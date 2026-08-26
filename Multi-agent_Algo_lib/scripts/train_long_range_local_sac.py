"""Fine-tune the unchanged SAC-DMP actor on independent long-range local patches."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
# On Windows, initialize pandas/pyarrow before PyTorch.  Importing it lazily via
# baseline.common.callbacks after torch can terminate this process inside the
# pyarrow DLL loader (0xc0000005) before Python can raise an exception.
import pandas as _pandas  # noqa: F401
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Controller.dmp_rl import DMPConfig  # noqa: E402
from Entity.dynamic_obstacles import MovingSphereObstacle  # noqa: E402
from Entity.static_obstacles import WorkspaceBoundaryPlaneObstacle  # noqa: E402
from Environment.single_agent_dmp_env import EnvConfig, SingleAgentDMPEnv  # noqa: E402
from baseline.common.callbacks import BaseCallback  # noqa: E402
from experiment_config import EXPERIMENT_CONFIG  # noqa: E402
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    propagate_historical_sac_dmp_action,
)
from planning.semi_structured_long_range_benchmark import (  # noqa: E402
    SENSOR_RANGE_M,
    WORKSPACE_BOUNDS,
    json_ready,
)
from runner_sac import build_model, load_checkpoint, save_checkpoint  # noqa: E402
from scripts.run_long_range_contract_pilot import obstacle_from_spec  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/training/long_range_local_sac_256_encoder_adapt.json"
SOURCE_PATHS = (
    "Environment/single_agent_dmp_env.py",
    "Entity/KinematicModel.py",
    "Environment/frozen_sac_dmp_execution.py",
    "planning/historical_forcing_gate.py",
    "runner_sac.py",
    "baseline/sac/sac.py",
    "baseline/sac/policies.py",
    "baseline/sac/net.py",
    "Multi-agent_Algo_lib/scripts/train_long_range_local_sac.py",
    "configs/training/long_range_local_sac_256_encoder_adapt.json",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_group_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _ray_directions(azimuth_bins: int, elevation_bins: int, elevation_range: Sequence[float]) -> np.ndarray:
    azimuth = np.linspace(-np.pi, np.pi, int(azimuth_bins), endpoint=False, dtype=float)
    elevation = np.deg2rad(
        np.linspace(float(elevation_range[0]), float(elevation_range[1]), int(elevation_bins), dtype=float)
    )
    directions = []
    for azimuth_value in azimuth:
        for elevation_value in elevation:
            cosine = float(np.cos(elevation_value))
            directions.append(
                [
                    cosine * float(np.cos(azimuth_value)),
                    cosine * float(np.sin(azimuth_value)),
                    float(np.sin(elevation_value)),
                ]
            )
    return np.asarray(directions, dtype=float)


def _old_to_new_ray_mapping(config: Mapping[str, Any]) -> list[int]:
    source = config["source_sensor"]
    target = config["sensor"]
    old_directions = _ray_directions(
        source["azimuth_bins"], source["elevation_bins"], source["elevation_range_deg"]
    )
    new_directions = _ray_directions(
        target["azimuth_bins"], target["elevation_bins"], target["elevation_range_deg"]
    )
    return np.argmax(old_directions @ new_directions.T, axis=1).astype(int).tolist()


def _expanded_encoder_state(
    target_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
    *,
    mapping: Sequence[int],
    source_ray_count: int,
    target_ray_count: int,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    first_layer = "sensor_encoder.layers.0.weight"
    expanded = {name: tensor.detach().clone() for name, tensor in target_state.items()}
    exact_names: list[str] = []
    for name, target_tensor in expanded.items():
        if name == first_layer:
            continue
        source_tensor = source_state[name]
        if source_tensor.shape != target_tensor.shape:
            raise RuntimeError(f"unexpected non-input checkpoint shape mismatch for {name}")
        expanded[name] = source_tensor.detach().clone()
        exact_names.append(name)

    source_weight = source_state[first_layer].detach()
    target_weight = torch.zeros_like(expanded[first_layer])
    prefix = 7
    if source_weight.shape[1] != prefix + 2 * int(source_ray_count):
        raise RuntimeError("source sensor encoder input width does not match its ray contract")
    if target_weight.shape[1] != prefix + 2 * int(target_ray_count):
        raise RuntimeError("target sensor encoder input width does not match its ray contract")
    target_weight[:, :prefix] = source_weight[:, :prefix]
    for source_index, target_index in enumerate(mapping):
        target_weight[:, prefix + int(target_index)] += source_weight[:, prefix + source_index]
        target_weight[:, prefix + target_ray_count + int(target_index)] += source_weight[
            :, prefix + source_ray_count + source_index
        ]
    expanded[first_layer] = target_weight
    return expanded, exact_names


def load_expanded_encoder_checkpoint(
    model: Any,
    checkpoint_path: Path,
    config: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    source_ray_count = int(config["source_sensor"]["rays_per_frame"])
    target_ray_count = int(config["sensor"]["rays_per_frame"])
    mapping = _old_to_new_ray_mapping(config)
    if len(mapping) != source_ray_count or len(set(mapping)) != source_ray_count:
        raise RuntimeError("nearest-direction transfer must map every old ray to a unique new ray")

    exact_by_module: dict[str, list[str]] = {}
    for module_name in ("actor", "critic", "critic_target"):
        module = getattr(model, module_name)
        expanded, exact_names = _expanded_encoder_state(
            module.state_dict(),
            checkpoint[module_name],
            mapping=mapping,
            source_ray_count=source_ray_count,
            target_ray_count=target_ray_count,
        )
        module.load_state_dict(expanded, strict=True)
        exact_by_module[module_name] = exact_names

    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in model.actor.named_parameters():
        trainable = name.startswith("sensor_encoder.")
        parameter.requires_grad_(trainable)
        (trainable_names if trainable else frozen_names).append(name)
    model.actor.optimizer = torch.optim.Adam(
        [parameter for parameter in model.actor.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
    )

    if "log_ent_coef" in checkpoint:
        model.log_ent_coef = None
        model.ent_coef_optimizer = None
        model.ent_coef_tensor = checkpoint["log_ent_coef"].to(model.device).exp().detach()

    transfer = {
        "schema_version": "sac_256_encoder_transfer_v1",
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_rays_per_frame": source_ray_count,
        "target_rays_per_frame": target_ray_count,
        "old_to_new_nearest_direction_indices": mapping,
        "mapped_old_ray_count": len(mapping),
        "unique_target_ray_count": len(set(mapping)),
        "new_input_columns_initialized_to_zero_except_mapped_old_directions": True,
        "actor_trainable_parameter_names": trainable_names,
        "actor_frozen_parameter_names": frozen_names,
        "actor_trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.actor.parameters() if parameter.requires_grad)
        ),
        "actor_frozen_parameter_count": int(
            sum(parameter.numel() for parameter in model.actor.parameters() if not parameter.requires_grad)
        ),
        "exact_checkpoint_parameter_names_by_module": exact_by_module,
        "critic_role": "training_only_value_estimator",
        "entropy_coefficient_frozen_from_source": "log_ent_coef" in checkpoint,
    }
    atomic_json(run_dir / "encoder_transfer_report.json", transfer)
    return transfer


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def boundary_obstacles() -> list[WorkspaceBoundaryPlaneObstacle]:
    lower = np.asarray(WORKSPACE_BOUNDS[0], dtype=float)
    upper = np.asarray(WORKSPACE_BOUNDS[1], dtype=float)
    result = []
    for axis in range(3):
        result.append(
            WorkspaceBoundaryPlaneObstacle(
                axis=axis, bound=lower[axis], lower_bounds=lower,
                upper_bounds=upper, is_lower=True,
            )
        )
        result.append(
            WorkspaceBoundaryPlaneObstacle(
                axis=axis, bound=upper[axis], lower_bounds=lower,
                upper_bounds=upper, is_lower=False,
            )
        )
    return result


class LongRangeLocalReferenceEnv(SingleAgentDMPEnv):
    """Single shared-policy training view of manifest-local reference tasks."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        training_config: Mapping[str, Any],
        *,
        seed: int,
        training: bool,
    ) -> None:
        self.entries = list(manifest["entries"])
        self.training_config = training_config
        self.training_mode = bool(training)
        self.episode_rng = np.random.default_rng(int(seed))
        self.reset_records: list[dict[str, Any]] = []
        self.outcome_counts = {"success": 0, "collision": 0, "timeout": 0}
        reward = training_config["reward"]
        base = EXPERIMENT_CONFIG
        env_config = EnvConfig(
            max_steps=int(training_config["local_episode_max_steps"]),
            goal_tolerance=float(training_config["goal_tolerance_m"]),
            obstacle_potential_weight=float(reward["obstacle_potential_weight"]),
            obstacle_influence_distance=float(reward["obstacle_influence_distance_m"]),
            obstacle_potential_penalty_max=40.0,
            boundary_influence_distance=float(reward["boundary_influence_distance_m"]),
            boundary_potential_weight=float(reward["boundary_potential_weight"]),
            boundary_potential_penalty_max=40.0,
            step_reward_weight=float(reward["step_reward_weight"]),
            step_penalty=float(reward["step_penalty"]),
            collision_penalty=float(reward["collision_penalty"]),
            timeout_penalty=float(reward["timeout_penalty"]),
            success_bonus=float(reward["success_bonus"]),
            collision_margin=0.0,
            workspace_bounds=tuple(tuple(row) for row in WORKSPACE_BOUNDS),
            action_guidance_enabled=bool(
                training and training_config["action_guidance"]["enabled_training_only"]
            ),
            action_guidance_radius=float(training_config["action_guidance"]["radius_m"]),
            action_guidance_initial_weight=float(training_config["action_guidance"]["initial_weight"]),
            action_guidance_decay_steps=int(training_config["action_guidance"]["decay_steps"]),
        )
        super().__init__(
            dynamics_config={
                **base.build_dynamics_config(),
                "maximum_speed_norm": float(training_config["maximum_speed_norm_mps"]),
            },
            sensor_config={
                **base.build_sensor_config(),
                "sensing_radius": float(training_config["sensor"]["range_m"]),
                "azimuth_bins": int(training_config["sensor"]["azimuth_bins"]),
                "elevation_bins": int(training_config["sensor"]["elevation_bins"]),
                "include_previous_scan": True,
                "goal_distance_clip": 2.0 * float(training_config["sensor"]["range_m"]),
            },
            dmp_config=DMPConfig(
                dt=float(base.time_step),
                dims=3,
                K_alpha=float(base.k_alpha),
                K_beta=float(base.k_beta),
                alpha_s=float(base.alpha_s),
                tau=float(base.tau),
                forcing_term_max=float(base.forcing_term_max),
                forcing_term_min=float(base.forcing_term_min),
                goal_offset_max=float(base.goal_offset_max),
                phase_mode="classic",
                phase_integrator="legacy_euler",
            ),
            env_config=env_config,
            transition_function=propagate_historical_sac_dmp_action,
        )
        expected_observation_dim = int(training_config["frozen_semantics"]["actor_observation_dim"])
        if self.observation_space.shape != (expected_observation_dim,) or self.action_space.shape != (6,):
            raise RuntimeError("training environment does not match the frozen retraining interface")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.episode_rng = np.random.default_rng(int(seed))
        entry = self.entries[int(self.episode_rng.integers(0, len(self.entries)))]
        sample = self._sample_patch(entry)
        observation, info = super().reset(
            seed=int(self.episode_rng.integers(0, np.iinfo(np.uint32).max)),
            options=sample,
        )
        self.current_entry = entry
        self.current_sample = sample
        if len(self.reset_records) < 20000:
            self.reset_records.append(
                {
                    "scenario_id": entry["scenario_id"],
                    "stage": entry["stage"],
                    "family": entry["family"],
                    "task_pattern": entry["task_pattern"],
                    "reference_distance_m": float(np.linalg.norm(sample["goal"] - sample["start"])),
                    "start_z_m": float(sample["start"][2]),
                    "goal_z_m": float(sample["goal"][2]),
                    "local_static_count": len(sample["static_obstacles"]) - 6,
                    "local_dynamic_count": len(sample["dynamic_obstacles"]),
                }
            )
        return observation, info

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            if info["success"]:
                self.outcome_counts["success"] += 1
            elif info["collision"]:
                self.outcome_counts["collision"] += 1
            else:
                self.outcome_counts["timeout"] += 1
        return observation, reward, terminated, truncated, info

    def _sample_patch(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        static_all = [obstacle_from_spec(row) for row in entry["static_obstacles"]]
        time_index = int(self.episode_rng.integers(0, int(entry["max_steps"]) + 1))
        dynamic_all: list[MovingSphereObstacle] = []
        for spec, track in zip(
            entry["dynamic_obstacles"], entry["dynamic_obstacle_trajectories"], strict=True
        ):
            dynamic_all.append(
                MovingSphereObstacle(
                    center=np.asarray(track[time_index], dtype=float),
                    velocity=np.asarray(spec["velocity"], dtype=float),
                    radius=float(spec["radius"]),
                    safety_margin=float(spec.get("safety_margin", 0.0)),
                    bounds=None,
                )
            )
        starts = np.asarray(entry["starts"], dtype=float)
        terminals = np.asarray(entry["goals"], dtype=float)
        lower = np.asarray(WORKSPACE_BOUNDS[0], dtype=float)
        upper = np.asarray(WORKSPACE_BOUNDS[1], dtype=float)
        ref_cfg = self.training_config["local_reference_distance_m"]
        for _ in range(500):
            agent = int(self.episode_rng.integers(0, len(starts)))
            alpha = float(self.episode_rng.uniform(0.0, 0.94))
            start = starts[agent] + alpha * (terminals[agent] - starts[agent])
            start[:2] += self.episode_rng.uniform(-2.5, 2.5, size=2)
            if self.episode_rng.random() < float(self.training_config["boundary_exposure_probability"]):
                if self.episode_rng.random() < 0.5:
                    start[2] = lower[2] + self.episode_rng.uniform(0.32, 0.95)
                else:
                    start[2] = upper[2] - self.episode_rng.uniform(0.32, 0.95)
            else:
                start[2] += self.episode_rng.uniform(-0.35, 0.35)
            start = np.clip(start, lower + 0.31, upper - 0.31)
            target_direction = terminals[agent] - start
            norm = float(np.linalg.norm(target_direction))
            if norm < 1.0e-8:
                continue
            target_direction /= norm
            target_direction += self.episode_rng.normal(0.0, 0.18, size=3)
            direction_norm = float(np.linalg.norm(target_direction))
            if direction_norm < 1.0e-8:
                continue
            target_direction /= direction_norm
            if self.episode_rng.random() < float(ref_cfg["near_probability"]):
                distance = float(self.episode_rng.uniform(*ref_cfg["near_range"]))
            else:
                distance = float(self.episode_rng.uniform(*ref_cfg["extended_range"]))
            goal = start + distance * target_direction
            if np.any(goal < lower + 0.26) or np.any(goal > upper - 0.26):
                continue
            if not self._segment_clear(start, goal, static_all, dynamic_all):
                continue
            local_static = [
                obstacle for obstacle in static_all
                if min(obstacle.signed_distance(start), obstacle.signed_distance(goal)) <= 8.0
            ]
            local_dynamic = [
                obstacle for obstacle in dynamic_all
                if min(obstacle.signed_distance(start), obstacle.signed_distance(goal)) <= 8.0
            ]
            return {
                "start": start,
                "goal": goal,
                "task_goal": terminals[agent].copy(),
                "source_agent_id": int(agent),
                "static_obstacles": [*local_static, *boundary_obstacles()],
                "dynamic_obstacles": local_dynamic,
            }
        raise RuntimeError(f"failed to sample valid local patch from {entry['scenario_id']}")

    @staticmethod
    def _segment_clear(
        start: np.ndarray, goal: np.ndarray, static: Sequence[Any], dynamic: Sequence[Any]
    ) -> bool:
        for alpha in np.linspace(0.0, 1.0, 11):
            point = (1.0 - alpha) * start + alpha * goal
            if any(obstacle.signed_distance(point) <= 0.30 for obstacle in static):
                return False
            if any(obstacle.signed_distance(point) <= 0.30 for obstacle in dynamic):
                return False
        return True


def evaluate_model(
    model: Any,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int,
    count: int,
    checkpoint_label: str,
    timestep: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = LongRangeLocalReferenceEnv(
        manifest, config, seed=int(seed), training=False
    )
    rows: list[dict[str, Any]] = []
    for episode_id in range(int(count)):
        obs, reset_info = env.reset(seed=int(seed + episode_id))
        total_reward = 0.0
        terminated = truncated = False
        info = reset_info
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
        position = env.dynamics.p.copy()
        lower = np.asarray(WORKSPACE_BOUNDS[0], dtype=float)
        upper = np.asarray(WORKSPACE_BOUNDS[1], dtype=float)
        outside = bool(np.any(position <= lower) or np.any(position >= upper))
        rows.append(
            {
                "checkpoint": checkpoint_label,
                "training_timestep": int(timestep),
                "episode_id": episode_id,
                "source_scenario_id": env.current_entry["scenario_id"],
                "stage": env.current_entry["stage"],
                "family": env.current_entry["family"],
                "success": bool(info["success"]),
                "collision": bool(info["collision"]),
                "boundary_collision": bool(info["collision"] and outside),
                "timeout": bool(truncated),
                "steps": int(env.steps),
                "reward": total_reward,
                "final_distance_m": float(info["distance_to_goal"]),
                "maximum_speed_norm_mps": float(np.linalg.norm(env.dynamics.v)),
            }
        )
    env.close()
    success = float(np.mean([row["success"] for row in rows]))
    collision = float(np.mean([row["collision"] for row in rows]))
    boundary = float(np.mean([row["boundary_collision"] for row in rows]))
    timeout = float(np.mean([row["timeout"] for row in rows]))
    summary = {
        "checkpoint": checkpoint_label,
        "training_timestep": int(timestep),
        "episode_count": len(rows),
        "success_rate": success,
        "collision_rate": collision,
        "boundary_collision_rate": boundary,
        "timeout_rate": timeout,
        "mean_final_distance_m": float(np.mean([row["final_distance_m"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
    }
    return summary, rows


class EvaluationCheckpointCallback(BaseCallback):
    def __init__(
        self,
        *,
        run_dir: Path,
        validation_manifest: Mapping[str, Any],
        training_config: Mapping[str, Any],
        interval: int,
        validation_count: int,
        seed: int,
    ) -> None:
        super().__init__(verbose=1)
        self.run_dir = run_dir
        self.validation_manifest = validation_manifest
        self.training_config = training_config
        self.interval = int(interval)
        self.validation_count = int(validation_count)
        self.seed = int(seed)
        self.summaries: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []
        self.best_key: tuple[float, ...] | None = None

    def _on_step(self) -> bool:
        if self.num_timesteps <= 0 or self.num_timesteps % self.interval:
            return True
        checkpoint = self.run_dir / "checkpoints" / f"checkpoint_{self.num_timesteps:07d}.pt"
        save_checkpoint(self.model, checkpoint, extra={"long_range_finetune_step": self.num_timesteps})
        summary, rows = evaluate_model(
            self.model,
            self.validation_manifest,
            self.training_config,
            seed=self.seed,
            count=self.validation_count,
            checkpoint_label=checkpoint.name,
            timestep=self.num_timesteps,
        )
        self.summaries.append(summary)
        self.rows.extend(rows)
        key = (
            summary["success_rate"],
            -summary["collision_rate"],
            -summary["boundary_collision_rate"],
            -summary["timeout_rate"],
            -summary["mean_final_distance_m"],
        )
        if self.best_key is None or key > self.best_key:
            self.best_key = key
            save_checkpoint(
                self.model,
                self.run_dir / "best_validation.pt",
                extra={"long_range_finetune_step": self.num_timesteps, "validation_summary": summary},
            )
        write_csv(self.run_dir / "validation_summary.csv", self.summaries)
        write_csv(self.run_dir / "validation_episode_results.csv", self.rows)
        print(f"[sac-finetune] validation step={self.num_timesteps} success={summary['success_rate']:.3f} collision={summary['collision_rate']:.3f}", flush=True)
        return True


def prepare(config_path: Path) -> None:
    config = load_json(config_path)
    root = (REPO_ROOT / config["artifact_root"]).resolve()
    run_dir = root / "07_training" / config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    initialization = str(config["initialization"])
    if initialization not in {"source_checkpoint_encoder_only"}:
        raise RuntimeError(f"unsupported initialization: {initialization}")
    checkpoint = REPO_ROOT / config["source_checkpoint"]
    expected_checkpoint_hash = config["source_checkpoint_sha256"]
    if sha256_file(checkpoint) != expected_checkpoint_hash:
        raise RuntimeError("source SAC checkpoint provenance hash mismatch")
    training_manifest = root / config["training_manifest"]
    validation_manifest = root / config["validation_manifest"]
    freeze = {
        "schema_version": "long_range_local_sac_finetune_freeze_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "configuration_sha256": sha256_file(config_path),
        "initialization": initialization,
        "source_checkpoint_loaded": True,
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "training_manifest_sha256": sha256_file(training_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "source_sha256": {path: sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS},
        "development_manifest_read": False,
        "formal_manifest_exists": False,
        "formal_result_count": 0,
        "actor_trainable_scope": config["frozen_semantics"]["actor_trainable_scope"],
        "actor_trunk_frozen": True,
        "action_heads_frozen": True,
        "sensor_input_width_changed": True,
        "action_semantics_changed": False,
    }
    atomic_json(run_dir / "training_freeze.json", freeze)
    print(json.dumps({"phase": "prepare", "run_dir": str(run_dir), "status": "PASS"}), flush=True)


def verify_freeze(config_path: Path, config: Mapping[str, Any], run_dir: Path) -> None:
    freeze = load_json(run_dir / "training_freeze.json")
    if sha256_file(config_path) != freeze["configuration_sha256"]:
        raise RuntimeError("training config changed after freeze")
    for path, expected in freeze["source_sha256"].items():
        if sha256_file(REPO_ROOT / path) != expected:
            raise RuntimeError(f"training source changed after freeze: {path}")


def train(config_path: Path, total_timesteps_override: int | None) -> None:
    config = load_json(config_path)
    root = (REPO_ROOT / config["artifact_root"]).resolve()
    run_dir = root / "07_training" / config["run_id"]
    verify_freeze(config_path, config, run_dir)
    training_manifest = load_json(root / config["training_manifest"])
    validation_manifest = load_json(root / config["validation_manifest"])
    training_env = LongRangeLocalReferenceEnv(
        training_manifest, config, seed=int(config["seed"]), training=True
    )
    model_config = replace(
        EXPERIMENT_CONFIG,
        learning_rate=float(config["learning_rate"]),
        buffer_size=int(config["buffer_size"]),
        batch_size=int(config["batch_size"]),
        learning_starts=int(config["learning_starts"]),
        train_freq=int(config["train_freq"]),
        gradient_steps=int(config["gradient_steps"]),
        ent_coef=config["ent_coef"],
        verbose=1,
    )
    # Keep the TensorBoard path below the legacy Windows MAX_PATH boundary;
    # the run directory itself retains all model-selection artifacts.
    tensorboard_dir = root / "07_training" / "tb256"
    model = build_model(training_env, config=model_config, tensorboard_log=str(tensorboard_dir), verbose=1)
    transfer = load_expanded_encoder_checkpoint(
        model, REPO_ROOT / config["source_checkpoint"], config, run_dir
    )
    initial_label = "source_checkpoint_256_encoder_expansion"
    frozen_actor_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.actor.named_parameters()
        if not parameter.requires_grad
    }
    frozen_actor_sha256_before = tensor_group_sha256(frozen_actor_before)
    initial_summary, initial_rows = evaluate_model(
        model,
        validation_manifest,
        config,
        seed=int(config["seed"]) + 100000,
        count=int(config["validation_episode_count"]),
        checkpoint_label=initial_label,
        timestep=0,
    )
    callback = EvaluationCheckpointCallback(
        run_dir=run_dir,
        validation_manifest=validation_manifest,
        training_config=config,
        interval=int(config["checkpoint_interval"]),
        validation_count=int(config["validation_episode_count"]),
        seed=int(config["seed"]) + 100000,
    )
    callback.summaries.append(initial_summary)
    callback.rows.extend(initial_rows)
    write_csv(run_dir / "validation_summary.csv", callback.summaries)
    write_csv(run_dir / "validation_episode_results.csv", callback.rows)
    print(f"[sac-finetune] initial success={initial_summary['success_rate']:.3f} collision={initial_summary['collision_rate']:.3f}", flush=True)
    total = int(config["total_timesteps"] if total_timesteps_override is None else total_timesteps_override)
    model.learn(total_timesteps=total, callback=callback, reset_num_timesteps=True)
    save_checkpoint(model, run_dir / "final_model.pt", extra={"long_range_finetune_step": total})
    frozen_actor_after = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.actor.named_parameters()
        if not parameter.requires_grad
    }
    frozen_actor_sha256_after = tensor_group_sha256(frozen_actor_after)
    frozen_actor_exact = frozen_actor_sha256_before == frozen_actor_sha256_after
    if not frozen_actor_exact:
        raise RuntimeError("frozen actor trunk/action-head parameters changed during encoder-only adaptation")
    write_csv(run_dir / "training_local_patch_sample.csv", training_env.reset_records)
    atomic_json(
        run_dir / "training_result.json",
        {
            "schema_version": "long_range_local_sac_finetune_result_v1",
            "status": "COMPLETE",
            "trained_timesteps": total,
            "initialization": config["initialization"],
            "source_checkpoint_loaded": True,
            "source_checkpoint": config["source_checkpoint"],
            "source_checkpoint_sha256": config["source_checkpoint_sha256"],
            "final_checkpoint": "final_model.pt",
            "final_checkpoint_sha256": sha256_file(run_dir / "final_model.pt"),
            "best_validation_checkpoint": "best_validation.pt" if (run_dir / "best_validation.pt").is_file() else None,
            "best_validation_checkpoint_sha256": sha256_file(run_dir / "best_validation.pt") if (run_dir / "best_validation.pt").is_file() else None,
            "validation_history": callback.summaries,
            "training_outcome_counts": training_env.outcome_counts,
            "actor_trainable_scope": "sensor_encoder_only",
            "actor_trainable_parameter_names": transfer["actor_trainable_parameter_names"],
            "actor_frozen_parameter_names": transfer["actor_frozen_parameter_names"],
            "frozen_actor_sha256_before": frozen_actor_sha256_before,
            "frozen_actor_sha256_after": frozen_actor_sha256_after,
            "frozen_actor_exact_match": frozen_actor_exact,
            "sensor_input_width_changed": True,
            "observation_semantics_changed": False,
            "action_semantics_changed": False,
            "forcing_gate": HISTORICAL_GATE_NAME,
            "development_data_used": False,
            "formal_data_used": False,
        },
    )
    training_env.close()
    print(json.dumps({"phase": "train", "status": "COMPLETE", "timesteps": total}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "train"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timesteps", type=int)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.phase == "prepare":
        prepare(config_path)
    else:
        train(config_path, args.timesteps)


if __name__ == "__main__":
    main()
