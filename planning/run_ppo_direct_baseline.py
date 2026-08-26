"""Throughput, training, and evaluation runner for PPO-Direct."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psutil

from planning.ppo_direct_baseline import (
    ACTION_DIM,
    ACTION_SCALE_MPS2,
    NUM_AGENTS,
    OBSERVATION_DIM,
    PPODirectRewardConfig,
    build_local_observations,
    install_pandas_import_guard,
    make_environment,
    make_sb3_vec_env,
    normalized_actions_to_accelerations,
)
from planning.final_four_stage_benchmark import step_direct_accelerations


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig" if not exists else "utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_snapshot() -> dict[str, float | str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip().splitlines()[0]
        utilization, used, total = [float(value.strip()) for value in output.split(",")]
        return {
            "gpu_utilization_percent": utilization,
            "gpu_memory_used_mb": used,
            "gpu_memory_total_mb": total,
        }
    except Exception as exc:
        return {
            "gpu_utilization_percent": "UNAVAILABLE",
            "gpu_memory_used_mb": "UNAVAILABLE",
            "gpu_memory_total_mb": "UNAVAILABLE",
            "gpu_query_error": str(exc),
        }


def synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def build_model(env: Any, *, device: str, seed: int, n_steps: int = 2048) -> Any:
    install_pandas_import_guard()
    from stable_baselines3 import PPO

    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=int(n_steps),
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        device=device,
        seed=int(seed),
        verbose=0,
    )


def throughput(root: Path) -> None:
    manifest = load_json(root / "03_training_environment/PPO_DIRECT_TRAIN_MANIFEST.json")
    entries = manifest["entries"]
    rows: list[dict[str, Any]] = []
    for num_worlds in (1, 4, 8):
        env = make_sb3_vec_env(
            entries,
            num_worlds=num_worlds,
            reward_config=PPODirectRewardConfig(),
            threaded=True,
        )
        model = build_model(env, device="cuda", seed=20260823, n_steps=128)
        obs = env.reset()
        for _ in range(5):
            actions, _ = model.predict(obs, deterministic=False)
            obs, _, _, _ = env.step(actions)
        synchronize_cuda()
        t0 = time.perf_counter()
        forward_calls = 100
        for _ in range(forward_calls):
            actions, _ = model.predict(obs, deterministic=False)
        synchronize_cuda()
        forward_elapsed = time.perf_counter() - t0

        process = psutil.Process(os.getpid())
        cpu_start = process.cpu_times()
        rss_start = process.memory_info().rss
        step_count = 60
        t0 = time.perf_counter()
        for _ in range(step_count):
            random_actions = np.random.default_rng(20260823).uniform(
                -1.0, 1.0, size=(env.num_envs, ACTION_DIM)
            ).astype(np.float32)
            obs, _, _, _ = env.step(random_actions)
        step_elapsed = time.perf_counter() - t0
        cpu_end = process.cpu_times()
        cpu_seconds = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
        logical = max(1, psutil.cpu_count(logical=True) or 1)
        cpu_percent = 100.0 * cpu_seconds / max(step_elapsed, 1e-9) / logical
        rss_peak = max(rss_start, process.memory_info().rss)

        # One representative PPO rollout/update with the requested ten epochs.
        update_start = time.perf_counter()
        model.learn(total_timesteps=env.num_envs * 128, reset_num_timesteps=True)
        update_elapsed = time.perf_counter() - update_start
        total_samples = env.num_envs * 128
        samples_per_second = total_samples / max(update_elapsed, 1e-9)
        environment_transitions_per_second = (
            step_count * env.num_envs / max(step_elapsed, 1e-9)
        )
        record = {
            "num_worlds": num_worlds,
            "num_agent_env_rows": env.num_envs,
            "threaded_world_steps": True,
            "benchmark_world_steps": step_count,
            "environment_transitions_per_second": environment_transitions_per_second,
            "environment_step_ms_per_joint_world": 1000.0 * step_elapsed / (step_count * num_worlds),
            "policy_forward_ms_per_agent_batch": 1000.0 * forward_elapsed / forward_calls,
            "ppo_benchmark_rollout_steps": 128,
            "ppo_update_plus_collection_s": update_elapsed,
            "ppo_end_to_end_transitions_per_second": samples_per_second,
            "cpu_utilization_percent_of_machine": cpu_percent,
            "process_rss_mb": rss_peak / (1024**2),
            **gpu_snapshot(),
        }
        rows.append(record)
        env.close()
        print(json.dumps(record), flush=True)
    fields = tuple(rows[0].keys())
    write_csv(root / "04_throughput/PPO_DIRECT_TRAINING_THROUGHPUT.csv", rows, fields)
    selected = max(rows, key=lambda row: float(row["ppo_end_to_end_transitions_per_second"]))
    rate = float(selected["ppo_end_to_end_transitions_per_second"])
    eta = {
        str(target): {
            "seconds": target / rate,
            "hours": target / rate / 3600.0,
        }
        for target in (200_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000)
    }
    write_json(
        root / "04_throughput/PPO_DIRECT_TRAINING_ETA.json",
        {
            "schema_version": "ppo_direct_training_eta_v1",
            "selected_num_worlds": int(selected["num_worlds"]),
            "selection_basis": "maximum measured PPO collection-plus-update agent transitions/s",
            "selected_transitions_per_second": rate,
            "eta": eta,
            "TRAINING_TIME_RISK": "HIGH" if eta["3000000"]["hours"] > 12.0 else "LOW",
            "benchmark_note": "128-step rollout engineering benchmark; phase wall clock is reported separately",
        },
    )
    config_path = root / "03_training_environment/PPO_DIRECT_TRAINING_CONFIG.json"
    config = load_json(config_path)
    config["num_worlds"] = int(selected["num_worlds"])
    config["throughput_selection_artifact"] = "04_throughput/PPO_DIRECT_TRAINING_THROUGHPUT.csv"
    write_json(config_path, config)


class TrainingHistoryCallback:
    """Factory wrapper to avoid importing SB3 before the pandas guard."""

    @staticmethod
    def create(root: Path):
        install_pandas_import_guard()
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)
                self.recent = deque(maxlen=100)
                self.rollout_actions: list[np.ndarray] = []
                self.rollout_rewards: list[np.ndarray] = []
                self.started = time.perf_counter()
                self.rows: list[dict[str, Any]] = []

            def _on_step(self) -> bool:
                actions = self.locals.get("actions")
                rewards = self.locals.get("rewards")
                infos = self.locals.get("infos", [])
                if actions is not None:
                    self.rollout_actions.append(np.asarray(actions, dtype=np.float32))
                if rewards is not None:
                    self.rollout_rewards.append(np.asarray(rewards, dtype=np.float32))
                for info in infos:
                    if "team_episode" in info:
                        self.recent.append(dict(info["team_episode"]))
                return True

            def _on_rollout_end(self) -> None:
                actions = (
                    np.concatenate(self.rollout_actions, axis=0)
                    if self.rollout_actions
                    else np.zeros((0, ACTION_DIM), dtype=np.float32)
                )
                rewards = (
                    np.concatenate(self.rollout_rewards, axis=0)
                    if self.rollout_rewards
                    else np.zeros(0, dtype=np.float32)
                )
                episodes = list(self.recent)
                row = {
                    "timesteps": int(self.model.num_timesteps),
                    "wallclock_s": time.perf_counter() - self.started,
                    "recent_team_episodes": len(episodes),
                    "recent_team_success": float(np.mean([e["success"] for e in episodes])) if episodes else math.nan,
                    "recent_collision": float(np.mean([e["collision"] for e in episodes])) if episodes else math.nan,
                    "recent_peer_collision": float(np.mean([e["peer_collision"] for e in episodes])) if episodes else math.nan,
                    "recent_timeout": float(np.mean([e["timeout"] for e in episodes])) if episodes else math.nan,
                    "recent_agent_completion": float(np.mean([e["agent_completion"] for e in episodes])) if episodes else math.nan,
                    "mean_step_reward": float(np.mean(rewards)) if rewards.size else math.nan,
                    "mean_abs_action": float(np.mean(np.abs(actions))) if actions.size else math.nan,
                    "action_saturation_rate": float(np.mean(np.abs(actions) >= 0.999)) if actions.size else math.nan,
                    "mean_episode_steps": float(np.mean([e["steps"] for e in episodes])) if episodes else math.nan,
                    "approx_kl": float(self.model.logger.name_to_value.get("train/approx_kl", math.nan)),
                    "entropy_loss": float(self.model.logger.name_to_value.get("train/entropy_loss", math.nan)),
                    "policy_gradient_loss": float(self.model.logger.name_to_value.get("train/policy_gradient_loss", math.nan)),
                    "value_loss": float(self.model.logger.name_to_value.get("train/value_loss", math.nan)),
                }
                self.rows.append(row)
                self.rollout_actions.clear()
                self.rollout_rewards.clear()

            def _on_training_end(self) -> None:
                fields = (
                    "timesteps", "wallclock_s", "recent_team_episodes", "recent_team_success",
                    "recent_collision", "recent_peer_collision", "recent_timeout",
                    "recent_agent_completion", "mean_step_reward", "mean_abs_action",
                    "action_saturation_rate", "mean_episode_steps", "approx_kl",
                    "entropy_loss", "policy_gradient_loss", "value_loss",
                )
                append_csv(root / "05_training/PPO_DIRECT_TRAINING_HISTORY.csv", self.rows, fields)
                append_csv(root / "05_training/PPO_DIRECT_LEARNING_CURVES.csv", self.rows, fields)

        return _Callback()


def train(root: Path, target: int, resume: Path | None, device: str) -> None:
    install_pandas_import_guard()
    from stable_baselines3 import PPO

    config = load_json(root / "03_training_environment/PPO_DIRECT_TRAINING_CONFIG.json")
    num_worlds = int(config["num_worlds"])
    manifest = load_json(root / "03_training_environment/PPO_DIRECT_TRAIN_MANIFEST.json")
    env = make_sb3_vec_env(
        manifest["entries"],
        num_worlds=num_worlds,
        reward_config=PPODirectRewardConfig(),
        threaded=True,
    )
    if resume is None:
        model = build_model(env, device=device, seed=int(config["seed"]), n_steps=int(config["n_steps"]))
        current = 0
    else:
        model = PPO.load(str(resume), env=env, device=device)
        current = int(model.num_timesteps)
    if target <= current:
        raise ValueError(f"target {target} must exceed checkpoint timesteps {current}")
    callback = TrainingHistoryCallback.create(root)
    started = time.perf_counter()
    model.learn(
        total_timesteps=int(target - current),
        callback=callback,
        reset_num_timesteps=False,
        progress_bar=False,
    )
    wall = time.perf_counter() - started
    actual = int(model.num_timesteps)
    checkpoint = root / f"05_training/checkpoints/ppo_direct_{actual}.zip"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint.with_suffix("")))
    record = {
        "nominal_target_timesteps": int(target),
        "actual_timesteps": actual,
        "phase_start_timesteps": current,
        "phase_wallclock_s": wall,
        "phase_agent_transitions_per_second": (actual - current) / max(wall, 1e-9),
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": sha256(checkpoint),
        "parameter_count": int(sum(parameter.numel() for parameter in model.policy.parameters())),
        "device": str(model.device),
    }
    write_json(root / f"05_training/phase_{target}_summary.json", record)
    env.close()
    print(json.dumps(record), flush=True)


def min_obstacle_clearance(env: Any) -> float:
    obstacles = list(env.static_obstacles) + list(env.dynamic_obstacles)
    return min(
        (
            float(obstacle.signed_distance(dynamic.p))
            for dynamic in env.dynamics
            for obstacle in obstacles
        ),
        default=float("inf"),
    )


def run_episode(model: Any, entry: Mapping[str, Any], save_trajectory: bool) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    env = make_environment(entry)
    obs = build_local_observations(env)
    positions = [env._positions().copy()]
    velocities = [env._velocities().copy()]
    accelerations: list[np.ndarray] = []
    min_obstacle = min_obstacle_clearance(env)
    min_peer = float(env._check_collision()["min_inter_agent_distance"])
    inference_s = 0.0
    observation_s = 0.0
    conversion_s = 0.0
    calls = 0
    last_info: dict[str, Any] | None = None
    while True:
        synchronize_cuda()
        t0 = time.perf_counter()
        actions, _ = model.predict(obs, deterministic=True)
        synchronize_cuda()
        inference_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        commanded = normalized_actions_to_accelerations(actions)
        conversion_s += time.perf_counter() - t0
        terminated, truncated, info = step_direct_accelerations(
            env, commanded, refresh_sensors=True
        )
        calls += NUM_AGENTS
        positions.append(env._positions().copy())
        velocities.append(env._velocities().copy())
        accelerations.append(np.asarray(info["applied_accelerations"], dtype=float))
        min_obstacle = min(min_obstacle, min_obstacle_clearance(env))
        min_peer = min(min_peer, float(info["min_inter_agent_distance"]))
        t0 = time.perf_counter()
        obs = build_local_observations(env)
        observation_s += time.perf_counter() - t0
        last_info = dict(info)
        if terminated or truncated:
            break
    assert last_info is not None
    position_array = np.asarray(positions, dtype=float)
    velocity_array = np.asarray(velocities, dtype=float)
    acceleration_array = np.asarray(accelerations, dtype=float)
    path_per_agent = np.sum(np.linalg.norm(np.diff(position_array, axis=0), axis=2), axis=0)
    straight = np.linalg.norm(np.asarray(entry["goals"]) - np.asarray(entry["starts"]), axis=1)
    efficiency = straight / np.maximum(path_per_agent, 1e-8)
    if len(acceleration_array) >= 2:
        jerk = np.diff(acceleration_array, axis=0) / 0.1
        smoothness = float(np.mean(np.sum(jerk**2, axis=2)))
    else:
        smoothness = 0.0
    obstacle_mask = np.asarray(last_info["obstacle_collision_mask"], dtype=bool)
    static_mask = np.asarray(last_info["static_obstacle_collision_mask"], dtype=bool)
    dynamic_mask = np.asarray(last_info["dynamic_obstacle_collision_mask"], dtype=bool)
    peer_mask = np.asarray(last_info["inter_agent_collision_mask"], dtype=bool)
    success_mask = np.asarray(last_info["success_mask"], dtype=bool)
    team = {
        "scenario_id": str(entry["scenario_id"]),
        "seed": int(entry["seed"]),
        "stage": str(entry["stage"]),
        "family": str(entry["family"]),
        "task_pattern": str(entry["task_pattern"]),
        "success": bool(last_info["success"]),
        "collision": bool(last_info["collision"]),
        "static_collision": bool(np.any(static_mask)),
        "dynamic_collision": bool(np.any(dynamic_mask)),
        "obstacle_collision": bool(np.any(obstacle_mask)),
        "peer_collision": bool(np.any(peer_mask)),
        "boundary_collision": bool(np.any(last_info["boundary_collision_mask"])),
        "timeout": bool(last_info["truncated"]),
        "agent_completion": float(np.mean(success_mask)),
        "agent_completion_count": int(np.sum(success_mask)),
        "steps": int(last_info["steps"]),
        "completion_time_s": 0.1 * int(last_info["steps"]) if last_info["success"] else "",
        "team_path_length_m": float(np.sum(path_per_agent)),
        "path_efficiency": float(np.mean(efficiency[success_mask])) if np.any(success_mask) else "",
        "smoothness": smoothness,
        "minimum_obstacle_clearance_m": float(min_obstacle),
        "minimum_peer_distance_m": float(min_peer),
        "policy_calls": calls,
        "ppo_inference_compute_ms": 1000.0 * inference_s,
        "observation_assembly_compute_ms": 1000.0 * observation_s,
        "action_conversion_compute_ms": 1000.0 * conversion_s,
        "total_online_compute_ms": 1000.0 * (inference_s + observation_s + conversion_s),
    }
    agents = [
        {
            "scenario_id": str(entry["scenario_id"]),
            "agent_id": agent_id,
            "completed": bool(success_mask[agent_id]),
            "collision": bool(np.asarray(last_info["collision_mask"])[agent_id]),
            "obstacle_collision": bool(obstacle_mask[agent_id]),
            "peer_collision": bool(peer_mask[agent_id]),
            "path_length_m": float(path_per_agent[agent_id]),
            "path_efficiency": float(efficiency[agent_id]),
        }
        for agent_id in range(NUM_AGENTS)
    ]
    trajectory = {
        "schema_version": "ppo_direct_trajectory_v1",
        "scenario_id": str(entry["scenario_id"]),
        "seed": int(entry["seed"]),
        "stage": str(entry["stage"]),
        "dt": 0.1,
        "starts": entry["starts"],
        "goals": entry["goals"],
        "positions": position_array.tolist() if save_trajectory else [],
        "velocities": velocity_array.tolist() if save_trajectory else [],
        "applied_accelerations": acceleration_array.tolist() if save_trajectory else [],
        "outcome": team,
    }
    env.close()
    return team, agents, trajectory


def summarize(team_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = [("overall", list(team_rows))]
    groups.extend(
        (stage, [row for row in team_rows if row["stage"] == stage])
        for stage in ("stage_1", "stage_2", "stage_3", "stage_4")
    )
    groups.append(("stage_3_4", [row for row in team_rows if row["stage"] in {"stage_3", "stage_4"}]))
    result = []
    for scope, rows in groups:
        n = len(rows)
        success_rows = [row for row in rows if row["success"]]
        result.append(
            {
                "scope": scope,
                "n": n,
                "success_count": sum(bool(row["success"]) for row in rows),
                "success_rate": np.mean([row["success"] for row in rows]) if n else math.nan,
                "collision_rate": np.mean([row["collision"] for row in rows]) if n else math.nan,
                "static_collision_rate": np.mean([row["static_collision"] for row in rows]) if n else math.nan,
                "dynamic_collision_rate": np.mean([row["dynamic_collision"] for row in rows]) if n else math.nan,
                "peer_collision_rate": np.mean([row["peer_collision"] for row in rows]) if n else math.nan,
                "timeout_rate": np.mean([row["timeout"] for row in rows]) if n else math.nan,
                "agent_completion": np.mean([row["agent_completion"] for row in rows]) if n else math.nan,
                "mean_steps": np.mean([row["steps"] for row in rows]) if n else math.nan,
                "mean_success_completion_time_s": np.mean([row["completion_time_s"] for row in success_rows]) if success_rows else math.nan,
                "mean_total_online_compute_ms": np.mean([row["total_online_compute_ms"] for row in rows]) if n else math.nan,
            }
        )
    return result


def evaluate(root: Path, checkpoint: Path, split: str, limit: int | None, trajectories: bool, device: str) -> None:
    install_pandas_import_guard()
    from stable_baselines3 import PPO

    filename = {
        "dev": "PPO_DIRECT_DEV_MANIFEST.json",
        "holdout": "PPO_DIRECT_HOLDOUT_MANIFEST.json",
        "formal": "FORMAL_V2_MANIFEST.json",
    }[split]
    if split == "formal":
        manifest_path = Path(
            "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2/FORMAL_V2_MANIFEST.json"
        )
    else:
        manifest_path = root / f"03_training_environment/{filename}"
    manifest = load_json(manifest_path)
    entries = list(manifest["entries"])
    if limit is not None:
        # Balanced prefix by taking the same count from each stage.
        per_stage = max(1, int(limit) // 4)
        entries = [
            entry
            for stage in ("stage_1", "stage_2", "stage_3", "stage_4")
            for entry in [row for row in entries if row["stage"] == stage][:per_stage]
        ]
    model = PPO.load(str(checkpoint), device=device)
    team_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    trajectory_manifest: list[dict[str, Any]] = []
    output_dir = {
        "dev": root / "06_development",
        "holdout": root / "07_holdout",
        "formal": root / "09_formal_v2",
    }[split]
    trajectory_dir = root / "12_trajectories/formal_records"
    for index, entry in enumerate(entries, start=1):
        team, agents, trajectory = run_episode(model, entry, trajectories)
        team_rows.append(team)
        agent_rows.extend(agents)
        if trajectories:
            path = trajectory_dir / f"{entry['scenario_id']}.json"
            write_json(path, trajectory)
            trajectory_manifest.append(
                {
                    "scenario_id": entry["scenario_id"],
                    "stage": entry["stage"],
                    "success": team["success"],
                    "path": str(path.relative_to(root)),
                    "sha256": sha256(path),
                }
            )
        if index % 10 == 0 or index == len(entries):
            print(
                json.dumps(
                    {
                        "split": split,
                        "completed": index,
                        "total": len(entries),
                        "success": float(np.mean([row["success"] for row in team_rows])),
                        "collision": float(np.mean([row["collision"] for row in team_rows])),
                    }
                ),
                flush=True,
            )
    summary = summarize(team_rows)
    team_fields = tuple(team_rows[0].keys())
    agent_fields = tuple(agent_rows[0].keys())
    summary_fields = tuple(summary[0].keys())
    if split == "formal":
        write_csv(output_dir / "ppo_direct_formal_team_results.csv", team_rows, team_fields)
        write_csv(output_dir / "ppo_direct_formal_agent_results.csv", agent_rows, agent_fields)
        write_csv(output_dir / "ppo_direct_stage_summary.csv", summary, summary_fields)
        write_csv(root / "12_trajectories/ppo_direct_trajectory_manifest.csv", trajectory_manifest, ("scenario_id", "stage", "success", "path", "sha256"))
    else:
        label = split.upper()
        write_csv(output_dir / f"PPO_DIRECT_{label}_RESULTS.csv", team_rows, team_fields)
        write_csv(output_dir / f"PPO_DIRECT_{label}_SUMMARY.csv", summary, summary_fields)
    write_json(
        output_dir / f"{split}_evaluation_summary.json",
        {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "summary": summary},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("throughput", "train", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("dev", "holdout", "formal"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trajectories", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "throughput":
        throughput(root)
    elif args.command == "train":
        if args.target is None:
            parser.error("--target is required for train")
        train(root, args.target, args.resume, args.device)
    else:
        if args.checkpoint is None or args.split is None:
            parser.error("--checkpoint and --split are required for evaluate")
        evaluate(root, args.checkpoint, args.split, args.limit, args.trajectories, args.device)


if __name__ == "__main__":
    main()
