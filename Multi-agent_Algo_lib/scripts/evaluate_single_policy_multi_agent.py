"""Evaluate a frozen single-agent SAC navigator in aligned multi-agent scenes.

Two observation-only baselines are evaluated with the same checkpoint, scene,
seed, dynamics, and termination rules:

``blind``
    Every UAV observes only the obstacles already present in the scene.

``peer_spheres``
    Every other UAV is exposed to the local LiDAR as a moving sphere.  No
    identity, communication, priority, or explicit inter-agent state is given
    to the policy.

The policy input remains exactly the historical 122-dimensional single-agent
input: 119 sensor features plus phase, K_alpha, and K_beta.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Environment.multi_agent_dmp_env import (  # noqa: E402
    MultiAgentDMPEnv,
    _AgentAsDynamicObstacle,
)
from MASAC.config import MASACExperimentConfig  # noqa: E402
from MASAC.curriculum import (  # noqa: E402
    build_stage_env_kwargs,
)
from experiment_config import EXPERIMENT_CONFIG as SINGLE_AGENT_CONFIG  # noqa: E402
from runner_sac import build_env as build_single_env  # noqa: E402
from runner_sac import build_model as build_single_model  # noqa: E402
from runner_sac import load_checkpoint  # noqa: E402
from scripts.validate_fcep_mechanism import (  # noqa: E402
    SCENARIO_NAMES as CANONICAL_SCENARIOS,
    build_scenario_options as build_canonical_scenario,
)
from scripts.validate_masac_multi_agent_scenarios import (  # noqa: E402
    SCENARIO_LABELS as DISTRIBUTION_LABELS,
    SCENARIO_NAMES as DISTRIBUTION_SCENARIOS,
    TRAINING_DISTRIBUTION_SCENARIO,
    _phase3_level3_stage,
    build_scenario as build_distribution_scenario,
)


OBSERVATION_MODES = ("blind", "peer_spheres")
MODE_LABELS = {
    "blind": "邻机不可见",
    "peer_spheres": "邻机动态球观测",
}
CANONICAL_LABELS = {
    "open_flight": "空旷飞行",
    "static_detour": "静态绕障",
    "dynamic_crossing": "动态横穿",
    "two_agent_crossing": "双机交叉",
    "narrow_yielding": "狭窄通道让行",
}
DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "evaluation" / "single_policy_multi_agent_no_coordination.json"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_jsonable(value), ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict, np.ndarray))
                    else value
                    for key, value in row.items()
                }
            )


def build_aligned_config(
    *,
    num_agents: int = 3,
    max_steps: int = 200,
) -> MASACExperimentConfig:
    """Combine current multi-agent task rules with checkpoint-matched control."""
    single = SINGLE_AGENT_CONFIG
    return replace(
        MASACExperimentConfig(),
        num_agents=int(num_agents),
        max_steps=int(max_steps),
        velocity_clip=tuple(single.velocity_clip),
        accelerate_clip=tuple(single.accelerate_clip),
        time_step=float(single.time_step),
        sensing_radius=float(single.sensing_radius),
        sensor_azimuth_bins=int(single.sensor_azimuth_bins),
        sensor_elevation_bins=int(single.sensor_elevation_bins),
        sensor_elevation_range_deg=(-80.0, 80.0),
        sensor_include_previous_scan=True,
        sensor_goal_distance_clip=2.0 * float(single.sensing_radius),
        k_alpha=float(single.k_alpha),
        k_beta=float(single.k_beta),
        alpha_s=float(single.alpha_s),
        dmp_tau=float(single.tau),
        forcing_term_min=float(single.forcing_term_min),
        forcing_term_max=float(single.forcing_term_max),
        forcing_gate_kappa=1.0,
        goal_offset_max=float(single.goal_offset_max),
        phase_mode="classic",
        phase_integrator="legacy_euler",
        phase_min=0.0,
        action_guidance_enabled=False,
    )


class SinglePolicyMultiAgentEnv(MultiAgentDMPEnv):
    """Multi-agent environment with an evaluation-only peer LiDAR adapter."""

    def __init__(
        self,
        *args: Any,
        observation_mode: str = "blind",
        peer_radius: float = 0.3,
        include_boundaries_in_sensor: bool = True,
        terminate_on_boundary_collision: bool = True,
        **kwargs: Any,
    ) -> None:
        if observation_mode not in OBSERVATION_MODES:
            raise ValueError(f"unsupported observation mode: {observation_mode}")
        if float(peer_radius) <= 0.0:
            raise ValueError("peer_radius must be positive")
        self.single_policy_observation_mode = str(observation_mode)
        self.peer_radius = float(peer_radius)
        self.include_boundaries_in_sensor = bool(include_boundaries_in_sensor)
        self.terminate_on_boundary_collision = bool(terminate_on_boundary_collision)
        super().__init__(*args, **kwargs)

    def _sensor_static_obstacles(self) -> list[Any]:
        if self.include_boundaries_in_sensor:
            return super()._sensor_static_obstacles()
        return list(self.static_obstacles)

    def _compute_boundary_collision_mask(self) -> np.ndarray:
        if self.terminate_on_boundary_collision:
            return super()._compute_boundary_collision_mask()
        return np.zeros(self.num_agents, dtype=bool)

    def _sensor_dynamic_obstacles(self, agent_index: int) -> list[Any]:
        obstacles = list(super()._sensor_dynamic_obstacles(agent_index))
        if self.single_policy_observation_mode == "blind":
            return obstacles
        for peer_index, dynamic in enumerate(self.dynamics):
            if peer_index == int(agent_index):
                continue
            obstacles.append(
                _AgentAsDynamicObstacle(
                    center=dynamic.p.copy(),
                    velocity=dynamic.v.copy(),
                    radius=self.peer_radius,
                    safety_margin=0.0,
                )
            )
        return obstacles


def build_policy_observations(
    env: MultiAgentDMPEnv,
    *,
    expected_observation_dim: int | None = None,
) -> np.ndarray:
    """Reconstruct the local SAC-DMP observation under the active ray contract."""
    rows: list[np.ndarray] = []
    for agent_index in range(env.num_agents):
        packet = env.latest_sensor_packets[agent_index]
        if packet is None:
            raise RuntimeError("environment must be reset before building policy observations")
        dmp = env.dmps[agent_index]
        extra = np.asarray(
            [dmp.phase, dmp.config.K_alpha, dmp.config.K_beta],
            dtype=np.float32,
        )
        rows.append(
            np.concatenate(
                [packet.observation.astype(np.float32, copy=False), extra],
                axis=0,
            )
        )
    observations = np.stack(rows, axis=0).astype(np.float32)
    if expected_observation_dim is not None and observations.shape[1] != int(expected_observation_dim):
        raise ValueError(
            f"policy requires {expected_observation_dim} features, got {observations.shape[1]}"
        )
    return observations


def _build_environment(
    config: MASACExperimentConfig,
    *,
    observation_mode: str,
    peer_radius: float,
    training_distribution: bool,
    include_boundaries_in_sensor: bool = True,
    terminate_on_boundary_collision: bool = True,
) -> SinglePolicyMultiAgentEnv:
    if training_distribution:
        kwargs = build_stage_env_kwargs(config, _phase3_level3_stage(config))
    else:
        kwargs = config.build_core_env_kwargs()
    return SinglePolicyMultiAgentEnv(
        **copy.deepcopy(kwargs),
        observation_mode=observation_mode,
        peer_radius=peer_radius,
        include_boundaries_in_sensor=include_boundaries_in_sensor,
        terminate_on_boundary_collision=terminate_on_boundary_collision,
    )


def _scenario_options(
    config: MASACExperimentConfig,
    *,
    suite: str,
    scenario_name: str,
    seed: int,
) -> tuple[dict[str, Any] | None, str]:
    if suite == "canonical":
        return build_canonical_scenario(config, scenario_name), CANONICAL_LABELS[scenario_name]
    scenario = build_distribution_scenario(scenario_name, config, seed=seed)
    if scenario_name == TRAINING_DISTRIBUTION_SCENARIO:
        return None, DISTRIBUTION_LABELS[scenario_name]
    return {
        "starts": np.asarray(scenario["starts"], dtype=float),
        "goals": np.asarray(scenario["goals"], dtype=float),
        "static_obstacles": copy.deepcopy(scenario["static_obstacles"]),
        "dynamic_obstacles": copy.deepcopy(scenario["dynamic_obstacles"]),
    }, DISTRIBUTION_LABELS[scenario_name]


def _safe_mean(values: list[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else 0.0


def run_episode(
    *,
    model: Any,
    config: MASACExperimentConfig,
    suite: str,
    scenario_name: str,
    seed: int,
    episode_index: int,
    observation_mode: str,
    peer_radius: float,
    scenario_options_override: dict[str, Any] | None = None,
    scenario_label_override: str | None = None,
    include_boundaries_in_sensor: bool = True,
    terminate_on_boundary_collision: bool = True,
) -> dict[str, Any]:
    training_distribution = (
        suite == "distribution" and scenario_name == TRAINING_DISTRIBUTION_SCENARIO
    )
    if scenario_options_override is None:
        options, label = _scenario_options(
            config,
            suite=suite,
            scenario_name=scenario_name,
            seed=seed,
        )
    else:
        options = copy.deepcopy(scenario_options_override)
        label = str(scenario_label_override or scenario_name)
    env = _build_environment(
        config,
        observation_mode=observation_mode,
        peer_radius=peer_radius,
        training_distribution=training_distribution,
        include_boundaries_in_sensor=include_boundaries_in_sensor,
        terminate_on_boundary_collision=terminate_on_boundary_collision,
    )
    try:
        _, info = env.reset(seed=int(seed), options=options)
        positions = env._positions()
        path_lengths = np.zeros(env.num_agents, dtype=float)
        ever_success = np.zeros(env.num_agents, dtype=bool)
        min_pairwise_distance = float(info["min_inter_agent_distance"])
        min_boundary_distance = float(np.min(info["min_boundary_distances"]))
        min_sensor_clearance = float("inf")
        total_reward = 0.0
        inference_times_ms: list[float] = []
        forcing_norms: list[float] = []
        offset_norms: list[float] = []
        forcing_axis_saturated = 0
        offset_axis_saturated = 0
        action_agent_count = 0
        acceleration_clipped = 0
        velocity_clipped = 0
        dynamics_axis_count = 0
        collision = False
        obstacle_collision = False
        inter_agent_collision = False
        boundary_collision = False
        terminated = False
        truncated = False

        while not (terminated or truncated):
            observations = build_policy_observations(env)
            active_mask = np.logical_not(env.success_rewarded_mask.copy())
            inference_start = time.perf_counter_ns()
            actions, _ = model.predict(observations, deterministic=True)
            inference_times_ms.append(
                float(time.perf_counter_ns() - inference_start) / 1_000_000.0
            )
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape != env.action_shape:
                raise ValueError(f"policy action shape {actions.shape} != {env.action_shape}")

            active_actions = actions[active_mask]
            if active_actions.size:
                forcing = active_actions[:, :3]
                offsets = active_actions[:, 3:]
                forcing_norms.extend(np.linalg.norm(forcing, axis=1).tolist())
                offset_norms.extend(np.linalg.norm(offsets, axis=1).tolist())
                forcing_axis_saturated += int(
                    np.sum(np.abs(forcing) >= 0.95 * float(config.forcing_term_max))
                )
                offset_axis_saturated += int(
                    np.sum(np.abs(offsets) >= 0.95 * float(config.goal_offset_max))
                )
                action_agent_count += int(active_actions.shape[0])

            previous_positions = positions.copy()
            _, rewards, terminated, truncated, info = env.step(actions)
            positions = env._positions()
            path_lengths += np.linalg.norm(positions - previous_positions, axis=1)
            total_reward += float(np.sum(rewards))
            ever_success |= np.asarray(info["success_mask"], dtype=bool)
            min_pairwise_distance = min(
                min_pairwise_distance,
                float(info["min_inter_agent_distance"]),
            )
            min_boundary_distance = min(
                min_boundary_distance,
                float(np.min(info["min_boundary_distances"])),
            )
            min_sensor_clearance = min(
                min_sensor_clearance,
                float(np.min(info["min_clearances"])),
            )
            collision |= bool(info["collision"])
            obstacle_collision |= bool(np.any(info["obstacle_collision_mask"]))
            inter_agent_collision |= bool(np.any(info["inter_agent_collision_mask"]))
            boundary_collision |= bool(np.any(info["boundary_collision_mask"]))
            acceleration_clipped += int(np.sum(info["acceleration_clip_mask"][active_mask]))
            velocity_clipped += int(np.sum(info["velocity_clip_mask"][active_mask]))
            dynamics_axis_count += int(np.sum(active_mask)) * 3

        team_success = bool(info.get("success", False))
        status = "success" if team_success else "collision" if collision else "timeout"
        starts = np.asarray(env.starts, dtype=float)
        goals = np.asarray(env.goals, dtype=float)
        direct_lengths = np.linalg.norm(goals - starts, axis=1)
        reached_efficiency = np.divide(
            direct_lengths,
            np.maximum(path_lengths, 1.0e-9),
        )
        reached_efficiency[~ever_success] = np.nan
        forcing_denominator = max(1, action_agent_count * 3)
        offset_denominator = max(1, action_agent_count * 3)
        dynamics_denominator = max(1, dynamics_axis_count)
        return {
            "mode": observation_mode,
            "mode_label": MODE_LABELS[observation_mode],
            "suite": suite,
            "scenario": scenario_name,
            "scenario_label": label,
            "seed": int(seed),
            "episode": int(episode_index),
            "status": status,
            "team_success": team_success,
            "agent_success_rate": float(np.mean(ever_success)),
            "reached_agent_count": int(np.sum(ever_success)),
            "collision": collision,
            "inter_agent_collision": inter_agent_collision,
            "obstacle_collision": obstacle_collision,
            "boundary_collision": boundary_collision,
            "timeout": bool(status == "timeout"),
            "steps": int(env.steps),
            "flight_time": float(env.steps * config.time_step),
            "total_reward": total_reward,
            "path_length_mean": float(np.mean(path_lengths)),
            "path_length_team": float(np.sum(path_lengths)),
            "path_lengths": path_lengths.tolist(),
            "path_efficiency_reached_mean": (
                float(np.nanmean(reached_efficiency)) if np.any(ever_success) else 0.0
            ),
            "min_pairwise_distance": min_pairwise_distance,
            "min_pairwise_clearance": (
                min_pairwise_distance - float(config.inter_agent_safe_distance)
            ),
            "min_boundary_distance": min_boundary_distance,
            "min_sensor_clearance": min_sensor_clearance,
            "final_goal_distance_mean": float(
                np.mean(np.linalg.norm(goals - positions, axis=1))
            ),
            "forcing_norm_mean": _safe_mean(forcing_norms),
            "offset_norm_mean": _safe_mean(offset_norms),
            "forcing_axis_saturation_rate": forcing_axis_saturated / forcing_denominator,
            "offset_axis_saturation_rate": offset_axis_saturated / offset_denominator,
            "acceleration_axis_clip_rate": acceleration_clipped / dynamics_denominator,
            "velocity_axis_clip_rate": velocity_clipped / dynamics_denominator,
            "inference_time_mean_ms": _safe_mean(inference_times_ms),
        }
    finally:
        env.close()


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = (
        "agent_success_rate",
        "path_length_mean",
        "path_length_team",
        "path_efficiency_reached_mean",
        "min_pairwise_distance",
        "min_pairwise_clearance",
        "min_boundary_distance",
        "min_sensor_clearance",
        "final_goal_distance_mean",
        "forcing_norm_mean",
        "offset_norm_mean",
        "forcing_axis_saturation_rate",
        "offset_axis_saturation_rate",
        "acceleration_axis_clip_rate",
        "velocity_axis_clip_rate",
        "inference_time_mean_ms",
        "steps",
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["mode"]), str(row["suite"]), str(row["scenario"]))
        grouped.setdefault(key, []).append(row)
    aggregates: list[dict[str, Any]] = []
    for (mode, suite, scenario), members in grouped.items():
        total = len(members)
        success_count = sum(bool(row["team_success"]) for row in members)
        ci_low, ci_high = _wilson_interval(success_count, total)
        result: dict[str, Any] = {
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "suite": suite,
            "scenario": scenario,
            "scenario_label": members[0]["scenario_label"],
            "episodes": total,
            "success_count": success_count,
            "team_success_rate": success_count / total,
            "team_success_ci95_low": ci_low,
            "team_success_ci95_high": ci_high,
            "collision_rate": float(np.mean([row["collision"] for row in members])),
            "inter_agent_collision_rate": float(
                np.mean([row["inter_agent_collision"] for row in members])
            ),
            "obstacle_collision_rate": float(
                np.mean([row["obstacle_collision"] for row in members])
            ),
            "boundary_collision_rate": float(
                np.mean([row["boundary_collision"] for row in members])
            ),
            "timeout_rate": float(np.mean([row["timeout"] for row in members])),
        }
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in members], dtype=float)
            finite_values = values[np.isfinite(values)]
            result[f"{key}_mean"] = _safe_mean(values.tolist())
            result[f"{key}_std"] = (
                float(np.std(finite_values, ddof=1)) if len(finite_values) > 1 else 0.0
            )
        aggregates.append(result)
    return aggregates


def build_comparison_rows(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["suite"], row["scenario"], row["mode"]): row
        for row in aggregates
    }
    comparisons: list[dict[str, Any]] = []
    scenario_keys = sorted({(row["suite"], row["scenario"]) for row in aggregates})
    for suite, scenario in scenario_keys:
        blind = lookup.get((suite, scenario, "blind"))
        peer = lookup.get((suite, scenario, "peer_spheres"))
        if blind is None or peer is None:
            continue
        comparisons.append(
            {
                "suite": suite,
                "scenario": scenario,
                "scenario_label": blind["scenario_label"],
                "episodes_per_mode": blind["episodes"],
                "blind_team_success_rate": blind["team_success_rate"],
                "peer_team_success_rate": peer["team_success_rate"],
                "success_rate_delta_peer_minus_blind": (
                    peer["team_success_rate"] - blind["team_success_rate"]
                ),
                "blind_agent_success_rate": blind["agent_success_rate_mean"],
                "peer_agent_success_rate": peer["agent_success_rate_mean"],
                "agent_success_rate_delta_peer_minus_blind": (
                    peer["agent_success_rate_mean"] - blind["agent_success_rate_mean"]
                ),
                "blind_inter_agent_collision_rate": blind["inter_agent_collision_rate"],
                "peer_inter_agent_collision_rate": peer["inter_agent_collision_rate"],
                "inter_agent_collision_delta_peer_minus_blind": (
                    peer["inter_agent_collision_rate"] - blind["inter_agent_collision_rate"]
                ),
                "blind_path_length_mean": blind["path_length_mean_mean"],
                "peer_path_length_mean": peer["path_length_mean_mean"],
                "path_length_delta_peer_minus_blind": (
                    peer["path_length_mean_mean"] - blind["path_length_mean_mean"]
                ),
                "blind_min_pairwise_distance": blind["min_pairwise_distance_mean"],
                "peer_min_pairwise_distance": peer["min_pairwise_distance_mean"],
            }
        )
    return comparisons


def _mode_totals(rows: list[dict[str, Any]], mode: str, suite: str) -> dict[str, float]:
    selected = [row for row in rows if row["mode"] == mode and row["suite"] == suite]
    if not selected:
        return {}
    return {
        "episodes": float(len(selected)),
        "team_success_rate": float(np.mean([row["team_success"] for row in selected])),
        "agent_success_rate": float(np.mean([row["agent_success_rate"] for row in selected])),
        "inter_agent_collision_rate": float(
            np.mean([row["inter_agent_collision"] for row in selected])
        ),
        "obstacle_collision_rate": float(
            np.mean([row["obstacle_collision"] for row in selected])
        ),
        "boundary_collision_rate": float(
            np.mean([row["boundary_collision"] for row in selected])
        ),
        "timeout_rate": float(np.mean([row["timeout"] for row in selected])),
        "path_length_mean": float(np.mean([row["path_length_mean"] for row in selected])),
        "min_pairwise_distance": float(
            np.mean([row["min_pairwise_distance"] for row in selected])
        ),
    }


def write_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    checkpoint: Path,
    peer_radius: float,
) -> None:
    lines = [
        "# 冻结单机策略在多机环境中的无协调基线",
        "",
        f"- Checkpoint：`{checkpoint}`",
        "- 策略执行：所有无人机共享同一冻结 SAC Actor，deterministic 推理",
        "- 邻机不可见：策略不接收任何邻机信息",
        f"- 邻机动态球观测：其他无人机以半径 `{peer_radius:.3f} m` 的动态球进入 LiDAR",
        "- 两种模式均不使用通信、身份、优先级、上层调度或显式 inter-agent observation",
        "",
        "## 分布场景总体结果",
        "",
        "| 模式 | 回合 | 团队成功率 | 单机成功率 | 机间碰撞率 | 障碍碰撞率 | 边界碰撞率 | 超时率 | 平均路径长度/m | 平均最小机距/m |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in OBSERVATION_MODES:
        total = _mode_totals(rows, mode, "distribution")
        if not total:
            continue
        lines.append(
            "| {label} | {episodes:.0f} | {team:.2%} | {agent:.2%} | {inter:.2%} | "
            "{obstacle:.2%} | {boundary:.2%} | {timeout:.2%} | {path:.3f} | {distance:.3f} |".format(
                label=MODE_LABELS[mode],
                episodes=total["episodes"],
                team=total["team_success_rate"],
                agent=total["agent_success_rate"],
                inter=total["inter_agent_collision_rate"],
                obstacle=total["obstacle_collision_rate"],
                boundary=total["boundary_collision_rate"],
                timeout=total["timeout_rate"],
                path=total["path_length_mean"],
                distance=total["min_pairwise_distance"],
            )
        )
    lines.extend(
        [
            "",
            "## 分场景结果",
            "",
            "| 场景组 | 场景 | 模式 | 回合 | 团队成功率 | 单机成功率 | 机间碰撞率 | 超时率 | 平均路径长度/m |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregates:
        lines.append(
            "| {suite} | {scenario} | {mode} | {episodes} | {success:.2%} | {agent:.2%} | "
            "{collision:.2%} | {timeout:.2%} | {path:.3f} |".format(
                suite=row["suite"],
                scenario=row["scenario_label"],
                mode=row["mode_label"],
                episodes=row["episodes"],
                success=row["team_success_rate"],
                agent=row["agent_success_rate_mean"],
                collision=row["inter_agent_collision_rate"],
                timeout=row["timeout_rate"],
                path=row["path_length_mean_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "邻机动态球模式只改变传感器障碍物集合，不改变真实动力学、碰撞阈值、场景障碍物或终止条件。"
            "因此，该模式属于局部反应式避碰，而不是多机协调策略。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_settings(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"evaluation config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--distribution-episodes", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--modes", nargs="+", choices=OBSERVATION_MODES, default=None)
    parser.add_argument("--canonical-only", action="store_true")
    return parser.parse_args()


def main() -> Path:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    settings = _load_settings(config_path)
    checkpoint = _resolve_path(args.checkpoint or settings["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    modes = tuple(args.modes or settings.get("modes", OBSERVATION_MODES))
    episodes_per_distribution = int(
        args.distribution_episodes
        if args.distribution_episodes is not None
        else settings.get("distribution_episodes", 50)
    )
    if episodes_per_distribution <= 0:
        raise ValueError("distribution_episodes must be positive")
    seed_base = int(args.seed_base or settings.get("seed_base", 202608040))
    peer_radius = float(settings.get("peer_radius", 0.3))
    num_agents = int(settings.get("num_agents", 3))
    max_steps = int(settings.get("max_steps", 200))
    canonical_scenarios = tuple(settings.get("canonical_scenarios", CANONICAL_SCENARIOS))
    distribution_scenarios = tuple(
        settings.get("distribution_scenarios", DISTRIBUTION_SCENARIOS)
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT / "artifacts" / f"single_policy_multi_agent_no_coordination_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    aligned_config = build_aligned_config(num_agents=num_agents, max_steps=max_steps)
    reference_env = build_single_env(
        config=SINGLE_AGENT_CONFIG,
        action_guidance_enabled=False,
    )
    model = build_single_model(reference_env, config=SINGLE_AGENT_CONFIG, verbose=0)
    load_checkpoint(model, checkpoint)
    model.actor.train(False)
    if int(reference_env.observation_space.shape[0]) != 122:
        raise ValueError("historical single-agent environment no longer exposes 122 features")

    snapshot = {
        "created_at": datetime.now().astimezone().isoformat(),
        "evaluation_config_path": config_path,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "modes": modes,
        "peer_radius": peer_radius,
        "peer_sphere_safety_margin": 0.0,
        "num_agents": num_agents,
        "max_steps": max_steps,
        "distribution_episodes": episodes_per_distribution,
        "seed_base": seed_base,
        "canonical_scenarios": canonical_scenarios,
        "distribution_scenarios": distribution_scenarios,
        "single_policy_observation_dim": 122,
        "single_policy_sensor_dim": 119,
        "policy_extra_fields": ["phase", "K_alpha", "K_beta"],
        "excluded_policy_fields": [
            "progress",
            "stagnation_counter",
            "inter_agent_observation",
            "agent_identity",
            "communication",
            "coordination_priority",
        ],
        "aligned_multi_agent_config": asdict(aligned_config),
        "historical_single_agent_config": asdict(SINGLE_AGENT_CONFIG),
    }
    _write_json(output_dir / "config_snapshot.json", snapshot)

    rows: list[dict[str, Any]] = []
    total_planned = len(modes) * len(canonical_scenarios)
    if not args.canonical_only:
        total_planned += len(modes) * len(distribution_scenarios) * episodes_per_distribution
    completed = 0
    for mode in modes:
        for scenario_index, scenario_name in enumerate(canonical_scenarios):
            seed = seed_base + 900_000 + scenario_index
            rows.append(
                run_episode(
                    model=model,
                    config=aligned_config,
                    suite="canonical",
                    scenario_name=scenario_name,
                    seed=seed,
                    episode_index=0,
                    observation_mode=mode,
                    peer_radius=peer_radius,
                )
            )
            completed += 1
            print(
                f"[{completed}/{total_planned}] {mode} canonical/{scenario_name}: "
                f"{rows[-1]['status']}"
            )
        if args.canonical_only:
            continue
        for scenario_offset, scenario_name in enumerate(distribution_scenarios):
            scenario_seed_base = seed_base + scenario_offset * 10_000
            for episode_index in range(episodes_per_distribution):
                seed = scenario_seed_base + episode_index
                rows.append(
                    run_episode(
                        model=model,
                        config=aligned_config,
                        suite="distribution",
                        scenario_name=scenario_name,
                        seed=seed,
                        episode_index=episode_index,
                        observation_mode=mode,
                        peer_radius=peer_radius,
                    )
                )
                completed += 1
                if completed % 10 == 0 or completed == total_planned:
                    print(f"[{completed}/{total_planned}] {mode} distribution/{scenario_name}")

    reference_env.close()
    aggregates = aggregate_rows(rows)
    comparisons = build_comparison_rows(aggregates)
    _write_csv(output_dir / "episodes.csv", rows)
    _write_csv(output_dir / "scenario_summary.csv", aggregates)
    _write_csv(output_dir / "mode_comparison.csv", comparisons)
    _write_json(output_dir / "summary.json", aggregates)
    write_report(
        output_dir / "report.md",
        rows=rows,
        aggregates=aggregates,
        checkpoint=checkpoint,
        peer_radius=peer_radius,
    )
    print(f"Artifacts written to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
