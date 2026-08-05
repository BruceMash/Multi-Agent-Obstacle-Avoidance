"""Evaluate real Guidance waypoints with a fully frozen single-agent policy.

Guidance is restricted to selecting the active goal. The action returned by
``model.predict`` is passed directly to the existing environment/DMP step.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
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

from Guidance.reference_point_proposal_demo import (  # noqa: E402
    Proposal,
    ProposalConfig,
    propose_reference_points,
)
from experiment_config import EXPERIMENT_CONFIG as SINGLE_AGENT_CONFIG  # noqa: E402
from runner_sac import build_env as build_single_env  # noqa: E402
from runner_sac import build_model as build_single_model  # noqa: E402
from runner_sac import load_checkpoint  # noqa: E402
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    STAGE_SPECS,
    build_single_distribution_multi_config,
    build_stage_scenario,
)
from scripts.evaluate_single_policy_guidance_multi_agent import (  # noqa: E402
    CONTROLLER_LABELS as LEGACY_CONTROLLER_LABELS,
    reference_point_clearance,
)
from scripts.evaluate_single_policy_multi_agent import (  # noqa: E402
    _build_environment,
    _safe_mean,
    _sha256,
    _write_csv,
    _write_json,
    aggregate_rows,
    build_policy_observations,
    run_episode as run_baseline_episode,
)


DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "evaluation" / "frozen_policy_waypoint_guidance.json"
)
CONTROLLER_LABELS = {
    "baseline": LEGACY_CONTROLLER_LABELS["baseline"],
    "w1": "冻结策略 + Guidance W1",
    "w2": "冻结策略 + Guidance W2",
    "w3": "冻结策略 + Guidance W3",
}
WAYPOINT_METRIC_KEYS = (
    "waypoint_request_count",
    "waypoint_generated_count",
    "waypoint_dead_end_count",
    "waypoint_invalid_replan_count",
    "waypoint_reached_replan_count",
    "waypoint_stagnation_replan_count",
    "waypoint_boundary_rejection_count",
    "waypoint_obstacle_rejection_count",
    "waypoint_segment_fallback_count",
    "waypoint_sequence_length_mean",
    "waypoint_selected_depth_mean",
    "waypoint_active_distance_mean",
    "waypoint_proposal_time_mean_ms",
    "waypoint_fallback_agent_step_rate",
    "waypoint_replans_per_step",
)


@dataclass
class ActiveWaypointState:
    point: np.ndarray | None = None
    selected_depth: int = 0
    previous_task_distance: float | None = None
    stagnation_steps: int = 0


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-9:
        return np.zeros_like(vector)
    return vector / norm


def point_inside_guidance_bounds(
    point: np.ndarray,
    workspace_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    margin: float,
) -> bool:
    lower, upper = np.asarray(workspace_bounds, dtype=float)
    point = np.asarray(point, dtype=float)
    return bool(np.all(point >= lower + float(margin)) and np.all(point <= upper - float(margin)))


def segment_is_clear(
    env: Any,
    agent_index: int,
    start: np.ndarray,
    end: np.ndarray,
    *,
    boundary_margin: float,
    collision_clearance: float,
    samples: int,
) -> bool:
    """Check only whether a candidate active-goal segment is selectable."""
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if not point_inside_guidance_bounds(end, env.env_config.workspace_bounds, boundary_margin):
        return False
    obstacles = list(env._sensor_static_obstacles())
    obstacles.extend(env._sensor_dynamic_obstacles(int(agent_index)))
    for ratio in np.linspace(0.0, 1.0, int(samples) + 1, dtype=float)[1:]:
        point = (1.0 - ratio) * np.asarray(start, dtype=float) + ratio * np.asarray(end, dtype=float)
        if not point_inside_guidance_bounds(point, env.env_config.workspace_bounds, boundary_margin):
            return False
        if any(float(obstacle.signed_distance(point)) <= float(collision_clearance) for obstacle in obstacles):
            return False
    return True


def set_dmp_active_goal_preserve_phase(dmp: Any, active_goal: np.ndarray) -> None:
    """Update only the DMP goal; never reset phase or controller state."""
    previous_phase = float(dmp.phase)
    dmp.goal = np.asarray(active_goal, dtype=float).copy()
    if float(dmp.phase) != previous_phase:
        raise RuntimeError("updating an active waypoint must preserve DMP phase")


def predict_actions_without_postprocessing(
    model: Any,
    observations: np.ndarray,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Return the deterministic network action without blending or correction."""
    actions, _ = model.predict(observations, deterministic=True)
    actions = np.asarray(actions)
    if actions.shape != expected_shape:
        raise ValueError(f"policy action shape {actions.shape} != {expected_shape}")
    return actions


def build_active_goal_observations(
    env: Any,
    active_goals: np.ndarray,
) -> np.ndarray:
    observations = build_policy_observations(env).copy()
    positions = np.asarray(env._positions(), dtype=float)
    for agent_index in range(int(env.num_agents)):
        delta = np.asarray(active_goals[agent_index], dtype=float) - positions[agent_index]
        distance = float(np.linalg.norm(delta))
        observations[agent_index, 3:6] = normalize(delta).astype(np.float32)
        observations[agent_index, 6] = np.float32(
            np.clip(
                distance / float(env.sensors[agent_index].goal_distance_clip),
                0.0,
                1.0,
            )
        )
    return observations


def _proposal_is_selectable(
    env: Any,
    agent_index: int,
    proposal: Proposal,
    *,
    boundary_margin: float,
    collision_clearance: float,
) -> tuple[bool, str | None]:
    if not point_inside_guidance_bounds(
        proposal.point,
        env.env_config.workspace_bounds,
        boundary_margin,
    ):
        return False, "boundary"
    if reference_point_clearance(env, agent_index, proposal.point) <= float(collision_clearance):
        return False, "obstacle"
    return True, None


def generate_waypoint_sequence(
    env: Any,
    agent_index: int,
    task_goal: np.ndarray,
    *,
    depth: int,
    proposal_config: ProposalConfig,
    boundary_margin: float,
    collision_clearance: float,
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Virtually roll out a frozen-obstacle Guidance prefix."""
    if depth <= 0:
        raise ValueError("depth must be positive")
    sensor = copy.deepcopy(env.sensors[agent_index])
    packet = copy.deepcopy(env.latest_sensor_packets[agent_index])
    virtual_position = np.asarray(env.dynamics[agent_index].p, dtype=float).copy()
    virtual_velocity = np.asarray(env.dynamics[agent_index].v, dtype=float).copy()
    static_obstacles = copy.deepcopy(env._sensor_static_obstacles())
    dynamic_obstacles = copy.deepcopy(env._sensor_dynamic_obstacles(agent_index))
    sequence: list[np.ndarray] = []
    counts = {"boundary": 0, "obstacle": 0}
    for _ in range(int(depth)):
        proposals = propose_reference_points(
            virtual_position,
            np.asarray(task_goal, dtype=float),
            virtual_velocity,
            packet,
            sensor,
            proposal_config,
            float(env.env_config.goal_tolerance),
        )
        selected: Proposal | None = None
        for proposal in proposals:
            selectable, reason = _proposal_is_selectable(
                env,
                agent_index,
                proposal,
                boundary_margin=boundary_margin,
                collision_clearance=collision_clearance,
            )
            if selectable:
                selected = proposal
                break
            if reason is not None:
                counts[reason] += 1
        if selected is None:
            break
        sequence.append(np.asarray(selected.point, dtype=float).copy())
        virtual_position = np.asarray(selected.point, dtype=float).copy()
        virtual_velocity = normalize(selected.direction) * float(proposal_config.nominal_speed)
        packet = sensor.sense(
            virtual_position,
            virtual_velocity,
            np.asarray(task_goal, dtype=float),
            static_obstacles,
            dynamic_obstacles,
        )
    return sequence, counts


def select_safe_lookahead(
    env: Any,
    agent_index: int,
    sequence: list[np.ndarray],
    *,
    requested_depth: int,
    boundary_margin: float,
    collision_clearance: float,
    segment_samples: int,
) -> tuple[np.ndarray | None, int, int]:
    """Use the deepest point whose direct segment remains collision-free."""
    start = np.asarray(env.dynamics[agent_index].p, dtype=float)
    fallback_count = 0
    maximum_depth = min(int(requested_depth), len(sequence))
    for depth in range(maximum_depth, 0, -1):
        point = np.asarray(sequence[depth - 1], dtype=float)
        if segment_is_clear(
            env,
            agent_index,
            start,
            point,
            boundary_margin=boundary_margin,
            collision_clearance=collision_clearance,
            samples=segment_samples,
        ):
            return point.copy(), depth, fallback_count
        fallback_count += 1
    return None, 0, fallback_count


def active_waypoint_invalid(
    env: Any,
    agent_index: int,
    point: np.ndarray,
    *,
    boundary_margin: float,
    collision_clearance: float,
    segment_samples: int,
) -> bool:
    return not segment_is_clear(
        env,
        agent_index,
        env.dynamics[agent_index].p,
        point,
        boundary_margin=boundary_margin,
        collision_clearance=collision_clearance,
        samples=segment_samples,
    )


def update_stagnation(
    state: ActiveWaypointState,
    task_distance: float,
    *,
    progress_epsilon: float,
) -> None:
    if state.previous_task_distance is None:
        state.stagnation_steps = 0
    elif state.previous_task_distance - float(task_distance) < float(progress_epsilon):
        state.stagnation_steps += 1
    else:
        state.stagnation_steps = 0
    state.previous_task_distance = float(task_distance)


def _zero_waypoint_metrics(row: dict[str, Any]) -> None:
    for key in WAYPOINT_METRIC_KEYS:
        row[key] = 0.0


def run_waypoint_episode(
    *,
    model: Any,
    config: Any,
    scenario_name: str,
    scenario_label: str,
    seed: int,
    episode_index: int,
    peer_radius: float,
    scenario_options: dict[str, Any],
    requested_depth: int,
    proposal_config: ProposalConfig,
    reached_tolerance: float,
    boundary_margin: float,
    collision_clearance: float,
    segment_samples: int,
    stagnation_steps: int,
    stagnation_progress_epsilon: float,
) -> dict[str, Any]:
    env = _build_environment(
        config,
        observation_mode="peer_spheres",
        peer_radius=peer_radius,
        training_distribution=False,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )
    try:
        _, info = env.reset(seed=int(seed), options=copy.deepcopy(scenario_options))
        task_goals = np.asarray(env.goals, dtype=float).copy()
        states = [ActiveWaypointState() for _ in range(int(env.num_agents))]
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
        proposal_times_ms: list[float] = []
        sequence_lengths: list[float] = []
        selected_depths: list[float] = []
        active_distances: list[float] = []
        forcing_axis_saturated = 0
        offset_axis_saturated = 0
        action_agent_count = 0
        acceleration_clipped = 0
        velocity_clipped = 0
        dynamics_axis_count = 0
        fallback_agent_steps = 0
        active_agent_steps = 0
        counters = {
            "request": 0,
            "generated": 0,
            "dead_end": 0,
            "invalid": 0,
            "reached": 0,
            "stagnation": 0,
            "boundary": 0,
            "obstacle": 0,
            "segment_fallback": 0,
        }
        collision = False
        obstacle_collision = False
        inter_agent_collision = False
        boundary_collision = False
        terminated = False
        truncated = False

        while not (terminated or truncated):
            active_mask = np.logical_not(env.success_rewarded_mask.copy())
            active_goals = task_goals.copy()
            for agent_index in range(int(env.num_agents)):
                if not bool(active_mask[agent_index]):
                    set_dmp_active_goal_preserve_phase(
                        env.dmps[agent_index], task_goals[agent_index]
                    )
                    continue
                active_agent_steps += 1
                state = states[agent_index]
                task_distance = float(
                    np.linalg.norm(task_goals[agent_index] - positions[agent_index])
                )
                update_stagnation(
                    state,
                    task_distance,
                    progress_epsilon=stagnation_progress_epsilon,
                )
                reason: str | None = None
                if state.point is None:
                    reason = "initial"
                elif active_waypoint_invalid(
                    env,
                    agent_index,
                    state.point,
                    boundary_margin=boundary_margin,
                    collision_clearance=collision_clearance,
                    segment_samples=segment_samples,
                ):
                    reason = "invalid"
                elif float(np.linalg.norm(state.point - positions[agent_index])) <= reached_tolerance:
                    reason = "reached"
                elif state.stagnation_steps >= int(stagnation_steps):
                    reason = "stagnation"

                if reason is not None:
                    counters["request"] += 1
                    if reason in counters:
                        counters[reason] += 1
                    proposal_start = time.perf_counter_ns()
                    sequence, rejected = generate_waypoint_sequence(
                        env,
                        agent_index,
                        task_goals[agent_index],
                        depth=requested_depth,
                        proposal_config=proposal_config,
                        boundary_margin=boundary_margin,
                        collision_clearance=collision_clearance,
                    )
                    proposal_times_ms.append(
                        float(time.perf_counter_ns() - proposal_start) / 1_000_000.0
                    )
                    sequence_lengths.append(float(len(sequence)))
                    counters["boundary"] += int(rejected["boundary"])
                    counters["obstacle"] += int(rejected["obstacle"])
                    point, selected_depth, segment_fallbacks = select_safe_lookahead(
                        env,
                        agent_index,
                        sequence,
                        requested_depth=requested_depth,
                        boundary_margin=boundary_margin,
                        collision_clearance=collision_clearance,
                        segment_samples=segment_samples,
                    )
                    counters["segment_fallback"] += int(segment_fallbacks)
                    if point is None:
                        counters["dead_end"] += 1
                        state.point = None
                        state.selected_depth = 0
                    else:
                        counters["generated"] += 1
                        state.point = point.copy()
                        state.selected_depth = int(selected_depth)
                        state.stagnation_steps = 0
                        selected_depths.append(float(selected_depth))

                if state.point is None:
                    fallback_agent_steps += 1
                    active_goals[agent_index] = task_goals[agent_index].copy()
                else:
                    active_goals[agent_index] = state.point.copy()
                    active_distances.append(
                        float(np.linalg.norm(state.point - positions[agent_index]))
                    )
                set_dmp_active_goal_preserve_phase(
                    env.dmps[agent_index], active_goals[agent_index]
                )

            observations = build_active_goal_observations(env, active_goals)
            inference_start = time.perf_counter_ns()
            actions = predict_actions_without_postprocessing(
                model,
                observations,
                env.action_shape,
            )
            inference_times_ms.append(
                float(time.perf_counter_ns() - inference_start) / 1_000_000.0
            )
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
            # Strict contract: the network action is passed to env.step unchanged.
            _, rewards, terminated, truncated, info = env.step(actions)
            positions = env._positions()
            path_lengths += np.linalg.norm(positions - previous_positions, axis=1)
            total_reward += float(np.sum(rewards))
            ever_success |= np.asarray(info["success_mask"], dtype=bool)
            min_pairwise_distance = min(
                min_pairwise_distance, float(info["min_inter_agent_distance"])
            )
            min_boundary_distance = min(
                min_boundary_distance, float(np.min(info["min_boundary_distances"]))
            )
            min_sensor_clearance = min(
                min_sensor_clearance, float(np.min(info["min_clearances"]))
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
        direct_lengths = np.linalg.norm(task_goals - starts, axis=1)
        reached_efficiency = np.divide(direct_lengths, np.maximum(path_lengths, 1.0e-9))
        reached_efficiency[~ever_success] = np.nan
        forcing_denominator = max(1, action_agent_count * 3)
        offset_denominator = max(1, action_agent_count * 3)
        dynamics_denominator = max(1, dynamics_axis_count)
        return {
            "mode": "peer_spheres",
            "mode_label": "邻机动态球观测",
            "suite": "frozen_policy_waypoint_guidance",
            "scenario": scenario_name,
            "scenario_label": scenario_label,
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
            "min_pairwise_clearance": min_pairwise_distance - float(config.inter_agent_safe_distance),
            "min_boundary_distance": min_boundary_distance,
            "min_sensor_clearance": min_sensor_clearance,
            "final_goal_distance_mean": float(
                np.mean(np.linalg.norm(task_goals - positions, axis=1))
            ),
            "forcing_norm_mean": _safe_mean(forcing_norms),
            "offset_norm_mean": _safe_mean(offset_norms),
            "forcing_axis_saturation_rate": forcing_axis_saturated / forcing_denominator,
            "offset_axis_saturation_rate": offset_axis_saturated / offset_denominator,
            "acceleration_axis_clip_rate": acceleration_clipped / dynamics_denominator,
            "velocity_axis_clip_rate": velocity_clipped / dynamics_denominator,
            "inference_time_mean_ms": _safe_mean(inference_times_ms),
            "waypoint_request_count": int(counters["request"]),
            "waypoint_generated_count": int(counters["generated"]),
            "waypoint_dead_end_count": int(counters["dead_end"]),
            "waypoint_invalid_replan_count": int(counters["invalid"]),
            "waypoint_reached_replan_count": int(counters["reached"]),
            "waypoint_stagnation_replan_count": int(counters["stagnation"]),
            "waypoint_boundary_rejection_count": int(counters["boundary"]),
            "waypoint_obstacle_rejection_count": int(counters["obstacle"]),
            "waypoint_segment_fallback_count": int(counters["segment_fallback"]),
            "waypoint_sequence_length_mean": _safe_mean(sequence_lengths),
            "waypoint_selected_depth_mean": _safe_mean(selected_depths),
            "waypoint_active_distance_mean": _safe_mean(active_distances),
            "waypoint_proposal_time_mean_ms": _safe_mean(proposal_times_ms),
            "waypoint_fallback_agent_step_rate": fallback_agent_steps / max(1, active_agent_steps),
            "waypoint_replans_per_step": counters["request"] / max(1, int(env.steps)),
        }
    finally:
        env.close()


def aggregate_by_controller(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for controller in sorted({str(row["controller"]) for row in rows}):
        members = [row for row in rows if row["controller"] == controller]
        for aggregate in aggregate_rows(members):
            scenario_members = [row for row in members if row["scenario"] == aggregate["scenario"]]
            aggregate["controller"] = controller
            aggregate["controller_label"] = CONTROLLER_LABELS[controller]
            aggregate["boundary_excursion_rate"] = float(
                np.mean([float(row["min_boundary_distance"]) < 0.0 for row in scenario_members])
            )
            for key in WAYPOINT_METRIC_KEYS:
                aggregate[f"{key}_mean"] = _safe_mean(
                    [float(row[key]) for row in scenario_members]
                )
            aggregates.append(aggregate)
    return aggregates


def build_paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["scenario"]), int(row["seed"]), str(row["controller"])): row
        for row in rows
    }
    results: list[dict[str, Any]] = []
    scenarios = sorted({str(row["scenario"]) for row in rows})
    variants = sorted(
        {str(row["controller"]) for row in rows if row["controller"] != "baseline"}
    )
    for scenario in scenarios:
        seeds = sorted({int(row["seed"]) for row in rows if row["scenario"] == scenario})
        for variant in variants:
            pairs = [
                (lookup[(scenario, seed, "baseline")], lookup[(scenario, seed, variant)])
                for seed in seeds
            ]
            results.append(
                {
                    "scenario": scenario,
                    "scenario_label": pairs[0][0]["scenario_label"],
                    "variant": variant,
                    "variant_label": CONTROLLER_LABELS[variant],
                    "paired_episodes": len(pairs),
                    "baseline_team_success_rate": float(np.mean([a["team_success"] for a, _ in pairs])),
                    "variant_team_success_rate": float(np.mean([b["team_success"] for _, b in pairs])),
                    "success_rate_delta": float(np.mean([b["team_success"] for _, b in pairs])) - float(np.mean([a["team_success"] for a, _ in pairs])),
                    "inter_agent_collision_delta": float(np.mean([b["inter_agent_collision"] for _, b in pairs])) - float(np.mean([a["inter_agent_collision"] for a, _ in pairs])),
                    "obstacle_collision_delta": float(np.mean([b["obstacle_collision"] for _, b in pairs])) - float(np.mean([a["obstacle_collision"] for a, _ in pairs])),
                    "timeout_delta": float(np.mean([b["timeout"] for _, b in pairs])) - float(np.mean([a["timeout"] for a, _ in pairs])),
                    "baseline_success_only": sum(bool(a["team_success"]) and not bool(b["team_success"]) for a, b in pairs),
                    "variant_success_only": sum(not bool(a["team_success"]) and bool(b["team_success"]) for a, b in pairs),
                }
            )
    return results


def write_report(
    path: Path,
    *,
    aggregates: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    checkpoint: Path,
) -> None:
    controller_order = {"baseline": 0, "w1": 1, "w2": 2, "w3": 3}
    rows = sorted(
        aggregates,
        key=lambda row: (row["scenario"], controller_order[row["controller"]]),
    )
    lines = [
        "# 冻结单机策略的真实路径点 Guidance 实验",
        "",
        f"- Checkpoint：`{checkpoint}`",
        "- 策略网络 action 原样进入现有 DMP，不执行残差叠加、投影或安全后处理。",
        "- Guidance 仅负责生成和切换真实 active waypoint。",
        "- 最终任务目标单独用于奖励和成功判定，路径点切换不重置 DMP phase。",
        "",
        "## 汇总结果",
        "",
        "| 场景 | 控制方式 | 团队成功率 | 单机到达率 | 机间碰撞率 | 障碍碰撞率 | 超时率 | 越界率 | 路径/m | active距离/m |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {controller} | {team:.2%} | {agent:.2%} | {inter:.2%} | {obstacle:.2%} | {timeout:.2%} | {boundary:.2%} | {path:.3f} | {active:.3f} |".format(
                scenario=row["scenario_label"],
                controller=row["controller_label"],
                team=row["team_success_rate"],
                agent=row["agent_success_rate_mean"],
                inter=row["inter_agent_collision_rate"],
                obstacle=row["obstacle_collision_rate"],
                timeout=row["timeout_rate"],
                boundary=row["boundary_excursion_rate"],
                path=row["path_length_mean_mean"],
                active=row["waypoint_active_distance_mean_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## 相对 Baseline 的配对变化",
            "",
            "| 场景 | 方案 | 成功率变化 | 机间碰撞变化 | 障碍碰撞变化 | 超时变化 | Baseline独有成功 | 新方案独有成功 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {scenario} | {variant} | {success:+.2%} | {inter:+.2%} | {obstacle:+.2%} | {timeout:+.2%} | {base_only} | {variant_only} |".format(
                scenario=row["scenario_label"],
                variant=row["variant_label"],
                success=row["success_rate_delta"],
                inter=row["inter_agent_collision_delta"],
                obstacle=row["obstacle_collision_delta"],
                timeout=row["timeout_delta"],
                base_only=row["baseline_success_only"],
                variant_only=row["variant_success_only"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes-per-stage", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--stages", type=str, default=None)
    return parser.parse_args()


def main() -> Path:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = _resolve_path(args.checkpoint or settings["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    episodes_per_stage = int(
        args.episodes_per_stage
        if args.episodes_per_stage is not None
        else settings.get("episodes_per_stage", 50)
    )
    seed_base = int(args.seed_base or settings.get("seed_base", 202608050))
    peer_radius = float(settings.get("peer_radius", 0.3))
    num_agents = int(settings.get("num_agents", 3))
    max_steps = int(settings.get("max_steps", 50))
    boundary_margin = float(settings.get("boundary_margin", 0.4))
    collision_clearance = float(settings.get("collision_clearance", 0.0))
    segment_samples = int(settings.get("segment_samples", 16))
    reached_tolerance = float(settings.get("reached_tolerance", 0.25))
    stagnation_steps = int(settings.get("stagnation_steps", 10))
    stagnation_progress_epsilon = float(settings.get("stagnation_progress_epsilon", 0.01))
    proposal_config = ProposalConfig(**dict(settings.get("proposal", {})))
    depths = [int(value) for value in settings.get("lookahead_depths", [1, 2, 3])]
    if episodes_per_stage <= 0 or any(depth <= 0 for depth in depths):
        raise ValueError("episode count and lookahead depths must be positive")

    requested_stages = (
        [value.strip() for value in args.stages.split(",") if value.strip()]
        if args.stages
        else list(settings.get("stages", []))
    )
    stage_lookup = {str(stage["name"]): stage for stage in STAGE_SPECS}
    stages = [stage_lookup[name] for name in requested_stages]
    if not stages:
        raise ValueError("at least one stage must be configured")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT / "artifacts" / f"frozen_policy_waypoint_guidance_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    config = build_single_distribution_multi_config(num_agents=num_agents, max_steps=max_steps)
    reference_env = build_single_env(config=SINGLE_AGENT_CONFIG, action_guidance_enabled=False)
    model = build_single_model(reference_env, config=SINGLE_AGENT_CONFIG, verbose=0)
    load_checkpoint(model, checkpoint)
    model.actor.train(False)

    _write_json(
        output_dir / "config_snapshot.json",
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "evaluation_config_path": config_path,
            "checkpoint": checkpoint,
            "checkpoint_sha256": _sha256(checkpoint),
            "episodes_per_stage": episodes_per_stage,
            "seed_base": seed_base,
            "stages": stages,
            "lookahead_depths": depths,
            "frozen_policy_contract": {
                "network_action_passed_to_env_step_unchanged": True,
                "action_residual": False,
                "action_projection": False,
                "forced_braking": False,
                "dmp_phase_reset_on_waypoint_switch": False,
            },
            "boundary_margin": boundary_margin,
            "collision_clearance": collision_clearance,
            "segment_samples": segment_samples,
            "reached_tolerance": reached_tolerance,
            "stagnation_steps": stagnation_steps,
            "stagnation_progress_epsilon": stagnation_progress_epsilon,
            "proposal_config": asdict(proposal_config),
            "aligned_config": asdict(config),
        },
    )

    rows: list[dict[str, Any]] = []
    controllers = ["baseline"] + [f"w{depth}" for depth in depths]
    total = len(stages) * episodes_per_stage * len(controllers)
    completed = 0
    labels = {
        "C_parallel_peer_spheres": "C 平行航路（邻机动态球）",
        "D_permuted_peer_spheres": "D 目标置换（邻机动态球）",
        "E_head_on_narrow_peer_spheres": "E 对向窄通道（邻机动态球）",
    }
    for stage in stages:
        for episode_index in range(episodes_per_stage):
            seed = seed_base + episode_index
            options = build_stage_scenario(config, stage, seed=seed)
            scenario_name = str(stage["name"])
            scenario_label = labels.get(scenario_name, scenario_name)
            baseline = run_baseline_episode(
                model=model,
                config=config,
                suite="frozen_policy_waypoint_guidance",
                scenario_name=scenario_name,
                seed=seed,
                episode_index=episode_index,
                observation_mode="peer_spheres",
                peer_radius=peer_radius,
                scenario_options_override=options,
                scenario_label_override=scenario_label,
                include_boundaries_in_sensor=False,
                terminate_on_boundary_collision=False,
            )
            baseline["controller"] = "baseline"
            baseline["controller_label"] = CONTROLLER_LABELS["baseline"]
            _zero_waypoint_metrics(baseline)
            rows.append(baseline)
            completed += 1

            for depth in depths:
                controller = f"w{depth}"
                row = run_waypoint_episode(
                    model=model,
                    config=config,
                    scenario_name=scenario_name,
                    scenario_label=scenario_label,
                    seed=seed,
                    episode_index=episode_index,
                    peer_radius=peer_radius,
                    scenario_options=options,
                    requested_depth=depth,
                    proposal_config=proposal_config,
                    reached_tolerance=reached_tolerance,
                    boundary_margin=boundary_margin,
                    collision_clearance=collision_clearance,
                    segment_samples=segment_samples,
                    stagnation_steps=stagnation_steps,
                    stagnation_progress_epsilon=stagnation_progress_epsilon,
                )
                row["controller"] = controller
                row["controller_label"] = CONTROLLER_LABELS[controller]
                rows.append(row)
                completed += 1
            if completed % 20 == 0 or completed == total:
                print(f"[{completed}/{total}] {scenario_name}")

    reference_env.close()
    aggregates = aggregate_by_controller(rows)
    comparisons = build_paired_comparisons(rows)
    _write_csv(output_dir / "episodes.csv", rows)
    _write_csv(output_dir / "summary.csv", aggregates)
    _write_csv(output_dir / "paired_comparisons.csv", comparisons)
    _write_json(
        output_dir / "summary.json",
        {"aggregates": aggregates, "paired_comparisons": comparisons},
    )
    write_report(
        output_dir / "report.md",
        aggregates=aggregates,
        comparisons=comparisons,
        checkpoint=checkpoint,
    )
    print(f"Artifacts written to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
