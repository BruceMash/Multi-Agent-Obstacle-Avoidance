"""Evaluate the historical single-agent SAC navigator with current Guidance.

The task goal remains owned by the environment. Guidance only replaces the
goal direction seen by the frozen SAC actor and the attraction direction used
by the DMP controller. This preserves task success semantics and the goal
distance scale seen during historical training.
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
from scripts.evaluate_single_policy_multi_agent import (  # noqa: E402
    _build_environment,
    _jsonable,
    _safe_mean,
    _sha256,
    _write_csv,
    _write_json,
    aggregate_rows,
    build_policy_observations,
    run_episode as run_baseline_episode,
)


DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "evaluation" / "single_policy_guidance_multi_agent.json"
)
CONTROLLER_LABELS = {
    "baseline": "单机 SAC/DMP + 邻机动态球",
    "guidance": "单机 SAC/DMP + 邻机动态球 + Guidance",
}
GUIDANCE_METRIC_KEYS = (
    "guidance_request_count",
    "guidance_generated_count",
    "guidance_dead_end_count",
    "guidance_collision_replan_count",
    "guidance_reached_replan_count",
    "guidance_periodic_replan_count",
    "guidance_candidate_collision_rejection_count",
    "guidance_proposal_count_mean",
    "guidance_proposal_time_mean_ms",
    "guidance_alignment_mean",
    "guidance_waypoint_distance_mean",
    "guidance_fallback_agent_step_rate",
    "guidance_replans_per_step",
)


@dataclass
class WaypointState:
    point: np.ndarray | None = None
    generated_step: int = -1


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-9:
        return np.zeros_like(vector)
    return vector / norm


def reference_point_clearance(
    env: Any,
    agent_index: int,
    point: np.ndarray,
) -> float:
    """Return signed clearance to obstacles visible to one agent."""
    obstacles = list(env._sensor_static_obstacles())
    obstacles.extend(env._sensor_dynamic_obstacles(int(agent_index)))
    if not obstacles:
        return float("inf")
    return float(
        min(obstacle.signed_distance(np.asarray(point, dtype=float)) for obstacle in obstacles)
    )


def reference_point_has_collision(
    env: Any,
    agent_index: int,
    point: np.ndarray,
    *,
    collision_clearance: float = 0.0,
) -> bool:
    """A point is invalid once an obstacle reaches its configured clearance."""
    return reference_point_clearance(env, agent_index, point) <= float(
        collision_clearance
    )


def waypoint_refresh_reason(
    env: Any,
    agent_index: int,
    state: WaypointState,
    *,
    step: int,
    reached_tolerance: float,
    refresh_steps: int,
    collision_clearance: float,
) -> str | None:
    if state.point is None:
        return "initial" if state.generated_step < 0 else "dead_end_retry"
    if reference_point_has_collision(
        env,
        agent_index,
        state.point,
        collision_clearance=collision_clearance,
    ):
        return "collision"
    position = np.asarray(env.dynamics[agent_index].p, dtype=float)
    if float(np.linalg.norm(state.point - position)) <= float(reached_tolerance):
        return "reached"
    if int(step) - int(state.generated_step) >= int(refresh_steps):
        return "periodic"
    return None


def choose_collision_free_proposal(
    env: Any,
    agent_index: int,
    proposals: list[Proposal],
    *,
    collision_clearance: float,
) -> tuple[Proposal | None, int]:
    rejected = 0
    for proposal in proposals:
        if reference_point_has_collision(
            env,
            agent_index,
            proposal.point,
            collision_clearance=collision_clearance,
        ):
            rejected += 1
            continue
        return proposal, rejected
    return None, rejected


def build_direction_proxy_goal(
    position: np.ndarray,
    task_goal: np.ndarray,
    waypoint: np.ndarray | None,
) -> np.ndarray:
    """Preserve task-distance magnitude while replacing only attraction direction."""
    position = np.asarray(position, dtype=float)
    task_goal = np.asarray(task_goal, dtype=float)
    if waypoint is None:
        return task_goal.copy()
    task_distance = float(np.linalg.norm(task_goal - position))
    direction = _normalize(np.asarray(waypoint, dtype=float) - position)
    if task_distance <= 1.0e-9 or not np.any(direction):
        return task_goal.copy()
    return position + task_distance * direction


def build_guided_policy_observations(
    env: Any,
    proxy_goals: np.ndarray,
    task_goals: np.ndarray,
) -> np.ndarray:
    """Override goal direction while preserving the historical goal-distance feature."""
    observations = build_policy_observations(env).copy()
    positions = np.asarray(env._positions(), dtype=float)
    for agent_index in range(int(env.num_agents)):
        proxy_delta = np.asarray(proxy_goals[agent_index], dtype=float) - positions[agent_index]
        observations[agent_index, 3:6] = _normalize(proxy_delta).astype(np.float32)
        task_distance = float(
            np.linalg.norm(np.asarray(task_goals[agent_index], dtype=float) - positions[agent_index])
        )
        observations[agent_index, 6] = np.float32(
            np.clip(task_distance / float(env.sensors[agent_index].goal_distance_clip), 0.0, 1.0)
        )
    return observations


def _zero_guidance_metrics(row: dict[str, Any]) -> None:
    for key in GUIDANCE_METRIC_KEYS:
        row[key] = 0.0


def run_guidance_episode(
    *,
    model: Any,
    config: Any,
    scenario_name: str,
    scenario_label: str,
    seed: int,
    episode_index: int,
    peer_radius: float,
    scenario_options: dict[str, Any],
    proposal_config: ProposalConfig,
    refresh_steps: int,
    reached_tolerance: float,
    collision_clearance: float,
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
        states = [WaypointState() for _ in range(int(env.num_agents))]
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
        proposal_counts: list[float] = []
        proposal_alignments: list[float] = []
        waypoint_distances: list[float] = []
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
            "collision": 0,
            "reached": 0,
            "periodic": 0,
            "candidate_collision_rejection": 0,
        }
        collision = False
        obstacle_collision = False
        inter_agent_collision = False
        boundary_collision = False
        terminated = False
        truncated = False

        while not (terminated or truncated):
            active_mask = np.logical_not(env.success_rewarded_mask.copy())
            proxy_goals = task_goals.copy()
            for agent_index in range(int(env.num_agents)):
                if not bool(active_mask[agent_index]):
                    env.dmps[agent_index].goal = task_goals[agent_index].copy()
                    continue
                active_agent_steps += 1
                state = states[agent_index]
                reason = waypoint_refresh_reason(
                    env,
                    agent_index,
                    state,
                    step=int(env.steps),
                    reached_tolerance=reached_tolerance,
                    refresh_steps=refresh_steps,
                    collision_clearance=collision_clearance,
                )
                if reason is not None:
                    counters["request"] += 1
                    if reason in counters:
                        counters[reason] += 1
                    proposal_start = time.perf_counter_ns()
                    proposals = propose_reference_points(
                        positions[agent_index],
                        task_goals[agent_index],
                        env.dynamics[agent_index].v,
                        env.latest_sensor_packets[agent_index],
                        env.sensors[agent_index],
                        proposal_config,
                        float(config.goal_tolerance),
                    )
                    proposal_times_ms.append(
                        float(time.perf_counter_ns() - proposal_start) / 1_000_000.0
                    )
                    proposal_counts.append(float(len(proposals)))
                    selected, rejected = choose_collision_free_proposal(
                        env,
                        agent_index,
                        proposals,
                        collision_clearance=collision_clearance,
                    )
                    counters["candidate_collision_rejection"] += int(rejected)
                    if selected is None:
                        counters["dead_end"] += 1
                        state.point = None
                        state.generated_step = int(env.steps)
                    else:
                        counters["generated"] += 1
                        state.point = np.asarray(selected.point, dtype=float).copy()
                        state.generated_step = int(env.steps)
                        proposal_alignments.append(float(selected.alignment))
                        waypoint_distances.append(float(selected.distance))

                if state.point is None:
                    fallback_agent_steps += 1
                proxy_goals[agent_index] = build_direction_proxy_goal(
                    positions[agent_index],
                    task_goals[agent_index],
                    state.point,
                )
                env.dmps[agent_index].goal = proxy_goals[agent_index].copy()

            observations = build_guided_policy_observations(
                env,
                proxy_goals,
                task_goals,
            )
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
        direct_lengths = np.linalg.norm(task_goals - starts, axis=1)
        reached_efficiency = np.divide(direct_lengths, np.maximum(path_lengths, 1.0e-9))
        reached_efficiency[~ever_success] = np.nan
        forcing_denominator = max(1, action_agent_count * 3)
        offset_denominator = max(1, action_agent_count * 3)
        dynamics_denominator = max(1, dynamics_axis_count)
        row = {
            "mode": "peer_spheres",
            "mode_label": "邻机动态球观测",
            "suite": "single_guidance_paired",
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
            "min_pairwise_clearance": (
                min_pairwise_distance - float(config.inter_agent_safe_distance)
            ),
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
            "guidance_request_count": int(counters["request"]),
            "guidance_generated_count": int(counters["generated"]),
            "guidance_dead_end_count": int(counters["dead_end"]),
            "guidance_collision_replan_count": int(counters["collision"]),
            "guidance_reached_replan_count": int(counters["reached"]),
            "guidance_periodic_replan_count": int(counters["periodic"]),
            "guidance_candidate_collision_rejection_count": int(
                counters["candidate_collision_rejection"]
            ),
            "guidance_proposal_count_mean": _safe_mean(proposal_counts),
            "guidance_proposal_time_mean_ms": _safe_mean(proposal_times_ms),
            "guidance_alignment_mean": _safe_mean(proposal_alignments),
            "guidance_waypoint_distance_mean": _safe_mean(waypoint_distances),
            "guidance_fallback_agent_step_rate": (
                fallback_agent_steps / max(1, active_agent_steps)
            ),
            "guidance_replans_per_step": counters["request"] / max(1, int(env.steps)),
        }
        return row
    finally:
        env.close()


def aggregate_by_controller(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    controllers = sorted({str(row["controller"]) for row in rows})
    for controller in controllers:
        members = [row for row in rows if row["controller"] == controller]
        for aggregate in aggregate_rows(members):
            aggregate["controller"] = controller
            aggregate["controller_label"] = CONTROLLER_LABELS[controller]
            scenario_members = [
                row for row in members if row["scenario"] == aggregate["scenario"]
            ]
            aggregate["boundary_excursion_rate"] = float(
                np.mean(
                    [float(row["min_boundary_distance"]) < 0.0 for row in scenario_members]
                )
            )
            for key in GUIDANCE_METRIC_KEYS:
                aggregate[f"{key}_mean"] = _safe_mean(
                    [float(row[key]) for row in scenario_members]
                )
            results.append(aggregate)
    return results


def build_paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["scenario"]), int(row["seed"]), str(row["controller"])): row
        for row in rows
    }
    comparisons: list[dict[str, Any]] = []
    scenarios = sorted({str(row["scenario"]) for row in rows})
    for scenario in scenarios:
        seeds = sorted({int(row["seed"]) for row in rows if row["scenario"] == scenario})
        paired = [
            (lookup[(scenario, seed, "baseline")], lookup[(scenario, seed, "guidance")])
            for seed in seeds
        ]
        baseline_only = sum(bool(a["team_success"]) and not bool(b["team_success"]) for a, b in paired)
        guidance_only = sum(not bool(a["team_success"]) and bool(b["team_success"]) for a, b in paired)
        comparisons.append(
            {
                "scenario": scenario,
                "scenario_label": paired[0][0]["scenario_label"],
                "paired_episodes": len(paired),
                "both_success": sum(bool(a["team_success"]) and bool(b["team_success"]) for a, b in paired),
                "baseline_success_only": baseline_only,
                "guidance_success_only": guidance_only,
                "both_fail": sum(not bool(a["team_success"]) and not bool(b["team_success"]) for a, b in paired),
                "net_success_change": guidance_only - baseline_only,
                "inter_agent_collision_resolved": sum(bool(a["inter_agent_collision"]) and not bool(b["inter_agent_collision"]) for a, b in paired),
                "inter_agent_collision_introduced": sum(not bool(a["inter_agent_collision"]) and bool(b["inter_agent_collision"]) for a, b in paired),
                "obstacle_collision_resolved": sum(bool(a["obstacle_collision"]) and not bool(b["obstacle_collision"]) for a, b in paired),
                "obstacle_collision_introduced": sum(not bool(a["obstacle_collision"]) and bool(b["obstacle_collision"]) for a, b in paired),
            }
        )
    return comparisons


def diagnose_failures(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["scenario"], row["controller"]): row for row in aggregates}
    diagnoses: list[dict[str, Any]] = []
    for scenario in sorted({key[0] for key in lookup}):
        baseline = lookup[(scenario, "baseline")]
        guidance = lookup[(scenario, "guidance")]
        causes: list[str] = []
        if guidance["inter_agent_collision_rate"] >= baseline["inter_agent_collision_rate"]:
            causes.append("Guidance仅做各机局部反应，未建立通行权、优先级或时序互斥")
        if guidance["obstacle_collision_rate"] > baseline["obstacle_collision_rate"]:
            causes.append("参考方向安全性未完全转化为SAC/DMP实际闭环轨迹安全性")
        if guidance["boundary_excursion_rate"] > baseline["boundary_excursion_rate"] + 0.20:
            causes.append(
                "历史单机语义未向LiDAR注册边界，方向代理又将短参考方向按剩余航程放大，导致越界显著增加"
            )
        if (
            guidance["forcing_axis_saturation_rate_mean"]
            > baseline["forcing_axis_saturation_rate_mean"] + 0.05
        ):
            causes.append("Guidance目标方向改变引起策略动作分布偏移与forcing饱和增加")
        if (
            guidance["offset_axis_saturation_rate_mean"]
            > baseline["offset_axis_saturation_rate_mean"] + 0.05
        ):
            causes.append("Guidance方向超出历史策略常见目标方向分布，goal offset饱和增加")
        if (
            guidance["acceleration_axis_clip_rate_mean"]
            > baseline["acceleration_axis_clip_rate_mean"] + 0.05
        ):
            causes.append("方向代理目标使DMP闭环驱动增大，动力学裁剪增加")
        if guidance["timeout_rate"] > baseline["timeout_rate"]:
            causes.append("周期重规划或局部绕行增加路径长度，50步时域不足")
        if guidance["guidance_dead_end_count_mean"] > 0.0:
            causes.append("部分时刻没有满足净空、制动距离与正向进度约束的候选点")
        if guidance["guidance_collision_replan_count_mean"] > 0.0:
            causes.append("移动邻机或障碍物占用当前参考点，触发碰撞失效重规划")
        if not causes and guidance["team_success_rate"] < 1.0:
            causes.append("失败未由单一诊断指标主导，需结合逐回合轨迹进一步检查")
        diagnoses.append(
            {
                "scenario": scenario,
                "baseline_team_success_rate": baseline["team_success_rate"],
                "guidance_team_success_rate": guidance["team_success_rate"],
                "success_rate_delta": guidance["team_success_rate"] - baseline["team_success_rate"],
                "possible_causes": causes,
            }
        )
    return diagnoses


def write_report(
    path: Path,
    *,
    aggregates: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    diagnoses: list[dict[str, Any]],
    checkpoint: Path,
) -> None:
    order = {"baseline": 0, "guidance": 1}
    rows = sorted(aggregates, key=lambda row: (row["scenario"], order[row["controller"]]))
    lines = [
        "# 单机 SAC/DMP 接入 Guidance 的多机配对实验",
        "",
        f"- Checkpoint：`{checkpoint}`",
        "- 最终目标用于奖励与成功判定；Guidance仅提供局部方向代理目标。",
        "- 当前参考点发生障碍物或邻机占用碰撞时，立即重新生成参考点。",
        "",
        "## 汇总结果",
        "",
        "| 场景 | 控制方式 | 团队成功率 | 单机到达率 | 机间碰撞率 | 障碍碰撞率 | 超时率 | 越界回合率 | 平均路径/m | forcing饱和率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {controller} | {team:.2%} | {agent:.2%} | {inter:.2%} | "
            "{obstacle:.2%} | {timeout:.2%} | {boundary:.2%} | {path:.3f} | {forcing:.2%} |".format(
                scenario=row["scenario_label"],
                controller=row["controller_label"],
                team=row["team_success_rate"],
                agent=row["agent_success_rate_mean"],
                inter=row["inter_agent_collision_rate"],
                obstacle=row["obstacle_collision_rate"],
                timeout=row["timeout_rate"],
                boundary=row["boundary_excursion_rate"],
                path=row["path_length_mean_mean"],
                forcing=row["forcing_axis_saturation_rate_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## 配对变化",
            "",
            "| 场景 | 两者成功 | Baseline独有成功 | Guidance独有成功 | 两者失败 | 净成功变化 | 机间碰撞消除/引入 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            "| {scenario} | {both_success} | {baseline_only} | {guidance_only} | {both_fail} | {net:+d} | {resolved}/{introduced} |".format(
                scenario=row["scenario_label"],
                both_success=row["both_success"],
                baseline_only=row["baseline_success_only"],
                guidance_only=row["guidance_success_only"],
                both_fail=row["both_fail"],
                net=row["net_success_change"],
                resolved=row["inter_agent_collision_resolved"],
                introduced=row["inter_agent_collision_introduced"],
            )
        )
    lines.extend(["", "## 失败原因诊断", ""])
    for row in diagnoses:
        lines.append(
            f"### {row['scenario']}：成功率变化 {row['success_rate_delta']:+.2%}"
        )
        lines.append("")
        for cause in row["possible_causes"]:
            lines.append(f"- {cause}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


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
    guidance_settings = dict(settings.get("guidance", {}))
    refresh_steps = int(guidance_settings.pop("refresh_steps", 5))
    reached_tolerance = float(guidance_settings.pop("reached_tolerance", 0.25))
    collision_clearance = float(guidance_settings.pop("collision_clearance", 0.0))
    proposal_config = ProposalConfig(**guidance_settings)
    if episodes_per_stage <= 0 or refresh_steps <= 0 or reached_tolerance <= 0.0:
        raise ValueError("episode count, refresh_steps and reached_tolerance must be positive")

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
        else REPO_ROOT / "artifacts" / f"single_policy_guidance_multi_agent_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    config = build_single_distribution_multi_config(num_agents=num_agents, max_steps=max_steps)
    reference_env = build_single_env(
        config=SINGLE_AGENT_CONFIG,
        action_guidance_enabled=False,
    )
    model = build_single_model(reference_env, config=SINGLE_AGENT_CONFIG, verbose=0)
    load_checkpoint(model, checkpoint)
    model.actor.train(False)

    snapshot = {
        "created_at": datetime.now().astimezone().isoformat(),
        "evaluation_config_path": config_path,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "episodes_per_stage": episodes_per_stage,
        "seed_base": seed_base,
        "peer_radius": peer_radius,
        "stages": stages,
        "guidance_adapter": {
            "interface": "direction_proxy_goal",
            "preserve_task_goal_for_reward_and_success": True,
            "preserve_task_goal_distance_feature": True,
            "refresh_steps": refresh_steps,
            "reached_tolerance": reached_tolerance,
            "collision_clearance": collision_clearance,
            "collision_triggers_immediate_replan": True,
        },
        "proposal_config": asdict(proposal_config),
        "aligned_config": asdict(config),
    }
    _write_json(output_dir / "config_snapshot.json", snapshot)

    rows: list[dict[str, Any]] = []
    total = len(stages) * episodes_per_stage * 2
    completed = 0
    clean_labels = {
        "C_parallel_peer_spheres": "C 平行航路（邻机动态球）",
        "D_permuted_peer_spheres": "D 目标置换（邻机动态球）",
        "E_head_on_narrow_peer_spheres": "E 对向窄通道（邻机动态球）",
    }
    for stage in stages:
        for episode_index in range(episodes_per_stage):
            seed = seed_base + episode_index
            options = build_stage_scenario(config, stage, seed=seed)
            label = clean_labels.get(str(stage["name"]), str(stage["name"]))
            baseline = run_baseline_episode(
                model=model,
                config=config,
                suite="single_guidance_paired",
                scenario_name=str(stage["name"]),
                seed=seed,
                episode_index=episode_index,
                observation_mode="peer_spheres",
                peer_radius=peer_radius,
                scenario_options_override=options,
                scenario_label_override=label,
                include_boundaries_in_sensor=False,
                terminate_on_boundary_collision=False,
            )
            baseline["controller"] = "baseline"
            baseline["controller_label"] = CONTROLLER_LABELS["baseline"]
            _zero_guidance_metrics(baseline)
            rows.append(baseline)
            completed += 1

            guidance = run_guidance_episode(
                model=model,
                config=config,
                scenario_name=str(stage["name"]),
                scenario_label=label,
                seed=seed,
                episode_index=episode_index,
                peer_radius=peer_radius,
                scenario_options=options,
                proposal_config=proposal_config,
                refresh_steps=refresh_steps,
                reached_tolerance=reached_tolerance,
                collision_clearance=collision_clearance,
            )
            guidance["controller"] = "guidance"
            guidance["controller_label"] = CONTROLLER_LABELS["guidance"]
            rows.append(guidance)
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"[{completed}/{total}] {stage['name']}")

    reference_env.close()
    aggregates = aggregate_by_controller(rows)
    paired = build_paired_comparisons(rows)
    diagnoses = diagnose_failures(aggregates)
    _write_csv(output_dir / "episodes.csv", rows)
    _write_csv(output_dir / "summary.csv", aggregates)
    _write_csv(output_dir / "paired_comparisons.csv", paired)
    _write_json(
        output_dir / "summary.json",
        {
            "aggregates": aggregates,
            "paired_comparisons": paired,
            "failure_diagnoses": diagnoses,
        },
    )
    write_report(
        output_dir / "report.md",
        aggregates=aggregates,
        paired=paired,
        diagnoses=diagnoses,
        checkpoint=checkpoint,
    )
    print(f"Artifacts written to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
