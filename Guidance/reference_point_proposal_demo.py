"""
基于当前 MASAC 训练环境的 3D LiDAR 参考点提议性能测试。

测试流程：
1. 直接使用 MASACExperimentConfig、MultiAgentDMPEnv 和课程场景生成器；
2. 每轮从 LocalObstacleSensor.current_scan 读取与训练一致的 3D LiDAR 扫描；
3. 按安全裕度、真实距离进度、速度连续性和可用前视距离生成并排序候选点；
4. 保留每轮 top-K 候选点，选择得分最高的候选点作为下一轮当前位置；
5. 持续首尾相连，直至全部智能体到达目标、无可行候选点或超过步数上限；
6. 输出逐回合指标、逐轮候选点、选择轨迹、汇总结果和三维可视化。

本脚本用于隔离评估参考点提议与评分机制，不执行强化学习策略和 DMP 动力学控制。
碰撞事件按训练环境标准持续记录，但不提前截断几何路径，以便同时评价
“能否形成完整起终点路径”和“路径是否满足训练环境安全约束”。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Microsoft JhengHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
ALGO_ROOT = PROJECT_ROOT / "Multi-agent_Algo_lib"
for import_path in (PROJECT_ROOT, ALGO_ROOT):
    import_path_text = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)

from Environment.multi_agent_dmp_env import MultiAgentDMPEnv
from MASAC.config import MASAC_EXPERIMENT_CONFIG, MASACExperimentConfig
from MASAC.curriculum import (
    CurriculumStage,
    build_curriculum_stages,
    build_stage_env_kwargs,
)


EPS = 1e-8
AGENT_COLORS = ("#00B8D9", "#FFAB00", "#E75480", "#7C4DFF", "#36B37E")


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < EPS:
        return np.zeros_like(vector, dtype=float)
    return vector / norm


@dataclass(frozen=True)
class ProposalConfig:
    """保持原 demo 评分结构，仅由训练传感器提供方向和扫描距离。"""

    safe_radius: float = 1.15
    terminal_safe_radius: float = 0.30
    braking_deceleration: float = 4.0
    reaction_time: float = 0.10
    eta: float = 0.58
    s_min: float = 0.35
    s_max: float = 1.05
    h_max: float = 2.5
    top_k: int = 10
    min_alignment: float = -0.10
    terminal_radius: float = 1.50
    terminal_step_ratio: float = 0.55
    terminal_min_step: float = 0.05
    positive_progress_epsilon: float = 1e-6
    scan_guard_azimuth_bins: int = 1
    scan_guard_elevation_bins: int = 1
    obstacle_motion_allowance: float = 0.06
    w_h: float = 1.10
    w_p: float = 1.45
    w_v: float = 0.20
    w_l: float = 0.85
    nominal_speed: float = 0.80
    distance_mode: str = "original"
    adaptive_r_min: float = 0.35
    adaptive_r_max: float = 4.50
    goal_alignment_gamma: float = 2.0

    def __post_init__(self) -> None:
        positive_fields = (
            self.safe_radius,
            self.terminal_safe_radius,
            self.braking_deceleration,
            self.reaction_time,
            self.eta,
            self.s_min,
            self.s_max,
            self.h_max,
            self.nominal_speed,
            self.adaptive_r_min,
            self.adaptive_r_max,
            self.goal_alignment_gamma,
            self.terminal_radius,
            self.terminal_step_ratio,
            self.terminal_min_step,
        )
        if any(value <= 0.0 for value in positive_fields):
            raise ValueError("proposal distance, safety and speed parameters must be positive")
        if self.s_min > self.s_max:
            raise ValueError("s_min must not exceed s_max")
        if self.terminal_safe_radius > self.safe_radius:
            raise ValueError("terminal_safe_radius must not exceed safe_radius")
        if self.terminal_step_ratio > 1.0:
            raise ValueError("terminal_step_ratio must not exceed 1")
        if self.terminal_min_step > self.s_min:
            raise ValueError("terminal_min_step must not exceed s_min")
        if self.positive_progress_epsilon < 0.0:
            raise ValueError("positive_progress_epsilon must be non-negative")
        if self.scan_guard_azimuth_bins < 0 or self.scan_guard_elevation_bins < 0:
            raise ValueError("scan guard bin counts must be non-negative")
        if self.obstacle_motion_allowance < 0.0:
            raise ValueError("obstacle_motion_allowance must be non-negative")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.distance_mode not in {"original", "goal_aligned_adaptive"}:
            raise ValueError("distance_mode must be original or goal_aligned_adaptive")
        if self.adaptive_r_min > self.adaptive_r_max:
            raise ValueError("adaptive_r_min must not exceed adaptive_r_max")
        if self.adaptive_r_min < self.s_min:
            raise ValueError("adaptive_r_min must respect the original minimum step")


@dataclass(frozen=True)
class Proposal:
    azimuth_index: int
    elevation_index: int
    direction: np.ndarray
    point: np.ndarray
    distance: float
    raw_obstacle_distance: float
    obstacle_distance: float
    effective_safe_radius: float
    braking_distance: float
    safety_margin: float
    normalized_margin: float
    distance_progress: float
    normalized_progress: float
    alignment: float
    smoothness: float
    usable_length: float
    score: float


@dataclass(frozen=True)
class SectorSafetyField:
    """Per-sector safety quantities shared by proposal and graph consumers."""

    raw_obstacle_distance: np.ndarray
    obstacle_distance: np.ndarray
    effective_safe_radius: float
    braking_distance: float
    safety_margin: np.ndarray
    normalized_margin: np.ndarray


@dataclass
class EpisodeResult:
    seed: int
    stage: str
    success: bool
    route_completed: bool
    termination: str
    proposal_steps: int
    reached_agent_count: int
    agent_success_rate: float
    obstacle_collision: bool
    boundary_collision: bool
    inter_agent_collision: bool
    dead_end: bool
    mean_path_length: float
    mean_direct_length: float
    mean_path_efficiency: float
    mean_final_goal_distance: float
    min_inter_agent_distance: float
    min_obstacle_clearance: float
    mean_retained_candidate_count: float
    mean_viable_candidate_count: float
    mean_proposal_time_ms: float
    max_proposal_time_ms: float


def build_stage(config: MASACExperimentConfig, stage_name: str) -> CurriculumStage:
    stages = build_curriculum_stages(
        config.curriculum_phase2_box_counts,
        config.curriculum_phase2_sphere_counts,
        config.curriculum_phase3_dynamic_counts,
    )
    if stage_name == "phase1":
        return stages[0]
    if stage_name == "static":
        phase2_stages = [stage for stage in stages if stage.phase == 2]
        return phase2_stages[-1]
    if stage_name == "final":
        return stages[-1]
    raise ValueError(f"unsupported stage: {stage_name}")


def build_environment(
    config: MASACExperimentConfig,
    stage: CurriculumStage,
) -> MultiAgentDMPEnv:
    return MultiAgentDMPEnv(**build_stage_env_kwargs(config, stage))


def verify_environment_alignment(
    env: MultiAgentDMPEnv,
    config: MASACExperimentConfig,
) -> dict[str, Any]:
    """检查脚本实际使用的传感器与训练配置是否完全一致。"""

    sensor = env.sensors[0]
    expected = {
        "sensing_radius": float(config.sensing_radius),
        "azimuth_bins": int(config.sensor_azimuth_bins),
        "elevation_bins": int(config.sensor_elevation_bins),
        "elevation_range_deg": tuple(float(v) for v in config.sensor_elevation_range_deg),
        "include_previous_scan": bool(config.sensor_include_previous_scan),
        "workspace_bounds": tuple(tuple(float(v) for v in row) for row in config.workspace_bounds),
        "num_agents": int(config.num_agents),
        "time_step": float(config.time_step),
        "max_steps": int(config.max_steps),
        "goal_tolerance": float(config.goal_tolerance),
    }
    actual = {
        "sensing_radius": float(sensor.sensing_radius),
        "azimuth_bins": int(sensor.azimuth_bins),
        "elevation_bins": int(sensor.elevation_bins),
        "elevation_range_deg": tuple(float(v) for v in sensor.elevation_range_deg),
        "include_previous_scan": bool(sensor.include_previous_scan),
        "workspace_bounds": tuple(
            tuple(float(v) for v in row)
            for row in np.asarray(env.env_config.workspace_bounds, dtype=float)
        ),
        "num_agents": int(env.num_agents),
        "time_step": float(env.dynamics[0].dt),
        "max_steps": int(env.env_config.max_steps),
        "goal_tolerance": float(env.env_config.goal_tolerance),
    }
    if actual != expected:
        mismatches = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if expected[key] != actual[key]
        }
        raise RuntimeError(f"environment alignment check failed: {mismatches}")
    if sensor.ray_directions.shape != (
        int(config.sensor_azimuth_bins),
        int(config.sensor_elevation_bins),
        3,
    ):
        raise RuntimeError("sensor ray direction tensor does not match configured scan organization")
    return actual


def compute_sector_safety_field(
    position: np.ndarray,
    goal: np.ndarray,
    velocity: np.ndarray,
    sensor_packet: Any,
    sensor: Any,
    config: ProposalConfig,
    goal_tolerance: float,
) -> SectorSafetyField:
    """Compute the proposal generator's existing sector-safety semantics."""

    position = np.asarray(position, dtype=float)
    goal = np.asarray(goal, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    goal_distance = float(np.linalg.norm(goal - position))
    speed = float(np.linalg.norm(velocity))

    scan = np.asarray(sensor_packet.current_scan, dtype=float)
    if scan.shape != sensor.scan_shape:
        raise ValueError(
            f"current_scan shape {scan.shape} does not match sensor scan_shape {sensor.scan_shape}"
        )
    raw_obstacle_distances = (
        np.clip(scan, 0.0, 1.0) * float(sensor.sensing_radius)
    )
    obstacle_distances = raw_obstacle_distances.copy()
    elevation_indices = np.arange(int(sensor.elevation_bins))
    for azimuth_offset in range(
        -config.scan_guard_azimuth_bins,
        config.scan_guard_azimuth_bins + 1,
    ):
        azimuth_shifted = np.roll(
            raw_obstacle_distances,
            shift=azimuth_offset,
            axis=0,
        )
        for elevation_offset in range(
            -config.scan_guard_elevation_bins,
            config.scan_guard_elevation_bins + 1,
        ):
            neighbor_elevation_indices = np.clip(
                elevation_indices + elevation_offset,
                0,
                int(sensor.elevation_bins) - 1,
            )
            obstacle_distances = np.minimum(
                obstacle_distances,
                azimuth_shifted[:, neighbor_elevation_indices],
            )
    obstacle_distances = np.maximum(
        obstacle_distances - config.obstacle_motion_allowance,
        0.0,
    )

    terminal_mode = goal_distance <= config.terminal_radius
    if terminal_mode:
        terminal_span = max(
            config.terminal_radius - float(goal_tolerance),
            EPS,
        )
        cruise_blend = float(
            np.clip(
                (goal_distance - float(goal_tolerance)) / terminal_span,
                0.0,
                1.0,
            )
        )
        effective_safe_radius = (
            config.terminal_safe_radius
            + cruise_blend * (config.safe_radius - config.terminal_safe_radius)
        )
    else:
        effective_safe_radius = config.safe_radius
    braking_distance = (
        speed * speed / (2.0 * config.braking_deceleration)
        + config.reaction_time * speed
    )
    safety_margin = (
        obstacle_distances - effective_safe_radius - braking_distance
    )
    normalized_margin = np.clip(
        safety_margin / config.h_max, 0.0, 1.0
    )
    return SectorSafetyField(
        raw_obstacle_distance=raw_obstacle_distances.copy(),
        obstacle_distance=obstacle_distances.copy(),
        effective_safe_radius=float(effective_safe_radius),
        braking_distance=float(braking_distance),
        safety_margin=safety_margin.copy(),
        normalized_margin=normalized_margin.copy(),
    )


def propose_reference_points(
    position: np.ndarray,
    goal: np.ndarray,
    velocity: np.ndarray,
    sensor_packet: Any,
    sensor: Any,
    config: ProposalConfig,
    goal_tolerance: float,
    *,
    timing_sink: dict[str, float] | None = None,
) -> list[Proposal]:
    """
    使用当前训练传感器的射线方向与 current_scan 生成所有可行候选点。

    current_scan 在环境中归一化到 [0, 1]，因此先乘 sensing_radius
    恢复物理距离。射线展平顺序与 LocalObstacleSensor._scan_obstacles
    的“方位角优先、俯仰角次之”组织方式保持一致。
    """

    position = np.asarray(position, dtype=float)
    goal = np.asarray(goal, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    goal_vector = goal - position
    goal_distance = float(np.linalg.norm(goal_vector))
    if goal_distance <= float(goal_tolerance):
        if timing_sink is not None:
            timing_sink["coarse_ranking_ms"] = 0.0
        return []

    goal_direction = normalize(goal_vector)
    velocity_direction = normalize(velocity)
    directions = np.asarray(sensor.ray_directions, dtype=float)
    sector_safety = compute_sector_safety_field(
        position,
        goal,
        velocity,
        sensor_packet,
        sensor,
        config,
        goal_tolerance,
    )

    proposals: list[Proposal] = []
    terminal_mode = goal_distance <= config.terminal_radius
    minimum_step = config.terminal_min_step if terminal_mode else config.s_min
    distance_limit = (
        config.adaptive_r_max
        if config.distance_mode == "goal_aligned_adaptive" and not terminal_mode
        else config.s_max
    )
    maximum_step = min(distance_limit, goal_distance)
    if terminal_mode:
        maximum_step = min(
            maximum_step,
            max(config.terminal_min_step, config.terminal_step_ratio * goal_distance),
        )
    for azimuth_index in range(int(sensor.azimuth_bins)):
        for elevation_index in range(int(sensor.elevation_bins)):
            direction = directions[azimuth_index, elevation_index]
            alignment = float(np.dot(direction, goal_direction))
            if alignment < config.min_alignment:
                continue

            raw_obstacle_distance = float(
                sector_safety.raw_obstacle_distance[azimuth_index, elevation_index]
            )
            obstacle_distance = float(
                sector_safety.obstacle_distance[azimuth_index, elevation_index]
            )
            safety_margin = float(
                sector_safety.safety_margin[azimuth_index, elevation_index]
            )
            if safety_margin <= 0.0:
                continue

            if config.distance_mode == "goal_aligned_adaptive" and not terminal_mode:
                q_safe = float(
                    sector_safety.normalized_margin[azimuth_index, elevation_index]
                )
                q_goal = 0.5 * (1.0 + float(np.clip(alignment, -1.0, 1.0)))
                desired_step = float(
                    np.clip(
                        config.adaptive_r_min
                        + (config.adaptive_r_max - config.adaptive_r_min)
                        * q_safe
                        * q_goal ** config.goal_alignment_gamma,
                        minimum_step,
                        maximum_step,
                    )
                )
            else:
                desired_step = float(
                    np.clip(config.eta * safety_margin, minimum_step, maximum_step)
                )
            step = min(
                desired_step,
                obstacle_distance - sector_safety.effective_safe_radius,
                float(sensor.sensing_radius),
                maximum_step,
            )
            if step < minimum_step:
                continue
            if step <= EPS:
                continue

            normalized_margin = float(
                sector_safety.normalized_margin[azimuth_index, elevation_index]
            )
            usable_length = float(np.clip(step / distance_limit, 0.0, 1.0))
            candidate_point = position + step * direction
            candidate_goal_distance = float(np.linalg.norm(goal - candidate_point))
            distance_progress = goal_distance - candidate_goal_distance
            normalized_progress = float(
                np.clip(distance_progress / max(step, EPS), -1.0, 1.0)
            )
            smoothness = (
                float(np.dot(direction, velocity_direction))
                if np.linalg.norm(velocity_direction) > 0.0
                else 0.0
            )
            score = (
                config.w_h * normalized_margin
                + config.w_p * normalized_progress
                + config.w_v * smoothness
                + config.w_l * usable_length
            )
            proposals.append(
                Proposal(
                    azimuth_index=azimuth_index,
                    elevation_index=elevation_index,
                    direction=direction.copy(),
                    point=candidate_point,
                    distance=float(step),
                    raw_obstacle_distance=raw_obstacle_distance,
                    obstacle_distance=obstacle_distance,
                    effective_safe_radius=sector_safety.effective_safe_radius,
                    braking_distance=sector_safety.braking_distance,
                    safety_margin=float(safety_margin),
                    normalized_margin=normalized_margin,
                    distance_progress=float(distance_progress),
                    normalized_progress=normalized_progress,
                    alignment=alignment,
                    smoothness=smoothness,
                    usable_length=usable_length,
                    score=float(score),
                )
            )

    coarse_started = time.perf_counter_ns()
    positive_proposals = [
        proposal
        for proposal in proposals
        if proposal.distance_progress > config.positive_progress_epsilon
    ]
    if positive_proposals:
        proposals = positive_proposals
    proposals.sort(key=lambda proposal: proposal.score, reverse=True)
    if timing_sink is not None:
        timing_sink["coarse_ranking_ms"] = (
            time.perf_counter_ns() - coarse_started
        ) / 1.0e6
    return proposals


def set_environment_agent_states(
    env: MultiAgentDMPEnv,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> None:
    for agent_index in range(int(env.num_agents)):
        env.dynamics[agent_index].p = np.asarray(
            positions[agent_index], dtype=float
        ).copy()
        env.dynamics[agent_index].v = np.asarray(
            velocities[agent_index], dtype=float
        ).copy()


def minimum_obstacle_clearance(
    positions: np.ndarray,
    obstacles: list[Any],
) -> float:
    if not obstacles:
        return float("inf")
    return float(
        min(
            obstacle.signed_distance(position)
            for position in np.asarray(positions, dtype=float)
            for obstacle in obstacles
        )
    )


def record_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    stage_name: str,
    step: int,
    agent_index: int,
    origin: np.ndarray,
    proposals: list[Proposal],
    top_k: int,
) -> None:
    retained = proposals[:top_k]
    for rank, proposal in enumerate(retained, start=1):
        rows.append(
            {
                "seed": int(seed),
                "stage": stage_name,
                "step": int(step),
                "agent": int(agent_index),
                "rank": int(rank),
                "selected": bool(rank == 1),
                "viable_candidate_count": int(len(proposals)),
                "origin_x": float(origin[0]),
                "origin_y": float(origin[1]),
                "origin_z": float(origin[2]),
                "point_x": float(proposal.point[0]),
                "point_y": float(proposal.point[1]),
                "point_z": float(proposal.point[2]),
                "direction_x": float(proposal.direction[0]),
                "direction_y": float(proposal.direction[1]),
                "direction_z": float(proposal.direction[2]),
                "azimuth_index": int(proposal.azimuth_index),
                "elevation_index": int(proposal.elevation_index),
                "score": float(proposal.score),
                "distance": float(proposal.distance),
                "raw_obstacle_distance": float(proposal.raw_obstacle_distance),
                "obstacle_distance": float(proposal.obstacle_distance),
                "effective_safe_radius": float(proposal.effective_safe_radius),
                "braking_distance": float(proposal.braking_distance),
                "safety_margin": float(proposal.safety_margin),
                "normalized_margin": float(proposal.normalized_margin),
                "distance_progress": float(proposal.distance_progress),
                "normalized_progress": float(proposal.normalized_progress),
                "alignment": float(proposal.alignment),
                "smoothness": float(proposal.smoothness),
                "usable_length": float(proposal.usable_length),
            }
        )


def obstacle_snapshot(obstacle: Any, dynamic: bool) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "type": type(obstacle).__name__,
        "dynamic": bool(dynamic),
        "center": np.asarray(obstacle.center, dtype=float).tolist(),
    }
    if hasattr(obstacle, "expanded_half_extents"):
        snapshot["half_extents"] = np.asarray(
            obstacle.expanded_half_extents, dtype=float
        ).tolist()
    else:
        snapshot["radius"] = float(obstacle.effective_radius)
    if hasattr(obstacle, "velocity"):
        snapshot["velocity"] = np.asarray(obstacle.velocity, dtype=float).tolist()
    return snapshot


def run_episode(
    env: MultiAgentDMPEnv,
    *,
    seed: int,
    stage_name: str,
    proposal_config: ProposalConfig,
    max_steps: int,
) -> tuple[
    EpisodeResult,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    _, reset_info = env.reset(seed=seed)
    positions = np.asarray(reset_info["starts"], dtype=float).copy()
    goals = np.asarray(reset_info["goals"], dtype=float).copy()
    velocities = np.zeros_like(positions, dtype=float)
    reached = np.zeros(int(env.num_agents), dtype=bool)

    trajectories: list[list[np.ndarray]] = [
        [positions[index].copy()] for index in range(int(env.num_agents))
    ]
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    candidate_sets_for_plot: list[dict[str, Any]] = []
    proposal_times_ms: list[float] = []
    retained_counts: list[int] = []
    viable_counts: list[int] = []
    min_inter_agent_distance = float(reset_info["min_inter_agent_distance"])
    min_obstacle_distance = minimum_obstacle_clearance(
        positions, list(env.static_obstacles) + list(env.dynamic_obstacles)
    )
    initial_obstacles = [
        obstacle_snapshot(obstacle, dynamic=False)
        for obstacle in env.static_obstacles
    ] + [
        obstacle_snapshot(obstacle, dynamic=True)
        for obstacle in env.dynamic_obstacles
    ]
    dynamic_traces: list[list[np.ndarray]] = [
        [np.asarray(obstacle.center, dtype=float).copy()]
        for obstacle in env.dynamic_obstacles
    ]

    termination = "step_limit"
    collision_info = env._check_collision()
    obstacle_collision_event = bool(
        np.any(collision_info["obstacle_collision_mask"])
    )
    boundary_collision_event = bool(
        np.any(collision_info["boundary_collision_mask"])
    )
    inter_agent_collision_event = bool(
        np.any(collision_info["inter_agent_collision_mask"])
    )
    proposal_steps = 0
    for step in range(max_steps):
        proposal_steps = step + 1
        selected_points = positions.copy()
        next_velocities = velocities.copy()
        dead_end_agents: list[int] = []

        for agent_index in range(int(env.num_agents)):
            goal_distance = float(np.linalg.norm(goals[agent_index] - positions[agent_index]))
            if goal_distance <= float(env.env_config.goal_tolerance):
                reached[agent_index] = True
                next_velocities[agent_index] = 0.0
                continue

            sensor = env.sensors[agent_index]
            sensor_packet = sensor.sense(
                positions[agent_index],
                velocities[agent_index],
                goals[agent_index],
                env._sensor_static_obstacles(),
                env._sensor_dynamic_obstacles(agent_index),
            )
            proposal_start = time.perf_counter()
            proposals = propose_reference_points(
                positions[agent_index],
                goals[agent_index],
                velocities[agent_index],
                sensor_packet,
                sensor,
                proposal_config,
                float(env.env_config.goal_tolerance),
            )
            proposal_times_ms.append((time.perf_counter() - proposal_start) * 1000.0)
            viable_counts.append(len(proposals))
            retained_counts.append(min(len(proposals), proposal_config.top_k))
            record_candidate_rows(
                candidate_rows,
                seed=seed,
                stage_name=stage_name,
                step=step,
                agent_index=agent_index,
                origin=positions[agent_index],
                proposals=proposals,
                top_k=proposal_config.top_k,
            )

            if not proposals:
                dead_end_agents.append(agent_index)
                next_velocities[agent_index] = 0.0
                continue

            retained = proposals[: proposal_config.top_k]
            best = retained[0]
            selected_points[agent_index] = best.point.copy()
            next_velocities[agent_index] = (
                proposal_config.nominal_speed * normalize(best.direction)
            )
            candidate_sets_for_plot.append(
                {
                    "step": int(step),
                    "agent": int(agent_index),
                    "origin": positions[agent_index].copy(),
                    "points": np.stack(
                        [proposal.point.copy() for proposal in retained], axis=0
                    ),
                }
            )
            selected_rows.append(
                {
                    "seed": int(seed),
                    "stage": stage_name,
                    "step": int(step),
                    "agent": int(agent_index),
                    "point_x": float(best.point[0]),
                    "point_y": float(best.point[1]),
                    "point_z": float(best.point[2]),
                    "goal_distance_before": goal_distance,
                    "score": float(best.score),
                    "safety_margin": float(best.safety_margin),
                    "distance_progress": float(best.distance_progress),
                    "normalized_progress": float(best.normalized_progress),
                    "alignment": float(best.alignment),
                    "smoothness": float(best.smoothness),
                }
            )

        if dead_end_agents:
            termination = "dead_end"
            break

        positions = selected_points
        velocities = next_velocities
        for agent_index in range(int(env.num_agents)):
            trajectories[agent_index].append(positions[agent_index].copy())

        for obstacle_index, obstacle in enumerate(env.dynamic_obstacles):
            obstacle.step(float(env.dynamics[0].dt))
            dynamic_traces[obstacle_index].append(
                np.asarray(obstacle.center, dtype=float).copy()
            )

        set_environment_agent_states(env, positions, velocities)
        collision_info = env._check_collision()
        obstacle_collision_event = obstacle_collision_event or bool(
            np.any(collision_info["obstacle_collision_mask"])
        )
        boundary_collision_event = boundary_collision_event or bool(
            np.any(collision_info["boundary_collision_mask"])
        )
        inter_agent_collision_event = inter_agent_collision_event or bool(
            np.any(collision_info["inter_agent_collision_mask"])
        )
        min_inter_agent_distance = min(
            min_inter_agent_distance,
            float(collision_info["min_inter_agent_distance"]),
        )
        min_obstacle_distance = min(
            min_obstacle_distance,
            minimum_obstacle_clearance(
                positions, list(env.static_obstacles) + list(env.dynamic_obstacles)
            ),
        )
        reached = np.linalg.norm(goals - positions, axis=1) <= float(
            env.env_config.goal_tolerance
        )

        if bool(np.all(reached)):
            if (
                obstacle_collision_event
                or boundary_collision_event
                or inter_agent_collision_event
            ):
                termination = "completed_with_collision"
            else:
                termination = "success"
            break

    final_distances = np.linalg.norm(goals - positions, axis=1)
    path_lengths = np.array(
        [
            float(
                np.sum(
                    np.linalg.norm(
                        np.diff(np.asarray(agent_trajectory, dtype=float), axis=0),
                        axis=1,
                    )
                )
            )
            for agent_trajectory in trajectories
        ],
        dtype=float,
    )
    direct_lengths = np.linalg.norm(goals - np.asarray(reset_info["starts"], dtype=float), axis=1)
    completed_path_lengths = path_lengths + final_distances
    path_efficiencies = np.divide(
        direct_lengths,
        completed_path_lengths,
        out=np.zeros_like(direct_lengths),
        where=completed_path_lengths > EPS,
    )
    obstacle_collision = bool(obstacle_collision_event)
    boundary_collision = bool(boundary_collision_event)
    inter_agent_collision = bool(inter_agent_collision_event)
    success = bool(termination == "success")
    route_completed = bool(np.all(reached))
    reached_count = int(np.sum(reached))
    result = EpisodeResult(
        seed=int(seed),
        stage=stage_name,
        success=success,
        route_completed=route_completed,
        termination=termination,
        proposal_steps=int(proposal_steps),
        reached_agent_count=reached_count,
        agent_success_rate=float(reached_count / int(env.num_agents)),
        obstacle_collision=obstacle_collision,
        boundary_collision=boundary_collision,
        inter_agent_collision=inter_agent_collision,
        dead_end=bool(termination == "dead_end"),
        mean_path_length=float(np.mean(path_lengths)),
        mean_direct_length=float(np.mean(direct_lengths)),
        mean_path_efficiency=float(np.mean(path_efficiencies)),
        mean_final_goal_distance=float(np.mean(final_distances)),
        min_inter_agent_distance=float(min_inter_agent_distance),
        min_obstacle_clearance=float(min_obstacle_distance),
        mean_retained_candidate_count=float(np.mean(retained_counts)) if retained_counts else 0.0,
        mean_viable_candidate_count=float(np.mean(viable_counts)) if viable_counts else 0.0,
        mean_proposal_time_ms=float(np.mean(proposal_times_ms)) if proposal_times_ms else 0.0,
        max_proposal_time_ms=float(np.max(proposal_times_ms)) if proposal_times_ms else 0.0,
    )
    plot_data = {
        "seed": int(seed),
        "starts": np.asarray(reset_info["starts"], dtype=float),
        "goals": goals.copy(),
        "trajectories": [np.asarray(points, dtype=float) for points in trajectories],
        "candidate_sets": candidate_sets_for_plot,
        "initial_obstacles": initial_obstacles,
        "dynamic_traces": [
            np.asarray(points, dtype=float) for points in dynamic_traces
        ],
        "termination": termination,
    }
    return result, candidate_rows, selected_rows, plot_data


def draw_sphere(
    axis: Any,
    center: np.ndarray,
    radius: float,
    color: str,
    alpha: float,
) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 18)
    v = np.linspace(0.0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    axis.plot_surface(
        x,
        y,
        z,
        color=color,
        alpha=alpha,
        linewidth=0.0,
        shade=True,
    )


def draw_episode(
    plot_data: dict[str, Any],
    result: EpisodeResult,
    workspace_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    output_path: Path,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(13.5, 8.5))
    axis = fig.add_subplot(111, projection="3d")
    lower, upper = np.asarray(workspace_bounds, dtype=float)

    for obstacle in plot_data["initial_obstacles"]:
        center = np.asarray(obstacle["center"], dtype=float)
        color = "#FF6B6B" if obstacle["dynamic"] else "#6C7A89"
        if "half_extents" in obstacle:
            half = np.asarray(obstacle["half_extents"], dtype=float)
            axis.bar3d(
                center[0] - half[0],
                center[1] - half[1],
                center[2] - half[2],
                2.0 * half[0],
                2.0 * half[1],
                2.0 * half[2],
                color=color,
                alpha=0.25,
                shade=True,
            )
        else:
            draw_sphere(
                axis,
                center,
                float(obstacle["radius"]),
                color=color,
                alpha=0.28 if obstacle["dynamic"] else 0.22,
            )

    for trace in plot_data["dynamic_traces"]:
        if len(trace) > 1:
            axis.plot(
                trace[:, 0],
                trace[:, 1],
                trace[:, 2],
                color="#FF6B6B",
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
            )

    candidate_label_used = False
    for candidate_set in plot_data["candidate_sets"]:
        agent_index = int(candidate_set["agent"])
        color = AGENT_COLORS[agent_index % len(AGENT_COLORS)]
        origin = np.asarray(candidate_set["origin"], dtype=float)
        points = np.asarray(candidate_set["points"], dtype=float)
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            s=9,
            alpha=0.20,
            label="每轮 top-K 候选点" if not candidate_label_used else None,
        )
        candidate_label_used = True
        for point in points:
            axis.plot(
                [origin[0], point[0]],
                [origin[1], point[1]],
                [origin[2], point[2]],
                color=color,
                linewidth=0.35,
                alpha=0.08,
            )

    starts = np.asarray(plot_data["starts"], dtype=float)
    goals = np.asarray(plot_data["goals"], dtype=float)
    for agent_index, trajectory in enumerate(plot_data["trajectories"]):
        color = AGENT_COLORS[agent_index % len(AGENT_COLORS)]
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            color=color,
            linewidth=2.6,
            label=f"UAV {agent_index} 选择轨迹",
        )
        axis.scatter(
            starts[agent_index, 0],
            starts[agent_index, 1],
            starts[agent_index, 2],
            color=color,
            marker="o",
            s=55,
            edgecolors="black",
            linewidths=0.4,
        )
        axis.scatter(
            goals[agent_index, 0],
            goals[agent_index, 1],
            goals[agent_index, 2],
            color=color,
            marker="*",
            s=145,
            edgecolors="black",
            linewidths=0.5,
        )

    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(lower[2], upper[2])
    axis.set_box_aspect(upper - lower)
    axis.set_xlabel("X / m")
    axis.set_ylabel("Y / m")
    axis.set_zlabel("Z / m")
    axis.set_title(
        f"参考点提议轨迹 | seed={result.seed} | {result.termination} | "
        f"steps={result.proposal_steps}"
    )
    axis.grid(alpha=0.20)
    axis.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: list[float]) -> float | None:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else None


def aggregate_results(results: list[EpisodeResult]) -> dict[str, Any]:
    count = max(1, len(results))
    return {
        "episode_count": int(len(results)),
        "team_success_rate": float(np.mean([row.success for row in results])),
        "route_completion_rate": float(
            np.mean([row.route_completed for row in results])
        ),
        "agent_success_rate": float(np.mean([row.agent_success_rate for row in results])),
        "collision_rate": float(
            np.mean(
                [
                    row.obstacle_collision
                    or row.boundary_collision
                    or row.inter_agent_collision
                    for row in results
                ]
            )
        ),
        "obstacle_collision_rate": float(
            sum(row.obstacle_collision for row in results) / count
        ),
        "boundary_collision_rate": float(
            sum(row.boundary_collision for row in results) / count
        ),
        "inter_agent_collision_rate": float(
            sum(row.inter_agent_collision for row in results) / count
        ),
        "dead_end_rate": float(sum(row.dead_end for row in results) / count),
        "step_limit_rate": float(
            sum(row.termination == "step_limit" for row in results) / count
        ),
        "mean_proposal_steps": float(np.mean([row.proposal_steps for row in results])),
        "mean_path_efficiency": float(
            np.mean([row.mean_path_efficiency for row in results])
        ),
        "mean_final_goal_distance": float(
            np.mean([row.mean_final_goal_distance for row in results])
        ),
        "mean_min_inter_agent_distance": finite_mean(
            [row.min_inter_agent_distance for row in results]
        ),
        "mean_min_obstacle_clearance": finite_mean(
            [row.min_obstacle_clearance for row in results]
        ),
        "mean_retained_candidate_count": float(
            np.mean([row.mean_retained_candidate_count for row in results])
        ),
        "mean_viable_candidate_count": float(
            np.mean([row.mean_viable_candidate_count for row in results])
        ),
        "mean_proposal_time_ms": float(
            np.mean([row.mean_proposal_time_ms for row in results])
        ),
        "max_proposal_time_ms": float(
            np.max([row.max_proposal_time_ms for row in results])
        ),
        "termination_counts": {
            termination: int(sum(row.termination == termination for row in results))
            for termination in (
                "success",
                "completed_with_collision",
                "dead_end",
                "step_limit",
            )
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the reference-point proposal score in the current MASAC environment."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reference_point_proposal_env"),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=321)
    parser.add_argument(
        "--stage",
        choices=("phase1", "static", "final"),
        default="final",
        help="phase1: 无障碍；static: 最高静态难度；final: 当前最终动态难度。",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--nominal-speed", type=float, default=0.80)
    parser.add_argument("--safe-radius", type=float, default=1.15)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    config = MASAC_EXPERIMENT_CONFIG
    stage = build_stage(config, args.stage)
    env = build_environment(config, stage)
    alignment = verify_environment_alignment(env, config)
    proposal_config = ProposalConfig(
        top_k=int(args.top_k),
        nominal_speed=float(args.nominal_speed),
        safe_radius=float(args.safe_radius),
        terminal_safe_radius=0.5 * float(env.env_config.inter_agent_safe_distance),
        braking_deceleration=float(
            np.min(
                np.maximum(
                    np.abs(np.asarray(env.dynamics[0].accelerate_min, dtype=float)),
                    np.abs(np.asarray(env.dynamics[0].accelerate_max, dtype=float)),
                )
            )
        ),
        reaction_time=float(env.dynamics[0].dt),
        obstacle_motion_allowance=(
            float(config.curriculum_dynamic_speed_range[1])
            * float(env.dynamics[0].dt)
        ),
    )
    max_steps = int(args.max_steps or config.max_steps)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    output_dir = args.output.resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    results: list[EpisodeResult] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for episode_index in range(int(args.episodes)):
        seed = int(args.seed_base) + episode_index
        result, episode_candidates, episode_selected, plot_data = run_episode(
            env,
            seed=seed,
            stage_name=stage.name,
            proposal_config=proposal_config,
            max_steps=max_steps,
        )
        results.append(result)
        candidate_rows.extend(episode_candidates)
        selected_rows.extend(episode_selected)
        if not args.no_figures:
            draw_episode(
                plot_data,
                result,
                config.workspace_bounds,
                figures_dir / f"proposal_trajectory_seed_{seed}.png",
                show=bool(args.show),
            )
        if not args.quiet:
            print(
                f"[{episode_index + 1:02d}/{args.episodes:02d}] seed={seed} "
                f"termination={result.termination} reached={result.reached_agent_count}/{config.num_agents} "
                f"steps={result.proposal_steps} efficiency={result.mean_path_efficiency:.3f} "
                f"proposal={result.mean_proposal_time_ms:.3f} ms"
            )

    episode_rows = [asdict(result) for result in results]
    write_csv(output_dir / "episode_metrics.csv", episode_rows)
    write_csv(output_dir / "candidate_sets.csv", candidate_rows)
    write_csv(output_dir / "selected_trajectory.csv", selected_rows)

    summary = {
        "environment_alignment": alignment,
        "curriculum_stage": stage.to_dict(),
        "proposal_config": asdict(proposal_config),
        "max_proposal_steps": max_steps,
        "aggregate": aggregate_results(results),
        "outputs": {
            "episode_metrics": "episode_metrics.csv",
            "candidate_sets": "candidate_sets.csv",
            "selected_trajectory": "selected_trajectory.csv",
            "figures": None if args.no_figures else "figures/",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    env.close()
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"结果已保存至: {output_dir}")
    return summary


if __name__ == "__main__":
    run(parse_args())
