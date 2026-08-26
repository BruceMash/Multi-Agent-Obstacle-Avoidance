"""Sensing-matched classical baselines for the final four-stage benchmark.

The planners in this module inherit the frozen DWA/RVO parameters and control
laws.  Their only changed contract is obstacle/peer information.  Every agent
receives the same current local range packet consumed by the execution policy.
When the environment explicitly freezes ``local_anonymous_ally_block``, the
planner also receives that same range-limited, identity-free relative
position/velocity block.  Hit identity, hidden geometry, out-of-range state,
and private future state remain unavailable.  Visible range endpoints are
represented as stationary, zero-radius surface samples for collision queries.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from planning.final_four_stage_benchmark import (
    DWAStyleConfig,
    RVOStyleConfig,
    _clip_norm,
    _reciprocal_correction,
    _rotate_xy,
    _trajectory_metrics,
    step_direct_accelerations,
)


ADAPTER_SCHEMA_VERSION = "sensing_matched_lidar_surface_adapter_v1"
DYNAMIC_STATE_ESTIMATOR = "zero_order_hold_untyped_visible_surface_samples"
HIT_EPSILON = 1.0e-6


def _episode_termination_reason(
    *,
    success: bool,
    static_obstacle_collision: bool,
    dynamic_obstacle_collision: bool,
    obstacle_collision: bool,
    inter_agent_collision: bool,
    boundary_collision: bool,
    timeout: bool,
    planner_infeasible_count: int,
) -> str:
    """Return an exclusive, typed episode outcome without changing dynamics."""

    if success:
        return "success"
    if static_obstacle_collision:
        return "static_obstacle_collision"
    if dynamic_obstacle_collision:
        return "dynamic_obstacle_collision"
    if obstacle_collision:
        return "obstacle_collision"
    if inter_agent_collision:
        return "inter_agent_collision"
    if boundary_collision:
        return "boundary_collision"
    if timeout and planner_infeasible_count:
        return "planner_infeasible"
    if timeout:
        return "timeout"
    return "other"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def observation_equivalence_hash(env: Any, agent_id: int) -> str:
    """Hash only signals admitted by the per-step sensing-matched contract."""

    agent_id = int(agent_id)
    packet = env.latest_sensor_packets[agent_id]
    if packet is None:
        raise RuntimeError("environment must be reset before planner observation")
    sensor = env.sensors[agent_id]
    dynamic = env.dynamics[agent_id]
    payload = {
        "schema": ADAPTER_SCHEMA_VERSION,
        "ego_position": np.asarray(dynamic.p, dtype=float).tolist(),
        "ego_velocity": np.asarray(dynamic.v, dtype=float).tolist(),
        "terminal_goal": np.asarray(env.goals[agent_id], dtype=float).tolist(),
        "current_scan": np.asarray(packet.current_scan, dtype=float).tolist(),
        "previous_scan": np.asarray(packet.previous_scan, dtype=float).tolist(),
        "ray_directions": np.asarray(sensor.ray_directions, dtype=float).tolist(),
        "sensing_radius": float(sensor.sensing_radius),
    }
    local_peers = _local_anonymous_peers(env, agent_id)
    if local_peers:
        payload["local_anonymous_peers"] = [
            {
                "slot": int(peer.agent_id),
                "position": np.asarray(peer.position, dtype=float).tolist(),
                "velocity": np.asarray(peer.velocity, dtype=float).tolist(),
            }
            for peer in local_peers
        ]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ObservedSurfacePoint:
    """A deterministic primitive recoverable from one visible LiDAR return."""

    point: np.ndarray

    def __post_init__(self) -> None:
        point = np.asarray(self.point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("surface point must be a finite 3-vector")
        point = point.copy()
        point.setflags(write=False)
        object.__setattr__(self, "point", point)

    @property
    def center(self) -> np.ndarray:
        return self.point

    @property
    def velocity(self) -> np.ndarray:
        return np.zeros(3, dtype=float)

    @property
    def radius(self) -> float:
        return 0.0

    @property
    def safety_margin(self) -> float:
        return 0.0

    @property
    def effective_radius(self) -> float:
        return 0.0

    def signed_distance(self, query: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(query, dtype=float) - self.point))

    def closest_point(self, query: np.ndarray) -> np.ndarray:
        del query
        return self.point.copy()


@dataclass(frozen=True)
class SensingMatchedPerception:
    local_surfaces: tuple[tuple[ObservedSurfacePoint, ...], ...]
    local_anonymous_peers: tuple[tuple[Any, ...], ...]
    observation_hashes: tuple[str, ...]
    adapter_runtime_ms: float
    visible_hit_count: int
    observable_peer_count: int


def _local_anonymous_peers(env: Any, agent_id: int) -> tuple[Any, ...]:
    """Return only peer state authorized by the active local contract.

    Historical sensing-matched artifacts use the legacy environment mode and
    retain their LiDAR-only behavior.  The long-range benchmark opts into the
    local anonymous mode explicitly; only then may this adapter call the
    existing public observation accessor.
    """

    mode = str(getattr(env.env_config, "peer_state_observation_mode", ""))
    if mode != "local_anonymous_ally_block":
        return ()
    accessor = getattr(env, "observable_neighbor_states", None)
    if accessor is None:
        raise RuntimeError("local anonymous peer mode requires the public peer accessor")
    return tuple(accessor(int(agent_id)))


def _visible_surface_points(env: Any, agent_id: int) -> tuple[ObservedSurfacePoint, ...]:
    packet = env.latest_sensor_packets[int(agent_id)]
    if packet is None:
        raise RuntimeError("environment must be reset before perception reconstruction")
    sensor = env.sensors[int(agent_id)]
    scan = np.asarray(packet.current_scan, dtype=float)
    directions = np.asarray(sensor.ray_directions, dtype=float)
    if directions.shape != scan.shape + (3,):
        raise ValueError("LiDAR scan and direction organization mismatch")
    hit_mask = scan < (1.0 - HIT_EPSILON)
    distances = scan[hit_mask] * float(sensor.sensing_radius)
    origin = np.asarray(env.dynamics[int(agent_id)].p, dtype=float)
    endpoints = origin[None, :] + directions[hit_mask] * distances[:, None]
    return tuple(ObservedSurfacePoint(point) for point in endpoints.reshape(-1, 3))


def reconstruct_sensing_matched_perception(env: Any) -> SensingMatchedPerception:
    """Build local occupancy from public online observations only."""

    started = time.perf_counter_ns()
    rows = tuple(_visible_surface_points(env, agent_id) for agent_id in range(int(env.num_agents)))
    peer_rows = tuple(
        _local_anonymous_peers(env, agent_id) for agent_id in range(int(env.num_agents))
    )
    hashes = tuple(observation_equivalence_hash(env, agent_id) for agent_id in range(int(env.num_agents)))
    elapsed = (time.perf_counter_ns() - started) / 1.0e6
    return SensingMatchedPerception(
        local_surfaces=rows,
        local_anonymous_peers=peer_rows,
        observation_hashes=hashes,
        adapter_runtime_ms=float(elapsed),
        visible_hit_count=sum(len(row) for row in rows),
        observable_peer_count=sum(len(row) for row in peer_rows),
    )


def _surface_clearance(point: np.ndarray, obstacles: Iterable[ObservedSurfacePoint]) -> float:
    values = [obstacle.signed_distance(point) for obstacle in obstacles]
    return min(values, default=float("inf"))


def _dwa_candidate_scores_vectorized(
    *,
    position: np.ndarray,
    goal: np.ndarray,
    preferred: np.ndarray,
    candidates: np.ndarray,
    obstacle_points: np.ndarray,
    peer_positions: np.ndarray,
    peer_velocities: np.ndarray,
    dt: float,
    steps: int,
    obstacle_collision_distance: float,
    peer_collision_distance: float,
    sensing_radius: float,
    peer_influence_distance: float,
    config: DWAStyleConfig,
) -> np.ndarray:
    """Batch the unchanged DWA candidate/horizon distance arithmetic."""

    times = np.arange(1, int(steps) + 1, dtype=float) * float(dt)
    predicted = (
        np.asarray(position, dtype=float)[None, None, :]
        + np.asarray(candidates, dtype=float)[:, None, :] * times[None, :, None]
    )
    candidate_count = int(predicted.shape[0])
    if obstacle_points.size:
        obstacle_delta = predicted[:, :, None, :] - obstacle_points[None, None, :, :]
        minimum_obstacle = np.min(np.linalg.norm(obstacle_delta, axis=-1), axis=(1, 2))
    else:
        minimum_obstacle = np.full(candidate_count, float("inf"), dtype=float)
    if peer_positions.size:
        predicted_peers = (
            peer_positions[None, None, :, :]
            + peer_velocities[None, None, :, :] * times[None, :, None, None]
        )
        peer_delta = predicted[:, :, None, :] - predicted_peers
        minimum_peer = np.min(np.linalg.norm(peer_delta, axis=-1), axis=(1, 2))
    else:
        minimum_peer = np.full(candidate_count, float("inf"), dtype=float)
    final_positions = predicted[:, -1, :]
    goal_delta_norm = float(np.linalg.norm(np.asarray(goal, dtype=float) - position))
    progress = goal_delta_norm - np.linalg.norm(goal[None, :] - final_positions, axis=1)
    clearance_term = np.minimum(minimum_obstacle, float(sensing_radius))
    peer_term = np.minimum(minimum_peer, 2.0 * float(peer_influence_distance))
    candidate_norm = np.linalg.norm(candidates, axis=1)
    preferred_norm = float(np.linalg.norm(preferred))
    denominators = np.maximum(candidate_norm * preferred_norm, 1.0e-9)
    speed_alignment = np.sum(candidates * preferred[None, :], axis=1) / denominators
    collision = (minimum_obstacle <= float(obstacle_collision_distance)) | (
        minimum_peer <= float(peer_collision_distance)
    )
    return (
        float(config.goal_weight) * progress
        + float(config.clearance_weight) * clearance_term
        + float(config.peer_weight) * peer_term
        + float(config.speed_weight) * speed_alignment
        - 1.0e6 * collision.astype(float)
        - 1.0e-12 * np.arange(candidate_count, dtype=float)
    )


def _dwa_candidate_scores_scalar_reference(
    **kwargs: Any,
) -> np.ndarray:
    """Slow development-only oracle used to verify the batched implementation."""

    position = np.asarray(kwargs["position"], dtype=float)
    goal = np.asarray(kwargs["goal"], dtype=float)
    preferred = np.asarray(kwargs["preferred"], dtype=float)
    candidates = np.asarray(kwargs["candidates"], dtype=float)
    obstacle_points = np.asarray(kwargs["obstacle_points"], dtype=float).reshape(-1, 3)
    peer_positions = np.asarray(kwargs["peer_positions"], dtype=float).reshape(-1, 3)
    peer_velocities = np.asarray(kwargs["peer_velocities"], dtype=float).reshape(-1, 3)
    dt = float(kwargs["dt"])
    steps = int(kwargs["steps"])
    config = kwargs["config"]
    scores = []
    for candidate_id, candidate in enumerate(candidates):
        minimum_obstacle = float("inf")
        minimum_peer = float("inf")
        final_position = position.copy()
        for horizon_step in range(1, steps + 1):
            final_position = position + candidate * (horizon_step * dt)
            if obstacle_points.size:
                minimum_obstacle = min(
                    minimum_obstacle,
                    float(np.min(np.linalg.norm(obstacle_points - final_position, axis=1))),
                )
            if peer_positions.size:
                peers = peer_positions + peer_velocities * (horizon_step * dt)
                minimum_peer = min(
                    minimum_peer,
                    float(np.min(np.linalg.norm(peers - final_position, axis=1))),
                )
        collision = (
            minimum_obstacle <= float(kwargs["obstacle_collision_distance"])
            or minimum_peer <= float(kwargs["peer_collision_distance"])
        )
        progress = float(np.linalg.norm(goal - position) - np.linalg.norm(goal - final_position))
        clearance_term = min(minimum_obstacle, float(kwargs["sensing_radius"]))
        peer_term = min(minimum_peer, 2.0 * float(kwargs["peer_influence_distance"]))
        alignment = float(np.dot(candidate, preferred)) / max(
            float(np.linalg.norm(candidate) * np.linalg.norm(preferred)), 1.0e-9
        )
        scores.append(
            config.goal_weight * progress
            + config.clearance_weight * clearance_term
            + config.peer_weight * peer_term
            + config.speed_weight * alignment
            - (1.0e6 if collision else 0.0)
            - 1.0e-12 * candidate_id
        )
    return np.asarray(scores, dtype=float)


def dwa_sensing_matched_accelerations(
    env: Any, config: DWAStyleConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    """Frozen DWA logic under the active range-limited observation contract."""

    total_started = time.perf_counter_ns()
    perception = reconstruct_sensing_matched_perception(env)
    core_started = time.perf_counter_ns()
    dt = float(env.dynamics[0].dt)
    a_min = float(np.min(np.asarray(env.dynamics[0].accelerate_min)))
    a_max = float(np.max(np.asarray(env.dynamics[0].accelerate_max)))
    v_min = float(np.min(np.asarray(env.dynamics[0].velocity_min)))
    v_max = float(np.max(np.asarray(env.dynamics[0].velocity_max)))
    steps = max(1, int(round(float(config.horizon_s) / dt)))
    result = np.zeros((int(env.num_agents), 3), dtype=float)
    evaluated = 0
    infeasible_agents = 0
    for agent_id in range(int(env.num_agents)):
        if bool(env.success_rewarded_mask[agent_id]):
            continue
        position = np.asarray(env.dynamics[agent_id].p, dtype=float)
        current_v = np.asarray(env.dynamics[agent_id].v, dtype=float)
        goal = np.asarray(env.goals[agent_id], dtype=float)
        local_obstacles = perception.local_surfaces[agent_id]
        local_peers = perception.local_anonymous_peers[agent_id]
        low = np.maximum(v_min, current_v + a_min * dt)
        high = np.minimum(v_max, current_v + a_max * dt)
        axes = [
            np.linspace(low[dim], high[dim], int(config.velocity_samples_per_axis))
            for dim in range(3)
        ]
        candidates = np.asarray(np.meshgrid(*axes, indexing="ij"), dtype=float).reshape(3, -1).T
        goal_delta = goal - position
        preferred = _clip_norm(
            goal_delta / max(float(np.linalg.norm(goal_delta)), 1e-9)
            * config.preferred_speed_mps,
            v_max,
        )
        candidates = np.vstack([candidates, np.clip(preferred, low, high), np.zeros(3)])
        score_kwargs = {
            "position": position,
            "goal": goal,
            "preferred": preferred,
            "candidates": candidates,
            "obstacle_points": np.asarray(
                [surface.point for surface in local_obstacles], dtype=float
            ).reshape(-1, 3),
            "peer_positions": np.asarray(
                [peer.position for peer in local_peers], dtype=float
            ).reshape(-1, 3),
            "peer_velocities": np.asarray(
                [peer.velocity for peer in local_peers], dtype=float
            ).reshape(-1, 3),
            "dt": dt,
            "steps": steps,
            "obstacle_collision_distance": float(env.env_config.collision_margin)
            + float(config.collision_buffer_m),
            "peer_collision_distance": float(env.env_config.inter_agent_safe_distance)
            + float(config.collision_buffer_m),
            "sensing_radius": float(env.sensors[agent_id].sensing_radius),
            "peer_influence_distance": float(env.env_config.inter_agent_influence_distance),
            "config": config,
        }
        scores = _dwa_candidate_scores_vectorized(**score_kwargs)
        evaluated += int(len(candidates))
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        best_velocity = candidates[best_index].copy()
        if best_score < -5.0e5:
            infeasible_agents += 1
            best_velocity = np.zeros(3, dtype=float)
        result[agent_id] = (best_velocity - current_v) / dt
    result = np.clip(result, a_min, a_max)
    core_ms = (time.perf_counter_ns() - core_started) / 1.0e6
    total_ms = (time.perf_counter_ns() - total_started) / 1.0e6
    return result, {
        "planner": "DWA-SensingMatched",
        "runtime_ms": float(total_ms),
        "perception_adapter_runtime_ms": perception.adapter_runtime_ms,
        "planner_core_runtime_ms": float(core_ms),
        "candidate_velocity_count": int(evaluated),
        "infeasible_agent_count": int(infeasible_agents),
        "visible_hit_count": int(perception.visible_hit_count),
        "observable_peer_count": int(perception.observable_peer_count),
        "peer_information_mode": str(
            getattr(env.env_config, "peer_state_observation_mode", "")
        ),
        "observation_equivalence_hashes": perception.observation_hashes,
    }


def _project_sensing_matched_velocity_candidate(
    *,
    env: Any,
    agent_id: int,
    preferred: np.ndarray,
    corrected: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    local_obstacles: tuple[ObservedSurfacePoint, ...],
    config: RVOStyleConfig,
) -> np.ndarray:
    dt = float(env.dynamics[agent_id].dt)
    horizon = max(float(config.peer_time_horizon_s), float(config.obstacle_time_horizon_s))
    sample_times = np.linspace(dt, horizon, 10)
    base_directions = [corrected, preferred]
    preferred_norm = float(np.linalg.norm(preferred))
    for degrees in (18.0, -18.0, 36.0, -36.0, 60.0, -60.0, 90.0, -90.0):
        base_directions.append(_rotate_xy(preferred, math.radians(degrees)))
    vertical = max(0.45, 0.35 * preferred_norm)
    for sign in (-1.0, 1.0):
        row = preferred.copy()
        row[2] += sign * vertical
        base_directions.append(_clip_norm(row, config.preferred_speed_mps))
        for degrees in (30.0, -30.0, 60.0, -60.0):
            row = _rotate_xy(preferred, math.radians(degrees))
            row[2] += sign * vertical
            base_directions.append(_clip_norm(row, config.preferred_speed_mps))
    candidates = [np.zeros(3), velocity.copy()]
    for scale in (0.55, 0.78, 1.0):
        candidates.extend(
            _clip_norm(row * scale, config.preferred_speed_mps) for row in base_directions
        )
    best_score = -float("inf")
    best = np.zeros(3)
    for candidate_id, candidate in enumerate(candidates):
        minimum_obstacle = float("inf")
        collision = False
        final_position = position.copy()
        for sample_time in sample_times:
            predicted = position + candidate * float(sample_time)
            final_position = predicted
            minimum_obstacle = min(
                minimum_obstacle, _surface_clearance(predicted, local_obstacles)
            )
            collision |= minimum_obstacle <= config.safety_buffer_m
        goal_distance = float(np.linalg.norm(np.asarray(env.goals[agent_id]) - final_position))
        preference_cost = float(np.linalg.norm(candidate - preferred))
        acceleration_cost = float(np.linalg.norm(candidate - velocity))
        clearance_reward = (
            min(minimum_obstacle, 2.5) if math.isfinite(minimum_obstacle) else 2.5
        )
        peer_reward = 2.5
        score = (
            -1.6 * preference_cost
            - 0.45 * goal_distance
            - 0.06 * acceleration_cost
            + 0.45 * clearance_reward
            + 0.55 * peer_reward
            - (1.0e6 if collision else 0.0)
            - 1.0e-12 * candidate_id
        )
        if score > best_score:
            best_score = score
            best = candidate.copy()
    return best


def rvo_sensing_matched_accelerations(
    env: Any, config: RVOStyleConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    """Frozen RVO-style logic using only current untyped LiDAR surfaces."""

    total_started = time.perf_counter_ns()
    perception = reconstruct_sensing_matched_perception(env)
    core_started = time.perf_counter_ns()
    dt = float(env.dynamics[0].dt)
    a_min = float(np.min(np.asarray(env.dynamics[0].accelerate_min)))
    a_max = float(np.max(np.asarray(env.dynamics[0].accelerate_max)))
    v_max_component = float(np.max(np.asarray(env.dynamics[0].velocity_max)))
    desired = np.zeros((int(env.num_agents), 3), dtype=float)
    correction_count = 0
    for agent_id in range(int(env.num_agents)):
        if bool(env.success_rewarded_mask[agent_id]):
            continue
        position = np.asarray(env.dynamics[agent_id].p, dtype=float)
        current_velocity = np.asarray(env.dynamics[agent_id].v, dtype=float)
        goal_delta = np.asarray(env.goals[agent_id], dtype=float) - position
        preferred = _clip_norm(goal_delta, config.preferred_speed_mps)
        velocity = preferred.copy()
        local_obstacles = perception.local_surfaces[agent_id]
        for _ in range(int(config.iterations)):
            correction = np.zeros(3, dtype=float)
            # No peer identity/velocity track is present in the 122-D actor
            # input.  Visible peers are still avoided through their LiDAR
            # surface returns in this same obstacle loop.
            for obstacle in local_obstacles:
                clearance = obstacle.signed_distance(position)
                if clearance >= config.obstacle_time_horizon_s * max(
                    np.linalg.norm(velocity), 0.5
                ):
                    continue
                away = position - obstacle.closest_point(position)
                away /= max(float(np.linalg.norm(away)), 1e-9)
                correction += config.static_gain * away * max(
                    0.0,
                    (config.safety_buffer_m + 0.75 - clearance)
                    / max(config.obstacle_time_horizon_s, 1e-6),
                )
                correction_count += 1
            velocity = _clip_norm(preferred + correction, config.preferred_speed_mps)
        velocity = _project_sensing_matched_velocity_candidate(
            env=env,
            agent_id=agent_id,
            preferred=preferred,
            corrected=velocity,
            position=position,
            velocity=current_velocity,
            local_obstacles=local_obstacles,
            config=config,
        )
        desired[agent_id] = np.clip(velocity, -v_max_component, v_max_component)
    accelerations = np.clip((desired - np.stack([d.v for d in env.dynamics])) / dt, a_min, a_max)
    core_ms = (time.perf_counter_ns() - core_started) / 1.0e6
    total_ms = (time.perf_counter_ns() - total_started) / 1.0e6
    return accelerations, {
        "planner": "RVO-SensingMatched",
        "runtime_ms": float(total_ms),
        "perception_adapter_runtime_ms": perception.adapter_runtime_ms,
        "planner_core_runtime_ms": float(core_ms),
        "correction_count": int(correction_count),
        "infeasible_agent_count": 0,
        "visible_hit_count": int(perception.visible_hit_count),
        "observation_equivalence_hashes": perception.observation_hashes,
    }


def run_sensing_matched_episode(
    *,
    environment_builder: Any,
    multi_config: Any,
    scenario: str,
    seed: int,
    peer_radius: float,
    method: str,
    planner_config: DWAStyleConfig | RVOStyleConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if method not in {"dwa_sensing_matched", "rvo_sensing_matched"}:
        raise ValueError(method)
    env, scene_metadata = environment_builder(
        config=multi_config,
        scenario=str(scenario),
        seed=int(seed),
        peer_radius=float(peer_radius),
    )
    episode_started = time.perf_counter_ns()
    try:
        starts = np.asarray(env.starts, dtype=float).copy()
        goals = np.asarray(env.goals, dtype=float).copy()
        positions = [env._positions().copy()]
        velocities = [env._velocities().copy()]
        applied_accelerations: list[np.ndarray] = []
        min_clearance_by_agent = np.full(int(env.num_agents), float("inf"))
        min_peer_by_agent = np.full(int(env.num_agents), float("inf"))
        completion_steps: list[int | None] = [None] * int(env.num_agents)
        runtime_rows: list[dict[str, Any]] = []
        obstacle_collision = False
        static_obstacle_collision = False
        dynamic_obstacle_collision = False
        inter_agent_collision = False
        boundary_collision = False
        planner_infeasible_count = 0
        terminated = truncated = False
        last_info: dict[str, Any] = {}
        while not (terminated or truncated):
            if method == "dwa_sensing_matched":
                acceleration, planner_info = dwa_sensing_matched_accelerations(
                    env, planner_config  # type: ignore[arg-type]
                )
            else:
                acceleration, planner_info = rvo_sensing_matched_accelerations(
                    env, planner_config  # type: ignore[arg-type]
                )
            planner_infeasible_count += int(planner_info.get("infeasible_agent_count", 0))
            runtime_rows.append(
                {
                    "stage": scene_metadata.get("stage"),
                    "scenario_id": str(scenario),
                    "seed": int(seed),
                    "method": method,
                    "decision_index": int(env.steps),
                    **planner_info,
                }
            )
            terminated, truncated, last_info = step_direct_accelerations(env, acceleration)
            positions.append(env._positions().copy())
            velocities.append(env._velocities().copy())
            applied = np.asarray(last_info["applied_accelerations"], dtype=float)
            applied_accelerations.append(applied)
            min_clearance_by_agent = np.minimum(
                min_clearance_by_agent, np.asarray(last_info["min_clearances"], dtype=float)
            )
            pairwise = np.asarray(last_info["pairwise_distances"], dtype=float)
            for agent_id in range(int(env.num_agents)):
                peers = np.delete(pairwise[agent_id], agent_id)
                if peers.size:
                    min_peer_by_agent[agent_id] = min(
                        min_peer_by_agent[agent_id], float(np.min(peers))
                    )
                if completion_steps[agent_id] is None and bool(last_info["success_mask"][agent_id]):
                    completion_steps[agent_id] = int(env.steps)
            obstacle_collision |= bool(np.any(last_info["obstacle_collision_mask"]))
            static_obstacle_collision |= bool(
                np.any(last_info["static_obstacle_collision_mask"])
            )
            dynamic_obstacle_collision |= bool(
                np.any(last_info["dynamic_obstacle_collision_mask"])
            )
            inter_agent_collision |= bool(np.any(last_info["inter_agent_collision_mask"]))
            boundary_collision |= bool(np.any(last_info["boundary_collision_mask"]))
        position_array = np.stack(positions)
        velocity_array = np.stack(velocities)
        acceleration_array = np.stack(applied_accelerations)
        trajectory = _trajectory_metrics(
            position_array, velocity_array, acceleration_array, float(env.dynamics[0].dt)
        )
        success = bool(last_info.get("success", False))
        collision = bool(obstacle_collision or inter_agent_collision or boundary_collision)
        timeout = bool(truncated)
        termination_reason = _episode_termination_reason(
            success=success,
            static_obstacle_collision=static_obstacle_collision,
            dynamic_obstacle_collision=dynamic_obstacle_collision,
            obstacle_collision=obstacle_collision,
            inter_agent_collision=inter_agent_collision,
            boundary_collision=boundary_collision,
            timeout=timeout,
            planner_infeasible_count=planner_infeasible_count,
        )
        path_lengths = np.asarray(trajectory["path_lengths"], dtype=float)
        straight = np.linalg.norm(goals - starts, axis=1)
        final_collision_mask = np.asarray(last_info["collision_mask"], dtype=bool)
        final_static_collision_mask = np.asarray(
            last_info["static_obstacle_collision_mask"], dtype=bool
        )
        final_dynamic_collision_mask = np.asarray(
            last_info["dynamic_obstacle_collision_mask"], dtype=bool
        )
        final_peer_collision_mask = np.asarray(
            last_info["inter_agent_collision_mask"], dtype=bool
        )
        final_boundary_collision_mask = np.asarray(
            last_info["boundary_collision_mask"], dtype=bool
        )
        agent_rows: list[dict[str, Any]] = []
        for agent_id in range(int(env.num_agents)):
            completed = completion_steps[agent_id] is not None and not final_collision_mask[agent_id]
            agent_rows.append(
                {
                    "stage": scene_metadata.get("stage"),
                    "scenario_id": str(scenario),
                    "seed": int(seed),
                    "method": method,
                    "agent_id": int(agent_id),
                    "agent_terminal_completed": bool(completed),
                    "agent_collision": bool(final_collision_mask[agent_id]),
                    "agent_static_obstacle_collision": bool(
                        final_static_collision_mask[agent_id]
                    ),
                    "agent_dynamic_obstacle_collision": bool(
                        final_dynamic_collision_mask[agent_id]
                    ),
                    "agent_inter_agent_collision": bool(
                        final_peer_collision_mask[agent_id]
                    ),
                    "agent_boundary_collision": bool(
                        final_boundary_collision_mask[agent_id]
                    ),
                    "agent_path_length_m": float(path_lengths[agent_id]),
                    "agent_path_efficiency": (
                        float(straight[agent_id] / max(path_lengths[agent_id], 1e-9))
                        if completed
                        else None
                    ),
                    "completion_step": completion_steps[agent_id],
                    "minimum_obstacle_clearance_m": float(min_clearance_by_agent[agent_id]),
                    "minimum_peer_distance_m": float(min_peer_by_agent[agent_id]),
                }
            )
        total_runtime = float(sum(row["runtime_ms"] for row in runtime_rows))
        adapter_runtime = float(
            sum(row["perception_adapter_runtime_ms"] for row in runtime_rows)
        )
        core_runtime = float(sum(row["planner_core_runtime_ms"] for row in runtime_rows))
        episode = {
            "stage": scene_metadata.get("stage"),
            "family": scene_metadata.get("family"),
            "scenario_id": str(scenario),
            "seed": int(seed),
            "method": method,
            "team_success": success,
            "any_collision": collision,
            "obstacle_collision": obstacle_collision,
            "static_obstacle_collision": static_obstacle_collision,
            "dynamic_obstacle_collision": dynamic_obstacle_collision,
            "inter_agent_collision": inter_agent_collision,
            "boundary_collision": boundary_collision,
            "timeout": timeout,
            "termination_reason": termination_reason,
            "completion_step": int(env.steps) if success else None,
            "completion_time_s": float(env.steps * env.dynamics[0].dt) if success else None,
            "termination_time_s": float(env.steps * env.dynamics[0].dt),
            "steps": int(env.steps),
            "team_path_length_m": float(trajectory["path_length_team_sum"]),
            "team_path_length_mean_agent_m": float(trajectory["path_length_team_mean"]),
            "trajectory_smoothness": float(trajectory["trajectory_smoothness_team_mean"]),
            "minimum_obstacle_clearance_m": float(np.min(min_clearance_by_agent)),
            "minimum_inter_agent_distance_m": float(np.min(min_peer_by_agent)),
            "planning_runtime_ms": total_runtime,
            "perception_adapter_runtime_ms": adapter_runtime,
            "planner_core_runtime_ms": core_runtime,
            "planning_decision_count": len(runtime_rows),
            "planning_runtime_per_decision_ms": total_runtime / max(1, len(runtime_rows)),
            "planner_infeasible_count": int(planner_infeasible_count),
            "initial_condition_hash": scene_metadata.get("environment_fingerprint"),
            "scenario_manifest_hash": scene_metadata.get("environment_fingerprint"),
            "end_to_end_runtime_ms": (time.perf_counter_ns() - episode_started) / 1.0e6,
            "adapter_schema": ADAPTER_SCHEMA_VERSION,
            "dynamic_state_estimator": DYNAMIC_STATE_ESTIMATOR,
            "sensing_matched_parameter_retuning": False,
        }
        trajectory_payload = {
            "positions": position_array,
            "velocities": velocity_array,
            "accelerations": acceleration_array,
            "starts": starts,
            "goals": goals,
        }
        return episode, agent_rows, runtime_rows, trajectory_payload
    finally:
        env.close()
