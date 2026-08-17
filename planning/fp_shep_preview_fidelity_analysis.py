"""Evaluation-only primitives for the FP-SHEP 2x2 fidelity diagnosis.

This module deliberately does not alter the operational FP-SHEP entry point.
It reuses the checkpoint observation builder, deterministic actor inference,
the transition symbol routed by the historical-gate context, and the existing
LiDAR ray/geometry intersection implementation.  The refreshed branch changes
only static-obstacle sensing as the virtual ego state advances.  Peer-visible
surface samples remain frozen and no peer trajectory is predicted.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import planning.policy_preview as policy_preview_module
from Environment.frozen_sac_dmp_execution import (
    build_historical_actor_observation,
    predict_frozen_action,
)
from planning.policy_preview import (
    CandidatePreview,
    PreviewInitialState,
    PreviewLocalContext,
    PreviewPerformance,
    PreviewTrajectory,
    _known_clearance,
    _reconstruct_scan_from_frozen_surfaces,
    point_to_segment_distance,
)


SCHEMA_VERSION = "fp_shep_preview_fidelity_control_v1"
SENSING_FROZEN = "frozen_visible_surface"
SENSING_REFRESHED_STATIC = "refreshed_static_geometry"
REFRESHED_SENSING_SCOPE = "STATIC_GEOMETRY_ONLY"
H4_TERMINATION_RULE = "record_only_no_early_stop_formal_h4_anchor"
H20_TERMINATION_RULE = "stop_on_static_collision_or_terminal_success"


def _vector3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return result.copy()


def _surface_points_from_scan(
    *,
    position: np.ndarray,
    ray_directions: np.ndarray,
    normalized_scan: np.ndarray,
    sensing_radius: float,
    hit_epsilon: float,
) -> np.ndarray:
    scan = np.asarray(normalized_scan, dtype=float)
    directions = np.asarray(ray_directions, dtype=float)
    if directions.shape != scan.shape + (3,):
        raise ValueError("ray directions do not match scan organization")
    hit_mask = scan < 1.0 - float(hit_epsilon)
    distances = scan[hit_mask] * float(sensing_radius)
    return (
        np.asarray(position, dtype=float)[None, :]
        + directions[hit_mask] * distances[:, None]
    ).reshape(-1, 3)


def _project_frozen_surface_points(
    *,
    position: np.ndarray,
    surface_points: np.ndarray,
    ray_directions: np.ndarray,
    sensing_radius: float,
    scan_shape: Sequence[int],
) -> np.ndarray:
    """Reuse the operational nearest-ray projection for one typed point set."""

    template = PreviewLocalContext(
        current_scan=np.ones(tuple(scan_shape), dtype=np.float32),
        previous_scan=np.ones(tuple(scan_shape), dtype=np.float32),
        ray_directions=np.asarray(ray_directions, dtype=float),
        sensing_radius=float(sensing_radius),
        goal_distance_clip=1.0,
        visible_surface_points=np.asarray(surface_points, dtype=float).reshape(-1, 3),
    )
    return _reconstruct_scan_from_frozen_surfaces(position, template)


@dataclass(frozen=True)
class RefreshedStaticSensingContext:
    """Read-only geometry required by the refreshed-static preview branch."""

    sensor: Any
    static_obstacles: tuple[Any, ...]
    peer_visible_surface_points: np.ndarray
    initial_static_scan: np.ndarray
    initial_peer_scan: np.ndarray
    initial_combined_scan: np.ndarray
    source_identity_verified: bool
    scope: str = REFRESHED_SENSING_SCOPE

    def __post_init__(self) -> None:
        points = np.asarray(self.peer_visible_surface_points, dtype=float)
        static_scan = np.asarray(self.initial_static_scan, dtype=np.float32)
        peer_scan = np.asarray(self.initial_peer_scan, dtype=np.float32)
        combined = np.asarray(self.initial_combined_scan, dtype=np.float32)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("peer_visible_surface_points must have shape (N,3)")
        if static_scan.shape != peer_scan.shape or static_scan.shape != combined.shape:
            raise ValueError("typed initial scans must share one shape")
        object.__setattr__(self, "peer_visible_surface_points", points.copy())
        object.__setattr__(self, "initial_static_scan", static_scan.copy())
        object.__setattr__(self, "initial_peer_scan", peer_scan.copy())
        object.__setattr__(self, "initial_combined_scan", combined.copy())
        object.__setattr__(
            self,
            "static_obstacles",
            tuple(copy.deepcopy(list(self.static_obstacles))),
        )
        if str(self.scope) != REFRESHED_SENSING_SCOPE:
            raise ValueError("refreshed sensing scope must remain STATIC_GEOMETRY_ONLY")


def build_refreshed_static_sensing_context(
    env: Any,
    agent_index: int,
    *,
    identity_atol: float = 2.0e-7,
    hit_epsilon: float = 1.0e-6,
) -> RefreshedStaticSensingContext:
    """Separate only currently visible peer hits from the t=0 combined scan.

    Static geometry is available to this offline control experiment.  Peer
    geometry is used only at t=0 to identify which *already visible* combined
    LiDAR hits came from peers.  Those points are then frozen exactly like the
    operational approximation; hidden peer surfaces and future peer motion are
    never introduced.
    """

    agent_index = int(agent_index)
    if len(getattr(env, "dynamic_obstacles", ())) != 0:
        raise ValueError("refreshed-static diagnosis requires zero dynamic obstacles")
    packet = env.latest_sensor_packets[agent_index]
    if packet is None:
        raise RuntimeError("environment must be reset before sensing diagnosis")
    sensor = env.sensors[agent_index]
    position = np.asarray(env.dynamics[agent_index].p, dtype=float)
    static_obstacles = tuple(copy.deepcopy(env._sensor_static_obstacles()))
    peer_obstacles = tuple(copy.deepcopy(env._sensor_dynamic_obstacles(agent_index)))
    if len(peer_obstacles) != int(env.num_agents) - 1:
        raise RuntimeError("peer obstacle set does not match frozen peer-sphere assumption")

    static_raw = sensor._scan_obstacles(position, static_obstacles)
    peer_raw = sensor._scan_obstacles(position, peer_obstacles)
    combined_raw = np.minimum(static_raw, peer_raw)
    radius = float(sensor.sensing_radius)
    static_scan = np.clip(static_raw / radius, 0.0, 1.0).astype(np.float32)
    peer_scan = np.clip(peer_raw / radius, 0.0, 1.0).astype(np.float32)
    combined_scan = np.clip(combined_raw / radius, 0.0, 1.0).astype(np.float32)
    actual_scan = np.asarray(packet.current_scan, dtype=np.float32)
    identity_verified = bool(
        np.allclose(combined_scan, actual_scan, rtol=0.0, atol=float(identity_atol))
    )
    if not identity_verified:
        raise RuntimeError("typed t=0 static+peer raycast does not reproduce current LiDAR")

    # Static obstacles are iterated before peer obstacles in the real sensor;
    # strict inequality therefore reproduces the real tie behavior.
    peer_visible = np.logical_and(
        peer_scan < static_scan - float(identity_atol),
        peer_scan < 1.0 - float(hit_epsilon),
    )
    peer_points = _surface_points_from_scan(
        position=position,
        ray_directions=sensor.ray_directions,
        normalized_scan=np.where(peer_visible, peer_scan, 1.0),
        sensing_radius=radius,
        hit_epsilon=hit_epsilon,
    )
    return RefreshedStaticSensingContext(
        sensor=sensor,
        static_obstacles=static_obstacles,
        peer_visible_surface_points=peer_points,
        initial_static_scan=static_scan,
        initial_peer_scan=peer_scan,
        initial_combined_scan=combined_scan,
        source_identity_verified=True,
    )


def reconstruct_refreshed_static_scan(
    position: np.ndarray,
    context: RefreshedStaticSensingContext,
) -> np.ndarray:
    """Rebuild static LiDAR at a virtual ego position and retain frozen peers."""

    position = _vector3(position, "position")
    sensor = context.sensor
    static_raw = sensor._scan_obstacles(position, context.static_obstacles)
    static_scan = np.clip(
        static_raw / float(sensor.sensing_radius), 0.0, 1.0
    ).astype(np.float32)
    peer_scan = _project_frozen_surface_points(
        position=position,
        surface_points=context.peer_visible_surface_points,
        ray_directions=sensor.ray_directions,
        sensing_radius=float(sensor.sensing_radius),
        scan_shape=sensor.scan_shape,
    )
    return np.minimum(static_scan, peer_scan).astype(np.float32)


def sensor_derived_clearance(
    normalized_scan: np.ndarray,
    sensing_radius: float,
    *,
    hit_epsilon: float = 1.0e-6,
) -> float:
    """Return nearest modeled LiDAR surface, never unseen ground truth."""

    scan = np.asarray(normalized_scan, dtype=float)
    hit = scan < 1.0 - float(hit_epsilon)
    if not np.any(hit):
        return float("inf")
    return float(np.min(scan[hit]) * float(sensing_radius))


def _ground_truth_static_collision(
    *,
    position: np.ndarray,
    static_obstacles: Sequence[Any],
    collision_margin: float,
) -> bool:
    return bool(
        any(
            obstacle.contains(np.asarray(position, dtype=float), margin=float(collision_margin))
            for obstacle in static_obstacles
        )
    )


def diagnostic_preview_rollout(
    *,
    initial_state: PreviewInitialState,
    local_context: PreviewLocalContext,
    candidate_goal: np.ndarray,
    policy: Any,
    horizon: int,
    dmp_config: Any,
    dynamics: Any,
    static_obstacles: Sequence[Any],
    collision_margin: float,
    terminal_tolerance: float,
    sensing_mode: str,
    refreshed_context: RefreshedStaticSensingContext | None = None,
    stop_on_diagnostic_termination: bool = False,
) -> CandidatePreview:
    """Roll out one frozen candidate with explicit sensing-factor control.

    H=4 callers must set ``stop_on_diagnostic_termination=False`` so the A
    anchor remains numerically identical to the operational preview.  H=20 C/D
    callers use the same True setting.  Ground-truth collision fields are
    diagnostics only and never replace the sensor-derived clearance feature.
    """

    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if sensing_mode not in (SENSING_FROZEN, SENSING_REFRESHED_STATIC):
        raise ValueError(f"unsupported sensing mode: {sensing_mode}")
    if sensing_mode == SENSING_REFRESHED_STATIC and refreshed_context is None:
        raise ValueError("refreshed sensing requires a refreshed context")
    candidate_goal = _vector3(candidate_goal, "candidate_goal")

    position = initial_state.position.copy()
    velocity = initial_state.velocity.copy()
    phase = float(initial_state.phase)
    current_scan = local_context.current_scan.copy()
    previous_scan = local_context.previous_scan.copy()
    positions = [position.copy()]
    velocities = [velocity.copy()]
    phases = [phase]
    accelerations: list[np.ndarray] = []
    commanded_accelerations: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    current_scans: list[np.ndarray] = []
    previous_scans: list[np.ndarray] = []
    clearances: list[float] = []
    infos: list[dict[str, Any]] = []
    observation_ns = policy_ns = transition_ns = sensing_ns = 0
    first_collision_step: int | None = None
    termination_step: int | None = None
    termination_reason: str | None = None
    total_started = time.perf_counter_ns()

    for step_index in range(horizon):
        started = time.perf_counter_ns()
        observation = build_historical_actor_observation(
            velocity=velocity,
            active_goal=candidate_goal,
            position=position,
            current_scan=current_scan,
            previous_scan=previous_scan,
            goal_distance_clip=local_context.goal_distance_clip,
            phase=phase,
            k_alpha=dmp_config.K_alpha,
            k_beta=dmp_config.K_beta,
        )
        observation_ns += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        action = predict_frozen_action(policy, observation)
        policy_ns += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        # Resolve the symbol dynamically so the established exception-safe
        # historical transition context remains authoritative.
        transition = policy_preview_module.propagate_sac_dmp_action(
            position=position,
            velocity=velocity,
            phase=phase,
            active_goal=candidate_goal,
            terminal_goal=initial_state.task_goal,
            action=action,
            dmp_config=dmp_config,
            dynamics=dynamics,
        )
        transition_ns += time.perf_counter_ns() - started

        observations.append(observation)
        actions.append(action)
        current_scans.append(current_scan.copy())
        previous_scans.append(previous_scan.copy())
        accelerations.append(transition.applied_acceleration.copy())
        commanded_accelerations.append(transition.commanded_acceleration.copy())
        infos.append(dict(transition.controller_info))
        position = transition.position.copy()
        velocity = transition.velocity.copy()
        phase = float(transition.phase)
        positions.append(position.copy())
        velocities.append(velocity.copy())
        phases.append(phase)

        started = time.perf_counter_ns()
        if sensing_mode == SENSING_FROZEN:
            clearance = _known_clearance(position, local_context)
            next_scan = _reconstruct_scan_from_frozen_surfaces(position, local_context)
            clearance_source = local_context.clearance_source
            observation_model = local_context.observation_model
        else:
            assert refreshed_context is not None
            next_scan = reconstruct_refreshed_static_scan(position, refreshed_context)
            clearance = sensor_derived_clearance(
                next_scan, float(local_context.sensing_radius)
            )
            clearance_source = (
                "refreshed_static_lidar_raycast_plus_frozen_peer_visible_surfaces"
            )
            observation_model = "refreshed_static_geometry_frozen_peer_surface"
        sensing_ns += time.perf_counter_ns() - started
        clearances.append(float(clearance))

        collision = _ground_truth_static_collision(
            position=position,
            static_obstacles=static_obstacles,
            collision_margin=float(collision_margin),
        )
        if collision and first_collision_step is None:
            first_collision_step = step_index + 1
        terminal_success = bool(
            np.linalg.norm(initial_state.task_goal - position)
            <= float(terminal_tolerance)
        )
        previous_scan, current_scan = current_scan.copy(), next_scan.copy()
        if stop_on_diagnostic_termination and (collision or terminal_success):
            termination_step = step_index + 1
            termination_reason = (
                "static_obstacle_collision" if collision else "terminal_success"
            )
            break

    trajectory = PreviewTrajectory(
        candidate_goal=candidate_goal,
        positions=np.stack(positions),
        velocities=np.stack(velocities),
        accelerations=np.stack(accelerations),
        commanded_accelerations=np.stack(commanded_accelerations),
        phases=np.asarray(phases, dtype=float),
        observations=np.stack(observations).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        current_scans=np.stack(current_scans).astype(np.float32),
        previous_scans=np.stack(previous_scans).astype(np.float32),
        clearances=np.asarray(clearances, dtype=float),
        controller_infos=tuple(infos),
    )
    task_progress = float(
        np.linalg.norm(initial_state.task_goal - initial_state.position)
        - np.linalg.norm(initial_state.task_goal - trajectory.positions[-1])
    )
    min_clearance = float(np.min(trajectory.clearances))
    max_deviation = float(
        max(
            point_to_segment_distance(
                point, initial_state.position, candidate_goal
            )
            for point in trajectory.positions[1:]
        )
    )
    terminal_speed = float(np.linalg.norm(trajectory.velocities[-1]))
    total_ns = time.perf_counter_ns() - total_started
    performance = PreviewPerformance(
        observation_ms=observation_ns / 1.0e6,
        policy_ms=policy_ns / 1.0e6,
        transition_ms=transition_ns / 1.0e6,
        sensor_reconstruction_ms=sensing_ns / 1.0e6,
        total_ms=total_ns / 1.0e6,
        policy_calls=trajectory.horizon,
    )
    completed = trajectory.horizon == horizon
    finite_clearance = bool(np.isfinite(min_clearance))
    valid_mask = {
        "task_progress": bool(np.isfinite(task_progress)),
        "min_clearance": finite_clearance,
        "max_execution_deviation": bool(np.isfinite(max_deviation)),
        "terminal_speed": bool(np.isfinite(terminal_speed)),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "sensing_mode": sensing_mode,
        "REFRESHED_SENSING_SCOPE": (
            REFRESHED_SENSING_SCOPE
            if sensing_mode == SENSING_REFRESHED_STATIC
            else "NOT_APPLICABLE"
        ),
        "clearance_source": clearance_source,
        "clearance_is_approximate": True,
        "clearance_semantics": "SENSOR_DERIVED_MODELED_VISIBLE_SURFACE_ONLY",
        "ground_truth_collision_used_in_clearance": False,
        "observation_model": observation_model,
        "lidar_hit_source_available": sensing_mode == SENSING_REFRESHED_STATIC,
        "dynamic_entity_extrapolation": False,
        "peer_future_motion_prediction": False,
        "joint_multi_agent_rollout": False,
        "peer_visible_surface_assumption": "frozen_t0_visible_surface_samples",
        "forcing_gate_distance_source": "historical_vector_goal_eff_gate",
        "history_is_preview_local": True,
        "requested_horizon_steps": horizon,
        "effective_horizon_steps": trajectory.horizon,
        "effective_horizon_ratio": float(trajectory.horizon / horizon),
        "preview_completed": completed,
        "termination_reason": termination_reason or "completed_horizon",
        "diagnostic_termination_enabled": bool(stop_on_diagnostic_termination),
        "preview_ground_truth_collision": first_collision_step is not None,
        "preview_collision_step": first_collision_step,
        "preview_termination_step": termination_step,
        "feature_valid_mask": valid_mask,
        "feature_full_horizon_mask": {
            key: bool(value and completed) for key, value in valid_mask.items()
        },
        "obstacle_clearance_source": clearance_source,
        "obstacle_clearance_is_approximate": True,
        "boundary_clearance_source": "not_used_boundary_free",
        "boundary_clearance_is_approximate": False,
        "clearance_finite_mask": finite_clearance,
        "open_space_flag": not finite_clearance,
    }
    return CandidatePreview(
        trajectory=trajectory,
        task_progress=task_progress,
        min_clearance=min_clearance,
        max_execution_deviation=max_deviation,
        terminal_speed=terminal_speed,
        performance=performance,
        metadata=metadata,
    )


def post_hoc_three_feature_score(
    preview: CandidatePreview,
    normalization: Mapping[str, float],
) -> float:
    """Apply the frozen formal scales for description, never selection."""

    progress = float(
        np.clip(
            preview.task_progress / float(normalization["task_progress_scale"]),
            -1.0,
            1.0,
        )
    )
    clearance = (
        1.0
        if not np.isfinite(preview.min_clearance)
        else float(
            np.clip(
                preview.min_clearance / float(normalization["min_clearance_scale"]),
                0.0,
                1.0,
            )
        )
    )
    deviation = float(
        np.clip(
            preview.max_execution_deviation
            / float(normalization["max_execution_deviation_scale"]),
            0.0,
            1.0,
        )
    )
    return progress + clearance - deviation


def rank_biserial(success_values: Sequence[float], failure_values: Sequence[float]) -> float | None:
    """Return a fixed non-parametric separation statistic with tie handling."""

    success = np.asarray(success_values, dtype=float)
    failure = np.asarray(failure_values, dtype=float)
    success = success[np.isfinite(success)]
    failure = failure[np.isfinite(failure)]
    if success.size == 0 or failure.size == 0:
        return None
    greater = 0.0
    for first in success:
        greater += float(np.sum(first > failure))
        greater += 0.5 * float(np.sum(first == failure))
    auc = greater / float(success.size * failure.size)
    return float(2.0 * auc - 1.0)


def descriptive(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p50": None, "p95": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
    }


def classify_directional_fraction(
    improved_count: int,
    total_count: int,
    thresholds: Mapping[str, float],
) -> str:
    """Apply pre-registered diagnostic labels; these are not significance tests."""

    if int(total_count) <= 0 or int(improved_count) <= 0:
        return "NONE"
    fraction = float(improved_count) / float(total_count)
    if fraction >= float(thresholds["strong_fraction"]):
        return "STRONG"
    if fraction >= float(thresholds["moderate_fraction"]):
        return "MODERATE"
    return "WEAK"


__all__ = [
    "H4_TERMINATION_RULE",
    "H20_TERMINATION_RULE",
    "REFRESHED_SENSING_SCOPE",
    "SCHEMA_VERSION",
    "SENSING_FROZEN",
    "SENSING_REFRESHED_STATIC",
    "RefreshedStaticSensingContext",
    "build_refreshed_static_sensing_context",
    "classify_directional_fraction",
    "descriptive",
    "diagnostic_preview_rollout",
    "post_hoc_three_feature_score",
    "rank_biserial",
    "reconstruct_refreshed_static_scan",
    "sensor_derived_clearance",
]
