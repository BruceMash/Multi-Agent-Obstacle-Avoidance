"""Frozen-policy short-horizon execution preview (FP-SHEP).

The preview is deliberately local-information-only.  The current LiDAR packet
does not retain hit identity, obstacle type, or obstacle velocity.  Therefore
the first implementation freezes the currently visible hit surfaces in world
coordinates and reconstructs later preview scans from those samples.  It is an
explicit approximation, not an exact future local-observation model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np

from Controller.dmp_rl import DMPConfig
from Environment.frozen_sac_dmp_execution import (
    build_actor_observation,
    predict_policy_action,
    predict_frozen_actions,
    propagate_sac_dmp_action,
)


CLEARANCE_SOURCE = "frozen_lidar_surface_samples"
OBSERVATION_MODEL = "frozen_visible_surface_approximation"
BOUNDARY_CLEARANCE_SOURCE = "not_separately_available_from_untyped_lidar"
PREVIEW_TERMINATION_COMPLETED = "completed_horizon"
PREVIEW_FEATURE_NAMES = (
    "task_progress",
    "min_clearance",
    "max_execution_deviation",
    "terminal_speed",
)


def _vector3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result.copy()


@dataclass(frozen=True)
class PreviewInitialState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    phase: float
    active_goal: np.ndarray
    task_goal: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector3(self.position, "position"))
        object.__setattr__(self, "velocity", _vector3(self.velocity, "velocity"))
        object.__setattr__(self, "acceleration", _vector3(self.acceleration, "acceleration"))
        object.__setattr__(self, "active_goal", _vector3(self.active_goal, "active_goal"))
        object.__setattr__(self, "task_goal", _vector3(self.task_goal, "task_goal"))
        phase = float(self.phase)
        if not np.isfinite(phase):
            raise ValueError("phase must be finite")
        object.__setattr__(self, "phase", phase)


@dataclass(frozen=True)
class PreviewLocalContext:
    current_scan: np.ndarray
    previous_scan: np.ndarray
    ray_directions: np.ndarray
    sensing_radius: float
    goal_distance_clip: float
    visible_surface_points: np.ndarray
    clearance_source: str = CLEARANCE_SOURCE
    clearance_is_approximate: bool = True
    observation_model: str = OBSERVATION_MODEL
    lidar_hit_source_available: bool = False
    dynamic_entity_extrapolation: bool = False

    def __post_init__(self) -> None:
        current = np.asarray(self.current_scan, dtype=np.float32)
        previous = np.asarray(self.previous_scan, dtype=np.float32)
        directions = np.asarray(self.ray_directions, dtype=float)
        points = np.asarray(self.visible_surface_points, dtype=float)
        if current.shape != previous.shape:
            raise ValueError("current_scan and previous_scan shapes must match")
        if directions.shape != current.shape + (3,):
            raise ValueError("ray_directions must have shape current_scan.shape + (3,)")
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("visible_surface_points must have shape (M, 3)")
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(previous)):
            raise ValueError("scan history must be finite")
        if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(points)):
            raise ValueError("ray directions and visible surface points must be finite")
        sensing_radius = float(self.sensing_radius)
        goal_distance_clip = float(self.goal_distance_clip)
        if sensing_radius <= 0.0 or not np.isfinite(sensing_radius):
            raise ValueError("sensing_radius must be positive and finite")
        if goal_distance_clip <= 0.0 or not np.isfinite(goal_distance_clip):
            raise ValueError("goal_distance_clip must be positive and finite")
        object.__setattr__(self, "current_scan", current.copy())
        object.__setattr__(self, "previous_scan", previous.copy())
        object.__setattr__(self, "ray_directions", directions.copy())
        object.__setattr__(self, "visible_surface_points", points.copy())
        object.__setattr__(self, "sensing_radius", sensing_radius)
        object.__setattr__(self, "goal_distance_clip", goal_distance_clip)

    @classmethod
    def from_sensor_packet(
        cls,
        *,
        position: np.ndarray,
        sensor_packet: Any,
        sensor: Any,
        hit_epsilon: float = 1.0e-6,
    ) -> "PreviewLocalContext":
        position = _vector3(position, "position")
        current_scan = np.asarray(sensor_packet.current_scan, dtype=np.float32)
        directions = np.asarray(sensor.ray_directions, dtype=float)
        if directions.shape != current_scan.shape + (3,):
            raise ValueError("sensor ray organization does not match current_scan")
        hit_mask = current_scan < (1.0 - float(hit_epsilon))
        hit_distances = current_scan[hit_mask].astype(float) * float(sensor.sensing_radius)
        surface_points = position[None, :] + directions[hit_mask] * hit_distances[:, None]
        return cls(
            current_scan=current_scan,
            previous_scan=np.asarray(sensor_packet.previous_scan, dtype=np.float32),
            ray_directions=directions,
            sensing_radius=float(sensor.sensing_radius),
            goal_distance_clip=float(sensor.goal_distance_clip),
            visible_surface_points=surface_points.reshape(-1, 3),
        )


@dataclass(frozen=True)
class PreviewPerformance:
    observation_ms: float
    policy_ms: float
    transition_ms: float
    sensor_reconstruction_ms: float
    total_ms: float
    policy_calls: int


@dataclass(frozen=True)
class PreviewTrajectory:
    candidate_goal: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    commanded_accelerations: np.ndarray
    phases: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    current_scans: np.ndarray
    previous_scans: np.ndarray
    clearances: np.ndarray
    controller_infos: tuple[dict[str, Any], ...]

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class CandidatePreview:
    trajectory: PreviewTrajectory
    task_progress: float
    min_clearance: float
    max_execution_deviation: float
    terminal_speed: float
    performance: PreviewPerformance
    metadata: dict[str, Any] = field(default_factory=dict)


def adapt_candidate_proposals(
    proposals: Iterable[Any],
    *,
    consumer_top_k: int | None = None,
) -> list[Any]:
    """Preserve the existing consumer's ordering and explicit truncation.

    There is intentionally no default K.  ``None`` represents consumers that
    inspect the complete ordered proposal list; an integer represents the
    runtime ``ProposalConfig.top_k`` used by existing top-K consumers.
    """
    ordered = list(proposals)
    if consumer_top_k is None:
        return ordered
    consumer_top_k = int(consumer_top_k)
    if consumer_top_k <= 0:
        raise ValueError("consumer_top_k must be positive")
    return ordered[: min(len(ordered), consumer_top_k)]


def build_preview_inputs_from_env(env: Any, agent_index: int) -> tuple[PreviewInitialState, PreviewLocalContext]:
    """Read only currently available agent/DMP/LiDAR state from an environment."""
    agent_index = int(agent_index)
    packet = env.latest_sensor_packets[agent_index]
    if packet is None:
        raise RuntimeError("environment must be reset before preview")
    dynamics = env.dynamics[agent_index]
    dmp = env.dmps[agent_index]
    controller_info = env.latest_controller_infos[agent_index]
    applied = np.asarray(
        controller_info.get("applied_acceleration", np.zeros(3, dtype=float)),
        dtype=float,
    )
    if applied.shape != (3,):
        applied = np.zeros(3, dtype=float)
    state = PreviewInitialState(
        position=dynamics.p,
        velocity=dynamics.v,
        acceleration=applied,
        phase=dmp.phase,
        active_goal=dmp.goal,
        task_goal=env.goals[agent_index],
    )
    context = PreviewLocalContext.from_sensor_packet(
        position=dynamics.p,
        sensor_packet=packet,
        sensor=env.sensors[agent_index],
    )
    return state, context


def _reconstruct_scan_from_frozen_surfaces(
    position: np.ndarray,
    context: PreviewLocalContext,
) -> np.ndarray:
    """Project frozen visible surface samples into the nearest current ray."""
    scan = np.ones(context.current_scan.shape, dtype=np.float32)
    points = context.visible_surface_points
    if points.shape[0] == 0:
        return scan
    relative = points - np.asarray(position, dtype=float)[None, :]
    ranges = np.linalg.norm(relative, axis=1)
    valid = np.logical_and(ranges > 1.0e-9, ranges <= context.sensing_radius)
    if not np.any(valid):
        return scan
    unit = relative[valid] / ranges[valid, None]
    flat_directions = context.ray_directions.reshape(-1, 3)
    similarities = unit @ flat_directions.T
    nearest = np.argmax(similarities, axis=1)
    normalized_ranges = np.clip(ranges[valid] / context.sensing_radius, 0.0, 1.0)
    flat_scan = scan.reshape(-1)
    for ray_index, normalized_range in zip(nearest, normalized_ranges, strict=True):
        flat_scan[int(ray_index)] = min(flat_scan[int(ray_index)], float(normalized_range))
    return scan


def _known_clearance(position: np.ndarray, context: PreviewLocalContext) -> float:
    if context.visible_surface_points.shape[0] == 0:
        return float("inf")
    distances = np.linalg.norm(context.visible_surface_points - position[None, :], axis=1)
    if not np.all(np.isfinite(distances)):
        raise ValueError("known-surface clearance produced a non-finite distance")
    return float(np.min(distances))


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    point = _vector3(point, "point")
    start = _vector3(start, "segment start")
    end = _vector3(end, "segment end")
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= 1.0e-16:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, segment) / length_squared, 0.0, 1.0))
    projection = start + fraction * segment
    return float(np.linalg.norm(point - projection))


def _append_task_aware_context(
    observation: np.ndarray,
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    task_goal: np.ndarray,
    current_scan: np.ndarray,
    local_context: PreviewLocalContext,
    switch_age_s: float,
    previous_applied_acceleration: np.ndarray,
    extension: dict[str, Any] | None,
) -> np.ndarray:
    """Append the same nine legal features used by task-aware SAC execution.

    The helper is opt-in.  Existing FP-SHEP callers therefore retain their
    byte-for-byte 522-D observation contract.  For an expanded SAC checkpoint,
    the preview branch treats the candidate as newly accepted (age starts at
    zero) and rolls previous acceleration forward from the cloned live state.
    """

    if extension is None:
        return np.asarray(observation, dtype=np.float32)
    from Guidance.reference_point_proposal_demo import (
        ProposalConfig,
        compute_sector_safety_field,
    )

    task_delta = np.asarray(task_goal, dtype=float) - np.asarray(position, dtype=float)
    task_distance = float(np.linalg.norm(task_delta))
    task_direction = (
        np.zeros(3, dtype=float)
        if task_distance < 1.0e-8
        else task_delta / task_distance
    )
    scan = np.asarray(current_scan, dtype=np.float32)
    sensor = SimpleNamespace(
        ray_directions=np.asarray(local_context.ray_directions, dtype=float),
        scan_shape=scan.shape,
        azimuth_bins=int(scan.shape[0]),
        elevation_bins=int(scan.shape[1]),
        sensing_radius=float(local_context.sensing_radius),
    )
    packet = SimpleNamespace(current_scan=scan)
    safety = compute_sector_safety_field(
        position,
        task_goal,
        velocity,
        packet,
        sensor,
        ProposalConfig(),
        float(extension.get("goal_tolerance_m", 0.25)),
    )
    directions = np.asarray(local_context.ray_directions, dtype=float).reshape(-1, 3)
    direction_index = int(np.argmax(directions @ task_direction))
    task_safety = float(safety.normalized_margin.reshape(-1)[direction_index])
    context = np.concatenate(
        [
            task_direction.astype(np.float32),
            np.asarray(
                [
                    np.clip(
                        task_distance
                        / float(extension.get("task_distance_scale_m", 100.0)),
                        0.0,
                        1.0,
                    ),
                    task_safety,
                    np.clip(
                        float(switch_age_s)
                        / float(extension.get("switch_age_scale_s", 5.0)),
                        0.0,
                        1.0,
                    ),
                ],
                dtype=np.float32,
            ),
            np.clip(
                np.asarray(previous_applied_acceleration, dtype=float)
                / float(extension.get("acceleration_scale_mps2", 4.0)),
                -1.0,
                1.0,
            ).astype(np.float32),
        ]
    )
    expanded = np.concatenate([np.asarray(observation, dtype=np.float32), context])
    if expanded.shape != (531,):
        raise RuntimeError(f"task-aware preview observation must be 531-D, got {expanded.shape}")
    return expanded.astype(np.float32)


def preview_candidate(
    *,
    initial_state: PreviewInitialState,
    local_context: PreviewLocalContext,
    candidate_goal: np.ndarray,
    policy: Any,
    horizon: int,
    dmp_config: DMPConfig,
    dynamics: Any,
    debug: bool = False,
    observation_extension: dict[str, Any] | None = None,
) -> CandidatePreview:
    """Execute one independent H-step closed-loop candidate branch."""
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    candidate_goal = _vector3(candidate_goal, "candidate_goal")
    position = initial_state.position.copy()
    velocity = initial_state.velocity.copy()
    phase = float(initial_state.phase)
    current_scan = local_context.current_scan.copy()
    previous_scan = local_context.previous_scan.copy()
    previous_applied_acceleration = initial_state.acceleration.copy()

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
    controller_infos: list[dict[str, Any]] = []
    observation_ns = policy_ns = transition_ns = sensor_ns = 0
    total_start = time.perf_counter_ns()

    if debug:
        print("=== FP-SHEP ===")
        print(f"candidate: {candidate_goal}")
        print(f"initial position: {position}")
        print(f"initial velocity: {velocity}")
        print("initial DMP state: phase-only controller (no separate z state)")
        print(f"initial phase: {phase}")

    for step in range(horizon):
        started = time.perf_counter_ns()
        observation = build_actor_observation(
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
        observation = _append_task_aware_context(
            observation,
            position=position,
            velocity=velocity,
            task_goal=initial_state.task_goal,
            current_scan=current_scan,
            local_context=local_context,
            switch_age_s=float(step) * float(dynamics.dt),
            previous_applied_acceleration=previous_applied_acceleration,
            extension=observation_extension,
        )
        observation_ns += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        action = predict_policy_action(policy, observation)
        policy_ns += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        transition = propagate_sac_dmp_action(
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
        controller_infos.append(transition.controller_info)
        previous_applied_acceleration = transition.applied_acceleration.copy()
        position = transition.position.copy()
        velocity = transition.velocity.copy()
        phase = transition.phase
        positions.append(position.copy())
        velocities.append(velocity.copy())
        phases.append(phase)
        clearances.append(_known_clearance(position, local_context))

        started = time.perf_counter_ns()
        next_scan = _reconstruct_scan_from_frozen_surfaces(position, local_context)
        sensor_ns += time.perf_counter_ns() - started
        previous_scan, current_scan = current_scan.copy(), next_scan

        if debug:
            info = transition.controller_info
            print(f"step {step + 1}:")
            print(f"    observation: {observation}")
            print(f"    SAC action: {action}")
            print(f"    goal modulation: {info['goal_offset']}")
            print(f"    forcing: {info['forcing']}")
            print(f"    acceleration: {transition.applied_acceleration}")
            print(f"    position: {position}")
            print(f"    velocity: {velocity}")
            print(f"    phase: {phase}")

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
        controller_infos=tuple(controller_infos),
    )
    initial_task_distance = float(np.linalg.norm(initial_state.task_goal - initial_state.position))
    terminal_task_distance = float(np.linalg.norm(initial_state.task_goal - trajectory.positions[-1]))
    task_progress = initial_task_distance - terminal_task_distance
    min_clearance = float(np.min(trajectory.clearances))
    deviations = [
        point_to_segment_distance(point, initial_state.position, candidate_goal)
        for point in trajectory.positions[1:]
    ]
    max_deviation = float(max(deviations))
    terminal_speed = float(np.linalg.norm(trajectory.velocities[-1]))
    clearance_finite = bool(np.isfinite(min_clearance))
    feature_valid_mask = {
        "task_progress": bool(np.isfinite(task_progress)),
        "min_clearance": clearance_finite,
        "max_execution_deviation": bool(np.isfinite(max_deviation)),
        "terminal_speed": bool(np.isfinite(terminal_speed)),
    }
    total_ns = time.perf_counter_ns() - total_start
    performance = PreviewPerformance(
        observation_ms=observation_ns / 1.0e6,
        policy_ms=policy_ns / 1.0e6,
        transition_ms=transition_ns / 1.0e6,
        sensor_reconstruction_ms=sensor_ns / 1.0e6,
        total_ms=total_ns / 1.0e6,
        policy_calls=horizon,
    )
    metadata = {
        "clearance_source": local_context.clearance_source,
        "clearance_is_approximate": local_context.clearance_is_approximate,
        "observation_model": local_context.observation_model,
        "lidar_hit_source_available": local_context.lidar_hit_source_available,
        "dynamic_entity_extrapolation": local_context.dynamic_entity_extrapolation,
        "forcing_gate_distance_source": "terminal_task_goal",
        "boundary_constraint_added": False,
        "history_is_preview_local": True,
        "requested_horizon_steps": horizon,
        "effective_horizon_steps": trajectory.horizon,
        "effective_horizon_ratio": float(trajectory.horizon / horizon),
        "preview_completed": True,
        "termination_reason": PREVIEW_TERMINATION_COMPLETED,
        "feature_valid_mask": feature_valid_mask,
        "feature_full_horizon_mask": feature_valid_mask.copy(),
        "obstacle_clearance_source": local_context.clearance_source,
        "obstacle_clearance_is_approximate": local_context.clearance_is_approximate,
        "boundary_clearance_source": BOUNDARY_CLEARANCE_SOURCE,
        "boundary_clearance_is_approximate": True,
        "clearance_finite_mask": clearance_finite,
        "open_space_flag": bool(local_context.visible_surface_points.shape[0] == 0),
    }
    if debug:
        print("features:")
        print(f"    task_progress: {task_progress}")
        print(f"    min_clearance: {min_clearance}")
        print(f"    max_execution_deviation: {max_deviation}")
        print(f"    terminal_speed: {terminal_speed}")
    return CandidatePreview(
        trajectory=trajectory,
        task_progress=task_progress,
        min_clearance=min_clearance,
        max_execution_deviation=max_deviation,
        terminal_speed=terminal_speed,
        performance=performance,
        metadata=metadata,
    )


def preview_candidates(
    *,
    initial_state: PreviewInitialState,
    local_context: PreviewLocalContext,
    candidates: Sequence[Any],
    policy: Any,
    horizon: int,
    dmp_config: DMPConfig,
    dynamics: Any,
    debug_candidate_index: int | None = None,
    observation_extension: dict[str, Any] | None = None,
) -> list[CandidatePreview]:
    """Preview K_t candidates; each call receives an independent history copy."""
    results: list[CandidatePreview] = []
    for index, candidate in enumerate(candidates):
        candidate_goal = getattr(candidate, "point", candidate)
        results.append(
            preview_candidate(
                initial_state=initial_state,
                local_context=local_context,
                candidate_goal=candidate_goal,
                policy=policy,
                horizon=horizon,
                dmp_config=dmp_config,
                dynamics=dynamics,
                debug=debug_candidate_index == index,
                observation_extension=observation_extension,
            )
        )
    return results


def preview_candidates_batched(
    *,
    initial_state: PreviewInitialState,
    local_context: PreviewLocalContext,
    candidates: Sequence[Any],
    policy: Any,
    horizon: int,
    dmp_config: DMPConfig,
    dynamics: Any,
    timing_sink: dict[str, float] | None = None,
    observation_extension: dict[str, Any] | None = None,
) -> list[CandidatePreview]:
    """Preview one agent's candidate branches with one actor batch per step.

    This is an exact-semantics engineering path: every branch keeps independent
    position, velocity, phase, scan history, DMP transition, and clearance
    history.  Only the frozen actor calls are grouped along the batch dimension.
    Candidate order and the serial H-step dependency within each branch are
    unchanged.
    """

    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    goals = [
        _vector3(getattr(candidate, "point", candidate), f"candidate_goal[{index}]")
        for index, candidate in enumerate(candidates)
    ]
    branch_count = len(goals)
    if branch_count == 0:
        if timing_sink is not None:
            timing_sink.update(
                {
                    "observation_ms": 0.0,
                    "actor_forward_ms": 0.0,
                    "dmp_rollout_ms": 0.0,
                    "sensor_reconstruction_ms": 0.0,
                    "geometry_metric_ms": 0.0,
                    "total_ms": 0.0,
                    "actor_batch_calls": 0.0,
                    "preview_branch_count": 0.0,
                }
            )
        return []

    total_started = time.perf_counter_ns()
    positions = [initial_state.position.copy() for _ in goals]
    velocities = [initial_state.velocity.copy() for _ in goals]
    phases = [float(initial_state.phase) for _ in goals]
    current_scans = [local_context.current_scan.copy() for _ in goals]
    previous_scans = [local_context.previous_scan.copy() for _ in goals]
    previous_applied_accelerations = [initial_state.acceleration.copy() for _ in goals]

    position_rows = [[value.copy()] for value in positions]
    velocity_rows = [[value.copy()] for value in velocities]
    phase_rows = [[value] for value in phases]
    acceleration_rows: list[list[np.ndarray]] = [[] for _ in goals]
    commanded_rows: list[list[np.ndarray]] = [[] for _ in goals]
    observation_rows: list[list[np.ndarray]] = [[] for _ in goals]
    action_rows: list[list[np.ndarray]] = [[] for _ in goals]
    current_scan_rows: list[list[np.ndarray]] = [[] for _ in goals]
    previous_scan_rows: list[list[np.ndarray]] = [[] for _ in goals]
    clearance_rows: list[list[float]] = [[] for _ in goals]
    controller_rows: list[list[dict[str, Any]]] = [[] for _ in goals]

    observation_ns = actor_ns = transition_ns = sensor_ns = 0
    for _ in range(horizon):
        started = time.perf_counter_ns()
        observations = np.stack(
            [
                _append_task_aware_context(
                    build_actor_observation(
                    velocity=velocities[index],
                    active_goal=goals[index],
                    position=positions[index],
                    current_scan=current_scans[index],
                    previous_scan=previous_scans[index],
                    goal_distance_clip=local_context.goal_distance_clip,
                    phase=phases[index],
                    k_alpha=dmp_config.K_alpha,
                    k_beta=dmp_config.K_beta,
                    ),
                    position=positions[index],
                    velocity=velocities[index],
                    task_goal=initial_state.task_goal,
                    current_scan=current_scans[index],
                    local_context=local_context,
                    switch_age_s=float(len(acceleration_rows[index])) * float(dynamics.dt),
                    previous_applied_acceleration=previous_applied_accelerations[index],
                    extension=observation_extension,
                )
                for index in range(branch_count)
            ]
        ).astype(np.float32, copy=False)
        observation_ns += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        actions = predict_frozen_actions(
            policy,
            observations,
            expected_shape=(branch_count, 6),
        )
        actor_ns += time.perf_counter_ns() - started

        for index in range(branch_count):
            started = time.perf_counter_ns()
            transition = propagate_sac_dmp_action(
                position=positions[index],
                velocity=velocities[index],
                phase=phases[index],
                active_goal=goals[index],
                terminal_goal=initial_state.task_goal,
                action=actions[index],
                dmp_config=dmp_config,
                dynamics=dynamics,
            )
            transition_ns += time.perf_counter_ns() - started

            observation_rows[index].append(observations[index].copy())
            action_rows[index].append(np.asarray(actions[index], dtype=np.float32).copy())
            current_scan_rows[index].append(current_scans[index].copy())
            previous_scan_rows[index].append(previous_scans[index].copy())
            acceleration_rows[index].append(transition.applied_acceleration.copy())
            commanded_rows[index].append(transition.commanded_acceleration.copy())
            controller_rows[index].append(transition.controller_info)
            previous_applied_accelerations[index] = transition.applied_acceleration.copy()
            positions[index] = transition.position.copy()
            velocities[index] = transition.velocity.copy()
            phases[index] = float(transition.phase)
            position_rows[index].append(positions[index].copy())
            velocity_rows[index].append(velocities[index].copy())
            phase_rows[index].append(phases[index])
            clearance_rows[index].append(_known_clearance(positions[index], local_context))

            started = time.perf_counter_ns()
            next_scan = _reconstruct_scan_from_frozen_surfaces(
                positions[index], local_context
            )
            sensor_ns += time.perf_counter_ns() - started
            previous_scans[index], current_scans[index] = (
                current_scans[index].copy(),
                next_scan,
            )

    geometry_started = time.perf_counter_ns()
    results: list[CandidatePreview] = []
    for index, goal in enumerate(goals):
        trajectory = PreviewTrajectory(
            candidate_goal=goal,
            positions=np.stack(position_rows[index]),
            velocities=np.stack(velocity_rows[index]),
            accelerations=np.stack(acceleration_rows[index]),
            commanded_accelerations=np.stack(commanded_rows[index]),
            phases=np.asarray(phase_rows[index], dtype=float),
            observations=np.stack(observation_rows[index]).astype(np.float32),
            actions=np.stack(action_rows[index]).astype(np.float32),
            current_scans=np.stack(current_scan_rows[index]).astype(np.float32),
            previous_scans=np.stack(previous_scan_rows[index]).astype(np.float32),
            clearances=np.asarray(clearance_rows[index], dtype=float),
            controller_infos=tuple(controller_rows[index]),
        )
        initial_task_distance = float(
            np.linalg.norm(initial_state.task_goal - initial_state.position)
        )
        terminal_task_distance = float(
            np.linalg.norm(initial_state.task_goal - trajectory.positions[-1])
        )
        task_progress = initial_task_distance - terminal_task_distance
        min_clearance = float(np.min(trajectory.clearances))
        deviations = [
            point_to_segment_distance(point, initial_state.position, goal)
            for point in trajectory.positions[1:]
        ]
        max_deviation = float(max(deviations))
        terminal_speed = float(np.linalg.norm(trajectory.velocities[-1]))
        clearance_finite = bool(np.isfinite(min_clearance))
        feature_valid_mask = {
            "task_progress": bool(np.isfinite(task_progress)),
            "min_clearance": clearance_finite,
            "max_execution_deviation": bool(np.isfinite(max_deviation)),
            "terminal_speed": bool(np.isfinite(terminal_speed)),
        }
        per_branch = 1.0 / float(branch_count)
        performance = PreviewPerformance(
            observation_ms=observation_ns / 1.0e6 * per_branch,
            policy_ms=actor_ns / 1.0e6 * per_branch,
            transition_ms=transition_ns / 1.0e6 * per_branch,
            sensor_reconstruction_ms=sensor_ns / 1.0e6 * per_branch,
            total_ms=0.0,
            policy_calls=horizon,
        )
        metadata = {
            "clearance_source": local_context.clearance_source,
            "clearance_is_approximate": local_context.clearance_is_approximate,
            "observation_model": local_context.observation_model,
            "lidar_hit_source_available": local_context.lidar_hit_source_available,
            "dynamic_entity_extrapolation": local_context.dynamic_entity_extrapolation,
            "forcing_gate_distance_source": "terminal_task_goal",
            "boundary_constraint_added": False,
            "history_is_preview_local": True,
            "requested_horizon_steps": horizon,
            "effective_horizon_steps": trajectory.horizon,
            "effective_horizon_ratio": float(trajectory.horizon / horizon),
            "preview_completed": True,
            "termination_reason": PREVIEW_TERMINATION_COMPLETED,
            "feature_valid_mask": feature_valid_mask,
            "feature_full_horizon_mask": feature_valid_mask.copy(),
            "obstacle_clearance_source": local_context.clearance_source,
            "obstacle_clearance_is_approximate": local_context.clearance_is_approximate,
            "boundary_clearance_source": BOUNDARY_CLEARANCE_SOURCE,
            "boundary_clearance_is_approximate": True,
            "clearance_finite_mask": clearance_finite,
            "open_space_flag": bool(local_context.visible_surface_points.shape[0] == 0),
            "actor_batching": "candidate_dimension_only",
        }
        results.append(
            CandidatePreview(
                trajectory=trajectory,
                task_progress=task_progress,
                min_clearance=min_clearance,
                max_execution_deviation=max_deviation,
                terminal_speed=terminal_speed,
                performance=performance,
                metadata=metadata,
            )
        )
    geometry_ns = time.perf_counter_ns() - geometry_started
    total_ns = time.perf_counter_ns() - total_started
    per_branch_total_ms = total_ns / 1.0e6 / float(branch_count)
    results = [
        CandidatePreview(
            trajectory=result.trajectory,
            task_progress=result.task_progress,
            min_clearance=result.min_clearance,
            max_execution_deviation=result.max_execution_deviation,
            terminal_speed=result.terminal_speed,
            performance=PreviewPerformance(
                observation_ms=result.performance.observation_ms,
                policy_ms=result.performance.policy_ms,
                transition_ms=result.performance.transition_ms,
                sensor_reconstruction_ms=result.performance.sensor_reconstruction_ms,
                total_ms=per_branch_total_ms,
                policy_calls=horizon,
            ),
            metadata=result.metadata,
        )
        for result in results
    ]
    if timing_sink is not None:
        timing_sink.update(
            {
                "observation_ms": observation_ns / 1.0e6,
                "actor_forward_ms": actor_ns / 1.0e6,
                "dmp_rollout_ms": transition_ns / 1.0e6,
                "sensor_reconstruction_ms": sensor_ns / 1.0e6,
                "geometry_metric_ms": geometry_ns / 1.0e6,
                "total_ms": total_ns / 1.0e6,
                "actor_batch_calls": float(horizon),
                "preview_branch_count": float(branch_count),
            }
        )
    return results
