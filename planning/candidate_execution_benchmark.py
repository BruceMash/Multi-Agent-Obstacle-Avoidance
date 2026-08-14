"""Independent FP-SHEP discriminability and real-execution validation.

This module is evaluation-only.  Preview branches only consume the current
local observation, while controlled real rollouts advance an isolated copy of
the real environment through ``MultiAgentDMPEnv.step``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from Environment.frozen_sac_dmp_execution import predict_frozen_actions
from planning.policy_preview import (
    CandidatePreview,
    adapt_candidate_proposals,
    build_preview_inputs_from_env,
    point_to_segment_distance,
    preview_candidates,
)


FEATURE_PAIRS = (
    ("task_progress", "preview_task_progress", "real_task_progress"),
    ("min_clearance", "preview_min_clearance", "real_min_clearance"),
    ("execution_deviation", "preview_execution_deviation", "real_execution_deviation"),
    ("terminal_speed", "preview_terminal_speed", "real_terminal_speed"),
)


def _vector3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return result.copy()


def _array_digest(hasher: Any, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(array.tobytes())


def environment_state_fingerprint(env: Any) -> str:
    """Fingerprint all execution state that the benchmark is allowed to mutate."""
    hasher = hashlib.sha256()
    for dynamics in env.dynamics:
        _array_digest(hasher, dynamics.p)
        _array_digest(hasher, dynamics.v)
        _array_digest(hasher, dynamics.state)
    for dmp in env.dmps:
        _array_digest(hasher, dmp.goal)
        _array_digest(hasher, [dmp.phase])
    for previous_velocity in getattr(env, "previous_velocities", []):
        _array_digest(hasher, previous_velocity)
    _array_digest(hasher, env.goals)
    for packet in env.latest_sensor_packets:
        if packet is None:
            hasher.update(b"no-packet")
        else:
            _array_digest(hasher, packet.current_scan)
            _array_digest(hasher, packet.previous_scan)
            _array_digest(hasher, [packet.min_clearance])
    for sensor in getattr(env, "sensors", []):
        previous_scan = getattr(sensor, "_previous_scan", None)
        if previous_scan is None:
            hasher.update(b"no-sensor-history")
        else:
            _array_digest(hasher, previous_scan)
    _array_digest(hasher, getattr(env, "success_rewarded_mask", []))
    _array_digest(hasher, getattr(env, "stagnation_counters", []))
    for controller_info in getattr(env, "latest_controller_infos", []):
        for key in sorted(controller_info):
            value = controller_info[key]
            hasher.update(str(key).encode("utf-8"))
            if isinstance(value, np.ndarray):
                _array_digest(hasher, value)
            elif isinstance(value, (str, int, float, bool, type(None))):
                hasher.update(repr(value).encode("utf-8"))
    for history in getattr(env, "stagnation_distance_histories", []):
        _array_digest(hasher, list(history))
    hasher.update(str(int(getattr(env, "steps", 0))).encode("ascii"))
    hasher.update(str(int(getattr(env, "action_guidance_step", 0))).encode("ascii"))
    random_generator = getattr(env, "np_random", None)
    if random_generator is not None and hasattr(random_generator, "bit_generator"):
        hasher.update(
            json.dumps(
                random_generator.bit_generator.state,
                sort_keys=True,
                default=lambda value: value.tolist()
                if isinstance(value, np.ndarray)
                else value,
            ).encode("utf-8")
        )
    for obstacle in getattr(env, "dynamic_obstacles", []):
        state = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in vars(obstacle).items()
            if isinstance(value, (str, int, float, bool, type(None), np.ndarray))
        }
        hasher.update(json.dumps(state, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class RealCandidateRollout:
    candidate_goal: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    task_progress: float
    min_clearance: float
    max_execution_deviation: float
    terminal_speed: float
    terminal_position_error: float | None
    mean_trajectory_error: float | None
    terminal_position_error_at_effective_horizon: float | None
    mean_trajectory_error_at_effective_horizon: float | None
    trajectory_error_is_full_horizon: bool
    collision: bool
    obstacle_collision: bool
    inter_agent_collision: bool
    boundary_collision: bool
    success: bool
    agent_success: bool
    terminated: bool
    truncated: bool
    effective_steps: int
    minimum_inter_agent_distance: float
    runtime_ms: float
    clearance_source: str = "real_environment_sensor_packet"
    metadata: dict[str, Any] = field(default_factory=dict)


def _active_goal_observations(env: Any, active_goals: np.ndarray) -> np.ndarray:
    """Build the historical 122-D input without importing an evaluation script."""
    from Environment.frozen_sac_dmp_execution import build_historical_actor_observation

    rows: list[np.ndarray] = []
    for agent_index in range(int(env.num_agents)):
        packet = env.latest_sensor_packets[agent_index]
        if packet is None:
            raise RuntimeError("environment must be reset before candidate rollout")
        dmp = env.dmps[agent_index]
        rows.append(
            build_historical_actor_observation(
                velocity=env.dynamics[agent_index].v,
                active_goal=active_goals[agent_index],
                position=env.dynamics[agent_index].p,
                current_scan=packet.current_scan,
                previous_scan=packet.previous_scan,
                goal_distance_clip=env.sensors[agent_index].goal_distance_clip,
                phase=dmp.phase,
                k_alpha=dmp.config.K_alpha,
                k_beta=dmp.config.K_beta,
            )
        )
    return np.stack(rows).astype(np.float32)


def real_candidate_rollout(
    *,
    initial_env: Any,
    agent_index: int,
    candidate_goal: np.ndarray,
    policy: Any,
    horizon: int,
    preview: CandidatePreview | None = None,
) -> RealCandidateRollout:
    """Run one candidate from an isolated, identical environment state."""
    horizon = int(horizon)
    agent_index = int(agent_index)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= agent_index < int(initial_env.num_agents):
        raise IndexError("agent_index is out of range")
    candidate_goal = _vector3(candidate_goal, "candidate_goal")
    source_fingerprint = environment_state_fingerprint(initial_env)
    env = copy.deepcopy(initial_env)
    if environment_state_fingerprint(env) != source_fingerprint:
        raise RuntimeError("deep-copied candidate environment does not match initial state")

    task_goal = np.asarray(env.goals[agent_index], dtype=float).copy()
    start = np.asarray(env.dynamics[agent_index].p, dtype=float).copy()
    initial_distance = float(np.linalg.norm(task_goal - start))
    active_goals = np.stack([np.asarray(dmp.goal, dtype=float) for dmp in env.dmps])
    active_goals[agent_index] = candidate_goal
    positions = [start]
    velocities = [np.asarray(env.dynamics[agent_index].v, dtype=float).copy()]
    clearances: list[float] = []
    minimum_inter_agent_distance = float("inf")
    collision = obstacle_collision = inter_agent_collision = boundary_collision = False
    success = agent_success = terminated = truncated = False
    last_info: dict[str, Any] = {}
    started = time.perf_counter_ns()
    try:
        for _ in range(horizon):
            for index, active_goal in enumerate(active_goals):
                env.dmps[index].goal = np.asarray(active_goal, dtype=float).copy()
            observations = _active_goal_observations(env, active_goals)
            actions = predict_frozen_actions(
                policy,
                observations,
                expected_shape=tuple(env.action_shape),
            )
            _, _, terminated, truncated, last_info = env.step(actions)
            positions.append(np.asarray(env.dynamics[agent_index].p, dtype=float).copy())
            velocities.append(np.asarray(env.dynamics[agent_index].v, dtype=float).copy())
            step_clearances = np.asarray(last_info.get("min_clearances", []), dtype=float)
            if step_clearances.size > agent_index:
                clearances.append(float(step_clearances[agent_index]))
            minimum_inter_agent_distance = min(
                minimum_inter_agent_distance,
                float(last_info.get("min_inter_agent_distance", float("inf"))),
            )
            collision = collision or bool(last_info.get("collision", False))
            success = success or bool(last_info.get("success", False))
            for key, name in (
                ("obstacle_collision_mask", "obstacle"),
                ("inter_agent_collision_mask", "inter_agent"),
                ("boundary_collision_mask", "boundary"),
            ):
                mask = np.asarray(last_info.get(key, []), dtype=bool)
                hit = bool(mask.size > agent_index and mask[agent_index])
                if name == "obstacle":
                    obstacle_collision = obstacle_collision or hit
                elif name == "inter_agent":
                    inter_agent_collision = inter_agent_collision or hit
                else:
                    boundary_collision = boundary_collision or hit
            success_mask = np.asarray(last_info.get("success_mask", []), dtype=bool)
            agent_success = agent_success or bool(
                success_mask.size > agent_index and success_mask[agent_index]
            )
            if terminated or truncated:
                break
    finally:
        runtime_ms = (time.perf_counter_ns() - started) / 1.0e6
        close = getattr(env, "close", None)
        if callable(close):
            close()

    position_array = np.stack(positions)
    velocity_array = np.stack(velocities)
    terminal_distance = float(np.linalg.norm(task_goal - position_array[-1]))
    deviations = [
        point_to_segment_distance(point, start, candidate_goal)
        for point in position_array[1:]
    ]
    terminal_position_error: float | None = None
    mean_trajectory_error: float | None = None
    terminal_position_error_at_effective_horizon: float | None = None
    mean_trajectory_error_at_effective_horizon: float | None = None
    if preview is not None:
        comparable = min(len(position_array), len(preview.trajectory.positions))
        errors = np.linalg.norm(
            preview.trajectory.positions[:comparable] - position_array[:comparable],
            axis=1,
        )
        terminal_position_error_at_effective_horizon = float(errors[-1])
        mean_trajectory_error_at_effective_horizon = (
            float(np.mean(errors[1:])) if comparable > 1 else 0.0
        )
        if len(position_array) - 1 == horizon:
            terminal_position_error = terminal_position_error_at_effective_horizon
            mean_trajectory_error = mean_trajectory_error_at_effective_horizon
    return RealCandidateRollout(
        candidate_goal=candidate_goal,
        positions=position_array,
        velocities=velocity_array,
        task_progress=initial_distance - terminal_distance,
        min_clearance=float(np.min(clearances)) if clearances else float("inf"),
        max_execution_deviation=float(max(deviations)) if deviations else 0.0,
        terminal_speed=float(np.linalg.norm(velocity_array[-1])),
        terminal_position_error=terminal_position_error,
        mean_trajectory_error=mean_trajectory_error,
        terminal_position_error_at_effective_horizon=terminal_position_error_at_effective_horizon,
        mean_trajectory_error_at_effective_horizon=mean_trajectory_error_at_effective_horizon,
        trajectory_error_is_full_horizon=(len(position_array) - 1 == horizon),
        collision=collision,
        obstacle_collision=obstacle_collision,
        inter_agent_collision=inter_agent_collision,
        boundary_collision=boundary_collision,
        success=success,
        agent_success=agent_success,
        terminated=bool(terminated),
        truncated=bool(truncated),
        effective_steps=len(position_array) - 1,
        minimum_inter_agent_distance=minimum_inter_agent_distance,
        runtime_ms=runtime_ms,
        metadata={
            "execution_semantics": "MultiAgentDMPEnv.step",
            "sensor_refresh": "real_each_step",
            "dynamic_obstacle_update": "real_each_step",
            "initial_state_fingerprint": source_fingerprint,
            "final_info_steps": int(last_info.get("steps", getattr(initial_env, "steps", 0))),
        },
    )


def _proposal_metadata(proposal: Any) -> dict[str, Any]:
    names = (
        "azimuth_index", "elevation_index", "direction", "point", "distance",
        "raw_obstacle_distance", "obstacle_distance", "effective_safe_radius",
        "braking_distance", "safety_margin", "normalized_margin",
        "distance_progress", "normalized_progress", "alignment", "smoothness",
        "usable_length", "score",
    )
    return {name: getattr(proposal, name) for name in names if hasattr(proposal, name)}


def evaluate_candidate_set(
    *,
    env: Any,
    agent_index: int,
    proposals: Sequence[Any],
    policy: Any,
    horizon: int,
    consumer_top_k: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate ordered candidates without mutating ``env`` or policy state."""
    before = environment_state_fingerprint(env)
    all_positions = np.stack([
        np.asarray(dynamics.p, dtype=float) for dynamics in env.dynamics
    ])
    all_velocities = np.stack([
        np.asarray(dynamics.v, dtype=float) for dynamics in env.dynamics
    ])
    selected = adapt_candidate_proposals(proposals, consumer_top_k=consumer_top_k)
    initial_state, local_context = build_preview_inputs_from_env(env, agent_index)
    previews = preview_candidates(
        initial_state=initial_state,
        local_context=local_context,
        candidates=selected,
        policy=policy,
        horizon=horizon,
        dmp_config=env.dmps[agent_index].config,
        dynamics=env.dynamics[agent_index],
    )
    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for candidate_index, (proposal, preview) in enumerate(zip(selected, previews, strict=True)):
        real = real_candidate_rollout(
            initial_env=env,
            agent_index=agent_index,
            candidate_goal=preview.trajectory.candidate_goal,
            policy=policy,
            horizon=horizon,
            preview=preview,
        )
        metadata = _proposal_metadata(proposal)
        row = {
            "agent_id": int(agent_index),
            "candidate_id": int(candidate_index),
            "candidate_rank": int(candidate_index + 1),
            "candidate_xyz": preview.trajectory.candidate_goal.tolist(),
            "proposal_score": float(metadata.get("score", float("nan"))),
            "geometric_progress": float(metadata.get("distance_progress", float("nan"))),
            "H": int(horizon),
            "K_requested": None if consumer_top_k is None else int(consumer_top_k),
            "K_t": len(selected),
            "preview_task_progress": preview.task_progress,
            "real_task_progress": real.task_progress,
            "preview_min_clearance": preview.min_clearance,
            "real_min_clearance": real.min_clearance,
            "preview_execution_deviation": preview.max_execution_deviation,
            "real_execution_deviation": real.max_execution_deviation,
            "preview_terminal_speed": preview.terminal_speed,
            "real_terminal_speed": real.terminal_speed,
            "terminal_position_error": real.terminal_position_error,
            "mean_trajectory_error": real.mean_trajectory_error,
            "terminal_position_error_at_effective_horizon": real.terminal_position_error_at_effective_horizon,
            "mean_trajectory_error_at_effective_horizon": real.mean_trajectory_error_at_effective_horizon,
            "trajectory_error_is_full_horizon": real.trajectory_error_is_full_horizon,
            "collision": real.collision,
            "obstacle_collision": real.obstacle_collision,
            "inter_agent_collision": real.inter_agent_collision,
            "boundary_collision": real.boundary_collision,
            "success": real.success,
            "agent_success": real.agent_success,
            "terminated": real.terminated,
            "truncated": real.truncated,
            "effective_steps": real.effective_steps,
            "minimum_real_inter_agent_distance": real.minimum_inter_agent_distance,
            "neighbor_positions_initial": np.delete(all_positions, agent_index, axis=0).tolist(),
            "neighbor_velocities_initial": np.delete(all_velocities, agent_index, axis=0).tolist(),
            "preview_runtime_ms": preview.performance.total_ms,
            "real_runtime_ms": real.runtime_ms,
            "preview_clearance_source": preview.metadata["clearance_source"],
            "preview_clearance_is_approximate": preview.metadata["clearance_is_approximate"],
            "real_clearance_source": real.clearance_source,
            **{
                f"proposal_{key}": value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in metadata.items()
                if key not in {"point", "score", "distance_progress"}
            },
        }
        rows.append(row)
        trajectories.append(
            {
                "candidate_id": candidate_index,
                "candidate_xyz": preview.trajectory.candidate_goal.copy(),
                "preview_positions": preview.trajectory.positions.copy(),
                "real_positions": real.positions.copy(),
                "real_velocities": real.velocities.copy(),
            }
        )
    if environment_state_fingerprint(env) != before:
        raise RuntimeError("candidate evaluation mutated its initial environment")
    return rows, trajectories


def _finite_for_score(values: Iterable[float], *, cap: float) -> np.ndarray:
    result = np.asarray(list(values), dtype=float)
    result = np.nan_to_num(result, nan=0.0, posinf=float(cap), neginf=-float(cap))
    return np.clip(result, -float(cap), float(cap))


def minmax(values: Iterable[float], *, cap: float = 1.0e6) -> np.ndarray:
    result = _finite_for_score(values, cap=cap)
    if result.size == 0:
        return result
    span = float(np.max(result) - np.min(result))
    if span <= 1.0e-12:
        return np.zeros_like(result)
    return (result - np.min(result)) / span


def average_ranks(values: Iterable[float], *, descending: bool = False) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return np.asarray([], dtype=float)
    ranked = -array if descending else array
    order = np.argsort(ranked, kind="mergesort")
    result = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and ranked[order[end]] == ranked[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return result


def pearson(x: Iterable[float], y: Iterable[float]) -> float | None:
    left = np.asarray(list(x), dtype=float)
    right = np.asarray(list(y), dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 2 or np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    left = np.asarray(list(x), dtype=float)
    right = np.asarray(list(y), dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(valid)) < 2:
        return None
    return pearson(average_ranks(left[valid]), average_ranks(right[valid]))


def pairwise_accuracy(predicted: Iterable[float], actual: Iterable[float]) -> float | None:
    predicted = np.asarray(list(predicted), dtype=float)
    actual = np.asarray(list(actual), dtype=float)
    correct = comparable = 0
    for left in range(len(actual)):
        for right in range(left + 1, len(actual)):
            actual_sign = np.sign(actual[left] - actual[right])
            if actual_sign == 0:
                continue
            comparable += 1
            correct += int(np.sign(predicted[left] - predicted[right]) == actual_sign)
    return None if comparable == 0 else float(correct / comparable)


def score_and_summarize_candidate_set(
    rows: list[dict[str, Any]],
    *,
    weights: dict[str, float],
    clearance_cap: float,
) -> dict[str, Any]:
    """Attach fixed benchmark-only scores/ranks and return set-level metrics."""
    if not rows:
        return {"K_t": 0}
    progress_p = minmax(row["preview_task_progress"] for row in rows)
    progress_r = minmax(row["real_task_progress"] for row in rows)
    clearance_p = minmax(
        (row["preview_min_clearance"] for row in rows), cap=clearance_cap
    )
    clearance_r = minmax(
        (row["real_min_clearance"] for row in rows), cap=clearance_cap
    )
    deviation_p = minmax(row["preview_execution_deviation"] for row in rows)
    deviation_r = minmax(row["real_execution_deviation"] for row in rows)
    speed_p = minmax(row["preview_terminal_speed"] for row in rows)
    speed_r = minmax(row["real_terminal_speed"] for row in rows)
    j_preview = (
        float(weights["progress"]) * progress_p
        + float(weights["clearance"]) * clearance_p
        - float(weights["deviation"]) * deviation_p
        - float(weights.get("terminal_speed", 0.0)) * speed_p
    )
    j_real = (
        float(weights["progress"]) * progress_r
        + float(weights["clearance"]) * clearance_r
        - float(weights["deviation"]) * deviation_r
        - float(weights.get("terminal_speed", 0.0)) * speed_r
    )
    geometry = np.asarray([row["proposal_score"] for row in rows], dtype=float)
    preview_order = np.argsort(-j_preview, kind="mergesort")
    real_order = np.argsort(-j_real, kind="mergesort")
    geometry_order = np.argsort(-geometry, kind="mergesort")
    preview_rank = np.empty(len(rows), dtype=int)
    real_rank = np.empty(len(rows), dtype=int)
    geometry_rank = np.empty(len(rows), dtype=int)
    preview_rank[preview_order] = np.arange(1, len(rows) + 1)
    real_rank[real_order] = np.arange(1, len(rows) + 1)
    geometry_rank[geometry_order] = np.arange(1, len(rows) + 1)
    for index, row in enumerate(rows):
        row.update(
            J_preview=float(j_preview[index]),
            J_real=float(j_real[index]),
            preview_rank=int(preview_rank[index]),
            geometry_rank=int(geometry_rank[index]),
            real_rank=int(real_rank[index]),
        )
    top_m = min(3, len(rows))
    oracle = int(real_order[0])
    def optional_mean(key: str) -> float | None:
        values = np.asarray(
            [row[key] for row in rows if row.get(key) is not None],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else None

    summary: dict[str, Any] = {
        "K_t": len(rows),
        "preview_real_spearman": spearman(j_preview, j_real),
        "geometry_real_spearman": spearman(geometry, j_real),
        "preview_top1_match": int(preview_order[0] == oracle),
        "geometry_top1_match": int(geometry_order[0] == oracle),
        "preview_top3_hit": int(oracle in preview_order[:top_m]),
        "geometry_top3_hit": int(oracle in geometry_order[:top_m]),
        "preview_pairwise_accuracy": pairwise_accuracy(j_preview, j_real),
        "geometry_pairwise_accuracy": pairwise_accuracy(geometry, j_real),
        "mean_preview_runtime_ms": float(np.mean([row["preview_runtime_ms"] for row in rows])),
        "total_preview_runtime_ms": float(np.sum([row["preview_runtime_ms"] for row in rows])),
        "mean_trajectory_error": optional_mean("mean_trajectory_error"),
        "terminal_position_error": optional_mean("terminal_position_error"),
        "mean_trajectory_error_at_effective_horizon": optional_mean(
            "mean_trajectory_error_at_effective_horizon"
        ),
        "terminal_position_error_at_effective_horizon": optional_mean(
            "terminal_position_error_at_effective_horizon"
        ),
        "full_horizon_trajectory_fraction": float(np.mean([
            float(row["trajectory_error_is_full_horizon"]) for row in rows
        ])),
    }
    for name, preview_key, _ in FEATURE_PAIRS:
        values = _finite_for_score((row[preview_key] for row in rows), cap=clearance_cap)
        mean = float(np.mean(values))
        summary[f"{name}_preview_std"] = float(np.std(values))
        summary[f"{name}_preview_normalized_range"] = float(
            (np.max(values) - np.min(values)) / (abs(mean) + 1.0e-8)
        )
    return summary


def correlation_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, preview_key, real_key in FEATURE_PAIRS:
        preview = np.asarray([row[preview_key] for row in rows], dtype=float)
        real = np.asarray([row[real_key] for row in rows], dtype=float)
        valid = np.isfinite(preview) & np.isfinite(real)
        results.append(
            {
                "feature": name,
                "Pearson": pearson(preview[valid], real[valid]),
                "Spearman": spearman(preview[valid], real[valid]),
                "MAE": float(np.mean(np.abs(preview[valid] - real[valid]))) if np.any(valid) else None,
                "sample_count": int(np.sum(valid)),
                "clearance_semantics_comparable": name != "min_clearance",
            }
        )
    return results


def classify_cases(
    rows: Sequence[dict[str, Any]],
    scenario_summaries: Sequence[dict[str, Any]],
    *,
    score_gap_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_state: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["scenario_id"], row["seed"], row["agent_id"], row["H"])
        by_state.setdefault(key, []).append(row)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for key, group in by_state.items():
        geometry_best = min(group, key=lambda row: row["geometry_rank"])
        preview_best = min(group, key=lambda row: row["preview_rank"])
        real_best = min(group, key=lambda row: row["real_rank"])
        geometry_gap = float(real_best["J_real"] - geometry_best["J_real"])
        preview_gap = float(real_best["J_real"] - preview_best["J_real"])
        base = {
            "scenario_id": key[0], "seed": key[1], "agent_id": key[2], "H": key[3],
            "geometry_best_candidate": geometry_best["candidate_id"],
            "preview_best_candidate": preview_best["candidate_id"],
            "real_best_candidate": real_best["candidate_id"],
            "geometry_real_score_gap": geometry_gap,
            "preview_real_score_gap": preview_gap,
        }
        if geometry_best is not real_best and preview_best is real_best and geometry_gap > score_gap_threshold:
            successes.append(base)
        if preview_best is not real_best and preview_gap > score_gap_threshold:
            failures.append(base)
    return successes, failures


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
