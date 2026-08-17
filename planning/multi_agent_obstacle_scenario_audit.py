"""Evaluation-only coupled static-obstacle and multi-agent scenario helpers.

This module deliberately does not register a training scenario.  The geometry
is frozen before closed-loop method evaluation and is used only by the
``multi_agent_obstacle`` diagnostic runner.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from Entity.static_obstacles import StaticSphereObstacle


SCHEMA_VERSION = "multi_agent_obstacle_scenario_audit_v1"
SCENARIO_ID = "multi_agent_obstacle"
SCENARIO_ROLE = "STATIC-OBSTACLE + MULTI-AGENT COUPLING DIAGNOSTIC"
SEED_SET_ROLE = "development_diagnostic_not_held_out_test"

# This specification was frozen after the seed-0..9 geometry prototype passed
# the gate.  Closed-loop results must never be used to alter these values.
FROZEN_LAYOUT_SPEC: dict[str, Any] = {
    "version": "multi_agent_obstacle_geometry_v1_frozen_before_method_evaluation",
    "num_agents": 3,
    "crossing_center_x_m": 4.0,
    "center_y_jitter_m": [-0.04, 0.04],
    "center_z_jitter_m": [-0.03, 0.03],
    "relative_starts_m": [
        [-3.8, -1.2, -0.4],
        [-3.8, 1.2, 0.4],
        [-3.8, 0.0, -0.7],
    ],
    "relative_goals_m": [
        [3.8, 1.2, 0.4],
        [3.8, -1.2, -0.4],
        [3.8, 0.0, -0.7],
    ],
    "static_obstacles": [
        {
            "route_agent": 0,
            "route_fraction": 0.32,
            "offset_m": [0.0, 0.0, 0.0],
            "radius_m": 0.45,
            "safety_margin_m": 0.10,
        },
        {
            "route_agent": 2,
            "route_fraction": 0.68,
            "offset_m": [0.0, 0.0, 0.15],
            "radius_m": 0.45,
            "safety_margin_m": 0.10,
        },
    ],
    "dynamic_obstacle_count": 0,
    "observation_mode": "peer_spheres",
    "peer_radius_m": 0.3,
    "include_boundaries_in_sensor": False,
    "terminate_on_boundary_collision": False,
    "seeds": list(range(10)),
    "seed_set_role": SEED_SET_ROLE,
    "geometry_frozen_before_method_results": True,
    "closed_loop_result_tuning_forbidden": True,
}


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


FROZEN_LAYOUT_HASH = stable_hash(FROZEN_LAYOUT_SPEC)


def build_multi_agent_obstacle_options(seed: int) -> dict[str, Any]:
    """Construct the frozen seed-deterministic diagnostic geometry."""

    rng = np.random.default_rng(int(seed))
    center = np.asarray(
        [
            float(FROZEN_LAYOUT_SPEC["crossing_center_x_m"]),
            rng.uniform(*FROZEN_LAYOUT_SPEC["center_y_jitter_m"]),
            rng.uniform(*FROZEN_LAYOUT_SPEC["center_z_jitter_m"]),
        ],
        dtype=float,
    )
    starts = center + np.asarray(FROZEN_LAYOUT_SPEC["relative_starts_m"], dtype=float)
    goals = center + np.asarray(FROZEN_LAYOUT_SPEC["relative_goals_m"], dtype=float)
    obstacles: list[StaticSphereObstacle] = []
    for item in FROZEN_LAYOUT_SPEC["static_obstacles"]:
        agent = int(item["route_agent"])
        fraction = float(item["route_fraction"])
        obstacle_center = (
            starts[agent]
            + fraction * (goals[agent] - starts[agent])
            + np.asarray(item["offset_m"], dtype=float)
        )
        obstacles.append(
            StaticSphereObstacle(
                center=obstacle_center,
                radius=float(item["radius_m"]),
                safety_margin=float(item["safety_margin_m"]),
            )
        )
    return {
        "starts": starts,
        "goals": goals,
        "static_obstacles": obstacles,
        "dynamic_obstacles": [],
    }


def build_multi_agent_obstacle_environment(
    *,
    config: Any,
    scenario: str,
    seed: int,
    peer_radius: float,
) -> tuple[Any, dict[str, Any]]:
    """Build the evaluation-only scene without touching training adapters."""

    if str(scenario) != SCENARIO_ID:
        raise ValueError(f"expected scenario {SCENARIO_ID!r}, got {scenario!r}")
    if int(config.num_agents) != int(FROZEN_LAYOUT_SPEC["num_agents"]):
        raise ValueError("frozen coupled scenario requires exactly three agents")
    if not math.isclose(float(peer_radius), float(FROZEN_LAYOUT_SPEC["peer_radius_m"])):
        raise ValueError("peer radius differs from the frozen geometry specification")

    # Lazy import keeps this planning module independent of the evaluation CLI.
    from scripts.evaluate_single_policy_multi_agent import _build_environment

    options = build_multi_agent_obstacle_options(int(seed))
    env = _build_environment(
        config,
        observation_mode="peer_spheres",
        peer_radius=float(peer_radius),
        training_distribution=False,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )
    env.reset(seed=int(seed), options=options)
    return env, {
        "scene_type": SCENARIO_ID,
        "scenario_role": SCENARIO_ROLE,
        "scenario_geometry_version": FROZEN_LAYOUT_SPEC["version"],
        "scenario_geometry_hash": FROZEN_LAYOUT_HASH,
        "seed_set_role": SEED_SET_ROLE,
        "observation_mode": "peer_spheres",
        "peer_spheres_enabled": True,
        "include_boundaries_in_sensor": False,
        "terminate_on_boundary_collision": False,
        "static_obstacle_count": len(options["static_obstacles"]),
        "dynamic_obstacle_count": 0,
        "geometry_frozen_before_method_results": True,
    }


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * delta)))


def minimum_static_surface_clearances(
    positions: np.ndarray,
    static_obstacles: Sequence[Any],
) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if not static_obstacles:
        return np.full(positions.shape[0], np.inf, dtype=float)
    return np.asarray(
        [
            min(float(obstacle.signed_distance(position)) for obstacle in static_obstacles)
            for position in positions
        ],
        dtype=float,
    )


def direct_path_static_clearances(
    starts: np.ndarray,
    goals: np.ndarray,
    static_obstacles: Sequence[Any],
) -> np.ndarray:
    """Exact direct-route clearance for the spherical audit scenarios."""

    starts = np.asarray(starts, dtype=float)
    goals = np.asarray(goals, dtype=float)
    if not static_obstacles:
        return np.full(starts.shape[0], np.inf, dtype=float)
    results = []
    for start, goal in zip(starts, goals, strict=True):
        per_obstacle = []
        for obstacle in static_obstacles:
            if not isinstance(obstacle, StaticSphereObstacle):
                raise TypeError("audit direct-path metric currently requires spherical obstacles")
            center_distance = point_to_segment_distance(obstacle.center, start, goal)
            per_obstacle.append(center_distance - float(obstacle.effective_radius))
        results.append(min(per_obstacle))
    return np.asarray(results, dtype=float)


def predicted_pair_conflict(
    starts: np.ndarray,
    goals: np.ndarray,
) -> dict[str, Any]:
    """Minimum synchronized straight-line pair distance over normalized time."""

    starts = np.asarray(starts, dtype=float)
    goals = np.asarray(goals, dtype=float)
    best = (float("inf"), 0.0, -1, -1)
    for first in range(len(starts)):
        for second in range(first + 1, len(starts)):
            relative_start = starts[first] - starts[second]
            relative_delta = (goals[first] - starts[first]) - (
                goals[second] - starts[second]
            )
            denominator = float(np.dot(relative_delta, relative_delta))
            fraction = (
                0.0
                if denominator <= 1e-12
                else float(
                    np.clip(
                        -np.dot(relative_start, relative_delta) / denominator,
                        0.0,
                        1.0,
                    )
                )
            )
            distance = float(np.linalg.norm(relative_start + fraction * relative_delta))
            best = min(best, (distance, fraction, first, second))
    return {
        "minimum_predicted_pair_distance_m": best[0],
        "closest_pair_time_fraction": best[1],
        "closest_pair": [best[2], best[3]],
    }


def obstacle_descriptor(obstacle: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": type(obstacle).__name__,
        "center": np.asarray(obstacle.center, dtype=float).tolist(),
        "safety_margin_m": float(getattr(obstacle, "safety_margin", 0.0)),
        "effective_radius_m": float(obstacle.effective_radius),
    }
    if hasattr(obstacle, "radius"):
        result["radius_m"] = float(obstacle.radius)
    if hasattr(obstacle, "half_extents"):
        result["half_extents_m"] = np.asarray(obstacle.half_extents, dtype=float).tolist()
    if hasattr(obstacle, "velocity"):
        result["velocity_mps"] = np.asarray(obstacle.velocity, dtype=float).tolist()
    return result


def geometry_record(
    env: Any,
    *,
    scenario: str,
    seed: int,
    obstacle_influence_distance: float,
    inter_agent_risk_distance: float,
) -> dict[str, Any]:
    from scripts.evaluate_single_policy_multi_agent import build_policy_observations

    starts = np.asarray(env.starts, dtype=float)
    goals = np.asarray(env.goals, dtype=float)
    direct = direct_path_static_clearances(starts, goals, env.static_obstacles)
    conflict = predicted_pair_conflict(starts, goals)
    initial_static = minimum_static_surface_clearances(starts, env.static_obstacles)
    goal_static = minimum_static_surface_clearances(goals, env.static_obstacles)
    pairwise = np.linalg.norm(starts[:, None, :] - starts[None, :, :], axis=-1)
    upper = pairwise[np.triu_indices(len(starts), k=1)]
    predicted_distance = float(conflict["minimum_predicted_pair_distance_m"])
    predicted_time = float(conflict["closest_pair_time_fraction"])
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": str(scenario),
        "seed": int(seed),
        "num_agents": int(env.num_agents),
        "static_obstacle_count": len(env.static_obstacles),
        "dynamic_obstacle_count": len(env.dynamic_obstacles),
        "static_obstacles": [obstacle_descriptor(item) for item in env.static_obstacles],
        "dynamic_obstacles": [obstacle_descriptor(item) for item in env.dynamic_obstacles],
        "starts": starts.tolist(),
        "terminal_goals": goals.tolist(),
        "direct_static_surface_clearances_m": direct.tolist(),
        "direct_path_intersection_agent_count": int(np.sum(direct <= 0.0)),
        "direct_path_influence_agent_count": int(
            np.sum(direct <= float(obstacle_influence_distance))
        ),
        "direct_obstacle_pressure_episode": bool(
            np.any(direct <= float(obstacle_influence_distance))
        ),
        "minimum_initial_static_surface_clearance_m": float(np.min(initial_static)),
        "minimum_goal_static_surface_clearance_m": float(np.min(goal_static)),
        "minimum_initial_inter_agent_distance_m": float(np.min(upper)),
        **conflict,
        "predicted_inter_agent_conflict": bool(
            predicted_distance <= float(inter_agent_risk_distance)
            and 0.05 <= predicted_time <= 0.95
        ),
        "initial_obstacle_collision": bool(np.any(initial_static <= 0.0)),
        "goal_obstacle_collision": bool(np.any(goal_static <= 0.0)),
        "initial_inter_agent_collision": bool(
            np.min(upper) <= float(env.env_config.inter_agent_safe_distance)
        ),
        "observation_mode": str(env.single_policy_observation_mode),
        "peer_spheres_enabled": str(env.single_policy_observation_mode) == "peer_spheres",
        "checkpoint_actor_observation_shape": list(build_policy_observations(env).shape),
        "native_observation_shape": list(env.get_observation().shape),
        "include_boundaries_in_sensor": bool(env.include_boundaries_in_sensor),
        "terminate_on_boundary_collision": bool(env.terminate_on_boundary_collision),
    }


def summarize_geometry(
    records: Sequence[Mapping[str, Any]],
    *,
    required_static_obstacle_count: int | None = None,
    required_pressure_rate: float = 0.70,
    required_conflict_rate: float = 0.70,
) -> dict[str, Any]:
    if not records:
        raise ValueError("geometry records must not be empty")
    count = len(records)
    finite = [
        float(value)
        for row in records
        for value in row["direct_static_surface_clearances_m"]
        if math.isfinite(float(value))
    ]
    pressure_count = sum(bool(row["direct_obstacle_pressure_episode"]) for row in records)
    conflict_count = sum(bool(row["predicted_inter_agent_conflict"]) for row in records)
    checks = {
        "deterministic_seed_set_complete": sorted(int(row["seed"]) for row in records)
        == list(range(10)),
        "no_initial_obstacle_collision": not any(
            bool(row["initial_obstacle_collision"]) for row in records
        ),
        "no_initial_inter_agent_collision": not any(
            bool(row["initial_inter_agent_collision"]) for row in records
        ),
        "goals_feasible": not any(bool(row["goal_obstacle_collision"]) for row in records),
        "obstacle_count_matches": (
            True
            if required_static_obstacle_count is None
            else all(
                int(row["static_obstacle_count"]) == int(required_static_obstacle_count)
                for row in records
            )
        ),
        "no_dynamic_obstacles": all(int(row["dynamic_obstacle_count"]) == 0 for row in records),
        "obstacle_pressure_rate_at_least_70pct": pressure_count / count
        >= float(required_pressure_rate),
        "predicted_conflict_rate_at_least_70pct": conflict_count / count
        >= float(required_conflict_rate),
        "peer_spheres_enabled": all(bool(row["peer_spheres_enabled"]) for row in records),
        "boundary_free": all(
            not bool(row["include_boundaries_in_sensor"])
            and not bool(row["terminate_on_boundary_collision"])
            for row in records
        ),
    }
    return {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "episode_count": count,
        "pressure_episode_count": pressure_count,
        "pressure_episode_rate": pressure_count / count,
        "predicted_conflict_episode_count": conflict_count,
        "predicted_conflict_episode_rate": conflict_count / count,
        "direct_path_intersection_agent_count": sum(
            int(row["direct_path_intersection_agent_count"]) for row in records
        ),
        "direct_path_influence_agent_count": sum(
            int(row["direct_path_influence_agent_count"]) for row in records
        ),
        "mean_direct_static_surface_clearance_m": (
            float(np.mean(finite)) if finite else None
        ),
        "minimum_direct_static_surface_clearance_m": min(finite) if finite else None,
        "maximum_direct_static_surface_clearance_m": max(finite) if finite else None,
    }
