"""Frozen scene and classical-planner contracts for the final benchmark.

This module deliberately contains no learned-method selection logic.  It owns
only deterministic scene construction, geometry fingerprints, difficulty
descriptors, and the two classical local-planner implementations used by the
final six-method comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from Entity.dynamic_obstacles import MovingSphereObstacle, PatternedMovingSphereObstacle
from Entity.static_obstacles import (
    AxisAlignedBoxObstacle,
    StaticCylinderObstacle,
    StaticSphereObstacle,
    WorkspaceBoundaryPlaneObstacle,
)


SCHEMA_VERSION = "final_four_stage_scene_v1"
STAGE_ORDER = ("stage_1", "stage_2", "stage_3", "stage_4")
STAGE_LABELS = {
    "stage_1": "Stage I — Nominal multi-UAV",
    "stage_2": "Stage II — Sparse static",
    "stage_3": "Stage III — Dense constrained",
    "stage_4": "Stage IV — Dense mixed dynamic",
}
FAMILY_ORDER = {
    "stage_1": (
        "open_crossing",
        "open_permutation",
        "open_fan",
        "open_head_on",
        "open_altitude_crossing",
    ),
    "stage_2": (
        "sparse_slalom",
        "sparse_crossing",
        "sparse_altitude_gate",
        "sparse_offset_gate",
        "sparse_staggered",
    ),
    "stage_3": (
        "dense_dual_gate",
        "dense_central_split",
        "dense_staggered_corridor",
        "dense_three_lane",
        "dense_offset_chicane",
    ),
    "stage_4": (
        "mixed_moving_gate",
        "mixed_cross_traffic",
        "mixed_curved_flow",
        "mixed_wandering_flow",
        "mixed_reciprocal_flow",
    ),
}
WORKSPACE_BOUNDS = ((-1.5, -4.2, -2.2), (12.5, 4.2, 2.2))


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sphere(center: Sequence[float], radius: float, margin: float = 0.08) -> dict[str, Any]:
    return {
        "type": "sphere",
        "center": list(map(float, center)),
        "radius": float(radius),
        "safety_margin": float(margin),
    }


def _box(
    center: Sequence[float], half_extents: Sequence[float], margin: float = 0.06
) -> dict[str, Any]:
    return {
        "type": "box",
        "center": list(map(float, center)),
        "half_extents": list(map(float, half_extents)),
        "safety_margin": float(margin),
    }


def _cylinder(
    center: Sequence[float], radius: float, half_height: float, margin: float = 0.06
) -> dict[str, Any]:
    return {
        "type": "cylinder",
        "center": list(map(float, center)),
        "radius": float(radius),
        "half_height": float(half_height),
        "safety_margin": float(margin),
    }


def _moving(
    center: Sequence[float],
    velocity: Sequence[float],
    *,
    radius: float,
    bounds: Sequence[Sequence[float]],
    mode: str,
    seed: int,
    margin: float = 0.06,
    turn_rate: float = 0.45,
    wandering_strength: float = 0.8,
) -> dict[str, Any]:
    return {
        "type": "patterned_moving_sphere",
        "center": list(map(float, center)),
        "velocity": list(map(float, velocity)),
        "radius": float(radius),
        "safety_margin": float(margin),
        "bounds": [list(map(float, row)) for row in bounds],
        "motion_mode": str(mode),
        "turn_rate": float(turn_rate),
        "wandering_strength": float(wandering_strength),
        "motion_seed": int(seed),
    }


def obstacle_from_spec(spec: Mapping[str, Any]) -> Any:
    kind = str(spec["type"])
    if kind == "sphere":
        return StaticSphereObstacle(
            center=np.asarray(spec["center"], dtype=float),
            radius=float(spec["radius"]),
            safety_margin=float(spec.get("safety_margin", 0.0)),
        )
    if kind == "box":
        return AxisAlignedBoxObstacle(
            center=np.asarray(spec["center"], dtype=float),
            half_extents=np.asarray(spec["half_extents"], dtype=float),
            safety_margin=float(spec.get("safety_margin", 0.0)),
        )
    if kind == "cylinder":
        return StaticCylinderObstacle(
            center=np.asarray(spec["center"], dtype=float),
            radius=float(spec["radius"]),
            half_height=float(spec["half_height"]),
            safety_margin=float(spec.get("safety_margin", 0.0)),
        )
    if kind == "patterned_moving_sphere":
        return PatternedMovingSphereObstacle(
            center=np.asarray(spec["center"], dtype=float),
            radius=float(spec["radius"]),
            velocity=np.asarray(spec["velocity"], dtype=float),
            safety_margin=float(spec.get("safety_margin", 0.0)),
            bounds=tuple(np.asarray(row, dtype=float) for row in spec["bounds"]),
            motion_mode=str(spec["motion_mode"]),
            turn_rate=float(spec.get("turn_rate", 0.45)),
            wandering_strength=float(spec.get("wandering_strength", 0.8)),
            seed=int(spec["motion_seed"]),
        )
    raise ValueError(f"unknown obstacle type: {kind}")


def obstacles_from_entry(entry: Mapping[str, Any]) -> tuple[list[Any], list[Any]]:
    static = [obstacle_from_spec(row) for row in entry["static_obstacles"]]
    dynamic = [obstacle_from_spec(row) for row in entry["dynamic_obstacles"]]
    return static, dynamic


def _jittered(value: Sequence[float], rng: np.random.Generator, scale: Sequence[float]) -> list[float]:
    return (
        np.asarray(value, dtype=float)
        + rng.uniform(-1.0, 1.0, size=3) * np.asarray(scale, dtype=float)
    ).tolist()


def _task_points(stage_index: int, family_index: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    span = 7.8 + 0.72 * stage_index + rng.uniform(-0.12, 0.12)
    x0 = -0.15 + rng.uniform(-0.10, 0.10)
    lateral_templates = (
        np.array([-1.55, 0.0, 1.55]),
        np.array([-1.75, 0.25, 1.45]),
        np.array([-1.35, -0.10, 1.75]),
        np.array([-1.80, 0.0, 1.80]),
        np.array([-1.45, 0.15, 1.60]),
    )
    goal_orders = (
        (2, 0, 1),
        (1, 2, 0),
        (2, 1, 0),
        (2, 0, 1),
        (1, 2, 0),
    )
    start_y = lateral_templates[family_index].copy()
    start_y += rng.uniform(-0.13, 0.13, size=3)
    goal_y_base = -lateral_templates[(family_index + 2) % len(lateral_templates)]
    goal_y = goal_y_base[np.asarray(goal_orders[family_index], dtype=int)]
    goal_y += rng.uniform(-0.13, 0.13, size=3)
    start_z_templates = (
        np.array([-0.45, 0.0, 0.45]),
        np.array([0.35, -0.45, 0.05]),
        np.array([-0.55, 0.45, 0.0]),
        np.array([0.15, -0.35, 0.45]),
        np.array([-0.70, 0.0, 0.70]),
    )
    start_z = start_z_templates[family_index] + rng.uniform(-0.08, 0.08, size=3)
    goal_z = -start_z_templates[(family_index + 1) % 5][np.asarray(goal_orders[family_index])]
    goal_z += rng.uniform(-0.08, 0.08, size=3)
    starts = np.stack(
        [np.full(3, x0), start_y, start_z], axis=1
    )
    goals = np.stack(
        [np.full(3, x0 + span), goal_y, goal_z], axis=1
    )
    starts = _expand_pairwise_spacing(starts, minimum=1.30)
    goals = _expand_pairwise_spacing(goals, minimum=1.30)
    return starts.astype(float), goals.astype(float)


def _expand_pairwise_spacing(points: np.ndarray, *, minimum: float) -> np.ndarray:
    """Expand only lateral/vertical offsets until the environment gate passes."""

    result = np.asarray(points, dtype=float).copy()
    center = np.mean(result[:, 1:], axis=0)
    for _ in range(32):
        pair_minimum = min(
            float(np.linalg.norm(result[left] - result[right]))
            for left in range(len(result))
            for right in range(left + 1, len(result))
        )
        if pair_minimum >= float(minimum):
            return result
        result[:, 1:] = center + 1.08 * (result[:, 1:] - center)
    raise RuntimeError("failed to enforce frozen start/goal spacing")


def _stage2_obstacles(family_index: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    patterns = (
        ((2.6, -0.8, 0.0), (4.6, 0.9, 0.1), (6.6, -0.7, -0.1)),
        ((2.8, 0.9, 0.0), (4.8, -0.9, -0.1), (6.8, 0.8, 0.1)),
        ((3.1, -0.2, -0.75), (5.0, 0.2, 0.75), (6.9, -0.3, 0.0)),
        ((3.0, -1.0, 0.0), (4.9, 0.1, 0.0), (6.8, 1.0, 0.0)),
        ((2.5, 1.0, 0.1), (4.2, -0.8, -0.1), (5.9, 0.9, 0.0), (7.3, -0.7, 0.1)),
    )
    specs: list[dict[str, Any]] = []
    for index, center in enumerate(patterns[family_index]):
        c = _jittered(center, rng, (0.18, 0.16, 0.10))
        radius = 0.36 + 0.04 * ((index + family_index) % 2) + rng.uniform(-0.02, 0.02)
        if (index + family_index) % 3 == 0:
            specs.append(_cylinder(c, radius, 0.62 + rng.uniform(-0.05, 0.05)))
        else:
            specs.append(_sphere(c, radius))
    return specs


def _stage3_obstacles(family_index: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    # Five topology families, each with two offset gates plus central blockers.
    y_offsets = (-0.15, 0.35, -0.40, 0.55, -0.65)
    shift = y_offsets[family_index]
    specs: list[dict[str, Any]] = []
    gate_x = (2.6, 4.8, 7.0)
    gaps = (1.15, -1.00, 0.85)
    for gate_id, (x, gap_y) in enumerate(zip(gate_x, gaps, strict=True)):
        gap_y += shift * (1.0 if gate_id != 1 else -1.0)
        for side in (-1.0, 1.0):
            y = gap_y + side * (1.25 + 0.08 * family_index)
            center = _jittered((x, y, 0.0), rng, (0.12, 0.08, 0.05))
            if (gate_id + family_index) % 2 == 0:
                specs.append(_box(center, (0.34, 0.58, 0.82), margin=0.07))
            else:
                specs.append(_cylinder(center, 0.43, 0.90, margin=0.07))
    central = (
        (3.7, -0.15 + 0.20 * family_index, 0.72),
        (5.9, 0.30 - 0.18 * family_index, -0.72),
    )
    for index, center in enumerate(central):
        c = _jittered(center, rng, (0.14, 0.12, 0.08))
        specs.append(_sphere(c, 0.42 + 0.025 * ((index + family_index) % 2), 0.07))
    return specs


def _stage4_dynamic(family_index: int, rng: np.random.Generator, seed: int) -> list[dict[str, Any]]:
    mode_sets = (
        ("linear", "linear", "curved"),
        ("linear", "curved", "linear"),
        ("curved", "curved", "linear"),
        ("wandering", "curved", "wandering"),
        ("linear", "wandering", "curved"),
    )
    centers = (
        (3.5, -2.20, -0.25),
        (5.2, 2.15, 0.45),
        (7.1, -2.05, 0.15),
    )
    velocities = (
        (0.05, 0.78, 0.08),
        (-0.08, -0.74, -0.06),
        (0.03, 0.70, -0.04),
    )
    bounds = ((1.2, -3.25, -1.45), (9.7, 3.25, 1.45))
    rows = []
    for index, (center, velocity, mode) in enumerate(
        zip(centers, velocities, mode_sets[family_index], strict=True)
    ):
        c = _jittered(center, rng, (0.16, 0.12, 0.12))
        speed_scale = 0.92 + 0.05 * family_index + rng.uniform(-0.03, 0.03)
        v = (np.asarray(velocity, dtype=float) * speed_scale).tolist()
        rows.append(
            _moving(
                c,
                v,
                radius=0.27 + 0.015 * ((family_index + index) % 2),
                bounds=bounds,
                mode=mode,
                seed=int(seed + 1009 * (index + 1)),
                turn_rate=0.30 + 0.07 * family_index,
                wandering_strength=0.55 + 0.08 * family_index,
            )
        )
    if family_index in (3, 4):
        rows.append(
            _moving(
                _jittered((8.3, 2.0, -0.55), rng, (0.12, 0.12, 0.10)),
                (-0.10, -0.67, 0.09),
                radius=0.25,
                bounds=bounds,
                mode="wandering" if family_index == 3 else "linear",
                seed=int(seed + 5003),
                turn_rate=0.42,
                wandering_strength=0.72,
            )
        )
    return rows


def _freeze_dynamic_trajectories(specs: Sequence[Mapping[str, Any]], *, dt: float, steps: int) -> list[list[list[float]]]:
    trajectories: list[list[list[float]]] = []
    for spec in specs:
        obstacle = obstacle_from_spec(spec)
        path = [np.asarray(obstacle.center, dtype=float).tolist()]
        for _ in range(int(steps)):
            path.append(np.asarray(obstacle.step(float(dt)), dtype=float).tolist())
        trajectories.append(path)
    return trajectories


def _polyline_samples(points: Sequence[Sequence[float]], count_per_segment: int = 80) -> np.ndarray:
    rows: list[np.ndarray] = []
    array = np.asarray(points, dtype=float)
    for index in range(len(array) - 1):
        endpoint = index == len(array) - 2
        alphas = np.linspace(0.0, 1.0, count_per_segment, endpoint=endpoint)
        rows.extend((1.0 - alpha) * array[index] + alpha * array[index + 1] for alpha in alphas)
    return np.asarray(rows, dtype=float)


def _witness_paths(starts: np.ndarray, goals: np.ndarray, stage_index: int) -> list[list[list[float]]]:
    if stage_index <= 1:
        return [[start.tolist(), goal.tolist()] for start, goal in zip(starts, goals, strict=True)]
    bypass_y = (-3.25, 3.25, -3.25)
    rows: list[list[list[float]]] = []
    for agent_id, (start, goal) in enumerate(zip(starts, goals, strict=True)):
        # The witness is an evaluation-only existence certificate.  Move to a
        # side lane before the first gate and leave it after the last gate so
        # the certificate never depends on a planner-specific local choice.
        x_mid_a = float(start[0] + 0.05 * (goal[0] - start[0]))
        x_mid_b = float(start[0] + 0.95 * (goal[0] - start[0]))
        y = bypass_y[(agent_id + stage_index) % 3]
        z = float(1.80 * (-1.0 if agent_id == 1 else 1.0))
        rows.append(
            [
                start.tolist(),
                [x_mid_a, y, z],
                [x_mid_b, y, z],
                goal.tolist(),
            ]
        )
    return rows


def _minimum_static_witness_clearance(paths: Sequence[Sequence[Sequence[float]]], specs: Sequence[Mapping[str, Any]]) -> float:
    if not specs:
        return float("inf")
    obstacles = [obstacle_from_spec(spec) for spec in specs]
    minimum = float("inf")
    for path in paths:
        for point in _polyline_samples(path):
            minimum = min(minimum, *(float(obstacle.signed_distance(point)) for obstacle in obstacles))
    return float(minimum)


def _straight_crossing_descriptor(starts: np.ndarray, goals: np.ndarray) -> dict[str, Any]:
    alphas = np.linspace(0.0, 1.0, 121)
    paths = starts[:, None, :] * (1.0 - alphas[None, :, None]) + goals[:, None, :] * alphas[None, :, None]
    pair_minima: list[float] = []
    for left in range(len(starts)):
        for right in range(left + 1, len(starts)):
            pair_minima.append(float(np.min(np.linalg.norm(paths[left] - paths[right], axis=1))))
    return {
        "straight_path_pair_minimum_m": float(min(pair_minima)),
        "crossing_pair_count_lt_1p2m": int(sum(value < 1.2 for value in pair_minima)),
        "crossing_intensity": float(np.mean([math.exp(-value / 1.2) for value in pair_minima])),
    }


def _obstacle_volume(spec: Mapping[str, Any]) -> float:
    margin = float(spec.get("safety_margin", 0.0))
    if spec["type"] in {"sphere", "patterned_moving_sphere"}:
        radius = float(spec["radius"]) + margin
        return 4.0 * math.pi * radius**3 / 3.0
    if spec["type"] == "box":
        half = np.asarray(spec["half_extents"], dtype=float) + margin
        return float(8.0 * np.prod(half))
    if spec["type"] == "cylinder":
        radius = float(spec["radius"]) + margin
        half_height = float(spec["half_height"]) + margin
        return float(math.pi * radius**2 * (2.0 * half_height))
    raise ValueError(spec["type"])


def _translation_invariant_signature(entry: Mapping[str, Any]) -> str:
    points = np.asarray(entry["starts"] + entry["goals"], dtype=float)
    origin = np.mean(points, axis=0)
    payload = {
        "starts": np.round(np.asarray(entry["starts"], dtype=float) - origin, 5).tolist(),
        "goals": np.round(np.asarray(entry["goals"], dtype=float) - origin, 5).tolist(),
        "static": [
            {
                **{key: value for key, value in row.items() if key != "center"},
                "center": np.round(np.asarray(row["center"], dtype=float) - origin, 5).tolist(),
            }
            for row in entry["static_obstacles"]
        ],
        "dynamic": [
            {
                **{key: value for key, value in row.items() if key not in {"center", "bounds"}},
                "center": np.round(np.asarray(row["center"], dtype=float) - origin, 5).tolist(),
            }
            for row in entry["dynamic_obstacles"]
        ],
    }
    return stable_hash(payload)


def generate_scenario_entry(
    *,
    stage: str,
    scenario_index: int,
    seed: int,
    prefix: str,
    max_steps: int = 220,
    dt: float = 0.1,
) -> dict[str, Any]:
    if stage not in STAGE_ORDER:
        raise ValueError(stage)
    stage_index = STAGE_ORDER.index(stage) + 1
    families = FAMILY_ORDER[stage]
    family_index = int(scenario_index) % len(families)
    family = families[family_index]
    rng = np.random.default_rng(int(seed))
    starts, goals = _task_points(stage_index, family_index, rng)
    if stage_index == 1:
        static_specs: list[dict[str, Any]] = []
    elif stage_index == 2:
        static_specs = _stage2_obstacles(family_index, rng)
    else:
        static_specs = _stage3_obstacles(family_index, rng)
    dynamic_specs = (
        _stage4_dynamic(family_index, rng, int(seed)) if stage_index == 4 else []
    )
    dynamic_trajectories = _freeze_dynamic_trajectories(
        dynamic_specs, dt=float(dt), steps=int(max_steps)
    )
    witness_paths = _witness_paths(starts, goals, stage_index)
    witness_clearance = _minimum_static_witness_clearance(witness_paths, static_specs)
    crossing = _straight_crossing_descriptor(starts, goals)
    task_distances = np.linalg.norm(goals - starts, axis=1)
    workspace = np.asarray(WORKSPACE_BOUNDS, dtype=float)
    workspace_volume = float(np.prod(workspace[1] - workspace[0]))
    dynamic_speeds = [float(np.linalg.norm(row["velocity"])) for row in dynamic_specs]
    expected_free_width = {1: 4.2, 2: 2.1, 3: 1.05, 4: 0.90}[stage_index]
    difficulty_score = (
        0.12 * float(np.mean(task_distances))
        + 0.22 * len(static_specs)
        + 0.35 * len(dynamic_specs)
        + 0.60 * crossing["crossing_intensity"]
        + 0.45 / expected_free_width
    )
    scene_id = f"{prefix}{stage_index}_{int(scenario_index):03d}"
    entry: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "stage_index": stage_index,
        "stage_label": STAGE_LABELS[stage],
        "scenario_id": scene_id,
        "scenario_index": int(scenario_index),
        "family": family,
        "family_index": family_index,
        "seed": int(seed),
        "starts": starts.tolist(),
        "initial_velocities": np.zeros_like(starts).tolist(),
        "goals": goals.tolist(),
        "static_obstacles": static_specs,
        "dynamic_obstacles": dynamic_specs,
        "dynamic_obstacle_trajectories": dynamic_trajectories,
        "workspace_bounds": json_ready(WORKSPACE_BOUNDS),
        "boundary_mode": "boundary_free",
        "max_steps": int(max_steps),
        "dt": float(dt),
        "witness_paths_evaluation_only": witness_paths,
        "difficulty": {
            "task_distance_mean_m": float(np.mean(task_distances)),
            "task_distance_min_m": float(np.min(task_distances)),
            "task_distance_max_m": float(np.max(task_distances)),
            "static_obstacle_count": len(static_specs),
            "dynamic_obstacle_count": len(dynamic_specs),
            "obstacle_count": len(static_specs) + len(dynamic_specs),
            "static_obstacle_volume_density": float(
                sum(_obstacle_volume(row) for row in static_specs) / workspace_volume
            ),
            "minimum_free_width_design_m": float(expected_free_width),
            "witness_path_minimum_static_clearance_m": (
                None if math.isinf(witness_clearance) else float(witness_clearance)
            ),
            "dynamic_speed_mean_mps": (
                float(np.mean(dynamic_speeds)) if dynamic_speeds else 0.0
            ),
            "dynamic_speed_max_mps": max(dynamic_speeds, default=0.0),
            "initial_minimum_inter_agent_distance_m": float(
                min(
                    np.linalg.norm(starts[left] - starts[right])
                    for left in range(3)
                    for right in range(left + 1, 3)
                )
            ),
            **crossing,
            "route_conflict_descriptor": float(
                crossing["crossing_intensity"]
                + 0.08 * len(static_specs)
                + 0.14 * len(dynamic_specs)
            ),
            "difficulty_score": float(difficulty_score),
        },
    }
    geometry_payload = {
        key: entry[key]
        for key in (
            "starts",
            "goals",
            "static_obstacles",
            "dynamic_obstacles",
            "dynamic_obstacle_trajectories",
        )
    }
    entry["geometry_fingerprint"] = stable_hash(geometry_payload)
    entry["translation_invariant_fingerprint"] = _translation_invariant_signature(entry)
    entry["environment_fingerprint"] = stable_hash(
        {**geometry_payload, "workspace_bounds": entry["workspace_bounds"], "dt": dt}
    )
    return entry


def generate_scenario_manifest(
    *,
    counts_per_stage: int,
    seed_base: int,
    prefix: str,
    max_steps: int = 220,
    dt: float = 0.1,
) -> dict[str, Any]:
    entries = []
    for stage_index, stage in enumerate(STAGE_ORDER):
        for scenario_index in range(int(counts_per_stage)):
            seed = int(seed_base + stage_index * 10_000 + scenario_index)
            entries.append(
                generate_scenario_entry(
                    stage=stage,
                    scenario_index=scenario_index,
                    seed=seed,
                    prefix=prefix,
                    max_steps=max_steps,
                    dt=dt,
                )
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage_order": list(STAGE_ORDER),
        "family_order": json_ready(FAMILY_ORDER),
        "counts_per_stage": int(counts_per_stage),
        "unique_scenario_count": len(entries),
        "seed_base": int(seed_base),
        "entries": entries,
    }
    manifest["manifest_sha256"] = stable_hash(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def validate_scenario_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = list(manifest["entries"])
    ids = [str(row["scenario_id"]) for row in entries]
    geometry = [str(row["geometry_fingerprint"]) for row in entries]
    translation = [str(row["translation_invariant_fingerprint"]) for row in entries]
    seeds = [int(row["seed"]) for row in entries]
    exact_duplicate_count = len(geometry) - len(set(geometry))
    translation_duplicate_count = len(translation) - len(set(translation))
    rows_by_stage = {
        stage: [row for row in entries if row["stage"] == stage]
        for stage in STAGE_ORDER
    }
    stage_mean_difficulty = {
        stage: float(np.mean([row["difficulty"]["difficulty_score"] for row in rows]))
        for stage, rows in rows_by_stage.items()
    }
    difficulty_values = [stage_mean_difficulty[stage] for stage in STAGE_ORDER]
    witness_valid = all(
        row["difficulty"]["witness_path_minimum_static_clearance_m"] is None
        or float(row["difficulty"]["witness_path_minimum_static_clearance_m"]) > 0.30
        for row in entries
    )
    dynamic_frozen = all(
        len(row["dynamic_obstacle_trajectories"]) == len(row["dynamic_obstacles"])
        and all(len(path) == int(row["max_steps"]) + 1 for path in row["dynamic_obstacle_trajectories"])
        for row in entries
    )
    start_goal_contract_valid = True
    initial_obstacle_clearance_valid = True
    for row in entries:
        starts = np.asarray(row["starts"], dtype=float)
        goals = np.asarray(row["goals"], dtype=float)
        start_min = min(
            float(np.linalg.norm(starts[left] - starts[right]))
            for left in range(3)
            for right in range(left + 1, 3)
        )
        goal_min = min(
            float(np.linalg.norm(goals[left] - goals[right]))
            for left in range(3)
            for right in range(left + 1, 3)
        )
        task_min = float(np.min(np.linalg.norm(goals - starts, axis=1)))
        start_goal_contract_valid &= start_min >= 1.20 and goal_min >= 1.20 and task_min >= 5.50
        obstacles = [
            obstacle_from_spec(spec)
            for spec in row["static_obstacles"] + row["dynamic_obstacles"]
        ]
        initial_obstacle_clearance_valid &= all(
            float(obstacle.signed_distance(point)) > 0.0
            for obstacle in obstacles
            for point in np.vstack([starts, goals])
        )
    checks = {
        "unique_scenario_ids": len(ids) == len(set(ids)),
        "unique_seeds": len(seeds) == len(set(seeds)),
        "exact_geometry_duplication_lt_5pct": exact_duplicate_count / max(1, len(entries)) < 0.05,
        "translation_equivalent_duplication_lt_5pct": translation_duplicate_count / max(1, len(entries)) < 0.05,
        "five_families_per_stage": all(
            len({row["family"] for row in rows}) == 5 for rows in rows_by_stage.values()
        ),
        "difficulty_mean_strictly_increasing": all(
            left < right for left, right in zip(difficulty_values, difficulty_values[1:])
        ),
        "witness_static_clearance_valid": witness_valid,
        "dynamic_trajectories_frozen": dynamic_frozen,
        "environment_start_goal_contract_valid": bool(start_goal_contract_valid),
        "initial_obstacle_clearance_valid": bool(initial_obstacle_clearance_valid),
        "boundary_free": all(row["boundary_mode"] == "boundary_free" for row in entries),
    }
    return {
        "checks": checks,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "failed_checks": [key for key, value in checks.items() if not value],
        "scenario_count": len(entries),
        "exact_duplicate_count": exact_duplicate_count,
        "translation_equivalent_duplicate_count": translation_duplicate_count,
        "scenario_duplication_rate": exact_duplicate_count / max(1, len(entries)),
        "translation_equivalent_duplication_rate": translation_duplicate_count / max(1, len(entries)),
        "stage_mean_difficulty": stage_mean_difficulty,
        "GEOMETRY_DIVERSITY_VALID": "YES" if checks["exact_geometry_duplication_lt_5pct"] and checks["translation_equivalent_duplication_lt_5pct"] else "NO",
        "DIFFICULTY_MONOTONICITY_VALID": "YES" if checks["difficulty_mean_strictly_increasing"] else "NO",
    }


@dataclass(frozen=True)
class DWAStyleConfig:
    horizon_s: float = 1.2
    velocity_samples_per_axis: int = 5
    preferred_speed_mps: float = 2.4
    goal_weight: float = 2.0
    clearance_weight: float = 1.3
    speed_weight: float = 0.18
    peer_weight: float = 1.2
    collision_buffer_m: float = 0.08


@dataclass(frozen=True)
class RVOStyleConfig:
    preferred_speed_mps: float = 2.4
    peer_time_horizon_s: float = 1.8
    obstacle_time_horizon_s: float = 1.4
    reciprocal_gain: float = 0.62
    static_gain: float = 1.0
    dynamic_gain: float = 0.9
    safety_buffer_m: float = 0.12
    iterations: int = 3


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= float(maximum) or norm < 1e-12:
        return vector.copy()
    return vector * (float(maximum) / norm)


def _surface_clearance(point: np.ndarray, obstacles: Iterable[Any]) -> float:
    values = [float(obstacle.signed_distance(point)) for obstacle in obstacles]
    return min(values, default=float("inf"))


def _signed_distance_batch(points: np.ndarray, obstacle: Any) -> np.ndarray:
    """Vectorized equivalent of the repository obstacle signed-distance API."""

    points = np.asarray(points, dtype=float)
    if points.shape[-1] != 3:
        raise ValueError("points must end in dimension 3")
    if isinstance(obstacle, WorkspaceBoundaryPlaneObstacle):
        coordinate = points[..., int(obstacle.axis)]
        if bool(obstacle.is_lower):
            return coordinate - float(obstacle.bound)
        return float(obstacle.bound) - coordinate
    if isinstance(obstacle, (StaticSphereObstacle, MovingSphereObstacle)):
        return (
            np.linalg.norm(points - np.asarray(obstacle.center, dtype=float), axis=-1)
            - float(obstacle.effective_radius)
        )
    if isinstance(obstacle, AxisAlignedBoxObstacle):
        q = (
            np.abs(points - np.asarray(obstacle.center, dtype=float))
            - np.asarray(obstacle.expanded_half_extents, dtype=float)
        )
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside
    if isinstance(obstacle, StaticCylinderObstacle):
        offset = points - np.asarray(obstacle.center, dtype=float)
        d_xy = np.linalg.norm(offset[..., :2], axis=-1) - float(
            obstacle.expanded_radius
        )
        d_z = np.abs(offset[..., 2]) - float(obstacle.expanded_half_height)
        outside_xy = np.maximum(d_xy, 0.0)
        outside_z = np.maximum(d_z, 0.0)
        inside = np.minimum(np.maximum(d_xy, d_z), 0.0)
        return np.sqrt(outside_xy * outside_xy + outside_z * outside_z) + inside

    flat = points.reshape(-1, 3)
    values = np.asarray(
        [float(obstacle.signed_distance(point)) for point in flat], dtype=float
    )
    return values.reshape(points.shape[:-1])


def _dwa_fullstate_candidate_scores_scalar_reference(
    *,
    position: np.ndarray,
    goal: np.ndarray,
    preferred: np.ndarray,
    candidates: np.ndarray,
    static_obstacles: Sequence[Any],
    dynamic_predictions_by_step: Sequence[Sequence[Any]],
    peer_positions: np.ndarray,
    peer_velocities: np.ndarray,
    dt: float,
    collision_margin: float,
    peer_safe_distance: float,
    sensing_radius: float,
    peer_influence_distance: float,
    config: DWAStyleConfig,
) -> np.ndarray:
    """Original scalar score retained as a numerical regression oracle."""

    scores = np.empty(len(candidates), dtype=float)
    position = np.asarray(position, dtype=float)
    goal = np.asarray(goal, dtype=float)
    goal_delta = goal - position
    for candidate_id, candidate_v in enumerate(np.asarray(candidates, dtype=float)):
        minimum_obstacle = float("inf")
        minimum_peer = float("inf")
        collision = False
        predicted = position.copy()
        for horizon_step, dynamic_obstacles in enumerate(
            dynamic_predictions_by_step, start=1
        ):
            predicted = position + candidate_v * (horizon_step * float(dt))
            obstacle_clearance = _surface_clearance(
                predicted, list(static_obstacles) + list(dynamic_obstacles)
            )
            minimum_obstacle = min(minimum_obstacle, obstacle_clearance)
            for peer_position, peer_velocity in zip(
                np.asarray(peer_positions, dtype=float),
                np.asarray(peer_velocities, dtype=float),
                strict=True,
            ):
                predicted_peer = peer_position + peer_velocity * (
                    horizon_step * float(dt)
                )
                minimum_peer = min(
                    minimum_peer,
                    float(np.linalg.norm(predicted - predicted_peer)),
                )
            if obstacle_clearance <= float(collision_margin) + config.collision_buffer_m:
                collision = True
            if minimum_peer <= float(peer_safe_distance) + config.collision_buffer_m:
                collision = True
        progress = float(np.linalg.norm(goal_delta) - np.linalg.norm(goal - predicted))
        clearance_term = min(minimum_obstacle, float(sensing_radius))
        peer_term = min(minimum_peer, float(peer_influence_distance) * 2.0)
        speed_alignment = float(np.dot(candidate_v, preferred)) / max(
            float(np.linalg.norm(candidate_v) * np.linalg.norm(preferred)), 1e-9
        )
        scores[candidate_id] = (
            config.goal_weight * progress
            + config.clearance_weight * clearance_term
            + config.peer_weight * peer_term
            + config.speed_weight * speed_alignment
            - (1.0e6 if collision else 0.0)
            - 1.0e-12 * candidate_id
        )
    return scores


def _dwa_fullstate_candidate_scores_vectorized(
    *,
    position: np.ndarray,
    goal: np.ndarray,
    preferred: np.ndarray,
    candidates: np.ndarray,
    static_obstacles: Sequence[Any],
    dynamic_predictions_by_step: Sequence[Sequence[Any]],
    peer_positions: np.ndarray,
    peer_velocities: np.ndarray,
    dt: float,
    collision_margin: float,
    peer_safe_distance: float,
    sensing_radius: float,
    peer_influence_distance: float,
    config: DWAStyleConfig,
) -> np.ndarray:
    """Batch the unchanged DWA score over velocity candidates and horizon."""

    position = np.asarray(position, dtype=float)
    goal = np.asarray(goal, dtype=float)
    preferred = np.asarray(preferred, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    horizon_steps = len(dynamic_predictions_by_step)
    times = float(dt) * np.arange(1, horizon_steps + 1, dtype=float)
    predicted = position[None, None, :] + candidates[:, None, :] * times[None, :, None]

    obstacle_clearance = np.full(
        (len(candidates), horizon_steps), float("inf"), dtype=float
    )
    for obstacle in static_obstacles:
        obstacle_clearance = np.minimum(
            obstacle_clearance, _signed_distance_batch(predicted, obstacle)
        )
    for horizon_index, dynamic_obstacles in enumerate(dynamic_predictions_by_step):
        horizon_points = predicted[:, horizon_index, :]
        for obstacle in dynamic_obstacles:
            obstacle_clearance[:, horizon_index] = np.minimum(
                obstacle_clearance[:, horizon_index],
                _signed_distance_batch(horizon_points, obstacle),
            )
    minimum_obstacle = np.min(obstacle_clearance, axis=1)

    peer_positions = np.asarray(peer_positions, dtype=float).reshape(-1, 3)
    peer_velocities = np.asarray(peer_velocities, dtype=float).reshape(-1, 3)
    if len(peer_positions):
        predicted_peers = (
            peer_positions[None, :, :]
            + times[:, None, None] * peer_velocities[None, :, :]
        )
        peer_distances = np.linalg.norm(
            predicted[:, :, None, :] - predicted_peers[None, :, :, :], axis=-1
        )
        minimum_peer = np.min(peer_distances, axis=(1, 2))
        peer_collision = np.any(
            peer_distances
            <= float(peer_safe_distance) + float(config.collision_buffer_m),
            axis=(1, 2),
        )
    else:
        minimum_peer = np.full(len(candidates), float("inf"), dtype=float)
        peer_collision = np.zeros(len(candidates), dtype=bool)

    obstacle_collision = np.any(
        obstacle_clearance
        <= float(collision_margin) + float(config.collision_buffer_m),
        axis=1,
    )
    collision = np.logical_or(obstacle_collision, peer_collision)
    goal_delta = goal - position
    final_positions = predicted[:, -1, :]
    progress = np.linalg.norm(goal_delta) - np.linalg.norm(
        goal[None, :] - final_positions, axis=1
    )
    clearance_term = np.minimum(minimum_obstacle, float(sensing_radius))
    peer_term = np.minimum(minimum_peer, float(peer_influence_distance) * 2.0)
    preferred_norm = float(np.linalg.norm(preferred))
    alignment_denominator = np.maximum(
        np.linalg.norm(candidates, axis=1) * preferred_norm, 1e-9
    )
    speed_alignment = (candidates @ preferred) / alignment_denominator
    return (
        float(config.goal_weight) * progress
        + float(config.clearance_weight) * clearance_term
        + float(config.peer_weight) * peer_term
        + float(config.speed_weight) * speed_alignment
        - np.where(collision, 1.0e6, 0.0)
        - 1.0e-12 * np.arange(len(candidates), dtype=float)
    )


def _planner_static_obstacles(env: Any) -> list[Any]:
    """Return every static surface enforced by the episode contract.

    The historical four-stage benchmark was boundary-free, whereas the
    long-range benchmark terminates on the frozen 100 x 100 x flight-band
    boundary.  A full-state planner must therefore score those boundary
    planes whenever the environment enforces them.  Keeping this conditional
    preserves the historical boundary-free behavior exactly.
    """

    obstacles = list(env.static_obstacles)
    boundary_enforced = bool(
        getattr(
            env,
            "terminate_on_boundary_collision",
            getattr(env.env_config, "terminate_on_boundary_collision", False),
        )
    )
    if boundary_enforced:
        obstacles.extend(list(getattr(env, "workspace_boundary_obstacles", ())))
    return obstacles


def current_state_dynamic_predictions(
    dynamic_obstacles: Iterable[Any], prediction_times_s: Iterable[float]
) -> list[list[MovingSphereObstacle]]:
    """Predict moving spheres from public current position and velocity only.

    Deliberately do not copy a live obstacle object: patterned obstacles carry
    private RNG state.  The corrected classical contract is the frozen
    constant-velocity model p_hat(t+h)=p(t)+h*v(t).
    """

    snapshots = [
        (
            np.asarray(obstacle.center, dtype=float).copy(),
            np.asarray(obstacle.velocity, dtype=float).copy(),
            float(obstacle.radius),
            float(obstacle.safety_margin),
        )
        for obstacle in dynamic_obstacles
    ]
    return [
        [
            MovingSphereObstacle(
                center=center + float(prediction_time) * velocity,
                radius=radius,
                velocity=velocity,
                safety_margin=safety_margin,
                bounds=None,
            )
            for center, velocity, radius, safety_margin in snapshots
        ]
        for prediction_time in prediction_times_s
    ]


def dwa_style_accelerations(env: Any, config: DWAStyleConfig) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter_ns()
    positions = env._positions().astype(float)
    velocities = env._velocities().astype(float)
    dt = float(env.dynamics[0].dt)
    a_min = float(np.min(np.asarray(env.dynamics[0].accelerate_min)))
    a_max = float(np.max(np.asarray(env.dynamics[0].accelerate_max)))
    v_min = float(np.min(np.asarray(env.dynamics[0].velocity_min)))
    v_max = float(np.max(np.asarray(env.dynamics[0].velocity_max)))
    steps = max(1, int(round(float(config.horizon_s) / dt)))
    result = np.zeros_like(positions)
    evaluated = 0
    infeasible_agents = 0
    dynamic_predictions_by_step = current_state_dynamic_predictions(
        env.dynamic_obstacles,
        (horizon_step * dt for horizon_step in range(1, steps + 1)),
    )
    static_obstacles = _planner_static_obstacles(env)
    for agent_id in range(int(env.num_agents)):
        if bool(env.success_rewarded_mask[agent_id]):
            continue
        current_v = velocities[agent_id]
        low = np.maximum(v_min, current_v + a_min * dt)
        high = np.minimum(v_max, current_v + a_max * dt)
        axes = [
            np.linspace(low[dim], high[dim], int(config.velocity_samples_per_axis))
            for dim in range(3)
        ]
        candidates = np.asarray(np.meshgrid(*axes, indexing="ij"), dtype=float).reshape(3, -1).T
        goal_delta = np.asarray(env.goals[agent_id], dtype=float) - positions[agent_id]
        preferred = _clip_norm(goal_delta / max(float(np.linalg.norm(goal_delta)), 1e-9) * config.preferred_speed_mps, v_max)
        candidates = np.vstack([candidates, np.clip(preferred, low, high), np.zeros(3)])
        peer_mask = np.arange(int(env.num_agents)) != int(agent_id)
        scores = _dwa_fullstate_candidate_scores_vectorized(
            position=positions[agent_id],
            goal=np.asarray(env.goals[agent_id], dtype=float),
            preferred=preferred,
            candidates=candidates,
            static_obstacles=static_obstacles,
            dynamic_predictions_by_step=dynamic_predictions_by_step,
            peer_positions=positions[peer_mask],
            peer_velocities=velocities[peer_mask],
            dt=dt,
            collision_margin=float(env.env_config.collision_margin),
            peer_safe_distance=float(env.env_config.inter_agent_safe_distance),
            sensing_radius=float(env.sensors[agent_id].sensing_radius),
            peer_influence_distance=float(env.env_config.inter_agent_influence_distance),
            config=config,
        )
        evaluated += len(candidates)
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        best_velocity = candidates[best_index].copy()
        if best_score < -5.0e5:
            infeasible_agents += 1
            best_velocity = np.zeros(3, dtype=float)
        result[agent_id] = (best_velocity - current_v) / dt
    result = np.clip(result, a_min, a_max)
    return result, {
        "planner": "3D-DWA-style",
        "runtime_ms": (time.perf_counter_ns() - started) / 1.0e6,
        "candidate_velocity_count": int(evaluated),
        "infeasible_agent_count": int(infeasible_agents),
    }


def _reciprocal_correction(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    combined_radius: float,
    horizon: float,
) -> np.ndarray:
    rel_pos = np.asarray(relative_position, dtype=float)
    rel_vel = np.asarray(relative_velocity, dtype=float)
    distance = float(np.linalg.norm(rel_pos))
    if distance < 1e-9:
        return np.array([combined_radius / max(horizon, 1e-6), 0.0, 0.0])
    speed_squared = float(np.dot(rel_vel, rel_vel))
    if speed_squared < 1e-12:
        return np.zeros(3)
    closest_time = float(
        np.clip(-np.dot(rel_pos, rel_vel) / speed_squared, 0.0, float(horizon))
    )
    closest = rel_pos + rel_vel * closest_time
    closest_distance = float(np.linalg.norm(closest))
    if closest_time <= 0.0 or closest_distance >= combined_radius:
        return np.zeros(3)
    normal = (
        closest / closest_distance
        if closest_distance >= 1e-9
        else rel_pos / distance
    )
    required = max(
        0.0,
        (combined_radius - closest_distance) / max(closest_time, 0.1),
    )
    return normal * required


def _rotate_xy(vector: np.ndarray, angle: float) -> np.ndarray:
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    result = np.asarray(vector, dtype=float).copy()
    result[0] = cosine * vector[0] - sine * vector[1]
    result[1] = sine * vector[0] + cosine * vector[1]
    return result


def _project_reciprocal_velocity_candidate(
    *,
    env: Any,
    agent_id: int,
    preferred: np.ndarray,
    corrected: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    config: RVOStyleConfig,
) -> np.ndarray:
    """Deterministic 3-D velocity-obstacle projection around the ORCA update."""

    dt = float(env.dynamics[0].dt)
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
    candidates = [np.zeros(3), velocities[agent_id].copy()]
    for scale in (0.55, 0.78, 1.0):
        candidates.extend(_clip_norm(row * scale, config.preferred_speed_mps) for row in base_directions)
    dynamic_by_time = current_state_dynamic_predictions(
        env.dynamic_obstacles, sample_times
    )
    static_obstacles = _planner_static_obstacles(env)
    best_score = -float("inf")
    best = np.zeros(3)
    for candidate_id, candidate in enumerate(candidates):
        minimum_obstacle = float("inf")
        minimum_peer = float("inf")
        collision = False
        final_position = positions[agent_id].copy()
        for time_index, sample_time in enumerate(sample_times):
            predicted = positions[agent_id] + candidate * float(sample_time)
            final_position = predicted
            minimum_obstacle = min(
                minimum_obstacle,
                _surface_clearance(
                    predicted,
                    static_obstacles + dynamic_by_time[time_index],
                ),
            )
            for peer_id in range(int(env.num_agents)):
                if peer_id == agent_id:
                    continue
                peer_position = positions[peer_id] + velocities[peer_id] * float(sample_time)
                minimum_peer = min(
                    minimum_peer, float(np.linalg.norm(predicted - peer_position))
                )
            collision |= minimum_obstacle <= config.safety_buffer_m
            collision |= minimum_peer <= float(env.env_config.inter_agent_safe_distance) + config.safety_buffer_m
        goal_distance = float(np.linalg.norm(np.asarray(env.goals[agent_id]) - final_position))
        preference_cost = float(np.linalg.norm(candidate - preferred))
        acceleration_cost = float(np.linalg.norm(candidate - velocities[agent_id]))
        clearance_reward = min(minimum_obstacle, 2.5) if math.isfinite(minimum_obstacle) else 2.5
        peer_reward = min(minimum_peer, 2.5) if math.isfinite(minimum_peer) else 2.5
        score = (
            -1.6 * preference_cost
            -0.45 * goal_distance
            -0.06 * acceleration_cost
            +0.45 * clearance_reward
            +0.55 * peer_reward
            -(1.0e6 if collision else 0.0)
            -1.0e-12 * candidate_id
        )
        if score > best_score:
            best_score = score
            best = candidate.copy()
    return best


def rvo_orca_style_accelerations(env: Any, config: RVOStyleConfig) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter_ns()
    positions = env._positions().astype(float)
    velocities = env._velocities().astype(float)
    dt = float(env.dynamics[0].dt)
    a_min = float(np.min(np.asarray(env.dynamics[0].accelerate_min)))
    a_max = float(np.max(np.asarray(env.dynamics[0].accelerate_max)))
    v_max_component = float(np.max(np.asarray(env.dynamics[0].velocity_max)))
    desired = np.zeros_like(positions)
    correction_count = 0
    static_obstacles = _planner_static_obstacles(env)
    for agent_id in range(int(env.num_agents)):
        if bool(env.success_rewarded_mask[agent_id]):
            continue
        goal_delta = np.asarray(env.goals[agent_id], dtype=float) - positions[agent_id]
        preferred = _clip_norm(goal_delta, config.preferred_speed_mps)
        velocity = preferred.copy()
        for _ in range(int(config.iterations)):
            correction = np.zeros(3, dtype=float)
            for peer_id in range(int(env.num_agents)):
                if peer_id == agent_id:
                    continue
                row = _reciprocal_correction(
                    positions[agent_id] - positions[peer_id],
                    velocity - velocities[peer_id],
                    float(env.env_config.inter_agent_safe_distance) + config.safety_buffer_m,
                    config.peer_time_horizon_s,
                )
                if np.linalg.norm(row) > 0.0:
                    correction_count += 1
                correction += config.reciprocal_gain * row
            for obstacle in static_obstacles:
                clearance = float(obstacle.signed_distance(positions[agent_id]))
                if clearance >= config.obstacle_time_horizon_s * max(np.linalg.norm(velocity), 0.5):
                    continue
                away = positions[agent_id] - np.asarray(obstacle.closest_point(positions[agent_id]), dtype=float)
                away /= max(float(np.linalg.norm(away)), 1e-9)
                correction += config.static_gain * away * max(
                    0.0, (config.safety_buffer_m + 0.75 - clearance) / max(config.obstacle_time_horizon_s, 1e-6)
                )
                correction_count += 1
            for obstacle in env.dynamic_obstacles:
                row = _reciprocal_correction(
                    positions[agent_id] - np.asarray(obstacle.center, dtype=float),
                    velocity - np.asarray(obstacle.velocity, dtype=float),
                    float(obstacle.effective_radius) + config.safety_buffer_m,
                    config.obstacle_time_horizon_s,
                )
                if np.linalg.norm(row) > 0.0:
                    correction_count += 1
                correction += config.dynamic_gain * row
            velocity = _clip_norm(preferred + correction, config.preferred_speed_mps)
        velocity = _project_reciprocal_velocity_candidate(
            env=env,
            agent_id=agent_id,
            preferred=preferred,
            corrected=velocity,
            positions=positions,
            velocities=velocities,
            config=config,
        )
        desired[agent_id] = np.clip(velocity, -v_max_component, v_max_component)
    accelerations = (desired - velocities) / dt
    accelerations = np.clip(accelerations, a_min, a_max)
    return accelerations, {
        "planner": "RVO/ORCA-style",
        "runtime_ms": (time.perf_counter_ns() - started) / 1.0e6,
        "correction_count": int(correction_count),
    }


def _typed_obstacle_collision_masks(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Decompose the environment's obstacle mask without changing termination."""

    static_mask = np.zeros(int(env.num_agents), dtype=bool)
    dynamic_mask = np.zeros(int(env.num_agents), dtype=bool)
    margin = float(env.env_config.collision_margin)
    for agent_id, dynamic in enumerate(env.dynamics):
        static_mask[agent_id] = any(
            obstacle.contains(dynamic.p, margin=margin)
            for obstacle in env.static_obstacles
        )
        dynamic_mask[agent_id] = any(
            obstacle.contains(dynamic.p, margin=margin)
            for obstacle in env.dynamic_obstacles
        )
    return static_mask, dynamic_mask


def step_direct_accelerations(
    env: Any,
    accelerations: np.ndarray,
    *,
    refresh_sensors: bool = True,
) -> tuple[bool, bool, dict[str, Any]]:
    """Advance a classical planner with the environment's exact point-mass limits.

    This evaluation-only path intentionally does not call the SAC-DMP action
    interface.  Collision, success, timeout, dynamic-obstacle motion, and
    completed-agent freezing remain the environment definitions.  Full-state
    callers may skip unused LiDAR refresh while retaining exact geometric
    clearance accounting; sensing-matched callers keep the default refresh.
    """

    accelerations = np.asarray(accelerations, dtype=float)
    expected = (int(env.num_agents), int(env.state_dim))
    if accelerations.shape != expected:
        raise ValueError(f"accelerations must have shape {expected}")
    if not np.all(np.isfinite(accelerations)):
        raise ValueError("accelerations must be finite")
    env.previous_velocities = env._velocities().astype(float, copy=True)
    previous_distances = np.linalg.norm(
        np.asarray(env.goals, dtype=float) - env._positions(), axis=1
    )
    applied = np.zeros_like(accelerations)
    for agent_id in range(int(env.num_agents)):
        if bool(env.success_rewarded_mask[agent_id]):
            env._freeze_agent(agent_id)
            continue
        before = env.dynamics[agent_id].v.copy()
        env.dynamics[agent_id].step(accelerations[agent_id])
        applied[agent_id] = (env.dynamics[agent_id].v - before) / float(
            env.dynamics[agent_id].dt
        )
    for obstacle in env.dynamic_obstacles:
        obstacle.step(float(env.dynamics[0].dt))
    env.steps += 1
    env.action_guidance_step += 1
    if refresh_sensors:
        for agent_id in range(int(env.num_agents)):
            env.latest_sensor_packets[agent_id] = env.sensors[agent_id].sense(
                env.dynamics[agent_id].p,
                env.dynamics[agent_id].v,
                env.goals[agent_id],
                env._sensor_static_obstacles(),
                env._sensor_dynamic_obstacles(agent_id),
            )
    current_distances = np.linalg.norm(
        np.asarray(env.goals, dtype=float) - env._positions(), axis=1
    )
    success_mask = current_distances <= float(env.env_config.goal_tolerance)
    collision_info = env._check_collision()
    static_obstacle_mask, dynamic_obstacle_mask = _typed_obstacle_collision_masks(env)
    env.latest_collision_info = collision_info
    new_success = np.logical_and(success_mask, np.logical_not(env.success_rewarded_mask))
    env.success_rewarded_mask = np.logical_or(env.success_rewarded_mask, new_success)
    for agent_id in np.flatnonzero(new_success):
        env._freeze_agent(int(agent_id))
    full_success = bool(np.all(success_mask))
    collision = bool(collision_info["collision"])
    success = bool(full_success and not collision)
    terminated = bool(success or collision)
    truncated = bool((not terminated) and env.steps >= int(env.env_config.max_steps))
    if refresh_sensors:
        min_clearances = np.asarray(
            [packet.min_clearance for packet in env.latest_sensor_packets], dtype=float
        )
    else:
        clearance_obstacles = _planner_static_obstacles(env) + list(
            env.dynamic_obstacles
        )
        min_clearances = np.asarray(
            [
                _surface_clearance(env.dynamics[agent_id].p, clearance_obstacles)
                for agent_id in range(int(env.num_agents))
            ],
            dtype=float,
        )
    info = {
        "success": success,
        "success_mask": success_mask.copy(),
        "new_success_mask": new_success.copy(),
        "collision": collision,
        "collision_mask": collision_info["collision_mask"].copy(),
        "obstacle_collision_mask": collision_info["obstacle_collision_mask"].copy(),
        "static_obstacle_collision_mask": static_obstacle_mask.copy(),
        "dynamic_obstacle_collision_mask": dynamic_obstacle_mask.copy(),
        "inter_agent_collision_mask": collision_info["inter_agent_collision_mask"].copy(),
        "boundary_collision_mask": collision_info["boundary_collision_mask"].copy(),
        "min_inter_agent_distance": float(collision_info["min_inter_agent_distance"]),
        "pairwise_distances": collision_info["pairwise_distances"].copy(),
        "min_clearances": min_clearances,
        "distance_to_goals": current_distances.copy(),
        "progress": previous_distances - current_distances,
        "applied_accelerations": applied.copy(),
        "steps": int(env.steps),
        "truncated": truncated,
    }
    return terminated, truncated, info


def _trajectory_metrics(
    positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray, dt: float
) -> dict[str, Any]:
    path_lengths = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=2), axis=0)
    acceleration_deltas = np.diff(accelerations, axis=0)
    if acceleration_deltas.size:
        jerk = acceleration_deltas / float(dt)
        smoothness = np.mean(np.sum(jerk**2, axis=2), axis=0)
    else:
        smoothness = np.zeros(positions.shape[1], dtype=float)
    return {
        "path_lengths": path_lengths,
        "path_length_team_sum": float(np.sum(path_lengths)),
        "path_length_team_mean": float(np.mean(path_lengths)),
        "trajectory_smoothness_per_agent": smoothness,
        "trajectory_smoothness_team_mean": float(np.mean(smoothness)),
    }


def run_classical_episode(
    *,
    environment_builder: Any,
    multi_config: Any,
    scenario: str,
    seed: int,
    peer_radius: float,
    method: str,
    planner_config: DWAStyleConfig | RVOStyleConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if method not in {"dwa_style", "rvo_orca_style"}:
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
        dynamic_positions = [
            [np.asarray(obstacle.center, dtype=float).copy() for obstacle in env.dynamic_obstacles]
        ]
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
        terminated = False
        truncated = False
        last_info: dict[str, Any] = {}
        while not (terminated or truncated):
            if method == "dwa_style":
                acceleration, planner_info = dwa_style_accelerations(env, planner_config)  # type: ignore[arg-type]
            else:
                acceleration, planner_info = rvo_orca_style_accelerations(env, planner_config)  # type: ignore[arg-type]
            planner_infeasible_count += int(planner_info.get("infeasible_agent_count", 0))
            runtime_rows.append(
                {
                    "stage": scene_metadata.get("stage"),
                    "scenario_id": str(scenario),
                    "method": method,
                    "decision_index": int(env.steps),
                    "runtime_ms": float(planner_info["runtime_ms"]),
                }
            )
            terminated, truncated, last_info = step_direct_accelerations(
                env, acceleration, refresh_sensors=False
            )
            positions.append(env._positions().copy())
            velocities.append(env._velocities().copy())
            applied = np.asarray(last_info["applied_accelerations"], dtype=float)
            applied_accelerations.append(applied)
            dynamic_positions.append(
                [np.asarray(obstacle.center, dtype=float).copy() for obstacle in env.dynamic_obstacles]
            )
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
        if success:
            termination_reason = "success"
        elif static_obstacle_collision:
            termination_reason = "static_obstacle_collision"
        elif dynamic_obstacle_collision:
            termination_reason = "dynamic_obstacle_collision"
        elif inter_agent_collision:
            termination_reason = "inter_agent_collision"
        elif boundary_collision:
            termination_reason = "boundary_collision"
        elif timeout and planner_infeasible_count:
            termination_reason = "planner_infeasible"
        elif timeout:
            termination_reason = "timeout"
        else:
            termination_reason = "other"
        path_lengths = np.asarray(trajectory["path_lengths"], dtype=float)
        straight = np.linalg.norm(goals - starts, axis=1)
        agent_rows: list[dict[str, Any]] = []
        final_collision_mask = np.asarray(last_info["collision_mask"], dtype=bool)
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
                    "agent_path_length_m": float(path_lengths[agent_id]),
                    "agent_path_efficiency": (
                        float(straight[agent_id] / max(path_lengths[agent_id], 1e-9))
                        if completed
                        else None
                    ),
                    "reference_selected": False,
                    "reference_reached": None,
                    "completion_step": completion_steps[agent_id],
                    "minimum_obstacle_clearance_m": (
                        float(min_clearance_by_agent[agent_id])
                        if math.isfinite(min_clearance_by_agent[agent_id])
                        else None
                    ),
                    "minimum_peer_distance_m": (
                        float(min_peer_by_agent[agent_id])
                        if math.isfinite(min_peer_by_agent[agent_id])
                        else None
                    ),
                }
            )
        total_planner_runtime = float(sum(row["runtime_ms"] for row in runtime_rows))
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
            "minimum_obstacle_clearance_m": (
                float(np.min(min_clearance_by_agent))
                if np.any(np.isfinite(min_clearance_by_agent))
                else None
            ),
            "minimum_inter_agent_distance_m": float(np.min(min_peer_by_agent)),
            "planning_runtime_ms": total_planner_runtime,
            "planning_decision_count": len(runtime_rows),
            "planning_runtime_per_decision_ms": (
                total_planner_runtime / max(1, len(runtime_rows))
            ),
            "lower_level_runtime_ms": 0.0,
            "environment_step_runtime_ms": None,
            "end_to_end_runtime_ms": (time.perf_counter_ns() - episode_started) / 1.0e6,
            "initial_condition_hash": scene_metadata.get("environment_fingerprint"),
            "scenario_manifest_hash": scene_metadata.get("environment_fingerprint"),
            "planner_infeasible_count": int(planner_infeasible_count),
        }
        trajectory_payload = {
            "positions": position_array,
            "velocities": velocity_array,
            "accelerations": acceleration_array,
            "dynamic_obstacle_positions": dynamic_positions,
            "starts": starts,
            "goals": goals,
        }
        return episode, agent_rows, runtime_rows, trajectory_payload
    finally:
        env.close()
