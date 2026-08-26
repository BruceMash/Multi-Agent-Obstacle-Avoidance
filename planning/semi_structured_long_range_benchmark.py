"""Frozen grammar for the large-scale semi-structured long-range benchmark.

The grammar is deliberately method independent.  Stage changes only obstacle
population; workspace, mission distribution, geometry families, task-pattern
distribution, obstacle sizes, motion speeds, sensing contract, and dynamics
remain invariant.  The route witness is evaluation infrastructure and is never
exposed to any planner.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "semi_structured_long_range_scene_v1"
APPLICATION_SETTING = "LARGE_SCALE_SEMI_STRUCTURED_DYNAMIC_OPERATIONAL_ENVIRONMENT"
WORKSPACE_BOUNDS = ((0.0, 0.0, 0.8), (100.0, 100.0, 3.2))
OPERATIONAL_FLIGHT_BAND_M = (0.8, 3.2)
MISSION_DISTANCE_RANGE_M = (65.0, 85.0)
# The long-range execution contract uses the adapted 16 x 16 sensor encoder.
# Earlier scene manifests retained the historical 56-direction metadata even
# though their execution configs already instantiated 256 directions.  Formal
# manifests must state the executed contract rather than that stale provenance.
SENSOR_DIRECTION_COUNT = 256
SENSOR_RANGE_M = 4.5
DT = 0.1
LONG_RANGE_MAX_STEPS = 1500
STATIC_DYNAMIC_RATIO = 4
MISSION_RELEVANCE_DISTANCE_M = 18.0
STATIC_WITNESS_RESOLUTION_M = 1.0
STATIC_WITNESS_INFLATION_M = 0.45
MAX_SCENE_GENERATION_ATTEMPTS = 200

STAGE_ORDER = ("stage_1", "stage_2", "stage_3", "stage_4")
STAGE_LABELS = {
    "stage_1": "Stage I — Low-Density Operation",
    "stage_2": "Stage II — Moderate-Density Operation",
    "stage_3": "Stage III — High-Density Operation",
    "stage_4": "Stage IV — Peak-Density Mixed Operation",
}
STAGE_POPULATION = {
    "stage_1": {"static": 8, "dynamic": 2},
    "stage_2": {"static": 16, "dynamic": 4},
    "stage_3": {"static": 24, "dynamic": 6},
    "stage_4": {"static": 32, "dynamic": 8},
}
FAMILY_ORDER = (
    "parallel_structural_corridors",
    "cross_intersection_operational_area",
    "staggered_equipment_storage_blocks",
    "open_transfer_area_distributed_structures",
    "merge_bottleneck_operational_lanes",
)
FAMILY_LABELS = {
    "parallel_structural_corridors": "Parallel Structural Corridors",
    "cross_intersection_operational_area": "Cross-Intersection Operational Area",
    "staggered_equipment_storage_blocks": "Staggered Equipment / Storage Blocks",
    "open_transfer_area_distributed_structures": "Open Transfer Area with Distributed Structures",
    "merge_bottleneck_operational_lanes": "Merge / Bottleneck Operational Lanes",
}
TASK_PATTERNS = (
    "crossing",
    "merge",
    "opposite_direction_encounter",
    "parallel_shared_lane",
    "intersection_interaction",
    "shared_transfer_area",
)


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _box(center: Sequence[float], half_extents: Sequence[float], margin: float = 0.10) -> dict[str, Any]:
    return {
        "type": "box",
        "center": [float(value) for value in center],
        "half_extents": [float(value) for value in half_extents],
        "safety_margin": float(margin),
    }


def _cylinder(center: Sequence[float], radius: float, half_height: float = 1.20, margin: float = 0.10) -> dict[str, Any]:
    return {
        "type": "cylinder",
        "center": [float(value) for value in center],
        "radius": float(radius),
        "half_height": float(half_height),
        "safety_margin": float(margin),
    }


def _moving_sphere(center: Sequence[float], velocity: Sequence[float], radius: float, margin: float = 0.08) -> dict[str, Any]:
    return {
        "type": "moving_sphere_constant_translation",
        "center": [float(value) for value in center],
        "velocity": [float(value) for value in velocity],
        "radius": float(radius),
        "safety_margin": float(margin),
        "motion_model": "constant_direction_translation",
        "future_available_to_planner": False,
    }


def _base_tasks() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    z_low = 1.35
    z_mid = 2.00
    z_high = 2.65
    return (
        (
            np.asarray([[8, 30, z_low], [8, 50, z_mid], [8, 70, z_high]], dtype=float),
            np.asarray([[82, 70, z_high], [82, 30, z_low], [82, 50, z_mid]], dtype=float),
        ),
        (
            np.asarray([[8, 25, z_low], [8, 50, z_mid], [8, 75, z_high]], dtype=float),
            np.asarray([[82, 45, z_high], [82, 50, z_low], [82, 55, z_mid]], dtype=float),
        ),
        (
            np.asarray([[8, 40, z_low], [84, 50, z_mid], [8, 60, z_high]], dtype=float),
            np.asarray([[84, 60, z_high], [8, 50, z_low], [84, 40, z_mid]], dtype=float),
        ),
        (
            np.asarray([[8, 38, z_low], [8, 50, z_mid], [8, 62, z_high]], dtype=float),
            np.asarray([[84, 42, z_high], [84, 50, z_low], [84, 58, z_mid]], dtype=float),
        ),
        (
            np.asarray([[8, 50, z_low], [50, 8, z_mid], [8, 62, z_high]], dtype=float),
            np.asarray([[84, 50, z_high], [50, 84, z_low], [84, 38, z_mid]], dtype=float),
        ),
        (
            np.asarray([[14, 35, z_low], [50, 12, z_mid], [86, 35, z_high]], dtype=float),
            np.asarray([[86, 65, z_high], [50, 88, z_low], [14, 65, z_mid]], dtype=float),
        ),
    )


def _task_points(pattern_index: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    starts, goals = _base_tasks()[int(pattern_index) % len(TASK_PATTERNS)]
    starts = starts.copy()
    goals = goals.copy()
    starts[:, :2] += rng.uniform(-0.8, 0.8, size=(3, 2))
    goals[:, :2] += rng.uniform(-0.8, 0.8, size=(3, 2))
    starts[:, 2] += rng.uniform(-0.12, 0.12, size=3)
    goals[:, 2] += rng.uniform(-0.12, 0.12, size=3)
    return starts, goals


def _family_slot_centers(family_index: int) -> list[tuple[float, float]]:
    xs = (18.0, 28.0, 38.0, 48.0, 58.0, 68.0, 78.0, 88.0)
    if family_index == 0:
        ys_by_x = [(35.0, 45.0, 55.0, 65.0) for _ in xs]
    elif family_index == 1:
        ys_by_x = [
            (30.0, 40.0, 60.0, 70.0) if abs(x - 50.0) > 13.0 else (25.0, 38.0, 62.0, 75.0)
            for x in xs
        ]
    elif family_index == 2:
        ys_by_x = [
            (31.0, 43.0, 57.0, 69.0) if index % 2 == 0 else (35.0, 47.0, 53.0, 65.0)
            for index, _ in enumerate(xs)
        ]
    elif family_index == 3:
        ys_by_x = [
            (24.0, 38.0, 62.0, 76.0) if index % 2 == 0 else (29.0, 42.0, 58.0, 71.0)
            for index, _ in enumerate(xs)
        ]
    elif family_index == 4:
        ys_by_x = [
            (27.0 + 1.4 * index, 39.0 + 0.7 * index, 61.0 - 0.7 * index, 73.0 - 1.4 * index)
            for index, _ in enumerate(xs)
        ]
    else:
        raise ValueError(f"Unknown family index: {family_index}")
    return [(x, y) for x, ys in zip(xs, ys_by_x, strict=True) for y in ys]


def _static_slots(family_index: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot_id, (x, y) in enumerate(_family_slot_centers(family_index)):
        center = (
            x + rng.uniform(-0.65, 0.65),
            y + rng.uniform(-0.65, 0.65),
            2.0,
        )
        if (slot_id + family_index) % 4 == 2:
            rows.append(_cylinder(center, radius=0.55 + rng.uniform(0.0, 0.18)))
        else:
            if family_index in (0, 4):
                half_xy = (1.55 + rng.uniform(-0.20, 0.25), 0.75 + rng.uniform(-0.10, 0.18))
            elif family_index == 1:
                half_xy = (1.05 + rng.uniform(-0.15, 0.20), 1.05 + rng.uniform(-0.15, 0.20))
            elif family_index == 2:
                half_xy = (1.25 + rng.uniform(-0.18, 0.22), 0.90 + rng.uniform(-0.12, 0.18))
            else:
                half_xy = (1.00 + rng.uniform(-0.18, 0.25), 0.85 + rng.uniform(-0.15, 0.25))
            rows.append(_box(center, (half_xy[0], half_xy[1], 1.20)))
    return rows


def _dynamic_slots(rng: np.random.Generator) -> list[dict[str, Any]]:
    # Every slot intersects the support of all six frozen mission-pattern
    # families. The complete 150 s constant-direction track also remains
    # inside the closed 100 x 100 m workspace; no reflection, teleportation,
    # or stochastic wandering is used to keep an obstacle artificially local.
    templates = (
        ((35.0, 35.0, 1.35), (0.40, 0.00, 0.0)),
        ((65.0, 35.0, 2.00), (-0.40, 0.00, 0.0)),
        ((35.0, 65.0, 2.65), (0.40, 0.00, 0.0)),
        ((65.0, 65.0, 1.65), (-0.40, 0.00, 0.0)),
        ((40.0, 35.0, 2.35), (0.00, 0.40, 0.0)),
        ((45.0, 65.0, 1.35), (0.00, -0.40, 0.0)),
        ((60.0, 35.0, 1.75), (0.00, 0.40, 0.0)),
        ((55.0, 65.0, 2.55), (0.00, -0.40, 0.0)),
    )
    rows: list[dict[str, Any]] = []
    for slot_id, (center, velocity) in enumerate(templates):
        c = np.asarray(center, dtype=float)
        c[:2] += rng.uniform(-0.8, 0.8, size=2)
        c[2] += rng.uniform(-0.10, 0.10)
        speed_scale = rng.uniform(0.94, 1.06)
        v = np.asarray(velocity, dtype=float) * speed_scale
        rows.append(_moving_sphere(c, v, radius=0.30 + 0.02 * (slot_id % 3)))
    return rows


def _distance_point_to_segment_2d(point: np.ndarray, start: np.ndarray, goal: np.ndarray) -> float:
    segment = goal[:2] - start[:2]
    denom = float(np.dot(segment, segment))
    if denom <= 1.0e-12:
        return float(np.linalg.norm(point[:2] - start[:2]))
    alpha = float(np.clip(np.dot(point[:2] - start[:2], segment) / denom, 0.0, 1.0))
    projection = start[:2] + alpha * segment
    return float(np.linalg.norm(point[:2] - projection))


def mission_relevant_ratio(specs: Sequence[Mapping[str, Any]], starts: np.ndarray, goals: np.ndarray) -> float:
    if not specs:
        return 1.0
    relevant = 0
    for spec in specs:
        center = np.asarray(spec["center"], dtype=float)
        minimum = min(
            _distance_point_to_segment_2d(center, start, goal)
            for start, goal in zip(starts, goals, strict=True)
        )
        relevant += int(minimum <= MISSION_RELEVANCE_DISTANCE_M)
    return relevant / len(specs)


def _obstacle_clear_of_points(spec: Mapping[str, Any], points: np.ndarray, clearance: float = 1.0) -> bool:
    center = np.asarray(spec["center"], dtype=float)
    if spec["type"] == "box":
        half = np.asarray(spec["half_extents"], dtype=float) + float(spec.get("safety_margin", 0.0)) + clearance
        return bool(np.all(np.any(np.abs(points - center) > half, axis=1)))
    if spec["type"] == "cylinder":
        radius = float(spec["radius"]) + float(spec.get("safety_margin", 0.0)) + clearance
        return bool(np.all(np.linalg.norm(points[:, :2] - center[:2], axis=1) > radius))
    raise ValueError(spec["type"])


def _freeze_dynamic_trajectories(specs: Sequence[Mapping[str, Any]], *, dt: float, max_steps: int) -> list[list[list[float]]]:
    rows: list[list[list[float]]] = []
    times = np.arange(int(max_steps) + 1, dtype=float) * float(dt)
    for spec in specs:
        center = np.asarray(spec["center"], dtype=float)
        velocity = np.asarray(spec["velocity"], dtype=float)
        path = center[None, :] + times[:, None] * velocity[None, :]
        rows.append(path.tolist())
    return rows


def _projected_blocked_mask(static_specs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    lower = np.asarray(WORKSPACE_BOUNDS[0], dtype=float)
    upper = np.asarray(WORKSPACE_BOUNDS[1], dtype=float)
    nx = int(round((upper[0] - lower[0]) / STATIC_WITNESS_RESOLUTION_M)) + 1
    ny = int(round((upper[1] - lower[1]) / STATIC_WITNESS_RESOLUTION_M)) + 1
    xs = lower[0] + np.arange(nx) * STATIC_WITNESS_RESOLUTION_M
    ys = lower[1] + np.arange(ny) * STATIC_WITNESS_RESOLUTION_M
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    blocked = np.zeros((nx, ny), dtype=bool)
    for spec in static_specs:
        center = np.asarray(spec["center"], dtype=float)
        margin = float(spec.get("safety_margin", 0.0)) + STATIC_WITNESS_INFLATION_M
        if spec["type"] == "box":
            half = np.asarray(spec["half_extents"], dtype=float)
            blocked |= (np.abs(grid_x - center[0]) <= half[0] + margin) & (np.abs(grid_y - center[1]) <= half[1] + margin)
        elif spec["type"] == "cylinder":
            radius = float(spec["radius"]) + margin
            blocked |= (grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2 <= radius**2
        else:
            raise ValueError(spec["type"])
    return blocked


def _grid_index(point: np.ndarray) -> tuple[int, int]:
    lower = np.asarray(WORKSPACE_BOUNDS[0], dtype=float)
    value = np.rint((point[:2] - lower[:2]) / STATIC_WITNESS_RESOLUTION_M).astype(int)
    return int(value[0]), int(value[1])


def _astar_length(blocked: np.ndarray, start: np.ndarray, goal: np.ndarray) -> float | None:
    start_index = _grid_index(start)
    goal_index = _grid_index(goal)
    if blocked[start_index] or blocked[goal_index]:
        return None
    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    queue: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start_index)]
    best = {start_index: 0.0}
    while queue:
        _, cost, node = heapq.heappop(queue)
        if cost > best.get(node, float("inf")) + 1.0e-12:
            continue
        if node == goal_index:
            return cost * STATIC_WITNESS_RESOLUTION_M
        for dx, dy, step_cost in moves:
            nxt = (node[0] + dx, node[1] + dy)
            if not (0 <= nxt[0] < blocked.shape[0] and 0 <= nxt[1] < blocked.shape[1]):
                continue
            if blocked[nxt]:
                continue
            new_cost = cost + step_cost
            if new_cost + 1.0e-12 < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                heuristic = math.hypot(goal_index[0] - nxt[0], goal_index[1] - nxt[1])
                heapq.heappush(queue, (new_cost + heuristic, new_cost, nxt))
    return None


def static_route_witness(starts: np.ndarray, goals: np.ndarray, static_specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocked = _projected_blocked_mask(static_specs)
    lengths = [_astar_length(blocked, start, goal) for start, goal in zip(starts, goals, strict=True)]
    return {
        "method": "2d_grid_astar_full_height_projection_evaluation_only",
        "resolution_m": STATIC_WITNESS_RESOLUTION_M,
        "inflation_m": STATIC_WITNESS_INFLATION_M,
        "all_agents_feasible": all(value is not None for value in lengths),
        "path_lengths_m": lengths,
    }


def _pairwise_minimum(points: np.ndarray) -> float:
    return min(
        float(np.linalg.norm(points[left] - points[right]))
        for left in range(len(points))
        for right in range(left + 1, len(points))
    )


def _entry_attempt(*, stage: str, scenario_index: int, seed: int, prefix: str, attempt: int) -> dict[str, Any]:
    family_index = int(scenario_index) % len(FAMILY_ORDER)
    family = FAMILY_ORDER[family_index]
    pattern_index = int(scenario_index) % len(TASK_PATTERNS)
    task_pattern = TASK_PATTERNS[pattern_index]
    rng = np.random.default_rng(int(seed) + 104729 * int(attempt))
    starts, goals = _task_points(pattern_index, rng)
    population = STAGE_POPULATION[stage]

    static_slots = _static_slots(family_index, rng)
    protected = np.vstack([starts, goals])
    static_slots = [row for row in static_slots if _obstacle_clear_of_points(row, protected)]
    if len(static_slots) < population["static"]:
        raise RuntimeError("insufficient protected static slots")
    static_order = rng.permutation(len(static_slots))
    static_specs = [static_slots[int(index)] for index in static_order[: population["static"]]]

    dynamic_slots = _dynamic_slots(rng)
    dynamic_order = rng.permutation(len(dynamic_slots))
    dynamic_specs = [dynamic_slots[int(index)] for index in dynamic_order[: population["dynamic"]]]
    dynamic_tracks = _freeze_dynamic_trajectories(dynamic_specs, dt=DT, max_steps=LONG_RANGE_MAX_STEPS)

    task_distances = np.linalg.norm(goals - starts, axis=1)
    witness = static_route_witness(starts, goals, static_specs)
    scene_id = f"{prefix}{STAGE_ORDER.index(stage) + 1}_{int(scenario_index):03d}"
    return {
        "schema_version": SCHEMA_VERSION,
        "application_setting": APPLICATION_SETTING,
        "scenario_id": scene_id,
        "scenario_index": int(scenario_index),
        "stage": stage,
        "stage_label": STAGE_LABELS[stage],
        "stage_index": STAGE_ORDER.index(stage) + 1,
        "family": family,
        "family_label": FAMILY_LABELS[family],
        "family_index": family_index,
        "task_pattern": task_pattern,
        "task_pattern_index": pattern_index,
        "seed": int(seed),
        "generation_attempt": int(attempt),
        "starts": starts.tolist(),
        "initial_velocities": np.zeros_like(starts).tolist(),
        "goals": goals.tolist(),
        "workspace_bounds": json_ready(WORKSPACE_BOUNDS),
        "operational_flight_band_m": json_ready(OPERATIONAL_FLIGHT_BAND_M),
        "boundary_mode": "closed_workspace_with_floor_and_ceiling",
        "dt": DT,
        "max_steps": LONG_RANGE_MAX_STEPS,
        "static_obstacles": static_specs,
        "dynamic_obstacles": dynamic_specs,
        "dynamic_obstacle_trajectories": dynamic_tracks,
        "static_route_witness_evaluation_only": witness,
        "difficulty": {
            "active_variable": "OBSTACLE_POPULATION_ONLY",
            "static_obstacle_count": len(static_specs),
            "dynamic_obstacle_count": len(dynamic_specs),
            "total_obstacle_count": len(static_specs) + len(dynamic_specs),
            "static_dynamic_ratio": STATIC_DYNAMIC_RATIO,
            "mission_relevant_static_ratio": mission_relevant_ratio(static_specs, starts, goals),
            "mission_relevant_dynamic_ratio": mission_relevant_ratio(dynamic_specs, starts, goals),
        },
        "mission": {
            "straight_line_distances_m": task_distances.tolist(),
            "straight_line_distance_min_m": float(np.min(task_distances)),
            "straight_line_distance_mean_m": float(np.mean(task_distances)),
            "straight_line_distance_max_m": float(np.max(task_distances)),
            "start_pairwise_minimum_m": _pairwise_minimum(starts),
            "goal_pairwise_minimum_m": _pairwise_minimum(goals),
        },
        "information_contract": {
            "future_dynamic_trajectory_available_to_environment": True,
            "future_dynamic_trajectory_available_to_planner": False,
            "complete_control_rate_global_dynamic_map_required": False,
        },
    }


def generate_scenario_entry(*, stage: str, scenario_index: int, seed: int, prefix: str) -> dict[str, Any]:
    if stage not in STAGE_ORDER:
        raise ValueError(stage)
    last_reason = "unknown"
    for attempt in range(MAX_SCENE_GENERATION_ATTEMPTS):
        try:
            entry = _entry_attempt(stage=stage, scenario_index=scenario_index, seed=seed, prefix=prefix, attempt=attempt)
        except RuntimeError as error:
            last_reason = str(error)
            continue
        distances = np.asarray(entry["mission"]["straight_line_distances_m"], dtype=float)
        if np.any(distances < MISSION_DISTANCE_RANGE_M[0]) or np.any(distances > MISSION_DISTANCE_RANGE_M[1]):
            last_reason = "mission_distance_out_of_contract"
            continue
        if not entry["static_route_witness_evaluation_only"]["all_agents_feasible"]:
            last_reason = "static_route_infeasible"
            continue
        if entry["difficulty"]["mission_relevant_static_ratio"] < 0.75:
            last_reason = "static_mission_relevance_too_low"
            continue
        if entry["difficulty"]["mission_relevant_dynamic_ratio"] < 0.75:
            last_reason = "dynamic_mission_relevance_too_low"
            continue
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
        entry["dynamic_track_fingerprint"] = stable_hash(entry["dynamic_obstacle_trajectories"])
        translation_origin = np.mean(np.asarray(entry["starts"] + entry["goals"], dtype=float), axis=0)
        translation_payload = {
            "starts": (np.asarray(entry["starts"], dtype=float) - translation_origin).round(5).tolist(),
            "goals": (np.asarray(entry["goals"], dtype=float) - translation_origin).round(5).tolist(),
            "static": [
                {
                    **{key: value for key, value in spec.items() if key != "center"},
                    "center": (np.asarray(spec["center"], dtype=float) - translation_origin).round(5).tolist(),
                }
                for spec in entry["static_obstacles"]
            ],
        }
        entry["translation_invariant_fingerprint"] = stable_hash(translation_payload)
        entry["environment_fingerprint"] = stable_hash(
            {**geometry_payload, "workspace_bounds": entry["workspace_bounds"], "dt": entry["dt"], "max_steps": entry["max_steps"]}
        )
        return entry
    raise RuntimeError(
        f"failed to generate valid scene after {MAX_SCENE_GENERATION_ATTEMPTS} attempts: {last_reason}"
    )


def generate_scenario_manifest(*, counts_per_stage: int, seed_base: int, prefix: str) -> dict[str, Any]:
    entries = []
    for stage_index, stage in enumerate(STAGE_ORDER):
        for scenario_index in range(int(counts_per_stage)):
            seed = int(seed_base + stage_index * 100_000 + scenario_index)
            entries.append(generate_scenario_entry(stage=stage, scenario_index=scenario_index, seed=seed, prefix=prefix))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "application_setting": APPLICATION_SETTING,
        "workspace_bounds": json_ready(WORKSPACE_BOUNDS),
        "operational_flight_band_m": json_ready(OPERATIONAL_FLIGHT_BAND_M),
        "mission_distance_range_m": json_ready(MISSION_DISTANCE_RANGE_M),
        "sensor_direction_count": SENSOR_DIRECTION_COUNT,
        "sensor_range_m": SENSOR_RANGE_M,
        "dt": DT,
        "max_steps": LONG_RANGE_MAX_STEPS,
        "stage_order": list(STAGE_ORDER),
        "stage_population": json_ready(STAGE_POPULATION),
        "static_dynamic_ratio": STATIC_DYNAMIC_RATIO,
        "family_order": list(FAMILY_ORDER),
        "task_patterns": list(TASK_PATTERNS),
        "counts_per_stage": int(counts_per_stage),
        "unique_scenario_count": len(entries),
        "seed_base": int(seed_base),
        "entries": entries,
    }
    manifest["manifest_sha256"] = stable_hash({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    return manifest


def validate_scenario_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = list(manifest["entries"])
    errors: list[str] = []
    expected = int(manifest["counts_per_stage"]) * len(STAGE_ORDER)
    if len(entries) != expected:
        errors.append("scenario_count")
    ids = [str(row["scenario_id"]) for row in entries]
    if len(set(ids)) != len(ids):
        errors.append("scenario_id_duplicate")
    geometry = [str(row["geometry_fingerprint"]) for row in entries]
    if len(set(geometry)) != len(geometry):
        errors.append("geometry_duplicate")
    translation = [str(row["translation_invariant_fingerprint"]) for row in entries]
    if len(set(translation)) != len(translation):
        errors.append("translation_equivalent_duplicate")

    for stage in STAGE_ORDER:
        members = [row for row in entries if row["stage"] == stage]
        if len(members) != int(manifest["counts_per_stage"]):
            errors.append(f"{stage}_count")
        family_counts = {family: sum(row["family"] == family for row in members) for family in FAMILY_ORDER}
        if max(family_counts.values(), default=0) - min(family_counts.values(), default=0) > 1:
            errors.append(f"{stage}_family_balance")
        task_counts = {task: sum(row["task_pattern"] == task for row in members) for task in TASK_PATTERNS}
        if max(task_counts.values(), default=0) - min(task_counts.values(), default=0) > 1:
            errors.append(f"{stage}_task_balance")
        target = STAGE_POPULATION[stage]
        for row in members:
            if len(row["static_obstacles"]) != target["static"] or len(row["dynamic_obstacles"]) != target["dynamic"]:
                errors.append(f"{stage}_population")
                break

    for row in entries:
        distances = np.asarray(row["mission"]["straight_line_distances_m"], dtype=float)
        if np.any(distances < MISSION_DISTANCE_RANGE_M[0]) or np.any(distances > MISSION_DISTANCE_RANGE_M[1]):
            errors.append("mission_distance")
            break
        if not bool(row["static_route_witness_evaluation_only"]["all_agents_feasible"]):
            errors.append("static_route_feasibility")
            break
        if row["boundary_mode"] != "closed_workspace_with_floor_and_ceiling":
            errors.append("boundary_contract")
            break
        if any(spec["motion_model"] != "constant_direction_translation" for spec in row["dynamic_obstacles"]):
            errors.append("dynamic_motion_model")
            break
        if any(len(track) != LONG_RANGE_MAX_STEPS + 1 for track in row["dynamic_obstacle_trajectories"]):
            errors.append("dynamic_track_length")
            break
        lower = np.asarray(WORKSPACE_BOUNDS[0], dtype=float)
        upper = np.asarray(WORKSPACE_BOUNDS[1], dtype=float)
        tracks = [np.asarray(track, dtype=float) for track in row["dynamic_obstacle_trajectories"]]
        if any(np.any(track < lower - 1.0e-9) or np.any(track > upper + 1.0e-9) for track in tracks):
            errors.append("dynamic_track_boundary")
            break
        if float(row["difficulty"]["mission_relevant_static_ratio"]) < 0.75:
            errors.append("static_mission_relevance")
            break
        if float(row["difficulty"]["mission_relevant_dynamic_ratio"]) < 0.75:
            errors.append("dynamic_mission_relevance")
            break

    population_sequence = [STAGE_POPULATION[stage]["static"] + STAGE_POPULATION[stage]["dynamic"] for stage in STAGE_ORDER]
    if population_sequence != sorted(population_sequence) or len(set(population_sequence)) != len(population_sequence):
        errors.append("population_monotonicity")
    ratios = {
        STAGE_POPULATION[stage]["static"] / STAGE_POPULATION[stage]["dynamic"]
        for stage in STAGE_ORDER
    }
    if ratios != {float(STATIC_DYNAMIC_RATIO)}:
        errors.append("static_dynamic_ratio")

    return {
        "schema_version": "semi_structured_long_range_manifest_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "scenario_count": len(entries),
        "stage_counts": {stage: sum(row["stage"] == stage for row in entries) for stage in STAGE_ORDER},
        "family_counts": {family: sum(row["family"] == family for row in entries) for family in FAMILY_ORDER},
        "task_pattern_counts": {task: sum(row["task_pattern"] == task for row in entries) for task in TASK_PATTERNS},
        "exact_geometry_duplicates": len(entries) - len(set(geometry)),
        "translation_equivalent_duplicates": len(entries) - len(set(translation)),
        "static_route_feasible_count": sum(bool(row["static_route_witness_evaluation_only"]["all_agents_feasible"]) for row in entries),
        "minimum_mission_relevant_static_ratio": min(float(row["difficulty"]["mission_relevant_static_ratio"]) for row in entries),
        "minimum_mission_relevant_dynamic_ratio": min(float(row["difficulty"]["mission_relevant_dynamic_ratio"]) for row in entries),
        "population_sequence": population_sequence,
        "static_dynamic_ratio": STATIC_DYNAMIC_RATIO,
        "active_difficulty_variable": "OBSTACLE_POPULATION_ONLY",
        "local_decision_recoverability": "YES" if not errors else "NOT_ESTABLISHED",
    }


__all__ = [
    "APPLICATION_SETTING",
    "DT",
    "FAMILY_LABELS",
    "FAMILY_ORDER",
    "LONG_RANGE_MAX_STEPS",
    "MISSION_DISTANCE_RANGE_M",
    "OPERATIONAL_FLIGHT_BAND_M",
    "SCHEMA_VERSION",
    "SENSOR_DIRECTION_COUNT",
    "SENSOR_RANGE_M",
    "STAGE_LABELS",
    "STAGE_ORDER",
    "STAGE_POPULATION",
    "STATIC_DYNAMIC_RATIO",
    "TASK_PATTERNS",
    "WORKSPACE_BOUNDS",
    "generate_scenario_entry",
    "generate_scenario_manifest",
    "json_ready",
    "stable_hash",
    "static_route_witness",
    "validate_scenario_manifest",
]
