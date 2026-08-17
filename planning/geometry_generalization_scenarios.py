"""Frozen non-translation-equivalent multi-agent geometry layouts.

This module is deliberately policy-free.  It contains only deterministic
geometry construction, analytic descriptors, and initial-state acceptance
checks.  Proposal, FP-SHEP, SAC, rollout, and GAT code must never be imported
or called from this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from Entity.static_obstacles import StaticSphereObstacle


SCHEMA_VERSION = "geometry_generalization_scenarios_v1"
SCENARIO_ID = "multi_agent_obstacle_generalization"
SCENARIO_ROLE = "NON-TRANSLATION-EQUIVALENT STATIC-OBSTACLE + MULTI-AGENT COUPLING AUDIT"
LAYOUT_SET_ROLE = "held_out_geometry_generalization_not_used_for_tuning"
NUM_AGENTS = 3
STATIC_OBSTACLE_COUNT = 2
DYNAMIC_OBSTACLE_COUNT = 0
OBSTACLE_RADIUS_M = 0.45
OBSTACLE_SAFETY_MARGIN_M = 0.10
OBSTACLE_EFFECTIVE_RADIUS_M = OBSTACLE_RADIUS_M + OBSTACLE_SAFETY_MARGIN_M
PEER_RADIUS_M = 0.30
WORKSPACE_BOUNDS = ((-0.5, -2.5, -1.2), (8.5, 2.0, 1.2))
MIN_START_GOAL_DISTANCE_M = 6.0
MIN_INITIAL_UAV_DISTANCE_M = 0.6
MIN_TERMINAL_UAV_DISTANCE_M = 1.2


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RouteSpec:
    direction: tuple[float, float, float]
    crossing_fraction: float = 0.5
    crossing_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def endpoints(self, crossing_center: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(crossing_center, dtype=float) + np.asarray(
            self.crossing_offset, dtype=float
        )
        direction = np.asarray(self.direction, dtype=float)
        fraction = float(self.crossing_fraction)
        return center - fraction * direction, center + (1.0 - fraction) * direction


@dataclass(frozen=True)
class ObstacleSpec:
    route_agent: int
    route_fraction: float
    offset: tuple[float, float, float]


@dataclass(frozen=True)
class FrozenLayoutSpec:
    layout_id: str
    family: str
    family_description: str
    crossing_center: tuple[float, float, float]
    routes: tuple[RouteSpec, RouteSpec, RouteSpec]
    obstacles: tuple[ObstacleSpec, ObstacleSpec]
    structural_variant: str

    def geometry(self) -> tuple[np.ndarray, np.ndarray, tuple[StaticSphereObstacle, ...]]:
        endpoints = [route.endpoints(self.crossing_center) for route in self.routes]
        starts = np.stack([item[0] for item in endpoints], axis=0)
        goals = np.stack([item[1] for item in endpoints], axis=0)
        obstacles: list[StaticSphereObstacle] = []
        for item in self.obstacles:
            agent = int(item.route_agent)
            center = (
                starts[agent]
                + float(item.route_fraction) * (goals[agent] - starts[agent])
                + np.asarray(item.offset, dtype=float)
            )
            obstacles.append(
                StaticSphereObstacle(
                    center=center,
                    radius=OBSTACLE_RADIUS_M,
                    safety_margin=OBSTACLE_SAFETY_MARGIN_M,
                )
            )
        return starts, goals, tuple(obstacles)

    def parameter_record(self) -> dict[str, Any]:
        return {
            "layout_id": self.layout_id,
            "family": self.family,
            "family_description": self.family_description,
            "crossing_center": list(self.crossing_center),
            "routes": [
                {
                    "direction": list(item.direction),
                    "crossing_fraction": float(item.crossing_fraction),
                    "crossing_offset": list(item.crossing_offset),
                }
                for item in self.routes
            ],
            "obstacles": [
                {
                    "route_agent": int(item.route_agent),
                    "route_fraction": float(item.route_fraction),
                    "offset": list(item.offset),
                    "radius_m": OBSTACLE_RADIUS_M,
                    "safety_margin_m": OBSTACLE_SAFETY_MARGIN_M,
                }
                for item in self.obstacles
            ],
            "structural_variant": self.structural_variant,
        }


def _routes(
    directions: Sequence[Sequence[float]],
    *,
    fractions: Sequence[float] = (0.5, 0.5, 0.5),
    offsets: Sequence[Sequence[float]] = ((0.0, 0.0, 0.0),) * 3,
) -> tuple[RouteSpec, RouteSpec, RouteSpec]:
    values = tuple(
        RouteSpec(
            direction=tuple(float(value) for value in direction),
            crossing_fraction=float(fraction),
            crossing_offset=tuple(float(value) for value in offset),
        )
        for direction, fraction, offset in zip(
            directions, fractions, offsets, strict=True
        )
    )
    if len(values) != NUM_AGENTS:
        raise ValueError("exactly three routes are required")
    return values  # type: ignore[return-value]


def _obstacles(
    first: tuple[int, float, Sequence[float]],
    second: tuple[int, float, Sequence[float]],
) -> tuple[ObstacleSpec, ObstacleSpec]:
    return tuple(
        ObstacleSpec(
            route_agent=int(agent),
            route_fraction=float(fraction),
            offset=tuple(float(value) for value in offset),
        )
        for agent, fraction, offset in (first, second)
    )  # type: ignore[return-value]


def _layout(
    family: str,
    index: int,
    description: str,
    center: Sequence[float],
    directions: Sequence[Sequence[float]],
    obstacles: tuple[ObstacleSpec, ObstacleSpec],
    *,
    fractions: Sequence[float] = (0.5, 0.5, 0.5),
    offsets: Sequence[Sequence[float]] = ((0.0, 0.0, 0.0),) * 3,
    variant: str,
) -> FrozenLayoutSpec:
    return FrozenLayoutSpec(
        layout_id=f"HOLDOUT_{family}_{index:02d}",
        family=family,
        family_description=description,
        crossing_center=tuple(float(value) for value in center),
        routes=_routes(directions, fractions=fractions, offsets=offsets),
        obstacles=obstacles,
        structural_variant=variant,
    )


_A = "moderate-angle crossing + obstacles away from exact crossing"
_B = "acute-angle crossing + obstacle near one approach corridor"
_C = "actual 3D near-orthogonal crossing + asymmetric obstacle placement"
_D = "two-agent crossing + third-agent nearby parallel passage"
_E = "terminal-assignment-induced crossing + obstacle-constrained detour"
_F = "asymmetric start spacing + asymmetric obstacle corridor"


# This explicit table is the complete formal held-out geometry set.  Its
# entries are structural variations, not random jitter samples.
FROZEN_HELD_OUT_LAYOUTS: tuple[FrozenLayoutSpec, ...] = (
    _layout("A", 0, _A, (4.0, -0.25, 0.0), ((7.2, 2.2, 0.6), (7.2, -2.2, -0.6), (7.4, -0.4, 1.4)), _obstacles((0, 0.28, (0.0, 0.20, 0.0)), (1, 0.72, (0.0, -0.20, 0.10))), variant="baseline_moderate"),
    _layout("A", 1, _A, (4.1, -0.20, 0.0), ((7.0, 2.5, 0.5), (7.3, -2.0, -0.8), (7.2, 0.2, 1.0)), _obstacles((1, 0.30, (0.0, 0.20, 0.05)), (2, 0.69, (0.0, -0.25, -0.10))), fractions=(0.48, 0.52, 0.50), variant="angle_and_crossing_time_shift"),
    _layout("A", 2, _A, (3.9, -0.35, 0.0), ((7.4, 1.9, 0.8), (7.0, -2.5, -0.5), (7.5, 0.1, -0.9)), _obstacles((0, 0.34, (0.0, -0.25, 0.0)), (1, 0.66, (0.0, 0.20, 0.15))), fractions=(0.51, 0.49, 0.50), variant="asymmetric_route_slopes"),
    _layout("A", 3, _A, (4.2, -0.15, 0.0), ((7.3, 2.3, -0.6), (7.1, -2.1, 0.7), (7.0, -0.1, -1.0)), _obstacles((2, 0.27, (0.0, 0.25, 0.10)), (0, 0.74, (0.0, -0.20, -0.05))), fractions=(0.49, 0.51, 0.50), variant="vertical_sign_swap"),

    _layout("B", 0, _B, (4.0, -0.30, 0.0), ((7.4, 1.8, 1.2), (7.2, -1.2, -1.0), (7.0, 1.4, -1.8)), _obstacles((0, 0.38, (0.0, 0.15, 0.0)), (2, 0.70, (0.0, -0.20, 0.10))), fractions=(0.50, 0.48, 0.52), variant="acute_three_route"),
    _layout("B", 1, _B, (4.1, -0.25, 0.0), ((7.2, 1.6, 1.2), (7.4, -1.4, -0.8), (7.0, 1.2, -1.8)), _obstacles((1, 0.35, (0.0, -0.18, 0.05)), (0, 0.73, (0.0, 0.20, -0.10))), fractions=(0.49, 0.51, 0.50), variant="acute_obstacle_side_swap"),
    _layout("B", 2, _B, (3.95, -0.20, 0.0), ((7.5, 1.7, 0.9), (7.0, -1.3, -1.1), (7.1, 1.4, -1.8)), _obstacles((2, 0.32, (0.0, 0.15, 0.05)), (1, 0.68, (0.0, -0.15, 0.10))), fractions=(0.52, 0.48, 0.50), variant="acute_longitudinal_shift"),
    _layout("B", 3, _B, (4.15, -0.35, 0.0), ((7.1, 1.6, -1.2), (7.3, -1.3, 0.9), (7.0, 1.2, 1.8)), _obstacles((0, 0.36, (0.0, -0.20, 0.10)), (2, 0.71, (0.0, 0.18, -0.05))), fractions=(0.48, 0.50, 0.52), variant="acute_vertical_mirror"),

    _layout("C", 0, _C, (4.0, -0.25, 0.0), ((6.0, 3.0, 1.2), (-3.5, 4.5, 2.4), (7.0, -1.5, -0.8)), _obstacles((0, 0.31, (0.0, 0.20, -0.05)), (1, 0.67, (0.15, -0.20, 0.0))), variant="3d_orthogonal_counterflow"),
    _layout("C", 1, _C, (4.1, -0.25, 0.0), ((6.2, 2.8, 1.0), (-3.6, 4.4, 2.2), (7.0, -1.2, -1.0)), _obstacles((1, 0.33, (0.10, 0.15, 0.0)), (2, 0.69, (0.0, -0.20, 0.10))), fractions=(0.49, 0.51, 0.50), variant="3d_orthogonal_angle_shift"),
    _layout("C", 2, _C, (3.9, -0.15, 0.0), ((5.8, 3.2, 1.0), (-3.65, 4.2, 2.3), (7.2, -1.4, -0.7)), _obstacles((0, 0.35, (0.0, -0.20, 0.10)), (1, 0.65, (-0.10, 0.20, -0.05))), fractions=(0.51, 0.49, 0.50), variant="3d_orthogonal_crossing_shift"),
    _layout("C", 3, _C, (4.2, -0.20, 0.0), ((6.4, 2.6, 1.2), (-3.5, 4.4, 2.4), (7.0, -1.6, -0.6)), _obstacles((2, 0.30, (0.0, 0.20, 0.05)), (0, 0.71, (0.0, -0.18, -0.10))), fractions=(0.50, 0.50, 0.48), variant="3d_orthogonal_obstacle_asymmetry"),

    _layout("D", 0, _D, (4.0, -0.30, 0.0), ((7.4, 2.4, 0.6), (7.4, -2.4, -0.6), (7.2, 0.2, 0.5)), _obstacles((0, 0.32, (0.0, 0.15, 0.0)), (1, 0.68, (0.0, -0.20, 0.10))), offsets=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, -0.75, 0.65)), variant="third_agent_upper_parallel"),
    _layout("D", 1, _D, (4.1, -0.20, 0.0), ((7.2, 2.5, 0.5), (7.3, -2.2, -0.7), (7.0, 0.0, 0.0)), _obstacles((1, 0.34, (0.0, 0.20, 0.0)), (2, 0.70, (0.0, -0.18, -0.05))), offsets=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.30, 1.00)), fractions=(0.49, 0.51, 0.50), variant="third_agent_lower_parallel"),
    _layout("D", 2, _D, (3.95, -0.30, 0.0), ((7.5, 2.1, 0.8), (7.1, -2.5, -0.4), (7.2, 0.0, 0.0)), _obstacles((2, 0.30, (0.0, -0.20, 0.0)), (0, 0.72, (0.0, 0.20, 0.10))), offsets=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, -0.20, -1.00)), fractions=(0.51, 0.49, 0.50), variant="third_agent_vertical_mirror"),
    _layout("D", 3, _D, (4.15, -0.25, 0.0), ((7.2, 2.3, -0.7), (7.4, -2.1, 0.6), (7.0, 0.0, 0.0)), _obstacles((0, 0.29, (0.0, -0.18, 0.05)), (1, 0.71, (0.0, 0.18, -0.10))), offsets=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.00, -1.00)), variant="third_agent_offset_and_sign_swap"),

    _layout("E", 0, _E, (4.0, -0.25, 0.0), ((7.4, 2.8, 0.6), (7.2, -2.8, -0.6), (-6.4, 1.8, 1.2)), _obstacles((0, 0.40, (0.0, 0.10, 0.0)), (2, 0.60, (0.0, -0.15, 0.10))), variant="counterflow_terminal_assignment"),
    _layout("E", 1, _E, (4.1, -0.20, 0.0), ((7.2, 2.6, 0.8), (7.4, -2.5, -0.5), (-6.2, 2.0, 1.4)), _obstacles((1, 0.38, (0.0, -0.15, 0.05)), (2, 0.63, (0.10, 0.15, -0.05))), fractions=(0.49, 0.51, 0.50), variant="counterflow_assignment_shift"),
    _layout("E", 2, _E, (3.95, -0.30, 0.0), ((7.5, 2.4, -0.7), (7.0, -2.9, 0.6), (-6.3, 1.9, -1.3)), _obstacles((2, 0.36, (-0.10, 0.15, 0.0)), (0, 0.65, (0.0, -0.15, 0.10))), fractions=(0.51, 0.49, 0.50), variant="counterflow_vertical_mirror"),
    _layout("E", 3, _E, (4.15, -0.15, 0.0), ((7.1, 2.9, 0.5), (7.3, -2.6, -0.8), (-6.4, 1.7, 1.1)), _obstacles((0, 0.42, (0.0, 0.12, -0.05)), (1, 0.58, (0.0, -0.12, 0.05))), fractions=(0.48, 0.52, 0.50), variant="terminal_assignment_crossing_time_shift"),

    _layout("F", 0, _F, (4.0, -0.25, 0.0), ((7.4, 2.0, 0.7), (7.2, -2.3, -0.5), (7.0, 0.4, -1.2)), _obstacles((0, 0.33, (0.0, 0.25, 0.0)), (2, 0.64, (0.0, -0.30, 0.10))), fractions=(0.42, 0.50, 0.58), variant="longitudinal_spacing_forward"),
    _layout("F", 1, _F, (4.05, -0.20, 0.0), ((7.2, 2.2, 0.5), (7.4, -2.0, -0.7), (7.0, 0.2, 1.2)), _obstacles((1, 0.31, (0.0, -0.25, 0.05)), (0, 0.67, (0.0, 0.30, -0.05))), fractions=(0.58, 0.42, 0.42), variant="longitudinal_spacing_reordered"),
    _layout("F", 2, _F, (3.95, -0.30, 0.0), ((7.5, 1.8, -0.8), (7.0, -2.5, 0.4), (7.1, 0.5, 1.0)), _obstacles((2, 0.35, (0.0, 0.30, -0.10)), (1, 0.62, (0.0, -0.25, 0.05))), fractions=(0.45, 0.55, 0.50), variant="asymmetric_corridor_vertical"),
    _layout("F", 3, _F, (4.1, -0.25, 0.0), ((7.1, 2.4, 0.6), (7.3, -2.1, -0.6), (6.4, -0.3, -1.1)), _obstacles((0, 0.29, (0.0, -0.30, 0.10)), (2, 0.70, (0.0, 0.28, -0.05))), fractions=(0.44, 0.52, 0.67), variant="maximum_asymmetry"),
)


FROZEN_LAYOUT_TABLE_SHA256 = stable_hash(
    [layout.parameter_record() for layout in FROZEN_HELD_OUT_LAYOUTS]
)


def layout_by_id(layout_id: str) -> FrozenLayoutSpec:
    matches = [item for item in FROZEN_HELD_OUT_LAYOUTS if item.layout_id == layout_id]
    if len(matches) != 1:
        raise KeyError(f"unknown or duplicate layout_id: {layout_id}")
    return matches[0]


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return float("nan")
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _segment_closest_parameters(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> tuple[float, float, float]:
    """Return distance and independent segment fractions at spatial closest approach."""

    p = np.asarray(first_start, dtype=float)
    q = np.asarray(second_start, dtype=float)
    d1 = np.asarray(first_end, dtype=float) - p
    d2 = np.asarray(second_end, dtype=float) - q
    r = p - q
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    epsilon = 1e-12
    if a <= epsilon and e <= epsilon:
        return float(np.linalg.norm(p - q)), 0.0, 0.0
    if a <= epsilon:
        first_t = 0.0
        second_t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= epsilon:
            second_t = 0.0
            first_t = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denominator = a * e - b * b
            first_t = (
                float(np.clip((b * f - c * e) / denominator, 0.0, 1.0))
                if abs(denominator) > epsilon
                else 0.0
            )
            second_t = (b * first_t + f) / e
            if second_t < 0.0:
                second_t = 0.0
                first_t = float(np.clip(-c / a, 0.0, 1.0))
            elif second_t > 1.0:
                second_t = 1.0
                first_t = float(np.clip((b - c) / a, 0.0, 1.0))
    first_point = p + first_t * d1
    second_point = q + second_t * d2
    return float(np.linalg.norm(first_point - second_point)), first_t, second_t


def pairwise_route_descriptors(starts: np.ndarray, goals: np.ndarray) -> list[dict[str, Any]]:
    starts = np.asarray(starts, dtype=float)
    goals = np.asarray(goals, dtype=float)
    rows: list[dict[str, Any]] = []
    for first in range(len(starts)):
        for second in range(first + 1, len(starts)):
            first_direction = goals[first] - starts[first]
            second_direction = goals[second] - starts[second]
            relative_start = starts[first] - starts[second]
            relative_delta = first_direction - second_direction
            denominator = float(np.dot(relative_delta, relative_delta))
            synchronized_t = (
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
            synchronized_delta = relative_start + synchronized_t * relative_delta
            spatial_distance, first_t, second_t = _segment_closest_parameters(
                starts[first], goals[first], starts[second], goals[second]
            )
            rows.append(
                {
                    "pair": [first, second],
                    "route_angle_3d_deg": _angle_degrees(
                        first_direction, second_direction
                    ),
                    "route_angle_xy_deg": _angle_degrees(
                        first_direction[:2], second_direction[:2]
                    ),
                    "direct_path_predicted_minimum_inter_agent_distance_m": float(
                        np.linalg.norm(synchronized_delta)
                    ),
                    "predicted_closest_approach_time_fraction": synchronized_t,
                    "spatial_path_minimum_distance_m": spatial_distance,
                    "spatial_closest_time_fraction_first": first_t,
                    "spatial_closest_time_fraction_second": second_t,
                    "predicted_closest_approach_time_difference_fraction": abs(
                        first_t - second_t
                    ),
                    "predicted_closest_vertical_separation_m": abs(
                        float(synchronized_delta[2])
                    ),
                }
            )
    return rows


def _point_to_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    denominator = float(np.dot(delta, delta))
    fraction = (
        0.0
        if denominator <= 1e-12
        else float(
            np.clip(
                np.dot(np.asarray(point, dtype=float) - start, delta) / denominator,
                0.0,
                1.0,
            )
        )
    )
    return float(np.linalg.norm(np.asarray(point, dtype=float) - (start + fraction * delta)))


def _pairwise_minimum(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    upper = distances[np.triu_indices(len(points), k=1)]
    return float(np.min(upper)) if upper.size else float("inf")


def relative_geometry_record(
    starts: np.ndarray,
    goals: np.ndarray,
    obstacles: Sequence[StaticSphereObstacle],
) -> dict[str, Any]:
    starts = np.asarray(starts, dtype=float)
    goals = np.asarray(goals, dtype=float)
    obstacle_centers = np.stack([np.asarray(item.center, dtype=float) for item in obstacles])
    anchor = np.mean(np.concatenate([starts, goals, obstacle_centers], axis=0), axis=0)

    def centered(value: np.ndarray) -> list[Any]:
        array = np.round(np.asarray(value, dtype=float) - anchor, 10)
        array[np.abs(array) < 1e-10] = 0.0
        return array.tolist()

    return {
        "starts": centered(starts),
        "goals": centered(goals),
        "obstacle_centers": centered(obstacle_centers),
        "obstacle_radii": [float(item.radius) for item in obstacles],
        "obstacle_safety_margins": [float(item.safety_margin) for item in obstacles],
    }


def layout_geometry_descriptor(layout: FrozenLayoutSpec) -> dict[str, Any]:
    starts, goals, obstacles = layout.geometry()
    pairs = pairwise_route_descriptors(starts, goals)
    direct_clearances = []
    for agent in range(NUM_AGENTS):
        direct_clearances.append(
            min(
                _point_to_segment_distance(obstacle.center, starts[agent], goals[agent])
                - float(obstacle.effective_radius)
                for obstacle in obstacles
            )
        )
    nearest_sync = min(
        pairs,
        key=lambda row: float(
            row["direct_path_predicted_minimum_inter_agent_distance_m"]
        ),
    )
    nearest_orthogonal = min(
        pairs, key=lambda row: abs(float(row["route_angle_3d_deg"]) - 90.0)
    )
    crossing_center = np.asarray(layout.crossing_center, dtype=float)
    obstacle_to_crossing = [
        float(np.linalg.norm(np.asarray(item.center, dtype=float) - crossing_center))
        for item in obstacles
    ]
    relative = relative_geometry_record(starts, goals, obstacles)
    return {
        "route_pair_descriptors": pairs,
        "minimum_initial_inter_agent_distance_m": _pairwise_minimum(starts),
        "minimum_terminal_inter_agent_distance_m": _pairwise_minimum(goals),
        "route_lengths_m": np.linalg.norm(goals - starts, axis=1).tolist(),
        "direct_path_static_surface_clearances_m": direct_clearances,
        "number_of_direct_obstacle_influence_paths": int(
            np.sum(np.asarray(direct_clearances) <= 1.5)
        ),
        "number_of_direct_obstacle_intersecting_paths": int(
            np.sum(np.asarray(direct_clearances) <= 0.0)
        ),
        "minimum_direct_path_obstacle_clearance_m": float(min(direct_clearances)),
        "obstacle_to_crossing_distances_m": obstacle_to_crossing,
        "minimum_obstacle_to_crossing_distance_m": float(min(obstacle_to_crossing)),
        "direct_path_predicted_minimum_inter_agent_distance_m": float(
            nearest_sync["direct_path_predicted_minimum_inter_agent_distance_m"]
        ),
        "predicted_closest_approach_pair": nearest_sync["pair"],
        "predicted_closest_approach_time_fraction": float(
            nearest_sync["predicted_closest_approach_time_fraction"]
        ),
        "predicted_closest_approach_time_difference_fraction": float(
            nearest_sync["predicted_closest_approach_time_difference_fraction"]
        ),
        "near_orthogonal_pair": nearest_orthogonal["pair"],
        "near_orthogonal_route_angle_3d_deg": float(
            nearest_orthogonal["route_angle_3d_deg"]
        ),
        "near_orthogonal_route_angle_xy_deg": float(
            nearest_orthogonal["route_angle_xy_deg"]
        ),
        "near_orthogonal_pair_predicted_minimum_distance_m": float(
            nearest_orthogonal[
                "direct_path_predicted_minimum_inter_agent_distance_m"
            ]
        ),
        "near_orthogonal_pair_time_difference_fraction": float(
            nearest_orthogonal[
                "predicted_closest_approach_time_difference_fraction"
            ]
        ),
        "near_orthogonal_pair_vertical_separation_m": float(
            nearest_orthogonal["predicted_closest_vertical_separation_m"]
        ),
        "third_agent_proximity_m": float(
            min(
                row["direct_path_predicted_minimum_inter_agent_distance_m"]
                for row in pairs
                if 2 in row["pair"]
            )
        ),
        "relative_geometry": relative,
        "relative_geometry_hash": stable_hash(relative),
    }


def _minimum_point_obstacle_clearance(
    points: np.ndarray, obstacles: Sequence[StaticSphereObstacle]
) -> float:
    return float(
        min(
            np.linalg.norm(np.asarray(point, dtype=float) - obstacle.center)
            - float(obstacle.effective_radius)
            for point in np.asarray(points, dtype=float)
            for obstacle in obstacles
        )
    )


def analytic_geometry_gate(
    layout: FrozenLayoutSpec,
    *,
    obstacle_influence_distance_m: float = 1.5,
    predicted_inter_agent_risk_distance_m: float = 0.6,
    near_orthogonal_tolerance_deg: float = 15.0,
    near_orthogonal_time_difference_max: float = 0.08,
) -> dict[str, Any]:
    """Evaluate only analytic geometry and initial-state constraints."""

    starts, goals, obstacles = layout.geometry()
    descriptor = layout_geometry_descriptor(layout)
    lower, upper = np.asarray(WORKSPACE_BOUNDS, dtype=float)
    start_clearance = _minimum_point_obstacle_clearance(starts, obstacles)
    goal_clearance = _minimum_point_obstacle_clearance(goals, obstacles)
    obstacle_centers = np.stack([np.asarray(item.center, dtype=float) for item in obstacles])
    finite_spheres = bool(
        np.all(np.isfinite(obstacle_centers))
        and all(float(item.effective_radius) > 0.0 for item in obstacles)
    )
    nearest_pair_distance = float(
        descriptor["direct_path_predicted_minimum_inter_agent_distance_m"]
    )
    nearest_pair_time = float(descriptor["predicted_closest_approach_time_fraction"])
    family_c_pressure = True
    if layout.family == "C":
        family_c_pressure = bool(
            abs(float(descriptor["near_orthogonal_route_angle_3d_deg"]) - 90.0)
            <= float(near_orthogonal_tolerance_deg)
            and float(descriptor["near_orthogonal_pair_predicted_minimum_distance_m"])
            <= float(predicted_inter_agent_risk_distance_m)
            and float(descriptor["near_orthogonal_pair_time_difference_fraction"])
            <= float(near_orthogonal_time_difference_max)
            and float(descriptor["near_orthogonal_pair_vertical_separation_m"])
            <= float(predicted_inter_agent_risk_distance_m)
        )
    checks = {
        "exactly_three_uavs": starts.shape == (NUM_AGENTS, 3),
        "exactly_two_static_obstacles": len(obstacles) == STATIC_OBSTACLE_COUNT,
        "zero_dynamic_obstacles": DYNAMIC_OBSTACLE_COUNT == 0,
        "peer_spheres_enabled": True,
        "boundary_free": True,
        "finite_geometry": bool(
            np.all(np.isfinite(starts))
            and np.all(np.isfinite(goals))
            and finite_spheres
        ),
        "starts_inside_validation_workspace": bool(
            np.all(starts >= lower) and np.all(starts <= upper)
        ),
        "goals_inside_validation_workspace": bool(
            np.all(goals >= lower) and np.all(goals <= upper)
        ),
        "minimum_initial_uav_distance": bool(
            descriptor["minimum_initial_inter_agent_distance_m"]
            > MIN_INITIAL_UAV_DISTANCE_M
        ),
        "minimum_terminal_uav_distance": bool(
            descriptor["minimum_terminal_inter_agent_distance_m"]
            >= MIN_TERMINAL_UAV_DISTANCE_M
        ),
        "minimum_route_length": bool(
            min(descriptor["route_lengths_m"]) >= MIN_START_GOAL_DISTANCE_M
        ),
        "starts_outside_obstacle_effective_regions": start_clearance > 0.0,
        "goals_outside_obstacle_effective_regions": goal_clearance > 0.0,
        "direct_path_enters_obstacle_influence_region": bool(
            min(descriptor["direct_path_static_surface_clearances_m"])
            <= float(obstacle_influence_distance_m)
        ),
        "meaningful_obstacle_pressure_relation": bool(
            descriptor["number_of_direct_obstacle_influence_paths"] >= 1
        ),
        "predicted_inter_agent_crossing_relation": bool(
            nearest_pair_distance <= float(predicted_inter_agent_risk_distance_m)
            and 0.05 <= nearest_pair_time <= 0.95
        ),
        "analytic_unbounded_free_space_exists": finite_spheres,
        "family_c_actual_3d_spatiotemporal_pressure": family_c_pressure,
    }
    return {
        "layout_id": layout.layout_id,
        "family": layout.family,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "minimum_start_obstacle_clearance_m": start_clearance,
        "minimum_goal_obstacle_clearance_m": goal_clearance,
        "solvability_statement": (
            "analytic_only: two finite spheres cannot seal boundary-free R3; "
            "no planner, policy, rollout, or method outcome was used"
        ),
        "method_outcome_used": False,
        "descriptor": descriptor,
    }


def build_layout_manifest_record(layout: FrozenLayoutSpec) -> dict[str, Any]:
    starts, goals, obstacles = layout.geometry()
    descriptor = layout_geometry_descriptor(layout)
    geometry_payload = {
        "layout_id": layout.layout_id,
        "family": layout.family,
        "starts": starts.tolist(),
        "terminal_goals": goals.tolist(),
        "obstacles": [
            {
                "center": np.asarray(item.center, dtype=float).tolist(),
                "radius_m": float(item.radius),
                "safety_margin_m": float(item.safety_margin),
                "effective_radius_m": float(item.effective_radius),
            }
            for item in obstacles
        ],
        "geometry_descriptors": descriptor,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "layout_id": layout.layout_id,
        "evaluation_seed": 1000 + FROZEN_HELD_OUT_LAYOUTS.index(layout),
        "family": layout.family,
        "family_description": layout.family_description,
        "structural_variant": layout.structural_variant,
        "layout_parameters": layout.parameter_record(),
        **geometry_payload,
        "layout_hash": stable_hash(geometry_payload),
        "relative_geometry_hash": descriptor["relative_geometry_hash"],
        "peer_sensing_mode": "peer_spheres",
        "peer_radius_m": PEER_RADIUS_M,
        "boundary_free": True,
        "dynamic_obstacle_count": DYNAMIC_OBSTACLE_COUNT,
        "layout_set_role": LAYOUT_SET_ROLE,
    }


def build_scenario_manifest() -> dict[str, Any]:
    layouts = [build_layout_manifest_record(item) for item in FROZEN_HELD_OUT_LAYOUTS]
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": SCENARIO_ID,
        "scenario_role": SCENARIO_ROLE,
        "layout_set_role": LAYOUT_SET_ROLE,
        "geometry_generation": "explicit_parameterized_deterministic_table",
        "geometry_frozen_before_method_evaluation": True,
        "outcome_based_rejection_forbidden": True,
        "layout_count": len(layouts),
        "families": sorted({item["family"] for item in layouts}),
        "layouts_per_family": {
            family: sum(item["family"] == family for item in layouts)
            for family in sorted({item["family"] for item in layouts})
        },
        "frozen_layout_table_sha256": FROZEN_LAYOUT_TABLE_SHA256,
        "layouts": layouts,
    }


def validate_manifest_geometry(
    manifest: Mapping[str, Any],
    *,
    legacy_relative_hashes: Iterable[str] = (),
) -> dict[str, Any]:
    layout_records = list(manifest["layouts"])
    relative_hashes = [str(item["relative_geometry_hash"]) for item in layout_records]
    layout_hashes = [str(item["layout_hash"]) for item in layout_records]
    legacy = set(str(value) for value in legacy_relative_hashes)
    gates = [analytic_geometry_gate(layout) for layout in FROZEN_HELD_OUT_LAYOUTS]
    family_c_gates = [item for item in gates if item["family"] == "C"]
    checks = {
        "exactly_24_layouts": len(layout_records) == 24,
        "families_A_to_F": sorted({item["family"] for item in layout_records})
        == list("ABCDEF"),
        "four_layouts_per_family": all(
            sum(item["family"] == family for item in layout_records) == 4
            for family in "ABCDEF"
        ),
        "all_layout_hashes_unique": len(set(layout_hashes)) == len(layout_hashes),
        "all_relative_geometry_hashes_unique": len(set(relative_hashes))
        == len(relative_hashes),
        "none_matches_legacy_relative_geometry": not legacy.intersection(relative_hashes),
        "all_geometry_gates_passed": all(item["status"] == "PASSED" for item in gates),
        "all_family_c_actual_3d_pressure_passed": all(
            item["checks"]["family_c_actual_3d_spatiotemporal_pressure"]
            for item in family_c_gates
        ),
        "method_outcome_not_used": all(not item["method_outcome_used"] for item in gates),
    }
    return {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "layout_count": len(layout_records),
        "relative_geometry_variant_count": len(set(relative_hashes)),
        "geometry_gates": gates,
    }


def build_environment_options(layout: FrozenLayoutSpec) -> dict[str, Any]:
    starts, goals, obstacles = layout.geometry()
    return {
        "starts": starts.copy(),
        "goals": goals.copy(),
        "static_obstacles": list(obstacles),
        "dynamic_obstacles": [],
    }


__all__ = [
    "DYNAMIC_OBSTACLE_COUNT",
    "FROZEN_HELD_OUT_LAYOUTS",
    "FROZEN_LAYOUT_TABLE_SHA256",
    "FrozenLayoutSpec",
    "LAYOUT_SET_ROLE",
    "NUM_AGENTS",
    "OBSTACLE_EFFECTIVE_RADIUS_M",
    "OBSTACLE_RADIUS_M",
    "OBSTACLE_SAFETY_MARGIN_M",
    "PEER_RADIUS_M",
    "SCENARIO_ID",
    "SCENARIO_ROLE",
    "STATIC_OBSTACLE_COUNT",
    "analytic_geometry_gate",
    "build_environment_options",
    "build_layout_manifest_record",
    "build_scenario_manifest",
    "layout_by_id",
    "layout_geometry_descriptor",
    "pairwise_route_descriptors",
    "relative_geometry_record",
    "stable_hash",
    "validate_manifest_geometry",
]
