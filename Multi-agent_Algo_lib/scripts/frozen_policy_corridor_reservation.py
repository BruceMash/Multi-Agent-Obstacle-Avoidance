"""Geometry and typed schemas for frozen-policy corridor coordination.

This module deliberately contains no policy, DMP, GAT, or environment mutation
logic.  It provides the deterministic geometry contract shared by the minimal
E-corridor experiments and by a future learned upper-level planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import acos
from typing import Any, Mapping, Sequence

import numpy as np


ArrayLike3 = Sequence[float] | np.ndarray


class WaypointType(str, Enum):
    """Finite active-waypoint vocabulary allowed to the upper level."""

    PROGRESS = "PROGRESS"
    DECELERATION = "DECELERATION"
    HOLD = "HOLD"
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    TERMINAL = "TERMINAL"


class CorridorDirection(str, Enum):
    """Travel direction relative to the inferred A-to-B centerline."""

    A_TO_B = "A_TO_B"
    B_TO_A = "B_TO_A"
    BYPASS = "BYPASS"


class AgentUpperState(str, Enum):
    """Upper-level states specified by the corridor validation protocol."""

    CRUISE = "CRUISE"
    APPROACH = "APPROACH"
    REQUEST = "REQUEST"
    DECELERATE = "DECELERATE"
    HOLD = "HOLD"
    AUTHORIZED = "AUTHORIZED"
    ENTER = "ENTER"
    OCCUPY = "OCCUPY"
    EXIT = "EXIT"
    RELEASED = "RELEASED"
    FINISHED = "FINISHED"


class ReservationStatus(str, Enum):
    EMPTY = "EMPTY"
    REQUESTED = "REQUESTED"
    AUTHORIZED = "AUTHORIZED"
    OCCUPIED = "OCCUPIED"
    EXITING = "EXITING"
    RELEASED = "RELEASED"
    CANCELLED = "CANCELLED"


class FirstFailureType(str, Enum):
    INTER_AGENT_COLLISION = "INTER_AGENT_COLLISION"
    OBSTACLE_COLLISION = "OBSTACLE_COLLISION"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    TIMEOUT = "TIMEOUT"
    DEADLOCK = "DEADLOCK"
    HOLD_TRACKING_FAILURE = "HOLD_TRACKING_FAILURE"
    ENTRY_TRACKING_FAILURE = "ENTRY_TRACKING_FAILURE"
    PREMATURE_ENTRY = "PREMATURE_ENTRY"
    PREMATURE_RELEASE = "PREMATURE_RELEASE"
    OTHER = "OTHER"


def _vector3(value: ArrayLike3, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-dimensional vector")
    return array.copy()


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class AxisAlignedRegion:
    """Closed axis-aligned region used for conflict and hold zones."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = _vector3(self.lower, name="lower")
        upper = _vector3(self.upper, name="upper")
        if np.any(upper < lower):
            raise ValueError("region upper bounds must not be below lower bounds")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.lower + self.upper)

    @property
    def size(self) -> np.ndarray:
        return self.upper - self.lower

    def contains(self, point: ArrayLike3, margin: float = 0.0) -> bool:
        point_array = _vector3(point, name="point")
        margin_value = float(margin)
        return bool(
            np.all(point_array >= self.lower - margin_value)
            and np.all(point_array <= self.upper + margin_value)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"lower": self.lower.tolist(), "upper": self.upper.tolist()}


@dataclass(frozen=True)
class CorridorGeometryConfig:
    """Centralised inference and waypoint-layout parameters.

    ``from_mapping`` accepts either the complete experiment configuration or
    its ``corridor_geometry`` subsection.  Therefore callers never need to
    repeat geometric constants in controller code.
    """

    corridor_id: str = "E_head_on_narrow_corridor"
    inference_mode: str = "parallel_axis_aligned_boxes"
    wall_pair_tolerance: float = 1.0e-6
    minimum_wall_overlap: float = 1.0
    minimum_corridor_width: float = 0.0
    hold_offset_from_entry: float = 0.35
    hold_vertical_offset: float = 0.70
    hold_zone_vertical_half_extent: float = 0.15
    deceleration_offset_from_hold: float = 0.20
    entry_offset_inside_corridor: float = 0.35
    exit_offset_outside_corridor: float = 0.30
    progress_offset_from_hold: float = 0.10
    conflict_lateral_margin: float = 0.0
    release_margin: float = 0.20
    minimum_static_clearance: float = 0.30
    hold_zone_half_length: float = 0.12
    numerical_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        nonnegative = (
            "wall_pair_tolerance",
            "minimum_wall_overlap",
            "minimum_corridor_width",
            "hold_offset_from_entry",
            "hold_vertical_offset",
            "hold_zone_vertical_half_extent",
            "deceleration_offset_from_hold",
            "entry_offset_inside_corridor",
            "exit_offset_outside_corridor",
            "progress_offset_from_hold",
            "conflict_lateral_margin",
            "release_margin",
            "minimum_static_clearance",
            "hold_zone_half_length",
            "numerical_tolerance",
        )
        for name in nonnegative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.inference_mode != "parallel_axis_aligned_boxes":
            raise ValueError(
                "only inference_mode='parallel_axis_aligned_boxes' is supported"
            )

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any] | "CorridorGeometryConfig" | None
    ) -> "CorridorGeometryConfig":
        if mapping is None:
            return cls()
        if isinstance(mapping, cls):
            return mapping
        values: Mapping[str, Any] = mapping
        nested = mapping.get("corridor_geometry")
        if isinstance(nested, Mapping):
            values = nested
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        return {
            name: _jsonable(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class CorridorMetadata:
    """Geometry inferred from a pair of parallel axis-aligned corridor walls."""

    corridor_id: str
    centerline: np.ndarray
    entry_gate_a: np.ndarray
    entry_gate_b: np.ndarray
    exit_gate_a: np.ndarray
    exit_gate_b: np.ndarray
    conflict_region: AxisAlignedRegion
    hold_zone_a: AxisAlignedRegion
    hold_zone_b: AxisAlignedRegion
    nominal_direction_a: np.ndarray
    nominal_direction_b: np.ndarray
    longitudinal_axis: int
    lateral_axis: int
    vertical_axis: int
    corridor_width: float
    usable_length: float
    wall_obstacle_indices: tuple[int, int]
    release_margin: float = 0.0
    inference_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        centerline = np.asarray(self.centerline, dtype=float)
        if centerline.shape != (2, 3) or not np.all(np.isfinite(centerline)):
            raise ValueError("centerline must contain two finite 3-D endpoints")
        object.__setattr__(self, "centerline", centerline.copy())
        for name in (
            "entry_gate_a",
            "entry_gate_b",
            "exit_gate_a",
            "exit_gate_b",
            "nominal_direction_a",
            "nominal_direction_b",
        ):
            object.__setattr__(self, name, _vector3(getattr(self, name), name=name))
        axes = {self.longitudinal_axis, self.lateral_axis, self.vertical_axis}
        if axes != {0, 1, 2}:
            raise ValueError("corridor axes must be a permutation of (0, 1, 2)")
        if self.usable_length <= 0.0 or self.corridor_width <= 0.0:
            raise ValueError("corridor length and width must be positive")
        for name in ("nominal_direction_a", "nominal_direction_b"):
            norm = float(np.linalg.norm(getattr(self, name)))
            if not np.isclose(norm, 1.0):
                raise ValueError(f"{name} must be a unit vector")

    @property
    def centerline_point(self) -> np.ndarray:
        return self.centerline[0].copy()

    @property
    def centerline_direction(self) -> np.ndarray:
        direction = self.centerline[1] - self.centerline[0]
        return direction / np.linalg.norm(direction)

    @property
    def conflict_lower(self) -> np.ndarray:
        return self.conflict_region.lower.copy()

    @property
    def conflict_upper(self) -> np.ndarray:
        return self.conflict_region.upper.copy()

    def longitudinal_coordinate(self, point: ArrayLike3) -> float:
        """Return distance along A-to-B centerline, with gate A at zero."""

        point_array = _vector3(point, name="point")
        return float(np.dot(point_array - self.entry_gate_a, self.centerline_direction))

    def inside_conflict_region(self, point: ArrayLike3, margin: float = 0.0) -> bool:
        return self.conflict_region.contains(point, margin=margin)

    def direction_for_route(
        self, start: ArrayLike3, goal: ArrayLike3
    ) -> CorridorDirection:
        if not route_intersects_conflict_region(start, goal, self):
            return CorridorDirection.BYPASS
        start_s = self.longitudinal_coordinate(start)
        goal_s = self.longitudinal_coordinate(goal)
        if goal_s > start_s:
            return CorridorDirection.A_TO_B
        if goal_s < start_s:
            return CorridorDirection.B_TO_A
        return CorridorDirection.BYPASS

    def has_crossed_entry(
        self,
        point: ArrayLike3,
        direction: CorridorDirection | str,
        margin: float = 0.0,
    ) -> bool:
        direction_value = CorridorDirection(direction)
        coordinate = self.longitudinal_coordinate(point)
        if direction_value is CorridorDirection.A_TO_B:
            return coordinate >= float(margin)
        if direction_value is CorridorDirection.B_TO_A:
            return coordinate <= self.usable_length - float(margin)
        return False

    def has_crossed_exit(
        self,
        point: ArrayLike3,
        direction: CorridorDirection | str,
        margin: float = 0.0,
    ) -> bool:
        direction_value = CorridorDirection(direction)
        coordinate = self.longitudinal_coordinate(point)
        if direction_value is CorridorDirection.A_TO_B:
            return coordinate >= self.usable_length + float(margin)
        if direction_value is CorridorDirection.B_TO_A:
            return coordinate <= -float(margin)
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "corridor_id": self.corridor_id,
            "centerline": self.centerline.tolist(),
            "centerline_point": self.centerline_point.tolist(),
            "centerline_direction": self.centerline_direction.tolist(),
            "entry_gate_a": self.entry_gate_a.tolist(),
            "entry_gate_b": self.entry_gate_b.tolist(),
            "exit_gate_a": self.exit_gate_a.tolist(),
            "exit_gate_b": self.exit_gate_b.tolist(),
            "conflict_region": self.conflict_region.to_dict(),
            "conflict_lower": self.conflict_lower.tolist(),
            "conflict_upper": self.conflict_upper.tolist(),
            "hold_zone_a": self.hold_zone_a.to_dict(),
            "hold_zone_b": self.hold_zone_b.to_dict(),
            "nominal_direction_a": self.nominal_direction_a.tolist(),
            "nominal_direction_b": self.nominal_direction_b.tolist(),
            "longitudinal_axis": self.longitudinal_axis,
            "lateral_axis": self.lateral_axis,
            "vertical_axis": self.vertical_axis,
            "corridor_width": self.corridor_width,
            "usable_length": self.usable_length,
            "wall_obstacle_indices": list(self.wall_obstacle_indices),
            "release_margin": self.release_margin,
            "inference_parameters": _jsonable(self.inference_parameters),
        }


@dataclass
class CandidateWaypoint:
    """One member of the finite upper-level active-waypoint set."""

    position: np.ndarray
    waypoint_type: str
    corridor_id: str
    direction: str
    min_clearance: float
    max_expected_turn: float
    source: str
    rollout_feasible: bool | None = None
    waypoint_id: str = ""
    ordinal: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = _vector3(self.position, name="position")
        self.waypoint_type = WaypointType(self.waypoint_type).value
        self.direction = CorridorDirection(self.direction).value
        self.min_clearance = float(self.min_clearance)
        self.max_expected_turn = float(self.max_expected_turn)
        if not np.isfinite(self.min_clearance):
            self.min_clearance = float("inf")
        if self.max_expected_turn < 0.0 or not np.isfinite(self.max_expected_turn):
            raise ValueError("max_expected_turn must be finite and non-negative")

    @property
    def type(self) -> str:
        """Compatibility alias matching the experiment protocol wording."""

        return self.waypoint_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "waypoint_id": self.waypoint_id,
            "position": self.position.tolist(),
            "waypoint_type": self.waypoint_type,
            "type": self.waypoint_type,
            "corridor_id": self.corridor_id,
            "direction": self.direction,
            "min_clearance": self.min_clearance,
            "max_expected_turn": self.max_expected_turn,
            "source": self.source,
            "rollout_feasible": self.rollout_feasible,
            "ordinal": self.ordinal,
            "metadata": _jsonable(self.metadata),
        }


@dataclass
class AgentCoordinationState:
    agent_id: int
    state: AgentUpperState = AgentUpperState.CRUISE
    direction: CorridorDirection = CorridorDirection.BYPASS
    bypass: bool = False
    state_entered_step: int = 0
    request_step: int | None = None
    authorized_step: int | None = None
    current_waypoint_id: str | None = None
    stable_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


@dataclass
class CorridorReservation:
    """Single-owner reservation record for a capacity-one corridor."""

    corridor_id: str
    owner_agent_id: int | None = None
    request_time: float | None = None
    authorized_time: float | None = None
    predicted_entry_time: float | None = None
    predicted_exit_time: float | None = None
    actual_entry_time: float | None = None
    actual_exit_time: float | None = None
    release_time: float | None = None
    direction: CorridorDirection = CorridorDirection.BYPASS
    status: ReservationStatus = ReservationStatus.EMPTY

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


@dataclass
class CandidateRolloutResult:
    """Policy-conditioned counterfactual rollout schema (no rollout logic)."""

    candidate_waypoint: CandidateWaypoint
    predicted_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    predicted_velocities: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    predicted_accelerations: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    predicted_waypoint_distance: float = float("inf")
    predicted_terminal_distance: float = float("inf")
    predicted_min_obstacle_clearance: float = float("inf")
    predicted_min_peer_distance: float = float("inf")
    predicted_corridor_entry_step: int | None = None
    predicted_corridor_exit_step: int | None = None
    predicted_settle_step: int | None = None
    predicted_max_turn_angle: float = 0.0
    predicted_max_lateral_deviation: float = 0.0
    predicted_collision: bool = False
    predicted_out_of_bounds: bool = False
    predicted_feasible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


@dataclass
class WaypointSwitchRecord:
    agent_id: int
    step: int
    previous_waypoint: str | None
    new_waypoint: str
    switch_reason: str
    agent_state: AgentUpperState
    speed: float
    distance_to_previous: float | None
    distance_to_new: float
    dmp_phase: float | None
    reservation_owner: int | None
    predicted_entry_time: float | None = None
    predicted_exit_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


def _expanded_box_bounds(obstacle: Any) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(obstacle, "center") or not hasattr(obstacle, "half_extents"):
        raise TypeError("corridor inference requires axis-aligned box obstacles")
    center = _vector3(obstacle.center, name="obstacle.center")
    half_extents = _vector3(obstacle.half_extents, name="obstacle.half_extents")
    safety_margin = float(getattr(obstacle, "safety_margin", 0.0))
    expanded = half_extents + safety_margin
    if np.any(expanded <= 0.0):
        raise ValueError("obstacle expanded half extents must be positive")
    return center - expanded, center + expanded


def _workspace_region(workspace_bounds: Any | None) -> AxisAlignedRegion | None:
    if workspace_bounds is None:
        return None
    array = np.asarray(workspace_bounds, dtype=float)
    if array.shape != (2, 3):
        raise ValueError("workspace_bounds must have shape (2, 3)")
    return AxisAlignedRegion(array[0], array[1])


def infer_corridor_metadata(
    static_obstacles: Sequence[Any],
    workspace_bounds: Any | None = None,
    config: Mapping[str, Any] | CorridorGeometryConfig | None = None,
) -> CorridorMetadata:
    """Infer the narrow corridor from the best parallel AABB wall pair."""

    settings = CorridorGeometryConfig.from_mapping(config)
    if len(static_obstacles) < 2:
        raise ValueError("at least two static obstacles are required")
    bounds = [_expanded_box_bounds(obstacle) for obstacle in static_obstacles]
    best: tuple[float, int, int, int, int, int, float, float, float, float] | None = None

    for first in range(len(bounds)):
        for second in range(first + 1, len(bounds)):
            lower_first, upper_first = bounds[first]
            lower_second, upper_second = bounds[second]
            half_first = 0.5 * (upper_first - lower_first)
            half_second = 0.5 * (upper_second - lower_second)
            if not np.allclose(
                half_first,
                half_second,
                atol=settings.wall_pair_tolerance,
                rtol=0.0,
            ):
                continue
            separated_axes: list[tuple[float, int, int, int]] = []
            for axis in range(3):
                if upper_first[axis] < lower_second[axis]:
                    separated_axes.append(
                        (lower_second[axis] - upper_first[axis], axis, first, second)
                    )
                elif upper_second[axis] < lower_first[axis]:
                    separated_axes.append(
                        (lower_first[axis] - upper_second[axis], axis, second, first)
                    )
            for width, lateral_axis, low_wall, high_wall in separated_axes:
                remaining = [axis for axis in range(3) if axis != lateral_axis]
                overlaps = {
                    axis: min(upper_first[axis], upper_second[axis])
                    - max(lower_first[axis], lower_second[axis])
                    for axis in remaining
                }
                longitudinal_axis = max(overlaps, key=overlaps.get)
                vertical_axis = next(axis for axis in remaining if axis != longitudinal_axis)
                length = overlaps[longitudinal_axis]
                vertical_overlap = overlaps[vertical_axis]
                if (
                    length < settings.minimum_wall_overlap
                    or vertical_overlap <= settings.wall_pair_tolerance
                    or width <= max(
                        settings.minimum_corridor_width,
                        settings.wall_pair_tolerance,
                    )
                ):
                    continue
                score = float(length * width)
                candidate = (
                    score,
                    low_wall,
                    high_wall,
                    longitudinal_axis,
                    lateral_axis,
                    vertical_axis,
                    float(length),
                    float(width),
                    float(vertical_overlap),
                    float(max(lower_first[longitudinal_axis], lower_second[longitudinal_axis])),
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate

    if best is None:
        raise ValueError("no parallel axis-aligned corridor wall pair was found")

    (
        _,
        low_wall_index,
        high_wall_index,
        longitudinal_axis,
        lateral_axis,
        vertical_axis,
        _,
        _,
        _,
        longitudinal_low,
    ) = best
    low_lower, low_upper = bounds[low_wall_index]
    high_lower, high_upper = bounds[high_wall_index]
    longitudinal_high = min(
        low_upper[longitudinal_axis], high_upper[longitudinal_axis]
    )
    lateral_low = low_upper[lateral_axis] - settings.conflict_lateral_margin
    lateral_high = high_lower[lateral_axis] + settings.conflict_lateral_margin
    vertical_low = max(low_lower[vertical_axis], high_lower[vertical_axis])
    vertical_high = min(low_upper[vertical_axis], high_upper[vertical_axis])

    conflict_lower = np.zeros(3, dtype=float)
    conflict_upper = np.zeros(3, dtype=float)
    conflict_lower[longitudinal_axis] = longitudinal_low
    conflict_upper[longitudinal_axis] = longitudinal_high
    conflict_lower[lateral_axis] = lateral_low
    conflict_upper[lateral_axis] = lateral_high
    conflict_lower[vertical_axis] = vertical_low
    conflict_upper[vertical_axis] = vertical_high
    workspace = _workspace_region(workspace_bounds)
    if workspace is not None:
        conflict_lower = np.maximum(conflict_lower, workspace.lower)
        conflict_upper = np.minimum(conflict_upper, workspace.upper)
    conflict_region = AxisAlignedRegion(conflict_lower, conflict_upper)

    gate_a = conflict_region.center
    gate_b = conflict_region.center
    gate_a[longitudinal_axis] = conflict_lower[longitudinal_axis]
    gate_b[longitudinal_axis] = conflict_upper[longitudinal_axis]
    direction_a = np.zeros(3, dtype=float)
    direction_a[longitudinal_axis] = 1.0
    direction_b = -direction_a

    safe_lower = conflict_lower.copy()
    safe_upper = conflict_upper.copy()
    for axis in (lateral_axis, vertical_axis):
        requested = settings.minimum_static_clearance
        if safe_upper[axis] - safe_lower[axis] >= 2.0 * requested:
            safe_lower[axis] += requested
            safe_upper[axis] -= requested

    def hold_zone(entry: np.ndarray, direction: np.ndarray) -> AxisAlignedRegion:
        center = entry - direction * settings.hold_offset_from_entry
        center[vertical_axis] += (
            float(np.sign(direction[longitudinal_axis]))
            * settings.hold_vertical_offset
        )
        lower = safe_lower.copy()
        upper = safe_upper.copy()
        lower[longitudinal_axis] = (
            center[longitudinal_axis] - settings.hold_zone_half_length
        )
        upper[longitudinal_axis] = (
            center[longitudinal_axis] + settings.hold_zone_half_length
        )
        lower[vertical_axis] = (
            center[vertical_axis] - settings.hold_zone_vertical_half_extent
        )
        upper[vertical_axis] = (
            center[vertical_axis] + settings.hold_zone_vertical_half_extent
        )
        if workspace is not None:
            lower = np.maximum(lower, workspace.lower)
            upper = np.minimum(upper, workspace.upper)
        return AxisAlignedRegion(lower, upper)

    centerline = np.stack([gate_a, gate_b], axis=0)
    return CorridorMetadata(
        corridor_id=settings.corridor_id,
        centerline=centerline,
        entry_gate_a=gate_a,
        entry_gate_b=gate_b,
        exit_gate_a=gate_b,
        exit_gate_b=gate_a,
        conflict_region=conflict_region,
        hold_zone_a=hold_zone(gate_a, direction_a),
        hold_zone_b=hold_zone(gate_b, direction_b),
        nominal_direction_a=direction_a,
        nominal_direction_b=direction_b,
        longitudinal_axis=longitudinal_axis,
        lateral_axis=lateral_axis,
        vertical_axis=vertical_axis,
        corridor_width=float(lateral_high - lateral_low),
        usable_length=float(longitudinal_high - longitudinal_low),
        wall_obstacle_indices=(low_wall_index, high_wall_index),
        release_margin=settings.release_margin,
        inference_parameters=settings.to_dict(),
    )


def inside_conflict_region(
    point: ArrayLike3, metadata: CorridorMetadata, margin: float = 0.0
) -> bool:
    return metadata.inside_conflict_region(point, margin=margin)


def route_intersects_conflict_region(
    start: ArrayLike3,
    goal: ArrayLike3,
    metadata: CorridorMetadata,
    margin: float = 0.0,
) -> bool:
    """Return whether the closed start-goal segment intersects the conflict AABB."""

    start_array = _vector3(start, name="start")
    goal_array = _vector3(goal, name="goal")
    lower = metadata.conflict_lower - float(margin)
    upper = metadata.conflict_upper + float(margin)
    delta = goal_array - start_array
    t_low, t_high = 0.0, 1.0
    tolerance = float(metadata.inference_parameters.get("numerical_tolerance", 1.0e-9))
    for axis in range(3):
        if abs(delta[axis]) <= tolerance:
            if start_array[axis] < lower[axis] or start_array[axis] > upper[axis]:
                return False
            continue
        first = (lower[axis] - start_array[axis]) / delta[axis]
        second = (upper[axis] - start_array[axis]) / delta[axis]
        near, far = min(first, second), max(first, second)
        t_low = max(t_low, near)
        t_high = min(t_high, far)
        if t_low > t_high:
            return False
    return True


def longitudinal_coordinate(point: ArrayLike3, metadata: CorridorMetadata) -> float:
    return metadata.longitudinal_coordinate(point)


def direction_for_route(
    start: ArrayLike3, goal: ArrayLike3, metadata: CorridorMetadata
) -> CorridorDirection:
    return metadata.direction_for_route(start, goal)


def has_crossed_entry(
    point: ArrayLike3,
    direction: CorridorDirection | str,
    metadata: CorridorMetadata,
    margin: float = 0.0,
) -> bool:
    return metadata.has_crossed_entry(point, direction, margin=margin)


def has_crossed_exit(
    point: ArrayLike3,
    direction: CorridorDirection | str,
    metadata: CorridorMetadata,
    margin: float = 0.0,
) -> bool:
    return metadata.has_crossed_exit(point, direction, margin=margin)


def minimum_point_clearance(
    point: ArrayLike3,
    static_obstacles: Sequence[Any],
    workspace_bounds: Any | None = None,
) -> float:
    """Minimum center-to-surface clearance to obstacles and workspace boundary."""

    point_array = _vector3(point, name="point")
    clearances: list[float] = []
    for obstacle in static_obstacles:
        if hasattr(obstacle, "signed_distance"):
            clearances.append(float(obstacle.signed_distance(point_array)))
            continue
        lower, upper = _expanded_box_bounds(obstacle)
        q = np.maximum(np.maximum(lower - point_array, point_array - upper), 0.0)
        outside = float(np.linalg.norm(q))
        if np.all(point_array >= lower) and np.all(point_array <= upper):
            inside = -float(np.min(np.minimum(point_array - lower, upper - point_array)))
            clearances.append(inside)
        else:
            clearances.append(outside)
    workspace = _workspace_region(workspace_bounds)
    if workspace is not None:
        boundary = np.concatenate(
            [point_array - workspace.lower, workspace.upper - point_array]
        )
        clearances.append(float(np.min(boundary)))
    return min(clearances, default=float("inf"))


def point_has_clearance(
    point: ArrayLike3,
    static_obstacles: Sequence[Any],
    required_clearance: float,
    workspace_bounds: Any | None = None,
) -> bool:
    return minimum_point_clearance(
        point, static_obstacles, workspace_bounds
    ) >= float(required_clearance)


def _turn_angle(previous: np.ndarray, current: np.ndarray, following: np.ndarray) -> float:
    incoming = current - previous
    outgoing = following - current
    incoming_norm = float(np.linalg.norm(incoming))
    outgoing_norm = float(np.linalg.norm(outgoing))
    if incoming_norm <= 1.0e-12 or outgoing_norm <= 1.0e-12:
        return 0.0
    cosine = float(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm))
    return float(acos(float(np.clip(cosine, -1.0, 1.0))))


def build_route_candidates(
    metadata: CorridorMetadata,
    start: ArrayLike3,
    goal: ArrayLike3,
    static_obstacles: Sequence[Any],
    geometry_config: Mapping[str, Any] | CorridorGeometryConfig | None = None,
    workspace_bounds: Any | None = None,
) -> list[CandidateWaypoint]:
    """Build the explicit finite candidate sequence for one route.

    Corridor routes always use the order ``PROGRESS -> DECELERATION -> HOLD ->
    ENTRY -> EXIT -> TERMINAL``.  Routes not intersecting the conflict region
    are explicitly marked ``BYPASS`` and receive only ``PROGRESS -> TERMINAL``.
    """

    settings = CorridorGeometryConfig.from_mapping(
        geometry_config or metadata.inference_parameters
    )
    start_array = _vector3(start, name="start")
    goal_array = _vector3(goal, name="goal")
    direction = metadata.direction_for_route(start_array, goal_array)

    if direction is CorridorDirection.BYPASS:
        positions = [0.5 * (start_array + goal_array), goal_array]
        types = [WaypointType.PROGRESS, WaypointType.TERMINAL]
    else:
        if direction is CorridorDirection.A_TO_B:
            entry_coordinate = 0.0
            sign = 1.0
        else:
            entry_coordinate = metadata.usable_length
            sign = -1.0

        hold_coordinate = entry_coordinate - sign * settings.hold_offset_from_entry
        deceleration_coordinate = (
            hold_coordinate - sign * settings.deceleration_offset_from_hold
        )
        progress_coordinate = (
            deceleration_coordinate - sign * settings.progress_offset_from_hold
        )
        entry_inside_coordinate = (
            entry_coordinate + sign * settings.entry_offset_inside_corridor
        )
        exit_coordinate = (
            metadata.usable_length + settings.exit_offset_outside_corridor
            if direction is CorridorDirection.A_TO_B
            else -settings.exit_offset_outside_corridor
        )

        route_cross_section = 0.5 * (start_array + goal_array)
        safe_cross_section = route_cross_section.copy()
        clearance = settings.minimum_static_clearance + settings.numerical_tolerance
        for axis in (metadata.lateral_axis, metadata.vertical_axis):
            lower = metadata.conflict_lower[axis] + clearance
            upper = metadata.conflict_upper[axis] - clearance
            if lower <= upper:
                safe_cross_section[axis] = np.clip(
                    safe_cross_section[axis], lower, upper
                )
            else:
                safe_cross_section[axis] = metadata.conflict_region.center[axis]

        hold_cross_section = safe_cross_section.copy()
        hold_cross_section[metadata.vertical_axis] += (
            sign * settings.hold_vertical_offset
        )

        def centerline_position(
            coordinate: float,
            cross_section: np.ndarray = safe_cross_section,
        ) -> np.ndarray:
            point = cross_section.copy()
            point[metadata.longitudinal_axis] = (
                metadata.entry_gate_a[metadata.longitudinal_axis]
                + coordinate * metadata.centerline_direction[metadata.longitudinal_axis]
            )
            return point

        positions = [
            centerline_position(progress_coordinate, hold_cross_section),
            centerline_position(deceleration_coordinate, hold_cross_section),
            centerline_position(hold_coordinate, hold_cross_section),
            centerline_position(entry_inside_coordinate, hold_cross_section),
            centerline_position(exit_coordinate),
            goal_array,
        ]
        types = [
            WaypointType.PROGRESS,
            WaypointType.DECELERATION,
            WaypointType.HOLD,
            WaypointType.ENTRY,
            WaypointType.EXIT,
            WaypointType.TERMINAL,
        ]

    turns = [0.0] * len(positions)
    for index in range(1, len(positions) - 1):
        turns[index] = _turn_angle(
            np.asarray(positions[index - 1]),
            np.asarray(positions[index]),
            np.asarray(positions[index + 1]),
        )

    candidates: list[CandidateWaypoint] = []
    for ordinal, (waypoint_type, position) in enumerate(zip(types, positions)):
        min_clearance = minimum_point_clearance(
            position, static_obstacles, workspace_bounds
        )
        feasible = min_clearance + settings.numerical_tolerance >= (
            settings.minimum_static_clearance
        )
        candidate = CandidateWaypoint(
            waypoint_id=(
                f"{metadata.corridor_id}:{direction.value}:"
                f"{ordinal}:{waypoint_type.value}"
            ),
            position=np.asarray(position, dtype=float),
            waypoint_type=waypoint_type.value,
            corridor_id=metadata.corridor_id,
            direction=direction.value,
            min_clearance=min_clearance,
            max_expected_turn=turns[ordinal],
            source="inferred_parallel_aabb_corridor",
            rollout_feasible=None,
            ordinal=ordinal,
            metadata={
                "bypass": direction is CorridorDirection.BYPASS,
                "geometrically_feasible": bool(feasible),
                "minimum_required_clearance": settings.minimum_static_clearance,
            },
        )
        candidates.append(candidate)
    return candidates


__all__ = [
    "AgentCoordinationState",
    "AgentUpperState",
    "AxisAlignedRegion",
    "CandidateRolloutResult",
    "CandidateWaypoint",
    "CorridorDirection",
    "CorridorGeometryConfig",
    "CorridorMetadata",
    "CorridorReservation",
    "FirstFailureType",
    "ReservationStatus",
    "WaypointSwitchRecord",
    "WaypointType",
    "build_route_candidates",
    "direction_for_route",
    "has_crossed_entry",
    "has_crossed_exit",
    "infer_corridor_metadata",
    "inside_conflict_region",
    "longitudinal_coordinate",
    "minimum_point_clearance",
    "point_has_clearance",
    "route_intersects_conflict_region",
]
