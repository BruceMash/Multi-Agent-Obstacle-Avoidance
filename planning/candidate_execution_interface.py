"""Stable graph-stage interface for FP-SHEP candidate execution information.

This module does not alter preview propagation.  It converts an already
computed :class:`CandidatePreview` into an explicitly masked representation,
provides finite normalization, and exposes an independent constant-velocity
neighbor-conflict diagnostic for a future graph builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from planning.policy_preview import (
    BOUNDARY_CLEARANCE_SOURCE,
    CLEARANCE_SOURCE,
    PREVIEW_FEATURE_NAMES,
    PREVIEW_TERMINATION_COMPLETED,
    CandidatePreview,
)


EXECUTION_FEATURE_ORDER = PREVIEW_FEATURE_NAMES
NORMALIZATION_SPEC_VERSION = "fp_shep_h4_v1"


def _readonly_array(
    value: Any,
    *,
    name: str,
    dtype: Any = float,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GraphReadyCandidateExecution:
    """Candidate-level execution representation before tensor construction.

    ``preview_positions[0]`` and ``preview_velocities[0]`` correspond to the
    first predicted step, i.e. paper index ``h = 1``.  The initial state is not
    included in these graph-stage arrays.
    """

    candidate_id: int
    candidate_world_position: np.ndarray
    preview_positions: np.ndarray
    preview_velocities: np.ndarray
    task_progress_raw: float
    min_clearance_raw: float
    max_execution_deviation_raw: float
    terminal_speed_raw: float
    requested_horizon_steps: int
    effective_horizon_steps: int
    effective_horizon_ratio: float
    preview_completed: bool
    termination_reason: str
    feature_valid_mask: np.ndarray
    feature_full_horizon_mask: np.ndarray
    obstacle_clearance_source: str
    obstacle_clearance_is_approximate: bool
    boundary_clearance_source: str
    boundary_clearance_is_approximate: bool
    clearance_finite_mask: bool
    open_space_flag: bool

    def __post_init__(self) -> None:
        requested = int(self.requested_horizon_steps)
        effective = int(self.effective_horizon_steps)
        if requested <= 0:
            raise ValueError("requested_horizon_steps must be positive")
        if effective < 0 or effective > requested:
            raise ValueError("effective_horizon_steps must be in [0, requested]")
        positions = np.asarray(self.preview_positions, dtype=float)
        velocities = np.asarray(self.preview_velocities, dtype=float)
        candidate_position = np.asarray(self.candidate_world_position, dtype=float)
        if candidate_position.shape != (3,) or not np.all(np.isfinite(candidate_position)):
            raise ValueError("candidate_world_position must be finite with shape (3,)")
        expected_trajectory_shape = (effective, 3)
        if positions.shape != expected_trajectory_shape:
            raise ValueError(
                f"preview_positions must have shape {expected_trajectory_shape}"
            )
        if velocities.shape != expected_trajectory_shape:
            raise ValueError(
                f"preview_velocities must have shape {expected_trajectory_shape}"
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("preview trajectory arrays must be finite")
        expected_ratio = float(effective / requested)
        if not np.isclose(float(self.effective_horizon_ratio), expected_ratio):
            raise ValueError("effective_horizon_ratio is inconsistent with step counts")
        if bool(self.preview_completed) != (effective == requested):
            raise ValueError("preview_completed is inconsistent with step counts")
        if not str(self.termination_reason):
            raise ValueError("termination_reason must be non-empty")
        feature_shape = (len(EXECUTION_FEATURE_ORDER),)
        valid_mask = np.asarray(self.feature_valid_mask, dtype=bool)
        full_mask = np.asarray(self.feature_full_horizon_mask, dtype=bool)
        if valid_mask.shape != feature_shape or full_mask.shape != feature_shape:
            raise ValueError(f"feature masks must have shape {feature_shape}")
        if np.any(full_mask & ~valid_mask):
            raise ValueError("a full-horizon feature must also be valid")
        if bool(self.clearance_finite_mask) != bool(
            np.isfinite(float(self.min_clearance_raw))
        ):
            raise ValueError("clearance_finite_mask does not match min_clearance_raw")

        object.__setattr__(self, "candidate_id", int(self.candidate_id))
        object.__setattr__(
            self,
            "candidate_world_position",
            _readonly_array(
                candidate_position,
                name="candidate_world_position",
                shape=(3,),
            ),
        )
        object.__setattr__(
            self,
            "preview_positions",
            _readonly_array(positions, name="preview_positions"),
        )
        object.__setattr__(
            self,
            "preview_velocities",
            _readonly_array(velocities, name="preview_velocities"),
        )
        object.__setattr__(self, "requested_horizon_steps", requested)
        object.__setattr__(self, "effective_horizon_steps", effective)
        object.__setattr__(self, "effective_horizon_ratio", expected_ratio)
        object.__setattr__(
            self,
            "feature_valid_mask",
            _readonly_array(valid_mask, name="feature_valid_mask", dtype=bool),
        )
        object.__setattr__(
            self,
            "feature_full_horizon_mask",
            _readonly_array(full_mask, name="feature_full_horizon_mask", dtype=bool),
        )

    @property
    def raw_feature_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.task_progress_raw,
                self.min_clearance_raw,
                self.max_execution_deviation_raw,
                self.terminal_speed_raw,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class ExecutionNormalizationSpec:
    """Fixed, non-fitted normalization specification for graph-stage input."""

    task_progress_scale: float = 0.32
    min_clearance_scale: float = 4.5
    max_execution_deviation_scale: float = 1.05
    terminal_speed_scale: float = 1.2
    version: str = NORMALIZATION_SPEC_VERSION

    def __post_init__(self) -> None:
        for name in (
            "task_progress_scale",
            "min_clearance_scale",
            "max_execution_deviation_scale",
            "terminal_speed_scale",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if not str(self.version):
            raise ValueError("normalization version must be non-empty")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExecutionNormalizationSpec":
        return cls(
            task_progress_scale=float(values["task_progress_scale"]),
            min_clearance_scale=float(values["min_clearance_scale"]),
            max_execution_deviation_scale=float(
                values["max_execution_deviation_scale"]
            ),
            terminal_speed_scale=float(values["terminal_speed_scale"]),
            version=str(values.get("version", NORMALIZATION_SPEC_VERSION)),
        )


@dataclass(frozen=True)
class NormalizedExecutionFeatures:
    values: np.ndarray
    valid_mask: np.ndarray
    full_horizon_mask: np.ndarray
    clipped_mask: np.ndarray
    clearance_finite_mask: bool
    open_space_flag: bool
    feature_order: tuple[str, ...] = EXECUTION_FEATURE_ORDER
    spec_version: str = NORMALIZATION_SPEC_VERSION

    def __post_init__(self) -> None:
        shape = (len(self.feature_order),)
        values = np.asarray(self.values, dtype=float)
        valid = np.asarray(self.valid_mask, dtype=bool)
        full = np.asarray(self.full_horizon_mask, dtype=bool)
        clipped = np.asarray(self.clipped_mask, dtype=bool)
        if any(array.shape != shape for array in (values, valid, full, clipped)):
            raise ValueError(f"normalized feature arrays must have shape {shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("normalized feature values must be finite")
        object.__setattr__(self, "values", _readonly_array(values, name="values"))
        object.__setattr__(
            self, "valid_mask", _readonly_array(valid, name="valid_mask", dtype=bool)
        )
        object.__setattr__(
            self,
            "full_horizon_mask",
            _readonly_array(full, name="full_horizon_mask", dtype=bool),
        )
        object.__setattr__(
            self,
            "clipped_mask",
            _readonly_array(clipped, name="clipped_mask", dtype=bool),
        )


def _mask_from_metadata(
    metadata: Mapping[str, Any],
    key: str,
    defaults: Mapping[str, bool],
) -> np.ndarray:
    values = metadata.get(key, defaults)
    if not isinstance(values, Mapping):
        raise ValueError(f"{key} must be a mapping keyed by feature name")
    return np.asarray(
        [bool(values.get(name, False)) for name in EXECUTION_FEATURE_ORDER],
        dtype=bool,
    )


def graph_ready_candidate_execution(
    candidate_id: int,
    preview: CandidatePreview,
) -> GraphReadyCandidateExecution:
    """Convert an existing FP-SHEP result without changing any raw feature."""

    trajectory = preview.trajectory
    metadata = preview.metadata
    effective = int(metadata.get("effective_horizon_steps", trajectory.horizon))
    requested = int(metadata.get("requested_horizon_steps", effective))
    completed = bool(metadata.get("preview_completed", effective == requested))
    termination_reason = str(
        metadata.get(
            "termination_reason",
            PREVIEW_TERMINATION_COMPLETED if completed else "preview_truncated",
        )
    )
    raw_values = {
        "task_progress": float(preview.task_progress),
        "min_clearance": float(preview.min_clearance),
        "max_execution_deviation": float(preview.max_execution_deviation),
        "terminal_speed": float(preview.terminal_speed),
    }
    default_valid = {
        name: bool(np.isfinite(value)) for name, value in raw_values.items()
    }
    valid_mask = _mask_from_metadata(
        metadata, "feature_valid_mask", default_valid
    )
    default_full = {
        name: bool(default_valid[name] and completed) for name in EXECUTION_FEATURE_ORDER
    }
    full_mask = _mask_from_metadata(
        metadata, "feature_full_horizon_mask", default_full
    )
    if trajectory.positions.shape[0] < effective + 1:
        raise ValueError("preview trajectory is shorter than effective_horizon_steps")
    if trajectory.velocities.shape[0] < effective + 1:
        raise ValueError("preview velocities are shorter than effective_horizon_steps")

    return GraphReadyCandidateExecution(
        candidate_id=int(candidate_id),
        candidate_world_position=trajectory.candidate_goal,
        preview_positions=trajectory.positions[1 : effective + 1],
        preview_velocities=trajectory.velocities[1 : effective + 1],
        task_progress_raw=raw_values["task_progress"],
        min_clearance_raw=raw_values["min_clearance"],
        max_execution_deviation_raw=raw_values["max_execution_deviation"],
        terminal_speed_raw=raw_values["terminal_speed"],
        requested_horizon_steps=requested,
        effective_horizon_steps=effective,
        effective_horizon_ratio=float(effective / requested),
        preview_completed=completed,
        termination_reason=termination_reason,
        feature_valid_mask=valid_mask,
        feature_full_horizon_mask=full_mask,
        obstacle_clearance_source=str(
            metadata.get("obstacle_clearance_source", CLEARANCE_SOURCE)
        ),
        obstacle_clearance_is_approximate=bool(
            metadata.get("obstacle_clearance_is_approximate", True)
        ),
        boundary_clearance_source=str(
            metadata.get("boundary_clearance_source", BOUNDARY_CLEARANCE_SOURCE)
        ),
        boundary_clearance_is_approximate=bool(
            metadata.get("boundary_clearance_is_approximate", True)
        ),
        clearance_finite_mask=bool(
            metadata.get("clearance_finite_mask", np.isfinite(preview.min_clearance))
        ),
        open_space_flag=bool(metadata.get("open_space_flag", False)),
    )


def normalize_execution_features(
    execution: GraphReadyCandidateExecution,
    spec: ExecutionNormalizationSpec | None = None,
) -> NormalizedExecutionFeatures:
    """Return a finite companion vector; raw values remain untouched."""

    spec = spec or ExecutionNormalizationSpec()
    raw = execution.raw_feature_vector
    scales = np.asarray(
        [
            spec.task_progress_scale,
            spec.min_clearance_scale,
            spec.max_execution_deviation_scale,
            spec.terminal_speed_scale,
        ],
        dtype=float,
    )
    lower = np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=float)
    upper = np.ones(4, dtype=float)
    scaled = np.zeros(4, dtype=float)
    finite = np.isfinite(raw)
    scaled[finite] = raw[finite] / scales[finite]
    clearance_index = EXECUTION_FEATURE_ORDER.index("min_clearance")
    if not finite[clearance_index] and execution.open_space_flag:
        scaled[clearance_index] = 1.0
    clipped = np.logical_or(scaled < lower, scaled > upper)
    values = np.clip(scaled, lower, upper)
    invalid_numeric = ~np.isfinite(values)
    if np.any(invalid_numeric):
        values[invalid_numeric] = 0.0
    valid_mask = execution.feature_valid_mask & finite
    if execution.open_space_flag:
        valid_mask[clearance_index] = False
    return NormalizedExecutionFeatures(
        values=values,
        valid_mask=valid_mask,
        full_horizon_mask=execution.feature_full_horizon_mask & valid_mask,
        clipped_mask=clipped,
        clearance_finite_mask=execution.clearance_finite_mask,
        open_space_flag=execution.open_space_flag,
        spec_version=spec.version,
    )


@dataclass(frozen=True)
class ConstantVelocityConflictDiagnostic:
    per_step_distance: np.ndarray
    minimum_separation: float
    time_to_minimum_separation: float
    risk_duration: float
    risk_step_count: int
    risk_separation_threshold: float
    dt: float

    def __post_init__(self) -> None:
        distances = np.asarray(self.per_step_distance, dtype=float)
        if distances.ndim != 1 or distances.size == 0:
            raise ValueError("per_step_distance must be a non-empty vector")
        if not np.all(np.isfinite(distances)):
            raise ValueError("per_step_distance must be finite")
        object.__setattr__(
            self,
            "per_step_distance",
            _readonly_array(distances, name="per_step_distance"),
        )


def constant_velocity_conflict_diagnostic(
    *,
    candidate_preview_positions: np.ndarray,
    neighbor_current_position: np.ndarray,
    neighbor_current_velocity: np.ndarray,
    dt: float,
    risk_separation_threshold: float,
) -> ConstantVelocityConflictDiagnostic:
    """Compare one candidate trajectory with one constant-velocity neighbor.

    Input row zero represents prediction step ``h=1``.  Consequently the
    minimum-separation time is ``(argmin_index + 1) * dt``.
    """

    candidate_positions = np.asarray(candidate_preview_positions, dtype=float)
    if candidate_positions.ndim != 2 or candidate_positions.shape[1:] != (3,):
        raise ValueError("candidate_preview_positions must have shape (H, 3)")
    if candidate_positions.shape[0] == 0:
        raise ValueError("candidate_preview_positions must contain at least one step")
    if not np.all(np.isfinite(candidate_positions)):
        raise ValueError("candidate_preview_positions must be finite")
    neighbor_position = _readonly_array(
        neighbor_current_position,
        name="neighbor_current_position",
        shape=(3,),
    )
    neighbor_velocity = _readonly_array(
        neighbor_current_velocity,
        name="neighbor_current_velocity",
        shape=(3,),
    )
    dt = float(dt)
    threshold = float(risk_separation_threshold)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("risk_separation_threshold must be positive and finite")

    steps = np.arange(1, candidate_positions.shape[0] + 1, dtype=float)
    neighbor_positions = (
        neighbor_position[None, :] + steps[:, None] * dt * neighbor_velocity[None, :]
    )
    distances = np.linalg.norm(candidate_positions - neighbor_positions, axis=1)
    minimum_index = int(np.argmin(distances))
    # Paper risk duration uses the strict set {h | d_h < d_safe}.
    risk_step_count = int(np.count_nonzero(distances < threshold))
    return ConstantVelocityConflictDiagnostic(
        per_step_distance=distances,
        minimum_separation=float(distances[minimum_index]),
        time_to_minimum_separation=float((minimum_index + 1) * dt),
        risk_duration=float(risk_step_count * dt),
        risk_step_count=risk_step_count,
        risk_separation_threshold=threshold,
        dt=dt,
    )
