"""Pure protocol primitives for the Pre-GAT closed-loop baseline.

This module does not train or invoke a GAT.  It keeps online candidate
selection limited to the existing Proposal generator and operational H=4
FP-SHEP.  Terminal task goals remain owned by the environment; selected
references are returned to the evaluation runner for assignment to ``dmp.goal``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from Guidance.reference_point_proposal_demo import ProposalConfig, propose_reference_points
from planning.candidate_execution_interface import (
    ExecutionNormalizationSpec,
    graph_ready_candidate_execution,
    normalize_execution_features,
)
from planning.policy_preview import (
    CandidatePreview,
    adapt_candidate_proposals,
    build_preview_inputs_from_env,
    preview_candidate,
    preview_candidates_batched,
)
from pre_gat_closed_loop_constants import (
    CLOSED_LOOP_SCHEMA_VERSION,
    FORMAL_PREVIEW_HORIZON,
    METHOD_DISPLAY_NAMES,
    METHOD_FP_SHEP,
    METHOD_FROZEN,
    METHOD_ORDER,
    METHOD_PROPOSAL,
    SELECTION_BASELINE_TERMINAL,
    SELECTION_FP_SHEP,
    SELECTION_NO_CANDIDATE_FALLBACK,
    SELECTION_PROPOSAL,
)


def _vector3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return result.copy()


@dataclass(frozen=True)
class FPSHEPOnlineScoreSpec:
    """Explicit deployment-only score applied to four recorded features.

    Terminal speed remains a required FP-SHEP output and is always logged, but
    its weight is fixed to zero for this baseline.  This score is not declared
    the unique or final FP-SHEP definition.
    """

    horizon: int = FORMAL_PREVIEW_HORIZON
    normalization: ExecutionNormalizationSpec = field(
        default_factory=ExecutionNormalizationSpec
    )
    progress_weight: float = 1.0
    clearance_weight: float = 1.0
    deviation_weight: float = 1.0
    terminal_speed_weight: float = 0.0
    name: str = "fixed_physical_scale_three_feature_online_score"
    definition_status: str = "goal_specific_baseline_not_global_fp_shep_definition"

    def __post_init__(self) -> None:
        if int(self.horizon) <= 0:
            raise ValueError("online FP-SHEP selector horizon must be positive")
        for name in ("progress_weight", "clearance_weight", "deviation_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if float(self.terminal_speed_weight) != 0.0:
            raise ValueError("terminal speed is recorded but excluded from online ranking")
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(self, "terminal_speed_weight", 0.0)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "FPSHEPOnlineScoreSpec":
        values = values or {}
        normalization_values = values.get("normalization", {})
        normalization = (
            normalization_values
            if isinstance(normalization_values, ExecutionNormalizationSpec)
            else ExecutionNormalizationSpec.from_mapping(normalization_values)
            if normalization_values
            else ExecutionNormalizationSpec()
        )
        weights = values.get("weights", values)
        return cls(
            horizon=int(values.get("H_preview", values.get("horizon", FORMAL_PREVIEW_HORIZON))),
            normalization=normalization,
            progress_weight=float(weights.get("progress", weights.get("progress_weight", 1.0))),
            clearance_weight=float(weights.get("clearance", weights.get("clearance_weight", 1.0))),
            deviation_weight=float(weights.get("deviation", weights.get("deviation_weight", 1.0))),
            terminal_speed_weight=float(
                weights.get("terminal_speed", weights.get("terminal_speed_weight", 0.0))
            ),
            name=str(values.get("name", cls.name)),
            definition_status=str(values.get("definition_status", cls.definition_status)),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "H_preview": int(self.horizon),
            "formula": "+progress +clearance -deviation",
            "feature_order": [
                "task_progress",
                "min_clearance",
                "max_execution_deviation",
                "terminal_speed",
            ],
            "weights": {
                "task_progress": self.progress_weight,
                "min_clearance": self.clearance_weight,
                "max_execution_deviation": -self.deviation_weight,
                "terminal_speed": self.terminal_speed_weight,
            },
            "used_for_online_ranking": {
                "task_progress": True,
                "min_clearance": True,
                "max_execution_deviation": True,
                "terminal_speed": False,
            },
            "all_four_preview_features_recorded": True,
            "normalization": vars(self.normalization).copy(),
            "normalization_is_candidate_set_fitted": False,
            "definition_status": self.definition_status,
            "uses_supervision_target": False,
            "uses_real_rollout": False,
            "uses_diagnostic_preview": False,
        }


@dataclass(frozen=True)
class FixedPeriodProtocol:
    m_upper: int
    consumer_top_k: int = 10
    goal_switch_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if int(self.m_upper) <= 0:
            raise ValueError("M_upper must be positive")
        if int(self.consumer_top_k) <= 0:
            raise ValueError("consumer_top_k must be positive")
        if float(self.goal_switch_epsilon) < 0.0:
            raise ValueError("goal_switch_epsilon must be non-negative")
        object.__setattr__(self, "m_upper", int(self.m_upper))
        object.__setattr__(self, "consumer_top_k", int(self.consumer_top_k))
        object.__setattr__(self, "goal_switch_epsilon", float(self.goal_switch_epsilon))

    def is_replanning_step(self, timestep: int) -> bool:
        timestep = int(timestep)
        if timestep < 0:
            raise ValueError("timestep must be non-negative")
        return timestep % self.m_upper == 0

    def metadata(self) -> dict[str, Any]:
        return {
            "M_upper": self.m_upper,
            "replanning_rule": "timestep % M_upper == 0",
            "waypoint_reached_triggers_replan": False,
            "stagnation_triggers_replan": False,
            "candidate_invalidation_triggers_replan": False,
            "termination_interrupts_hold": True,
            "consumer_top_k": self.consumer_top_k,
        }


@dataclass(frozen=True)
class PreviewScoreRecord:
    candidate_id: int
    candidate_world_position: np.ndarray
    score: float
    preview_task_progress: float
    preview_min_clearance: float
    preview_max_execution_deviation: float
    preview_terminal_speed: float
    normalized_features: np.ndarray
    valid_mask: np.ndarray
    clipped_mask: np.ndarray
    runtime_ms: float
    preview: CandidatePreview

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_world_position",
            _vector3(self.candidate_world_position, "candidate_world_position"),
        )
        for name in ("normalized_features", "valid_mask", "clipped_mask"):
            dtype = bool if name.endswith("mask") else float
            value = np.asarray(getattr(self, name), dtype=dtype)
            if value.shape != (4,):
                raise ValueError(f"{name} must have shape (4,)")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.isfinite(float(self.score)):
            raise ValueError("online FP-SHEP score must be finite")

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": int(self.candidate_id),
            "candidate_world_position": self.candidate_world_position.tolist(),
            "fp_shep_online_score": float(self.score),
            "preview_task_progress": float(self.preview_task_progress),
            "preview_min_clearance": float(self.preview_min_clearance),
            "preview_max_execution_deviation": float(
                self.preview_max_execution_deviation
            ),
            "preview_terminal_speed": float(self.preview_terminal_speed),
            "terminal_speed_used_for_online_ranking": False,
            "normalized_preview_features": self.normalized_features.tolist(),
            "preview_feature_valid_mask": self.valid_mask.tolist(),
            "preview_feature_clipped_mask": self.clipped_mask.tolist(),
            "preview_runtime_ms": float(self.runtime_ms),
        }


@dataclass(frozen=True)
class SelectionDecision:
    method: str
    execution_reference: np.ndarray
    selection_kind: str
    proposals: tuple[Any, ...]
    selected_candidate_id: int | None
    selected_candidate_original_index: int | None
    fp_shep_records: tuple[PreviewScoreRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.method not in METHOD_ORDER:
            raise ValueError(f"unknown method: {self.method}")
        object.__setattr__(
            self,
            "execution_reference",
            _vector3(self.execution_reference, "execution_reference"),
        )
        if self.selection_kind == SELECTION_NO_CANDIDATE_FALLBACK:
            if self.selected_candidate_id is not None:
                raise ValueError("no-candidate fallback must not count as a selection")
        if self.selected_candidate_id is not None:
            if not 0 <= int(self.selected_candidate_id) < len(self.proposals):
                raise IndexError("selected_candidate_id is outside proposal list")

    @property
    def no_candidate_fallback(self) -> bool:
        return self.selection_kind == SELECTION_NO_CANDIDATE_FALLBACK


def generate_candidate_set(
    env: Any,
    agent_index: int,
    proposal_config: ProposalConfig,
    *,
    consumer_top_k: int,
) -> tuple[tuple[Any, ...], int]:
    """Generate once and apply the established explicit consumer truncation."""

    agent_index = int(agent_index)
    packet = env.latest_sensor_packets[agent_index]
    if packet is None:
        raise RuntimeError("environment must be reset before candidate generation")
    all_proposals = propose_reference_points(
        env.dynamics[agent_index].p,
        env.goals[agent_index],
        env.dynamics[agent_index].v,
        packet,
        env.sensors[agent_index],
        proposal_config,
        float(env.env_config.goal_tolerance),
    )
    selected = adapt_candidate_proposals(
        all_proposals,
        consumer_top_k=int(consumer_top_k),
    )
    return tuple(selected), len(all_proposals)


def score_fp_shep_candidates(
    *,
    env: Any,
    agent_index: int,
    proposals: Sequence[Any],
    policy: Any,
    spec: FPSHEPOnlineScoreSpec | None = None,
    observation_extension: dict[str, Any] | None = None,
) -> tuple[PreviewScoreRecord, ...]:
    """Evaluate one proposal-only set with the explicitly configured preview."""

    spec = spec or FPSHEPOnlineScoreSpec()
    initial_state, local_context = build_preview_inputs_from_env(env, int(agent_index))
    records: list[PreviewScoreRecord] = []
    for candidate_id, proposal in enumerate(proposals):
        preview = preview_candidate(
            initial_state=initial_state,
            local_context=local_context,
            candidate_goal=np.asarray(proposal.point, dtype=float),
            policy=policy,
            horizon=int(spec.horizon),
            dmp_config=env.dmps[int(agent_index)].config,
            dynamics=env.dynamics[int(agent_index)],
            observation_extension=observation_extension,
        )
        execution = graph_ready_candidate_execution(candidate_id, preview)
        normalized = normalize_execution_features(execution, spec.normalization)
        values = normalized.values
        score = (
            spec.progress_weight * values[0]
            + spec.clearance_weight * values[1]
            - spec.deviation_weight * values[2]
            - spec.terminal_speed_weight * values[3]
        )
        records.append(
            PreviewScoreRecord(
                candidate_id=candidate_id,
                candidate_world_position=proposal.point,
                score=float(score),
                preview_task_progress=float(preview.task_progress),
                preview_min_clearance=float(preview.min_clearance),
                preview_max_execution_deviation=float(preview.max_execution_deviation),
                preview_terminal_speed=float(preview.terminal_speed),
                normalized_features=normalized.values,
                valid_mask=normalized.valid_mask,
                clipped_mask=normalized.clipped_mask,
                runtime_ms=float(preview.performance.total_ms),
                preview=preview,
            )
        )
    return tuple(records)


def score_fp_shep_candidates_batched(
    *,
    env: Any,
    agent_index: int,
    proposals: Sequence[Any],
    policy: Any,
    spec: FPSHEPOnlineScoreSpec | None = None,
    timing_sink: dict[str, float] | None = None,
    observation_extension: dict[str, Any] | None = None,
) -> tuple[PreviewScoreRecord, ...]:
    """Evaluate the configured score with candidate-batched actor calls."""

    import time

    spec = spec or FPSHEPOnlineScoreSpec()
    prepare_started = time.perf_counter_ns()
    initial_state, local_context = build_preview_inputs_from_env(env, int(agent_index))
    state_prepare_ms = (time.perf_counter_ns() - prepare_started) / 1.0e6
    preview_timing: dict[str, float] = {}
    previews = preview_candidates_batched(
        initial_state=initial_state,
        local_context=local_context,
        candidates=proposals,
        policy=policy,
        horizon=int(spec.horizon),
        dmp_config=env.dmps[int(agent_index)].config,
        dynamics=env.dynamics[int(agent_index)],
        timing_sink=preview_timing,
        observation_extension=observation_extension,
    )
    metric_started = time.perf_counter_ns()
    records: list[PreviewScoreRecord] = []
    for candidate_id, (proposal, preview) in enumerate(
        zip(proposals, previews, strict=True)
    ):
        execution = graph_ready_candidate_execution(candidate_id, preview)
        normalized = normalize_execution_features(execution, spec.normalization)
        values = normalized.values
        score = (
            spec.progress_weight * values[0]
            + spec.clearance_weight * values[1]
            - spec.deviation_weight * values[2]
            - spec.terminal_speed_weight * values[3]
        )
        records.append(
            PreviewScoreRecord(
                candidate_id=candidate_id,
                candidate_world_position=proposal.point,
                score=float(score),
                preview_task_progress=float(preview.task_progress),
                preview_min_clearance=float(preview.min_clearance),
                preview_max_execution_deviation=float(
                    preview.max_execution_deviation
                ),
                preview_terminal_speed=float(preview.terminal_speed),
                normalized_features=normalized.values,
                valid_mask=normalized.valid_mask,
                clipped_mask=normalized.clipped_mask,
                runtime_ms=float(preview.performance.total_ms),
                preview=preview,
            )
        )
    score_metric_ms = (time.perf_counter_ns() - metric_started) / 1.0e6
    if timing_sink is not None:
        timing_sink.update(
            {
                "fp_shep_state_prepare_ms": float(
                    state_prepare_ms + preview_timing.get("observation_ms", 0.0)
                ),
                "fp_shep_actor_forward_ms": float(
                    preview_timing.get("actor_forward_ms", 0.0)
                ),
                "fp_shep_dmp_rollout_ms": float(
                    preview_timing.get("dmp_rollout_ms", 0.0)
                ),
                "fp_shep_geometry_metric_ms": float(
                    preview_timing.get("sensor_reconstruction_ms", 0.0)
                    + preview_timing.get("geometry_metric_ms", 0.0)
                    + score_metric_ms
                ),
                "fp_shep_actor_batch_calls": float(
                    preview_timing.get("actor_batch_calls", 0.0)
                ),
                "fp_shep_preview_branch_count": float(
                    preview_timing.get("preview_branch_count", 0.0)
                ),
            }
        )
    return tuple(records)


def choose_execution_reference(
    *,
    method: str,
    terminal_task_goal: np.ndarray,
    proposals: Sequence[Any],
    policy: Any | None = None,
    env: Any | None = None,
    agent_index: int | None = None,
    score_spec: FPSHEPOnlineScoreSpec | None = None,
) -> SelectionDecision:
    """Choose an execution reference without adding a selectable null class."""

    terminal_task_goal = _vector3(terminal_task_goal, "terminal_task_goal")
    proposals = tuple(proposals)
    if method == METHOD_FROZEN:
        return SelectionDecision(
            method=method,
            execution_reference=terminal_task_goal,
            selection_kind=SELECTION_BASELINE_TERMINAL,
            proposals=(),
            selected_candidate_id=None,
            selected_candidate_original_index=None,
        )
    if method not in {METHOD_PROPOSAL, METHOD_FP_SHEP}:
        raise ValueError(f"unknown method: {method}")
    if not proposals:
        return SelectionDecision(
            method=method,
            execution_reference=terminal_task_goal,
            selection_kind=SELECTION_NO_CANDIDATE_FALLBACK,
            proposals=(),
            selected_candidate_id=None,
            selected_candidate_original_index=None,
        )
    if method == METHOD_PROPOSAL:
        return SelectionDecision(
            method=method,
            execution_reference=proposals[0].point,
            selection_kind=SELECTION_PROPOSAL,
            proposals=proposals,
            selected_candidate_id=0,
            selected_candidate_original_index=0,
        )
    if policy is None or env is None or agent_index is None:
        raise ValueError("FP-SHEP selection requires policy, environment and agent_index")
    scored = score_fp_shep_candidates(
        env=env,
        agent_index=int(agent_index),
        proposals=proposals,
        policy=policy,
        spec=score_spec,
    )
    scores = np.asarray([item.score for item in scored], dtype=float)
    selected_id = int(np.argmax(scores))
    return SelectionDecision(
        method=method,
        execution_reference=proposals[selected_id].point,
        selection_kind=SELECTION_FP_SHEP,
        proposals=proposals,
        selected_candidate_id=selected_id,
        selected_candidate_original_index=selected_id,
        fp_shep_records=scored,
    )


def goal_switch(previous: np.ndarray, current: np.ndarray, epsilon: float) -> tuple[bool, float]:
    jump = float(np.linalg.norm(_vector3(current, "current") - _vector3(previous, "previous")))
    return bool(jump > float(epsilon)), jump


def minimum_pairwise_distance(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(points) < 2:
        return float("inf")
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return float(np.min(distances[np.triu_indices(len(points), k=1)]))


def trajectory_metrics(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    *,
    dt: float,
) -> dict[str, Any]:
    """Compute established path and jerk-based smoothness metrics."""

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    accelerations = np.asarray(accelerations, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape (T+1, N, 3)")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must match positions")
    if accelerations.ndim != 3 or accelerations.shape[1:] != positions.shape[1:]:
        raise ValueError("accelerations must have shape (T, N, 3)")
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    path_lengths = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=2), axis=0)
    velocity_deltas = np.diff(velocities, axis=0)
    velocity_variation = (
        np.mean(np.linalg.norm(velocity_deltas, axis=2), axis=0)
        if velocity_deltas.size
        else np.zeros(positions.shape[1], dtype=float)
    )
    acceleration_deltas = np.diff(accelerations, axis=0)
    acceleration_variation = (
        np.mean(np.linalg.norm(acceleration_deltas, axis=2), axis=0)
        if acceleration_deltas.size
        else np.zeros(positions.shape[1], dtype=float)
    )
    if acceleration_deltas.size:
        jerk = acceleration_deltas / dt
        smoothness = np.mean(np.sum(jerk ** 2, axis=2), axis=0)
        jerk_rms = np.sqrt(np.mean(jerk ** 2, axis=(0, 2)))
    else:
        smoothness = np.zeros(positions.shape[1], dtype=float)
        jerk_rms = np.zeros(positions.shape[1], dtype=float)
    return {
        "path_lengths": path_lengths,
        "path_length_team_mean": float(np.mean(path_lengths)),
        "path_length_team_sum": float(np.sum(path_lengths)),
        "velocity_variation_per_agent": velocity_variation,
        "velocity_variation_team_mean": float(np.mean(velocity_variation)),
        "acceleration_variation_per_agent": acceleration_variation,
        "acceleration_variation_team_mean": float(np.mean(acceleration_variation)),
        "trajectory_smoothness_per_agent": smoothness,
        "trajectory_smoothness_team_mean": float(np.mean(smoothness)),
        "jerk_rms_per_agent": jerk_rms,
        "jerk_rms_team_mean": float(np.mean(jerk_rms)),
        "smoothness_definition": "mean_over_time(sum_xyz((delta_acceleration/dt)^2))",
    }


def termination_reason(*, success: bool, collision: bool, terminated: bool, truncated: bool) -> str:
    if success:
        return "success"
    if collision:
        return "collision"
    if truncated:
        return "timeout"
    if terminated:
        return "terminated"
    return "running"
