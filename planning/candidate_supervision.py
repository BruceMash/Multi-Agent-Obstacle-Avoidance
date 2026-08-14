"""Candidate-supervision labels and audits built on controlled SAC-DMP rollouts.

This module is deliberately independent from the graph network.  It defines
fixed physical normalization, two provisional quality variants, failure-tier
ordering, soft targets, ranking metrics, and leakage-free dataset splits.
It does not train or select a GAT model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from planning.candidate_execution_interface import ExecutionNormalizationSpec


SUPERVISION_SCHEMA_VERSION = "candidate_supervision_v1"
PRIMARY_TARGET_NAME = "provisional_primary_target"
COMPANION_TARGET_NAME = "companion_target"
FORMAL_PREVIEW_HORIZON = 4
QUALITY_FEATURE_ORDER = (
    "task_progress",
    "min_clearance",
    "max_execution_deviation",
    "terminal_speed",
)
EPS = 1.0e-12


def _readonly(value: Any, *, dtype: Any = float) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CandidateQualitySpec:
    """Configurable, non-fitted candidate-quality definition.

    The three-feature variant is the provisional primary target.  The
    four-feature variant is a companion target for label audit.  Neither is
    declared final by this schema.
    """

    normalization: ExecutionNormalizationSpec = field(
        default_factory=ExecutionNormalizationSpec
    )
    progress_weight: float = 1.0
    clearance_weight: float = 1.0
    deviation_weight: float = 1.0
    terminal_speed_weight: float = 1.0
    failure_margin: float = 1.0e-3
    primary_target_name: str = PRIMARY_TARGET_NAME
    companion_target_name: str = COMPANION_TARGET_NAME
    primary_feature_count: int = 3
    companion_feature_count: int = 4
    final_target_frozen: bool = False

    def __post_init__(self) -> None:
        for name in (
            "progress_weight",
            "clearance_weight",
            "deviation_weight",
            "terminal_speed_weight",
            "failure_margin",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if int(self.primary_feature_count) != 3:
            raise ValueError("the provisional primary target must use three features")
        if int(self.companion_feature_count) != 4:
            raise ValueError("the companion target must use four features")
        if bool(self.final_target_frozen):
            raise ValueError("this supervision goal must not freeze the final target")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CandidateQualitySpec":
        normalization_values = values.get("normalization", {})
        normalization = (
            normalization_values
            if isinstance(normalization_values, ExecutionNormalizationSpec)
            else ExecutionNormalizationSpec.from_mapping(normalization_values)
            if normalization_values
            else ExecutionNormalizationSpec()
        )
        return cls(
            normalization=normalization,
            progress_weight=float(values.get("progress_weight", 1.0)),
            clearance_weight=float(values.get("clearance_weight", 1.0)),
            deviation_weight=float(values.get("deviation_weight", 1.0)),
            terminal_speed_weight=float(values.get("terminal_speed_weight", 1.0)),
            failure_margin=float(values.get("failure_margin", 1.0e-3)),
            primary_target_name=str(values.get("primary_target_name", PRIMARY_TARGET_NAME)),
            companion_target_name=str(values.get("companion_target_name", COMPANION_TARGET_NAME)),
            final_target_frozen=bool(values.get("final_target_frozen", False)),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SUPERVISION_SCHEMA_VERSION,
            "normalization": vars(self.normalization).copy(),
            "weights": {
                "task_progress": self.progress_weight,
                "min_clearance": self.clearance_weight,
                "max_execution_deviation": self.deviation_weight,
                "terminal_speed": self.terminal_speed_weight,
            },
            "provisional_primary_target": {
                "feature_count": 3,
                "formula": "+progress +clearance -deviation",
            },
            "companion_target": {
                "feature_count": 4,
                "formula": "+progress +clearance -deviation -terminal_speed",
            },
            "failure_policy": "collision_lexicographic_below_every_non_collision_branch",
            "failure_margin": self.failure_margin,
            "final_target_frozen": False,
        }


@dataclass(frozen=True)
class NormalizedQualityBatch:
    values: np.ndarray
    valid_mask: np.ndarray
    clipped_mask: np.ndarray
    clearance_finite_mask: np.ndarray
    clearance_positive_inf_flag: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(QUALITY_FEATURE_ORDER):
            raise ValueError("quality values must have shape (N, 4)")
        for name, value in (
            ("valid_mask", self.valid_mask),
            ("clipped_mask", self.clipped_mask),
        ):
            array = np.asarray(value, dtype=bool)
            if array.shape != values.shape:
                raise ValueError(f"{name} must match quality values")
            object.__setattr__(self, name, _readonly(array, dtype=bool))
        for name, value in (
            ("clearance_finite_mask", self.clearance_finite_mask),
            ("clearance_positive_inf_flag", self.clearance_positive_inf_flag),
        ):
            array = np.asarray(value, dtype=bool)
            if array.shape != (values.shape[0],):
                raise ValueError(f"{name} must have shape (N,)")
            object.__setattr__(self, name, _readonly(array, dtype=bool))
        if not np.all(np.isfinite(values)):
            raise ValueError("normalized quality values must be finite")
        object.__setattr__(self, "values", _readonly(values))


@dataclass(frozen=True)
class CandidateQualityBatch:
    normalized: NormalizedQualityBatch
    nominal_three_feature: np.ndarray
    nominal_four_feature: np.ndarray
    target_three_feature: np.ndarray
    target_four_feature: np.ndarray
    failure_mask: np.ndarray
    three_feature_bounds: tuple[float, float]
    four_feature_bounds: tuple[float, float]
    three_feature_failure_offset: float
    four_feature_failure_offset: float

    def __post_init__(self) -> None:
        count = int(self.normalized.values.shape[0])
        for name in (
            "nominal_three_feature",
            "nominal_four_feature",
            "target_three_feature",
            "target_four_feature",
        ):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != (count,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape (N,)")
            object.__setattr__(self, name, _readonly(array))
        failure = np.asarray(self.failure_mask, dtype=bool)
        if failure.shape != (count,):
            raise ValueError("failure_mask must have shape (N,)")
        object.__setattr__(self, "failure_mask", _readonly(failure, dtype=bool))


def _metric(row: Any, name: str) -> float:
    value = row[name] if isinstance(row, Mapping) else getattr(row, name)
    return float(value)


def normalize_quality_metrics(
    rows: Sequence[Any],
    spec: CandidateQualitySpec | None = None,
) -> NormalizedQualityBatch:
    """Normalize raw physical metrics without fitting to a candidate set."""

    spec = spec or CandidateQualitySpec()
    count = len(rows)
    values = np.zeros((count, 4), dtype=float)
    valid = np.zeros((count, 4), dtype=bool)
    clipped = np.zeros((count, 4), dtype=bool)
    clearance_finite = np.zeros(count, dtype=bool)
    clearance_inf = np.zeros(count, dtype=bool)
    scales = np.asarray(
        [
            spec.normalization.task_progress_scale,
            spec.normalization.min_clearance_scale,
            spec.normalization.max_execution_deviation_scale,
            spec.normalization.terminal_speed_scale,
        ],
        dtype=float,
    )
    lows = np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=float)
    highs = np.ones(4, dtype=float)
    for index, row in enumerate(rows):
        raw = np.asarray([_metric(row, name) for name in QUALITY_FEATURE_ORDER], dtype=float)
        clearance_finite[index] = bool(np.isfinite(raw[1]))
        clearance_inf[index] = bool(np.isposinf(raw[1]))
        for feature_index, raw_value in enumerate(raw):
            meaningful = bool(np.isfinite(raw_value))
            if feature_index == 1 and np.isposinf(raw_value):
                meaningful = True
                scaled = 1.0
            elif meaningful:
                scaled = float(raw_value / scales[feature_index])
            else:
                scaled = 0.0
            normalized = float(np.clip(scaled, lows[feature_index], highs[feature_index]))
            values[index, feature_index] = normalized
            valid[index, feature_index] = meaningful
            clipped[index, feature_index] = meaningful and not np.isclose(scaled, normalized)
    return NormalizedQualityBatch(
        values=values,
        valid_mask=valid,
        clipped_mask=clipped,
        clearance_finite_mask=clearance_finite,
        clearance_positive_inf_flag=clearance_inf,
    )


def quality_bounds(
    spec: CandidateQualitySpec,
    *,
    include_terminal_speed: bool,
) -> tuple[float, float]:
    lower = -spec.progress_weight - spec.deviation_weight
    upper = spec.progress_weight + spec.clearance_weight
    if include_terminal_speed:
        lower -= spec.terminal_speed_weight
    return float(lower), float(upper)


def _apply_failure_tier(
    nominal: np.ndarray,
    failure_mask: np.ndarray,
    bounds: tuple[float, float],
    margin: float,
) -> tuple[np.ndarray, float]:
    offset = float(bounds[1] - bounds[0] + margin)
    target = np.asarray(nominal, dtype=float).copy()
    target[np.asarray(failure_mask, dtype=bool)] -= offset
    return target, offset


def compute_candidate_quality(
    rows: Sequence[Any],
    *,
    failure_mask: Sequence[bool] | None = None,
    spec: CandidateQualitySpec | None = None,
) -> CandidateQualityBatch:
    """Compute auditable 3-feature and 4-feature quality vectors."""

    spec = spec or CandidateQualitySpec()
    normalized = normalize_quality_metrics(rows, spec)
    features = normalized.values
    nominal_three = (
        spec.progress_weight * features[:, 0]
        + spec.clearance_weight * features[:, 1]
        - spec.deviation_weight * features[:, 2]
    )
    nominal_four = nominal_three - spec.terminal_speed_weight * features[:, 3]
    failure = np.zeros(len(rows), dtype=bool) if failure_mask is None else np.asarray(failure_mask, dtype=bool)
    if failure.shape != (len(rows),):
        raise ValueError("failure_mask must have shape (N,)")
    three_bounds = quality_bounds(spec, include_terminal_speed=False)
    four_bounds = quality_bounds(spec, include_terminal_speed=True)
    target_three, three_offset = _apply_failure_tier(
        nominal_three, failure, three_bounds, spec.failure_margin
    )
    target_four, four_offset = _apply_failure_tier(
        nominal_four, failure, four_bounds, spec.failure_margin
    )
    return CandidateQualityBatch(
        normalized=normalized,
        nominal_three_feature=nominal_three,
        nominal_four_feature=nominal_four,
        target_three_feature=target_three,
        target_four_feature=target_four,
        failure_mask=failure,
        three_feature_bounds=three_bounds,
        four_feature_bounds=four_bounds,
        three_feature_failure_offset=three_offset,
        four_feature_failure_offset=four_offset,
    )


def termination_reason(rollout: Any, horizon: int) -> str:
    collision_types: list[str] = []
    for attribute, label in (
        ("obstacle_collision", "obstacle"),
        ("inter_agent_collision", "inter_agent"),
        ("boundary_collision", "boundary"),
    ):
        if bool(getattr(rollout, attribute, False)):
            collision_types.append(label)
    if bool(getattr(rollout, "collision", False)):
        suffix = "+".join(collision_types) if collision_types else "unspecified"
        return f"collision:{suffix}"
    if bool(getattr(rollout, "success", False)):
        return "team_success"
    if bool(getattr(rollout, "truncated", False)):
        return "timeout"
    if int(getattr(rollout, "effective_steps", 0)) < int(horizon):
        return "terminated_without_classified_outcome"
    if bool(getattr(rollout, "agent_success", False)):
        return "completed_horizon_agent_success"
    return "completed_horizon"


def branch_is_failure(rollout: Any) -> bool:
    """Current environment has an explicit collision failure signal only."""

    return bool(getattr(rollout, "collision", False))


def soft_target(values: Iterable[float], temperature: float) -> np.ndarray:
    values_array = np.asarray(list(values), dtype=float)
    temperature = float(temperature)
    if values_array.ndim != 1 or values_array.size == 0:
        raise ValueError("values must be a non-empty vector")
    if not np.all(np.isfinite(values_array)):
        raise ValueError("soft-target values must be finite")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    shifted = (values_array - float(np.max(values_array))) / temperature
    weights = np.exp(np.clip(shifted, -745.0, 0.0))
    return weights / float(np.sum(weights))


def soft_target_statistics(probabilities: Iterable[float]) -> dict[str, float]:
    probabilities = np.asarray(list(probabilities), dtype=float)
    positive = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return {
        "entropy": entropy,
        "normalized_entropy": (
            float(entropy / math.log(probabilities.size)) if probabilities.size > 1 else 0.0
        ),
        "effective_candidate_count": float(np.exp(entropy)),
        "maximum_probability": float(np.max(probabilities)),
    }


def average_ranks(values: Iterable[float], *, descending: bool = True) -> np.ndarray:
    values_array = np.asarray(list(values), dtype=float)
    if values_array.ndim != 1:
        raise ValueError("rank values must be one-dimensional")
    work = -values_array if descending else values_array
    order = np.argsort(work, kind="mergesort")
    ranks = np.empty(values_array.size, dtype=float)
    index = 0
    while index < values_array.size:
        end = index + 1
        while end < values_array.size and work[order[end]] == work[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    if x_array.shape != y_array.shape or x_array.ndim != 1 or x_array.size < 2:
        return None
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        return None
    x_rank = average_ranks(x_array)
    y_rank = average_ranks(y_array)
    if np.std(x_rank) < EPS or np.std(y_rank) < EPS:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def pearson(x: Iterable[float], y: Iterable[float]) -> float | None:
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    if x_array.shape != y_array.shape or x_array.ndim != 1 or x_array.size < 2:
        return None
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        return None
    if np.std(x_array) < EPS or np.std(y_array) < EPS:
        return None
    return float(np.corrcoef(x_array, y_array)[0, 1])


def pairwise_ranking_accuracy(
    predicted: Iterable[float],
    target: Iterable[float],
) -> float | None:
    predicted = np.asarray(list(predicted), dtype=float)
    target = np.asarray(list(target), dtype=float)
    if predicted.shape != target.shape or predicted.ndim != 1:
        raise ValueError("predicted and target must have matching vector shapes")
    scores: list[float] = []
    for left in range(predicted.size):
        for right in range(left + 1, predicted.size):
            actual_delta = float(target[left] - target[right])
            if abs(actual_delta) <= EPS:
                continue
            predicted_delta = float(predicted[left] - predicted[right])
            if abs(predicted_delta) <= EPS:
                scores.append(0.5)
            else:
                scores.append(float(np.sign(predicted_delta) == np.sign(actual_delta)))
    return float(np.mean(scores)) if scores else None


def descending_order(values: Iterable[float]) -> np.ndarray:
    values_array = np.asarray(list(values), dtype=float)
    if values_array.ndim != 1:
        raise ValueError("ordering values must be one-dimensional")
    return np.argsort(-values_array, kind="mergesort")


def top_m_oracle_recall(
    predicted: Iterable[float],
    target: Iterable[float],
    m: int,
) -> float:
    predicted_order = descending_order(predicted)
    target_order = descending_order(target)
    available = min(int(m), predicted_order.size)
    return float(int(target_order[0]) in set(predicted_order[:available].tolist()))


def candidate_set_ranking_metrics(
    *,
    proposal_scores_with_null: Iterable[float],
    formal_preview_quality: Iterable[float],
    real_target_quality: Iterable[float],
) -> dict[str, Any]:
    """Compare proposal-only and execution-aware ranking to the full oracle.

    Proposal Spearman/pairwise metrics exclude null because null has no
    Proposal.score.  Proposal Top-M metrics remain full-class metrics: an
    oracle-null state is therefore a proposal miss, as required by the schema.
    """

    proposal = np.asarray(list(proposal_scores_with_null), dtype=float)
    preview = np.asarray(list(formal_preview_quality), dtype=float)
    target = np.asarray(list(real_target_quality), dtype=float)
    if proposal.shape != preview.shape or proposal.shape != target.shape:
        raise ValueError("all candidate-set score vectors must have matching shapes")
    if proposal.size == 0 or not np.all(np.isfinite(preview)) or not np.all(np.isfinite(target)):
        raise ValueError("candidate-set quality vectors must be non-empty and finite")
    proposal_indices = np.flatnonzero(np.isfinite(proposal))
    proposal_order = proposal_indices[
        np.argsort(-proposal[proposal_indices], kind="mergesort")
    ] if proposal_indices.size else np.empty(0, dtype=int)
    oracle = int(descending_order(target)[0])
    fp_order = descending_order(preview)
    proposal_top1 = int(proposal_order[0]) if proposal_order.size else None
    result: dict[str, Any] = {
        "oracle_top1_class_index": oracle,
        "proposal_top1_class_index": proposal_top1,
        "fp_shep_top1_class_index": int(fp_order[0]),
        "proposal_spearman_proposals_only": (
            spearman(proposal[proposal_indices], target[proposal_indices])
            if proposal_indices.size >= 2 else None
        ),
        "fp_shep_spearman_all_classes": spearman(preview, target),
        "proposal_pairwise_accuracy_proposals_only": (
            pairwise_ranking_accuracy(proposal[proposal_indices], target[proposal_indices])
            if proposal_indices.size >= 2 else None
        ),
        "fp_shep_pairwise_accuracy_all_classes": pairwise_ranking_accuracy(preview, target),
        "proposal_top1_hit_full_target": float(proposal_top1 == oracle),
        "fp_shep_top1_hit_full_target": float(int(fp_order[0]) == oracle),
        "proposal_top3_oracle_recall_full_target": float(
            oracle in set(proposal_order[: min(3, proposal_order.size)].tolist())
        ),
        "fp_shep_top3_oracle_recall_full_target": top_m_oracle_recall(preview, target, 3),
        "proposal_null_score_available": False,
        "proposal_rank_cannot_select_null": True,
    }
    sorted_target = np.sort(target)[::-1]
    result["top1_top2_target_gap"] = (
        float(sorted_target[0] - sorted_target[1]) if sorted_target.size > 1 else 0.0
    )
    result["candidate_quality_std"] = float(np.std(target))
    return result


def target_bundle(
    quality: CandidateQualityBatch,
    temperatures: Sequence[float],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "provisional_primary_target": quality.target_three_feature.copy(),
        "companion_target": quality.target_four_feature.copy(),
        "hard_target_three_feature": int(np.argmax(quality.target_three_feature)),
        "hard_target_four_feature": int(np.argmax(quality.target_four_feature)),
        "soft_targets": {},
    }
    for temperature in temperatures:
        key = f"tau_{float(temperature):g}"
        primary = soft_target(quality.target_three_feature, float(temperature))
        companion = soft_target(quality.target_four_feature, float(temperature))
        result["soft_targets"][key] = {
            "temperature": float(temperature),
            "provisional_primary_target": primary,
            "companion_target": companion,
            "primary_statistics": soft_target_statistics(primary),
            "companion_statistics": soft_target_statistics(companion),
        }
    return result


def validate_sample_timesteps(
    timesteps: Sequence[int],
    *,
    maximum_label_horizon: int,
) -> tuple[int, ...]:
    resolved = tuple(int(value) for value in timesteps)
    if not resolved or resolved[0] < 0 or any(b <= a for a, b in zip(resolved, resolved[1:])):
        raise ValueError("sample timesteps must be a non-negative strictly increasing sequence")
    if any((b - a) < int(maximum_label_horizon) for a, b in zip(resolved, resolved[1:])):
        raise ValueError("sample timestep gaps must be at least H_label_max")
    return resolved


def assign_seed_split(
    seed: int,
    split_seeds: Mapping[str, Sequence[int]],
) -> str:
    matches = [name for name, seeds in split_seeds.items() if int(seed) in {int(item) for item in seeds}]
    if len(matches) != 1:
        raise ValueError(f"seed {seed} must belong to exactly one dataset split")
    return matches[0]


def validate_split_seeds(split_seeds: Mapping[str, Sequence[int]]) -> None:
    required = {"train", "validation", "test"}
    if set(split_seeds) != required:
        raise ValueError(f"split names must be exactly {sorted(required)}")
    flattened = [int(seed) for seeds in split_seeds.values() for seed in seeds]
    if len(flattened) != len(set(flattened)):
        raise ValueError("dataset split seed sets must be disjoint")


def state_group_id(*, scenario: str, seed: int, episode: int, timestep: int) -> str:
    return f"{scenario}__seed{int(seed):03d}__episode{int(episode):03d}__t{int(timestep):04d}"


def portable_raw_metric(value: float) -> dict[str, Any]:
    value = float(value)
    return {
        "value": value if np.isfinite(value) else None,
        "positive_inf_flag": bool(np.isposinf(value)),
        "negative_inf_flag": bool(np.isneginf(value)),
        "nan_flag": bool(np.isnan(value)),
    }

