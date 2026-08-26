"""Minimal post-GAT transient-aware candidate veto (TACV).

The module consumes the already-computed frozen FP-SHEP preview records and
the already-computed GAT interaction diagnostics.  It never changes the
candidate graph, logits, preview rollout, or learned checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TACVConfig:
    dt_s: float
    activation_threshold: float
    minimum_relative_reduction: float
    maximum_gat_candidate_rank: int = 3
    apply_to_initial_selection: bool = False
    numeric_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        if not np.isfinite(self.activation_threshold) or self.activation_threshold < 0.0:
            raise ValueError("activation_threshold must be finite and nonnegative")
        if not 0.0 < self.minimum_relative_reduction < 1.0:
            raise ValueError("minimum_relative_reduction must lie in (0,1)")
        if self.maximum_gat_candidate_rank < 2:
            raise ValueError("maximum_gat_candidate_rank must permit an alternative")
        if self.numeric_tolerance < 0.0:
            raise ValueError("numeric_tolerance must be nonnegative")


def preview_transient_metrics(
    current_applied_acceleration: Sequence[float] | np.ndarray,
    predicted_applied_accelerations: Sequence[Sequence[float]] | np.ndarray,
    *,
    dt_s: float,
) -> dict[str, float]:
    current = np.asarray(current_applied_acceleration, dtype=float)
    predicted = np.asarray(predicted_applied_accelerations, dtype=float)
    if current.shape != (3,) or not np.all(np.isfinite(current)):
        raise ValueError("current applied acceleration must be a finite 3-vector")
    if predicted.ndim != 2 or predicted.shape[1] != 3 or not predicted.shape[0]:
        raise ValueError("predicted accelerations must have shape (H,3)")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("predicted accelerations must be finite")
    acceleration = np.vstack([current[None, :], predicted])
    jerk = np.diff(acceleration, axis=0) / float(dt_s)
    norm = np.linalg.norm(jerk, axis=1)
    return {
        "J_preview": float(np.mean(np.sum(jerk**2, axis=1))),
        "J_preview_peak": float(np.max(norm)),
        "J_preview_mean": float(np.mean(norm)),
        "J_preview_vertical": float(np.mean(jerk[:, 2] ** 2)),
        "J_preview_lateral": float(np.mean(np.sum(jerk[:, :2] ** 2, axis=1))),
    }


def safety_noninferior(
    selected: Mapping[str, Any],
    alternative: Mapping[str, Any],
    *,
    tolerance: float = 1.0e-12,
) -> bool:
    """Conservative Pareto check using only frozen safety descriptors."""

    if bool(alternative["risky"]) and not bool(selected["risky"]):
        return False
    if (
        float(alternative["preview_min_clearance_m"]) + tolerance
        < float(selected["preview_min_clearance_m"])
    ):
        return False
    if bool(selected["risky"]) and bool(alternative["risky"]):
        if (
            float(alternative["maximum_risk_duration_s"])
            > float(selected["maximum_risk_duration_s"]) + tolerance
        ):
            return False
        selected_separation = selected["minimum_predicted_separation_m"]
        alternative_separation = alternative["minimum_predicted_separation_m"]
        selected_value = (
            float("inf") if selected_separation is None else float(selected_separation)
        )
        alternative_value = (
            float("inf")
            if alternative_separation is None
            else float(alternative_separation)
        )
        if alternative_value + tolerance < selected_value:
            return False
    return True


def _candidate_ranks(class_logits: Sequence[float], candidate_count: int) -> dict[int, int]:
    logits = np.asarray(class_logits, dtype=float)
    if logits.shape != (candidate_count + 1,) or not np.all(np.isfinite(logits)):
        raise ValueError("class_logits must be finite null-plus-candidate logits")
    order = np.argsort(-logits[1:], kind="stable")
    return {int(candidate_id): int(rank + 1) for rank, candidate_id in enumerate(order)}


def select_tacv_candidate(
    *,
    original_selected_candidate_id: int | None,
    class_logits: Sequence[float],
    preview_records: Sequence[Any],
    interaction_records: Sequence[Mapping[str, Any]],
    current_applied_acceleration: Sequence[float] | np.ndarray,
    event_type: str,
    config: TACVConfig,
) -> dict[str, Any]:
    """Apply TACV after the unchanged GAT selection and safety mask."""

    candidate_count = len(preview_records)
    candidate_ids = [int(record.candidate_id) for record in preview_records]
    if sorted(candidate_ids) != list(range(candidate_count)):
        raise ValueError("preview records must preserve the frozen candidate id mapping")
    interaction_by_id = {
        int(row["candidate_id"]): dict(row) for row in interaction_records
    }
    if sorted(interaction_by_id) != list(range(candidate_count)):
        raise ValueError("interaction records must cover the full Top-K candidate set")
    ranks = _candidate_ranks(class_logits, candidate_count)
    logits = np.asarray(class_logits, dtype=float)
    candidates: list[dict[str, Any]] = []
    for record in preview_records:
        candidate_id = int(record.candidate_id)
        interaction = interaction_by_id[candidate_id]
        transient = preview_transient_metrics(
            current_applied_acceleration,
            record.preview.trajectory.accelerations,
            dt_s=config.dt_s,
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "gat_candidate_rank": ranks[candidate_id],
                "gat_logit": float(logits[candidate_id + 1]),
                "fp_shep_score": float(record.score),
                "preview_min_clearance_m": float(record.preview.min_clearance),
                "interaction_edge_count": int(
                    interaction.get("interaction_edge_count", 0)
                ),
                "minimum_predicted_separation_m": (
                    None
                    if interaction.get("minimum_predicted_separation_m") is None
                    else float(interaction["minimum_predicted_separation_m"])
                ),
                "maximum_risk_duration_s": float(
                    interaction.get("maximum_risk_duration_s", 0.0)
                ),
                "risky": bool(interaction.get("risky", False)),
                **transient,
            }
        )

    base = {
        "event_type": str(event_type),
        "original_selected_candidate_id": original_selected_candidate_id,
        "effective_selected_candidate_id": original_selected_candidate_id,
        "activation_threshold": float(config.activation_threshold),
        "minimum_relative_reduction": float(config.minimum_relative_reduction),
        "maximum_gat_candidate_rank": int(config.maximum_gat_candidate_rank),
        "candidate_count": candidate_count,
        "activated": False,
        "replaced": False,
        "reason": "UNCHANGED",
        "original_J_preview": None,
        "replacement_J_preview": None,
        "predicted_relative_reduction": None,
        "original_gat_candidate_rank": None,
        "replacement_gat_candidate_rank": None,
        "safety_noninferior_alternative_count": 0,
        "material_alternative_count": 0,
        "candidate_diagnostics": candidates,
    }
    if original_selected_candidate_id is None:
        return {**base, "reason": "ORIGINAL_NULL_SELECTION"}
    if not 0 <= int(original_selected_candidate_id) < candidate_count:
        raise ValueError("original selected candidate id is outside Top-K")
    original = candidates[int(original_selected_candidate_id)]
    base["original_J_preview"] = float(original["J_preview"])
    base["original_gat_candidate_rank"] = int(original["gat_candidate_rank"])
    if str(event_type) == "INITIAL_SELECTION" and not config.apply_to_initial_selection:
        return {**base, "reason": "INITIAL_SELECTION_EXCLUDED"}
    if float(original["J_preview"]) < float(config.activation_threshold):
        return {**base, "reason": "BELOW_ACTIVATION_THRESHOLD"}

    base["activated"] = True
    ordered = sorted(candidates, key=lambda row: (int(row["gat_candidate_rank"]), int(row["candidate_id"])))
    safety_count = 0
    material_count = 0
    selected_alternative: Mapping[str, Any] | None = None
    for alternative in ordered:
        if int(alternative["candidate_id"]) == int(original_selected_candidate_id):
            continue
        if int(alternative["gat_candidate_rank"]) > config.maximum_gat_candidate_rank:
            continue
        if not safety_noninferior(
            original, alternative, tolerance=config.numeric_tolerance
        ):
            continue
        safety_count += 1
        relative_reduction = 1.0 - float(alternative["J_preview"]) / max(
            float(original["J_preview"]), 1.0e-300
        )
        if relative_reduction + config.numeric_tolerance < config.minimum_relative_reduction:
            continue
        material_count += 1
        selected_alternative = alternative
        break
    base["safety_noninferior_alternative_count"] = safety_count
    base["material_alternative_count"] = material_count
    if selected_alternative is None:
        return {**base, "reason": "NO_ADMISSIBLE_MATERIAL_ALTERNATIVE"}

    relative_reduction = 1.0 - float(selected_alternative["J_preview"]) / max(
        float(original["J_preview"]), 1.0e-300
    )
    return {
        **base,
        "effective_selected_candidate_id": int(selected_alternative["candidate_id"]),
        "replaced": True,
        "reason": "REPLACED_BY_FIRST_GAT_RANKED_ADMISSIBLE_ALTERNATIVE",
        "replacement_J_preview": float(selected_alternative["J_preview"]),
        "predicted_relative_reduction": float(relative_reduction),
        "replacement_gat_candidate_rank": int(
            selected_alternative["gat_candidate_rank"]
        ),
    }


__all__ = [
    "TACVConfig",
    "preview_transient_metrics",
    "safety_noninferior",
    "select_tacv_candidate",
]
