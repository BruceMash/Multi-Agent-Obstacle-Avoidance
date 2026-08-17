"""Pure post-hoc analysis for the frozen FP-SHEP failure-mode audit.

This module consumes recorded artifacts and diagnostic traces.  It does not
import or alter the candidate generator, FP-SHEP propagation, SAC, DMP, or
environment execution code.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "fp_shep_failure_mode_analysis_v1"
UNREACHABLE_PRIMARY_CATEGORIES = (
    "TIME_BUDGET_ONLY",
    "STAGNATION",
    "OSCILLATION",
    "ACTION_SATURATION",
    "OBSTACLE_AVOIDANCE_DEADLOCK",
    "INTER_AGENT_BLOCKING",
    "UNKNOWN",
)


def parse_json_cell(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list, tuple)):
        return value
    return json.loads(str(value))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else result


def vector3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(parse_json_cell(value, value), dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape (3,)")
    return result


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1.0e-12:
        return None
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def point_to_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1.0e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * delta)))


def segment_sphere_surface_clearance(
    start: np.ndarray,
    end: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_effective_radius: float,
) -> float:
    return point_to_segment_distance(obstacle_center, start, end) - float(
        obstacle_effective_radius
    )


def _obstacle_arrays(layout: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    obstacles = list(layout["obstacles"])
    centers = np.asarray([row["center"] for row in obstacles], dtype=float)
    radii = np.asarray(
        [
            float(row.get("effective_radius_m", row["radius_m"] + row.get("safety_margin_m", 0.0)))
            for row in obstacles
        ],
        dtype=float,
    )
    return centers, radii


def candidate_geometry_record(
    *,
    layout: Mapping[str, Any],
    agent_id: int,
    candidate: np.ndarray,
    proposal_rank: int | None,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute evaluation-only geometry without filtering the candidate."""

    agent_id = int(agent_id)
    starts = np.asarray(layout["starts"], dtype=float)
    goals = np.asarray(layout["terminal_goals"], dtype=float)
    start = starts[agent_id]
    terminal = goals[agent_id]
    candidate = np.asarray(candidate, dtype=float)
    centers, radii = _obstacle_arrays(layout)
    point_clearances = np.linalg.norm(centers - candidate[None, :], axis=1) - radii
    start_candidate_clearances = np.asarray(
        [
            segment_sphere_surface_clearance(start, candidate, center, radius)
            for center, radius in zip(centers, radii, strict=True)
        ]
    )
    candidate_terminal_clearances = np.asarray(
        [
            segment_sphere_surface_clearance(candidate, terminal, center, radius)
            for center, radius in zip(centers, radii, strict=True)
        ]
    )
    influence = float(thresholds["candidate_influence_distance_m"])
    narrow_clearance = float(thresholds["candidate_narrow_clearance_m"])
    turn = angle_degrees(candidate - start, terminal - candidate)

    terminal_delta = terminal - candidate
    terminal_length_sq = float(np.dot(terminal_delta, terminal_delta))
    obstacle_between_candidate_and_terminal = []
    for center in centers:
        if terminal_length_sq <= 1.0e-12:
            fraction = 0.0
        else:
            fraction = float(np.dot(center - candidate, terminal_delta) / terminal_length_sq)
        obstacle_between_candidate_and_terminal.append(0.0 < fraction < 1.0)
    backside = bool(
        any(obstacle_between_candidate_and_terminal)
        and float(np.min(candidate_terminal_clearances)) <= influence
    )

    route = terminal - start
    route_length_sq = float(np.dot(route, route))
    candidate_projection = (
        float(np.dot(candidate - start, route) / route_length_sq)
        if route_length_sq > 1.0e-12
        else 0.0
    )
    obstacle_projections = (
        np.dot(centers - start[None, :], route) / route_length_sq
        if route_length_sq > 1.0e-12
        else np.zeros(len(centers), dtype=float)
    )
    lateral_side = bool(
        np.any(np.abs(obstacle_projections - candidate_projection) <= 0.15)
        and np.min(point_clearances) <= influence
    )
    narrow_region = bool(
        len(point_clearances) >= 2
        and np.partition(point_clearances, 1)[1] <= narrow_clearance
    )
    min_candidate_terminal = float(np.min(candidate_terminal_clearances))
    poor = bool(
        min_candidate_terminal < 0.0
        or (
            backside
            and turn is not None
            and turn >= float(thresholds["poor_handoff_turning_angle_deg"])
        )
        or (
            min_candidate_terminal <= float(thresholds["poor_handoff_clearance_m"])
            and turn is not None
            and turn >= 90.0
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "layout_id": layout["layout_id"],
        "family": layout["family"],
        "agent_id": agent_id,
        "proposal_rank": proposal_rank,
        "candidate_world_position": candidate.tolist(),
        "candidate_distance_from_ego_m": float(np.linalg.norm(candidate - start)),
        "candidate_distance_to_terminal_m": float(np.linalg.norm(terminal - candidate)),
        "candidate_to_terminal_turning_angle_deg": turn,
        "candidate_minimum_static_surface_clearance_m": float(np.min(point_clearances)),
        "candidate_distance_to_obstacle_influence_region_m": float(
            np.min(point_clearances) - influence
        ),
        "start_to_candidate_minimum_static_clearance_m": float(
            np.min(start_candidate_clearances)
        ),
        "candidate_to_terminal_minimum_static_clearance_m": min_candidate_terminal,
        "obstacle_lateral_side": lateral_side,
        "obstacle_backside": backside,
        "narrow_geometric_region": narrow_region,
        "geometrically_poor_handoff_candidate": poor,
        "geometry_used_for_selection": False,
    }


def score_margin_record(
    row: Mapping[str, Any], *, epsilon: float
) -> dict[str, Any]:
    records = list(parse_json_cell(row.get("fp_shep_candidate_records"), []))
    scores = sorted(
        [float(item["fp_shep_online_score"]) for item in records], reverse=True
    )
    best = scores[0] if scores else None
    second = scores[1] if len(scores) >= 2 else None
    third = scores[2] if len(scores) >= 3 else None
    score_range = (max(scores) - min(scores)) if len(scores) >= 2 else 0.0
    margin_1_2 = (best - second) if second is not None else None
    margin_1_3 = (best - third) if third is not None else None
    normalized = (
        margin_1_2 / max(score_range, float(epsilon))
        if margin_1_2 is not None
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "layout_id": row["layout_id"],
        "family": row["family"],
        "agent_id": int(row["agent_id"]),
        "candidate_count": len(scores),
        "best_score": best,
        "second_best_score": second,
        "third_best_score": third,
        "margin_1_2": margin_1_2,
        "margin_1_3": margin_1_3,
        "score_range": score_range,
        "normalized_margin_1_2": normalized,
        "normalization_definition": "margin_1_2/max(candidate_score_range,epsilon)",
    }


def _sign_reversal_count(values: np.ndarray, epsilon: float = 1.0e-9) -> int:
    values = np.asarray(values, dtype=float)
    signs = np.sign(values[np.abs(values) > epsilon])
    return int(np.sum(signs[1:] != signs[:-1])) if signs.size >= 2 else 0


def classify_reference_unreachable(
    trace_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one mutually-exclusive primary subtype plus overlapping flags."""

    if not trace_rows:
        return {
            "primary_subcategory": "UNKNOWN",
            "diagnostic_flags": ["TRACE_UNAVAILABLE"],
        }
    rows = sorted(trace_rows, key=lambda row: int(row["step"]))
    window_size = min(int(thresholds["trace_window_steps"]), len(rows))
    window = rows[-window_size:]
    distances = np.asarray([float(row["distance_to_reference_m"]) for row in window])
    deltas = np.diff(distances)
    window_progress = float(distances[0] - distances[-1]) if len(distances) >= 2 else 0.0
    total_variation = float(np.sum(np.abs(deltas))) if deltas.size else 0.0
    reversals = _sign_reversal_count(deltas)
    saturation_rate = float(
        np.mean([float(row["action_saturation_fraction"]) for row in window])
    )
    forcing_saturation_rate = float(
        np.mean([float(row["forcing_saturation_fraction"]) for row in window])
    )
    goal_offset_saturation_rate = float(
        np.mean([float(row["goal_offset_saturation_fraction"]) for row in window])
    )
    mean_speed = float(np.mean([float(row["speed_mps"]) for row in window]))
    min_static = float(
        np.min([float(row["static_obstacle_clearance_m"]) for row in window])
    )
    min_peer = float(
        np.min([float(row["minimum_inter_agent_distance_m"]) for row in window])
    )
    stagnation = window_progress <= float(thresholds["stagnation_progress_m"])
    oscillation = bool(
        reversals >= int(thresholds["oscillation_distance_reversal_count"])
        and total_variation >= float(thresholds["oscillation_total_variation_m"])
    )
    action_saturation = saturation_rate >= float(thresholds["action_saturation_rate"])
    velocity_collapse = mean_speed <= float(thresholds["velocity_collapse_mps"])
    obstacle_deadlock = bool(
        stagnation
        and min_static <= float(thresholds["obstacle_pressure_distance_m"])
    )
    inter_agent_blocking = bool(
        stagnation
        and min_peer <= float(thresholds["inter_agent_pressure_distance_m"])
    )
    time_budget_only = bool(
        window_progress > float(thresholds["time_budget_progress_m"])
        and not oscillation
        and not action_saturation
        and not obstacle_deadlock
        and not inter_agent_blocking
    )
    flags = {
        "TIME_BUDGET_PROGRESS_CONTINUING": time_budget_only,
        "STAGNATION": stagnation,
        "OSCILLATION": oscillation,
        "ACTION_SATURATION": action_saturation,
        "FORCING_SATURATION": forcing_saturation_rate
        >= float(thresholds["action_saturation_rate"]),
        "GOAL_OFFSET_SATURATION": goal_offset_saturation_rate
        >= float(thresholds["action_saturation_rate"]),
        "VELOCITY_COLLAPSE": velocity_collapse,
        "OBSTACLE_AVOIDANCE_DEADLOCK": obstacle_deadlock,
        "INTER_AGENT_BLOCKING": inter_agent_blocking,
        "PROGRESS_REVERSAL": bool(np.any(deltas > 0.0)),
    }
    if inter_agent_blocking:
        primary = "INTER_AGENT_BLOCKING"
    elif obstacle_deadlock:
        primary = "OBSTACLE_AVOIDANCE_DEADLOCK"
    elif action_saturation:
        primary = "ACTION_SATURATION"
    elif oscillation:
        primary = "OSCILLATION"
    elif stagnation:
        primary = "STAGNATION"
    elif time_budget_only:
        primary = "TIME_BUDGET_ONLY"
    else:
        primary = "UNKNOWN"
    if primary not in UNREACHABLE_PRIMARY_CATEGORIES:
        raise RuntimeError("unreachable subtype is not mutually-exclusive schema member")
    return {
        "primary_subcategory": primary,
        "diagnostic_flags": [name for name, active in flags.items() if active],
        "window_steps": window_size,
        "window_reference_progress_m": window_progress,
        "window_reference_distance_total_variation_m": total_variation,
        "window_reference_distance_reversal_count": reversals,
        "window_action_saturation_rate": saturation_rate,
        "window_forcing_saturation_rate": forcing_saturation_rate,
        "window_goal_offset_saturation_rate": goal_offset_saturation_rate,
        "window_mean_speed_mps": mean_speed,
        "window_minimum_static_clearance_m": min_static,
        "window_minimum_inter_agent_distance_m": min_peer,
    }


def descriptive_statistics(values: Iterable[Any]) -> dict[str, Any]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=float,
    )
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def family_failure_summary(
    cohort_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cohort_rows:
        groups[str(row["family"])].append(row)
    results: list[dict[str, Any]] = []
    for family in sorted(groups):
        members = groups[family]
        categories = Counter(str(row["primary_failure_category"]) for row in members)
        results.append(
            {
                "schema_version": SCHEMA_VERSION,
                "family": family,
                "layout_count": len(members),
                "success_count": sum(parse_bool(row["team_success"]) for row in members),
                "reference_unreachable_count": sum(
                    "REFERENCE_UNREACHABLE" in str(row["primary_failure_category"])
                    for row in members
                ),
                "obstacle_collision_count": sum(
                    "OBSTACLE_COLLISION" in str(row["primary_failure_category"])
                    for row in members
                ),
                "inter_agent_failure_count": sum(
                    "INTER_AGENT_COLLISION" in str(row["primary_failure_category"])
                    for row in members
                ),
                "horizon_limitation_count": sum(
                    parse_bool(row.get("horizon_limitation_signal", False)) for row in members
                ),
                "geometric_handoff_issue_count": sum(
                    parse_bool(row.get("geometrically_poor_handoff_candidate", False))
                    for row in members
                ),
                "lower_policy_compatibility_issue_count": sum(
                    parse_bool(row.get("lower_policy_local_compatibility_failure", False))
                    for row in members
                ),
                "primary_category_counts": dict(categories),
            }
        )
    return results


def evidence_label(fraction: float, thresholds: Mapping[str, Any]) -> str:
    if fraction >= float(thresholds["mechanism_yes_fraction"]):
        return "YES"
    if fraction >= float(thresholds["mechanism_partial_fraction"]):
        return "PARTIAL"
    return "NO"


def build_mechanism_conclusion(
    *,
    failed_layouts: Sequence[Mapping[str, Any]],
    horizon_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    lower_policy_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    terminal_speed_diagnostic_value: str = "NOT_ESTABLISHED",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failure_count = max(1, len(failed_layouts))
    failed_ids = {str(row["layout_id"]) for row in failed_layouts}
    all_ids = {str(row["layout_id"]) for row in horizon_rows}
    control_ids = all_ids - failed_ids
    control_count = max(1, len(control_ids))
    selected_long = [
        row
        for row in horizon_rows
        if str(row.get("candidate_role")) == "selected"
        and int(row["H_diag"]) > 4
    ]
    horizon_layouts = {
        str(row["layout_id"])
        for row in selected_long
        if str(row["layout_id"]) in failed_ids
        and parse_bool(row.get("new_failure_signal_beyond_h4", False))
    }
    horizon_control_layouts = {
        str(row["layout_id"])
        for row in selected_long
        if str(row["layout_id"]) in control_ids
        and parse_bool(row.get("new_failure_signal_beyond_h4", False))
    }
    h4_failure_signal_layouts = {
        str(row["layout_id"])
        for row in horizon_rows
        if str(row["layout_id"]) in failed_ids
        and str(row.get("candidate_role")) == "selected"
        and int(row["H_diag"]) == 4
        and parse_bool(row.get("diagnostic_failure_signal", False))
    }
    geometric_layouts = {
        str(row["layout_id"])
        for row in geometry_rows
        if str(row["layout_id"]) in failed_ids
        and parse_bool(row["geometrically_poor_handoff_candidate"])
    }
    geometric_control_layouts = {
        str(row["layout_id"])
        for row in geometry_rows
        if str(row["layout_id"]) in control_ids
        and parse_bool(row["geometrically_poor_handoff_candidate"])
    }
    lower_layouts = {
        str(row["layout_id"])
        for row in lower_policy_rows
        if parse_bool(row.get("lower_policy_local_compatibility_failure", False))
    }
    residual_coordination = {
        str(row["layout_id"])
        for row in failed_layouts
        if str(row["primary_failure_category"])
        == "RESIDUAL_INTER_AGENT_COLLISION_AFTER_INDIVIDUAL_EXECUTION"
    }
    h4_control_signal_layouts = {
        str(row["layout_id"])
        for row in horizon_rows
        if str(row["layout_id"]) in control_ids
        and str(row.get("candidate_role")) == "selected"
        and int(row["H_diag"]) == 4
        and parse_bool(row.get("diagnostic_failure_signal", False))
    }

    # Longer-horizon progress separation is a cohort-level secondary signal.
    # It uses a threshold frozen before the diagnostic run and cannot by itself
    # produce a YES label because it is not a layout-level termination signal.
    median_progress: dict[tuple[str, int], float | None] = {}
    for cohort_name, ids in (("failure", failed_ids), ("control", control_ids)):
        for horizon in (4, 8, 12, 20):
            values = [
                float(row["task_progress"])
                for row in horizon_rows
                if str(row["layout_id"]) in ids
                and str(row.get("candidate_role")) == "selected"
                and int(row["H_diag"]) == horizon
                and math.isfinite(float(row["task_progress"]))
            ]
            median_progress[(cohort_name, horizon)] = (
                float(np.median(values)) if values else None
            )
    h4_failure_median = median_progress[("failure", 4)]
    h4_control_median = median_progress[("control", 4)]
    h4_gap = (
        abs(float(h4_control_median) - float(h4_failure_median))
        if h4_failure_median is not None and h4_control_median is not None
        else 0.0
    )
    longer_gaps = []
    for horizon in (8, 12, 20):
        failure_median = median_progress[("failure", horizon)]
        control_median = median_progress[("control", horizon)]
        if failure_median is not None and control_median is not None:
            longer_gaps.append(
                (horizon, abs(float(control_median) - float(failure_median)))
            )
    best_horizon, best_gap = max(longer_gaps, key=lambda item: item[1], default=(None, 0.0))
    progress_separation_gain = float(best_gap - h4_gap)
    progress_separation_supported = progress_separation_gain >= float(
        thresholds["progress_degradation_from_h4_m"]
    )

    mechanisms = [
        (
            "SHORT_PREVIEW_HORIZON_LIMITATION",
            horizon_layouts,
            horizon_control_layouts,
            "longer_horizon_failure_signal",
        ),
        (
            "PREVIEW_FEATURE_LIMITATION",
            h4_failure_signal_layouts,
            h4_control_signal_layouts,
            "H4_failure_signal",
        ),
        (
            "CANDIDATE_ADMISSIBILITY_LIMITATION",
            geometric_layouts,
            geometric_control_layouts,
            "geometrically_poor_handoff_candidate",
        ),
        (
            "LOWER_POLICY_LOCAL_COMPATIBILITY_LIMITATION",
            lower_layouts,
            set(),
            "strict_lower_policy_compatibility_rule",
        ),
        (
            "RESIDUAL_COORDINATION_LIMITATION",
            residual_coordination,
            set(),
            "residual_inter_agent_failure_after_individual_execution",
        ),
    ]
    rows = []
    for name, layout_ids, control_layout_ids, evidence_definition in mechanisms:
        fraction = len(layout_ids) / failure_count
        control_fraction = len(control_layout_ids) / control_count
        excess_fraction = max(0.0, fraction - control_fraction)
        label = evidence_label(excess_fraction, thresholds)
        if (
            name == "SHORT_PREVIEW_HORIZON_LIMITATION"
            and label == "NO"
            and progress_separation_supported
        ):
            label = "PARTIAL"
        if (
            name == "PREVIEW_FEATURE_LIMITATION"
            and label == "NO"
            and str(terminal_speed_diagnostic_value) in {"MODERATE", "HIGH"}
        ):
            # Terminal speed is already recorded but excluded from the formal
            # three-feature score.  A descriptive cohort effect supports only
            # PARTIAL diagnostic value; it does not justify adding the feature.
            label = "PARTIAL"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "mechanism": name,
                "supporting_layout_count": len(layout_ids),
                "failure_layout_count": len(failed_layouts),
                "supporting_fraction": fraction,
                "success_control_supporting_layout_count": len(control_layout_ids),
                "success_control_layout_count": len(control_ids),
                "success_control_supporting_fraction": control_fraction,
                "failure_specific_excess_fraction": excess_fraction,
                "label": label,
                "supporting_layout_ids": sorted(layout_ids),
                "success_control_supporting_layout_ids": sorted(control_layout_ids),
                "evidence_definition": evidence_definition,
                "h4_progress_median_gap_m": (
                    h4_gap if name == "SHORT_PREVIEW_HORIZON_LIMITATION" else None
                ),
                "maximum_longer_horizon_progress_median_gap_m": (
                    best_gap if name == "SHORT_PREVIEW_HORIZON_LIMITATION" else None
                ),
                "maximum_gap_horizon": (
                    best_horizon if name == "SHORT_PREVIEW_HORIZON_LIMITATION" else None
                ),
                "progress_separation_gain_over_h4_m": (
                    progress_separation_gain
                    if name == "SHORT_PREVIEW_HORIZON_LIMITATION"
                    else None
                ),
                "cohort_progress_separation_secondary_support": (
                    progress_separation_supported
                    if name == "SHORT_PREVIEW_HORIZON_LIMITATION"
                    else None
                ),
                "terminal_speed_secondary_support": (
                    str(terminal_speed_diagnostic_value)
                    if name == "PREVIEW_FEATURE_LIMITATION"
                    else None
                ),
            }
        )
    label_rank = {"YES": 2, "PARTIAL": 1, "NO": 0}
    ordered = sorted(
        rows,
        key=lambda row: (
            -label_rank[str(row["label"])],
            -float(row["failure_specific_excess_fraction"]),
            row["mechanism"],
        ),
    )
    yes_rows = [row for row in ordered if row["label"] == "YES"]
    partial_rows = [row for row in ordered if row["label"] == "PARTIAL"]
    supported = yes_rows + partial_rows
    if yes_rows:
        primary = yes_rows[0]["mechanism"]
        secondary = (
            supported[1]["mechanism"] if len(supported) >= 2 else "NOT_ESTABLISHED"
        )
    elif len(partial_rows) >= 2:
        primary = "MIXED"
        secondary = " + ".join(sorted(str(row["mechanism"]) for row in partial_rows))
    else:
        primary = partial_rows[0]["mechanism"] if partial_rows else "NOT_ESTABLISHED"
        secondary = "NOT_ESTABLISHED"
    label_by_name = {row["mechanism"]: row["label"] for row in rows}
    conclusion = {
        **label_by_name,
        "TERMINAL_SPEED_DIAGNOSTIC_VALUE": "NOT_ESTABLISHED",
        "PRIMARY_LIMITATION": primary,
        "SECONDARY_LIMITATION": secondary,
        "FP_SHEP_REDESIGN_RECOMMENDED": (
            "YES"
            if label_by_name.get("SHORT_PREVIEW_HORIZON_LIMITATION")
            in {"YES", "PARTIAL"}
            or label_by_name.get("PREVIEW_FEATURE_LIMITATION")
            in {"YES", "PARTIAL"}
            else "NO"
        ),
        "SAC_FINE_TUNING_RECOMMENDED": (
            "YES"
            if label_by_name.get("LOWER_POLICY_LOCAL_COMPATIBILITY_LIMITATION") == "YES"
            else "NOT_ESTABLISHED"
        ),
        "GAT_STAGE_I_RECOMMENDED": (
            "YES"
            if label_by_name.get("RESIDUAL_COORDINATION_LIMITATION") == "YES"
            else "NO"
        ),
    }
    return rows, conclusion


__all__ = [
    "SCHEMA_VERSION",
    "UNREACHABLE_PRIMARY_CATEGORIES",
    "angle_degrees",
    "build_mechanism_conclusion",
    "candidate_geometry_record",
    "classify_reference_unreachable",
    "descriptive_statistics",
    "evidence_label",
    "family_failure_summary",
    "parse_bool",
    "parse_json_cell",
    "score_margin_record",
    "segment_sphere_surface_clearance",
]
