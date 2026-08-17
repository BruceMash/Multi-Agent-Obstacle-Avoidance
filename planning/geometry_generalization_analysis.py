"""Pure post-hoc analysis for the frozen geometry generalization audit."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from planning.geometry_generalization_scenarios import pairwise_route_descriptors
from planning.pre_gat_220step_revalidation import (
    METHOD_FP_SHEP,
    METHOD_ORDER,
    METHOD_PROPOSAL,
    METHOD_TERMINAL,
)


SCHEMA_VERSION = "geometry_generalization_analysis_v1"


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator / denominator)


def _mean(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _median(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def aggregate_method_summary(
    episode_rows: Sequence[Mapping[str, Any]],
    agent_rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str] = ("method",),
) -> list[dict[str, Any]]:
    episode_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    agent_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        episode_groups[tuple(row[field] for field in group_fields)].append(row)
    for row in agent_rows:
        agent_groups[tuple(row[field] for field in group_fields)].append(row)
    results: list[dict[str, Any]] = []
    for key in sorted(episode_groups, key=lambda item: tuple(str(value) for value in item)):
        episodes = episode_groups[key]
        agents = agent_groups.get(key, [])
        method = str(episodes[0]["method"])
        successes = [row for row in episodes if bool(row["team_success"])]
        collisions = [row for row in episodes if bool(row["collision"])]
        obstacle_collisions = [row for row in episodes if bool(row["obstacle_collision"])]
        inter_collisions = [row for row in episodes if bool(row["inter_agent_collision"])]
        timeouts = [row for row in episodes if bool(row["timeout"])]
        available = [row for row in agents if bool(row.get("reference_available", False))]
        reached = [row for row in available if bool(row.get("reference_reached", False))]
        completed = [
            row for row in reached if bool(row.get("terminal_completed_after_reference", False))
        ]
        record = {field: value for field, value in zip(group_fields, key, strict=True)}
        record.update(
            {
                "schema_version": SCHEMA_VERSION,
                "episode_count": len(episodes),
                "team_success_count": len(successes),
                "team_success_rate": _rate(len(successes), len(episodes)),
                "collision_count": len(collisions),
                "collision_rate": _rate(len(collisions), len(episodes)),
                "obstacle_collision_count": len(obstacle_collisions),
                "obstacle_collision_rate": _rate(len(obstacle_collisions), len(episodes)),
                "inter_agent_collision_count": len(inter_collisions),
                "inter_agent_collision_rate": _rate(len(inter_collisions), len(episodes)),
                "timeout_count": len(timeouts),
                "timeout_rate": _rate(len(timeouts), len(episodes)),
                "mean_completion_step": _mean(row["steps"] for row in successes),
                "median_completion_step": _median(row["steps"] for row in successes),
                "mean_path_length_m": _mean(row["path_length_m"] for row in episodes),
                "mean_path_length_team_sum_m": _mean(
                    row["path_length_team_sum_m"] for row in episodes
                ),
                "mean_trajectory_smoothness": _mean(
                    row["trajectory_smoothness"] for row in episodes
                ),
                "mean_minimum_static_obstacle_clearance_m": _mean(
                    row["minimum_static_obstacle_clearance_m"] for row in episodes
                ),
                "mean_minimum_inter_agent_distance_m": _mean(
                    row["minimum_inter_agent_distance_m"] for row in episodes
                ),
                "reference_available_count": len(available),
                "reference_reached_count": len(reached),
                "reference_reached_rate": (
                    _rate(len(reached), len(available))
                    if method != METHOD_TERMINAL
                    else None
                ),
                "reached_then_terminal_count": len(completed),
                "reached_then_terminal_rate": (
                    _rate(len(completed), len(reached))
                    if method != METHOD_TERMINAL
                    else None
                ),
                "collision_before_reference_count": sum(
                    bool(row.get("collision_before_reference", False)) for row in available
                ),
                "collision_after_reference_count": sum(
                    bool(row.get("collision_after_reference", False)) for row in reached
                ),
                "timeout_before_reference_count": sum(
                    bool(row.get("reference_timeout", False)) for row in available
                ),
                "timeout_after_reference_count": sum(
                    bool(row.get("timeout_after_reference", False)) for row in reached
                ),
                "mean_reference_reached_step": _mean(
                    row.get("reference_reached_step") for row in reached
                ),
                "mean_terminal_completion_step": _mean(
                    row.get("terminal_completed_step") for row in completed
                ),
                "no_candidate_fallback_count": sum(
                    bool(row.get("no_candidate_fallback", False)) for row in agents
                ),
            }
        )
        results.append(record)
    return results


def build_selected_tuple_compatibility(
    episode_rows: Sequence[Mapping[str, Any]],
    agent_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    layout_index = {str(row["layout_id"]): row for row in manifest["layouts"]}
    episodes = {
        (str(row["layout_id"]), str(row["method"])): row for row in episode_rows
    }
    agents: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    selections: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in agent_rows:
        agents[(str(row["layout_id"]), str(row["method"]))].append(row)
    for row in selection_rows:
        selections[(str(row["layout_id"]), str(row["method"]))].append(row)
    results: list[dict[str, Any]] = []
    for key in sorted(selections):
        layout_id, method = key
        if method not in {METHOD_PROPOSAL, METHOD_FP_SHEP}:
            continue
        rows = sorted(selections[key], key=lambda row: int(row["agent_id"]))
        episode = episodes[key]
        agent_group = sorted(agents[key], key=lambda row: int(row["agent_id"]))
        starts = np.asarray(layout_index[layout_id]["starts"], dtype=float)
        references: list[np.ndarray] = []
        selected_indices: list[int | None] = []
        for row, start in zip(rows, starts, strict=True):
            value = row.get("selected_world_reference")
            references.append(start.copy() if value is None else np.asarray(value, dtype=float))
            selected = row.get("selected_candidate_index")
            selected_indices.append(None if selected is None else int(selected))
        reference_array = np.stack(references)
        pairs = pairwise_route_descriptors(starts, reference_array)
        nearest = min(
            pairs,
            key=lambda row: float(
                row["direct_path_predicted_minimum_inter_agent_distance_m"]
            ),
        )
        pairwise_reference = np.linalg.norm(
            reference_array[:, None, :] - reference_array[None, :, :], axis=-1
        )
        upper = pairwise_reference[np.triu_indices(len(reference_array), k=1)]
        reached_steps = [
            row.get("reference_reached_step")
            for row in agent_group
            if row.get("reference_reached_step") is not None
        ]
        actual_spread = (
            max(int(value) for value in reached_steps)
            - min(int(value) for value in reached_steps)
            if len(reached_steps) >= 2
            else None
        )
        results.append(
            {
                "schema_version": SCHEMA_VERSION,
                "layout_id": layout_id,
                "family": episode["family"],
                "method": method,
                "selected_candidate_tuple": selected_indices,
                "K_t_tuple": [int(row.get("K_t", 0)) for row in rows],
                "selected_reference_positions": reference_array.tolist(),
                "minimum_selected_reference_pairwise_distance_m": float(np.min(upper)),
                "predicted_straight_path_minimum_pairwise_distance_m": float(
                    nearest["direct_path_predicted_minimum_inter_agent_distance_m"]
                ),
                "predicted_closest_pair": nearest["pair"],
                "predicted_closest_approach_time_fraction": float(
                    nearest["predicted_closest_approach_time_fraction"]
                ),
                "predicted_closest_approach_time_difference_fraction": float(
                    nearest["predicted_closest_approach_time_difference_fraction"]
                ),
                "predicted_path_crossing_relation": bool(
                    float(nearest["spatial_path_minimum_distance_m"]) <= 0.6
                ),
                "predicted_arrival_time_overlap": bool(
                    float(nearest["spatial_path_minimum_distance_m"]) <= 0.6
                    and float(
                        nearest["predicted_closest_approach_time_difference_fraction"]
                    )
                    <= 0.08
                ),
                "reference_reached_count": sum(
                    bool(row.get("reference_reached", False)) for row in agent_group
                ),
                "all_available_references_reached": bool(
                    agent_group
                    and all(
                        not bool(row.get("reference_available", False))
                        or bool(row.get("reference_reached", False))
                        for row in agent_group
                    )
                ),
                "actual_reference_reached_step_spread": actual_spread,
                "team_success": bool(episode["team_success"]),
                "obstacle_collision": bool(episode["obstacle_collision"]),
                "inter_agent_collision": bool(episode["inter_agent_collision"]),
                "timeout": bool(episode["timeout"]),
                "online_selection_changed": False,
                "diagnostic_scope": "POST_HOC_FAILURE_EXPLANATION_ONLY",
            }
        )
    return results


def build_fp_shep_failure_analysis(
    episode_rows: Sequence[Mapping[str, Any]],
    agent_rows: Sequence[Mapping[str, Any]],
    compatibility_rows: Sequence[Mapping[str, Any]],
    *,
    inter_agent_pressure_distance_m: float = 1.2,
) -> list[dict[str, Any]]:
    compatibility = {
        (str(row["layout_id"]), str(row["method"])): row
        for row in compatibility_rows
    }
    grouped_agents: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in agent_rows:
        grouped_agents[(str(row["layout_id"]), str(row["method"]))].append(row)
    results: list[dict[str, Any]] = []
    for episode in episode_rows:
        if episode["method"] != METHOD_FP_SHEP or bool(episode["team_success"]):
            continue
        key = (str(episode["layout_id"]), METHOD_FP_SHEP)
        agents = sorted(grouped_agents[key], key=lambda row: int(row["agent_id"]))
        compat = compatibility[key]
        k_zero = any(int(row.get("K_t", 0)) == 0 for row in agents)
        all_reached = bool(
            agents
            and all(
                not bool(row.get("reference_available", False))
                or bool(row.get("reference_reached", False))
                for row in agents
            )
        )
        interaction_pressure = bool(
            float(episode["minimum_inter_agent_distance_m"])
            <= float(inter_agent_pressure_distance_m)
            or compat["predicted_path_crossing_relation"]
            or compat["predicted_arrival_time_overlap"]
        )
        if k_zero:
            primary = "K0_NO_CANDIDATE"
        elif bool(episode["obstacle_collision"]) and not bool(
            episode["inter_agent_collision"]
        ):
            primary = "PURE_OBSTACLE_COLLISION"
        elif bool(episode["inter_agent_collision"]) and all_reached:
            primary = "RESIDUAL_INTER_AGENT_COLLISION_AFTER_INDIVIDUAL_EXECUTION"
        elif bool(episode["inter_agent_collision"]):
            primary = "INTER_AGENT_COLLISION_BEFORE_REFERENCE_EXECUTABILITY"
        elif bool(episode["timeout"]) and all_reached and interaction_pressure:
            primary = "RESIDUAL_COORDINATION_RELATED_TIMEOUT"
        elif bool(episode["timeout"]) and not all_reached:
            primary = "REFERENCE_UNREACHABLE_OR_LOWER_POLICY_INSTABILITY"
        elif bool(episode["timeout"]):
            primary = "STAGE2_TERMINAL_TIMEOUT_WITHOUT_COORDINATION_SIGNATURE"
        else:
            primary = "OTHER_EXECUTION_FAILURE"
        coordination_signature = primary in {
            "RESIDUAL_INTER_AGENT_COLLISION_AFTER_INDIVIDUAL_EXECUTION",
            "RESIDUAL_COORDINATION_RELATED_TIMEOUT",
        }
        results.append(
            {
                "schema_version": SCHEMA_VERSION,
                "layout_id": episode["layout_id"],
                "family": episode["family"],
                "primary_failure_category": primary,
                "residual_multi_agent_coordination_signature": coordination_signature,
                "individual_reference_executability_established": all_reached,
                "K0_present": k_zero,
                "obstacle_collision": bool(episode["obstacle_collision"]),
                "inter_agent_collision": bool(episode["inter_agent_collision"]),
                "timeout": bool(episode["timeout"]),
                "minimum_inter_agent_distance_m": float(
                    episode["minimum_inter_agent_distance_m"]
                ),
                "minimum_static_obstacle_clearance_m": float(
                    episode["minimum_static_obstacle_clearance_m"]
                ),
                "reference_reached_count": sum(
                    bool(row.get("reference_reached", False)) for row in agents
                ),
                "reference_available_count": sum(
                    bool(row.get("reference_available", False)) for row in agents
                ),
                "collision_before_reference_count": sum(
                    bool(row.get("collision_before_reference", False)) for row in agents
                ),
                "collision_after_reference_count": sum(
                    bool(row.get("collision_after_reference", False)) for row in agents
                ),
                "selected_candidate_indices": [
                    row.get("selected_candidate_index") for row in agents
                ],
                "selected_proposal_ranks": [
                    row.get("selected_proposal_rank") for row in agents
                ],
                "preview_scores": [row.get("selected_fp_shep_score") for row in agents],
                "predicted_progress": [
                    row.get("selected_preview_task_progress") for row in agents
                ],
                "predicted_clearance": [
                    row.get("selected_preview_min_clearance") for row in agents
                ],
                "predicted_deviation": [
                    row.get("selected_preview_max_execution_deviation") for row in agents
                ],
                "selected_reference_positions": compat["selected_reference_positions"],
                "minimum_selected_reference_pairwise_distance_m": compat[
                    "minimum_selected_reference_pairwise_distance_m"
                ],
                "predicted_straight_path_minimum_pairwise_distance_m": compat[
                    "predicted_straight_path_minimum_pairwise_distance_m"
                ],
                "predicted_path_crossing_relation": compat[
                    "predicted_path_crossing_relation"
                ],
                "predicted_arrival_time_overlap": compat[
                    "predicted_arrival_time_overlap"
                ],
                "diagnostic_flags_overlap_allowed": True,
            }
        )
    return results


def build_paired_method_analysis(
    episode_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in episode_rows:
        episodes[str(row["layout_id"])][str(row["method"])] = row
    selections: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in selection_rows:
        selections[(str(row["layout_id"]), str(row["method"]), int(row["agent_id"]))] = row
    rows: list[dict[str, Any]] = []
    disagreement_numerator = 0
    disagreement_denominator = 0
    outcomes = Counter()
    for layout_id in sorted(episodes):
        methods = episodes[layout_id]
        if set(methods) != set(METHOD_ORDER):
            raise ValueError(f"incomplete method triplet for {layout_id}: {sorted(methods)}")
        proposal = methods[METHOD_PROPOSAL]
        fp = methods[METHOD_FP_SHEP]
        agent_pair_status: list[str] = []
        for agent_id in range(3):
            proposal_row = selections[(layout_id, METHOD_PROPOSAL, agent_id)]
            fp_row = selections[(layout_id, METHOD_FP_SHEP, agent_id)]
            if int(proposal_row.get("K_t", 0)) == 0:
                agent_pair_status.append("NO_CANDIDATE_PAIR")
                continue
            disagreement_denominator += 1
            disagree = int(proposal_row["selected_candidate_index"]) != int(
                fp_row["selected_candidate_index"]
            )
            disagreement_numerator += int(disagree)
            agent_pair_status.append("DISAGREE" if disagree else "AGREE")
        proposal_success = bool(proposal["team_success"])
        fp_success = bool(fp["team_success"])
        if proposal_success and fp_success:
            paired_outcome = "BOTH_SUCCESS"
        elif proposal_success:
            paired_outcome = "PROPOSAL_ONLY_SUCCESS"
        elif fp_success:
            paired_outcome = "FP_SHEP_ONLY_SUCCESS"
        else:
            paired_outcome = "BOTH_FAIL"
        outcomes[paired_outcome] += 1
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "layout_id": layout_id,
                "family": proposal["family"],
                "terminal_success": bool(methods[METHOD_TERMINAL]["team_success"]),
                "proposal_success": proposal_success,
                "fp_shep_success": fp_success,
                "paired_selector_outcome": paired_outcome,
                "agent_selector_pair_status": agent_pair_status,
                "selector_disagreement_count": sum(
                    item == "DISAGREE" for item in agent_pair_status
                ),
                "selector_comparable_agent_count": sum(
                    item != "NO_CANDIDATE_PAIR" for item in agent_pair_status
                ),
                "proposal_reference_reached_count": int(
                    proposal["temporary_reference_reached_count"]
                ),
                "fp_shep_reference_reached_count": int(
                    fp["temporary_reference_reached_count"]
                ),
                "proposal_reached_then_terminal_count": int(
                    proposal["reached_then_terminal_completion_count"]
                ),
                "fp_shep_reached_then_terminal_count": int(
                    fp["reached_then_terminal_completion_count"]
                ),
                "proposal_inter_agent_failure": bool(proposal["inter_agent_collision"]),
                "fp_shep_inter_agent_failure": bool(fp["inter_agent_collision"]),
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "layout_count": len(rows),
        "both_success_count": outcomes["BOTH_SUCCESS"],
        "proposal_only_success_count": outcomes["PROPOSAL_ONLY_SUCCESS"],
        "fp_shep_only_success_count": outcomes["FP_SHEP_ONLY_SUCCESS"],
        "both_fail_count": outcomes["BOTH_FAIL"],
        "selector_disagreement_count": disagreement_numerator,
        "selector_comparable_agent_count": disagreement_denominator,
        "selector_disagreement_rate": _rate(
            disagreement_numerator, disagreement_denominator
        ),
        "K0_excluded_from_selector_denominator": True,
    }
    return rows, summary


def build_conclusion(
    method_summary: Sequence[Mapping[str, Any]],
    family_summary: Sequence[Mapping[str, Any]],
    paired_summary: Mapping[str, Any],
    failure_rows: Sequence[Mapping[str, Any]],
    *,
    geometry_valid: bool,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    methods = {str(row["method"]): row for row in method_summary}
    proposal_rate = float(methods[METHOD_PROPOSAL]["team_success_rate"] or 0.0)
    fp_rate = float(methods[METHOD_FP_SHEP]["team_success_rate"] or 0.0)
    terminal_rate = float(methods[METHOD_TERMINAL]["team_success_rate"] or 0.0)
    residual = [
        row for row in failure_rows if bool(row["residual_multi_agent_coordination_signature"])
    ]
    residual_layouts = {str(row["layout_id"]) for row in residual}
    residual_families = {str(row["family"]) for row in residual}
    failure_count = len(failure_rows)
    residual_fraction = len(residual) / max(1, failure_count)
    thresholds = config["gat_decision_thresholds"]
    stable_residual = bool(
        len(residual_layouts) >= int(thresholds["minimum_residual_layout_count"])
        and len(residual_families) >= int(thresholds["minimum_residual_family_count"])
    )
    strong_threshold = float(thresholds["strong_fp_success_rate"])
    support_lower = float(thresholds["coordination_support_lower_success_rate"])
    residual_dominance = float(thresholds["residual_failure_fraction"])
    if fp_rate >= strong_threshold and not stable_residual:
        coordination_gap = "NO"
        gat_necessity = "NOT_ESTABLISHED"
        proceed = "NO"
    elif fp_rate >= strong_threshold and stable_residual:
        coordination_gap = "CONDITIONAL"
        gat_necessity = "POTENTIALLY_SUPPORTED"
        proceed = (
            "YES"
            if len(residual_layouts)
            >= int(thresholds["conditional_proceed_residual_layout_count"])
            else "NO"
        )
    elif fp_rate >= support_lower and stable_residual and residual_fraction >= residual_dominance:
        coordination_gap = "YES"
        gat_necessity = "SUPPORTED"
        proceed = "YES"
    else:
        coordination_gap = "NO"
        gat_necessity = "NOT_ESTABLISHED"
        proceed = "NO"
    individual_joint_cases = [
        row
        for row in residual
        if bool(row["individual_reference_executability_established"])
        and bool(row["predicted_path_crossing_relation"])
    ]
    individual_joint_gap = (
        "YES"
        if len(individual_joint_cases)
        >= int(thresholds["minimum_individual_joint_gap_case_count"])
        else "NOT_ESTABLISHED"
    )
    family_fp = {
        str(row["family"]): float(row["team_success_rate"] or 0.0)
        for row in family_summary
        if row["method"] == METHOD_FP_SHEP
    }
    family_advantage = 0
    family_proposal = {
        str(row["family"]): float(row["team_success_rate"] or 0.0)
        for row in family_summary
        if row["method"] == METHOD_PROPOSAL
    }
    for family, value in family_fp.items():
        family_advantage += int(value > family_proposal.get(family, 0.0))
    if fp_rate >= 0.75 and fp_rate - proposal_rate >= 0.20 and family_advantage >= 4:
        fp_signal = "STRONG"
    elif fp_rate > proposal_rate + 0.05:
        fp_signal = "MODERATE"
    else:
        fp_signal = "WEAK"
    primary_counts = Counter(str(row["primary_failure_category"]) for row in failure_rows)
    if not geometry_valid:
        primary_limitation = "GEOMETRY_MANIFEST_OR_ACCEPTANCE_GATE_INVALID"
        next_step = "Fix code/manifest semantics only; do not run method evaluation."
    elif proceed == "YES":
        primary_limitation = "INDIVIDUAL_SELECTOR_DOES_NOT_MODEL_JOINT_CANDIDATE_COMPATIBILITY"
        next_step = "Proceed to separately approved GAT Stage-I using the frozen evidence."
    elif fp_rate >= strong_threshold:
        primary_limitation = "RESIDUAL_COORDINATION_GAP_TOO_WEAK_OR_UNSTABLE"
        next_step = "Keep FP-SHEP as the primary method and treat GAT as an optional extension."
    else:
        primary_limitation = (
            primary_counts.most_common(1)[0][0] if primary_counts else "NO_FP_SHEP_FAILURE"
        )
        next_step = "Resolve the observed non-coordination bottleneck before GAT training."
    return {
        "GEOMETRY_GENERALIZATION_VALID": "YES" if geometry_valid else "NO",
        "FP_SHEP_GENERALIZATION_SIGNAL": fp_signal,
        "RESIDUAL_MULTI_AGENT_COORDINATION_GAP": coordination_gap,
        "INDIVIDUAL_VS_JOINT_COMPATIBILITY_GAP": individual_joint_gap,
        "GAT_CORE_NECESSITY": gat_necessity,
        "PROCEED_TO_GAT_STAGE_I": proceed,
        "PRIMARY_LIMITATION": primary_limitation,
        "NEXT_STEP": next_step,
        "terminal_team_success_rate": terminal_rate,
        "proposal_team_success_rate": proposal_rate,
        "fp_shep_team_success_rate": fp_rate,
        "fp_shep_minus_proposal_team_success_rate": fp_rate - proposal_rate,
        "fp_shep_failure_count": failure_count,
        "residual_coordination_failure_count": len(residual),
        "residual_coordination_layout_count": len(residual_layouts),
        "residual_coordination_family_count": len(residual_families),
        "residual_coordination_failure_fraction": residual_fraction,
        "individual_vs_joint_case_count": len(individual_joint_cases),
        "fp_shep_family_success_rates": family_fp,
        "proposal_family_success_rates": family_proposal,
        "fp_shep_advantage_family_count": family_advantage,
        "failure_primary_category_counts": dict(primary_counts),
        "paired_outcomes": dict(paired_summary),
        "statistical_significance_claimed": False,
        "decision_thresholds": dict(thresholds),
    }


__all__ = [
    "aggregate_method_summary",
    "build_conclusion",
    "build_fp_shep_failure_analysis",
    "build_paired_method_analysis",
    "build_selected_tuple_compatibility",
]
