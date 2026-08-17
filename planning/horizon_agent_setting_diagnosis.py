"""Pure analysis helpers for the horizon × agent-setting diagnosis.

The module contains no controller, policy, environment, or training logic.  It
only validates horizon-only protocol pairs and aggregates already completed
evaluation records.  Single-vs-multi claims are deliberately restricted to
the shared ``open`` and ``sparse_static`` scenarios.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "horizon_agent_setting_diagnosis_v1"
HORIZON_SHORT = 50
HORIZON_LONG = 220
SHARED_AGENT_SETTING_SCENARIOS = ("open", "sparse_static")
EFFECT_NONE_THRESHOLD_PP = 10.0
EFFECT_STRONG_THRESHOLD_PP = 30.0


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


def horizon_independent_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(contract))
    if "max_steps" not in result:
        raise ValueError("protocol contract must contain max_steps")
    result.pop("max_steps")
    return result


def validate_horizon_only_pair(
    short_contract: Mapping[str, Any],
    long_contract: Mapping[str, Any],
) -> dict[str, Any]:
    short_steps = int(short_contract.get("max_steps", -1))
    long_steps = int(long_contract.get("max_steps", -1))
    if (short_steps, long_steps) != (HORIZON_SHORT, HORIZON_LONG):
        raise ValueError("horizon pair must be exactly 50 and 220 steps")
    short_common = horizon_independent_contract(short_contract)
    long_common = horizon_independent_contract(long_contract)
    if short_common != long_common:
        raise ValueError("horizon pair differs in fields other than max_steps")
    return {
        "status": "PASSED",
        "short_max_steps": short_steps,
        "long_max_steps": long_steps,
        "horizon_independent_contract_hash": stable_hash(short_common),
        "only_active_difference": "max_steps",
    }


def classify_effect(rate_difference: float) -> str:
    """Classify absolute effect size; this is not a significance test."""

    difference_pp = abs(float(rate_difference)) * 100.0
    if difference_pp < EFFECT_NONE_THRESHOLD_PP:
        return "NONE"
    if difference_pp < EFFECT_STRONG_THRESHOLD_PP:
        return "WEAK"
    return "STRONG"


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _rate_from_counts(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: str,
    scenario_scope: str,
) -> dict[str, Any]:
    selected = [row for row in rows if str(row["protocol"]) == protocol]
    if scenario_scope == "shared":
        selected = [
            row
            for row in selected
            if str(row["scenario"]) in SHARED_AGENT_SETTING_SCENARIOS
        ]
    elif scenario_scope != "all":
        selected = [row for row in selected if str(row["scenario"]) == scenario_scope]

    available = sum(int(row["temporary_reference_available_count"]) for row in selected)
    reached = sum(int(row["temporary_reference_reached_count"]) for row in selected)
    reached_terminal = sum(
        int(row["reached_then_terminal_completion_count"]) for row in selected
    )
    terminal_success_count = sum(bool(row["team_terminal_success"]) for row in selected)
    collision_count = sum(bool(row["collision"]) for row in selected)
    obstacle_collision_count = sum(bool(row["obstacle_collision"]) for row in selected)
    inter_agent_collision_count = sum(
        bool(row["inter_agent_collision"]) for row in selected
    )
    timeout_count = sum(bool(row["timeout"]) for row in selected)
    team_stage1_eligible = sum(
        int(row["temporary_reference_available_count"]) > 0 for row in selected
    )
    team_stage1_success = sum(bool(row["team_stage1_success"]) for row in selected)
    reached_steps: list[float] = []
    completion_steps: list[float] = []
    remaining_steps: list[float] = []
    stage2_steps: list[float] = []
    for row in selected:
        reached_steps.extend(float(value) for value in row.get("reference_reached_steps", []))
        completion_steps.extend(
            float(value) for value in row.get("terminal_completion_steps", [])
        )
        remaining_steps.extend(
            float(value) for value in row.get("remaining_steps_after_reference", [])
        )
        stage2_steps.extend(
            float(value) for value in row.get("stage2_completion_steps", [])
        )

    stage2_eligible_episodes = sum(
        int(row["temporary_reference_reached_count"]) > 0 for row in selected
    )
    stage1_collision_count = sum(
        int(row["collision_before_reference_reached_count"]) for row in selected
    )
    stage2_collision_count = sum(
        int(row["collision_after_reference_reached_count"]) for row in selected
    )
    stage2_timeout_count = sum(
        bool(row["timeout_after_reference_reached"]) for row in selected
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "agent_setting": selected[0]["agent_setting"] if selected else None,
        "max_steps": int(selected[0]["max_steps"]) if selected else None,
        "scenario_scope": scenario_scope,
        "scenarios": sorted({str(row["scenario"]) for row in selected}),
        "episode_count": len(selected),
        "terminal_success_count": terminal_success_count,
        "terminal_success_rate": _rate_from_counts(terminal_success_count, len(selected)),
        "collision_count": collision_count,
        "collision_rate": _rate_from_counts(collision_count, len(selected)),
        "obstacle_collision_count": obstacle_collision_count,
        "obstacle_collision_rate": _rate_from_counts(
            obstacle_collision_count, len(selected)
        ),
        "inter_agent_collision_count": inter_agent_collision_count,
        "inter_agent_collision_rate": _rate_from_counts(
            inter_agent_collision_count, len(selected)
        ),
        "timeout_count": timeout_count,
        "timeout_rate": _rate_from_counts(timeout_count, len(selected)),
        "agent_reference_available_count": available,
        "agent_reference_reached_count": reached,
        "agent_reference_reached_rate": _rate_from_counts(reached, available),
        "agent_reached_then_terminal_count": reached_terminal,
        "agent_reached_then_terminal_rate": _rate_from_counts(
            reached_terminal, reached
        ),
        "team_stage1_eligible_episode_count": team_stage1_eligible,
        "team_stage1_success_episode_count": team_stage1_success,
        "team_stage1_success_rate": _rate_from_counts(
            team_stage1_success, team_stage1_eligible
        ),
        "stage1_reach_count": reached,
        "stage1_reach_rate": _rate_from_counts(reached, available),
        "stage1_mean_steps": float(np.mean(reached_steps)) if reached_steps else None,
        "stage1_collision_count": stage1_collision_count,
        "stage1_collision_rate": _rate_from_counts(
            stage1_collision_count, available
        ),
        "stage2_completion_count": reached_terminal,
        "stage2_completion_rate": _rate_from_counts(reached_terminal, reached),
        "stage2_mean_steps": float(np.mean(stage2_steps)) if stage2_steps else None,
        "stage2_collision_count": stage2_collision_count,
        "stage2_collision_rate": _rate_from_counts(stage2_collision_count, reached),
        "stage2_timeout_count": stage2_timeout_count,
        "stage2_timeout_eligible_episode_count": stage2_eligible_episodes,
        "stage2_timeout_rate": _rate_from_counts(
            stage2_timeout_count, stage2_eligible_episodes
        ),
        "collision_before_reference_reached_count": stage1_collision_count,
        "collision_after_reference_reached_count": stage2_collision_count,
        "timeout_after_reference_reached_count": stage2_timeout_count,
        "mean_reference_reached_step": (
            float(np.mean(reached_steps)) if reached_steps else None
        ),
        "mean_terminal_completion_step": (
            float(np.mean(completion_steps)) if completion_steps else None
        ),
        "mean_remaining_steps_after_reference": (
            float(np.mean(remaining_steps)) if remaining_steps else None
        ),
        "mean_terminal_progress_m": _mean(selected, "terminal_progress_m"),
        "mean_path_length_m": _mean(selected, "path_length_m"),
        "mean_trajectory_smoothness": _mean(selected, "trajectory_smoothness"),
    }


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return str(row["agent_setting"]), str(row["scenario"]), int(row["seed"])


def build_paired_comparisons(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[_pair_key(row)][int(row["max_steps"])] = row

    comparisons: list[dict[str, Any]] = []
    errors: list[str] = []
    for key, by_horizon in sorted(grouped.items()):
        if set(by_horizon) != {HORIZON_SHORT, HORIZON_LONG}:
            errors.append(f"{key}: missing 50/220 pair")
            continue
        short = by_horizon[HORIZON_SHORT]
        long = by_horizon[HORIZON_LONG]
        initial_match = short["initial_condition_hash"] == long["initial_condition_hash"]
        reference_match = short["temporary_reference_hash"] == long["temporary_reference_hash"]
        contract_match = (
            short["horizon_independent_contract_hash"]
            == long["horizon_independent_contract_hash"]
        )
        short_steps = int(short["episode_steps"])
        if short_steps < HORIZON_SHORT:
            prefix_match = (
                short["final_state_hash"] == long["final_state_hash"]
                and int(long["episode_steps"]) == short_steps
                and short["termination_reason"] == long["termination_reason"]
            )
        else:
            prefix_match = (
                short["final_state_hash"] == long["state_at_step_50_hash"]
            )
        if not all((initial_match, reference_match, contract_match, prefix_match)):
            errors.append(
                f"{key}: initial={initial_match}, reference={reference_match}, "
                f"contract={contract_match}, prefix={prefix_match}"
            )
        actual_success = bool(long["team_terminal_success"])
        counterfactual_success = bool(
            actual_success
            and long.get("team_terminal_completion_step") is not None
            and int(long["team_terminal_completion_step"]) <= HORIZON_SHORT
        )
        short_reached_failed = bool(
            int(short["temporary_reference_reached_count"]) > 0
            and not bool(short["team_terminal_success"])
        )
        completion_after_50 = bool(
            actual_success
            and int(long["team_terminal_completion_step"]) > HORIZON_SHORT
        )
        comparisons.append(
            {
                "schema_version": SCHEMA_VERSION,
                "agent_setting": key[0],
                "scenario": key[1],
                "seed": key[2],
                "short_protocol": short["protocol"],
                "long_protocol": long["protocol"],
                "initial_condition_match": initial_match,
                "temporary_reference_match": reference_match,
                "horizon_independent_contract_match": contract_match,
                "trajectory_prefix_through_step_50_match": prefix_match,
                "short_terminal_success": bool(short["team_terminal_success"]),
                "long_terminal_success": actual_success,
                "short_reference_reached_count": int(
                    short["temporary_reference_reached_count"]
                ),
                "short_reference_available_count": int(
                    short["temporary_reference_available_count"]
                ),
                "short_reached_but_terminal_failed": short_reached_failed,
                "long_terminal_completion_step": long.get(
                    "team_terminal_completion_step"
                ),
                "counterfactual_success_at_50": counterfactual_success,
                "actual_success_at_220": actual_success,
                "success_censored_by_50_step_horizon": bool(
                    actual_success and not counterfactual_success
                ),
                "terminal_completion_after_step_50": completion_after_50,
                "m50_reached_failure_recovered_after_50": bool(
                    key[0] == "multi_agent"
                    and short_reached_failed
                    and completion_after_50
                ),
            }
        )
    integrity = {
        "status": "PASSED" if not errors else "FAILED",
        "pair_count": len(comparisons),
        "errors": errors,
        "all_initial_conditions_paired": not errors,
        "all_step_50_prefixes_matched": not errors,
    }
    return comparisons, integrity


def _effect_record(
    *,
    name: str,
    baseline: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_rate = float(baseline["terminal_success_rate"] or 0.0)
    comparison_rate = float(comparison["terminal_success_rate"] or 0.0)
    difference = comparison_rate - baseline_rate
    return {
        "name": name,
        "baseline_protocol": baseline["protocol"],
        "comparison_protocol": comparison["protocol"],
        "baseline_success_count": int(baseline["terminal_success_count"]),
        "baseline_episode_count": int(baseline["episode_count"]),
        "baseline_success_rate": baseline_rate,
        "comparison_success_count": int(comparison["terminal_success_count"]),
        "comparison_episode_count": int(comparison["episode_count"]),
        "comparison_success_rate": comparison_rate,
        "rate_difference": difference,
        "rate_difference_percentage_points": difference * 100.0,
        "classification": classify_effect(difference),
        "classification_is_statistical_significance": False,
    }


def analyze_diagnosis(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    protocols = ("S50", "S220", "M50", "M220")
    scenario_names = sorted({str(row["scenario"]) for row in rows})
    stage_rows: list[dict[str, Any]] = []
    for protocol in protocols:
        stage_rows.append(aggregate_rows(rows, protocol=protocol, scenario_scope="all"))
        stage_rows.append(aggregate_rows(rows, protocol=protocol, scenario_scope="shared"))
        for scenario in scenario_names:
            selected = [
                row
                for row in rows
                if str(row["protocol"]) == protocol
                and str(row["scenario"]) == scenario
            ]
            if selected:
                stage_rows.append(
                    aggregate_rows(rows, protocol=protocol, scenario_scope=scenario)
                )

    index = {(row["protocol"], row["scenario_scope"]): row for row in stage_rows}
    single_effect = _effect_record(
        name="HORIZON_EFFECT_SINGLE",
        baseline=index[("S50", "all")],
        comparison=index[("S220", "all")],
    )
    multi_effect = _effect_record(
        name="HORIZON_EFFECT_MULTI",
        baseline=index[("M50", "all")],
        comparison=index[("M220", "all")],
    )
    multi_at_50 = _effect_record(
        name="MULTI_AGENT_EFFECT_SHARED_SCENARIOS_AT_50",
        baseline=index[("M50", "shared")],
        comparison=index[("S50", "shared")],
    )
    # Positive difference means Single-Agent outperforms Multi-Agent.
    multi_at_220 = _effect_record(
        name="MULTI_AGENT_EFFECT_SHARED_SCENARIOS_AT_220",
        baseline=index[("M220", "shared")],
        comparison=index[("S220", "shared")],
    )

    pairs, pairing = build_paired_comparisons(rows)
    long_pairs = [pair for pair in pairs if bool(pair["actual_success_at_220"])]
    censored = [
        pair for pair in long_pairs if bool(pair["success_censored_by_50_step_horizon"])
    ]
    multi_long_success = [
        pair
        for pair in long_pairs
        if pair["agent_setting"] == "multi_agent"
    ]
    multi_after_50 = [
        pair
        for pair in multi_long_success
        if bool(pair["terminal_completion_after_step_50"])
    ]
    m50_reached_failures = [
        pair
        for pair in pairs
        if pair["agent_setting"] == "multi_agent"
        and bool(pair["short_reached_but_terminal_failed"])
    ]
    m50_horizon_recoveries = [
        pair
        for pair in m50_reached_failures
        if bool(pair["m50_reached_failure_recovered_after_50"])
    ]

    horizon_single_class = single_effect["classification"]
    horizon_multi_class = multi_effect["classification"]
    multi_shared_class = multi_at_220["classification"]
    multi_gap = float(multi_at_220["rate_difference"])
    multi_gain = float(multi_effect["rate_difference"])
    single_gain = float(single_effect["rate_difference"])
    if multi_shared_class == "NONE" and horizon_multi_class in {"WEAK", "STRONG"}:
        primary = "SHORT_EVALUATION_HORIZON"
    elif multi_shared_class in {"WEAK", "STRONG"} and horizon_multi_class in {
        "WEAK",
        "STRONG",
    }:
        primary = "HORIZON_PLUS_MULTI_AGENT_INTERACTION"
    elif multi_shared_class in {"WEAK", "STRONG"} and horizon_multi_class == "NONE":
        primary = "MULTI_AGENT_REFERENCE_TRANSITION_DISTRIBUTION"
    elif single_gain > 0.0 and multi_gain > 0.0:
        primary = "EPISODE_HORIZON_MISMATCH"
    else:
        primary = "NO_MATERIAL_LIMITATION_ESTABLISHED"

    if primary in {"SHORT_EVALUATION_HORIZON", "EPISODE_HORIZON_MISMATCH"}:
        fine_tuning = "NO"
        return_to_gat = "NO"
        next_step = "Set a defensible closed-loop horizon before reassessing coordination methods."
    elif primary == "HORIZON_PLUS_MULTI_AGENT_INTERACTION":
        # The residual gap is not, by itself, evidence that the frozen lower
        # policy is the cause.  Candidate, collision, and inter-agent effects
        # remain confounded, so this diagnosis must not recommend fine-tuning.
        fine_tuning = "NO"
        return_to_gat = "YES"
        next_step = (
            "Separate residual inter-agent/candidate compatibility failures from lower-policy "
            "adaptation before any SAC fine-tuning decision."
        )
    elif primary == "MULTI_AGENT_REFERENCE_TRANSITION_DISTRIBUTION":
        fine_tuning = "NOT_ESTABLISHED"
        return_to_gat = "YES"
        next_step = (
            "Return to multi-agent coordination analysis and evaluate FP-SHEP/GAT relevance."
        )
    else:
        fine_tuning = "NO"
        return_to_gat = "NO"
        next_step = "Retain the frozen lower policy and review evaluation assumptions."

    conclusion = {
        "HORIZON_EFFECT_SINGLE": horizon_single_class,
        "HORIZON_EFFECT_MULTI": horizon_multi_class,
        "MULTI_AGENT_EFFECT_SHARED_SCENARIOS": multi_shared_class,
        "HORIZON_EFFECT": horizon_multi_class,
        "MULTI_AGENT_EFFECT": multi_shared_class,
        "PRIMARY_LIMITATION": primary,
        "SAC_FINE_TUNING_RECOMMENDED": fine_tuning,
        "RETURN_TO_GAT_COORDINATION": return_to_gat,
        "NEXT_STEP": next_step,
        "effect_classification_is_statistical_significance": False,
        "single_horizon_effect": single_effect,
        "multi_horizon_effect": multi_effect,
        "multi_agent_effect_shared_at_50": multi_at_50,
        "multi_agent_effect_shared_at_220": multi_at_220,
        "setting_specific_scenarios_are_supplementary_only": [
            "historical_base",
            "multi_agent",
        ],
        "terminal_completion_after_step_50_count_multi": len(multi_after_50),
        "terminal_completion_after_step_50_total_multi_success": len(
            multi_long_success
        ),
        "terminal_completion_after_step_50_rate_multi": _rate_from_counts(
            len(multi_after_50), len(multi_long_success)
        ),
        "success_censored_by_50_count_all": len(censored),
        "success_censored_by_50_total_220_success": len(long_pairs),
        "success_censored_by_50_rate_all": _rate_from_counts(
            len(censored), len(long_pairs)
        ),
        "m50_reached_failure_recovered_after_50_count": len(m50_horizon_recoveries),
        "m50_reached_failure_count": len(m50_reached_failures),
        "m50_reached_failure_horizon_attribution_rate": _rate_from_counts(
            len(m50_horizon_recoveries), len(m50_reached_failures)
        ),
        "shared_scenario_gap_at_220": multi_gap,
    }
    return {
        "stage_rows": stage_rows,
        "horizon_rows": [single_effect, multi_effect, multi_at_50, multi_at_220],
        "paired_rows": pairs,
        "censoring_rows": pairs,
        "pairing_integrity": pairing,
        "conclusion": conclusion,
    }


def render_final_report(
    *,
    config: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> str:
    conclusion = analysis["conclusion"]
    rows = {
        (row["protocol"], row["scenario_scope"]): row
        for row in analysis["stage_rows"]
    }
    pct = lambda value: "N/A" if value is None else f"{100.0 * float(value):.2f}%"
    lines = [
        "# Horizon × Agent-Setting Reference-Transition Diagnosis",
        "",
        "## Protocol",
        "",
        "- Frozen deterministic SAC checkpoint; no training or gradient update.",
        "- Historical vector forcing gate; one-shot Proposal.score top-1 reference.",
        "- S50/S220 and M50/M220 are horizon-only paired comparisons.",
        "- Single-vs-multi primary analysis uses only open and sparse_static.",
        "- historical_base and multi_agent are supplementary setting-specific diagnostics.",
        "",
        "## Core results",
        "",
        "| Protocol | Scope | Episodes | Terminal success | Agent ref reached | Agent reached→terminal | Collision | Timeout |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol in ("S50", "S220", "M50", "M220"):
        for scope in ("all", "shared"):
            row = rows[(protocol, scope)]
            lines.append(
                f"| {protocol} | {scope} | {row['episode_count']} | "
                f"{pct(row['terminal_success_rate'])} | "
                f"{pct(row['agent_reference_reached_rate'])} | "
                f"{pct(row['agent_reached_then_terminal_rate'])} | "
                f"{pct(row['collision_rate'])} | {pct(row['timeout_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Reference-transition stage diagnostics",
            "",
            "| Protocol | Stage-1 reach | Stage-1 mean steps | Stage-1 collision | Stage-2 completion | Stage-2 mean steps | Stage-2 collision | Stage-2 timeout |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for protocol in ("S50", "S220", "M50", "M220"):
        row = rows[(protocol, "all")]
        mean1 = (
            "N/A"
            if row["stage1_mean_steps"] is None
            else f"{row['stage1_mean_steps']:.2f}"
        )
        mean2 = (
            "N/A"
            if row["stage2_mean_steps"] is None
            else f"{row['stage2_mean_steps']:.2f}"
        )
        lines.append(
            f"| {protocol} | {row['stage1_reach_count']}/"
            f"{row['agent_reference_available_count']} ({pct(row['stage1_reach_rate'])}) | "
            f"{mean1} | {row['stage1_collision_count']}/"
            f"{row['agent_reference_available_count']} ({pct(row['stage1_collision_rate'])}) | "
            f"{row['stage2_completion_count']}/{row['agent_reference_reached_count']} "
            f"({pct(row['stage2_completion_rate'])}) | {mean2} | "
            f"{row['stage2_collision_count']}/{row['agent_reference_reached_count']} "
            f"({pct(row['stage2_collision_rate'])}) | "
            f"{row['stage2_timeout_count']}/{row['stage2_timeout_eligible_episode_count']} "
            f"({pct(row['stage2_timeout_rate'])}) |"
        )
    lines.extend(
        [
            "",
            "## Effect sizes",
            "",
            "| Effect | Baseline | Comparison | Rate difference | Label |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for effect in analysis["horizon_rows"]:
        lines.append(
            f"| {effect['name']} | {effect['baseline_success_count']}/"
            f"{effect['baseline_episode_count']} ({pct(effect['baseline_success_rate'])}) | "
            f"{effect['comparison_success_count']}/{effect['comparison_episode_count']} "
            f"({pct(effect['comparison_success_rate'])}) | "
            f"{effect['rate_difference_percentage_points']:.2f} pp | "
            f"{effect['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Censoring diagnosis",
            "",
            "- M220 terminal completions after step 50: "
            f"{conclusion['terminal_completion_after_step_50_count_multi']}/"
            f"{conclusion['terminal_completion_after_step_50_total_multi_success']} "
            f"({pct(conclusion['terminal_completion_after_step_50_rate_multi'])}).",
            "- All 220-step successes censored by a step-50 cutoff: "
            f"{conclusion['success_censored_by_50_count_all']}/"
            f"{conclusion['success_censored_by_50_total_220_success']} "
            f"({pct(conclusion['success_censored_by_50_rate_all'])}).",
            "- M50 reached-but-failed episodes recovered after step 50 in paired M220: "
            f"{conclusion['m50_reached_failure_recovered_after_50_count']}/"
            f"{conclusion['m50_reached_failure_count']} "
            f"({pct(conclusion['m50_reached_failure_horizon_attribution_rate'])}).",
            "",
            "## Effect diagnosis",
            "",
            f"- HORIZON_EFFECT_SINGLE = {conclusion['HORIZON_EFFECT_SINGLE']}",
            f"- HORIZON_EFFECT_MULTI = {conclusion['HORIZON_EFFECT_MULTI']}",
            "- MULTI_AGENT_EFFECT_SHARED_SCENARIOS = "
            f"{conclusion['MULTI_AGENT_EFFECT_SHARED_SCENARIOS']}",
            f"- PRIMARY_LIMITATION = {conclusion['PRIMARY_LIMITATION']}",
            "- SAC_FINE_TUNING_RECOMMENDED = "
            f"{conclusion['SAC_FINE_TUNING_RECOMMENDED']}",
            f"- RETURN_TO_GAT_COORDINATION = {conclusion['RETURN_TO_GAT_COORDINATION']}",
            f"- NEXT_STEP = {conclusion['NEXT_STEP']}",
            "",
            "Effect thresholds are descriptive labels, not statistical significance tests.",
            "No fine-tuning, FP-SHEP execution, GAT training, or horizon search was performed.",
            "",
            "## Reproducibility",
            "",
            f"- Schema: {SCHEMA_VERSION}",
            f"- Checkpoint: {config['checkpoint']}",
            f"- Checkpoint SHA256: {config['checkpoint_sha256_expected']}",
        ]
    )
    return "\n".join(lines) + "\n"
