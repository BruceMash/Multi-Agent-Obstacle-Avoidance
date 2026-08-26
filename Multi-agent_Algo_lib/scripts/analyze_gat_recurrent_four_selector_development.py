#!/usr/bin/env python3
"""Final four-selector Dev analysis and hard-gate decision."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.analyze_gat_r_vs_fp_development import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    aggregate,
    bootstrap_mean_ci,
    exact_mcnemar,
    finite,
    flag,
    read_csv,
    scope_rows,
    write_csv,
    write_json,
)


SCHEMA_VERSION = "gat_recurrent_four_selector_development_analysis_v1"
ARTIFACT_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
ARM_PATHS = {
    "fp_shep": "FP_SHEP_RERR_DEV400/development_team_results.csv",
    "gat_v1": "GAT_V1_RERR_DEV400/development_team_results.csv",
    "gat_r": "GAT_R_RERR_DEV400/development_team_results.csv",
    "gat_rs": "GAT_RS_RERR_DEV400/development_team_results.csv",
}
COMPARISONS = (
    ("gat_v1", "fp_shep"),
    ("gat_r", "fp_shep"),
    ("gat_rs", "fp_shep"),
    ("gat_rs", "gat_r"),
    ("gat_rs", "gat_v1"),
)
SCOPES = ("overall", "stage_1", "stage_2", "stage_3", "stage_4", "high_density")
BINARY_FIELDS = ("team_success", "collision", "obstacle_collision", "inter_agent_collision", "timeout")
CONTINUOUS_FIELDS = (
    "completion_time_s",
    "team_path_length_m",
    "trajectory_smoothness",
    "minimum_inter_agent_distance_m",
    "minimum_static_obstacle_clearance_m",
    "total_online_algorithm_compute_ms",
)


def paired_binary(
    left_name: str,
    right_name: str,
    left: Mapping[str, dict[str, str]],
    right: Mapping[str, dict[str, str]],
    scenario_ids: list[str],
    field: str,
) -> dict[str, Any]:
    both_true = left_only = right_only = both_false = 0
    for scenario_id in scenario_ids:
        left_value = flag(left[scenario_id][field])
        right_value = flag(right[scenario_id][field])
        if left_value and right_value:
            both_true += 1
        elif left_value:
            left_only += 1
        elif right_value:
            right_only += 1
        else:
            both_false += 1
    return {
        "field": field,
        "n": len(scenario_ids),
        "both_true": both_true,
        f"{left_name}_only_true": left_only,
        f"{right_name}_only_true": right_only,
        "both_false": both_false,
        "discordant_count": left_only + right_only,
        "exact_two_sided_mcnemar_p": exact_mcnemar(left_only, right_only),
        f"{left_name}_minus_{right_name}_rate_pp": 100.0 * (left_only - right_only) / len(scenario_ids),
    }


def continuous_comparison(
    left_name: str,
    right_name: str,
    left: Mapping[str, dict[str, str]],
    right: Mapping[str, dict[str, str]],
) -> dict[str, Any]:
    both_success = sorted(
        scenario_id
        for scenario_id in left
        if flag(left[scenario_id]["team_success"]) and flag(right[scenario_id]["team_success"])
    )
    metrics: dict[str, Any] = {}
    for field in CONTINUOUS_FIELDS:
        pairs = [
            (finite(left[scenario_id][field]), finite(right[scenario_id][field]))
            for scenario_id in both_success
        ]
        pairs = [(left_value, right_value) for left_value, right_value in pairs if left_value is not None and right_value is not None]
        differences = np.asarray([left_value - right_value for left_value, right_value in pairs], dtype=np.float64)
        low, high = bootstrap_mean_ci(differences)
        try:
            p_value = float(wilcoxon(differences, alternative="two-sided").pvalue) if np.any(differences != 0.0) else 1.0
        except ValueError:
            p_value = 1.0
        metrics[field] = {
            "subset": "both_success_only",
            "n": int(differences.size),
            f"{left_name}_minus_{right_name}_mean": float(np.mean(differences)) if differences.size else None,
            f"{left_name}_minus_{right_name}_median": float(np.median(differences)) if differences.size else None,
            "paired_bootstrap_95_ci": [low, high] if differences.size else None,
            "wilcoxon_two_sided_p": p_value,
        }
    return {
        "left": left_name,
        "right": right_name,
        "both_success_count": len(both_success),
        "failed_completion_times_filled_with_zero": False,
        "metrics": metrics,
    }


def main() -> None:
    dev = ARTIFACT_ROOT / "07_development"
    arms = {selector: read_csv(dev / relative) for selector, relative in ARM_PATHS.items()}
    maps = {selector: {row["scenario_id"]: row for row in rows} for selector, rows in arms.items()}
    expected_ids = set(maps["fp_shep"])
    if len(expected_ids) != 400 or any(set(index) != expected_ids for index in maps.values()):
        raise RuntimeError("four-selector scenario keys do not match exactly")
    for scenario_id in expected_ids:
        identity = tuple(maps["fp_shep"][scenario_id][field] for field in ("seed", "stage", "family", "task_pattern"))
        if any(tuple(index[scenario_id][field] for field in ("seed", "stage", "family", "task_pattern")) != identity for index in maps.values()):
            raise RuntimeError(f"four-selector identity mismatch: {scenario_id}")

    comparison_rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        for selector, rows in arms.items():
            row = aggregate(selector, scope, scope_rows(rows, scope))
            row["schema_version"] = SCHEMA_VERSION
            comparison_rows.append(row)

    paired_tests: dict[str, Any] = {}
    continuous_tests: dict[str, Any] = {}
    for left_name, right_name in COMPARISONS:
        comparison_name = f"{left_name}_vs_{right_name}"
        paired_tests[comparison_name] = {}
        for scope in SCOPES:
            scenario_ids = sorted(row["scenario_id"] for row in scope_rows(arms[left_name], scope))
            paired_tests[comparison_name][scope] = {
                field: paired_binary(
                    left_name,
                    right_name,
                    maps[left_name],
                    maps[right_name],
                    scenario_ids,
                    field,
                )
                for field in BINARY_FIELDS
            }
        continuous_tests[comparison_name] = continuous_comparison(
            left_name,
            right_name,
            maps[left_name],
            maps[right_name],
        )

    overall = {
        row["selector"]: row
        for row in comparison_rows
        if row["scope"] == "overall"
    }
    high_density = {
        row["selector"]: row
        for row in comparison_rows
        if row["scope"] == "high_density"
    }
    rs_vs_fp_smoothness = continuous_tests["gat_rs_vs_fp_shep"]["metrics"]["trajectory_smoothness"]
    rs_vs_r_smoothness = continuous_tests["gat_rs_vs_gat_r"]["metrics"]["trajectory_smoothness"]
    rs_fp_smoothness_delta = rs_vs_fp_smoothness["gat_rs_minus_fp_shep_mean"]
    rs_r_smoothness_delta = rs_vs_r_smoothness["gat_rs_minus_gat_r_mean"]
    success_gate = overall["gat_rs"]["success_rate"] > overall["fp_shep"]["success_rate"]
    safety_gate = overall["gat_rs"]["collision_rate"] <= overall["fp_shep"]["collision_rate"]
    peer_gate = overall["gat_rs"]["peer_collision_rate"] <= overall["fp_shep"]["peer_collision_rate"]
    high_density_gate = high_density["gat_rs"]["success_rate"] > high_density["fp_shep"]["success_rate"]
    smoothness_noninferior_fp = rs_fp_smoothness_delta <= 0.0
    smoothness_noninferior_r = rs_r_smoothness_delta <= 0.0
    smoothness_supervision_rejected = not (smoothness_noninferior_fp and smoothness_noninferior_r)
    dev_gate = success_gate and safety_gate and peer_gate and high_density_gate and not smoothness_supervision_rejected

    decision = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "scenario_count_per_selector": 400,
        "selector_count": 4,
        "paired_identity_match": True,
        "formal_v1_used_for_tuning": False,
        "fp_dev_success": overall["fp_shep"]["success_rate"],
        "gat_v1_dev_success": overall["gat_v1"]["success_rate"],
        "gat_r_dev_success": overall["gat_r"]["success_rate"],
        "gat_rs_dev_success": overall["gat_rs"]["success_rate"],
        "gat_r_gain_over_fp_pp": 100.0 * (overall["gat_r"]["success_rate"] - overall["fp_shep"]["success_rate"]),
        "gat_rs_gain_over_fp_pp": 100.0 * (overall["gat_rs"]["success_rate"] - overall["fp_shep"]["success_rate"]),
        "gat_rs_gain_over_gat_r_pp": 100.0 * (overall["gat_rs"]["success_rate"] - overall["gat_r"]["success_rate"]),
        "gat_rs_gain_over_gat_v1_pp": 100.0 * (overall["gat_rs"]["success_rate"] - overall["gat_v1"]["success_rate"]),
        "fp_dev_collision": overall["fp_shep"]["collision_rate"],
        "gat_rs_dev_collision": overall["gat_rs"]["collision_rate"],
        "fp_dev_peer_collision": overall["fp_shep"]["peer_collision_rate"],
        "gat_rs_dev_peer_collision": overall["gat_rs"]["peer_collision_rate"],
        "high_density_fp_success": high_density["fp_shep"]["success_rate"],
        "high_density_gat_rs_success": high_density["gat_rs"]["success_rate"],
        "high_density_gat_rs_gain_pp": 100.0 * (high_density["gat_rs"]["success_rate"] - high_density["fp_shep"]["success_rate"]),
        "gat_rs_smoothness_paired_delta_vs_fp": rs_fp_smoothness_delta,
        "gat_rs_smoothness_pvalue_vs_fp": rs_vs_fp_smoothness["wilcoxon_two_sided_p"],
        "gat_rs_smoothness_paired_delta_vs_gat_r": rs_r_smoothness_delta,
        "gat_rs_smoothness_pvalue_vs_gat_r": rs_vs_r_smoothness["wilcoxon_two_sided_p"],
        "success_gate": success_gate,
        "safety_gate": safety_gate,
        "peer_safety_gate": peer_gate,
        "high_density_gate": high_density_gate,
        "smoothness_noninferior_to_fp": smoothness_noninferior_fp,
        "smoothness_noninferior_to_gat_r": smoothness_noninferior_r,
        "smoothness_supervision_rejected": smoothness_supervision_rejected,
        "DEV_GATE": "PASS" if dev_gate else "FAIL",
        "HOLDOUT_GATE": "NOT_RUN",
        "FINAL_GAT_RS_FREEZE": "NO",
        "holdout_opened": False,
        "formal_v2_generated": False,
        "GAT_INCREMENT_UNDER_RERR": "POSITIVE" if overall["gat_rs"]["success_rate"] > overall["fp_shep"]["success_rate"] else "NEGATIVE",
        "GAT_CORE_CONTRIBUTION_SUPPORTED": "NO",
        "RECOMMENDED_NEXT_STEP": "RECONSIDER_GAT_TRAINING_OBJECTIVE",
    }
    write_csv(dev / "dev_selector_comparison.csv", comparison_rows)
    write_json(
        dev / "dev_paired_success_tests.json",
        {
            "schema_version": SCHEMA_VERSION,
            "test": "exact two-sided McNemar via paired binomial discordances",
            "comparisons": paired_tests,
        },
    )
    write_json(
        dev / "dev_both_success_continuous_tests.json",
        {
            "schema_version": SCHEMA_VERSION,
            "subset": "pair-specific both-success episodes only",
            "failed_completion_times_filled_with_zero": False,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "comparisons": continuous_tests,
        },
    )
    write_json(dev / "DEV_GATE_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
