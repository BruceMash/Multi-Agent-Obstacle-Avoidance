#!/usr/bin/env python3
"""Analyze the corrected FP-anchored GAT-R/GAT-RS Dev-400 arms.

The original four-arm development results remain immutable.  This analysis
reuses the same FP-SHEP and GAT-V1 controls, substitutes only the two corrected
training arms, and applies the preregistered safety-first selection rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.analyze_gat_r_vs_fp_development import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    aggregate,
    flag,
    read_csv,
    scope_rows,
    write_csv,
    write_json,
)
from scripts.analyze_gat_recurrent_four_selector_development import (  # noqa: E402
    BINARY_FIELDS,
    CONTINUOUS_FIELDS,
    paired_binary,
    continuous_comparison,
)


SCHEMA_VERSION = "gat_recurrent_corrected_development_analysis_v1"
ARTIFACT_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
OUTPUT_ROOT = ARTIFACT_ROOT / "13_objective_revision/07_development"
ARM_PATHS = {
    "fp_shep": ARTIFACT_ROOT / "07_development/FP_SHEP_RERR_DEV400/development_team_results.csv",
    "gat_v1": ARTIFACT_ROOT / "07_development/GAT_V1_RERR_DEV400/development_team_results.csv",
    "gat_r": OUTPUT_ROOT / "GAT_R_FP_ANCHOR_DEV400/development_team_results.csv",
    "gat_rs": OUTPUT_ROOT / "GAT_RS_FP_ANCHOR_DEV400/development_team_results.csv",
}
COMPARISONS = (
    ("gat_v1", "fp_shep"),
    ("gat_r", "fp_shep"),
    ("gat_rs", "fp_shep"),
    ("gat_rs", "gat_r"),
)
SCOPES = ("overall", "stage_1", "stage_2", "stage_3", "stage_4", "high_density")


def _rate(rows: list[dict[str, str]], field: str) -> float:
    return sum(flag(row[field]) for row in rows) / len(rows)


def main() -> None:
    arms = {selector: read_csv(path) for selector, path in ARM_PATHS.items()}
    maps = {selector: {row["scenario_id"]: row for row in rows} for selector, rows in arms.items()}
    expected_ids = set(maps["fp_shep"])
    if len(expected_ids) != 400 or any(set(index) != expected_ids for index in maps.values()):
        raise RuntimeError("corrected four-selector scenario keys do not match exactly")
    for scenario_id in expected_ids:
        identity = tuple(maps["fp_shep"][scenario_id][field] for field in ("seed", "stage", "family", "task_pattern"))
        if any(tuple(index[scenario_id][field] for field in ("seed", "stage", "family", "task_pattern")) != identity for index in maps.values()):
            raise RuntimeError(f"corrected four-selector identity mismatch: {scenario_id}")

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
                field: paired_binary(left_name, right_name, maps[left_name], maps[right_name], scenario_ids, field)
                for field in BINARY_FIELDS
            }
        continuous_tests[comparison_name] = continuous_comparison(left_name, right_name, maps[left_name], maps[right_name])

    overall = {row["selector"]: row for row in comparison_rows if row["scope"] == "overall"}
    high_density = {row["selector"]: row for row in comparison_rows if row["scope"] == "high_density"}
    rs_fp_smoothness = continuous_tests["gat_rs_vs_fp_shep"]["metrics"]["trajectory_smoothness"]
    rs_r_smoothness = continuous_tests["gat_rs_vs_gat_r"]["metrics"]["trajectory_smoothness"]
    rs_fp_delta = rs_fp_smoothness["gat_rs_minus_fp_shep_mean"]
    rs_r_delta = rs_r_smoothness["gat_rs_minus_gat_r_mean"]

    selector_gates: dict[str, dict[str, Any]] = {}
    for selector in ("gat_r", "gat_rs"):
        success_gain_pp = 100.0 * (overall[selector]["success_rate"] - overall["fp_shep"]["success_rate"])
        high_density_gain_pp = 100.0 * (high_density[selector]["success_rate"] - high_density["fp_shep"]["success_rate"])
        selector_gates[selector] = {
            "success_strictly_greater_than_fp": overall[selector]["success_rate"] > overall["fp_shep"]["success_rate"],
            "collision_noninferior_to_fp": overall[selector]["collision_rate"] <= overall["fp_shep"]["collision_rate"],
            "peer_collision_noninferior_to_fp": overall[selector]["peer_collision_rate"] <= overall["fp_shep"]["peer_collision_rate"],
            "high_density_success_strictly_greater_than_fp": high_density[selector]["success_rate"] > high_density["fp_shep"]["success_rate"],
            "recommended_overall_gain_at_least_2pp": success_gain_pp >= 2.0,
            "recommended_high_density_gain_at_least_3pp": high_density_gain_pp >= 3.0,
            "success_gain_over_fp_pp": success_gain_pp,
            "high_density_gain_over_fp_pp": high_density_gain_pp,
        }
        selector_gates[selector]["hard_gate"] = all(
            selector_gates[selector][key]
            for key in (
                "success_strictly_greater_than_fp",
                "collision_noninferior_to_fp",
                "peer_collision_noninferior_to_fp",
                "high_density_success_strictly_greater_than_fp",
            )
        )

    rs_smoothness_gate = bool(rs_fp_delta <= 0.0 and rs_r_delta <= 0.0)
    smoothness_supervision_rejected = not rs_smoothness_gate
    selected_selector = "gat_r" if smoothness_supervision_rejected else "gat_rs"
    selected_gate = bool(selector_gates[selected_selector]["hard_gate"])

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
        "fp_dev_collision": overall["fp_shep"]["collision_rate"],
        "gat_r_dev_collision": overall["gat_r"]["collision_rate"],
        "gat_rs_dev_collision": overall["gat_rs"]["collision_rate"],
        "fp_dev_peer_collision": overall["fp_shep"]["peer_collision_rate"],
        "gat_r_dev_peer_collision": overall["gat_r"]["peer_collision_rate"],
        "gat_rs_dev_peer_collision": overall["gat_rs"]["peer_collision_rate"],
        "gat_rs_smoothness_paired_delta_vs_fp": rs_fp_delta,
        "gat_rs_smoothness_pvalue_vs_fp": rs_fp_smoothness["wilcoxon_two_sided_p"],
        "gat_rs_smoothness_paired_delta_vs_gat_r": rs_r_delta,
        "gat_rs_smoothness_pvalue_vs_gat_r": rs_r_smoothness["wilcoxon_two_sided_p"],
        "lower_smoothness_is_better": True,
        "gat_rs_smoothness_gate": rs_smoothness_gate,
        "SMOOTHNESS_SUPERVISION_REJECTED": "YES" if smoothness_supervision_rejected else "NO",
        "selected_selector_for_holdout": selected_selector,
        "selector_gates": selector_gates,
        "DEV_GATE": "PASS" if selected_gate else "FAIL",
        "HOLDOUT_OPEN_AUTHORIZED": selected_gate,
        "HOLDOUT_GATE": "NOT_RUN",
        "FINAL_GAT_RS_FREEZE": "NO",
        "formal_v2_generated": False,
        "GAT_INCREMENT_UNDER_RERR": "POSITIVE" if selected_gate else "NOT_ESTABLISHED",
        "GAT_CORE_CONTRIBUTION_SUPPORTED": "NOT_YET_HOLDOUT_VALIDATED",
        "RECOMMENDED_NEXT_STEP": "RUN_SEALED_HOLDOUT_WITH_GAT_R" if selected_gate and selected_selector == "gat_r" else "STOP_BEFORE_HOLDOUT",
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "dev_selector_comparison.csv", comparison_rows)
    write_json(
        OUTPUT_ROOT / "dev_paired_success_tests.json",
        {
            "schema_version": SCHEMA_VERSION,
            "test": "exact two-sided McNemar via paired binomial discordances",
            "comparisons": paired_tests,
        },
    )
    write_json(
        OUTPUT_ROOT / "dev_both_success_continuous_tests.json",
        {
            "schema_version": SCHEMA_VERSION,
            "subset": "pair-specific both-success episodes only",
            "failed_completion_times_filled_with_zero": False,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "continuous_fields": list(CONTINUOUS_FIELDS),
            "comparisons": continuous_tests,
        },
    )
    write_json(OUTPUT_ROOT / "DEV_GATE_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
