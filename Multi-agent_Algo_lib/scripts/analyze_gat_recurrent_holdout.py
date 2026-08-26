#!/usr/bin/env python3
"""Analyze the one-shot sealed Holdout comparison for selected GAT-R vs FP."""

from __future__ import annotations

import json
import shutil
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
    read_csv,
    scope_rows,
    write_csv,
    write_json,
)
from scripts.analyze_gat_recurrent_four_selector_development import (  # noqa: E402
    BINARY_FIELDS,
    CONTINUOUS_FIELDS,
    continuous_comparison,
    paired_binary,
)


SCHEMA_VERSION = "gat_recurrent_holdout_analysis_v1"
ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
REVISION = ROOT / "13_objective_revision"
OUTPUT = REVISION / "08_holdout"
REQUIRED_OUTPUT = ROOT / "08_holdout"
ARM_PATHS = {
    "fp_shep": OUTPUT / "FP_SHEP_HOLDOUT400/holdout_team_results.csv",
    "gat_r": OUTPUT / "GAT_R_FP_ANCHOR_HOLDOUT400/holdout_team_results.csv",
}
SCOPES = ("overall", "stage_1", "stage_2", "stage_3", "stage_4", "high_density")


def main() -> None:
    arms = {selector: read_csv(path) for selector, path in ARM_PATHS.items()}
    maps = {selector: {row["scenario_id"]: row for row in rows} for selector, rows in arms.items()}
    expected_ids = set(maps["fp_shep"])
    if len(expected_ids) != 400 or set(maps["gat_r"]) != expected_ids:
        raise RuntimeError("Holdout scenario keys do not match exactly")
    for scenario_id in expected_ids:
        left = maps["fp_shep"][scenario_id]
        right = maps["gat_r"][scenario_id]
        identity_fields = ("seed", "stage", "family", "task_pattern")
        if tuple(left[field] for field in identity_fields) != tuple(right[field] for field in identity_fields):
            raise RuntimeError(f"Holdout identity mismatch: {scenario_id}")

    comparison_rows: list[dict[str, Any]] = []
    paired: dict[str, Any] = {}
    for scope in SCOPES:
        for selector, rows in arms.items():
            row = aggregate(selector, scope, scope_rows(rows, scope))
            row["schema_version"] = SCHEMA_VERSION
            comparison_rows.append(row)
        scenario_ids = sorted(row["scenario_id"] for row in scope_rows(arms["gat_r"], scope))
        paired[scope] = {
            field: paired_binary("gat_r", "fp_shep", maps["gat_r"], maps["fp_shep"], scenario_ids, field)
            for field in BINARY_FIELDS
        }

    continuous = continuous_comparison("gat_r", "fp_shep", maps["gat_r"], maps["fp_shep"])
    overall = {row["selector"]: row for row in comparison_rows if row["scope"] == "overall"}
    high_density = {row["selector"]: row for row in comparison_rows if row["scope"] == "high_density"}
    smoothness = continuous["metrics"]["trajectory_smoothness"]
    smoothness_delta = smoothness["gat_r_minus_fp_shep_mean"]
    success_gate = overall["gat_r"]["success_rate"] > overall["fp_shep"]["success_rate"]
    collision_gate = overall["gat_r"]["collision_rate"] <= overall["fp_shep"]["collision_rate"]
    peer_gate = overall["gat_r"]["peer_collision_rate"] <= overall["fp_shep"]["peer_collision_rate"]
    high_density_gate = high_density["gat_r"]["success_rate"] > high_density["fp_shep"]["success_rate"]
    smoothness_noninferior = smoothness_delta <= 0.0
    hard_gate = success_gate and collision_gate and peer_gate and high_density_gate

    tests_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "selector": "gat_r",
        "reference": "fp_shep",
        "test": "exact two-sided McNemar via paired binomial discordances",
        "paired_binary": paired,
        "continuous_both_success": continuous,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "failed_completion_times_filled_with_zero": False,
        "success_gate": success_gate,
        "collision_gate": collision_gate,
        "peer_collision_gate": peer_gate,
        "high_density_success_gate": high_density_gate,
        "smoothness_noninferior_to_fp": smoothness_noninferior,
        "smoothness_objective_status": "REJECTED_AT_DEV" if not smoothness_noninferior else "NONINFERIOR",
        "HOLDOUT_GATE": "PASS" if hard_gate else "FAIL",
        "FORMAL_V2_OPEN_AUTHORIZED": hard_gate,
        "gate_interpretation": (
            "The selected checkpoint is GAT-R after the preregistered Dev rejection of GAT-RS; "
            "therefore Formal authorization uses the success/collision/peer/high-density hard gates. "
            "Smoothness remains reported but cannot restore the rejected smoothness objective."
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison_path = OUTPUT / "holdout_selector_comparison.csv"
    tests_path = OUTPUT / "holdout_paired_tests.json"
    write_csv(comparison_path, comparison_rows)
    write_json(tests_path, tests_payload)
    # Update the originally requested canonical artifact locations only after
    # the sealed comparison is complete and reconciled.
    shutil.copy2(comparison_path, REQUIRED_OUTPUT / "holdout_selector_comparison.csv")
    shutil.copy2(tests_path, REQUIRED_OUTPUT / "holdout_paired_tests.json")
    print(json.dumps(tests_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
