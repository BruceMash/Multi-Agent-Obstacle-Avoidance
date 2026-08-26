#!/usr/bin/env python3
"""Independent second-pass verifier for the Formal V2 runtime audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2"
METHODS = (
    "M1_DWA_FullState",
    "M2_DWA_SensingMatched",
    "M8_RERR_FP_SHEP_SAC_DMP",
    "M9_Proposed_RERR_GAT_SAC_DMP",
)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit = args.audit.resolve()

    raw = csv_rows(FORMAL_ROOT / "formal_v2_team_results.csv")
    conclusion = json.loads((audit / "conclusion.json").read_text(encoding="utf-8"))
    step_table = {row["method_id"]: row for row in csv_rows(audit / "compute_per_control_step.csv")}
    delta_rows = csv_rows(audit / "fp_vs_gat_runtime_delta.csv")
    event_rows = csv_rows(audit / "rerr_event_latency_summary.csv")
    reconciliation_path = audit / "final_reconciliation.json"
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))

    checks: dict[str, bool] = {}
    means: dict[str, float] = {}
    for method_id in METHODS:
        subset = [row for row in raw if row["method_id"] == method_id]
        checks[f"{method_id}_row_count_400"] = len(subset) == 400
        mean_compute = sum(float(row["total_online_algorithm_compute_ms"]) for row in subset) / len(subset)
        mean_ratio = sum(
            float(row["total_online_algorithm_compute_ms"]) / float(row["steps"])
            for row in subset
        ) / len(subset)
        means[method_id] = mean_compute
        checks[f"{method_id}_mean_compute"] = close(
            mean_compute, float(step_table[method_id]["mean_online_compute_ms_per_episode"])
        )
        checks[f"{method_id}_mean_episode_ratio"] = close(
            mean_ratio, float(step_table[method_id]["mean_of_episode_compute_ms_per_step"])
        )

    proposed = [row for row in raw if row["method_id"] == "M9_Proposed_RERR_GAT_SAC_DMP"]
    proposed_success = [row for row in proposed if row["team_success"].lower() == "true"]
    proposed_failure = [row for row in proposed if row["team_success"].lower() == "false"]
    checks["proposed_mean_steps"] = close(
        sum(float(row["steps"]) for row in proposed) / len(proposed),
        float(conclusion["PROPOSED_MEAN_STEPS_PER_EPISODE"]),
    )
    checks["proposed_success_steps"] = close(
        sum(float(row["steps"]) for row in proposed_success) / len(proposed_success),
        float(conclusion["PROPOSED_MEAN_SUCCESS_STEPS"]),
    )
    checks["proposed_failure_steps"] = close(
        sum(float(row["steps"]) for row in proposed_failure) / len(proposed_failure),
        float(conclusion["PROPOSED_MEAN_FAILURE_STEPS"]),
    )
    total_delta = means["M9_Proposed_RERR_GAT_SAC_DMP"] - means["M8_RERR_FP_SHEP_SAC_DMP"]
    contribution_sum = sum(
        float(row["contribution_ms_per_episode"])
        for row in delta_rows
        if row["component"] != "TOTAL_OBSERVED_DELTA"
    )
    checks["delta_reconstruction"] = close(total_delta, contribution_sum)
    checks["conclusion_delta"] = close(
        total_delta, float(conclusion["GAT_EXTRA_COMPUTE_PER_EPISODE_MS"])
    )

    proposed_event = next(
        row
        for row in event_rows
        if row["method_id"] == "M9_Proposed_RERR_GAT_SAC_DMP"
        and row["event_class"] == "ALL_UPPER_RECONSTRUCTION_EVENTS"
    )
    checks["event_p95_conclusion"] = close(
        float(proposed_event["p95_ms"]), float(conclusion["PROPOSED_RERR_EVENT_P95_MS"])
    )
    checks["event_p99_conclusion"] = close(
        float(proposed_event["p99_ms"]), float(conclusion["PROPOSED_RERR_EVENT_P99_MS"])
    )
    checks["true_step_latency_not_fabricated"] = (
        conclusion["TRUE_PER_STEP_LATENCY_AVAILABLE"] == "NO"
        and not (audit / "per_step_latency_summary.csv").exists()
    )
    checks["no_formal_rerun"] = conclusion["FORMAL_V2_RERUN"] == "NO"

    independent = {
        "verification_mode": "INDEPENDENT_SECOND_PASS_RAW_CSV_REDUCTION",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "independently_recomputed_primary_means_ms": means,
        "independently_recomputed_gat_minus_fp_delta_ms": total_delta,
    }
    reconciliation["independent_second_pass"] = independent
    reconciliation["status"] = (
        "PASS"
        if reconciliation.get("status") == "PASS" and independent["status"] == "PASS"
        else "FAIL"
    )
    reconciliation_path.write_text(
        json.dumps(reconciliation, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if reconciliation["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"independent verification failed: {failed}")


if __name__ == "__main__":
    main()
