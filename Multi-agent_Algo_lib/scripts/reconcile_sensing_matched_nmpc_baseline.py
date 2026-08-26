"""Independent raw-record reconciliation for the SM-NMPC-style goal."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/sensing_matched_nmpc_baseline/20260819_211048"


def load_json(name: str) -> Any:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def main() -> None:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, observed: Any) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "observed": observed})

    dev_manifest = load_json("development_scenario_manifest.json")
    val_manifest = load_json("validation_scenario_manifest.json")
    dev = read_csv("development_results.csv")
    val = read_csv("validation_episode_results.csv")
    paired = read_csv("paired_validation.csv")
    runtime = read_csv("nmpc_validation_runtime.csv")
    freeze = load_json("NMPC_BASELINE_FREEZE.json")
    conclusion = load_json("conclusion.json")
    separation = load_json("scenario_separation_audit.json")

    check("development_manifest_count", len(dev_manifest["entries"]) == 32, len(dev_manifest["entries"]))
    check("validation_manifest_count", len(val_manifest["entries"]) == 40, len(val_manifest["entries"]))
    check("development_row_count", len(dev) == 256, len(dev))
    dev_keys = {(row["config_id"], row["scenario_id"], int(row["seed"])) for row in dev}
    check("development_unique_keys", len(dev_keys) == 256, len(dev_keys))
    config_counts = Counter(row["config_id"] for row in dev)
    check("eight_configs_32_each", len(config_counts) == 8 and set(config_counts.values()) == {32}, dict(config_counts))
    check("validation_row_count", len(val) == 80, len(val))
    arm_counts = Counter(row["validation_arm"] for row in val)
    check("validation_two_arms_40_each", arm_counts == {"SM_NMPC": 40, "PROPOSED": 40}, dict(arm_counts))
    check("paired_row_count", len(paired) == 40, len(paired))
    pair_keys = {(row["scenario_id"], int(row["seed"])) for row in paired}
    check("paired_unique_keys", len(pair_keys) == 40, len(pair_keys))
    manifest_keys = {(row["scenario_id"], int(row["seed"])) for row in val_manifest["entries"]}
    check("paired_manifest_exact_match", pair_keys == manifest_keys, len(pair_keys ^ manifest_keys))
    check("scenario_separation", separation["status"] == "PASS" and all(value == 0 for key, value in separation.items() if key.endswith("overlap")), separation)

    nmpc = [row for row in val if row["validation_arm"] == "SM_NMPC"]
    proposed = [row for row in val if row["validation_arm"] == "PROPOSED"]
    nmpc_success = sum(as_bool(row["success"]) for row in nmpc)
    nmpc_collision = sum(as_bool(row["collision"]) for row in nmpc)
    nmpc_timeout = sum(as_bool(row["timeout"]) for row in nmpc)
    proposed_success = sum(as_bool(row["success"]) for row in proposed)
    proposed_collision = sum(as_bool(row["collision"]) for row in proposed)
    proposed_timeout = sum(as_bool(row["timeout"]) for row in proposed)
    check("nmpc_outcomes_reproduced", (nmpc_success, nmpc_collision, nmpc_timeout) == (9, 31, 0), [nmpc_success, nmpc_collision, nmpc_timeout])
    check("proposed_outcomes_reproduced", (proposed_success, proposed_collision, proposed_timeout) == (32, 3, 5), [proposed_success, proposed_collision, proposed_timeout])
    stage_success = {
        stage: sum(as_bool(row["success"]) for row in nmpc if row["stage"] == stage)
        for stage in ("stage_1", "stage_2", "stage_3", "stage_4")
    }
    check("nmpc_stage_success_reproduced", stage_success == {"stage_1": 4, "stage_2": 4, "stage_3": 1, "stage_4": 0}, stage_success)

    times = np.asarray([float(row["runtime_ms"]) for row in runtime], dtype=float)
    check("runtime_decision_count", len(times) == 515, len(times))
    check("runtime_finite_positive", bool(np.all(np.isfinite(times)) and np.all(times > 0.0)), [float(np.min(times)), float(np.max(times))])
    check("runtime_mean_reproduced", math.isclose(float(np.mean(times)), float(conclusion["NMPC_MEAN_DECISION_LATENCY_MS"]), rel_tol=0.0, abs_tol=1e-12), float(np.mean(times)))
    check("runtime_p95_reproduced", math.isclose(float(np.percentile(times, 95)), float(conclusion["NMPC_P95_DECISION_LATENCY_MS"]), rel_tol=0.0, abs_tol=1e-12), float(np.percentile(times, 95)))
    check("deadline_miss_reproduced", float(np.mean(times > 100.0)) == float(conclusion["NMPC_DEADLINE_MISS_RATE"]) == 0.0, int(np.sum(times > 100.0)))
    check("zero_fallback", sum(as_bool(row["fallback_used"]) for row in runtime) == 0, sum(as_bool(row["fallback_used"]) for row in runtime))
    check("finite_objectives", all(row["objective_final"] not in {"", "NaN", "Infinity", "-Infinity"} for row in runtime), len(runtime))

    source_hashes = freeze["source_hashes"]
    observed_hashes = {path: sha256(ROOT / path) for path in source_hashes}
    check("frozen_source_hashes", observed_hashes == source_hashes, {path: observed_hashes[path] == source_hashes[path] for path in source_hashes})
    check("selected_config", freeze["selected_config_id"] == "N07" and freeze["selected_config"]["horizon_steps"] == 10, freeze["selected_config_id"])
    check("validation_not_seen_at_freeze", freeze["validation_results_observed_at_freeze"] is False, freeze["validation_results_observed_at_freeze"])
    check("unit_tests", load_json("unit_tests.json")["status"] == "PASS" and load_json("unit_tests.json")["test_count"] >= 15, load_json("unit_tests.json")["test_count"])
    check("information_equivalence", load_json("nmpc_information_equivalence_tests.json")["HIDDEN_STATE_LEAKAGE"] == "NO", load_json("nmpc_information_equivalence_tests.json")["status"])

    required_fields = {
        "NMPC_IMPLEMENTED",
        "NMPC_METHOD_LABEL",
        "NMPC_USES_FULL_STATE",
        "NMPC_USES_HIDDEN_STATIC_GEOMETRY",
        "NMPC_USES_EXACT_DYNAMIC_VELOCITY",
        "NMPC_USES_FUTURE_DYNAMIC_STATE",
        "NMPC_USES_CONTINUOUS_EXACT_PEER_STATE",
        "NMPC_INFORMATION_CONTRACT_MATCH",
        "NMPC_DYNAMICS_CONTRACT_MATCH",
        "NMPC_SOLVER",
        "NMPC_HORIZON_STEPS",
        "NMPC_HORIZON_SECONDS",
        "NMPC_UPDATE_FREQUENCY",
        "NMPC_CONFIGS_EVALUATED",
        "NMPC_ENGINEERING_PHASE_CLOSED",
        "NMPC_VALIDATION_SUCCESS",
        "NMPC_VALIDATION_STAGE1_SUCCESS",
        "NMPC_VALIDATION_STAGE2_SUCCESS",
        "NMPC_VALIDATION_STAGE3_SUCCESS",
        "NMPC_VALIDATION_STAGE4_SUCCESS",
        "NMPC_VALIDATION_COLLISION",
        "NMPC_VALIDATION_TIMEOUT",
        "NMPC_MEAN_DECISION_LATENCY_MS",
        "NMPC_P95_DECISION_LATENCY_MS",
        "NMPC_DEADLINE_MISS_RATE",
        "NMPC_TOTAL_COMPUTE_MS",
        "REALTIME_FEASIBLE",
        "NMPC_SOFTWARE_FAILURE_COUNT",
        "NMPC_PHYSICAL_CONSTRAINT_VIOLATIONS",
        "NMPC_BASELINE_READY",
        "PROPOSED_CHANGED",
        "NEW_FINAL_BENCHMARK_RUN",
        "FINAL_PROTOCOL_AMENDMENT_NMPC",
        "RECOMMENDED_NEXT_STEP",
    }
    check("mandatory_conclusion_fields", not (required_fields - set(conclusion)), sorted(required_fields - set(conclusion)))
    check("forced_stop_decision", conclusion["NMPC_BASELINE_READY"] == "NO" and conclusion["FINAL_PROTOCOL_AMENDMENT_NMPC"] == "NOT_APPLIED" and conclusion["RECOMMENDED_NEXT_STEP"] == "PROCEED_WITHOUT_NMPC", {key: conclusion[key] for key in ("NMPC_BASELINE_READY", "FINAL_PROTOCOL_AMENDMENT_NMPC", "RECOMMENDED_NEXT_STEP")})
    check("scope_integrity", conclusion["PROPOSED_CHANGED"] == "NO" and conclusion["NEW_FINAL_BENCHMARK_RUN"] == "NO", [conclusion["PROPOSED_CHANGED"], conclusion["NEW_FINAL_BENCHMARK_RUN"]])

    failed = [row for row in checks if row["status"] != "PASS"]
    result = {
        "schema_version": "sensing_matched_nmpc_independent_reconciliation_v1",
        "status": "PASS" if not failed else "FAIL",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "raw_reproduced": {
            "nmpc": {"success": nmpc_success, "collision": nmpc_collision, "timeout": nmpc_timeout},
            "proposed": {"success": proposed_success, "collision": proposed_collision, "timeout": proposed_timeout},
            "runtime_decisions": len(times),
            "runtime_mean_ms": float(np.mean(times)),
            "runtime_p95_ms": float(np.percentile(times, 95)),
        },
    }
    for name in ("independent_reconciliation.json", "final_reconciliation.json"):
        (OUTPUT / name).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError(failed)
    print(json.dumps({"status": result["status"], "checks": len(checks)}))


if __name__ == "__main__":
    main()
