#!/usr/bin/env python3
"""Independent final reconciliation for the TACV audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "artifacts/transient_aware_candidate_veto/20260824_184551"
ORIGINAL = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552/04_development/records/original/episode_records"
VARIANTS = ("tacv_mild", "tacv_medium", "tacv_strong")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.stem: load_json(path)
        for path in sorted(directory.glob("*.json"))
        if not path.stem.endswith("_SOFTWARE_ERROR")
    }


def main() -> None:
    checks: dict[str, bool] = {}
    original = records(ORIGINAL)
    checks["original_count_100"] = len(original) == 100
    paired_csv_path = ROOT / "06_development/TACV_DEVELOPMENT_PAIRED_SUMMARY.csv"
    with paired_csv_path.open(newline="", encoding="utf-8-sig") as handle:
        paired = {row["variant"]: row for row in csv.DictReader(handle)}
    recomputed: dict[str, Any] = {}
    for variant in VARIANTS:
        directory = ROOT / f"06_development/records/{variant}/episode_records"
        arm = records(directory)
        checks[f"{variant}_count_100"] = len(arm) == 100
        checks[f"{variant}_scenario_pairing"] = set(arm) == set(original)
        checks[f"{variant}_zero_errors"] = not list(directory.glob("*_SOFTWARE_ERROR.json"))
        ids = sorted(original)
        source = [original[sid]["episode"] for sid in ids]
        target = [arm[sid]["episode"] for sid in ids]
        both = [index for index, (a, b) in enumerate(zip(source, target)) if a["team_success"] and b["team_success"]]
        success_original = float(np.mean([bool(row["team_success"]) for row in source]))
        success_target = float(np.mean([bool(row["team_success"]) for row in target]))
        peer_delta = 100.0 * (
            float(np.mean([bool(row["inter_agent_collision"]) for row in target]))
            - float(np.mean([bool(row["inter_agent_collision"]) for row in source]))
        )
        smooth_original = float(np.mean([float(source[index]["trajectory_smoothness"]) for index in both]))
        smooth_target = float(np.mean([float(target[index]["trajectory_smoothness"]) for index in both]))
        smooth_reduction = 100.0 * (smooth_original - smooth_target) / smooth_original
        row = paired[variant]
        checks[f"{variant}_success_reproduced"] = (
            abs(success_original - float(row["original_success_rate"])) < 1.0e-12
            and abs(success_target - float(row["tacv_success_rate"])) < 1.0e-12
        )
        checks[f"{variant}_peer_delta_reproduced"] = abs(peer_delta - float(row["peer_collision_delta_pp"])) < 1.0e-12
        checks[f"{variant}_smoothness_reproduced"] = abs(smooth_reduction - float(row["smoothness_reduction_percent"])) < 1.0e-12
        recomputed[variant] = {
            "success_original": success_original,
            "success_tacv": success_target,
            "both_success_count": len(both),
            "smoothness_reduction_percent": smooth_reduction,
            "peer_collision_delta_pp": peer_delta,
        }

    selection = load_json(ROOT / "06_development/TACV_DEVELOPMENT_SELECTION.json")
    checks["no_arm_passed_development_gate"] = not any(
        bool(row["development_gate_pass"]) for row in selection["paired_results"]
    )
    checks["selection_none"] = (
        selection["selected_variant"] == "NONE" and not selection["holdout_authorized"]
    )
    checks["no_holdout_episode_records"] = not list((ROOT / "08_holdout").rglob("*_trajectory.npz"))
    checks["formal_v2_not_run"] = (
        load_json(ROOT / "TACV_FORMAL_GO_NO_GO.json")["FORMAL_V2_EXECUTED"] == "NO"
    )
    prefreeze = load_json(ROOT / "11_freeze/PRE_DEVELOPMENT_TACV_FREEZE.json")
    for relative, expected in prefreeze["source_sha256"].items():
        checks[f"frozen_source_{relative}"] = sha256(REPO_ROOT / relative) == expected
    required = (
        "01_preview_contract/FP_PREVIEW_TRANSIENT_CONTRACT.json",
        "02_predictability/PREVIEW_VS_REALIZED_TRANSIENT.csv",
        "02_predictability/PREVIEW_TRANSIENT_PREDICTABILITY.json",
        "03_replaceability/TACV_SAFETY_ADMISSIBILITY_CONTRACT.json",
        "03_replaceability/SAFE_ALTERNATIVE_REPLACEABILITY.csv",
        "03_replaceability/NECESSARY_VS_AVOIDABLE_TRANSIENTS.csv",
        "04_gate_decision/TACV_GATE_DECISION.json",
        "05_tacv_implementation/TACV_IMPLEMENTATION_CONTRACT.json",
        "06_development/TACV_DEVELOPMENT_RESULTS.csv",
        "06_development/TACV_EVENT_LEVEL_DECISIONS.csv",
        "07_failure_audit/TACV_PAIRED_FAILURE_AUDIT.csv",
        "08_holdout/TACV_HOLDOUT_RESULTS.csv",
        "08_holdout/TACV_HOLDOUT_PAIRED_STATISTICS.json",
        "09_runtime/TACV_RUNTIME_SUMMARY.json",
        "11_freeze/FINAL_TACV_FREEZE.json",
        "12_paper_ready/PAPER_TACV_DIAGNOSTIC_PARAGRAPH.md",
        "TACV_FORMAL_GO_NO_GO.json",
        "conclusion.json",
        "FINAL_REPORT.md",
    )
    for relative in required:
        checks[f"required_{relative}"] = (ROOT / relative).is_file()
    payload = {
        "schema_version": "tacv_final_independent_reconciliation_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recomputed_development": recomputed,
        "selected_variant": selection["selected_variant"],
        "new_holdout_episode_count": 0,
        "formal_v2_episode_count": 0,
        "retraining_performed": False,
    }
    target = ROOT / "INDEPENDENT_RECONCILIATION.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
