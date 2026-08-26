#!/usr/bin/env python3
"""Independent artifact and stop-rule reconciliation for the parallel study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "artifacts/parallel_zigzag_resolution/20260826_091334"


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def yes(value: Any) -> bool:
    return value is True or str(value).upper() in {"YES", "PASS", "TRUE"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    common = load(root / "00_context/COMMON_EXPERIMENT_FREEZE.json")
    for item in common["authority_sources"]:
        source = REPO_ROOT / item["path"]
        check(f"authority_hash:{item['path']}", source.is_file() and sha256(source) == item["sha256"], item["sha256"])

    a_dir = root / "track_A_motion_proposal"
    b_dir = root / "track_B_learned_turn"
    c_dir = root / "track_C_visualization"
    a_required = [
        "MOTION_ALIGNED_PROPOSAL_THRESHOLD.json",
        "MOTION_PROPOSAL_CONTRACT.json",
        "MOTION_PROPOSAL_DEV_RESULTS.csv",
        "MOTION_PROPOSAL_DEV_RAW_TRAJECTORIES.pdf",
        "TRACK_A_CONCLUSION.json",
    ]
    b_required = [
        "TURN_OBSERVATION_EXTENSION.json",
        "ZERO_INIT_EQUIVALENCE.json",
        "TARGET_ENCODER_FREEZE_AUDIT.json",
        "TURN_TEACHER_CONTRACT.json",
        "TURN_CONTINUITY_REWARD.json",
        "TURN_FINETUNE_CONFIG.json",
        "TURN_TRAINING_HISTORY.csv",
        "TURN_DEV_RESULTS.csv",
        "TURN_DEV_RAW_TRAJECTORIES.pdf",
        "TRACK_B_CONCLUSION.json",
    ]
    c_required = [
        "PAPER_VISUAL_STYLE_CONTRACT.json",
        "METHOD_VISUAL_CONTRACT.json",
        "OBSTACLE_VISUAL_CONTRACT.json",
        "CAMERA_PRESETS.json",
        "AXIS_SCALE_CONTRACT.json",
        "FIGURE_DATA_INTEGRITY_AUDIT.csv",
        "PDF_RENDER_VALIDATION.csv",
        "figure_manifest.csv",
        "VISUALIZATION_GUIDE.md",
        "TRACK_C_CONCLUSION.json",
    ]
    for directory, required, label in ((a_dir, a_required, "A"), (b_dir, b_required, "B"), (c_dir, c_required, "C")):
        for name in required:
            check(f"required_{label}:{name}", (directory / name).is_file())

    a = load(a_dir / "TRACK_A_CONCLUSION.json")
    b = load(b_dir / "TRACK_B_CONCLUSION.json")
    c = load(c_dir / "TRACK_C_CONCLUSION.json")
    a_accepted = yes(a.get("TRACK_A_ACCEPTED", "NO"))
    b_accepted = yes(b.get("TRACK_B_ACCEPTED", "NO"))
    check("track_a_no_formal", not yes(a.get("FORMAL_EXECUTED", a.get("FORMAL_V2_EXECUTED", "NO"))))
    check("track_b_no_formal", not yes(b.get("FORMAL_EXECUTED", b.get("FORMAL_V2_EXECUTED", "NO"))))
    check("track_c_no_scientific_change", not yes(c.get("SCIENTIFIC_RESULTS_CHANGED", "NO")))
    check("track_b_zero_init", yes(b.get("ZERO_INIT_EQUIVALENCE", "FAIL")))
    drift = b.get("sensor_encoder_drift", b.get("TRACK_B_SENSOR_ENCODER_DRIFT", 1.0))
    if isinstance(drift, dict):
        micro = drift.get("pretraining_mechanism_microtest", {})
        drift_ok = (
            isinstance(micro, dict)
            and set(micro) == {"actor", "online_critic", "target_critic"}
            and all(float(value) == 0.0 for value in micro.values())
            and str(drift.get("post_training", "")).upper() == "NOT_RUN"
        )
    else:
        drift_ok = float(drift) == 0.0
    check("track_b_encoder_drift", drift_ok, drift)

    if a_accepted:
        for name in ("FINAL_MOTION_PROPOSAL_FREEZE.json", "MOTION_PROPOSAL_HOLDOUT_RESULTS.csv", "MOTION_PROPOSAL_HOLDOUT_RAW_TRAJECTORIES.pdf"):
            check(f"accepted_A:{name}", (a_dir / name).is_file())
        check("accepted_A_holdout_pass", yes(a.get("HOLDOUT_GATE", "FAIL")))
    else:
        check("rejected_A_not_accepted", True)
    if b_accepted:
        for name in ("FINAL_TURN_POLICY_FREEZE.json", "TURN_HOLDOUT_RESULTS.csv", "TURN_HOLDOUT_RAW_TRAJECTORIES.pdf"):
            check(f"accepted_B:{name}", (b_dir / name).is_file())
        check("accepted_B_holdout_pass", yes(b.get("HOLDOUT_GATE", "FAIL")))
    else:
        check("rejected_B_not_accepted", True)

    combined = root / "optional_combined/COMBINATION_CONCLUSION.json"
    combination_executed = False
    if combined.is_file():
        combination_executed = yes(load(combined).get("COMBINATION_EXECUTED", "NO"))
    check("combination_authorization_rule", (not combination_executed) or (a_accepted and b_accepted))

    integrity_rows = rows(c_dir / "FIGURE_DATA_INTEGRITY_AUDIT.csv")
    check("figure_integrity_nonempty", bool(integrity_rows), len(integrity_rows))
    check(
        "figure_raw_trajectory_only",
        bool(integrity_rows) and all(yes(row.get("raw_trajectory_used", "NO")) for row in integrity_rows),
    )
    check(
        "figure_no_post_processing",
        bool(integrity_rows) and all(not yes(row.get("post_processing_applied", "YES")) for row in integrity_rows),
    )
    render_rows = rows(c_dir / "PDF_RENDER_VALIDATION.csv")
    check("pdf_render_validation_nonempty", bool(render_rows), len(render_rows))
    check(
        "pdf_render_validation_pass",
        bool(render_rows)
        and all(
            yes(row.get("status", row.get("validation", row.get("visual_inspection", "FAIL"))))
            for row in render_rows
        ),
    )

    # Require the independently authored, track-specific audits in the final
    # closure rather than treating the local track reconciliations as sufficient.
    for label in ("A", "B", "C"):
        audit_path = root / f"final_selection/INDEPENDENT_TRACK_{label}_AUDIT.json"
        check(f"independent_track_{label.lower()}_audit_exists", audit_path.is_file())
        if audit_path.is_file():
            audit = load(audit_path)
            audit_status = audit.get("overall_status", audit.get("status", "FAIL"))
            blocking = audit.get("blocking_failures", audit.get("blocking_failure_count", 0))
            if isinstance(blocking, list):
                blocking_ok = len(blocking) == 0
            else:
                try:
                    blocking_ok = int(blocking) == 0
                except (TypeError, ValueError):
                    blocking_ok = False
            check(
                f"independent_track_{label.lower()}_audit_pass",
                yes(audit_status) and blocking_ok,
                {"status": audit_status, "blocking_failures": blocking},
            )

    failures = [item for item in checks if not item["passed"]]
    result = {
        "schema_version": "parallel_zigzag_independent_reconciliation_v1",
        "status": "PASS" if not failures else "FAIL",
        "check_count": len(checks),
        "failed_check_count": len(failures),
        "failed_checks": failures,
        "checks": checks,
        "track_a_accepted": a_accepted,
        "track_b_accepted": b_accepted,
        "formal_v2_executed": False,
        "original_formal_success": 0.9525,
    }
    for name in ("independent_reconciliation.json", "final_reconciliation.json"):
        with (root / "final_selection" / name).open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    print(json.dumps({"status": result["status"], "failed_checks": failures}, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
