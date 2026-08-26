#!/usr/bin/env python3
"""Independent raw-artifact reconciliation for the Track-B hard-stop branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts/parallel_zigzag_resolution/20260826_091334/track_B_learned_turn"
COMMON = REPO_ROOT / "artifacts/parallel_zigzag_resolution/20260826_091334/00_context/COMMON_EXPERIMENT_FREEZE.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    required = [
        "TURN_OBSERVATION_EXTENSION.json",
        "ZERO_INIT_EQUIVALENCE.json",
        "TARGET_ENCODER_FREEZE_AUDIT.json",
        "TURN_SMOOTH_TEACHER_ACTION_RECOVERABILITY.json",
        "TURN_TEACHER_CONTRACT.json",
        "TURN_CONTINUITY_REWARD.json",
        "TURN_FINETUNE_CONFIG.json",
        "TURN_TRAINING_HISTORY.csv",
        "TURN_DEV_RESULTS.csv",
        "TURN_DEV_RAW_TRAJECTORIES.pdf",
        "TRACK_B_CONCLUSION.json",
        "FINAL_REPORT.md",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    common = load(COMMON)
    checkpoint = REPO_ROOT / common["checkpoints"]["sac_dmp"]["path"]
    checkpoint_hash_ok = sha256(checkpoint) == common["checkpoints"]["sac_dmp"]["sha256"]
    zero = load(ROOT / "ZERO_INIT_EQUIVALENCE.json")
    encoder = load(ROOT / "TARGET_ENCODER_FREEZE_AUDIT.json")
    teacher = load(ROOT / "TURN_SMOOTH_TEACHER_ACTION_RECOVERABILITY.json")
    conclusion = load(ROOT / "TRACK_B_CONCLUSION.json")
    checkpoints = sorted(str(path.relative_to(ROOT)) for path in ROOT.rglob("*.pt"))
    dev_csv = (ROOT / "TURN_DEV_RESULTS.csv").read_text(encoding="utf-8-sig")
    history_csv = (ROOT / "TURN_TRAINING_HISTORY.csv").read_text(encoding="utf-8-sig")
    checks = {
        "required_files_present": not missing,
        "common_checkpoint_hash_match": checkpoint_hash_ok,
        "zero_init_pass": zero["ZERO_INIT_EQUIVALENCE"] == "PASS",
        "encoder_freeze_mechanism_pass": encoder["pretraining_freeze_mechanism_test"] == "PASS",
        "teacher_gate_failed": teacher["TEACHER_ACTION_CONTRACT_VALID"] == "NO",
        "training_unauthorized": teacher["TRAINING_AUTHORIZED"] == "NO",
        "no_training_checkpoint_created": not checkpoints,
        "training_history_not_run": "NOT_RUN" in history_csv,
        "development_not_run": "NOT_RUN" in dev_csv,
        "holdout_not_run": conclusion["HOLDOUT_EXECUTED"] == "NO",
        "formal_not_run": conclusion["FORMAL_EXECUTED"] is False,
        "track_b_rejected": conclusion["TRACK_B_ACCEPTED"] == "NO",
        "formal_result_unchanged": conclusion["FORMAL_RESULT_CHANGED"] == "NO",
    }
    payload = {
        "schema_version": "track_b_independent_reconciliation_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing_files": missing,
        "training_checkpoints_found": checkpoints,
        "artifact_hashes": {
            name: sha256(ROOT / name)
            for name in required
            if (ROOT / name).exists()
        },
    }
    write(ROOT / "independent_reconciliation.json", payload)
    write(ROOT / "final_reconciliation.json", payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
