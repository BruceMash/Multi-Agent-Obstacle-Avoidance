#!/usr/bin/env python3
"""Freeze the corrected Dev-selected GAT checkpoint before Holdout is opened."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
REVISION = ROOT / "13_objective_revision"
DEV_DECISION = REVISION / "07_development/DEV_GATE_DECISION.json"
HOLDOUT_MANIFEST = ROOT / "08_holdout/GAT_RS_HOLDOUT_SCENE_MANIFEST.json"
HOLDOUT_VALIDATION = ROOT / "08_holdout/GAT_RS_HOLDOUT_SCENE_MANIFEST_validation.json"
HOLDOUT_CONFIG = REPO_ROOT / "configs/evaluation/gat_recurrent_r_fp_anchor_holdout.json"
CHECKPOINT = REVISION / "05_gat_r_fp_anchor_training/checkpoints/best_validation.pt"
TRAINING_MANIFEST = REVISION / "05_gat_r_fp_anchor_training/GAT_R_TRAINING_MANIFEST.json"
TARGET_CONTRACT = REVISION / "dataset_v2/FP_ANCHORED_TARGET_CONTRACT_FREEZE.json"
OUTPUT = REVISION / "09_final_freeze/PRE_HOLDOUT_GAT_R_FREEZE.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    decision = load_json(DEV_DECISION)
    validation = load_json(HOLDOUT_VALIDATION)
    config = load_json(HOLDOUT_CONFIG)
    if decision["DEV_GATE"] != "PASS" or decision["selected_selector_for_holdout"] != "gat_r":
        raise RuntimeError("corrected Dev decision does not authorize GAT-R Holdout")
    if decision["SMOOTHNESS_SUPERVISION_REJECTED"] != "YES":
        raise RuntimeError("GAT-RS rejection was not frozen")
    if validation["status"] != "PASS" or int(validation["performance_result_count_at_freeze"]) != 0:
        raise RuntimeError("Holdout split was not cleanly frozen before performance")
    checkpoint_hash = sha256_file(CHECKPOINT)
    if checkpoint_hash != config["gat_r_checkpoint_sha256_expected"]:
        raise RuntimeError("selected GAT-R checkpoint hash mismatch")
    existing_records = []
    for relative in (
        "08_holdout/FP_SHEP_HOLDOUT400/episode_records",
        "08_holdout/GAT_R_FP_ANCHOR_HOLDOUT400/episode_records",
    ):
        directory = REVISION / relative
        existing_records.extend(directory.glob("*.json") if directory.exists() else [])
    if existing_records:
        raise RuntimeError("Holdout performance already exists before selection freeze")

    payload = {
        "schema_version": "gat_recurrent_pre_holdout_freeze_v1",
        "status": "FROZEN_BEFORE_HOLDOUT_PERFORMANCE",
        "DEV_GATE": "PASS",
        "selected_selector": "gat_r",
        "selected_checkpoint": str(CHECKPOINT.relative_to(REPO_ROOT).as_posix()),
        "selected_checkpoint_sha256": checkpoint_hash,
        "SMOOTHNESS_SUPERVISION_REJECTED": "YES",
        "FINAL_GAT_RS_FREEZE": "NO",
        "GAT_R_SELECTED_FOR_HOLDOUT": "YES",
        "selection_reason": (
            "GAT-R and GAT-RS tied on Dev success and any-collision rate, while "
            "GAT-RS had significantly higher paired smoothness cost than both FP-SHEP and GAT-R."
        ),
        "dev_decision_path": str(DEV_DECISION.relative_to(REPO_ROOT).as_posix()),
        "dev_decision_sha256": sha256_file(DEV_DECISION),
        "training_manifest_path": str(TRAINING_MANIFEST.relative_to(REPO_ROOT).as_posix()),
        "training_manifest_sha256": sha256_file(TRAINING_MANIFEST),
        "target_contract_path": str(TARGET_CONTRACT.relative_to(REPO_ROOT).as_posix()),
        "target_contract_sha256": sha256_file(TARGET_CONTRACT),
        "holdout_config_path": str(HOLDOUT_CONFIG.relative_to(REPO_ROOT).as_posix()),
        "holdout_config_sha256": sha256_file(HOLDOUT_CONFIG),
        "holdout_manifest_path": str(HOLDOUT_MANIFEST.relative_to(REPO_ROOT).as_posix()),
        "holdout_manifest_sha256": sha256_file(HOLDOUT_MANIFEST),
        "holdout_manifest_semantic_sha256": load_json(HOLDOUT_MANIFEST)["manifest_semantic_sha256"],
        "holdout_validation_sha256": sha256_file(HOLDOUT_VALIDATION),
        "holdout_performance_rows_at_freeze": 0,
        "formal_v1_used_for_training_or_selection": False,
        "runtime_input_changed": False,
        "runtime_role_changed": False,
        "sac_dmp_rerr_contract_changed": False,
        "formal_v2_generated": False,
    }
    write_json(OUTPUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
