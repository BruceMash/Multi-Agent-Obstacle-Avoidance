#!/usr/bin/env python3
"""Freeze the Holdout-validated GAT-R checkpoint before Formal V2 creation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
REVISION = ROOT / "13_objective_revision"
CHECKPOINT = REVISION / "05_gat_r_fp_anchor_training/checkpoints/best_validation.pt"
PRE_HOLDOUT = REVISION / "09_final_freeze/PRE_HOLDOUT_GAT_R_FREEZE.json"
DEV_DECISION = REVISION / "07_development/DEV_GATE_DECISION.json"
HOLDOUT_TESTS = REVISION / "08_holdout/holdout_paired_tests.json"
CANONICAL_OUTPUT = ROOT / "09_final_freeze/FINAL_GAT_RS_FREEZE.json"
REVISION_OUTPUT = REVISION / "09_final_freeze/FINAL_GAT_R_FREEZE.json"


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
    pre = load_json(PRE_HOLDOUT)
    dev = load_json(DEV_DECISION)
    holdout = load_json(HOLDOUT_TESTS)
    checkpoint_hash = sha256_file(CHECKPOINT)
    if pre["selected_checkpoint_sha256"] != checkpoint_hash:
        raise RuntimeError("selected checkpoint changed after pre-Holdout freeze")
    if dev["DEV_GATE"] != "PASS" or dev["selected_selector_for_holdout"] != "gat_r":
        raise RuntimeError("Dev gate does not select GAT-R")
    if holdout["HOLDOUT_GATE"] != "PASS" or not bool(holdout["FORMAL_V2_OPEN_AUTHORIZED"]):
        raise RuntimeError("Holdout gate does not authorize Formal V2")
    overall = holdout["paired_binary"]["overall"]
    payload = {
        "schema_version": "gat_recurrent_final_checkpoint_freeze_v2",
        "status": "FROZEN_BEFORE_FORMAL_V2_MANIFEST",
        "FINAL_GAT_RS_FREEZE": "NO",
        "FINAL_GAT_R_FREEZE": "YES",
        "SMOOTHNESS_SUPERVISION_REJECTED": "YES",
        "selected_selector": "gat_r",
        "selected_checkpoint": str(CHECKPOINT.relative_to(REPO_ROOT).as_posix()),
        "selected_checkpoint_sha256": checkpoint_hash,
        "DEV_GATE": "PASS",
        "HOLDOUT_GATE": "PASS",
        "formal_v2_open_authorized": True,
        "holdout_success_comparison": overall["team_success"],
        "holdout_collision_comparison": overall["collision"],
        "holdout_peer_collision_comparison": overall["inter_agent_collision"],
        "holdout_smoothness_noninferior_to_fp": holdout["smoothness_noninferior_to_fp"],
        "smoothness_interpretation": (
            "GAT-RS was rejected at Dev. The selected GAT-R passes success and safety gates "
            "but does not claim a smoothness improvement."
        ),
        "pre_holdout_freeze_sha256": sha256_file(PRE_HOLDOUT),
        "dev_decision_sha256": sha256_file(DEV_DECISION),
        "holdout_tests_sha256": sha256_file(HOLDOUT_TESTS),
        "formal_v1_used_for_training_tuning_or_selection": False,
        "post_holdout_checkpoint_or_threshold_change_allowed": False,
        "runtime_input_changed": False,
        "runtime_role_changed": False,
        "sac_dmp_rerr_contract_changed": False,
        "formal_v2_manifest_created_at_freeze": False,
        "formal_v2_episode_count_at_freeze": 0,
    }
    write_json(CANONICAL_OUTPUT, payload)
    write_json(REVISION_OUTPUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
