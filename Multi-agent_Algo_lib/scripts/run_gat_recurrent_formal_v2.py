#!/usr/bin/env python3
"""Run the Holdout-authorized untouched Formal V2 with frozen GAT-R.

This is a thin, explicit re-binding of the reconciled eight-method long-range
formal engine.  It changes only the artifact namespace, seed block, selected
GAT checkpoint/training config, and the long-range goal-distance normalization
already exercised in Dev and Holdout.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts import run_long_range_formal_benchmark as base  # noqa: E402


ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
REVISION = ROOT / "13_objective_revision"
FREEZE_DIR = ROOT / "09_final_freeze"
FORMAL_DIR = ROOT / "10_formal_v2"
RECORD_DIR = FORMAL_DIR / "formal_records"
FINAL_CHECKPOINT_FREEZE = FREEZE_DIR / "FINAL_GAT_RS_FREEZE.json"
DEV_DECISION = REVISION / "07_development/DEV_GATE_DECISION.json"
HOLDOUT_TESTS = REVISION / "08_holdout/holdout_paired_tests.json"
PRE_HOLDOUT_FREEZE = REVISION / "09_final_freeze/PRE_HOLDOUT_GAT_R_FREEZE.json"
USED_REGISTRY = ROOT / "02_recurrent_dataset/ALL_USED_SCENE_REGISTRY.csv"
OLD_ARTIFACT = REPO_ROOT / "artifacts/semi_structured_long_range_main_benchmark/20260820_193228"
OLD_M9 = OLD_ARTIFACT / "10_final_freeze/method_configs/M9_Proposed_RERR_GAT_SAC_DMP.json"
SELECTED_CHECKPOINT = REVISION / "05_gat_r_fp_anchor_training/checkpoints/best_validation.pt"
SELECTED_TRAINING_CONFIG = REPO_ROOT / "configs/training/gat_recurrent_r_fp_anchor.json"


METHODS = tuple(
    {
        **copy.deepcopy(method),
        **(
            {
                "display_name": "R-ERR + GAT-R + SAC-DMP",
                "role": "FINAL_HOLDOUT_VALIDATED_METHOD",
            }
            if method["method_id"] == "M9_Proposed_RERR_GAT_SAC_DMP"
            else {}
        ),
    }
    for method in base.METHODS
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def development_and_holdout_gate() -> dict[str, Any]:
    dev = load_json(DEV_DECISION)
    holdout = load_json(HOLDOUT_TESTS)
    checkpoint = load_json(FINAL_CHECKPOINT_FREEZE)
    pre = load_json(PRE_HOLDOUT_FREEZE)
    if dev["DEV_GATE"] != "PASS" or dev["selected_selector_for_holdout"] != "gat_r":
        raise RuntimeError("Dev gate does not select GAT-R")
    if holdout["HOLDOUT_GATE"] != "PASS" or not bool(holdout["FORMAL_V2_OPEN_AUTHORIZED"]):
        raise RuntimeError("Holdout gate does not authorize Formal V2")
    if checkpoint["FINAL_GAT_R_FREEZE"] != "YES" or checkpoint["FINAL_GAT_RS_FREEZE"] != "NO":
        raise RuntimeError("final checkpoint freeze is inconsistent")
    if checkpoint["selected_checkpoint_sha256"] != base.sha256_file(SELECTED_CHECKPOINT):
        raise RuntimeError("selected GAT-R checkpoint changed")
    return {
        "DEV_GATE": {
            "path": str(DEV_DECISION.relative_to(REPO_ROOT).as_posix()),
            "sha256": base.sha256_file(DEV_DECISION),
            "status": "PASS",
            "selected_selector": "gat_r",
        },
        "HOLDOUT_GATE": {
            "path": str(HOLDOUT_TESTS.relative_to(REPO_ROOT).as_posix()),
            "sha256": base.sha256_file(HOLDOUT_TESTS),
            "status": "PASS",
            "success_delta_pp": holdout["paired_binary"]["overall"]["team_success"]["gat_r_minus_fp_shep_rate_pp"],
            "collision_delta_pp": holdout["paired_binary"]["overall"]["collision"]["gat_r_minus_fp_shep_rate_pp"],
            "peer_collision_delta_pp": holdout["paired_binary"]["overall"]["inter_agent_collision"]["gat_r_minus_fp_shep_rate_pp"],
        },
        "FINAL_CHECKPOINT_FREEZE": {
            "path": str(FINAL_CHECKPOINT_FREEZE.relative_to(REPO_ROOT).as_posix()),
            "sha256": base.sha256_file(FINAL_CHECKPOINT_FREEZE),
            "selected_checkpoint_sha256": checkpoint["selected_checkpoint_sha256"],
            "smoothness_supervision_rejected": True,
        },
        "PRE_HOLDOUT_FREEZE": {
            "path": str(PRE_HOLDOUT_FREEZE.relative_to(REPO_ROOT).as_posix()),
            "sha256": base.sha256_file(PRE_HOLDOUT_FREEZE),
            "checkpoint_unchanged": pre["selected_checkpoint_sha256"] == checkpoint["selected_checkpoint_sha256"],
        },
        "formal_v1_used_for_training_tuning_or_selection": False,
    }


def derived_method_config(method: Mapping[str, str]) -> dict[str, Any]:
    if method["method_id"] != "M9_Proposed_RERR_GAT_SAC_DMP":
        return _ORIGINAL_RESOLVE(method)
    config = load_json(OLD_M9)
    config.update(
        {
            "artifact_root": str(ROOT.relative_to(REPO_ROOT).as_posix()),
            "output_subdir": "10_formal_v2/formal_records/M9_Proposed_RERR_GAT_SAC_DMP",
            "development_manifest": "10_formal_v2/FORMAL_V2_MANIFEST.json",
            "stage1_config": str(SELECTED_TRAINING_CONFIG.relative_to(REPO_ROOT).as_posix()),
            "gat_checkpoint": str(SELECTED_CHECKPOINT.relative_to(REPO_ROOT).as_posix()),
            "gat_checkpoint_sha256_expected": base.sha256_file(SELECTED_CHECKPOINT),
            "configuration_id": "FORMAL_V2_GAT_R_FP_ANCHOR",
            "final_method_description": (
                "Proposal + FP-SHEP H4 + focal-safe FP-anchored GAT-R + "
                "R-ERR + adapted 256-ray SAC-DMP"
            ),
        }
    )
    config["graph"] = copy.deepcopy(config["graph"])
    config["graph"]["task_goal_distance_scale_m"] = 100.0
    config["graph"]["runtime_feature_semantics_changed"] = False
    return config


def frozen_method_config(method: Mapping[str, str]) -> dict[str, Any]:
    return load_json(FREEZE_DIR / "method_configs" / f"{method['method_id']}.json")


def historical_registry() -> tuple[list[dict[str, Any]], dict[str, set[Any]]]:
    rows: list[dict[str, Any]] = []
    sets: dict[str, set[Any]] = {
        "seed": set(),
        "geometry_fingerprint": set(),
        "dynamic_track_fingerprint": set(),
        "translation_invariant_fingerprint": set(),
        "start_goal_fingerprint": set(),
    }
    with USED_REGISTRY.open("r", newline="", encoding="utf-8-sig") as handle:
        for source in csv.DictReader(handle):
            row = dict(source)
            if row.get("seed") not in (None, ""):
                row["seed"] = int(row["seed"])
            rows.append(row)
            for key in sets:
                value = row.get(key)
                if value not in (None, ""):
                    sets[key].add(int(value) if key == "seed" else value)
    return rows, sets


def configure(*, frozen: bool) -> None:
    base.ARTIFACT_ROOT = ROOT
    base.FREEZE_DIR = FREEZE_DIR
    base.MANIFEST_DIR = FORMAL_DIR
    base.RECORD_DIR = RECORD_DIR
    base.FINAL_METHOD_FREEZE = FREEZE_DIR / "FINAL_LONG_RANGE_METHOD_FREEZE.json"
    base.FORMAL_RUN_FREEZE = FREEZE_DIR / "FORMAL_V2_RUN_FREEZE.json"
    base.FORMAL_MANIFEST = FORMAL_DIR / "FORMAL_V2_MANIFEST.json"
    base.FORMAL_REGISTRY = FORMAL_DIR / "ALL_USED_SCENE_REGISTRY.csv"
    base.DEVELOPMENT_SELECTION_FREEZE = DEV_DECISION
    base.SHORT_COORDINATION_RECONCILIATION = PRE_HOLDOUT_FREEZE
    base.FORMAL_ENGINE_PREFLIGHT = FREEZE_DIR / "formal_v2_engine_preflight.json"
    base.SEED_BASE = 3_400_000_000
    base.SCENARIOS_PER_STAGE = 100
    base.EXPECTED_SCENARIOS = 400
    base.METHODS = METHODS
    base.METHOD_BY_ID = {row["method_id"]: row for row in METHODS}
    base.METHOD_ORDER = tuple(base.METHOD_BY_ID)
    base.DEVELOPMENT_EVIDENCE = {}
    base.SOURCE_PATHS = tuple(
        dict.fromkeys(
            (*base.SOURCE_PATHS,
             "Multi-agent_Algo_lib/scripts/run_gat_recurrent_formal_v2.py",
             "Multi-agent_Algo_lib/scripts/run_gat_recurrent_r_development.py",
             "Multi-agent_Algo_lib/scripts/train_gat_recurrent_fp_anchored.py",
             str(FINAL_CHECKPOINT_FREEZE.relative_to(REPO_ROOT).as_posix()))
        )
    )
    base.development_gate = development_and_holdout_gate
    base.historical_scene_registry = historical_registry
    base.resolved_method_config = frozen_method_config if frozen else derived_method_config


def preflight() -> None:
    development_and_holdout_gate()
    marker = base.FORMAL_MANIFEST
    if marker.exists():
        payload = load_json(marker)
        if payload.get("status") not in {"NOT_RUN", "NOT_GENERATED"}:
            raise RuntimeError("Formal V2 manifest already generated")
    if RECORD_DIR.exists() and any(RECORD_DIR.rglob("*.json")):
        raise RuntimeError("Formal V2 records already exist")
    manifest_path = OLD_ARTIFACT / "08_development/development_manifest.json"
    manifest = load_json(manifest_path)
    entry = dict(manifest["entries"][0])
    checks: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for method in METHODS:
        try:
            runtime = base.build_method_runtime(method, manifest)
            row, agents, trajectory, events = base.evaluate_formal_episode(method, runtime, entry)
            positions = np.asarray(trajectory["positions"], dtype=float)
            if row.get("method_id") != method["method_id"] or row.get("scenario_id") != entry["scenario_id"]:
                raise RuntimeError("standardized identity mismatch")
            if len(agents) != 3 or positions.ndim != 3 or positions.shape[1:] != (3, 3):
                raise RuntimeError("preflight result shape mismatch")
            if not np.isfinite(positions).all():
                raise RuntimeError("non-finite preflight trajectory")
            checks.append(
                {
                    "method_id": method["method_id"],
                    "engine": method["engine"],
                    "status": "PASS",
                    "agent_row_count": len(agents),
                    "trajectory_shape": list(positions.shape),
                    "event_schema_exercised": isinstance(events, list),
                    "performance_values_retained": False,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "method_id": method["method_id"],
                    "engine": method["engine"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    payload = {
        "schema_version": "gat_recurrent_formal_v2_engine_preflight_v1",
        "status": "PASS" if len(checks) == len(METHODS) and not errors else "FAIL",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_split": "preexisting_development_integration_scene",
        "source_manifest": str(manifest_path.relative_to(REPO_ROOT).as_posix()),
        "source_manifest_sha256": base.sha256_file(manifest_path),
        "scenario_id": entry["scenario_id"],
        "method_count": len(checks),
        "expected_method_count": len(METHODS),
        "checks": checks,
        "errors": errors,
        "performance_values_retained": False,
        "formal_v2_manifest_generated": False,
        "formal_v2_episode_count": 0,
        "formal_data_used": False,
    }
    base.atomic_json(base.FORMAL_ENGINE_PREFLIGHT, payload)
    print(json.dumps({"phase": "preflight", "status": payload["status"], "method_count": len(checks), "errors": errors}), flush=True)
    if payload["status"] != "PASS":
        raise RuntimeError(f"Formal V2 engine preflight failed: {errors}")


_ORIGINAL_RESOLVE = base.resolved_method_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "prepare", "run", "status", "finalize"))
    parser.add_argument("--method-id", choices=tuple(row["method_id"] for row in METHODS))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    configure(frozen=args.phase in {"run", "status", "finalize"})
    if args.phase == "preflight":
        preflight()
    elif args.phase == "prepare":
        base.prepare()
    elif args.phase == "run":
        if args.method_id is None:
            raise ValueError("--method-id is required for run")
        base.run(args.method_id, args.shard_index, args.shard_count, args.limit)
    elif args.phase == "status":
        base.status()
    else:
        base.finalize()


if __name__ == "__main__":
    main()
