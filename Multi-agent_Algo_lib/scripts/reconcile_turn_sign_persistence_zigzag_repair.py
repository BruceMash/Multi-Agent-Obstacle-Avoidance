#!/usr/bin/env python3
"""Independent raw-artifact reconciliation for turn-sign persistence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts/turn_sign_persistence_zigzag_repair/20260825_222130"
FREEZE = ROOT / "TURN_PERSISTENCE_RUNTIME_FREEZE.json"


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
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def reconcile(block: str) -> dict[str, Any]:
    suffix = "DEV" if block == "development" else "HOLDOUT"
    manifest_path = ROOT / f"TURN_PERSISTENCE_{suffix}100_MANIFEST.json"
    decision_path = ROOT / ("TURN_PERSISTENCE_DEV_GO_NO_GO.json" if block == "development" else "TURN_PERSISTENCE_HOLDOUT_DECISION.json")
    manifest = load(manifest_path)
    decision = load(decision_path)
    freeze = load(FREEZE)
    ids = [str(row["scenario_id"]) for row in manifest["entries"]]
    checks: dict[str, bool] = {
        "manifest_count_100": len(ids) == 100,
        "manifest_unique_ids": len(set(ids)) == 100,
        "microtest_pass": load(ROOT / "TURN_PERSISTENCE_MICROTEST.json").get("status") == "PASS",
        "formal_not_executed": decision.get("FORMAL_V2_EXECUTED") is False,
    }
    cell_counts: dict[tuple[str, str], int] = {}
    for row in manifest["entries"]:
        key = (str(row["stage"]), str(row["family"]))
        cell_counts[key] = cell_counts.get(key, 0) + 1
        scene = REPO_ROOT / row["scenario_file"]
        checks[f"scene_hash::{row['scenario_id']}"] = scene.exists() and sha256(scene) == row["scenario_file_sha256"]
    checks["balanced_4x5x5"] = len(cell_counts) == 20 and set(cell_counts.values()) == {5}
    for name, expected in freeze["source_sha256"].items():
        checks[f"source_hash::{name}"] = sha256(REPO_ROOT / name) == expected
    for name, expected in freeze["frozen_artifact_sha256"].items():
        checks[f"frozen_artifact_hash::{name}"] = sha256(REPO_ROOT / name) == expected

    outcomes: dict[str, dict[str, int]] = {}
    trajectory_keys: dict[str, set[str]] = {}
    repaired_trace_count = 0
    bypass_rows = 0
    bypass_contract_violations = 0
    tangent_checks = 0
    tangent_violations = 0
    nonfinite_rows = 0
    persistence_modified = 0
    for arm in ("strong", "repaired"):
        root = ROOT / block / arm / "episode_records"
        software_errors = list(root.glob("*_SOFTWARE_ERROR.json"))
        checks[f"no_software_errors::{arm}"] = len(software_errors) == 0
        rows = []
        trajectory_keys[arm] = set()
        for sid in ids:
            record_path = root / f"{sid}.json"
            checks[f"record_exists::{arm}::{sid}"] = record_path.exists()
            if not record_path.exists():
                continue
            record = load(record_path)
            rows.append(record["episode"])
            checks[f"identity::{arm}::{sid}"] = record["entry_identity"]["scenario_id"] == sid
            trajectory = root / record["trajectory_file"]
            trace_path = root / record["limiter_trace_file"]
            checks[f"trajectory_hash::{arm}::{sid}"] = sha256(trajectory) == record["trajectory_sha256"]
            checks[f"trace_hash::{arm}::{sid}"] = sha256(trace_path) == record["limiter_trace_sha256"]
            trajectory_keys[arm].add(record["trajectory_sha256"])
            arrays = np.load(trajectory)
            nonfinite_rows += int(np.sum(~np.isfinite(arrays["positions"])))
            if arm == "repaired":
                trace = np.load(trace_path)
                required = {
                    "raw_acceleration", "strong_limited_acceleration", "persistence_output_acceleration",
                    "executed_acceleration", "execution_velocity", "horizontal_requested_mps2",
                    "horizontal_executed_mps2", "vertical_requested_mps2", "vertical_executed_mps2",
                    "horizontal_accepted_sign", "vertical_accepted_sign", "horizontal_pending_sign",
                    "vertical_pending_sign", "horizontal_pending_count", "vertical_pending_count",
                    "persistence_bypassed_for_safety", "persistence_modified",
                }
                checks[f"trace_schema::{sid}"] = required.issubset(trace.files)
                repaired_trace_count += len(trace["step"])
                raw = np.asarray(trace["raw_acceleration"], dtype=float)
                strong = np.asarray(trace["strong_limited_acceleration"], dtype=float)
                filtered = np.asarray(trace["persistence_output_acceleration"], dtype=float)
                velocity = np.asarray(trace["execution_velocity"], dtype=float)
                bypass = np.asarray(trace["persistence_bypassed_for_safety"], dtype=bool)
                modified = np.asarray(trace["persistence_modified"], dtype=bool)
                bypass_rows += int(np.sum(bypass))
                persistence_modified += int(np.sum(modified))
                bypass_contract_violations += int(np.sum(np.linalg.norm(filtered[bypass] - raw[bypass], axis=1) > 1e-10))
                valid = (~bypass) & (np.linalg.norm(velocity[:, :2], axis=1) > 1e-9)
                tangent = np.zeros_like(velocity[:, :2])
                tangent[valid] = velocity[valid, :2] / np.linalg.norm(velocity[valid, :2], axis=1, keepdims=True)
                tangent_checks += int(np.sum(valid))
                tangent_violations += int(np.sum(np.abs(np.sum((filtered[:, :2] - strong[:, :2]) * tangent, axis=1))[valid] > 1e-8))
                nonfinite_rows += int(np.sum(~np.isfinite(raw)) + np.sum(~np.isfinite(filtered)))
        outcomes[arm] = {
            "n": len(rows),
            "success": sum(bool(row["team_success"]) for row in rows),
            "collision": sum(bool(row["collision"]) for row in rows),
            "obstacle_collision": sum(bool(row["obstacle_collision"]) for row in rows),
            "peer_collision": sum(bool(row["inter_agent_collision"]) for row in rows),
            "timeout": sum(bool(row["timeout"]) for row in rows),
        }
        checks[f"record_count_100::{arm}"] = len(rows) == 100
        expected_success = int(round(100 * float(decision["team_success"][arm])))
        checks[f"success_reproduced::{arm}"] = outcomes[arm]["success"] == expected_success
    checks["paired_trajectory_sets_unique"] = all(len(trajectory_keys[arm]) == 100 for arm in trajectory_keys)
    checks["repaired_trace_nonempty"] = repaired_trace_count > 0
    checks["safety_bypass_contract"] = bypass_contract_violations == 0
    checks["horizontal_tangent_preserved"] = tangent_checks > 0 and tangent_violations == 0
    checks["finite_execution_records"] = nonfinite_rows == 0
    checks["persistence_intervened"] = persistence_modified > 0
    payload = {
        "schema_version": f"turn_persistence_{block}_independent_reconciliation_v1",
        "block": block,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "outcomes_recomputed": outcomes,
        "trace_reconciliation": {
            "repaired_trace_count": repaired_trace_count,
            "persistence_modified_count": persistence_modified,
            "safety_bypass_row_count": bypass_rows,
            "safety_bypass_contract_violations": bypass_contract_violations,
            "horizontal_tangent_checks": tangent_checks,
            "horizontal_tangent_violations": tangent_violations,
            "nonfinite_value_count": nonfinite_rows,
        },
    }
    output = ROOT / ("independent_reconciliation.json" if block == "development" else "holdout_independent_reconciliation.json")
    write(output, payload)
    if block == "development":
        write(ROOT / "final_reconciliation.json", payload)
    print(json.dumps({"status": payload["status"], "outcomes": outcomes, "failed_checks": [key for key, value in checks.items() if not value]}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=("development", "holdout"), required=True)
    args = parser.parse_args()
    result = reconcile(args.block)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
