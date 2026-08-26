#!/usr/bin/env python3
"""Final raw-record reconciliation for Frozen-Strong Formal V2."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts/frozen_strong_formal_v2/20260826_110337"
FORMAL_STUDY = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
MANIFEST_PATH = FORMAL_STUDY / "10_formal_v2/FORMAL_V2_MANIFEST.json"
ORIGINAL_DIR = FORMAL_STUDY / "10_formal_v2/formal_records/M9_Proposed_RERR_GAT_SAC_DMP"
STRONG_DIR = ROOT / "03_formal_run/episode_records"
PREFREEZE = ROOT / "01_prefreeze/FINAL_FROZEN_STRONG_PREFREEZE.json"
IDENTITY = ROOT / "00_identity/FINAL_STRONG_IDENTITY_AUDIT.json"
TEAM_CSV = ROOT / "04_formal_results/strong_formal_team_results.csv"
AGENT_CSV = ROOT / "04_formal_results/strong_formal_agent_results.csv"
STAGE_CSV = ROOT / "04_formal_results/strong_formal_stage_summary.csv"
CONTINUOUS_CSV = ROOT / "05_paired_statistics/strong_vs_original_continuous.csv"
PAIR_JSON = ROOT / "05_paired_statistics/strong_vs_original_paired_success.json"
TRAJECTORY_MANIFEST = ROOT / "07_trajectories/strong_trajectory_manifest.csv"
FIGURE_INTEGRITY = ROOT / "09_paper_ready/FORMAL_FIGURE_DATA_INTEGRITY.csv"
PDF_VALIDATION = ROOT / "09_paper_ready/FORMAL_PDF_RENDER_VALIDATION.csv"
OUTPUT = ROOT / "10_reconciliation/final_reconciliation.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def common_jerk(record: Mapping[str, Any], path: Path) -> dict[str, float]:
    with np.load(path) as arrays:
        velocity = np.asarray(arrays["velocities"], dtype=float)
    active = np.zeros(velocity.shape[:2], dtype=bool)
    for agent in record["agents"]:
        agent_id = int(agent["agent_id"])
        stop = agent.get("terminal_completion_step")
        if stop is None:
            stop = velocity.shape[0] - 1
        active[: min(int(stop) + 1, velocity.shape[0]), agent_id] = True
    acceleration = np.diff(velocity, axis=0) / 0.1
    jerk = np.diff(acceleration, axis=0) / 0.1
    valid = active[2:] & active[1:-1] & active[:-2] & np.all(np.isfinite(jerk), axis=2)
    norm = np.linalg.norm(jerk, axis=2)[valid]
    lateral = np.linalg.norm(jerk[:, :, :2], axis=2)[valid]
    vertical = jerk[:, :, 2][valid]
    return {
        "vertical_jerk_mean_squared": float(np.mean(vertical**2)),
        "lateral_jerk_mean_squared": float(np.mean(lateral**2)),
        "jerk_p95_mps3": float(np.percentile(norm, 95)),
    }


def main() -> None:
    errors: list[str] = []
    manifest = load_json(MANIFEST_PATH)
    entries = list(manifest["entries"])
    ids = [str(entry["scenario_id"]) for entry in entries]
    id_set = set(ids)
    if len(ids) != 400 or len(id_set) != 400:
        errors.append("manifest_identity")
    prefreeze = load_json(PREFREEZE)
    identity = load_json(IDENTITY)

    strong_json = sorted(path for path in STRONG_DIR.glob("FORMAL_LR_*.json") if "SOFTWARE_ERROR" not in path.name)
    strong_npz = sorted(STRONG_DIR.glob("FORMAL_LR_*_trajectory.npz"))
    strong_trace = sorted(STRONG_DIR.glob("FORMAL_LR_*_limiter_trace.npz"))
    current_errors = sorted(STRONG_DIR.glob("*_SOFTWARE_ERROR.json"))
    if not (len(strong_json) == len(strong_npz) == len(strong_trace) == 400):
        errors.append("strong_record_counts")
    if current_errors:
        errors.append("current_software_errors")

    strong_success: dict[str, bool] = {}
    original_success: dict[str, bool] = {}
    strong_collision: dict[str, tuple[bool, bool, bool]] = {}
    strong_agents = 0
    nonfinite = 0
    trajectory_hash_mismatch = 0
    limiter_hash_mismatch = 0
    strong_smooth: dict[str, float] = {}
    original_smooth: dict[str, float] = {}
    strong_jerk: dict[str, dict[str, float]] = {}
    original_jerk: dict[str, dict[str, float]] = {}
    original_aggregate = hashlib.sha256()
    for sid in ids:
        strong_path = STRONG_DIR / f"{sid}.json"
        original_path = ORIGINAL_DIR / f"{sid}.json"
        if not strong_path.is_file() or not original_path.is_file():
            errors.append(f"missing_record:{sid}")
            continue
        strong = load_json(strong_path)
        original = load_json(original_path)
        strong_npz_path = STRONG_DIR / f"{sid}_trajectory.npz"
        strong_trace_path = STRONG_DIR / f"{sid}_limiter_trace.npz"
        original_npz_path = ORIGINAL_DIR / f"{sid}_trajectory.npz"
        if sha256_file(strong_npz_path) != strong["trajectory_sha256"]:
            trajectory_hash_mismatch += 1
        if sha256_file(strong_trace_path) != strong["limiter_trace_sha256"]:
            limiter_hash_mismatch += 1
        # The retained Original Formal-V2 schema names the file digest
        # ``trajectory_file_sha256``; the newly serialized Strong record uses
        # ``trajectory_sha256``.  Reconcile each archive against its native,
        # immutable schema instead of assuming the new field name retroactively.
        if sha256_file(original_npz_path) != original["trajectory_file_sha256"]:
            trajectory_hash_mismatch += 1
        original_aggregate.update(sid.encode())
        original_aggregate.update(bytes.fromhex(sha256_file(original_path)))
        original_aggregate.update(bytes.fromhex(sha256_file(original_npz_path)))
        with np.load(strong_npz_path) as arrays:
            positions = np.asarray(arrays["positions"], dtype=float)
            velocity = np.asarray(arrays["velocities"], dtype=float)
        if positions.ndim != 3 or positions.shape[1:] != (3, 3) or not np.isfinite(positions).all() or not np.isfinite(velocity).all():
            nonfinite += 1
        strong_ep = strong["episode"]
        original_ep = original["episode"]
        strong_success[sid] = bool(strong_ep["team_success"])
        original_success[sid] = bool(original_ep["team_success"])
        strong_collision[sid] = (
            bool(strong_ep["collision"]),
            bool(strong_ep["obstacle_collision"]),
            bool(strong_ep["inter_agent_collision"]),
        )
        strong_agents += len(strong["agents"])
        strong_smooth[sid] = float(strong_ep["trajectory_smoothness"])
        original_smooth[sid] = float(original_ep["trajectory_smoothness"])
        strong_jerk[sid] = common_jerk(strong, strong_npz_path)
        original_jerk[sid] = common_jerk(original, original_npz_path)

    if trajectory_hash_mismatch:
        errors.append("trajectory_hash_mismatch")
    if limiter_hash_mismatch:
        errors.append("limiter_hash_mismatch")
    if nonfinite:
        errors.append("trajectory_shape_or_nonfinite")
    if strong_agents != 1200:
        errors.append("raw_agent_count")

    team_rows = read_csv(TEAM_CSV)
    agent_rows = read_csv(AGENT_CSV)
    stage_rows = read_csv(STAGE_CSV)
    if len(team_rows) != 400 or {row["scenario_id"] for row in team_rows} != id_set:
        errors.append("team_csv_identity")
    if len(agent_rows) != 1200 or len({(row["scenario_id"], row["agent_id"]) for row in agent_rows}) != 1200:
        errors.append("agent_csv_identity")
    if len(stage_rows) != 6:
        errors.append("stage_summary_scope_count")

    raw_strong_success_count = sum(strong_success.values())
    raw_original_success_count = sum(original_success.values())
    csv_success_count = sum(row["team_success"].lower() == "true" for row in team_rows)
    if raw_strong_success_count != csv_success_count:
        errors.append("team_csv_success_reproduction")
    overall = next((row for row in stage_rows if row["scope"] == "Overall"), None)
    if overall is None or int(overall["team_success_count"]) != raw_strong_success_count:
        errors.append("stage_summary_success_reproduction")

    both = [sid for sid in ids if strong_success.get(sid, False) and original_success.get(sid, False)]
    original_only = [sid for sid in ids if original_success.get(sid, False) and not strong_success.get(sid, False)]
    strong_only = [sid for sid in ids if strong_success.get(sid, False) and not original_success.get(sid, False)]
    both_fail = [sid for sid in ids if not original_success.get(sid, False) and not strong_success.get(sid, False)]
    pair = load_json(PAIR_JSON)
    expected_pair = (len(both), len(original_only), len(strong_only), len(both_fail))
    observed_pair = (int(pair["both_success"]), int(pair["original_only_success"]), int(pair["strong_only_success"]), int(pair["both_failure"]))
    if expected_pair != observed_pair:
        errors.append("paired_success_reproduction")
    if raw_original_success_count != 381:
        errors.append("original_success_changed")

    continuous_rows = {row["metric"]: row for row in read_csv(CONTINUOUS_CSV)}
    raw_metrics = {
        "trajectory_smoothness": (original_smooth, strong_smooth),
        "vertical_jerk_mean_squared": (
            {sid: original_jerk[sid]["vertical_jerk_mean_squared"] for sid in both},
            {sid: strong_jerk[sid]["vertical_jerk_mean_squared"] for sid in both},
        ),
        "lateral_jerk_mean_squared": (
            {sid: original_jerk[sid]["lateral_jerk_mean_squared"] for sid in both},
            {sid: strong_jerk[sid]["lateral_jerk_mean_squared"] for sid in both},
        ),
        "jerk_p95_mps3": (
            {sid: original_jerk[sid]["jerk_p95_mps3"] for sid in both},
            {sid: strong_jerk[sid]["jerk_p95_mps3"] for sid in both},
        ),
    }
    continuous_checks: dict[str, bool] = {}
    for metric, (original_map, strong_map) in raw_metrics.items():
        original_mean = float(np.mean([original_map[sid] for sid in both]))
        strong_mean = float(np.mean([strong_map[sid] for sid in both]))
        row = continuous_rows.get(metric)
        passed = bool(
            row is not None
            and int(row["paired_n"]) == len(both)
            and abs(float(row["original_mean"]) - original_mean) <= 1.0e-9
            and abs(float(row["strong_mean"]) - strong_mean) <= 1.0e-9
        )
        continuous_checks[metric] = passed
        if not passed:
            errors.append(f"continuous_reproduction:{metric}")

    trajectory_rows = read_csv(TRAJECTORY_MANIFEST)
    if len(trajectory_rows) != 800 or any(row["raw_trajectory_used"].lower() != "true" or row["post_processing_applied"].lower() != "false" for row in trajectory_rows):
        errors.append("trajectory_manifest_integrity")
    figure_rows = read_csv(FIGURE_INTEGRITY)
    for row in figure_rows:
        source = REPO_ROOT / row["trajectory_source"]
        if not source.is_file() or sha256_file(source) != row["trajectory_sha256"] or row["raw_trajectory_used"] != "YES" or row["post_processing_applied"] != "NO":
            errors.append(f"figure_data_integrity:{row.get('figure_id')}:{row.get('scenario_id')}:{row.get('method')}")
    pdf_rows = read_csv(PDF_VALIDATION)
    if len(pdf_rows) != 5 or any(row["visual_inspection"] != "PASS" for row in pdf_rows):
        errors.append("pdf_render_validation")

    source_mismatch = []
    for source, expected in prefreeze["runtime_source_sha256"].items():
        observed = sha256_file(REPO_ROOT / source)
        if observed != expected:
            source_mismatch.append(source)
    if source_mismatch:
        errors.append("runtime_source_drift")
    if sha256_file(REPO_ROOT / identity["SAC_checkpoint"]) != identity["SAC_checkpoint_sha256"]:
        errors.append("sac_checkpoint_hash")
    if sha256_file(REPO_ROOT / identity["GAT_checkpoint"]) != identity["GAT_checkpoint_sha256"]:
        errors.append("gat_checkpoint_hash")

    payload = {
        "schema_version": "frozen_strong_formal_final_reconciliation_v1",
        "FINAL_RECONCILIATION": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": {
            "manifest_scenario_count": len(ids),
            "manifest_unique_count": len(id_set),
            "strong_record_count": len(strong_json),
            "strong_trajectory_count": len(strong_npz),
            "strong_limiter_trace_count": len(strong_trace),
            "strong_raw_agent_count": strong_agents,
            "team_csv_count": len(team_rows),
            "agent_csv_count": len(agent_rows),
            "strong_success_count_raw": raw_strong_success_count,
            "original_success_count_raw": raw_original_success_count,
            "paired_counts_raw": {
                "both_success": len(both),
                "original_only": len(original_only),
                "strong_only": len(strong_only),
                "both_failure": len(both_fail),
            },
            "continuous_metric_reproduction": continuous_checks,
            "trajectory_hash_mismatch_count": trajectory_hash_mismatch,
            "limiter_hash_mismatch_count": limiter_hash_mismatch,
            "nonfinite_or_shape_failure_count": nonfinite,
            "runtime_source_mismatch": source_mismatch,
            "figure_integrity_row_count": len(figure_rows),
            "pdf_validation_row_count": len(pdf_rows),
            "archived_prelaunch_software_error_count": len(list((ROOT / "03_formal_run/prelaunch_failure_attempt").glob("*_SOFTWARE_ERROR.json"))),
            "current_formal_software_error_count": len(current_errors),
        },
        "original_record_set_aggregate_sha256": original_aggregate.hexdigest(),
        "original_result_preserved": raw_original_success_count == 381,
        "all_raw_strong_trajectories_reopened": len(strong_npz) == 400,
        "all_raw_original_trajectories_reopened": raw_original_success_count == 381 and len(original_success) == 400,
        "figure_raw_trajectory_contract_verified": not any(error.startswith("figure_data_integrity") for error in errors),
        "source_and_checkpoint_hashes_verified": not source_mismatch and not any(error.endswith("checkpoint_hash") for error in errors),
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps({"status": payload["FINAL_RECONCILIATION"], "errors": errors, "strong_success": raw_strong_success_count}, indent=2))
    if errors:
        raise RuntimeError(f"final reconciliation failed: {errors}")


if __name__ == "__main__":
    main()
