#!/usr/bin/env python3
"""Read-only paired analysis for the frozen Strong Holdout100."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for search_path in (REPO_ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import analyze_safety_adaptive_jerk_limiter as dev


ARTIFACT_ROOT = REPO_ROOT / "artifacts/safety_adaptive_jerk_limiter/20260825_115704"
SOURCE_STUDY_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
ORIGINAL_ROOT = SOURCE_STUDY_ROOT / "13_objective_revision/08_holdout/GAT_R_FP_ANCHOR_HOLDOUT400/episode_records"
STRONG_ROOT = ARTIFACT_ROOT / "holdout_records/strong/episode_records"
FREEZE_PATH = ARTIFACT_ROOT / "FINAL_JERK_LIMITER_FREEZE.json"
RESULT_PATH = ARTIFACT_ROOT / "JERK_LIMITER_HOLDOUT_RESULTS.csv"
DECISION_PATH = ARTIFACT_ROOT / "holdout_decision.json"


def load_records(root: Path, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for sid in ids:
        path = root / f"{sid}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload["entry_identity"]["scenario_id"]) != sid:
            raise RuntimeError(f"record identity mismatch: {sid}")
        payload["json_path"] = path
        payload["npz_path"] = path.with_name(f"{sid}_trajectory.npz")
        records[sid] = payload
    return records


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in names} for row in rows])
    temporary.replace(path)


def trace_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    trace_path = Path(record["json_path"]).with_name(str(record["limiter_trace_file"]))
    trace = np.load(trace_path)
    active = np.asarray(trace["limiter_active"], dtype=bool)
    bypass = np.asarray(trace["hard_bypass"], dtype=bool)
    modification = np.asarray(trace["limiter_modification_norm_mps2"], dtype=float)
    runtime = np.asarray(trace["limiter_runtime_ns"], dtype=np.int64)
    return {
        "activation_rate": float(np.mean(active)),
        "bypass_rate": float(np.mean(bypass)),
        "unchanged_rate": float(np.mean(modification <= 1e-12)),
        "mean_modification_mps2": float(np.mean(modification)),
        "p95_modification_mps2": float(np.percentile(modification, 95)),
        "runtime_ms_episode": float(np.sum(runtime) / 1e6),
        "runtime_us_agent_step": float(np.mean(runtime) / 1e3),
    }


def common_velocity_jerk_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    """Compute a common raw jerk proxy from stored executed velocities for both arms."""
    arrays = np.load(record["npz_path"])
    velocity = np.asarray(arrays["velocities"], dtype=float)
    if velocity.ndim == 2:
        steps = np.asarray(arrays["steps"], dtype=int)
        agent_ids = np.asarray(arrays["agent_ids"], dtype=int)
        dense = np.full((int(np.max(steps)) + 1, 3, 3), np.nan, dtype=float)
        dense[steps, agent_ids] = velocity
        velocity = dense
    if velocity.ndim != 3 or velocity.shape[1:] != (3, 3):
        raise RuntimeError(f"unexpected velocity archive shape: {velocity.shape}")
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
    vertical = jerk[:, :, 2][valid]
    lateral = np.linalg.norm(jerk[:, :, :2], axis=2)[valid]
    return {
        "jerk_mean_squared": float(np.mean(norm**2)),
        "vertical_jerk_mean_squared": float(np.mean(vertical**2)),
        "lateral_jerk_mean_squared": float(np.mean(lateral**2)),
        "jerk_p90_mps3": float(np.percentile(norm, 90)),
        "jerk_p95_mps3": float(np.percentile(norm, 95)),
    }


def pct_reduction(original: np.ndarray, candidate: np.ndarray) -> float:
    return 100.0 * (float(np.mean(original)) - float(np.mean(candidate))) / float(np.mean(original))


def episode_fields(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record["episode"]


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    ids = list(map(str, freeze["scenario_ids"]))
    if len(ids) != 100:
        raise RuntimeError("frozen Holdout100 ID count changed")
    software_errors = list(STRONG_ROOT.glob("*_SOFTWARE_ERROR.json"))
    if software_errors:
        raise RuntimeError(f"Holdout software errors: {software_errors}")
    original = load_records(ORIGINAL_ROOT, ids)
    strong = load_records(STRONG_ROOT, ids)
    original_metric = {sid: common_velocity_jerk_metrics(record) for sid, record in original.items()}
    strong_metric = {sid: common_velocity_jerk_metrics(record) for sid, record in strong.items()}
    both = [sid for sid in ids if episode_fields(original[sid])["team_success"] and episode_fields(strong[sid])["team_success"]]

    def values(records: Mapping[str, Mapping[str, Any]], field: str) -> np.ndarray:
        return np.asarray([episode_fields(records[sid])[field] for sid in both], dtype=float)

    original_smooth = values(original, "trajectory_smoothness")
    strong_smooth = values(strong, "trajectory_smoothness")
    original_p90 = np.asarray([original_metric[sid]["jerk_p90_mps3"] for sid in both])
    strong_p90 = np.asarray([strong_metric[sid]["jerk_p90_mps3"] for sid in both])
    original_p95 = np.asarray([original_metric[sid]["jerk_p95_mps3"] for sid in both])
    strong_p95 = np.asarray([strong_metric[sid]["jerk_p95_mps3"] for sid in both])
    original_vertical = np.asarray([original_metric[sid]["vertical_jerk_mean_squared"] for sid in both])
    strong_vertical = np.asarray([strong_metric[sid]["vertical_jerk_mean_squared"] for sid in both])
    original_lateral = np.asarray([original_metric[sid]["lateral_jerk_mean_squared"] for sid in both])
    strong_lateral = np.asarray([strong_metric[sid]["lateral_jerk_mean_squared"] for sid in both])

    smooth_reduction = pct_reduction(original_smooth, strong_smooth)
    vertical_reduction = pct_reduction(original_vertical, strong_vertical)
    lateral_reduction = pct_reduction(original_lateral, strong_lateral)
    p90_reduction = pct_reduction(original_p90, strong_p90)
    p95_reduction = pct_reduction(original_p95, strong_p95)
    smooth_pairs = np.column_stack([original_smooth, strong_smooth])
    smooth_ci = dev.paired_bootstrap(smooth_pairs, lambda x: pct_reduction(x[:, 0], x[:, 1]), 20261125)
    p90_change_ci = dev.paired_bootstrap(strong_p90 - original_p90, np.mean, 20261225)
    p95_change_ci = dev.paired_bootstrap(strong_p95 - original_p95, np.mean, 20261325)

    original_success = float(np.mean([episode_fields(record)["team_success"] for record in original.values()]))
    strong_success = float(np.mean([episode_fields(record)["team_success"] for record in strong.values()]))
    original_peer = float(np.mean([episode_fields(record)["inter_agent_collision"] for record in original.values()]))
    strong_peer = float(np.mean([episode_fields(record)["inter_agent_collision"] for record in strong.values()]))
    success_delta_pp = 100.0 * (strong_success - original_success)
    peer_delta_pp = 100.0 * (strong_peer - original_peer)
    trace = [trace_metrics(strong[sid]) for sid in ids]

    fixed_ids = []
    for stage in range(1, 5):
        fixed_ids.append(min(sid for sid in ids if sid.startswith(f"GATRS_HOLDOUT_{stage}_")))
    fixed_details = []
    fixed_positive = 0
    fixed_worse = 0
    for sid in fixed_ids:
        both_success = bool(episode_fields(original[sid])["team_success"] and episode_fields(strong[sid])["team_success"])
        if both_success:
            smooth_fixed = 100.0 * (float(episode_fields(original[sid])["trajectory_smoothness"]) - float(episode_fields(strong[sid])["trajectory_smoothness"])) / float(episode_fields(original[sid])["trajectory_smoothness"])
            p95_fixed = 100.0 * (original_metric[sid]["jerk_p95_mps3"] - strong_metric[sid]["jerk_p95_mps3"]) / original_metric[sid]["jerk_p95_mps3"]
        else:
            smooth_fixed = p95_fixed = -100.0
        if both_success and smooth_fixed >= 15.0 and p95_fixed >= 15.0:
            fixed_positive += 1
        if not both_success or smooth_fixed < -10.0 or p95_fixed < -10.0:
            fixed_worse += 1
        fixed_details.append({"scenario_id": sid, "both_success": both_success, "smoothness_reduction_percent": smooth_fixed, "p95_jerk_reduction_percent": p95_fixed})

    reliability_gate = bool(success_delta_pp >= -1.0 - 1e-12)
    peer_gate = bool(peer_delta_pp <= 1.0 + 1e-12)
    smooth_gate = bool(smooth_reduction >= 15.0 and smooth_ci[0] > 0.0)
    tail_gate = bool(p90_reduction > 0.0 and p95_reduction > 0.0 and p90_change_ci[1] < 0.0 and p95_change_ci[1] < 0.0)
    visual_gate = bool(fixed_positive >= 3 and fixed_worse == 0)
    passed = bool(reliability_gate and peer_gate and smooth_gate and tail_gate and visual_gate)

    rows = []
    for arm, records in (("Original", original), ("Strong", strong)):
        episodes = [episode_fields(record) for record in records.values()]
        rows.append({
            "arm": arm,
            "n": len(records),
            "team_success": float(np.mean([row["team_success"] for row in episodes])),
            "collision": float(np.mean([row["collision"] for row in episodes])),
            "obstacle_collision": float(np.mean([row["obstacle_collision"] for row in episodes])),
            "peer_collision": float(np.mean([row["inter_agent_collision"] for row in episodes])),
            "timeout": float(np.mean([row["timeout"] for row in episodes])),
            "agent_completion": float(np.mean([row["agent_completion_rate"] for row in episodes])),
            "both_success_n": len(both),
            "smoothness_reduction_percent": 0.0 if arm == "Original" else smooth_reduction,
            "vertical_jerk_reduction_percent": 0.0 if arm == "Original" else vertical_reduction,
            "lateral_jerk_reduction_percent": 0.0 if arm == "Original" else lateral_reduction,
            "p90_jerk_reduction_percent": 0.0 if arm == "Original" else p90_reduction,
            "p95_jerk_reduction_percent": 0.0 if arm == "Original" else p95_reduction,
            "mean_team_path_m_both_success": float(np.mean(values(records, "team_path_length_m"))),
            "mean_completion_time_s_both_success": float(np.mean(values(records, "completion_time_s"))),
            "mean_min_obstacle_clearance_m": float(np.mean([row["minimum_obstacle_clearance_m"] for row in episodes])),
            "mean_min_peer_distance_m": float(np.mean([row["minimum_inter_agent_distance_m"] for row in episodes])),
            "limiter_activation_rate": 0.0 if arm == "Original" else float(np.mean([row["activation_rate"] for row in trace])),
            "emergency_bypass_rate": 0.0 if arm == "Original" else float(np.mean([row["bypass_rate"] for row in trace])),
            "unchanged_control_fraction": 1.0 if arm == "Original" else float(np.mean([row["unchanged_rate"] for row in trace])),
            "mean_acceleration_modification_mps2": 0.0 if arm == "Original" else float(np.mean([row["mean_modification_mps2"] for row in trace])),
            "p95_acceleration_modification_mps2": 0.0 if arm == "Original" else float(np.mean([row["p95_modification_mps2"] for row in trace])),
            "limiter_runtime_ms_episode": 0.0 if arm == "Original" else float(np.mean([row["runtime_ms_episode"] for row in trace])),
            "limiter_runtime_us_agent_step": 0.0 if arm == "Original" else float(np.mean([row["runtime_us_agent_step"] for row in trace])),
        })
    write_csv(RESULT_PATH, rows)
    decision = {
        "schema_version": "safety_adaptive_jerk_limiter_holdout_decision_v1",
        "status": "PASS" if passed else "FAIL",
        "selected_variant": "STRONG" if passed else "NONE",
        "scenario_count": len(ids),
        "both_success_n": len(both),
        "success_delta_pp": success_delta_pp,
        "peer_collision_delta_pp": peer_delta_pp,
        "smoothness_improvement_percent": smooth_reduction,
        "smoothness_improvement_95ci": list(smooth_ci),
        "vertical_jerk_improvement_percent": vertical_reduction,
        "lateral_jerk_improvement_percent": lateral_reduction,
        "p90_jerk_improvement_percent": p90_reduction,
        "p95_jerk_improvement_percent": p95_reduction,
        "p90_change_mps3_95ci": list(p90_change_ci),
        "p95_change_mps3_95ci": list(p95_change_ci),
        "fixed_visual_scene_ids": fixed_ids,
        "fixed_visual_details": fixed_details,
        "gates": {
            "reliability": reliability_gate,
            "peer_safety": peer_gate,
            "smoothness_replication": smooth_gate,
            "jerk_tail_replication": tail_gate,
            "fixed_raw_visual": visual_gate,
        },
        "holdout_passed": passed,
        "formal_v2_executed": False,
        "formal_reevaluation_recommended": passed,
        "jerk_tail_reconstruction_contract": "common second finite difference of raw stored executed velocity at dt=0.1 s; historical Original Holdout did not retain applied-acceleration arrays",
    }
    DECISION_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
