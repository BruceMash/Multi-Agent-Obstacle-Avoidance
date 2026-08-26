#!/usr/bin/env python3
"""Finalize failure audit, conclusion, GO/NO-GO, and concise report."""

from __future__ import annotations

import csv
import hashlib
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


ROOT = REPO_ROOT / "artifacts/safety_adaptive_jerk_limiter/20260825_115704"
DEV_ORIGINAL = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552/04_development/records/original/episode_records"
DEV_STRONG = ROOT / "dev_records/strong/episode_records"
HOLDOUT_ORIGINAL = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/13_objective_revision/08_holdout/GAT_R_FP_ANCHOR_HOLDOUT400/episode_records"
HOLDOUT_STRONG = ROOT / "holdout_records/strong/episode_records"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])
    temporary.replace(path)


def load_record(root: Path, sid: str) -> dict[str, Any]:
    path = root / f"{sid}.json"
    record = load_json(path)
    record["json_path"] = path
    record["npz_path"] = root / f"{sid}_trajectory.npz"
    return record


def dev_pair_metrics() -> dict[str, float]:
    ids = sorted(path.stem for path in DEV_ORIGINAL.glob("GATRS_DEV_*.json") if not path.name.endswith("_SOFTWARE_ERROR.json"))
    original = {sid: load_record(DEV_ORIGINAL, sid) for sid in ids}
    strong = {sid: load_record(DEV_STRONG, sid) for sid in ids}
    both = [sid for sid in ids if original[sid]["episode"]["team_success"] and strong[sid]["episode"]["team_success"]]
    om = {sid: dev.trajectory_metrics(original[sid]) for sid in both}
    sm = {sid: dev.trajectory_metrics(strong[sid]) for sid in both}
    def reduction(field: str) -> float:
        a = np.asarray([om[sid][field] for sid in both], dtype=float)
        b = np.asarray([sm[sid][field] for sid in both], dtype=float)
        return 100.0 * (float(np.mean(a)) - float(np.mean(b))) / float(np.mean(a))
    original_path = float(np.mean([original[sid]["episode"]["team_path_length_m"] for sid in both]))
    strong_path = float(np.mean([strong[sid]["episode"]["team_path_length_m"] for sid in both]))
    return {
        "vertical_reduction_percent": reduction("vertical_jerk_mean_squared"),
        "lateral_reduction_percent": reduction("lateral_jerk_mean_squared"),
        "p95_reduction_percent": reduction("jerk_p95_mps3"),
        "path_change_percent": 100.0 * (strong_path - original_path) / original_path,
    }


def holdout_failure_rows() -> list[dict[str, Any]]:
    freeze = load_json(ROOT / "FINAL_JERK_LIMITER_FREEZE.json")
    rows: list[dict[str, Any]] = []
    for sid in freeze["scenario_ids"]:
        original = load_record(HOLDOUT_ORIGINAL, sid)
        strong = load_record(HOLDOUT_STRONG, sid)
        if not (original["episode"]["team_success"] and not strong["episode"]["team_success"]):
            continue
        trace = np.load(HOLDOUT_STRONG / str(strong["limiter_trace_file"]))
        final_step = int(strong["episode"]["steps"]) - 1
        for agent_id in range(3):
            indices = np.flatnonzero((np.asarray(trace["agent_id"]) == agent_id) & (np.asarray(trace["step"]) <= final_step))
            if not len(indices):
                continue
            index = int(indices[-1])
            recent = indices[np.asarray(trace["step"])[indices] >= max(0, final_step - 10)]
            latest_reference_step = max(
                [int(event["step"]) for event in strong.get("events", []) if int(event["agent_id"]) == agent_id and bool(event.get("goal_changed", True)) and int(event["step"]) <= final_step],
                default=0,
            )
            possible_suppression = bool(
                np.any(np.asarray(trace["limiter_active"])[recent])
                and np.max(np.asarray(trace["limiter_modification_norm_mps2"])[recent]) > 1e-9
                and not np.any(np.asarray(trace["hard_bypass"])[recent])
            )
            rows.append({
                "block": "holdout",
                "arm": "strong",
                "scenario_id": sid,
                "stage": strong["entry_identity"]["stage"],
                "family": strong["entry_identity"]["family"],
                "termination_reason": strong["episode"]["termination_reason"],
                "steps": strong["episode"]["steps"],
                "collision_time_s": 0.1 * int(strong["episode"]["steps"]),
                "obstacle_collision": strong["episode"]["obstacle_collision"],
                "peer_collision": strong["episode"]["inter_agent_collision"],
                "agent_id": agent_id,
                "last_control_step": int(trace["step"][index]),
                "safety_margin_m": float(trace["safety_margin_m"][index]),
                "q_safe": float(trace["q_safe"][index]),
                "hard_bypass": bool(trace["hard_bypass"][index]),
                "limiter_active": bool(trace["limiter_active"][index]),
                "raw_acceleration": json.dumps(np.asarray(trace["raw_acceleration"][index]).tolist()),
                "limited_acceleration": json.dumps(np.asarray(trace["limited_acceleration"][index]).tolist()),
                "previous_executed_acceleration": json.dumps(np.asarray(trace["previous_executed_acceleration"][index]).tolist()),
                "executed_acceleration": json.dumps(np.asarray(trace["executed_acceleration"][index]).tolist()),
                "modification_norm_mps2": float(trace["limiter_modification_norm_mps2"][index]),
                "time_since_latest_reference_change_s": 0.1 * (final_step - latest_reference_step),
                "recent_1s_any_limiter_active": bool(np.any(np.asarray(trace["limiter_active"])[recent])),
                "recent_1s_any_hard_bypass": bool(np.any(np.asarray(trace["hard_bypass"])[recent])),
                "recent_1s_max_modification_mps2": float(np.max(np.asarray(trace["limiter_modification_norm_mps2"])[recent])),
                "necessary_aggressive_maneuver_suppressed": "POSSIBLE" if possible_suppression else "NO_EVIDENCE_IN_LAST_1S",
            })
    return rows


def main() -> None:
    dev_results = {row["arm"].lower(): row for row in csv.DictReader((ROOT / "JERK_LIMITER_DEV_RESULTS.csv").open(encoding="utf-8-sig"))}
    holdout_results = {row["arm"].lower(): row for row in csv.DictReader((ROOT / "JERK_LIMITER_HOLDOUT_RESULTS.csv").open(encoding="utf-8-sig"))}
    for row in dev_results.values():
        row["static_collision"] = "UNAVAILABLE_NOT_RETAINED_SEPARATELY"
        row["dynamic_collision"] = "UNAVAILABLE_NOT_RETAINED_SEPARATELY"
    for row in holdout_results.values():
        row["static_collision"] = "UNAVAILABLE_NOT_RETAINED_SEPARATELY"
        row["dynamic_collision"] = "UNAVAILABLE_NOT_RETAINED_SEPARATELY"
    write_csv(ROOT / "JERK_LIMITER_DEV_RESULTS.csv", list(dev_results.values()))
    write_csv(ROOT / "JERK_LIMITER_HOLDOUT_RESULTS.csv", list(holdout_results.values()))
    holdout_decision = load_json(ROOT / "holdout_decision.json")
    thresholds = load_json(ROOT / "JERK_LIMITER_THRESHOLDS.json")
    pair = dev_pair_metrics()

    audit_path = ROOT / "JERK_LIMITER_FAILURE_AUDIT.csv"
    existing = [row for row in csv.DictReader(audit_path.open(encoding="utf-8-sig")) if row.get("block") != "holdout"]
    for row in existing:
        row.setdefault("block", "development")
    holdout_audit = holdout_failure_rows()
    write_csv(audit_path, existing + holdout_audit)

    dev_original = dev_results["original"]
    dev_strong = dev_results["strong"]
    holdout_original = holdout_results["original"]
    holdout_strong = holdout_results["strong"]
    holdout_path_change = 100.0 * (float(holdout_strong["mean_team_path_m_both_success"]) - float(holdout_original["mean_team_path_m_both_success"])) / float(holdout_original["mean_team_path_m_both_success"])
    conclusion = {
        "RETRAINING_PERFORMED": "NO",
        "SAC_CHECKPOINT_CHANGED": "NO",
        "PROPOSAL_CHANGED": "NO",
        "FP_SHEP_CHANGED": "NO",
        "GAT_CHANGED": "NO",
        "ERR_CHANGED": "NO",
        "DMP_CHANGED": "NO",
        "JERK_LIMITER_ADDED": "YES",
        "LIMITER_TYPE": "SAFETY_ADAPTIVE_VECTOR_JERK_LIMIT",
        "MILD_JMAX": thresholds["variants"]["mild"]["j_smooth_mps3"],
        "MEDIUM_JMAX": thresholds["variants"]["medium"]["j_smooth_mps3"],
        "STRONG_JMAX": thresholds["variants"]["strong"]["j_smooth_mps3"],
        "SELECTED_VARIANT": "STRONG",
        "DEV_ORIGINAL_SUCCESS": float(dev_original["team_success"]),
        "DEV_SELECTED_SUCCESS": float(dev_strong["team_success"]),
        "DEV_PEER_COLLISION_DELTA_PP": float(dev_strong["peer_collision_delta_pp_vs_original"]),
        "DEV_SMOOTHNESS_IMPROVEMENT_PERCENT": float(dev_strong["both_success_smoothness_reduction_percent"]),
        "DEV_VERTICAL_JERK_IMPROVEMENT_PERCENT": pair["vertical_reduction_percent"],
        "DEV_P95_JERK_IMPROVEMENT_PERCENT": pair["p95_reduction_percent"],
        "DEV_PATH_LENGTH_CHANGE_PERCENT": pair["path_change_percent"],
        "LIMITER_ACTIVATION_RATE": float(holdout_strong["limiter_activation_rate"]),
        "EMERGENCY_BYPASS_RATE": float(holdout_strong["emergency_bypass_rate"]),
        "HOLDOUT_PASSED": "YES",
        "HOLDOUT_SMOOTHNESS_IMPROVEMENT_PERCENT": holdout_decision["smoothness_improvement_percent"],
        "HOLDOUT_SUCCESS_DELTA_PP": holdout_decision["success_delta_pp"],
        "HOLDOUT_PEER_COLLISION_DELTA_PP": holdout_decision["peer_collision_delta_pp"],
        "HOLDOUT_PATH_LENGTH_CHANGE_PERCENT": holdout_path_change,
        "RAW_TRAJECTORY_VISUALLY_IMPROVED": "PARTIAL",
        "ZIGZAG_ELIMINATED": "NO",
        "FORMAL_REEVALUATION_RECOMMENDED": "YES",
        "FORMAL_V2_EXECUTED": "NO",
        "ORIGINAL_FORMAL_SUCCESS": 0.9525,
        "ORIGINAL_FORMAL_RESULT_CHANGED": "NO",
        "FINAL_RECOMMENDATION": "Freeze Strong for an explicitly authorized Formal V2 reevaluation; keep Original as the current formal method until then.",
    }
    atomic_json(ROOT / "conclusion.json", conclusion)
    go_no_go = {
        "schema_version": "safety_adaptive_jerk_limiter_formal_go_no_go_v1",
        "decision": "GO_AWAIT_EXPLICIT_USER_AUTHORIZATION",
        "selected_variant": "STRONG",
        "holdout_gate": "PASS",
        "formal_v2_executed": False,
        "formal_v2_original_result_preserved": "381/400 (95.25%)",
        "authorized_next_action": "STOP_AND_WAIT",
        "selected_freeze_sha256": hashlib.sha256((ROOT / "FINAL_JERK_LIMITER_FREEZE.json").read_bytes()).hexdigest(),
        "holdout_results_sha256": hashlib.sha256((ROOT / "JERK_LIMITER_HOLDOUT_RESULTS.csv").read_bytes()).hexdigest(),
        "holdout_raw_trajectories_sha256": hashlib.sha256((ROOT / "JERK_LIMITER_HOLDOUT_RAW_TRAJECTORIES.pdf").read_bytes()).hexdigest(),
    }
    atomic_json(ROOT / "JERK_LIMITER_FORMAL_GO_NO_GO.json", go_no_go)

    report = f"""# Safety-Adaptive Jerk-Limited SAC-DMP Execution

## Result

Strong is the only Development variant that passed all frozen gates. It achieved **97/100** Development success versus **96/100** for Original and reduced paired both-success smoothness by **{float(dev_strong['both_success_smoothness_reduction_percent']):.2f}%**.

On the independent balanced Holdout100, Strong achieved **98/100 (98.0%)** success versus **99/100 (99.0%)** for Original. Peer collision changed from **1.0% to 0.0%**, while obstacle collision changed from **0.0% to 2.0%**. The reliability and peer-safety gates pass at their frozen boundaries.

## Smoothness and raw morphology

On the 97 both-success Holdout pairs, Strong reduced the existing smoothness/jerk cost by **{holdout_decision['smoothness_improvement_percent']:.2f}%** (95% bootstrap CI **{holdout_decision['smoothness_improvement_95ci'][0]:.2f}%–{holdout_decision['smoothness_improvement_95ci'][1]:.2f}%**). Vertical and lateral jerk decreased by **{holdout_decision['vertical_jerk_improvement_percent']:.2f}%** and **{holdout_decision['lateral_jerk_improvement_percent']:.2f}%**. Common velocity-derived P90/P95 jerk proxies decreased by **{holdout_decision['p90_jerk_improvement_percent']:.2f}%/{holdout_decision['p95_jerk_improvement_percent']:.2f}%**.

The four outcome-blind fixed Holdout scenes all succeeded and improved smoothness by 38.3%–47.5%. Raw 3-D and altitude traces show lower-frequency sharp reversals, but visible recurrent lateral/vertical zigzag remains. Therefore `RAW_TRAJECTORY_VISUALLY_IMPROVED = PARTIAL` and `ZIGZAG_ELIMINATED = NO`.

Historical Original Holdout archives retain raw position/velocity but not applied-acceleration arrays. Holdout tail plots and P90/P95 statistics therefore use the same second finite difference of stored executed velocity for both arms; the retained existing smoothness metric remains the primary comparison.

## Path, limiter, and failures

Both-success Holdout team path length increased by **{holdout_path_change:.2f}%** and completion time increased by **{float(holdout_strong['mean_completion_time_s_both_success']) - float(holdout_original['mean_completion_time_s_both_success']):.3f} s**. The limiter activated on **{100*float(holdout_strong['limiter_activation_rate']):.2f}%** of agent control steps; emergency hard bypass occurred on **{100*float(holdout_strong['emergency_bypass_rate']):.4f}%**. Limiter-only cost was **{float(holdout_strong['limiter_runtime_us_agent_step']):.3f} microseconds/agent-step**.

Two Original-success Holdout episodes became Strong obstacle collisions. Their per-agent pre-collision raw/limited/previous accelerations, safety margins, bypass states, recent reference-change timing, and last-1-s limiter activity are retained in `JERK_LIMITER_FAILURE_AUDIT.csv`. Strong also recovered one Original failure, yielding a net -1 pp success change.

The retained evaluator records obstacle collision as one combined flag and do not retain static-versus-dynamic collision subtype. Those two requested subtype rates are marked `UNAVAILABLE_NOT_RETAINED_SEPARATELY` rather than reconstructed post hoc.

## Decision

- `SELECTED_VARIANT = STRONG`
- `HOLDOUT_PASSED = YES`
- `FORMAL_REEVALUATION_RECOMMENDED = YES`
- `FORMAL_V2_EXECUTED = NO`
- Original Formal V2 remains **381/400 (95.25%)**.

Strong is frozen for a possible new Formal V2 evaluation, but no Formal episode is authorized or run by this goal. The current paper-formal result remains attached to Original until explicit authorization and a new formal comparison.
"""
    (ROOT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "PASS", "holdout_failure_episode_count": len({row['scenario_id'] for row in holdout_audit}), "conclusion": conclusion}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
