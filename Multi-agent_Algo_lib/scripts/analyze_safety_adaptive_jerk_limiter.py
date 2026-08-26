#!/usr/bin/env python3
"""Read-only paired analysis for the safety-adaptive jerk-limiter screen."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ARTIFACT_ROOT = REPO_ROOT / "artifacts/safety_adaptive_jerk_limiter/20260825_115704"
SOURCE_STUDY_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
SOURCE_MANIFEST = SOURCE_STUDY_ROOT / "07_development/GAT_RS_DEV_SCENE_MANIFEST.json"
SOURCE_ORIGINAL = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552/04_development/records/original/episode_records"
DEV_ROOT = ARTIFACT_ROOT / "dev_records"
ARMS = ("original", "mild", "medium", "strong")
FIXED_SCENES = tuple(f"GATRS_DEV_{stage}_000" for stage in range(1, 5))
DT = 0.1


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    names = list(fields or [])
    for row in rows:
        for key in row:
            if key not in names:
                names.append(str(key))
    if not names:
        names = ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows([{key: json_ready(row.get(key)) for key in names} for row in rows])
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def root_for(arm: str) -> Path:
    return SOURCE_ORIGINAL if arm == "original" else DEV_ROOT / arm / "episode_records"


def load_records(arm: str) -> dict[str, dict[str, Any]]:
    root = root_for(arm)
    records = {}
    for path in sorted(root.glob("GATRS_DEV_*.json")):
        if path.name.endswith("_SOFTWARE_ERROR.json"):
            continue
        payload = load_json(path)
        sid = str(payload["entry_identity"]["scenario_id"])
        payload["json_path"] = path
        payload["npz_path"] = path.with_name(path.stem + "_trajectory.npz")
        records[sid] = payload
    return records


def active_acceleration(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.load(record["npz_path"])
    acceleration = np.asarray(arrays["applied_accelerations_full"], dtype=float)
    acceleration = acceleration.copy()
    if not np.all(np.isfinite(acceleration[0])):
        acceleration[0] = 0.0
    active = np.zeros(acceleration.shape[:2], dtype=bool)
    for agent in record["agents"]:
        agent_id = int(agent["agent_id"])
        stop = agent.get("terminal_completion_step")
        if stop is None:
            stop = acceleration.shape[0] - 1
        active[: min(int(stop) + 1, acceleration.shape[0]), agent_id] = True
    return acceleration, active


def trajectory_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    acceleration, active_state = active_acceleration(record)
    jerk = np.diff(acceleration, axis=0) / DT
    valid = active_state[1:] & active_state[:-1] & np.all(np.isfinite(jerk), axis=2)
    norm = np.linalg.norm(jerk, axis=2)[valid]
    vertical = jerk[:, :, 2][valid]
    lateral = np.linalg.norm(jerk[:, :, :2], axis=2)[valid]
    return {
        "jerk_sample_count": int(norm.size),
        "jerk_mean_squared": float(np.mean(norm**2)) if norm.size else float("nan"),
        "vertical_jerk_mean_squared": float(np.mean(vertical**2)) if norm.size else float("nan"),
        "lateral_jerk_mean_squared": float(np.mean(lateral**2)) if norm.size else float("nan"),
        "jerk_p90_mps3": float(np.percentile(norm, 90)) if norm.size else float("nan"),
        "jerk_p95_mps3": float(np.percentile(norm, 95)) if norm.size else float("nan"),
        "jerk_peak_mps3": float(np.max(norm)) if norm.size else float("nan"),
    }


def limiter_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    if "limiter_trace_file" not in record:
        return {
            "limiter_activation_rate": 0.0,
            "hard_bypass_count": 0,
            "mean_limiter_modification_mps2": 0.0,
            "p95_limiter_modification_mps2": 0.0,
            "limiter_runtime_ms": 0.0,
            "limiter_runtime_per_agent_step_us": 0.0,
        }
    trace_path = Path(record["json_path"]).with_name(str(record["limiter_trace_file"]))
    trace = np.load(trace_path)
    active = np.asarray(trace["limiter_active"], dtype=bool)
    bypass = np.asarray(trace["hard_bypass"], dtype=bool)
    modification = np.asarray(trace["limiter_modification_norm_mps2"], dtype=float)
    runtime_ns = np.asarray(trace["limiter_runtime_ns"], dtype=np.int64)
    return {
        "limiter_activation_rate": float(np.mean(active)) if active.size else 0.0,
        "hard_bypass_count": int(np.sum(bypass)),
        "mean_limiter_modification_mps2": float(np.mean(modification)) if modification.size else 0.0,
        "p95_limiter_modification_mps2": float(np.percentile(modification, 95)) if modification.size else 0.0,
        "limiter_runtime_ms": float(np.sum(runtime_ns) / 1.0e6),
        "limiter_runtime_per_agent_step_us": float(np.mean(runtime_ns) / 1.0e3) if runtime_ns.size else 0.0,
    }


def paired_bootstrap(values: np.ndarray, statistic, seed: int, repeats: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = np.empty(repeats, dtype=float)
    for index in range(repeats):
        take = rng.integers(0, len(values), size=len(values))
        sampled[index] = statistic(values[take])
    return tuple(float(value) for value in np.percentile(sampled, (2.5, 97.5)))


def percent_reduction(original: np.ndarray, candidate: np.ndarray) -> float:
    return 100.0 * (float(np.mean(original)) - float(np.mean(candidate))) / float(np.mean(original))


def analyze() -> None:
    if load_json(ARTIFACT_ROOT / "JERK_LIMITER_MICROTEST.json")["status"] != "PASS":
        raise RuntimeError("microtest must pass before analysis")
    records = {arm: load_records(arm) for arm in ARMS}
    expected = set(records["original"])
    if len(expected) != 100 or any(set(records[arm]) != expected for arm in ARMS):
        raise RuntimeError({arm: len(records[arm]) for arm in ARMS})
    metrics = {
        arm: {sid: trajectory_metrics(record) for sid, record in records[arm].items()}
        for arm in ARMS
    }
    rows: list[dict[str, Any]] = []
    gates: dict[str, dict[str, Any]] = {}
    original = records["original"]
    original_success = float(np.mean([row["episode"]["team_success"] for row in original.values()]))
    original_peer = float(np.mean([row["episode"]["inter_agent_collision"] for row in original.values()]))
    for arm in ARMS:
        arm_records = records[arm]
        all_episode = [arm_records[sid]["episode"] for sid in sorted(expected)]
        both = [
            sid for sid in sorted(expected)
            if original[sid]["episode"]["team_success"] and arm_records[sid]["episode"]["team_success"]
        ]
        original_smooth = np.asarray([original[sid]["episode"]["trajectory_smoothness"] for sid in both], dtype=float)
        arm_smooth = np.asarray([arm_records[sid]["episode"]["trajectory_smoothness"] for sid in both], dtype=float)
        original_p90 = np.asarray([metrics["original"][sid]["jerk_p90_mps3"] for sid in both], dtype=float)
        arm_p90 = np.asarray([metrics[arm][sid]["jerk_p90_mps3"] for sid in both], dtype=float)
        original_p95 = np.asarray([metrics["original"][sid]["jerk_p95_mps3"] for sid in both], dtype=float)
        arm_p95 = np.asarray([metrics[arm][sid]["jerk_p95_mps3"] for sid in both], dtype=float)
        success = float(np.mean([episode["team_success"] for episode in all_episode]))
        collision = float(np.mean([episode["collision"] for episode in all_episode]))
        peer = float(np.mean([episode["inter_agent_collision"] for episode in all_episode]))
        obstacle = float(np.mean([episode["obstacle_collision"] for episode in all_episode]))
        smooth_reduction = percent_reduction(original_smooth, arm_smooth) if arm != "original" else 0.0
        p90_change = float(np.mean(arm_p90 - original_p90)) if arm != "original" else 0.0
        p95_change = float(np.mean(arm_p95 - original_p95)) if arm != "original" else 0.0
        if arm == "original":
            smooth_ci = (0.0, 0.0)
            p90_ci = (0.0, 0.0)
            p95_ci = (0.0, 0.0)
        else:
            ratio_values = np.column_stack([original_smooth, arm_smooth])
            smooth_ci = paired_bootstrap(
                ratio_values,
                lambda sample: percent_reduction(sample[:, 0], sample[:, 1]),
                20260825 + list(ARMS).index(arm),
            )
            p90_ci = paired_bootstrap(
                arm_p90 - original_p90, np.mean,
                20260925 + list(ARMS).index(arm),
            )
            p95_ci = paired_bootstrap(
                arm_p95 - original_p95, np.mean,
                20261025 + list(ARMS).index(arm),
            )
        limiter = [limiter_metrics(arm_records[sid]) for sid in sorted(expected)]
        pooled_p90 = float(np.percentile(np.concatenate([
            np.asarray([metrics[arm][sid]["jerk_p90_mps3"]]) for sid in sorted(expected)
        ]), 90))
        fixed_positive = 0
        fixed_worse = 0
        fixed_details = []
        for sid in FIXED_SCENES:
            both_fixed = bool(original[sid]["episode"]["team_success"] and arm_records[sid]["episode"]["team_success"])
            if arm == "original":
                smooth_fixed = p95_fixed = 0.0
            elif both_fixed:
                smooth_fixed = 100.0 * (
                    float(original[sid]["episode"]["trajectory_smoothness"])
                    - float(arm_records[sid]["episode"]["trajectory_smoothness"])
                ) / float(original[sid]["episode"]["trajectory_smoothness"])
                p95_fixed = 100.0 * (
                    metrics["original"][sid]["jerk_p95_mps3"] - metrics[arm][sid]["jerk_p95_mps3"]
                ) / metrics["original"][sid]["jerk_p95_mps3"]
            else:
                smooth_fixed = p95_fixed = -100.0
            if arm != "original" and both_fixed and smooth_fixed >= 15.0 and p95_fixed >= 15.0:
                fixed_positive += 1
            if arm != "original" and (not both_fixed or smooth_fixed < -10.0 or p95_fixed < -10.0):
                fixed_worse += 1
            fixed_details.append({"scenario_id": sid, "both_success": both_fixed, "smoothness_reduction_percent": smooth_fixed, "p95_jerk_reduction_percent": p95_fixed})
        success_loss_pp = 100.0 * (original_success - success)
        peer_delta_pp = 100.0 * (peer - original_peer)
        reliability_gate = bool(success_loss_pp <= 1.0 + 1e-12)
        peer_gate = bool(peer_delta_pp <= 1.0 + 1e-12)
        smooth_gate = bool(arm != "original" and smooth_reduction >= 15.0)
        tail_gate = bool(arm != "original" and p90_change < 0.0 and p95_change < 0.0 and p90_ci[1] < 0.0 and p95_ci[1] < 0.0)
        visual_gate = bool(arm != "original" and fixed_positive >= 3 and fixed_worse == 0)
        pass_gate = bool(reliability_gate and peer_gate and smooth_gate and tail_gate and visual_gate)
        gates[arm] = {
            "pass": pass_gate,
            "reliability_gate": reliability_gate,
            "peer_safety_gate": peer_gate,
            "smoothness_gate": smooth_gate,
            "jerk_tail_gate": tail_gate,
            "fixed_visual_gate": visual_gate,
            "fixed_positive_scene_count": fixed_positive,
            "fixed_worse_scene_count": fixed_worse,
            "fixed_scene_details": fixed_details,
        }
        rows.append({
            "arm": arm,
            "n": len(all_episode),
            "team_success": success,
            "collision": collision,
            "obstacle_collision": obstacle,
            "peer_collision": peer,
            "timeout": float(np.mean([episode["timeout"] for episode in all_episode])),
            "agent_completion": float(np.mean([episode["agent_completion_rate"] for episode in all_episode])),
            "success_loss_pp_vs_original": success_loss_pp,
            "peer_collision_delta_pp_vs_original": peer_delta_pp,
            "both_success_n": len(both),
            "both_success_smoothness_reduction_percent": smooth_reduction,
            "smoothness_reduction_ci_low": smooth_ci[0],
            "smoothness_reduction_ci_high": smooth_ci[1],
            "both_success_jerk_p90_change_mps3": p90_change,
            "jerk_p90_change_ci_low": p90_ci[0],
            "jerk_p90_change_ci_high": p90_ci[1],
            "both_success_jerk_p95_change_mps3": p95_change,
            "jerk_p95_change_ci_low": p95_ci[0],
            "jerk_p95_change_ci_high": p95_ci[1],
            "mean_vertical_jerk_mse": float(np.mean([metrics[arm][sid]["vertical_jerk_mean_squared"] for sid in sorted(expected)])),
            "mean_lateral_jerk_mse": float(np.mean([metrics[arm][sid]["lateral_jerk_mean_squared"] for sid in sorted(expected)])),
            "mean_jerk_peak_mps3": float(np.mean([metrics[arm][sid]["jerk_peak_mps3"] for sid in sorted(expected)])),
            "mean_team_path_m_both_success": float(np.mean([arm_records[sid]["episode"]["team_path_length_m"] for sid in both])),
            "mean_completion_time_s_both_success": float(np.mean([arm_records[sid]["episode"]["completion_time_s"] for sid in both])),
            "mean_min_obstacle_clearance_m": float(np.mean([episode["minimum_obstacle_clearance_m"] for episode in all_episode])),
            "mean_min_peer_distance_m": float(np.mean([episode["minimum_inter_agent_distance_m"] for episode in all_episode])),
            "mean_total_online_compute_ms": float(np.mean([episode["total_online_algorithm_compute_ms"] for episode in all_episode])),
            "mean_limiter_activation_rate": float(np.mean([row["limiter_activation_rate"] for row in limiter])),
            "hard_bypass_count": int(np.sum([row["hard_bypass_count"] for row in limiter])),
            "mean_limiter_modification_mps2": float(np.mean([row["mean_limiter_modification_mps2"] for row in limiter])),
            "p95_limiter_modification_mps2": float(np.percentile([row["p95_limiter_modification_mps2"] for row in limiter], 95)),
            "mean_limiter_runtime_ms_episode": float(np.mean([row["limiter_runtime_ms"] for row in limiter])),
            "mean_limiter_runtime_us_agent_step": float(np.mean([row["limiter_runtime_per_agent_step_us"] for row in limiter])),
            "fixed_visual_positive_count": fixed_positive,
            "fixed_visual_worse_count": fixed_worse,
            "reliability_gate": reliability_gate,
            "peer_safety_gate": peer_gate,
            "smoothness_gate": smooth_gate,
            "jerk_tail_gate": tail_gate,
            "visual_gate": visual_gate,
            "overall_dev_gate": pass_gate,
        })
    passing = [arm for arm in ("mild", "medium", "strong") if gates[arm]["pass"]]
    selected = "NONE"
    if passing:
        reductions = {row["arm"]: float(row["both_success_smoothness_reduction_percent"]) for row in rows}
        best = max(reductions[arm] for arm in passing)
        comparable = [arm for arm in passing if best - reductions[arm] <= 2.0]
        selected = next(arm for arm in ("mild", "medium", "strong") if arm in comparable)
    failures: list[dict[str, Any]] = []
    for arm in ("mild", "medium", "strong"):
        for sid in sorted(expected):
            if not original[sid]["episode"]["team_success"] or records[arm][sid]["episode"]["team_success"]:
                continue
            episode = records[arm][sid]["episode"]
            trace_path = Path(records[arm][sid]["json_path"]).with_name(records[arm][sid]["limiter_trace_file"])
            trace = np.load(trace_path)
            last_step = int(np.max(trace["step"]))
            indices = np.flatnonzero(np.asarray(trace["step"]) == last_step)
            for index in indices:
                failures.append({
                    "arm": arm, "scenario_id": sid,
                    "stage": records[arm][sid]["entry_identity"]["stage"],
                    "family": records[arm][sid]["entry_identity"]["family"],
                    "termination_reason": episode["termination_reason"],
                    "steps": episode["steps"],
                    "obstacle_collision": episode["obstacle_collision"],
                    "peer_collision": episode["inter_agent_collision"],
                    "agent_id": int(trace["agent_id"][index]),
                    "last_control_step": last_step,
                    "safety_margin_m": float(trace["safety_margin_m"][index]),
                    "q_safe": float(trace["q_safe"][index]),
                    "hard_bypass": bool(trace["hard_bypass"][index]),
                    "limiter_active": bool(trace["limiter_active"][index]),
                    "raw_acceleration": np.asarray(trace["raw_acceleration"][index]).tolist(),
                    "limited_acceleration": np.asarray(trace["limited_acceleration"][index]).tolist(),
                    "previous_executed_acceleration": np.asarray(trace["previous_executed_acceleration"][index]).tolist(),
                    "executed_acceleration": np.asarray(trace["executed_acceleration"][index]).tolist(),
                    "modification_norm_mps2": float(trace["limiter_modification_norm_mps2"][index]),
                })
    write_csv(ARTIFACT_ROOT / "JERK_LIMITER_DEV_RESULTS.csv", rows)
    write_csv(
        ARTIFACT_ROOT / "JERK_LIMITER_FAILURE_AUDIT.csv", failures,
        fields=("arm", "scenario_id", "stage", "family", "termination_reason", "steps", "obstacle_collision", "peer_collision", "agent_id", "last_control_step", "safety_margin_m", "q_safe", "hard_bypass", "limiter_active", "raw_acceleration", "limited_acceleration", "previous_executed_acceleration", "executed_acceleration", "modification_norm_mps2"),
    )
    decision = {
        "schema_version": "safety_adaptive_jerk_limiter_development_decision_v1",
        "status": "PASS" if selected != "NONE" else "FAIL",
        "gates": gates,
        "passing_variants": passing,
        "selected_variant": selected,
        "selection_rule": "among passers, highest smoothness reduction; variants within 2 pp are comparable and use Mild>Medium>Strong",
        "holdout_authorized": selected != "NONE",
        "formal_v2_authorized": False,
        "contract_sha256": sha256_file(ARTIFACT_ROOT / "JERK_LIMITER_CONTROL_CONTRACT.json"),
        "thresholds_sha256": sha256_file(ARTIFACT_ROOT / "JERK_LIMITER_THRESHOLDS.json"),
        "microtest_sha256": sha256_file(ARTIFACT_ROOT / "JERK_LIMITER_MICROTEST.json"),
    }
    atomic_json(ARTIFACT_ROOT / "development_decision.json", decision)
    print(json.dumps(json_ready({"rows": rows, "decision": decision}), indent=2))


if __name__ == "__main__":
    analyze()
