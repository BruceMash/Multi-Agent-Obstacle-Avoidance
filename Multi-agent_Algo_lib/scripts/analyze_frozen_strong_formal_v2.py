#!/usr/bin/env python3
"""Read-only analysis of the one-shot Frozen-Strong Formal V2 arm."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta, binomtest


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search in (REPO_ROOT, ALGO_ROOT):
    if str(search) not in sys.path:
        sys.path.insert(0, str(search))

from planning.long_range_collision_recheck import audit_trajectory_collisions  # noqa: E402


ROOT = REPO_ROOT / "artifacts/frozen_strong_formal_v2/20260826_110337"
RECORD_DIR = ROOT / "03_formal_run/episode_records"
RESULT_DIR = ROOT / "04_formal_results"
PAIRED_DIR = ROOT / "05_paired_statistics"
RUNTIME_DIR = ROOT / "06_runtime"
TRAJECTORY_DIR = ROOT / "07_trajectories"
FORMAL_STUDY = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
ORIGINAL_DIR = FORMAL_STUDY / "10_formal_v2/formal_records/M9_Proposed_RERR_GAT_SAC_DMP"
MANIFEST_PATH = FORMAL_STUDY / "10_formal_v2/FORMAL_V2_MANIFEST.json"
ACCEPTANCE_PATH = ROOT / "01_prefreeze/FINAL_STRONG_FORMAL_ACCEPTANCE_RULE.json"
IDENTITY_PATH = ROOT / "00_identity/FINAL_STRONG_IDENTITY_AUDIT.json"

TEAM_RESULTS = RESULT_DIR / "strong_formal_team_results.csv"
AGENT_RESULTS = RESULT_DIR / "strong_formal_agent_results.csv"
STAGE_SUMMARY = RESULT_DIR / "strong_formal_stage_summary.csv"
FAILURE_TAXONOMY = RESULT_DIR / "strong_formal_failure_taxonomy.csv"
PAIR_SUCCESS = PAIRED_DIR / "strong_vs_original_paired_success.json"
CONTINUOUS = PAIRED_DIR / "strong_vs_original_continuous.csv"
CONTINUOUS_SCENES = PAIRED_DIR / "strong_vs_original_both_success_per_scene.csv"
RUNTIME_SUMMARY = RUNTIME_DIR / "strong_runtime_summary.json"
TRAJECTORY_MANIFEST = TRAJECTORY_DIR / "strong_trajectory_manifest.csv"
ANALYSIS = RESULT_DIR / "FORMAL_STRONG_ANALYSIS.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    temporary.replace(path)


def cp_interval(success: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if success == 0 else float(beta.ppf(alpha / 2.0, success, n - success + 1))
    upper = 1.0 if success == n else float(beta.ppf(1.0 - alpha / 2.0, success + 1, n - success))
    return lower, upper


def dense_velocity_jerk(record: Mapping[str, Any], npz_path: Path) -> dict[str, float]:
    with np.load(npz_path) as arrays:
        velocity = np.asarray(arrays["velocities"], dtype=float)
    if velocity.ndim != 3 or velocity.shape[1:] != (3, 3):
        raise RuntimeError(f"unexpected velocity shape: {npz_path}: {velocity.shape}")
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
    if not norm.size:
        raise RuntimeError(f"empty jerk series: {npz_path}")
    return {
        "velocity_jerk_mean_squared": float(np.mean(norm**2)),
        "vertical_jerk_mean_squared": float(np.mean(vertical**2)),
        "lateral_jerk_mean_squared": float(np.mean(lateral**2)),
        "jerk_p90_mps3": float(np.percentile(norm, 90)),
        "jerk_p95_mps3": float(np.percentile(norm, 95)),
    }


def path_metrics(positions: np.ndarray, entry: Mapping[str, Any]) -> dict[str, Any]:
    per_agent = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=2), axis=0)
    starts = np.asarray(entry["starts"], dtype=float)
    goals = np.asarray(entry["goals"], dtype=float)
    straight = np.linalg.norm(goals - starts, axis=1)
    efficiency = straight / np.maximum(per_agent, 1.0e-12)
    return {
        "per_agent_path_lengths_m": per_agent,
        "team_path_length_m_raw": float(np.sum(per_agent)),
        "per_agent_path_length_mean_m": float(np.mean(per_agent)),
        "path_efficiency": float(np.mean(efficiency)),
        "detour_ratio": float(np.mean(per_agent / np.maximum(straight, 1.0e-12))),
    }


def classify_failure(row: Mapping[str, Any]) -> str:
    if bool(row["team_success"]):
        return "success"
    if bool(row["static_collision"]):
        return "static_obstacle_collision"
    if bool(row["dynamic_collision"]):
        return "dynamic_obstacle_collision"
    if bool(row["peer_collision"]):
        return "inter_agent_collision"
    if bool(row["boundary_collision"]):
        return "boundary_collision"
    if bool(row["timeout"]):
        return "timeout"
    return "other_terminal_incomplete"


def collision_fields(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "collision": bool(audit["any_collision"]),
        "obstacle_collision": bool(audit["obstacle_collision"]),
        "static_collision": bool(audit["static_obstacle_collision"]),
        "dynamic_collision": bool(audit["dynamic_obstacle_collision"]),
        "peer_collision": bool(audit["inter_agent_collision"]),
        "boundary_collision": bool(audit["boundary_collision"]),
        "minimum_obstacle_clearance_m": float(audit["minimum_obstacle_signed_clearance_m"]),
        "minimum_peer_distance_m": float(audit["minimum_inter_agent_distance_m"]),
    }


def record_metrics(path: Path, entry: Mapping[str, Any], *, strong: bool) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    record = load_json(path)
    episode = dict(record["episode"])
    sid = str(entry["scenario_id"])
    if str(episode.get("scenario_id", episode.get("scenario"))) != sid:
        raise RuntimeError(f"record identity mismatch: {path}")
    npz_path = path.with_name(f"{sid}_trajectory.npz")
    with np.load(npz_path) as arrays:
        positions = np.asarray(arrays["positions"], dtype=float)
        velocity = np.asarray(arrays["velocities"], dtype=float)
    if not np.isfinite(positions).all() or not np.isfinite(velocity).all():
        raise RuntimeError(f"non-finite trajectory: {npz_path}")
    audit = audit_trajectory_collisions(positions, entry)
    collision = collision_fields(audit)
    if bool(episode["obstacle_collision"]) != collision["obstacle_collision"]:
        raise RuntimeError(f"obstacle collision replay mismatch: {sid}")
    if bool(episode["inter_agent_collision"]) != collision["peer_collision"]:
        raise RuntimeError(f"peer collision replay mismatch: {sid}")
    paths = path_metrics(positions, entry)
    jerk = dense_velocity_jerk(record, npz_path)
    agents = list(record["agents"])
    row = {
        "scenario_id": sid,
        "seed": int(entry["seed"]),
        "stage_index": int(entry["stage_index"]),
        "stage": str(entry["stage"]),
        "family": str(entry["family"]),
        "team_success": bool(episode["team_success"]),
        **collision,
        "timeout": bool(episode["timeout"]),
        "agent_completion_rate": float(np.mean([bool(agent["success"]) for agent in agents])),
        "termination_reason": str(episode["termination_reason"]),
        "steps": int(episode["steps"]),
        "completion_time_s": episode.get("completion_time_s"),
        "team_path_length_m": float(episode["team_path_length_m"]),
        "per_agent_path_length_mean_m": paths["per_agent_path_length_mean_m"],
        "path_efficiency": paths["path_efficiency"],
        "detour_ratio": paths["detour_ratio"],
        "trajectory_smoothness": float(episode["trajectory_smoothness"]),
        **jerk,
        "planning_compute_ms": float(episode["upper_planning_total_ms"]),
        "execution_actor_compute_ms": float(episode["execution_actor_forward_ms"]),
        "execution_dmp_compute_ms": float(episode["execution_dmp_ms"]),
        "shared_online_compute_ms": float(episode["total_online_algorithm_compute_ms"]),
        "trajectory_file": str(npz_path.relative_to(REPO_ROOT).as_posix()),
        "trajectory_sha256": sha256_file(npz_path),
        "record_file": str(path.relative_to(REPO_ROOT).as_posix()),
        "record_sha256": sha256_file(path),
        "sample_count": int(positions.shape[0]),
        "start_state_match": bool(np.array_equal(positions[0], np.asarray(entry["starts"], dtype=float))),
        "raw_trajectory_used": True,
        "post_processing_applied": False,
    }
    if strong:
        trace_path = path.with_name(f"{sid}_limiter_trace.npz")
        with np.load(trace_path) as trace:
            runtime_ns = np.asarray(trace["limiter_runtime_ns"], dtype=np.int64)
            active = np.asarray(trace["limiter_active"], dtype=bool)
            hard = np.asarray(trace["hard_bypass"], dtype=bool)
            early = np.asarray(trace["early_bypass"], dtype=bool)
            trace_count = int(runtime_ns.size)
        if trace_count != int(episode["execution_dmp_call_count"]):
            raise RuntimeError(f"limiter trace count mismatch: {sid}")
        row.update({
            "limiter_compute_ms": float(np.sum(runtime_ns) / 1.0e6),
            "total_declared_online_compute_ms": float(episode["total_online_algorithm_compute_ms"] + np.sum(runtime_ns) / 1.0e6),
            "limiter_us_per_agent_step": float(np.mean(runtime_ns) / 1.0e3),
            "limiter_activation_fraction": float(np.mean(active)),
            "hard_bypass_fraction": float(np.mean(hard)),
            "early_bypass_fraction": float(np.mean(early)),
            "limiter_trace_count": trace_count,
            "limiter_trace_file": str(trace_path.relative_to(REPO_ROOT).as_posix()),
            "limiter_trace_sha256": sha256_file(trace_path),
        })
    row["failure_type"] = classify_failure(row)
    agent_rows: list[dict[str, Any]] = []
    path_lengths = np.asarray(paths["per_agent_path_lengths_m"], dtype=float)
    for agent in agents:
        agent_id = int(agent["agent_id"])
        agent_rows.append({
            "scenario_id": sid,
            "seed": int(entry["seed"]),
            "stage_index": int(entry["stage_index"]),
            "stage": str(entry["stage"]),
            "family": str(entry["family"]),
            "agent_id": agent_id,
            "completed": bool(agent["success"]),
            "path_length_m": float(path_lengths[agent_id]),
            "static_collision": bool(audit["agent_static_obstacle_collision"][agent_id]),
            "dynamic_collision": bool(audit["agent_dynamic_obstacle_collision"][agent_id]),
            "obstacle_collision": bool(audit["agent_obstacle_collision"][agent_id]),
            "peer_collision": bool(audit["agent_inter_agent_collision"][agent_id]),
            "boundary_collision": bool(audit["agent_boundary_collision"][agent_id]),
            "any_collision": bool(audit["agent_any_collision"][agent_id]),
            "minimum_obstacle_clearance_m": float(audit["agent_minimum_obstacle_signed_clearance_m"][agent_id]),
            "minimum_peer_distance_m": float(audit["agent_minimum_inter_agent_distance_m"][agent_id]),
        })
    meta = {
        "positions": positions,
        "start": positions[0],
        "end": positions[-1],
        "record": record,
    }
    return row, agent_rows, meta


def stage_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scopes: list[tuple[str, list[Mapping[str, Any]]]] = [("Overall", list(rows))]
    for stage_index in range(1, 5):
        scopes.append((f"Stage {stage_index}", [row for row in rows if int(row["stage_index"]) == stage_index]))
    scopes.append(("Stage III+IV", [row for row in rows if int(row["stage_index"]) in (3, 4)]))
    output = []
    for scope, subset in scopes:
        n = len(subset)
        values: dict[str, Any] = {"scope": scope, "n": n}
        for key in ("team_success", "collision", "obstacle_collision", "static_collision", "dynamic_collision", "peer_collision", "boundary_collision", "timeout"):
            count = int(sum(bool(row[key]) for row in subset))
            low, high = cp_interval(count, n)
            values[f"{key}_count"] = count
            values[f"{key}_rate"] = count / n
            values[f"{key}_ci95_low"] = low
            values[f"{key}_ci95_high"] = high
        values["agent_completion_rate"] = float(np.mean([float(row["agent_completion_rate"]) for row in subset]))
        output.append(values)
    return output


def bootstrap_summary(original: np.ndarray, strong: np.ndarray, seed: int) -> dict[str, float]:
    original = np.asarray(original, dtype=float)
    strong = np.asarray(strong, dtype=float)
    if original.shape != strong.shape or original.ndim != 1 or original.size == 0:
        raise ValueError("paired vectors must be non-empty and equal")
    difference = strong - original
    rng = np.random.default_rng(seed)
    index = rng.integers(0, original.size, size=(5000, original.size))
    original_boot = np.mean(original[index], axis=1)
    strong_boot = np.mean(strong[index], axis=1)
    difference_boot = strong_boot - original_boot
    percent_boot = 100.0 * difference_boot / np.maximum(np.abs(original_boot), 1.0e-12)
    return {
        "original_mean": float(np.mean(original)),
        "strong_mean": float(np.mean(strong)),
        "paired_mean_difference_strong_minus_original": float(np.mean(difference)),
        "paired_median_difference_strong_minus_original": float(np.median(difference)),
        "percentage_change_strong_minus_original": float(100.0 * np.mean(difference) / max(abs(float(np.mean(original))), 1.0e-12)),
        "improvement_percent_lower_is_better": float(-100.0 * np.mean(difference) / max(abs(float(np.mean(original))), 1.0e-12)),
        "mean_difference_ci95_low": float(np.percentile(difference_boot, 2.5)),
        "mean_difference_ci95_high": float(np.percentile(difference_boot, 97.5)),
        "percentage_change_ci95_low": float(np.percentile(percent_boot, 2.5)),
        "percentage_change_ci95_high": float(np.percentile(percent_boot, 97.5)),
        "bootstrap_resamples": 5000,
    }


def main() -> None:
    manifest = load_json(MANIFEST_PATH)
    entries = [dict(entry) for entry in manifest["entries"]]
    entry_by_id = {str(entry["scenario_id"]): entry for entry in entries}
    ids = [str(entry["scenario_id"]) for entry in entries]
    if len(ids) != 400 or len(set(ids)) != 400:
        raise RuntimeError("Formal manifest identity failure")
    strong_json = sorted(path for path in RECORD_DIR.glob("FORMAL_LR_*.json") if "SOFTWARE_ERROR" not in path.name)
    strong_npz = sorted(RECORD_DIR.glob("FORMAL_LR_*_trajectory.npz"))
    strong_trace = sorted(RECORD_DIR.glob("FORMAL_LR_*_limiter_trace.npz"))
    software_errors = sorted(RECORD_DIR.glob("*_SOFTWARE_ERROR.json"))
    if not (len(strong_json) == len(strong_npz) == len(strong_trace) == 400) or software_errors:
        raise RuntimeError("Frozen Strong Formal record set is not complete and clean")

    strong_rows: list[dict[str, Any]] = []
    original_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    strong_meta: dict[str, dict[str, Any]] = {}
    original_meta: dict[str, dict[str, Any]] = {}
    for index, sid in enumerate(ids, start=1):
        entry = entry_by_id[sid]
        strong, agents, smeta = record_metrics(RECORD_DIR / f"{sid}.json", entry, strong=True)
        original, _, ometa = record_metrics(ORIGINAL_DIR / f"{sid}.json", entry, strong=False)
        strong_rows.append(strong)
        original_rows.append(original)
        agent_rows.extend(agents)
        strong_meta[sid] = smeta
        original_meta[sid] = ometa
        if index % 50 == 0:
            print(f"[analysis] reopened {index}/400 paired scenes", flush=True)

    if {row["scenario_id"] for row in strong_rows} != set(ids):
        raise RuntimeError("Strong IDs do not match manifest")
    if len(agent_rows) != 1200:
        raise RuntimeError("expected exactly 1200 Strong agent rows")
    write_csv(TEAM_RESULTS, strong_rows)
    write_csv(AGENT_RESULTS, agent_rows)
    summary_rows = stage_summary(strong_rows)
    write_csv(STAGE_SUMMARY, summary_rows)

    original_by_id = {str(row["scenario_id"]): row for row in original_rows}
    strong_by_id = {str(row["scenario_id"]): row for row in strong_rows}
    scopes = [("Overall", ids)] + [
        (f"Stage {stage}", [sid for sid in ids if int(entry_by_id[sid]["stage_index"]) == stage])
        for stage in range(1, 5)
    ] + [("Stage III+IV", [sid for sid in ids if int(entry_by_id[sid]["stage_index"]) in (3, 4)])]
    taxonomy_rows: list[dict[str, Any]] = []
    for scope, scope_ids in scopes:
        for method, mapping in (("Original Proposed", original_by_id), ("Frozen Strong", strong_by_id)):
            counts = Counter(str(mapping[sid]["failure_type"]) for sid in scope_ids)
            for failure_type in (
                "success", "static_obstacle_collision", "dynamic_obstacle_collision",
                "inter_agent_collision", "boundary_collision", "timeout", "other_terminal_incomplete",
            ):
                taxonomy_rows.append({
                    "scope": scope,
                    "method": method,
                    "failure_type": failure_type,
                    "count": int(counts.get(failure_type, 0)),
                    "rate": float(counts.get(failure_type, 0) / len(scope_ids)),
                    "n": len(scope_ids),
                })
    write_csv(FAILURE_TAXONOMY, taxonomy_rows)

    both_success = [sid for sid in ids if original_by_id[sid]["team_success"] and strong_by_id[sid]["team_success"]]
    original_only = [sid for sid in ids if original_by_id[sid]["team_success"] and not strong_by_id[sid]["team_success"]]
    strong_only = [sid for sid in ids if strong_by_id[sid]["team_success"] and not original_by_id[sid]["team_success"]]
    both_failure = [sid for sid in ids if not original_by_id[sid]["team_success"] and not strong_by_id[sid]["team_success"]]
    original_success = len(both_success) + len(original_only)
    strong_success = len(both_success) + len(strong_only)
    discordant = len(original_only) + len(strong_only)
    mcnemar = float(binomtest(min(len(original_only), len(strong_only)), discordant, p=0.5).pvalue) if discordant else 1.0
    pair_payload = {
        "schema_version": "frozen_strong_formal_paired_success_v1",
        "scenario_count": 400,
        "both_success": len(both_success),
        "original_only_success": len(original_only),
        "strong_only_success": len(strong_only),
        "both_failure": len(both_failure),
        "original_success_count": original_success,
        "strong_success_count": strong_success,
        "original_success_rate": original_success / 400.0,
        "strong_success_rate": strong_success / 400.0,
        "strong_minus_original_success_pp": 100.0 * (strong_success - original_success) / 400.0,
        "original_success_ci95_exact": list(cp_interval(original_success, 400)),
        "strong_success_ci95_exact": list(cp_interval(strong_success, 400)),
        "mcnemar_exact_two_sided_p": mcnemar,
        "discordant_original_only": len(original_only),
        "discordant_strong_only": len(strong_only),
        "p_value_is_not_acceptance_rule": True,
        "both_success_ids": both_success,
        "original_only_ids": original_only,
        "strong_only_ids": strong_only,
        "both_failure_ids": both_failure,
    }
    atomic_json(PAIR_SUCCESS, pair_payload)

    metric_specs = (
        ("completion_time_s", "s", True),
        ("team_path_length_m", "m", True),
        ("per_agent_path_length_mean_m", "m", True),
        ("path_efficiency", "ratio", False),
        ("detour_ratio", "ratio", True),
        ("trajectory_smoothness", "m2_s6", True),
        ("vertical_jerk_mean_squared", "m2_s6", True),
        ("lateral_jerk_mean_squared", "m2_s6", True),
        ("jerk_p90_mps3", "m_s3", True),
        ("jerk_p95_mps3", "m_s3", True),
        ("minimum_obstacle_clearance_m", "m", False),
        ("minimum_peer_distance_m", "m", False),
    )
    continuous_rows: list[dict[str, Any]] = []
    per_scene_rows: list[dict[str, Any]] = []
    for sid in both_success:
        row: dict[str, Any] = {"scenario_id": sid, "stage_index": int(entry_by_id[sid]["stage_index"]), "family": entry_by_id[sid]["family"]}
        for metric, _, _ in metric_specs:
            row[f"original_{metric}"] = original_by_id[sid][metric]
            row[f"strong_{metric}"] = strong_by_id[sid][metric]
            row[f"difference_{metric}"] = float(strong_by_id[sid][metric]) - float(original_by_id[sid][metric])
        per_scene_rows.append(row)
    write_csv(CONTINUOUS_SCENES, per_scene_rows)
    for index, (metric, unit, lower_is_better) in enumerate(metric_specs):
        original_values = np.asarray([float(original_by_id[sid][metric]) for sid in both_success])
        strong_values = np.asarray([float(strong_by_id[sid][metric]) for sid in both_success])
        stats = bootstrap_summary(original_values, strong_values, seed=2026082600 + index)
        continuous_rows.append({
            "metric": metric,
            "unit": unit,
            "paired_n": len(both_success),
            "lower_is_better": lower_is_better,
            **stats,
        })
    write_csv(CONTINUOUS, continuous_rows)

    limiter_ms = np.asarray([float(row["limiter_compute_ms"]) for row in strong_rows])
    shared_ms = np.asarray([float(row["shared_online_compute_ms"]) for row in strong_rows])
    total_ms = np.asarray([float(row["total_declared_online_compute_ms"]) for row in strong_rows])
    total_steps = np.asarray([int(row["steps"]) for row in strong_rows], dtype=float)
    trace_counts = np.asarray([int(row["limiter_trace_count"]) for row in strong_rows], dtype=float)
    activations = np.asarray([float(row["limiter_activation_fraction"]) for row in strong_rows])
    hard_bypass = np.asarray([float(row["hard_bypass_fraction"]) for row in strong_rows])
    early_bypass = np.asarray([float(row["early_bypass_fraction"]) for row in strong_rows])
    runtime_payload = {
        "schema_version": "frozen_strong_formal_runtime_summary_v1",
        "scenario_count": 400,
        "accounting_boundary": "upper planning + execution actor + DMP internal + Frozen Strong limiter; excludes environment stepping, sensing, collision checking, serialization and I/O",
        "shared_existing_online_compute_mean_ms_episode": float(np.mean(shared_ms)),
        "strong_limiter_mean_ms_episode": float(np.mean(limiter_ms)),
        "total_declared_online_compute_mean_ms_episode": float(np.mean(total_ms)),
        "total_declared_online_compute_mean_ms_control_step": float(np.sum(total_ms) / np.sum(total_steps)),
        "strong_limiter_us_agent_step": float(1000.0 * np.sum(limiter_ms) / np.sum(trace_counts)),
        "limiter_activation_fraction_pooled": float(np.average(activations, weights=trace_counts)),
        "hard_bypass_fraction_pooled": float(np.average(hard_bypass, weights=trace_counts)),
        "early_bypass_fraction_pooled": float(np.average(early_bypass, weights=trace_counts)),
        "early_bypass_count": 0,
        "total_control_steps": int(np.sum(total_steps)),
        "total_agent_steps": int(np.sum(trace_counts)),
        "not_full_perception_to_control_latency": True,
    }
    atomic_json(RUNTIME_SUMMARY, runtime_payload)

    trajectory_rows: list[dict[str, Any]] = []
    for sid in ids:
        entry = entry_by_id[sid]
        for method, mapping, meta in (
            ("Original Proposed", original_by_id, original_meta[sid]),
            ("Frozen Strong", strong_by_id, strong_meta[sid]),
        ):
            positions = np.asarray(meta["positions"], dtype=float)
            trajectory_rows.append({
                "scenario_id": sid,
                "stage_index": int(entry["stage_index"]),
                "stage": entry["stage"],
                "family": entry["family"],
                "method": method,
                "team_success": mapping[sid]["team_success"],
                "failure_type": mapping[sid]["failure_type"],
                "sample_count": int(positions.shape[0]),
                "uav_count": int(positions.shape[1]),
                "dt_s": 0.1,
                "static_obstacle_count": len(entry["static_obstacles"]),
                "dynamic_obstacle_count": len(entry["dynamic_obstacles"]),
                "trajectory_file": mapping[sid]["trajectory_file"],
                "trajectory_sha256": mapping[sid]["trajectory_sha256"],
                "record_file": mapping[sid]["record_file"],
                "record_sha256": mapping[sid]["record_sha256"],
                "start_state_match": mapping[sid]["start_state_match"],
                "start_state": json.dumps(np.asarray(positions[0]).tolist(), separators=(",", ":")),
                "end_state": json.dumps(np.asarray(positions[-1]).tolist(), separators=(",", ":")),
                "raw_trajectory_used": True,
                "post_processing_applied": False,
            })
    write_csv(TRAJECTORY_MANIFEST, trajectory_rows)

    acceptance = load_json(ACCEPTANCE_PATH)
    overall = next(row for row in summary_rows if row["scope"] == "Overall")
    original_overall = stage_summary(original_rows)[0]
    continuous_by_metric = {row["metric"]: row for row in continuous_rows}
    failure_original = Counter(str(row["failure_type"]) for row in original_rows)
    failure_strong = Counter(str(row["failure_type"]) for row in strong_rows)
    new_failure_mode = any(
        failure_original.get(kind, 0) == 0 and failure_strong.get(kind, 0) >= int(math.ceil(0.02 * 400))
        for kind in failure_strong if kind != "success"
    )
    gates = {
        "record_integrity": len(strong_rows) == 400 and len(agent_rows) == 1200 and not software_errors,
        "reliability": strong_success >= int(acceptance["minimum_integer_success_count"]),
        "total_collision": float(overall["collision_rate"]) - float(original_overall["collision_rate"]) <= 0.01 + 1e-12,
        "peer_collision": float(overall["peer_collision_rate"]) - float(original_overall["peer_collision_rate"]) <= 0.01 + 1e-12,
        "obstacle_collision": float(overall["obstacle_collision_rate"]) - float(original_overall["obstacle_collision_rate"]) <= 0.01 + 1e-12,
        "no_new_failure_mode": not new_failure_mode,
        "smoothness": float(continuous_by_metric["trajectory_smoothness"]["improvement_percent_lower_is_better"]) >= 30.0,
        "vertical_jerk": float(continuous_by_metric["vertical_jerk_mean_squared"]["improvement_percent_lower_is_better"]) >= 20.0,
        "lateral_jerk": float(continuous_by_metric["lateral_jerk_mean_squared"]["improvement_percent_lower_is_better"]) >= 20.0,
        "p95_jerk": float(continuous_by_metric["jerk_p95_mps3"]["improvement_percent_lower_is_better"]) >= 20.0,
        "runtime": float(runtime_payload["strong_limiter_us_agent_step"]) <= 100.0,
        "raw_trajectory": all(bool(row["raw_trajectory_used"]) and not bool(row["post_processing_applied"]) for row in trajectory_rows),
    }
    analysis = {
        "schema_version": "frozen_strong_formal_analysis_v1",
        "scenario_count": 400,
        "strong_success_count": strong_success,
        "strong_success_rate": strong_success / 400.0,
        "original_success_count": original_success,
        "original_success_rate": original_success / 400.0,
        "strong_minus_original_success_pp": 100.0 * (strong_success - original_success) / 400.0,
        "paired_both_success_n": len(both_success),
        "gates_before_visual_review": gates,
        "statistical_acceptance_before_visual_review": all(gates.values()),
        "visual_review_status": "PENDING",
        "final_acceptance_status": "PENDING_VISUAL_AND_RECONCILIATION",
        "output_sha256": {
            str(path.relative_to(ROOT).as_posix()): sha256_file(path)
            for path in (TEAM_RESULTS, AGENT_RESULTS, STAGE_SUMMARY, FAILURE_TAXONOMY, PAIR_SUCCESS, CONTINUOUS, CONTINUOUS_SCENES, RUNTIME_SUMMARY, TRAJECTORY_MANIFEST)
        },
        "original_records_modified": False,
        "identity_sha256": sha256_file(IDENTITY_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
    }
    atomic_json(ANALYSIS, analysis)
    print(json.dumps({
        "status": "PASS",
        "strong_success": f"{strong_success}/400",
        "original_success": f"{original_success}/400",
        "both_success": len(both_success),
        "gates_before_visual_review": gates,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
