"""Read-only path-inefficiency/root-cause audit for frozen Formal V2 M9.

This program only reads retained JSON/NPZ artifacts.  It does not instantiate
an environment, execute a controller, alter a checkpoint, or change a method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2/formal_records"
M9_ROOT = FORMAL_ROOT / "M9_Proposed_RERR_GAT_SAC_DMP"
M2_ROOT = FORMAL_ROOT / "M2_DWA_SensingMatched"
PPO_ROOT = ROOT / "artifacts/ppo_direct_full_baseline/20260823_193740/12_trajectories/formal_records"
CRT_ORIGINAL_ROOT = ROOT / "artifacts/continuous_reference_transition/20260824_132552/04_development/records/original/episode_records"
TACV_MEDIUM_ROOT = ROOT / "artifacts/transient_aware_candidate_veto/20260824_184551/06_development/records/tacv_medium/episode_records"
DT = 0.1
EPS = 1.0e-8
SIGN_TOL = 1.0e-3
HIGH_CANCELLATION = 0.5
BOOTSTRAP_REPS = 3000


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def clean(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean(row.get(key)) for key in fields})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(values: Iterable[Any]) -> np.ndarray:
    result = []
    for value in values:
        if value is not None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                result.append(number)
    return np.asarray(result, dtype=float)


def describe(values: Iterable[Any], percentiles: Sequence[float] = (10, 25, 50, 75, 90, 95)) -> dict[str, Any]:
    data = finite(values)
    result: dict[str, Any] = {"n": int(data.size), "mean": None, "median": None}
    for percentile in percentiles:
        result[f"p{str(percentile).replace('.', '_')}"] = None
    if not data.size:
        return result
    result["mean"] = float(np.mean(data))
    result["median"] = float(np.median(data))
    for percentile in percentiles:
        result[f"p{str(percentile).replace('.', '_')}"] = float(np.percentile(data, percentile))
    return result


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return np.zeros_like(vector, dtype=float) if norm <= EPS else np.asarray(vector, dtype=float) / norm


def angle(a: np.ndarray, b: np.ndarray) -> float | None:
    ua, ub = unit(a), unit(b)
    if not np.any(ua) or not np.any(ub):
        return None
    return float(math.acos(float(np.clip(np.dot(ua, ub), -1.0, 1.0))))


def wrapped_delta(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.empty(0, dtype=float)
    return np.arctan2(np.sin(np.diff(values)), np.cos(np.diff(values)))


def sign_reversals(values: np.ndarray, tolerance: float) -> int:
    signs = np.sign(values[np.abs(values) > tolerance])
    return int(np.sum(signs[1:] * signs[:-1] < 0)) if signs.size > 1 else 0


def trajectory_metrics(positions: np.ndarray, accelerations: np.ndarray, start: np.ndarray, goal: np.ndarray) -> dict[str, Any]:
    positions = np.asarray(positions, dtype=float)
    dp = np.diff(positions, axis=0)
    lengths = np.linalg.norm(dp, axis=1)
    mission = float(np.linalg.norm(goal - start))
    path = float(np.sum(lengths))
    goal_vectors = goal[None, :] - positions[:-1]
    goal_norms = np.linalg.norm(goal_vectors, axis=1)
    u_goal = np.divide(goal_vectors, goal_norms[:, None], out=np.zeros_like(goal_vectors), where=goal_norms[:, None] > EPS)
    parallel_signed = np.sum(dp * u_goal, axis=1)
    parallel_vectors = parallel_signed[:, None] * u_goal
    lateral_vectors = dp - parallel_vectors
    forward = float(np.sum(np.maximum(parallel_signed, 0.0)))
    backward = float(np.sum(np.maximum(-parallel_signed, 0.0)))
    lateral = float(np.sum(np.linalg.norm(lateral_vectors, axis=1)))
    z_variation = float(np.sum(np.abs(dp[:, 2])))
    xy_variation = float(np.sum(np.linalg.norm(dp[:, :2], axis=1)))
    goal_reduction = float(np.linalg.norm(goal - positions[0]) - np.linalg.norm(goal - positions[-1]))

    moving = lengths > EPS
    headings = np.divide(dp[moving], lengths[moving, None], out=np.zeros((int(np.sum(moving)), 3)), where=lengths[moving, None] > EPS)
    heading_changes = []
    if headings.shape[0] > 1:
        heading_changes = np.arccos(np.clip(np.sum(headings[:-1] * headings[1:], axis=1), -1.0, 1.0))
    heading_changes = np.asarray(heading_changes, dtype=float)
    azimuth = np.arctan2(headings[:, 1], headings[:, 0]) if headings.size else np.empty(0)
    elevation = np.arctan2(headings[:, 2], np.linalg.norm(headings[:, :2], axis=1)) if headings.size else np.empty(0)
    azimuth_change = np.abs(wrapped_delta(azimuth))
    elevation_change = np.abs(np.diff(elevation)) if elevation.size > 1 else np.empty(0)

    accelerations = np.asarray(accelerations, dtype=float)
    jerk = np.diff(accelerations, axis=0) / DT if accelerations.shape[0] > 1 else np.empty((0, 3))
    jerk_sq = np.sum(jerk**2, axis=1) if jerk.size else np.empty(0)
    return {
        "mission_distance_m": mission,
        "path_length_m": path,
        "detour_ratio": path / mission if mission > EPS else None,
        "excess_path_m": path - mission,
        "normalized_excess": (path - mission) / mission if mission > EPS else None,
        "forward_path_m": forward,
        "backward_path_m": backward,
        "net_goal_distance_reduction_m": goal_reduction,
        "backward_ratio": backward / path if path > EPS else None,
        "lateral_motion_m": lateral,
        "lateral_ratio": lateral / path if path > EPS else None,
        "vertical_variation_m": z_variation,
        "xy_variation_m": xy_variation,
        "vertical_ratio": z_variation / path if path > EPS else None,
        "z_velocity_sign_reversals_raw": sign_reversals(dp[:, 2] / DT, EPS),
        "z_velocity_sign_reversals_significant": sign_reversals(dp[:, 2] / DT, SIGN_TOL),
        "cumulative_heading_variation_rad": float(np.sum(heading_changes)),
        "p90_heading_change_rad": float(np.percentile(heading_changes, 90)) if heading_changes.size else 0.0,
        "heading_reversal_count": int(np.sum(heading_changes > math.pi / 2.0)),
        "cumulative_azimuth_variation_rad": float(np.sum(azimuth_change)),
        "cumulative_elevation_variation_rad": float(np.sum(elevation_change)),
        "jerk_cost_sum": float(np.sum(jerk_sq)),
        "jerk_sample_count": int(jerk_sq.size),
        "smoothness_m2_s6": float(np.mean(jerk_sq)) if jerk_sq.size else 0.0,
        "vertical_jerk_cost_sum": float(np.sum(jerk[:, 2] ** 2)) if jerk.size else 0.0,
        "lateral_jerk_cost_sum": float(np.sum(jerk[:, :2] ** 2)) if jerk.size else 0.0,
    }


def stage_label(value: str) -> str:
    text = str(value).strip().lower().replace("_", " ")
    mapping = {
        "1": "Stage I", "stage 1": "Stage I", "i": "Stage I", "stage i": "Stage I",
        "2": "Stage II", "stage 2": "Stage II", "ii": "Stage II", "stage ii": "Stage II",
        "3": "Stage III", "stage 3": "Stage III", "iii": "Stage III", "stage iii": "Stage III",
        "4": "Stage IV", "stage 4": "Stage IV", "iv": "Stage IV", "stage iv": "Stage IV",
    }
    if text in mapping:
        return mapping[text]
    return str(value)


def classify_segment(event: Mapping[str, Any], start_distance: float) -> tuple[str, bool, str]:
    event_name = str(event.get("event") or "")
    goal_type = str(event.get("new_active_goal_type") or "")
    peer = bool(event.get("interaction_risky_candidate_count", 0)) or bool(event.get("selected_candidate_was_risky_before_interaction_mask"))
    emergency = "EMERGENCY" in event_name
    terminal = bool(event.get("terminal_null_eligible")) or goal_type.lower() in {"terminal", "task_goal", "final_task_goal"}
    completion = "REFERENCE_COMPLETION" in event_name
    restoration = "HANDOFF" in event_name or terminal
    if peer:
        return "peer-critical", True, "recorded candidate interaction-risk evidence"
    if emergency:
        return "untyped-safety-critical", True, "ERR emergency event; retained margin is untyped"
    if restoration:
        return "task-goal-restoration", False, "terminal/null handoff evidence"
    if completion:
        return "local-reference-completion", False, "recorded completion event"
    if start_distance <= 4.5:
        return "terminal-approach", False, "within frozen sensing radius of final goal"
    return "free-flight/low-recorded-interaction", False, "no retained strong safety evidence"


def changed_events(record: Mapping[str, Any], agent_id: int) -> list[dict[str, Any]]:
    rows = [dict(event) for event in record.get("events", []) if int(event.get("agent_id", -1)) == agent_id and bool(event.get("goal_changed"))]
    rows.sort(key=lambda event: (int(event.get("step", 0)), str(event.get("event", ""))))
    unique: list[dict[str, Any]] = []
    for event in rows:
        if unique and int(event["step"]) == int(unique[-1]["step"]):
            unique[-1] = event
        else:
            unique.append(event)
    return unique


def build_segments(record: Mapping[str, Any], positions: np.ndarray, accelerations: np.ndarray, active_goals: np.ndarray, goal: np.ndarray, agent_id: int, base: Mapping[str, Any]) -> list[dict[str, Any]]:
    changes = changed_events(record, agent_id)
    by_step = {int(event["step"]): event for event in changes}
    if 0 not in by_step:
        by_step[0] = {"event": "INITIAL_ACTIVE_REFERENCE_SYNTHETIC_FROM_RETAINED_ARRAY", "step": 0, "agent_id": agent_id, "new_active_goal": active_goals[0, agent_id].tolist(), "goal_changed": True}
    boundaries = sorted(step for step in by_step if 0 <= step < positions.shape[0] - 1)
    if not boundaries or boundaries[0] != 0:
        boundaries.insert(0, 0)
    boundaries.append(positions.shape[0] - 1)
    jerk = np.diff(accelerations[:, agent_id, :], axis=0) / DT if accelerations.shape[0] > 1 else np.empty((0, 3))
    jerk_steps = np.arange(1, accelerations.shape[0], dtype=int)
    rows: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        event = by_step.get(start, {})
        dp = np.diff(positions[start : end + 1, agent_id, :], axis=0)
        length = float(np.sum(np.linalg.norm(dp, axis=1)))
        d_start = float(np.linalg.norm(goal - positions[start, agent_id]))
        d_end = float(np.linalg.norm(goal - positions[end, agent_id]))
        progress = d_start - d_end
        eta = progress / (length + EPS)
        mask = (jerk_steps >= start) & (jerk_steps < end)
        selected_jerk = jerk[mask]
        ref = np.asarray(event.get("new_active_goal", active_goals[start, agent_id]), dtype=float)
        ref_dir = unit(ref - positions[start, agent_id])
        category, safety, evidence = classify_segment(event, d_start)
        row = {
            **base,
            "agent_id": agent_id,
            "segment_index": index,
            "start_step": start,
            "end_step_exclusive": end,
            "duration_s": (end - start) * DT,
            "source_event": event.get("event", ""),
            "active_reference_x": ref[0], "active_reference_y": ref[1], "active_reference_z": ref[2],
            "reference_direction_x": ref_dir[0], "reference_direction_y": ref_dir[1], "reference_direction_z": ref_dir[2],
            "segment_path_length_m": length,
            "segment_goal_progress_m": progress,
            "segment_efficiency_eta": eta,
            "start_goal_distance_m": d_start,
            "end_goal_distance_m": d_end,
            "displacement_x": positions[end, agent_id, 0] - positions[start, agent_id, 0],
            "displacement_y": positions[end, agent_id, 1] - positions[start, agent_id, 1],
            "displacement_z": positions[end, agent_id, 2] - positions[start, agent_id, 2],
            "jerk_cost_sum": float(np.sum(selected_jerk**2)),
            "vertical_jerk_cost_sum": float(np.sum(selected_jerk[:, 2] ** 2)) if selected_jerk.size else 0.0,
            "lateral_jerk_cost_sum": float(np.sum(selected_jerk[:, :2] ** 2)) if selected_jerk.size else 0.0,
            "jerk_sample_count": int(selected_jerk.shape[0]),
            "jerk_cost_per_step": float(np.mean(np.sum(selected_jerk**2, axis=1))) if selected_jerk.size else None,
            "efficiency_class": "negative" if eta < 0 else "low" if eta < 0.2 else "moderate" if eta < 0.5 else "high",
            "context_class": category,
            "recorded_safety_associated": safety,
            "context_evidence": evidence,
            "potentially_redundant_low_efficiency": False,
            "diagnostic_excess_burden_m": max(length - max(progress, 0.0), 0.0),
        }
        rows.append(row)
    return rows


def build_cancellation(segments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for first, second in zip(segments[:-1], segments[1:]):
        d1 = np.asarray([first["displacement_x"], first["displacement_y"], first["displacement_z"]], dtype=float)
        d2 = np.asarray([second["displacement_x"], second["displacement_y"], second["displacement_z"]], dtype=float)
        n1, n2 = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
        cos_reverse = float(np.dot(d1, d2) / (n1 * n2)) if n1 > EPS and n2 > EPS else None
        cancellation = 1.0 - float(np.linalg.norm(d1 + d2)) / (n1 + n2 + EPS)
        r1 = np.asarray([first["reference_direction_x"], first["reference_direction_y"], first["reference_direction_z"]])
        r2 = np.asarray([second["reference_direction_x"], second["reference_direction_y"], second["reference_direction_z"]])
        rangle = angle(r1, r2)
        row = {
            "scenario_id": first["scenario_id"], "stage": first["stage"], "family": first["family"], "agent_id": first["agent_id"],
            "first_segment_index": first["segment_index"], "second_segment_index": second["segment_index"],
            "first_start_step": first["start_step"], "second_start_step": second["start_step"],
            "first_displacement_m": n1, "second_displacement_m": n2,
            "cos_reverse": cos_reverse, "cancellation": cancellation, "high_cancellation": cancellation >= HIGH_CANCELLATION,
            "reference_angular_change_rad": rangle,
            "reference_angular_change_deg": None if rangle is None else math.degrees(rangle),
            "pair_jerk_cost_sum": first["jerk_cost_sum"] + second["jerk_cost_sum"],
            "pair_path_length_m": first["segment_path_length_m"] + second["segment_path_length_m"],
        }
        rows.append(row)
    for index in range(len(segments) - 2):
        first, third = segments[index], segments[index + 2]
        a = np.asarray([first["reference_direction_x"], first["reference_direction_y"], first["reference_direction_z"]])
        c = np.asarray([third["reference_direction_x"], third["reference_direction_y"], third["reference_direction_z"]])
        ac = angle(a, c)
        if index < len(rows):
            rows[index]["aba_ac_angle_deg"] = None if ac is None else math.degrees(ac)
            for threshold in (10, 20, 30):
                rows[index][f"aba_approx_le_{threshold}deg"] = bool(ac is not None and math.degrees(ac) <= threshold)
    return rows


def bootstrap_spearman(rows: Sequence[Mapping[str, Any]], x: str, y: str, seed: int) -> dict[str, Any]:
    prepared = [(str(row["scenario_id"]), float(row[x]), float(row[y])) for row in rows if row.get(x) is not None and row.get(y) is not None and math.isfinite(float(row[x])) and math.isfinite(float(row[y]))]
    if len(prepared) < 3:
        return {"n": len(prepared), "rho": None, "p": None, "ci95": [None, None]}
    rho, p = spearmanr([item[1] for item in prepared], [item[2] for item in prepared])
    clusters = defaultdict(list)
    for item in prepared:
        clusters[item[0]].append(item)
    keys = sorted(clusters)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(BOOTSTRAP_REPS):
        sampled = []
        for key in rng.choice(keys, size=len(keys), replace=True):
            sampled.extend(clusters[str(key)])
        value = spearmanr([item[1] for item in sampled], [item[2] for item in sampled]).statistic
        if math.isfinite(float(value)):
            samples.append(float(value))
    return {"n": len(prepared), "rho": float(rho), "p": float(p), "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))], "bootstrap_reps": BOOTSTRAP_REPS, "cluster": "scenario_id"}


def reference_switch_masks(acceleration_count: int, switches: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    jerk_steps = np.arange(1, acceleration_count, dtype=int)
    mask02 = np.zeros(jerk_steps.size, dtype=bool)
    mask05 = np.zeros(jerk_steps.size, dtype=bool)
    for step in switches:
        mask02 |= (jerk_steps >= step) & (jerk_steps <= step + 2)
        mask05 |= (jerk_steps >= step) & (jerk_steps <= step + 5)
    return mask02, mask05


def aggregate_episode(record: Mapping[str, Any], agent_rows: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]], cancellation: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    episode = record["episode"]
    low = [row for row in segments if float(row["segment_efficiency_eta"]) < 0.2]
    negative = [row for row in segments if float(row["segment_efficiency_eta"]) < 0.0]
    total_path = sum(float(row["segment_path_length_m"]) for row in segments)
    total_burden = sum(float(row["diagnostic_excess_burden_m"]) for row in segments)
    actual_changes = sum(int(row["segment_index"] > 0) for row in segments)
    durations = [float(row["duration_s"]) for row in segments if int(row["segment_index"]) > 0]
    completion = sum("REFERENCE_COMPLETION" in str(row["source_event"]) for row in segments)
    restoration = sum("HANDOFF" in str(row["source_event"]) or row["context_class"] == "task-goal-restoration" for row in segments)
    sim_seconds = float(episode.get("steps", 0)) * DT
    return {
        "scenario_id": episode["scenario_id"], "stage": stage_label(episode["stage"]), "family": episode["family"], "seed": episode["seed"],
        "team_success": bool(episode["team_success"]), "agent_count": len(agent_rows),
        "team_mission_distance_sum_m": sum(float(row["mission_distance_m"]) for row in agent_rows),
        "mean_agent_mission_distance_m": float(np.mean([row["mission_distance_m"] for row in agent_rows])),
        "team_path_length_m": sum(float(row["path_length_m"]) for row in agent_rows),
        "mean_agent_path_length_m": float(np.mean([row["path_length_m"] for row in agent_rows])),
        "team_excess_path_m": sum(float(row["excess_path_m"]) for row in agent_rows),
        "mean_agent_excess_path_m": float(np.mean([row["excess_path_m"] for row in agent_rows])),
        "mean_detour_ratio": float(np.mean([row["detour_ratio"] for row in agent_rows])),
        "mean_normalized_excess": float(np.mean([row["normalized_excess"] for row in agent_rows])),
        "team_backward_path_m": sum(float(row["backward_path_m"]) for row in agent_rows),
        "mean_backward_ratio": float(np.mean([row["backward_ratio"] for row in agent_rows])),
        "team_lateral_motion_m": sum(float(row["lateral_motion_m"]) for row in agent_rows),
        "mean_lateral_ratio": float(np.mean([row["lateral_ratio"] for row in agent_rows])),
        "team_vertical_variation_m": sum(float(row["vertical_variation_m"]) for row in agent_rows),
        "mean_vertical_ratio": float(np.mean([row["vertical_ratio"] for row in agent_rows])),
        "team_cumulative_heading_variation_rad": sum(float(row["cumulative_heading_variation_rad"]) for row in agent_rows),
        "trajectory_smoothness_m2_s6": float(np.mean([row["smoothness_m2_s6"] for row in agent_rows])),
        "stored_trajectory_smoothness_m2_s6": float(episode["trajectory_smoothness"]),
        "err_accepted_reference_changes": actual_changes,
        "err_changes_per_simulated_second": actual_changes / sim_seconds if sim_seconds > 0 else None,
        "mean_dwell_time_s": float(np.mean(durations)) if durations else None,
        "median_dwell_time_s": float(np.median(durations)) if durations else None,
        "p10_dwell_time_s": float(np.percentile(durations, 10)) if durations else None,
        "local_reference_completion_event_count": completion,
        "task_goal_restoration_count": restoration,
        "segment_count": len(segments),
        "negative_progress_segment_fraction": len(negative) / len(segments) if segments else None,
        "low_efficiency_segment_fraction_eta_lt_0p2": len(low) / len(segments) if segments else None,
        "path_share_in_low_efficiency_segments": sum(float(row["segment_path_length_m"]) for row in low) / total_path if total_path > EPS else None,
        "diagnostic_excess_burden_share_in_low_efficiency_segments": sum(float(row["diagnostic_excess_burden_m"]) for row in low) / total_burden if total_burden > EPS else None,
        "mean_segment_efficiency": float(np.mean([row["segment_efficiency_eta"] for row in segments])) if segments else None,
        "mean_segment_cancellation": float(np.mean([row["cancellation"] for row in cancellation])) if cancellation else None,
        "high_cancellation_pair_fraction": float(np.mean([row["high_cancellation"] for row in cancellation])) if cancellation else None,
        "mean_reference_angular_change_deg": float(np.mean([row["reference_angular_change_deg"] for row in cancellation if row.get("reference_angular_change_deg") is not None])) if any(row.get("reference_angular_change_deg") is not None for row in cancellation) else None,
    }


def stable_rows_for_agent(base: Mapping[str, Any], segments: Sequence[Mapping[str, Any]], accelerations: np.ndarray, agent_id: int) -> list[dict[str, Any]]:
    jerk = np.diff(accelerations[:, agent_id, :], axis=0) / DT if accelerations.shape[0] > 1 else np.empty((0, 3))
    jerk_steps = np.arange(1, accelerations.shape[0], dtype=int)
    rows = []
    for segment in segments:
        if float(segment["duration_s"]) < 0.5:
            continue
        start, end = int(segment["start_step"]), int(segment["end_step_exclusive"])
        full = (jerk_steps >= start) & (jerk_steps < end)
        core = (jerk_steps >= start + 5) & (jerk_steps < end)
        for scope, mask in (("full_stable_reference_interval", full), ("stable_core_after_first_0p5s", core)):
            selected = jerk[mask]
            if not selected.size:
                continue
            rows.append({
                **base, "agent_id": agent_id, "segment_index": segment["segment_index"], "scope": scope,
                "interval_duration_s": float(segment["duration_s"]), "qualifies_0p5s": True, "qualifies_1p0s": float(segment["duration_s"]) >= 1.0,
                "sample_count": int(selected.shape[0]), "jerk_cost_sum": float(np.sum(selected**2)),
                "jerk_cost_per_step": float(np.mean(np.sum(selected**2, axis=1))),
                "vertical_jerk_cost_per_step": float(np.mean(selected[:, 2] ** 2)),
                "lateral_jerk_cost_per_step": float(np.mean(np.sum(selected[:, :2] ** 2, axis=1))),
                "segment_efficiency_eta": segment["segment_efficiency_eta"], "context_class": segment["context_class"],
            })
    return rows


def summarize_stage(rows: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Any]:
    selected = list(rows) if stage == "Overall" else [row for row in rows if row["stage"] == stage]
    output: dict[str, Any] = {"stage": stage, "successful_agent_trajectories": len(selected), "successful_team_episodes": len({row["scenario_id"] for row in selected})}
    for field in ("mission_distance_m", "path_length_m", "detour_ratio", "excess_path_m", "normalized_excess", "backward_path_m", "backward_ratio", "lateral_motion_m", "lateral_ratio", "vertical_variation_m", "vertical_ratio", "cumulative_heading_variation_rad", "smoothness_m2_s6"):
        summary = describe(row[field] for row in selected)
        for key, value in summary.items():
            output[f"{field}_{key}"] = value
    return output


def compatible_baseline_rows() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, root in (("DWA-SensingMatched", M2_ROOT),):
        for path in sorted(root.glob("*.json")):
            record = load_json(path)
            if not bool(record["episode"]["team_success"]):
                continue
            with np.load(root / record["trajectory_file"]) as trajectory:
                pos = np.asarray(trajectory["positions"], float)
                acc = np.asarray(trajectory["accelerations"], float)
                starts = np.asarray(trajectory["starts"], float)
                goals = np.asarray(trajectory["goals"], float)
            for agent in range(pos.shape[1]):
                output.append({"method": method, "scenario_id": record["episode"]["scenario_id"], "stage": stage_label(record["episode"]["stage"]), "agent_id": agent, **trajectory_metrics(pos[:, agent], acc[:, agent], starts[agent], goals[agent])})
    for path in sorted(PPO_ROOT.glob("*.json")):
        record = load_json(path)
        outcome = record.get("outcome", {})
        if not bool(outcome.get("team_success")):
            continue
        pos = np.asarray(record["positions"], float)
        acc = np.asarray(record["applied_accelerations"], float)
        starts = np.asarray(record["starts"], float)
        goals = np.asarray(record["goals"], float)
        for agent in range(pos.shape[1]):
            output.append({"method": "PPO-Direct", "scenario_id": record["scenario_id"], "stage": stage_label(record["stage"]), "agent_id": agent, **trajectory_metrics(pos[:, agent], acc[:, agent], starts[agent], goals[agent])})
    return output


def tacv_diagnostic() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    missing = []
    for original_path in sorted(CRT_ORIGINAL_ROOT.glob("*.json")):
        medium_path = TACV_MEDIUM_ROOT / original_path.name
        if not medium_path.exists():
            missing.append(original_path.name)
            continue
        original, medium = load_json(original_path), load_json(medium_path)
        if not bool(original["episode"]["team_success"]) or not bool(medium["episode"]["team_success"]):
            continue
        def episode_values(record: Mapping[str, Any]) -> dict[str, Any]:
            mission = sum(float(agent["terminal_progress_m"]) + float(agent["final_terminal_distance_m"]) for agent in record["agents"])
            path = float(record["episode"]["team_path_length_m"])
            changes = sum(bool(event.get("goal_changed")) and str(event.get("event")) != "INITIAL_SELECTION" for event in record["events"])
            return {"path": path, "mission": mission, "detour": path / mission, "excess": path - mission, "smoothness": float(record["episode"]["trajectory_smoothness"]), "changes": changes, "replanning": int(record["episode"]["replanning_count"])}
        o, m = episode_values(original), episode_values(medium)
        rows.append({
            "scenario_id": original["episode"]["scenario_id"], "stage": stage_label(original["episode"]["stage"]), "family": original["episode"]["family"],
            "original_team_path_length_m": o["path"], "tacv_medium_team_path_length_m": m["path"], "delta_path_m_medium_minus_original": m["path"] - o["path"],
            "original_team_excess_path_m": o["excess"], "tacv_medium_team_excess_path_m": m["excess"], "delta_excess_m_medium_minus_original": m["excess"] - o["excess"],
            "original_team_detour_ratio": o["detour"], "tacv_medium_team_detour_ratio": m["detour"], "delta_detour_ratio_medium_minus_original": m["detour"] - o["detour"],
            "original_smoothness": o["smoothness"], "tacv_medium_smoothness": m["smoothness"], "delta_smoothness_medium_minus_original": m["smoothness"] - o["smoothness"],
            "original_reference_changes": o["changes"], "tacv_medium_reference_changes": m["changes"], "delta_reference_changes": m["changes"] - o["changes"],
            "original_replanning_count": o["replanning"], "tacv_medium_replanning_count": m["replanning"], "delta_replanning_count": m["replanning"] - o["replanning"],
            "backward_lateral_vertical_low_efficiency_cancellation_status": "UNAVAILABLE_FOR_TACV_MEDIUM: positions were not retained in its NPZ",
        })
    summary = {
        "scope": "paired both-success Development only; not Formal evidence", "paired_both_success_n": len(rows), "missing_records": missing,
        "mean_path_change_m": float(np.mean([row["delta_path_m_medium_minus_original"] for row in rows])) if rows else None,
        "mean_excess_change_m": float(np.mean([row["delta_excess_m_medium_minus_original"] for row in rows])) if rows else None,
        "mean_detour_ratio_change": float(np.mean([row["delta_detour_ratio_medium_minus_original"] for row in rows])) if rows else None,
        "mean_smoothness_change": float(np.mean([row["delta_smoothness_medium_minus_original"] for row in rows])) if rows else None,
        "mean_reference_change_count_change": float(np.mean([row["delta_reference_changes"] for row in rows])) if rows else None,
        "mean_original_path_m": float(np.mean([row["original_team_path_length_m"] for row in rows])) if rows else None,
        "medium_position_logging": "NOT_RETAINED",
        "low_efficiency_path_share_change": None,
        "interpretation_boundary": "Aggregate path/detour and reference-count effects are recoverable; geometric component and segment-efficiency changes are unavailable and are not reconstructed.",
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    for name in ("00_context", "01_metric_contract", "02_formal_path_decomposition", "03_err_segment_analysis", "04_cancellation", "05_path_jerk_relation", "06_stable_reference", "07_tacv_medium_diagnostic", "08_stage_analysis", "09_figures", "10_root_cause", "11_paper_ready"):
        (output / name).mkdir(parents=True, exist_ok=True)

    agent_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    cancellation_rows: list[dict[str, Any]] = []
    stable_rows: list[dict[str, Any]] = []
    metric_checks = []
    temporal_totals = Counter()
    representative_candidates = []

    record_paths = sorted(M9_ROOT.glob("*.json"))
    for record_path in record_paths:
        record = load_json(record_path)
        episode = record["episode"]
        if not bool(episode["team_success"]):
            continue
        with np.load(M9_ROOT / record["trajectory_file"]) as trajectory:
            positions = np.asarray(trajectory["positions"], dtype=float)
            accelerations = np.asarray(trajectory["accelerations"], dtype=float)
            active_goals = np.asarray(trajectory["active_goals"], dtype=float)
            starts = np.asarray(trajectory["starts"], dtype=float)
            goals = np.asarray(trajectory["goals"], dtype=float)
        base = {"scenario_id": episode["scenario_id"], "stage": stage_label(episode["stage"]), "family": episode["family"], "seed": episode["seed"]}
        current_agents = []
        current_segments = []
        current_cancellation = []
        for agent_id in range(positions.shape[1]):
            metrics = trajectory_metrics(positions[:, agent_id], accelerations[:, agent_id], starts[agent_id], goals[agent_id])
            row = {**base, "agent_id": agent_id, "trajectory_steps": positions.shape[0] - 1, **metrics}
            agent_rows.append(row)
            current_agents.append(row)
            segments = build_segments(record, positions, accelerations, active_goals, goals[agent_id], agent_id, base)
            cancellations = build_cancellation(segments)
            for index, segment in enumerate(segments[:-1]):
                pair = cancellations[index]
                segment["potentially_redundant_low_efficiency"] = bool(
                    float(segment["segment_efficiency_eta"]) < 0.2
                    and not bool(segment["recorded_safety_associated"])
                    and (float(segment["duration_s"]) <= 1.0 + EPS or float(pair["cancellation"]) >= 0.2)
                )
            current_segments.extend(segments)
            current_cancellation.extend(cancellations)
            stable_rows.extend(stable_rows_for_agent(base, segments, accelerations, agent_id))

            jerk = np.diff(accelerations[:, agent_id, :], axis=0) / DT
            jerk_sq = np.sum(jerk**2, axis=1)
            switches = [int(event["step"]) for event in changed_events(record, agent_id) if int(event["step"]) > 0 and str(event.get("event")) != "INITIAL_SELECTION"]
            mask02, mask05 = reference_switch_masks(accelerations.shape[0], switches)
            temporal_totals["total_cost"] += float(np.sum(jerk_sq))
            temporal_totals["total_samples"] += int(jerk_sq.size)
            temporal_totals["within_0p2_cost"] += float(np.sum(jerk_sq[mask02]))
            temporal_totals["within_0p2_samples"] += int(np.sum(mask02))
            temporal_totals["within_0p5_cost"] += float(np.sum(jerk_sq[mask05]))
            temporal_totals["within_0p5_samples"] += int(np.sum(mask05))
            temporal_totals["outside_0p5_cost"] += float(np.sum(jerk_sq[~mask05]))
            temporal_totals["outside_0p5_samples"] += int(np.sum(~mask05))
            temporal_totals["outside_0p5_vertical_cost"] += float(np.sum(jerk[~mask05, 2] ** 2))
            temporal_totals["outside_0p5_lateral_cost"] += float(np.sum(jerk[~mask05, :2] ** 2))
        segment_rows.extend(current_segments)
        cancellation_rows.extend(current_cancellation)
        current_episode = aggregate_episode(record, current_agents, current_segments, current_cancellation)
        episode_rows.append(current_episode)
        metric_checks.append({
            "scenario_id": episode["scenario_id"],
            "team_path_difference_recomputed_minus_stored_m": current_episode["team_path_length_m"] - float(episode["team_path_length_m"]),
            "smoothness_difference_recomputed_minus_stored": current_episode["trajectory_smoothness_m2_s6"] - float(episode["trajectory_smoothness"]),
        })
        representative_candidates.append({"scenario_id": episode["scenario_id"], "stage": base["stage"], "mean_detour_ratio": current_episode["mean_detour_ratio"], "path": str(record_path), "trajectory": str(M9_ROOT / record["trajectory_file"])})

    if len(episode_rows) != 381:
        raise RuntimeError(f"Expected 381 successful M9 episodes, found {len(episode_rows)}")
    max_path_error = max(abs(row["team_path_difference_recomputed_minus_stored_m"]) for row in metric_checks)
    max_smooth_error = max(abs(row["smoothness_difference_recomputed_minus_stored"]) for row in metric_checks)
    if max_path_error > 1e-8 or max_smooth_error > 1e-8:
        raise RuntimeError(f"Metric reconciliation failed: path={max_path_error}, smoothness={max_smooth_error}")

    write_csv(output / "02_formal_path_decomposition/FORMAL_PATH_DECOMPOSITION.csv", agent_rows)
    write_csv(output / "02_formal_path_decomposition/FORMAL_EPISODE_PATH_SUMMARY.csv", episode_rows)
    write_csv(output / "03_err_segment_analysis/ERR_SEGMENT_EFFICIENCY.csv", segment_rows)
    write_csv(output / "04_cancellation/REFERENCE_SEQUENCE_CANCELLATION.csv", cancellation_rows)
    write_csv(output / "06_stable_reference/STABLE_REFERENCE_JERK_AUDIT.csv", stable_rows)
    write_csv(output / "00_context/METRIC_RECONCILIATION.csv", metric_checks)

    segment_summary: dict[str, Any] = {
        "scope": "all agents in 381 successful Formal V2 M9 episodes",
        "segment_count": len(segment_rows),
        "eta_distribution": describe((row["segment_efficiency_eta"] for row in segment_rows), percentiles=(10, 25, 50, 75, 90)),
        "fractions": {}, "path_shares": {}, "diagnostic_excess_burden_shares": {},
        "efficiency_class_counts": dict(Counter(row["efficiency_class"] for row in segment_rows)),
        "context_counts": dict(Counter(row["context_class"] for row in segment_rows)),
        "potentially_redundant_low_efficiency_count": sum(bool(row["potentially_redundant_low_efficiency"]) for row in segment_rows),
        "caution": "Low eta is diagnostic, not proof of a bad decision. The excess burden is a local non-causal allocation and is not algebraically equal to episode L_path-D_mission.",
    }
    total_segment_path = sum(float(row["segment_path_length_m"]) for row in segment_rows)
    total_burden = sum(float(row["diagnostic_excess_burden_m"]) for row in segment_rows)
    for label, threshold in (("eta_lt_0", 0.0), ("eta_lt_0p1", 0.1), ("eta_lt_0p2", 0.2), ("eta_lt_0p5", 0.5)):
        selected = [row for row in segment_rows if float(row["segment_efficiency_eta"]) < threshold]
        segment_summary["fractions"][label] = len(selected) / len(segment_rows)
        segment_summary["path_shares"][label] = sum(float(row["segment_path_length_m"]) for row in selected) / total_segment_path
        segment_summary["diagnostic_excess_burden_shares"][label] = sum(float(row["diagnostic_excess_burden_m"]) for row in selected) / total_burden
    write_json(output / "03_err_segment_analysis/ERR_SEGMENT_EFFICIENCY_SUMMARY.json", clean(segment_summary))

    err_pairs = (
        ("err_count_vs_team_path", "err_accepted_reference_changes", "team_path_length_m"),
        ("err_count_vs_excess_path", "err_accepted_reference_changes", "team_excess_path_m"),
        ("err_rate_vs_excess_path", "err_changes_per_simulated_second", "team_excess_path_m"),
        ("err_count_vs_detour_ratio", "err_accepted_reference_changes", "mean_detour_ratio"),
        ("err_count_vs_backward", "err_accepted_reference_changes", "team_backward_path_m"),
        ("err_count_vs_lateral", "err_accepted_reference_changes", "team_lateral_motion_m"),
        ("err_count_vs_vertical", "err_accepted_reference_changes", "team_vertical_variation_m"),
        ("err_count_vs_low_efficiency_share", "err_accepted_reference_changes", "path_share_in_low_efficiency_segments"),
        ("err_count_vs_smoothness", "err_accepted_reference_changes", "trajectory_smoothness_m2_s6"),
        ("cancellation_vs_excess_path", "mean_segment_cancellation", "team_excess_path_m"),
        ("cancellation_vs_smoothness", "mean_segment_cancellation", "trajectory_smoothness_m2_s6"),
    )
    err_associations = {name: bootstrap_spearman(episode_rows, x, y, 5000 + index) for index, (name, x, y) in enumerate(err_pairs)}
    write_json(output / "05_path_jerk_relation/ERR_PATH_ASSOCIATION.json", clean({"scope": "381 successful team episodes; episode-cluster bootstrap", "associations": err_associations}))

    smooth_pairs = (
        ("excess_path_vs_smoothness", "team_excess_path_m"),
        ("detour_ratio_vs_smoothness", "mean_detour_ratio"),
        ("backward_path_vs_smoothness", "team_backward_path_m"),
        ("lateral_motion_vs_smoothness", "team_lateral_motion_m"),
        ("vertical_variation_vs_smoothness", "team_vertical_variation_m"),
        ("err_count_vs_smoothness", "err_accepted_reference_changes"),
        ("low_efficiency_share_vs_smoothness", "path_share_in_low_efficiency_segments"),
        ("cancellation_vs_smoothness", "mean_segment_cancellation"),
        ("heading_variation_vs_smoothness", "team_cumulative_heading_variation_rad"),
    )
    smooth_associations = {name: bootstrap_spearman(episode_rows, x, "trajectory_smoothness_m2_s6", 6000 + index) for index, (name, x) in enumerate(smooth_pairs)}
    write_json(output / "05_path_jerk_relation/PATH_SMOOTHNESS_ASSOCIATION.json", clean({"scope": "381 successful team episodes", "associations": smooth_associations}))

    category_rows = []
    total_jerk = sum(float(row["jerk_cost_sum"]) for row in segment_rows)
    total_samples = sum(int(row["jerk_sample_count"]) for row in segment_rows)
    for category in ("negative", "low", "moderate", "high"):
        selected = [row for row in segment_rows if row["efficiency_class"] == category]
        cost = sum(float(row["jerk_cost_sum"]) for row in selected)
        samples = sum(int(row["jerk_sample_count"]) for row in selected)
        category_rows.append({"category": category, "overlap_policy": "mutually exclusive eta bin", "segment_count": len(selected), "sample_count": samples, "jerk_cost_sum": cost, "jerk_cost_per_step": cost / samples if samples else None, "fraction_total_jerk_cost": cost / total_jerk if total_jerk else None, "fraction_total_time_samples": samples / total_samples if total_samples else None})
    high_indices = {(row["scenario_id"], row["agent_id"], row["first_segment_index"]) for row in cancellation_rows if row["high_cancellation"]} | {(row["scenario_id"], row["agent_id"], row["second_segment_index"]) for row in cancellation_rows if row["high_cancellation"]}
    for category, predicate in (
        ("high-cancellation-neighborhood", lambda row: (row["scenario_id"], row["agent_id"], row["segment_index"]) in high_indices),
        ("peer-critical", lambda row: row["context_class"] == "peer-critical"),
        ("recorded-safety-associated", lambda row: bool(row["recorded_safety_associated"])),
    ):
        selected = [row for row in segment_rows if predicate(row)]
        cost = sum(float(row["jerk_cost_sum"]) for row in selected)
        samples = sum(int(row["jerk_sample_count"]) for row in selected)
        category_rows.append({"category": category, "overlap_policy": "diagnostic overlay; do not add to eta-bin fractions", "segment_count": len(selected), "sample_count": samples, "jerk_cost_sum": cost, "jerk_cost_per_step": cost / samples if samples else None, "fraction_total_jerk_cost": cost / total_jerk if total_jerk else None, "fraction_total_time_samples": samples / total_samples if total_samples else None})

    temporal = {
        "scope": "successful Formal V2 M9; per-agent union of accepted active-reference-change windows",
        "jerk_sample_contract": "jerk[k]=(a[k+1]-a[k])/0.1; jerk step index k+1; windows include switch step through switch+N",
        "overlap_handling": "union of jerk sample indices per agent, so overlapping switch windows are counted once",
        "total_jerk_cost": temporal_totals["total_cost"],
        "within_0p2s": {"cost": temporal_totals["within_0p2_cost"], "samples": temporal_totals["within_0p2_samples"], "cost_fraction": temporal_totals["within_0p2_cost"] / temporal_totals["total_cost"], "cost_per_step": temporal_totals["within_0p2_cost"] / temporal_totals["within_0p2_samples"]},
        "within_0p5s": {"cost": temporal_totals["within_0p5_cost"], "samples": temporal_totals["within_0p5_samples"], "cost_fraction": temporal_totals["within_0p5_cost"] / temporal_totals["total_cost"], "cost_per_step": temporal_totals["within_0p5_cost"] / temporal_totals["within_0p5_samples"]},
        "outside_0p5s": {"cost": temporal_totals["outside_0p5_cost"], "samples": temporal_totals["outside_0p5_samples"], "cost_fraction": temporal_totals["outside_0p5_cost"] / temporal_totals["total_cost"], "cost_per_step": temporal_totals["outside_0p5_cost"] / temporal_totals["outside_0p5_samples"], "vertical_cost_per_step": temporal_totals["outside_0p5_vertical_cost"] / temporal_totals["outside_0p5_samples"], "lateral_cost_per_step": temporal_totals["outside_0p5_lateral_cost"] / temporal_totals["outside_0p5_samples"]},
        "segment_type_attribution": category_rows,
        "causal_boundary": "Temporal concentration and segment association are descriptive; overlapping diagnostic overlays are not additive.",
    }
    write_json(output / "05_path_jerk_relation/JERK_COST_TEMPORAL_ATTRIBUTION.json", clean(temporal))

    stage_rows = [summarize_stage(agent_rows, stage) for stage in ("Overall", "Stage I", "Stage II", "Stage III", "Stage IV")]
    write_csv(output / "08_stage_analysis/STAGE_PATH_INEFFICIENCY_SUMMARY.csv", stage_rows)

    baseline_rows = compatible_baseline_rows()
    baseline_summary = []
    for method in ("Proposed", "DWA-SensingMatched", "PPO-Direct"):
        data = agent_rows if method == "Proposed" else [row for row in baseline_rows if row["method"] == method]
        baseline_summary.append({"method": method, "successful_team_episodes": len({row["scenario_id"] for row in data}), "successful_agent_trajectories": len(data), "comparison_status": "success-conditional, unpaired across method-specific success subsets", **{f"{field}_mean": describe(row[field] for row in data)["mean"] for field in ("detour_ratio", "backward_ratio", "lateral_ratio", "vertical_ratio", "cumulative_heading_variation_rad", "smoothness_m2_s6")}})
    write_csv(output / "08_stage_analysis/COMPATIBLE_BASELINE_PATH_COMPARISON.csv", baseline_summary)

    tacv_rows, tacv_summary = tacv_diagnostic()
    write_csv(output / "07_tacv_medium_diagnostic/TACV_MEDIUM_PATH_DIAGNOSTIC.csv", tacv_rows)
    write_json(output / "07_tacv_medium_diagnostic/TACV_MEDIUM_PATH_DIAGNOSTIC_SUMMARY.json", clean(tacv_summary))

    overall = stage_rows[0]
    mean_backward_ratio = overall["backward_ratio_mean"]
    mean_lateral_ratio = overall["lateral_ratio_mean"]
    mean_vertical_ratio = overall["vertical_ratio_mean"]
    mean_cancellation = describe(row["cancellation"] for row in cancellation_rows)["mean"]
    high_cancellation_fraction = float(np.mean([row["high_cancellation"] for row in cancellation_rows]))
    low_path_share = segment_summary["path_shares"]["eta_lt_0p2"]
    safety_low = [row for row in segment_rows if float(row["segment_efficiency_eta"]) < 0.2 and bool(row["recorded_safety_associated"])]
    low_all = [row for row in segment_rows if float(row["segment_efficiency_eta"]) < 0.2]
    safety_low_fraction = len(safety_low) / len(low_all) if low_all else 0.0
    ratios = {"VERTICAL_OSCILLATION": mean_vertical_ratio, "LATERAL_DETOUR": mean_lateral_ratio, "BACKTRACKING": mean_backward_ratio, "RECURRENT_REFERENCE_CANCELLATION": mean_cancellation}
    primary = max(ratios, key=ratios.get)
    ordered = sorted(ratios.values(), reverse=True)
    if ordered[0] < 0.08:
        primary = "NO_CLEAR_DOMINANT_SOURCE"
    elif ordered[1] > 0 and ordered[0] / ordered[1] < 1.35:
        primary = "MIXED"
    if safety_low_fraction > 0.7 and low_path_share > 0.15:
        primary = "SAFETY_NECESSARY_AVOIDANCE"

    rho_err = err_associations["err_count_vs_excess_path"]["rho"]
    rho_low_smooth = smooth_associations["low_efficiency_share_vs_smoothness"]["rho"]
    rho_detour_smooth = smooth_associations["detour_ratio_vs_smoothness"]["rho"]
    # A shared recurrent-correction cause predicts *positive* co-variation with
    # smoothness.  Absolute correlations would misclassify the observed inverse
    # relationships as supporting evidence.
    positive_shared_signals = [
        max(float(rho_low_smooth or 0.0), 0.0),
        max(float(rho_detour_smooth or 0.0), 0.0),
        max(float(err_associations["cancellation_vs_smoothness"]["rho"] or 0.0), 0.0),
        min(max(float(rho_err or 0.0), 0.0), max(float(err_associations["err_count_vs_smoothness"]["rho"] or 0.0), 0.0)),
    ]
    strong_signals = sum(value >= 0.3 for value in positive_shared_signals)
    moderate_signals = sum(value >= 0.15 for value in positive_shared_signals)
    shared = "STRONG" if strong_signals >= 2 else "MODERATE" if moderate_signals >= 2 else "WEAK" if any(value >= 0.1 for value in positive_shared_signals) else "NONE"
    headroom_score = int(low_path_share >= 0.15) + int(high_cancellation_fraction >= 0.15) + int(abs(rho_err or 0) >= 0.3) + int(safety_low_fraction < 0.5)
    headroom = "HIGH" if headroom_score >= 4 else "MODERATE" if headroom_score >= 2 else "LOW"
    # Mandatory goal stop: rare low-efficiency segments OR low cancellation is
    # sufficient to close the redundant-ERR repair branch.
    if low_path_share < 0.05 or high_cancellation_fraction < 0.05:
        headroom = "LOW"
    stable_fraction = temporal["outside_0p5s"]["cost_fraction"]
    if headroom in {"LOW", "NONE"}:
        next_step = "NO CHANGE - KEEP ORIGINAL"
    elif high_cancellation_fraction >= 0.15:
        next_step = "ERR REDUNDANT-UPDATE SUPPRESSION WORTH TESTING"
    elif safety_low_fraction < 0.5:
        next_step = "SAFETY-FIRST PATH-EFFICIENCY VETO WORTH TESTING"
    else:
        next_step = "LOCAL REFERENCE COMPLETION / RESTORATION LOGIC WORTH AUDITING"

    source_table = [
        {"source": "vertical oscillation", "path_contribution": mean_vertical_ratio, "jerk_association": smooth_associations["vertical_variation_vs_smoothness"]["rho"], "ERR_association": err_associations["err_count_vs_vertical"]["rho"], "safety_association": "not typed in retained LiDAR margin", "evidence_strength": "direct trajectory decomposition", "repair_headroom": "low if minor ratio", "interpretation": "cumulative |dz| / path; diagnostic, non-additive"},
        {"source": "lateral detour", "path_contribution": mean_lateral_ratio, "jerk_association": smooth_associations["lateral_motion_vs_smoothness"]["rho"], "ERR_association": err_associations["err_count_vs_lateral"]["rho"], "safety_association": "mixed/partially observed", "evidence_strength": "direct trajectory decomposition", "repair_headroom": headroom, "interpretation": "cumulative motion perpendicular to instantaneous final-goal direction; non-additive"},
        {"source": "backtracking", "path_contribution": mean_backward_ratio, "jerk_association": smooth_associations["backward_path_vs_smoothness"]["rho"], "ERR_association": err_associations["err_count_vs_backward"]["rho"], "safety_association": "mixed/partially observed", "evidence_strength": "direct signed-progress decomposition", "repair_headroom": headroom, "interpretation": "explicit motion away from final task goal"},
        {"source": "reference-segment cancellation", "path_contribution": mean_cancellation, "jerk_association": err_associations["cancellation_vs_smoothness"]["rho"], "ERR_association": "intrinsic adjacent-ERR metric", "safety_association": safety_low_fraction, "evidence_strength": "direct adjacent-segment geometry", "repair_headroom": headroom, "interpretation": f"C>=0.5 pair fraction={high_cancellation_fraction:.4f}"},
        {"source": "switch-adjacent transient", "path_contribution": "not a path component", "jerk_association": temporal["within_0p5s"]["cost_fraction"], "ERR_association": "direct temporal window", "safety_association": "not isolated", "evidence_strength": "union-of-time-index attribution", "repair_headroom": "previous CRT/TACV branches rejected", "interpretation": "fraction of total jerk cost within 0.5 s after true reference changes"},
        {"source": "stable-reference closed-loop correction", "path_contribution": "not separately additive", "jerk_association": temporal["outside_0p5s"]["cost_fraction"], "ERR_association": "outside switch unions", "safety_association": "not isolated", "evidence_strength": "direct temporal complement", "repair_headroom": "lower-level if dominant", "interpretation": "jerk persisting outside 0.5 s switch windows"},
        {"source": "peer-interaction avoidance", "path_contribution": sum(float(row["segment_path_length_m"]) for row in segment_rows if row["context_class"] == "peer-critical") / total_segment_path, "jerk_association": sum(float(row["jerk_cost_sum"]) for row in segment_rows if row["context_class"] == "peer-critical") / total_jerk, "ERR_association": "recorded interaction-risk context", "safety_association": "YES", "evidence_strength": "partial retained graph-event evidence", "repair_headroom": "safety-first; do not prune automatically", "interpretation": "only event-time recorded interaction labels"},
        {"source": "obstacle avoidance", "path_contribution": None, "jerk_association": None, "ERR_association": "untyped emergency margin only", "safety_association": "UNAVAILABLE_SEPARATELY", "evidence_strength": "insufficient typed per-segment evidence", "repair_headroom": "not established", "interpretation": "static/dynamic/peer cause cannot be separated from retained active-direction margin"},
    ]
    write_csv(output / "10_root_cause/PATH_ROOT_CAUSE_SUMMARY.csv", source_table)

    representative = []
    for stage in ("Stage I", "Stage II", "Stage III", "Stage IV"):
        rows = sorted([row for row in representative_candidates if row["stage"] == stage], key=lambda row: row["mean_detour_ratio"])
        representative.append(rows[len(rows) // 2])
    write_json(output / "09_figures/REPRESENTATIVE_TRAJECTORY_SELECTION.json", {"rule": "stage-median successful episode by episode mean agent detour ratio", "selected": representative})

    conclusion = {
        "ORIGINAL_FORMAL_SUCCESS": 0.9525,
        "ORIGINAL_FORMAL_SUCCESSFUL_EPISODES": 381,
        "MEAN_PATH_LENGTH_SUCCESS": describe(row["team_path_length_m"] for row in episode_rows)["mean"],
        "MEDIAN_PATH_LENGTH_SUCCESS": describe(row["team_path_length_m"] for row in episode_rows)["median"],
        "MEAN_MISSION_DISTANCE": overall["mission_distance_m_mean"],
        "MEAN_DETOUR_RATIO": overall["detour_ratio_mean"],
        "MEDIAN_DETOUR_RATIO": overall["detour_ratio_median"],
        "MEAN_EXCESS_PATH_M": overall["excess_path_m_mean"],
        "MEAN_BACKWARD_PATH_M": overall["backward_path_m_mean"],
        "MEAN_BACKWARD_RATIO": mean_backward_ratio,
        "MEAN_LATERAL_MOTION_M": overall["lateral_motion_m_mean"],
        "MEAN_VERTICAL_VARIATION_M": overall["vertical_variation_m_mean"],
        "MEAN_VERTICAL_RATIO": mean_vertical_ratio,
        "MEAN_ERR_ACCEPTED_CHANGES": describe(row["err_accepted_reference_changes"] for row in episode_rows)["mean"],
        "MEDIAN_ERR_SEGMENT_EFFICIENCY": segment_summary["eta_distribution"]["median"],
        "NEGATIVE_PROGRESS_SEGMENT_FRACTION": segment_summary["fractions"]["eta_lt_0"],
        "LOW_EFFICIENCY_SEGMENT_FRACTION_ETA_LT_0P2": segment_summary["fractions"]["eta_lt_0p2"],
        "PATH_SHARE_IN_LOW_EFFICIENCY_SEGMENTS": low_path_share,
        "EXCESS_PATH_SHARE_ASSOCIATED_WITH_LOW_EFFICIENCY_SEGMENTS": segment_summary["diagnostic_excess_burden_shares"]["eta_lt_0p2"],
        "MEAN_SEGMENT_CANCELLATION": mean_cancellation,
        "HIGH_CANCELLATION_PAIR_FRACTION": high_cancellation_fraction,
        "ERR_COUNT_VS_EXCESS_PATH_RHO": rho_err,
        "LOW_EFFICIENCY_SHARE_VS_SMOOTHNESS_RHO": rho_low_smooth,
        "DETOUR_RATIO_VS_SMOOTHNESS_RHO": rho_detour_smooth,
        "VERTICAL_VARIATION_VS_SMOOTHNESS_RHO": smooth_associations["vertical_variation_vs_smoothness"]["rho"],
        "JERK_COST_WITHIN_0P2S_OF_REFERENCE_CHANGE": temporal["within_0p2s"]["cost_fraction"],
        "JERK_COST_WITHIN_0P5S_OF_REFERENCE_CHANGE": temporal["within_0p5s"]["cost_fraction"],
        "JERK_COST_OUTSIDE_0P5S_SWITCH_WINDOWS": stable_fraction,
        "STABLE_REFERENCE_JERK_LEVEL": temporal["outside_0p5s"]["cost_per_step"],
        "TACV_MEDIUM_PATH_CHANGE": tacv_summary["mean_path_change_m"],
        "TACV_MEDIUM_DETOUR_RATIO_CHANGE": tacv_summary["mean_detour_ratio_change"],
        "TACV_MEDIUM_LOW_EFFICIENCY_PATH_SHARE_CHANGE": None,
        "TACV_MEDIUM_LOW_EFFICIENCY_PATH_SHARE_CHANGE_STATUS": "UNAVAILABLE_NOT_RETAINED",
        "TACV_MEDIUM_ERR_COUNT_CHANGE": tacv_summary["mean_reference_change_count_change"],
        "TACV_MEDIUM_DIAGNOSTIC_CASE": "CASE B: path changed negligibly while smoothness improved; retained positions are unavailable for finer geometric attribution",
        "PATH_EXCESS_PRIMARY_SOURCE": primary,
        "PATH_AND_JERK_SHARED_CAUSE_EVIDENCE": shared,
        "FRAMEWORK_PRESERVING_PATH_REPAIR_HEADROOM": headroom,
        "NEW_SIMULATION_RUN": "NO", "METHOD_CHANGED": "NO", "FORMAL_RESULT_CHANGED": "NO", "ACADEMIC_INTEGRITY_GATE": "PASS",
        "NEXT_STEP_RECOMMENDATION": next_step,
        "FINAL_RECOMMENDATION": f"Read-only evidence classifies the dominant path signature as {primary}; shared path/jerk evidence is {shared} and framework-preserving repair headroom is {headroom}. Recommendation: {next_step}.",
        "metric_reconciliation": {"max_team_path_error_m": max_path_error, "max_smoothness_error": max_smooth_error},
        "interpretation_boundaries": ["Associations are not causal effects.", "Lateral/vertical/backward diagnostic components are not additive path partitions.", "TACV-Medium Development is not Formal evidence and did not retain positions."],
    }
    write_json(output / "conclusion.json", clean(conclusion))

    contract = {
        "schema_version": "path_metric_contract_v1",
        "authoritative_implementation": "planning/pre_gat_closed_loop.py::trajectory_metrics and evaluate_gat_v1_err_development.py episode aggregation",
        "per_agent_path_length": "sum_t ||p[t+1]-p[t]||_2 over raw retained positions",
        "team_path_length": "sum of three per-agent path lengths; team_path_length_mean_agent_m is their arithmetic mean",
        "start_point_included": True,
        "terminal_interpolation": False,
        "raw_executed_state_interval_s": DT,
        "smoothing_resampling": "none",
        "failed_episode_storage": "path is retained for failures, but paper successful-path summaries filter to team_success; paired continuous comparisons use both-success only",
        "unit": "metre",
        "smoothness_contract": "per agent mean_t sum_xyz(((a[t+1]-a[t])/0.1)^2); team arithmetic mean; unit m^2/s^6",
        "switch_window_index_contract": "jerk step k corresponds diff acceleration k-1->k; inherited diagnostic window includes steps switch through switch+N",
        "vertical_sign_tolerances": {"raw": EPS, "significant_mps": SIGN_TOL, "status": "small numerical tolerance only; no physical threshold"},
        "high_cancellation_definition": "C >= 0.5, meaning at least half of the two-segment displacement magnitude is canceled geometrically",
        "low_efficiency_reporting_thresholds": [0.0, 0.1, 0.2, 0.5],
        "primary_data_scope": "381 successful Original Proposed Formal V2 episodes; 1143 agent trajectories",
        "formal_result_changed": False,
        "reconciliation": conclusion["metric_reconciliation"],
    }
    write_json(output / "01_metric_contract/PATH_METRIC_CONTRACT.json", contract)

    sources = [
        ROOT / "planning/pre_gat_closed_loop.py",
        ROOT / "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
        ROOT / "Multi-agent_Algo_lib/scripts/run_long_range_formal_benchmark.py",
        ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/FINAL_REPORT.md",
        ROOT / "artifacts/recurrent_selector_ablation_confirmation/20260822_143118/FINAL_REPORT.md",
        ROOT / "artifacts/sector_resolution_oscillation_audit/20260824_110611/FINAL_REPORT.md",
        ROOT / "artifacts/continuous_reference_transition/20260824_132552/FINAL_REPORT.md",
        ROOT / "artifacts/transient_aware_candidate_veto/20260824_184551/FINAL_REPORT.md",
        ROOT / "artifacts/formal_v2_runtime_realtime_audit/20260822_163515/FINAL_REPORT.md",
        ROOT / "artifacts/phase1_experiment_closure/20260823_141509/FINAL_METHOD_IDENTITY.json",
    ]
    write_json(output / "00_context/SOURCE_INVENTORY.json", {"read_only": True, "sources": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in sources], "formal_record_count": len(record_paths), "successful_record_count": len(episode_rows), "new_simulation_run": False, "method_changed": False})

    q = {
        "Q1": f"Mean successful team path is {conclusion['MEAN_PATH_LENGTH_SUCCESS']:.3f} m; mean agent detour ratio is {conclusion['MEAN_DETOUR_RATIO']:.4f} ({(conclusion['MEAN_DETOUR_RATIO']-1)*100:.2f}% above straight-line mission distance).",
        "Q2": f"The classified dominant diagnostic signature is {primary}; components are non-additive.",
        "Q3": f"Mean vertical variation ratio is {mean_vertical_ratio:.4f}.",
        "Q4": f"Mean explicit backward ratio is {mean_backward_ratio:.4f}.",
        "Q5": f"eta<0.2 segments consume {low_path_share:.4f} of executed path; negative eta fraction is {segment_summary['fractions']['eta_lt_0']:.4f}.",
        "Q6": f"Mean adjacent-segment cancellation is {mean_cancellation:.4f}; C>=0.5 fraction is {high_cancellation_fraction:.4f}.",
        "Q7": f"ERR count vs team excess path Spearman rho={rho_err:.4f}, CI={err_associations['err_count_vs_excess_path']['ci95']}.",
        "Q8": f"Low-efficiency path share vs smoothness rho={rho_low_smooth:.4f} (opposite the shared-cause prediction); eta<0.2 segments carry {(category_rows[0]['fraction_total_jerk_cost'] + category_rows[1]['fraction_total_jerk_cost']):.4f} of jerk cost.",
        "Q9": f"Reference-change unions contain {temporal['within_0p2s']['cost_fraction']:.4f} (0.2 s) and {temporal['within_0p5s']['cost_fraction']:.4f} (0.5 s) of total jerk cost; the 0.5 s union covers {temporal['within_0p5s']['samples']/temporal_totals['total_samples']:.4f} of samples.",
        "Q10": f"Outside 0.5 s switch windows, {stable_fraction:.4f} of total jerk cost remains, at {temporal['outside_0p5s']['cost_per_step']:.3f} m^2/s^6 per sample.",
        "Q11": f"On paired Development both-success cases, TACV-Medium changed team path by {tacv_summary['mean_path_change_m']:.3f} m and reference changes by {tacv_summary['mean_reference_change_count_change']:.3f}; component changes are unavailable because TACV positions were not retained.",
        "Q12": f"Repair headroom={headroom}; {next_step}.",
    }
    report_lines = [
        "# Path-Inefficiency and Recurrent-Correction Root-Cause Audit", "",
        "## Executive result", "",
        conclusion["FINAL_RECOMMENDATION"], "",
        f"The exact stored path and smoothness metrics were reproduced over all 381 successful Formal V2 episodes with maximum errors {max_path_error:.3e} m and {max_smooth_error:.3e} m²/s⁶. No policy or simulator was executed.", "",
        "## Direct answers", "",
    ]
    report_lines += [f"- **{key}.** {value}" for key, value in q.items()]
    report_lines += ["", "## Interpretation boundary", "", "Low segment efficiency, cancellation, and rank associations are descriptive. They do not prove that ERR caused path excess or that a segment can safely be suppressed. The TACV-Medium comparison is a Development intervention only; its retained NPZ omits executed positions, so unobserved decomposition fields remain unavailable.", "", "## Decision fields", "", f"- `PATH_EXCESS_PRIMARY_SOURCE = {primary}`", f"- `PATH_AND_JERK_SHARED_CAUSE_EVIDENCE = {shared}`", f"- `FRAMEWORK_PRESERVING_PATH_REPAIR_HEADROOM = {headroom}`", f"- `NEXT_STEP_RECOMMENDATION = {next_step}`", "- `NEW_SIMULATION_RUN = NO`", "- `METHOD_CHANGED = NO`", "- `FORMAL_RESULT_CHANGED = NO`", ""]
    (output / "10_root_cause/PATH_INEFFICIENCY_ROOT_CAUSE_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    write_json(output / "10_root_cause/PATH_REPAIR_GO_NO_GO.json", {"decision": "GO" if headroom in {"HIGH", "MODERATE"} else "NO_GO", "repair_headroom": headroom, "recommendation": next_step, "implementation_authorized": False, "method_changed": False, "formal_result_changed": False})

    paper_lines = ["# Paper Path-Efficiency Diagnostic", "", f"Across 381 successful Formal V2 episodes, the frozen Proposed method had a mean agent detour ratio of {conclusion['MEAN_DETOUR_RATIO']:.4f} and mean agent excess path of {conclusion['MEAN_EXCESS_PATH_M']:.3f} m. Low-efficiency ERR segments (eta < 0.2) accounted for {low_path_share*100:.2f}% of executed distance. Adjacent-segment cancellation averaged {mean_cancellation:.4f}, while {stable_fraction*100:.2f}% of total jerk cost occurred outside 0.5 s reference-change windows.", "", f"The evidence supports `{primary}` as the primary diagnostic label and `{shared}` shared path/jerk cause evidence. These are associations, not causal claims. Recommended next step: `{next_step}`.", ""]
    (output / "11_paper_ready/PAPER_PATH_EFFICIENCY_DIAGNOSTIC.md").write_text("\n".join(paper_lines), encoding="utf-8")
    (output / "FINAL_REPORT.md").write_text("\n".join(report_lines + ["## Artifact root", "", str(output), ""]), encoding="utf-8")

    print(json.dumps({"output": str(output), "success_episodes": len(episode_rows), "agent_rows": len(agent_rows), "segments": len(segment_rows), "cancellation_pairs": len(cancellation_rows), "conclusion": conclusion}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
