"""Read-only audit of Proposal sector geometry, reference switching, and jerk.

This script never executes a policy or changes a checkpoint.  It reconstructs
the frozen Proposal direction contract from source/config constants and reads
the stored Formal V2 GAT-R event/trajectory records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/sector_resolution_oscillation_audit/20260824_110611"
METHOD_CONFIG = REPO_ROOT / (
    "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/09_final_freeze/"
    "method_configs/M9_Proposed_RERR_GAT_SAC_DMP.json"
)
FORMAL_RECORDS = REPO_ROOT / (
    "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2/"
    "formal_records/M9_Proposed_RERR_GAT_SAC_DMP"
)
FORMAL_MANIFEST = REPO_ROOT / (
    "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2/"
    "FORMAL_V2_MANIFEST.json"
)
DT = 0.1
POST_SWITCH_WINDOW_S = 0.5
BOOTSTRAP_REPLICATES_SPEARMAN = 300
BOOTSTRAP_REPLICATES_CONTRAST = 2000
BOOTSTRAP_SEED = 2026082401


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(str(key))
        fields = ordered
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(json_ready(value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (np.bool_, bool)):
        return int(bool(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else ""
    return value


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return None
    return vector / norm


def angle_between(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return float(np.arccos(cosine))


def direction_angles(direction: np.ndarray | None) -> tuple[float | None, float | None]:
    if direction is None:
        return None, None
    direction = np.asarray(direction, dtype=float)
    return float(math.atan2(direction[1], direction[0])), float(math.asin(np.clip(direction[2], -1.0, 1.0)))


def wrapped_angle_difference(new: float | None, old: float | None) -> float | None:
    if new is None or old is None:
        return None
    return float((new - old + math.pi) % (2.0 * math.pi) - math.pi)


def sector_geometry(config: Mapping[str, Any]) -> dict[str, Any]:
    sensor = config["sensor"]
    azimuth_bins = int(sensor["azimuth_bins"])
    elevation_bins = int(sensor["elevation_bins"])
    training_config = load_json(REPO_ROOT / config["training_config"])
    elevation_range = tuple(float(value) for value in training_config["sensor"]["elevation_range_deg"])
    azimuth = np.linspace(-math.pi, math.pi, azimuth_bins, endpoint=False, dtype=float)
    elevation = np.deg2rad(np.linspace(elevation_range[0], elevation_range[1], elevation_bins, dtype=float))
    directions: list[list[float]] = []
    metadata: list[dict[str, Any]] = []
    for azimuth_index, azimuth_value in enumerate(azimuth):
        for elevation_index, elevation_value in enumerate(elevation):
            cosine = float(math.cos(elevation_value))
            direction = [
                cosine * float(math.cos(azimuth_value)),
                cosine * float(math.sin(azimuth_value)),
                float(math.sin(elevation_value)),
            ]
            sector_id = azimuth_index * elevation_bins + elevation_index
            directions.append(direction)
            metadata.append(
                {
                    "sector_id": sector_id,
                    "sector_label": f"A{azimuth_index:02d}_E{elevation_index:02d}",
                    "azimuth_index": azimuth_index,
                    "elevation_index": elevation_index,
                    "azimuth_rad": float(azimuth_value),
                    "azimuth_deg": float(math.degrees(azimuth_value)),
                    "elevation_rad": float(elevation_value),
                    "elevation_deg": float(math.degrees(elevation_value)),
                    "direction": direction,
                }
            )
    return {
        "azimuth_bins": azimuth_bins,
        "elevation_bins": elevation_bins,
        "azimuth": azimuth,
        "elevation": elevation,
        "directions": np.asarray(directions, dtype=float),
        "metadata": metadata,
        "elevation_range_deg": elevation_range,
    }


def sector_indices(sector_id: int, elevation_bins: int) -> tuple[int, int]:
    return int(sector_id) // int(elevation_bins), int(sector_id) % int(elevation_bins)


def nearest_sector(direction: np.ndarray | None, geometry: Mapping[str, Any]) -> tuple[int | None, float | None]:
    if direction is None:
        return None, None
    dots = np.asarray(geometry["directions"], dtype=float) @ np.asarray(direction, dtype=float)
    index = int(np.argmax(dots))
    return index, float(math.acos(np.clip(dots[index], -1.0, 1.0)))


def cyclic_azimuth_delta(left: int, right: int, count: int) -> int:
    direct = abs(int(left) - int(right))
    return min(direct, int(count) - direct)


def adjacent(left: int | None, right: int | None, geometry: Mapping[str, Any]) -> tuple[bool | None, str | None]:
    if left is None or right is None or left == right:
        return (False if left == right and left is not None else None), None
    left_azimuth, left_elevation = sector_indices(left, int(geometry["elevation_bins"]))
    right_azimuth, right_elevation = sector_indices(right, int(geometry["elevation_bins"]))
    azimuth_delta = cyclic_azimuth_delta(left_azimuth, right_azimuth, int(geometry["azimuth_bins"]))
    elevation_delta = abs(left_elevation - right_elevation)
    is_adjacent = azimuth_delta <= 1 and elevation_delta <= 1
    if not is_adjacent:
        return False, "non_adjacent"
    if azimuth_delta == 1 and elevation_delta == 0:
        return True, "azimuth_only"
    if azimuth_delta == 0 and elevation_delta == 1:
        return True, "elevation_only"
    return True, "diagonal"


def source_adjacency_pairs(geometry: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    azimuth_bins = int(geometry["azimuth_bins"])
    elevation_bins = int(geometry["elevation_bins"])
    pairs: set[tuple[int, int]] = set()
    rows: list[tuple[int, int, str]] = []
    for azimuth_index in range(azimuth_bins):
        for elevation_index in range(elevation_bins):
            source = azimuth_index * elevation_bins + elevation_index
            for azimuth_offset in (-1, 0, 1):
                for elevation_offset in (-1, 0, 1):
                    if azimuth_offset == 0 and elevation_offset == 0:
                        continue
                    neighbor_azimuth = (azimuth_index + azimuth_offset) % azimuth_bins
                    neighbor_elevation = elevation_index + elevation_offset
                    if not 0 <= neighbor_elevation < elevation_bins:
                        continue
                    target = neighbor_azimuth * elevation_bins + neighbor_elevation
                    key = tuple(sorted((source, target)))
                    if key in pairs:
                        continue
                    pairs.add(key)
                    _, kind = adjacent(source, target, geometry)
                    rows.append((key[0], key[1], str(kind)))
    return sorted(rows)


def finite(values: Iterable[Any]) -> np.ndarray:
    result = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=float)
    return result


def describe(values: Iterable[Any]) -> dict[str, Any]:
    data = finite(values)
    if not data.size:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None, "p90": None, "p95": None, "min": None, "max": None}
    return {
        "n": int(data.size),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def event_jerk_metrics(accelerations: np.ndarray, agent_id: int, step: int) -> dict[str, Any]:
    acceleration = np.asarray(accelerations[:, int(agent_id), :], dtype=float)
    if acceleration.shape[0] < 2:
        return {"pre_switch_jerk_peak": None, "pre_switch_vertical_jerk_peak": None, "post_switch_jerk_peak": None, "post_switch_vertical_jerk_peak": None, "post_switch_lateral_jerk_peak": None, "post_switch_mean_squared_jerk": None, "post_switch_jerk_sample_count": 0, "pre_switch_jerk_sample_count": 0}
    jerk = np.diff(acceleration, axis=0) / DT
    jerk_steps = np.arange(1, acceleration.shape[0], dtype=int)
    last_step = int(step + round(POST_SWITCH_WINDOW_S / DT))
    first_pre_step = int(step - round(POST_SWITCH_WINDOW_S / DT))
    pre_mask = (jerk_steps >= first_pre_step) & (jerk_steps < int(step))
    mask = (jerk_steps >= int(step)) & (jerk_steps <= last_step)
    pre_selected = jerk[pre_mask]
    selected = jerk[mask]
    if not selected.size:
        return {"pre_switch_jerk_peak": None, "pre_switch_vertical_jerk_peak": None, "post_switch_jerk_peak": None, "post_switch_vertical_jerk_peak": None, "post_switch_lateral_jerk_peak": None, "post_switch_mean_squared_jerk": None, "post_switch_jerk_sample_count": 0, "pre_switch_jerk_sample_count": int(pre_selected.shape[0])}
    pre_peak = float(np.max(np.linalg.norm(pre_selected, axis=1))) if pre_selected.size else None
    pre_vertical_peak = float(np.max(np.abs(pre_selected[:, 2]))) if pre_selected.size else None
    post_peak = float(np.max(np.linalg.norm(selected, axis=1)))
    post_vertical_peak = float(np.max(np.abs(selected[:, 2])))
    return {
        "pre_switch_jerk_peak": pre_peak,
        "pre_switch_vertical_jerk_peak": pre_vertical_peak,
        "post_switch_jerk_peak": post_peak,
        "post_switch_vertical_jerk_peak": post_vertical_peak,
        "post_switch_lateral_jerk_peak": float(np.max(np.linalg.norm(selected[:, :2], axis=1))),
        "post_switch_mean_squared_jerk": float(np.mean(np.sum(selected**2, axis=1))),
        "post_switch_jerk_sample_count": int(selected.shape[0]),
        "pre_switch_jerk_sample_count": int(pre_selected.shape[0]),
        "post_minus_pre_jerk_peak": None if pre_peak is None else post_peak - pre_peak,
        "post_minus_pre_vertical_jerk_peak": None if pre_vertical_peak is None else post_vertical_peak - pre_vertical_peak,
    }


def build_event_rows(geometry: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    record_paths = sorted(FORMAL_RECORDS.glob("FORMAL_LR_*.json"))
    rows: list[dict[str, Any]] = []
    counters: defaultdict[str, int] = defaultdict(int)
    for record_index, record_path in enumerate(record_paths, start=1):
        record = load_json(record_path)
        if not bool(record["episode"]["team_success"]):
            continue
        trajectory_path = FORMAL_RECORDS / str(record["trajectory_file"])
        with np.load(trajectory_path) as trajectory:
            positions = np.asarray(trajectory["positions"], dtype=float)
            accelerations = np.asarray(trajectory["accelerations"], dtype=float)
        stage = str(record["episode"]["stage"])
        family = str(record["episode"]["family"])
        events_by_agent: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for event in record["events"]:
            if bool(event.get("goal_changed", False)):
                events_by_agent[int(event["agent_id"])].append(event)
        for agent_id, events in events_by_agent.items():
            events = sorted(events, key=lambda item: (int(item["step"]), str(item["event"])))
            selected_history: list[int | None] = []
            previous_time: float | None = None
            previous_switch_time: float | None = None
            for sequence_index, event in enumerate(events):
                step = int(event["step"])
                position = positions[min(step, positions.shape[0] - 1), agent_id]
                old_reference = np.asarray(event["old_active_goal"], dtype=float)
                new_reference = np.asarray(event["new_active_goal"], dtype=float)
                old_direction = unit(old_reference - position)
                new_direction = unit(new_reference - position)
                old_active_nearest_sector, old_active_center_error = nearest_sector(old_direction, geometry)
                selected_null = bool(event.get("selected_null", False))
                new_active_goal_type = str(event.get("new_active_goal_type") or "")
                selected_candidate_id = event.get("selected_candidate_id")
                points = event.get("candidate_world_points", [])
                is_candidate_reference = bool(
                    new_active_goal_type == "reference"
                    and selected_candidate_id is not None
                    and 0 <= int(selected_candidate_id) < len(points)
                )
                new_sector: int | None = None
                new_center_error: float | None = None
                candidate_point_error: float | None = None
                if is_candidate_reference:
                    selected_point = np.asarray(points[int(selected_candidate_id)], dtype=float)
                    candidate_point_error = float(np.linalg.norm(selected_point - new_reference))
                    candidate_direction = unit(selected_point - position)
                    new_sector, new_center_error = nearest_sector(candidate_direction, geometry)
                old_selected_sector = selected_history[-1] if selected_history else None
                is_adjacent, adjacent_type = adjacent(old_selected_sector, new_sector, geometry)
                actual_angular_jump = angle_between(old_direction, new_direction)
                old_azimuth, old_elevation = direction_angles(old_direction)
                new_azimuth, new_elevation = direction_angles(new_direction)
                azimuth_jump = wrapped_angle_difference(new_azimuth, old_azimuth)
                elevation_jump = None if new_elevation is None or old_elevation is None else float(new_elevation - old_elevation)
                center_jump = None
                sector_azimuth_jump = None
                sector_elevation_jump = None
                if old_selected_sector is not None and new_sector is not None:
                    center_jump = angle_between(
                        np.asarray(geometry["directions"])[old_selected_sector],
                        np.asarray(geometry["directions"])[new_sector],
                    )
                    old_meta = geometry["metadata"][old_selected_sector]
                    new_meta = geometry["metadata"][new_sector]
                    sector_azimuth_jump = wrapped_angle_difference(float(new_meta["azimuth_rad"]), float(old_meta["azimuth_rad"]))
                    sector_elevation_jump = float(new_meta["elevation_rad"] - old_meta["elevation_rad"])
                reverse = bool(
                    len(selected_history) >= 2
                    and new_sector is not None
                    and selected_history[-1] is not None
                    and selected_history[-2] is not None
                    and new_sector == selected_history[-2]
                    and new_sector != selected_history[-1]
                )
                abab = bool(
                    len(selected_history) >= 3
                    and new_sector is not None
                    and selected_history[-3] is not None
                    and selected_history[-2] is not None
                    and selected_history[-1] is not None
                    and selected_history[-3] == selected_history[-1]
                    and selected_history[-2] == new_sector
                    and selected_history[-3] != selected_history[-2]
                )
                event_time = float(event.get("time_s", step * DT))
                switchback_latency = (
                    event_time - previous_switch_time if reverse and previous_switch_time is not None else None
                )
                gat_margin = event.get("top1_top2_probability_margin")
                row = {
                    "episode_id": str(record["scenario_id"]),
                    "stage": stage,
                    "family": family,
                    "agent_id": agent_id,
                    "event_sequence_index": sequence_index,
                    "time_step": step,
                    "time_s": event_time,
                    "event_type": str(event["event"]),
                    "rerr_trigger_reason": "|".join(map(str, event.get("trigger_reasons", []))) or str(event["event"]),
                    "counts_as_reproposal": bool(event.get("counts_as_reproposal", False)),
                    "old_sector_id": old_selected_sector,
                    "new_sector_id": new_sector,
                    "old_active_direction_nearest_sector_id": old_active_nearest_sector,
                    "old_reference": old_reference.tolist(),
                    "new_reference": new_reference.tolist(),
                    "position_at_switch": position.tolist(),
                    "old_direction": None if old_direction is None else old_direction.tolist(),
                    "new_direction": None if new_direction is None else new_direction.tolist(),
                    "angular_jump_rad": actual_angular_jump,
                    "angular_jump_deg": None if actual_angular_jump is None else math.degrees(actual_angular_jump),
                    "sector_center_angular_jump_rad": center_jump,
                    "sector_center_angular_jump_deg": None if center_jump is None else math.degrees(center_jump),
                    "azimuth_jump_rad": azimuth_jump,
                    "azimuth_jump_deg": None if azimuth_jump is None else math.degrees(azimuth_jump),
                    "elevation_jump_rad": elevation_jump,
                    "elevation_jump_deg": None if elevation_jump is None else math.degrees(elevation_jump),
                    "sector_azimuth_jump_rad": sector_azimuth_jump,
                    "sector_azimuth_jump_deg": None if sector_azimuth_jump is None else math.degrees(sector_azimuth_jump),
                    "sector_elevation_jump_rad": sector_elevation_jump,
                    "sector_elevation_jump_deg": None if sector_elevation_jump is None else math.degrees(sector_elevation_jump),
                    "candidate_changed": bool(old_selected_sector is not None and new_sector is not None and old_selected_sector != new_sector),
                    "candidate_to_candidate_switch": bool(old_selected_sector is not None and new_sector is not None),
                    "sectors_adjacent": is_adjacent,
                    "adjacency_type": adjacent_type,
                    "reverse_of_previous_switch": reverse,
                    "abab_pattern_completed": abab,
                    "switchback_latency_s": switchback_latency,
                    "dwell_time_since_previous_reference_change_s": None if previous_time is None else event_time - previous_time,
                    "old_reference_distance_m": float(np.linalg.norm(old_reference - position)),
                    "new_reference_distance_m": float(np.linalg.norm(new_reference - position)),
                    "gat_top1_top2_probability_margin": gat_margin,
                    "selected_candidate_id_event_local": event.get("selected_candidate_id"),
                    "selected_null": selected_null,
                    "new_active_goal_type": new_active_goal_type,
                    "is_candidate_reference": is_candidate_reference,
                    "candidate_point_match_error_m": candidate_point_error,
                    "new_direction_to_sector_center_error_rad": new_center_error,
                    "new_direction_to_sector_center_error_deg": None if new_center_error is None else math.degrees(new_center_error),
                    "old_active_direction_to_nearest_center_error_deg": None if old_active_center_error is None else math.degrees(old_active_center_error),
                    **event_jerk_metrics(accelerations, agent_id, step),
                }
                rows.append(row)
                counters["actual_reference_changes"] += 1
                if is_candidate_reference:
                    counters["candidate_reference_changes"] += 1
                else:
                    counters["noncandidate_reference_changes"] += 1
                if selected_null:
                    counters["null_reference_changes"] += 1
                if candidate_point_error is not None and candidate_point_error > 1.0e-8:
                    counters["candidate_point_mismatch"] += 1
                if new_center_error is not None and new_center_error > 1.0e-7:
                    counters["candidate_direction_center_mismatch"] += 1
                selected_history.append(new_sector)
                previous_time = event_time
                if old_selected_sector is not None and new_sector is not None:
                    previous_switch_time = event_time
        if record_index % 50 == 0:
            print(f"[sector-audit] read {record_index}/{len(record_paths)} records", flush=True)
    integrity = {
        "formal_record_count": len(record_paths),
        "successful_record_count": len({row["episode_id"] for row in rows}),
        **dict(counters),
        "event_table_scope": "successful Formal V2 M9 episodes only",
        "trajectory_sampling_s": DT,
        "post_switch_jerk_window_s": POST_SWITCH_WINDOW_S,
        "candidate_direction_exact_center_tolerance_rad": 1.0e-7,
    }
    return rows, integrity


def candidate_switch_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row["candidate_to_candidate_switch"] and row["candidate_changed"]]


def cluster_bootstrap_spearman(rows: Sequence[Mapping[str, Any]], x_field: str, y_field: str) -> dict[str, Any]:
    valid = [row for row in rows if row.get(x_field) is not None and row.get(y_field) is not None]
    if len(valid) < 3:
        return {"n": len(valid), "estimate": None, "ci_low": None, "ci_high": None, "p_value": None, "bootstrap_replicates": 0}
    x = np.asarray([float(row[x_field]) for row in valid], dtype=float)
    if "jump" in x_field:
        # Direction jumps are exactly discrete sector-center quantities.  Round
        # below physical/reporting precision so parser-dependent binary noise
        # does not break rank ties during independent reconciliation.
        x = np.round(x, 9)
    y = np.asarray([float(row[y_field]) for row in valid], dtype=float)
    estimate = float(spearmanr(x, y).statistic)
    p_value = float(spearmanr(x, y).pvalue)
    grouped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for episode_id in sorted({str(row["episode_id"]) for row in valid}):
        members = [row for row in valid if str(row["episode_id"]) == episode_id]
        group_x = np.asarray([float(row[x_field]) for row in members])
        if "jump" in x_field:
            group_x = np.round(group_x, 9)
        grouped[episode_id] = (
            group_x,
            np.asarray([float(row[y_field]) for row in members]),
        )
    keys = list(grouped)
    rng = np.random.default_rng(BOOTSTRAP_SEED + sum(ord(char) for char in x_field + y_field))
    boot: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES_SPEARMAN):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        sample_x = np.concatenate([grouped[key][0] for key in sampled])
        sample_y = np.concatenate([grouped[key][1] for key in sampled])
        value = float(spearmanr(sample_x, sample_y).statistic)
        if math.isfinite(value):
            boot.append(value)
    return {
        "n": len(valid),
        "episode_cluster_count": len(keys),
        "estimate": estimate,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_value": p_value,
        "bootstrap_replicates": len(boot),
        "bootstrap_unit": "episode",
    }


def cluster_bootstrap_mean_contrast(rows: Sequence[Mapping[str, Any]], group_field: str, y_field: str) -> dict[str, Any]:
    valid = [row for row in rows if row.get(group_field) is not None and row.get(y_field) is not None]
    positive = np.asarray([float(row[y_field]) for row in valid if bool(row[group_field])], dtype=float)
    negative = np.asarray([float(row[y_field]) for row in valid if not bool(row[group_field])], dtype=float)
    if not positive.size or not negative.size:
        return {"n": len(valid), "n_positive": int(positive.size), "n_negative": int(negative.size), "estimate": None, "ci_low": None, "ci_high": None, "p_value": None, "bootstrap_replicates": 0}
    estimate = float(np.mean(positive) - np.mean(negative))
    p_value = float(mannwhitneyu(positive, negative, alternative="two-sided").pvalue)
    episode_ids = sorted({str(row["episode_id"]) for row in valid})
    per_episode: dict[str, tuple[float, int, float, int]] = {}
    for episode_id in episode_ids:
        members = [row for row in valid if str(row["episode_id"]) == episode_id]
        pos = [float(row[y_field]) for row in members if bool(row[group_field])]
        neg = [float(row[y_field]) for row in members if not bool(row[group_field])]
        per_episode[episode_id] = (sum(pos), len(pos), sum(neg), len(neg))
    rng = np.random.default_rng(BOOTSTRAP_SEED + sum(ord(char) for char in group_field + y_field))
    boot: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES_CONTRAST):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        pos_sum = sum(per_episode[key][0] for key in sampled)
        pos_n = sum(per_episode[key][1] for key in sampled)
        neg_sum = sum(per_episode[key][2] for key in sampled)
        neg_n = sum(per_episode[key][3] for key in sampled)
        if pos_n and neg_n:
            boot.append(pos_sum / pos_n - neg_sum / neg_n)
    return {
        "n": len(valid),
        "n_positive": int(positive.size),
        "n_negative": int(negative.size),
        "positive_mean": float(np.mean(positive)),
        "negative_mean": float(np.mean(negative)),
        "positive_median": float(np.median(positive)),
        "negative_median": float(np.median(negative)),
        "estimate": estimate,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_value": p_value,
        "bootstrap_replicates": len(boot),
        "bootstrap_unit": "episode",
    }


def cluster_bootstrap_paired_delta(rows: Sequence[Mapping[str, Any]], post_field: str, pre_field: str) -> dict[str, Any]:
    valid = [row for row in rows if row.get(post_field) is not None and row.get(pre_field) is not None]
    if not valid:
        return {"n": 0, "estimate": None, "ci_low": None, "ci_high": None, "p_value": None, "bootstrap_replicates": 0}
    post = np.asarray([float(row[post_field]) for row in valid], dtype=float)
    pre = np.asarray([float(row[pre_field]) for row in valid], dtype=float)
    delta = post - pre
    episode_ids = sorted({str(row["episode_id"]) for row in valid})
    per_episode: dict[str, tuple[float, int]] = {}
    for episode_id in episode_ids:
        members = [row for row in valid if str(row["episode_id"]) == episode_id]
        values = [float(row[post_field]) - float(row[pre_field]) for row in members]
        per_episode[episode_id] = (sum(values), len(values))
    rng = np.random.default_rng(BOOTSTRAP_SEED + sum(ord(char) for char in post_field + pre_field))
    boot: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES_CONTRAST):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        total = sum(per_episode[key][0] for key in sampled)
        count = sum(per_episode[key][1] for key in sampled)
        if count:
            boot.append(total / count)
    return {
        "n": len(valid),
        "post_mean": float(np.mean(post)),
        "pre_mean": float(np.mean(pre)),
        "post_median": float(np.median(post)),
        "pre_median": float(np.median(pre)),
        "fraction_post_greater_than_pre": float(np.mean(delta > 0.0)),
        "estimate": float(np.mean(delta)),
        "median_paired_delta": float(np.median(delta)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_value": float(wilcoxon(post, pre, alternative="two-sided").pvalue),
        "bootstrap_replicates": len(boot),
        "bootstrap_unit": "episode",
    }


def summarize_switch_scope(rows: Sequence[Mapping[str, Any]], scope: str) -> dict[str, Any]:
    data = list(rows) if scope == "overall" else [row for row in rows if row["stage"] == scope]
    adjacent_rows = [row for row in data if bool(row.get("sectors_adjacent"))]
    reverse_rows = [row for row in data if bool(row.get("reverse_of_previous_switch"))]
    high_rows = [row for row in data if bool(row.get("high_jerk_switch"))]
    elevation_dominant = [
        row for row in data
        if abs(float(row.get("sector_elevation_jump_deg") or 0.0))
        > abs(float(row.get("sector_azimuth_jump_deg") or 0.0))
    ]
    return {
        "scope": scope,
        "candidate_sector_switch_count": len(data),
        "adjacent_switch_count": len(adjacent_rows),
        "adjacent_switch_fraction": len(adjacent_rows) / len(data) if data else None,
        "high_jerk_switch_count": len(high_rows),
        "high_jerk_adjacent_switch_count": sum(bool(row.get("sectors_adjacent")) for row in high_rows),
        "high_jerk_adjacent_switch_fraction": (
            sum(bool(row.get("sectors_adjacent")) for row in high_rows) / len(high_rows)
            if high_rows else None
        ),
        "switchback_count_1s": sum(bool(row.get("switchback_within_1_0s")) for row in data),
        "switchback_rate_1s": (
            sum(bool(row.get("switchback_within_1_0s")) for row in data) / len(data)
            if data else None
        ),
        "abab_pattern_count": sum(bool(row.get("abab_pattern_completed")) for row in data),
        "elevation_dominant_switch_fraction": len(elevation_dominant) / len(data) if data else None,
        "mean_sector_center_angular_jump_deg": describe(row.get("sector_center_angular_jump_deg") for row in data)["mean"],
        "median_sector_center_angular_jump_deg": describe(row.get("sector_center_angular_jump_deg") for row in data)["median"],
        "mean_abs_sector_elevation_jump_deg": describe(abs(float(row.get("sector_elevation_jump_deg") or 0.0)) for row in data)["mean"],
        "median_dwell_time_s": describe(row.get("dwell_time_since_previous_reference_change_s") for row in data)["median"],
        "mean_post_switch_jerk_peak_mps3": describe(row.get("post_switch_jerk_peak") for row in data)["mean"],
        "median_post_switch_jerk_peak_mps3": describe(row.get("post_switch_jerk_peak") for row in data)["median"],
        "mean_vertical_jerk_peak_mps3": describe(row.get("post_switch_vertical_jerk_peak") for row in data)["mean"],
        "mean_post_switch_squared_jerk_m2_s6": describe(row.get("post_switch_mean_squared_jerk") for row in data)["mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    config = load_json(METHOD_CONFIG)
    geometry = sector_geometry(config)
    pairs = source_adjacency_pairs(geometry)

    event_rows, integrity = build_event_rows(geometry)
    switches = candidate_switch_rows(event_rows)
    high_jerk_threshold = float(np.percentile(finite(row["post_switch_jerk_peak"] for row in switches), 90))
    for row in event_rows:
        row["high_jerk_threshold_mps3"] = high_jerk_threshold
        row["high_jerk_switch"] = bool(
            row.get("post_switch_jerk_peak") is not None
            and float(row["post_switch_jerk_peak"]) >= high_jerk_threshold
        )
        latency = row.get("switchback_latency_s")
        for window in (0.2, 0.5, 1.0, 2.0):
            row[f"switchback_within_{str(window).replace('.', '_')}s"] = bool(
                row.get("reverse_of_previous_switch")
                and latency is not None
                and float(latency) <= window + 1.0e-9
            )
    switches = candidate_switch_rows(event_rows)

    reference_distances = finite(
        row["new_reference_distance_m"]
        for row in event_rows
        if row["is_candidate_reference"]
    )
    distance_quantiles = {
        "P25": float(np.percentile(reference_distances, 25)),
        "P50": float(np.percentile(reference_distances, 50)),
        "P75": float(np.percentile(reference_distances, 75)),
        "P95": float(np.percentile(reference_distances, 95)),
    }
    resolution_rows: list[dict[str, Any]] = []
    adjacent_angles: list[float] = []
    for left, right, pair_type in pairs:
        angle_rad = float(angle_between(geometry["directions"][left], geometry["directions"][right]))
        adjacent_angles.append(angle_rad)
        left_meta = geometry["metadata"][left]
        right_meta = geometry["metadata"][right]
        for quantile_name, radius in distance_quantiles.items():
            resolution_rows.append(
                {
                    "left_sector_id": left,
                    "right_sector_id": right,
                    "adjacency_type": pair_type,
                    "left_azimuth_index": left_meta["azimuth_index"],
                    "left_elevation_index": left_meta["elevation_index"],
                    "right_azimuth_index": right_meta["azimuth_index"],
                    "right_elevation_index": right_meta["elevation_index"],
                    "angular_distance_rad": angle_rad,
                    "angular_distance_deg": math.degrees(angle_rad),
                    "reference_distance_quantile": quantile_name,
                    "reference_distance_m": radius,
                    "expected_reference_displacement_m": 2.0 * radius * math.sin(angle_rad / 2.0),
                }
            )
    azimuth_spacing = 2.0 * math.pi / int(geometry["azimuth_bins"])
    elevation_spacing = float(np.diff(geometry["elevation"])[0])
    angle_stats = describe(adjacent_angles)

    contract = {
        "schema_version": "proposal_sector_contract_v1",
        "source_of_truth": {
            "proposal": "Guidance/reference_point_proposal_demo.py::propose_reference_points",
            "sensor_geometry": "Entity/sensors.py::LocalObstacleSensor",
            "top_k_adapter": "planning/policy_preview.py::adapt_candidate_proposals",
            "gat_projection": "planning/heterogeneous_candidate_graph.py::_canonical_gat_sector_projection",
            "formal_method_config": str(METHOD_CONFIG.relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "proposal_candidate_sector_count": int(geometry["directions"].shape[0]),
        "proposal_azimuth_bins": int(geometry["azimuth_bins"]),
        "proposal_elevation_bins": int(geometry["elevation_bins"]),
        "proposal_unique_direction_center_count": int(np.unique(np.round(geometry["directions"], 14), axis=0).shape[0]),
        "proposal_geometric_direction_center_count": 256,
        "actual_feasible_candidate_count_per_state": "1..256 after source feasibility/progress filters; exact pre-Top-K count is not retained in Formal event records",
        "raw_candidate_upper_bound_before_top_k": 256,
        "top_k": int(config["top_k"]),
        "fp_shep_input": "ordered feasible Proposal list truncated to Top-K=10",
        "gat_receives_proposal_sector_id": False,
        "gat_receives_continuous_candidate_direction_xyz": True,
        "gat_input_dim_depends_on_proposal_sector_count": False,
        "gat_canonical_projection_direction_count": 56,
        "gat_canonical_projection_depends_on_proposal_list_or_count": False,
        "gat_canonical_projection_depends_on_shared_sensor_scan_grid": True,
        "proposal_and_gat_direction_contract": "COUPLED",
        "coupling_explanation": "Proposal directions and the 56-value GAT safety field both read the same 16x16 sensor grid, but GAT projects that grid to a fixed 8x7 basis and does not consume Proposal sector IDs.",
        "azimuth_grid": "uniform [-pi,pi), endpoint excluded",
        "elevation_grid": "uniform in elevation angle from -80 to +80 degrees, endpoints included",
        "uniform_on_sphere": False,
        "uniformity_explanation": "uniform angle spacing is not equal-solid-angle sampling; azimuth-neighbor great-circle spacing contracts near +/-80 degrees",
        "source_supported_adjacency": "Moore neighborhood induced by scan_guard_azimuth_bins=1 and scan_guard_elevation_bins=1; azimuth wraps and elevation does not",
        "source_supported_unique_adjacency_pair_count": len(pairs),
        "reference_distance_rule": {
            "desired_step": "clip(eta*safety_margin, minimum_step, maximum_step)",
            "actual_step": "min(desired_step, obstacle_distance-effective_safe_radius, sensing_radius, maximum_step)",
            "s_min_m": 0.35,
            "s_max_m": 1.05,
            "terminal_min_step_m": 0.05,
            "terminal_step_ratio": 0.55,
            "eta": 0.58,
            "formal_observed_selected_reference_distance_quantiles_m": distance_quantiles,
        },
        "sector_centers": geometry["metadata"],
    }
    write_json(output / "01_sector_contract/PROPOSAL_SECTOR_CONTRACT.json", contract)
    write_csv(output / "01_sector_contract/SECTOR_ANGULAR_RESOLUTION.csv", resolution_rows)
    write_json(
        output / "01_sector_contract/SECTOR_ANGULAR_RESOLUTION_SUMMARY.json",
        {
            "azimuth_spacing_rad": azimuth_spacing,
            "azimuth_spacing_deg": math.degrees(azimuth_spacing),
            "elevation_spacing_rad": elevation_spacing,
            "elevation_spacing_deg": math.degrees(elevation_spacing),
            "adjacent_pair_angular_distance_rad": angle_stats,
            "adjacent_pair_angular_distance_deg": {
                key: (math.degrees(value) if isinstance(value, float) else value)
                for key, value in angle_stats.items()
            },
            "reference_distance_quantiles_m": distance_quantiles,
        },
    )

    candidate_elevations = np.asarray([meta["elevation_deg"] for meta in geometry["metadata"]])
    unique_elevations = np.asarray([math.degrees(value) for value in geometry["elevation"]])
    sign_sequences: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        if row["new_sector_id"] is not None:
            sign_sequences[(str(row["episode_id"]), int(row["agent_id"]))].append(row)
    positive_negative_positive = 0
    negative_positive_negative = 0
    elevation_level_aba_1s = 0
    elevation_eligible = 0
    for sequence in sign_sequences.values():
        sequence = sorted(sequence, key=lambda row: float(row["time_s"]))
        elevations = [float(geometry["metadata"][int(row["new_sector_id"])]["elevation_deg"]) for row in sequence]
        for index in range(2, len(sequence)):
            first, middle, last = elevations[index - 2 : index + 1]
            positive_negative_positive += int(first > 0.0 and middle < 0.0 and last > 0.0)
            negative_positive_negative += int(first < 0.0 and middle > 0.0 and last < 0.0)
            elevation_eligible += 1
            latency = float(sequence[index]["time_s"]) - float(sequence[index - 1]["time_s"])
            elevation_level_aba_1s += int(first == last and first != middle and latency <= 1.0 + 1.0e-9)
    vertical_audit = {
        "proposal_elevation_level_count": int(geometry["elevation_bins"]),
        "unique_elevation_angles_deg": unique_elevations.tolist(),
        "adjacent_elevation_differences_deg": np.diff(unique_elevations).tolist(),
        "upward_sector_count": int(np.sum(candidate_elevations > 0.0)),
        "exact_level_sector_count": int(np.sum(candidate_elevations == 0.0)),
        "downward_sector_count": int(np.sum(candidate_elevations < 0.0)),
        "near_level_definition_deg": abs(float(unique_elevations[7])),
        "near_level_sector_count_within_half_elevation_step": int(np.sum(np.abs(candidate_elevations) <= abs(float(unique_elevations[7])) + 1.0e-9)),
        "positive_negative_positive_pattern_count": positive_negative_positive,
        "negative_positive_negative_pattern_count": negative_positive_negative,
        "elevation_level_ABA_within_1s_count": elevation_level_aba_1s,
        "elevation_level_ABA_within_1s_rate": elevation_level_aba_1s / elevation_eligible if elevation_eligible else None,
        "eligible_three_selection_sequences": elevation_eligible,
        "stage_switch_statistics": [summarize_switch_scope(switches, scope) for scope in ("Stage I", "Stage II", "Stage III", "Stage IV")],
    }
    write_json(output / "01_sector_contract/VERTICAL_SECTOR_RESOLUTION_AUDIT.json", vertical_audit)
    write_csv(output / "02_existing_event_analysis/SECTOR_SWITCH_EVENT_TABLE.csv", event_rows)

    switch_statistics = [summarize_switch_scope(switches, scope) for scope in ("overall", "Stage I", "Stage II", "Stage III", "Stage IV")]
    write_csv(output / "02_existing_event_analysis/SECTOR_SWITCH_STATISTICS.csv", switch_statistics)
    chatter_windows: dict[str, Any] = {}
    for window in (0.2, 0.5, 1.0, 2.0):
        field = f"switchback_within_{str(window).replace('.', '_')}s"
        chatter_windows[str(window)] = {
            "count": int(sum(bool(row[field]) for row in switches)),
            "rate": float(np.mean([bool(row[field]) for row in switches])) if switches else None,
        }
    overall_stats = switch_statistics[0]
    chatter = {
        "scope": "successful Formal V2 M9 candidate-to-candidate changed-sector switches",
        "adjacency_definition": contract["source_supported_adjacency"],
        "high_jerk_definition": "post-switch 0.5 s jerk-norm peak at or above the successful-Formal P90",
        "high_jerk_threshold_mps3": high_jerk_threshold,
        "all_switches": overall_stats,
        "switchback_windows_s": chatter_windows,
        "abab_pattern_count": int(sum(bool(row["abab_pattern_completed"]) for row in switches)),
        "by_adjacency_type": {
            kind: summarize_switch_scope([row for row in switches if row.get("adjacency_type") == kind], "overall")
            for kind in ("azimuth_only", "elevation_only", "diagonal", "non_adjacent")
        },
        "direction_subsets": {
            "azimuth_only": summarize_switch_scope(
                [row for row in switches if abs(float(row.get("sector_elevation_jump_deg") or 0.0)) < 1.0e-9], "overall"
            ),
            "elevation_dominant": summarize_switch_scope(
                [row for row in switches if abs(float(row.get("sector_elevation_jump_deg") or 0.0)) > abs(float(row.get("sector_azimuth_jump_deg") or 0.0))], "overall"
            ),
            "all_directions": overall_stats,
        },
    }
    write_json(output / "02_existing_event_analysis/ADJACENT_SECTOR_CHATTER_AUDIT.json", chatter)

    associations: list[dict[str, Any]] = []
    for name, x_field, y_field in (
        ("sector_center_angular_jump_vs_post_switch_jerk_peak", "sector_center_angular_jump_deg", "post_switch_jerk_peak"),
        ("absolute_elevation_jump_vs_vertical_jerk_peak", "abs_sector_elevation_jump_deg", "post_switch_vertical_jerk_peak"),
        ("gat_margin_vs_post_switch_jerk_peak", "gat_top1_top2_probability_margin", "post_switch_jerk_peak"),
    ):
        prepared = []
        for row in switches:
            item = dict(row)
            item["abs_sector_elevation_jump_deg"] = abs(float(row.get("sector_elevation_jump_deg") or 0.0))
            prepared.append(item)
        result = cluster_bootstrap_spearman(prepared, x_field, y_field)
        associations.append({"relationship": name, "analysis": "Spearman rank association", **result})
    for name, group_field, y_field in (
        ("adjacent_vs_nonadjacent_post_switch_jerk_peak", "sectors_adjacent", "post_switch_jerk_peak"),
        ("switchback_vs_ordinary_post_switch_jerk_peak", "switchback_within_1_0s", "post_switch_jerk_peak"),
        ("elevation_switch_vs_no_elevation_switch_vertical_jerk_peak", "has_elevation_change", "post_switch_vertical_jerk_peak"),
    ):
        prepared = []
        for row in switches:
            item = dict(row)
            item["has_elevation_change"] = abs(float(row.get("sector_elevation_jump_deg") or 0.0)) > 1.0e-9
            prepared.append(item)
        result = cluster_bootstrap_mean_contrast(prepared, group_field, y_field)
        associations.append({"relationship": name, "analysis": "positive-minus-negative mean contrast; Mann-Whitney p; episode-cluster bootstrap CI", **result})
    transient_tests = []
    for name, post_field, pre_field in (
        ("post_vs_pre_switch_jerk_peak", "post_switch_jerk_peak", "pre_switch_jerk_peak"),
        ("post_vs_pre_switch_vertical_jerk_peak", "post_switch_vertical_jerk_peak", "pre_switch_vertical_jerk_peak"),
    ):
        result = cluster_bootstrap_paired_delta(switches, post_field, pre_field)
        transient_tests.append({"relationship": name, **result})
        associations.append(
            {
                "relationship": name,
                "analysis": "post-minus-pre paired mean; Wilcoxon p; episode-cluster bootstrap CI",
                **result,
            }
        )
    write_csv(output / "02_existing_event_analysis/SECTOR_JERK_ASSOCIATION.csv", associations)
    write_json(
        output / "02_existing_event_analysis/SWITCH_TRANSIENT_CONTROL_AUDIT.json",
        {
            "window_s": POST_SWITCH_WINDOW_S,
            "scope": "successful Formal V2 candidate-sector changes",
            "tests": transient_tests,
            "limitation": "Frequent recurrent changes make some pre-switch windows overlap the response to an earlier switch; this paired test identifies temporal concentration, not isolated causality.",
        },
    )

    boundary_audit = {
        "continuous_pre_quantization_direction_available": False,
        "distance_to_sector_boundary": "UNAVAILABLE",
        "reason": "Proposal enumerates exact sensor-ray centers directly; no continuous desired direction is quantized into a sector before candidate generation.",
        "selected_candidate_direction_center_error": describe(row.get("new_direction_to_sector_center_error_deg") for row in event_rows if row["new_sector_id"] is not None),
        "gat_top1_top2_margin_available": True,
        "gat_margin_used_as_sector_boundary_margin": False,
        "small_directional_margin_to_switch_probability": "NOT_IDENTIFIABLE_FROM_STORED EVENTS; non-event candidate rankings are not retained",
        "small_margin_to_AB_switching": cluster_bootstrap_mean_contrast(
            [row for row in switches if row.get("gat_top1_top2_probability_margin") is not None],
            "switchback_within_1_0s",
            "gat_top1_top2_probability_margin",
        ),
    }
    write_json(output / "02_existing_event_analysis/SECTOR_BOUNDARY_INSTABILITY.json", boundary_audit)
    write_json(output / "02_existing_event_analysis/EVENT_AUDIT_INTEGRITY.json", integrity)

    angular_bin_edges = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 180.0)
    angular_bin_rows: list[dict[str, Any]] = []
    for lower, upper in zip(angular_bin_edges[:-1], angular_bin_edges[1:]):
        members = [
            row for row in switches
            if row.get("sector_center_angular_jump_deg") is not None
            and float(row["sector_center_angular_jump_deg"]) > lower - (1.0e-9 if lower == 0.0 else 0.0)
            and float(row["sector_center_angular_jump_deg"]) <= upper + 1.0e-9
        ]
        angular_bin_rows.append(
            {
                "angular_jump_bin_deg": f"({lower:g},{upper:g}]",
                "lower_exclusive_deg": lower,
                "upper_inclusive_deg": upper,
                "event_count": len(members),
                "event_fraction": len(members) / len(switches) if switches else None,
                "high_jerk_event_count": sum(bool(row["high_jerk_switch"]) for row in members),
                "high_jerk_rate": float(np.mean([bool(row["high_jerk_switch"]) for row in members])) if members else None,
                "mean_post_switch_jerk_peak_mps3": describe(row["post_switch_jerk_peak"] for row in members)["mean"],
                "median_post_switch_jerk_peak_mps3": describe(row["post_switch_jerk_peak"] for row in members)["median"],
                "adjacent_switch_fraction": float(np.mean([bool(row["sectors_adjacent"]) for row in members])) if members else None,
            }
        )
    write_csv(output / "02_existing_event_analysis/ANGULAR_JUMP_JERK_BINS.csv", angular_bin_rows)

    compatibility_conditions = [
        {"condition": 1, "requirement": "Proposal sector resolution is independently configurable", "pass": False, "evidence": "Proposal loops over sensor.ray_directions and requires current_scan.shape == sensor.scan_shape; there is no separate Proposal azimuth/elevation configuration."},
        {"condition": 2, "requirement": "GAT-R input dimensionality remains identical", "pass": True, "evidence": "GAT safety field is explicitly projected to fixed 8x7=56 and Top-K remains 10."},
        {"condition": 3, "requirement": "GAT-R node feature semantics remain identical", "pass": True, "evidence": "Candidate nodes use continuous direction xyz and scalar proposal features; no Proposal sector ID is embedded."},
        {"condition": 4, "requirement": "56-direction canonical projection remains identical", "pass": True, "evidence": "GAT_CANONICAL_AZIMUTH_BINS=8 and GAT_CANONICAL_ELEVATION_BINS=7 are fixed independently of the proposal list."},
        {"condition": 5, "requirement": "Top-K interface remains valid", "pass": True, "evidence": "adapt_candidate_proposals preserves ordering and truncates to K=10 regardless of raw list length."},
        {"condition": 6, "requirement": "No learned checkpoint must be retrained", "pass": False, "evidence": "Changing the shared sensor grid changes the frozen SAC observation dimension 522; an independent denser Proposal grid would require a new, source-unsupported scan interpolation/feasibility contract."},
        {"condition": 7, "requirement": "Reference meaning remains unchanged", "pass": True, "evidence": "The distance rule and world-frame temporary reference meaning could remain unchanged."},
        {"condition": 8, "requirement": "Candidate geometry remains inside existing physical SAC-DMP reference support", "pass": True, "evidence": "Keeping the distance rule would retain the existing 0.05-1.05 m generated-reference support."},
    ]
    compatibility = {
        "SECTOR_DENSIFICATION_WITHOUT_RETRAINING": "NO",
        "SECTOR_DENSIFICATION_ABLATION_AUTHORIZED": "NO",
        "all_eight_conditions_pass": all(bool(row["pass"]) for row in compatibility_conditions),
        "conditions": compatibility_conditions,
        "mandatory_stop_reason": "Conditions 1 and 6 fail. Direct grid densification changes the actor observation size; decoupled densification would invent new interpolation semantics absent from the frozen implementation.",
        "top_k_frozen": 10,
        "gat_canonical_direction_count_frozen": 56,
        "retraining_performed": False,
    }
    write_json(output / "03_root_cause/SECTOR_DENSIFICATION_COMPATIBILITY.json", compatibility)
    write_json(
        output / "04_resolution_variants/SECTOR_VARIANT_CONTRACTS.json",
        {
            "status": "NOT_RUN_DENSIFICATION_HARD_GATE_FAILED",
            "S0": {"sector_geometry": "16x16 shared sensor grid", "sector_count": 256, "top_k": 10, "status": "frozen current method"},
            "S1": {"status": "NOT_AUTHORIZED"},
            "S2": {"status": "NOT_AUTHORIZED"},
            "hysteresis_factor": {"status": "NOT_RUN_AFTER_MANDATORY_STOP", "reason": "The requested primary 2x2 design requires an authorized denser-sector arm."},
        },
    )
    not_run_metric_fields = {
        "team_success_rate": None,
        "collision_rate": None,
        "static_collision_rate": None,
        "dynamic_collision_rate": None,
        "peer_collision_rate": None,
        "timeout_rate": None,
        "agent_completion_rate": None,
        "trajectory_smoothness_m2_s6": None,
        "vertical_jerk_metric": None,
        "lateral_jerk_metric": None,
        "completion_time_s": None,
        "team_path_length_m": None,
        "path_efficiency": None,
        "minimum_obstacle_clearance_m": None,
        "minimum_peer_distance_m": None,
        "reproposals_per_episode": None,
        "active_reference_changes_per_episode": None,
        "reference_switches_per_second": None,
        "mean_angular_reference_jump_deg": None,
        "median_angular_reference_jump_deg": None,
        "mean_absolute_elevation_jump_deg": None,
        "switchback_rate_1s": None,
        "median_dwell_time_s": None,
        "fp_shep_compute_ms": None,
        "gat_compute_ms": None,
        "total_upper_compute_ms": None,
        "total_online_compute_ms": None,
    }
    not_run_rows = [
        {"variant": label, "sector_resolution": sector, "hysteresis": hysteresis, "status": "NOT_RUN_DENSIFICATION_HARD_GATE_FAILED", **not_run_metric_fields}
        for label, sector, hysteresis in (
            ("A", "S0_current", "H0_current"),
            ("B", "S1_denser", "H0_current"),
            ("C", "S0_current", "H1_minimal"),
            ("D", "S1_denser", "H1_minimal"),
        )
    ]
    write_csv(output / "06_development/SECTOR_DEVELOPMENT_RESULTS.csv", not_run_rows)
    write_csv(output / "06_development/SECTOR_RELIABILITY_SMOOTHNESS_PARETO.csv", not_run_rows)
    write_csv(output / "07_holdout/SECTOR_HOLDOUT_RESULTS.csv", [{"status": "NOT_RUN_NO_DEVELOPMENT_WINNER"}])
    formal_go = {
        "status": "NO_GO",
        "dev_success_delta": "NOT_RUN",
        "holdout_success_delta": "NOT_RUN",
        "dev_jerk_reduction": "NOT_RUN",
        "holdout_jerk_reduction": "NOT_RUN",
        "vertical_jerk_reduction": "NOT_RUN",
        "reference_switch_reduction": "NOT_RUN",
        "compute_increase": "NOT_RUN",
        "paired_failure_inspection": "NOT_RUN",
        "recommendation": "Do not create a Formal sector variant under the current frozen shared sensor/Proposal contract.",
        "original_formal_v2_success_preserved": 0.9525,
    }
    write_json(output / "10_freeze/SECTOR_VARIANT_FORMAL_GO_NO_GO.json", formal_go)
    write_json(
        output / "10_freeze/SECTOR_SMOOTHNESS_VARIANT_FREEZE.json",
        {"status": "NOT_CREATED_NO_WINNER", "winner": None, "checkpoint_changed": False, "formal_variant_authorized": False},
    )

    source_files = [
        METHOD_CONFIG,
        REPO_ROOT / "Guidance/reference_point_proposal_demo.py",
        REPO_ROOT / "Entity/sensors.py",
        REPO_ROOT / "planning/policy_preview.py",
        REPO_ROOT / "planning/heterogeneous_candidate_graph.py",
        REPO_ROOT / "planning/pre_gat_closed_loop.py",
        FORMAL_MANIFEST,
        REPO_ROOT / config["gat_checkpoint"],
        REPO_ROOT / config["sac_checkpoint"],
    ]
    write_json(
        output / "00_context/SOURCE_AND_CHECKPOINT_FREEZE.json",
        {
            "method_config_sha256": sha256(METHOD_CONFIG),
            "formal_manifest_sha256": sha256(FORMAL_MANIFEST),
            "files": [
                {"path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha256(path), "size_bytes": path.stat().st_size}
                for path in source_files
            ],
            "formal_results_modified": False,
            "policy_execution_performed": False,
        },
    )
    write_json(
        output / "00_context/ANALYSIS_CONTRACT.json",
        {
            "scope": "read-only successful Formal V2 M9 event/trajectory alignment",
            "high_jerk_threshold": "global P90 post-switch 0.5 s jerk-norm peak, determined descriptively after data extraction",
            "headline_switchback_window_s": 1.0,
            "all_switchback_windows_s": [0.2, 0.5, 1.0, 2.0],
            "jerk_definition": "delta applied acceleration / 0.1 s; established smoothness is mean(sum(jerk_xyz^2))",
            "association": "event-level Spearman with episode-cluster percentile bootstrap",
            "causal_claim_authorized": False,
        },
    )
    print(json.dumps({"status": "PASS", "output": str(output), "switch_rows": len(switches), "event_rows": len(event_rows), "compatibility": "NO"}), flush=True)


if __name__ == "__main__":
    main()
