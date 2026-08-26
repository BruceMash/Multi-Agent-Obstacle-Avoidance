#!/usr/bin/env python3
"""Read-only analysis and raw visualization for the final zigzag experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

ARTIFACT_ROOT = REPO_ROOT / "artifacts/final_residual_zigzag_resolution/20260825_183146"
RUNTIME_CONTRACT = ARTIFACT_ROOT / "FINAL_ZIGZAG_RUNTIME_CONTRACT.json"
LAG_CSV = ARTIFACT_ROOT / "LAGGED_REFERENCE_EXECUTION_COUPLING.csv"
MANIFEST_PATHS = {
    "development": ARTIFACT_ROOT / "FINAL_ZIGZAG_DEV100_MANIFEST.json",
    "holdout": ARTIFACT_ROOT / "FINAL_ZIGZAG_HOLDOUT100_MANIFEST.json",
}
RECORD_ROOTS = {
    block: {
        arm: ARTIFACT_ROOT / f"{block}/{arm}/episode_records"
        for arm in ("strong", "repaired")
    }
    for block in ("development", "holdout")
}
DISPLAY = {"strong": "Frozen Strong", "repaired": "Early Bypass + DCTB"}
EPS_SPEED = 1.0e-9
EPS_ANGLE = 1.0e-9


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_ready(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _json_ready(row.get(name)) for name in names})
    temporary.replace(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def sign_runs(values: Iterable[float]) -> list[int]:
    result: list[int] = []
    previous = 0
    length = 0
    for value in values:
        sign = 1 if value > EPS_ANGLE else -1 if value < -EPS_ANGLE else 0
        if sign and sign == previous:
            length += 1
        else:
            if length:
                result.append(length)
            previous = sign
            length = 1 if sign else 0
    if length:
        result.append(length)
    return result


def direction_signals(velocity: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    speed = np.linalg.norm(velocity, axis=1)
    valid = speed > EPS_SPEED
    yaw = np.full(len(speed), np.nan)
    pitch = np.full(len(speed), np.nan)
    yaw[valid] = np.arctan2(velocity[valid, 1], velocity[valid, 0])
    horizontal = np.linalg.norm(velocity[:, :2], axis=1)
    pitch[valid] = np.arctan2(velocity[valid, 2], horizontal[valid])
    omega_yaw = np.full(len(speed), np.nan)
    omega_pitch = np.full(len(speed), np.nan)
    consecutive = valid[1:] & valid[:-1]
    index = np.flatnonzero(consecutive) + 1
    omega_yaw[index] = wrap_angle(yaw[index] - yaw[index - 1]) / dt
    omega_pitch[index] = (pitch[index] - pitch[index - 1]) / dt
    output: dict[str, np.ndarray] = {
        "yaw_rate": omega_yaw,
        "pitch_rate": omega_pitch,
    }
    for name, omega in (("yaw", omega_yaw), ("pitch", omega_pitch)):
        eligible = np.zeros(len(speed), dtype=bool)
        reversal = np.zeros(len(speed), dtype=bool)
        pair = (
            np.isfinite(omega[1:])
            & np.isfinite(omega[:-1])
            & (np.abs(omega[1:]) > EPS_ANGLE)
            & (np.abs(omega[:-1]) > EPS_ANGLE)
        )
        indexes = np.flatnonzero(pair) + 1
        eligible[indexes] = True
        reversal[indexes] = omega[indexes] * omega[indexes - 1] < 0.0
        output[f"{name}_eligible"] = eligible
        output[f"{name}_reversal"] = reversal
    return output


def _active_end(record: Mapping[str, Any], agent_id: int, length: int) -> int:
    row = next(item for item in record["agents"] if int(item["agent_id"]) == agent_id)
    stop = row.get("terminal_completion_step")
    return length if stop is None else min(length, int(stop) + 1)


def trajectory_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    arrays = np.load(record["npz_path"])
    velocity = np.asarray(arrays["velocities"], dtype=float)
    applied = np.asarray(arrays["applied_accelerations_full"], dtype=float).copy()
    dt = float(arrays["dt"])
    if not np.all(np.isfinite(applied[0])):
        applied[0] = 0.0
    metrics: dict[str, Any] = {
        "yaw_reversal_count": 0,
        "yaw_eligible_count": 0,
        "pitch_reversal_count": 0,
        "pitch_eligible_count": 0,
        "yaw_directional_tv": 0.0,
        "pitch_directional_tv": 0.0,
        "yaw_run_lengths": [],
        "pitch_run_lengths": [],
    }
    jerk_norms: list[np.ndarray] = []
    vertical_jerk: list[np.ndarray] = []
    lateral_jerk: list[np.ndarray] = []
    for agent_id in range(velocity.shape[1]):
        end = _active_end(record, agent_id, len(velocity))
        signal = direction_signals(velocity[:end, agent_id], dt)
        for axis in ("yaw", "pitch"):
            rate = signal[f"{axis}_rate"]
            finite = rate[np.isfinite(rate)]
            metrics[f"{axis}_reversal_count"] += int(np.sum(signal[f"{axis}_reversal"]))
            metrics[f"{axis}_eligible_count"] += int(np.sum(signal[f"{axis}_eligible"]))
            metrics[f"{axis}_directional_tv"] += float(np.sum(np.abs(np.diff(finite)))) if len(finite) > 1 else 0.0
            metrics[f"{axis}_run_lengths"].extend(sign_runs(finite))
        jerk = np.diff(applied[:end, agent_id], axis=0) / dt
        finite = np.all(np.isfinite(jerk), axis=1)
        jerk = jerk[finite]
        if len(jerk):
            jerk_norms.append(np.linalg.norm(jerk, axis=1))
            vertical_jerk.append(jerk[:, 2])
            lateral_jerk.append(np.linalg.norm(jerk[:, :2], axis=1))
    norm = np.concatenate(jerk_norms) if jerk_norms else np.asarray([])
    vertical = np.concatenate(vertical_jerk) if vertical_jerk else np.asarray([])
    lateral = np.concatenate(lateral_jerk) if lateral_jerk else np.asarray([])
    for axis in ("yaw", "pitch"):
        eligible = metrics[f"{axis}_eligible_count"]
        metrics[f"{axis}_reversal_rate"] = (
            metrics[f"{axis}_reversal_count"] / eligible if eligible else float("nan")
        )
        runs = metrics[f"{axis}_run_lengths"]
        metrics[f"{axis}_median_run_s"] = float(np.median(runs) * dt) if runs else float("nan")
        metrics[f"{axis}_mean_run_s"] = float(np.mean(runs) * dt) if runs else float("nan")
        metrics[f"{axis}_p90_run_s"] = float(np.percentile(runs, 90) * dt) if runs else float("nan")
    metrics.update(
        jerk_mean_squared=float(np.mean(norm**2)) if len(norm) else float("nan"),
        vertical_jerk_mean_squared=float(np.mean(vertical**2)) if len(vertical) else float("nan"),
        lateral_jerk_mean_squared=float(np.mean(lateral**2)) if len(lateral) else float("nan"),
        jerk_p90=float(np.percentile(norm, 90)) if len(norm) else float("nan"),
        jerk_p95=float(np.percentile(norm, 95)) if len(norm) else float("nan"),
    )
    return metrics


def load_records(block: str, arm: str, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    root = RECORD_ROOTS[block][arm]
    errors = sorted(root.glob("*_SOFTWARE_ERROR.json"))
    if errors:
        raise RuntimeError(f"software errors in {block}/{arm}: {errors}")
    records: dict[str, dict[str, Any]] = {}
    for sid in ids:
        path = root / f"{sid}.json"
        if not path.exists():
            raise RuntimeError(f"missing record: {path}")
        payload = load_json(path)
        payload["json_path"] = path
        payload["npz_path"] = path.with_name(str(payload["trajectory_file"]))
        records[sid] = payload
    return records


def pct_reduction(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference if reference else float("nan")


def bootstrap(values: np.ndarray, statistic: Any, seed: int, repeats: int = 3000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for index in range(repeats):
        take = rng.integers(0, len(values), size=len(values))
        samples[index] = statistic(values[take])
    return tuple(map(float, np.percentile(samples, (2.5, 97.5))))


def _pooled_direction(members: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for axis in ("yaw", "pitch"):
        reversal = sum(int(row[f"{axis}_reversal_count"]) for row in members)
        eligible = sum(int(row[f"{axis}_eligible_count"]) for row in members)
        runs = [value for row in members for value in row[f"{axis}_run_lengths"]]
        output.update(
            {
                f"{axis}_reversal_count": reversal,
                f"{axis}_eligible_count": eligible,
                f"{axis}_reversal_rate": reversal / eligible if eligible else float("nan"),
                f"{axis}_directional_tv": sum(float(row[f"{axis}_directional_tv"]) for row in members),
                f"{axis}_median_run_s": float(np.median(runs) * 0.1) if runs else float("nan"),
                f"{axis}_mean_run_s": float(np.mean(runs) * 0.1) if runs else float("nan"),
                f"{axis}_p90_run_s": float(np.percentile(runs, 90) * 0.1) if runs else float("nan"),
            }
        )
    return output


def analyze(block: str) -> None:
    contract = load_json(RUNTIME_CONTRACT)
    manifest = load_json(MANIFEST_PATHS[block])
    ids = [str(row["scenario_id"]) for row in manifest["entries"]]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise RuntimeError(f"{block} manifest is not unique 100")
    records = {arm: load_records(block, arm, ids) for arm in DISPLAY}
    metric = {
        arm: {sid: trajectory_metrics(record) for sid, record in records[arm].items()}
        for arm in DISPLAY
    }
    both = [
        sid for sid in ids
        if records["strong"][sid]["episode"]["team_success"]
        and records["repaired"][sid]["episode"]["team_success"]
    ]
    result_rows: list[dict[str, Any]] = []
    morphology_rows: list[dict[str, Any]] = []
    for scope in ["overall", *list(dict.fromkeys(row["stage"] for row in manifest["entries"]))]:
        scope_ids = ids if scope == "overall" else [sid for sid in ids if records["strong"][sid]["entry_identity"]["stage"] == scope]
        for arm in DISPLAY:
            episodes = [records[arm][sid]["episode"] for sid in scope_ids]
            result_rows.append(
                {
                    "scope": scope,
                    "arm": arm,
                    "display_name": DISPLAY[arm],
                    "n": len(episodes),
                    "team_success_count": sum(bool(row["team_success"]) for row in episodes),
                    "team_success_rate": float(np.mean([row["team_success"] for row in episodes])),
                    "collision_rate": float(np.mean([row["collision"] for row in episodes])),
                    "obstacle_collision_rate": float(np.mean([row["obstacle_collision"] for row in episodes])),
                    "peer_collision_rate": float(np.mean([row["inter_agent_collision"] for row in episodes])),
                    "timeout_rate": float(np.mean([row["timeout"] for row in episodes])),
                    "agent_completion_rate": float(np.mean([row["agent_completion_rate"] for row in episodes])),
                    "mean_total_online_compute_ms": float(np.mean([row["total_online_algorithm_compute_ms"] for row in episodes])),
                }
            )
            pooled = _pooled_direction([metric[arm][sid] for sid in scope_ids])
            morphology_rows.append({"scope": scope, "arm": arm, **pooled})
    overall = {(row["arm"]): row for row in result_rows if row["scope"] == "overall"}
    morph = {(row["arm"]): row for row in morphology_rows if row["scope"] == "overall"}
    smooth_strong = np.asarray([records["strong"][sid]["episode"]["trajectory_smoothness"] for sid in both], dtype=float)
    smooth_repaired = np.asarray([records["repaired"][sid]["episode"]["trajectory_smoothness"] for sid in both], dtype=float)
    path_strong = np.asarray([records["strong"][sid]["episode"]["team_path_length_m"] for sid in both], dtype=float)
    path_repaired = np.asarray([records["repaired"][sid]["episode"]["team_path_length_m"] for sid in both], dtype=float)
    completion_strong = np.asarray([records["strong"][sid]["episode"]["completion_time_s"] for sid in both], dtype=float)
    completion_repaired = np.asarray([records["repaired"][sid]["episode"]["completion_time_s"] for sid in both], dtype=float)
    paired_quality_sources = {
        "trajectory_smoothness_cost": {
            arm: np.asarray([records[arm][sid]["episode"]["trajectory_smoothness"] for sid in both], dtype=float)
            for arm in DISPLAY
        },
        "jerk_mean_squared": {
            arm: np.asarray([metric[arm][sid]["jerk_mean_squared"] for sid in both], dtype=float)
            for arm in DISPLAY
        },
        "vertical_jerk_mean_squared": {
            arm: np.asarray([metric[arm][sid]["vertical_jerk_mean_squared"] for sid in both], dtype=float)
            for arm in DISPLAY
        },
        "lateral_jerk_mean_squared": {
            arm: np.asarray([metric[arm][sid]["lateral_jerk_mean_squared"] for sid in both], dtype=float)
            for arm in DISPLAY
        },
        "episode_p95_jerk_mps3": {
            arm: np.asarray([metric[arm][sid]["jerk_p95"] for sid in both], dtype=float)
            for arm in DISPLAY
        },
        "team_path_length_m": {
            arm: np.asarray([records[arm][sid]["episode"]["team_path_length_m"] for sid in both], dtype=float)
            for arm in DISPLAY
        },
        "completion_time_s": {
            arm: np.asarray([records[arm][sid]["episode"]["completion_time_s"] for sid in both], dtype=float)
            for arm in DISPLAY
        },
        "reference_selection_count": {
            arm: np.asarray([records[arm][sid]["episode"]["reference_selection_count"] for sid in both], dtype=float)
            for arm in DISPLAY
        },
    }
    paired_quality_rows = []
    for name, values in paired_quality_sources.items():
        strong_mean = float(np.mean(values["strong"]))
        repaired_mean = float(np.mean(values["repaired"]))
        paired_quality_rows.append(
            {
                "metric": name,
                "both_success_n": len(both),
                "strong_mean": strong_mean,
                "repaired_mean": repaired_mean,
                "repaired_minus_strong": repaired_mean - strong_mean,
                "reduction_percent": pct_reduction(strong_mean, repaired_mean),
            }
        )
    smooth_pct = pct_reduction(float(np.mean(smooth_strong)), float(np.mean(smooth_repaired)))
    smooth_ci = bootstrap(
        np.column_stack([smooth_strong, smooth_repaired]),
        lambda sample: pct_reduction(float(np.mean(sample[:, 0])), float(np.mean(sample[:, 1]))),
        2026082501,
    )
    yaw_reduction = pct_reduction(morph["strong"]["yaw_reversal_rate"], morph["repaired"]["yaw_reversal_rate"])
    pitch_reduction = pct_reduction(morph["strong"]["pitch_reversal_rate"], morph["repaired"]["pitch_reversal_rate"])
    yaw_tv_reduction = pct_reduction(morph["strong"]["yaw_directional_tv"], morph["repaired"]["yaw_directional_tv"])
    pitch_tv_reduction = pct_reduction(morph["strong"]["pitch_directional_tv"], morph["repaired"]["pitch_directional_tv"])
    fixed_ids = (
        contract["development"]["fixed_visual_scene_ids_before_outcomes"]
        if block == "development"
        else contract["holdout"]["fixed_visual_scene_ids_before_outcomes"]
    )
    fixed_rows = []
    fixed_positive = 0
    for sid in fixed_ids:
        strong_m = metric["strong"][sid]
        repaired_m = metric["repaired"][sid]
        strong_reversal = strong_m["yaw_reversal_rate"] + strong_m["pitch_reversal_rate"]
        repaired_reversal = repaired_m["yaw_reversal_rate"] + repaired_m["pitch_reversal_rate"]
        strong_tv = strong_m["yaw_directional_tv"] + strong_m["pitch_directional_tv"]
        repaired_tv = repaired_m["yaw_directional_tv"] + repaired_m["pitch_directional_tv"]
        strong_run = np.nanmean([strong_m["yaw_mean_run_s"], strong_m["pitch_mean_run_s"]])
        repaired_run = np.nanmean([repaired_m["yaw_mean_run_s"], repaired_m["pitch_mean_run_s"]])
        positive = bool(repaired_reversal < strong_reversal and repaired_tv < strong_tv and repaired_run > strong_run)
        fixed_positive += int(positive)
        fixed_rows.append(
            {
                "block": block,
                "scenario_id": sid,
                "stage": records["strong"][sid]["entry_identity"]["stage"],
                "both_success": bool(sid in both),
                "combined_reversal_reduction_percent": pct_reduction(strong_reversal, repaired_reversal),
                "combined_directional_tv_reduction_percent": pct_reduction(strong_tv, repaired_tv),
                "mean_same_direction_run_change_s": repaired_run - strong_run,
                "objective_longer_arc_proxy": positive,
            }
        )
    repaired_events = [event for sid in ids for event in records["repaired"][sid]["events"]]
    dctb_trace = [row for sid in ids for row in records["repaired"][sid].get("direction_continuity_trace", [])]
    dctb_reference_eligible = [
        row
        for row in dctb_trace
        if int(row.get("history_length_before", 0)) >= 2 and not row.get("history_reset", False)
    ]
    dctb_horizontal_trigger_count = sum(
        bool(row.get("horizontal_reversal_trigger")) for row in dctb_reference_eligible
    )
    dctb_vertical_trigger_count = sum(
        bool(row.get("vertical_reversal_trigger")) for row in dctb_reference_eligible
    )
    limiter_summaries = [records["repaired"][sid]["limiter_summary"] for sid in ids]
    dctb_summary = {
        "trace_count": len(dctb_trace),
        "reference_turn_history_eligible_count": len(dctb_reference_eligible),
        "gat_top1_horizontal_reference_reversal_trigger_count": dctb_horizontal_trigger_count,
        "gat_top1_horizontal_reference_reversal_trigger_rate": dctb_horizontal_trigger_count / max(1, len(dctb_reference_eligible)),
        "gat_top1_vertical_reference_reversal_trigger_count": dctb_vertical_trigger_count,
        "gat_top1_vertical_reference_reversal_trigger_rate": dctb_vertical_trigger_count / max(1, len(dctb_reference_eligible)),
        "activation_count": sum(bool(row.get("activated")) for row in dctb_trace),
        "replacement_count": sum(bool(row.get("replaced")) for row in dctb_trace),
        "rank2_replacement_count": sum(row.get("replacement_gat_rank") == 2 for row in dctb_trace),
        "rank3_replacement_count": sum(row.get("replacement_gat_rank") == 3 for row in dctb_trace),
        "gat_top1_retention_rate": 1.0 - sum(bool(row.get("replaced")) for row in dctb_trace) / max(1, sum(not row.get("history_reset", False) for row in dctb_trace)),
        "safety_critical_bypass_count": sum(row.get("reason") == "EXISTING_SAFETY_CRITICAL_BYPASS" for row in dctb_trace),
        "reason_counts": dict(Counter(str(row.get("reason")) for row in dctb_trace)),
    }
    early_summary = {
        "early_bypass_count": sum(int(row.get("early_bypass_count", 0)) for row in limiter_summaries),
        "hard_bypass_count": sum(int(row.get("hard_bypass_count", 0)) for row in limiter_summaries),
        "combined_bypass_count": sum(int(row.get("combined_bypass_count", 0)) for row in limiter_summaries),
    }
    success_loss_pp = 100.0 * (overall["strong"]["team_success_rate"] - overall["repaired"]["team_success_rate"])
    peer_delta_pp = 100.0 * (overall["repaired"]["peer_collision_rate"] - overall["strong"]["peer_collision_rate"])
    obstacle_delta_pp = 100.0 * (overall["repaired"]["obstacle_collision_rate"] - overall["strong"]["obstacle_collision_rate"])
    gate_config = contract["development_gates_frozen_before_results"]
    direction_axis = "yaw" if yaw_reduction >= pitch_reduction else "pitch"
    direction_reduction = max(yaw_reduction, pitch_reduction)
    same_axis_tv_reduction = yaw_tv_reduction if direction_axis == "yaw" else pitch_tv_reduction
    gates = {
        "reliability": success_loss_pp <= float(gate_config["maximum_team_success_loss_pp"]) + 1e-12,
        "peer_safety": peer_delta_pp <= float(gate_config["maximum_peer_collision_increase_pp"]) + 1e-12,
        "obstacle_safety": obstacle_delta_pp <= float(gate_config["maximum_obstacle_collision_increase_pp"]) + 1e-12,
        "executed_reversal": direction_reduction >= float(gate_config["minimum_executed_yaw_or_pitch_reversal_reduction_percent"]),
        "same_axis_directional_tv": same_axis_tv_reduction >= float(gate_config["minimum_same_axis_directional_total_variation_reduction_percent"]),
        "strong_smoothness_preserved": smooth_pct >= -float(gate_config["maximum_smoothness_cost_increase_percent_on_both_success"]),
        "fixed_raw_visual_objective_proxy": fixed_positive >= 3,
    }
    gate_pass = bool(all(gates.values()))
    paired_failure_rows = []
    for sid in ids:
        strong_success = bool(records["strong"][sid]["episode"]["team_success"])
        repaired_success = bool(records["repaired"][sid]["episode"]["team_success"])
        if strong_success and not repaired_success:
            repaired_record = records["repaired"][sid]
            limiter_trace = np.load(repaired_record["json_path"].with_name(str(repaired_record["limiter_trace_file"])))
            final_start_step = max(0, int(repaired_record["episode"]["steps"]) - 10)
            final_mask = np.asarray(limiter_trace["step"], dtype=int) >= final_start_step
            warning = np.asarray(limiter_trace["warning_low_margin"], dtype=bool)[final_mask]
            hard = np.asarray(limiter_trace["hard_bypass"], dtype=bool)[final_mask]
            early = np.asarray(limiter_trace["early_bypass"], dtype=bool)[final_mask]
            bypass = np.asarray(limiter_trace["bypass_active"], dtype=bool)[final_mask]
            limiter_active = np.asarray(limiter_trace["limiter_active"], dtype=bool)[final_mask]
            margin = np.asarray(limiter_trace["safety_margin_m"], dtype=float)[final_mask]
            delta = np.asarray(limiter_trace["safety_margin_delta_m"], dtype=float)[final_mask]
            modification = np.asarray(limiter_trace["limiter_modification_norm_mps2"], dtype=float)[final_mask]
            raw = np.asarray(limiter_trace["raw_acceleration"], dtype=float)[final_mask]
            executed = np.asarray(limiter_trace["executed_acceleration"], dtype=float)[final_mask]
            dctb_safety_bypass = sum(
                row.get("reason") == "EXISTING_SAFETY_CRITICAL_BYPASS"
                for row in repaired_record.get("direction_continuity_trace", [])
            )
            event_steps = [
                int(row["step"])
                for row in repaired_record.get("direction_continuity_trace", [])
                if row.get("replaced")
            ]
            paired_failure_rows.append(
                {
                    "scenario_id": sid,
                    "stage": records["strong"][sid]["entry_identity"]["stage"],
                    "strong_outcome": records["strong"][sid]["episode"]["termination_reason"],
                    "repaired_outcome": records["repaired"][sid]["episode"]["termination_reason"],
                    "repaired_obstacle_collision": records["repaired"][sid]["episode"]["obstacle_collision"],
                    "repaired_peer_collision": records["repaired"][sid]["episode"]["inter_agent_collision"],
                    "dctb_replacement_count": len(event_steps),
                    "last_dctb_replacement_step": max(event_steps) if event_steps else None,
                    "early_bypass_count": records["repaired"][sid]["limiter_summary"].get("early_bypass_count", 0),
                    "hard_bypass_count": records["repaired"][sid]["limiter_summary"].get("hard_bypass_count", 0),
                    "final_1s_trace_row_count": int(np.sum(final_mask)),
                    "final_1s_minimum_safety_margin_m": float(np.nanmin(margin)) if len(margin) else None,
                    "final_1s_deteriorating_margin_row_count": int(np.sum(np.isfinite(delta) & (delta < 0.0))),
                    "final_1s_warning_row_count": int(np.sum(warning)),
                    "final_1s_early_bypass_row_count": int(np.sum(early)),
                    "final_1s_hard_bypass_row_count": int(np.sum(hard)),
                    "final_1s_combined_bypass_row_count": int(np.sum(bypass)),
                    "final_1s_limiter_active_row_count": int(np.sum(limiter_active)),
                    "final_1s_nonzero_suppression_row_count": int(np.sum(modification > 1.0e-9)),
                    "final_1s_max_raw_to_executed_acceleration_delta_mps2": (
                        float(np.max(np.linalg.norm(raw - executed, axis=1))) if len(raw) else None
                    ),
                    "final_1s_max_recorded_limiter_modification_mps2": float(np.max(modification)) if len(modification) else None,
                    "warning_state_without_bypass_row_count": int(np.sum(warning & ~bypass)),
                    "hard_state_without_bypass_row_count": int(np.sum(hard & ~bypass)),
                    "dctb_safety_critical_bypass_count": int(dctb_safety_bypass),
                    "necessary_emergency_maneuver_suppressed_by_residual_repair": bool(
                        np.any(warning & ~bypass) or np.any(hard & ~bypass)
                    ),
                }
            )
    decision = {
        "schema_version": f"final_zigzag_{block}_decision_v1",
        "block": block,
        "scenario_count": len(ids),
        "both_success_count": len(both),
        "team_success": {arm: overall[arm]["team_success_rate"] for arm in DISPLAY},
        "success_loss_pp": success_loss_pp,
        "peer_collision_delta_pp": peer_delta_pp,
        "obstacle_collision_delta_pp": obstacle_delta_pp,
        "smoothness_reduction_percent_both_success": smooth_pct,
        "smoothness_reduction_bootstrap_95ci": smooth_ci,
        "path_length_reduction_percent_both_success": pct_reduction(float(np.mean(path_strong)), float(np.mean(path_repaired))),
        "completion_time_delta_s_both_success": float(np.mean(completion_repaired - completion_strong)),
        "yaw_reversal_reduction_percent": yaw_reduction,
        "pitch_reversal_reduction_percent": pitch_reduction,
        "yaw_directional_tv_reduction_percent": yaw_tv_reduction,
        "pitch_directional_tv_reduction_percent": pitch_tv_reduction,
        "selected_direction_gate_axis": direction_axis,
        "selected_direction_gate_reduction_percent": direction_reduction,
        "selected_axis_tv_reduction_percent": same_axis_tv_reduction,
        "fixed_visual_objective_positive_count": fixed_positive,
        "dctb": dctb_summary,
        "early_bypass": early_summary,
        "gates": gates,
        ("DEV_GATE" if block == "development" else "HOLDOUT_GATE"): "PASS" if gate_pass else "FAIL",
        "HOLDOUT_AUTHORIZED": bool(gate_pass) if block == "development" else None,
        "FORMAL_V2_EXECUTED": False,
        "manifest_semantic_sha256": manifest["manifest_semantic_sha256"],
        "record_set_semantic_sha256": stable_hash(
            {
                arm: {
                    sid: {
                        "success": records[arm][sid]["episode"]["team_success"],
                        "reason": records[arm][sid]["episode"]["termination_reason"],
                        "trajectory_sha256": records[arm][sid]["trajectory_sha256"],
                    }
                    for sid in ids
                }
                for arm in DISPLAY
            }
        ),
    }
    prefix = "DEV" if block == "development" else "HOLDOUT"
    write_csv(ARTIFACT_ROOT / f"FINAL_ZIGZAG_{prefix}_RESULTS.csv", result_rows)
    write_csv(ARTIFACT_ROOT / f"FINAL_ZIGZAG_{prefix}_PAIRED_MORPHOLOGY.csv", morphology_rows)
    write_csv(ARTIFACT_ROOT / f"FINAL_ZIGZAG_{prefix}_PAIRED_QUALITY.csv", paired_quality_rows)
    write_csv(ARTIFACT_ROOT / f"FINAL_ZIGZAG_{prefix}_FIXED_VISUAL_SCENE_METRICS.csv", fixed_rows)
    write_csv(ARTIFACT_ROOT / f"FINAL_ZIGZAG_{prefix}_PAIRED_FAILURE_AUDIT.csv", paired_failure_rows)
    if block == "development":
        write_csv(ARTIFACT_ROOT / "FINAL_ZIGZAG_FAILURE_AUDIT.csv", paired_failure_rows)
    write_csv(
        ARTIFACT_ROOT / f"FINAL_ZIGZAG_{prefix}_DCTB_INTERVENTIONS.csv",
        [
            {
                "scenario_id": sid,
                **row,
            }
            for sid in ids
            for row in records["repaired"][sid].get("direction_continuity_trace", [])
            if row.get("replaced")
        ],
    )
    atomic_json(
        ARTIFACT_ROOT / ("FINAL_ZIGZAG_DEV_GO_NO_GO.json" if block == "development" else "FINAL_ZIGZAG_HOLDOUT_DECISION.json"),
        decision,
    )
    print(json.dumps(_json_ready(decision), indent=2))


def plot_lag() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with LAG_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharex=True, constrained_layout=True)
    styles = {"Original": ("#4C78A8", "--"), "Strong": ("#E45756", "-")}
    for axis_name, axis in zip(("yaw", "pitch"), axes):
        for method in ("Original", "Strong"):
            subset = sorted(
                [row for row in rows if row["method"] == method and row["axis"] == axis_name],
                key=lambda row: float(row["lag_s"]),
            )
            x = np.asarray([float(row["lag_s"]) for row in subset])
            y = np.asarray([float(row["rho"]) for row in subset])
            lo = np.asarray([float(row["rho_ci95_low"]) for row in subset])
            hi = np.asarray([float(row["rho_ci95_high"]) for row in subset])
            color, linestyle = styles[method]
            axis.plot(x, y, color=color, linestyle=linestyle, marker="o", markersize=3.0, linewidth=1.4, label=method)
            axis.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
        axis.axhline(0.0, color="0.45", linewidth=0.65)
        axis.set_title(axis_name.capitalize())
        axis.set_xlabel("Lag (s)")
        axis.grid(True, color="0.88", linewidth=0.5)
    axes[0].set_ylabel("Spearman correlation")
    axes[1].legend(frameon=False, loc="upper right")
    output = ARTIFACT_ROOT / "paper_ready/pdf/figure_L_lagged_reference_to_turn_response.pdf"
    png = ARTIFACT_ROOT / "paper_ready/png/figure_L_lagged_reference_to_turn_response.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_raw(block: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from planning.plot_continuous_reference_transition import draw_obstacles_3d, draw_obstacles_xy

    contract = load_json(RUNTIME_CONTRACT)
    manifest = load_json(MANIFEST_PATHS[block])
    index = {str(row["scenario_id"]): row for row in manifest["entries"]}
    fixed = (
        contract["development"]["fixed_visual_scene_ids_before_outcomes"]
        if block == "development" else contract["holdout"]["fixed_visual_scene_ids_before_outcomes"]
    )
    output = ARTIFACT_ROOT / (
        "FINAL_ZIGZAG_DEV_RAW_TRAJECTORIES.pdf"
        if block == "development"
        else "FINAL_ZIGZAG_HOLDOUT_RAW_TRAJECTORIES.pdf"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    styles = {"strong": "--", "repaired": "-"}
    with PdfPages(output) as pdf:
        for sid in fixed:
            scene = load_json(REPO_ROOT / index[sid]["scenario_file"])
            payloads = {arm: load_json(RECORD_ROOTS[block][arm] / f"{sid}.json") for arm in DISPLAY}
            arrays = {arm: np.load(RECORD_ROOTS[block][arm] / payloads[arm]["trajectory_file"]) for arm in DISPLAY}
            figure = plt.figure(figsize=(14.5, 7.7))
            # Reserve a fixed header band for the page title and the two-row
            # method/UAV legend.  ``constrained_layout`` does not consistently
            # account for figure-level legends when saving a multi-page PDF.
            grid = figure.add_gridspec(
                2,
                3,
                left=0.055,
                right=0.985,
                bottom=0.075,
                top=0.805,
                wspace=0.18,
                hspace=0.32,
            )
            ax3d = figure.add_subplot(grid[0, 0], projection="3d")
            axxy = figure.add_subplot(grid[0, 1])
            axz = figure.add_subplot(grid[0, 2])
            axyaw = figure.add_subplot(grid[1, 0])
            axpitch = figure.add_subplot(grid[1, 1])
            axref = figure.add_subplot(grid[1, 2])
            maximum_steps = max(len(arrays[arm]["positions"]) for arm in DISPLAY)
            draw_obstacles_3d(ax3d, scene, maximum_steps)
            draw_obstacles_xy(axxy, scene, maximum_steps)
            for arm in DISPLAY:
                position = np.asarray(arrays[arm]["positions"], dtype=float)
                velocity = np.asarray(arrays[arm]["velocities"], dtype=float)
                active_goal = np.asarray(arrays[arm]["active_goals"], dtype=float)
                t = np.arange(len(position)) * 0.1
                for agent_id in range(3):
                    label = f"UAV {agent_id + 1} - {DISPLAY[arm]}"
                    ax3d.plot(*position[:, agent_id].T, color=colors[agent_id], linestyle=styles[arm], linewidth=1.2 if arm == "repaired" else 0.9, label=label)
                    axxy.plot(position[:, agent_id, 0], position[:, agent_id, 1], color=colors[agent_id], linestyle=styles[arm], linewidth=1.2 if arm == "repaired" else 0.9)
                    axz.plot(t, position[:, agent_id, 2], color=colors[agent_id], linestyle=styles[arm], linewidth=1.0)
                    signal = direction_signals(velocity[:, agent_id], 0.1)
                    axyaw.plot(t, signal["yaw_rate"], color=colors[agent_id], linestyle=styles[arm], linewidth=0.8)
                    axpitch.plot(t, signal["pitch_rate"], color=colors[agent_id], linestyle=styles[arm], linewidth=0.8)
                    for axis, name in ((axyaw, "yaw"), (axpitch, "pitch")):
                        indexes = np.flatnonzero(signal[f"{name}_reversal"])
                        axis.scatter(t[indexes], signal[f"{name}_rate"][indexes], s=6, facecolors=(colors[agent_id] if arm == "repaired" else "none"), edgecolors=colors[agent_id], linewidths=0.45, alpha=0.7)
                    changes = np.r_[True, np.linalg.norm(np.diff(active_goal[:, agent_id], axis=0), axis=1) > 1e-9]
                    change_steps = np.flatnonzero(changes)
                    axref.scatter(t[change_steps], np.full(len(change_steps), agent_id + (0.18 if arm == "repaired" else -0.18)), color=colors[agent_id], marker=("o" if arm == "repaired" else "x"), s=8, linewidths=0.5)
                ax3d.scatter(position[0, :, 0], position[0, :, 1], position[0, :, 2], marker="o", s=15, color=colors)
                ax3d.scatter(position[-1, :, 0], position[-1, :, 1], position[-1, :, 2], marker="^", s=18, color=colors)
                if arm == "repaired":
                    for row in payloads[arm].get("direction_continuity_trace", []):
                        if row.get("replaced"):
                            step, agent_id = int(row["step"]), int(row["agent_id"])
                            if step < len(position):
                                axxy.scatter(position[step, agent_id, 0], position[step, agent_id, 1], marker="D", s=11, color=colors[agent_id], edgecolor="k", linewidth=0.25)
                                axref.scatter(step * 0.1, agent_id + 0.38, marker="D", s=10, color=colors[agent_id], edgecolor="k", linewidth=0.25)
                    trace_path = RECORD_ROOTS[block][arm] / payloads[arm]["limiter_trace_file"]
                    trace = np.load(trace_path)
                    bypass = np.asarray(trace["bypass_active"], dtype=bool)
                    for step, agent_id in zip(np.asarray(trace["step"])[bypass], np.asarray(trace["agent_id"])[bypass]):
                        if int(step) < len(position):
                            axxy.scatter(position[int(step), int(agent_id), 0], position[int(step), int(agent_id), 1], marker="*", s=24, color=colors[int(agent_id)], edgecolor="k", linewidth=0.3)
                            axref.scatter(float(step) * 0.1, int(agent_id) + 0.62, marker="*", s=18, color=colors[int(agent_id)], edgecolor="k", linewidth=0.3)
            starts = np.asarray(scene["starts"], dtype=float)
            goals = np.asarray(scene["goals"], dtype=float)
            axxy.scatter(starts[:, 0], starts[:, 1], marker="o", s=18, color=colors, edgecolor="k", linewidth=0.3)
            axxy.scatter(goals[:, 0], goals[:, 1], marker="^", s=22, color=colors, edgecolor="k", linewidth=0.3)
            ax3d.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="Raw 3-D trajectories")
            ax3d.view_init(elev=24, azim=-58)
            axxy.set(xlabel="x (m)", ylabel="y (m)", title="XY paths and interventions", aspect="equal")
            axz.set(xlabel="time (s)", ylabel="z (m)", title="Raw altitude")
            axyaw.set(xlabel="time (s)", ylabel="yaw rate (rad/s)", title="Raw yaw rate; dots = reversals")
            axpitch.set(xlabel="time (s)", ylabel="pitch rate (rad/s)", title="Raw pitch rate; dots = reversals")
            axref.set(xlabel="time (s)", ylabel="UAV index / event band", title="Reference changes (o/x), DCTB (D), bypass (*)")
            axref.set_yticks([0, 1, 2], ["UAV 1", "UAV 2", "UAV 3"])
            for axis in (axxy, axz, axyaw, axpitch, axref):
                axis.grid(True, color="0.88", linewidth=0.45)
            handles, labels = ax3d.get_legend_handles_labels()
            figure.suptitle(
                f"{scene['stage'].replace('_', ' ').title()} - {sid} - raw 0.1 s data",
                y=0.982,
            )
            figure.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.945),
                ncol=3,
                frameon=False,
            )
            pdf.savefig(figure)
            plt.close(figure)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--block", choices=("development", "holdout"), required=True)
    plot_parser = sub.add_parser("plot-raw")
    plot_parser.add_argument("--block", choices=("development", "holdout"), required=True)
    sub.add_parser("plot-lag")
    args = parser.parse_args()
    if args.command == "analyze":
        analyze(args.block)
    elif args.command == "plot-raw":
        plot_raw(args.block)
    else:
        plot_lag()


if __name__ == "__main__":
    main()
