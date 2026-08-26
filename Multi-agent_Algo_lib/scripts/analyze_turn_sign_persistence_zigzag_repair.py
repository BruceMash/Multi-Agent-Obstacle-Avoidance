#!/usr/bin/env python3
"""Analyze the frozen turn-sign persistence Dev/Holdout blocks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts import analyze_final_residual_zigzag_resolution as legacy  # noqa: E402


ARTIFACT_ROOT = REPO_ROOT / "artifacts/turn_sign_persistence_zigzag_repair/20260825_222130"
RUNTIME_CONTRACT = ARTIFACT_ROOT / "TURN_PERSISTENCE_RUNTIME_FREEZE.json"
MANIFEST_PATHS = {
    "development": ARTIFACT_ROOT / "TURN_PERSISTENCE_DEV100_MANIFEST.json",
    "holdout": ARTIFACT_ROOT / "TURN_PERSISTENCE_HOLDOUT100_MANIFEST.json",
}
RECORD_ROOTS = {
    block: {
        arm: ARTIFACT_ROOT / block / arm / "episode_records"
        for arm in ("strong", "repaired")
    }
    for block in ("development", "holdout")
}
DISPLAY = {
    "strong": "Frozen Strong",
    "repaired": "Strong + Early Bypass + Turn-Sign Persistence",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
    else:
        names = list(fieldnames)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([json_ready(dict(row)) for row in rows])


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def reduction(reference: float, repaired: float) -> float:
    return 100.0 * (reference - repaired) / reference if reference else float("nan")


def bootstrap_smoothness(strong: np.ndarray, repaired: np.ndarray, seed: int) -> tuple[float, float]:
    pairs = np.column_stack([strong, repaired])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(3000):
        take = rng.integers(0, len(pairs), len(pairs))
        selected = pairs[take]
        samples.append(reduction(float(np.mean(selected[:, 0])), float(np.mean(selected[:, 1]))))
    return tuple(map(float, np.percentile(samples, (2.5, 97.5))))


def trace_summary(records: Mapping[str, Mapping[str, Any]], ids: Sequence[str]) -> dict[str, Any]:
    summaries = [records[sid]["limiter_summary"] for sid in ids]
    totals = {
        key: int(sum(int(row.get(key, 0)) for row in summaries))
        for key in (
            "trace_count", "activation_count", "early_bypass_count", "hard_bypass_count",
            "combined_bypass_count", "persistence_activation_count",
            "horizontal_persistence_activation_count", "vertical_persistence_activation_count",
            "horizontal_confirmed_reversal_count", "vertical_confirmed_reversal_count",
            "horizontal_magnitude_override_count", "vertical_magnitude_override_count",
            "persistence_safety_bypass_count", "persistence_modified_count",
        )
    }
    totals["persistence_unchanged_fraction"] = (
        1.0 - totals["persistence_modified_count"] / totals["trace_count"]
        if totals["trace_count"] else 1.0
    )
    totals["horizontal_persistence_activation_rate"] = (
        totals["horizontal_persistence_activation_count"] / totals["trace_count"]
        if totals["trace_count"] else 0.0
    )
    totals["vertical_persistence_activation_rate"] = (
        totals["vertical_persistence_activation_count"] / totals["trace_count"]
        if totals["trace_count"] else 0.0
    )
    totals["magnitude_override_count"] = (
        totals["horizontal_magnitude_override_count"]
        + totals["vertical_magnitude_override_count"]
    )
    totals["mean_filter_runtime_ms_per_episode"] = float(np.mean([row.get("runtime_ms", 0.0) for row in summaries]))
    return totals


def analyze(block: str) -> dict[str, Any]:
    contract = load_json(RUNTIME_CONTRACT)
    manifest = load_json(MANIFEST_PATHS[block])
    ids = [str(row["scenario_id"]) for row in manifest["entries"]]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise RuntimeError("manifest must contain exactly 100 unique scenarios")
    records = {arm: load_records(block, arm, ids) for arm in DISPLAY}
    metrics = {
        arm: {sid: legacy.trajectory_metrics(record) for sid, record in records[arm].items()}
        for arm in DISPLAY
    }
    both = [
        sid for sid in ids
        if records["strong"][sid]["episode"]["team_success"]
        and records["repaired"][sid]["episode"]["team_success"]
    ]
    result_rows: list[dict[str, Any]] = []
    morphology_rows: list[dict[str, Any]] = []
    stages = list(dict.fromkeys(str(row["stage"]) for row in manifest["entries"]))
    for scope in ["overall", *stages]:
        scope_ids = ids if scope == "overall" else [sid for sid in ids if records["strong"][sid]["entry_identity"]["stage"] == scope]
        for arm in DISPLAY:
            episodes = [records[arm][sid]["episode"] for sid in scope_ids]
            result_rows.append(
                {
                    "block": block,
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
            morphology_rows.append(
                {"block": block, "scope": scope, "arm": arm, **legacy._pooled_direction([metrics[arm][sid] for sid in scope_ids])}
            )
    overall = {row["arm"]: row for row in result_rows if row["scope"] == "overall"}
    morphology = {row["arm"]: row for row in morphology_rows if row["scope"] == "overall"}
    yaw_reduction = reduction(morphology["strong"]["yaw_reversal_rate"], morphology["repaired"]["yaw_reversal_rate"])
    pitch_reduction = reduction(morphology["strong"]["pitch_reversal_rate"], morphology["repaired"]["pitch_reversal_rate"])
    yaw_tv_reduction = reduction(morphology["strong"]["yaw_directional_tv"], morphology["repaired"]["yaw_directional_tv"])
    pitch_tv_reduction = reduction(morphology["strong"]["pitch_directional_tv"], morphology["repaired"]["pitch_directional_tv"])
    strong_smooth = np.asarray([records["strong"][sid]["episode"]["trajectory_smoothness"] for sid in both], dtype=float)
    repaired_smooth = np.asarray([records["repaired"][sid]["episode"]["trajectory_smoothness"] for sid in both], dtype=float)
    smooth_reduction = reduction(float(np.mean(strong_smooth)), float(np.mean(repaired_smooth)))
    path_strong = np.asarray([records["strong"][sid]["episode"]["team_path_length_m"] for sid in both], dtype=float)
    path_repaired = np.asarray([records["repaired"][sid]["episode"]["team_path_length_m"] for sid in both], dtype=float)
    completion_strong = np.asarray([records["strong"][sid]["episode"]["completion_time_s"] for sid in both], dtype=float)
    completion_repaired = np.asarray([records["repaired"][sid]["episode"]["completion_time_s"] for sid in both], dtype=float)
    paired_quality: dict[str, dict[str, float]] = {}
    quality_sources = {
        "existing_smoothness_cost": lambda arm, sid: records[arm][sid]["episode"]["trajectory_smoothness"],
        "vertical_jerk_mean_squared": lambda arm, sid: metrics[arm][sid]["vertical_jerk_mean_squared"],
        "lateral_jerk_mean_squared": lambda arm, sid: metrics[arm][sid]["lateral_jerk_mean_squared"],
        "episode_p95_jerk_mps3": lambda arm, sid: metrics[arm][sid]["jerk_p95"],
        "team_path_length_m": lambda arm, sid: records[arm][sid]["episode"]["team_path_length_m"],
        "completion_time_s": lambda arm, sid: records[arm][sid]["episode"]["completion_time_s"],
    }
    for name, getter in quality_sources.items():
        paired_quality[name] = {
            arm: float(np.mean([getter(arm, sid) for sid in both])) for arm in DISPLAY
        }
        paired_quality[name]["reduction_percent"] = reduction(
            paired_quality[name]["strong"], paired_quality[name]["repaired"]
        )

    fixed_ids = contract["development"]["fixed_visual_scene_ids_before_outcomes"] if block == "development" else contract["holdout"]["fixed_visual_scene_ids_before_outcomes"]
    visual_rows: list[dict[str, Any]] = []
    visual_positive = 0
    for sid in fixed_ids:
        strong = metrics["strong"][sid]
        repaired = metrics["repaired"][sid]
        strong_rev = strong["yaw_reversal_rate"] + strong["pitch_reversal_rate"]
        repaired_rev = repaired["yaw_reversal_rate"] + repaired["pitch_reversal_rate"]
        strong_tv = strong["yaw_directional_tv"] + strong["pitch_directional_tv"]
        repaired_tv = repaired["yaw_directional_tv"] + repaired["pitch_directional_tv"]
        strong_run = float(np.nanmean([strong["yaw_mean_run_s"], strong["pitch_mean_run_s"]]))
        repaired_run = float(np.nanmean([repaired["yaw_mean_run_s"], repaired["pitch_mean_run_s"]]))
        positive = bool(repaired_rev < strong_rev and repaired_tv < strong_tv and repaired_run > strong_run)
        visual_positive += int(positive)
        visual_rows.append(
            {
                "block": block,
                "scenario_id": sid,
                "stage": records["strong"][sid]["entry_identity"]["stage"],
                "strong_success": bool(records["strong"][sid]["episode"]["team_success"]),
                "repaired_success": bool(records["repaired"][sid]["episode"]["team_success"]),
                "combined_reversal_reduction_percent": reduction(strong_rev, repaired_rev),
                "combined_directional_tv_reduction_percent": reduction(strong_tv, repaired_tv),
                "mean_same_direction_run_change_s": repaired_run - strong_run,
                "objective_longer_arc_proxy": positive,
            }
        )

    success_loss_pp = 100.0 * (overall["strong"]["team_success_rate"] - overall["repaired"]["team_success_rate"])
    peer_delta_pp = 100.0 * (overall["repaired"]["peer_collision_rate"] - overall["strong"]["peer_collision_rate"])
    obstacle_delta_pp = 100.0 * (overall["repaired"]["obstacle_collision_rate"] - overall["strong"]["obstacle_collision_rate"])
    gate = contract["development_gates_frozen_before_results"]
    gates = {
        "reliability": success_loss_pp <= float(gate["maximum_team_success_loss_pp"]) + 1e-12,
        "peer_safety": peer_delta_pp <= float(gate["maximum_peer_collision_increase_pp"]) + 1e-12,
        "obstacle_safety": obstacle_delta_pp <= float(gate["maximum_obstacle_collision_increase_pp"]) + 1e-12,
        "yaw_reversal": yaw_reduction >= float(gate["minimum_executed_yaw_reversal_reduction_percent"]),
        "pitch_reversal": pitch_reduction >= float(gate["minimum_executed_pitch_reversal_reduction_percent"]),
        "yaw_rate_tv": yaw_tv_reduction >= float(gate["minimum_yaw_rate_total_variation_reduction_percent"]),
        "pitch_rate_tv": pitch_tv_reduction >= float(gate["minimum_pitch_rate_total_variation_reduction_percent"]),
        "strong_smoothness_preserved": smooth_reduction >= -float(gate["maximum_smoothness_cost_increase_percent_on_both_success"]),
        "fixed_raw_visual_longer_arc_proxy": visual_positive >= int(gate["minimum_visual_scenes_with_longer_arcs"]),
    }
    gate_pass = bool(all(gates.values()))

    failure_rows: list[dict[str, Any]] = []
    for sid in ids:
        strong_success = bool(records["strong"][sid]["episode"]["team_success"])
        repaired_success = bool(records["repaired"][sid]["episode"]["team_success"])
        if not (strong_success and not repaired_success):
            continue
        trace = np.load(RECORD_ROOTS[block]["repaired"] / records["repaired"][sid]["limiter_trace_file"])
        start_step = max(0, int(records["repaired"][sid]["episode"]["steps"]) - 10)
        mask = np.asarray(trace["step"], dtype=int) >= start_step
        indexes = np.flatnonzero(mask)
        for index in indexes:
            failure_rows.append(
                {
                    "block": block,
                    "scenario_id": sid,
                    "stage": records["strong"][sid]["entry_identity"]["stage"],
                    "repaired_outcome": records["repaired"][sid]["episode"]["termination_reason"],
                    "step": int(trace["step"][index]),
                    "agent_id": int(trace["agent_id"][index]),
                    "safety_margin_m": float(trace["safety_margin_m"][index]),
                    "raw_acceleration": json.dumps(trace["raw_acceleration"][index].tolist()),
                    "strong_limited_acceleration": json.dumps(trace["strong_limited_acceleration"][index].tolist()),
                    "persistence_output_acceleration": json.dumps(trace["persistence_output_acceleration"][index].tolist()),
                    "final_physical_acceleration": json.dumps(trace["executed_acceleration"][index].tolist()),
                    "horizontal_accepted_sign": int(trace["horizontal_accepted_sign"][index]),
                    "vertical_accepted_sign": int(trace["vertical_accepted_sign"][index]),
                    "horizontal_pending_sign": int(trace["horizontal_pending_sign"][index]),
                    "vertical_pending_sign": int(trace["vertical_pending_sign"][index]),
                    "horizontal_pending_count": int(trace["horizontal_pending_count"][index]),
                    "vertical_pending_count": int(trace["vertical_pending_count"][index]),
                    "horizontal_activation": bool(trace["horizontal_persistence_activation"][index]),
                    "vertical_activation": bool(trace["vertical_persistence_activation"][index]),
                    "horizontal_override": bool(trace["horizontal_magnitude_override"][index]),
                    "vertical_override": bool(trace["vertical_magnitude_override"][index]),
                    "safety_bypass": bool(trace["persistence_bypassed_for_safety"][index]),
                    "persistence_modified": bool(trace["persistence_modified"][index]),
                    "non_emergency_turn_suppression": bool(
                        trace["persistence_modified"][index]
                        and not trace["persistence_bypassed_for_safety"][index]
                    ),
                }
            )

    trace_stats = trace_summary(records["repaired"], ids)
    decision = {
        "schema_version": f"turn_persistence_{block}_decision_v1",
        "block": block,
        "scenario_count": len(ids),
        "both_success_count": len(both),
        "team_success": {arm: overall[arm]["team_success_rate"] for arm in DISPLAY},
        "success_loss_pp": success_loss_pp,
        "peer_collision_delta_pp": peer_delta_pp,
        "obstacle_collision_delta_pp": obstacle_delta_pp,
        "yaw_reversal_reduction_percent": yaw_reduction,
        "pitch_reversal_reduction_percent": pitch_reduction,
        "yaw_reversal_rate": {
            "strong": morphology["strong"]["yaw_reversal_rate"],
            "repaired": morphology["repaired"]["yaw_reversal_rate"],
        },
        "pitch_reversal_rate": {
            "strong": morphology["strong"]["pitch_reversal_rate"],
            "repaired": morphology["repaired"]["pitch_reversal_rate"],
        },
        "same_sign_run_duration_s": {
            "yaw_median_strong": morphology["strong"]["yaw_median_run_s"],
            "yaw_median_repaired": morphology["repaired"]["yaw_median_run_s"],
            "yaw_median_change": morphology["repaired"]["yaw_median_run_s"] - morphology["strong"]["yaw_median_run_s"],
            "yaw_p90_strong": morphology["strong"]["yaw_p90_run_s"],
            "yaw_p90_repaired": morphology["repaired"]["yaw_p90_run_s"],
            "pitch_median_strong": morphology["strong"]["pitch_median_run_s"],
            "pitch_median_repaired": morphology["repaired"]["pitch_median_run_s"],
            "pitch_median_change": morphology["repaired"]["pitch_median_run_s"] - morphology["strong"]["pitch_median_run_s"],
            "pitch_p90_strong": morphology["strong"]["pitch_p90_run_s"],
            "pitch_p90_repaired": morphology["repaired"]["pitch_p90_run_s"],
        },
        "yaw_rate_total_variation_reduction_percent": yaw_tv_reduction,
        "pitch_rate_total_variation_reduction_percent": pitch_tv_reduction,
        "smoothness_reduction_percent_both_success": smooth_reduction,
        "smoothness_reduction_bootstrap_95ci": bootstrap_smoothness(strong_smooth, repaired_smooth, 2026082521),
        "path_length_reduction_percent_both_success": reduction(float(np.mean(path_strong)), float(np.mean(path_repaired))),
        "completion_time_delta_s_both_success": float(np.mean(completion_repaired - completion_strong)),
        "paired_both_success_quality": paired_quality,
        "fixed_visual_longer_arc_proxy_count": visual_positive,
        "turn_persistence": trace_stats,
        "gates": gates,
        ("DEV_GATE" if block == "development" else "HOLDOUT_GATE"): "PASS" if gate_pass else "FAIL",
        "HOLDOUT_AUTHORIZED": bool(gate_pass) if block == "development" else None,
        "FORMAL_V2_EXECUTED": False,
        "record_set_semantic_sha256": stable_hash(
            {
                arm: {
                    sid: {
                        "success": records[arm][sid]["episode"]["team_success"],
                        "termination": records[arm][sid]["episode"]["termination_reason"],
                        "trajectory_sha256": records[arm][sid]["trajectory_sha256"],
                    }
                    for sid in ids
                }
                for arm in DISPLAY
            }
        ),
    }
    suffix = "DEV" if block == "development" else "HOLDOUT"
    for row in result_rows:
        if row["scope"] == "overall":
            arm = str(row["arm"])
            row.update(
                {
                    f"both_success_{name}": values[arm]
                    for name, values in paired_quality.items()
                }
            )
    write_csv(ARTIFACT_ROOT / f"TURN_PERSISTENCE_{suffix}_RESULTS.csv", result_rows)
    write_csv(ARTIFACT_ROOT / f"TURN_PERSISTENCE_{suffix}_MORPHOLOGY.csv", [*morphology_rows, *visual_rows])
    write_csv(
        ARTIFACT_ROOT / f"TURN_PERSISTENCE_{suffix}_FAILURE_AUDIT.csv",
        failure_rows,
        fieldnames=(
            "block", "scenario_id", "stage", "repaired_outcome", "step", "agent_id",
            "safety_margin_m", "raw_acceleration", "strong_limited_acceleration",
            "persistence_output_acceleration", "final_physical_acceleration",
            "horizontal_accepted_sign", "vertical_accepted_sign", "horizontal_pending_sign",
            "vertical_pending_sign", "horizontal_pending_count", "vertical_pending_count",
            "horizontal_activation", "vertical_activation", "horizontal_override",
            "vertical_override", "safety_bypass", "persistence_modified",
            "non_emergency_turn_suppression",
        ),
    )
    if block == "development":
        write_csv(
            ARTIFACT_ROOT / "TURN_PERSISTENCE_FAILURE_AUDIT.csv",
            failure_rows,
            fieldnames=(
                "block", "scenario_id", "stage", "repaired_outcome", "step", "agent_id",
                "safety_margin_m", "raw_acceleration", "strong_limited_acceleration",
                "persistence_output_acceleration", "final_physical_acceleration",
                "horizontal_accepted_sign", "vertical_accepted_sign", "horizontal_pending_sign",
                "vertical_pending_sign", "horizontal_pending_count", "vertical_pending_count",
                "horizontal_activation", "vertical_activation", "horizontal_override",
                "vertical_override", "safety_bypass", "persistence_modified",
                "non_emergency_turn_suppression",
            ),
        )
    atomic_json(
        ARTIFACT_ROOT / ("TURN_PERSISTENCE_DEV_GO_NO_GO.json" if block == "development" else "TURN_PERSISTENCE_HOLDOUT_DECISION.json"),
        decision,
    )
    print(json.dumps(json_ready(decision), indent=2))
    return decision


def plot_raw(block: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import importlib.util

    # Load the plotting helpers by file path so this read-only renderer does
    # not initialize the heavyweight planning package and a second OpenMP
    # runtime through the policy-preview import chain.
    helper_path = REPO_ROOT / "planning/plot_continuous_reference_transition.py"
    helper_spec = importlib.util.spec_from_file_location("_turn_persistence_plot_helpers", helper_path)
    if helper_spec is None or helper_spec.loader is None:
        raise RuntimeError("could not load obstacle plotting helpers")
    helper_module = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper_module)
    draw_obstacles_3d = helper_module.draw_obstacles_3d
    draw_obstacles_xy = helper_module.draw_obstacles_xy

    contract = load_json(RUNTIME_CONTRACT)
    manifest = load_json(MANIFEST_PATHS[block])
    index = {str(row["scenario_id"]): row for row in manifest["entries"]}
    fixed = contract["development"]["fixed_visual_scene_ids_before_outcomes"] if block == "development" else contract["holdout"]["fixed_visual_scene_ids_before_outcomes"]
    output = ARTIFACT_ROOT / ("TURN_PERSISTENCE_DEV_RAW_TRAJECTORIES.pdf" if block == "development" else "TURN_PERSISTENCE_HOLDOUT_RAW_TRAJECTORIES.pdf")
    colors = ("#0072B2", "#D55E00", "#009E73")
    styles = {"strong": "--", "repaired": "-"}
    labels = {"strong": "Frozen Strong", "repaired": "Repaired"}
    with PdfPages(output) as pdf:
        for sid in fixed:
            scene = load_json(REPO_ROOT / index[sid]["scenario_file"])
            payloads = {arm: load_json(RECORD_ROOTS[block][arm] / f"{sid}.json") for arm in DISPLAY}
            arrays = {arm: np.load(RECORD_ROOTS[block][arm] / payloads[arm]["trajectory_file"]) for arm in DISPLAY}
            figure = plt.figure(figsize=(14.5, 8.2))
            grid = figure.add_gridspec(2, 3, left=0.055, right=0.985, bottom=0.075, top=0.81, wspace=0.20, hspace=0.34)
            ax3d = figure.add_subplot(grid[0, 0], projection="3d")
            axxy = figure.add_subplot(grid[0, 1])
            axz = figure.add_subplot(grid[0, 2])
            axyaw = figure.add_subplot(grid[1, 0])
            axpitch = figure.add_subplot(grid[1, 1])
            axevents = figure.add_subplot(grid[1, 2])
            max_steps = max(len(arrays[arm]["positions"]) for arm in DISPLAY)
            draw_obstacles_3d(ax3d, scene, max_steps)
            draw_obstacles_xy(axxy, scene, max_steps)
            for arm in DISPLAY:
                position = np.asarray(arrays[arm]["positions"], dtype=float)
                velocity = np.asarray(arrays[arm]["velocities"], dtype=float)
                active_goal = np.asarray(arrays[arm]["active_goals"], dtype=float)
                time = np.arange(len(position)) * 0.1
                for agent in range(3):
                    line_label = f"UAV {agent + 1} - {labels[arm]}"
                    width = 1.35 if arm == "repaired" else 0.9
                    ax3d.plot(*position[:, agent].T, color=colors[agent], linestyle=styles[arm], linewidth=width, label=line_label)
                    axxy.plot(position[:, agent, 0], position[:, agent, 1], color=colors[agent], linestyle=styles[arm], linewidth=width)
                    axz.plot(time, position[:, agent, 2], color=colors[agent], linestyle=styles[arm], linewidth=width)
                    signal = legacy.direction_signals(velocity[:, agent], 0.1)
                    axyaw.plot(time, signal["yaw_rate"], color=colors[agent], linestyle=styles[arm], linewidth=0.8)
                    axpitch.plot(time, signal["pitch_rate"], color=colors[agent], linestyle=styles[arm], linewidth=0.8)
                    for axis, name in ((axyaw, "yaw"), (axpitch, "pitch")):
                        reversals = np.flatnonzero(signal[f"{name}_reversal"])
                        axis.scatter(time[reversals], signal[f"{name}_rate"][reversals], s=7, facecolors=(colors[agent] if arm == "repaired" else "none"), edgecolors=colors[agent], linewidths=0.45)
                    changes = np.r_[True, np.linalg.norm(np.diff(active_goal[:, agent], axis=0), axis=1) > 1e-9]
                    steps = np.flatnonzero(changes)
                    if arm == "repaired":
                        axevents.scatter(
                            time[steps], np.full(len(steps), 2 * agent + 0.5),
                            color=colors[agent], marker="o", s=8, linewidths=0.4,
                            facecolors="none",
                        )
                if arm == "repaired":
                    trace = np.load(RECORD_ROOTS[block][arm] / payloads[arm]["limiter_trace_file"])
                    for agent in range(3):
                        agent_mask = np.asarray(trace["agent_id"], dtype=int) == agent
                        trace_time = np.asarray(trace["step"], dtype=float)[agent_mask] * 0.1
                        horizontal_sign = np.asarray(trace["horizontal_accepted_sign"], dtype=float)[agent_mask]
                        vertical_sign = np.asarray(trace["vertical_accepted_sign"], dtype=float)[agent_mask]
                        horizontal_pending = np.asarray(trace["horizontal_pending_count"], dtype=int)[agent_mask] > 0
                        vertical_pending = np.asarray(trace["vertical_pending_count"], dtype=int)[agent_mask] > 0
                        horizontal_lane = 2 * agent
                        vertical_lane = 2 * agent + 1
                        axevents.step(
                            trace_time, horizontal_lane + 0.24 * horizontal_sign,
                            where="post", color=colors[agent], linewidth=0.75,
                        )
                        axevents.step(
                            trace_time, vertical_lane + 0.24 * vertical_sign,
                            where="post", color=colors[agent], linestyle=":", linewidth=0.75,
                        )
                        axevents.fill_between(
                            trace_time, horizontal_lane - 0.38, horizontal_lane + 0.38,
                            where=horizontal_pending, step="post", color=colors[agent], alpha=0.10,
                        )
                        axevents.fill_between(
                            trace_time, vertical_lane - 0.38, vertical_lane + 0.38,
                            where=vertical_pending, step="post", color=colors[agent], alpha=0.10,
                        )
                    event_specs = (
                        ("early_bypass", "v", 0.5),
                        ("hard_bypass", "*", 0.5),
                    )
                    for key, marker, lane_offset in event_specs:
                        mask = np.asarray(trace[key], dtype=bool)
                        steps = np.asarray(trace["step"], dtype=int)[mask]
                        agents = np.asarray(trace["agent_id"], dtype=int)[mask]
                        y = 2 * agents + lane_offset
                        axevents.scatter(
                            steps * 0.1, y, c=[colors[i] for i in agents], marker=marker,
                            s=13 if marker != "*" else 22, linewidths=0.45,
                        )
            starts = np.asarray(scene["starts"], dtype=float)
            goals = np.asarray(scene["goals"], dtype=float)
            axxy.scatter(starts[:, 0], starts[:, 1], marker="o", s=20, color=colors, edgecolor="k", linewidth=0.3)
            axxy.scatter(goals[:, 0], goals[:, 1], marker="^", s=24, color=colors, edgecolor="k", linewidth=0.3)
            ax3d.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="Raw 3-D trajectories")
            ax3d.view_init(elev=24, azim=-58)
            axxy.set(xlabel="x (m)", ylabel="y (m)", title="Raw XY paths", aspect="equal")
            axz.set(xlabel="time (s)", ylabel="z (m)", title="Raw altitude")
            axyaw.set(xlabel="time (s)", ylabel="yaw rate (rad/s)", title="Raw yaw rate; dots are reversals")
            axpitch.set(xlabel="time (s)", ylabel="pitch rate (rad/s)", title="Raw pitch rate; dots are reversals")
            axevents.set(xlabel="time (s)", ylabel="state lane", title="Accepted sign, pending state, reference, bypass")
            axevents.set_yticks(
                list(range(6)),
                ["U1-H", "U1-V", "U2-H", "U2-V", "U3-H", "U3-V"],
            )
            axevents.set_ylim(-0.55, 5.55)
            for axis in (axxy, axz, axyaw, axpitch, axevents):
                axis.grid(True, color="0.88", linewidth=0.45)
            handles, legend_labels = ax3d.get_legend_handles_labels()
            figure.suptitle(f"{scene['stage'].replace('_', ' ').title()} - {sid} - unsmoothed 0.1 s records", y=0.982)
            figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=False)
            pdf.savefig(figure)
            plt.close(figure)
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--block", choices=("development", "holdout"), required=True)
    plot_parser = sub.add_parser("plot-raw")
    plot_parser.add_argument("--block", choices=("development", "holdout"), required=True)
    args = parser.parse_args()
    if args.command == "analyze":
        analyze(args.block)
    else:
        plot_raw(args.block)


if __name__ == "__main__":
    main()
