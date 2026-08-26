#!/usr/bin/env python3
"""Analyze FP-SHEP preview transient predictability and replaceability gates."""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "artifacts/transient_aware_candidate_veto/20260824_184551"
CRT_ROOT = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552"
BLOCKS = {
    "development": {
        "runs": ROOT / "02_predictability/diagnostic_runs/development/episode_records",
        "source": CRT_ROOT / "04_development/records/original/episode_records",
    },
    "holdout": {
        "runs": ROOT / "02_predictability/diagnostic_runs/holdout/episode_records",
        "source": CRT_ROOT / "07_holdout/records/original/episode_records",
    },
}
DT = 0.1
H = 4
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260824


def average_ranks(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return one-based average ranks with deterministic tie handling."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("rank input must be a finite one-dimensional array")
    order = np.argsort(array, kind="mergesort")
    ranked = np.empty(array.size, dtype=float)
    position = 0
    while position < array.size:
        end = position + 1
        while end < array.size and array[order[end]] == array[order[position]]:
            end += 1
        ranked[order[position:end]] = 0.5 * ((position + 1) + end)
        position = end
    return ranked


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    data = list(rows)
    fieldnames = list(fields or [])
    if not fieldnames:
        for row in data:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(str(key))
    if not fieldnames:
        fieldnames = ["status", "reason"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow({key: scalar(row.get(key)) for key in fieldnames})
    temporary.replace(path)


def load_preview_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def event_jerk_metrics(accelerations: np.ndarray, agent_id: int, step: int) -> dict[str, Any]:
    acceleration = np.asarray(accelerations[:, int(agent_id), :], dtype=float)
    jerk = np.diff(acceleration, axis=0) / DT
    jerk_steps = np.arange(1, acceleration.shape[0], dtype=int)

    def selected(last_offset: int) -> np.ndarray:
        mask = (jerk_steps >= int(step)) & (jerk_steps <= int(step + last_offset))
        return jerk[mask]

    post_02 = selected(2)
    post_05 = selected(5)
    post_h = selected(H - 1)

    def peak_norm(value: np.ndarray) -> float | None:
        return float(np.max(np.linalg.norm(value, axis=1))) if value.size else None

    def peak_vertical(value: np.ndarray) -> float | None:
        return float(np.max(np.abs(value[:, 2]))) if value.size else None

    def peak_lateral(value: np.ndarray) -> float | None:
        return float(np.max(np.linalg.norm(value[:, :2], axis=1))) if value.size else None

    return {
        "J_real_peak_0p2": peak_norm(post_02),
        "J_real_peak_0p5": peak_norm(post_05),
        "J_real_mean_0p5": float(np.mean(np.sum(post_05 ** 2, axis=1))) if post_05.size else None,
        "J_real_mean_H0p4": float(np.mean(np.sum(post_h ** 2, axis=1))) if post_h.size else None,
        "J_real_vertical_peak": peak_vertical(post_05),
        "J_real_lateral_peak": peak_lateral(post_05),
        "realized_H_sample_count": int(post_h.shape[0]),
        "realized_0p5_sample_count": int(post_05.shape[0]),
    }


def candidate_rank(logits: Sequence[float], candidate_id: int) -> int:
    candidate_logits = np.asarray(logits[1:], dtype=float)
    order = np.argsort(-candidate_logits, kind="stable")
    location = np.flatnonzero(order == int(candidate_id))
    if location.size != 1:
        raise RuntimeError("invalid candidate/logit mapping")
    return int(location[0] + 1)


def candidate_interaction(event: Mapping[str, Any], candidate_id: int) -> dict[str, Any]:
    records = {
        int(row["candidate_id"]): row
        for row in event.get("all_candidate_interaction_records", [])
    }
    row = records.get(int(candidate_id), {})
    minimum = row.get("minimum_predicted_separation_m")
    return {
        "interaction_edge_count": int(row.get("interaction_edge_count", 0)),
        "minimum_predicted_separation_m": None if minimum is None else float(minimum),
        "maximum_risk_duration_s": float(row.get("maximum_risk_duration_s", 0.0)),
        "d_safe_m": float(row.get("d_safe_m", 0.6)),
        "risky": bool(row.get("risky", False)),
    }


def selected_events(expected_per_block: int | None = 100) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block, contract in BLOCKS.items():
        reproduction_paths = sorted(contract["runs"].glob("*_reproduction.json"))
        if expected_per_block is not None and len(reproduction_paths) != expected_per_block:
            raise RuntimeError(
                f"{block} diagnostic is incomplete: "
                f"{len(reproduction_paths)}/{expected_per_block}"
            )
        if any(load_json(path)["status"] != "PASS" for path in reproduction_paths):
            raise RuntimeError(f"{block} includes a failed exact reproduction")
        for reproduction_path in reproduction_paths:
            scenario_id = reproduction_path.name.removesuffix("_reproduction.json")
            source_path = contract["source"] / f"{scenario_id}.json"
            source = load_json(source_path)
            preview_path = contract["runs"] / f"{scenario_id}_candidate_previews.jsonl.gz"
            candidates = load_preview_rows(preview_path)
            by_key = {
                (int(row["step"]), int(row["agent_id"]), int(row["candidate_id"])): row
                for row in candidates
            }
            event_by_agent: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
            for event in source["events"]:
                if bool(event.get("goal_changed", False)):
                    event_by_agent[int(event["agent_id"])].append(event)
            next_changed: dict[tuple[int, int], int | None] = {}
            for agent_id, events in event_by_agent.items():
                ordered = sorted(events, key=lambda row: int(row["step"]))
                for index, event in enumerate(ordered):
                    next_changed[(int(event["step"]), agent_id)] = (
                        int(ordered[index + 1]["step"]) if index + 1 < len(ordered) else None
                    )
            trajectory_path = source_path.with_name(str(source["trajectory_file"]))
            with np.load(trajectory_path, allow_pickle=False) as trajectory:
                accelerations = np.asarray(trajectory["applied_accelerations"], dtype=float)
                for event in source["events"]:
                    if event.get("event") == "INITIAL_SELECTION":
                        continue
                    if not bool(event.get("goal_changed", False)):
                        continue
                    selected = event.get("selected_candidate_id")
                    if selected is None or bool(event.get("selected_null", False)):
                        continue
                    step = int(event["step"])
                    agent_id = int(event["agent_id"])
                    key = (step, agent_id, int(selected))
                    if key not in by_key:
                        raise RuntimeError(f"selected preview row missing: {block} {scenario_id} {key}")
                    candidate = by_key[key]
                    next_step = next_changed.get((step, agent_id))
                    delta = None if next_step is None else int(next_step - step)
                    realized = event_jerk_metrics(accelerations, agent_id, step)
                    full_realized_horizon = int(realized["realized_H_sample_count"]) == H
                    clean = (delta is None or delta > H) and full_realized_horizon
                    interaction = candidate_interaction(event, int(selected))
                    logits = event.get("class_logits") or []
                    result.append(
                        {
                            "block": block,
                            "scenario_id": scenario_id,
                            "seed": int(source["summary"]["seed"]),
                            "stage": str(source["summary"]["stage"]),
                            "family": str(source["summary"]["family"]),
                            "task_pattern": str(source["summary"].get("task_pattern", "")),
                            "episode_team_success": bool(source["episode"]["team_success"]),
                            "episode_collision": bool(source["episode"]["collision"]),
                            "episode_peer_collision": bool(source["episode"].get("inter_agent_collision", False)),
                            "agent_id": agent_id,
                            "step": step,
                            "time_s": float(step * DT),
                            "event_type": str(event["event"]),
                            "selected_candidate_id": int(selected),
                            "selected_gat_candidate_rank": candidate_rank(logits, int(selected)),
                            "selected_gat_logit": float(logits[int(selected) + 1]),
                            "selected_gat_probability": float((event.get("class_probabilities") or [])[int(selected) + 1]),
                            "next_command_step": next_step,
                            "steps_to_next_command": delta,
                            "clean_window_H4": clean,
                            "full_realized_H4_available": full_realized_horizon,
                            "terminal_approach_source_flag": bool(event.get("terminal_null_eligible", False)),
                            "candidate_world_x_m": float(candidate["candidate_world_position"][0]),
                            "candidate_world_y_m": float(candidate["candidate_world_position"][1]),
                            "candidate_world_z_m": float(candidate["candidate_world_position"][2]),
                            "J_preview": float(candidate["J_preview"]),
                            "J_preview_peak": float(candidate["J_preview_peak"]),
                            "J_preview_mean": float(candidate["J_preview_mean"]),
                            "J_preview_vertical": float(candidate["J_preview_vertical"]),
                            "J_preview_lateral": float(candidate["J_preview_lateral"]),
                            "preview_min_clearance_m": float(candidate["preview_min_clearance"]),
                            "preview_time_to_min_clearance_s": float(candidate["preview_time_to_min_clearance_s"]),
                            "preview_task_progress_m": float(candidate["preview_task_progress"]),
                            "preview_max_execution_deviation_m": float(candidate["preview_max_execution_deviation"]),
                            "preview_terminal_speed_mps": float(candidate["preview_terminal_speed"]),
                            "interaction_edge_count": interaction["interaction_edge_count"],
                            "minimum_predicted_peer_separation_m": interaction["minimum_predicted_separation_m"],
                            "maximum_peer_risk_duration_s": interaction["maximum_risk_duration_s"],
                            "selected_candidate_risky": interaction["risky"],
                            **realized,
                        }
                    )
    return result


def weighted_cluster_spearman_ci(rows: Sequence[Mapping[str, Any]], x_name: str, y_name: str) -> tuple[float, float, float]:
    x = np.asarray([float(row[x_name]) for row in rows], dtype=float)
    y = np.asarray([float(row[y_name]) for row in rows], dtype=float)
    clusters = np.asarray([str(row["scenario_id"]) for row in rows])
    rx = average_ranks(x)
    ry = average_ranks(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    unique = np.unique(clusters)
    summaries: list[np.ndarray] = []
    for cluster in unique:
        mask = clusters == cluster
        a = rx[mask]
        b = ry[mask]
        summaries.append(
            np.asarray(
                [len(a), np.sum(a), np.sum(b), np.sum(a * a), np.sum(b * b), np.sum(a * b)],
                dtype=float,
            )
        )
    matrix = np.stack(summaries)
    rng = np.random.default_rng(BOOTSTRAP_SEED + len(rows))
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(unique), size=len(unique))
        n, sx, sy, sxx, syy, sxy = np.sum(matrix[sampled], axis=0)
        covariance = sxy - sx * sy / n
        variance_x = sxx - sx * sx / n
        variance_y = syy - sy * sy / n
        values[index] = covariance / math.sqrt(max(variance_x * variance_y, 1.0e-300))
    return rho, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positive = int(np.count_nonzero(labels))
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        return None
    ranks = average_ranks(scores)
    return float((np.sum(ranks[labels]) - positive * (positive + 1) / 2.0) / (positive * negative))


def rank_calibration(rows: Sequence[Mapping[str, Any]], block: str, subset: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    preview = np.asarray([float(row["J_preview"]) for row in rows], dtype=float)
    realized = np.asarray([float(row["J_real_mean_H0p4"]) for row in rows], dtype=float)
    output: list[dict[str, Any]] = []
    ratios: dict[str, float] = {}
    for bins, name in ((4, "quartile"), (10, "decile")):
        edges = np.quantile(preview, np.linspace(0.0, 1.0, bins + 1))
        membership = np.searchsorted(edges[1:-1], preview, side="right")
        means: list[float] = []
        for index in range(bins):
            values = realized[membership == index]
            means.append(float(np.mean(values)) if values.size else float("nan"))
            output.append(
                {
                    "block": block,
                    "subset": subset,
                    "binning": name,
                    "bin": index + 1,
                    "event_count": int(values.size),
                    "preview_low_inclusive": float(edges[index]),
                    "preview_high_inclusive": float(edges[index + 1]),
                    "realized_mean_H0p4": None if not values.size else float(np.mean(values)),
                    "realized_median_H0p4": None if not values.size else float(np.median(values)),
                    "realized_peak_0p5_mean": None if not values.size else float(
                        np.mean([float(row["J_real_peak_0p5"]) for row, member in zip(rows, membership == index) if member])
                    ),
                }
            )
        if bins == 4 and means[0] > 0.0:
            ratios["quartile_q4_q1_ratio"] = float(means[-1] / means[0])
    return output, ratios


def predictability(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sections: dict[str, Any] = {}
    calibration_rows: list[dict[str, Any]] = []
    unscorable_by_block: dict[str, int] = {}
    for block in BLOCKS:
        raw_block_rows = [row for row in rows if row["block"] == block]
        block_rows = [
            row
            for row in raw_block_rows
            if row["J_real_mean_H0p4"] is not None
            and row["J_real_peak_0p5"] is not None
        ]
        unscorable_by_block[block] = len(raw_block_rows) - len(block_rows)
        for subset_name, subset_rows in (
            ("all_event", block_rows),
            ("clean_window", [row for row in block_rows if row["clean_window_H4"]]),
        ):
            if len(subset_rows) < 10:
                raise RuntimeError(f"insufficient {block} {subset_name} events")
            rho, ci_low, ci_high = weighted_cluster_spearman_ci(
                subset_rows, "J_preview", "J_real_mean_H0p4"
            )
            preview = np.asarray([float(row["J_preview"]) for row in subset_rows])
            realized_peak = np.asarray([float(row["J_real_peak_0p5"]) for row in subset_rows])
            high_real_threshold = float(np.percentile(realized_peak, 90))
            high_real = realized_peak >= high_real_threshold
            auc = auc_score(high_real, preview)
            section_calibration, ratios = rank_calibration(subset_rows, block, subset_name)
            calibration_rows.extend(section_calibration)
            threshold_metrics: dict[str, Any] = {}
            for percentile in (90, 95):
                threshold = float(np.percentile(preview, percentile))
                predicted = preview >= threshold
                tp = int(np.count_nonzero(predicted & high_real))
                threshold_metrics[f"P{percentile}"] = {
                    "threshold": threshold,
                    "selected_count": int(np.count_nonzero(predicted)),
                    "precision": float(tp / max(np.count_nonzero(predicted), 1)),
                    "recall": float(tp / max(np.count_nonzero(high_real), 1)),
                    "mean_realized_peak_in_tail": float(np.mean(realized_peak[predicted])),
                    "mean_realized_peak_outside_tail": float(np.mean(realized_peak[~predicted])),
                }
            sections[f"{block}_{subset_name}"] = {
                "event_count": len(subset_rows),
                "episode_count": len({row["scenario_id"] for row in subset_rows}),
                "spearman_rho": rho,
                "episode_cluster_bootstrap_95ci": [ci_low, ci_high],
                "bootstrap_rank_semantics": "Pearson correlation of globally fixed average ranks with episode-cluster resampling weights",
                "high_realized_jerk_threshold_P90": high_real_threshold,
                "high_realized_event_count": int(np.count_nonzero(high_real)),
                "high_jerk_auroc": auc,
                **ratios,
                "preview_percentile_screen": threshold_metrics,
            }

    clean = [sections[f"{block}_clean_window"] for block in BLOCKS]
    strong = all(
        section["spearman_rho"] >= 0.35
        and section["episode_cluster_bootstrap_95ci"][0] > 0.0
        and section["quartile_q4_q1_ratio"] >= 1.25
        and section["high_jerk_auroc"] is not None
        and section["high_jerk_auroc"] >= 0.65
        for section in clean
    )
    moderate = all(
        section["spearman_rho"] >= 0.15
        and section["episode_cluster_bootstrap_95ci"][0] > 0.0
        and section["quartile_q4_q1_ratio"] >= 1.10
        and section["high_jerk_auroc"] is not None
        and section["high_jerk_auroc"] >= 0.58
        for section in clean
    )
    consistently_positive = all(section["spearman_rho"] > 0.0 for section in clean)
    label = "STRONG" if strong else "MODERATE" if moderate else "WEAK" if consistently_positive else "NONE"
    dev_clean = [row for row in rows if row["block"] == "development" and row["clean_window_H4"]]
    activation_threshold = float(np.percentile([float(row["J_preview"]) for row in dev_clean], 90))
    return (
        {
            "schema_version": "preview_transient_predictability_v1",
            "primary_analysis": "CLEAN_WINDOW",
            "metric": "J_preview vs J_real_mean_H0p4",
            "clean_window_data_requirement": "no later accepted command through s+4 and exactly four realized jerk samples",
            "sections": sections,
            "development_P90_preview_activation_threshold": activation_threshold,
            "PREVIEW_TRANSIENT_PREDICTABILITY": label,
            "TACV_PREDICTABILITY_GATE_PASS": label in {"STRONG", "MODERATE"},
            "unscorable_terminal_tail_event_count_by_block": unscorable_by_block,
            "classifier_trained": False,
            "formal_v2_used_for_parameter_selection": False,
        },
        calibration_rows,
    )


def safety_contract() -> dict[str, Any]:
    return {
        "schema_version": "tacv_safety_admissibility_contract_v1",
        "frozen_before_replaceability_result": True,
        "proposal_feasibility": "Every member of the frozen ordered Top-K was already accepted by the existing Proposal feasibility/progress filter.",
        "interaction_acceptance": "Reuse the existing mask_risky_when_safe_alternative_exists binary risky descriptor: risky iff d_min<d_safe or T_risk>0.",
        "obstacle_preview_semantics": "Reuse FP-SHEP frozen-LiDAR approximate minimum clearance; no typed obstacle or new threshold is introduced.",
        "safety_noninferior_rule": [
            "alternative is an existing Top-K proposal",
            "alternative interaction-risk class is no worse than selected (safe<=risky)",
            "alternative preview minimum clearance is at least selected preview minimum clearance",
            "if both are interaction-risky: alternative risk duration is no greater and minimum predicted separation is no smaller",
        ],
        "numeric_comparison_tolerance": 1.0e-12,
        "smoothness_never_overrides_safety": True,
        "new_weighted_safety_score": False,
        "clearance_source_limitation": "untyped frozen visible-surface approximation; static/dynamic dominance cannot be inferred",
    }


def noninferior(selected: Mapping[str, Any], alternative: Mapping[str, Any]) -> bool:
    tolerance = 1.0e-12
    if bool(alternative["risky"]) and not bool(selected["risky"]):
        return False
    if float(alternative["preview_min_clearance"]) + tolerance < float(selected["preview_min_clearance"]):
        return False
    if bool(selected["risky"]) and bool(alternative["risky"]):
        if float(alternative["maximum_risk_duration_s"]) > float(selected["maximum_risk_duration_s"]) + tolerance:
            return False
        selected_sep = float("inf") if selected["minimum_predicted_separation_m"] is None else float(selected["minimum_predicted_separation_m"])
        alternative_sep = float("inf") if alternative["minimum_predicted_separation_m"] is None else float(alternative["minimum_predicted_separation_m"])
        if alternative_sep + tolerance < selected_sep:
            return False
    return True


def replaceability(
    selected_rows: Sequence[Mapping[str, Any]],
    activation_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected_index = {
        (row["block"], row["scenario_id"], int(row["step"]), int(row["agent_id"])): row
        for row in selected_rows
    }
    candidate_output: list[dict[str, Any]] = []
    event_output: list[dict[str, Any]] = []
    for block, contract in BLOCKS.items():
        for reproduction_path in sorted(contract["runs"].glob("*_reproduction.json")):
            scenario_id = reproduction_path.name.removesuffix("_reproduction.json")
            source_path = contract["source"] / f"{scenario_id}.json"
            source = load_json(source_path)
            preview_rows = load_preview_rows(contract["runs"] / f"{scenario_id}_candidate_previews.jsonl.gz")
            by_event: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for row in preview_rows:
                by_event[(int(row["step"]), int(row["agent_id"]))].append(row)
            for event in source["events"]:
                selected_id = event.get("selected_candidate_id")
                if event.get("event") == "INITIAL_SELECTION" or not bool(event.get("goal_changed", False)) or selected_id is None:
                    continue
                key = (block, scenario_id, int(event["step"]), int(event["agent_id"]))
                selected_summary = selected_index.get(key)
                if selected_summary is None or float(selected_summary["J_preview"]) < activation_threshold:
                    continue
                candidates = by_event[(int(event["step"]), int(event["agent_id"]))]
                logits = event["class_logits"]
                interaction_map = {
                    int(row["candidate_id"]): candidate_interaction(event, int(row["candidate_id"]))
                    for row in candidates
                }
                enriched: list[dict[str, Any]] = []
                for row in candidates:
                    cid = int(row["candidate_id"])
                    interaction = interaction_map[cid]
                    point = np.asarray(row["candidate_world_position"], dtype=float)
                    position = np.asarray(row["current_position"], dtype=float)
                    direction = point - position
                    norm = float(np.linalg.norm(direction))
                    direction = direction / norm if norm > 0.0 else np.zeros(3)
                    enriched.append(
                        {
                            **row,
                            **interaction,
                            "gat_rank": candidate_rank(logits, cid),
                            "gat_logit": float(logits[cid + 1]),
                            "direction": direction,
                        }
                    )
                selected = next(row for row in enriched if int(row["candidate_id"]) == int(selected_id))
                eligible: list[dict[str, Any]] = []
                for row in enriched:
                    is_selected = int(row["candidate_id"]) == int(selected_id)
                    safe_noninferior = (not is_selected) and noninferior(selected, row)
                    lower = (not is_selected) and float(row["J_preview"]) < float(selected["J_preview"]) - 1.0e-12
                    replacement_eligible = safe_noninferior and lower
                    if replacement_eligible:
                        eligible.append(row)
                    selected_direction = np.asarray(selected["direction"], dtype=float)
                    direction = np.asarray(row["direction"], dtype=float)
                    angle = float(np.degrees(np.arccos(np.clip(np.dot(selected_direction, direction), -1.0, 1.0))))
                    candidate_output.append(
                        {
                            "block": block,
                            "scenario_id": scenario_id,
                            "stage": selected_summary["stage"],
                            "family": selected_summary["family"],
                            "agent_id": int(event["agent_id"]),
                            "step": int(event["step"]),
                            "selected_candidate_id": int(selected_id),
                            "candidate_id": int(row["candidate_id"]),
                            "candidate_is_original_selected": is_selected,
                            "candidate_world_x_m": float(row["candidate_world_position"][0]),
                            "candidate_world_y_m": float(row["candidate_world_position"][1]),
                            "candidate_world_z_m": float(row["candidate_world_position"][2]),
                            "gat_rank": int(row["gat_rank"]),
                            "gat_logit": float(row["gat_logit"]),
                            "gat_logit_gap_selected_minus_candidate": float(selected["gat_logit"] - row["gat_logit"]),
                            "fp_shep_online_score": float(row["fp_shep_online_score"]),
                            "proposal_feasible_by_source_contract": True,
                            "preview_min_clearance_m": float(row["preview_min_clearance"]),
                            "minimum_predicted_peer_separation_m": row["minimum_predicted_separation_m"],
                            "maximum_peer_risk_duration_s": float(row["maximum_risk_duration_s"]),
                            "candidate_risky": bool(row["risky"]),
                            "J_preview": float(row["J_preview"]),
                            "preview_transient_reduction_vs_selected_fraction": float(1.0 - float(row["J_preview"]) / max(float(selected["J_preview"]), 1.0e-300)),
                            "direction_difference_deg": angle,
                            "safety_noninferior": safe_noninferior,
                            "strictly_lower_transient": lower,
                            "replacement_eligible": replacement_eligible,
                        }
                    )
                eligible_by_rank = sorted(eligible, key=lambda row: (int(row["gat_rank"]), int(row["candidate_id"])))
                best = min(eligible, key=lambda row: (float(row["J_preview"]), int(row["gat_rank"]))) if eligible else None
                first = eligible_by_rank[0] if eligible_by_rank else None
                reduction_sets = {
                    threshold: [
                        row
                        for row in eligible
                        if 1.0
                        - float(row["J_preview"])
                        / max(float(selected["J_preview"]), 1.0e-300)
                        >= threshold - 1.0e-12
                    ]
                    for threshold in (0.20, 0.35, 0.50)
                }
                high_peer = any(int(row["interaction_edge_count"]) > 0 for row in enriched)
                terminal = bool(selected_summary["terminal_approach_source_flag"])
                if high_peer:
                    interaction_type = "high_peer_interaction"
                elif terminal:
                    interaction_type = "terminal_approach"
                else:
                    interaction_type = "ordinary_free_flight_low_peer"
                reduction = None if best is None else float(1.0 - float(best["J_preview"]) / max(float(selected["J_preview"]), 1.0e-300))
                event_output.append(
                    {
                        "block": block,
                        "scenario_id": scenario_id,
                        "stage": selected_summary["stage"],
                        "family": selected_summary["family"],
                        "agent_id": int(event["agent_id"]),
                        "step": int(event["step"]),
                        "interaction_type": interaction_type,
                        "static_obstacle_dominance": "UNAVAILABLE_UNTYPED_FP_CLEARANCE",
                        "dynamic_obstacle_dominance": "UNAVAILABLE_UNTYPED_FP_CLEARANCE",
                        "selected_candidate_id": int(selected_id),
                        "selected_gat_rank": int(selected["gat_rank"]),
                        "selected_gat_logit": float(selected["gat_logit"]),
                        "selected_J_preview": float(selected["J_preview"]),
                        "selected_preview_min_clearance_m": float(selected["preview_min_clearance"]),
                        "selected_candidate_risky": bool(selected["risky"]),
                        "admissible_lower_transient_alternative_count": len(eligible),
                        "replaceable": bool(eligible),
                        "replaceable_ge_20pct": bool(reduction_sets[0.20]),
                        "replaceable_ge_35pct": bool(reduction_sets[0.35]),
                        "replaceable_ge_50pct": bool(reduction_sets[0.50]),
                        "top3_replaceable_ge_20pct": any(
                            int(row["gat_rank"]) <= 3 for row in reduction_sets[0.20]
                        ),
                        "top3_replaceable_ge_35pct": any(
                            int(row["gat_rank"]) <= 3 for row in reduction_sets[0.35]
                        ),
                        "top3_replaceable_ge_50pct": any(
                            int(row["gat_rank"]) <= 3 for row in reduction_sets[0.50]
                        ),
                        "classification": "POTENTIALLY_AVOIDABLE_HIGH_TRANSIENT" if eligible else "NECESSARY_HIGH_TRANSIENT",
                        "best_alternative_candidate_id": None if best is None else int(best["candidate_id"]),
                        "best_alternative_gat_rank": None if best is None else int(best["gat_rank"]),
                        "best_alternative_gat_logit": None if best is None else float(best["gat_logit"]),
                        "first_ranked_eligible_candidate_id": None if first is None else int(first["candidate_id"]),
                        "first_ranked_eligible_gat_rank": None if first is None else int(first["gat_rank"]),
                        "best_achievable_transient_reduction_fraction": reduction,
                        "best_alternative_clearance_gap_m": None if best is None else float(best["preview_min_clearance"] - selected["preview_min_clearance"]),
                        "best_alternative_gat_logit_gap": None if best is None else float(selected["gat_logit"] - best["gat_logit"]),
                        "best_alternative_direction_difference_deg": None if best is None else float(np.degrees(np.arccos(np.clip(np.dot(np.asarray(selected["direction"]), np.asarray(best["direction"])), -1.0, 1.0)))),
                    }
                )

    by_block: dict[str, Any] = {}
    for block in BLOCKS:
        values = [row for row in event_output if row["block"] == block]
        replaceable_rows = [row for row in values if row["replaceable"]]
        reductions = [float(row["best_achievable_transient_reduction_fraction"]) for row in replaceable_rows]
        by_block[block] = {
            "high_transient_event_count": len(values),
            "replaceable_event_count": len(replaceable_rows),
            "replaceable_rate": float(len(replaceable_rows) / max(len(values), 1)),
            "necessary_fraction": float((len(values) - len(replaceable_rows)) / max(len(values), 1)),
            "potentially_avoidable_fraction": float(len(replaceable_rows) / max(len(values), 1)),
            "median_best_transient_reduction": None if not reductions else float(np.median(reductions)),
            "material_replaceable_rate": {
                f"minimum_reduction_{percent}pct": float(
                    np.mean([bool(row[f"replaceable_ge_{percent}pct"]) for row in values])
                )
                for percent in (20, 35, 50)
            },
            "top3_material_replaceable_rate": {
                f"minimum_reduction_{percent}pct": float(
                    np.mean([bool(row[f"top3_replaceable_ge_{percent}pct"]) for row in values])
                )
                for percent in (20, 35, 50)
            },
        }
    replicated_rate = min(by_block["development"]["replaceable_rate"], by_block["holdout"]["replaceable_rate"])
    label = "HIGH" if replicated_rate >= 0.35 else "MODERATE" if replicated_rate >= 0.15 else "LOW" if replicated_rate >= 0.05 else "NONE"
    aggregate = {
        "schema_version": "tacv_safe_alternative_replaceability_v1",
        "activation_threshold_source": "Development CLEAN-WINDOW P90 J_preview",
        "activation_threshold": activation_threshold,
        "blocks": by_block,
        "replicated_rate_for_gate": replicated_rate,
        "SAFE_ALTERNATIVE_HEADROOM": label,
        "TACV_REPLACEABILITY_GATE_PASS": label in {"HIGH", "MODERATE"},
        "material_reduction_diagnostics": "20%, 35%, and 50% are the user-bounded TACV Development settings; they are reported without changing the frozen strict-lower headroom gate",
        "static_dynamic_stratification": "UNAVAILABLE because frozen FP clearance is untyped; no labels fabricated",
    }
    return candidate_output, event_output, aggregate


def stopped_outputs(predictability_payload: Mapping[str, Any]) -> None:
    reason = "PREVIEW_TRANSIENT_PREDICTABILITY is WEAK/NONE; replaceability and TACV are not authorized"
    atomic_json(ROOT / "03_replaceability/TACV_SAFETY_ADMISSIBILITY_CONTRACT.json", safety_contract())
    write_csv(ROOT / "03_replaceability/SAFE_ALTERNATIVE_REPLACEABILITY.csv", [], fields=["status", "reason"])
    write_csv(ROOT / "03_replaceability/NECESSARY_VS_AVOIDABLE_TRANSIENTS.csv", [], fields=["status", "reason"])
    decision = {
        "schema_version": "tacv_gate_decision_v1",
        "PREVIEW_TRANSIENT_PREDICTABILITY": predictability_payload["PREVIEW_TRANSIENT_PREDICTABILITY"],
        "SAFE_ALTERNATIVE_HEADROOM": "NOT_RUN",
        "TACV_AUTHORIZED": "NO",
        "stop_reason": reason,
    }
    atomic_json(ROOT / "04_gate_decision/TACV_GATE_DECISION.json", decision)


def main() -> None:
    rows = selected_events()
    write_csv(ROOT / "02_predictability/PREVIEW_VS_REALIZED_TRANSIENT.csv", rows)
    payload, calibration = predictability(rows)
    atomic_json(ROOT / "02_predictability/PREVIEW_TRANSIENT_PREDICTABILITY.json", payload)
    write_csv(ROOT / "02_predictability/PREVIEW_RANK_CALIBRATION.csv", calibration)
    atomic_json(ROOT / "03_replaceability/TACV_SAFETY_ADMISSIBILITY_CONTRACT.json", safety_contract())
    if not payload["TACV_PREDICTABILITY_GATE_PASS"]:
        stopped_outputs(payload)
        print(json.dumps({"predictability": payload["PREVIEW_TRANSIENT_PREDICTABILITY"], "TACV_AUTHORIZED": "NO"}, indent=2))
        return
    candidate_rows, event_rows, replaceability_payload = replaceability(
        rows, float(payload["development_P90_preview_activation_threshold"])
    )
    write_csv(ROOT / "03_replaceability/SAFE_ALTERNATIVE_REPLACEABILITY.csv", candidate_rows)
    write_csv(ROOT / "03_replaceability/NECESSARY_VS_AVOIDABLE_TRANSIENTS.csv", event_rows)
    atomic_json(ROOT / "03_replaceability/SAFE_ALTERNATIVE_REPLACEABILITY.json", replaceability_payload)
    authorized = bool(replaceability_payload["TACV_REPLACEABILITY_GATE_PASS"])
    decision = {
        "schema_version": "tacv_gate_decision_v1",
        "PREVIEW_TRANSIENT_PREDICTABILITY": payload["PREVIEW_TRANSIENT_PREDICTABILITY"],
        "SAFE_ALTERNATIVE_HEADROOM": replaceability_payload["SAFE_ALTERNATIVE_HEADROOM"],
        "TACV_AUTHORIZED": "YES" if authorized else "NO",
        "development_activation_threshold": payload["development_P90_preview_activation_threshold"],
        "formal_v2_used_for_parameter_selection": False,
        "stop_reason": None if authorized else "safe-alternative headroom is LOW/NONE",
    }
    atomic_json(ROOT / "04_gate_decision/TACV_GATE_DECISION.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
