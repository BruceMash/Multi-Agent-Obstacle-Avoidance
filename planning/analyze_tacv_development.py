#!/usr/bin/env python3
"""Analyze TACV Development, select at most one arm, and freeze Holdout."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "artifacts/transient_aware_candidate_veto/20260824_184551"
CRT_ROOT = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552"
SOURCE_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
ORIGINAL_DEV = CRT_ROOT / "04_development/records/original/episode_records"
VARIANTS = ("tacv_mild", "tacv_medium", "tacv_strong")
FULL_HOLDOUT_MANIFEST = SOURCE_ROOT / "08_holdout/GAT_RS_HOLDOUT_SCENE_MANIFEST.json"
DIAGNOSTIC_HOLDOUT_MANIFEST = CRT_ROOT / "00_context/CRT_HOLDOUT_MANIFEST.json"
DEVELOPMENT_MANIFEST = CRT_ROOT / "00_context/CRT_DEVELOPMENT_MANIFEST.json"
DT = 0.1


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not finite else float(np.mean(finite))


def percent_change_reduction(original: float | None, variant: float | None) -> float | None:
    if original is None or variant is None or original == 0.0:
        return None
    return float(100.0 * (original - variant) / original)


def records(directory: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.stem.endswith("_SOFTWARE_ERROR"):
            continue
        payload = load_json(path)
        result[str(payload["entry_identity"]["scenario_id"])] = payload
    return result


def variant_records(variant: str) -> dict[str, dict[str, Any]]:
    return records(ROOT / f"06_development/records/{variant}/episode_records")


def event_jerk(accelerations: np.ndarray, agent_id: int, step: int) -> dict[str, Any]:
    acceleration = np.asarray(accelerations[:, int(agent_id), :], dtype=float)
    jerk = np.diff(acceleration, axis=0) / DT
    jerk_steps = np.arange(1, acceleration.shape[0], dtype=int)
    mask = (jerk_steps >= int(step)) & (jerk_steps <= int(step + 5))
    selected = jerk[mask]
    return {
        "realized_post_switch_0p5_peak_mps3": (
            None if not selected.size else float(np.max(np.linalg.norm(selected, axis=1)))
        ),
        "realized_post_switch_0p5_mean_squared_m2_s6": (
            None if not selected.size else float(np.mean(np.sum(selected**2, axis=1)))
        ),
        "realized_post_switch_vertical_peak_mps3": (
            None if not selected.size else float(np.max(np.abs(selected[:, 2])))
        ),
        "realized_post_switch_lateral_peak_mps3": (
            None if not selected.size else float(np.max(np.linalg.norm(selected[:, :2], axis=1)))
        ),
        "realized_sample_count": int(selected.shape[0]),
    }


def switch_rows(payloads: Mapping[str, Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scenario_id, payload in payloads.items():
        source_directory = (
            ORIGINAL_DEV
            if method == "original"
            else ROOT / f"06_development/records/{method}/episode_records"
        )
        path = source_directory / str(payload["trajectory_file"])
        with np.load(path, allow_pickle=False) as trajectory:
            acceleration = np.asarray(trajectory["applied_accelerations"], dtype=float)
        for event in payload["events"]:
            if event.get("event") == "INITIAL_SELECTION":
                continue
            if not bool(event.get("goal_changed", False)):
                continue
            if event.get("selected_candidate_id") is None or bool(event.get("selected_null", False)):
                continue
            output.append(
                {
                    "method": method,
                    "scenario_id": scenario_id,
                    "stage": payload["entry_identity"]["stage"],
                    "family": payload["entry_identity"]["family"],
                    "agent_id": int(event["agent_id"]),
                    "step": int(event["step"]),
                    **event_jerk(acceleration, int(event["agent_id"]), int(event["step"])),
                }
            )
    return output


def method_summary(method: str, payloads: Mapping[str, Mapping[str, Any]], scope: str) -> dict[str, Any]:
    selected = [
        payload
        for payload in payloads.values()
        if scope == "overall" or str(payload["entry_identity"]["stage"]) == scope
    ]
    episodes = [payload["episode"] for payload in selected]
    return {
        "scope": scope,
        "method": method,
        "episode_count": len(episodes),
        "team_success_count": sum(bool(row["team_success"]) for row in episodes),
        "team_success_rate": float(np.mean([bool(row["team_success"]) for row in episodes])),
        "collision_rate": float(np.mean([bool(row["collision"]) for row in episodes])),
        "static_collision_rate": float(np.mean([bool(row.get("static_obstacle_collision", False)) for row in episodes])),
        "dynamic_collision_rate": float(np.mean([bool(row.get("dynamic_obstacle_collision", False)) for row in episodes])),
        "peer_collision_rate": float(np.mean([bool(row["inter_agent_collision"]) for row in episodes])),
        "timeout_rate": float(np.mean([bool(row["timeout"]) for row in episodes])),
        "agent_completion_rate": mean(row["agent_completion_rate"] for row in episodes),
        "trajectory_smoothness_mean_all": mean(row["trajectory_smoothness"] for row in episodes),
        "vertical_jerk_mean_squared_all": mean(row["vertical_jerk_mean_squared_m2_s6"] for row in episodes),
        "lateral_jerk_mean_squared_all": mean(row["lateral_jerk_mean_squared_m2_s6"] for row in episodes),
        "completion_time_success_mean_s": mean(row["completion_time_s"] for row in episodes if row["team_success"]),
        "team_path_length_success_mean_m": mean(row["team_path_length_m"] for row in episodes if row["team_success"]),
        "path_efficiency_success_mean": mean(row["path_efficiency_mean"] for row in episodes if row["team_success"]),
        "minimum_obstacle_clearance_mean_m": mean(row["minimum_obstacle_signed_clearance_rechecked_m"] for row in episodes),
        "minimum_peer_distance_mean_m": mean(row["minimum_inter_agent_distance_m"] for row in episodes),
        "online_compute_mean_ms": mean(
            row.get("total_online_algorithm_compute_plus_tacv_ms", row["total_online_algorithm_compute_ms"])
            for row in episodes
        ),
        "tacv_activation_rate": (
            None if method == "original" else float(
                sum(int(row["tacv_activation_count"]) for row in episodes)
                / max(sum(int(row["tacv_candidate_decision_count"]) for row in episodes), 1)
            )
        ),
        "tacv_replacement_rate": (
            None if method == "original" else float(
                sum(int(row["tacv_replacement_count"]) for row in episodes)
                / max(sum(int(row["tacv_candidate_decision_count"]) for row in episodes), 1)
            )
        ),
        "gat_effective_top1_retention_rate": (
            None if method == "original" else float(
                1.0
                - sum(int(row["tacv_replacement_count"]) for row in episodes)
                / max(sum(int(row["tacv_candidate_decision_count"]) for row in episodes), 1)
            )
        ),
        "mean_replacement_gat_rank": (
            None if method == "original" else mean(row.get("tacv_mean_replacement_gat_rank") for row in episodes)
        ),
        "mean_predicted_transient_reduction": (
            None if method == "original" else mean(row.get("tacv_mean_predicted_relative_reduction") for row in episodes)
        ),
    }


def exact_mcnemar(success_original: Sequence[bool], success_variant: Sequence[bool]) -> dict[str, Any]:
    original_only = sum(bool(a) and not bool(b) for a, b in zip(success_original, success_variant))
    variant_only = sum(not bool(a) and bool(b) for a, b in zip(success_original, success_variant))
    discordant = original_only + variant_only
    if not discordant:
        p_value = 1.0
    else:
        lower = min(original_only, variant_only)
        tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "original_only_success": original_only,
        "tacv_only_success": variant_only,
        "exact_two_sided_p": float(p_value),
    }


def paired_summary(original: Mapping[str, Mapping[str, Any]], variant: Mapping[str, Mapping[str, Any]], name: str, switches: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ids = sorted(original)
    if ids != sorted(variant):
        raise RuntimeError(f"{name} is not paired with Original")
    original_episode = [original[sid]["episode"] for sid in ids]
    variant_episode = [variant[sid]["episode"] for sid in ids]
    both = [index for index, (a, b) in enumerate(zip(original_episode, variant_episode)) if a["team_success"] and b["team_success"]]

    def paired_means(key: str) -> tuple[float | None, float | None]:
        return (
            mean(original_episode[index].get(key) for index in both),
            mean(variant_episode[index].get(key) for index in both),
        )

    smooth_original, smooth_variant = paired_means("trajectory_smoothness")
    vertical_original, vertical_variant = paired_means("vertical_jerk_mean_squared_m2_s6")
    lateral_original, lateral_variant = paired_means("lateral_jerk_mean_squared_m2_s6")
    completion_original, completion_variant = paired_means("completion_time_s")
    path_original, path_variant = paired_means("team_path_length_m")
    efficiency_original, efficiency_variant = paired_means("path_efficiency_mean")
    clearance_original, clearance_variant = paired_means("minimum_obstacle_signed_clearance_rechecked_m")
    peer_distance_original, peer_distance_variant = paired_means("minimum_inter_agent_distance_m")
    compute_original = mean(row["total_online_algorithm_compute_ms"] for row in original_episode)
    compute_variant = mean(row["total_online_algorithm_compute_plus_tacv_ms"] for row in variant_episode)
    switch_original = mean(row["realized_post_switch_0p5_peak_mps3"] for row in switches["original"])
    switch_variant = mean(row["realized_post_switch_0p5_peak_mps3"] for row in switches[name])
    peer_original_rate = float(np.mean([bool(row["inter_agent_collision"]) for row in original_episode]))
    peer_variant_rate = float(np.mean([bool(row["inter_agent_collision"]) for row in variant_episode]))
    success_original_rate = float(np.mean([bool(row["team_success"]) for row in original_episode]))
    success_variant_rate = float(np.mean([bool(row["team_success"]) for row in variant_episode]))
    candidate_decision_count = sum(int(row["tacv_candidate_decision_count"]) for row in variant_episode)
    activation_count = sum(int(row["tacv_activation_count"]) for row in variant_episode)
    replacement_count = sum(int(row["tacv_replacement_count"]) for row in variant_episode)
    replacement_weight = [int(row["tacv_replacement_count"]) for row in variant_episode]
    mean_predicted_reduction = (
        None
        if not replacement_count
        else float(
            sum(
                int(row["tacv_replacement_count"])
                * float(row["tacv_mean_predicted_relative_reduction"])
                for row in variant_episode
                if row.get("tacv_mean_predicted_relative_reduction") is not None
            )
            / replacement_count
        )
    )
    peer_interaction_replacement_fraction = (
        None
        if not replacement_count
        else float(
            sum(
                weight * float(row["tacv_peer_interaction_replacement_fraction"])
                for weight, row in zip(replacement_weight, variant_episode)
                if weight and row.get("tacv_peer_interaction_replacement_fraction") is not None
            )
            / replacement_count
        )
    )
    failed_episode_replacement_count = sum(
        int(row["tacv_replacement_count"])
        for row in variant_episode
        if not bool(row["team_success"])
    )
    original_success_tacv_failure_count = sum(
        bool(a["team_success"]) and not bool(b["team_success"])
        for a, b in zip(original_episode, variant_episode)
    )
    summary = {
        "variant": name,
        "episode_count": len(ids),
        "both_success_count": len(both),
        "original_success_rate": success_original_rate,
        "tacv_success_rate": success_variant_rate,
        "success_delta_pp": 100.0 * (success_variant_rate - success_original_rate),
        "original_peer_collision_rate": peer_original_rate,
        "tacv_peer_collision_rate": peer_variant_rate,
        "peer_collision_delta_pp": 100.0 * (peer_variant_rate - peer_original_rate),
        **exact_mcnemar(
            [bool(row["team_success"]) for row in original_episode],
            [bool(row["team_success"]) for row in variant_episode],
        ),
        "both_success_original_smoothness": smooth_original,
        "both_success_tacv_smoothness": smooth_variant,
        "smoothness_reduction_percent": percent_change_reduction(smooth_original, smooth_variant),
        "vertical_jerk_reduction_percent": percent_change_reduction(vertical_original, vertical_variant),
        "lateral_jerk_reduction_percent": percent_change_reduction(lateral_original, lateral_variant),
        "switch_jerk_peak_original_mean_mps3": switch_original,
        "switch_jerk_peak_tacv_mean_mps3": switch_variant,
        "switch_jerk_peak_reduction_percent": percent_change_reduction(switch_original, switch_variant),
        "both_success_completion_time_delta_s": None if completion_original is None or completion_variant is None else completion_variant - completion_original,
        "both_success_team_path_delta_m": None if path_original is None or path_variant is None else path_variant - path_original,
        "both_success_path_efficiency_delta": None if efficiency_original is None or efficiency_variant is None else efficiency_variant - efficiency_original,
        "both_success_obstacle_clearance_delta_m": None if clearance_original is None or clearance_variant is None else clearance_variant - clearance_original,
        "both_success_peer_distance_delta_m": None if peer_distance_original is None or peer_distance_variant is None else peer_distance_variant - peer_distance_original,
        "online_compute_original_mean_ms": compute_original,
        "online_compute_tacv_mean_ms": compute_variant,
        "online_compute_delta_ms": None if compute_original is None or compute_variant is None else compute_variant - compute_original,
        "tacv_candidate_decision_count": candidate_decision_count,
        "tacv_activation_count": activation_count,
        "tacv_replacement_count": replacement_count,
        "tacv_activation_rate": float(activation_count / max(candidate_decision_count, 1)),
        "tacv_replacement_rate": float(replacement_count / max(candidate_decision_count, 1)),
        "gat_effective_top1_retention_rate": float(1.0 - replacement_count / max(candidate_decision_count, 1)),
        "mean_predicted_transient_reduction": mean_predicted_reduction,
        "peer_interaction_replacement_fraction": peer_interaction_replacement_fraction,
        "replacement_fraction_in_failed_episodes": float(failed_episode_replacement_count / max(replacement_count, 1)),
        "original_success_tacv_failure_count": original_success_tacv_failure_count,
        "original_success_tacv_failure_fraction": float(original_success_tacv_failure_count / len(ids)),
    }
    summary["development_gate_pass"] = bool(
        summary["success_delta_pp"] >= -1.0 - 1.0e-12
        and summary["peer_collision_delta_pp"] <= 1.0 + 1.0e-12
        and summary["smoothness_reduction_percent"] is not None
        and summary["smoothness_reduction_percent"] >= 5.0
        and summary["switch_jerk_peak_reduction_percent"] is not None
        and summary["switch_jerk_peak_reduction_percent"] >= 5.0
    )
    return summary


def aggregate_tacv_events(variant: str, payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_id, payload in payloads.items():
        path = ROOT / f"06_development/records/{variant}/tacv_event_records/{scenario_id}.json"
        record = load_json(path)
        trajectory_path = ROOT / f"06_development/records/{variant}/episode_records/{payload['trajectory_file']}"
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            acceleration = np.asarray(trajectory["applied_accelerations"], dtype=float)
        for row in record["event_rows"]:
            if row["event_type"] == "INITIAL_SELECTION":
                continue
            realized = event_jerk(acceleration, int(row["agent_id"]), int(row["step"]))
            rows.append(
                {
                    "variant": variant,
                    "stage": payload["entry_identity"]["stage"],
                    "family": payload["entry_identity"]["family"],
                    "episode_team_success": bool(payload["episode"]["team_success"]),
                    "episode_collision": bool(payload["episode"]["collision"]),
                    "episode_peer_collision": bool(payload["episode"]["inter_agent_collision"]),
                    **row,
                    **realized,
                }
            )
    return rows


def failure_audit(original: Mapping[str, Mapping[str, Any]], variants: Mapping[str, Mapping[str, Mapping[str, Any]]], tacv_events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in tacv_events:
        if row["replaced"]:
            events_by_key[(str(row["variant"]), str(row["scenario_id"]))].append(row)
    rows: list[dict[str, Any]] = []
    for variant, payloads in variants.items():
        for scenario_id in sorted(original):
            source = original[scenario_id]["episode"]
            target = payloads[scenario_id]["episode"]
            if not source["team_success"] or target["team_success"]:
                continue
            replacements = events_by_key.get((variant, scenario_id), [])
            failure_time = float(target["steps"] * target["dt"])
            if not replacements:
                rows.append(
                    {
                        "variant": variant,
                        "scenario_id": scenario_id,
                        "stage": target["stage"],
                        "family": target["family"],
                        "termination_reason": target["termination_reason"],
                        "collision": target["collision"],
                        "peer_collision": target["inter_agent_collision"],
                        "replacement_event_present": False,
                        "peer_exact_state": "UNAVAILABLE_NOT_RETAINED",
                    }
                )
                continue
            for event in replacements:
                rows.append(
                    {
                        "variant": variant,
                        "scenario_id": scenario_id,
                        "stage": target["stage"],
                        "family": target["family"],
                        "termination_reason": target["termination_reason"],
                        "collision": target["collision"],
                        "static_collision": target.get("static_obstacle_collision"),
                        "dynamic_collision": target.get("dynamic_obstacle_collision"),
                        "peer_collision": target["inter_agent_collision"],
                        "replacement_event_present": True,
                        "agent_id": event["agent_id"],
                        "veto_step": event["step"],
                        "veto_time_s": event["time_s"],
                        "time_from_veto_to_failure_s": failure_time - float(event["time_s"]),
                        "original_candidate_id": event["original_selected_candidate_id"],
                        "replacement_candidate_id": event["effective_selected_candidate_id"],
                        "original_gat_rank": event["original_gat_candidate_rank"],
                        "replacement_gat_rank": event["replacement_gat_candidate_rank"],
                        "original_J_preview": event["original_J_preview"],
                        "replacement_J_preview": event["replacement_J_preview"],
                        "predicted_relative_reduction": event["predicted_relative_reduction"],
                        "original_preview_min_clearance_m": event["original_preview_min_clearance_m"],
                        "replacement_preview_min_clearance_m": event["replacement_preview_min_clearance_m"],
                        "original_candidate_risky": event["original_candidate_risky"],
                        "replacement_candidate_risky": event["replacement_candidate_risky"],
                        "original_minimum_predicted_separation_m": event["original_minimum_predicted_separation_m"],
                        "replacement_minimum_predicted_separation_m": event["replacement_minimum_predicted_separation_m"],
                        "peer_exact_state": "UNAVAILABLE_NOT_RETAINED",
                    }
                )
    return rows


def freeze_holdout(selected_variant: str, selection: Mapping[str, Any]) -> None:
    full = load_json(FULL_HOLDOUT_MANIFEST)
    diagnostic = load_json(DIAGNOSTIC_HOLDOUT_MANIFEST)
    diagnostic_ids = {str(row["scenario_id"]) for row in diagnostic["entries"]}
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in full["entries"]:
        if str(row["scenario_id"]) not in diagnostic_ids:
            groups[(str(row["stage"]), str(row["family"]))].append(row)
    chosen: list[Mapping[str, Any]] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda row: str(row["scenario_id"]))
        if len(members) < 5:
            raise RuntimeError(f"insufficient unused Holdout scenes in {key}")
        chosen.extend(members[:5])
    if len(chosen) != 100:
        raise RuntimeError(f"sealed TACV Holdout must contain 100 scenes, found {len(chosen)}")
    dev = load_json(DEVELOPMENT_MANIFEST)
    seen_ids = diagnostic_ids | {str(row["scenario_id"]) for row in dev["entries"]}
    if any(str(row["scenario_id"]) in seen_ids for row in chosen):
        raise RuntimeError("sealed Holdout scenario id overlap")
    for key in ("geometry_fingerprint", "translation_invariant_fingerprint"):
        previous = {
            str(row[key])
            for row in [*diagnostic["entries"], *dev["entries"]]
            if row.get(key) is not None
        }
        if any(str(row[key]) in previous for row in chosen if row.get(key) is not None):
            raise RuntimeError(f"sealed Holdout {key} overlap")
    manifest = {
        "schema_version": "tacv_sealed_holdout_manifest_v1",
        "source_manifest": str(FULL_HOLDOUT_MANIFEST.relative_to(REPO_ROOT)),
        "selection_rule": "first five lexicographic unused scenario ids per stage-family cell; no performance read",
        "excluded_diagnostic_holdout_scenario_count": len(diagnostic_ids),
        "entry_count": len(chosen),
        "entries": chosen,
    }
    atomic_json(ROOT / "11_freeze/TACV_HOLDOUT_MANIFEST.json", manifest)
    freeze = {
        "schema_version": "final_tacv_freeze_v1",
        "status": "FROZEN_BEFORE_HOLDOUT",
        "selected_variant": selected_variant,
        "selected_configuration": load_json(ROOT / "05_tacv_implementation/TACV_IMPLEMENTATION_CONTRACT.json")["variants"][selected_variant],
        "development_selection_sha256": sha256_file(ROOT / "06_development/TACV_DEVELOPMENT_SELECTION.json"),
        "holdout_manifest_sha256": sha256_file(ROOT / "11_freeze/TACV_HOLDOUT_MANIFEST.json"),
        "holdout_scenario_count": 100,
        "source_sha256": {
            relative: sha256_file(REPO_ROOT / relative)
            for relative in (
                "planning/transient_aware_candidate_veto.py",
                "Multi-agent_Algo_lib/scripts/run_tacv_development.py",
            )
        },
        "formal_v2_authorized": False,
        "development_result_snapshot": dict(selection),
    }
    atomic_json(ROOT / "11_freeze/FINAL_TACV_FREEZE.json", freeze)


def main() -> None:
    original = records(ORIGINAL_DEV)
    variants = {variant: variant_records(variant) for variant in VARIANTS}
    if len(original) != 100 or any(len(values) != 100 for values in variants.values()):
        raise RuntimeError(
            f"incomplete Development: original={len(original)}, "
            + ", ".join(f"{name}={len(values)}" for name, values in variants.items())
        )
    switches = {"original": switch_rows(original, "original")}
    switches.update({name: switch_rows(values, name) for name, values in variants.items()})
    tacv_events = [row for name, values in variants.items() for row in aggregate_tacv_events(name, values)]
    write_csv(ROOT / "06_development/TACV_EVENT_LEVEL_DECISIONS.csv", tacv_events)

    result_rows: list[dict[str, Any]] = []
    for scope in ("overall", "stage_1", "stage_2", "stage_3", "stage_4"):
        result_rows.append(method_summary("original", original, scope))
        for name, values in variants.items():
            result_rows.append(method_summary(name, values, scope))
    write_csv(ROOT / "06_development/TACV_DEVELOPMENT_RESULTS.csv", result_rows)
    paired = [paired_summary(original, values, name, switches) for name, values in variants.items()]
    write_csv(ROOT / "06_development/TACV_DEVELOPMENT_PAIRED_SUMMARY.csv", paired)
    failures = failure_audit(original, variants, tacv_events)
    write_csv(ROOT / "07_failure_audit/TACV_PAIRED_FAILURE_AUDIT.csv", failures)

    eligible = [row for row in paired if row["development_gate_pass"]]
    if eligible:
        selected = sorted(
            eligible,
            key=lambda row: (
                -float(row["tacv_success_rate"]),
                float(row["tacv_peer_collision_rate"]),
                -float(row["smoothness_reduction_percent"]),
                -float(row["switch_jerk_peak_reduction_percent"]),
                float(row["both_success_completion_time_delta_s"] or 0.0),
                float(row["online_compute_tacv_mean_ms"] or float("inf")),
            ),
        )[0]
        selected_variant = str(selected["variant"])
        holdout_authorized = True
        reason = "at least one arm passed the frozen reliability/smoothness gate"
    else:
        selected = None
        selected_variant = "NONE"
        holdout_authorized = False
        reason = "no arm passed the frozen Development gate"
    selection = {
        "schema_version": "tacv_development_selection_v1",
        "selected_variant": selected_variant,
        "holdout_authorized": holdout_authorized,
        "selection_reason": reason,
        "selection_priority": [
            "team reliability",
            "peer-collision behavior",
            "smoothness reduction",
            "switch-transient reduction",
            "completion time",
            "compute",
        ],
        "paired_results": paired,
        "selected_result": selected,
        "formal_v2_run": False,
    }
    atomic_json(ROOT / "06_development/TACV_DEVELOPMENT_SELECTION.json", selection)
    if holdout_authorized:
        freeze_holdout(selected_variant, selected)
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
