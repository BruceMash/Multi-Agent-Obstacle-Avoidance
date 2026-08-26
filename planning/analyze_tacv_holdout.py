#!/usr/bin/env python3
"""Analyze the single frozen TACV sealed Holdout and stop before Formal V2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analyze_tacv_development import (
    DT,
    ROOT,
    atomic_json,
    event_jerk,
    exact_mcnemar,
    load_json,
    mean,
    method_summary,
    percent_change_reduction,
    records,
    write_csv,
)


SOURCE_ROOT = ROOT.parents[1] / "gat_recurrent_smoothness_rescue/20260821_180244"
ORIGINAL_HOLDOUT = (
    SOURCE_ROOT
    / "13_objective_revision/08_holdout/GAT_R_FP_ANCHOR_HOLDOUT400/episode_records"
)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260824


def switch_rows(
    payloads: Mapping[str, Mapping[str, Any]], directory: Path, method: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scenario_id, payload in payloads.items():
        with np.load(directory / str(payload["trajectory_file"]), allow_pickle=False) as trajectory:
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
                    "agent_id": int(event["agent_id"]),
                    "step": int(event["step"]),
                    **event_jerk(acceleration, int(event["agent_id"]), int(event["step"])),
                }
            )
    return output


def paired_binary(
    original: Sequence[bool], variant: Sequence[bool], *, event_name: str
) -> dict[str, Any]:
    first_only = sum(bool(a) and not bool(b) for a, b in zip(original, variant))
    second_only = sum(not bool(a) and bool(b) for a, b in zip(original, variant))
    result = exact_mcnemar(original, variant)
    return {
        "event": event_name,
        "original_only": first_only,
        "tacv_only": second_only,
        "exact_two_sided_p": result["exact_two_sided_p"],
    }


def paired_bootstrap(
    differences: Sequence[float], name: str, seed_offset: int
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    if not values.size:
        return {"metric": name, "pair_count": 0, "mean_difference": None, "ci95": None}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    sampled = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    estimates = np.mean(values[sampled], axis=1)
    return {
        "metric": name,
        "pair_count": int(values.size),
        "mean_difference_tacv_minus_original": float(np.mean(values)),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED + seed_offset,
    }


def aggregate_events(
    variant: str, payloads: Mapping[str, Mapping[str, Any]], directory: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_id, payload in payloads.items():
        event_payload = load_json(directory.parent / "tacv_event_records" / f"{scenario_id}.json")
        with np.load(directory / str(payload["trajectory_file"]), allow_pickle=False) as trajectory:
            acceleration = np.asarray(trajectory["applied_accelerations"], dtype=float)
        for row in event_payload["event_rows"]:
            if row["event_type"] == "INITIAL_SELECTION":
                continue
            rows.append(
                {
                    "variant": variant,
                    "stage": payload["entry_identity"]["stage"],
                    "family": payload["entry_identity"]["family"],
                    "episode_team_success": bool(payload["episode"]["team_success"]),
                    "episode_collision": bool(payload["episode"]["collision"]),
                    "episode_peer_collision": bool(payload["episode"]["inter_agent_collision"]),
                    **row,
                    **event_jerk(acceleration, int(row["agent_id"]), int(row["step"])),
                }
            )
    return rows


def main() -> None:
    freeze = load_json(ROOT / "11_freeze/FINAL_TACV_FREEZE.json")
    variant = str(freeze["selected_variant"])
    manifest = load_json(ROOT / "11_freeze/TACV_HOLDOUT_MANIFEST.json")
    scenario_ids = {str(row["scenario_id"]) for row in manifest["entries"]}
    original_all = records(ORIGINAL_HOLDOUT)
    original = {scenario_id: original_all[scenario_id] for scenario_id in scenario_ids}
    tacv_directory = ROOT / f"08_holdout/records/{variant}/episode_records"
    tacv = records(tacv_directory)
    if len(original) != len(tacv) or len(tacv) != 100 or set(original) != set(tacv):
        raise RuntimeError(
            f"sealed Holdout is incomplete or unpaired: original={len(original)}, tacv={len(tacv)}"
        )
    result_rows: list[dict[str, Any]] = []
    for scope in ("overall", "stage_1", "stage_2", "stage_3", "stage_4"):
        result_rows.append(method_summary("original", original, scope))
        result_rows.append(method_summary(variant, tacv, scope))
    write_csv(ROOT / "08_holdout/TACV_HOLDOUT_RESULTS.csv", result_rows)

    ids = sorted(original)
    original_episode = [original[sid]["episode"] for sid in ids]
    tacv_episode = [tacv[sid]["episode"] for sid in ids]
    both_ids = [
        sid
        for sid in ids
        if original[sid]["episode"]["team_success"] and tacv[sid]["episode"]["team_success"]
    ]
    original_switch = switch_rows(original, ORIGINAL_HOLDOUT, "original")
    tacv_switch = switch_rows(tacv, tacv_directory, variant)
    events = aggregate_events(variant, tacv, tacv_directory)
    write_csv(ROOT / "08_holdout/TACV_HOLDOUT_EVENT_LEVEL_DECISIONS.csv", events)

    binary = [
        paired_binary(
            [bool(row["team_success"]) for row in original_episode],
            [bool(row["team_success"]) for row in tacv_episode],
            event_name="team_success",
        ),
        paired_binary(
            [bool(row["collision"]) for row in original_episode],
            [bool(row["collision"]) for row in tacv_episode],
            event_name="any_collision",
        ),
        paired_binary(
            [bool(row["inter_agent_collision"]) for row in original_episode],
            [bool(row["inter_agent_collision"]) for row in tacv_episode],
            event_name="peer_collision",
        ),
    ]
    continuous_keys = (
        "trajectory_smoothness",
        "vertical_jerk_mean_squared_m2_s6",
        "lateral_jerk_mean_squared_m2_s6",
        "completion_time_s",
        "team_path_length_m",
        "path_efficiency_mean",
        "minimum_obstacle_signed_clearance_rechecked_m",
        "minimum_inter_agent_distance_m",
    )
    bootstrap: list[dict[str, Any]] = []
    for offset, key in enumerate(continuous_keys):
        bootstrap.append(
            paired_bootstrap(
                [float(tacv[sid]["episode"][key]) - float(original[sid]["episode"][key]) for sid in both_ids],
                key,
                offset,
            )
        )
    statistics = {
        "schema_version": "tacv_holdout_paired_statistics_v1",
        "variant": variant,
        "episode_count": 100,
        "both_success_count": len(both_ids),
        "binary_exact_mcnemar": binary,
        "continuous_both_success_paired_bootstrap": bootstrap,
        "failed_completion_times_filled_with_zero": False,
    }
    atomic_json(ROOT / "08_holdout/TACV_HOLDOUT_PAIRED_STATISTICS.json", statistics)

    success_original = float(np.mean([bool(row["team_success"]) for row in original_episode]))
    success_tacv = float(np.mean([bool(row["team_success"]) for row in tacv_episode]))
    peer_original = float(np.mean([bool(row["inter_agent_collision"]) for row in original_episode]))
    peer_tacv = float(np.mean([bool(row["inter_agent_collision"]) for row in tacv_episode]))
    smooth_original = mean(original[sid]["episode"]["trajectory_smoothness"] for sid in both_ids)
    smooth_tacv = mean(tacv[sid]["episode"]["trajectory_smoothness"] for sid in both_ids)
    switch_original = mean(row["realized_post_switch_0p5_peak_mps3"] for row in original_switch)
    switch_tacv = mean(row["realized_post_switch_0p5_peak_mps3"] for row in tacv_switch)
    replacements = [row for row in events if row["replaced"]]
    candidate_decisions = [row for row in events if row["original_selected_candidate_id"] is not None]
    runtime = {
        "schema_version": "tacv_runtime_summary_v1",
        "variant": variant,
        "original_online_compute_mean_ms": mean(row["total_online_algorithm_compute_ms"] for row in original_episode),
        "tacv_online_compute_mean_ms": mean(row["total_online_algorithm_compute_plus_tacv_ms"] for row in tacv_episode),
        "tacv_veto_only_mean_ms": mean(row["tacv_runtime_ms"] for row in tacv_episode),
        "tacv_veto_per_candidate_decision_ms": float(
            sum(float(row["tacv_runtime_ms"]) for row in tacv_episode)
            / max(sum(int(row["tacv_candidate_decision_count"]) for row in tacv_episode), 1)
        ),
        "additional_preview_rollout_count": 0,
    }
    atomic_json(ROOT / "09_runtime/TACV_RUNTIME_SUMMARY.json", runtime)

    holdout_result = {
        "variant": variant,
        "original_success": success_original,
        "tacv_success": success_tacv,
        "success_delta_pp": 100.0 * (success_tacv - success_original),
        "original_peer_collision": peer_original,
        "tacv_peer_collision": peer_tacv,
        "peer_collision_delta_pp": 100.0 * (peer_tacv - peer_original),
        "original_smoothness_both_success": smooth_original,
        "tacv_smoothness_both_success": smooth_tacv,
        "smoothness_reduction_percent": percent_change_reduction(smooth_original, smooth_tacv),
        "switch_jerk_peak_reduction_percent": percent_change_reduction(switch_original, switch_tacv),
        "replacement_rate": float(len(replacements) / max(len(candidate_decisions), 1)),
        "gat_effective_top1_retention": float(1.0 - len(replacements) / max(len(candidate_decisions), 1)),
        "mean_predicted_transient_reduction": mean(row["predicted_relative_reduction"] for row in replacements),
    }
    accepted = bool(
        holdout_result["smoothness_reduction_percent"] is not None
        and holdout_result["smoothness_reduction_percent"] > 0.0
        and holdout_result["switch_jerk_peak_reduction_percent"] is not None
        and holdout_result["switch_jerk_peak_reduction_percent"] > 0.0
        and holdout_result["success_delta_pp"] >= -1.0 - 1.0e-12
        and holdout_result["peer_collision_delta_pp"] <= 1.0 + 1.0e-12
    )
    development = load_json(ROOT / "06_development/TACV_DEVELOPMENT_SELECTION.json")
    go_no_go = {
        "schema_version": "tacv_formal_go_no_go_v1",
        "selected_variant": variant,
        "development": development["selected_result"],
        "holdout": holdout_result,
        "HOLDOUT_ACCEPTANCE": "PASS" if accepted else "FAIL",
        "FORMAL_REEVALUATION_RECOMMENDED": "YES" if accepted else "NO",
        "FORMAL_V2_EXECUTED": "NO",
        "ORIGINAL_FORMAL_SUCCESS": 0.9525,
        "FORMAL_RESULT_MODIFIED": "NO",
        "stop_before_formal": True,
    }
    atomic_json(ROOT / "TACV_FORMAL_GO_NO_GO.json", go_no_go)
    print(json.dumps(go_no_go, indent=2), flush=True)


if __name__ == "__main__":
    main()
