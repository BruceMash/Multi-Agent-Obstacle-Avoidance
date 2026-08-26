"""Analyze paired Proposed vs Proposed-CRT Development and Holdout blocks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552"
VARIANTS = ("original", "crt_0p2", "crt_0p3", "crt_0p5")
T_REF = {"original": None, "crt_0p2": 0.2, "crt_0p3": 0.3, "crt_0p5": 0.5}
DT = 0.1
RNG_SEED = 20260824

FREEZE_SOURCE_PATHS = (
    "planning/continuous_reference_transition.py",
    "planning/analyze_continuous_reference_transition.py",
    "planning/event_triggered_reference_reconstruction.py",
    "planning/goal_semantics_diagnosis.py",
    "Environment/frozen_sac_dmp_execution.py",
    "Environment/multi_agent_dmp_env.py",
    "Controller/dmp_rl.py",
    "Entity/KinematicModel.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
    "Multi-agent_Algo_lib/scripts/run_continuous_reference_transition.py",
    "test/test_continuous_reference_transition.py",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def record_dir(block: str, variant: str) -> Path:
    phase = "04_development" if block == "development" else "07_holdout"
    return ROOT / phase / "records" / variant / "episode_records"


def records(block: str, variant: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(record_dir(block, variant).glob("*.json")):
        if path.stem.endswith("_SOFTWARE_ERROR"):
            continue
        payload = load_json(path)
        payload["_record_path"] = path
        payload["_variant"] = variant
        result.append(payload)
    return result


def finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(
        [float(value) for value in values if value is not None], dtype=float
    )
    return array[np.isfinite(array)]


def mean(values: Iterable[Any]) -> float | None:
    array = finite(values)
    return float(np.mean(array)) if array.size else None


def percentile(values: Iterable[Any], q: float) -> float | None:
    array = finite(values)
    return float(np.percentile(array, q)) if array.size else None


def episode_value(record: Mapping[str, Any], name: str) -> Any:
    return record["episode"].get(name)


def summarize_scope(
    variant: str,
    scope: str,
    data: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    success = [row for row in data if bool(episode_value(row, "team_success"))]
    event_rows = [event for row in data for event in row["events"]]
    changed = [
        row
        for row in event_rows
        if bool(row.get("goal_changed")) and row.get("event") != "INITIAL_SELECTION"
    ]
    return {
        "block_scope": scope,
        "variant": variant,
        "T_ref_s": T_REF[variant],
        "episode_count": len(data),
        "team_success_count": len(success),
        "team_success_rate": mean(episode_value(row, "team_success") for row in data),
        "collision_count": sum(bool(episode_value(row, "collision")) for row in data),
        "collision_rate": mean(episode_value(row, "collision") for row in data),
        "static_collision_count": sum(bool(episode_value(row, "static_obstacle_collision")) for row in data),
        "static_collision_rate": mean(episode_value(row, "static_obstacle_collision") for row in data),
        "dynamic_collision_count": sum(bool(episode_value(row, "dynamic_obstacle_collision")) for row in data),
        "dynamic_collision_rate": mean(episode_value(row, "dynamic_obstacle_collision") for row in data),
        "peer_collision_count": sum(bool(episode_value(row, "inter_agent_collision")) for row in data),
        "peer_collision_rate": mean(episode_value(row, "inter_agent_collision") for row in data),
        "timeout_count": sum(bool(episode_value(row, "timeout")) for row in data),
        "timeout_rate": mean(episode_value(row, "timeout") for row in data),
        "agent_completion_rate": mean(episode_value(row, "agent_completion_rate") for row in data),
        "successful_smoothness_mean": mean(episode_value(row, "trajectory_smoothness") for row in success),
        "all_episode_smoothness_mean": mean(episode_value(row, "trajectory_smoothness") for row in data),
        "successful_vertical_jerk_mean_squared": mean(episode_value(row, "vertical_jerk_mean_squared_m2_s6") for row in success),
        "successful_lateral_jerk_mean_squared": mean(episode_value(row, "lateral_jerk_mean_squared_m2_s6") for row in success),
        "successful_completion_time_s": mean(episode_value(row, "completion_time_s") for row in success),
        "successful_team_path_length_m": mean(episode_value(row, "team_path_length_m") for row in success),
        "successful_path_efficiency": mean(episode_value(row, "path_efficiency_mean") for row in success),
        "minimum_obstacle_clearance_mean_m": mean(episode_value(row, "minimum_obstacle_signed_clearance_rechecked_m") for row in data),
        "minimum_peer_distance_mean_m": mean(episode_value(row, "minimum_inter_agent_distance_m") for row in data),
        "mean_reproposals_per_episode": mean(episode_value(row, "replanning_count") for row in data),
        "mean_actual_command_changes_per_episode": len(changed) / len(data) if data else None,
        "mean_safety_bandwidth_overrides_per_episode": mean(episode_value(row, "crt_bandwidth_override_count") for row in data),
        "mean_safety_instant_replacements_per_episode": mean(episode_value(row, "crt_instant_replacement_count") for row in data),
        "mean_online_compute_ms": mean(episode_value(row, "total_online_algorithm_compute_ms") for row in data),
        "mean_crt_runtime_ms": mean(episode_value(row, "crt_total_runtime_ms") for row in data),
        "mean_crt_filter_only_runtime_ms": mean(episode_value(row, "crt_filter_only_runtime_ms") for row in data),
        "mean_crt_safety_runtime_ms": mean(episode_value(row, "crt_safety_evaluation_runtime_ms") for row in data),
        "mean_command_execution_error_m": mean(episode_value(row, "crt_mean_command_execution_error_m") for row in data),
        "p95_command_execution_error_m": percentile(
            (episode_value(row, "crt_p95_command_execution_error_m") for row in data), 95
        ),
        "mean_reference_filter_velocity_mps": mean(episode_value(row, "crt_mean_filter_speed_mps") for row in data),
        "p95_reference_filter_velocity_mps": percentile(
            (episode_value(row, "crt_p95_filter_speed_mps") for row in data), 95
        ),
        "mean_settle_time_s": mean(episode_value(row, "crt_mean_settle_time_s") for row in data),
        "p95_settle_time_s": percentile(
            (episode_value(row, "crt_p95_settle_time_s") for row in data), 95
        ),
    }


def event_jerk_metrics(
    accelerations: np.ndarray,
    agent_id: int,
    step: int,
) -> dict[str, Any]:
    acceleration = np.asarray(accelerations[:, int(agent_id), :], dtype=float)
    jerk = np.diff(acceleration, axis=0) / DT
    jerk_steps = np.arange(1, acceleration.shape[0], dtype=int)

    def selected(first: int, last: int, *, inclusive_last: bool) -> np.ndarray:
        mask = jerk_steps >= int(first)
        mask &= jerk_steps <= int(last) if inclusive_last else jerk_steps < int(last)
        return jerk[mask]

    pre = selected(step - 5, step, inclusive_last=False)
    post_02 = selected(step, step + 2, inclusive_last=True)
    post_05 = selected(step, step + 5, inclusive_last=True)

    def peak_norm(value: np.ndarray) -> float | None:
        return float(np.max(np.linalg.norm(value, axis=1))) if value.size else None

    def peak_vertical(value: np.ndarray) -> float | None:
        return float(np.max(np.abs(value[:, 2]))) if value.size else None

    def peak_lateral(value: np.ndarray) -> float | None:
        return float(np.max(np.linalg.norm(value[:, :2], axis=1))) if value.size else None

    return {
        "pre_switch_0p5_jerk_peak_mps3": peak_norm(pre),
        "pre_switch_0p5_vertical_jerk_peak_mps3": peak_vertical(pre),
        "post_switch_0p2_jerk_peak_mps3": peak_norm(post_02),
        "post_switch_0p5_jerk_peak_mps3": peak_norm(post_05),
        "post_switch_0p5_vertical_jerk_peak_mps3": peak_vertical(post_05),
        "post_switch_0p5_lateral_jerk_peak_mps3": peak_lateral(post_05),
        "post_minus_pre_0p5_jerk_peak_mps3": (
            None if not pre.size or not post_05.size else peak_norm(post_05) - peak_norm(pre)
        ),
        "post_minus_pre_0p5_vertical_jerk_peak_mps3": (
            None if not pre.size or not post_05.size else peak_vertical(post_05) - peak_vertical(pre)
        ),
        "pre_sample_count": int(pre.shape[0]),
        "post_0p2_sample_count": int(post_02.shape[0]),
        "post_0p5_sample_count": int(post_05.shape[0]),
    }


def event_table(block: str, variants: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for record in records(block, variant):
            trajectory_path = Path(record["_record_path"]).with_name(
                record["trajectory_file"]
            )
            trajectory = np.load(trajectory_path)
            accelerations = np.asarray(trajectory["applied_accelerations"], dtype=float)
            sequence_by_agent: dict[int, int] = {}
            for event in record["events"]:
                if event.get("event") == "INITIAL_SELECTION" or not bool(event.get("goal_changed")):
                    continue
                agent = int(event["agent_id"])
                sequence_by_agent[agent] = sequence_by_agent.get(agent, 0) + 1
                old_goal = np.asarray(event["old_active_goal"], dtype=float)
                new_goal = np.asarray(event["new_active_goal"], dtype=float)
                rows.append(
                    {
                        "block": block,
                        "variant": variant,
                        "T_ref_s": T_REF[variant],
                        "scenario_id": record["summary"]["scenario_id"],
                        "stage": record["summary"]["stage"],
                        "family": record["summary"]["family"],
                        "agent_id": agent,
                        "agent_switch_sequence": sequence_by_agent[agent],
                        "step": int(event["step"]),
                        "time_s": float(event["step"]) * DT,
                        "event": event["event"],
                        "command_jump_m": float(np.linalg.norm(new_goal - old_goal)),
                        "crt_initial_tracking_error_m": event.get("crt_initial_tracking_error_m"),
                        "crt_settle_time_s": event.get("crt_settle_time_s"),
                        "crt_settle_censored_by_new_command": event.get("crt_settle_censored_by_new_command"),
                        "episode_team_success": bool(record["episode"]["team_success"]),
                        **event_jerk_metrics(accelerations, agent, int(event["step"])),
                    }
                )
    return rows


def event_summary(rows: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    data = [row for row in rows if row["variant"] == variant]
    return {
        "variant": variant,
        "event_count": len(data),
        "post_0p2_jerk_peak_mean": mean(row["post_switch_0p2_jerk_peak_mps3"] for row in data),
        "post_0p5_jerk_peak_mean": mean(row["post_switch_0p5_jerk_peak_mps3"] for row in data),
        "post_0p5_vertical_jerk_peak_mean": mean(row["post_switch_0p5_vertical_jerk_peak_mps3"] for row in data),
        "post_0p5_lateral_jerk_peak_mean": mean(row["post_switch_0p5_lateral_jerk_peak_mps3"] for row in data),
        "post_minus_pre_0p5_jerk_peak_mean": mean(row["post_minus_pre_0p5_jerk_peak_mps3"] for row in data),
        "post_minus_pre_0p5_vertical_jerk_peak_mean": mean(row["post_minus_pre_0p5_vertical_jerk_peak_mps3"] for row in data),
        "mean_settle_time_s": mean(row["crt_settle_time_s"] for row in data),
        "settled_event_count": sum(row["crt_settle_time_s"] is not None for row in data),
        "censored_event_count": sum(bool(row["crt_settle_censored_by_new_command"]) for row in data),
    }


def cluster_bootstrap_paired(
    pairs: Sequence[tuple[str, float, float]],
    *,
    replicates: int = 5000,
) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "mean_delta": None, "ci_low": None, "ci_high": None}
    values = np.asarray([[left, right] for _, left, right in pairs], dtype=float)
    delta = values[:, 1] - values[:, 0]
    rng = np.random.default_rng(RNG_SEED)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    boot = np.mean(delta[indices], axis=1)
    statistic = wilcoxon(delta).pvalue if np.any(delta != 0.0) else 1.0
    return {
        "n": len(values),
        "original_mean": float(np.mean(values[:, 0])),
        "crt_mean": float(np.mean(values[:, 1])),
        "mean_delta_crt_minus_original": float(np.mean(delta)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "wilcoxon_p": float(statistic),
    }


def paired_variant(
    original: Sequence[Mapping[str, Any]],
    variant_records: Sequence[Mapping[str, Any]],
    variant: str,
) -> dict[str, Any]:
    left = {row["summary"]["scenario_id"]: row for row in original}
    right = {row["summary"]["scenario_id"]: row for row in variant_records}
    if set(left) != set(right):
        raise RuntimeError(f"paired scenario mismatch for {variant}")
    keys = sorted(left)
    both_success = [
        key
        for key in keys
        if bool(left[key]["episode"]["team_success"])
        and bool(right[key]["episode"]["team_success"])
    ]
    original_only = sum(
        bool(left[key]["episode"]["team_success"])
        and not bool(right[key]["episode"]["team_success"])
        for key in keys
    )
    crt_only = sum(
        not bool(left[key]["episode"]["team_success"])
        and bool(right[key]["episode"]["team_success"])
        for key in keys
    )
    binary_p = (
        float(binomtest(min(original_only, crt_only), original_only + crt_only, 0.5).pvalue)
        if original_only + crt_only
        else 1.0
    )

    def continuous(name: str) -> dict[str, Any]:
        pairs = [
            (
                key,
                float(left[key]["episode"][name]),
                float(right[key]["episode"][name]),
            )
            for key in both_success
            if left[key]["episode"].get(name) is not None
            and right[key]["episode"].get(name) is not None
        ]
        return cluster_bootstrap_paired(pairs)

    return {
        "variant": variant,
        "pair_count": len(keys),
        "both_success": len(both_success),
        "original_only_success": int(original_only),
        "crt_only_success": int(crt_only),
        "both_failure": int(len(keys) - len(both_success) - original_only - crt_only),
        "mcnemar_exact_p": binary_p,
        "continuous_both_success": {
            name: continuous(name)
            for name in (
                "trajectory_smoothness",
                "vertical_jerk_mean_squared_m2_s6",
                "lateral_jerk_mean_squared_m2_s6",
                "completion_time_s",
                "team_path_length_m",
                "path_efficiency_mean",
                "minimum_obstacle_signed_clearance_rechecked_m",
                "minimum_inter_agent_distance_m",
                "total_online_algorithm_compute_ms",
            )
        },
    }


def failure_classification(original: Mapping[str, Any], crt: Mapping[str, Any]) -> dict[str, Any]:
    episode = crt["episode"]
    events = crt["events"]
    terminal_handoff = any(row.get("event") == "REFERENCE_HANDOFF" for row in events)
    mean_error = episode.get("crt_mean_command_execution_error_m") or 0.0
    bandwidth = int(episode.get("crt_bandwidth_override_count") or 0)
    instant = int(episode.get("crt_instant_replacement_count") or 0)
    if bool(episode.get("dynamic_obstacle_collision")):
        primary = "dynamic-obstacle interaction"
    elif bool(episode.get("inter_agent_collision")):
        primary = "peer interaction"
    elif bool(episode.get("static_obstacle_collision")) and bandwidth > 0:
        primary = "slow safety response"
    elif bool(episode.get("static_obstacle_collision")):
        primary = "static-obstacle interaction"
    elif bool(episode.get("timeout")) and mean_error > 0.05:
        primary = "goal lag"
    elif bool(episode.get("timeout")) and terminal_handoff:
        primary = "terminal-goal handoff"
    else:
        primary = "unrelated deterministic trajectory divergence"
    return {
        "scenario_id": crt["summary"]["scenario_id"],
        "stage": crt["summary"]["stage"],
        "family": crt["summary"]["family"],
        "variant": crt["_variant"],
        "original_success": bool(original["episode"]["team_success"]),
        "crt_success": bool(episode["team_success"]),
        "crt_collision": bool(episode["collision"]),
        "crt_static_collision": bool(episode.get("static_obstacle_collision")),
        "crt_dynamic_collision": bool(episode.get("dynamic_obstacle_collision")),
        "crt_peer_collision": bool(episode.get("inter_agent_collision")),
        "crt_timeout": bool(episode.get("timeout")),
        "slow_safety_response_flag": bool(bandwidth > 0),
        "goal_lag_flag": bool(mean_error > 0.05),
        "dynamic_obstacle_interaction_flag": bool(episode.get("dynamic_obstacle_collision")),
        "peer_interaction_flag": bool(episode.get("inter_agent_collision")),
        "altitude_transition_flag": bool(
            (episode.get("vertical_jerk_peak_mps3") or 0.0)
            > (episode.get("jerk_norm_peak_mps3") or float("inf")) * 0.7
        ),
        "terminal_goal_handoff_flag": terminal_handoff,
        "unrelated_stochastic_variation_flag": false_value(),
        "mean_command_execution_error_m": mean_error,
        "bandwidth_override_count": bandwidth,
        "instant_replacement_count": instant,
        "primary_diagnosis": primary,
        "inspection_basis": "deterministic paired trajectory, collision decomposition, command lag, event history",
    }


def false_value() -> bool:
    return False


def percent_reduction(original: float | None, changed: float | None) -> float | None:
    if original is None or changed is None or original == 0.0:
        return None
    return float(100.0 * (original - changed) / original)


def development() -> None:
    all_records = {variant: records("development", variant) for variant in VARIANTS}
    if any(len(value) != 100 for value in all_records.values()):
        raise RuntimeError(
            f"Development is incomplete: { {key: len(value) for key, value in all_records.items()} }"
        )
    result_rows: list[dict[str, Any]] = []
    for variant, data in all_records.items():
        result_rows.append(summarize_scope(variant, "overall", data))
        for stage in ("stage_1", "stage_2", "stage_3", "stage_4"):
            result_rows.append(
                summarize_scope(
                    variant,
                    stage,
                    [row for row in data if row["summary"]["stage"] == stage],
                )
            )
    event_rows = event_table("development", VARIANTS)
    event_summaries = {row["variant"]: row for row in map(lambda value: event_summary(event_rows, value), VARIANTS)}
    for row in result_rows:
        row.update(
            {
                f"event_{key}": value
                for key, value in event_summaries[row["variant"]].items()
                if key != "variant"
            }
        )
    write_csv(ROOT / "04_development/CRT_DEVELOPMENT_RESULTS.csv", result_rows)
    write_csv(ROOT / "05_event_alignment/CRT_EVENT_ALIGNED_JERK.csv", event_rows)
    write_csv(ROOT / "05_event_alignment/CRT_EVENT_ALIGNED_SUMMARY.csv", list(event_summaries.values()))

    paired = {
        variant: paired_variant(all_records["original"], all_records[variant], variant)
        for variant in VARIANTS[1:]
    }
    write_json(ROOT / "04_development/CRT_DEVELOPMENT_PAIRED_STATISTICS.json", paired)

    original_by_id = {
        row["summary"]["scenario_id"]: row for row in all_records["original"]
    }
    failures: list[dict[str, Any]] = []
    for variant in VARIANTS[1:]:
        for row in all_records[variant]:
            source = original_by_id[row["summary"]["scenario_id"]]
            if bool(source["episode"]["team_success"]) and not bool(row["episode"]["team_success"]):
                failures.append(failure_classification(source, row))
    write_csv(ROOT / "04_development/CRT_PAIRED_FAILURE_AUDIT.csv", failures)

    overall = {
        row["variant"]: row
        for row in result_rows
        if row["block_scope"] == "overall"
    }
    base = overall["original"]
    candidates: list[dict[str, Any]] = []
    for variant in VARIANTS[1:]:
        row = overall[variant]
        pair = paired[variant]
        continuous = pair["continuous_both_success"]
        smooth = continuous["trajectory_smoothness"]
        smooth_reduction = percent_reduction(smooth["original_mean"], smooth["crt_mean"])
        path = continuous["team_path_length_m"]
        path_reduction = percent_reduction(path["original_mean"], path["crt_mean"])
        vertical = continuous["vertical_jerk_mean_squared_m2_s6"]
        vertical_reduction = percent_reduction(
            vertical["original_mean"], vertical["crt_mean"]
        )
        lateral = continuous["lateral_jerk_mean_squared_m2_s6"]
        lateral_reduction = percent_reduction(
            lateral["original_mean"], lateral["crt_mean"]
        )
        event_reduction = percent_reduction(
            event_summaries["original"]["post_0p5_jerk_peak_mean"],
            event_summaries[variant]["post_0p5_jerk_peak_mean"],
        )
        vertical_event_reduction = percent_reduction(
            event_summaries["original"]["post_0p5_vertical_jerk_peak_mean"],
            event_summaries[variant]["post_0p5_vertical_jerk_peak_mean"],
        )
        success_delta_pp = 100.0 * (row["team_success_rate"] - base["team_success_rate"])
        collision_delta_pp = 100.0 * (row["collision_rate"] - base["collision_rate"])
        peer_delta_pp = 100.0 * (row["peer_collision_rate"] - base["peer_collision_rate"])
        dynamic_delta_pp = 100.0 * (row["dynamic_collision_rate"] - base["dynamic_collision_rate"])
        command_event_count = int(event_summaries[variant]["event_count"])
        instant_count = sum(int(episode_value(item, "crt_instant_replacement_count") or 0) for item in all_records[variant])
        bandwidth_count = sum(int(episode_value(item, "crt_bandwidth_override_count") or 0) for item in all_records[variant])
        preferred_reliability = success_delta_pp >= -1.0 - 1.0e-12
        exceptional_reliability = bool(
            success_delta_pp >= -2.0 - 1.0e-12
            and (smooth_reduction or -float("inf")) >= 25.0
            and peer_delta_pp <= 0.0
            and dynamic_delta_pp <= 0.0
        )
        override_regular = bool(
            command_event_count > 0
            and (
                instant_count / command_event_count > 0.01
                or bandwidth_count / command_event_count > 0.05
            )
        )
        no_systematic_collision = bool(
            collision_delta_pp <= 2.0 + 1.0e-12
            and peer_delta_pp <= 2.0 + 1.0e-12
            and dynamic_delta_pp <= 2.0 + 1.0e-12
        )
        clear_transient_reduction = bool(
            (smooth_reduction or -float("inf")) > 0.0
            and (event_reduction or -float("inf")) > 0.0
            and (vertical_event_reduction or -float("inf")) > 0.0
        )
        eligible = bool(
            (preferred_reliability or exceptional_reliability)
            and no_systematic_collision
            and clear_transient_reduction
            and not override_regular
        )
        candidates.append(
            {
                "variant": variant,
                "T_ref_s": T_REF[variant],
                "omega_rad_s": 5.83392170191739 / float(T_REF[variant]),
                "success_delta_pp": success_delta_pp,
                "collision_delta_pp": collision_delta_pp,
                "peer_collision_delta_pp": peer_delta_pp,
                "dynamic_collision_delta_pp": dynamic_delta_pp,
                "both_success_smoothness_reduction_percent": smooth_reduction,
                "both_success_path_length_reduction_percent": path_reduction,
                "both_success_path_length_delta_m": path["mean_delta_crt_minus_original"],
                "both_success_vertical_jerk_reduction_percent": vertical_reduction,
                "both_success_lateral_jerk_reduction_percent": lateral_reduction,
                "post_switch_0p5_jerk_reduction_percent": event_reduction,
                "post_switch_vertical_jerk_reduction_percent": vertical_event_reduction,
                "completion_time_delta_s_both_success": continuous["completion_time_s"]["mean_delta_crt_minus_original"],
                "compute_delta_ms_both_success": continuous["total_online_algorithm_compute_ms"]["mean_delta_crt_minus_original"],
                "bandwidth_override_count": bandwidth_count,
                "instant_replacement_count": instant_count,
                "override_regular": override_regular,
                "preferred_reliability_envelope": preferred_reliability,
                "exceptional_two_pp_envelope": exceptional_reliability,
                "no_new_systematic_collision_mode": no_systematic_collision,
                "clear_transient_reduction": clear_transient_reduction,
                "eligible": eligible,
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    selected = (
        sorted(
            eligible,
            key=lambda row: (
                row["success_delta_pp"],
                row["both_success_smoothness_reduction_percent"],
                row["post_switch_0p5_jerk_reduction_percent"],
                -row["completion_time_delta_s_both_success"],
                -row["compute_delta_ms_both_success"],
            ),
            reverse=True,
        )[0]
        if eligible
        else None
    )
    selection = {
        "schema_version": "crt_development_selection_v1",
        "status": "SELECTED" if selected is not None else "NO_ACCEPTABLE_CRT",
        "selection_priority": [
            "reliability",
            "jerk/smoothness reduction",
            "switch-aligned transient reduction",
            "completion time",
            "computation",
        ],
        "screening_envelope": {
            "preferred_success_delta_pp_min": -1.0,
            "exceptional_success_delta_pp_min": -2.0,
            "exceptional_smoothness_reduction_percent_min": 25.0,
        },
        "candidates": candidates,
        "selected_variant": None if selected is None else selected["variant"],
        "selected_T_ref_s": None if selected is None else selected["T_ref_s"],
        "selected_omega_rad_s": None if selected is None else selected["omega_rad_s"],
        "reason": (
            "No CRT setting passed the preregistered reliability and transient-reduction screen."
            if selected is None
            else "Selected by lexicographic reliability-first priority among eligible CRT settings."
        ),
        "holdout_authorized": selected is not None,
        "formal_v2_authorized": false_value(),
    }
    write_json(ROOT / "06_selection/CRT_DEVELOPMENT_SELECTION.json", selection)

    pre = load_json(ROOT / "00_context/CRT_PREDEVELOPMENT_FREEZE.json")
    freeze = {
        "schema_version": "final_crt_freeze_v1",
        "status": "FROZEN_BEFORE_HOLDOUT" if selected is not None else "NO_VARIANT_TO_FREEZE",
        "selected_variant": selection["selected_variant"],
        "selected_T_ref_s": selection["selected_T_ref_s"],
        "selected_omega_rad_s": selection["selected_omega_rad_s"],
        "transition_type": "SECOND_ORDER_CRITICALLY_DAMPED",
        "integrator": "EXACT_DISCRETE",
        "dt_s": 0.1,
        "safety_hard_margin_m": 0.0,
        "safety_bandwidth_multiplier": 2.0,
        "source_sha256": {
            path: sha256_file(REPO_ROOT / path) for path in FREEZE_SOURCE_PATHS
        },
        "checkpoint_sha256": pre["checkpoint_sha256"],
        "development_manifest_sha256": sha256_file(
            ROOT / "00_context/CRT_DEVELOPMENT_MANIFEST.json"
        ),
        "holdout_manifest_sha256": sha256_file(
            ROOT / "00_context/CRT_HOLDOUT_MANIFEST.json"
        ),
        "development_selection_sha256": sha256_file(
            ROOT / "06_selection/CRT_DEVELOPMENT_SELECTION.json"
        ),
        "holdout_opened": false_value(),
        "post_holdout_tuning_authorized": false_value(),
        "formal_v2_execution_authorized": false_value(),
    }
    write_json(ROOT / "10_freeze/FINAL_CRT_FREEZE.json", freeze)


def holdout() -> None:
    selection = load_json(ROOT / "06_selection/CRT_DEVELOPMENT_SELECTION.json")
    selected = selection["selected_variant"]
    if selected is None:
        raise RuntimeError("no CRT variant was selected for Holdout")
    original = records("holdout", "original")
    changed = records("holdout", selected)
    if len(original) != 100 or len(changed) != 100:
        raise RuntimeError(
            f"Holdout incomplete: original={len(original)} selected={len(changed)}"
        )
    event_rows = event_table("holdout", ("original", selected))
    event_summaries = {
        variant: event_summary(event_rows, variant)
        for variant in ("original", selected)
    }
    summaries = [
        summarize_scope("original", "overall", original),
        summarize_scope(selected, "overall", changed),
    ]
    for row in summaries:
        row.update(
            {
                f"event_{key}": value
                for key, value in event_summaries[row["variant"]].items()
                if key != "variant"
            }
        )
    write_csv(ROOT / "07_holdout/CRT_HOLDOUT_RESULTS.csv", summaries)
    write_csv(ROOT / "05_event_alignment/CRT_HOLDOUT_EVENT_ALIGNED_JERK.csv", event_rows)
    paired = paired_variant(original, changed, selected)
    write_json(
        ROOT / "07_holdout/CRT_HOLDOUT_PAIRED_STATISTICS.json",
        {
            "schema_version": "crt_holdout_paired_statistics_v1",
            "selected_variant": selected,
            "paired": paired,
            "event_summary": event_summaries,
            "tuning_after_holdout": false_value(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("development", "holdout"))
    args = parser.parse_args()
    if args.phase == "development":
        development()
    else:
        holdout()


if __name__ == "__main__":
    main()
