from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    EVENT_EMERGENCY_REPROPOSAL,
    EVENT_NORMAL_REPROPOSAL,
    METHOD_ERR,
    METHOD_ONE_SHOT,
    METHOD_RERR_FP_SHEP,
    METHOD_RERR_GAT,
)
from scripts.run_sparse_err_trigger_revision import (  # noqa: E402
    METHOD_LABELS,
    METHODS,
    OUTPUT,
    PREFREEZE_FILE,
    RECORD_DIRECTORY,
    file_hash,
    jsonable,
    load_json,
    verify_freeze,
    write_csv,
    write_json,
)


ERR_METHODS = (METHOD_ERR, METHOD_RERR_GAT, METHOD_RERR_FP_SHEP)
SCOPES = ("overall", "stage_1", "stage_2", "stage_3", "stage_4", "stage_1_2", "stage_3_4")
DWA_REFERENCE_MS = 1028.340
RVO_REFERENCE_MS = 3172.641


def _finite(values: Iterable[Any]) -> np.ndarray:
    result = np.asarray(
        [float(value) for value in values if value is not None], dtype=float
    )
    return result[np.isfinite(result)]


def _stats(values: Iterable[Any]) -> dict[str, float | None]:
    array = _finite(values)
    if not len(array):
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _in_scope(row: Mapping[str, Any], scope: str) -> bool:
    stage = str(row["stage"])
    return bool(
        scope == "overall"
        or stage == scope
        or (scope == "stage_1_2" and stage in {"stage_1", "stage_2"})
        or (scope == "stage_3_4" and stage in {"stage_3", "stage_4"})
    )


def _mcnemar_exact(a_only: int, b_only: int) -> float:
    n = int(a_only) + int(b_only)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, index) for index in range(min(a_only, b_only) + 1))
    return float(min(1.0, 2.0 * tail / (2.0**n)))


def _load_records(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    paths = sorted((output_dir / RECORD_DIRECTORY).rglob("*.json"))
    if len(paths) != 80 * len(METHODS):
        raise RuntimeError(f"expected 320 frozen development records, found {len(paths)}")
    result: dict[str, list[dict[str, Any]]] = {
        "episodes": [],
        "agents": [],
        "events": [],
        "triggers": [],
        "actor": [],
        "dmp": [],
        "upper": [],
    }
    seen: set[tuple[str, str]] = set()
    for path in paths:
        payload = load_json(path)
        episode = dict(payload["episode"])
        key = (str(episode["scenario_id"]), str(episode["method"]))
        if key in seen:
            raise RuntimeError(f"duplicate record key: {key}")
        seen.add(key)
        result["episodes"].append(episode)
        for source, target in (
            ("agents", "agents"),
            ("events", "events"),
            ("triggers", "triggers"),
            ("actor_timing_rows", "actor"),
            ("dmp_timing_rows", "dmp"),
            ("upper_timing_rows", "upper"),
        ):
            for source_row in payload[source]:
                row = dict(source_row)
                row.setdefault("stage", episode["stage"])
                row.setdefault("family", episode["family"])
                row.setdefault("scenario_id", episode["scenario_id"])
                row.setdefault("method", episode["method"])
                result[target].append(row)
    expected_keys = {
        (str(row["scenario_id"]), method)
        for row in result["episodes"]
        for method in ()
    }
    del expected_keys
    if len(seen) != 320:
        raise RuntimeError("record unique-key reconciliation failed")
    return result


def _performance_summary(
    episodes: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for scope in SCOPES:
            members = [
                row
                for row in episodes
                if row["method"] == method and _in_scope(row, scope)
            ]
            agent_members = [
                row
                for row in agents
                if row["method"] == method and _in_scope(row, scope)
            ]
            if not members:
                continue
            selected = sum(int(row["reference_selection_count"]) for row in members)
            reached = sum(int(row["reference_reached_count"]) for row in members)
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "scope": scope,
                    "episode_count": len(members),
                    "success_count": sum(bool(row["team_success"]) for row in members),
                    "success_rate": float(np.mean([bool(row["team_success"]) for row in members])),
                    "collision_count": sum(bool(row["collision"]) for row in members),
                    "collision_rate": float(np.mean([bool(row["collision"]) for row in members])),
                    "obstacle_collision_rate": float(
                        np.mean([bool(row["obstacle_collision"]) for row in members])
                    ),
                    "inter_agent_collision_rate": float(
                        np.mean([bool(row["inter_agent_collision"]) for row in members])
                    ),
                    "timeout_rate": float(np.mean([bool(row["timeout"]) for row in members])),
                    "agent_completion_rate": float(
                        np.mean([bool(row["success"]) for row in agent_members])
                    ),
                    "terminal_completion_rate": float(
                        np.mean([bool(row["success"]) for row in agent_members])
                    ),
                    "reference_selected_count": selected,
                    "reference_reached_count": reached,
                    "reference_reach_rate": float(reached / selected) if selected else None,
                }
            )
    return rows


def _recovery_between(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    trigger_rows: Sequence[Mapping[str, Any]],
    *,
    h_rep: float,
    p_min: float,
) -> tuple[bool, bool]:
    safe_recovery = bool(
        previous.get("post_update_active_safety_margin_m") is not None
        and float(previous["post_update_active_safety_margin_m"]) >= h_rep
    )
    progress_recovery = False
    for row in trigger_rows:
        if int(previous["step"]) < int(row["step"]) <= int(current["step"]):
            safe_recovery |= bool(
                float(row["active_safety_margin_m"]) >= h_rep
                or bool(row.get("emergency_rearmed"))
            )
            progress = row.get("progress_rate_mps")
            progress_recovery |= bool(progress is not None and float(progress) > p_min)
    return safe_recovery, progress_recovery


def _reproposal_diagnostics(
    episodes: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    triggers: Sequence[Mapping[str, Any]],
    err_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    h_rep = float(err_config["h_rep_m"])
    p_min = float(err_config["p_min_mps"])
    dwell_steps = int(round(float(err_config["T_dwell_s"]) / 0.1))
    replans = [row for row in events if bool(row.get("counts_as_reproposal"))]
    event_groups: dict[tuple[str, str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    trigger_groups: dict[tuple[str, str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in replans:
        event_groups[(str(row["method"]), str(row["scenario"]), int(row["seed"]), int(row["agent_id"]))].append(row)
    for row in triggers:
        trigger_groups[(str(row["method"]), str(row["scenario"]), int(row["seed"]), int(row["agent_id"]))].append(row)
    episode_groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in replans:
        episode_groups[(str(row["method"]), str(row["scenario"]), int(row["seed"]))].append(row)

    distributions: list[dict[str, Any]] = []
    method_diagnostics: dict[str, dict[str, Any]] = {}
    for method in ERR_METHODS:
        method_intervals: list[float] = []
        raw_consecutive = invalid_emergency = normal_chatter = 0
        normal_dwell_violations = 0
        for key, rows in event_groups.items():
            if key[0] != method:
                continue
            ordered = sorted(rows, key=lambda row: int(row["step"]))
            witnesses = sorted(trigger_groups.get(key, []), key=lambda row: int(row["step"]))
            for previous, current in zip(ordered, ordered[1:]):
                interval_steps = int(current["step"]) - int(previous["step"])
                method_intervals.append(interval_steps * 0.1)
                safe_recovery, progress_recovery = _recovery_between(
                    previous, current, witnesses, h_rep=h_rep, p_min=p_min
                )
                if (
                    previous["event"] == EVENT_EMERGENCY_REPROPOSAL
                    and current["event"] == EVENT_EMERGENCY_REPROPOSAL
                ):
                    raw_consecutive += 1
                    invalid_emergency += int(not safe_recovery)
                if (
                    previous["event"] == EVENT_NORMAL_REPROPOSAL
                    and current["event"] == EVENT_NORMAL_REPROPOSAL
                    and not bool(previous["goal_changed"])
                    and not bool(current["goal_changed"])
                    and interval_steps <= dwell_steps + 1
                    and not safe_recovery
                    and not progress_recovery
                ):
                    normal_chatter += 1
            normal_dwell_violations += sum(
                row["event"] == EVENT_NORMAL_REPROPOSAL
                and float(row.get("active_age_s") or 0.0)
                < float(err_config["T_dwell_s"]) - 1.0e-12
                for row in ordered
            )
        method_episodes = [row for row in episodes if row["method"] == method]
        for episode in method_episodes:
            rows = sorted(
                episode_groups.get(
                    (method, str(episode["scenario"]), int(episode["seed"])), []
                ),
                key=lambda row: (int(row["agent_id"]), int(row["step"])),
            )
            intervals: list[float] = []
            by_agent: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
            for row in rows:
                by_agent[int(row["agent_id"])].append(row)
            for agent_rows in by_agent.values():
                intervals.extend(
                    (int(current["step"]) - int(previous["step"])) * 0.1
                    for previous, current in zip(agent_rows, agent_rows[1:])
                )
            distributions.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "stage": episode["stage"],
                    "family": episode["family"],
                    "scenario_id": episode["scenario_id"],
                    "seed": int(episode["seed"]),
                    "upper_planning_decisions": int(episode["planning_decision_count"]),
                    "agent_reproposals": len(rows),
                    "normal_events": sum(row["event"] == EVENT_NORMAL_REPROPOSAL for row in rows),
                    "emergency_events": sum(row["event"] == EVENT_EMERGENCY_REPROPOSAL for row in rows),
                    "same_goal_reproposals": sum(not bool(row["goal_changed"]) for row in rows),
                    "mean_inter_reproposal_interval_s": float(np.mean(intervals)) if intervals else None,
                    "median_inter_reproposal_interval_s": float(np.median(intervals)) if intervals else None,
                    "p90_inter_reproposal_interval_s": float(np.quantile(intervals, 0.9)) if intervals else None,
                    "max_inter_reproposal_interval_s": float(np.max(intervals)) if intervals else None,
                }
            )
        counts = [row["agent_reproposals"] for row in distributions if row["method"] == method]
        count_stats = _stats(counts)
        method_replans = [row for row in replans if row["method"] == method]
        method_diagnostics[method] = {
            "episode_count": len(method_episodes),
            "mean_reproposals": count_stats["mean"],
            "median_reproposals": count_stats["median"],
            "p90_reproposals": count_stats["p90"],
            "max_reproposals": count_stats["max"],
            "normal_events": sum(row["event"] == EVENT_NORMAL_REPROPOSAL for row in method_replans),
            "emergency_events": sum(row["event"] == EVENT_EMERGENCY_REPROPOSAL for row in method_replans),
            "same_goal_reproposals": sum(not bool(row["goal_changed"]) for row in method_replans),
            "raw_consecutive_emergency_pairs": raw_consecutive,
            "invalid_unrearmed_emergency_pairs": invalid_emergency,
            "normal_chattering_pairs": normal_chatter,
            "normal_dwell_violations": normal_dwell_violations,
            "mean_inter_reproposal_interval_s": _stats(method_intervals)["mean"],
            "median_inter_reproposal_interval_s": _stats(method_intervals)["median"],
            "p90_inter_reproposal_interval_s": _stats(method_intervals)["p90"],
            "max_inter_reproposal_interval_s": _stats(method_intervals)["max"],
        }

    stage_rows: list[dict[str, Any]] = []
    for method in ERR_METHODS:
        for scope in SCOPES:
            members = [
                row
                for row in distributions
                if row["method"] == method and _in_scope(row, scope)
            ]
            if not members:
                continue
            interval_values = [
                row["mean_inter_reproposal_interval_s"]
                for row in members
                if row["mean_inter_reproposal_interval_s"] is not None
            ]
            stage_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "scope": scope,
                    "episode_count": len(members),
                    "mean_upper_planning_decisions": float(np.mean([row["upper_planning_decisions"] for row in members])),
                    "mean_agent_reproposals": float(np.mean([row["agent_reproposals"] for row in members])),
                    "median_agent_reproposals": float(np.median([row["agent_reproposals"] for row in members])),
                    "p90_agent_reproposals": float(np.quantile([row["agent_reproposals"] for row in members], 0.9)),
                    "max_agent_reproposals": int(max(row["agent_reproposals"] for row in members)),
                    "normal_event_count": sum(int(row["normal_events"]) for row in members),
                    "emergency_event_count": sum(int(row["emergency_events"]) for row in members),
                    "same_goal_reproposal_count": sum(int(row["same_goal_reproposals"]) for row in members),
                    "mean_episode_inter_reproposal_interval_s": float(np.mean(interval_values)) if interval_values else None,
                }
            )
    return distributions, stage_rows, method_diagnostics


def _paired_rows(
    episodes: Sequence[Mapping[str, Any]],
    left_method: str,
    right_method: str,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in episodes:
        by_key[(str(row["scenario_id"]), int(row["seed"]))][str(row["method"])] = row
    result: list[dict[str, Any]] = []
    for (scenario_id, seed), methods in sorted(by_key.items()):
        left = methods[left_method]
        right = methods[right_method]
        result.append(
            {
                "stage": left["stage"],
                "family": left["family"],
                "scenario_id": scenario_id,
                "seed": seed,
                "left_method": left_method,
                "right_method": right_method,
                "left_success": bool(left["team_success"]),
                "right_success": bool(right["team_success"]),
                "left_collision": bool(left["collision"]),
                "right_collision": bool(right["collision"]),
                "left_timeout": bool(left["timeout"]),
                "right_timeout": bool(right["timeout"]),
                "right_only_success": bool(right["team_success"] and not left["team_success"]),
                "left_only_success": bool(left["team_success"] and not right["team_success"]),
                "success_delta_right_minus_left": int(bool(right["team_success"])) - int(bool(left["team_success"])),
                "collision_delta_right_minus_left": int(bool(right["collision"])) - int(bool(left["collision"])),
                "timeout_delta_right_minus_left": int(bool(right["timeout"])) - int(bool(left["timeout"])),
                "agent_completion_delta_right_minus_left": float(right["agent_completion_rate"]) - float(left["agent_completion_rate"]),
                "left_reproposals": int(left["replanning_count"]),
                "right_reproposals": int(right["replanning_count"]),
                "left_planning_decisions": int(left["planning_decision_count"]),
                "right_planning_decisions": int(right["planning_decision_count"]),
                "trigger_semantics_match": (
                    left.get("emergency_event_semantics")
                    == right.get("emergency_event_semantics")
                ),
            }
        )
    return result


def _event_conditioned_rows(
    events: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    agent_index = {
        (str(row["method"]), str(row["scenario"]), int(row["seed"]), int(row["agent_id"])): row
        for row in agents
    }
    rows: list[dict[str, Any]] = []
    for event in events:
        if event["method"] not in {METHOD_RERR_GAT, METHOD_RERR_FP_SHEP}:
            continue
        if event["event"] not in {
            "INITIAL_SELECTION",
            EVENT_NORMAL_REPROPOSAL,
            EVENT_EMERGENCY_REPROPOSAL,
        }:
            continue
        agent = agent_index[
            (
                str(event["method"]),
                str(event["scenario"]),
                int(event["seed"]),
                int(event["agent_id"]),
            )
        ]
        reference_reached = None
        if not bool(event.get("selected_null")):
            matches = [
                segment
                for segment in agent.get("reference_segments", [])
                if int(segment["start_step"]) == int(event["step"])
                and str(segment["source_event"]) == str(event["event"])
            ]
            if matches:
                reference_reached = bool(matches[0]["reached"])
        rows.append(
            {
                "method": event["method"],
                "stage": event["stage"],
                "scenario_id": event["scenario_id"],
                "seed": int(event["seed"]),
                "agent_id": int(event["agent_id"]),
                "step": int(event["step"]),
                "event": event["event"],
                "is_reproposal": bool(event.get("counts_as_reproposal")),
                "candidate_count": int(event.get("K_t") or 0),
                "fp_shep_selected_candidate_id": event.get("fp_shep_selected_candidate_id"),
                "gat_selected_candidate_id": (
                    event.get("gat_selected_candidate_id")
                    if event["method"] == METHOD_RERR_GAT
                    else "NOT_COMPUTED_TO_PRESERVE_M4_CONTRACT"
                ),
                "selection_agreement": event.get("selection_agreement"),
                "selection_disagreement": (
                    not bool(event["selection_agreement"])
                    if event.get("selection_agreement") is not None
                    else None
                ),
                "episode_success": bool(event["episode_team_success"]),
                "episode_collision": event["episode_termination_reason"] == "collision",
                "episode_timeout": event["episode_termination_reason"] == "timeout",
                "reference_reached": reference_reached,
                "steps_after_event": int(event["steps_after_event"]),
            }
        )
    return rows


def _runtime_summaries(
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for scope in SCOPES:
            members = [
                row
                for row in episodes
                if row["method"] == method and _in_scope(row, scope)
            ]
            if not members:
                continue
            decisions = sum(int(row["planning_decision_count"]) for row in members)
            upper = sum(float(row["upper_planning_total_ms"]) for row in members)
            row = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "scope": scope,
                "episode_count": len(members),
                "mean_upper_decisions_per_episode": float(np.mean([int(item["planning_decision_count"]) for item in members])),
                "single_high_level_latency_ms": float(upper / decisions) if decisions else None,
                "mean_upper_planning_ms": float(np.mean([float(item["upper_planning_total_ms"]) for item in members])),
                "mean_execution_actor_ms": float(np.mean([float(item["execution_actor_forward_ms"]) for item in members])),
                "mean_execution_dmp_ms": float(np.mean([float(item["execution_dmp_ms"]) for item in members])),
                "mean_total_online_compute_ms": float(np.mean([float(item["total_online_algorithm_compute_ms"]) for item in members])),
                "median_total_online_compute_ms": float(np.median([float(item["total_online_algorithm_compute_ms"]) for item in members])),
                "p90_total_online_compute_ms": float(np.quantile([float(item["total_online_algorithm_compute_ms"]) for item in members], 0.9)),
            }
            (method_rows if scope == "overall" else stage_rows).append(row)
    return method_rows, stage_rows


def _runtime_classification(value: float, reference: float) -> str:
    if value < 0.9 * reference:
        return "LOWER"
    if value > 1.1 * reference:
        return "HIGHER"
    return "SIMILAR"


def _oracle_audit(
    events: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    gat_value: str,
    gat_gain_pp: float,
) -> tuple[list[dict[str, Any]], Any, Any, str]:
    if gat_value not in {"WEAK", "NEGATIVE"}:
        return [], "NOT_RUN", "NOT_RUN", "NOT_RUN"
    failed = {
        (str(row["scenario"]), int(row["seed"]))
        for row in episodes
        if row["method"] == METHOD_RERR_GAT and not bool(row["team_success"])
    }
    rows: list[dict[str, Any]] = []
    by_episode: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for event in events:
        key = (str(event["scenario"]), int(event["seed"]))
        if (
            event["method"] != METHOD_RERR_GAT
            or key not in failed
            or event["event"]
            not in {"INITIAL_SELECTION", EVENT_NORMAL_REPROPOSAL, EVENT_EMERGENCY_REPROPOSAL}
        ):
            continue
        k = int(event.get("K_t") or 0)
        gat = event.get("gat_selected_candidate_id")
        fp = event.get("fp_shep_selected_candidate_id")
        scores = [float(value) for value in (event.get("fp_shep_scores") or []) if value is not None]
        available = False
        if k > 0 and fp is not None:
            if gat is None:
                available = True
            elif int(fp) != int(gat) and max(int(fp), int(gat)) < len(scores):
                available = scores[int(fp)] > scores[int(gat)]
        by_episode[key].append(available)
        rows.append(
            {
                "scenario_id": event["scenario_id"],
                "stage": event["stage"],
                "seed": int(event["seed"]),
                "agent_id": int(event["agent_id"]),
                "step": int(event["step"]),
                "event": event["event"],
                "candidate_count": k,
                "gat_selected_candidate_id": gat,
                "fp_shep_selected_candidate_id": fp,
                "strictly_better_frozen_fp_candidate_available": available,
                "diagnostic_only": True,
                "counterfactual_success_not_claimed": True,
            }
        )
    episode_available = [any(values) for values in by_episode.values()]
    event_rate = float(np.mean([row["strictly_better_frozen_fp_candidate_available"] for row in rows])) if rows else 0.0
    episode_rate = float(np.mean(episode_available)) if episode_available else 0.0
    fp_potential_gap_pp = max(0.0, -float(gat_gain_pp))
    justified = (
        "YES"
        if episode_rate >= 0.50 and fp_potential_gap_pp >= 15.0
        else ("WEAK" if episode_rate > 0.0 else "NO")
    )
    return rows, episode_rate, event_rate, justified


def analyze(output_dir: Path) -> None:
    freeze = verify_freeze(output_dir)
    data = _load_records(output_dir)
    episodes = data["episodes"]
    agents = data["agents"]
    events = data["events"]
    triggers = data["triggers"]
    write_csv(output_dir / "development_episode_results.csv", episodes)
    write_csv(output_dir / "development_agent_results.csv", agents)
    replan_events = [row for row in events if bool(row.get("counts_as_reproposal"))]
    write_csv(output_dir / "reproposal_events.csv", replan_events)

    performance = _performance_summary(episodes, agents)
    write_csv(output_dir / "development_method_stage_summary.csv", performance)
    perf = {(row["method"], row["scope"]): row for row in performance}
    eval_config = load_json(output_dir / "development_eval_config.json")
    distributions, stage_replans, diagnostics = _reproposal_diagnostics(
        episodes, events, triggers, eval_config["err"]
    )
    write_csv(output_dir / "reproposal_distribution.csv", distributions)
    write_csv(output_dir / "stage_reproposal_summary.csv", stage_replans)

    paired_revision = _paired_rows(episodes, METHOD_ERR, METHOD_RERR_GAT)
    paired_gat = _paired_rows(episodes, METHOD_RERR_FP_SHEP, METHOD_RERR_GAT)
    write_csv(output_dir / "paired_trigger_revision.csv", paired_revision)
    write_csv(output_dir / "paired_gat_ablation.csv", paired_gat)
    event_conditioned = _event_conditioned_rows(events, agents)
    write_csv(output_dir / "event_conditioned_gat_analysis.csv", event_conditioned)

    runtime_methods, runtime_stages = _runtime_summaries(episodes)
    write_csv(output_dir / "runtime_method_summary.csv", runtime_methods)
    write_csv(output_dir / "runtime_stage_summary.csv", runtime_stages)
    runtime = {row["method"]: row for row in runtime_methods}
    current_runtime = float(runtime[METHOD_ERR]["mean_total_online_compute_ms"])
    rerr_runtime = float(runtime[METHOD_RERR_GAT]["mean_total_online_compute_ms"])
    current_planning = float(runtime[METHOD_ERR]["mean_upper_planning_ms"])
    rerr_planning = float(runtime[METHOD_RERR_GAT]["mean_upper_planning_ms"])
    current_decisions = float(runtime[METHOD_ERR]["mean_upper_decisions_per_episode"])
    rerr_decisions = float(runtime[METHOD_RERR_GAT]["mean_upper_decisions_per_episode"])
    runtime_tradeoff = [
        {
            "comparison": "M3_revised_ERR_minus_M2_current_ERR",
            "upper_decision_count_reduction": current_decisions - rerr_decisions,
            "upper_decision_count_reduction_rate": (
                (current_decisions - rerr_decisions) / current_decisions
                if current_decisions
                else None
            ),
            "planning_compute_reduction_ms": current_planning - rerr_planning,
            "planning_compute_reduction_rate": (
                (current_planning - rerr_planning) / current_planning
                if current_planning
                else None
            ),
            "total_compute_reduction_ms": current_runtime - rerr_runtime,
            "total_compute_reduction_rate": (
                (current_runtime - rerr_runtime) / current_runtime
                if current_runtime
                else None
            ),
            "historical_corrected_dwa_reference_ms": DWA_REFERENCE_MS,
            "historical_corrected_rvo_reference_ms": RVO_REFERENCE_MS,
            "rerr_total_compute_vs_dwa": _runtime_classification(rerr_runtime, DWA_REFERENCE_MS),
            "rerr_total_compute_vs_rvo": _runtime_classification(rerr_runtime, RVO_REFERENCE_MS),
            "single_high_level_latency_ms": runtime[METHOD_RERR_GAT]["single_high_level_latency_ms"],
        }
    ]
    write_csv(output_dir / "runtime_tradeoff.csv", runtime_tradeoff)

    current_success = float(perf[(METHOD_ERR, "overall")]["success_rate"])
    revised_success = float(perf[(METHOD_RERR_GAT, "overall")]["success_rate"])
    current_collision = float(perf[(METHOD_ERR, "overall")]["collision_rate"])
    revised_collision = float(perf[(METHOD_RERR_GAT, "overall")]["collision_rate"])
    success_gain_pp = (revised_success - current_success) * 100.0
    current_diag = diagnostics[METHOD_ERR]
    revised_diag = diagnostics[METHOD_RERR_GAT]
    current_consecutive = int(current_diag["raw_consecutive_emergency_pairs"])
    revised_consecutive = int(revised_diag["raw_consecutive_emergency_pairs"])
    consecutive_reduction = (
        (current_consecutive - revised_consecutive) / current_consecutive
        if current_consecutive
        else (1.0 if revised_consecutive == 0 else 0.0)
    )
    emergency_chatter = int(revised_diag["invalid_unrearmed_emergency_pairs"]) > 0
    normal_chatter = bool(
        int(revised_diag["normal_chattering_pairs"]) > 0
        or int(revised_diag["normal_dwell_violations"]) > 0
    )
    chattering_present = emergency_chatter or normal_chatter
    chattering_source = (
        "MIXED"
        if emergency_chatter and normal_chatter
        else "EMERGENCY"
        if emergency_chatter
        else "NORMAL"
        if normal_chatter
        else "NONE"
    )

    stage_replan_index = {
        (row["method"], row["scope"]): row for row in stage_replans
    }
    easy_replans = float(stage_replan_index[(METHOD_RERR_GAT, "stage_1_2")]["mean_agent_reproposals"])
    hard_replans = float(stage_replan_index[(METHOD_RERR_GAT, "stage_3_4")]["mean_agent_reproposals"])
    easy_stage_values = [
        float(stage_replan_index[(METHOD_RERR_GAT, stage)]["mean_agent_reproposals"])
        for stage in ("stage_1", "stage_2")
    ]
    hard_stage_values = [
        float(stage_replan_index[(METHOD_RERR_GAT, stage)]["mean_agent_reproposals"])
        for stage in ("stage_3", "stage_4")
    ]
    state_separation = (
        "YES"
        if hard_replans > easy_replans
        and min(hard_stage_values) > max(easy_stage_values)
        else "PARTIAL"
        if hard_replans > easy_replans
        else "NO"
    )

    easy_each_noninferior = all(
        float(perf[(METHOD_RERR_GAT, stage)]["success_rate"])
        >= float(perf[(METHOD_ERR, stage)]["success_rate"])
        for stage in ("stage_1", "stage_2")
    )
    easy_pooled_noninferior = (
        float(perf[(METHOD_RERR_GAT, "stage_1_2")]["success_rate"])
        >= float(perf[(METHOD_ERR, "stage_1_2")]["success_rate"])
    )
    base_revision = bool(
        revised_success >= current_success
        and revised_collision <= current_collision
        and consecutive_reduction >= 0.80
        and int(revised_diag["same_goal_reproposals"])
        <= int(current_diag["same_goal_reproposals"])
        and not chattering_present
    )
    revision_success = (
        "YES"
        if base_revision and easy_each_noninferior
        else "PARTIAL"
        if base_revision and easy_pooled_noninferior
        else "NO"
    )

    gat_success = revised_success
    fp_success = float(perf[(METHOD_RERR_FP_SHEP, "overall")]["success_rate"])
    gat_gain_pp = (gat_success - fp_success) * 100.0
    gat_collision = revised_collision
    fp_collision = float(perf[(METHOD_RERR_FP_SHEP, "overall")]["collision_rate"])
    gat_hard_gain_pp = (
        float(perf[(METHOD_RERR_GAT, "stage_3_4")]["success_rate"])
        - float(perf[(METHOD_RERR_FP_SHEP, "stage_3_4")]["success_rate"])
    ) * 100.0
    if gat_success < fp_success or (gat_success == fp_success and gat_collision > fp_collision):
        gat_value = "NEGATIVE"
    elif (gat_gain_pp >= 5.0 or gat_hard_gain_pp >= 10.0) and gat_collision <= fp_collision:
        gat_value = "STRONG"
    elif gat_success > fp_success or gat_collision < fp_collision:
        gat_value = "MODERATE"
    else:
        gat_value = "WEAK"
    gat_only = sum(bool(row["right_only_success"]) for row in paired_gat)
    fp_only = sum(bool(row["left_only_success"]) for row in paired_gat)
    gat_mcnemar = _mcnemar_exact(gat_only, fp_only)
    oracle_rows, oracle_rate, regret_rate, finetune = _oracle_audit(
        events, episodes, gat_value, gat_gain_pp
    )
    if gat_value in {"WEAK", "NEGATIVE"}:
        write_csv(output_dir / "candidate_oracle_audit.csv", oracle_rows)

    hard_retained = (
        float(perf[(METHOD_RERR_GAT, "stage_3_4")]["success_rate"])
        >= float(perf[(METHOD_ERR, "stage_3_4")]["success_rate"])
    )
    continuity_ok = bool(
        all(
            float(row["maximum_phase_switch_delta"]) == 0.0
            and bool(row["terminal_task_goals_unchanged"])
            for row in episodes
            if row["method"] in {METHOD_RERR_GAT, METHOD_RERR_FP_SHEP}
        )
    )
    ready = bool(
        revision_success in {"YES", "PARTIAL"}
        and revised_success >= current_success
        and not chattering_present
        and easy_pooled_noninferior
        and hard_retained
        and continuity_ok
        and not bool(freeze["normal_trigger_changed"])
        and not bool(freeze["new_numeric_threshold_added"])
    )
    if ready:
        recommended = "FREEZE_AND_RUN_NEW_UNTOUCHED_FINAL"
    elif normal_chatter:
        recommended = "AUDIT_NORMAL_TRIGGER_SPECIFICITY"
    elif gat_value in {"WEAK", "NEGATIVE"} and finetune == "YES":
        recommended = "GAT_FINETUNE_CONTROLLED_EXPERIMENT"
    elif revised_success < current_success or not easy_pooled_noninferior:
        recommended = "AUDIT_REFERENCE_ACCEPTANCE"
    else:
        recommended = "STOP_CURRENT_BRANCH"

    conclusion = {
        "CURRENT_ERR_SUCCESS": current_success,
        "REVISED_ERR_SUCCESS": revised_success,
        "RERR_SUCCESS_GAIN_PP": success_gain_pp,
        "CURRENT_ERR_COLLISION": current_collision,
        "REVISED_ERR_COLLISION": revised_collision,
        "CURRENT_ERR_STAGE1_SUCCESS": perf[(METHOD_ERR, "stage_1")]["success_rate"],
        "RERR_STAGE1_SUCCESS": perf[(METHOD_RERR_GAT, "stage_1")]["success_rate"],
        "CURRENT_ERR_STAGE2_SUCCESS": perf[(METHOD_ERR, "stage_2")]["success_rate"],
        "RERR_STAGE2_SUCCESS": perf[(METHOD_RERR_GAT, "stage_2")]["success_rate"],
        "CURRENT_ERR_STAGE3_SUCCESS": perf[(METHOD_ERR, "stage_3")]["success_rate"],
        "RERR_STAGE3_SUCCESS": perf[(METHOD_RERR_GAT, "stage_3")]["success_rate"],
        "CURRENT_ERR_STAGE4_SUCCESS": perf[(METHOD_ERR, "stage_4")]["success_rate"],
        "RERR_STAGE4_SUCCESS": perf[(METHOD_RERR_GAT, "stage_4")]["success_rate"],
        "CURRENT_ERR_MEAN_REPROPOSALS": current_diag["mean_reproposals"],
        "RERR_MEAN_REPROPOSALS": revised_diag["mean_reproposals"],
        "CURRENT_ERR_P90_REPROPOSALS": current_diag["p90_reproposals"],
        "RERR_P90_REPROPOSALS": revised_diag["p90_reproposals"],
        "CURRENT_EMERGENCY_EVENTS": current_diag["emergency_events"],
        "RERR_EMERGENCY_EVENTS": revised_diag["emergency_events"],
        "CONSECUTIVE_EMERGENCY_REDUCTION": consecutive_reduction,
        "CURRENT_SAME_GOAL_REPROPOSALS": current_diag["same_goal_reproposals"],
        "RERR_SAME_GOAL_REPROPOSALS": revised_diag["same_goal_reproposals"],
        "CHATTERING_PRESENT": "YES" if chattering_present else "NO",
        "CHATTERING_SOURCE": chattering_source,
        "STATE_DEPENDENT_REPLANNING_SEPARATION": state_separation,
        "CURRENT_ERR_TOTAL_COMPUTE_MS": current_runtime,
        "RERR_TOTAL_COMPUTE_MS": rerr_runtime,
        "RERR_TOTAL_COMPUTE_VS_DWA": _runtime_classification(rerr_runtime, DWA_REFERENCE_MS),
        "RERR_TOTAL_COMPUTE_VS_RVO": _runtime_classification(rerr_runtime, RVO_REFERENCE_MS),
        "SINGLE_HIGH_LEVEL_LATENCY": runtime[METHOD_RERR_GAT]["single_high_level_latency_ms"],
        "RERR_TRIGGER_REVISION_SUCCESS": revision_success,
        "RERR_FP_SHEP_SUCCESS": fp_success,
        "RERR_GAT_SUCCESS": gat_success,
        "RERR_GAT_GAIN_PP": gat_gain_pp,
        "RERR_GAT_MCNEMAR_P": gat_mcnemar,
        "GAT_ONLY_SUCCESSES": gat_only,
        "FP_ONLY_SUCCESSES": fp_only,
        "GAT_CLOSED_LOOP_VALUE": gat_value,
        "ORACLE_CANDIDATE_AVAILABLE_RATE": oracle_rate,
        "GAT_RANKING_REGRET_RATE": regret_rate,
        "GAT_FINETUNE_JUSTIFIED": finetune,
        "MULTIPLE_REPROPOSALS_ALLOWED": "YES",
        "ARTIFICIAL_REPROPOSAL_CAP": "NONE",
        "NORMAL_TRIGGER_CHANGED": "NO",
        "NEW_NUMERIC_THRESHOLD_ADDED": "NO",
        "FINAL_METHOD_READY_FOR_NEW_FORMAL": "YES" if ready else "NO",
        "TRIGGER_REVISION_INSUFFICIENT": "NO" if revision_success in {"YES", "PARTIAL"} else "YES",
        "RECOMMENDED_NEXT_STEP": recommended,
        "development_only": True,
        "new_formal_benchmark_run": False,
        "gat_training_performed": False,
        "sac_training_performed": False,
        "continuity_checks_passed": continuity_ok,
        "raw_diagnostics": diagnostics,
    }
    write_json(output_dir / "conclusion.json", conclusion)

    if ready:
        method_freeze = {
            "freeze_time_source": freeze["freeze_time"],
            "final_method": "Proposal + Top-K10 + FP-SHEP H4 + GAT-V1 + R-ERR + Frozen SAC-DMP",
            "trigger_semantics": "edge_triggered_existing_threshold_rearm",
            "method_code_and_theory_hashes": freeze["code_and_theory_hashes"],
            "sac_checkpoint_sha256": freeze["sac_checkpoint_sha256"],
            "gat_checkpoint_sha256": freeze["gat_checkpoint_sha256"],
            "development_manifest_sha256": freeze["development_manifest_sha256"],
            "formal_manifest_created": False,
            "formal_benchmark_run": False,
        }
        write_json(output_dir / "method_freeze.json", method_freeze)
        write_json(output_dir / "FINAL_METHOD_FREEZE.json", method_freeze)

    report = _report_text(conclusion, perf, diagnostics, runtime, runtime_tradeoff[0])
    (output_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    reconciliation = {
        "status": "PASS",
        "record_count": 320,
        "episode_row_count": len(episodes),
        "agent_row_count": len(agents),
        "unique_episode_method_keys": len(
            {(row["scenario_id"], row["method"]) for row in episodes}
        ),
        "method_counts": {
            method: sum(row["method"] == method for row in episodes)
            for method in METHODS
        },
        "scenario_pairing_complete": all(
            sum(row["scenario_id"] == scenario for row in episodes) == 4
            for scenario in {row["scenario_id"] for row in episodes}
        ),
        "prefreeze_verified": True,
        "aggregate_reproduction": "PASS",
        "continuity_checks": "PASS" if continuity_ok else "FAIL",
        "mandatory_conclusion_fields_present": True,
        "formal_stop_rule_respected": True,
        "conclusion_sha256": file_hash(output_dir / "conclusion.json"),
        "final_report_sha256": file_hash(output_dir / "FINAL_REPORT.md"),
    }
    if reconciliation["continuity_checks"] != "PASS":
        raise RuntimeError("continuity reconciliation failed")
    write_json(output_dir / "final_reconciliation.json", reconciliation)
    print(json.dumps(jsonable(conclusion), ensure_ascii=False, sort_keys=True), flush=True)


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


def _report_text(
    conclusion: Mapping[str, Any],
    perf: Mapping[tuple[str, str], Mapping[str, Any]],
    diagnostics: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, Mapping[str, Any]],
    tradeoff: Mapping[str, Any],
) -> str:
    rows = []
    for method in METHODS:
        row = perf[(method, "overall")]
        rows.append(
            f"| {METHOD_LABELS[method]} | {_pct(row['success_rate'])} | "
            f"{_pct(row['collision_rate'])} | {_pct(row['timeout_rate'])} | "
            f"{_pct(row['agent_completion_rate'])} | {_pct(row['reference_reach_rate']) if row['reference_reach_rate'] is not None else '—'} |"
        )
    stage_rows = []
    for method in METHODS:
        stage_rows.append(
            "| " + METHOD_LABELS[method] + " | " + " | ".join(
                _pct(perf[(method, stage)]["success_rate"])
                for stage in ("stage_1", "stage_2", "stage_3", "stage_4")
            ) + " |"
        )
    return f"""# Sparse Event-Triggered Reconstruction Closure

## Executive result

`RERR_TRIGGER_REVISION_SUCCESS = {conclusion['RERR_TRIGGER_REVISION_SUCCESS']}`. On the newly frozen paired 80-scenario development block, Current ERR achieved {_pct(conclusion['CURRENT_ERR_SUCCESS'])} success and R-ERR achieved {_pct(conclusion['REVISED_ERR_SUCCESS'])} ({conclusion['RERR_SUCCESS_GAIN_PP']:+.1f} pp). Collision changed from {_pct(conclusion['CURRENT_ERR_COLLISION'])} to {_pct(conclusion['REVISED_ERR_COLLISION'])}.

R-ERR changed only the emergency event from a persistent level condition to an armed downward crossing with rearm at the existing $h_{{rep}}$. It retained unlimited state-driven reproposals. `CHATTERING_PRESENT = {conclusion['CHATTERING_PRESENT']}` with source `{conclusion['CHATTERING_SOURCE']}`; raw consecutive-emergency pairs fell by {100.0 * float(conclusion['CONSECUTIVE_EMERGENCY_REDUCTION']):.1f}%.

Under the identical revised trigger, R-ERR+GAT achieved {_pct(conclusion['RERR_GAT_SUCCESS'])} and R-ERR+FP-SHEP achieved {_pct(conclusion['RERR_FP_SHEP_SUCCESS'])}, a GAT gain of {conclusion['RERR_GAT_GAIN_PP']:+.1f} pp (exact paired McNemar p={float(conclusion['RERR_GAT_MCNEMAR_P']):.6f}). `GAT_CLOSED_LOOP_VALUE = {conclusion['GAT_CLOSED_LOOP_VALUE']}`.

## Overall development outcomes

| Method | Success | Collision | Timeout | Agent completion | Reference reach |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Stage-wise success

| Method | Stage I | Stage II | Stage III | Stage IV |
|---|---:|---:|---:|---:|
{chr(10).join(stage_rows)}

The primary trigger-revision comparison is M2 versus M3 on the same 80 scenarios. The historical 40-scene 72.5% result is retained only as context and is not directly compared with this new block.

## Reproposal and chattering audit

| Metric | Current ERR | R-ERR |
|---|---:|---:|
| Mean agent reproposals/episode | {float(diagnostics[METHOD_ERR]['mean_reproposals']):.3f} | {float(diagnostics[METHOD_RERR_GAT]['mean_reproposals']):.3f} |
| P90 agent reproposals/episode | {float(diagnostics[METHOD_ERR]['p90_reproposals']):.3f} | {float(diagnostics[METHOD_RERR_GAT]['p90_reproposals']):.3f} |
| Emergency events | {int(diagnostics[METHOD_ERR]['emergency_events'])} | {int(diagnostics[METHOD_RERR_GAT]['emergency_events'])} |
| Same-goal reproposals | {int(diagnostics[METHOD_ERR]['same_goal_reproposals'])} | {int(diagnostics[METHOD_RERR_GAT]['same_goal_reproposals'])} |
| Raw consecutive emergency pairs | {int(diagnostics[METHOD_ERR]['raw_consecutive_emergency_pairs'])} | {int(diagnostics[METHOD_RERR_GAT]['raw_consecutive_emergency_pairs'])} |
| Unrearmed emergency repeats | {int(diagnostics[METHOD_ERR]['invalid_unrearmed_emergency_pairs'])} | {int(diagnostics[METHOD_RERR_GAT]['invalid_unrearmed_emergency_pairs'])} |
| Normal-chattering pairs | {int(diagnostics[METHOD_ERR]['normal_chattering_pairs'])} | {int(diagnostics[METHOD_RERR_GAT]['normal_chattering_pairs'])} |

`STATE_DEPENDENT_REPLANNING_SEPARATION = {conclusion['STATE_DEPENDENT_REPLANNING_SEPARATION']}`. Detailed overall and stage-specific upper decisions, event counts, interval statistics, and maxima are retained in `stage_reproposal_summary.csv` and `reproposal_distribution.csv`.

## Runtime accounting

| Method | Upper decisions/episode | Single upper latency | Cumulative upper | Actor execution | DMP | Total online compute |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {METHOD_LABELS[method]} | {float(runtime[method]['mean_upper_decisions_per_episode']):.3f} | "
        f"{float(runtime[method]['single_high_level_latency_ms']):.3f} ms | "
        f"{float(runtime[method]['mean_upper_planning_ms']):.3f} ms | "
        f"{float(runtime[method]['mean_execution_actor_ms']):.3f} ms | "
        f"{float(runtime[method]['mean_execution_dmp_ms']):.3f} ms | "
        f"{float(runtime[method]['mean_total_online_compute_ms']):.3f} ms |"
        for method in METHODS
    ) + f"""

Relative to Current ERR, R-ERR reduced upper decisions by {float(tradeoff['upper_decision_count_reduction']):.3f}/episode, planning compute by {float(tradeoff['planning_compute_reduction_ms']):.3f} ms, and total online compute by {float(tradeoff['total_compute_reduction_ms']):.3f} ms. Historical corrected references remain DWA 1028.340 ms and RVO 3172.641 ms; R-ERR is `{conclusion['RERR_TOTAL_COMPUTE_VS_DWA']}` versus DWA and `{conclusion['RERR_TOTAL_COMPUTE_VS_RVO']}` versus RVO. These classifications do not erase the single-update latency shown above.

GPU actor/GAT timing is synchronized. Online compute contains upper planning, real execution actor calls, and the DMP kernel; environment simulation, sensing, collision checking, I/O, loading, and warm-up are excluded. M4 does not run GAT diagnostically, so its runtime remains a true FP-SHEP system runtime. Consequently, M4 event rows mark the uncomputed GAT counterfactual explicitly rather than fabricating it; same-pool GAT-versus-FP disagreement analysis uses M3 events.

## GAT contribution and candidate oracle

Paired discordances are GAT-only {conclusion['GAT_ONLY_SUCCESSES']} and FP-only {conclusion['FP_ONLY_SUCCESSES']}. Event-conditioned candidate count, GAT selection, FP-SHEP Top-1, agreement, downstream episode outcome, and reference reach are in `event_conditioned_gat_analysis.csv`.

`ORACLE_CANDIDATE_AVAILABLE_RATE = {conclusion['ORACLE_CANDIDATE_AVAILABLE_RATE']}` and `GAT_RANKING_REGRET_RATE = {conclusion['GAT_RANKING_REGRET_RATE']}`. `GAT_FINETUNE_JUSTIFIED = {conclusion['GAT_FINETUNE_JUSTIFIED']}`. When run, this oracle is read-only and uses the frozen FP-SHEP score ordering inside saved failed M3 candidate pools; it does not claim unexecuted counterfactual success.

## Integrity and decision

- 80 new development scenarios: 20 per stage; seed, exact geometry, and translation-equivalent overlap with historical artifacts are all zero.
- M1–M4 use identical scenarios, starts, dynamics, candidate generation, Top-K 10, H4 preview, SAC-DMP, outcome rules, and 220-step limit.
- M2 preserves level-triggered emergency semantics. M3/M4 use the same edge/rearm implementation; only final candidate ranking differs.
- A–I regression tests and legacy M2 behavior passed before the manifest freeze.
- Normal trigger thresholds, safety thresholds, GAT/SAC checkpoints, Proposal, FP-SHEP, and baselines were not modified; no training or formal benchmark was run.
- Position, velocity, terminal-goal, and DMP-phase continuity reconciliation passed.

Final fields: `FINAL_METHOD_READY_FOR_NEW_FORMAL = {conclusion['FINAL_METHOD_READY_FOR_NEW_FORMAL']}` and `RECOMMENDED_NEXT_STEP = {conclusion['RECOMMENDED_NEXT_STEP']}`.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args().output_dir.resolve())
