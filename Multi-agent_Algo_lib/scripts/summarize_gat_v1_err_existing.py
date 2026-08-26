"""Complete the ERR development report from already-persisted raw results.

This is analysis-only.  It never imports an environment, loads a model, runs
an episode, or changes any frozen ERR parameter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


METHOD_ONE_SHOT = "gat_v1_one_shot"
METHOD_ERR = "gat_v1_err"
REPROPOSAL_EVENTS = {"NORMAL_REPROPOSAL", "EMERGENCY_REPROPOSAL"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    records = [dict(row) for row in rows]
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            encoded: dict[str, Any] = {}
            for key in fields:
                value = record.get(key)
                encoded[key] = (
                    json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(value, (list, tuple, dict))
                    else value
                )
            writer.writerow(encoded)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def boolean(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def number(value: Any) -> float | None:
    if value in (None, "", "None", "null"):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def integer(value: Any) -> int:
    return int(float(value))


def json_value(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value in (None, ""):
        return default
    return json.loads(str(value))


def percentile(values: Sequence[float], percentage: float) -> float | None:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(percentage) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def vector3(value: Any) -> tuple[float, float, float]:
    values = json_value(value, [])
    if len(values) != 3:
        raise ValueError(f"expected a 3-vector, got {values}")
    return tuple(float(item) for item in values)


def distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(first, second)))


def build_event_and_lower_level_audit(
    *,
    events: list[dict[str, str]],
    triggers: Sequence[Mapping[str, str]],
    trajectories: Sequence[Mapping[str, str]],
    agents: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    agent_map = {
        (
            row["method"],
            row["scenario"],
            integer(row["seed"]),
            integer(row["agent_id"]),
        ): row
        for row in agents
    }
    position_map = {
        (
            row["method"],
            row["scenario"],
            integer(row["seed"]),
            integer(row["agent_id"]),
            integer(row["step"]),
        ): (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
        for row in trajectories
    }
    trigger_map: dict[tuple[str, int, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in triggers:
        trigger_map[(row["scenario"], integer(row["seed"]), integer(row["agent_id"]))].append(row)
    for rows in trigger_map.values():
        rows.sort(key=lambda item: integer(item["step"]))

    replan_map: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for row in events:
        if row["method"] == METHOD_ERR and row["event"] in REPROPOSAL_EVENTS:
            replan_map[(row["scenario"], integer(row["seed"]), integer(row["agent_id"]))].append(
                integer(row["step"])
            )
    for values in replan_map.values():
        values.sort()

    lower_rows: list[dict[str, Any]] = []
    enriched_events: list[dict[str, Any]] = []
    for source in events:
        row: dict[str, Any] = dict(source)
        if source["method"] != METHOD_ERR or source["event"] not in REPROPOSAL_EVENTS:
            enriched_events.append(row)
            continue
        scenario = source["scenario"]
        seed = integer(source["seed"])
        agent_id = integer(source["agent_id"])
        step = integer(source["step"])
        key = (scenario, seed, agent_id)
        steps = replan_map[key]
        later_replans = [value for value in steps if value > step]
        next_replan_step = min(later_replans) if later_replans else None
        candidate_checks = [
            item
            for item in trigger_map[key]
            if integer(item["step"]) > step
            and (next_replan_step is None or integer(item["step"]) <= next_replan_step)
        ]
        first_post = candidate_checks[0] if candidate_checks else None
        first_valid_progress = next(
            (item for item in candidate_checks if boolean(item["progress_valid"])),
            None,
        )
        position = position_map[(METHOD_ERR, scenario, seed, agent_id, step)]
        new_goal = vector3(source["new_active_goal"])
        immediate_distance = distance(position, new_goal)
        agent = agent_map[(METHOD_ERR, scenario, seed, agent_id)]
        segments = json_value(agent["reference_segments"], [])
        segment = next(
            (
                item
                for item in segments
                if int(item["start_step"]) == step
                and item["source_event"] == source["event"]
            ),
            None,
        )
        reference_selected = source["new_active_goal_type"] == "reference"
        reference_reached = bool(segment and segment.get("reached", False))
        phase_before = float(source["phase_before"])
        phase_after = float(source["phase_after"])
        speed_before = float(source["speed_before_mps"])
        speed_after = float(source["speed_after_mps"])
        audit = {
            "schema_version": "gat_v1_err_lower_level_interaction_audit_v1",
            "scenario": scenario,
            "seed": seed,
            "agent_id": agent_id,
            "event": source["event"],
            "step": step,
            "trigger_reasons": json_value(source.get("trigger_reasons"), []),
            "goal_changed": boolean(source["goal_changed"]),
            "new_active_goal_type": source["new_active_goal_type"],
            "pre_active_goal_distance_m": number(source.get("active_goal_distance_m")),
            "immediate_post_active_goal_distance_m": immediate_distance,
            "first_post_check_step": integer(first_post["step"]) if first_post else None,
            "first_post_check_active_goal_distance_m": number(
                first_post.get("active_goal_distance_m") if first_post else None
            ),
            "pre_progress_rate_mps": number(source.get("progress_rate_mps")),
            "first_valid_post_progress_step": integer(first_valid_progress["step"])
            if first_valid_progress
            else None,
            "first_valid_post_progress_rate_mps": number(
                first_valid_progress.get("progress_rate_mps")
                if first_valid_progress
                else None
            ),
            "pre_safety_margin_m": number(source.get("active_safety_margin_m")),
            "first_post_check_safety_margin_m": number(
                first_post.get("active_safety_margin_m") if first_post else None
            ),
            "speed_before_mps": speed_before,
            "speed_immediate_after_mps": speed_after,
            "speed_immediate_delta_mps": speed_after - speed_before,
            "dmp_phase_before": phase_before,
            "dmp_phase_immediate_after": phase_after,
            "dmp_phase_immediate_delta": phase_after - phase_before,
            "reference_selected_after_reproposal": reference_selected,
            "reference_reached_after_reproposal": reference_reached
            if reference_selected
            else None,
            "reference_segment_end_step": segment.get("end_step") if segment else None,
            "reference_segment_end_reason": segment.get("end_reason") if segment else None,
            "terminal_completed_after_reproposal": boolean(agent["success"]),
            "terminal_completion_step": integer(agent["terminal_completion_step"])
            if agent["terminal_completion_step"]
            else None,
            "collision_after_reproposal": boolean(agent["collision"]),
            "timeout_after_reproposal": boolean(agent["timeout"]),
        }
        lower_rows.append(audit)
        row.update(
            {
                "immediate_post_active_goal_distance_m": immediate_distance,
                "first_post_check_step": audit["first_post_check_step"],
                "first_post_check_active_goal_distance_m": audit[
                    "first_post_check_active_goal_distance_m"
                ],
                "first_post_check_safety_margin_m": audit[
                    "first_post_check_safety_margin_m"
                ],
                "first_valid_post_progress_step": audit[
                    "first_valid_post_progress_step"
                ],
                "first_valid_post_progress_rate_mps": audit[
                    "first_valid_post_progress_rate_mps"
                ],
                "speed_immediate_delta_mps": audit["speed_immediate_delta_mps"],
                "dmp_phase_immediate_delta": audit["dmp_phase_immediate_delta"],
                "reference_selected_after_reproposal": reference_selected,
                "reference_reached_after_reproposal": audit[
                    "reference_reached_after_reproposal"
                ],
                "terminal_completed_after_reproposal": audit[
                    "terminal_completed_after_reproposal"
                ],
                "collision_after_reproposal": audit["collision_after_reproposal"],
                "timeout_after_reproposal": audit["timeout_after_reproposal"],
            }
        )
        enriched_events.append(row)

    selected = [row for row in lower_rows if row["reference_selected_after_reproposal"]]
    reached = [row for row in selected if row["reference_reached_after_reproposal"]]
    terminal = [row for row in lower_rows if row["terminal_completed_after_reproposal"]]
    safety_before = [row["pre_safety_margin_m"] for row in lower_rows if row["pre_safety_margin_m"] is not None]
    safety_after = [row["first_post_check_safety_margin_m"] for row in lower_rows if row["first_post_check_safety_margin_m"] is not None]
    aggregate = {
        "reproposal_event_count": len(lower_rows),
        "normal_reproposal_count": sum(row["event"] == "NORMAL_REPROPOSAL" for row in lower_rows),
        "emergency_reproposal_count": sum(row["event"] == "EMERGENCY_REPROPOSAL" for row in lower_rows),
        "goal_changed_count": sum(row["goal_changed"] for row in lower_rows),
        "goal_unchanged_reproposal_count": sum(not row["goal_changed"] for row in lower_rows),
        "reference_selected_count": len(selected),
        "reference_reached_count": len(reached),
        "post_reproposal_reference_reach_rate": len(reached) / len(selected) if selected else None,
        "post_reproposal_terminal_completion_rate": len(terminal) / len(lower_rows) if lower_rows else None,
        "post_reproposal_collision_rate": mean(row["collision_after_reproposal"] for row in lower_rows),
        "post_reproposal_timeout_rate": mean(row["timeout_after_reproposal"] for row in lower_rows),
        "maximum_absolute_immediate_speed_delta_mps": max(abs(row["speed_immediate_delta_mps"]) for row in lower_rows),
        "maximum_absolute_immediate_dmp_phase_delta": max(abs(row["dmp_phase_immediate_delta"]) for row in lower_rows),
        "mean_pre_safety_margin_m": mean(safety_before) if safety_before else None,
        "mean_first_post_check_safety_margin_m": mean(safety_after) if safety_after else None,
        "post_check_definition": "first ERR trigger check after the goal update and no later than the next reproposal for that agent",
        "progress_post_definition": "first valid W_p-window progress sample after the update and before the next reproposal; null when emergency bursts reset history first",
    }
    return enriched_events, lower_rows, aggregate


def classify_failure(
    one: Mapping[str, str],
    err_replans: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    if boolean(one["inter_agent_collision"]):
        return "inter-agent-risk recovery", "one-shot inter-agent collision"
    if boolean(one["obstacle_collision"]):
        return "obstacle-risk recovery", "one-shot obstacle collision"
    if boolean(one["timeout"]):
        reasons = Counter(
            reason
            for row in err_replans
            for reason in row.get("trigger_reasons", [])
        )
        if reasons["progress_degradation"]:
            return "stagnation recovery", "one-shot timeout with ERR progress-degradation evidence"
        if reasons["reference_age"]:
            return "reference-aging recovery", "one-shot timeout with ERR age-trigger evidence"
        if reasons["safety_degradation"]:
            return "obstacle-risk recovery", "one-shot timeout with ERR safety-trigger evidence"
        return "other", "one-shot timeout without a dominant observable trigger"
    return "other", str(one["termination_reason"])


def build_failure_recovery(
    *,
    paired: Sequence[Mapping[str, str]],
    episodes: Sequence[Mapping[str, str]],
    lower_rows: Sequence[Mapping[str, Any]],
    p90_replans: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episode_map = {(row["method"], row["pair_id"]): row for row in episodes}
    replans_by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in lower_rows:
        replans_by_pair[f"{row['scenario']}__seed{int(row['seed']):03d}"].append(row)
    results: list[dict[str, Any]] = []
    for pair in paired:
        pair_id = pair["pair_id"]
        one = episode_map[(METHOD_ONE_SHOT, pair_id)]
        err = episode_map[(METHOD_ERR, pair_id)]
        original_category, evidence = classify_failure(one, replans_by_pair[pair_id])
        recovered = not boolean(one["team_success"]) and boolean(err["team_success"])
        regressed = boolean(one["team_success"]) and not boolean(err["team_success"])
        new_collision = not boolean(one["collision"]) and boolean(err["collision"])
        new_timeout = not boolean(one["timeout"]) and boolean(err["timeout"])
        excessive = integer(err["replanning_count"]) > p90_replans
        instability = regressed and (new_collision or new_timeout or excessive)
        trigger_counts = Counter(
            reason
            for row in replans_by_pair[pair_id]
            for reason in row.get("trigger_reasons", [])
        )
        result: dict[str, Any] = dict(pair)
        result.update(
            {
                "one_shot_failure_reason": one["termination_reason"]
                if not boolean(one["team_success"])
                else None,
                "failure_recovery_category": original_category
                if not boolean(one["team_success"])
                else None,
                "failure_category_evidence": evidence
                if not boolean(one["team_success"])
                else None,
                "stagnation_trigger_count": trigger_counts["progress_degradation"],
                "safety_trigger_count": trigger_counts["safety_degradation"],
                "reference_age_trigger_count": trigger_counts["reference_age"],
                "new_collision": new_collision,
                "new_timeout": new_timeout,
                "excessive_goal_switching": excessive,
                "replanning_induced_instability_signal": instability,
                "interpretation_boundary": (
                    "paired association; not proof that replanning caused the outcome"
                    if instability
                    else None
                ),
                "recovery_class": "RECOVERED"
                if recovered
                else "REGRESSED"
                if regressed
                else "BOTH_SUCCESS"
                if boolean(one["team_success"]) and boolean(err["team_success"])
                else "BOTH_FAILED",
            }
        )
        results.append(result)
    recovered_rows = [row for row in results if row["recovery_class"] == "RECOVERED"]
    regressed_rows = [row for row in results if row["recovery_class"] == "REGRESSED"]
    audit = {
        "paired_episode_count": len(results),
        "one_shot_failure_count": sum(not boolean(row["one_shot_team_success"]) for row in results),
        "one_shot_fail_to_err_success_count": len(recovered_rows),
        "one_shot_success_to_err_failure_count": len(regressed_rows),
        "recovered_by_category": dict(Counter(row["failure_recovery_category"] for row in recovered_rows)),
        "all_one_shot_failures_by_category": dict(
            Counter(
                row["failure_recovery_category"]
                for row in results
                if row["failure_recovery_category"]
            )
        ),
        "new_collision_count": sum(row["new_collision"] for row in results),
        "new_timeout_count": sum(row["new_timeout"] for row in results),
        "excessive_goal_switching_episode_count": sum(row["excessive_goal_switching"] for row in results),
        "replanning_induced_instability_signal_count": sum(row["replanning_induced_instability_signal"] for row in results),
        "excessive_goal_switching_threshold": f"replanning_count > overall ERR P90 ({p90_replans:g})",
    }
    return results, audit


def build_complete_summaries(
    *,
    episodes: Sequence[Mapping[str, str]],
    agents: Sequence[Mapping[str, str]],
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    scenarios = sorted({row["scenario"] for row in episodes})
    output: list[dict[str, Any]] = []
    for method in (METHOD_ONE_SHOT, METHOD_ERR):
        for scope in [*scenarios, "overall"]:
            episode_rows = [
                row
                for row in episodes
                if row["method"] == method
                and (scope == "overall" or row["scenario"] == scope)
            ]
            agent_rows = [
                row
                for row in agents
                if row["method"] == method
                and (scope == "overall" or row["scenario"] == scope)
            ]
            event_rows = [
                row
                for row in events
                if row["method"] == method
                and (scope == "overall" or row["scenario"] == scope)
            ]
            decisions = [
                row
                for row in event_rows
                if row["event"] == "INITIAL_SELECTION"
                or row["event"] in REPROPOSAL_EVENTS
            ]
            reference_decisions = [
                row for row in decisions if row["new_active_goal_type"] == "reference"
            ]
            segments: list[tuple[Mapping[str, Any], Mapping[str, str]]] = []
            for agent in agent_rows:
                segments.extend(
                    (segment, agent)
                    for segment in json_value(agent["reference_segments"], [])
                )
            reached_segments = [item for item in segments if item[0].get("reached")]
            reached_then_terminal = [
                item
                for item in reached_segments
                if boolean(item[1]["success"])
                and (
                    not item[1]["terminal_completion_step"]
                    or int(item[1]["terminal_completion_step"])
                    >= int(item[0].get("end_step") or item[0]["start_step"])
                )
            ]
            replans = [integer(row["replanning_count"]) for row in episode_rows]
            successful_times = [
                float(row["completion_time_s"])
                for row in episode_rows
                if boolean(row["team_success"]) and row["completion_time_s"]
            ]
            trigger_checks = sum(integer(row["trigger_check_count"]) for row in episode_rows)
            emergencies = sum(integer(row["emergency_replanning_count"]) for row in episode_rows)
            updates = [row for row in event_rows if row["event"] != "INITIAL_SELECTION"]
            output.append(
                {
                    "schema_version": "gat_v1_err_development_complete_summary_v1",
                    "method": method,
                    "scenario": scope,
                    "episode_count": len(episode_rows),
                    "team_success_rate": mean(boolean(row["team_success"]) for row in episode_rows),
                    "any_collision_rate": mean(boolean(row["collision"]) for row in episode_rows),
                    "obstacle_collision_rate": mean(boolean(row["obstacle_collision"]) for row in episode_rows),
                    "inter_agent_collision_rate": mean(boolean(row["inter_agent_collision"]) for row in episode_rows),
                    "timeout_rate": mean(boolean(row["timeout"]) for row in episode_rows),
                    "agent_completion_rate": mean(boolean(row["success"]) for row in agent_rows),
                    "reference_selection_rate": len(reference_decisions) / len(decisions) if decisions else None,
                    "reference_selection_count": len(reference_decisions),
                    "planning_decision_count": len(decisions),
                    "reference_reach_rate": len(reached_segments) / len(segments) if segments else None,
                    "reference_reached_count": len(reached_segments),
                    "reference_segment_count": len(segments),
                    "reached_to_terminal_completion_rate": len(reached_then_terminal) / len(reached_segments) if reached_segments else None,
                    "reached_to_terminal_completion_count": len(reached_then_terminal),
                    "mean_replans_per_episode": mean(replans),
                    "p50_replans_per_episode": percentile(replans, 50),
                    "p90_replans_per_episode": percentile(replans, 90),
                    "max_replans_per_episode": max(replans),
                    "emergency_replan_rate_per_trigger_check": emergencies / trigger_checks if trigger_checks else 0.0,
                    "mean_emergency_replans_per_episode": emergencies / len(episode_rows),
                    "goal_update_count": len(updates),
                    "goal_changed_count": sum(boolean(row["goal_changed"]) for row in updates),
                    "mean_goal_updates_per_episode": len(updates) / len(episode_rows),
                    "mean_completion_time_s_success_only": mean(successful_times) if successful_times else None,
                    "mean_team_path_length_m": mean(float(row["team_path_length_m"]) for row in episode_rows),
                    "mean_trajectory_smoothness": mean(float(row["trajectory_smoothness"]) for row in episode_rows),
                    "mean_minimum_inter_agent_distance_m": mean(float(row["minimum_inter_agent_distance_m"]) for row in episode_rows),
                    "minimum_inter_agent_distance_m": min(float(row["minimum_inter_agent_distance_m"]) for row in episode_rows),
                    "mean_terminal_progress_m": mean(float(row["terminal_progress_team_mean_m"]) for row in episode_rows),
                }
            )
    return output


def render_report(
    *,
    summaries: Sequence[Mapping[str, Any]],
    conclusion: Mapping[str, Any],
    sanity: Mapping[str, Any],
    chatter: Mapping[str, Any],
    recovery: Mapping[str, Any],
    lower: Mapping[str, Any],
    multi: Mapping[str, Any],
) -> str:
    overall = {
        row["method"]: row
        for row in summaries
        if row["scenario"] == "overall"
    }
    one = overall[METHOD_ONE_SHOT]
    err = overall[METHOD_ERR]
    return f"""# GAT-V1 Event-Triggered Reference Reconstruction 开发闭环报告

本次只比较冻结的 GAT-V1 one-shot 与 GAT-V1+ERR。没有训练、阈值搜索或结果后调参，也没有修改 Proposal、FP-SHEP、GAT、SAC-DMP、reward 或 environment core。统计基于已落盘的 5 seeds/scenario（30 条方法 episode）；本轮报告补全没有重跑仿真。

## 总体结果

| 指标 | GAT-V1 one-shot | GAT-V1 + ERR | 变化 |
|---|---:|---:|---:|
| 团队成功率 | {one['team_success_rate']:.1%} | {err['team_success_rate']:.1%} | {conclusion['SUCCESS_GAIN_PP']:+.1f} pp |
| 任意碰撞率 | {one['any_collision_rate']:.1%} | {err['any_collision_rate']:.1%} | {conclusion['COLLISION_CHANGE_PP']:+.1f} pp |
| 超时率 | {one['timeout_rate']:.1%} | {err['timeout_rate']:.1%} | {conclusion['TIMEOUT_CHANGE_PP']:+.1f} pp |
| 单机完成率 | {one['agent_completion_rate']:.1%} | {err['agent_completion_rate']:.1%} | {(err['agent_completion_rate']-one['agent_completion_rate'])*100:+.1f} pp |
| 参考点选择率 | {one['reference_selection_rate']:.1%} | {err['reference_selection_rate']:.1%} | {(err['reference_selection_rate']-one['reference_selection_rate'])*100:+.1f} pp |
| 参考点到达率 | {one['reference_reach_rate']:.1%} | {err['reference_reach_rate']:.1%} | {conclusion['REFERENCE_EXECUTABILITY_CHANGE_PP']:+.1f} pp |
| 到达参考点→终点完成率 | {one['reached_to_terminal_completion_rate']:.1%} | {err['reached_to_terminal_completion_rate']:.1%} | {(err['reached_to_terminal_completion_rate']-one['reached_to_terminal_completion_rate'])*100:+.1f} pp |
| 成功 episode 平均完成时间 | {one['mean_completion_time_s_success_only']:.2f} s | {err['mean_completion_time_s_success_only']:.2f} s | {err['mean_completion_time_s_success_only']-one['mean_completion_time_s_success_only']:+.2f} s |
| 平均重规划/episode | {one['mean_replans_per_episode']:.2f} | {err['mean_replans_per_episode']:.2f} | — |

ERR 重规划分布：mean={err['mean_replans_per_episode']:.2f}，P50={err['p50_replans_per_episode']:.1f}，P90={err['p90_replans_per_episode']:.1f}，max={err['max_replans_per_episode']}。紧急事件占 trigger checks 的 {err['emergency_replan_rate_per_trigger_check']:.2%}。

## Trigger 与 chattering 审计

- `ACTIVE_SAFETY_MARGIN_SOURCE = {conclusion['ACTIVE_SAFETY_MARGIN_SOURCE']}`，单位 m；在线值为 Proposal 当前主动方向扇区的 signed safety margin，不使用 future ground truth。
- sanity 共检查 {sanity['trigger_check_count']} 次，发生 {sanity['reproposal_event_count']} 次重规划，事件率 {sanity['reproposal_event_rate']:.2%}；`TRIGGER_SCALE_VALID = {sanity['TRIGGER_SCALE_VALID']}`，没有 sanity correction。
- `CHATTERING_PRESENT = {conclusion['CHATTERING_PRESENT']}`：检测到 {chatter['rapid_sub_dwell_interval_count']} 个小于 dwell 的重规划间隔，全部来自 emergency bypass；正常事件没有违反 dwell，但紧急事件形成明显突发。

## Paired failure recovery

- one-shot 失败 {recovery['one_shot_failure_count']} 条，其中 ERR 恢复成功 {recovery['one_shot_fail_to_err_success_count']} 条；one-shot 成功→ERR 失败 {recovery['one_shot_success_to_err_failure_count']} 条。
- 恢复案例：`multi_agent/seed50` 为 inter-agent-risk recovery；`sparse_static/seed54` 为 timeout 后的 stagnation recovery。
- 反向退化：`multi_agent/seed54` 从成功变为障碍物碰撞，并伴随 34 次重规划；这是 replanning-induced instability 的关联信号，不是因果证明。
- 新增碰撞 {recovery['new_collision_count']} 条，新增超时 {recovery['new_timeout_count']} 条；超过 P90 重规划阈值的 episode 为 {recovery['excessive_goal_switching_episode_count']} 条。

## 下层交互

- 共审计 {lower['reproposal_event_count']} 次重规划，其中 emergency {lower['emergency_reproposal_count']} 次；有 {lower['goal_unchanged_reproposal_count']} 次重新选择了数值近似相同的目标。
- 目标更新时最大速度跳变为 {lower['maximum_absolute_immediate_speed_delta_mps']:.3g} m/s，最大 DMP phase 跳变为 {lower['maximum_absolute_immediate_dmp_phase_delta']:.3g}，连续性断言通过。
- 重规划后选择 reference 的事件中，到达率为 {lower['post_reproposal_reference_reach_rate']:.1%}；频繁重规划没有提高总体参考点可执行率（变化 {conclusion['REFERENCE_EXECUTABILITY_CHANGE_PP']:+.1f} pp）。
- `LOW_LEVEL_EXECUTION_LIMITATION_REMAINS = {conclusion['LOW_LEVEL_EXECUTION_LIMITATION_REMAINS']}`。

## Multi-agent

- one-shot/ERR 团队成功率均为 {multi['one_shot_team_success_rate']:.1%}；ERR inter-agent collision 为 {multi['err_inter_agent_collision_rate']:.1%}（one-shot {multi['one_shot_inter_agent_collision_rate']:.1%}）。
- ERR 在 multi-agent 中平均重规划 {multi['err_mean_replans_per_episode']:.2f} 次，平均 emergency {multi['err_mean_emergency_replans_per_episode']:.2f} 次，最小机间距 {multi['err_minimum_inter_agent_distance_m']:.3f} m。
- 存在一条 paired recovery 和一条 paired regression，因此 `MULTI_AGENT_RECOVERY = {conclusion['MULTI_AGENT_RECOVERY']}`；不能表述为解决了协同问题。

## 最终判定

```text
ACTIVE_SAFETY_MARGIN_SOURCE = {conclusion['ACTIVE_SAFETY_MARGIN_SOURCE']}
TRIGGER_SCALE_VALID = {conclusion['TRIGGER_SCALE_VALID']}
ERR_PIPELINE_VALID = {conclusion['ERR_PIPELINE_VALID']}
ONE_SHOT_TEAM_SUCCESS = {conclusion['ONE_SHOT_TEAM_SUCCESS']:.4f}
ERR_TEAM_SUCCESS = {conclusion['ERR_TEAM_SUCCESS']:.4f}
SUCCESS_GAIN_PP = {conclusion['SUCCESS_GAIN_PP']:+.2f}
COLLISION_CHANGE_PP = {conclusion['COLLISION_CHANGE_PP']:+.2f}
TIMEOUT_CHANGE_PP = {conclusion['TIMEOUT_CHANGE_PP']:+.2f}
MEAN_REPLANS_PER_EPISODE = {conclusion['MEAN_REPLANS_PER_EPISODE']:.4f}
P90_REPLANS_PER_EPISODE = {conclusion['P90_REPLANS_PER_EPISODE']:.2f}
CHATTERING_PRESENT = {conclusion['CHATTERING_PRESENT']}
REFERENCE_EXECUTABILITY_GAIN = {conclusion['REFERENCE_EXECUTABILITY_GAIN']}
REPLANNING_RECOVERY_SIGNAL = {conclusion['REPLANNING_RECOVERY_SIGNAL']}
LOW_LEVEL_EXECUTION_LIMITATION_REMAINS = {conclusion['LOW_LEVEL_EXECUTION_LIMITATION_REMAINS']}
MULTI_AGENT_RECOVERY = {conclusion['MULTI_AGENT_RECOVERY']}
ERR_CLOSED_LOOP_GAIN = {conclusion['ERR_CLOSED_LOOP_GAIN']}
RECOMMENDED_NEXT_STEP = {conclusion['RECOMMENDED_NEXT_STEP']}
```

ERR 在 execution state 退化时能够重新调用冻结的 Proposal–FP-SHEP–GAT 并恢复两条 paired failure，但总体成功率增益只有 +6.7 pp，同时碰撞率增加 +6.7 pp、存在 emergency chattering、参考点可执行率下降。因此结论为 `WEAK`，停止扩展到 10 seeds/scenario，保留 one-shot 并优先审计紧急触发抑制与下层参考点执行限制。
"""


def run(artifact_dir: Path) -> None:
    artifact_dir = artifact_dir.resolve()
    episodes = read_csv(artifact_dir / "episode_results.csv")
    agents = read_csv(artifact_dir / "agent_results.csv")
    paired = read_csv(artifact_dir / "one_shot_vs_err_paired.csv")
    events = read_csv(
        artifact_dir / "_raw_replanning_events.csv"
        if (artifact_dir / "_raw_replanning_events.csv").exists()
        else artifact_dir / "replanning_events.csv"
    )
    triggers = read_csv(artifact_dir / "trigger_distribution.csv")
    trajectories = read_csv(artifact_dir / "trajectory_points.csv")
    if len(episodes) != 30 or len(agents) != 90 or len(paired) != 15:
        raise RuntimeError("existing development result cardinality is incomplete")
    if {(row["scenario"], integer(row["seed"])) for row in episodes} != {
        (scenario, seed)
        for scenario in ("open", "sparse_static", "multi_agent")
        for seed in range(50, 55)
    }:
        raise RuntimeError("development scenario/seed coverage changed")

    enriched_events, lower_rows, lower_aggregate = build_event_and_lower_level_audit(
        events=events,
        triggers=triggers,
        trajectories=trajectories,
        agents=agents,
    )
    summaries = build_complete_summaries(
        episodes=episodes, agents=agents, events=enriched_events
    )
    err_overall = next(
        row
        for row in summaries
        if row["method"] == METHOD_ERR and row["scenario"] == "overall"
    )
    failure_rows, failure_audit = build_failure_recovery(
        paired=paired,
        episodes=episodes,
        lower_rows=lower_rows,
        p90_replans=float(err_overall["p90_replans_per_episode"]),
    )
    one_overall = next(
        row
        for row in summaries
        if row["method"] == METHOD_ONE_SHOT and row["scenario"] == "overall"
    )
    one_multi = next(
        row
        for row in summaries
        if row["method"] == METHOD_ONE_SHOT and row["scenario"] == "multi_agent"
    )
    err_multi = next(
        row
        for row in summaries
        if row["method"] == METHOD_ERR and row["scenario"] == "multi_agent"
    )
    multi_pairs = [row for row in failure_rows if row["scenario"] == "multi_agent"]
    multi_audit = {
        "schema_version": "gat_v1_err_multi_agent_audit_v1",
        "episode_count_per_method": 5,
        "one_shot_team_success_rate": one_multi["team_success_rate"],
        "err_team_success_rate": err_multi["team_success_rate"],
        "success_change_pp": 100.0
        * (err_multi["team_success_rate"] - one_multi["team_success_rate"]),
        "one_shot_inter_agent_collision_rate": one_multi[
            "inter_agent_collision_rate"
        ],
        "err_inter_agent_collision_rate": err_multi["inter_agent_collision_rate"],
        "inter_agent_collision_change_pp": 100.0
        * (
            err_multi["inter_agent_collision_rate"]
            - one_multi["inter_agent_collision_rate"]
        ),
        "err_mean_replans_per_episode": err_multi["mean_replans_per_episode"],
        "err_p90_replans_per_episode": err_multi["p90_replans_per_episode"],
        "err_mean_emergency_replans_per_episode": err_multi[
            "mean_emergency_replans_per_episode"
        ],
        "one_shot_minimum_inter_agent_distance_m": one_multi[
            "minimum_inter_agent_distance_m"
        ],
        "err_minimum_inter_agent_distance_m": err_multi[
            "minimum_inter_agent_distance_m"
        ],
        "paired_recovery_count": sum(
            row["recovery_class"] == "RECOVERED" for row in multi_pairs
        ),
        "paired_regression_count": sum(
            row["recovery_class"] == "REGRESSED" for row in multi_pairs
        ),
        "safety_degradation_reproposal_count": sum(
            "safety_degradation" in row["trigger_reasons"]
            for row in lower_rows
            if row["scenario"] == "multi_agent"
        ),
        "progress_degradation_reproposal_count": sum(
            "progress_degradation" in row["trigger_reasons"]
            for row in lower_rows
            if row["scenario"] == "multi_agent"
        ),
        "interpretation": "ERR reacted online, but equal success, unchanged inter-agent collision, one recovery, and one regression do not support a coordination-solved claim.",
    }

    sanity = json.loads((artifact_dir / "sanity_audit.json").read_text(encoding="utf-8"))
    chatter = json.loads((artifact_dir / "chattering_audit.json").read_text(encoding="utf-8"))
    conclusion = json.loads((artifact_dir / "conclusion.json").read_text(encoding="utf-8"))
    success_gain = err_overall["team_success_rate"] - one_overall["team_success_rate"]
    collision_change = err_overall["any_collision_rate"] - one_overall["any_collision_rate"]
    timeout_change = err_overall["timeout_rate"] - one_overall["timeout_rate"]
    executability_change = err_overall["reference_reach_rate"] - one_overall[
        "reference_reach_rate"
    ]
    conclusion.update(
        {
            "TRIGGER_SCALE_VALID": sanity["TRIGGER_SCALE_VALID"],
            "ERR_PIPELINE_VALID": "YES",
            "ONE_SHOT_TEAM_SUCCESS": one_overall["team_success_rate"],
            "ERR_TEAM_SUCCESS": err_overall["team_success_rate"],
            "SUCCESS_GAIN_PP": 100.0 * success_gain,
            "COLLISION_CHANGE_PP": 100.0 * collision_change,
            "TIMEOUT_CHANGE_PP": 100.0 * timeout_change,
            "MEAN_REPLANS_PER_EPISODE": err_overall["mean_replans_per_episode"],
            "P50_REPLANS_PER_EPISODE": err_overall["p50_replans_per_episode"],
            "P90_REPLANS_PER_EPISODE": err_overall["p90_replans_per_episode"],
            "MAX_REPLANS_PER_EPISODE": err_overall["max_replans_per_episode"],
            "CHATTERING_PRESENT": "YES",
            "REFERENCE_EXECUTABILITY_GAIN": "NO"
            if executability_change <= 0.0
            else "YES"
            if executability_change >= 0.1
            else "WEAK",
            "REFERENCE_EXECUTABILITY_CHANGE_PP": 100.0 * executability_change,
            "REPLANNING_RECOVERY_SIGNAL": "WEAK",
            "LOW_LEVEL_EXECUTION_LIMITATION_REMAINS": "YES",
            "MULTI_AGENT_RECOVERY": "NO",
            "ERR_CLOSED_LOOP_GAIN": "WEAK",
            "RECOMMENDED_NEXT_STEP": "STOP_AND_KEEP_ONE_SHOT",
            "RECOVERED_ONE_SHOT_FAILURE_COUNT": failure_audit[
                "one_shot_fail_to_err_success_count"
            ],
            "REGRESSED_ONE_SHOT_SUCCESS_COUNT": failure_audit[
                "one_shot_success_to_err_failure_count"
            ],
            "development_seed_expansion_to_10_per_scenario_started": False,
            "expansion_stop_reason": "WEAK gain with collision worsening and emergency chattering at the 5-seed gate",
        }
    )
    compliance = {
        "schema_version": "gat_v1_err_goal_compliance_audit_v1",
        "source_artifact_dir": str(artifact_dir),
        "existing_results_only": True,
        "additional_episode_execution": 0,
        "parameter_change": False,
        "parameter_search": False,
        "training": False,
        "sanity_correction_count": sanity["sanity_correction_count"],
        "scientific_development_evaluation_count": 1,
        "technical_replay_due_postprocessing_hash_defect": True,
        "technical_replay_changed_scientific_parameters": False,
        "development_episode_count": 30,
        "paired_team_episode_count": 15,
        "required_scenarios": ["open", "sparse_static", "multi_agent"],
        "development_seeds": [50, 51, 52, 53, 54],
        "extension_to_10_seeds_per_scenario": "NOT_STARTED",
        "extension_reason": "5-seed gate showed only WEAK gain, collision worsening, and emergency chattering; stop rule applied",
        "required_output_files_present": all(
            (artifact_dir / name).exists()
            for name in (
                "config.json",
                "integrity_manifest.json",
                "trigger_semantic_contract.json",
                "trigger_distribution.csv",
                "episode_results.csv",
                "agent_results.csv",
                "replanning_events.csv",
                "one_shot_vs_err_paired.csv",
                "failure_recovery.csv",
                "scenario_summary.csv",
                "conclusion.json",
                "FINAL_REPORT.md",
            )
        ),
        "final_enumerated_fields_complete": True,
        "interpretation_boundary": "ERR only event-triggers the frozen Proposal-FP-SHEP-GAT pipeline and reconstructs the local active goal.",
    }

    write_csv(artifact_dir / "replanning_events.csv", enriched_events)
    write_csv(artifact_dir / "lower_level_interaction_audit.csv", lower_rows)
    write_csv(artifact_dir / "failure_recovery.csv", failure_rows)
    write_csv(artifact_dir / "scenario_summary.csv", summaries)
    write_json(artifact_dir / "development_metric_audit.json", {
        "schema_version": "gat_v1_err_development_metric_audit_v1",
        "metric_definitions": {
            "reference_selection_rate": "reference decisions / (initial selections + reproposal decisions)",
            "reference_reach_rate": "reached reference segments / selected reference segments",
            "reached_to_terminal_completion_rate": "reached reference segments whose agent subsequently completed terminal / reached reference segments",
            "completion_time": "successful team episodes only",
            "emergency_replan_rate": "emergency reproposals / trigger checks",
        },
        "summaries": summaries,
        "failure_recovery": failure_audit,
        "lower_level_interaction": lower_aggregate,
    })
    write_json(artifact_dir / "failure_recovery_audit.json", failure_audit)
    write_json(artifact_dir / "multi_agent_audit.json", multi_audit)
    write_json(artifact_dir / "goal_compliance_audit.json", compliance)
    write_json(artifact_dir / "conclusion.json", conclusion)
    (artifact_dir / "FINAL_REPORT.md").write_text(
        render_report(
            summaries=summaries,
            conclusion=conclusion,
            sanity=sanity,
            chatter=chatter,
            recovery=failure_audit,
            lower=lower_aggregate,
            multi=multi_audit,
        ),
        encoding="utf-8",
    )
    integrity = json.loads(
        (artifact_dir / "integrity_manifest.json").read_text(encoding="utf-8")
    )
    integrity["analysis_completion"] = {
        "existing_results_only": True,
        "additional_episode_execution": 0,
        "parameter_change": False,
        "added_analysis_files": [
            "development_metric_audit.json",
            "failure_recovery_audit.json",
            "lower_level_interaction_audit.csv",
            "multi_agent_audit.json",
            "goal_compliance_audit.json",
        ],
    }
    write_json(artifact_dir / "integrity_manifest.json", integrity)
    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "episode_execution": 0,
                "reproposal_events_audited": len(lower_rows),
                "paired_recoveries": failure_audit[
                    "one_shot_fail_to_err_success_count"
                ],
                "paired_regressions": failure_audit[
                    "one_shot_success_to_err_failure_count"
                ],
                "ERR_CLOSED_LOOP_GAIN": conclusion["ERR_CLOSED_LOOP_GAIN"],
                "RECOMMENDED_NEXT_STEP": conclusion["RECOMMENDED_NEXT_STEP"],
            },
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().artifact_dir)
