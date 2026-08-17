"""Analyze Pre-GAT closed-loop records and render publication-facing artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pre_gat_closed_loop_constants import (  # noqa: E402
    METHOD_DISPLAY_NAMES,
    METHOD_FP_SHEP,
    METHOD_FROZEN,
    METHOD_ORDER,
    METHOD_PROPOSAL,
)


RATE_METRICS = {
    "success": "Success Rate (%)",
    "collision": "Collision Rate (%)",
    "inter_agent_collision": "Inter-Agent Collision Rate (%)",
    "obstacle_collision": "Obstacle Collision Rate (%)",
}
CONTINUOUS_METRICS = {
    "path_length_success_team_mean_m": ("Path Length", "m", True),
    "completion_time_success_s": ("Completion Time", "s", True),
    "minimum_inter_agent_distance_m": ("Minimum Inter-Agent Distance", "m", False),
    "minimum_obstacle_clearance_m": ("Minimum Obstacle Clearance", "m", False),
    "trajectory_smoothness_team_mean": ("Trajectory Smoothness", "m^2/s^6", False),
    "trajectory_smoothness_success_team_mean": (
        "Trajectory Smoothness of Successful Episodes", "m^2/s^6", True
    ),
    "goal_switch_count": ("Goal Switch Count", "count", False),
    "mean_goal_jump_m": ("Mean Goal Jump", "m", False),
}


def _coerce(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "null":
        return "null"
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return [
            {key: _coerce(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(_jsonable(row.get(key)), ensure_ascii=False)
                if isinstance(row.get(key), (dict, list, tuple, np.ndarray))
                else row.get(key)
                for key in fields
            })


def _finite(values: Iterable[Any]) -> np.ndarray:
    result = np.asarray([value for value in values if value is not None], dtype=float)
    return result[np.isfinite(result)]


def _continuous_summary(values: Iterable[Any]) -> dict[str, Any]:
    array = _finite(values)
    if not array.size:
        return {"mean": None, "std": None, "median": None, "n": 0, "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    half = 1.96 * std / math.sqrt(array.size) if array.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "median": float(np.median(array)),
        "n": int(array.size),
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = 1.96
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


def aggregate_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"episode_count": len(rows)}
    for key in RATE_METRICS:
        count = int(sum(bool(row[key]) for row in rows))
        low, high = _wilson(count, len(rows))
        result.update({
            f"{key}_count": count,
            f"{key}_total": len(rows),
            f"{key}_rate": count / len(rows) if rows else None,
            f"{key}_ci95_low": low,
            f"{key}_ci95_high": high,
        })
    for key in CONTINUOUS_METRICS:
        summary = _continuous_summary(row.get(key) for row in rows)
        for statistic, value in summary.items():
            result[f"{key}_{statistic}"] = value
    for key in (
        "unsafe_step_count", "unsafe_duration_s", "upper_replanning_cycle_count",
        "upper_replanning_agent_event_count", "same_goal_retention_count",
        "maximum_goal_jump_m", "reference_reached_early_count",
        "reference_hold_steps_after_reached_sum", "no_candidate_fallback_count",
        "no_candidate_fallback_rate", "proposal_fp_shep_disagreement_rate",
        "candidate_generation_runtime_mean_ms", "fp_shep_preview_runtime_mean_ms",
        "candidate_selection_runtime_mean_ms", "upper_replanning_runtime_mean_ms",
        "frozen_sac_inference_runtime_mean_ms", "low_level_step_runtime_mean_ms",
        "episode_runtime_ms",
    ):
        summary = _continuous_summary(row.get(key) for row in rows)
        result[f"{key}_mean"] = summary["mean"]
        result[f"{key}_std"] = summary["std"]
        result[f"{key}_n"] = summary["n"]
    return result


def build_aggregates(
    formal_rows: list[dict[str, Any]],
    scenario_order: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overall: list[dict[str, Any]] = []
    scenario: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        selected = [row for row in formal_rows if row["method"] == method]
        overall.append({
            "method": method,
            "method_display_name": METHOD_DISPLAY_NAMES[method],
            **aggregate_group(selected),
        })
        for scene in scenario_order:
            subset = [row for row in selected if row["scenario"] == scene]
            scenario.append({
                "scenario": scene,
                "method": method,
                "method_display_name": METHOD_DISPLAY_NAMES[method],
                **aggregate_group(subset),
            })
    return overall, scenario


def build_paired_analysis(formal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in formal_rows:
        grouped[str(row["pair_id"])][str(row["method"])] = row
    transitions = {
        "proposal_failure_to_fp_shep_success": 0,
        "proposal_success_to_fp_shep_failure": 0,
        "both_success": 0,
        "both_fail": 0,
    }
    difference_rows: list[dict[str, Any]] = []
    metrics = (
        "path_length_team_mean_m", "completion_time_all_s",
        "minimum_inter_agent_distance_m", "trajectory_smoothness_team_mean",
        "goal_switch_count",
    )
    for pair_id, methods in grouped.items():
        if set(methods) != set(METHOD_ORDER):
            continue
        proposal = methods[METHOD_PROPOSAL]
        fp = methods[METHOD_FP_SHEP]
        if not proposal["success"] and fp["success"]:
            transitions["proposal_failure_to_fp_shep_success"] += 1
        elif proposal["success"] and not fp["success"]:
            transitions["proposal_success_to_fp_shep_failure"] += 1
        elif proposal["success"] and fp["success"]:
            transitions["both_success"] += 1
        else:
            transitions["both_fail"] += 1
        record = {
            "pair_id": pair_id,
            "scenario": fp["scenario"],
            "seed": fp["seed"],
            "proposal_success": proposal["success"],
            "fp_shep_success": fp["success"],
            "proposal_collision": proposal["collision"],
            "fp_shep_collision": fp["collision"],
        }
        for metric in metrics:
            record[f"delta_fp_shep_minus_proposal__{metric}"] = float(fp[metric]) - float(proposal[metric])
        difference_rows.append(record)
    win_tie_loss: list[dict[str, Any]] = []
    beneficial_positive = {"minimum_inter_agent_distance_m"}
    for metric in metrics:
        values = np.asarray(
            [row[f"delta_fp_shep_minus_proposal__{metric}"] for row in difference_rows],
            dtype=float,
        )
        signs = np.sign(values)
        wins = int(np.sum(signs > 0)) if metric in beneficial_positive else int(np.sum(signs < 0))
        losses = int(np.sum(signs < 0)) if metric in beneficial_positive else int(np.sum(signs > 0))
        win_tie_loss.append({
            "metric": metric,
            "wins": wins,
            "ties": int(np.sum(signs == 0)),
            "losses": losses,
            "mean_delta_fp_shep_minus_proposal": float(np.mean(values)) if values.size else None,
            "median_delta_fp_shep_minus_proposal": float(np.median(values)) if values.size else None,
            "n": int(values.size),
        })
    return {
        "paired_complete_count": len(difference_rows),
        "transitions": transitions,
        "difference_rows": difference_rows,
        "win_tie_loss": win_tie_loss,
    }


def build_disagreement_analysis(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formal = [row for row in event_rows if row["phase"] == "formal"]
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in formal:
        key = (row["pair_id"], int(row["timestep"]), int(row["agent_id"]))
        by_key[key][row["method"]] = row
    results: list[dict[str, Any]] = []
    for key, methods in by_key.items():
        fp = methods.get(METHOD_FP_SHEP)
        proposal = methods.get(METHOD_PROPOSAL)
        if fp is None or proposal is None or not bool(fp["proposal_fp_shep_disagreement"]):
            continue
        results.append({
            "pair_id": key[0],
            "timestep": key[1],
            "agent_id": key[2],
            "scenario": fp["scenario"],
            "seed": fp["seed"],
            "comparison_scope": "paired_closed_loop_windows_after_method_states_may_have_diverged",
            "proposal_window_task_progress": proposal.get("execution_window_task_progress"),
            "fp_shep_window_task_progress": fp.get("execution_window_task_progress"),
            "proposal_window_minimum_clearance": proposal.get("execution_window_minimum_clearance"),
            "fp_shep_window_minimum_clearance": fp.get("execution_window_minimum_clearance"),
            "proposal_window_minimum_inter_agent_distance": proposal.get("execution_window_minimum_inter_agent_distance"),
            "fp_shep_window_minimum_inter_agent_distance": fp.get("execution_window_minimum_inter_agent_distance"),
            "proposal_window_collision": proposal.get("execution_window_collision"),
            "fp_shep_window_collision": fp.get("execution_window_collision"),
            "proposal_window_max_deviation": proposal.get("execution_window_max_deviation"),
            "fp_shep_window_max_deviation": fp.get("execution_window_max_deviation"),
            "proposal_goal_jump": proposal.get("goal_jump"),
            "fp_shep_goal_jump": fp.get("goal_jump"),
        })
    return results


def build_fallback_summary(
    event_rows: list[dict[str, Any]], scenario_order: Sequence[str]
) -> list[dict[str, Any]]:
    formal = [row for row in event_rows if row["phase"] == "formal"]
    output: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        for scenario in scenario_order:
            selected = [
                row for row in formal
                if row["method"] == method and row["scenario"] == scenario
            ]
            fallback = sum(bool(row["no_candidate_fallback"]) for row in selected)
            output.append({
                "method": method,
                "method_display_name": METHOD_DISPLAY_NAMES[method],
                "scenario": scenario,
                "replanning_agent_event_count": len(selected),
                "K_t_zero_count": fallback,
                "no_candidate_fallback_count": fallback,
                "no_candidate_fallback_rate": fallback / len(selected) if selected else None,
                "fallback_counted_as_selection": False,
                "candidate_pipeline_applicable": method != METHOD_FROZEN,
                "rate_interpretation": (
                    "not_applicable"
                    if method == METHOD_FROZEN
                    else "fallback events / replanning agent events"
                ),
            })
    return output


def build_switching_diagnostics(
    event_rows: list[dict[str, Any]], scenario_order: Sequence[str]
) -> list[dict[str, Any]]:
    """Aggregate read-only handoff diagnostics without changing the protocol."""

    formal = [row for row in event_rows if row["phase"] == "formal"]
    output: list[dict[str, Any]] = []
    for method in (METHOD_PROPOSAL, METHOD_FP_SHEP):
        for scenario in scenario_order:
            selected = [
                row for row in formal
                if row["method"] == method and row["scenario"] == scenario
            ]
            reached = sum(bool(row.get("reference_reached_early", False)) for row in selected)
            switched = sum(bool(row.get("selection_changed", False)) for row in selected)
            goal_jumps = _finite(row.get("goal_jump") for row in selected)
            hold_steps = _finite(
                row.get("reference_hold_steps_after_reached") for row in selected
            )
            output.append({
                "method": method,
                "method_display_name": METHOD_DISPLAY_NAMES[method],
                "scenario": scenario,
                "replanning_agent_event_count": len(selected),
                "selection_changed_count": switched,
                "selection_changed_rate": switched / len(selected) if selected else None,
                "same_goal_retention_count": len(selected) - switched,
                "mean_goal_jump_m": float(np.mean(goal_jumps)) if goal_jumps.size else None,
                "maximum_goal_jump_m": float(np.max(goal_jumps)) if goal_jumps.size else None,
                "reference_reached_early_count": reached,
                "reference_reached_early_rate": reached / len(selected) if selected else None,
                "reference_hold_steps_after_reached_sum": (
                    int(np.sum(hold_steps)) if hold_steps.size else 0
                ),
                "reference_hold_steps_after_reached_mean": (
                    float(np.mean(hold_steps)) if hold_steps.size else None
                ),
                "diagnostic_changes_control_behavior": False,
                "replanning_rule": "timestep % M_upper == 0",
            })
    return output


def build_runtime_summary(
    formal_rows: list[dict[str, Any]], scenario_order: Sequence[str]
) -> list[dict[str, Any]]:
    runtime_fields = (
        "candidate_generation_runtime_mean_ms",
        "fp_shep_preview_runtime_mean_ms",
        "candidate_selection_runtime_mean_ms",
        "upper_replanning_runtime_mean_ms",
        "frozen_sac_inference_runtime_mean_ms",
        "low_level_step_runtime_mean_ms",
        "episode_runtime_ms",
    )
    output: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        for scope, scenario in [("overall", None)] + [
            ("scenario", item) for item in scenario_order
        ]:
            selected = [
                row for row in formal_rows
                if row["method"] == method
                and (scenario is None or row["scenario"] == scenario)
            ]
            record: dict[str, Any] = {
                "scope": scope,
                "scenario": scenario,
                "method": method,
                "method_display_name": METHOD_DISPLAY_NAMES[method],
                "episode_count": len(selected),
                "GAT_forward_runtime_included": False,
            }
            for field in runtime_fields:
                summary = _continuous_summary(row.get(field) for row in selected)
                for statistic in ("mean", "std", "median", "n"):
                    record[f"{field}_{statistic}"] = summary[statistic]
            output.append(record)
    return output


def build_disagreement_summary(
    rows: list[dict[str, Any]], event_rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    metrics = (
        ("task_progress", True),
        ("minimum_clearance", True),
        ("minimum_inter_agent_distance", True),
        ("max_deviation", False),
        ("goal_jump", False),
    )
    formal_fp_events = [
        row for row in (event_rows or [])
        if row.get("phase") == "formal" and row.get("method") == METHOD_FP_SHEP
        and int(row.get("K_t", 0)) > 0
    ]
    denominator = len(formal_fp_events) if event_rows is not None else None
    result: dict[str, Any] = {
        "disagreement_event_count": len(rows),
        "eligible_fp_shep_event_count": denominator,
        "proposal_fp_shep_disagreement_rate": (
            len(rows) / denominator if denominator else None
        ),
    }
    for metric, beneficial_positive in metrics:
        deltas = _finite(
            float(row[f"fp_shep_window_{metric}"] if f"fp_shep_window_{metric}" in row else row[f"fp_shep_{metric}"])
            - float(row[f"proposal_window_{metric}"] if f"proposal_window_{metric}" in row else row[f"proposal_{metric}"])
            for row in rows
            if (row.get(f"fp_shep_window_{metric}") is not None or row.get(f"fp_shep_{metric}") is not None)
            and (row.get(f"proposal_window_{metric}") is not None or row.get(f"proposal_{metric}") is not None)
        )
        result[f"delta_fp_shep_minus_proposal__{metric}_mean"] = (
            float(np.mean(deltas)) if deltas.size else None
        )
        result[f"delta_fp_shep_minus_proposal__{metric}_median"] = (
            float(np.median(deltas)) if deltas.size else None
        )
        result[f"{metric}_beneficial_count"] = int(
            np.sum(deltas > 0.0) if beneficial_positive else np.sum(deltas < 0.0)
        )
        result[f"{metric}_n"] = int(deltas.size)
    result["proposal_window_collision_count"] = sum(
        bool(row.get("proposal_window_collision", False)) for row in rows
    )
    result["fp_shep_window_collision_count"] = sum(
        bool(row.get("fp_shep_window_collision", False)) for row in rows
    )
    result["comparison_scope"] = (
        "paired closed-loop windows; method states may differ after trajectory divergence"
    )
    return result


def select_representative_cases(formal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in formal_rows:
        grouped[str(row["pair_id"])][str(row["method"])] = row
    complete = [methods for methods in grouped.values() if set(methods) == set(METHOD_ORDER)]

    def median_case(candidates: list[dict[str, Any]], score) -> dict[str, Any] | None:
        if not candidates:
            return None
        values = np.asarray([score(item) for item in candidates], dtype=float)
        target = float(np.median(values))
        return candidates[int(np.argmin(np.abs(values - target)))]

    cases: list[dict[str, Any]] = []
    definitions = [
        (
            "A_frozen_failure_proposal_success",
            "frozen_failure_proposal_success",
            [m for m in complete if not m[METHOD_FROZEN]["success"] and m[METHOD_PROPOSAL]["success"]],
            lambda m: float(m[METHOD_FROZEN]["path_length_team_mean_m"] - m[METHOD_PROPOSAL]["path_length_team_mean_m"]),
            False,
        ),
        (
            "B_proposal_failure_fp_shep_success",
            "proposal_failure_fp_shep_success",
            [m for m in complete if not m[METHOD_PROPOSAL]["success"] and m[METHOD_FP_SHEP]["success"]],
            lambda m: float(m[METHOD_PROPOSAL]["path_length_team_mean_m"] - m[METHOD_FP_SHEP]["path_length_team_mean_m"]),
            False,
        ),
        (
            "C_proposal_success_fp_shep_failure",
            "proposal_success_fp_shep_failure",
            [m for m in complete if m[METHOD_PROPOSAL]["success"] and not m[METHOD_FP_SHEP]["success"]],
            lambda m: float(m[METHOD_FP_SHEP]["path_length_team_mean_m"] - m[METHOD_PROPOSAL]["path_length_team_mean_m"]),
            False,
        ),
        (
            "D_both_success_fp_shep_efficiency",
            "both_success_fp_shep_path_or_smoothness_improvement",
            [
                m for m in complete
                if m[METHOD_PROPOSAL]["success"] and m[METHOD_FP_SHEP]["success"]
                and (
                    m[METHOD_FP_SHEP]["path_length_team_mean_m"] < m[METHOD_PROPOSAL]["path_length_team_mean_m"]
                    or m[METHOD_FP_SHEP]["trajectory_smoothness_team_mean"] < m[METHOD_PROPOSAL]["trajectory_smoothness_team_mean"]
                )
            ],
            lambda m: float(m[METHOD_PROPOSAL]["path_length_team_mean_m"] - m[METHOD_FP_SHEP]["path_length_team_mean_m"]),
            False,
        ),
        (
            "E_both_fail_multi_agent_conflict",
            "both_fail_multi_agent_conflict",
            [
                m for m in complete
                if not m[METHOD_PROPOSAL]["success"] and not m[METHOD_FP_SHEP]["success"]
                and (m[METHOD_PROPOSAL]["inter_agent_collision"] or m[METHOD_FP_SHEP]["inter_agent_collision"])
            ],
            lambda m: float(m[METHOD_FP_SHEP]["minimum_inter_agent_distance_m"]),
            False,
        ),
    ]
    for case_id, reason, candidates, score, extreme in definitions:
        selected = median_case(candidates, score)
        cases.append({
            "case_id": case_id,
            "selection_reason": reason,
            "found": selected is not None,
            "pair_id": None if selected is None else selected[METHOD_FP_SHEP]["pair_id"],
            "scenario": None if selected is None else selected[METHOD_FP_SHEP]["scenario"],
            "seed": None if selected is None else selected[METHOD_FP_SHEP]["seed"],
            "extreme_example": extreme,
        })
    if complete:
        high_switch = max(complete, key=lambda m: float(m[METHOD_FP_SHEP]["goal_switch_rate"]))
        large_jump = max(complete, key=lambda m: float(m[METHOD_FP_SHEP]["maximum_goal_jump_m"]))
        for case_id, reason, selected in (
            ("F_high_frequency_goal_switching", "switching_instability_case", high_switch),
            ("G_large_goal_jump", "large_goal_jump_event", large_jump),
        ):
            cases.append({
                "case_id": case_id,
                "selection_reason": reason,
                "found": True,
                "pair_id": selected[METHOD_FP_SHEP]["pair_id"],
                "scenario": selected[METHOD_FP_SHEP]["scenario"],
                "seed": selected[METHOD_FP_SHEP]["seed"],
                "extreme_example": True,
            })
    residual = [
        m for m in complete
        if m[METHOD_FP_SHEP]["inter_agent_collision"]
        and m[METHOD_FP_SHEP]["scenario"] in {"multi_agent", "narrow_head_on"}
    ]
    selected = median_case(residual, lambda m: float(m[METHOD_FP_SHEP]["minimum_inter_agent_distance_m"]))
    cases.append({
        "case_id": "H_residual_multi_agent_conflict",
        "selection_reason": "residual_multi_agent_failure",
        "found": selected is not None,
        "pair_id": None if selected is None else selected[METHOD_FP_SHEP]["pair_id"],
        "scenario": None if selected is None else selected[METHOD_FP_SHEP]["scenario"],
        "seed": None if selected is None else selected[METHOD_FP_SHEP]["seed"],
        "extreme_example": False,
    })
    return cases


def _configure_style(settings: dict[str, Any]) -> str:
    preferences = settings["paper_ready"]["font_family_preference"]
    installed = {item.name for item in font_manager.fontManager.ttflist}
    font = next((item for item in preferences if item in installed), "DejaVu Serif")
    plt.rcParams.update({
        "font.family": font,
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    return font


def _figure_metadata(
    *,
    figure_id: str,
    title: str,
    source: str,
    methods: Sequence[str],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    metric: str,
    unit: str,
    aggregation: str,
    filter_rule: str,
    m_upper: int,
    success_only: bool,
    status: str,
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "publication_title": title,
        "source_artifact": source,
        "methods": [METHOD_DISPLAY_NAMES[item] for item in methods],
        "scenarios": list(scenarios),
        "seeds": list(seeds),
        "metric_definition": metric,
        "unit": unit,
        "aggregation": aggregation,
        "filter_rule": filter_rule,
        "M_upper": int(m_upper),
        "successful_only": bool(success_only),
        "timestamp": datetime.now().astimezone().isoformat(),
        "code_version": "pre_gat_closed_loop_v1",
        "publication_status": status,
    }


def _save_figure(
    fig: Any,
    *,
    figure_id: str,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    paper_dir: Path,
    dpi: int,
) -> None:
    figures = paper_dir / "figures"
    data_dir = paper_dir / "figure_data"
    figures.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{figure_id}.pdf", bbox_inches="tight")
    fig.savefig(figures / f"{figure_id}.png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    _write_csv(data_dir / f"{figure_id}.csv", rows)
    _write_json(data_dir / f"{figure_id}.json", metadata)


def _method_style(settings: dict[str, Any], method: str) -> dict[str, Any]:
    return settings["paper_ready"]["method_styles"][method]


def _plot_m_upper_sensitivity(
    *,
    development_rows: list[dict[str, Any]],
    settings: dict[str, Any],
    run_dir: Path,
) -> None:
    """Save development-only timescale diagnostics outside paper_ready."""

    if not development_rows:
        return
    methods = (METHOD_PROPOSAL, METHOD_FP_SHEP)
    m_values = sorted({int(row["M_upper"]) for row in development_rows})
    rows: list[dict[str, Any]] = []
    for method in methods:
        for value in m_values:
            selected = [
                row for row in development_rows
                if row["method"] == method and int(row["M_upper"]) == value
            ]
            summary = aggregate_group(selected)
            rows.append({
                "method": method,
                "method_display_name": METHOD_DISPLAY_NAMES[method],
                "M_upper": value,
                **summary,
            })
    panels = (
        ("success_rate", "Success Rate (%)", 100.0),
        ("collision_rate", "Collision Rate (%)", 100.0),
        ("inter_agent_collision_rate", "Inter-Agent Collision Rate (%)", 100.0),
        ("path_length_success_team_mean_m_mean", "Successful Path Length (m)", 1.0),
        ("completion_time_success_s_mean", "Successful Completion Time (s)", 1.0),
        ("goal_switch_count_mean", "Goal Switch Count", 1.0),
        ("mean_goal_jump_m_mean", "Mean Goal Jump (m)", 1.0),
        ("trajectory_smoothness_team_mean_mean", "Trajectory Smoothness", 1.0),
        ("minimum_inter_agent_distance_m_mean", "Minimum Inter-Agent Distance (m)", 1.0),
        ("upper_replanning_runtime_mean_ms_mean", "Upper Replanning Runtime (ms)", 1.0),
    )
    fig, axes = plt.subplots(2, 5, figsize=(12.0, 5.2), sharex=True)
    for axis, (key, label, scale) in zip(axes.flat, panels, strict=True):
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method]
            values = [
                (float(row[key]) * scale) if row.get(key) is not None else np.nan
                for row in method_rows
            ]
            style = _method_style(settings, method)
            axis.plot(
                m_values,
                values,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["line_style"],
                linewidth=1.2,
                label=METHOD_DISPLAY_NAMES[method],
            )
        axis.set_xlabel("M_upper")
        axis.set_ylabel(label)
        axis.set_xticks(m_values)
        axis.grid(alpha=0.25, linewidth=0.6)
    axes.flat[0].set_ylim(0.0, 100.0)
    axes.flat[1].set_ylim(0.0, 100.0)
    axes.flat[2].set_ylim(0.0, 100.0)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    target = run_dir / "upper_period_sensitivity"
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / "m_upper_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(
        target / "m_upper_sensitivity.png",
        dpi=int(settings["paper_ready"]["png_dpi"]),
        bbox_inches="tight",
    )
    plt.close(fig)
    _write_csv(target / "m_upper_sensitivity_by_method.csv", rows)
    _write_json(target / "m_upper_sensitivity_figure.json", {
        "title": "Development M_upper Sensitivity",
        "source_artifact": "per_episode/all_episode_records.csv",
        "phase_filter": "development only",
        "methods": [METHOD_DISPLAY_NAMES[item] for item in methods],
        "M_upper_values": m_values,
        "formal_seeds_used": False,
        "publication_status": "DIAGNOSTIC_ONLY",
        "paper_ready_included": False,
    })


def _plot_rate_overall(
    aggregate: list[dict[str, Any]], settings: dict[str, Any], paper_dir: Path,
    *, key: str, figure_id: str, title: str, formal_seeds: Sequence[int], m_upper: int,
    status: str,
) -> dict[str, Any]:
    rows = []
    for item in aggregate:
        rows.append({
            "method": item["method"],
            "method_display_name": item["method_display_name"],
            "count": item[f"{key}_count"],
            "total": item[f"{key}_total"],
            "rate_percent": 100.0 * item[f"{key}_rate"],
            "ci95_low_percent": 100.0 * item[f"{key}_ci95_low"],
            "ci95_high_percent": 100.0 * item[f"{key}_ci95_high"],
        })
    fig, axis = plt.subplots(figsize=(settings["paper_ready"]["single_column_width_in"], 2.7))
    x = np.arange(len(rows))
    values = np.asarray([row["rate_percent"] for row in rows])
    lower = values - np.asarray([row["ci95_low_percent"] for row in rows])
    upper = np.asarray([row["ci95_high_percent"] for row in rows]) - values
    for index, (method, value) in enumerate(zip(METHOD_ORDER, values, strict=True)):
        style = _method_style(settings, method)
        axis.bar(index, value, color=style["color"], hatch=style["hatch"], width=0.65)
    axis.errorbar(x, values, yerr=np.stack([lower, upper]), fmt="none", color="black", capsize=3, linewidth=0.8)
    axis.set_xticks(x, [METHOD_DISPLAY_NAMES[item] for item in METHOD_ORDER], rotation=18, ha="right")
    axis.set_ylabel(RATE_METRICS[key])
    axis.set_ylim(*settings["paper_ready"]["success_collision_axis_percent_range"])
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    metadata = _figure_metadata(
        figure_id=figure_id, title=title, source="per_episode/all_episode_records.csv",
        methods=METHOD_ORDER, scenarios=settings["scenario_types"], seeds=formal_seeds,
        metric=RATE_METRICS[key], unit="percent", aggregation="episode count / total",
        filter_rule="formal episodes", m_upper=m_upper, success_only=False, status=status,
    )
    _save_figure(fig, figure_id=figure_id, rows=rows, metadata=metadata, paper_dir=paper_dir, dpi=settings["paper_ready"]["png_dpi"])
    return metadata


def _plot_rate_scenario(
    aggregate: list[dict[str, Any]], settings: dict[str, Any], paper_dir: Path,
    *, key: str, figure_id: str, title: str, formal_seeds: Sequence[int], m_upper: int,
    status: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in aggregate:
        rows.append({
            "scenario": item["scenario"],
            "scenario_display_name": settings["scenario_display_names"][item["scenario"]],
            "method": item["method"],
            "method_display_name": item["method_display_name"],
            "count": item[f"{key}_count"],
            "total": item[f"{key}_total"],
            "rate_percent": 100.0 * item[f"{key}_rate"],
        })
    width = settings["paper_ready"]["double_column_width_in"]
    fig, axis = plt.subplots(figsize=(width, 3.0))
    scenarios = settings["scenario_types"]
    x = np.arange(len(scenarios))
    bar_width = 0.24
    for method_index, method in enumerate(METHOD_ORDER):
        values = [next(row["rate_percent"] for row in rows if row["scenario"] == scene and row["method"] == method) for scene in scenarios]
        style = _method_style(settings, method)
        axis.bar(x + (method_index - 1) * bar_width, values, width=bar_width, label=METHOD_DISPLAY_NAMES[method], color=style["color"], hatch=style["hatch"])
    axis.set_xticks(x, [settings["scenario_display_names"][item] for item in scenarios], rotation=25, ha="right")
    axis.set_ylabel(RATE_METRICS[key])
    axis.set_ylim(*settings["paper_ready"]["success_collision_axis_percent_range"])
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    fig.tight_layout()
    metadata = _figure_metadata(
        figure_id=figure_id, title=title, source="per_episode/all_episode_records.csv",
        methods=METHOD_ORDER, scenarios=scenarios, seeds=formal_seeds,
        metric=RATE_METRICS[key], unit="percent", aggregation="scenario-wise episode count / total",
        filter_rule="formal episodes", m_upper=m_upper, success_only=False, status=status,
    )
    _save_figure(fig, figure_id=figure_id, rows=rows, metadata=metadata, paper_dir=paper_dir, dpi=settings["paper_ready"]["png_dpi"])
    return metadata


def _plot_continuous_overall(
    aggregate: list[dict[str, Any]], settings: dict[str, Any], paper_dir: Path,
    *, metric_key: str, figure_id: str, title: str, formal_seeds: Sequence[int],
    m_upper: int, status: str,
) -> dict[str, Any]:
    label, unit, success_only = CONTINUOUS_METRICS[metric_key]
    rows: list[dict[str, Any]] = []
    for item in aggregate:
        rows.append({
            "method": item["method"],
            "method_display_name": item["method_display_name"],
            "mean": item[f"{metric_key}_mean"],
            "std": item[f"{metric_key}_std"],
            "median": item[f"{metric_key}_median"],
            "n": item[f"{metric_key}_n"],
            "ci95_low": item[f"{metric_key}_ci95_low"],
            "ci95_high": item[f"{metric_key}_ci95_high"],
        })
    fig, axis = plt.subplots(figsize=(settings["paper_ready"]["single_column_width_in"], 2.7))
    x = np.arange(len(rows))
    values = np.asarray([
        row["mean"] if row["mean"] is not None else np.nan for row in rows
    ], dtype=float)
    lower = np.asarray([
        value - row["ci95_low"]
        if np.isfinite(value) and row["ci95_low"] is not None else np.nan
        for value, row in zip(values, rows, strict=True)
    ])
    upper = np.asarray([
        row["ci95_high"] - value
        if np.isfinite(value) and row["ci95_high"] is not None else np.nan
        for value, row in zip(values, rows, strict=True)
    ])
    finite_values = values[np.isfinite(values)]
    annotation_y = 0.03 * float(np.max(finite_values)) if finite_values.size else 0.03
    for index, (method, value) in enumerate(zip(METHOD_ORDER, values, strict=True)):
        style = _method_style(settings, method)
        if np.isfinite(value):
            axis.bar(index, value, color=style["color"], hatch=style["hatch"], width=0.65)
        else:
            axis.text(
                index, annotation_y, "N/A\n(n=0)",
                ha="center", va="bottom", fontsize=7.0, color=style["color"],
            )
    axis.errorbar(x, values, yerr=np.stack([lower, upper]), fmt="none", color="black", capsize=3, linewidth=0.8)
    axis.set_xticks(x, [METHOD_DISPLAY_NAMES[item] for item in METHOD_ORDER], rotation=18, ha="right")
    axis.set_ylabel(f"{label} ({unit})")
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    metadata = _figure_metadata(
        figure_id=figure_id, title=title, source="per_episode/all_episode_records.csv",
        methods=METHOD_ORDER, scenarios=settings["scenario_types"], seeds=formal_seeds,
        metric=label, unit=unit, aggregation="mean with normal-approximation 95% CI",
        filter_rule="successful formal episodes" if success_only else "all formal episodes",
        m_upper=m_upper, success_only=success_only, status=status,
    )
    _save_figure(fig, figure_id=figure_id, rows=rows, metadata=metadata, paper_dir=paper_dir, dpi=settings["paper_ready"]["png_dpi"])
    return metadata


def _draw_box(axis: Any, center: np.ndarray, half: np.ndarray) -> None:
    center = np.asarray(center, dtype=float)
    half = np.asarray(half, dtype=float)
    vertices = np.asarray([
        center + np.asarray([sx, sy, sz]) * half
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ])
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(vertices[i] != vertices[j]) == 1]
    for left, right in edges:
        axis.plot(*vertices[[left, right]].T, color="0.65", linewidth=0.45, alpha=0.65)


def _draw_scene_obstacles(axis: Any, snapshot: dict[str, Any]) -> None:
    for obstacle in snapshot.get("static_obstacles", []) + snapshot.get("dynamic_obstacles", []):
        center = obstacle.get("center", obstacle.get("_center", obstacle.get("initial_center")))
        half = obstacle.get("half_extents", obstacle.get("_half_extents"))
        radius = obstacle.get("radius", obstacle.get("_radius"))
        if center is not None and half is not None:
            _draw_box(axis, np.asarray(center, dtype=float), np.asarray(half, dtype=float))
        elif center is not None:
            size = 16.0 + 80.0 * float(radius or 0.2)
            axis.scatter(*np.asarray(center, dtype=float), marker="s", s=size, color="0.65", alpha=0.55)


def _plot_trajectory_case(
    *, run_dir: Path, case: dict[str, Any], formal_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    settings: dict[str, Any], paper_dir: Path, figure_id: str, title: str,
    formal_seeds: Sequence[int], m_upper: int,
) -> dict[str, Any] | None:
    if not case.get("found"):
        return None
    pair_id = case["pair_id"]
    selected = {row["method"]: row for row in formal_rows if row["pair_id"] == pair_id}
    if set(selected) != set(METHOD_ORDER):
        return None
    fig = plt.figure(figsize=(settings["paper_ready"]["double_column_width_in"], 2.7))
    rows: list[dict[str, Any]] = []
    uav_colors = ("#1b9e77", "#d95f02", "#7570b3")
    for method_index, method in enumerate(METHOD_ORDER, start=1):
        episode = selected[method]
        path = run_dir / "trajectories" / f"{episode['method_episode_id']}.npz"
        bundle = np.load(path, allow_pickle=False)
        positions = np.asarray(bundle["positions"], dtype=float)
        starts = np.asarray(bundle["starts"], dtype=float)
        goals = np.asarray(bundle["terminal_goals"], dtype=float)
        snapshot = json.loads(str(bundle["scene_snapshot_json"].item()))
        method_events = [
            row for row in event_rows
            if row["phase"] == "formal"
            and row["pair_id"] == pair_id
            and row["method"] == method
        ]
        axis = fig.add_subplot(1, 3, method_index, projection="3d")
        _draw_scene_obstacles(axis, snapshot)
        for agent in range(positions.shape[1]):
            axis.plot(*positions[:, agent].T, color=uav_colors[agent], linewidth=1.2, label=f"UAV {agent + 1}")
            axis.scatter(*starts[agent], color=uav_colors[agent], marker="o", s=18)
            axis.scatter(*goals[agent], color=uav_colors[agent], marker="*", s=42)
            for step, point in enumerate(positions[:, agent]):
                rows.append({
                    "pair_id": pair_id, "method": method,
                    "method_display_name": METHOD_DISPLAY_NAMES[method],
                    "uav_id": agent + 1, "step": step,
                    "x": point[0], "y": point[1], "z": point[2],
                    "success": episode["success"], "collision": episode["collision"],
                    "record_type": "trajectory",
                })
            references = [
                np.asarray(row["new_execution_reference"], dtype=float)
                for row in method_events
                if int(row["agent_id"]) == agent
            ]
            if references:
                reference_array = np.asarray(references, dtype=float)
                axis.plot(
                    *reference_array.T,
                    color=uav_colors[agent],
                    linewidth=0.75,
                    linestyle=":",
                    alpha=0.75,
                )
                axis.scatter(
                    *reference_array.T,
                    color=uav_colors[agent],
                    marker="x",
                    s=16,
                )
                for reference_index, point in enumerate(reference_array):
                    rows.append({
                        "pair_id": pair_id,
                        "method": method,
                        "method_display_name": METHOD_DISPLAY_NAMES[method],
                        "uav_id": agent + 1,
                        "step": None,
                        "reference_index": reference_index,
                        "x": point[0],
                        "y": point[1],
                        "z": point[2],
                        "success": episode["success"],
                        "collision": episode["collision"],
                        "record_type": "selected_reference",
                    })
        if episode["collision"]:
            axis.scatter(
                *positions[-1].T,
                color="black",
                marker="x",
                s=30,
            )
        axis.set_title(METHOD_DISPLAY_NAMES[method], fontsize=8.5)
        axis.set_xlabel("X (m)", fontsize=7.0, labelpad=-1)
        axis.set_ylabel("Y (m)", fontsize=7.0, labelpad=-1)
        axis.set_zlabel("Z (m)", fontsize=7.0, labelpad=-1)
        axis.tick_params(axis="both", which="major", labelsize=6.0, pad=0)
        axis.set_box_aspect((1.0, 1.0, 0.78))
        axis.view_init(elev=24, azim=-62)
        axis.grid(alpha=0.2)
    handles = [
        Line2D([0], [0], color=uav_colors[index], linewidth=1.4, label=f"UAV {index + 1}")
        for index in range(3)
    ] + [
        Line2D([0], [0], color="0.25", marker="o", linestyle="none", markersize=4, label="Start"),
        Line2D([0], [0], color="0.25", marker="*", linestyle="none", markersize=6, label="Terminal Goal"),
        Line2D([0], [0], color="0.25", marker="x", linestyle=":", markersize=4, label="Selected Reference"),
        Line2D([0], [0], color="black", marker="x", linestyle="none", markersize=5, label="Collision Termination"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=4,
        fontsize=6.5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        handlelength=1.8,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.79, wspace=0.03)
    metadata = _figure_metadata(
        figure_id=figure_id, title=title, source="trajectories/*.npz",
        methods=METHOD_ORDER, scenarios=[case["scenario"]], seeds=[int(case["seed"])],
        metric="three-dimensional closed-loop trajectories and selected references", unit="m",
        aggregation="representative paired episode",
        filter_rule=case["selection_reason"], m_upper=m_upper,
        success_only=False, status="DIAGNOSTIC_ONLY",
    )
    metadata["representative_case_selection"] = case
    _save_figure(fig, figure_id=figure_id, rows=rows, metadata=metadata, paper_dir=paper_dir, dpi=settings["paper_ready"]["png_dpi"])
    return metadata


def _plot_case_time_series(
    *,
    run_dir: Path,
    case: dict[str, Any],
    formal_rows: list[dict[str, Any]],
    settings: dict[str, Any],
    m_upper: int,
) -> None:
    """Save traceable velocity, acceleration, separation and switch diagnostics."""

    if not case.get("found"):
        return
    selected = {
        row["method"]: row
        for row in formal_rows
        if row["pair_id"] == case["pair_id"]
    }
    if set(selected) != set(METHOD_ORDER):
        return
    case_dir = run_dir / "paper_ready" / "representative_cases" / str(case["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    series: dict[str, dict[str, np.ndarray]] = {}
    dt = float(settings["dt"])
    for method in METHOD_ORDER:
        bundle = np.load(
            run_dir / "trajectories" / f"{selected[method]['method_episode_id']}.npz",
            allow_pickle=False,
        )
        velocities = np.asarray(bundle["velocities"], dtype=float)
        accelerations = np.asarray(bundle["accelerations"], dtype=float)
        min_inter = np.asarray(bundle["minimum_inter_agent_distance"], dtype=float)
        replans = np.asarray(bundle["replanning_mask"], dtype=bool)
        speed = np.mean(np.linalg.norm(velocities, axis=2), axis=1)
        acceleration = np.mean(np.linalg.norm(accelerations, axis=2), axis=1)
        replan_count = np.sum(replans, axis=1).astype(float)
        series[method] = {
            "speed": speed,
            "acceleration": acceleration,
            "minimum_inter_agent_distance": min_inter,
            "replan_count": replan_count,
        }
        for step in range(len(speed)):
            rows.append({
                "case_id": case["case_id"],
                "pair_id": case["pair_id"],
                "method": method,
                "method_display_name": METHOD_DISPLAY_NAMES[method],
                "step": step,
                "time_s": step * dt,
                "team_mean_speed_m_per_s": speed[step],
                "team_mean_acceleration_m_per_s2": (
                    acceleration[step] if step < len(acceleration) else None
                ),
                "minimum_inter_agent_distance_m": (
                    min_inter[step] if step < len(min_inter) else None
                ),
                "replanned_agent_count": (
                    replan_count[step] if step < len(replan_count) else 0
                ),
            })
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(settings["paper_ready"]["double_column_width_in"], 6.2),
        sharex=True,
    )
    for method in METHOD_ORDER:
        style = _method_style(settings, method)
        value = series[method]
        axes[0].plot(np.arange(len(value["speed"])) * dt, value["speed"], label=METHOD_DISPLAY_NAMES[method], color=style["color"], linestyle=style["line_style"])
        axes[1].plot((np.arange(len(value["acceleration"])) + 1) * dt, value["acceleration"], color=style["color"], linestyle=style["line_style"])
        axes[2].plot(np.arange(len(value["minimum_inter_agent_distance"])) * dt, value["minimum_inter_agent_distance"], color=style["color"], linestyle=style["line_style"])
        active = np.flatnonzero(value["replan_count"] > 0)
        axes[3].scatter(active * dt, value["replan_count"][active], color=style["color"], marker=style["marker"], s=18)
    axes[0].set_ylabel("Velocity (m/s)")
    axes[1].set_ylabel("Acceleration (m/s²)")
    axes[2].set_ylabel("Minimum Distance (m)")
    axes[3].set_ylabel("Replanned UAVs")
    axes[3].set_xlabel("Time (s)")
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.35))
    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(case_dir / "time_series.pdf", bbox_inches="tight")
    fig.savefig(case_dir / "time_series.png", dpi=int(settings["paper_ready"]["png_dpi"]), bbox_inches="tight")
    plt.close(fig)
    _write_csv(case_dir / "time_series.csv", rows)
    _write_json(case_dir / "time_series.json", {
        "publication_title": "Closed-Loop Execution Time Series",
        "case_id": case["case_id"],
        "pair_id": case["pair_id"],
        "selection_reason": case["selection_reason"],
        "extreme_example": case["extreme_example"],
        "methods": [METHOD_DISPLAY_NAMES[item] for item in METHOD_ORDER],
        "metrics": [
            "team mean velocity", "team mean acceleration",
            "minimum inter-agent distance", "replanned UAV count",
        ],
        "M_upper": int(m_upper),
        "publication_status": "DIAGNOSTIC_ONLY",
    })


def _write_tables(
    overall: list[dict[str, Any]], scenario: list[dict[str, Any]],
    settings: dict[str, Any], paper_dir: Path,
) -> None:
    tables = paper_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    overall_rows: list[dict[str, Any]] = []
    for row in overall:
        output = {
            "Method": row["method_display_name"],
            "Success Count": row["success_count"],
            "Episode Count": row["success_total"],
            "Success Rate (%)": 100.0 * row["success_rate"],
            "Collision Count": row["collision_count"],
            "Collision Rate (%)": 100.0 * row["collision_rate"],
            "Inter-Agent Collision Rate (%)": 100.0 * row["inter_agent_collision_rate"],
            "Obstacle Collision Rate (%)": 100.0 * row["obstacle_collision_rate"],
        }
        for key, (label, unit, _) in CONTINUOUS_METRICS.items():
            output[f"{label} Mean ({unit})"] = row[f"{key}_mean"]
            output[f"{label} Std ({unit})"] = row[f"{key}_std"]
            output[f"{label} Median ({unit})"] = row[f"{key}_median"]
            output[f"{label} N"] = row[f"{key}_n"]
        overall_rows.append(output)
    scenario_rows = []
    for row in scenario:
        output = {
            "Scenario": settings["scenario_display_names"][row["scenario"]],
            "Method": row["method_display_name"],
            "Success Count": row["success_count"],
            "Episode Count": row["success_total"],
            "Success Rate (%)": 100.0 * row["success_rate"],
            "Collision Rate (%)": 100.0 * row["collision_rate"],
            "Inter-Agent Collision Rate (%)": 100.0 * row["inter_agent_collision_rate"],
            "Obstacle Collision Rate (%)": 100.0 * row["obstacle_collision_rate"],
        }
        for key, (label, unit, _) in CONTINUOUS_METRICS.items():
            output[f"{label} Mean ({unit})"] = row[f"{key}_mean"]
            output[f"{label} Std ({unit})"] = row[f"{key}_std"]
            output[f"{label} Median ({unit})"] = row[f"{key}_median"]
            output[f"{label} N"] = row[f"{key}_n"]
        scenario_rows.append(output)
    _write_csv(tables / "table_overall_metrics.csv", overall_rows)
    _write_csv(tables / "table_scenario_metrics.csv", scenario_rows)

    def latex(path: Path, rows: list[dict[str, Any]], first_columns: Sequence[str]) -> None:
        selected_columns = list(first_columns) + [
            "Success Rate (%)", "Collision Rate (%)",
            "Path Length Mean (m)", "Completion Time Mean (s)",
            "Minimum Inter-Agent Distance Mean (m)", "Goal Switch Count Mean (count)",
        ]
        align = "l" * len(first_columns) + "r" * (len(selected_columns) - len(first_columns))
        lines = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(selected_columns) + " \\\\", "\\midrule"]
        for row in rows:
            values = []
            for column in selected_columns:
                value = row.get(column)
                values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
            lines.append(" & ".join(values) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    latex(tables / "table_overall_metrics.tex", overall_rows, ["Method"])
    latex(tables / "table_scenario_metrics.tex", scenario_rows, ["Scenario", "Method"])


def _diagnosis(
    formal_rows: list[dict[str, Any]], development_rows: list[dict[str, Any]],
    scenario_aggregates: list[dict[str, Any]],
) -> dict[str, Any]:
    by_m: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in development_rows:
        by_m[int(row["M_upper"])].append(row)
    handoff = False
    if 4 in by_m and any(value in by_m for value in (8, 12)):
        m4_switch = float(np.mean([row["goal_switch_rate"] for row in by_m[4]]))
        m4_smooth = float(np.mean([row["trajectory_smoothness_team_mean"] for row in by_m[4]]))
        alternatives = [row for value in (8, 12) for row in by_m.get(value, [])]
        if alternatives:
            alt_switch = float(np.mean([row["goal_switch_rate"] for row in alternatives]))
            alt_smooth = float(np.mean([row["trajectory_smoothness_team_mean"] for row in alternatives]))
            handoff = bool(m4_switch > 1.05 * alt_switch and m4_smooth > 1.05 * alt_smooth)
    obstacle_scenes = {"sparse_static", "dense_static", "dynamic_obstacle"}
    interaction_scenes = {"multi_agent", "narrow_head_on"}
    lookup = {(row["scenario"], row["method"]): row for row in scenario_aggregates}
    obstacle_improvement = any(
        lookup[(scene, METHOD_FP_SHEP)]["success_rate"] > lookup[(scene, METHOD_PROPOSAL)]["success_rate"]
        for scene in obstacle_scenes
    )
    residual_collision = any(
        lookup[(scene, METHOD_FP_SHEP)]["inter_agent_collision_rate"] > 0.0
        for scene in interaction_scenes
    )
    return {
        "execution_handoff_limitation": handoff,
        "execution_handoff_label": "EXECUTION_HANDOFF_LIMITATION" if handoff else "NOT_ESTABLISHED",
        "fp_shep_obstacle_scene_improvement_observed": obstacle_improvement,
        "fp_shep_residual_inter_agent_collision_observed": residual_collision,
        "residual_multi_agent_interaction_limitation": bool(obstacle_improvement and residual_collision),
        "residual_multi_agent_label": (
            "RESIDUAL_MULTI_AGENT_INTERACTION_LIMITATION"
            if obstacle_improvement and residual_collision else "NOT_ESTABLISHED"
        ),
        "diagnosis_threshold_note": "handoff flag requires M=4 switch rate and smoothness each to exceed the pooled M=8/12 mean by more than 5%",
    }


def _manifest(path: Path, entries: list[dict[str, Any]], settings: dict[str, Any], font: str) -> None:
    lines = [
        "# Pre-GAT Closed-Loop Paper-Ready Manifest",
        "",
        f"- Publication language: English",
        f"- Actual font: {font}",
        f"- PNG DPI: {settings['paper_ready']['png_dpi']}",
        "- GAT training: disabled",
        "",
        "| File | Publication title | Metric | Source data | Methods | Scenarios | Seeds | M_upper | Aggregation | Filter | Status |",
        "|---|---|---|---|---|---|---|---:|---|---|---|",
    ]
    for item in entries:
        lines.append(
            "| {file} | {title} | {metric} | {source} | {methods} | {scenarios} | {seeds} | {m} | {aggregation} | {filter_rule} | {status} |".format(
                file=item["file"], title=item["publication_title"], metric=item["metric"],
                source=item["source_data"], methods=item["methods"], scenarios=item["scenarios"],
                seeds=item["seeds"], m=item["M_upper"], aggregation=item["aggregation"],
                filter_rule=item["filter_rule"], status=item["status"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    settings = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    episode_rows = _read_csv(run_dir / "per_episode" / "all_episode_records.csv")
    event_rows = _read_csv(run_dir / "per_replanning_event" / "all_replanning_events.csv")
    formal_rows = [row for row in episode_rows if row["phase"] == "formal"]
    development_rows = [row for row in episode_rows if row["phase"] == "development"]
    if not formal_rows:
        raise ValueError("paper-ready analysis requires formal episode records")
    formal_seeds = sorted({int(row["seed"]) for row in formal_rows})
    m_values = sorted({int(row["M_upper"]) for row in formal_rows})
    if len(m_values) != 1:
        raise ValueError("formal results must use one frozen M_upper")
    m_upper = m_values[0]
    expected_pairs = len(settings["scenario_types"]) * len(formal_seeds)
    seed_ready = len(formal_seeds) >= int(settings["paper_ready"]["minimum_formal_seed_count"])
    paired_complete = all(
        len([row for row in formal_rows if row["pair_id"] == pair_id]) == len(METHOD_ORDER)
        for pair_id in {row["pair_id"] for row in formal_rows}
    ) and len({row["pair_id"] for row in formal_rows}) == expected_pairs
    core_status = "READY" if seed_ready and paired_complete else "NEEDS_MORE_SEEDS"
    paper_dir = run_dir / "paper_ready"
    for name in ("figures", "figure_data", "tables", "representative_cases", "metadata"):
        (paper_dir / name).mkdir(parents=True, exist_ok=True)
    font = _configure_style(settings)

    overall, scenario = build_aggregates(formal_rows, settings["scenario_types"])
    paired = build_paired_analysis(formal_rows)
    disagreement = build_disagreement_analysis(event_rows)
    disagreement_summary = build_disagreement_summary(disagreement, event_rows)
    fallback = build_fallback_summary(event_rows, settings["scenario_types"])
    switching = build_switching_diagnostics(event_rows, settings["scenario_types"])
    runtime = build_runtime_summary(formal_rows, settings["scenario_types"])
    cases = select_representative_cases(formal_rows)
    diagnosis = _diagnosis(formal_rows, development_rows, scenario)
    _write_csv(run_dir / "summary" / "overall_metrics.csv", overall)
    _write_csv(run_dir / "summary" / "scenario_metrics.csv", scenario)
    _write_csv(run_dir / "summary" / "paired_differences.csv", paired["difference_rows"])
    _write_csv(run_dir / "summary" / "paired_win_tie_loss.csv", paired["win_tie_loss"])
    _write_json(run_dir / "summary" / "paired_transitions.json", paired["transitions"])
    _write_csv(run_dir / "summary" / "proposal_fp_shep_disagreement.csv", disagreement)
    _write_json(
        run_dir / "summary" / "proposal_fp_shep_disagreement_summary.json",
        disagreement_summary,
    )
    _write_csv(run_dir / "summary" / "fallback_by_method_scenario.csv", fallback)
    _write_csv(
        run_dir / "switching_diagnostics" / "switching_by_method_scenario.csv",
        switching,
    )
    _write_csv(run_dir / "runtime" / "runtime_by_method_scenario.csv", runtime)
    _write_csv(run_dir / "representative_cases" / "case_index.csv", cases)
    _write_json(run_dir / "summary" / "diagnosis.json", diagnosis)
    _write_json(paper_dir / "metadata" / "publication_name_mapping.json", {
        "methods": settings["method_display_names"],
        "scenarios": settings["scenario_display_names"],
    })
    _write_json(paper_dir / "metadata" / "method_style_mapping.json", settings["paper_ready"]["method_styles"])
    _write_json(paper_dir / "metadata" / "experiment_metadata.json", {
        "formal_seed_count": len(formal_seeds), "formal_seeds": formal_seeds,
        "formal_M_upper": m_upper, "paired_complete": paired_complete,
        "statistical_evidence_status": core_status, "actual_font": font,
        "figure_layout_quality_target": "paper_final",
        "statistical_evidence_is_not_inferred_from_dpi": True,
    })
    _plot_m_upper_sensitivity(
        development_rows=development_rows,
        settings=settings,
        run_dir=run_dir,
    )

    figure_metadata: list[dict[str, Any]] = []
    for metadata in (
        _plot_rate_overall(overall, settings, paper_dir, key="success", figure_id="fig_p1_overall_success_rate", title="Overall Success Rate", formal_seeds=formal_seeds, m_upper=m_upper, status=core_status),
        _plot_rate_overall(overall, settings, paper_dir, key="collision", figure_id="fig_p2_overall_collision_rate", title="Overall Collision Rate", formal_seeds=formal_seeds, m_upper=m_upper, status=core_status),
        _plot_rate_scenario(scenario, settings, paper_dir, key="success", figure_id="fig_p3_scenario_success_rate", title="Scenario-Wise Success Rate", formal_seeds=formal_seeds, m_upper=m_upper, status=core_status),
        _plot_rate_scenario(scenario, settings, paper_dir, key="collision", figure_id="fig_p4_scenario_collision_rate", title="Scenario-Wise Collision Rate", formal_seeds=formal_seeds, m_upper=m_upper, status=core_status),
        _plot_continuous_overall(overall, settings, paper_dir, metric_key="path_length_success_team_mean_m", figure_id="fig_p5_successful_path_length", title="Path Length of Successful Episodes", formal_seeds=formal_seeds, m_upper=m_upper, status=core_status),
        _plot_continuous_overall(overall, settings, paper_dir, metric_key="completion_time_success_s", figure_id="fig_p6_successful_completion_time", title="Completion Time of Successful Episodes", formal_seeds=formal_seeds, m_upper=m_upper, status=core_status),
        _plot_continuous_overall(overall, settings, paper_dir, metric_key="minimum_inter_agent_distance_m", figure_id="fig_p7_minimum_inter_agent_distance", title="Minimum Inter-Agent Distance", formal_seeds=formal_seeds, m_upper=m_upper, status=core_status),
        _plot_continuous_overall(overall, settings, paper_dir, metric_key="goal_switch_count", figure_id="fig_p8_goal_switch_count", title="Goal-Switching Frequency", formal_seeds=formal_seeds, m_upper=m_upper, status=core_status),
    ):
        figure_metadata.append(metadata)
    case_b = next(item for item in cases if item["case_id"].startswith("B_"))
    case_h = next(item for item in cases if item["case_id"].startswith("H_"))
    for metadata in (
        _plot_trajectory_case(run_dir=run_dir, case=case_b, formal_rows=formal_rows, event_rows=event_rows, settings=settings, paper_dir=paper_dir, figure_id="fig_p9_representative_closed_loop_trajectories", title="Representative Closed-Loop Trajectories", formal_seeds=formal_seeds, m_upper=m_upper),
        _plot_trajectory_case(run_dir=run_dir, case=case_h, formal_rows=formal_rows, event_rows=event_rows, settings=settings, paper_dir=paper_dir, figure_id="fig_p10_residual_multi_agent_conflict", title="Residual Multi-Agent Conflict", formal_seeds=formal_seeds, m_upper=m_upper),
    ):
        if metadata is not None:
            figure_metadata.append(metadata)
    for case in cases:
        _plot_case_time_series(
            run_dir=run_dir,
            case=case,
            formal_rows=formal_rows,
            settings=settings,
            m_upper=m_upper,
        )
    _write_tables(overall, scenario, settings, paper_dir)

    manifest_entries = []
    for metadata in figure_metadata:
        figure_id = metadata["figure_id"]
        manifest_entries.append({
            "file": f"figures/{figure_id}.pdf + .png",
            "publication_title": metadata["publication_title"],
            "metric": metadata["metric_definition"],
            "source_data": f"figure_data/{figure_id}.csv + .json",
            "methods": ", ".join(metadata["methods"]),
            "scenarios": ", ".join(metadata["scenarios"]),
            "seeds": f"{len(metadata['seeds'])} formal seeds",
            "M_upper": metadata["M_upper"],
            "aggregation": metadata["aggregation"],
            "filter_rule": metadata["filter_rule"],
            "status": metadata["publication_status"],
        })
    for table in ("table_overall_metrics", "table_scenario_metrics"):
        manifest_entries.append({
            "file": f"tables/{table}.csv + .tex",
            "publication_title": "Overall Metrics" if "overall" in table else "Scenario-Wise Metrics",
            "metric": "closed-loop performance metrics",
            "source_data": "per_episode/all_episode_records.csv",
            "methods": ", ".join(METHOD_DISPLAY_NAMES[item] for item in METHOD_ORDER),
            "scenarios": ", ".join(settings["scenario_display_names"].values()),
            "seeds": f"{len(formal_seeds)} formal seeds",
            "M_upper": m_upper,
            "aggregation": "mean, standard deviation, median, n and raw rate counts",
            "filter_rule": "formal episodes; efficiency fields use successful episodes only",
            "status": core_status,
        })
    _manifest(paper_dir / "PAPER_READY_MANIFEST.md", manifest_entries, settings, font)
    summary = {
        "formal_episode_count": len(formal_rows),
        "formal_pair_count": len({row["pair_id"] for row in formal_rows}),
        "expected_formal_pair_count": expected_pairs,
        "formal_seed_count": len(formal_seeds),
        "formal_M_upper": m_upper,
        "paired_complete": paired_complete,
        "paper_ready_status": core_status,
        "figure_count": len(figure_metadata),
        "table_count": 2,
        "representative_cases": cases,
        "paired_transitions": paired["transitions"],
        "proposal_fp_shep_disagreement_summary": disagreement_summary,
        "diagnosis": diagnosis,
        "paper_ready_dir": str(paper_dir),
        "publication_language": "English",
        "actual_font": font,
    }
    _write_json(run_dir / "summary" / "analysis_summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = analyze_run(args.run_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
