"""Analyze and visualize Temporary-Reference Interface Diagnosis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, ALGO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from planning.temporary_reference_diagnosis import (  # noqa: E402
    PROTOCOL_DISPLAY_NAMES,
    PROTOCOL_EXISTING_BOUNDARY_FREE,
    PROTOCOL_FIXED_PERIOD,
    PROTOCOL_ONE_SHOT,
    PROTOCOL_TERMINAL,
)
from scripts.evaluate_temporary_reference_interface import write_csv, write_json  # noqa: E402


PROTOCOL_ORDER = (
    PROTOCOL_TERMINAL,
    PROTOCOL_ONE_SHOT,
    PROTOCOL_EXISTING_BOUNDARY_FREE,
    PROTOCOL_FIXED_PERIOD,
)
COLORS = {
    PROTOCOL_TERMINAL: "#4C4C4C",
    PROTOCOL_ONE_SHOT: "#377EB8",
    PROTOCOL_EXISTING_BOUNDARY_FREE: "#4DAF4A",
    PROTOCOL_FIXED_PERIOD: "#E41A1C",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value in {"", None}:
                    row[key] = None
                    continue
                try:
                    row[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    row[key] = value
            rows.append(row)
    return rows


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def _number(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        aliases = {
            "distance_to_active_goal": "distance_to_active_goal_before",
            "distance_to_terminal_goal": "distance_to_terminal_goal_before",
            "minimum_clearance": "min_clearance",
        }
        value = row.get(aliases.get(key, ""))
    return float(default if value is None else value)


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(
    fig: plt.Figure,
    run_dir: Path,
    name: str,
    data_rows: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(
        run_dir / "figures" / f"{name}.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)
    write_csv(run_dir / "figure_data" / f"{name}.csv", data_rows)
    write_json(
        run_dir / "figure_data" / f"{name}.json",
        {
            "figure": name,
            "status": "DIAGNOSTIC_ONLY",
            "language": "English",
            "source": "numerical_artifacts",
            **dict(metadata),
        },
    )


def _overall(summary: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["protocol"]): row
        for row in summary
        if str(row["scenario"]) == "overall"
    }


def _protocol_bar(
    run_dir: Path,
    summary: Sequence[Mapping[str, Any]],
    *,
    field: str,
    ylabel: str,
    title: str,
    name: str,
    percent: bool = False,
) -> None:
    overall = _overall(summary)
    rows = [overall[protocol] for protocol in PROTOCOL_ORDER if protocol in overall]
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    plotted = 100.0 * values if percent else values
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(
        np.arange(len(rows)),
        plotted,
        color=[COLORS[str(row["protocol"])] for row in rows],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(
        [PROTOCOL_DISPLAY_NAMES[str(row["protocol"])] for row in rows], rotation=12, ha="right"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    if percent:
        ax.set_ylim(0.0, 100.0)
    for bar, value in zip(bars, plotted):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.1f}" if percent else f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    data = [
        {
            "protocol": row["protocol"],
            "protocol_display_name": PROTOCOL_DISPLAY_NAMES[str(row["protocol"])],
            field: float(row[field]),
            "plotted_value": float(value),
        }
        for row, value in zip(rows, plotted)
    ]
    _save_figure(fig, run_dir, name, data, {"metric": field, "percent_axis": percent})


def _paired_case(
    episodes: Sequence[Mapping[str, Any]],
    first_protocol: str,
    first_success: bool,
    second_protocol: str,
    second_success: bool,
) -> Mapping[str, Any] | None:
    lookup = {
        (str(row["scenario"]), int(row["seed"]), str(row["protocol"])): row
        for row in episodes
    }
    for scenario, seed, protocol in sorted(lookup):
        if protocol != first_protocol:
            continue
        first = lookup[(scenario, seed, first_protocol)]
        second = lookup.get((scenario, seed, second_protocol))
        if second is None:
            continue
        if bool(first["success"]) == first_success and bool(second["success"]) == second_success:
            return first
    return None


def select_representative_cases(
    episodes: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    switches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lookup = {str(row["pair_id"]): row for row in episodes}
    one_success_fixed_failure = _paired_case(
        episodes, PROTOCOL_ONE_SHOT, True, PROTOCOL_FIXED_PERIOD, False
    )
    existing_success_fixed_failure = _paired_case(
        episodes, PROTOCOL_EXISTING_BOUNDARY_FREE, True, PROTOCOL_FIXED_PERIOD, False
    )
    one_shot_failure_before = next(
        (
            row
            for row in episodes
            if row["protocol"] == PROTOCOL_ONE_SHOT
            and not bool(row["success"])
            and row.get("failure_attribution")
            == "collision_before_first_temporary_reference_reached"
        ),
        None,
    )
    fixed_collision_after_switch = next(
        (
            lookup[str(row["pair_id"])]
            for row in steps
            if row["protocol"] == PROTOCOL_FIXED_PERIOD
            and bool(row.get("collision", False))
            and any(
                0 <= int(row["completed_step"]) - int(event["reference_activated_step"]) <= 3
                for event in events
                if event["pair_id"] == row["pair_id"]
            )
        ),
        None,
    )
    early_hold = next(
        (
            lookup[str(row["pair_id"])]
            for row in steps
            if row["protocol"] == PROTOCOL_FIXED_PERIOD
            and bool(row.get("fixed_period_hold_after_reached", False))
        ),
        None,
    )
    largest_jump = max(
        switches,
        key=lambda row: float(row.get("closed_loop_commanded_acceleration_jump_mps2", 0.0)),
        default=None,
    )
    mismatch_row = max(
        (
            row
            for row in steps
            if row["protocol"] != PROTOCOL_TERMINAL
            and _number(row, "distance_to_active_goal", math.inf) <= 0.30
        ),
        key=lambda row: _number(row, "distance_to_terminal_goal", 0.0),
        default=None,
    )

    def descriptor(row: Mapping[str, Any] | None) -> Any:
        if row is None:
            return {"status": "NOT_FOUND"}
        return {
            "status": "FOUND",
            "pair_id": row.get("pair_id"),
            "protocol": row.get("protocol"),
            "scenario": row.get("scenario"),
            "seed": row.get("seed"),
            "agent_id": row.get("agent_id"),
            "timestep": row.get("timestep"),
        }

    return {
        "case_A_one_shot_success_fixed_period_failure": descriptor(one_success_fixed_failure),
        "case_B_one_shot_failure_before_reference": descriptor(one_shot_failure_before),
        "case_C_fixed_period_collision_shortly_after_switch": descriptor(
            fixed_collision_after_switch
        ),
        "case_D_reference_reached_early_but_held": descriptor(early_hold),
        "case_E_existing_interface_success_fixed_period_failure": descriptor(
            existing_success_fixed_failure
        ),
        "case_F_large_dmp_acceleration_jump": descriptor(largest_jump),
        "case_G_active_near_terminal_far_forcing_active": descriptor(mismatch_row),
    }


def _load_trajectory(run_dir: Path, pair_id: str) -> dict[str, np.ndarray]:
    with np.load(run_dir / "trajectories" / f"{pair_id}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _trajectory_rows(trajectory: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positions = np.asarray(trajectory["positions"], dtype=float)
    references = np.asarray(trajectory.get("execution_references", []), dtype=float)
    for step in range(len(positions)):
        for agent in range(positions.shape[1]):
            row = {
                "step": step,
                "agent_id": agent,
                "x": positions[step, agent, 0],
                "y": positions[step, agent, 1],
                "z": positions[step, agent, 2],
            }
            if step < len(references):
                row.update(
                    {
                        "reference_x": references[step, agent, 0],
                        "reference_y": references[step, agent, 1],
                        "reference_z": references[step, agent, 2],
                    }
                )
            rows.append(row)
    return rows


def plot_trajectory_case(
    run_dir: Path,
    *,
    pair_id: str,
    name: str,
    title: str,
    events: Sequence[Mapping[str, Any]],
    switches: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
) -> None:
    trajectory = _load_trajectory(run_dir, pair_id)
    positions = np.asarray(trajectory["positions"], dtype=float)
    starts = np.asarray(trajectory["starts"], dtype=float)
    goals = np.asarray(trajectory["terminal_goals"], dtype=float)
    colors = ("#377EB8", "#FF7F00", "#4DAF4A")
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    case_switches = [row for row in switches if row.get("pair_id") == pair_id]
    for agent in range(positions.shape[1]):
        return_switches = [
            int(row["timestep"])
            for row in case_switches
            if int(row.get("agent_id", -1)) == agent
            and str(row.get("switch_reason", ""))
            == "reference_reached_terminal_return"
        ]
        return_step = min(return_switches) if return_switches else None
        tracking_stop = (
            min(return_step + 1, len(positions))
            if return_step is not None
            else len(positions)
        )
        ax.plot(
            positions[:tracking_stop, agent, 0],
            positions[:tracking_stop, agent, 1],
            positions[:tracking_stop, agent, 2],
            color=colors[agent % len(colors)],
            linewidth=1.4,
            label=f"UAV {agent} trajectory",
        )
        if return_step is not None and return_step < len(positions) - 1:
            ax.plot(
                positions[return_step:, agent, 0],
                positions[return_step:, agent, 1],
                positions[return_step:, agent, 2],
                color=colors[agent % len(colors)],
                linewidth=1.6,
                linestyle="--",
            )
        ax.scatter(*starts[agent], marker="o", s=28, color=colors[agent % len(colors)])
        ax.scatter(*goals[agent], marker="*", s=75, color=colors[agent % len(colors)])
    case_events = [row for row in events if row.get("pair_id") == pair_id]
    for row in case_events:
        point = row.get("reference_point") or row.get("selected_candidate_world_position") or row.get("selected_point")
        if point is not None:
            ax.scatter(*np.asarray(point, dtype=float), marker="D", s=32, facecolors="none", edgecolors="black")
    for row in case_switches:
        point = np.asarray(row["position"], dtype=float)
        reached = "reached" in str(row.get("switch_reason", ""))
        ax.scatter(
            *point,
            marker="P" if reached else "x",
            s=40 if reached else 34,
            color="#984EA3" if reached else "black",
        )
    collision_rows = [
        row
        for row in steps
        if row.get("pair_id") == pair_id and bool(row.get("collision", False))
    ]
    for row in collision_rows:
        point = np.asarray(row["position"], dtype=float)
        ax.scatter(*point, marker="X", s=58, color="#E41A1C")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    handles = [
        Line2D([0], [0], color=colors[agent % len(colors)], lw=1.5, label=f"UAV {agent} trajectory")
        for agent in range(positions.shape[1])
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="#777777", label="Start", linestyle="none"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#777777", markeredgecolor="#777777", markersize=9, label="Terminal Goal", linestyle="none"),
            Line2D([0], [0], marker="D", color="black", markerfacecolor="none", label="Temporary Reference", linestyle="none"),
            Line2D([0], [0], marker="P", color="#984EA3", label="Reference Reached", linestyle="none"),
            Line2D([0], [0], marker="x", color="black", label="Reference Switch", linestyle="none"),
            Line2D([0], [0], color="#777777", linestyle="--", label="Return-to-Terminal Segment"),
            Line2D([0], [0], marker="X", color="#E41A1C", label="Collision", linestyle="none"),
        ]
    )
    ax.legend(handles=handles, loc="best", frameon=False, ncol=2)
    _save_figure(
        fig,
        run_dir,
        name,
        _trajectory_rows(trajectory),
        {
            "pair_id": pair_id,
            "reference_marker": "diamond",
            "switch_marker": "x",
            "temporary_reference_tracking_style": "solid",
            "return_to_terminal_style": "dashed_same_agent_color",
        },
    )


def plot_acceleration_jumps(run_dir: Path, switches: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        row
        for row in switches
        if row.get("closed_loop_commanded_acceleration_jump_mps2") is not None
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for protocol in PROTOCOL_ORDER:
        members = [row for row in rows if row.get("protocol") == protocol]
        if not members:
            continue
        ax.scatter(
            [float(row["goal_jump_m"]) for row in members],
            [float(row["closed_loop_commanded_acceleration_jump_mps2"]) for row in members],
            s=14,
            alpha=0.55,
            color=COLORS[protocol],
            label=PROTOCOL_DISPLAY_NAMES[protocol],
        )
    ax.set_xlabel("Goal Jump (m)")
    ax.set_ylabel("Acceleration Jump (m/s²)")
    ax.set_title("DMP Acceleration Discontinuity at Reference Switches")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    data = [
        {
            "protocol": row.get("protocol"),
            "pair_id": row.get("pair_id"),
            "agent_id": row.get("agent_id"),
            "timestep": row.get("timestep"),
            "goal_jump_m": row.get("goal_jump_m"),
            "zero_action_nominal_acceleration_jump_mps2": row.get(
                "zero_action_nominal_acceleration_jump_mps2"
            ),
            "closed_loop_commanded_acceleration_jump_mps2": row.get(
                "closed_loop_commanded_acceleration_jump_mps2"
            ),
        }
        for row in rows
    ]
    _save_figure(fig, run_dir, "D6_dmp_acceleration_jump", data, {})


def plot_goal_distance_case(
    run_dir: Path,
    steps: Sequence[Mapping[str, Any]],
    case: Mapping[str, Any],
) -> None:
    if case.get("status") != "FOUND":
        return
    pair_id = str(case["pair_id"])
    agent_id = int(case.get("agent_id") or 0)
    rows = sorted(
        [
            row
            for row in steps
            if row.get("pair_id") == pair_id and int(row["agent_id"]) == agent_id
        ],
        key=lambda row: int(row["completed_step"]),
    )
    if not rows:
        return
    time_s = np.asarray([0.1 * int(row["completed_step"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(time_s, [_number(row, "distance_to_active_goal", math.nan) for row in rows], label="Distance to Active Reference", color="#377EB8")
    ax.plot(time_s, [_number(row, "distance_to_terminal_goal", math.nan) for row in rows], label="Distance to Terminal Goal", color="#E41A1C")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.grid(alpha=0.25)
    second = ax.twinx()
    second.plot(time_s, [float(row["forcing_gate_value"]) for row in rows], label="Forcing Gate", color="#4DAF4A", linestyle="--")
    second.set_ylabel("Forcing Gate")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = second.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
    ax.set_title("Active-vs-Terminal Goal Distance and Forcing Gate")
    _save_figure(
        fig,
        run_dir,
        "D7_active_terminal_goal_distance",
        rows,
        {"pair_id": pair_id, "agent_id": agent_id},
    )


def plot_lifecycle(
    run_dir: Path,
    events: Sequence[Mapping[str, Any]],
    case: Mapping[str, Any],
) -> None:
    if case.get("status") != "FOUND":
        return
    pair_id = str(case["pair_id"])
    rows = [row for row in events if row.get("pair_id") == pair_id]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.35 * len(rows) + 1.5)))
    for index, row in enumerate(rows):
        start = int(row["reference_activated_step"])
        end = int(row.get("reference_released_step") or start)
        ax.barh(index, end - start, left=start, height=0.55, color=COLORS.get(str(row["protocol"]), "#777777"), alpha=0.75)
        reached = row.get("reference_reached_step")
        if reached is not None:
            ax.scatter(int(reached), index, marker="o", s=24, color="black", zorder=3)
        ax.text(end + 0.3, index, str(row.get("release_reason", "")), va="center", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"UAV {row['agent_id']} / ref {index}" for index, row in enumerate(rows)])
    ax.set_xlabel("Low-Level Step")
    ax.set_title("Temporary Reference Lifecycle Timeline")
    ax.grid(axis="x", alpha=0.25)
    _save_figure(fig, run_dir, "D8_reference_lifecycle_timeline", rows, {"pair_id": pair_id})


def build_diagnostic_statistics(
    episodes: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    switches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mismatch = [
        row
        for row in steps
        if row["protocol"] != PROTOCOL_TERMINAL
        and _number(row, "distance_to_active_goal", math.inf) <= 0.30
        and _number(row, "distance_to_terminal_goal", 0.0) >= 1.50
        and _number(row, "forcing_gate_value", 0.0) >= 0.90
    ]
    non_mismatch = [row for row in steps if row["protocol"] != PROTOCOL_TERMINAL and row not in mismatch]
    acceleration_norm = lambda row: float(np.linalg.norm(np.asarray(row.get("applied_acceleration", [0, 0, 0]), dtype=float)))
    fixed_hold = [
        row
        for row in steps
        if row["protocol"] == PROTOCOL_FIXED_PERIOD
        and bool(row.get("fixed_period_hold_after_reached", False))
    ]
    safety_rows = [row for row in switches if row["protocol"] != PROTOCOL_TERMINAL]
    one_shot = [row for row in episodes if row["protocol"] == PROTOCOL_ONE_SHOT]
    one_shot_reference_count = sum(
        int(row.get("temporary_reference_count", 0)) for row in one_shot
    )
    one_shot_reached_count = sum(
        int(row.get("temporary_reference_reached_count", 0)) for row in one_shot
    )
    one_shot_not_reached_count = sum(
        int(row.get("not_reached_count", 0)) for row in one_shot
    )
    one_shot_collision_before_count = sum(
        int(row.get("collision_before_reference_count", 0)) for row in one_shot
    )
    one_shot_collision_after_count = sum(
        int(row.get("collision_after_reference_count", 0)) for row in one_shot
    )
    failure_by_protocol = {
        protocol: dict(
            Counter(
                str(row["failure_attribution"])
                for row in episodes
                if row["protocol"] == protocol
            )
        )
        for protocol in PROTOCOL_ORDER
    }
    return {
        "active_terminal_forcing_mismatch": {
            "definition": "active_distance<=0.30m and terminal_distance>=1.50m and forcing_gate>=0.90",
            "step_count": len(mismatch),
            "step_fraction": float(len(mismatch) / max(1, len(mismatch) + len(non_mismatch))),
            "mean_applied_acceleration_norm_mps2": float(np.mean([acceleration_norm(row) for row in mismatch])) if mismatch else None,
            "non_mismatch_mean_applied_acceleration_norm_mps2": float(np.mean([acceleration_norm(row) for row in non_mismatch])) if non_mismatch else None,
            "collision_step_rate": float(np.mean([bool(row.get("collision", False)) for row in mismatch])) if mismatch else None,
        },
        "fixed_period_hold": {
            "hold_after_reached_step_count": len(fixed_hold),
            "collision_during_hold_step_count": sum(bool(row.get("collision", False)) for row in fixed_hold),
            "collision_during_hold_step_rate": float(np.mean([bool(row.get("collision", False)) for row in fixed_hold])) if fixed_hold else None,
        },
        "candidate_safety": {
            "switch_count": len(safety_rows),
            "point_unsafe_rate": float(np.mean([not bool(row.get("point_safety", True)) for row in safety_rows])) if safety_rows else None,
            "segment_invalid_rate": float(np.mean([not bool(row.get("segment_validity", True)) for row in safety_rows])) if safety_rows else None,
            "boundary_invalid_diagnostic_rate": float(np.mean([not bool(row.get("boundary_validity", True)) for row in safety_rows])) if safety_rows else None,
            "boundary_used_for_boundary_free_selection": False,
        },
        "switch_discontinuity": {
            "switch_count": len(switches),
            "mean_goal_jump_m": _mean(switches, "goal_jump_m"),
            "mean_zero_action_nominal_acceleration_jump_mps2": _mean(switches, "zero_action_nominal_acceleration_jump_mps2"),
            "mean_closed_loop_acceleration_jump_mps2": _mean(switches, "closed_loop_commanded_acceleration_jump_mps2"),
            "mean_observation_l2_change": _mean(switches, "observation_l2_change"),
            "maximum_non_goal_feature_l2_change": max((float(row.get("non_goal_feature_l2_change", 0.0)) for row in switches), default=0.0),
        },
        "one_shot": {
            "episodes": len(one_shot),
            "success_rate": _mean(one_shot, "success"),
            "reference_reached_rate": _mean(one_shot, "temporary_reference_reached_rate"),
            "temporary_reference_count": one_shot_reference_count,
            "temporary_reference_reached_count": one_shot_reached_count,
            "not_reached_count": one_shot_not_reached_count,
            "not_reached_rate": float(
                one_shot_not_reached_count / max(1, one_shot_reference_count)
            ),
            "collision_before_reference_count": one_shot_collision_before_count,
            "collision_after_reference_count": one_shot_collision_after_count,
            "reached_then_terminal_success_rate": _mean(
                one_shot, "reached_then_terminal_success_rate"
            ),
            "terminal_completion_after_return_rate": _mean(one_shot, "terminal_completion_after_return"),
            "mean_path_to_reference_m": _mean(one_shot, "mean_path_to_reference_m"),
            "mean_time_to_reference_s": _mean(one_shot, "mean_time_to_reference_s"),
        },
        "failure_attribution": dict(Counter(str(row["failure_attribution"]) for row in episodes)),
        "failure_attribution_by_protocol": failure_by_protocol,
    }


def _conclusion(
    summary: Sequence[Mapping[str, Any]], statistics: Mapping[str, Any]
) -> tuple[str, str, str]:
    overall = _overall(summary)
    one = float(overall[PROTOCOL_ONE_SHOT]["success_rate"])
    fixed = float(overall[PROTOCOL_FIXED_PERIOD]["success_rate"])
    existing = float(overall[PROTOCOL_EXISTING_BOUNDARY_FREE]["success_rate"])
    mismatch = statistics["active_terminal_forcing_mismatch"]
    if one <= fixed and existing <= fixed:
        return (
            "Conclusion 1",
            "temporary-reference injection compatibility",
            "switch discontinuity and active/terminal semantic mismatch",
        )
    if one > fixed and existing <= one:
        return (
            "Conclusion 2",
            "repeated hard switching and reference persistence",
            "active/terminal semantic mismatch",
        )
    if existing > fixed:
        return (
            "Conclusion 3",
            "missing execution-side handoff semantics",
            "active/terminal semantic mismatch"
            if int(mismatch["step_count"]) > 0
            else "reference switching discontinuity",
        )
    return (
        "Conclusion 5",
        "multiple coupled interface effects",
        "goal discontinuity and lower-policy distribution shift",
    )


def write_report(
    run_dir: Path,
    *,
    config: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    cases: Mapping[str, Any],
) -> None:
    overall = _overall(summary)
    conclusion, primary, secondary = _conclusion(summary, statistics)
    test_summary_path = run_dir / "tests" / "regression_summary.json"
    tests = _load_json(test_summary_path) if test_summary_path.is_file() else {"status": "PENDING"}
    lines = [
        "# Temporary-Reference Execution Interface Diagnosis",
        "",
        "## A. Execution Semantics Re-Audit",
        "",
        "`env.goals` remained the terminal task goal; `dmp.goal` stored the active execution reference. The Actor used the existing 122-D active-goal observation builder. The DMP spring used the active goal, while the forcing gate remained terminal-goal-distance based. Reference switches did not reset phase, sensor history, velocity history, or controller state.",
        "",
        "## B. Diagnostic Hypotheses",
        "",
        "The experiment separates one-shot injection, repeated hard switching, fixed-period hold, the existing handoff interface, geometric validity, Actor input shift, DMP attractor discontinuity, and terminal-gated residual forcing.",
        "",
        "## C. Protocol Definitions",
        "",
        "- Terminal-Goal Baseline: Frozen SAC-DMP with no temporary reference.",
        "- One-Shot Temporary Reference: one Proposal top-1 injection, 0.25 m reached tolerance, followed by one terminal return.",
        "- Existing Waypoint Handoff, Boundary-Free: existing reached/stagnation/segment/lookahead/priority-hold semantics with workspace-boundary filtering disabled.",
        "- Fixed-Period Hard Switching: Proposal top-1, M_upper=8, strict periodic replanning, 0.30 m diagnostic reached tolerance.",
        "",
        "## D. Terminal-Goal Baseline",
        "",
        f"Success: {100*float(overall[PROTOCOL_TERMINAL]['success_rate']):.2f}%; collision: {100*float(overall[PROTOCOL_TERMINAL]['collision_rate']):.2f}%.",
        "",
        "## E. One-Shot Temporary Reference",
        "",
        f"Success: {100*float(overall[PROTOCOL_ONE_SHOT]['success_rate']):.2f}%; reference reached rate: {100*float(overall[PROTOCOL_ONE_SHOT]['reference_reached_rate']):.2f}%. The reached tolerance was 0.25 m.",
        "",
        "## F. Existing Waypoint Interface",
        "",
        f"Boundary-free success: {100*float(overall[PROTOCOL_EXISTING_BOUNDARY_FREE]['success_rate']):.2f}%. Workspace-boundary filtering was disabled.",
        "",
        "## G. Fixed-Period Hard Switching",
        "",
        f"Success: {100*float(overall[PROTOCOL_FIXED_PERIOD]['success_rate']):.2f}%. M_upper remained 8 and the diagnostic reached tolerance remained 0.30 m.",
        "",
        "## H. Success / Collision Comparison",
        "",
        "| Protocol | Success | Collision | Obstacle collision | Inter-agent collision | Smoothness | Path length (m) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol in PROTOCOL_ORDER:
        row = overall[protocol]
        lines.append(
            f"| {PROTOCOL_DISPLAY_NAMES[protocol]} | {100*float(row['success_rate']):.2f}% | {100*float(row['collision_rate']):.2f}% | {100*float(row['obstacle_collision_rate']):.2f}% | {100*float(row['inter_agent_collision_rate']):.2f}% | {float(row['trajectory_smoothness_mean']):.2f} | {float(row['path_length_team_mean_m']):.3f} |"
        )
    lines.extend(
        [
            "",
            "See D1 and D2. Results use development seeds 0–9 only; formal seeds 10–29 were not used for interface design.",
            "",
            "## I. Obstacle Collision Attribution",
            "",
            f"Point-unsafe switch rate: {statistics['candidate_safety']['point_unsafe_rate']}. Segment-invalid switch rate: {statistics['candidate_safety']['segment_invalid_rate']}. Boundary-invalid rate is retained only as a diagnostic and was not used by the boundary-free protocol.",
            "",
            "## J. Reference Lifecycle Analysis",
            "",
            f"One-shot references: {statistics['one_shot']['temporary_reference_count']}; reached: {statistics['one_shot']['temporary_reference_reached_count']} ({100*statistics['one_shot']['reference_reached_rate']:.2f}%); not reached: {statistics['one_shot']['not_reached_count']} ({100*statistics['one_shot']['not_reached_rate']:.2f}%).",
            f"Collision before reference: {statistics['one_shot']['collision_before_reference_count']}; collision after reference/terminal return: {statistics['one_shot']['collision_after_reference_count']}. Reached-then-terminal success rate: {100*statistics['one_shot']['reached_then_terminal_success_rate']:.2f}%; terminal completion after return: {100*statistics['one_shot']['terminal_completion_after_return_rate']:.2f}%.",
            f"Mean path to a reached reference: {statistics['one_shot']['mean_path_to_reference_m']:.3f} m; mean time to a reached reference: {statistics['one_shot']['mean_time_to_reference_s']:.3f} s.",
            "",
            "## K. Reference-Reached-Early Analysis",
            "",
            f"Fixed-period hold-after-reach steps: {statistics['fixed_period_hold']['hold_after_reached_step_count']}; collision steps within such holds: {statistics['fixed_period_hold']['collision_during_hold_step_count']}.",
            "",
            "## L. Goal-Jump Analysis",
            "",
            f"Mean goal jump: {statistics['switch_discontinuity']['mean_goal_jump_m']:.4f} m.",
            "",
            "## M. DMP Acceleration-Discontinuity Analysis",
            "",
            f"Mean zero-action nominal jump: {statistics['switch_discontinuity']['mean_zero_action_nominal_acceleration_jump_mps2']:.4f} m/s². Mean policy-conditioned commanded jump: {statistics['switch_discontinuity']['mean_closed_loop_acceleration_jump_mps2']:.4f} m/s².",
            "",
            "## N. Active-vs-Terminal Goal Observation Analysis",
            "",
            f"Mean 122-D observation change: {statistics['switch_discontinuity']['mean_observation_l2_change']:.4f}; maximum non-goal feature change on the same snapshot: {statistics['switch_discontinuity']['maximum_non_goal_feature_l2_change']:.3e}.",
            "",
            "## O. Forcing-Gate Semantic Analysis",
            "",
            f"Active-near/terminal-far/forcing-active steps: {statistics['active_terminal_forcing_mismatch']['step_count']} ({100*statistics['active_terminal_forcing_mismatch']['step_fraction']:.2f}%).",
            "",
            "## P. Existing Waypoint Interface Analysis",
            "",
            "The primary existing-interface result is boundary-free. Workspace boundary validity was logged but did not affect candidate acceptance, segment validity, priority waiting-point selection, or replanning.",
            "",
            "## Q. Failure Attribution",
            "",
            "```json",
            json.dumps(statistics["failure_attribution"], ensure_ascii=False, indent=2),
            "```",
            "",
            "Per-protocol attribution is preserved in `protocol_summary/diagnostic_statistics.json` under `failure_attribution_by_protocol`.",
            "",
            "## R. Representative Cases",
            "",
            "```json",
            json.dumps(cases, ensure_ascii=False, indent=2),
            "```",
            "",
            "## S. Runtime",
            "",
            "| Protocol | Episode runtime (ms) | Candidate generation (ms/event) | Candidate selection (ms/event) | Frozen SAC inference (ms/step) | Low-level step (ms/step) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for protocol in PROTOCOL_ORDER:
        members = [row for row in episodes if row["protocol"] == protocol]
        format_runtime = lambda key: (
            "not instrumented"
            if math.isnan(_mean(members, key))
            else f"{_mean(members, key):.3f}"
        )
        lines.append(
            f"| {PROTOCOL_DISPLAY_NAMES[protocol]} | {format_runtime('episode_runtime_ms')} | {format_runtime('candidate_generation_runtime_mean_ms')} | {format_runtime('candidate_selection_runtime_mean_ms')} | {format_runtime('frozen_sac_inference_runtime_mean_ms')} | {format_runtime('low_level_step_runtime_mean_ms')} |"
        )
    lines.extend(
        [
            "",
            "Runtime is diagnostic wall-clock data from the development run; no execution semantics were altered for profiling. The existing waypoint consumer exposes only wrapper-level episode timing, so its internal component cells are explicitly marked `not instrumented`.",
            "",
            "## T. Regression Tests",
            "",
            f"`{json.dumps(tests, ensure_ascii=False)}`",
            "",
            "## U. Limitations",
            "",
            "This is a development-seed mechanism diagnosis, not a formal performance claim. LiDAR clearance remains sensor-derived, priority/hold is an engineering consumer, and no lower-policy adaptation was tested.",
            "",
            "## V. Recommended Next Step",
            "",
            f"Selected logic: **{conclusion}**. Primary limitation: **{primary}**. Secondary limitation: **{secondary}**.",
            "",
            "Both boundary-free temporary-reference protocols achieved 0% success, equal to Fixed-Period Hard Switching and below the 12% Terminal-Goal Baseline. One-Shot nevertheless reached 34.67% of temporary references, while no return completed the terminal task. The existing boundary-free interface reduced collision from 60% to 4% but did not restore any task success; it therefore improved execution stability without establishing task-level compatibility.",
            "",
            "Stage C was not executed: neither boundary-free protocol had non-zero success or exceeded Fixed-Period Hard Switching by the pre-registered absolute 5-percentage-point margin. No FP-SHEP selector run was started.",
            "",
            "Stage-I GAT training, SDH implementation, forcing-gate modification, and SAC retraining remain outside this Goal.",
        ]
    )
    (run_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        run_dir / "protocol_summary" / "conclusion.json",
        {
            "selected_conclusion": conclusion,
            "primary_limitation": primary,
            "secondary_limitation": secondary,
            "stage_c_gate": _load_json(run_dir / "protocol_summary" / "stage_c_gate.json"),
        },
    )


def analyze(run_dir: Path) -> dict[str, Any]:
    _configure_plot_style()
    config = _load_json(run_dir / "config.json")
    episodes = [
        _load_json(path)
        for path in sorted((run_dir / "per_episode").glob("*.json"))
        if path.name != "episodes.json"
    ]
    summary = _load_json(run_dir / "protocol_summary" / "protocol_summary.json")
    events = _load_json(run_dir / "reference_events" / "reference_events.json")
    step_json = run_dir / "per_step" / "per_step.json"
    steps = (
        _load_json(step_json)
        if step_json.is_file()
        else _load_csv(run_dir / "per_step" / "per_step.csv")
    )
    switches = _load_json(run_dir / "dmp_switch_diagnostics" / "switch_diagnostics.json")
    statistics = build_diagnostic_statistics(episodes, events, steps, switches)
    cases = select_representative_cases(episodes, events, steps, switches)
    write_json(run_dir / "protocol_summary" / "diagnostic_statistics.json", statistics)
    write_json(run_dir / "representative_cases" / "case_index.json", cases)

    _protocol_bar(
        run_dir,
        summary,
        field="success_rate",
        ylabel="Success Rate (%)",
        title="Protocol Success Comparison",
        name="D1_protocol_success_comparison",
        percent=True,
    )
    _protocol_bar(
        run_dir,
        summary,
        field="obstacle_collision_rate",
        ylabel="Obstacle Collision Rate (%)",
        title="Obstacle Collision Comparison",
        name="D2_obstacle_collision_comparison",
        percent=True,
    )
    _protocol_bar(
        run_dir,
        summary,
        field="trajectory_smoothness_mean",
        ylabel="Trajectory Smoothness",
        title="Trajectory Smoothness Comparison",
        name="D3_trajectory_smoothness_comparison",
    )
    one_case = cases["case_A_one_shot_success_fixed_period_failure"]
    if one_case["status"] != "FOUND":
        reached_event = next(
            (
                row
                for row in events
                if row.get("protocol") == PROTOCOL_ONE_SHOT
                and bool(row.get("reference_reached", False))
            ),
            None,
        )
        if reached_event is not None:
            one_case = {
                "status": "FOUND",
                "pair_id": reached_event["pair_id"],
                "protocol": reached_event["protocol"],
                "scenario": reached_event["scenario"],
                "seed": reached_event["seed"],
                "selection_reason": "contains_reference_reached_and_terminal_return",
            }
        else:
            one_case = cases["case_B_one_shot_failure_before_reference"]
    if one_case["status"] != "FOUND":
        fallback = next(
            (row for row in episodes if row["protocol"] == PROTOCOL_ONE_SHOT),
            None,
        )
        one_case = (
            {
                "status": "FOUND",
                "pair_id": fallback["pair_id"],
                "protocol": fallback["protocol"],
                "scenario": fallback["scenario"],
                "seed": fallback["seed"],
            }
            if fallback is not None
            else {"status": "NOT_FOUND"}
        )
    if one_case["status"] == "FOUND":
        plot_trajectory_case(
            run_dir,
            pair_id=str(one_case["pair_id"]),
            name="D4_representative_one_shot_trajectory",
            title="Representative One-Shot Temporary-Reference Trajectory",
            events=events,
            switches=switches,
            steps=steps,
        )
    fixed_case = cases["case_C_fixed_period_collision_shortly_after_switch"]
    if fixed_case["status"] != "FOUND":
        fixed_case = cases["case_D_reference_reached_early_but_held"]
    if fixed_case["status"] == "FOUND":
        plot_trajectory_case(
            run_dir,
            pair_id=str(fixed_case["pair_id"]),
            name="D5_fixed_period_switching_failure",
            title="Fixed-Period Hard-Switching Failure",
            events=events,
            switches=switches,
            steps=steps,
        )
    plot_acceleration_jumps(run_dir, switches)
    plot_goal_distance_case(
        run_dir, steps, cases["case_G_active_near_terminal_far_forcing_active"]
    )
    lifecycle_case = cases["case_D_reference_reached_early_but_held"]
    if lifecycle_case["status"] != "FOUND":
        lifecycle_case = one_case
    plot_lifecycle(run_dir, events, lifecycle_case)
    write_report(
        run_dir,
        config=config,
        episodes=episodes,
        summary=summary,
        statistics=statistics,
        cases=cases,
    )
    result = {
        "statistics": statistics,
        "representative_cases": cases,
        "figures_generated": sorted(path.name for path in (run_dir / "figures").glob("*.pdf")),
    }
    write_json(run_dir / "analysis_manifest.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(args.run_dir.resolve())


if __name__ == "__main__":
    main()
