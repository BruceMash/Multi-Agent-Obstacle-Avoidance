from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROFILE_NAMES = (
    "A_baseline",
    "B_phase_only",
    "C_constraint_only",
    "D_full",
)
PROFILE_COLORS = {
    "A_baseline": "#475569",
    "B_phase_only": "#2563eb",
    "C_constraint_only": "#d97706",
    "D_full": "#dc2626",
}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for raw in csv.DictReader(file):
            converted: dict[str, Any] = {}
            for key, value in raw.items():
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
    return rows


def discover_runs(input_root: Path) -> list[dict[str, Any]]:
    runs = []
    for config_path in input_root.rglob("config.json"):
        profile = next(
            (name for name in PROFILE_NAMES if name in config_path.parts),
            None,
        )
        if profile is None:
            continue
        run_dir = config_path.parent
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "profile": profile,
                "run_dir": run_dir,
                "seed": int(payload.get("script_args", {}).get("seed", -1)),
                "config": payload,
                "train": _read_csv(run_dir / "train_metrics.csv"),
                "episode": _read_csv(run_dir / "metrics.csv"),
                "eval": _read_csv(run_dir / "eval_metrics.csv"),
            }
        )
    return sorted(runs, key=lambda row: (row["profile"], row["seed"], str(row["run_dir"])))


def _series(run: dict[str, Any], source: str, x_key: str, y_key: str):
    points = []
    for row in run[source]:
        x = row.get(x_key)
        y = row.get(y_key)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if math.isfinite(float(x)) and math.isfinite(float(y)):
                points.append((float(x), float(y)))
    return points


def plot_metric(
    runs: list[dict[str, Any]],
    output_path: Path,
    *,
    source: str,
    x_key: str,
    y_key: str,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for profile in PROFILE_NAMES:
        profile_runs = [run for run in runs if run["profile"] == profile]
        for index, run in enumerate(profile_runs):
            points = _series(run, source, x_key, y_key)
            if not points:
                continue
            x, y = np.asarray(points).T
            axis.plot(
                x,
                y,
                color=PROFILE_COLORS[profile],
                alpha=0.35,
                linewidth=1.0,
                label=profile if index == 0 else None,
            )
    axis.set_title(title)
    axis.set_xlabel("training step")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_phase_flow_overlay(runs: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, profile in zip(axes.flat, PROFILE_NAMES):
        for run in [item for item in runs if item["profile"] == profile]:
            phase_points = _series(run, "train", "global_step", "phase_rate")
            flow_points = _series(run, "train", "global_step", "flow_consistency_mean")
            if phase_points:
                x, y = np.asarray(phase_points).T
                axis.plot(x, y, color="#2563eb", alpha=0.5, label="phase rate")
            if flow_points:
                x, y = np.asarray(flow_points).T
                axis.plot(x, y, color="#dc2626", alpha=0.5, label="flow consistency")
        axis.set_title(profile)
        axis.set_xlabel("training step")
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_scatter(
    runs: list[dict[str, Any]],
    output_path: Path,
    *,
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for profile in PROFILE_NAMES:
        x_values, y_values = [], []
        for run in [item for item in runs if item["profile"] == profile]:
            for row in run["train"]:
                x, y = row.get(x_key), row.get(y_key)
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    if math.isfinite(float(x)) and math.isfinite(float(y)):
                        x_values.append(float(x))
                        y_values.append(float(y))
        if x_values:
            axis.scatter(
                x_values,
                y_values,
                s=12,
                alpha=0.35,
                color=PROFILE_COLORS[profile],
                label=profile,
            )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for profile in PROFILE_NAMES:
        profile_runs = [run for run in runs if run["profile"] == profile]
        final_rows = [run["eval"][-1] for run in profile_runs if run["eval"]]
        train_rows = [row for run in profile_runs for row in run["train"]]

        def mean_value(rows, key):
            values = [
                float(row[key])
                for row in rows
                if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
            ]
            return float(np.mean(values)) if values else None

        divergence = any(
            not math.isfinite(float(row[key]))
            for row in train_rows
            for key in ("actor_loss", "critic_loss")
            if isinstance(row.get(key), (int, float))
        )
        summary[profile] = {
            "seeds": sorted({run["seed"] for run in profile_runs}),
            "run_count": len(profile_runs),
            "eval_success_rate_mean": mean_value(final_rows, "eval_success_rate"),
            "eval_obstacle_collision_rate_mean": mean_value(
                final_rows, "eval_obstacle_collision_rate"
            ),
            "eval_inter_agent_collision_rate_mean": mean_value(
                final_rows, "eval_inter_agent_collision_rate"
            ),
            "flow_consistency_mean": mean_value(train_rows, "flow_consistency_mean"),
            "negative_consistency_rate": mean_value(train_rows, "negative_consistency_rate"),
            "residual_forcing_norm_mean": mean_value(train_rows, "residual_forcing_norm"),
            "phase_pause_fraction": mean_value(train_rows, "phase_pause_fraction"),
            "training_nonfinite_loss_detected": divergence,
        }

    baseline = summary.get("A_baseline", {})
    for profile, row in summary.items():
        baseline_forcing = baseline.get("residual_forcing_norm_mean")
        current_forcing = row.get("residual_forcing_norm_mean")
        row["possible_forcing_collapse"] = bool(
            baseline_forcing is not None
            and current_forcing is not None
            and current_forcing < 0.2 * baseline_forcing
        )
        row["possible_long_phase_freeze"] = bool(
            row.get("phase_pause_fraction") is not None
            and row["phase_pause_fraction"] > 0.5
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze real FCEP ablation run data")
    parser.add_argument("--input-root", type=Path, default=Path("artifacts/fcep_ablation"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/fcep_analysis"))
    args = parser.parse_args()

    runs = discover_runs(args.input_root)
    if not runs:
        raise FileNotFoundError(f"no completed FCEP ablation runs found under {args.input_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    plots = (
        ("phase.png", "train", "global_step", "phase", "Phase", "phase"),
        ("flow_consistency.png", "train", "global_step", "flow_consistency_mean", "Flow consistency", "consistency"),
        ("residual_forcing.png", "train", "global_step", "residual_forcing_norm", "Residual forcing norm", "norm"),
        ("active_goal_error.png", "train", "global_step", "active_goal_error", "Goal error", "distance / m"),
        ("training_return.png", "episode", "global_step", "mean_reward", "Episode return", "mean return"),
        ("eval_success_rate.png", "eval", "global_step", "eval_success_rate", "Evaluation success rate", "success rate"),
        ("eval_collision_rate.png", "eval", "global_step", "eval_obstacle_collision_rate", "Evaluation obstacle collision rate", "collision rate"),
    )
    for filename, source, x_key, y_key, title, ylabel in plots:
        plot_metric(
            runs,
            args.output_root / filename,
            source=source,
            x_key=x_key,
            y_key=y_key,
            title=title,
            ylabel=ylabel,
        )
    plot_phase_flow_overlay(runs, args.output_root / "phase_rate_flow_overlay.png")
    plot_scatter(
        runs,
        args.output_root / "flow_vs_clearance.png",
        x_key="flow_consistency_mean",
        y_key="minimum_obstacle_clearance",
        title="Flow consistency vs. minimum obstacle clearance",
        xlabel="flow consistency",
        ylabel="clearance / m",
    )
    plot_scatter(
        runs,
        args.output_root / "flow_vs_action_saturation.png",
        x_key="flow_consistency_mean",
        y_key="action_saturation_rate",
        title="Flow consistency vs. actor action saturation",
        xlabel="flow consistency",
        ylabel="saturation rate",
    )

    report = {
        "input_root": str(args.input_root),
        "run_directories": [str(run["run_dir"]) for run in runs],
        "summary": summarize_runs(runs),
        "active_goal_switch_metrics": "not_applicable: no active/pending goal switch exists",
        "trajectory_figures": "generated by validate_fcep_mechanism.py",
        "data_policy": "all figures are generated only from discovered experiment CSV files",
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"FCEP ablation analysis output: {args.output_root}")


if __name__ == "__main__":
    main()
