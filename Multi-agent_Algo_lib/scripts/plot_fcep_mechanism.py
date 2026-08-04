from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for raw in csv.DictReader(file):
            row = dict(raw)
            for key in (
                "step",
                "agent",
                "phase",
                "flow_consistency",
                "phase_rate",
                "forcing_norm",
                "goal_error",
                "x",
                "y",
            ):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def plot_scenario(output_path: Path, scenario_name: str, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    colors = {"classic": "#2563eb", "fcep": "#dc2626"}
    for phase_mode in ("classic", "fcep"):
        selected = [row for row in rows if row["phase_mode"] == phase_mode]
        steps = sorted({int(row["step"]) for row in selected})
        for axis, key, label in (
            (axes[0, 0], "phase", "phase"),
            (axes[0, 1], "flow_consistency", "flow consistency"),
            (axes[1, 0], "phase_rate", "phase rate"),
            (axes[1, 1], "forcing_norm", "residual forcing norm"),
            (axes[2, 0], "goal_error", "active goal error"),
        ):
            values = [
                np.mean([row[key] for row in selected if int(row["step"]) == step])
                for step in steps
            ]
            axis.plot(steps, values, label=phase_mode, color=colors[phase_mode])
            axis.set_xlabel("step")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
        for agent_index in sorted({int(row["agent"]) for row in selected}):
            trajectory = [row for row in selected if int(row["agent"]) == agent_index]
            axes[2, 1].plot(
                [row["x"] for row in trajectory],
                [row["y"] for row in trajectory],
                color=colors[phase_mode],
                linestyle="-" if phase_mode == "classic" else "--",
                alpha=0.8,
                label=f"{phase_mode}/agent_{agent_index}",
            )
    for axis in axes.flat[:5]:
        axis.legend()
    axes[2, 1].set_xlabel("x / m")
    axes[2, 1].set_ylabel("y / m")
    axes[2, 1].set_title("trajectory projection")
    axes[2, 1].grid(alpha=0.25)
    axes[2, 1].legend(fontsize=8, ncol=2)
    figure.suptitle(f"Frozen-policy FCEP mechanism: {scenario_name}")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FCEP mechanism traces without PyTorch")
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.trace_csv)
    for scenario_name in sorted({row["scenario"] for row in rows}):
        plot_scenario(
            args.output_dir / f"{scenario_name}.png",
            scenario_name,
            [row for row in rows if row["scenario"] == scenario_name],
        )


if __name__ == "__main__":
    main()
