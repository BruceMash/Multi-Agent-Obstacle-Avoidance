from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.input_dir
    action_rows = read_rows(output_dir / "action_traces.csv")
    active_rows = [row for row in action_rows if row.get("agent_active", "True") == "True"]
    if active_rows:
        action_rows = active_rows
    q_rows = read_rows(output_dir / "q_action_scan.csv")
    outcomes = read_rows(output_dir / "scaled_action_outcomes.csv")
    reward_rows = read_rows(output_dir / "reward_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].hist([number(row, "raw_forcing_norm") for row in action_rows], bins=40, alpha=0.65, label="raw")
    axes[0, 0].hist([number(row, "effective_forcing_norm") for row in action_rows], bins=40, alpha=0.65, label="effective")
    axes[0, 0].set_title("Forcing norm distribution")
    axes[0, 0].legend()
    axes[0, 1].hist([number(row, "effective_residual_ratio") for row in action_rows], bins=40)
    axes[0, 1].set_title("Effective residual/nominal ratio")
    scenarios = list(dict.fromkeys(row["scenario"] for row in action_rows))
    axes[1, 0].bar(
        scenarios,
        [np.mean([number(row, "forcing_saturation_rate") for row in action_rows if row["scenario"] == name]) for name in scenarios],
    )
    axes[1, 0].set_title("Forcing saturation rate")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 1].bar(
        scenarios,
        [np.mean([row["acceleration_clipped"] == "True" for row in action_rows if row["scenario"] == name]) for name in scenarios],
    )
    axes[1, 1].set_title("Acceleration clipping rate")
    axes[1, 1].tick_params(axis="x", rotation=20)
    fig.savefig(output_dir / "action_diagnostics.png", dpi=160)
    plt.close(fig)

    if q_rows:
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        lambdas = sorted(set(number(row, "lambda") for row in q_rows))
        axis.plot(
            lambdas,
            [np.mean([number(row, "q_min") for row in q_rows if number(row, "lambda") == value]) for value in lambdas],
            marker="o",
        )
        axis.set_title("Qmin versus action scale")
        axis.set_xlabel("lambda")
        axis.set_ylabel("Qmin")
        axis.grid(True, alpha=0.3)
        fig.savefig(output_dir / "q_action_scale.png", dpi=160)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for scenario in sorted(set(row["scenario"] for row in outcomes)):
        rows = sorted(
            [row for row in outcomes if row["scenario"] == scenario],
            key=lambda row: number(row, "lambda"),
        )
        axis.plot(
            [number(row, "lambda") for row in rows],
            [number(row, "total_reward_mean") for row in rows],
            marker="o",
            label=scenario,
        )
    axis.set_title("Frozen-action replay return versus action scale")
    axis.set_xlabel("lambda")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.savefig(output_dir / "scaled_action_outcomes.png", dpi=160)
    plt.close(fig)

    components = sorted(set(row["component"] for row in reward_rows if row["component"] != "total_reward"))
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.barh(
        components,
        [np.mean([number(row, "absolute_mean") for row in reward_rows if row["component"] == component]) for component in components],
    )
    axis.set_title("Mean absolute reward component scale")
    fig.savefig(output_dir / "reward_components.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
