#!/usr/bin/env python3
"""Create the mandatory, data-honest Track-B NOT_RUN audit PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


DEFAULT_OUTPUT = Path(
    "artifacts/parallel_zigzag_resolution/20260826_091334/"
    "track_B_learned_turn/TURN_DEV_RAW_TRAJECTORIES.pdf"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.69, 8.27), facecolor="#F7FAFC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(
        0.5, 0.91, "Track B - Development Raw Trajectory Audit",
        ha="center", va="center", fontsize=23, weight="bold", color="#17324D",
    )
    banner = FancyBboxPatch(
        (0.11, 0.70), 0.78, 0.12,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.5, edgecolor="#7B2020", facecolor="#B83A3A",
        transform=ax.transAxes,
    )
    ax.add_patch(banner)
    ax.text(0.5, 0.76, "NOT RUN", ha="center", va="center", fontsize=30, weight="bold", color="white")
    ax.text(
        0.5, 0.63,
        "Zero Development trajectories generated  |  Zero Development scenes instantiated\n"
        "No training checkpoint produced or evaluated",
        ha="center", va="center", fontsize=13, linespacing=1.5, color="#273746",
    )
    cells = [
        ["Pre-training gate", "Result"],
        ["522-to-529 zero-column expansion", "PASS"],
        ["Smooth-teacher SAC-action recoverability", "FAIL"],
        ["Training authorization", "NO"],
        ["Development / Holdout / Formal", "NOT RUN / NOT RUN / NOT RUN"],
    ]
    table = ax.table(
        cellText=cells, cellLoc="center", bbox=[0.18, 0.29, 0.64, 0.26],
        colWidths=[0.50, 0.50],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#A6B8C5")
        cell.set_linewidth(0.8)
        cell.set_facecolor("#DDEAF3" if row == 0 else "#FFFFFF")
        if row == 0:
            cell.get_text().set_weight("bold")
            cell.get_text().set_color("#17324D")
    ax.text(
        0.5, 0.17,
        "Integrity note: Strong + Turn-Sign Persistence operates after the DMP in 3-D acceleration space.\n"
        "The auxiliary teacher loss requires a 6-D SAC action, and no authoritative unique inverse exists.\n"
        "This page is a gate record, not a trajectory figure.",
        ha="center", va="center", fontsize=10.5, linespacing=1.45, color="#52687A",
    )
    ax.text(
        0.5, 0.055,
        "Raw trajectory count: 0   |   Post-processing applied: NO",
        ha="center", fontsize=9, color="#6B7F8E",
    )
    fig.savefig(output, format="pdf", bbox_inches=None)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
