"""Plot raw Proposed trajectories only on the four frozen shared-scene anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from planning.generate_ppo_direct_paper_figures import (
    AGENT_COLORS,
    load_positions,
    plot_static,
    save_figure,
    select_shared_scenes,
    setup_style,
    write_csv,
)


PROPOSED = "M9_Proposed_RERR_GAT_SAC_DMP"


def generate(root: Path) -> dict[str, str]:
    setup_style()
    output = root / "13_paper_ready/proposed_only"
    anchors = select_shared_scenes(root)
    source_rows: list[dict[str, object]] = []

    figure = plt.figure(figsize=(9.0, 7.4))
    for panel, anchor in enumerate(anchors, start=1):
        axis = figure.add_subplot(2, 2, panel, projection="3d")
        scenario_id = anchor["scenario_id"]
        record = json.loads(
            (root / f"12_trajectories/formal_records/{scenario_id}.json").read_text(
                encoding="utf-8"
            )
        )
        for obstacle in record["static_obstacles"]:
            plot_static(axis, obstacle)
        for obstacle in record["dynamic_obstacles"]:
            center = np.asarray(obstacle["center"], dtype=float)
            velocity = np.asarray(obstacle["velocity"], dtype=float)
            end = np.clip(center + 5.0 * velocity, [0, 0, 0], [100, 100, 4])
            axis.scatter(*center, color="0.25", marker="x", s=13, linewidth=0.7)
            axis.plot(
                [center[0], end[0]],
                [center[1], end[1]],
                [center[2], end[2]],
                color="0.35",
                linestyle=(0, (1, 2)),
                linewidth=0.55,
                alpha=0.7,
            )

        positions = load_positions(root, PROPOSED, scenario_id)
        for agent in range(3):
            path = positions[:, agent]
            axis.plot(
                path[:, 0],
                path[:, 1],
                path[:, 2],
                color=AGENT_COLORS[agent],
                linewidth=1.25,
                alpha=0.95,
            )
            for step, point in enumerate(path):
                source_rows.append(
                    {
                        "stage": anchor["stage"],
                        "scenario_id": scenario_id,
                        "agent_id": agent + 1,
                        "step": step,
                        "time_s": 0.1 * step,
                        "x_m": point[0],
                        "y_m": point[1],
                        "z_m": point[2],
                    }
                )

        starts = np.asarray(record["starts"], dtype=float)
        goals = np.asarray(record["goals"], dtype=float)
        for agent in range(3):
            axis.scatter(
                *starts[agent],
                color=AGENT_COLORS[agent],
                marker="^",
                s=22,
                edgecolor="black",
                linewidth=0.3,
            )
            axis.scatter(
                *goals[agent],
                color=AGENT_COLORS[agent],
                marker="*",
                s=38,
                edgecolor="black",
                linewidth=0.3,
            )

        axis.set_xlim(0, 100)
        axis.set_ylim(0, 100)
        axis.set_zlim(0, 4)
        axis.set_xlabel("x (m)", labelpad=1)
        axis.set_ylabel("y (m)", labelpad=1)
        axis.set_zlabel("z (m)", labelpad=0)
        axis.set_title(anchor["stage"], pad=2)
        axis.view_init(elev=28, azim=-57)
        axis.grid(True, linewidth=0.25, color="0.9")

    handles = [
        *[
            Line2D([0], [0], color=AGENT_COLORS[index], linewidth=1.5, label=f"UAV {index + 1}")
            for index in range(3)
        ],
        Line2D([0], [0], color="0.2", marker="^", linestyle="None", label="start"),
        Line2D([0], [0], color="0.2", marker="*", linestyle="None", label="goal"),
        Line2D([0], [0], color="0.25", marker="x", linestyle="None", label="moving obstacle"),
    ]
    figure.legend(
        handles=handles,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
    )
    figure.subplots_adjust(top=0.91, wspace=0.02, hspace=0.09)

    pdf, png = save_figure(figure, output, "proposed_only_four_stage_3d_trajectories")
    source = output / "source_data/proposed_only_four_stage_3d_trajectories.csv"
    write_csv(source, source_rows)
    selection = output / "source_data/proposed_only_scene_selection.csv"
    write_csv(selection, anchors)
    caption = output / "captions/proposed_only_four_stage_3d_trajectories_caption.txt"
    caption.parent.mkdir(parents=True, exist_ok=True)
    caption.write_text(
        "Proposed-only raw 3-D trajectories on the four deterministic shared-scene anchors. "
        "Every stored 0.1 s position is plotted without smoothing or display downsampling. "
        "UAV identity is encoded by color; triangles and stars mark starts and goals. "
        "Static structures are gray and moving-obstacle initial states are crosses. "
        "All axes start at zero.\n",
        encoding="utf-8",
    )
    result = {
        "pdf": pdf.relative_to(root).as_posix(),
        "png_600dpi": png.relative_to(root).as_posix(),
        "source_data": source.relative_to(root).as_posix(),
        "selection": selection.relative_to(root).as_posix(),
        "caption": caption.relative_to(root).as_posix(),
    }
    manifest = output / "PROPOSED_ONLY_FIGURE_MANIFEST.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
