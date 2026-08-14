"""Debug-only visualization for a heterogeneous candidate graph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def plot_heterogeneous_candidate_graph(
    data: Any,
    *,
    ego_position: np.ndarray,
    task_goal: np.ndarray,
    neighbor_positions: np.ndarray,
    neighbor_velocities: np.ndarray,
    output_path: str | Path,
    title: str = "Heterogeneous Candidate Graph Debug",
) -> Path:
    """Plot only current/preview graph inputs; no global map is accessed."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ego = np.asarray(ego_position, dtype=float)
    goal = np.asarray(task_goal, dtype=float)
    neighbor_positions = np.asarray(neighbor_positions, dtype=float).reshape(-1, 3)
    neighbor_velocities = np.asarray(neighbor_velocities, dtype=float).reshape(-1, 3)
    dt = float(data.graph_metadata["dt"])
    horizon = int(data.graph_metadata["H"])

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(*ego, c="#0B6E4F", s=90, marker="o", label="ego UAV")
    axis.scatter(*goal, c="#D62828", s=100, marker="*", label="task goal")

    world_candidates = data["proposal"].world_position.detach().cpu().numpy()
    candidate_ids = data["proposal"].candidate_id.detach().cpu().numpy()
    plotted_points: list[np.ndarray] = [ego, goal]
    for node, (candidate, candidate_id) in enumerate(zip(world_candidates, candidate_ids)):
        axis.scatter(
            *candidate,
            c="#1D4ED8",
            s=55,
            marker="^",
            label="candidate" if node == 0 else None,
        )
        axis.text(*candidate, f"c{int(candidate_id)}", fontsize=8)
        trajectory = np.asarray(data.candidate_preview_positions[node], dtype=float)
        if trajectory.size:
            full = np.vstack([ego, trajectory])
            axis.plot(
                full[:, 0],
                full[:, 1],
                full[:, 2],
                c="#60A5FA",
                alpha=0.65,
                label="FP-SHEP trajectory" if node == 0 else None,
            )
            plotted_points.extend(full)
        plotted_points.append(candidate)

    for node, (position, velocity) in enumerate(zip(neighbor_positions, neighbor_velocities)):
        agent_id = int(data["align"].agent_id[node])
        steps = np.arange(0, horizon + 1, dtype=float)[:, None]
        trajectory = position[None, :] + steps * dt * velocity[None, :]
        axis.scatter(
            *position,
            c="#F59E0B",
            s=65,
            marker="s",
            label="observable neighbor" if node == 0 else None,
        )
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            "--",
            c="#F59E0B",
            label="neighbor constant-velocity trajectory" if node == 0 else None,
        )
        axis.text(*position, f"n{agent_id}", fontsize=8)
        plotted_points.extend(trajectory)

    conflict = data["align", "spatiotemporal", "proposal"]
    for edge in range(conflict.edge_index.shape[1]):
        candidate_point = conflict.candidate_minimum_position[edge].detach().cpu().numpy()
        neighbor_point = conflict.neighbor_minimum_position[edge].detach().cpu().numpy()
        t_min, d_min, risk = conflict.edge_attr[edge].detach().cpu().numpy()
        axis.plot(
            [candidate_point[0], neighbor_point[0]],
            [candidate_point[1], neighbor_point[1]],
            [candidate_point[2], neighbor_point[2]],
            c="#DC2626",
            linewidth=2.4,
            label="constructed spatiotemporal edge" if edge == 0 else None,
        )
        midpoint = 0.5 * (candidate_point + neighbor_point)
        axis.text(*midpoint, f"d={d_min:.2f}\nt={t_min:.2f}\nT={risk:.2f}", fontsize=7)
        plotted_points.extend([candidate_point, neighbor_point])

    if plotted_points:
        stacked = np.asarray(plotted_points, dtype=float).reshape(-1, 3)
        lower = np.min(stacked, axis=0)
        upper = np.max(stacked, axis=0)
        center = 0.5 * (lower + upper)
        radius = max(float(np.max(upper - lower)) * 0.6, 1.0)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_xlabel("world x / m")
    axis.set_ylabel("world y / m")
    axis.set_zlabel("world z / m")
    axis.set_title(title)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(
        "This module visualizes in-memory HeteroData; use "
        "validate_heterogeneous_candidate_graph.py to generate a complete artifact."
    )
