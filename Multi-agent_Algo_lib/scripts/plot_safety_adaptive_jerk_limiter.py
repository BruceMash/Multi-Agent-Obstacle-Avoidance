#!/usr/bin/env python3
"""Generate the preregistered raw 0.1 s Development trajectory comparison PDF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "artifacts/safety_adaptive_jerk_limiter/20260825_115704"
SOURCE_STUDY_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
SOURCE_MANIFEST = SOURCE_STUDY_ROOT / "07_development/GAT_RS_DEV_SCENE_MANIFEST.json"
SOURCE_ORIGINAL = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552/04_development/records/original/episode_records"
DEV_ROOT = ARTIFACT_ROOT / "dev_records"
OUTPUT = ARTIFACT_ROOT / "JERK_LIMITER_DEV_RAW_TRAJECTORIES.pdf"
ARMS = ("original", "mild", "medium", "strong")
ARM_LABELS = {"original": "Original", "mild": "Mild", "medium": "Medium", "strong": "Strong"}
LINE_STYLES = {"original": "-", "mild": "--", "medium": "-.", "strong": ":"}
UAV_COLORS = ("#0072B2", "#D55E00", "#009E73")
FIXED_SCENES = tuple(f"GATRS_DEV_{stage}_000" for stage in range(1, 5))
DT = 0.1
ACCELERATION_TITLE = "Executed acceleration norm"
ACCELERATION_YLABEL = r"$\|a_{exec}\|$ (m/s$^2$)"
JERK_TITLE = "Raw 0.1 s vector jerk"
JERK_YLABEL = r"$\|j\|$ (m/s$^3$)"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def record_root(arm: str) -> Path:
    return SOURCE_ORIGINAL if arm == "original" else DEV_ROOT / arm / "episode_records"


def load_record(arm: str, sid: str) -> tuple[dict[str, Any], Any]:
    root = record_root(arm)
    return load_json(root / f"{sid}.json"), np.load(root / f"{sid}_trajectory.npz")


def finite_rows(array: np.ndarray, agent_id: int) -> np.ndarray:
    values = np.asarray(array[:, agent_id], dtype=float)
    return values[np.all(np.isfinite(values), axis=1)]


def add_box_3d(ax: Any, obstacle: dict[str, Any]) -> None:
    center = np.asarray(obstacle["center"], dtype=float)
    half = np.asarray(obstacle["half_extents"], dtype=float)
    lo, hi = center - half, center + half
    vertices = np.asarray([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ])
    faces = [[vertices[i] for i in face] for face in ((0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(0,3,7,4))]
    ax.add_collection3d(Poly3DCollection(faces, facecolors="0.78", edgecolors="0.55", linewidths=0.25, alpha=0.22))


def add_cylinder_3d(ax: Any, obstacle: dict[str, Any]) -> None:
    center = np.asarray(obstacle["center"], dtype=float)
    radius = float(obstacle["radius"])
    half = float(obstacle["half_height"])
    theta = np.linspace(0.0, 2.0 * np.pi, 20)
    z = np.asarray([center[2] - half, center[2] + half])
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = center[0] + radius * np.cos(theta_grid)
    y = center[1] + radius * np.sin(theta_grid)
    ax.plot_surface(x, y, z_grid, color="0.72", edgecolor="none", alpha=0.20, shade=False)


def style_axis(ax: Any, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linewidth=0.45)


def generate() -> None:
    decision = load_json(ARTIFACT_ROOT / "development_decision.json")
    if not decision.get("gates"):
        raise RuntimeError("Development analysis must be complete before plotting")
    manifest = load_json(SOURCE_MANIFEST)
    entries = {str(row["scenario_id"]): row for row in manifest["entries"]}
    with PdfPages(OUTPUT) as pdf:
        for stage_index, sid in enumerate(FIXED_SCENES, start=1):
            entry = entries[sid]
            scene = load_json(SOURCE_STUDY_ROOT / str(entry["scenario_file"]))
            records = {arm: load_record(arm, sid) for arm in ARMS}
            fig = plt.figure(figsize=(15.8, 9.2), constrained_layout=True)
            grid = fig.add_gridspec(2, 3)
            ax3d = fig.add_subplot(grid[0, 0], projection="3d")
            axxy = fig.add_subplot(grid[0, 1])
            axz = fig.add_subplot(grid[0, 2])
            axa = fig.add_subplot(grid[1, 0])
            axj = fig.add_subplot(grid[1, 1])
            axe = fig.add_subplot(grid[1, 2])
            fig.suptitle(f"Stage {stage_index}", fontsize=14, fontweight="normal")

            for obstacle in scene["static_obstacles"]:
                if obstacle["type"] == "box":
                    add_box_3d(ax3d, obstacle)
                    center = np.asarray(obstacle["center"], dtype=float)
                    half = np.asarray(obstacle["half_extents"], dtype=float)
                    axxy.add_patch(Rectangle(center[:2] - half[:2], 2 * half[0], 2 * half[1], facecolor="0.78", edgecolor="0.58", linewidth=0.35, alpha=0.25))
                elif obstacle["type"] == "cylinder":
                    add_cylinder_3d(ax3d, obstacle)
                    axxy.add_patch(Circle(np.asarray(obstacle["center"], dtype=float)[:2], float(obstacle["radius"]), facecolor="0.78", edgecolor="0.58", linewidth=0.35, alpha=0.25))
            for track in scene.get("dynamic_obstacle_trajectories", []):
                track_array = np.asarray(track, dtype=float)
                ax3d.plot(track_array[:, 0], track_array[:, 1], track_array[:, 2], color="0.45", linewidth=0.55, alpha=0.5)
                axxy.plot(track_array[:, 0], track_array[:, 1], color="0.45", linewidth=0.55, alpha=0.5)

            event_row = 0
            event_labels = []
            for arm in ARMS:
                record, arrays = records[arm]
                positions = np.asarray(arrays["positions"], dtype=float)
                acceleration = np.asarray(arrays["applied_accelerations_full"], dtype=float).copy()
                if not np.all(np.isfinite(acceleration[0])):
                    acceleration[0] = 0.0
                for agent_id in range(positions.shape[1]):
                    path = finite_rows(positions, agent_id)
                    color = UAV_COLORS[agent_id]
                    line = LINE_STYLES[arm]
                    ax3d.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linestyle=line, linewidth=1.15, alpha=0.88)
                    axxy.plot(path[:, 0], path[:, 1], color=color, linestyle=line, linewidth=1.15, alpha=0.88)
                    time_state = np.arange(len(path)) * DT
                    axz.plot(time_state, path[:, 2], color=color, linestyle=line, linewidth=1.0, alpha=0.88)
                    valid_acc = acceleration[:, agent_id]
                    finite_acc = np.all(np.isfinite(valid_acc), axis=1)
                    acc_norm = np.linalg.norm(valid_acc[finite_acc], axis=1)
                    acc_time = np.flatnonzero(finite_acc) * DT
                    axa.plot(acc_time, acc_norm, color=color, linestyle=line, linewidth=0.95, alpha=0.86)
                    jerk = np.linalg.norm(np.diff(valid_acc, axis=0) / DT, axis=1)
                    jerk_time = np.arange(1, len(valid_acc)) * DT
                    finite_jerk = np.isfinite(jerk)
                    axj.plot(jerk_time[finite_jerk], jerk[finite_jerk], color=color, linestyle=line, linewidth=0.85, alpha=0.80)
                    y = event_row
                    event_labels.append(f"{ARM_LABELS[arm]} U{agent_id + 1}")
                    for event in record.get("events", []):
                        if int(event["agent_id"]) != agent_id or not bool(event.get("goal_changed", True)):
                            continue
                        event_type = str(event.get("event"))
                        marker = "x" if "EMERGENCY" in event_type else ("|" if "NORMAL" in event_type else ".")
                        axe.scatter(float(event["step"]) * DT, y, s=18 if marker != "." else 9, marker=marker, color=color, alpha=0.72, linewidths=0.75)
                    if arm != "original":
                        root = record_root(arm)
                        trace = np.load(root / str(record["limiter_trace_file"]))
                        mask = (np.asarray(trace["agent_id"], dtype=int) == agent_id) & np.asarray(trace["hard_bypass"], dtype=bool)
                        if np.any(mask):
                            axe.scatter(np.asarray(trace["step"], dtype=float)[mask] * DT, np.full(int(np.sum(mask)), y), s=24, marker="^", facecolors="none", edgecolors="#CC3311", linewidths=0.8)
                    event_row += 1

            for agent_id, color in enumerate(UAV_COLORS):
                start = np.asarray(scene["starts"][agent_id], dtype=float)
                goal = np.asarray(scene["goals"][agent_id], dtype=float)
                ax3d.scatter(*start, color=color, marker="o", s=25)
                ax3d.scatter(*goal, color=color, marker="*", s=45)
                axxy.scatter(start[0], start[1], color=color, marker="o", s=20)
                axxy.scatter(goal[0], goal[1], color=color, marker="*", s=38)

            ax3d.set(xlim=(0, 100), ylim=(0, 100), zlim=(0.8, 3.2), xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
            ax3d.view_init(elev=25, azim=-58)
            ax3d.set_title("3-D trajectories")
            axxy.set(xlim=(0, 100), ylim=(0, 100), aspect="equal")
            style_axis(axxy, "x (m)", "y (m)")
            axxy.set_title("XY projection")
            style_axis(axz, "Time (s)", "z (m)")
            axz.set_title("Altitude")
            style_axis(axa, "Time (s)", ACCELERATION_YLABEL)
            axa.set_title(ACCELERATION_TITLE)
            style_axis(axj, "Time (s)", JERK_YLABEL)
            axj.set_title(JERK_TITLE)
            style_axis(axe, "Time (s)", "Method / UAV")
            axe.set_yticks(np.arange(len(event_labels)), event_labels, fontsize=7)
            axe.set_ylim(-0.7, len(event_labels) - 0.3)
            axe.set_title("Reference/ERR updates and hard bypass")
            axe.invert_yaxis()

            method_handles = [Line2D([0], [0], color="0.18", linestyle=LINE_STYLES[arm], linewidth=1.4, label=ARM_LABELS[arm]) for arm in ARMS]
            uav_handles = [Line2D([0], [0], color=color, linestyle="-", linewidth=1.4, label=f"UAV {idx + 1}") for idx, color in enumerate(UAV_COLORS)]
            marker_handles = [
                Line2D([0], [0], color="0.25", marker=".", linestyle="None", label="reference update"),
                Line2D([0], [0], color="0.25", marker="|", linestyle="None", label="normal ERR"),
                Line2D([0], [0], color="0.25", marker="x", linestyle="None", label="emergency ERR"),
                Line2D([0], [0], color="#CC3311", marker="^", markerfacecolor="none", linestyle="None", label="hard bypass"),
            ]
            fig.legend(handles=method_handles + uav_handles + marker_handles, loc="outside lower center", ncol=6, frameon=False, fontsize=8)
            fig.text(0.995, 0.006, f"Fixed scene: {sid}; raw dt = 0.1 s; no smoothing or interpolation", ha="right", va="bottom", fontsize=7, color="0.35")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
    print(json.dumps({"status": "PASS", "output": str(OUTPUT), "pages": len(FIXED_SCENES)}, indent=2))


if __name__ == "__main__":
    generate()
