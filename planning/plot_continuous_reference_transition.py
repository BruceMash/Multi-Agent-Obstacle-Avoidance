"""Paper-ready CRT diagnostics from frozen Development/Holdout records.

No trajectory, reference, acceleration, or jerk signal is smoothed.  Scene
selection is deterministic and based on the stage-median paired smoothness
change among both-success Holdout episodes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552"
OUT = ROOT / "11_paper_ready"
PDF = OUT / "pdf"
PNG = OUT / "png_600dpi"
SOURCE = OUT / "source_data"
DT = 0.1

UAV_COLORS = ("#0072B2", "#D55E00", "#009E73")
STAGE_COLORS = {
    "stage_1": "#0072B2",
    "stage_2": "#009E73",
    "stage_3": "#E69F00",
    "stage_4": "#CC79A7",
}
STAGE_LABELS = {
    "stage_1": "Stage I",
    "stage_2": "Stage II",
    "stage_3": "Stage III",
    "stage_4": "Stage IV",
}
ORIGINAL_STYLE = (0, (4.5, 2.0))
CRT_STYLE = "-"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def save(fig: Any, stem: str) -> None:
    PDF.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(PNG / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def record_dir(block: str, variant: str) -> Path:
    phase = "04_development" if block == "development" else "07_holdout"
    return ROOT / phase / "records" / variant / "episode_records"


def load_records(block: str, variant: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(record_dir(block, variant).glob("*.json")):
        if path.stem.endswith("_SOFTWARE_ERROR"):
            continue
        row = load_json(path)
        row["_path"] = path
        result[row["summary"]["scenario_id"]] = row
    return result


def load_trajectory(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = Path(record["_path"]).with_name(str(record["trajectory_file"]))
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def source_root(manifest: Mapping[str, Any]) -> Path:
    source_manifest = REPO_ROOT / str(manifest["source_manifest"])
    return source_manifest.parent.parent


def load_scene(manifest: Mapping[str, Any], scenario_id: str) -> dict[str, Any]:
    entry = next(row for row in manifest["entries"] if row["scenario_id"] == scenario_id)
    return load_json(source_root(manifest) / str(entry["scenario_file"]))


def paired_delta(original: Mapping[str, Any], crt: Mapping[str, Any], key: str) -> float:
    return float(crt["episode"][key]) - float(original["episode"][key])


def select_representatives(
    original: Mapping[str, Mapping[str, Any]],
    crt: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in STAGE_LABELS:
        candidates: list[dict[str, Any]] = []
        for scenario_id in sorted(set(original) & set(crt)):
            left, right = original[scenario_id], crt[scenario_id]
            if left["summary"]["stage"] != stage:
                continue
            if not left["episode"]["team_success"] or not right["episode"]["team_success"]:
                continue
            candidates.append(
                {
                    "stage": stage,
                    "scenario_id": scenario_id,
                    "family": left["summary"]["family"],
                    "smoothness_delta": paired_delta(left, right, "trajectory_smoothness"),
                    "path_delta_m": paired_delta(left, right, "team_path_length_m"),
                    "original_smoothness": float(left["episode"]["trajectory_smoothness"]),
                    "crt_smoothness": float(right["episode"]["trajectory_smoothness"]),
                    "original_path_m": float(left["episode"]["team_path_length_m"]),
                    "crt_path_m": float(right["episode"]["team_path_length_m"]),
                }
            )
        if not candidates:
            raise RuntimeError(f"no both-success Holdout scene in {stage}")
        median = float(np.median([row["smoothness_delta"] for row in candidates]))
        selected = min(
            candidates,
            key=lambda row: (abs(row["smoothness_delta"] - median), row["scenario_id"]),
        )
        selected["both_success_count"] = len(candidates)
        selected["stage_median_smoothness_delta"] = median
        selected["selection_rule"] = (
            "closest paired smoothness delta to the stage median among both-success Holdout episodes"
        )
        rows.append(selected)
    return rows


def box_faces(center: Iterable[float], half: Iterable[float]) -> list[list[tuple[float, float, float]]]:
    c = np.asarray(center, dtype=float)
    h = np.asarray(half, dtype=float)
    points = [tuple(c + h * np.asarray([sx, sy, sz])) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return [[points[index] for index in face] for face in ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))]


def draw_sphere_3d(axis: Any, center: Iterable[float], radius: float, alpha: float = 0.14) -> None:
    center = np.asarray(center, dtype=float)
    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 13)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    axis.plot_surface(x, y, z, color="#777777", alpha=alpha, linewidth=0, shade=False)


def draw_obstacles_3d(axis: Any, scene: Mapping[str, Any], trajectory_steps: int) -> None:
    for obstacle in scene["static_obstacles"]:
        kind = obstacle["type"]
        margin = float(obstacle.get("safety_margin", 0.0))
        if kind == "box":
            half = np.asarray(obstacle["half_extents"], dtype=float) + margin
            axis.add_collection3d(
                Poly3DCollection(
                    box_faces(obstacle["center"], half),
                    facecolor="#7F8C8D",
                    edgecolor="#596465",
                    linewidth=0.28,
                    alpha=0.16,
                )
            )
        elif kind == "sphere":
            draw_sphere_3d(axis, obstacle["center"], float(obstacle["radius"]) + margin)
        elif kind == "cylinder":
            center = np.asarray(obstacle["center"], dtype=float)
            radius = float(obstacle["radius"]) + margin
            half_height = float(obstacle["half_height"]) + margin
            theta, z = np.meshgrid(
                np.linspace(0.0, 2.0 * np.pi, 28),
                np.asarray([center[2] - half_height, center[2] + half_height]),
            )
            axis.plot_surface(
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
                z,
                color="#7F8C8D",
                alpha=0.16,
                linewidth=0,
            )
    for obstacle, track in zip(scene["dynamic_obstacles"], scene["dynamic_obstacle_trajectories"]):
        points = np.asarray(track[:trajectory_steps], dtype=float)
        if points.size:
            axis.plot(points[:, 0], points[:, 1], points[:, 2], color="#555555", linestyle=(0, (2, 2)), linewidth=0.8, alpha=0.6)
            draw_sphere_3d(axis, points[0], float(obstacle["radius"]), alpha=0.18)


def draw_obstacles_xy(axis: Any, scene: Mapping[str, Any], trajectory_steps: int) -> None:
    for obstacle in scene["static_obstacles"]:
        kind = obstacle["type"]
        margin = float(obstacle.get("safety_margin", 0.0))
        center = np.asarray(obstacle["center"], dtype=float)
        if kind == "box":
            half = np.asarray(obstacle["half_extents"], dtype=float) + margin
            axis.add_patch(Rectangle((center[0] - half[0], center[1] - half[1]), 2 * half[0], 2 * half[1], facecolor="#7F8C8D", edgecolor="#596465", linewidth=0.35, alpha=0.18))
        else:
            radius = float(obstacle["radius"]) + margin
            axis.add_patch(Circle((center[0], center[1]), radius, facecolor="#7F8C8D", edgecolor="#596465", linewidth=0.35, alpha=0.18))
    for track in scene["dynamic_obstacle_trajectories"]:
        points = np.asarray(track[:trajectory_steps], dtype=float)
        if points.size:
            axis.plot(points[:, 0], points[:, 1], color="#555555", linestyle=(0, (2, 2)), linewidth=0.8, alpha=0.6)


def scene_bounds(scene: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(scene["workspace_bounds"], dtype=float)
    lower = np.minimum(bounds[0], 0.0)
    return lower, bounds[1]


def figure_trajectories(
    representatives: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    original: Mapping[str, Mapping[str, Any]],
    crt: Mapping[str, Mapping[str, Any]],
    *,
    mode: str,
) -> None:
    projection = "3d" if mode == "3d" else None
    fig = plt.figure(figsize=(12.8, 8.2) if mode == "3d" else (10.0, 8.6))
    for panel, row in enumerate(representatives, start=1):
        axis = fig.add_subplot(2, 2, panel, projection=projection)
        scene = load_scene(manifest, str(row["scenario_id"]))
        left = load_trajectory(original[str(row["scenario_id"])])
        right = load_trajectory(crt[str(row["scenario_id"])])
        steps = max(left["positions"].shape[0], right["positions"].shape[0])
        if mode == "3d":
            draw_obstacles_3d(axis, scene, steps)
        else:
            draw_obstacles_xy(axis, scene, steps)
        for agent, color in enumerate(UAV_COLORS):
            for trajectory, style, width in ((left, ORIGINAL_STYLE, 1.25), (right, CRT_STYLE, 2.0)):
                points = trajectory["positions"][:, agent]
                if mode == "3d":
                    axis.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linestyle=style, linewidth=width, alpha=0.92)
                else:
                    axis.plot(points[:, 0], points[:, 1], color=color, linestyle=style, linewidth=width, alpha=0.92)
            starts = np.asarray(scene["starts"], dtype=float)
            goals = np.asarray(scene["goals"], dtype=float)
            if mode == "3d":
                axis.scatter(*starts[agent], marker="^", s=34, facecolor="white", edgecolor=color, linewidth=1.1, depthshade=False)
                axis.scatter(*goals[agent], marker="*", s=65, facecolor=color, edgecolor="white", linewidth=0.5, depthshade=False)
            else:
                axis.scatter(starts[agent, 0], starts[agent, 1], marker="^", s=34, facecolor="white", edgecolor=color, linewidth=1.1)
                axis.scatter(goals[agent, 0], goals[agent, 1], marker="*", s=65, facecolor=color, edgecolor="white", linewidth=0.5)
        lower, upper = scene_bounds(scene)
        axis.set_xlim(float(lower[0]), float(upper[0]))
        axis.set_ylim(float(lower[1]), float(upper[1]))
        if mode == "3d":
            axis.set_zlim(0.0, float(upper[2]))
            axis.set_zlabel("z (m)", labelpad=0)
            axis.view_init(elev=27, azim=-61)
            axis.set_box_aspect((1.0, 1.0, 0.38))
        else:
            axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_title(STAGE_LABELS[str(row["stage"])], fontweight="semibold")
        axis.grid(True, color="#D7DCE1", linewidth=0.35)
    uav = [Line2D([0], [0], color=color, lw=2.4, label=f"UAV {index + 1}") for index, color in enumerate(UAV_COLORS)]
    methods = [
        Line2D([0], [0], color="#222222", lw=1.4, linestyle=ORIGINAL_STYLE, label="Original abrupt handoff"),
        Line2D([0], [0], color="#222222", lw=2.1, linestyle=CRT_STYLE, label="Selected CRT"),
    ]
    fig.legend(handles=uav + methods, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.97, bottom=0.09, wspace=0.05, hspace=0.14)
    save(fig, "figure_A_matched_holdout_trajectories_3d" if mode == "3d" else "figure_B_matched_holdout_trajectories_xy")


def focus_scene(representatives: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    median = float(np.median([float(row["smoothness_delta"]) for row in representatives]))
    return min(representatives, key=lambda row: (abs(float(row["smoothness_delta"]) - median), str(row["scenario_id"])))


def jerk(trajectory: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.diff(np.asarray(trajectory["applied_accelerations"], dtype=float), axis=0) / DT


def figure_focus_timeseries(
    focus: Mapping[str, Any],
    original_record: Mapping[str, Any],
    crt_record: Mapping[str, Any],
) -> None:
    left, right = load_trajectory(original_record), load_trajectory(crt_record)
    for kind, stem, ylabel in (("z", "figure_C_altitude_profiles", "z (m)"), ("jerk", "figure_D_raw_jerk_profiles", r"$\|j\|$ (m s$^{-3}$)")):
        fig, axes = plt.subplots(3, 1, figsize=(9.0, 6.6), sharex=False)
        for agent, axis in enumerate(axes):
            for trajectory, style, width, label in ((left, ORIGINAL_STYLE, 1.2, "Original"), (right, CRT_STYLE, 1.7, "CRT")):
                if kind == "z":
                    values = trajectory["positions"][:, agent, 2]
                    time = np.arange(len(values)) * DT
                else:
                    values = np.linalg.norm(jerk(trajectory)[:, agent], axis=1)
                    time = np.arange(2, 2 + len(values)) * DT
                axis.plot(time, values, color=UAV_COLORS[agent], linestyle=style, linewidth=width, label=label)
            axis.set_ylabel(f"UAV {agent + 1}\n{ylabel}")
            axis.grid(True, color="#D7DCE1", linewidth=0.4)
            axis.spines[["top", "right"]].set_visible(False)
        axes[-1].set_xlabel("Time (s)")
        axes[0].legend(handles=[Line2D([0], [0], color="#222222", linestyle=ORIGINAL_STYLE, lw=1.3, label="Original"), Line2D([0], [0], color="#222222", linestyle=CRT_STYLE, lw=1.8, label="Selected CRT")], frameon=False, ncol=2, loc="upper right")
        fig.suptitle(STAGE_LABELS[str(focus["stage"])], fontweight="semibold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save(fig, stem)

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 6.6), sharex=True)
    time = np.arange(right["g_cmd"].shape[0]) * DT
    for agent, axis in enumerate(axes):
        axis.plot(time, right["g_cmd"][:, agent, 2], color=UAV_COLORS[agent], linestyle=ORIGINAL_STYLE, linewidth=1.2, label=r"$g_{cmd,z}$")
        axis.plot(time, right["g_exec"][:, agent, 2], color=UAV_COLORS[agent], linestyle=CRT_STYLE, linewidth=1.8, label=r"$g_{exec,z}$")
        axis.set_ylabel(f"UAV {agent + 1}\nz (m)")
        axis.grid(True, color="#D7DCE1", linewidth=0.4)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    fig.suptitle(STAGE_LABELS[str(focus["stage"])], fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "figure_E_command_vs_executed_reference_altitude")

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 6.6), sharex=False)
    for agent, axis in enumerate(axes):
        for trajectory, style, width, label in ((left, ORIGINAL_STYLE, 1.2, "Original"), (right, CRT_STYLE, 1.7, "CRT")):
            direction = trajectory["g_exec"][:, agent] - trajectory["positions"][:, agent]
            norm = np.linalg.norm(direction, axis=1, keepdims=True)
            unit = direction / np.maximum(norm, 1.0e-12)
            angle = np.degrees(np.arccos(np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)))
            axis.plot(np.arange(1, len(unit)) * DT, angle, color=UAV_COLORS[agent], linestyle=style, linewidth=width, label=label)
        axis.set_ylabel(f"UAV {agent + 1}\nangle (deg)")
        axis.grid(True, color="#D7DCE1", linewidth=0.4)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    fig.suptitle(STAGE_LABELS[str(focus["stage"])], fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "figure_F_reference_direction_jump_timeline")

    peak_agent = int(np.argmax(np.max(np.linalg.norm(jerk(left), axis=2), axis=0)))
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.4), sharex=False)
    for axis, trajectory, record, label, style in ((axes[0], left, original_record, "Original", ORIGINAL_STYLE), (axes[1], right, crt_record, "Selected CRT", CRT_STYLE)):
        values = np.linalg.norm(jerk(trajectory)[:, peak_agent], axis=1)
        axis.plot(np.arange(2, 2 + len(values)) * DT, values, color=UAV_COLORS[peak_agent], linestyle=style, linewidth=1.5)
        for event in record["events"]:
            if event.get("event") != "INITIAL_SELECTION" and event.get("goal_changed") and int(event["agent_id"]) == peak_agent:
                axis.axvline(float(event["step"]) * DT, color="#D55E00", linewidth=0.45, alpha=0.28)
        axis.set_ylabel(label + "\n" + r"$\|j\|$ (m s$^{-3}$)")
        axis.grid(True, color="#D7DCE1", linewidth=0.4)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(STAGE_LABELS[str(focus["stage"])], fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "figure_G_raw_jerk_with_reference_switches")


def aligned_samples(records: Mapping[str, Mapping[str, Any]], component: str) -> tuple[np.ndarray, np.ndarray]:
    relative = np.arange(-5, 11, dtype=int)
    samples: list[np.ndarray] = []
    for record in records.values():
        trajectory = load_trajectory(record)
        raw = jerk(trajectory)
        for event in record["events"]:
            if event.get("event") == "INITIAL_SELECTION" or not event.get("goal_changed"):
                continue
            agent = int(event["agent_id"])
            step = int(event["step"])
            row = np.full(relative.shape, np.nan, dtype=float)
            for index, offset in enumerate(relative):
                jerk_index = step + int(offset) - 1
                if 0 <= jerk_index < raw.shape[0]:
                    vector = raw[jerk_index, agent]
                    row[index] = abs(vector[2]) if component == "vertical" else float(np.linalg.norm(vector))
            if np.sum(np.isfinite(row)) >= 8:
                samples.append(row)
    return relative.astype(float) * DT, np.asarray(samples, dtype=float)


def figure_event_alignment(original: Mapping[str, Mapping[str, Any]], crt: Mapping[str, Mapping[str, Any]]) -> None:
    for component, stem, ylabel in (("norm", "figure_H_event_aligned_jerk", r"$\|j\|$ (m s$^{-3}$)"), ("vertical", "figure_I_event_aligned_vertical_jerk", r"$|j_z|$ (m s$^{-3}$)")):
        fig, axis = plt.subplots(figsize=(7.8, 4.3))
        for data, color, label in ((original, "#4C78A8", "Original"), (crt, "#D55E00", "Selected CRT")):
            time, samples = aligned_samples(data, component)
            median = np.nanmedian(samples, axis=0)
            lower = np.nanpercentile(samples, 25, axis=0)
            upper = np.nanpercentile(samples, 75, axis=0)
            axis.plot(time, median, color=color, linewidth=2.0, label=f"{label} (events={len(samples)})")
            axis.fill_between(time, lower, upper, color=color, alpha=0.16, linewidth=0)
        axis.axvline(0.0, color="#222222", linestyle=(0, (2, 2)), linewidth=0.9)
        axis.set_xlabel("Time relative to command switch (s)")
        axis.set_ylabel(ylabel)
        axis.grid(True, color="#D7DCE1", linewidth=0.45)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
        fig.tight_layout()
        save(fig, stem)


def figure_tradeoff(original: Mapping[str, Mapping[str, Any]], crt: Mapping[str, Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for scenario_id in sorted(set(original) & set(crt)):
        left, right = original[scenario_id], crt[scenario_id]
        if not left["episode"]["team_success"] or not right["episode"]["team_success"]:
            continue
        rows.append(
            {
                "scenario_id": scenario_id,
                "stage": left["summary"]["stage"],
                "path_delta_m": paired_delta(left, right, "team_path_length_m"),
                "smoothness_delta": paired_delta(left, right, "trajectory_smoothness"),
            }
        )
    write_csv(SOURCE / "CRT_HOLDOUT_PAIRED_PATH_SMOOTHNESS.csv", rows)
    fig, axis = plt.subplots(figsize=(7.4, 5.4))
    for stage, label in STAGE_LABELS.items():
        data = [row for row in rows if row["stage"] == stage]
        axis.scatter([row["path_delta_m"] for row in data], [row["smoothness_delta"] for row in data], s=24, color=STAGE_COLORS[stage], alpha=0.58, edgecolor="none", label=label)
        axis.scatter(np.mean([row["path_delta_m"] for row in data]), np.mean([row["smoothness_delta"] for row in data]), s=85, color=STAGE_COLORS[stage], marker="D", edgecolor="white", linewidth=0.8)
    axis.axhline(0.0, color="#555555", linewidth=0.7)
    axis.axvline(0.0, color="#555555", linewidth=0.7)
    axis.set_xlabel("CRT − Original team path length (m)")
    axis.set_ylabel("CRT − Original smoothness/jerk cost")
    axis.text(0.02, 0.03, "lower-left = shorter and smoother", transform=axis.transAxes, fontsize=8, color="#444444")
    axis.grid(True, color="#D7DCE1", linewidth=0.4)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "figure_J_path_length_smoothness_tradeoff")


def main() -> None:
    setup_style()
    selection = load_json(ROOT / "06_selection/CRT_DEVELOPMENT_SELECTION.json")
    variant = selection["selected_variant"]
    if not variant:
        raise RuntimeError("Development did not select a CRT variant")
    manifest = load_json(ROOT / "00_context/CRT_HOLDOUT_MANIFEST.json")
    original = load_records("holdout", "original")
    crt = load_records("holdout", variant)
    if len(original) != 100 or len(crt) != 100:
        raise RuntimeError(f"Holdout records incomplete: original={len(original)}, CRT={len(crt)}")
    representatives = select_representatives(original, crt)
    for row in representatives:
        scenario_id = str(row["scenario_id"])
        for prefix, record in (("original", original[scenario_id]), ("crt", crt[scenario_id])):
            trajectory = load_trajectory(record)
            positions = np.asarray(trajectory["positions"], dtype=float)
            raw_jerk = jerk(trajectory)
            row[f"{prefix}_vertical_total_variation_m"] = float(
                np.sum(np.abs(np.diff(positions[:, :, 2], axis=0)))
            )
            row[f"{prefix}_raw_jerk_peak_mps3"] = float(
                np.max(np.linalg.norm(raw_jerk, axis=2))
            )
        row["vertical_total_variation_delta_m"] = float(
            row["crt_vertical_total_variation_m"]
            - row["original_vertical_total_variation_m"]
        )
        row["raw_jerk_peak_delta_mps3"] = float(
            row["crt_raw_jerk_peak_mps3"] - row["original_raw_jerk_peak_mps3"]
        )
    SOURCE.mkdir(parents=True, exist_ok=True)
    write_csv(SOURCE / "CRT_REPRESENTATIVE_SCENE_MANIFEST.csv", representatives)
    figure_trajectories(representatives, manifest, original, crt, mode="3d")
    figure_trajectories(representatives, manifest, original, crt, mode="xy")
    focus = focus_scene(representatives)
    figure_focus_timeseries(focus, original[str(focus["scenario_id"])], crt[str(focus["scenario_id"])])
    figure_event_alignment(original, crt)
    figure_tradeoff(original, crt)
    artifact = {
        "status": "PASS",
        "selected_variant": variant,
        "selection_rule": representatives[0]["selection_rule"],
        "representatives": representatives,
        "focus_scenario": dict(focus),
        "trajectory_smoothing_applied": False,
        "signal_smoothing_applied": False,
        "axis_contract": "physical coordinates; x/y begin at 0 and z display begins at 0",
        "figures": sorted(path.name for path in PDF.glob("figure_*.pdf")),
    }
    captions = {
        "figure_A_matched_holdout_trajectories_3d": "Matched raw 3-D Holdout trajectories. Color denotes UAV identity; dashed and solid lines denote the original abrupt handoff and selected CRT. Each panel is a stage-median representative among both-success pairs.",
        "figure_B_matched_holdout_trajectories_xy": "Top-down view of the same matched Holdout scenes and raw trajectories used in Figure A.",
        "figure_C_altitude_profiles": "Raw altitude histories for the deterministic focus scene; no trajectory smoothing was applied.",
        "figure_D_raw_jerk_profiles": "Raw jerk-norm histories computed by finite differencing the applied acceleration records.",
        "figure_E_command_vs_executed_reference_altitude": "Commanded and continuously executed reference altitude under the selected CRT.",
        "figure_F_reference_direction_jump_timeline": "Raw inter-step angular changes in the reference direction seen by the frozen SAC-DMP actor.",
        "figure_G_raw_jerk_with_reference_switches": "Raw jerk for the focus-scene UAV with the largest original jerk peak; vertical markers are actual upper command changes.",
        "figure_H_event_aligned_jerk": "Median and interquartile raw jerk norm aligned to upper command-update time over all Holdout switches.",
        "figure_I_event_aligned_vertical_jerk": "Median and interquartile raw vertical jerk aligned to upper command-update time over all Holdout switches.",
        "figure_J_path_length_smoothness_tradeoff": "Paired CRT-minus-Original path-length and smoothness changes for every both-success Holdout scenario; diamonds are stage means.",
    }
    caption_dir = OUT / "captions"
    caption_dir.mkdir(parents=True, exist_ok=True)
    for stem, caption in captions.items():
        (caption_dir / f"{stem}_caption.txt").write_text(caption + "\n", encoding="utf-8")
    artifact["captions"] = captions
    (OUT / "CRT_FIGURE_MANIFEST.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path = ROOT / "09_figures/CRT_FIGURE_INDEX.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
