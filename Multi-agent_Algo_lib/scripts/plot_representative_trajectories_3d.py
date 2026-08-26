"""Render reproducible 3-D views of the frozen representative trajectories.

This script is presentation-only. It reads the existing formal trajectory NPZ
files and frozen scenario manifest; it never executes an environment episode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image


METHODS = (
    "dwa_style",
    "rvo_orca_style",
    "terminal",
    "proposal",
    "fp_shep",
    "gat_v1",
)
METHOD_LABELS = {
    "dwa_style": "3D-DWA-style",
    "rvo_orca_style": "RVO/ORCA-style",
    "terminal": "Terminal",
    "proposal": "Proposal",
    "fp_shep": "FP-SHEP",
    "gat_v1": "Proposed",
}
STAGES = ("stage_1", "stage_2", "stage_3", "stage_4")
STAGE_LABELS = {
    "stage_1": "Stage I",
    "stage_2": "Stage II",
    "stage_3": "Stage III",
    "stage_4": "Stage IV",
}
AGENT_COLORS = ("#0072B2", "#D55E00", "#009E73")
AGENT_LINESTYLES = ("-", "--", ":")
STATIC_FACE = "#9A9A9A"
STATIC_EDGE = "#4A4A4A"
DYNAMIC_COLOR = "#CC79A7"
REFERENCE_COLOR = "#6A3D9A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=601)
    return parser.parse_args()


def configure_style() -> str:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = "Times New Roman" if "Times New Roman" in installed else "DejaVu Serif"
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return family


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def outcome_label(row: Mapping[str, str]) -> str:
    if truth(row["team_success"]):
        outcome = "Success"
    else:
        outcome = row.get("failure_taxonomy", "failure").replace("_", " ").title()
    computation_ms = float(row["planning_runtime_ms"])
    return f"{outcome}; compute {computation_ms:.1f} ms"


def load_context(artifact: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, str]]]:
    representative = read_csv(artifact / "representative_trajectory_manifest.csv")
    anchors = {
        row["stage"]: row["scenario_id"]
        for row in representative
        if row["selection_type"] == "stage_anchor_success"
    }
    if set(anchors) != set(STAGES):
        raise RuntimeError(f"Expected four frozen stage anchors, found {anchors}")
    payload = json.loads((artifact / "scenario_manifest.json").read_text(encoding="utf-8"))
    scenes = {entry["scenario_id"]: entry for entry in payload["entries"]}
    formal = read_csv(artifact / "formal_episode_results.csv")
    results = {
        (row["scenario_id"], row["method"]): row
        for row in formal
        if row["scenario_id"] in set(anchors.values())
    }
    return anchors, scenes, results


def sphere_mesh(center: Sequence[float], radius: float, resolution: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, 2 * resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def draw_sphere(ax: plt.Axes, center: Sequence[float], radius: float, color: str, alpha: float) -> None:
    x, y, z = sphere_mesh(center, radius)
    ax.plot_surface(x, y, z, color=color, linewidth=0.15, edgecolor=color, alpha=alpha, shade=True)


def draw_cylinder(ax: plt.Axes, obstacle: Mapping[str, Any]) -> None:
    cx, cy, cz = map(float, obstacle["center"])
    radius = float(obstacle["radius"])
    half_height = float(obstacle["half_height"])
    theta = np.linspace(0.0, 2.0 * np.pi, 32)
    z = np.array([cz - half_height, cz + half_height])
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = cx + radius * np.cos(theta_grid)
    y = cy + radius * np.sin(theta_grid)
    ax.plot_surface(x, y, z_grid, color=STATIC_FACE, edgecolor=STATIC_EDGE, linewidth=0.15, alpha=0.28, shade=True)
    for cap_z in (cz - half_height, cz + half_height):
        ax.plot_trisurf(
            np.r_[cx, cx + radius * np.cos(theta)],
            np.r_[cy, cy + radius * np.sin(theta)],
            np.full(theta.size + 1, cap_z),
            color=STATIC_FACE,
            edgecolor=STATIC_EDGE,
            linewidth=0.12,
            alpha=0.20,
        )


def box_faces(center: Sequence[float], half_extents: Sequence[float]) -> list[list[tuple[float, float, float]]]:
    center_array = np.asarray(center, dtype=float)
    half = np.asarray(half_extents, dtype=float)
    vertices = np.asarray(
        [
            center_array + half * np.array([sx, sy, sz], dtype=float)
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    # Vertex index is binary-coded by x, y, z signs in the nested loops above.
    indices = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
    return [[tuple(vertices[index]) for index in face] for face in indices]


def draw_box(ax: plt.Axes, obstacle: Mapping[str, Any]) -> None:
    collection = Poly3DCollection(
        box_faces(obstacle["center"], obstacle["half_extents"]),
        facecolor=STATIC_FACE,
        edgecolor=STATIC_EDGE,
        linewidth=0.35,
        alpha=0.25,
    )
    ax.add_collection3d(collection)


def draw_static_obstacles(ax: plt.Axes, scene: Mapping[str, Any]) -> None:
    for obstacle in scene["static_obstacles"]:
        obstacle_type = obstacle["type"]
        if obstacle_type == "sphere":
            draw_sphere(ax, obstacle["center"], float(obstacle["radius"]), STATIC_FACE, 0.30)
        elif obstacle_type == "cylinder":
            draw_cylinder(ax, obstacle)
        elif obstacle_type == "box":
            draw_box(ax, obstacle)
        else:
            raise ValueError(f"Unsupported static obstacle type: {obstacle_type}")


def draw_dynamic_obstacles(ax: plt.Axes, scene: Mapping[str, Any]) -> None:
    for obstacle, raw_track in zip(scene["dynamic_obstacles"], scene["dynamic_obstacle_trajectories"]):
        track = np.asarray(raw_track, dtype=float)
        ax.plot(track[:, 0], track[:, 1], track[:, 2], color=DYNAMIC_COLOR, linestyle=(0, (4, 2)), linewidth=1.0, alpha=0.72)
        draw_sphere(ax, track[0], float(obstacle["radius"]), DYNAMIC_COLOR, 0.22)
        ax.scatter(*track[-1], marker="x", s=18, color=DYNAMIC_COLOR, linewidth=0.8, depthshade=False)


def obstacle_bounds(obstacle: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(obstacle["center"], dtype=float)
    if obstacle["type"] == "sphere":
        half = np.full(3, float(obstacle["radius"]))
    elif obstacle["type"] == "cylinder":
        half = np.asarray([obstacle["radius"], obstacle["radius"], obstacle["half_height"]], dtype=float)
    elif obstacle["type"] == "box":
        half = np.asarray(obstacle["half_extents"], dtype=float)
    else:
        raise ValueError(obstacle["type"])
    return center - half, center + half


def iter_points(scene: Mapping[str, Any], trajectories: Mapping[str, Mapping[str, np.ndarray]]) -> Iterable[np.ndarray]:
    yield np.asarray(scene["starts"], dtype=float).reshape(-1, 3)
    yield np.asarray(scene["goals"], dtype=float).reshape(-1, 3)
    for data in trajectories.values():
        yield data["positions"].reshape(-1, 3)
        if "temporary_references" in data:
            yield data["temporary_references"].reshape(-1, 3)
    for obstacle in scene["static_obstacles"]:
        lower, upper = obstacle_bounds(obstacle)
        yield np.vstack((lower, upper))
    for track in scene["dynamic_obstacle_trajectories"]:
        yield np.asarray(track, dtype=float).reshape(-1, 3)


def axis_limits(scene: Mapping[str, Any], trajectories: Mapping[str, Mapping[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    points = np.vstack(list(iter_points(scene, trajectories)))
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = np.maximum(upper - lower, 0.1)
    padding = np.maximum(0.06 * span, np.asarray([0.25, 0.25, 0.20]))
    return lower - padding, upper + padding


def load_trajectories(artifact: Path, stage: str, scenario_id: str) -> dict[str, dict[str, np.ndarray]]:
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        path = artifact / "trajectories" / stage / scenario_id / f"{method}.npz"
        with np.load(path, allow_pickle=False) as payload:
            loaded[method] = {key: np.asarray(payload[key], dtype=float) for key in payload.files}
    return loaded


def style_3d_axis(
    ax: plt.Axes,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str,
    show_labels: bool = True,
) -> None:
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    span = upper - lower
    ax.set_box_aspect(tuple(span))
    ax.view_init(elev=23, azim=-62)
    ax.set_proj_type("persp", focal_length=0.92)
    if show_labels:
        ax.set_xlabel("X (m)", labelpad=2)
        ax.set_ylabel("Y (m)", labelpad=2)
        ax.set_zlabel("Z (m)", labelpad=2)
    ax.set_title(title, pad=4)
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.96, 0.96, 0.96, 0.22))
        axis.pane.set_edgecolor((0.72, 0.72, 0.72, 0.65))
        axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.55)
        axis._axinfo["grid"]["linewidth"] = 0.45


def draw_trajectory_panel(
    ax: plt.Axes,
    scene: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    title: str,
) -> None:
    draw_static_obstacles(ax, scene)
    draw_dynamic_obstacles(ax, scene)
    positions = data["positions"]
    starts = np.asarray(scene["starts"], dtype=float)
    goals = np.asarray(scene["goals"], dtype=float)
    references = data.get("temporary_references")
    for agent_id in range(positions.shape[1]):
        path = positions[:, agent_id, :]
        color = AGENT_COLORS[agent_id]
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=color,
            linestyle=AGENT_LINESTYLES[agent_id],
            linewidth=2.0,
            alpha=0.96,
        )
        ax.scatter(*starts[agent_id], marker="o", s=28, facecolor="white", edgecolor=color, linewidth=1.1, depthshade=False)
        ax.scatter(*goals[agent_id], marker="*", s=52, facecolor=color, edgecolor="#222222", linewidth=0.35, depthshade=False)
        if references is not None:
            ax.scatter(
                *references[agent_id],
                marker="D",
                s=25,
                facecolor="white",
                edgecolor=REFERENCE_COLOR,
                linewidth=0.9,
                depthshade=False,
            )
    style_3d_axis(ax, lower, upper, title)


def legend_handles(include_method_note: bool = False) -> list[Line2D]:
    handles = [
        Line2D([], [], color=AGENT_COLORS[index], linestyle=AGENT_LINESTYLES[index], linewidth=2.0, label=f"UAV {index + 1}")
        for index in range(3)
    ]
    handles.extend(
        [
            Line2D([], [], color="#444444", marker="o", markerfacecolor="white", linestyle="None", label="Start"),
            Line2D([], [], color="#444444", marker="*", markersize=8, linestyle="None", label="Goal"),
            Line2D([], [], color=REFERENCE_COLOR, marker="D", markerfacecolor="white", linestyle="None", label="Temporary reference"),
            Line2D([], [], color=STATIC_EDGE, linewidth=5.0, alpha=0.45, label="Static obstacle"),
            Line2D([], [], color=DYNAMIC_COLOR, linestyle=(0, (4, 2)), linewidth=1.2, label="Dynamic obstacle track"),
        ]
    )
    if include_method_note:
        handles.append(Line2D([], [], color="none", label="Same frozen scenario in every panel"))
    return handles


def save_figure(fig: plt.Figure, png: Path, pdf: Path, dpi: int) -> dict[str, Any]:
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.06, metadata={"Creator": "Matplotlib; frozen benchmark data only"})
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.06, pil_kwargs={"compress_level": 7})
    plt.close(fig)
    with Image.open(png) as image:
        dimensions = image.size
        stored_dpi = image.info.get("dpi", (0.0, 0.0))
    return {
        "png": str(png),
        "pdf": str(pdf),
        "width_px": dimensions[0],
        "height_px": dimensions[1],
        "dpi_x": float(stored_dpi[0]),
        "dpi_y": float(stored_dpi[1]),
        "png_sha256": sha256(png),
        "pdf_sha256": sha256(pdf),
    }


def source_rows(
    stage: str,
    scenario_id: str,
    scene: Mapping[str, Any],
    trajectories: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = {
        "stage": stage,
        "scenario_id": scenario_id,
        "method": "",
        "record_type": "",
        "entity_id": "",
        "step": "",
        "x": "",
        "y": "",
        "z": "",
        "obstacle_type": "",
        "radius": "",
        "half_x": "",
        "half_y": "",
        "half_z": "",
    }

    def append(**values: Any) -> None:
        row = dict(base)
        row.update(values)
        rows.append(row)

    for method, data in trajectories.items():
        for step, frame in enumerate(data["positions"]):
            for agent_id, point in enumerate(frame):
                append(method=method, record_type="trajectory", entity_id=agent_id, step=step, x=point[0], y=point[1], z=point[2])
        if "temporary_references" in data:
            for agent_id, point in enumerate(data["temporary_references"]):
                append(method=method, record_type="temporary_reference", entity_id=agent_id, step=0, x=point[0], y=point[1], z=point[2])
    for kind in ("starts", "goals"):
        for agent_id, point in enumerate(scene[kind]):
            append(method="all", record_type=kind[:-1], entity_id=agent_id, step=0, x=point[0], y=point[1], z=point[2])
    for obstacle_id, obstacle in enumerate(scene["static_obstacles"]):
        half = obstacle.get("half_extents", ("", "", obstacle.get("half_height", "")))
        append(
            method="all",
            record_type="static_obstacle",
            entity_id=obstacle_id,
            step=0,
            x=obstacle["center"][0],
            y=obstacle["center"][1],
            z=obstacle["center"][2],
            obstacle_type=obstacle["type"],
            radius=obstacle.get("radius", ""),
            half_x=half[0] if obstacle["type"] == "box" else "",
            half_y=half[1] if obstacle["type"] == "box" else "",
            half_z=half[2],
        )
    for obstacle_id, (obstacle, track) in enumerate(zip(scene["dynamic_obstacles"], scene["dynamic_obstacle_trajectories"])):
        for step, point in enumerate(track):
            append(
                method="all",
                record_type="dynamic_obstacle",
                entity_id=obstacle_id,
                step=step,
                x=point[0],
                y=point[1],
                z=point[2],
                obstacle_type=obstacle["type"],
                radius=obstacle["radius"],
            )
    return rows


def make_method_comparison(
    artifact: Path,
    stage: str,
    scenario_id: str,
    scene: Mapping[str, Any],
    trajectories: Mapping[str, Mapping[str, np.ndarray]],
    results: Mapping[tuple[str, str], Mapping[str, str]],
    png_dir: Path,
    pdf_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    lower, upper = axis_limits(scene, trajectories)
    fig = plt.figure(figsize=(13.6, 7.8))
    axes = [fig.add_subplot(2, 3, index + 1, projection="3d") for index in range(6)]
    for ax, method in zip(axes, METHODS):
        result = results[(scenario_id, method)]
        title = f"{METHOD_LABELS[method]} — {outcome_label(result)}"
        draw_trajectory_panel(ax, scene, trajectories[method], lower, upper, title)
    fig.legend(
        handles=legend_handles(include_method_note=True),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        columnspacing=1.25,
        handlelength=2.4,
    )
    fig.suptitle(f"{STAGE_LABELS[stage]} matched 3-D trajectories — {scenario_id}", y=0.985, fontsize=12)
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.12, top=0.92, wspace=0.02, hspace=0.10)
    stem = f"{stage}_matched_methods_3d"
    saved = save_figure(fig, png_dir / f"{stem}.png", pdf_dir / f"{stem}.pdf", dpi)
    return {"kind": "matched_six_method_comparison", "stage": stage, "scenario_id": scenario_id, **saved}


def make_proposed_overview(
    entries: Sequence[tuple[str, str, Mapping[str, Any], Mapping[str, Mapping[str, np.ndarray]]]],
    results: Mapping[tuple[str, str], Mapping[str, str]],
    png_dir: Path,
    pdf_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    fig = plt.figure(figsize=(11.2, 8.6))
    axes = [fig.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
    for ax, (stage, scenario_id, scene, trajectories) in zip(axes, entries):
        lower, upper = axis_limits(scene, trajectories)
        title = f"{STAGE_LABELS[stage]} — {scenario_id} — {outcome_label(results[(scenario_id, 'gat_v1')])}"
        draw_trajectory_panel(ax, scene, trajectories["gat_v1"], lower, upper, title)
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        columnspacing=1.3,
        handlelength=2.4,
    )
    fig.suptitle("Proposed method: representative 3-D trajectories", y=0.985, fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.11, top=0.93, wspace=0.02, hspace=0.10)
    stem = "proposed_four_stage_3d_overview"
    saved = save_figure(fig, png_dir / f"{stem}.png", pdf_dir / f"{stem}.pdf", dpi)
    return {"kind": "proposed_four_stage_overview", "stage": "all", "scenario_id": "four frozen anchors", **saved}


def main() -> None:
    args = parse_args()
    artifact = args.artifact.resolve()
    configure_style()
    anchors, scenes, results = load_context(artifact)
    root = artifact / "paper_ready" / "3d_trajectories"
    png_dir = root / "png_600dpi"
    pdf_dir = root / "pdf"
    source_dir = root / "source_data"
    captions_dir = root / "captions"
    script_dir = root / "scripts"
    for directory in (png_dir, pdf_dir, source_dir, captions_dir, script_dir):
        directory.mkdir(parents=True, exist_ok=True)

    stage_entries = []
    all_source: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for stage in STAGES:
        scenario_id = anchors[stage]
        scene = scenes[scenario_id]
        trajectories = load_trajectories(artifact, stage, scenario_id)
        stage_entries.append((stage, scenario_id, scene, trajectories))
        all_source.extend(source_rows(stage, scenario_id, scene, trajectories))
        manifest.append(
            make_method_comparison(
                artifact,
                stage,
                scenario_id,
                scene,
                trajectories,
                results,
                png_dir,
                pdf_dir,
                args.dpi,
            )
        )
    manifest.append(make_proposed_overview(stage_entries, results, png_dir, pdf_dir, args.dpi))

    source_path = source_dir / "representative_trajectories_3d.csv"
    write_csv(source_path, all_source)
    statistics_path = source_dir / "representative_trajectory_panel_compute_time.csv"
    statistics_rows = []
    for stage in STAGES:
        scenario_id = anchors[stage]
        for method in METHODS:
            result = results[(scenario_id, method)]
            statistics_rows.append(
                {
                    "stage": stage,
                    "scenario_id": scenario_id,
                    "method": method,
                    "method_display_name": METHOD_LABELS[method],
                    "outcome": "success" if truth(result["team_success"]) else result["failure_taxonomy"],
                    "planning_runtime_ms": result["planning_runtime_ms"],
                    "planning_decision_count": result["planning_decision_count"],
                    "planning_runtime_per_decision_ms": result["planning_runtime_per_decision_ms"],
                }
            )
    write_csv(statistics_path, statistics_rows)
    caption = (
        "Three-dimensional views of the frozen representative trajectories. "
        "Each stage uses the same deterministic anchor scenario as Figure 8. "
        "The six-method figures use identical axis limits and camera angles within a stage. "
        "Panel titles report measured planning computation time per episode (planning_runtime_ms); "
        "they do not report simulated trajectory execution or completion time. "
        "Lines show executed UAV trajectories; circles, stars, and diamonds mark starts, terminal goals, "
        "and one-shot temporary references. Translucent solids are static obstacles, while magenta dashed "
        "curves show the full frozen dynamic-obstacle tracks. These figures are derived from persisted formal "
        "NPZ trajectories and do not involve rerunning any episode."
    )
    (captions_dir / "representative_trajectories_3d_caption.txt").write_text(caption + "\n", encoding="utf-8")
    copied_script = script_dir / Path(__file__).name
    if copied_script.resolve() != Path(__file__).resolve():
        shutil.copy2(Path(__file__).resolve(), copied_script)
    source_hash = sha256(source_path)
    statistics_hash = sha256(statistics_path)
    script_hash = sha256(copied_script)
    for row in manifest:
        row["source_data"] = str(source_path)
        row["source_data_sha256"] = source_hash
        row["computation_time_source_data"] = str(statistics_path)
        row["computation_time_source_data_sha256"] = statistics_hash
        row["script"] = str(copied_script)
        row["script_sha256"] = script_hash
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "PASSED",
                "time_annotation_field": "planning_runtime_ms",
                "time_annotation_semantics": "planning computation time per episode; execution/completion time is not displayed",
                "figures": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASSED", "figure_count": len(manifest), "output": str(root)}, indent=2))


if __name__ == "__main__":
    main()
