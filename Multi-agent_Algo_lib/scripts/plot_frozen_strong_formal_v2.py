#!/usr/bin/env python3
"""Track-C style raw Formal V2 Original-vs-Frozen-Strong figures."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts/frozen_strong_formal_v2/20260826_110337"
FORMAL_STUDY = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
MANIFEST_PATH = FORMAL_STUDY / "10_formal_v2/FORMAL_V2_MANIFEST.json"
ORIGINAL_DIR = FORMAL_STUDY / "10_formal_v2/formal_records/M9_Proposed_RERR_GAT_SAC_DMP"
STRONG_DIR = ROOT / "03_formal_run/episode_records"
SCENE_FREEZE = ROOT / "01_prefreeze/FORMAL_REPRESENTATIVE_SCENE_FREEZE.json"
TRACK_C = REPO_ROOT / "artifacts/parallel_zigzag_resolution/20260826_091334/track_C_visualization"
STYLE_DIR = TRACK_C / "visualization"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from paper_plot_style import (  # noqa: E402
    TRAJECTORY_WIDTH_PT,
    UAV_COLORS,
    add_identity_legends,
    apply_paper_style,
    draw_environment_3d,
    draw_environment_xy,
    draw_start_goal_3d,
    draw_start_goal_xy,
    draw_trajectory_3d,
    draw_trajectory_xy,
    method_style,
    scale_annotation,
)


PDF_DIR = ROOT / "08_figures/pdf"
SVG_DIR = ROOT / "08_figures/svg"
PNG_DIR = ROOT / "08_figures/png_450dpi"
INTEGRITY = ROOT / "09_paper_ready/FORMAL_FIGURE_DATA_INTEGRITY.csv"
MANIFEST_OUT = ROOT / "09_paper_ready/formal_figure_manifest.csv"
CAPTION_DIR = ROOT / "09_paper_ready/captions"

FIGURES = {
    "Figure_A": "figure_A_formal_representative_3d_original_vs_strong",
    "Figure_B": "figure_B_formal_xy_original_vs_strong",
    "Figure_C": "figure_C_formal_altitude_original_vs_strong",
    "Figure_D": "figure_D_formal_raw_jerk_morphology",
    "Figure_E": "figure_E_formal_high_density_true_scale",
}
METHODS = ("Original", "Frozen Strong")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    temporary.replace(path)


def method_paths(sid: str, method: str) -> tuple[Path, Path]:
    root = ORIGINAL_DIR if method == "Original" else STRONG_DIR
    return root / f"{sid}.json", root / f"{sid}_trajectory.npz"


def trajectory(sid: str, method: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    record_path, npz_path = method_paths(sid, method)
    record = load_json(record_path)
    with np.load(npz_path) as arrays:
        positions = np.asarray(arrays["positions"], dtype=float)
        velocities = np.asarray(arrays["velocities"], dtype=float)
    if positions.ndim != 3 or positions.shape[1:] != (3, 3):
        raise RuntimeError(f"invalid raw trajectory shape: {npz_path}")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise RuntimeError(f"non-finite raw trajectory: {npz_path}")
    return record, positions, velocities


def by_agent(positions: np.ndarray) -> dict[int, np.ndarray]:
    return {agent: positions[:, agent, :] for agent in range(positions.shape[1])}


def method_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#2F3A42", lw=TRAJECTORY_WIDTH_PT, linestyle=method_style(method), label=method)
        for method in METHODS
    ]


def uav_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=UAV_COLORS[agent], lw=TRAJECTORY_WIDTH_PT, label=f"UAV {agent + 1}")
        for agent in sorted(UAV_COLORS)
    ]


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = PDF_DIR / f"{stem}.pdf"
    svg = SVG_DIR / f"{stem}.svg"
    png = PNG_DIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(svg)
    fig.savefig(png, dpi=450)
    plt.close(fig)
    return {
        "pdf": str(pdf.relative_to(REPO_ROOT).as_posix()),
        "svg": str(svg.relative_to(REPO_ROOT).as_posix()),
        "png_450dpi": str(png.relative_to(REPO_ROOT).as_posix()),
    }


def shared_legend(fig: plt.Figure, *, y: float = 0.015) -> None:
    handles = uav_handles() + method_handles()
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=5, frameon=False, handlelength=3.0)


def add_audit(rows: list[dict[str, Any]], figure_id: str, sid: str, method: str, entry: Mapping[str, Any]) -> None:
    record_path, npz_path = method_paths(sid, method)
    record, positions, _ = trajectory(sid, method)
    episode = record["episode"]
    rows.append({
        "figure_id": figure_id,
        "scenario_id": sid,
        "stage_index": int(entry["stage_index"]),
        "family": entry["family"],
        "method": method,
        "trajectory_source": str(npz_path.relative_to(REPO_ROOT).as_posix()),
        "trajectory_sha256": sha256_file(npz_path),
        "record_source": str(record_path.relative_to(REPO_ROOT).as_posix()),
        "record_sha256": sha256_file(record_path),
        "sample_count": int(positions.shape[0]),
        "uav_count": int(positions.shape[1]),
        "start_state": json.dumps(positions[0].tolist(), separators=(",", ":")),
        "end_state": json.dumps(positions[-1].tolist(), separators=(",", ":")),
        "start_state_match": bool(np.array_equal(positions[0], np.asarray(entry["starts"], dtype=float))),
        "static_obstacle_count": len(entry["static_obstacles"]),
        "dynamic_obstacle_count": len(entry["dynamic_obstacles"]),
        "team_success": bool(episode["team_success"]),
        "termination_reason": episode["termination_reason"],
        "raw_trajectory_used": "YES",
        "post_processing_applied": "NO",
        "dt_s": 0.1,
    })


def main() -> None:
    apply_paper_style()
    manifest = load_json(MANIFEST_PATH)
    entry_by_id = {str(entry["scenario_id"]): dict(entry) for entry in manifest["entries"]}
    freeze = load_json(SCENE_FREEZE)
    stage_ids = [freeze["stage_scene_ids"][f"stage_{stage}"] for stage in range(1, 5)]
    primary = str(freeze["primary_comparison_scene"])
    high_density = str(freeze["high_density_scene"])
    if len(list(STRONG_DIR.glob("FORMAL_LR_*.json"))) != 400:
        raise RuntimeError("all 400 Strong records are required before plotting")
    audit_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []

    # A. Four frozen scenes, paired columns, identical per-scene camera/bounds.
    fig = plt.figure(figsize=(12.6, 13.0))
    for row_index, sid in enumerate(stage_ids):
        entry = entry_by_id[sid]
        for col_index, method in enumerate(METHODS):
            ax = fig.add_subplot(4, 2, row_index * 2 + col_index + 1, projection="3d")
            _, positions, _ = trajectory(sid, method)
            draw_environment_3d(ax, entry, camera="OVERVIEW", scale_mode="PRESENTATION_3D")
            draw_start_goal_3d(ax, entry)
            draw_trajectory_3d(ax, by_agent(positions), method)
            ax.set_title(f"Stage {row_index + 1} | {method}", pad=2)
            add_audit(audit_rows, "Figure_A", sid, method, entry)
    fig.text(0.5, 0.048, scale_annotation("PRESENTATION_3D"), ha="center", fontsize=7.4, color="#4A5560")
    shared_legend(fig, y=0.008)
    fig.suptitle("Frozen Formal V2 scenes: raw Original vs Frozen Strong trajectories", y=0.995)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.975, bottom=0.075, hspace=0.18, wspace=0.02)
    paths = save_figure(fig, FIGURES["Figure_A"])
    figure_rows.append({"figure_id": "Figure_A", "title": "Frozen representative 3-D Original vs Strong comparison", "scene_ids": ";".join(stage_ids), "methods": ";".join(METHODS), "layout": "4x2 paired 3-D", "camera": "OVERVIEW", "scale_mode": "PRESENTATION_3D", **paths})

    # B. Stage-wise true XY overlays.
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 10.2), constrained_layout=False)
    for index, (ax, sid) in enumerate(zip(axes.flat, stage_ids, strict=True)):
        entry = entry_by_id[sid]
        draw_environment_xy(ax, entry)
        draw_start_goal_xy(ax, entry)
        for method in METHODS:
            _, positions, _ = trajectory(sid, method)
            draw_trajectory_xy(ax, by_agent(positions), method, alpha=0.92)
            add_audit(audit_rows, "Figure_B", sid, method, entry)
        ax.set_title(f"Stage {index + 1}")
    shared_legend(fig, y=0.008)
    fig.suptitle("Formal V2 raw XY trajectories (shared scene geometry and bounds)", y=0.99)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.945, bottom=0.08, hspace=0.18, wspace=0.16)
    paths = save_figure(fig, FIGURES["Figure_B"])
    figure_rows.append({"figure_id": "Figure_B", "title": "XY top-view comparison", "scene_ids": ";".join(stage_ids), "methods": ";".join(METHODS), "layout": "2x2 stage overlay", "camera": "TOP_VIEW", "scale_mode": "TRUE_SCALE_XY", **paths})

    # C. Raw altitude traces, same stage IDs; no interpolation/downsampling.
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 8.3), sharex=False)
    for stage_index, (ax, sid) in enumerate(zip(axes, stage_ids, strict=True), start=1):
        entry = entry_by_id[sid]
        for method in METHODS:
            _, positions, _ = trajectory(sid, method)
            time_s = np.arange(positions.shape[0]) * 0.1
            for agent in range(3):
                ax.plot(time_s, positions[:, agent, 2], color=UAV_COLORS[agent], linestyle=method_style(method), lw=1.35, alpha=0.92)
            add_audit(audit_rows, "Figure_C", sid, method, entry)
        ax.set_ylabel("z (m)")
        ax.set_title(f"Stage {stage_index}", loc="left", pad=1)
        ax.set_ylim(0.72, 3.28)
        ax.grid(True)
    axes[-1].set_xlabel("Time (s)")
    shared_legend(fig, y=0.005)
    fig.suptitle("Raw executed altitude histories on frozen Formal scenes", y=0.992)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, hspace=0.28)
    paths = save_figure(fig, FIGURES["Figure_C"])
    figure_rows.append({"figure_id": "Figure_C", "title": "z(t) comparison", "scene_ids": ";".join(stage_ids), "methods": ";".join(METHODS), "layout": "4x1 raw altitude", "camera": "TIME_SERIES", "scale_mode": "PHYSICAL_UNITS", **paths})

    # D. Primary scene raw jerk and direction-rate morphology.
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.4))
    entry = entry_by_id[primary]
    for method in METHODS:
        _, _, velocity = trajectory(primary, method)
        acceleration = np.diff(velocity, axis=0) / 0.1
        jerk = np.diff(acceleration, axis=0) / 0.1
        jerk_time = (np.arange(jerk.shape[0]) + 2) * 0.1
        speed_xy = np.linalg.norm(velocity[:, :, :2], axis=2)
        yaw = np.unwrap(np.arctan2(velocity[:, :, 1], velocity[:, :, 0]), axis=0)
        pitch = np.unwrap(np.arctan2(velocity[:, :, 2], np.maximum(speed_xy, 1.0e-12)), axis=0)
        yaw_rate = np.diff(yaw, axis=0) / 0.1
        pitch_rate = np.diff(pitch, axis=0) / 0.1
        rate_time = (np.arange(yaw_rate.shape[0]) + 1) * 0.1
        for agent in range(3):
            kwargs = {"color": UAV_COLORS[agent], "linestyle": method_style(method), "lw": 1.18, "alpha": 0.88}
            axes[0, 0].plot(jerk_time, np.linalg.norm(jerk[:, agent, :], axis=1), **kwargs)
            axes[0, 1].plot(jerk_time, jerk[:, agent, 2], **kwargs)
            axes[1, 0].plot(rate_time, yaw_rate[:, agent], **kwargs)
            axes[1, 1].plot(rate_time, pitch_rate[:, agent], **kwargs)
        add_audit(audit_rows, "Figure_D", primary, method, entry)
    labels = (("Raw 3-D jerk norm", "||j|| (m/s³)"), ("Raw vertical jerk", "jz (m/s³)"), ("Raw yaw rate", "rad/s"), ("Raw pitch rate", "rad/s"))
    for ax, (title, ylabel) in zip(axes.flat, labels, strict=True):
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
    shared_legend(fig, y=0.003)
    fig.suptitle(f"Formal Stage II raw execution morphology | {primary}", y=0.99)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.93, bottom=0.11, hspace=0.33, wspace=0.20)
    paths = save_figure(fig, FIGURES["Figure_D"])
    figure_rows.append({"figure_id": "Figure_D", "title": "Raw jerk and trajectory-morphology comparison", "scene_ids": primary, "methods": ";".join(METHODS), "layout": "2x2 time series", "camera": "TIME_SERIES", "scale_mode": "PHYSICAL_UNITS", **paths})

    # E. Frozen Stage IV high-density scene, paired true-scale 3-D panels.
    fig = plt.figure(figsize=(12.6, 5.4))
    entry = entry_by_id[high_density]
    for index, method in enumerate(METHODS, start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        _, positions, _ = trajectory(high_density, method)
        draw_environment_3d(ax, entry, camera="INTERACTION", scale_mode="TRUE_SCALE")
        draw_start_goal_3d(ax, entry)
        draw_trajectory_3d(ax, by_agent(positions), method)
        # At the physical 100 m x 100 m x roughly 2 m aspect, numerical z tick
        # labels occupy the same few screen pixels.  Keep the true-scale axis
        # geometry and replace only those overlapping labels with an explicit
        # physical-range annotation.
        ax.set_zticks([])
        ax.set_zlabel("")
        ax.text2D(0.91, 0.46, "z = 1--3 m\n(true scale)", transform=ax.transAxes,
                  ha="center", va="center", fontsize=6.3, color="#4A5560")
        ax.set_title(f"Stage IV | {method}", pad=2)
        add_audit(audit_rows, "Figure_E", high_density, method, entry)
    fig.text(0.5, 0.065, scale_annotation("TRUE_SCALE"), ha="center", fontsize=7.4, color="#4A5560")
    shared_legend(fig, y=0.005)
    fig.suptitle(f"High-density Formal interaction scene | {high_density}", y=0.985)
    fig.subplots_adjust(left=0.02, right=0.995, top=0.93, bottom=0.12, wspace=0.02)
    paths = save_figure(fig, FIGURES["Figure_E"])
    figure_rows.append({"figure_id": "Figure_E", "title": "High-density Stage IV interaction scene", "scene_ids": high_density, "methods": ";".join(METHODS), "layout": "1x2 paired 3-D", "camera": "INTERACTION", "scale_mode": "TRUE_SCALE", **paths})

    for row in figure_rows:
        row["status"] = "GENERATED_PENDING_RENDER_VALIDATION"
    write_csv(INTEGRITY, audit_rows)
    write_csv(MANIFEST_OUT, figure_rows)
    CAPTION_DIR.mkdir(parents=True, exist_ok=True)
    captions = {
        "Figure_A": "Raw executed trajectories for the outcome-independent first Formal scene in each Stage. UAV identity is color and method identity is line style. The declared 10x vertical display factor affects presentation only; numerical axes remain in meters.",
        "Figure_B": "True-scale XY overlays for the four frozen Formal scenes. Original and Frozen Strong share identical geometry, obstacle tracks, starts, goals, bounds, and UAV colors.",
        "Figure_C": "Raw 0.1 s altitude histories without interpolation, smoothing, or downsampling. Different trace lengths are the actual episode durations.",
        "Figure_D": "Raw velocity-derived jerk, yaw-rate, and pitch-rate histories on the pre-frozen Stage II primary scene. Values are finite differences of stored executed velocities at 0.1 s.",
        "Figure_E": "True-scale 3-D comparison on the pre-frozen Stage IV high-density scene. Static primitives and moving-obstacle tracks are rendered directly from the Formal manifest.",
    }
    for figure_id, caption in captions.items():
        (CAPTION_DIR / f"{FIGURES[figure_id]}_caption.txt").write_text(caption + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "figure_count": len(figure_rows), "integrity_rows": len(audit_rows)}, indent=2))


if __name__ == "__main__":
    main()
