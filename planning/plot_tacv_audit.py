#!/usr/bin/env python3
"""Render TACV audit figures from raw diagnostic and Development artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "artifacts/transient_aware_candidate_veto/20260824_184551"
CRT = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552"
SOURCE = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
ORIGINAL_DIR = CRT / "04_development/records/original/episode_records"
MEDIUM_DIR = ROOT / "06_development/records/tacv_medium/episode_records"
MANIFEST = CRT / "00_context/CRT_DEVELOPMENT_MANIFEST.json"
PDF_DIR = ROOT / "12_paper_ready/pdf"
PNG_DIR = ROOT / "12_paper_ready/png_600dpi"
DATA_DIR = ROOT / "12_paper_ready/source_data"
CAPTION_DIR = ROOT / "12_paper_ready/captions"
FIGURE_DIR = ROOT / "10_figures"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
LIGHT = "#E5E7EB"
AGENT_COLORS = (BLUE, ORANGE, GREEN)
BLOCK_COLORS = {"development": BLUE, "holdout": ORANGE}
VARIANT_COLORS = {
    "original": "#111827",
    "tacv_mild": BLUE,
    "tacv_medium": ORANGE,
    "tacv_strong": GREEN,
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D1D5DB",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, name: str, caption: str) -> None:
    for directory in (PDF_DIR, PNG_DIR, FIGURE_DIR, CAPTION_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(PNG_DIR / f"{name}.png", dpi=600, bbox_inches="tight")
    shutil.copy2(PNG_DIR / f"{name}.png", FIGURE_DIR / f"{name}.png")
    (CAPTION_DIR / f"{name}_caption.txt").write_text(caption.strip() + "\n", encoding="utf-8")
    plt.close(fig)


def write_source(name: str, frame: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")


def stage_label(value: str) -> str:
    return {"stage_1": "Stage I", "stage_2": "Stage II", "stage_3": "Stage III", "stage_4": "Stage IV"}[value]


def plot_a() -> None:
    data = pd.read_csv(ROOT / "02_predictability/PREVIEW_VS_REALIZED_TRANSIENT.csv")
    clean = data[data["clean_window_H4"].astype(bool)].copy()
    clean = clean[(clean["J_preview"] > 0) & (clean["J_real_mean_H0p4"] > 0)]
    pred = load_json(ROOT / "02_predictability/PREVIEW_TRANSIENT_PREDICTABILITY.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True, sharey=True)
    for axis, block in zip(axes, ("development", "holdout")):
        subset = clean[clean["block"] == block]
        image = axis.hexbin(
            subset["J_preview"],
            subset["J_real_mean_H0p4"],
            gridsize=48,
            xscale="log",
            yscale="log",
            mincnt=1,
            bins="log",
            cmap="viridis",
        )
        section = pred["sections"][f"{block}_clean_window"]
        axis.set_title("Development" if block == "development" else "Diagnostic Holdout")
        axis.set_xlabel(r"Preview transient $J_{preview}$")
        axis.text(
            0.04,
            0.96,
            f"$n$={len(subset):,}\n" + rf"$\rho$={section['spearman_rho']:.3f}",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": LIGHT, "pad": 3},
        )
    axes[0].set_ylabel(r"Realized H4 mean-squared jerk")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.025, shrink=0.88)
    colorbar.set_label("log event count")
    fig.suptitle("FP-SHEP preview transient predicts realized switch response only moderately", y=1.02)
    frame = clean[["block", "scenario_id", "stage", "agent_id", "step", "J_preview", "J_real_mean_H0p4"]]
    write_source("figure_A_preview_vs_realized_transient", frame)
    save_figure(
        fig,
        "figure_A_preview_vs_realized_transient",
        "CLEAN-window event density for the existing FP-SHEP preview transient versus raw realized H4 mean-squared jerk. Both independent blocks show a stable positive rank relationship, but substantial dispersion remains.",
    )


def plot_b() -> None:
    data = pd.read_csv(ROOT / "02_predictability/PREVIEW_RANK_CALIBRATION.csv")
    subset = data[(data["subset"] == "clean_window") & (data["binning"] == "decile")].copy()
    fig, axis = plt.subplots(figsize=(5.7, 3.35))
    for block in ("development", "holdout"):
        part = subset[subset["block"] == block].sort_values("bin")
        axis.plot(
            part["bin"],
            part["realized_mean_H0p4"],
            marker="o",
            linewidth=1.8,
            color=BLOCK_COLORS[block],
            label="Development" if block == "development" else "Diagnostic Holdout",
        )
    axis.set_xlabel("Preview-transient decile")
    axis.set_ylabel("Mean realized H4 jerk cost")
    axis.set_xticks(range(1, 11))
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.set_title("Realized jerk rises across preview-transient deciles")
    write_source("figure_B_preview_decile_calibration", subset)
    save_figure(
        fig,
        "figure_B_preview_decile_calibration",
        "Decile calibration on CLEAN-window events. The monotone tendency replicates, while overlap between adjacent deciles limits event-level classification accuracy.",
    )


def plot_c() -> None:
    data = pd.read_csv(ROOT / "03_replaceability/NECESSARY_VS_AVOIDABLE_TRANSIENTS.csv")
    subset = data[data["replaceable"].astype(bool)].dropna(
        subset=["selected_J_preview", "best_achievable_transient_reduction_fraction"]
    ).copy()
    subset["best_alternative_J_preview"] = subset["selected_J_preview"] * (
        1.0 - subset["best_achievable_transient_reduction_fraction"]
    )
    fig, axis = plt.subplots(figsize=(4.6, 4.0))
    for block in ("development", "holdout"):
        part = subset[subset["block"] == block]
        axis.scatter(
            part["selected_J_preview"],
            part["best_alternative_J_preview"],
            s=6,
            alpha=0.22,
            color=BLOCK_COLORS[block],
            edgecolors="none",
            label="Development" if block == "development" else "Diagnostic Holdout",
        )
    x_low = float(subset["selected_J_preview"].min() * 0.92)
    x_high = float(subset["selected_J_preview"].max() * 1.05)
    y_low = float(subset["best_alternative_J_preview"].min() * 0.82)
    y_high = float(subset["selected_J_preview"].max() * 1.05)
    axis.plot([x_low, x_high], [x_low, x_high], color="#111827", linestyle=":", linewidth=1.1, label="No reduction")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Selected candidate $J_{preview}$")
    axis.set_ylabel("Best safety-noninferior alternative")
    axis.set_xlim(x_low, x_high)
    axis.set_ylim(y_low, y_high)
    axis.legend(frameon=False, loc="lower right")
    axis.set_title("Most high-transient selections have a lower-transient alternative")
    write_source(
        "figure_C_selected_vs_best_admissible",
        subset[["block", "scenario_id", "stage", "selected_J_preview", "best_alternative_J_preview", "best_alternative_gat_rank"]],
    )
    save_figure(
        fig,
        "figure_C_selected_vs_best_admissible",
        "Selected high-preview-transient candidates versus the lowest-transient safety-noninferior alternative in the same frozen Top-K bundle. Points below the diagonal are potentially replaceable.",
    )


def grouped_rate(data: pd.DataFrame, group: str) -> pd.DataFrame:
    return (
        data.groupby(["block", group], observed=True)["replaceable"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "replaceable_rate", "count": "event_count"})
    )


def plot_d() -> None:
    data = pd.read_csv(ROOT / "03_replaceability/NECESSARY_VS_AVOIDABLE_TRANSIENTS.csv")
    data["replaceable"] = data["replaceable"].astype(bool)
    summary = grouped_rate(data, "stage")
    stages = ["stage_1", "stage_2", "stage_3", "stage_4"]
    x = np.arange(4)
    width = 0.36
    fig, axis = plt.subplots(figsize=(5.8, 3.25))
    for offset, block in zip((-width / 2, width / 2), ("development", "holdout")):
        part = summary.set_index(["block", "stage"])
        values = [100.0 * float(part.loc[(block, stage), "replaceable_rate"]) for stage in stages]
        axis.bar(x + offset, values, width=width, color=BLOCK_COLORS[block], label="Development" if block == "development" else "Diagnostic Holdout")
    axis.set_xticks(x, [stage_label(stage) for stage in stages])
    axis.set_ylim(70, 100)
    axis.set_ylabel("Replaceable high-transient events (%)")
    axis.legend(frameon=False, ncol=2, loc="lower right")
    axis.set_title("Safe-alternative headroom is high across all stages")
    write_source("figure_D_replaceable_by_stage", summary)
    save_figure(
        fig,
        "figure_D_replaceable_by_stage",
        "Fraction of high-preview-transient selected events with at least one safety-noninferior lower-transient alternative, stratified by stage and replicated diagnostic block.",
    )


def plot_e() -> None:
    data = pd.read_csv(ROOT / "03_replaceability/NECESSARY_VS_AVOIDABLE_TRANSIENTS.csv")
    data["replaceable"] = data["replaceable"].astype(bool)
    summary = grouped_rate(data, "interaction_type")
    categories = sorted(summary["interaction_type"].dropna().unique())
    labels = [value.replace("ordinary_free_flight_", "free-flight\n").replace("_", " ") for value in categories]
    x = np.arange(len(categories))
    width = 0.36
    fig, axis = plt.subplots(figsize=(7.0, 3.3))
    indexed = summary.set_index(["block", "interaction_type"])
    for offset, block in zip((-width / 2, width / 2), ("development", "holdout")):
        values = [100.0 * float(indexed.loc[(block, category), "replaceable_rate"]) for category in categories]
        axis.bar(x + offset, values, width=width, color=BLOCK_COLORS[block], label="Development" if block == "development" else "Diagnostic Holdout")
    axis.set_xticks(x, labels, rotation=15, ha="right")
    axis.set_ylim(70, 100)
    axis.set_ylabel("Replaceable (%)")
    axis.legend(frameon=False, ncol=2, loc="lower right")
    axis.set_title("Safe-alternative headroom is lower in high-peer states")
    write_source("figure_E_replaceable_by_interaction", summary)
    save_figure(
        fig,
        "figure_E_replaceable_by_interaction",
        "Replaceability by source-supported interaction type. High-peer states are rare (22 Development and 16 diagnostic-Holdout events) and show lower replaceability (77.3% and 81.3%) than ordinary free flight. Static-versus-dynamic dominance is omitted because the frozen FP-SHEP clearance observation is untyped.",
    )


def plot_f() -> None:
    data = pd.read_csv(ROOT / "03_replaceability/NECESSARY_VS_AVOIDABLE_TRANSIENTS.csv")
    subset = data[data["replaceable"].astype(bool)].dropna(subset=["best_alternative_gat_rank"]).copy()
    subset["best_alternative_gat_rank"] = subset["best_alternative_gat_rank"].astype(int)
    summary = (
        subset.groupby(["block", "best_alternative_gat_rank"], observed=True)
        .size()
        .reset_index(name="event_count")
    )
    fig, axis = plt.subplots(figsize=(5.8, 3.3))
    bins = np.arange(0.5, 11.5, 1.0)
    for block in ("development", "holdout"):
        part = subset[subset["block"] == block]
        axis.hist(
            part["best_alternative_gat_rank"],
            bins=bins,
            histtype="step",
            linewidth=2.0,
            density=True,
            color=BLOCK_COLORS[block],
            label="Development" if block == "development" else "Diagnostic Holdout",
        )
    axis.axvspan(1.5, 3.5, color=GREEN, alpha=0.08, label="TACV search depth")
    axis.set_xticks(range(1, 11))
    axis.set_xlabel("GAT rank of best admissible alternative")
    axis.set_ylabel("Event density")
    axis.legend(frameon=False, ncol=2)
    axis.set_title("Best transient alternative is not always near the top of GAT ranking")
    write_source("figure_F_admissible_alternative_gat_rank", summary)
    save_figure(
        fig,
        "figure_F_admissible_alternative_gat_rank",
        "Distribution of the GAT rank of the lowest-preview-transient safety-noninferior alternative. The shaded region denotes the frozen TACV rank-2-to-rank-3 search depth.",
    )


def record_map(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.stem: load_json(path)
        for path in sorted(directory.glob("*.json"))
        if not path.stem.endswith("_SOFTWARE_ERROR")
    }


def representatives() -> list[tuple[str, str]]:
    original = record_map(ORIGINAL_DIR)
    medium = record_map(MEDIUM_DIR)
    selected: list[tuple[str, str]] = []
    for stage in ("stage_1", "stage_2", "stage_3", "stage_4"):
        candidates = []
        for scenario_id in sorted(original):
            if original[scenario_id]["entry_identity"]["stage"] != stage:
                continue
            if not original[scenario_id]["episode"]["team_success"] or not medium[scenario_id]["episode"]["team_success"]:
                continue
            delta = float(medium[scenario_id]["episode"]["trajectory_smoothness"]) - float(original[scenario_id]["episode"]["trajectory_smoothness"])
            candidates.append((scenario_id, delta))
        median = float(np.median([value for _, value in candidates]))
        scenario_id = min(candidates, key=lambda pair: (abs(pair[1] - median), pair[0]))[0]
        selected.append((stage, scenario_id))
    return selected


def load_trajectory(directory: Path, record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with np.load(directory / str(record["trajectory_file"]), allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def scene_map() -> dict[str, Path]:
    manifest = load_json(MANIFEST)
    return {str(row["scenario_id"]): SOURCE / str(row["scenario_file"]) for row in manifest["entries"]}


def add_scene_3d(axis: Any, scene: Mapping[str, Any]) -> None:
    for obstacle in scene["static_obstacles"]:
        center = np.asarray(obstacle["center"], dtype=float)
        if obstacle["type"] == "box":
            half = np.asarray(obstacle["half_extents"], dtype=float)
            axis.bar3d(
                center[0] - half[0], center[1] - half[1], center[2] - half[2],
                2 * half[0], 2 * half[1], 2 * half[2],
                color="#9CA3AF", alpha=0.16, shade=False, linewidth=0,
            )
        else:
            axis.scatter(center[0], center[1], center[2], s=18, color="#6B7280", alpha=0.45, marker="s")
    for track in scene.get("dynamic_obstacle_trajectories", []):
        values = np.asarray(track, dtype=float)
        axis.plot(values[:, 0], values[:, 1], values[:, 2], color="#9CA3AF", linestyle=":", linewidth=0.7, alpha=0.6)


def plot_g_h_i() -> None:
    original_records = record_map(ORIGINAL_DIR)
    medium_records = record_map(MEDIUM_DIR)
    scenes = scene_map()
    reps = representatives()
    trajectory_rows: list[dict[str, Any]] = []
    jerk_rows: list[dict[str, Any]] = []
    peer_rows: list[dict[str, Any]] = []
    fig_g = plt.figure(figsize=(8.0, 6.3))
    axes_g = [fig_g.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
    fig_h, axes_h = plt.subplots(2, 2, figsize=(8.0, 5.5), sharex=False)
    fig_i, axes_i = plt.subplots(2, 2, figsize=(8.0, 5.5), sharex=False)
    for index, (stage, scenario_id) in enumerate(reps):
        scene = load_json(scenes[scenario_id])
        add_scene_3d(axes_g[index], scene)
        for method, directory, records, style in (
            ("Original", ORIGINAL_DIR, original_records, "-"),
            ("TACV-Medium (unselected)", MEDIUM_DIR, medium_records, "--"),
        ):
            trajectory = load_trajectory(directory, records[scenario_id])
            positions = trajectory["positions"]
            dt = float(trajectory["dt"])
            for agent in range(positions.shape[1]):
                axes_g[index].plot(
                    positions[:, agent, 0], positions[:, agent, 1], positions[:, agent, 2],
                    color=AGENT_COLORS[agent], linestyle=style, linewidth=1.25 if style == "-" else 1.0,
                    alpha=0.95,
                )
                axes_g[index].scatter(*positions[0, agent], color=AGENT_COLORS[agent], marker="o", s=12)
                axes_g[index].scatter(*positions[-1, agent], color=AGENT_COLORS[agent], marker="*", s=28)
                for step, point in enumerate(positions[:, agent]):
                    trajectory_rows.append(
                        {"stage": stage_label(stage), "scenario_id": scenario_id, "method": method, "agent_id": agent, "time_s": step * dt, "x_m": point[0], "y_m": point[1], "z_m": point[2]}
                    )
            acceleration = trajectory["applied_accelerations"]
            jerk = np.diff(acceleration, axis=0) / dt
            team_jerk = np.mean(np.linalg.norm(jerk, axis=2), axis=1)
            jerk_time = np.arange(1, acceleration.shape[0]) * dt
            axes_h.flat[index].plot(jerk_time, team_jerk, color=VARIANT_COLORS["original"] if method == "Original" else ORANGE, linestyle=style, linewidth=1.0, label=method)
            for time_value, value in zip(jerk_time, team_jerk):
                jerk_rows.append({"stage": stage_label(stage), "scenario_id": scenario_id, "method": method, "time_s": time_value, "team_mean_raw_jerk_mps3": value})
            pairs = [np.linalg.norm(positions[:, first] - positions[:, second], axis=1) for first, second in ((0, 1), (0, 2), (1, 2))]
            min_peer = np.min(np.vstack(pairs), axis=0)
            time = np.arange(positions.shape[0]) * dt
            axes_i.flat[index].plot(time, min_peer, color=VARIANT_COLORS["original"] if method == "Original" else ORANGE, linestyle=style, linewidth=1.0, label=method)
            for time_value, value in zip(time, min_peer):
                peer_rows.append({"stage": stage_label(stage), "scenario_id": scenario_id, "method": method, "time_s": time_value, "minimum_peer_center_distance_m": value})
        axes_g[index].set_title(stage_label(stage))
        axes_g[index].set_xlabel("x (m)", labelpad=1)
        axes_g[index].set_ylabel("y (m)", labelpad=1)
        axes_g[index].set_zlabel("z (m)", labelpad=1)
        axes_g[index].set_xlim(0, 100)
        axes_g[index].set_ylim(0, 100)
        axes_g[index].set_zlim(0, 3.5)
        axes_g[index].view_init(elev=24, azim=-62)
        axes_h.flat[index].set_title(stage_label(stage))
        axes_h.flat[index].set_xlabel("Time (s)")
        axes_h.flat[index].set_ylabel("Raw jerk (m/s³)")
        axes_i.flat[index].set_title(stage_label(stage))
        axes_i.flat[index].set_xlabel("Time (s)")
        axes_i.flat[index].set_ylabel("Minimum peer distance (m)")
        axes_i.flat[index].axhline(0.6, color="#B91C1C", linestyle=":", linewidth=0.9, label="$d_{safe}$")
    handles = [
        mpl.lines.Line2D([0], [0], color="#111827", linestyle="-", label="Original"),
        mpl.lines.Line2D([0], [0], color="#111827", linestyle="--", label="TACV-Medium (unselected)"),
    ] + [mpl.lines.Line2D([0], [0], color=color, label=f"UAV {index + 1}") for index, color in enumerate(AGENT_COLORS)]
    fig_g.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig_g.suptitle("Raw matched Development trajectories: reliability-best TACV arm was not selected", y=0.98)
    fig_g.subplots_adjust(bottom=0.12, hspace=0.16, wspace=0.04)
    handles_h, labels_h = axes_h.flat[0].get_legend_handles_labels()
    fig_h.legend(handles_h, labels_h, loc="lower center", ncol=2, frameon=False)
    fig_h.suptitle("Raw team-mean jerk in stage-median matched examples")
    fig_h.tight_layout(rect=(0, 0.06, 1, 0.96))
    handles_i, labels_i = axes_i.flat[0].get_legend_handles_labels()
    unique = dict(zip(labels_i, handles_i))
    fig_i.legend(unique.values(), unique.keys(), loc="lower center", ncol=3, frameon=False)
    fig_i.suptitle("Raw minimum inter-agent distance in matched examples")
    fig_i.tight_layout(rect=(0, 0.06, 1, 0.96))
    write_source("figure_G_original_vs_tacv_raw_trajectories", pd.DataFrame(trajectory_rows))
    write_source("figure_H_original_vs_tacv_raw_jerk", pd.DataFrame(jerk_rows))
    write_source("figure_I_original_vs_tacv_peer_distance", pd.DataFrame(peer_rows))
    save_figure(
        fig_g,
        "figure_G_original_vs_tacv_raw_trajectories",
        "Raw matched Development trajectories for Original (solid) and the reliability-best but unselected TACV-Medium arm (dashed). UAV identity is encoded by color. Each example is closest to the stage-median paired smoothness change among both-success scenes; no trajectory was smoothed.",
    )
    save_figure(
        fig_h,
        "figure_H_original_vs_tacv_raw_jerk",
        "Raw finite-difference team-mean jerk for the same matched Development examples. TACV-Medium changes local peaks but does not consistently suppress the switch-transient process.",
    )
    save_figure(
        fig_i,
        "figure_I_original_vs_tacv_peer_distance",
        "Raw minimum center-to-center inter-agent distance for the same matched Development examples. The dotted line is the existing 0.6 m safe-distance condition.",
    )


def plot_j() -> None:
    paired = pd.read_csv(ROOT / "06_development/TACV_DEVELOPMENT_PAIRED_SUMMARY.csv")
    rows = pd.concat(
        [
            pd.DataFrame(
                [{"method": "Original", "success_rate_percent": 96.0, "smoothness_reduction_percent": 0.0, "gate_pass": False}]
            ),
            paired.assign(
                method=paired["variant"].map(
                    {"tacv_mild": "TACV-Mild", "tacv_medium": "TACV-Medium", "tacv_strong": "TACV-Strong"}
                ),
                success_rate_percent=100.0 * paired["tacv_success_rate"],
                gate_pass=paired["development_gate_pass"],
            )[["method", "success_rate_percent", "smoothness_reduction_percent", "gate_pass"]],
        ],
        ignore_index=True,
    )
    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    x_upper = max(5.55, float(rows["smoothness_reduction_percent"].max()) + 0.5)
    axis.add_patch(
        mpl.patches.Rectangle(
            (5.0, 95.0),
            x_upper - 5.0,
            2.7,
            facecolor=GREEN,
            edgecolor="none",
            alpha=0.10,
            zorder=0,
        )
    )
    for _, row in rows.iterrows():
        key = "original" if row["method"] == "Original" else row["method"].lower().replace("-", "_")
        axis.scatter(row["smoothness_reduction_percent"], row["success_rate_percent"], s=68, color=VARIANT_COLORS[key], zorder=3)
        offsets = {
            "Original": (6, 6),
            "TACV-Mild": (6, 6),
            "TACV-Medium": (-92, 12),
            "TACV-Strong": (-18, -22),
        }
        axis.annotate(
            row["method"],
            (row["smoothness_reduction_percent"], row["success_rate_percent"]),
            xytext=offsets[row["method"]],
            textcoords="offset points",
        )
    axis.axvline(5.0, color=GREEN, linestyle="--", linewidth=0.9)
    axis.axhline(95.0, color=GREEN, linestyle="--", linewidth=0.9)
    axis.set_xlim(-0.35, x_upper)
    axis.set_ylim(93.4, 97.6)
    axis.set_xlabel("Paired both-success smoothness reduction (%)")
    axis.set_ylabel("Development team success (%)")
    axis.set_title("No TACV arm enters the frozen acceptance region")
    axis.text(x_upper - 0.04, 95.12, "Required region", ha="right", va="bottom", color="#047857")
    write_source("figure_J_reliability_smoothness_pareto", rows)
    save_figure(
        fig,
        "figure_J_reliability_smoothness_pareto",
        "Development reliability-smoothness screening. The frozen gate required success of at least 95% and at least 5% paired both-success smoothness reduction, in addition to peer-collision and switch-jerk conditions. No arm passed all conditions.",
    )


def write_manifest() -> None:
    names = [
        "figure_A_preview_vs_realized_transient",
        "figure_B_preview_decile_calibration",
        "figure_C_selected_vs_best_admissible",
        "figure_D_replaceable_by_stage",
        "figure_E_replaceable_by_interaction",
        "figure_F_admissible_alternative_gat_rank",
        "figure_G_original_vs_tacv_raw_trajectories",
        "figure_H_original_vs_tacv_raw_jerk",
        "figure_I_original_vs_tacv_peer_distance",
        "figure_J_reliability_smoothness_pareto",
    ]
    rows = [
        {
            "figure": name,
            "pdf": str((PDF_DIR / f"{name}.pdf").relative_to(ROOT)),
            "png_600dpi": str((PNG_DIR / f"{name}.png").relative_to(ROOT)),
            "source_data": str((DATA_DIR / f"{name}.csv").relative_to(ROOT)),
            "caption": str((CAPTION_DIR / f"{name}_caption.txt").relative_to(ROOT)),
            "status": "YES" if (PDF_DIR / f"{name}.pdf").is_file() else "NO",
        }
        for name in names
    ]
    with (ROOT / "12_paper_ready/figure_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    setup_style()
    plot_a()
    plot_b()
    plot_c()
    plot_d()
    plot_e()
    plot_f()
    plot_g_h_i()
    plot_j()
    write_manifest()
    print(json.dumps({"status": "PASS", "figure_count": 10}, indent=2))


if __name__ == "__main__":
    main()
