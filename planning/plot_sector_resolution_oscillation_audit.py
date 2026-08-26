"""Paper-ready diagnostic figures for the read-only sector audit."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "artifacts/sector_resolution_oscillation_audit/20260824_110611"
FORMAL_RECORDS = REPO_ROOT / (
    "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2/"
    "formal_records/M9_Proposed_RERR_GAT_SAC_DMP"
)
DT = 0.1


COLORS = {
    "blue": "#2369A8",
    "orange": "#E07A1F",
    "green": "#2A8C6A",
    "red": "#C94745",
    "purple": "#7554A3",
    "gray": "#646B73",
    "light": "#D7DCE1",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.75,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.25,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, root: Path, stem: str, caption: str) -> None:
    pdf_dir = root / "09_figures/pdf"
    png_dir = root / "09_figures/png_600dpi"
    paper_pdf = root / "11_paper_ready/pdf"
    paper_png = root / "11_paper_ready/png_600dpi"
    caption_dir = root / "11_paper_ready/captions"
    for directory in (pdf_dir, png_dir, paper_pdf, paper_png, caption_dir):
        directory.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{stem}.pdf"
    png_path = png_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    shutil.copy2(pdf_path, paper_pdf / pdf_path.name)
    shutil.copy2(png_path, paper_png / png_path.name)
    (caption_dir / f"{stem}_caption.txt").write_text(caption + "\n", encoding="utf-8")


def sector_arrays(contract: dict):
    centers = contract["sector_centers"]
    directions = np.asarray([row["direction"] for row in centers], dtype=float)
    azimuth = np.asarray([row["azimuth_deg"] for row in centers], dtype=float)
    elevation = np.asarray([row["elevation_deg"] for row in centers], dtype=float)
    return directions, azimuth, elevation


def figure_a(root: Path, contract: dict) -> None:
    directions, _, elevation = sector_arrays(contract)
    fig = plt.figure(figsize=(6.4, 5.1))
    axis = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=-80.0, vmax=80.0)
    color = plt.get_cmap("coolwarm")(norm(elevation))
    axis.scatter(directions[:, 0], directions[:, 1], directions[:, 2], c=color, s=11, depthshade=False)
    for direction, rgba in zip(directions, color):
        axis.plot([0.0, direction[0]], [0.0, direction[1]], [0.0, direction[2]], color=rgba, alpha=0.16, linewidth=0.45)
    axis.set_xlabel("Direction x")
    axis.set_ylabel("Direction y")
    axis.set_zlabel("Direction z")
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.set_zlim(-1.05, 1.05)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=24, azim=38)
    axis.set_title("Current Proposal direction centers (16 azimuth x 16 elevation)")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=axis, shrink=0.7, pad=0.08)
    cbar.set_label("Elevation (deg)")
    save_figure(
        fig,
        root,
        "figure_A_current_proposal_sector_geometry_3d",
        "Current Proposal geometry. The 256 centers are the frozen 16 x 16 LiDAR ray directions used directly by Proposal; color denotes elevation. The grid is uniform in azimuth/elevation angle but not in solid angle.",
    )


def figure_b(root: Path, contract: dict) -> None:
    _, azimuth, elevation = sector_arrays(contract)
    fig, axis = plt.subplots(figsize=(6.4, 4.1))
    axis.scatter(azimuth, elevation, s=13, color=COLORS["blue"], alpha=0.9)
    axis.set_xticks(np.arange(-180, 181, 45))
    axis.set_yticks(np.arange(-80, 81, 20))
    axis.set_xlim(-188, 180)
    axis.set_ylim(-86, 86)
    axis.set_xlabel("Azimuth center (deg)")
    axis.set_ylabel("Elevation center (deg)")
    axis.grid(True)
    axis.set_title("Proposal azimuth/elevation center distribution")
    axis.text(0.01, 0.98, "22.5 deg azimuth spacing\n10.667 deg elevation spacing\nNo exact 0 deg elevation level", transform=axis.transAxes, ha="left", va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0})
    save_figure(
        fig,
        root,
        "figure_B_azimuth_elevation_distribution",
        "Proposal center distribution in angular coordinates. The frozen grid uses 16 uniformly spaced azimuth levels and 16 uniformly spaced elevation levels from -80 to +80 degrees; zero elevation is not a center.",
    )


def figure_c(root: Path, resolution: pd.DataFrame) -> None:
    data = resolution[resolution["reference_distance_quantile"] == "P50"].copy()
    fig, axis = plt.subplots(figsize=(6.4, 4.1))
    bins = np.linspace(data["angular_distance_deg"].min(), data["angular_distance_deg"].max(), 28)
    for kind, color in (("azimuth_only", COLORS["blue"]), ("elevation_only", COLORS["orange"]), ("diagonal", COLORS["green"])):
        values = data.loc[data["adjacency_type"] == kind, "angular_distance_deg"]
        axis.hist(values, bins=bins, alpha=0.55, label=kind.replace("_", " "), color=color)
    axis.axvline(data["angular_distance_deg"].median(), color=COLORS["red"], linestyle="--", linewidth=1.2, label=f"median {data['angular_distance_deg'].median():.00f} deg")
    axis.set_xlabel("Great-circle angle between source-adjacent centers (deg)")
    axis.set_ylabel("Adjacent sector-pair count")
    axis.set_title("Source-adjacent angular-distance distribution")
    axis.legend(frameon=False)
    axis.grid(axis="y")
    save_figure(
        fig,
        root,
        "figure_C_adjacent_sector_angular_distance_distribution",
        "Great-circle distance across the 976 unique source-adjacent Proposal pairs. Uniform angular coordinates do not imply uniform spherical spacing: observed adjacency ranges from 3.88 to 24.87 degrees, with a 15.00-degree median.",
    )


def switch_data(events: pd.DataFrame) -> pd.DataFrame:
    result = events[(events["candidate_to_candidate_switch"] == 1) & (events["candidate_changed"] == 1)].copy()
    return result


def figure_d(root: Path, switches: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.1))
    values = switches["sector_center_angular_jump_deg"].dropna()
    axis.hist(values, bins=np.arange(0, min(181, math.ceil(values.max()) + 5), 5), color=COLORS["blue"], alpha=0.85)
    axis.axvline(values.median(), color=COLORS["red"], linestyle="--", linewidth=1.2, label=f"median {values.median():.2f} deg")
    axis.set_xlabel("Selected sector-center angular jump (deg)")
    axis.set_ylabel("Changed-reference event count")
    axis.set_title("Reference direction jumps in successful Formal V2 episodes")
    axis.grid(axis="y")
    axis.legend(frameon=False)
    save_figure(
        fig,
        root,
        "figure_D_reference_angular_jump_histogram",
        "Distribution of selected Proposal-sector center jumps for 45,116 actual changed-sector events in 381 successful Formal V2 Proposed episodes. Initial selections, terminal handoffs, and same-sector reference changes are excluded.",
    )


def figure_e(root: Path, switches: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.1))
    values = switches["sector_elevation_jump_deg"].dropna()
    limit = max(35, math.ceil(np.percentile(np.abs(values), 99) / 5) * 5)
    bins = np.arange(-limit - 2.5, limit + 5, 5)
    axis.hist(values.clip(-limit, limit), bins=bins, color=COLORS["orange"], alpha=0.88)
    axis.axvline(0.0, color=COLORS["gray"], linewidth=0.8)
    axis.set_xlabel("Selected elevation-center jump (deg)")
    axis.set_ylabel("Changed-reference event count")
    axis.set_title("Elevation jump distribution")
    axis.grid(axis="y")
    axis.text(0.98, 0.95, "10.667 deg per elevation level", transform=axis.transAxes, ha="right", va="top")
    save_figure(
        fig,
        root,
        "figure_E_elevation_jump_histogram",
        "Signed elevation-center changes for the same changed-sector event set. Values are raw selection events; the display clips only the horizontal plotting range at the 99th percentile and does not alter any statistic.",
    )


def relationship_figure(root: Path, switches: pd.DataFrame, x: str, y: str, stem: str, title: str, xlabel: str, ylabel: str, rho: float, caption: str, color: str) -> None:
    data = switches[[x, y]].dropna()
    x_limit = float(np.percentile(data[x], 99.5))
    y_limit = float(np.percentile(data[y], 99.5))
    fig, axis = plt.subplots(figsize=(6.4, 4.5))
    clipped = data[(data[x] <= x_limit) & (data[y] <= y_limit)]
    hexbin = axis.hexbin(clipped[x], clipped[y], gridsize=45, mincnt=1, cmap=color, bins="log")
    cbar = fig.colorbar(hexbin, ax=axis)
    cbar.set_label("log10 event count")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True)
    axis.text(0.98, 0.96, f"Spearman rho = {rho:+.3f}\nn = {len(data):,}", transform=axis.transAxes, ha="right", va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.0})
    save_figure(fig, root, stem, caption)


def select_aba_examples(switches: pd.DataFrame) -> pd.DataFrame:
    candidates = switches[
        (switches["switchback_within_1_0s"] == 1)
        & (switches["sector_elevation_jump_deg"].abs() > 1.0e-9)
        & switches["stage"].isin(["Stage I", "Stage II", "Stage III", "Stage IV"])
    ].copy()
    examples = []
    for stage in ("Stage I", "Stage II", "Stage III", "Stage IV"):
        stage_data = candidates[candidates["stage"] == stage].copy()
        median = stage_data["post_switch_jerk_peak"].median()
        stage_data["distance_to_stage_median_jerk"] = (stage_data["post_switch_jerk_peak"] - median).abs()
        examples.append(stage_data.sort_values(["distance_to_stage_median_jerk", "episode_id", "agent_id", "time_s"]).iloc[0])
    return pd.DataFrame(examples)


def figure_h(root: Path, switches: pd.DataFrame, contract: dict) -> pd.DataFrame:
    examples = select_aba_examples(switches)
    metadata = {int(row["sector_id"]): row for row in contract["sector_centers"]}
    fig, axes = plt.subplots(4, 2, figsize=(7.4, 8.2), sharex=False)
    for row_index, (_, example) in enumerate(examples.iterrows()):
        scenario = str(example["episode_id"])
        agent = int(example["agent_id"])
        center_time = float(example["time_s"])
        stage = str(example["stage"])
        trajectory_path = FORMAL_RECORDS / f"{scenario}_trajectory.npz"
        with np.load(trajectory_path) as trajectory:
            acceleration = np.asarray(trajectory["accelerations"][:, agent, :], dtype=float)
        jerk = np.diff(acceleration, axis=0) / DT
        jerk_time = np.arange(1, acceleration.shape[0]) * DT
        vertical_jerk = jerk[:, 2]
        sequence = switches[(switches["episode_id"] == scenario) & (switches["agent_id"] == agent)].copy().sort_values("time_s")
        start = center_time - 2.0
        end = center_time + 2.0
        local = sequence[(sequence["time_s"] >= start) & (sequence["time_s"] <= end)]
        selection_time = local["time_s"].to_numpy(dtype=float)
        elevation = np.asarray([metadata[int(value)]["elevation_deg"] for value in local["new_sector_id"]], dtype=float)
        sector_a = int(example["new_sector_id"])
        sector_b = int(example["old_sector_id"])
        elevation_a = float(metadata[sector_a]["elevation_deg"])
        elevation_b = float(metadata[sector_b]["elevation_deg"])
        b_time = center_time - float(example["switchback_latency_s"])
        highlight_time = np.asarray([b_time - 0.25, b_time, center_time, center_time + 0.25])
        highlight_elevation = np.asarray([elevation_a, elevation_b, elevation_a, elevation_a])
        axis_sector = axes[row_index, 0]
        axis_jerk = axes[row_index, 1]
        axis_sector.step(selection_time, elevation, where="post", color=COLORS["gray"], linewidth=0.75, alpha=0.6)
        axis_sector.scatter(selection_time, elevation, color=COLORS["gray"], s=9, alpha=0.6, zorder=2)
        axis_sector.step(highlight_time, highlight_elevation, where="post", color=COLORS["blue"], linewidth=1.6, zorder=4)
        axis_sector.scatter(highlight_time[:3], highlight_elevation[:3], color=[COLORS["blue"], COLORS["orange"], COLORS["blue"]], s=20, zorder=5)
        for label, time_value, elevation_value in zip(("A", "B", "A"), highlight_time[:3], highlight_elevation[:3]):
            axis_sector.annotate(label, (time_value, elevation_value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
        axis_sector.axvline(center_time, color=COLORS["red"], linestyle="--", linewidth=0.9)
        axis_sector.set_xlim(start, end)
        axis_sector.set_ylabel("Elevation (deg)")
        axis_sector.set_title(f"{stage}: {scenario}, UAV {agent} sector sequence")
        axis_sector.grid(True)
        mask = (jerk_time >= start) & (jerk_time <= end)
        axis_jerk.plot(jerk_time[mask], vertical_jerk[mask], color=COLORS["orange"], linewidth=0.9)
        axis_jerk.axvline(center_time, color=COLORS["red"], linestyle="--", linewidth=0.9)
        axis_jerk.axhline(0.0, color=COLORS["gray"], linewidth=0.55)
        axis_jerk.set_xlim(start, end)
        axis_jerk.set_ylabel("Vertical jerk (m/s$^3$)")
        axis_jerk.set_title(f"{stage}: raw vertical jerk")
        axis_jerk.grid(True)
        if row_index == 3:
            axis_sector.set_xlabel("Time (s)")
            axis_jerk.set_xlabel("Time (s)")
    fig.suptitle("A-B-A switchback examples selected closest to each stage median switchback jerk", y=0.995)
    fig.tight_layout()
    save_figure(
        fig,
        root,
        "figure_H_ABA_switchback_examples",
        "Raw 0.1 s A-B-A switchback examples, one per stage. Each example is selected before plotting as the switchback whose post-switch jerk peak is closest to that stage's median, preventing manual visual cherry-picking. Dashed lines mark the A-return event.",
    )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    configure_style()
    contract = load_json(root / "01_sector_contract/PROPOSAL_SECTOR_CONTRACT.json")
    resolution = pd.read_csv(root / "01_sector_contract/SECTOR_ANGULAR_RESOLUTION.csv")
    events = pd.read_csv(root / "02_existing_event_analysis/SECTOR_SWITCH_EVENT_TABLE.csv")
    switches = switch_data(events)
    associations = pd.read_csv(root / "02_existing_event_analysis/SECTOR_JERK_ASSOCIATION.csv")
    angle_row = associations.loc[associations["relationship"] == "sector_center_angular_jump_vs_post_switch_jerk_peak"].iloc[0]
    elevation_row = associations.loc[associations["relationship"] == "absolute_elevation_jump_vs_vertical_jerk_peak"].iloc[0]
    rho_angle = float(angle_row["estimate"])
    rho_elevation = float(elevation_row["estimate"])
    figure_a(root, contract)
    figure_b(root, contract)
    figure_c(root, resolution)
    figure_d(root, switches)
    figure_e(root, switches)
    relationship_figure(
        root,
        switches,
        "sector_center_angular_jump_deg",
        "post_switch_jerk_peak",
        "figure_F_angular_jump_vs_jerk_peak",
        "Sector-center angular jump versus post-switch jerk",
        "Sector-center angular jump (deg)",
        "0.5 s post-switch jerk peak (m/s$^3$)",
        rho_angle,
        f"Event-density plot for 45,116 successful-Formal changed-sector events. The association is weakly negative (Spearman rho={rho_angle:+.3f}; episode-cluster 95% CI {float(angle_row['ci_low']):+.3f} to {float(angle_row['ci_high']):+.3f}), contradicting the expected positive signature of coarse angular jumps as the primary jerk driver.",
        "Blues",
    )
    switches["abs_sector_elevation_jump_deg"] = switches["sector_elevation_jump_deg"].abs()
    relationship_figure(
        root,
        switches,
        "abs_sector_elevation_jump_deg",
        "post_switch_vertical_jerk_peak",
        "figure_G_elevation_jump_vs_vertical_jerk_peak",
        "Elevation jump versus post-switch vertical jerk",
        "Absolute elevation-center jump (deg)",
        "0.5 s vertical jerk peak (m/s$^3$)",
        rho_elevation,
        f"Event-density plot for the same changed-sector events. Elevation jump has a weak positive association with vertical jerk (Spearman rho={rho_elevation:+.3f}; episode-cluster 95% CI {float(elevation_row['ci_low']):+.3f} to {float(elevation_row['ci_high']):+.3f}), which is directional but too small to establish vertical quantization as the main cause.",
        "Oranges",
    )
    examples = figure_h(root, switches, contract)
    source_dir = root / "11_paper_ready/source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    examples.to_csv(source_dir / "figure_H_ABA_switchback_examples.csv", index=False, encoding="utf-8-sig")
    availability = {
        "A_current_proposal_sector_geometry_3d": "YES",
        "B_azimuth_elevation_distribution": "YES",
        "C_adjacent_sector_angular_distance_distribution": "YES",
        "D_reference_angular_jump_histogram": "YES",
        "E_elevation_jump_histogram": "YES",
        "F_angular_jump_vs_jerk_peak": "YES",
        "G_elevation_jump_vs_vertical_jerk_peak": "YES",
        "H_ABA_switchback_examples": "YES",
        "I_current_vs_denser_raw_trajectories": "NOT_RUN_DENSIFICATION_HARD_GATE_FAILED",
        "J_current_vs_hysteresis_trajectories": "NOT_RUN_AFTER_MANDATORY_STOP",
        "K_2x2_reliability_smoothness_pareto": "NOT_RUN_DENSIFICATION_HARD_GATE_FAILED",
    }
    (root / "09_figures/FIGURE_AVAILABILITY.json").write_text(json.dumps(availability, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "figure_count": 8, "root": str(root)}))


if __name__ == "__main__":
    main()
