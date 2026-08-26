"""Generate frozen paper figures from reconciled final benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
ALGO = ROOT / "Multi-agent_Algo_lib"
for _search in (ROOT, ALGO):
    if str(_search) not in sys.path:
        sys.path.insert(0, str(_search))

from scripts.analyze_final_untouched_paper_benchmark import (  # noqa: E402
    ABLATION_METHODS,
    MAIN_METHODS,
    PROPOSED,
    STAGES,
    refresh_report,
)
from scripts.run_final_untouched_paper_benchmark import (  # noqa: E402
    DEFAULT_OUTPUT,
    METHOD_BY_ID,
    METHOD_ORDER,
    load_json,
    write_csv,
    write_json,
)


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#7F7F7F")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2, 1, 2)))


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    return float(value) if value not in (None, "") else float("nan")


def _save(fig: plt.Figure, output: Path, stem: str) -> tuple[str, str]:
    pdf = output / "paper_ready" / "pdf" / f"{stem}.pdf"
    png = output / "paper_ready" / "png_600dpi" / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    return str(pdf.relative_to(output)).replace("\\", "/"), str(png.relative_to(output)).replace("\\", "/")


def _caption(output: Path, stem: str, text: str) -> str:
    path = output / "paper_ready" / "captions" / f"{stem}_caption.txt"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return str(path.relative_to(output)).replace("\\", "/")


def _clean_axes(axis: plt.Axes, grid: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(axis="y", color="0.86", linewidth=0.6, zorder=0)
    axis.tick_params(direction="out", length=3, width=0.7)


def figure_1(output: Path, stage_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source = [row for row in stage_rows if row["method_id"] in MAIN_METHODS]
    write_csv(output / "paper_ready" / "source_data" / "figure_01_success_vs_difficulty.csv", source)
    fig, axis = plt.subplots(figsize=(7.35, 3.55))
    x = np.arange(len(STAGES), dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(MAIN_METHODS))
    for index, method_id in enumerate(MAIN_METHODS):
        rows = [next(row for row in source if row["method_id"] == method_id and row["scope"] == stage) for stage in STAGES]
        y = np.asarray([100 * float(row["success_rate"]) for row in rows])
        lower = np.asarray([100 * float(row["success_ci95_lower"]) for row in rows])
        upper = np.asarray([100 * float(row["success_ci95_upper"]) for row in rows])
        axis.errorbar(
            x + offsets[index], y, yerr=np.vstack([y - lower, upper - y]),
            color=COLORS[index], marker=MARKERS[index], linestyle=LINESTYLES[index],
            markersize=4.5, capsize=2.3, label=METHOD_BY_ID[method_id]["display_name"], zorder=3,
        )
    axis.set_xticks(x, STAGES)
    axis.set_ylim(-3, 105)
    axis.set_ylabel("Team success rate (%)")
    axis.set_xlabel("Scenario difficulty stage")
    axis.set_title("Team success across the frozen four-stage benchmark")
    _clean_axes(axis)
    axis.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01))
    pdf, png = _save(fig, output, "figure_01_success_vs_difficulty")
    caption = _caption(
        output,
        "figure_01_success_vs_difficulty",
        "Figure 1. Team success rate across Stage I-IV on the single untouched 400-scenario manifest. Error bars are exact 95% Clopper-Pearson intervals (100 paired scenarios per stage and method). DWA-FullState is a strong system-level reference; DWA-SensingMatched is the frozen local-sensing comparison.",
    )
    return {"figure": 1, "title": "Team success vs difficulty", "pdf": pdf, "png_600dpi": png, "source": "paper_ready/source_data/figure_01_success_vs_difficulty.csv", "caption": caption, "status": "YES"}


def figure_2(output: Path, overall: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods = (METHOD_ORDER[0], METHOD_ORDER[1], METHOD_ORDER[2], PROPOSED)
    source = [next(row for row in overall if row["method_id"] == method_id) for method_id in methods]
    write_csv(output / "paper_ready" / "source_data" / "figure_02_outcome_composition.csv", source)
    fig, axis = plt.subplots(figsize=(7.1, 3.2))
    x = np.arange(len(methods))
    success = np.asarray([100 * float(row["success_rate"]) for row in source])
    collision = np.asarray([100 * float(row["collision_rate"]) for row in source])
    timeout = np.asarray([100 * float(row["timeout_rate"]) for row in source])
    axis.bar(x, success, width=0.62, color="0.20", label="Success", zorder=3)
    axis.bar(x, collision, width=0.62, bottom=success, color="0.58", hatch="///", label="Collision", zorder=3)
    axis.bar(x, timeout, width=0.62, bottom=success + collision, color="0.86", hatch="..", label="Timeout", zorder=3)
    for index in range(len(x)):
        for bottom, value in ((0, success[index]), (success[index], collision[index]), (success[index] + collision[index], timeout[index])):
            if value >= 5:
                axis.text(index, bottom + value / 2, f"{value:.1f}", ha="center", va="center", fontsize=7, color="white" if bottom == 0 else "black")
    axis.set_xticks(x, [METHOD_BY_ID[method_id]["display_name"].replace(" + SAC-DMP", "") for method_id in methods], rotation=12, ha="right")
    axis.set_ylim(0, 100)
    axis.set_ylabel("Episode composition (%)")
    axis.set_title("Overall categorical outcomes")
    _clean_axes(axis)
    axis.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.01))
    pdf, png = _save(fig, output, "figure_02_outcome_composition")
    caption = _caption(output, "figure_02_outcome_composition", "Figure 2. Overall success, collision, and timeout composition for the two DWA information contracts, Direct SAC-DMP, and the final Proposed method. Categories are mutually exclusive terminal outcomes under the frozen discrete post-transition collision contract.")
    return {"figure": 2, "title": "Outcome composition", "pdf": pdf, "png_600dpi": png, "source": "paper_ready/source_data/figure_02_outcome_composition.csv", "caption": caption, "status": "YES"}


def figure_3(output: Path, overall: Sequence[Mapping[str, Any]], difficulty: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source: list[dict[str, Any]] = []
    for method_id in ABLATION_METHODS:
        source.append(
            {
                "method_id": method_id,
                "display_name": METHOD_BY_ID[method_id]["display_name"],
                "overall_success_rate": next(row for row in overall if row["method_id"] == method_id)["success_rate"],
                "complex_success_rate": next(row for row in difficulty if row["method_id"] == method_id and row["scope"] == "Complex III+IV")["success_rate"],
            }
        )
    write_csv(output / "paper_ready" / "source_data" / "figure_03_ablation_success.csv", source)
    fig, axis = plt.subplots(figsize=(7.2, 3.5))
    x = np.arange(len(source))
    width = 0.36
    overall_values = [100 * float(row["overall_success_rate"]) for row in source]
    complex_values = [100 * float(row["complex_success_rate"]) for row in source]
    axis.bar(x - width / 2, overall_values, width, color="0.22", label="Overall", zorder=3)
    axis.bar(x + width / 2, complex_values, width, color="0.78", edgecolor="0.20", hatch="///", label="Complex (Stage III+IV)", zorder=3)
    for index, values in enumerate(zip(overall_values, complex_values)):
        for offset, value in zip((-width / 2, width / 2), values):
            axis.text(index + offset, value + 1.5, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    labels = [row["display_name"].replace(" + SAC-DMP", "").replace("Proposed R-ERR + GAT", "Proposed") for row in source]
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set_ylim(0, 108)
    axis.set_ylabel("Team success rate (%)")
    axis.set_title("Frozen learned-chain ablation")
    _clean_axes(axis)
    axis.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.01))
    pdf, png = _save(fig, output, "figure_03_ablation_success")
    caption = _caption(output, "figure_03_ablation_success", "Figure 3. Overall and preregistered Complex (Stage III+IV) team success for the complete frozen learned-chain ablation. Direct SAC-DMP executes terminal goals directly; subsequent rows add reference generation, physical preview, one-shot GAT selection, recurrent reconstruction, and GAT under the same recurrent trigger.")
    return {"figure": 3, "title": "Ablation performance", "pdf": pdf, "png_600dpi": png, "source": "paper_ready/source_data/figure_03_ablation_success.csv", "caption": caption, "status": "YES"}


def figure_4(output: Path, contribution: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scopes = ("overall", *STAGES, "Complex III+IV")
    source = [row for row in contribution if row["scope"] in scopes]
    write_csv(output / "paper_ready" / "source_data" / "figure_04_module_contributions.csv", source)
    fig, axis = plt.subplots(figsize=(7.2, 3.3))
    x = np.arange(len(scopes))
    width = 0.36
    labels = ("GAT contribution under R-ERR", "ERR contribution")
    for index, label in enumerate(labels):
        values = [float(next(row for row in source if row["contribution"] == label and row["scope"] == scope)["success_gain_pp"]) for scope in scopes]
        axis.bar(x + (index - 0.5) * width, values, width, color=("0.25" if index == 0 else "0.78"), edgecolor="0.20", hatch=(None if index == 0 else "///"), label=label, zorder=3)
    axis.axhline(0, color="0.25", linewidth=0.8)
    axis.set_xticks(x, ("Overall", "Stage I", "Stage II", "Stage III", "Stage IV", "Complex"), rotation=0)
    axis.set_ylabel("Proposed success contrast (pp)")
    axis.set_title("Separate GAT and recurrent-reconstruction contrasts")
    _clean_axes(axis)
    axis.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.01))
    pdf, png = _save(fig, output, "figure_04_module_contributions")
    caption = _caption(output, "figure_04_module_contributions", "Figure 4. Two distinct paired system contrasts: GAT contribution compares Proposed with R-ERR+FP-SHEP, while ERR contribution compares Proposed with One-Shot GAT. The contrasts are shown separately and are not interpreted as linearly additive module effects.")
    return {"figure": 4, "title": "GAT and ERR contributions", "pdf": pdf, "png_600dpi": png, "source": "paper_ready/source_data/figure_04_module_contributions.csv", "caption": caption, "status": "YES"}


def figure_5(output: Path, overall: Sequence[Mapping[str, Any]], runtime: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source: list[dict[str, Any]] = []
    for method_id in MAIN_METHODS:
        outcome = next(row for row in overall if row["method_id"] == method_id)
        timing = next(row for row in runtime if row["method_id"] == method_id and row["scope"] == "overall")
        source.append({**timing, "success_rate": outcome["success_rate"]})
    write_csv(output / "paper_ready" / "source_data" / "figure_05_runtime_tradeoff.csv", source)
    fig, axis = plt.subplots(figsize=(6.5, 3.8))
    for index, row in enumerate(source):
        x = float(row["total_online_algorithm_compute_mean_ms"])
        y = 100 * float(row["success_rate"])
        axis.scatter(x, y, s=48, color=COLORS[index], marker=MARKERS[index], edgecolor="black", linewidth=0.45, zorder=4)
        axis.annotate(row["display_name"].replace(" + SAC-DMP", "").replace("Proposed R-ERR + GAT", "Proposed"), (x, y), xytext=(5, 4 if index % 2 == 0 else -10), textcoords="offset points", fontsize=7, ha="left", va="center")
    axis.set_xscale("log")
    axis.set_xlabel("Total online algorithm compute per episode (ms, log scale)")
    axis.set_ylabel("Overall team success rate (%)")
    axis.set_title("Performance-compute tradeoff")
    axis.set_ylim(-3, 105)
    _clean_axes(axis, grid=True)
    pdf, png = _save(fig, output, "figure_05_runtime_tradeoff")
    caption = _caption(output, "figure_05_runtime_tradeoff", "Figure 5. Overall team success versus mean total online algorithm compute per episode. DWA compute is cumulative per-step local planning; learned-method compute includes upper planning, execution actor, and DMP kernels. The logarithmic x-axis visualizes accumulated compute and does not imply lower single-decision latency or shorter physical execution time.")
    return {"figure": 5, "title": "Runtime tradeoff", "pdf": pdf, "png_600dpi": png, "source": "paper_ready/source_data/figure_05_runtime_tradeoff.csv", "caption": caption, "status": "YES"}


def figure_6(output: Path, family: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source = [row for row in family if row["method_id"] in MAIN_METHODS]
    write_csv(output / "paper_ready" / "source_data" / "figure_06_stage_family_heatmap.csv", source)
    cells = sorted({(row["stage"], row["family"]) for row in source}, key=lambda item: (STAGES.index(item[0]), item[1]))
    matrix = np.asarray(
        [
            [100 * float(next(row for row in source if row["method_id"] == method_id and row["stage"] == stage and row["family"] == family_name)["success_rate"]) for stage, family_name in cells]
            for method_id in MAIN_METHODS
        ]
    )
    fig, axis = plt.subplots(figsize=(9.0, 3.2))
    image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=6.3, color="white" if value < 42 or value > 82 else "black")
    axis.set_yticks(np.arange(len(MAIN_METHODS)), [METHOD_BY_ID[method_id]["display_name"].replace(" + SAC-DMP", "").replace("Proposed R-ERR + GAT", "Proposed") for method_id in MAIN_METHODS])
    labels = [f"{stage.replace('Stage ', 'S')}\n{family_name}" for stage, family_name in cells]
    axis.set_xticks(np.arange(len(cells)), labels, rotation=55, ha="right")
    axis.set_title("Team success by frozen stage-family cell")
    axis.set_xlabel("Stage-family cell (20 scenarios each)")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.015)
    colorbar.set_label("Success rate (%)")
    pdf, png = _save(fig, output, "figure_06_stage_family_heatmap")
    caption = _caption(output, "figure_06_stage_family_heatmap", "Figure 6. Descriptive team success rate for the six main methods across the 20 preregistered stage-family cells (20 scenarios per cell). Numeric annotations preserve readability in grayscale. No family-level significance tests were performed.")
    return {"figure": 6, "title": "Stage-family success heatmap", "pdf": pdf, "png_600dpi": png, "source": "paper_ready/source_data/figure_06_stage_family_heatmap.csv", "caption": caption, "status": "YES"}


def _sphere(axis: Any, center: Sequence[float], radius: float, *, color: str = "0.72", alpha: float = 0.18) -> None:
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    axis.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=False)


def _box(axis: Any, center: Sequence[float], half: Sequence[float]) -> None:
    center = np.asarray(center, dtype=float)
    half = np.asarray(half, dtype=float)
    corners = np.asarray([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float) * half + center
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(np.abs((corners[i] - corners[j]) / np.maximum(half, 1e-9))) == 2]
    for left, right in edges:
        axis.plot(*np.vstack([corners[left], corners[right]]).T, color="0.55", linewidth=0.45, alpha=0.65)


def _cylinder(axis: Any, center: Sequence[float], radius: float, half_height: float) -> None:
    theta = np.linspace(0, 2 * np.pi, 28)
    for z in (center[2] - half_height, center[2] + half_height):
        axis.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta), np.full_like(theta, z), color="0.55", linewidth=0.45, alpha=0.65)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        axis.plot([center[0] + radius * np.cos(angle)] * 2, [center[1] + radius * np.sin(angle)] * 2, [center[2] - half_height, center[2] + half_height], color="0.55", linewidth=0.4, alpha=0.55)


def figure_7(output: Path, representatives: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = {row["scenario_id"]: row for row in manifest["entries"]}
    source_rows: list[dict[str, Any]] = []
    fig = plt.figure(figsize=(9.1, 7.0))
    axes = [fig.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
    for panel, stage in enumerate(STAGES):
        axis = axes[panel]
        rows = [row for row in representatives if row.get("stage") == stage and row.get("method_id")]
        if not rows:
            axis.text2D(0.5, 0.5, "No Proposed success", transform=axis.transAxes, ha="center")
            axis.set_title(stage)
            continue
        scenario_id = rows[0]["anchor_scenario_id"]
        entry = entries[scenario_id]
        for method_index, method_id in enumerate(MAIN_METHODS):
            trajectory_path = output / "trajectories" / scenario_id / f"{method_id}.npz"
            with np.load(trajectory_path, allow_pickle=False) as archive:
                positions = np.asarray(archive["positions"], dtype=float)
            for agent_id in range(positions.shape[1]):
                points = positions[:, agent_id]
                axis.plot(points[:, 0], points[:, 1], points[:, 2], color=COLORS[method_index], linestyle=("-", "--", ":")[agent_id], linewidth=1.0, alpha=0.92)
                for step, point in enumerate(points):
                    source_rows.append({"stage": stage, "scenario_id": scenario_id, "method_id": method_id, "agent_id": agent_id, "step": step, "x_m": point[0], "y_m": point[1], "z_m": point[2]})
        starts = np.asarray(entry["starts"], dtype=float)
        goals = np.asarray(entry["goals"], dtype=float)
        axis.scatter(starts[:, 0], starts[:, 1], starts[:, 2], marker="^", color="black", s=20, label="Starts")
        axis.scatter(goals[:, 0], goals[:, 1], goals[:, 2], marker="*", color="black", s=35, label="Goals")
        for obstacle in entry["static_obstacles"]:
            if obstacle["type"] == "sphere":
                _sphere(axis, obstacle["center"], float(obstacle["radius"]) + float(obstacle.get("safety_margin", 0.0)))
            elif obstacle["type"] == "box":
                _box(axis, obstacle["center"], np.asarray(obstacle["half_extents"], dtype=float) + float(obstacle.get("safety_margin", 0.0)))
            elif obstacle["type"] == "cylinder":
                _cylinder(axis, obstacle["center"], float(obstacle["radius"]), float(obstacle["half_height"]))
        for track in entry["dynamic_obstacle_trajectories"]:
            points = np.asarray(track, dtype=float)
            axis.plot(points[:, 0], points[:, 1], points[:, 2], color="0.45", linewidth=0.7, linestyle=(0, (2, 2)), alpha=0.8)
            if len(points):
                _sphere(axis, points[0], 0.32, color="0.65", alpha=0.14)
        axis.set_title(f"{stage}: {scenario_id}")
        axis.set_xlabel("x (m)", labelpad=1)
        axis.set_ylabel("y (m)", labelpad=1)
        axis.set_zlabel("z (m)", labelpad=1)
        axis.tick_params(labelsize=6, pad=0)
        axis.view_init(elev=24, azim=-64)
        axis.grid(True, linewidth=0.35, color="0.88")
    method_handles = [Line2D([0], [0], color=COLORS[index], linestyle="-", label=METHOD_BY_ID[method_id]["display_name"].replace(" + SAC-DMP", "").replace("Proposed R-ERR + GAT", "Proposed")) for index, method_id in enumerate(MAIN_METHODS)]
    agent_handles = [Line2D([0], [0], color="0.25", linestyle=style, label=f"UAV {agent+1}") for agent, style in enumerate(("-", "--", ":"))]
    fig.legend(handles=[*method_handles, *agent_handles], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.subplots_adjust(top=0.88, wspace=0.02, hspace=0.18)
    write_csv(output / "paper_ready" / "source_data" / "figure_07_representative_trajectories.csv", source_rows)
    pdf, png = _save(fig, output, "figure_07_representative_trajectories")
    caption = _caption(output, "figure_07_representative_trajectories", "Figure 7. Matched three-dimensional trajectories on one preregistered anchor per stage. Each anchor is the Proposed successful episode whose completion time is closest to that stage's median Proposed-success completion time (scenario-id tie break); all six main methods are then plotted on the identical frozen scene. Solid, dashed, and dotted lines denote UAVs 1-3. Gray geometry and dotted tracks show static and moving obstacles. No scene was selected for visual appearance.")
    return {"figure": 7, "title": "Matched representative 3-D trajectories", "pdf": pdf, "png_600dpi": png, "source": "paper_ready/source_data/figure_07_representative_trajectories.csv", "caption": caption, "status": "YES"}


def figure_8(output: Path, runtime_stage: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods = (METHOD_ORDER[0], METHOD_ORDER[1], METHOD_ORDER[5], METHOD_ORDER[6], PROPOSED)
    source = [row for row in runtime_stage if row["method_id"] in methods]
    write_csv(output / "paper_ready" / "source_data" / "figure_08_planning_frequency.csv", source)
    fig, axis = plt.subplots(figsize=(7.1, 3.3))
    x = np.arange(len(STAGES))
    for index, method_id in enumerate(methods):
        rows = [next(row for row in source if row["method_id"] == method_id and row["scope"] == stage) for stage in STAGES]
        values = [float(row["planning_decisions_mean_per_episode"]) for row in rows]
        axis.plot(x, values, color=COLORS[index], marker=MARKERS[index], linestyle=LINESTYLES[index], label=METHOD_BY_ID[method_id]["display_name"].replace(" + SAC-DMP", "").replace("Proposed R-ERR + GAT", "Proposed"))
    axis.set_xticks(x, STAGES)
    axis.set_xlabel("Scenario difficulty stage")
    axis.set_ylabel("Planning decisions per episode")
    axis.set_title("Per-step and event-triggered planning frequency")
    _clean_axes(axis)
    axis.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.01))
    pdf, png = _save(fig, output, "figure_08_planning_frequency")
    caption = _caption(output, "figure_08_planning_frequency", "Figure 8. Mean planning decisions per episode across difficulty. DWA replans at every executed control step, One-Shot GAT plans only at initialization, and the R-ERR methods plan initially and at state-triggered reconstruction events. Planning frequency and latency are reported separately from accumulated compute.")
    return {"figure": 8, "title": "Planning frequency", "pdf": pdf, "png_600dpi": png, "source": "paper_ready/source_data/figure_08_planning_frequency.csv", "caption": caption, "status": "YES"}


def plot_all(output: Path) -> list[dict[str, Any]]:
    reconciliation = load_json(output / "final_reconciliation.json")
    if reconciliation["FINAL_RECONCILIATION"] != "PASS":
        raise RuntimeError("figures are forbidden until final reconciliation passes")
    cache = load_json(output / "analysis_cache.json")
    overall = cache["overall"]
    stage = cache["stage"]
    difficulty = cache["difficulty"]
    family = cache["family"]
    runtime = _read_csv(output / "runtime_method_summary.csv")
    runtime_stage = _read_csv(output / "runtime_stage_summary.csv")
    contribution = _read_csv(output / "module_contribution.csv")
    representatives = _read_csv(output / "representative_trajectory_manifest.csv")
    manifest = load_json(output / "FINAL_UNTOUCHED_SCENARIO_MANIFEST.json")
    _setup_style()
    figures = [
        figure_1(output, stage),
        figure_2(output, overall),
        figure_3(output, overall, difficulty),
        figure_4(output, contribution),
        figure_5(output, overall, runtime),
        figure_6(output, family),
        figure_7(output, representatives, manifest),
        figure_8(output, runtime_stage),
    ]
    write_csv(output / "paper_ready" / "figure_manifest.csv", figures)
    script_destination = output / "paper_ready" / "plotting_scripts" / Path(__file__).name
    shutil.copy2(Path(__file__), script_destination)
    conclusion = load_json(output / "conclusion.json")
    conclusion["FINAL_PAPER_FIGURES_READY"] = "YES"
    conclusion["FINAL_PAPER_TABLES_READY"] = "YES"
    conclusion["PAPER_FIGURE_COUNT"] = len(figures)
    conclusion["RECOMMENDED_NEXT_STEP"] = "WRITE_PAPER_RESULTS"
    write_json(output / "conclusion.json", conclusion)
    refresh_report(output, figures_ready=True)
    return figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    figures = plot_all(args.output.resolve())
    print(json.dumps({"plotting": "PASS", "figure_count": len(figures)}), flush=True)


if __name__ == "__main__":
    main()
