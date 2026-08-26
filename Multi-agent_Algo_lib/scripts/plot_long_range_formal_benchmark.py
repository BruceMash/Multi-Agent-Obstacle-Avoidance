"""Generate the eleven preregistered paper figures for the long-range study."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
ALGO = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ALGO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.run_long_range_formal_benchmark import (  # noqa: E402
    ARTIFACT_ROOT,
    FORMAL_MANIFEST,
    METHOD_BY_ID,
    METHOD_ORDER,
    RECORD_DIR,
    load_json,
    write_csv,
)


STATISTICS_DIR = ARTIFACT_ROOT / "13_statistics"
FAILURE_DIR = ARTIFACT_ROOT / "14_failure_analysis"
RUNTIME_DIR = ARTIFACT_ROOT / "15_runtime_scaling"
PAPER_DIR = ARTIFACT_ROOT / "16_paper_ready"
PDF_DIR = PAPER_DIR / "figures_pdf"
PNG_DIR = PAPER_DIR / "figures_png_600dpi"
SOURCE_DIR = PAPER_DIR / "source_data"
CAPTION_DIR = PAPER_DIR / "captions"
SCRIPT_DIR = PAPER_DIR / "plot_scripts"
TABLE_DIR = PAPER_DIR / "tables"
STAGES = ("Stage I", "Stage II", "Stage III", "Stage IV")
PROPOSED = METHOD_ORDER[-1]
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#7A3E9D")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (7, 2)))
AGENT_COLORS = ("#0072B2", "#D55E00", "#009E73")
STAGE_POPULATION = {"Stage I": 10, "Stage II": 20, "Stage III": 30, "Stage IV": 40}
SHORT_DIR = ARTIFACT_ROOT / "02_short_coordination"
FIGURE9_METHODS = (
    "M1_DWA_FullState",
    "M2_DWA_SensingMatched",
    "M8_RERR_FP_SHEP_SAC_DMP",
    "M9_Proposed_RERR_GAT_SAC_DMP",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float:
    return float(value) if value not in (None, "", "None") else float("nan")


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="0.9", linewidth=0.45, zorder=0)


def caption(name: str, text: str) -> str:
    path = CAPTION_DIR / f"{name}_caption.txt"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return str(path.relative_to(ARTIFACT_ROOT)).replace("\\", "/")


def save(fig: Any, name: str) -> tuple[str, str]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = PDF_DIR / f"{name}.pdf"
    png = PNG_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return (
        str(pdf.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
        str(png.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
    )


def register(index: int, title: str, name: str, source: Path, text: str, fig: Any) -> dict[str, Any]:
    pdf, png = save(fig, name)
    return {
        "figure": index,
        "title": title,
        "pdf": pdf,
        "png_600dpi": png,
        "source_data": str(source.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
        "caption": caption(name, text),
        "status": "YES",
    }


def line_by_stage(rows: Sequence[Mapping[str, str]], field: str, ylabel: str, title: str, methods: Sequence[str]) -> Any:
    fig, axis = plt.subplots(figsize=(7.1, 3.35))
    x = np.arange(4)
    for index, method_id in enumerate(methods):
        values = [
            100.0 * number(next(row for row in rows if row["method_id"] == method_id and row["scope"] == stage)[field])
            for stage in STAGES
        ]
        axis.plot(x, values, color=COLORS[index], marker=MARKERS[index], linestyle=LINESTYLES[index], linewidth=1.5, label=METHOD_BY_ID[method_id]["display_name"])
    axis.set_xticks(x, STAGES)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Obstacle-population stage")
    axis.set_ylim(0, 103)
    axis.set_title(title)
    clean(axis)
    axis.legend(frameon=False, ncol=2, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    fig.subplots_adjust(right=0.72)
    return fig


def figure_1(stage: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    source = SOURCE_DIR / "figure_01_success_vs_stage.csv"
    write_csv(source, stage)
    fig = line_by_stage(stage, "success_rate", "Team success (%)", "Long-range team success versus obstacle population", METHOD_ORDER)
    return register(1, "Team success versus stage", "figure_01_success_vs_stage", source, "Figure 1. Team success rate across the four frozen obstacle-population stages. Each point uses 100 untouched scenarios; exact intervals are retained in the source table.", fig)


def figure_2(stage: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    source = SOURCE_DIR / "figure_02_collision_vs_stage.csv"
    write_csv(source, stage)
    fig = line_by_stage(stage, "collision_rate", "Any collision (%)", "Collision rate versus obstacle population", METHOD_ORDER)
    return register(2, "Collision rate versus stage", "figure_02_collision_vs_stage", source, "Figure 2. Any-collision rate across the four frozen stages. Collision types are independently recomputed from every stored three-dimensional trajectory.", fig)


def figure_3(overall: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    source = SOURCE_DIR / "figure_03_overall_outcomes.csv"
    write_csv(source, overall)
    fig, axis = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(METHOD_ORDER))
    width = 0.24
    for offset, (field, label, color, hatch) in enumerate((
        ("success_rate", "Success", "#009E73", ""),
        ("collision_rate", "Collision", "#D55E00", "//"),
        ("timeout_rate", "Timeout", "#0072B2", ".."),
    )):
        axis.bar(x + (offset - 1) * width, [100 * number(next(row for row in overall if row["method_id"] == method)[field]) for method in METHOD_ORDER], width, label=label, color=color, hatch=hatch, edgecolor="black", linewidth=0.35)
    axis.set_xticks(x, [f"M{index+1}" for index in range(len(METHOD_ORDER))])
    axis.set_ylabel("Episode rate (%)")
    axis.set_title("Overall categorical outcomes")
    clean(axis)
    axis.legend(frameon=False, ncol=3)
    return register(3, "Overall outcomes", "figure_03_overall_outcomes", source, "Figure 3. Overall success, collision, and timeout rates for the eight frozen methods. M1–M8 follow the left-to-right method order in the paper table.", fig)


def figure_4(overall: Sequence[Mapping[str, str]], runtime: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    source_rows = []
    for method_id in METHOD_ORDER:
        summary = next(row for row in overall if row["method_id"] == method_id)
        timing = next(row for row in runtime if row["method_id"] == method_id and row["scope"] == "overall")
        source_rows.append({"method_id": method_id, "display_name": METHOD_BY_ID[method_id]["display_name"], "success_rate": summary["success_rate"], "compute_ms": timing["total_online_compute_mean_ms"]})
    source = SOURCE_DIR / "figure_04_compute_success.csv"
    write_csv(source, source_rows)
    fig, axis = plt.subplots(figsize=(6.3, 3.6))
    for index, row in enumerate(source_rows):
        x = max(number(row["compute_ms"]), 0.1)
        y = 100 * number(row["success_rate"])
        axis.scatter(x, y, s=46, color=COLORS[index], marker=MARKERS[index], edgecolor="black", linewidth=0.35)
        axis.annotate(f"M{index+1}", (x, y), xytext=(4, 3), textcoords="offset points")
    axis.set_xscale("log")
    axis.set_xlabel("Total online algorithm compute per episode (ms, log scale)")
    axis.set_ylabel("Team success (%)")
    axis.set_title("Compute–success trade-off")
    clean(axis)
    return register(4, "Compute–success trade-off", "figure_04_compute_success", source, "Figure 4. Mean total online algorithm compute versus formal team success. Compute excludes environment stepping, sensing physics, collision checks, checkpoint loading, warm-up, and I/O.", fig)


def figure_5(path_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    rows = [row for row in path_rows if row["scope"] == "overall"]
    source = SOURCE_DIR / "figure_05_path_quality.csv"
    write_csv(source, rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    x = np.arange(len(rows))
    axes[0].bar(x, [number(row["successful_team_path_length_mean_m"]) for row in rows], color=COLORS, edgecolor="black", linewidth=0.35)
    axes[1].bar(x, [100 * number(row["successful_team_path_efficiency_mean"]) for row in rows], color=COLORS, edgecolor="black", linewidth=0.35)
    for axis, ylabel in zip(axes, ("Successful team path (m)", "Successful path efficiency (%)")):
        axis.set_xticks(x, [f"M{i+1}" for i in range(len(rows))])
        axis.set_ylabel(ylabel)
        clean(axis)
    axes[0].set_title("Path length")
    axes[1].set_title("Path efficiency")
    return register(5, "Successful path quality", "figure_05_path_quality", source, "Figure 5. Mean team path length and straight-line path efficiency over successful episodes only; failures are never filled with zeros.", fig)


def figure_6(overall: Sequence[Mapping[str, str]], high: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    learned = METHOD_ORDER[2:]
    source_rows = []
    for method_id in learned:
        source_rows.append({"method_id": method_id, "display_name": METHOD_BY_ID[method_id]["display_name"], "overall_success": next(row for row in overall if row["method_id"] == method_id)["success_rate"], "complex_success": next(row for row in high if row["method_id"] == method_id)["success_rate"]})
    source = SOURCE_DIR / "figure_06_ablation.csv"
    write_csv(source, source_rows)
    fig, axis = plt.subplots(figsize=(7.1, 3.5))
    x = np.arange(len(learned)); width = 0.36
    axis.bar(x-width/2, [100*number(row["overall_success"]) for row in source_rows], width, label="Overall", color="#56B4E9", edgecolor="black", linewidth=0.35)
    axis.bar(x+width/2, [100*number(row["complex_success"]) for row in source_rows], width, label="Stage III+IV", color="#E69F00", hatch="//", edgecolor="black", linewidth=0.35)
    axis.set_xticks(x, ["Direct", "Proposal", "FP-SHEP", "One-Shot GAT", "R-ERR+FP", "Proposed"], rotation=18, ha="right")
    axis.set_ylabel("Team success (%)"); axis.set_title("Frozen learned-chain ablation"); clean(axis); axis.legend(frameon=False)
    return register(6, "Learned-chain ablation", "figure_06_ablation", source, "Figure 6. Frozen learned-chain ablation for overall and high-density Stage III+IV success. All arms share the adapted 256-ray SAC-DMP execution policy.", fig)


def figure_7(contribution: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    rows = [row for row in contribution if row["scope"] in ("overall", "Complex III+IV")]
    source = SOURCE_DIR / "figure_07_module_contribution.csv"
    write_csv(source, rows)
    labels = [f"{row['contribution']}\n{row['scope']}" for row in rows]
    values = [number(row["success_gain_pp"]) for row in rows]
    fig, axis = plt.subplots(figsize=(6.5, 3.4))
    axis.bar(np.arange(len(rows)), values, color=("#0072B2", "#56B4E9", "#D55E00", "#E69F00"), edgecolor="black", linewidth=0.4)
    axis.axhline(0, color="black", linewidth=0.6)
    axis.set_xticks(np.arange(len(rows)), labels)
    axis.set_ylabel("Proposed success gain (pp)"); axis.set_title("ERR and GAT contribution contrasts"); clean(axis)
    return register(7, "Module contribution", "figure_07_module_contribution", source, "Figure 7. Paired success-rate contrasts isolating recurrent reconstruction relative to One-Shot GAT and GAT ranking relative to R-ERR+FP-SHEP. Contributions are not asserted to be additive.", fig)


def figure_8(family: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    rows = [row for row in family if row["method_id"] == PROPOSED]
    families = sorted({row["family"] for row in rows})
    matrix = np.asarray([[100*number(next(row for row in rows if row["stage"] == stage and row["family"] == family)["success_rate"]) for family in families] for stage in STAGES])
    source = SOURCE_DIR / "figure_08_stage_family_heatmap.csv"
    write_csv(source, rows)
    fig, axis = plt.subplots(figsize=(7.2, 3.35))
    image = axis.imshow(matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    axis.set_xticks(np.arange(len(families)), [name.replace("_", " ") for name in families], rotation=22, ha="right")
    axis.set_yticks(np.arange(4), STAGES)
    for i in range(4):
        for j in range(len(families)):
            axis.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center", color="white" if matrix[i,j] < 55 else "black", fontsize=7.5)
    fig.colorbar(image, ax=axis, label="Proposed success (%)", fraction=0.035, pad=0.02)
    axis.set_title("Proposed stage–family success")
    return register(8, "Stage–family heatmap", "figure_08_stage_family_heatmap", source, "Figure 8. Descriptive Proposed success rates for each frozen stage–geometry-family cell (20 scenarios per cell). Family-level results are not separate confirmatory tests.", fig)


def _plot_static(axis: Any, spec: Mapping[str, Any]) -> None:
    center = np.asarray(spec["center"], dtype=float)
    if spec["type"] == "box":
        half = np.asarray(spec["half_extents"], dtype=float)
        for sx in (-1, 1):
            for sy in (-1, 1):
                axis.plot([center[0]+sx*half[0]]*2, [center[1]+sy*half[1]]*2, [max(0,center[2]-half[2]), center[2]+half[2]], color="0.65", linewidth=0.35, alpha=0.5)
    else:
        theta = np.linspace(0, 2*np.pi, 20)
        radius = float(spec["radius"])
        for z in (max(0, center[2]-float(spec["half_height"])), center[2]+float(spec["half_height"])):
            axis.plot(center[0]+radius*np.cos(theta), center[1]+radius*np.sin(theta), z, color="0.65", linewidth=0.35, alpha=0.5)


def figure_9(representatives: Sequence[Mapping[str, str]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = {row["scenario_id"]: row for row in manifest["entries"]}
    anchors = {stage: next(row["anchor_scenario_id"] for row in representatives if row["stage"] == stage and row.get("anchor_scenario_id")) for stage in STAGES}
    methods = (METHOD_ORDER[0], METHOD_ORDER[1], METHOD_ORDER[2], METHOD_ORDER[5], METHOD_ORDER[6], METHOD_ORDER[7])
    source_rows: list[dict[str, Any]] = []
    fig = plt.figure(figsize=(9.0, 7.4))
    for panel, stage in enumerate(STAGES, start=1):
        axis = fig.add_subplot(2, 2, panel, projection="3d")
        scenario_id = anchors[stage]
        entry = entries[scenario_id]
        for spec in entry["static_obstacles"]:
            _plot_static(axis, spec)
        for track in entry["dynamic_obstacle_trajectories"]:
            sampled = np.asarray(track, dtype=float)[::30]
            axis.plot(sampled[:,0], sampled[:,1], sampled[:,2], color="0.35", linestyle=":", linewidth=0.6, alpha=0.7)
        for method_index, method_id in enumerate(methods):
            trajectory_path = RECORD_DIR / method_id / f"{scenario_id}_trajectory.npz"
            with np.load(trajectory_path, allow_pickle=False) as archive:
                positions = np.asarray(archive["positions"], dtype=float)
            for agent_id in range(3):
                path = positions[:, agent_id]
                axis.plot(path[:,0], path[:,1], path[:,2], color=AGENT_COLORS[agent_id], linestyle=LINESTYLES[method_index], linewidth=0.9, alpha=0.82)
                for step in range(0, len(path), max(1, len(path)//80)):
                    source_rows.append({"stage": stage, "scenario_id": scenario_id, "method_id": method_id, "agent_id": agent_id, "step": step, "x_m": path[step,0], "y_m": path[step,1], "z_m": path[step,2]})
        starts = np.asarray(entry["starts"], dtype=float); goals = np.asarray(entry["goals"], dtype=float)
        for agent_id in range(3):
            axis.scatter(*starts[agent_id], color=AGENT_COLORS[agent_id], marker="^", s=18)
            axis.scatter(*goals[agent_id], color=AGENT_COLORS[agent_id], marker="*", s=28)
        axis.set_xlim(0,100); axis.set_ylim(0,100); axis.set_zlim(0,4)
        axis.set_xlabel("x (m)", labelpad=2); axis.set_ylabel("y (m)", labelpad=2); axis.set_zlabel("z (m)", labelpad=0)
        axis.set_title(stage, pad=2)
        axis.view_init(elev=27, azim=-57)
        axis.grid(True, linewidth=0.25, color="0.9")
    agent_handles = [Line2D([0],[0],color=AGENT_COLORS[i],linestyle="-",label=f"UAV {i+1}") for i in range(3)]
    method_handles = [Line2D([0],[0],color="0.2",linestyle=LINESTYLES[i],label=METHOD_BY_ID[m]["display_name"].replace(" + SAC-DMP", "")) for i,m in enumerate(methods)]
    fig.legend(handles=[*agent_handles,*method_handles], ncol=5, loc="upper center", bbox_to_anchor=(0.5,0.995), frameon=False)
    fig.subplots_adjust(top=0.90, wspace=0.01, hspace=0.10)
    source = SOURCE_DIR / "figure_09_matched_3d_trajectories.csv"
    write_csv(source, source_rows)
    return register(9, "Matched representative 3-D trajectories", "figure_09_matched_3d_trajectories", source, "Figure 9. Matched 3-D trajectories on one preregistered median-success Proposed anchor per stage. UAV identity is encoded by color and method by line style. Static structures are gray; dotted gray paths are frozen moving-obstacle tracks. Axes begin at zero and no negative coordinate is introduced.", fig)


def figure_10(density: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    rows = [row for row in density if row["scope"] == "overall"]
    source = SOURCE_DIR / "figure_10_replanning_density.csv"; write_csv(source, rows)
    fig, axis = plt.subplots(figsize=(7.1,3.4)); x=np.arange(len(rows))
    axis.bar(x,[number(row["planning_decisions_mean_per_100m"]) for row in rows],color=COLORS,edgecolor="black",linewidth=0.35)
    axis.set_xticks(x,[f"M{i+1}" for i in range(len(rows))]); axis.set_ylabel("Planning decisions per 100 m"); axis.set_title("Planning and reconstruction density"); clean(axis)
    return register(10, "Planning density", "figure_10_replanning_density", source, "Figure 10. Mean online planning decisions normalized by frozen mean mission distance. One-shot methods plan once; DWA plans per control step; R-ERR plans at state-triggered events.", fig)


def figure_11(failures: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    rows = [row for row in failures if row["scope"] == "overall"]
    reasons = sorted({row["failure_type"] for row in rows})
    source = SOURCE_DIR / "figure_11_failure_taxonomy.csv"; write_csv(source, rows)
    fig, axis = plt.subplots(figsize=(7.2,3.6)); x=np.arange(len(METHOD_ORDER)); bottom=np.zeros(len(METHOD_ORDER))
    palette=plt.get_cmap("tab10")
    for index, reason in enumerate(reasons):
        values=np.asarray([number(next((row["count"] for row in rows if row["method_id"]==method and row["failure_type"]==reason),0)) for method in METHOD_ORDER])
        axis.bar(x,values,bottom=bottom,label=reason.replace("_"," "),color=palette(index),edgecolor="black",linewidth=0.25)
        bottom += values
    axis.set_xticks(x,[f"M{i+1}" for i in range(len(METHOD_ORDER))]); axis.set_ylabel("Failed episodes"); axis.set_title("Exclusive failure taxonomy"); clean(axis); axis.legend(frameon=False,ncol=3,bbox_to_anchor=(0.5,1.18),loc="upper center")
    return register(11, "Failure taxonomy", "figure_11_failure_taxonomy", source, "Figure 11. Exclusive formal failure taxonomy. Collision labels are decomposed independently from stored trajectories; timeout and other terminal causes retain the frozen evaluator semantics.", fig)


def generate_tables(overall: Sequence[Mapping[str, str]], stage: Sequence[Mapping[str, str]], path: Sequence[Mapping[str, str]], runtime: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    TABLE_DIR.mkdir(parents=True,exist_ok=True)
    table1=TABLE_DIR/"table_01_overall.csv"; write_csv(table1,overall)
    table2=TABLE_DIR/"table_02_stage.csv"; write_csv(table2,stage)
    table3=TABLE_DIR/"table_03_path_quality.csv"; write_csv(table3,[row for row in path if row["scope"]=="overall"])
    table4=TABLE_DIR/"table_04_runtime.csv"; write_csv(table4,[row for row in runtime if row["scope"]=="overall"])
    return [{"table":i,"title":title,"source":str(file.relative_to(ARTIFACT_ROOT)).replace("\\","/"),"status":"YES"} for i,(title,file) in enumerate((("Overall outcomes",table1),("Stage-wise outcomes",table2),("Path quality",table3),("Runtime and scaling",table4)),start=1)]


# The functions below implement the paper figure contract for the long-range
# rebuild.  They deliberately keep the historical plotting helpers above as
# provenance, but ``plot_all`` only calls this contract-v2 implementation.


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def method_code(method_id: str) -> str:
    return str(method_id).split("_", 1)[0]


def _load_formal_positions(method_id: str, scenario_id: str) -> np.ndarray:
    path = RECORD_DIR / method_id / f"{scenario_id}_trajectory.npz"
    with np.load(path, allow_pickle=False) as archive:
        positions = np.asarray(archive["positions"], dtype=float)
        if positions.ndim == 3:
            return positions
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise RuntimeError(f"unexpected trajectory shape for {method_id}/{scenario_id}: {positions.shape}")
        steps = np.asarray(archive["steps"], dtype=int)
        agents = np.asarray(archive["agent_ids"], dtype=int)
    unique_steps = np.unique(steps)
    shaped = np.empty((len(unique_steps), 3, 3), dtype=float)
    for frame_index, step in enumerate(unique_steps):
        for agent_id in range(3):
            match = np.flatnonzero((steps == step) & (agents == agent_id))
            if len(match) != 1:
                raise RuntimeError("trajectory is not a complete step-agent grid")
            shaped[frame_index, agent_id] = positions[int(match[0])]
    return shaped


def _load_short_positions(scenario_id: str) -> np.ndarray:
    path = SHORT_DIR / "episode_records" / f"{scenario_id}_trajectory.npz"
    with np.load(path, allow_pickle=False) as archive:
        flat = np.asarray(archive["positions"], dtype=float)
        steps = np.asarray(archive["steps"], dtype=int)
        agents = np.asarray(archive["agent_ids"], dtype=int)
    unique_steps = np.unique(steps)
    positions = np.empty((len(unique_steps), 3, 3), dtype=float)
    for frame_index, step in enumerate(unique_steps):
        for agent_id in range(3):
            match = np.flatnonzero((steps == step) & (agents == agent_id))
            if len(match) != 1:
                raise RuntimeError("short trajectory is not a complete step-agent grid")
            positions[frame_index, agent_id] = flat[int(match[0])]
    return positions


def _plot_static_topdown(axis: Any, spec: Mapping[str, Any], *, alpha: float = 0.32) -> None:
    center = np.asarray(spec["center"], dtype=float)
    if spec["type"] == "box":
        half = np.asarray(spec["half_extents"], dtype=float)
        patch = Rectangle(
            (center[0] - half[0], center[1] - half[1]),
            2.0 * half[0],
            2.0 * half[1],
            facecolor="0.58",
            edgecolor="0.35",
            linewidth=0.35,
            alpha=alpha,
        )
    else:
        patch = Circle(
            (center[0], center[1]),
            float(spec["radius"]),
            facecolor="0.58",
            edgecolor="0.35",
            linewidth=0.35,
            alpha=alpha,
        )
    axis.add_patch(patch)


def _paper_figure_1_short_coordination() -> dict[str, Any]:
    rows = read_csv(SHORT_DIR / "short_coordination_results.csv")
    representatives = read_csv(SHORT_DIR / "representative_trajectory_manifest.csv")
    manifest = load_json(SHORT_DIR / "SHORT_COORDINATION_MANIFEST.json")
    entries = {str(row["scenario_id"]): row for row in manifest["entries"]}
    order = ("crossing", "merge", "conflict")
    selected = {
        row["task_pattern"]: row for row in representatives
        if row.get("scenario_id") and row["task_pattern"] in order
    }
    if set(selected) != set(order):
        raise RuntimeError("Figure 1 requires a successful representative for every short task family")
    source_rows: list[dict[str, Any]] = []
    fig = plt.figure(figsize=(9.1, 3.25))
    labels = {"crossing": "Crossing", "merge": "Merge", "conflict": "Conflict"}
    for panel, pattern in enumerate(order, start=1):
        axis = fig.add_subplot(1, 3, panel, projection="3d")
        scenario_id = str(selected[pattern]["scenario_id"])
        entry = entries[scenario_id]
        positions = _load_short_positions(scenario_id)
        starts = np.asarray(entry["starts"], dtype=float)
        goals = np.asarray(entry["goals"], dtype=float)
        result = next(row for row in rows if row["scenario_id"] == scenario_id)
        if not truth(result["team_success"]):
            raise RuntimeError("short representative is not a team success")
        stride = max(1, len(positions) // 160)
        for agent_id, color in enumerate(AGENT_COLORS):
            path = positions[:, agent_id]
            axis.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=1.7)
            axis.scatter(*starts[agent_id], color=color, marker="^", s=24, depthshade=False)
            axis.scatter(*goals[agent_id], color=color, marker="*", s=52, depthshade=False)
            for step in range(0, len(path), stride):
                source_rows.append(
                    {
                        "task_pattern": pattern,
                        "scenario_id": scenario_id,
                        "agent_id": agent_id,
                        "step": step,
                        "x_m": path[step, 0],
                        "y_m": path[step, 1],
                        "z_m": path[step, 2],
                        "team_success": True,
                    }
                )
        axis.set_xlim(0.0, 9.0)
        axis.set_ylim(0.0, 4.5)
        axis.set_zlim(0.0, 2.4)
        axis.set_xlabel("x (m)", labelpad=1)
        axis.set_ylabel("y (m)", labelpad=1)
        axis.set_zlabel("z (m)", labelpad=0)
        axis.set_title(labels[pattern], pad=1)
        axis.view_init(elev=24, azim=-62)
        axis.grid(True, linewidth=0.25, color="0.90")
    fig.legend(
        handles=[Line2D([0], [0], color=AGENT_COLORS[i], label=f"UAV {i + 1}") for i in range(3)],
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(top=0.86, wspace=0.04)
    source = SOURCE_DIR / "figure_01_short_horizon_coordination.csv"
    write_csv(source, source_rows)
    return register(
        1,
        "Short-horizon local coordination",
        "figure_01_short_horizon_coordination",
        source,
        "Figure 1. Successful three-UAV short-horizon coordination in the crossing, merge, and conflict tasks. UAV identity is encoded by color; triangles and stars denote starts and goals. The 100-scene capability test was run only after long-range development was closed and did not participate in method selection.",
        fig,
    )


def _paper_figure_2_benchmark(manifest: Mapping[str, Any]) -> dict[str, Any]:
    family_order = list(manifest.get("family_order", []))
    if not family_order:
        family_order = sorted({str(row["family"]) for row in manifest["entries"]})
    entries = []
    for family in family_order:
        candidates = [
            row for row in manifest["entries"]
            if row["stage"] == "stage_4" and row["family"] == family
        ]
        if not candidates:
            raise RuntimeError(f"no Stage IV benchmark scene for family {family}")
        entries.append(min(candidates, key=lambda row: str(row["scenario_id"])))
    source_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.0))
    flat_axes = list(axes.flat)
    for panel, (axis, entry) in enumerate(zip(flat_axes, entries), start=1):
        for obstacle_id, spec in enumerate(entry["static_obstacles"]):
            _plot_static_topdown(axis, spec)
            center = spec["center"]
            source_rows.append(
                {
                    "panel": panel,
                    "scenario_id": entry["scenario_id"],
                    "family": entry["family"],
                    "entity": "static_obstacle",
                    "entity_id": obstacle_id,
                    "x_m": center[0],
                    "y_m": center[1],
                    "z_m": center[2],
                    "shape": spec["type"],
                }
            )
        for obstacle_id, track in enumerate(entry["dynamic_obstacle_trajectories"]):
            points = np.asarray(track, dtype=float)
            stride = max(1, len(points) // 80)
            sampled = points[::stride]
            axis.plot(sampled[:, 0], sampled[:, 1], color="0.20", linestyle=":", linewidth=0.75)
            axis.scatter(points[0, 0], points[0, 1], marker="o", s=13, color="0.20")
            for step, point in enumerate(points[::stride]):
                source_rows.append(
                    {
                        "panel": panel,
                        "scenario_id": entry["scenario_id"],
                        "family": entry["family"],
                        "entity": "dynamic_track",
                        "entity_id": obstacle_id,
                        "step": step * stride,
                        "x_m": point[0],
                        "y_m": point[1],
                        "z_m": point[2],
                    }
                )
        starts = np.asarray(entry["starts"], dtype=float)
        goals = np.asarray(entry["goals"], dtype=float)
        for agent_id, color in enumerate(AGENT_COLORS):
            axis.scatter(starts[agent_id, 0], starts[agent_id, 1], marker="^", s=24, color=color, zorder=4)
            axis.scatter(goals[agent_id, 0], goals[agent_id, 1], marker="*", s=40, color=color, zorder=4)
            axis.plot(
                [starts[agent_id, 0], goals[agent_id, 0]],
                [starts[agent_id, 1], goals[agent_id, 1]],
                color=color,
                linestyle="--",
                linewidth=0.45,
                alpha=0.45,
            )
            for role, point in (("start", starts[agent_id]), ("goal", goals[agent_id])):
                source_rows.append(
                    {
                        "panel": panel,
                        "scenario_id": entry["scenario_id"],
                        "family": entry["family"],
                        "entity": role,
                        "entity_id": agent_id,
                        "x_m": point[0],
                        "y_m": point[1],
                        "z_m": point[2],
                    }
                )
        axis.add_patch(
            Circle(
                (starts[0, 0], starts[0, 1]),
                4.5,
                facecolor="none",
                edgecolor=AGENT_COLORS[0],
                linestyle="--",
                linewidth=0.8,
                alpha=0.8,
            )
        )
        axis.set_xlim(0.0, 100.0)
        axis.set_ylim(0.0, 100.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_title(str(entry.get("family_label", entry["family"])).replace("_", " "))
        axis.grid(True, color="0.92", linewidth=0.35)
    legend_axis = flat_axes[-1]
    legend_axis.axis("off")
    legend_axis.legend(
        handles=[
            *[Line2D([0], [0], marker="^", linestyle="none", color=AGENT_COLORS[i], label=f"UAV {i + 1} start") for i in range(3)],
            Line2D([0], [0], marker="*", linestyle="none", color="0.25", label="terminal goal"),
            Patch(facecolor="0.58", edgecolor="0.35", alpha=0.32, label="static structure"),
            Line2D([0], [0], color="0.20", linestyle=":", label="moving-obstacle track"),
            Line2D([0], [0], color=AGENT_COLORS[0], linestyle="--", label="4.5 m local sensing radius"),
        ],
        frameon=False,
        loc="center",
    )
    fig.subplots_adjust(wspace=0.28, hspace=0.34)
    source = SOURCE_DIR / "figure_02_operational_benchmark.csv"
    write_csv(source, source_rows)
    return register(
        2,
        "Semi-structured operational benchmark",
        "figure_02_operational_benchmark",
        source,
        "Figure 2. Deterministically selected Stage IV examples of the five balanced geometry families in the 100 m by 100 m operational workspace. Gray footprints are static structures, dotted curves are frozen moving-obstacle tracks, and the dashed circle illustrates the 4.5 m local sensing radius. Straight start-goal connectors are task descriptors, not executed trajectories.",
        fig,
    )


def _paper_figure_3_success_population(stage: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    source_rows: list[dict[str, Any]] = []
    fig, axis = plt.subplots(figsize=(7.35, 3.55))
    for index, method_id in enumerate(METHOD_ORDER):
        values = []
        for stage_name in STAGES:
            row = next(row for row in stage if row["method_id"] == method_id and row["scope"] == stage_name)
            population = STAGE_POPULATION[stage_name]
            value = 100.0 * number(row["success_rate"])
            values.append(value)
            source_rows.append(
                {
                    "method_id": method_id,
                    "method": METHOD_BY_ID[method_id]["display_name"],
                    "stage": stage_name,
                    "obstacle_population": population,
                    "static_obstacles": int(population * 0.8),
                    "dynamic_obstacles": int(population * 0.2),
                    "team_success_rate": number(row["success_rate"]),
                }
            )
        axis.plot(
            [STAGE_POPULATION[stage_name] for stage_name in STAGES],
            values,
            color=COLORS[index],
            marker=MARKERS[index],
            linestyle=LINESTYLES[index],
            linewidth=1.45,
            label=f"{method_code(method_id)} {METHOD_BY_ID[method_id]['display_name']}",
        )
    axis.set_xticks((10, 20, 30, 40))
    axis.set_xlim(8, 42)
    axis.set_ylim(0, 103)
    axis.set_xlabel("Obstacle population (static + moving)")
    axis.set_ylabel("Team success (%)")
    axis.set_title("Long-range success versus obstacle population")
    clean(axis)
    axis.legend(frameon=False, ncol=2, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    fig.subplots_adjust(right=0.68)
    source = SOURCE_DIR / "figure_03_success_vs_obstacle_population.csv"
    write_csv(source, source_rows)
    return register(
        3,
        "Success versus obstacle population",
        "figure_03_success_vs_obstacle_population",
        source,
        "Figure 3. Untouched formal team success as the obstacle population increases from 8 static plus 2 moving objects to 32 static plus 8 moving objects. Every point contains 100 matched scenarios per method.",
        fig,
    )


def _paper_figure_4_failure_population(stage: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    fields = (
        ("static_obstacle_collision_rate", "Static collision", "o", "-"),
        ("dynamic_obstacle_collision_rate", "Moving-object collision", "s", "--"),
        ("inter_agent_collision_rate", "Inter-UAV collision", "^", "-."),
        ("timeout_rate", "Timeout", "D", ":"),
    )
    proposed_rows = [row for row in stage if row["method_id"] == PROPOSED]
    source_rows: list[dict[str, Any]] = []
    fig, axis = plt.subplots(figsize=(6.7, 3.55))
    for index, (field, label, marker, linestyle) in enumerate(fields):
        values = []
        for stage_name in STAGES:
            row = next(row for row in proposed_rows if row["scope"] == stage_name)
            value = 100.0 * number(row[field])
            values.append(value)
            source_rows.append(
                {
                    "stage": stage_name,
                    "obstacle_population": STAGE_POPULATION[stage_name],
                    "failure_mode": field,
                    "rate": number(row[field]),
                }
            )
        axis.plot(
            [STAGE_POPULATION[stage_name] for stage_name in STAGES],
            values,
            color=COLORS[index],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.55,
            label=label,
        )
    axis.set_xticks((10, 20, 30, 40))
    axis.set_xlim(8, 42)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Obstacle population (static + moving)")
    axis.set_ylabel("Proposed episode rate (%)")
    axis.set_title("Proposed failure modes versus obstacle population")
    clean(axis)
    axis.legend(frameon=False, ncol=2)
    source = SOURCE_DIR / "figure_04_failure_modes_vs_population.csv"
    write_csv(source, source_rows)
    return register(
        4,
        "Failure modes versus obstacle population",
        "figure_04_failure_modes_vs_population",
        source,
        "Figure 4. Proposed-method static-obstacle, moving-obstacle, inter-UAV collision, and timeout rates across the four formal obstacle populations. Collision types were independently recomputed from stored three-dimensional trajectories.",
        fig,
    )


def _paper_figure_5_full_ablation(
    overall: Sequence[Mapping[str, str]], high: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    methods = METHOD_ORDER[2:]
    source_rows: list[dict[str, Any]] = []
    for method_id in methods:
        overall_row = next(row for row in overall if row["method_id"] == method_id)
        high_row = next(row for row in high if row["method_id"] == method_id)
        source_rows.append(
            {
                "method_id": method_id,
                "method": METHOD_BY_ID[method_id]["display_name"],
                "overall_success_rate": number(overall_row["success_rate"]),
                "stage_iii_iv_success_rate": number(high_row["success_rate"]),
                "overall_collision_rate": number(overall_row["collision_rate"]),
                "overall_timeout_rate": number(overall_row["timeout_rate"]),
            }
        )
    fig, axis = plt.subplots(figsize=(7.45, 3.6))
    x = np.arange(len(methods))
    width = 0.36
    axis.bar(
        x - width / 2,
        [100.0 * row["overall_success_rate"] for row in source_rows],
        width,
        color="#56B4E9",
        edgecolor="black",
        linewidth=0.35,
        label="Overall",
    )
    axis.bar(
        x + width / 2,
        [100.0 * row["stage_iii_iv_success_rate"] for row in source_rows],
        width,
        color="#E69F00",
        edgecolor="black",
        linewidth=0.35,
        hatch="//",
        label="Stage III+IV",
    )
    labels = ("Direct", "Proposal", "FP-SHEP", "One-shot GAT", "R-ERR+FP", "Proposed")
    axis.set_xticks(x, labels, rotation=16, ha="right")
    axis.set_ylim(0, 103)
    axis.set_ylabel("Team success (%)")
    axis.set_title("Full learned-chain ablation")
    clean(axis)
    axis.legend(frameon=False, ncol=2)
    source = SOURCE_DIR / "figure_05_full_ablation.csv"
    write_csv(source, source_rows)
    return register(
        5,
        "Full learned-chain ablation",
        "figure_05_full_ablation",
        source,
        "Figure 5. Overall and high-density Stage III+IV success for Direct SAC-DMP, Proposal, FP-SHEP, One-Shot GAT, R-ERR+FP-SHEP, and the full Proposed method. All six arms use the same adapted 256-ray SAC-DMP policy and formal scenarios.",
        fig,
    )


def _paper_figure_6_strong_baselines(
    overall: Sequence[Mapping[str, str]], high: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    ordered = (
        "M2_DWA_SensingMatched",
        "M3_Waypoint_PPO",
        PROPOSED,
        "M1_DWA_FullState",
    )
    source_rows: list[dict[str, Any]] = []
    for method_id in ordered:
        if method_id == "M3_Waypoint_PPO":
            source_rows.append(
                {
                    "method_id": method_id,
                    "method": "Waypoint-PPO",
                    "evaluation_status": "NOT_EVALUATED_READINESS_GATE_FAILED",
                    "overall_success_rate": None,
                    "stage_iii_iv_success_rate": None,
                }
            )
            continue
        overall_row = next(row for row in overall if row["method_id"] == method_id)
        high_row = next(row for row in high if row["method_id"] == method_id)
        source_rows.append(
            {
                "method_id": method_id,
                "method": METHOD_BY_ID[method_id]["display_name"],
                "evaluation_status": "EVALUATED_UNTOUCHED_FORMAL",
                "overall_success_rate": number(overall_row["success_rate"]),
                "stage_iii_iv_success_rate": number(high_row["success_rate"]),
            }
        )
    fig, axis = plt.subplots(figsize=(6.8, 3.55))
    x = np.arange(len(ordered))
    width = 0.36
    overall_values = [
        np.nan if row["overall_success_rate"] is None else 100.0 * float(row["overall_success_rate"])
        for row in source_rows
    ]
    high_values = [
        np.nan if row["stage_iii_iv_success_rate"] is None else 100.0 * float(row["stage_iii_iv_success_rate"])
        for row in source_rows
    ]
    axis.bar(
        x - width / 2,
        overall_values,
        width,
        color="#56B4E9",
        edgecolor="black",
        linewidth=0.35,
        label="Overall",
    )
    axis.bar(
        x + width / 2,
        high_values,
        width,
        color="#E69F00",
        edgecolor="black",
        linewidth=0.35,
        hatch="//",
        label="Stage III+IV",
    )
    axis.add_patch(
        Rectangle(
            (x[1] - 0.39, 0.0),
            0.78,
            8.0,
            facecolor="none",
            edgecolor="0.35",
            linewidth=0.65,
            hatch="xx",
        )
    )
    axis.text(x[1], 10.0, "not evaluated", ha="center", va="bottom", fontsize=7.5)
    axis.set_xticks(x, ("DWA-SM", "Waypoint-PPO", "Proposed", "DWA-FS"))
    axis.set_ylim(0, 103)
    axis.set_ylabel("Team success (%)")
    axis.set_title("Strong-baseline comparison")
    clean(axis)
    axis.legend(frameon=False, ncol=2)
    source = SOURCE_DIR / "figure_06_strong_baselines.csv"
    write_csv(source, source_rows)
    return register(
        6,
        "Strong-baseline comparison",
        "figure_06_strong_baselines",
        source,
        "Figure 6. Equal-information DWA, Proposed, and FullState DWA performance on the untouched formal benchmark. Waypoint-PPO is shown only as a preregistered empty position because its preformal readiness gate failed; no PPO performance is imputed or fabricated.",
        fig,
    )


def _paper_figure_7_long_range_scaling(
    runtime: Sequence[Mapping[str, str]], density: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    source_rows: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        timing = next(
            row for row in runtime
            if row["method_id"] == method_id and row["scope"] == "overall"
        )
        planning = next(
            row for row in density
            if row["method_id"] == method_id and row["scope"] == "overall"
        )
        source_rows.append(
            {
                "method_id": method_id,
                "method": METHOD_BY_ID[method_id]["display_name"],
                "compute_mean_ms_per_100m": number(timing["compute_mean_ms_per_100m"]),
                "planning_decisions_mean_per_100m": number(planning["planning_decisions_mean_per_100m"]),
                "replanning_mean_per_100m": number(planning["replanning_mean_per_100m"]),
            }
        )
    fig, axes = plt.subplots(1, 3, figsize=(9.15, 3.45))
    x = np.arange(len(METHOD_ORDER))
    panels = (
        ("compute_mean_ms_per_100m", "Online compute (ms/100 m)", True),
        ("planning_decisions_mean_per_100m", "Planner decisions / 100 m", True),
        ("replanning_mean_per_100m", "Reproposals / 100 m", False),
    )
    for axis, (field, ylabel, log_scale) in zip(axes, panels):
        values = np.asarray([float(row[field]) for row in source_rows], dtype=float)
        axis.bar(x, values, color=COLORS, edgecolor="black", linewidth=0.3)
        axis.set_xticks(x, [method_code(method_id) for method_id in METHOD_ORDER], rotation=45, ha="right")
        axis.set_ylabel(ylabel)
        if log_scale and np.all(values > 0):
            axis.set_yscale("log")
        axis.set_ylim(bottom=0.08 if log_scale and np.all(values > 0) else 0.0)
        clean(axis)
    axes[0].set_title("Compute scaling")
    axes[1].set_title("Decision density")
    axes[2].set_title("Reconstruction density")
    fig.subplots_adjust(wspace=0.38, bottom=0.24)
    source = SOURCE_DIR / "figure_07_long_range_scaling.csv"
    write_csv(source, source_rows)
    return register(
        7,
        "Long-range compute and reconstruction scaling",
        "figure_07_long_range_scaling",
        source,
        "Figure 7. Total online algorithm compute, planner-decision density, and R-ERR reproposal density normalized by the frozen mean straight-line mission distance. Values are computation measurements rather than physical execution time.",
        fig,
    )


def _paper_figure_8_paired_path_motion(
    paired: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    metrics = (
        ("team_path_length_m", "Team path difference (m)"),
        ("completion_time_s", "Completion-time difference (s)"),
        ("team_path_efficiency", "Path-efficiency difference"),
        ("trajectory_smoothness", "Smoothness difference"),
    )
    rows = [row for row in paired if row["metric"] in {metric for metric, _ in metrics}]
    comparison_order = ("P1", "P2", "P3", "P4", "P5")
    baseline_label = {
        "P1": "DWA-SM",
        "P2": "Direct",
        "P3": "One-shot GAT",
        "P4": "R-ERR+FP",
        "P5": "DWA-FS",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6))
    for axis, (metric, xlabel) in zip(axes.flat, metrics):
        metric_rows = {row["comparison_id"]: row for row in rows if row["metric"] == metric}
        available = [comparison for comparison in comparison_order if comparison in metric_rows]
        y = np.arange(len(available))
        means = np.asarray(
            [number(metric_rows[comparison]["mean_proposed_minus_baseline"]) for comparison in available],
            dtype=float,
        )
        lower = np.asarray(
            [number(metric_rows[comparison]["paired_bootstrap_ci95_lower"]) for comparison in available],
            dtype=float,
        )
        upper = np.asarray(
            [number(metric_rows[comparison]["paired_bootstrap_ci95_upper"]) for comparison in available],
            dtype=float,
        )
        error = np.vstack((means - lower, upper - means))
        axis.errorbar(
            means,
            y,
            xerr=error,
            fmt="o",
            color="#0072B2",
            ecolor="0.35",
            elinewidth=0.9,
            capsize=2.5,
            markersize=4,
        )
        axis.axvline(0.0, color="0.2", linewidth=0.65, linestyle="--")
        axis.set_yticks(y, [baseline_label[comparison] for comparison in available])
        axis.set_xlabel(xlabel)
        axis.set_title(metric.replace("_", " "))
        clean(axis)
    fig.subplots_adjust(wspace=0.43, hspace=0.43)
    source = SOURCE_DIR / "figure_08_paired_path_motion.csv"
    write_csv(source, rows)
    return register(
        8,
        "Paired path and motion quality",
        "figure_08_paired_path_motion",
        source,
        "Figure 8. Mean Proposed-minus-baseline differences with 95% paired bootstrap intervals on the both-success subset only. Failed completion times are never filled with zero. Lower path length, completion time, and smoothness differences are favorable; higher path efficiency is favorable.",
        fig,
    )


def _joint_success_anchors(
    formal_rows: Sequence[Mapping[str, str]],
    methods: Sequence[str] = FIGURE9_METHODS,
) -> list[dict[str, Any]]:
    rows_by_key = {
        (str(row["scenario_id"]), str(row["method_id"])): row for row in formal_rows
    }
    anchors: list[dict[str, Any]] = []
    for stage_name in STAGES:
        scenario_ids = sorted(
            {
                str(row["scenario_id"])
                for row in formal_rows
                if row["stage"] == stage_name
            }
        )
        joint = [
            scenario_id for scenario_id in scenario_ids
            if all(
                (scenario_id, method_id) in rows_by_key
                and truth(rows_by_key[(scenario_id, method_id)]["team_success"])
                for method_id in methods
            )
        ]
        if not joint:
            raise RuntimeError(
                f"Figure 9 requires at least one jointly successful scene in {stage_name} "
                f"for {tuple(methods)}"
            )
        proposed_times = np.asarray(
            [number(rows_by_key[(scenario_id, PROPOSED)]["completion_time_s"]) for scenario_id in joint],
            dtype=float,
        )
        target = float(np.median(proposed_times))
        scenario_id = min(
            joint,
            key=lambda candidate: (
                abs(number(rows_by_key[(candidate, PROPOSED)]["completion_time_s"]) - target),
                candidate,
            ),
        )
        anchors.append(
            {
                "stage": stage_name,
                "scenario_id": scenario_id,
                "joint_success_method_ids": list(methods),
                "joint_success_candidate_count": len(joint),
                "selection_rule": "joint success for all plotted methods; Proposed completion time closest to joint-success stage median; scenario-id tie break",
                "post_hoc_visual_outcome_cherry_pick": False,
            }
        )
    return anchors


def _critical_peer_window(positions: np.ndarray, radius: float = 6.0) -> dict[str, Any]:
    best = (float("inf"), 0, 0, 1)
    for left in range(3):
        for right in range(left + 1, 3):
            distances = np.linalg.norm(positions[:, left] - positions[:, right], axis=1)
            step = int(np.argmin(distances))
            candidate = (float(distances[step]), step, left, right)
            if candidate < best:
                best = candidate
    distance, step, left, right = best
    center = 0.5 * (positions[step, left, :2] + positions[step, right, :2])
    return {
        "minimum_peer_distance_m": distance,
        "step": step,
        "agent_left": left,
        "agent_right": right,
        "x_min": max(0.0, float(center[0]) - radius),
        "x_max": min(100.0, float(center[0]) + radius),
        "y_min": max(0.0, float(center[1]) - radius),
        "y_max": min(100.0, float(center[1]) + radius),
    }


def _paper_figure_9_matched_3d(
    manifest: Mapping[str, Any], formal_rows: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    entries = {str(row["scenario_id"]): row for row in manifest["entries"]}
    anchors = _joint_success_anchors(formal_rows)
    selection_source = SOURCE_DIR / "figure_09_joint_success_selection.csv"
    write_csv(selection_source, anchors)
    source_rows: list[dict[str, Any]] = []
    method_styles = {method_id: LINESTYLES[index] for index, method_id in enumerate(FIGURE9_METHODS)}
    fig = plt.figure(figsize=(9.1, 7.45))
    for panel, anchor in enumerate(anchors, start=1):
        stage_name = str(anchor["stage"])
        scenario_id = str(anchor["scenario_id"])
        entry = entries[scenario_id]
        axis = fig.add_subplot(2, 2, panel, projection="3d")
        for spec in entry["static_obstacles"]:
            _plot_static(axis, spec)
        for track in entry["dynamic_obstacle_trajectories"]:
            points = np.asarray(track, dtype=float)
            stride = max(1, len(points) // 80)
            sampled = points[::stride]
            axis.plot(
                sampled[:, 0], sampled[:, 1], sampled[:, 2],
                color="0.30", linestyle=":", linewidth=0.55, alpha=0.65,
            )
        trajectories: dict[str, np.ndarray] = {}
        for method_id in FIGURE9_METHODS:
            positions = _load_formal_positions(method_id, scenario_id)
            trajectories[method_id] = positions
            stride = max(1, len(positions) // 180)
            for agent_id in range(3):
                path = positions[:, agent_id]
                axis.plot(
                    path[:, 0], path[:, 1], path[:, 2],
                    color=AGENT_COLORS[agent_id],
                    linestyle=method_styles[method_id],
                    linewidth=0.95,
                    alpha=0.82,
                )
                for step in range(0, len(path), stride):
                    source_rows.append(
                        {
                            "stage": stage_name,
                            "scenario_id": scenario_id,
                            "joint_success_all_plotted_methods": True,
                            "method_id": method_id,
                            "agent_id": agent_id,
                            "step": step,
                            "x_m": path[step, 0],
                            "y_m": path[step, 1],
                            "z_m": path[step, 2],
                        }
                    )
        starts = np.asarray(entry["starts"], dtype=float)
        goals = np.asarray(entry["goals"], dtype=float)
        for agent_id, color in enumerate(AGENT_COLORS):
            axis.scatter(*starts[agent_id], color=color, marker="^", s=18, depthshade=False)
            axis.scatter(*goals[agent_id], color=color, marker="*", s=30, depthshade=False)
        axis.set_xlim(0.0, 100.0)
        axis.set_ylim(0.0, 100.0)
        axis.set_zlim(0.0, 4.0)
        axis.set_xlabel("x (m)", labelpad=1)
        axis.set_ylabel("y (m)", labelpad=1)
        axis.set_zlabel("z (m)", labelpad=0)
        axis.set_title(stage_name, pad=1)
        axis.view_init(elev=27, azim=-57)
        axis.grid(True, linewidth=0.25, color="0.90")

        critical = _critical_peer_window(trajectories[PROPOSED])
        anchor.update({f"critical_{key}": value for key, value in critical.items()})
        inset = axis.inset_axes([0.58, 0.055, 0.37, 0.34])
        for spec in entry["static_obstacles"]:
            center = np.asarray(spec["center"], dtype=float)
            if (
                critical["x_min"] - 2.5 <= center[0] <= critical["x_max"] + 2.5
                and critical["y_min"] - 2.5 <= center[1] <= critical["y_max"] + 2.5
            ):
                _plot_static_topdown(inset, spec, alpha=0.22)
        for method_id, positions in trajectories.items():
            for agent_id in range(3):
                path = positions[:, agent_id]
                inside = (
                    (path[:, 0] >= critical["x_min"])
                    & (path[:, 0] <= critical["x_max"])
                    & (path[:, 1] >= critical["y_min"])
                    & (path[:, 1] <= critical["y_max"])
                )
                x_values = np.where(inside, path[:, 0], np.nan)
                y_values = np.where(inside, path[:, 1], np.nan)
                inset.plot(
                    x_values,
                    y_values,
                    color=AGENT_COLORS[agent_id],
                    linestyle=method_styles[method_id],
                    linewidth=0.8,
                    alpha=0.86,
                )
        proposed = trajectories[PROPOSED]
        critical_step = int(critical["step"])
        for agent_id in (int(critical["agent_left"]), int(critical["agent_right"])):
            inset.scatter(
                proposed[critical_step, agent_id, 0],
                proposed[critical_step, agent_id, 1],
                color=AGENT_COLORS[agent_id],
                marker="o",
                s=17,
                edgecolor="black",
                linewidth=0.3,
                zorder=5,
            )
        inset.set_xlim(float(critical["x_min"]), float(critical["x_max"]))
        inset.set_ylim(float(critical["y_min"]), float(critical["y_max"]))
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color("0.25")
    write_csv(selection_source, anchors)
    agent_handles = [
        Line2D([0], [0], color=AGENT_COLORS[index], linestyle="-", label=f"UAV {index + 1}")
        for index in range(3)
    ]
    method_handles = [
        Line2D(
            [0], [0], color="0.20", linestyle=method_styles[method_id],
            label=f"{method_code(method_id)} {METHOD_BY_ID[method_id]['display_name'].replace(' + SAC-DMP', '')}",
        )
        for method_id in FIGURE9_METHODS
    ]
    fig.legend(
        handles=[*agent_handles, *method_handles],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
    )
    fig.subplots_adjust(top=0.90, wspace=0.02, hspace=0.10)
    source = SOURCE_DIR / "figure_09_matched_joint_success_3d.csv"
    write_csv(source, source_rows)
    return register(
        9,
        "Matched jointly successful 3-D trajectories",
        "figure_09_matched_joint_success_3d",
        source,
        "Figure 9. Matched three-dimensional trajectories on one jointly successful DWA-FullState, DWA-SensingMatched, R-ERR+FP-SHEP, and Proposed scenario per stage. The deterministic anchor is the Proposed completion time closest to the median among all jointly successful scenes in that stage. UAV identity is color-coded and method identity uses line style. Insets show the local neighborhood around the minimum Proposed inter-UAV distance. Panel titles contain only Stage I through Stage IV; all axes start at zero.",
        fig,
    )


def _event_subset(events: Sequence[Mapping[str, Any]], maximum: int = 30) -> set[int]:
    if len(events) <= maximum:
        return set(range(len(events)))
    indices = np.linspace(0, len(events) - 1, maximum).round().astype(int)
    return set(int(value) for value in indices)


def _paper_figure_10_reference_chain(
    manifest: Mapping[str, Any], formal_rows: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    anchors = _joint_success_anchors(formal_rows)
    anchor = next(row for row in anchors if row["stage"] == "Stage IV")
    scenario_id = str(anchor["scenario_id"])
    entry = next(row for row in manifest["entries"] if row["scenario_id"] == scenario_id)
    payload = load_json(RECORD_DIR / PROPOSED / f"{scenario_id}.json")
    positions = _load_formal_positions(PROPOSED, scenario_id)
    events = list(payload.get("events", []))
    source_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.45), sharex=True, sharey=True)
    for agent_id, axis in enumerate(axes):
        for spec in entry["static_obstacles"]:
            _plot_static_topdown(axis, spec, alpha=0.20)
        path = positions[:, agent_id]
        axis.plot(path[:, 0], path[:, 1], color=AGENT_COLORS[agent_id], linewidth=1.45)
        axis.scatter(path[0, 0], path[0, 1], marker="^", color=AGENT_COLORS[agent_id], s=24, zorder=5)
        terminal = np.asarray(entry["goals"], dtype=float)[agent_id]
        axis.scatter(terminal[0], terminal[1], marker="*", color=AGENT_COLORS[agent_id], s=55, zorder=5)
        agent_events = [
            event for event in events
            if int(event.get("agent_id", -1)) == agent_id
            and event.get("new_active_goal") is not None
            and (truth(event.get("goal_changed", True)) or str(event.get("event")) == "INITIAL_SELECTION")
        ]
        plotted = _event_subset(agent_events)
        for event_index, event in enumerate(agent_events):
            step = min(max(int(event.get("step", 0)), 0), len(path) - 1)
            vehicle = path[step]
            reference = np.asarray(event["new_active_goal"], dtype=float)
            is_plotted = event_index in plotted
            source_rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage": "Stage IV",
                    "agent_id": agent_id,
                    "event_index": event_index,
                    "step": step,
                    "time_s": event.get("time_s"),
                    "event": event.get("event"),
                    "counts_as_reproposal": truth(event.get("counts_as_reproposal", False)),
                    "vehicle_x_m": vehicle[0],
                    "vehicle_y_m": vehicle[1],
                    "vehicle_z_m": vehicle[2],
                    "reference_x_m": reference[0],
                    "reference_y_m": reference[1],
                    "reference_z_m": reference[2],
                    "plotted_after_deterministic_decimation": is_plotted,
                }
            )
            if not is_plotted:
                continue
            axis.plot(
                [vehicle[0], reference[0]],
                [vehicle[1], reference[1]],
                color=AGENT_COLORS[agent_id],
                linestyle=":",
                linewidth=0.42,
                alpha=0.48,
            )
            axis.scatter(
                reference[0], reference[1],
                marker="o", facecolor="none", edgecolor=AGENT_COLORS[agent_id],
                linewidth=0.55, s=15, zorder=4,
            )
            if truth(event.get("counts_as_reproposal", False)):
                axis.scatter(vehicle[0], vehicle[1], marker="x", color="0.15", linewidth=0.65, s=14, zorder=4)
        for fraction, label in ((0.0, "0%"), (0.5, "50%"), (1.0, "100%")):
            index = min(int(round(fraction * (len(path) - 1))), len(path) - 1)
            axis.annotate(label, (path[index, 0], path[index, 1]), xytext=(2, 2), textcoords="offset points", fontsize=7)
        axis.set_xlim(0.0, 100.0)
        axis.set_ylim(0.0, 100.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x (m)")
        if agent_id == 0:
            axis.set_ylabel("y (m)")
        axis.set_title(f"UAV {agent_id + 1}")
        axis.grid(True, color="0.92", linewidth=0.35)
    fig.legend(
        handles=[
            Line2D([0], [0], color="0.25", marker="^", linestyle="none", label="start"),
            Line2D([0], [0], color="0.25", marker="*", linestyle="none", label="terminal goal"),
            Line2D([0], [0], color="0.25", marker="o", markerfacecolor="none", linestyle=":", label="active local reference"),
            Line2D([0], [0], color="0.15", marker="x", linestyle="none", label="R-ERR event position"),
        ],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
    )
    fig.subplots_adjust(top=0.84, wspace=0.18)
    source = SOURCE_DIR / "figure_10_reference_reconstruction_chain.csv"
    write_csv(source, source_rows)
    return register(
        10,
        "Reference reconstruction chain",
        "figure_10_reference_reconstruction_chain",
        source,
        "Figure 10. Proposed terminal goals, executed mission paths, active local references, and R-ERR event positions on the jointly successful Stage IV anchor. Percentage labels show executed mission progress. If an agent has more than 30 goal-changing events, display markers are deterministically decimated while every event remains in the source table.",
        fig,
    )


def _paper_figure_11_stage_family(
    family: Sequence[Mapping[str, str]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    rows = [row for row in family if row["method_id"] == PROPOSED]
    families = list(manifest.get("family_order", []))
    if not families:
        families = sorted({str(row["family"]) for row in rows})
    matrix = np.asarray(
        [
            [
                100.0 * number(
                    next(
                        row for row in rows
                        if row["stage"] == stage_name and row["family"] == family_name
                    )["success_rate"]
                )
                for family_name in families
            ]
            for stage_name in STAGES
        ],
        dtype=float,
    )
    source = SOURCE_DIR / "figure_11_stage_family_heatmap.csv"
    write_csv(source, rows)
    fig, axis = plt.subplots(figsize=(8.0, 3.55))
    image = axis.imshow(matrix, vmin=0.0, vmax=100.0, cmap="viridis", aspect="auto")
    family_labels = []
    for family_name in families:
        entry = next(row for row in manifest["entries"] if row["family"] == family_name)
        family_labels.append(str(entry.get("family_label", family_name)).replace("_", " "))
    axis.set_xticks(np.arange(len(families)), family_labels, rotation=18, ha="right")
    axis.set_yticks(np.arange(len(STAGES)), STAGES)
    for row_index in range(len(STAGES)):
        for column_index in range(len(families)):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.0f}%",
                ha="center",
                va="center",
                color="white" if value < 55.0 else "black",
                fontsize=7.5,
            )
    fig.colorbar(image, ax=axis, label="Proposed team success (%)", fraction=0.035, pad=0.02)
    axis.set_title("Proposed performance across stage and geometry family")
    fig.subplots_adjust(bottom=0.27)
    return register(
        11,
        "Stage-family success heatmap",
        "figure_11_stage_family_heatmap",
        source,
        "Figure 11. Descriptive Proposed team success for the balanced five-family by four-stage grid. Each cell contains 20 untouched formal scenarios; family-level cells are descriptive and are not separate confirmatory tests.",
        fig,
    )


def _paper_tables(
    overall: Sequence[Mapping[str, str]],
    stage: Sequence[Mapping[str, str]],
    high: Sequence[Mapping[str, str]],
    runtime: Sequence[Mapping[str, str]],
    paired: Sequence[Mapping[str, str]],
    information: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    main_rows: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        summary = next(row for row in overall if row["method_id"] == method_id)
        complex_row = next(row for row in high if row["method_id"] == method_id)
        timing = next(
            row for row in runtime
            if row["method_id"] == method_id and row["scope"] == "overall"
        )
        stage_rates = {
            stage_name: number(
                next(
                    row for row in stage
                    if row["method_id"] == method_id and row["scope"] == stage_name
                )["success_rate"]
            )
            for stage_name in STAGES
        }
        main_rows.append(
            {
                "method_id": method_id,
                "method": METHOD_BY_ID[method_id]["display_name"],
                "overall_success_rate": number(summary["success_rate"]),
                "stage_i_success_rate": stage_rates["Stage I"],
                "stage_ii_success_rate": stage_rates["Stage II"],
                "stage_iii_success_rate": stage_rates["Stage III"],
                "stage_iv_success_rate": stage_rates["Stage IV"],
                "stage_iii_iv_success_rate": number(complex_row["success_rate"]),
                "any_collision_rate": number(summary["collision_rate"]),
                "static_obstacle_collision_rate": number(summary["static_obstacle_collision_rate"]),
                "dynamic_obstacle_collision_rate": number(summary["dynamic_obstacle_collision_rate"]),
                "inter_agent_collision_rate": number(summary["inter_agent_collision_rate"]),
                "timeout_rate": number(summary["timeout_rate"]),
                "agent_completion_rate": number(summary["agent_completion_rate"]),
                "compute_mean_ms_per_100m": number(timing["compute_mean_ms_per_100m"]),
            }
        )
    table1 = TABLE_DIR / "table_01_main_performance.csv"
    write_csv(table1, main_rows)

    ablation_rows = [row for row in main_rows if row["method_id"] in METHOD_ORDER[2:]]
    table2 = TABLE_DIR / "table_02_full_ablation.csv"
    write_csv(table2, ablation_rows)

    table3 = TABLE_DIR / "table_03_paired_path_motion_quality.csv"
    write_csv(table3, paired)

    table4 = TABLE_DIR / "table_04_information_contract.csv"
    requested_information_methods = {
        "M1_DWA_FullState",
        "M2_DWA_SensingMatched",
        "M3_Waypoint_PPO",
        "M4_Direct_SAC_DMP",
        PROPOSED,
    }
    write_csv(
        table4,
        [row for row in information if row["method_id"] in requested_information_methods],
    )
    definitions = (
        ("Main performance", table1),
        ("Full learned-chain ablation", table2),
        ("Both-success paired path and motion quality", table3),
        ("Information and planning-frequency contract", table4),
    )
    return [
        {
            "table": index,
            "title": title,
            "source": str(path.relative_to(ARTIFACT_ROOT)).replace("\\", "/"),
            "status": "YES",
        }
        for index, (title, path) in enumerate(definitions, start=1)
    ]


def plot_all() -> list[dict[str, Any]]:
    conclusion = load_json(ARTIFACT_ROOT / "conclusion.json")
    if conclusion.get("FINAL_RECONCILIATION") != "PASS":
        raise RuntimeError("plotting is forbidden until final reconciliation passes")
    short_reconciliation = load_json(SHORT_DIR / "short_coordination_reconciliation.json")
    if short_reconciliation.get("status") != "PASS":
        raise RuntimeError("Figure 1 requires a reconciled 100-scene short coordination test")
    for directory in (PDF_DIR, PNG_DIR, SOURCE_DIR, CAPTION_DIR, SCRIPT_DIR, TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    setup()
    overall = read_csv(STATISTICS_DIR / "overall_summary.csv")
    stage = read_csv(STATISTICS_DIR / "stage_summary.csv")
    high = read_csv(STATISTICS_DIR / "high_density_summary.csv")
    family = read_csv(STATISTICS_DIR / "family_summary.csv")
    runtime = read_csv(RUNTIME_DIR / "runtime_summary.csv")
    density = read_csv(RUNTIME_DIR / "replanning_density.csv")
    paired = read_csv(STATISTICS_DIR / "paired_path_quality_summary.csv")
    information = read_csv(STATISTICS_DIR / "information_contract.csv")
    formal_rows = read_csv(RECORD_DIR / "formal_team_results.csv")
    manifest = load_json(FORMAL_MANIFEST)
    figures = [
        _paper_figure_1_short_coordination(),
        _paper_figure_2_benchmark(manifest),
        _paper_figure_3_success_population(stage),
        _paper_figure_4_failure_population(stage),
        _paper_figure_5_full_ablation(overall, high),
        _paper_figure_6_strong_baselines(overall, high),
        _paper_figure_7_long_range_scaling(runtime, density),
        _paper_figure_8_paired_path_motion(paired),
        _paper_figure_9_matched_3d(manifest, formal_rows),
        _paper_figure_10_reference_chain(manifest, formal_rows),
        _paper_figure_11_stage_family(family, manifest),
    ]
    if [int(row["figure"]) for row in figures] != list(range(1, 12)):
        raise RuntimeError("paper figure numbering is incomplete")
    write_csv(PAPER_DIR / "figure_manifest.csv", figures)
    tables = _paper_tables(overall, stage, high, runtime, paired, information)
    write_csv(PAPER_DIR / "table_manifest.csv", tables)
    shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)
    conclusion["FINAL_PAPER_FIGURES_READY"] = "YES"
    conclusion["FINAL_PAPER_TABLES_READY"] = "YES"
    conclusion["PAPER_FIGURE_COUNT"] = 11
    conclusion["PAPER_TABLE_COUNT"] = 4
    conclusion["REPRESENTATIVE_TRAJECTORY_SELECTION"] = "JOINT_SUCCESS_DETERMINISTIC_STAGE_MEDIAN"
    conclusion["PPO_PERFORMANCE_FABRICATED"] = "NO"
    conclusion["RECOMMENDED_NEXT_STEP"] = "WRITE_PAPER_RESULTS"
    (ARTIFACT_ROOT / "conclusion.json").write_text(
        json.dumps(conclusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return figures


def main() -> None:
    parser=argparse.ArgumentParser(); parser.parse_args(); figures=plot_all(); print(json.dumps({"plotting":"PASS","figure_count":len(figures)}),flush=True)


if __name__=="__main__":
    main()
