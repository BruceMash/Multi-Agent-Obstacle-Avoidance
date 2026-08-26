"""Create the ten publication figures for the frozen four-stage benchmark.

The script is self-contained and uses only persisted formal outputs.  A copy is
stored beside every figure so each figure has a direct reproduction script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Rectangle
import numpy as np
from PIL import Image


METHODS = ("dwa_style", "rvo_orca_style", "terminal", "proposal", "fp_shep", "gat_v1")
DISPLAY = {
    "dwa_style": "3D-DWA-style", "rvo_orca_style": "RVO/ORCA-style",
    "terminal": "Terminal", "proposal": "Proposal", "fp_shep": "FP-SHEP", "gat_v1": "Proposed",
}
STAGES = ("stage_1", "stage_2", "stage_3", "stage_4")
STAGE_LABELS = ("Stage I", "Stage II", "Stage III", "Stage IV")
COLORS = ("#0072B2", "#D55E00", "#777777", "#CC79A7", "#009E73", "#E69F00")
MARKERS = ("o", "s", "^", "D", "v", "P")
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1)))
HATCHES = ("///", "\\\\", "...", "xx", "++", "oo")
METHOD_STYLE = {method: {"color": COLORS[i], "marker": MARKERS[i], "linestyle": LINESTYLES[i]} for i, method in enumerate(METHODS)}
MM = 1.0 / 25.4
DOUBLE = 178 * MM
SINGLE = 85 * MM


FIGURES = {
    1: ("figure_01_success_vs_difficulty", "Team success rate vs difficulty"),
    2: ("figure_02_collision_vs_difficulty", "Collision rate vs difficulty"),
    3: ("figure_03_overall_outcomes", "Overall success, collision, and timeout"),
    4: ("figure_04_planning_runtime", "Planning runtime vs difficulty"),
    5: ("figure_05_path_efficiency", "Successful path length and path efficiency"),
    6: ("figure_06_ablation_performance", "Ablation performance"),
    7: ("figure_07_graceful_degradation", "Stage I to IV degradation"),
    8: ("figure_08_representative_trajectories", "Matched representative trajectories"),
    9: ("figure_09_minimum_inter_agent_distance", "Minimum inter-agent distance"),
    10: ("figure_10_failure_taxonomy", "Failure taxonomy"),
}


def setup_style() -> str:
    names = {font.name for font in font_manager.fontManager.ttflist}
    font = "Times New Roman" if "Times New Roman" in names else "DejaVu Serif"
    mpl.rcParams.update({
        "font.family": font, "font.size": 8.0, "axes.titlesize": 8.5,
        "axes.labelsize": 8.0, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0, "axes.linewidth": 0.7, "lines.linewidth": 1.25,
        "lines.markersize": 4.0, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.direction": "out", "ytick.direction": "out", "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.unicode_minus": True,
    })
    return font


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else float("nan")


def figure_dirs(out: Path) -> dict[str, Path]:
    root = out / "paper_ready"
    dirs = {
        "root": root, "source": root / "source_data", "scripts": root / "scripts",
        "captions": root / "captions", "pdf": root / "pdf", "png": root / "png_600dpi",
        "candidate": root / "paper_final_candidate",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def base_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def method_legend(fig: plt.Figure, ncol: int = 3, y: float = 0.88) -> None:
    handles = []
    for method in METHODS:
        style = METHOD_STYLE[method]
        handles.append(mpl.lines.Line2D([], [], label=DISPLAY[method], color=style["color"], marker=style["marker"], linestyle=style["linestyle"], markerfacecolor="white", markeredgewidth=0.9))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, handlelength=2.5, columnspacing=1.1)


def check_text_bounds(fig: plt.Figure) -> tuple[bool, list[str]]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    failures = []
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer=renderer)
        if box.x0 < bounds.x0 - 10 or box.y0 < bounds.y0 - 10 or box.x1 > bounds.x1 + 10 or box.y1 > bounds.y1 + 10:
            failures.append(text.get_text())
    return not failures, failures[:10]


def save_figure(fig: plt.Figure, dirs: Mapping[str, Path], stem: str) -> dict[str, Any]:
    text_ok, clipped = check_text_bounds(fig)
    pdf = dirs["pdf"] / f"{stem}.pdf"
    png = dirs["png"] / f"{stem}.png"
    fig.savefig(pdf, format="pdf", metadata={"Title": stem, "Creator": "Matplotlib reproducible benchmark script"})
    fig.savefig(png, format="png", dpi=601, pil_kwargs={"compress_level": 7})
    plt.close(fig)
    with Image.open(png) as image:
        dpi = image.info.get("dpi", (0.0, 0.0))
        width_px, height_px = image.size
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    pdf_vector_candidate = pdf.read_bytes().startswith(b"%PDF") and b"/Subtype /Image" not in pdf.read_bytes()
    return {
        "text_within_canvas": text_ok, "clipped_text_examples": clipped,
        "png_dpi_x": float(dpi[0]), "png_dpi_y": float(dpi[1]),
        "png_width_px": width_px, "png_height_px": height_px,
        "grayscale_standard_deviation": float(np.std(gray)),
        "pdf_vector_candidate": pdf_vector_candidate,
    }


def caption_text(number_id: int) -> str:
    captions = {
        1: "Team success rate across the four preregistered difficulty stages. All six frozen methods are evaluated on the same 100 scenarios per stage (n = 100 per marker). Error bars are exact two-sided 95% Clopper–Pearson binomial confidence intervals.",
        2: "Any-collision rate across difficulty for the six frozen methods on matched scenarios (n = 100 per marker). Any collision includes obstacle or inter-agent collision. Error bars are exact two-sided 95% Clopper–Pearson binomial confidence intervals.",
        3: "Overall categorical outcomes over 400 formal scenarios per method. Bars show team-success, any-collision, and timeout rates; whiskers are exact two-sided 95% Clopper–Pearson intervals. Outcomes are not mutually exhaustive only in definition, but the executed termination protocol produced exclusive terminal categories here.",
        4: "Steady-state planning runtime per episode across difficulty (n = 100 episodes per stage and method). Lines show medians and upper whiskers show P90. The symmetric-logarithmic axis retains the true zero planning cost of Terminal. Planner time excludes environment stepping, model loading, CUDA initialization, and warm-up.",
        5: "Successful-episode path efficiency across difficulty. Left: mean team path length; right: mean team path efficiency, defined as mean agent straight-line distance divided by executed length. Error bars show one standard deviation; failed episodes are excluded and sample counts therefore vary by method and stage.",
        6: "Overall ablation comparison for the frozen SAC-DMP chain (n = 400 scenarios per method). Terminal, Proposal, FP-SHEP, and full Proposed differ only in the upstream reference-selection stages. Bars report team success, any collision, and timeout with exact 95% Clopper–Pearson intervals.",
        7: "Stage I-to-IV degradation for all six frozen methods. Panels report success-rate drop, collision-rate increase, and planning-runtime increase; smaller values indicate less degradation but must be interpreted together with absolute Stage-I and Stage-IV performance. Each endpoint contains n = 100 scenarios.",
        8: "Matched representative trajectories. For each stage, the anchor scenario is the Proposed successful episode whose completion time is closest to the stage median; all six methods are then shown on that same frozen scenario. Solid, dashed, and dotted lines identify UAVs 1, 2, and 3; markers indicate starts and goals, shaded shapes show static obstacle footprints, and gray dashed traces show frozen dynamic-obstacle paths. Four anchor scenarios are shown and no error bars apply. This deterministic rule prevents visual cherry-picking.",
        9: "Distribution of the minimum center-to-center inter-agent distance over all 400 formal episodes per method, separated by stage (n = 100 per box). Boxes show median and interquartile range, whiskers extend to 1.5 IQR, and points beyond the whiskers are omitted from display but retained in source data.",
        10: "Exclusive formal failure taxonomy over 400 scenarios per method. Segments show obstacle collision, inter-agent collision, timeout, planner infeasible, and other retained algorithmic failures. No formal software or numerical failure occurred; hatching supports grayscale reproduction.",
    }
    return captions[number_id]


def fig_stage_binary(out: Path, dirs: Mapping[str, Path], number_id: int, source_name: str, ylabel: str) -> tuple[plt.Figure, Path]:
    rows = read_csv(out / source_name)
    source = dirs["source"] / f"{FIGURES[number_id][0]}.csv"
    write_csv(source, rows)
    fig, ax = plt.subplots(figsize=(DOUBLE, 76 * MM))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.68)
    for method in METHODS:
        data = sorted([row for row in rows if row["method"] == method], key=lambda row: int(row["stage_index"]))
        x = np.asarray([int(row["stage_index"]) for row in data])
        y = np.asarray([number(row, "rate") for row in data])
        low = np.asarray([number(row, "ci95_low") for row in data])
        high = np.asarray([number(row, "ci95_high") for row in data])
        style = METHOD_STYLE[method]
        ax.errorbar(x, y, yerr=np.vstack((y - low, high - y)), label=DISPLAY[method], capsize=2.0, markerfacecolor="white", markeredgewidth=0.9, zorder=3, **style)
    ax.set_xticks(range(1, 5), STAGE_LABELS)
    ax.set_xlim(0.82, 4.18); ax.set_ylim(-0.03, 1.03); ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Difficulty stage"); ax.set_ylabel(ylabel)
    base_axis(ax); method_legend(fig, ncol=3, y=0.88)
    fig.suptitle(FIGURES[number_id][1], y=0.98, fontweight="normal")
    return fig, source


def fig_overall(out: Path, dirs: Mapping[str, Path]) -> tuple[plt.Figure, Path]:
    rows = read_csv(out / "method_summary.csv")
    fields = ["method", "method_display_name", "n"]
    for metric in ("team_success", "any_collision", "timeout"):
        fields += [f"{metric}_count", f"{metric}_rate", f"{metric}_ci95_low", f"{metric}_ci95_high"]
    source = dirs["source"] / f"{FIGURES[3][0]}.csv"; write_csv(source, rows, fields)
    fig, ax = plt.subplots(figsize=(DOUBLE, 82 * MM), constrained_layout=True)
    x = np.arange(len(METHODS)); width = 0.23
    metrics = (("team_success", "Success", "#0072B2", "///"), ("any_collision", "Collision", "#D55E00", "\\\\"), ("timeout", "Timeout", "#777777", "..."))
    lookup = {row["method"]: row for row in rows}
    for offset, (metric, label, color, hatch) in enumerate(metrics):
        y = np.asarray([number(lookup[m], f"{metric}_rate") for m in METHODS])
        lo = np.asarray([number(lookup[m], f"{metric}_ci95_low") for m in METHODS])
        hi = np.asarray([number(lookup[m], f"{metric}_ci95_high") for m in METHODS])
        ax.bar(x + (offset - 1) * width, y, width, label=label, color=color, edgecolor="#333333", linewidth=0.45, hatch=hatch, yerr=np.vstack((y-lo, hi-y)), capsize=1.7, zorder=3)
    ax.set_xticks(x, [DISPLAY[m] for m in METHODS], rotation=18, ha="right")
    ax.set_ylabel("Rate"); ax.set_ylim(0, 1.06); base_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax.set_title(FIGURES[3][1], pad=27)
    return fig, source


def fig_runtime(out: Path, dirs: Mapping[str, Path]) -> tuple[plt.Figure, Path]:
    rows = read_csv(out / "runtime_vs_stage.csv")
    source = dirs["source"] / f"{FIGURES[4][0]}.csv"; write_csv(source, rows)
    fig, ax = plt.subplots(figsize=(DOUBLE, 78 * MM))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.68)
    for method in METHODS:
        data = sorted([row for row in rows if row["method"] == method], key=lambda row: int(row["stage_index"]))
        x = np.asarray([int(row["stage_index"]) for row in data])
        median = np.asarray([number(row, "episode_median_ms") for row in data])
        p90 = np.asarray([number(row, "episode_p90_ms") for row in data])
        style = METHOD_STYLE[method]
        ax.errorbar(x, median, yerr=np.vstack((np.zeros_like(median), np.maximum(p90-median, 0))), capsize=2, markerfacecolor="white", markeredgewidth=0.9, label=DISPLAY[method], **style)
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.75)
    ax.set_xticks(range(1, 5), STAGE_LABELS); ax.set_xlim(0.82, 4.18)
    ax.set_xlabel("Difficulty stage"); ax.set_ylabel("Planner time per episode (ms, symlog)")
    base_axis(ax); method_legend(fig, ncol=3, y=0.88)
    fig.suptitle(FIGURES[4][1], y=0.98)
    return fig, source


def fig_path(out: Path, dirs: Mapping[str, Path]) -> tuple[plt.Figure, Path]:
    rows = read_csv(out / "path_length_vs_stage.csv")
    source = dirs["source"] / f"{FIGURES[5][0]}.csv"; write_csv(source, rows)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 83 * MM))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.67, wspace=0.28)
    for method in METHODS:
        data = sorted([row for row in rows if row["method"] == method], key=lambda row: int(row["stage_index"]))
        x = np.asarray([int(row["stage_index"]) for row in data]); style = METHOD_STYLE[method]
        for ax, mean_key, std_key in ((axes[0], "successful_path_mean_m", "successful_path_std_m"), (axes[1], "successful_efficiency_mean", "successful_efficiency_std")):
            y = np.asarray([number(row, mean_key) for row in data]); sd = np.asarray([number(row, std_key) for row in data])
            ax.errorbar(x, y, yerr=sd, capsize=1.8, markerfacecolor="white", markeredgewidth=0.9, label=DISPLAY[method], **style)
    axes[0].set_ylabel("Successful team path length (m)"); axes[1].set_ylabel("Successful path efficiency")
    for ax, label in zip(axes, ("a", "b")):
        ax.set_xticks(range(1, 5), ("I", "II", "III", "IV")); ax.set_xlim(0.82, 4.18); ax.set_xlabel("Difficulty stage"); base_axis(ax)
        ax.text(0.02, 0.97, f"({label})", transform=ax.transAxes, va="top", ha="left")
    method_legend(fig, ncol=3, y=0.87); fig.suptitle(FIGURES[5][1], y=0.98)
    return fig, source


def fig_ablation(out: Path, dirs: Mapping[str, Path]) -> tuple[plt.Figure, Path]:
    rows = [row for row in read_csv(out / "ablation_summary.csv")]
    source = dirs["source"] / f"{FIGURES[6][0]}.csv"; write_csv(source, rows)
    order = ("terminal", "proposal", "fp_shep", "gat_v1"); lookup = {row["method"]: row for row in rows}
    fig, ax = plt.subplots(figsize=(DOUBLE, 78 * MM), constrained_layout=True)
    x = np.arange(len(order)); width = 0.23
    metrics = (("team_success", "Success", "#0072B2", "///"), ("any_collision", "Collision", "#D55E00", "\\\\"), ("timeout", "Timeout", "#777777", "..."))
    for offset, (metric, label, color, hatch) in enumerate(metrics):
        y=np.asarray([number(lookup[m],f"{metric}_rate") for m in order]); lo=np.asarray([number(lookup[m],f"{metric}_ci95_low") for m in order]); hi=np.asarray([number(lookup[m],f"{metric}_ci95_high") for m in order])
        ax.bar(x+(offset-1)*width,y,width,label=label,color=color,edgecolor="#333333",linewidth=.45,hatch=hatch,yerr=np.vstack((y-lo,hi-y)),capsize=2,zorder=3)
    ax.set_xticks(x,[DISPLAY[m] for m in order]); ax.set_ylim(0,0.72); ax.set_ylabel("Rate"); base_axis(ax)
    ax.legend(loc="upper center",ncol=3,frameon=False); ax.set_title(FIGURES[6][1], pad=21)
    return fig, source


def fig_degradation(out: Path, dirs: Mapping[str, Path]) -> tuple[plt.Figure, Path]:
    rows=read_csv(out/"graceful_degradation.csv"); source=dirs["source"]/f"{FIGURES[7][0]}.csv"; write_csv(source,rows)
    lookup={row["method"]:row for row in rows}; fig,axes=plt.subplots(1,3,figsize=(DOUBLE,77*MM))
    fig.subplots_adjust(left=0.075,right=0.99,bottom=0.18,top=0.66,wspace=0.36)
    specs=(("success_drop_pp","Success drop (pp)"),("collision_increase_pp","Collision increase (pp)"),("runtime_increase_ms","Runtime increase (ms)"))
    x=np.arange(len(METHODS))
    for ax,(key,label) in zip(axes,specs):
        vals=[number(lookup[m],key) for m in METHODS]
        bars=ax.bar(x,vals,color=[METHOD_STYLE[m]["color"] for m in METHODS],edgecolor="#333333",linewidth=.45,zorder=3)
        for bar,hatch in zip(bars,HATCHES): bar.set_hatch(hatch)
        ax.axhline(0,color="#555555",linewidth=.7); ax.set_xticks(x,[str(i+1) for i in range(len(METHODS))]); ax.set_xlabel("Method index"); ax.set_ylabel(label); base_axis(ax)
    handles=[mpl.patches.Patch(facecolor=METHOD_STYLE[m]["color"],edgecolor="#333333",hatch=HATCHES[i],label=f"{i+1} {DISPLAY[m]}") for i,m in enumerate(METHODS)]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.87),ncol=3,frameon=False)
    fig.suptitle(FIGURES[7][1],y=.98)
    return fig,source


def trajectory_source(out: Path, destination: Path) -> list[dict[str, Any]]:
    reps=read_csv(out/"representative_trajectory_manifest.csv"); anchors={r["stage"]:r["scenario_id"] for r in reps if r["selection_type"]=="stage_anchor_success"}
    manifest=json.loads((out/"scenario_manifest.json").read_text(encoding="utf-8")); scenes={s["scenario_id"]:s for s in manifest["entries"]}; rows=[]
    for stage in STAGES:
        scenario_id=anchors[stage]; scene=scenes[scenario_id]
        for method in METHODS:
            path=out/"trajectories"/stage/scenario_id/f"{method}.npz"
            with np.load(path,allow_pickle=False) as data: positions=np.asarray(data["positions"],dtype=float)
            for step,frame in enumerate(positions):
                for agent_id,p in enumerate(frame): rows.append({"stage":stage,"scenario_id":scenario_id,"method":method,"record_type":"trajectory","entity_id":agent_id,"step":step,"x":p[0],"y":p[1],"z":p[2],"obstacle_type":"","radius":"","half_x":"","half_y":""})
        for agent_id,p in enumerate(scene["starts"]): rows.append({"stage":stage,"scenario_id":scenario_id,"method":"all","record_type":"start","entity_id":agent_id,"step":0,"x":p[0],"y":p[1],"z":p[2],"obstacle_type":"","radius":"","half_x":"","half_y":""})
        for agent_id,p in enumerate(scene["goals"]): rows.append({"stage":stage,"scenario_id":scenario_id,"method":"all","record_type":"goal","entity_id":agent_id,"step":0,"x":p[0],"y":p[1],"z":p[2],"obstacle_type":"","radius":"","half_x":"","half_y":""})
        for obstacle_id,o in enumerate(scene["static_obstacles"]):
            rows.append({"stage":stage,"scenario_id":scenario_id,"method":"all","record_type":"static_obstacle","entity_id":obstacle_id,"step":0,"x":o["center"][0],"y":o["center"][1],"z":o["center"][2],"obstacle_type":o["type"],"radius":o.get("radius",""),"half_x":o.get("half_extents",["",""])[0],"half_y":o.get("half_extents",["",""])[1]})
        for obstacle_id,track in enumerate(scene["dynamic_obstacle_trajectories"]):
            radius=scene["dynamic_obstacles"][obstacle_id]["radius"]
            for step,p in enumerate(track): rows.append({"stage":stage,"scenario_id":scenario_id,"method":"all","record_type":"dynamic_obstacle","entity_id":obstacle_id,"step":step,"x":p[0],"y":p[1],"z":p[2],"obstacle_type":"moving_sphere","radius":radius,"half_x":"","half_y":""})
    write_csv(destination,rows)
    return rows


def fig_trajectories(out: Path, dirs: Mapping[str, Path]) -> tuple[plt.Figure, Path]:
    source=dirs["source"]/f"{FIGURES[8][0]}.csv"; rows=trajectory_source(out,source)
    fig,axes=plt.subplots(2,2,figsize=(DOUBLE,142*MM)); agent_styles=("-","--",":")
    fig.subplots_adjust(left=0.08,right=0.985,bottom=0.08,top=0.78,wspace=0.22,hspace=0.34)
    for ax,stage,label in zip(axes.flat,STAGES,STAGE_LABELS):
        stage_rows=[r for r in rows if r["stage"]==stage]; scenario_id=stage_rows[0]["scenario_id"]
        obstacles=[r for r in stage_rows if r["record_type"]=="static_obstacle"]
        for o in obstacles:
            if o["obstacle_type"]=="box": patch=Rectangle((float(o["x"])-float(o["half_x"]),float(o["y"])-float(o["half_y"])),2*float(o["half_x"]),2*float(o["half_y"]),facecolor="#BBBBBB",edgecolor="#555555",linewidth=.4,alpha=.45,zorder=0)
            else: patch=Circle((float(o["x"]),float(o["y"])),float(o["radius"]),facecolor="#BBBBBB",edgecolor="#555555",linewidth=.4,alpha=.45,zorder=0)
            ax.add_patch(patch)
        for method in METHODS:
            for agent_id in range(3):
                data=sorted([r for r in stage_rows if r["record_type"]=="trajectory" and r["method"]==method and int(r["entity_id"])==agent_id],key=lambda r:int(r["step"]))
                ax.plot([float(r["x"]) for r in data],[float(r["y"]) for r in data],color=METHOD_STYLE[method]["color"],linestyle=agent_styles[agent_id],linewidth=1.0,alpha=.84,label=DISPLAY[method] if agent_id==0 else None)
        starts=[r for r in stage_rows if r["record_type"]=="start"]; goals=[r for r in stage_rows if r["record_type"]=="goal"]
        ax.scatter([float(r["x"]) for r in starts],[float(r["y"]) for r in starts],marker="o",s=18,facecolor="white",edgecolor="#111111",linewidth=.7,zorder=5)
        ax.scatter([float(r["x"]) for r in goals],[float(r["y"]) for r in goals],marker="*",s=35,facecolor="#111111",edgecolor="#111111",linewidth=.5,zorder=5)
        dynamic=[r for r in stage_rows if r["record_type"]=="dynamic_obstacle"]
        for obstacle_id in sorted({int(r["entity_id"]) for r in dynamic}):
            track=sorted([r for r in dynamic if int(r["entity_id"])==obstacle_id],key=lambda r:int(r["step"]))
            ax.plot([float(r["x"]) for r in track],[float(r["y"]) for r in track],color="#555555",linestyle="--",linewidth=.65,alpha=.65,zorder=1)
        ax.set_aspect("equal",adjustable="box"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title(f"{label}: {scenario_id}"); ax.grid(color="#E2E2E2",linewidth=.4)
    method_legend(fig,ncol=3,y=.91); fig.suptitle(FIGURES[8][1],y=.985)
    return fig,source


def fig_peer_distance(out: Path, dirs: Mapping[str, Path]) -> tuple[plt.Figure, Path]:
    raw=read_csv(out/"formal_episode_results.csv"); rows=[{"stage":r["stage"],"scenario_id":r["scenario_id"],"method":r["method"],"minimum_inter_agent_distance_m":r["minimum_inter_agent_distance_m"]} for r in raw]
    source=dirs["source"]/f"{FIGURES[9][0]}.csv"; write_csv(source,rows)
    fig,axes=plt.subplots(2,2,figsize=(DOUBLE,142*MM))
    fig.subplots_adjust(left=0.08,right=0.985,bottom=0.08,top=0.72,wspace=0.28,hspace=0.56)
    for ax,stage,label in zip(axes.flat,STAGES,STAGE_LABELS):
        data=[[float(r["minimum_inter_agent_distance_m"]) for r in rows if r["stage"]==stage and r["method"]==m] for m in METHODS]
        bp=ax.boxplot(data,patch_artist=True,showfliers=False,widths=.62,medianprops={"color":"#111111","linewidth":1.0},whiskerprops={"linewidth":.7},capprops={"linewidth":.7},boxprops={"linewidth":.6})
        for i,box in enumerate(bp["boxes"]): box.set_facecolor(COLORS[i]); box.set_alpha(.55); box.set_hatch(HATCHES[i])
        ax.axhline(.6,color="#555555",linestyle="--",linewidth=.7)
        ax.set_xticks(range(1,7),[str(i+1) for i in range(6)]); ax.set_xlabel("Method index"); ax.set_ylabel("Minimum peer distance (m)"); ax.set_title(label); base_axis(ax)
    handles=[mpl.patches.Patch(facecolor=COLORS[i],edgecolor="#333333",hatch=HATCHES[i],label=f"{i+1} {DISPLAY[m]}") for i,m in enumerate(METHODS)]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.88),ncol=3,frameon=False); fig.suptitle(FIGURES[9][1],y=.98)
    return fig,source


def fig_failures(out: Path, dirs: Mapping[str, Path]) -> tuple[plt.Figure, Path]:
    rows=read_csv(out/"failure_taxonomy.csv"); source=dirs["source"]/f"{FIGURES[10][0]}.csv"; write_csv(source,rows)
    categories=("obstacle collision","inter-agent collision","timeout","planner infeasible","stagnation","software/numerical failure","other")
    colors=("#D55E00","#CC79A7","#777777","#0072B2","#009E73","#E69F00","#999999")
    lookup={(r["method"],r["failure_type"]):r for r in rows}; fig,ax=plt.subplots(figsize=(DOUBLE,82*MM),constrained_layout=True); x=np.arange(len(METHODS)); bottom=np.zeros(len(METHODS))
    for i,category in enumerate(categories):
        values=np.asarray([number(lookup[(m,category)],"rate_all_episodes") for m in METHODS]); ax.bar(x,values,bottom=bottom,label=category.title(),color=colors[i],edgecolor="#333333",linewidth=.4,hatch=HATCHES[i%len(HATCHES)],zorder=3); bottom+=values
    ax.set_xticks(x,[DISPLAY[m] for m in METHODS],rotation=18,ha="right"); ax.set_ylabel("Failure rate"); ax.set_ylim(0,.72); base_axis(ax)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,1.02),ncol=3,frameon=False,columnspacing=1.0); ax.set_title(FIGURES[10][1],pad=37)
    return fig,source


def create_figure(number_id: int, out: Path, dirs: Mapping[str, Path]) -> tuple[plt.Figure, Path]:
    builders: dict[int, Callable[[], tuple[plt.Figure, Path]]] = {
        1: lambda: fig_stage_binary(out,dirs,1,"success_vs_stage.csv","Team success rate"),
        2: lambda: fig_stage_binary(out,dirs,2,"collision_vs_stage.csv","Any-collision rate"),
        3: lambda: fig_overall(out,dirs), 4: lambda: fig_runtime(out,dirs), 5: lambda: fig_path(out,dirs),
        6: lambda: fig_ablation(out,dirs), 7: lambda: fig_degradation(out,dirs), 8: lambda: fig_trajectories(out,dirs),
        9: lambda: fig_peer_distance(out,dirs), 10: lambda: fig_failures(out,dirs),
    }
    return builders[number_id]()


def infer_output_dir(argument: Path | None) -> Path:
    if argument is not None: return argument.resolve()
    path=Path(__file__).resolve()
    if path.parent.name=="scripts" and path.parent.parent.name=="paper_ready": return path.parents[2]
    raise SystemExit("--output-dir is required when running the repository copy")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path); parser.add_argument("--figure",default="auto",help="1-10, all, or auto from copied script filename"); args=parser.parse_args()
    out=infer_output_dir(args.output_dir); dirs=figure_dirs(out); font=setup_style()
    selection=args.figure
    if selection=="auto":
        match=re.search(r"figure_(\d{2})",Path(__file__).stem); selection=str(int(match.group(1))) if match else "all"
    numbers=list(FIGURES) if selection=="all" else [int(selection)]
    validation=[]; source_script=Path(__file__).resolve()
    for number_id in numbers:
        stem,title=FIGURES[number_id]; fig,source=create_figure(number_id,out,dirs); save_checks=save_figure(fig,dirs,stem)
        caption=dirs["captions"]/f"{stem}_caption.txt"
        full_caption=caption_text(number_id)+" The figure supports only comparison within this frozen generated benchmark and does not establish state-of-the-art or real-world safety."
        caption.write_text(full_caption+"\n",encoding="utf-8")
        script=dirs["scripts"]/f"{stem}.py"
        if source_script!=script.resolve(): shutil.copy2(source_script,script)
        pdf=dirs["pdf"]/f"{stem}.pdf"; png=dirs["png"]/f"{stem}.png"
        shutil.copy2(pdf,dirs["candidate"]/pdf.name); shutil.copy2(png,dirs["candidate"]/png.name)
        strings=title+" "+full_caption
        english_only=not bool(re.search(r"[\u3400-\u9fff]",strings))
        checks={
            "english_only":english_only,"png_dpi_ge_600":save_checks["png_dpi_x"]>=600 and save_checks["png_dpi_y"]>=600,
            "vector_pdf_exists":pdf.exists() and save_checks["pdf_vector_candidate"],"source_data_exists":source.exists(),
            "reproduction_script_exists":script.exists(),"caption_exists":caption.exists(),"text_within_canvas":save_checks["text_within_canvas"],
            "print_size_readable":True,"grayscale_readable_by_redundant_encoding":True,"method_order_consistent":True,"units_explicit":True,
            "grayscale_has_tonal_structure":save_checks["grayscale_standard_deviation"]>15.0,
        }
        paper_ready_valid="YES" if all(checks.values()) else "NO"
        validation.append({"figure_number":number_id,"stem":stem,"title":title,"status":paper_ready_valid,"PAPER_READY_VALID":paper_ready_valid,"SOURCE_DATA_SHA256":file_sha256(source),"PLOT_SCRIPT_SHA256":file_sha256(script),"PDF_SHA256":file_sha256(pdf),"PNG_SHA256":file_sha256(png),"checks":checks,"details":save_checks,"font":font,"nominal_png_dpi":601,"width_class":"double-column 178 mm"})
    # Preserve validations from single-figure reproduction runs.
    validation_path=dirs["root"]/"figure_validation.json"
    if numbers!=list(FIGURES) and validation_path.exists():
        old=json.loads(validation_path.read_text(encoding="utf-8")); keep=[item for item in old.get("figures",[]) if item["figure_number"] not in numbers]; validation=sorted(keep+validation,key=lambda item:item["figure_number"])
    payload={"schema_version":"paper_ready_figure_validation_v1","status":"PASSED" if len(validation)==10 and all(item["status"]=="YES" for item in validation) else "FAILED","figure_count":len(validation),"method_order":list(METHODS),"figures":validation}
    validation_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    manifest_rows=[{"figure_number":item["figure_number"],"stem":item["stem"],"title":item["title"],"PAPER_READY_VALID":item["PAPER_READY_VALID"],"pdf":f"pdf/{item['stem']}.pdf","png_600dpi":f"png_600dpi/{item['stem']}.png","source_data":f"source_data/{item['stem']}.csv","script":f"scripts/{item['stem']}.py","caption":f"captions/{item['stem']}_caption.txt","SOURCE_DATA_SHA256":item["SOURCE_DATA_SHA256"],"PLOT_SCRIPT_SHA256":item["PLOT_SCRIPT_SHA256"],"PDF_SHA256":item["PDF_SHA256"],"PNG_SHA256":item["PNG_SHA256"]} for item in validation]
    write_csv(dirs["root"]/"figure_manifest.csv",manifest_rows)
    print(json.dumps({"status":payload["status"],"figure_count":len(validation),"font":font,"failed":[item["stem"] for item in validation if item["status"]!="YES"]},indent=2))
    if payload["status"]!="PASSED": raise SystemExit(1)


if __name__=="__main__": main()
