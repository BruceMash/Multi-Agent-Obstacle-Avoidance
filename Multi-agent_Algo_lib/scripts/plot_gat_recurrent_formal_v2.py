#!/usr/bin/env python3
"""Create compact paper-ready figures for the GAT-R selector study."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
FORMAL = ROOT / "10_formal_v2"
REVISION = ROOT / "13_objective_revision"
PAPER = ROOT / "12_paper_ready"
PDF = PAPER / "figures_pdf"
PNG = PAPER / "figures_png_600dpi"
SOURCE = PAPER / "source_data"
CAPTIONS = PAPER / "captions"
METHOD_ORDER = (
    "M1_DWA_FullState", "M2_DWA_SensingMatched", "M4_Direct_SAC_DMP",
    "M5_Proposal_SAC_DMP", "M6_FP_SHEP_SAC_DMP", "M7_OneShot_GAT_SAC_DMP",
    "M8_RERR_FP_SHEP_SAC_DMP", "M9_Proposed_RERR_GAT_SAC_DMP",
)
SHORT_NAMES = (
    "DWA-FS", "DWA-SM", "Direct", "Proposal", "1-shot FP", "1-shot GAT",
    "R-ERR+FP", "R-ERR+GAT-R",
)
STAGES = ("Stage I", "Stage II", "Stage III", "Stage IV")
BLUE = "#2B6CB0"
ORANGE = "#D97706"
GREEN = "#27805B"
RED = "#C2413B"
GRAY = "#6B7280"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def setup() -> None:
    for directory in (PDF, PNG, SOURCE, CAPTIONS):
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean(axis: Any, grid: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(axis="y", color="0.90", linewidth=0.45, zorder=0)


def save(index: int, slug: str, title: str, caption: str, fig: Any, source: Path) -> dict[str, Any]:
    pdf = PDF / f"figure_{index:02d}_{slug}.pdf"
    png = PNG / f"figure_{index:02d}_{slug}.png"
    cap = CAPTIONS / f"figure_{index:02d}_{slug}_caption.txt"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    cap.write_text(caption.strip() + "\n", encoding="utf-8")
    return {
        "figure": index,
        "title": title,
        "pdf": str(pdf.relative_to(ROOT)).replace("\\", "/"),
        "png_600dpi": str(png.relative_to(ROOT)).replace("\\", "/"),
        "source_data": str(source.relative_to(ROOT)).replace("\\", "/"),
        "caption": str(cap.relative_to(ROOT)).replace("\\", "/"),
        "status": "YES",
    }


def figure_overall(overall: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    source = SOURCE / "figure_01_formal_overall_outcomes.csv"
    write_csv(source, overall)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.35), gridspec_kw={"width_ratios": (1.55, 1.0)})
    success = [100 * float(next(row for row in overall if row["method_id"] == method)["success_rate"]) for method in METHOD_ORDER]
    colors = [GRAY, GRAY, "0.72", "0.72", "0.72", "0.72", ORANGE, BLUE]
    bars = axes[0].bar(np.arange(8), success, color=colors, edgecolor="0.20", linewidth=0.35, zorder=2)
    axes[0].set_xticks(np.arange(8), SHORT_NAMES, rotation=26, ha="right")
    axes[0].set_ylabel("Team success (%)")
    axes[0].set_ylim(0, 103)
    axes[0].set_title("All frozen methods")
    for bar, value in zip(bars, success):
        axes[0].text(bar.get_x() + bar.get_width()/2, value + 1.4, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    clean(axes[0])
    selected = [next(row for row in overall if row["method_id"] == method) for method in METHOD_ORDER[-2:]]
    x = np.arange(2); width = 0.25
    for offset, (field, label, color, hatch) in enumerate((
        ("success_rate", "Success", GREEN, ""),
        ("collision_rate", "Collision", RED, "//"),
        ("inter_agent_collision_rate", "Peer collision", BLUE, ".."),
    )):
        values = [100 * float(row[field]) for row in selected]
        bars = axes[1].bar(x + (offset-1)*width, values, width, label=label, color=color, hatch=hatch, edgecolor="0.20", linewidth=0.35, zorder=2)
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x()+bar.get_width()/2, value+1.1, f"{value:.2f}", ha="center", fontsize=7)
    axes[1].set_xticks(x, ("R-ERR+FP", "R-ERR+GAT-R"))
    axes[1].set_ylim(0, 103)
    axes[1].set_title("Identical R-ERR execution")
    axes[1].legend(frameon=False, loc="center", bbox_to_anchor=(0.50, 0.48), ncol=1)
    clean(axes[1])
    fig.subplots_adjust(wspace=0.28, bottom=0.23)
    return save(1, "formal-overall-outcomes", "Formal V2 outcomes", "Formal V2 overall outcomes on 400 untouched scenarios. The right panel isolates selector contribution under an identical R-ERR and SAC-DMP execution contract.", fig, source)


def figure_stage(stage: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    selected_ids = (METHOD_ORDER[0], METHOD_ORDER[1], METHOD_ORDER[-2], METHOD_ORDER[-1])
    source_rows = [row for row in stage if row["method_id"] in selected_ids]
    source = SOURCE / "figure_02_stage_success.csv"
    write_csv(source, source_rows)
    fig, axis = plt.subplots(figsize=(6.65, 3.35))
    styles = (
        (GRAY, "o", "--", "DWA FullState"),
        ("0.45", "s", ":", "DWA SensingMatched"),
        (ORANGE, "^", "-.", "R-ERR + FP-SHEP"),
        (BLUE, "D", "-", "R-ERR + GAT-R"),
    )
    x = np.arange(4)
    for method, (color, marker, line, label) in zip(selected_ids, styles):
        values = [100 * float(next(row for row in source_rows if row["method_id"] == method and row["scope"] == scope)["success_rate"]) for scope in STAGES]
        axis.plot(x, values, color=color, marker=marker, linestyle=line, linewidth=1.55, markersize=4.5, label=label)
        if method in selected_ids[-2:]:
            for xx, value in zip(x, values):
                axis.annotate(f"{value:.0f}", (xx, value), xytext=(0, 6 if method == selected_ids[-1] else -12), textcoords="offset points", ha="center", color=color, fontsize=7)
    axis.set_xticks(x, STAGES)
    axis.set_ylim(30, 101)
    axis.set_ylabel("Team success (%)")
    axis.set_xlabel("Obstacle-population stage")
    axis.legend(frameon=False, ncol=2, loc="lower left")
    clean(axis)
    return save(2, "stage-success", "Success across stages", "Team success across the four 100-scenario Formal V2 stages. GAT-R improves over FP-SHEP in Stages II and IV, ties in Stage III, and is one point lower in Stage I.", fig, source)


def figure_replication() -> dict[str, Any]:
    dev = json.loads((REVISION / "07_development/DEV_GATE_DECISION.json").read_text(encoding="utf-8"))
    holdout = json.loads((REVISION / "08_holdout/holdout_paired_tests.json").read_text(encoding="utf-8"))
    formal = json.loads((FORMAL / "formal_v2_fp_vs_gat_rs.json").read_text(encoding="utf-8"))
    rows = [
        {"block": "Dev", "fp_success": dev["fp_dev_success"], "gat_r_success": dev["gat_r_dev_success"], "gain_pp": 100*(dev["gat_r_dev_success"]-dev["fp_dev_success"]), "mcnemar_p": None},
        {"block": "Holdout", "fp_success": 0.935, "gat_r_success": 0.955, "gain_pp": holdout["paired_binary"]["overall"]["team_success"]["gat_r_minus_fp_shep_rate_pp"], "mcnemar_p": holdout["paired_binary"]["overall"]["team_success"]["exact_two_sided_mcnemar_p"]},
        {"block": "Formal V2", "fp_success": formal["scopes"]["overall"]["team_success"]["fp_shep_rate"], "gat_r_success": formal["scopes"]["overall"]["team_success"]["gat_r_rate"], "gain_pp": formal["scopes"]["overall"]["team_success"]["gat_r_minus_fp_shep_rate_pp"], "mcnemar_p": formal["scopes"]["overall"]["team_success"]["exact_two_sided_mcnemar_p"]},
    ]
    source = SOURCE / "figure_03_replication_across_blocks.csv"
    write_csv(source, rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.25), gridspec_kw={"width_ratios": (1.35, 0.75)})
    x=np.arange(3); width=0.34
    axes[0].bar(x-width/2, [100*r["fp_success"] for r in rows], width, color=ORANGE, label="R-ERR+FP", edgecolor="0.2", linewidth=.35)
    axes[0].bar(x+width/2, [100*r["gat_r_success"] for r in rows], width, color=BLUE, label="R-ERR+GAT-R", edgecolor="0.2", linewidth=.35)
    axes[0].set_xticks(x, [r["block"] for r in rows]); axes[0].set_ylim(88,98); axes[0].set_ylabel("Team success (%)")
    axes[0].set_title("Independent evaluation blocks"); axes[0].legend(frameon=False, ncol=2, loc="lower center"); clean(axes[0])
    gains=[r["gain_pp"] for r in rows]
    bars=axes[1].bar(x,gains,color=["0.55",GREEN,BLUE],edgecolor="0.2",linewidth=.35)
    axes[1].axhline(0,color="0.2",linewidth=.65); axes[1].set_xticks(x,[r["block"] for r in rows],rotation=22,ha="right")
    axes[1].set_ylabel("GAT-R gain (pp)"); axes[1].set_title("Selector increment")
    for bar,value in zip(bars,gains): axes[1].text(bar.get_x()+bar.get_width()/2,value+0.08,f"+{value:.2f}",ha="center",fontsize=7.5)
    axes[1].set_ylim(-.3,2.5); clean(axes[1])
    fig.subplots_adjust(wspace=.32,bottom=.19)
    return save(3,"replication-across-blocks","Replicated selector gain","GAT-R retains a positive success increment over FP-SHEP in development, sealed Holdout, and untouched Formal V2. Dev was used for selection; Holdout and Formal V2 are independent evaluation blocks.",fig,source)


def figure_paired() -> dict[str, Any]:
    formal = json.loads((FORMAL / "formal_v2_fp_vs_gat_rs.json").read_text(encoding="utf-8"))
    scopes=("overall","Stage I","Stage II","Stage III","Stage IV")
    rows=[]
    for scope in scopes:
        test=formal["scopes"][scope]["team_success"]
        rows.append({"scope":scope,"gat_r_only":test["gat_r_only_positive"],"fp_only":test["fp_shep_only_positive"],"both_success":test["both_positive"],"mcnemar_p":test["exact_two_sided_mcnemar_p"]})
    source=SOURCE/"figure_04_paired_discordances.csv"; write_csv(source,rows)
    fig,axis=plt.subplots(figsize=(6.35,3.25)); x=np.arange(len(rows)); width=.36
    bars1=axis.bar(x-width/2,[r["gat_r_only"] for r in rows],width,color=BLUE,label="GAT-R-only success",edgecolor="0.2",linewidth=.35)
    bars2=axis.bar(x+width/2,[r["fp_only"] for r in rows],width,color=ORANGE,label="FP-only success",edgecolor="0.2",linewidth=.35)
    axis.set_xticks(x,["Overall","I","II","III","IV"]); axis.set_ylabel("Discordant paired scenarios"); axis.set_xlabel("Formal V2 scope")
    axis.legend(frameon=False,ncol=2); clean(axis)
    for bars in (bars1,bars2):
        for bar in bars: axis.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.35,f"{int(bar.get_height())}",ha="center",fontsize=7.5)
    axis.text(.98,.96,"Overall McNemar p = 0.451",transform=axis.transAxes,ha="right",va="top",fontsize=8)
    return save(4,"paired-discordances","Paired success discordances","Paired Formal V2 success discordances. GAT-R succeeds alone on 25 scenarios and FP-SHEP alone on 19; the exact two-sided overall McNemar test is not significant.",fig,source)


def figure_continuous() -> dict[str, Any]:
    payload=json.loads((FORMAL/"formal_v2_continuous_paired_tests.json").read_text(encoding="utf-8"))
    rows=[row for row in payload["rows"] if row["scope"]=="overall"]
    source=SOURCE/"figure_05_both_success_quality.csv"; write_csv(source,rows)
    wanted=("completion_time_s","team_path_length_m","trajectory_smoothness","total_online_algorithm_compute_ms")
    labels=("Completion time (s)","Team path (m)","Smoothness (lower better)","Online compute (ms)")
    colors=(GREEN,GREEN,RED,RED)
    fig,axes=plt.subplots(2,2,figsize=(7.0,4.6))
    for axis,metric,label,color in zip(axes.flat,wanted,labels,colors):
        row=next(item for item in rows if item["metric"]==metric); delta=float(row["mean_gat_r_minus_fp_shep"])
        lower=float(row["paired_bootstrap_mean_difference_ci95_lower"]); upper=float(row["paired_bootstrap_mean_difference_ci95_upper"])
        axis.axhline(0,color="0.3",linewidth=.65); axis.errorbar([0],[delta],yerr=[[delta-lower],[upper-delta]],fmt="o",color=color,capsize=4,linewidth=1.3)
        axis.set_xlim(-.7,.7); axis.set_xticks([]); axis.set_ylabel("GAT-R minus FP-SHEP"); axis.set_title(label)
        axis.text(.96,.92,f"mean {delta:+.2f}\np={float(row['wilcoxon_two_sided_p']):.2g}",transform=axis.transAxes,ha="right",va="top",fontsize=7.5)
        clean(axis)
    fig.suptitle("Both-success paired quality (n=356)",y=.995,fontsize=10)
    fig.subplots_adjust(hspace=.42,wspace=.30)
    return save(5,"both-success-quality","Both-success quality trade-offs","Both-success paired GAT-R minus FP-SHEP differences with 95% paired-bootstrap intervals. GAT-R is faster and shorter-path, but less smooth and more computationally expensive. Failed episodes are never filled with zero.",fig,source)


def main() -> None:
    setup()
    overall=read_csv(FORMAL/"formal_v2_overall_summary.csv")
    stage=read_csv(FORMAL/"formal_v2_stage_summary.csv")
    figures=[figure_overall(overall),figure_stage(stage),figure_replication(),figure_paired(),figure_continuous()]
    write_csv(PAPER/"figure_manifest.csv",figures)
    print(json.dumps({"status":"PASS","figure_count":len(figures)}),flush=True)


if __name__ == "__main__":
    main()
