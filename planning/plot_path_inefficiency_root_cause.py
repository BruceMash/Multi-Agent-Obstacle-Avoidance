"""Generate paper/diagnostic figures A-N from the read-only path audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DT = 0.1
STAGES = ["Stage I", "Stage II", "Stage III", "Stage IV"]
STAGE_COLORS = ["#2463A7", "#2B8C6B", "#D48A23", "#B7485A"]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def numeric(rows: Iterable[Mapping[str, Any]], field: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, float)


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.0, "axes.titlesize": 10.0,
        "axes.labelsize": 9.0, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0, "figure.dpi": 140, "savefig.dpi": 600,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#D8DDE5", "grid.linewidth": 0.55,
        "grid.alpha": 0.75, "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig: plt.Figure, code: str, title: str, caption: str, dirs: Mapping[str, Path], source_rows: Sequence[Mapping[str, Any]], manifest: list[dict[str, Any]]) -> None:
    stem = f"figure_{code}_{title}"
    pdf = dirs["pdf"] / f"{stem}.pdf"
    png = dirs["png"] / f"{stem}.png"
    source = dirs["source"] / f"{stem}.csv"
    cap = dirs["captions"] / f"{stem}_caption.txt"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    write_csv(source, list(source_rows))
    cap.write_text(caption.strip() + "\n", encoding="utf-8")
    manifest.append({"figure": f"Figure {code}", "title": title.replace("_", " "), "pdf": str(pdf), "png_600dpi": str(png), "source_data": str(source), "caption": str(cap), "status": "YES"})


def scatter_by_stage(ax: plt.Axes, rows: Sequence[Mapping[str, Any]], x: str, y: str, xlabel: str, ylabel: str) -> None:
    for stage, color in zip(STAGES, STAGE_COLORS):
        selected = [row for row in rows if row["stage"] == stage]
        ax.scatter(numeric(selected, x), numeric(selected, y), s=13, alpha=0.55, color=color, edgecolors="none", label=stage)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); args = parser.parse_args()
    root = args.root.resolve(); style()
    dirs = {"pdf": root / "09_figures/pdf", "png": root / "09_figures/png_600dpi", "source": root / "09_figures/source_data", "captions": root / "09_figures/captions"}
    for path in dirs.values(): path.mkdir(parents=True, exist_ok=True)
    agents = read_csv(root / "02_formal_path_decomposition/FORMAL_PATH_DECOMPOSITION.csv")
    episodes = read_csv(root / "02_formal_path_decomposition/FORMAL_EPISODE_PATH_SUMMARY.csv")
    segments = read_csv(root / "03_err_segment_analysis/ERR_SEGMENT_EFFICIENCY.csv")
    cancellation = read_csv(root / "04_cancellation/REFERENCE_SEQUENCE_CANCELLATION.csv")
    tacv = read_csv(root / "07_tacv_medium_diagnostic/TACV_MEDIUM_PATH_DIAGNOSTIC.csv")
    temporal = json.loads((root / "05_path_jerk_relation/JERK_COST_TEMPORAL_ATTRIBUTION.json").read_text(encoding="utf-8"))
    representative = json.loads((root / "09_figures/REPRESENTATIVE_TRAJECTORY_SELECTION.json").read_text(encoding="utf-8"))["selected"]
    manifest: list[dict[str, Any]] = []

    # A
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    data = [numeric([r for r in agents if r["stage"] == stage], "detour_ratio") for stage in STAGES]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], STAGE_COLORS): body.set_facecolor(color); body.set_alpha(0.62)
    parts["cmedians"].set_color("#27313F"); parts["cmedians"].set_linewidth(1.3)
    ax.axhline(1.0, color="#596574", ls="--", lw=1); ax.set_xticks(range(1, 5), STAGES); ax.set_ylabel("Detour ratio $L/D$"); ax.set_title("Mission-normalized successful path length")
    save(fig, "A", "mission_normalized_path_distribution", "Successful Formal V2 Proposed agent trajectories. Violins show the full distribution; horizontal marks show medians; the dashed line is the straight-line lower reference L/D=1.", dirs, [{"stage": r["stage"], "scenario_id": r["scenario_id"], "agent_id": r["agent_id"], "detour_ratio": r["detour_ratio"]} for r in agents], manifest)

    # B
    fig, ax = plt.subplots(figsize=(7.2, 3.9)); fields = [("normalized_excess", "Excess/path mission"), ("backward_ratio", "Backward/path"), ("lateral_ratio", "Lateral/path"), ("vertical_ratio", "Vertical/path")]
    x = np.arange(4); width = 0.19
    source = []
    for index, (field, label) in enumerate(fields):
        vals=[]
        for stage in STAGES:
            value=float(np.mean(numeric([r for r in agents if r["stage"] == stage], field))); vals.append(value); source.append({"stage":stage,"component":field,"mean_ratio":value})
        ax.bar(x+(index-1.5)*width, vals, width, label=label)
    ax.set_xticks(x, STAGES); ax.set_ylabel("Mean diagnostic ratio"); ax.set_title("Path-excess signatures by stage"); ax.legend(ncol=2, frameon=False)
    ax.text(0.99, 0.98, "Components are non-additive", transform=ax.transAxes, ha="right", va="top", color="#596574")
    save(fig, "B", "path_excess_decomposition_by_stage", "Stage-wise means of mission-normalized excess and diagnostic backward, lateral, and vertical variation ratios. These vector-derived diagnostics are not additive path partitions.", dirs, source, manifest)

    # C
    fig, ax = plt.subplots(figsize=(6.4, 3.7)); data=[numeric([r for r in agents if r["stage"]==s],"backward_ratio") for s in STAGES]
    ax.boxplot(data, labels=STAGES, showfliers=False, patch_artist=True, boxprops={"facecolor":"#C9D6E8"}, medianprops={"color":"#27313F"}); ax.set_yscale("symlog", linthresh=1e-4); ax.set_ylabel("Backward motion / path"); ax.set_title("Explicit movement away from the final task goal")
    save(fig,"C","backward_motion_ratio_distribution","Successful Formal V2 agent-level backward-motion ratios. The symmetric-log scale preserves zeros and displays the sparse upper tail.",dirs,[{"stage":r["stage"],"scenario_id":r["scenario_id"],"agent_id":r["agent_id"],"backward_ratio":r["backward_ratio"]} for r in agents],manifest)

    for code, title, xfield, xlabel in (("D","vertical_variation_vs_path_excess","vertical_variation_m","Vertical total variation (m)"),("E","lateral_motion_vs_path_excess","lateral_motion_m","Cumulative lateral motion (m)")):
        fig, ax=plt.subplots(figsize=(6.2,4.0)); scatter_by_stage(ax,agents,xfield,"excess_path_m",xlabel,"Excess path (m)"); ax.legend(frameon=False,ncol=2); ax.set_title(title.replace("_"," ").title())
        save(fig,code,title,f"Successful Formal V2 agent trajectories; points are raw-trajectory decompositions colored by stage. The displayed association is descriptive.",dirs,[{"stage":r["stage"],"scenario_id":r["scenario_id"],"agent_id":r["agent_id"],xfield:r[xfield],"excess_path_m":r["excess_path_m"]} for r in agents],manifest)

    # F
    fig, ax=plt.subplots(figsize=(6.5,3.8)); eta=numeric(segments,"segment_efficiency_eta"); clipped=np.clip(eta,-0.25,1.05)
    ax.hist(clipped,bins=70,color="#2463A7",alpha=.78); ax.axvline(.2,color="#B7485A",ls="--",lw=1.2,label=r"diagnostic $\eta=0.2$"); ax.axvline(0,color="#27313F",lw=1); ax.set_yscale("log"); ax.set_xlabel(r"Segment progress efficiency $\eta$"); ax.set_ylabel("Segment count (log)"); ax.set_title("ERR segment progress-efficiency distribution"); ax.legend(frameon=False)
    save(fig,"F","err_segment_efficiency_distribution","All accepted-active-reference segments from successful Formal V2 trajectories. Values are clipped only for this display; source data retain exact eta. Thresholds are descriptive reporting bins, not pruning rules.",dirs,[{"scenario_id":r["scenario_id"],"stage":r["stage"],"agent_id":r["agent_id"],"segment_index":r["segment_index"],"eta":r["segment_efficiency_eta"],"path_m":r["segment_path_length_m"]} for r in segments],manifest)

    # G
    source=[]
    for stage in ["Overall"]+STAGES:
        s=segments if stage=="Overall" else [r for r in segments if r["stage"]==stage]; total=np.sum(numeric(s,"segment_path_length_m")); low=np.sum([float(r["segment_path_length_m"]) for r in s if float(r["segment_efficiency_eta"])<.2]); neg=np.sum([float(r["segment_path_length_m"]) for r in s if float(r["segment_efficiency_eta"])<0]); source.append({"stage":stage,"low_eta_lt_0p2_path_share":low/total,"negative_eta_path_share":neg/total})
    fig,ax=plt.subplots(figsize=(6.4,3.7)); xx=np.arange(len(source)); ax.bar(xx,[r["low_eta_lt_0p2_path_share"] for r in source],color="#D48A23",label=r"$\eta<0.2$"); ax.bar(xx,[r["negative_eta_path_share"] for r in source],color="#B7485A",label=r"$\eta<0$"); ax.set_xticks(xx,[r["stage"] for r in source]); ax.set_ylabel("Fraction of executed path"); ax.set_title("Path spent in low-progress ERR segments"); ax.legend(frameon=False)
    save(fig,"G","low_efficiency_path_share","Fractions are path-length weighted. Negative-progress path is a subset of eta<0.2 path; bars overlap and must not be added.",dirs,source,manifest)

    for code,title,xfield,xlabel,yfield,ylabel in (("H","segment_cancellation_vs_path_excess","mean_segment_cancellation","Mean adjacent-segment cancellation","team_excess_path_m","Team excess path (m)"),("I","err_frequency_vs_path_excess","err_accepted_reference_changes","Accepted reference changes / episode","team_excess_path_m","Team excess path (m)"),("J","path_excess_vs_smoothness","team_excess_path_m","Team excess path (m)","trajectory_smoothness_m2_s6",r"Smoothness ($m^2s^{-6}$)"),("K","low_efficiency_share_vs_smoothness","path_share_in_low_efficiency_segments",r"Path share in $\eta<0.2$ segments","trajectory_smoothness_m2_s6",r"Smoothness ($m^2s^{-6}$)")):
        fig,ax=plt.subplots(figsize=(6.2,4.0)); scatter_by_stage(ax,episodes,xfield,yfield,xlabel,ylabel); ax.legend(frameon=False,ncol=2); ax.set_title(title.replace("_"," ").title())
        save(fig,code,title,"Successful Formal V2 team episodes. Rank associations and episode-cluster confidence intervals are reported in the JSON analysis artifacts; scatter is descriptive.",dirs,[{"scenario_id":r["scenario_id"],"stage":r["stage"],xfield:r[xfield],yfield:r[yfield]} for r in episodes],manifest)

    # L
    total_samples=temporal["within_0p5s"]["samples"]+temporal["outside_0p5s"]["samples"]
    source=[
        {"category":"within 0.2 s","jerk_cost_fraction":temporal["within_0p2s"]["cost_fraction"],"time_sample_fraction":temporal["within_0p2s"]["samples"]/total_samples},
        {"category":"0.2-0.5 s increment","jerk_cost_fraction":temporal["within_0p5s"]["cost_fraction"]-temporal["within_0p2s"]["cost_fraction"],"time_sample_fraction":(temporal["within_0p5s"]["samples"]-temporal["within_0p2s"]["samples"])/total_samples},
        {"category":"outside 0.5 s","jerk_cost_fraction":temporal["outside_0p5s"]["cost_fraction"],"time_sample_fraction":temporal["outside_0p5s"]["samples"]/total_samples},]
    fig,ax=plt.subplots(figsize=(7.0,3.2)); colors=["#B7485A","#D48A23","#7E8794"]
    for yi,field in enumerate(("jerk_cost_fraction","time_sample_fraction")):
        left=0
        for row,color in zip(source,colors): ax.barh(yi,row[field],left=left,color=color,label=row["category"] if yi==0 else None); left+=row[field]
    ax.set_yticks([0,1],["Jerk-cost share","Time-sample share"]); ax.set_xlim(0,1); ax.set_xlabel("Fraction"); ax.set_title("Union-of-time-indices switch-window attribution"); ax.legend(frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(.5,-.48))
    save(fig,"L","switch_window_vs_stable_jerk_contribution","Per-agent accepted reference-change windows are unioned before cost aggregation, preventing double counting. The middle band is the incremental 0.2-0.5 s union; outside 0.5 s is the stable complement.",dirs,source,manifest)

    # M: stage-median raw trajectories and raw signals.
    fig,axes=plt.subplots(4,4,figsize=(14.2,12.8)); msource=[]; cmap=mpl.colormaps["viridis"]
    for row_i,selected in enumerate(representative):
        record=json.loads(Path(selected["path"]).read_text(encoding="utf-8")); traj=np.load(selected["trajectory"]); pos=np.asarray(traj["positions"],float); acc=np.asarray(traj["accelerations"],float); goals=np.asarray(traj["goals"],float); scenario=selected["scenario_id"]
        segs=[r for r in segments if r["scenario_id"]==scenario]; by_agent={a:[r for r in segs if int(r["agent_id"])==a] for a in range(3)}
        ax=axes[row_i,0]
        for agent,ls in zip(range(3),("-","--",":")):
            for seg in by_agent[agent]:
                s,e=int(seg["start_step"]),int(seg["end_step_exclusive"]); eta=float(seg["segment_efficiency_eta"]); ax.plot(pos[s:e+1,agent,0],pos[s:e+1,agent,1],ls=ls,color=cmap(np.clip((eta+.1)/1.1,0,1)),lw=1.0)
            ax.scatter(pos[0,agent,0],pos[0,agent,1],s=18,marker="o",color="#27313F"); ax.scatter(goals[agent,0],goals[agent,1],s=32,marker="*",color="#B7485A")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title(f"{selected['stage']} - XY")
        t=np.arange(pos.shape[0])*DT
        ax=axes[row_i,1]
        for agent,ls in zip(range(3),("-","--",":")): ax.plot(t,pos[:,agent,2],ls=ls,lw=.9,label=f"UAV {agent+1}")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("z (m)"); ax.set_title("Raw altitude")
        ax=axes[row_i,2]
        for agent,ls in zip(range(3),("-","--",":")): ax.plot(t,np.linalg.norm(goals[agent]-pos[:,agent],axis=1),ls=ls,lw=.9)
        for seg in segs:
            if int(seg["agent_id"])==0: ax.axvline(float(seg["start_step"])*DT,color="#7E8794",alpha=.12,lw=.5)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Goal distance (m)"); ax.set_title("Final-goal distance + ERR markers")
        ax=axes[row_i,3]
        for agent,ls in zip(range(3),("-","--",":")):
            jerk=np.linalg.norm(np.diff(acc[:,agent,:],axis=0)/DT,axis=1); ax.plot(np.arange(1,acc.shape[0])*DT,jerk,ls=ls,lw=.75)
        ax.set_xlabel("Time (s)"); ax.set_ylabel(r"$||j||$ ($ms^{-3}$)"); ax.set_title("Raw jerk")
        for agent in range(3):
            eta_by_step=np.full(pos.shape[0],np.nan); change_by_step=np.zeros(pos.shape[0],dtype=bool)
            for seg in by_agent[agent]:
                s,e=int(seg["start_step"]),int(seg["end_step_exclusive"]); eta_by_step[s:e+1]=float(seg["segment_efficiency_eta"]); change_by_step[s]=True
            jerk_norm=np.full(pos.shape[0],np.nan)
            raw_jerk=np.linalg.norm(np.diff(acc[:,agent,:],axis=0)/DT,axis=1); jerk_norm[1:1+raw_jerk.size]=raw_jerk
            for step in range(pos.shape[0]): msource.append({"stage":selected["stage"],"scenario_id":scenario,"agent_id":agent,"step":step,"time_s":step*DT,"x":pos[step,agent,0],"y":pos[step,agent,1],"z":pos[step,agent,2],"goal_distance_m":np.linalg.norm(goals[agent]-pos[step,agent]),"segment_efficiency_eta":eta_by_step[step],"active_reference_change":change_by_step[step],"jerk_norm_mps3":jerk_norm[step]})
        traj.close()
    axes[0,1].legend(frameon=False,ncol=3,loc="upper right")
    scalar = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-0.1, vmax=1.0), cmap=cmap)
    scalar.set_array([])
    fig.suptitle("Stage-median successful raw trajectories and recurrent-reference signals",y=.995)
    fig.tight_layout(rect=(0,0.055,1,0.985))
    color_axis = fig.add_axes([0.055, 0.012, 0.22, 0.012])
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label(r"Segment efficiency $\eta$ (display clipped to [-0.1, 1.0])")
    save(fig,"M","representative_raw_trajectories_by_segment_efficiency","One successful episode per stage, selected by the stage median episode detour ratio. XY segments are colored by eta (purple low, yellow high); line style indexes UAV. Altitude, final-goal distance, ERR markers for UAV 1, and jerk are unsmoothed raw signals.",dirs,msource,manifest)

    # N
    fields=[("delta_path_m_medium_minus_original","Path (m)"),("delta_detour_ratio_medium_minus_original","Detour ratio"),("delta_smoothness_medium_minus_original","Smoothness"),("delta_reference_changes","Reference changes")]
    fig,axes=plt.subplots(1,4,figsize=(10.5,3.4)); source=[]
    for ax,(field,label) in zip(axes,fields):
        vals=numeric(tacv,field); ax.boxplot(vals,showfliers=True,patch_artist=True,boxprops={"facecolor":"#C9D6E8"},medianprops={"color":"#27313F"}); ax.axhline(0,color="#596574",ls="--",lw=.9); ax.set_xticks([1],["Medium - Original"]); ax.set_ylabel(label); ax.set_title(label)
        source.extend({"scenario_id":r["scenario_id"],"stage":r["stage"],"metric":field,"delta":r[field]} for r in tacv)
    fig.suptitle("TACV-Medium paired both-success Development changes"); fig.tight_layout()
    save(fig,"N","original_vs_tacv_medium_path_component_changes","Paired both-success Development only (not Formal evidence). Medium retained aggregate path, detour, smoothness, and event counts but not executed positions; unavailable backward/lateral/vertical/segment-efficiency changes are not reconstructed.",dirs,source,manifest)

    write_csv(root/"11_paper_ready/figure_manifest.csv",manifest); write_csv(root/"09_figures/figure_manifest.csv",manifest)
    print(json.dumps({"figure_count":len(manifest),"manifest":str(root/"11_paper_ready/figure_manifest.csv")},indent=2))


if __name__ == "__main__": main()
