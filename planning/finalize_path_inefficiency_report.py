"""Compose the final narrative from frozen path-audit outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value): return f"{100*float(value):.3f}%"
def num(value, digits=3): return f"{float(value):.{digits}f}"


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True); args=parser.parse_args(); root=args.root.resolve()
    c=load_json(root/"conclusion.json")
    seg=load_json(root/"03_err_segment_analysis/ERR_SEGMENT_EFFICIENCY_SUMMARY.json")
    err=load_json(root/"05_path_jerk_relation/ERR_PATH_ASSOCIATION.json")["associations"]
    smooth=load_json(root/"05_path_jerk_relation/PATH_SMOOTHNESS_ASSOCIATION.json")["associations"]
    temporal=load_json(root/"05_path_jerk_relation/JERK_COST_TEMPORAL_ATTRIBUTION.json")
    tacv=load_json(root/"07_tacv_medium_diagnostic/TACV_MEDIUM_PATH_DIAGNOSTIC_SUMMARY.json")
    stages=load_csv(root/"08_stage_analysis/STAGE_PATH_INEFFICIENCY_SUMMARY.csv")
    baselines=load_csv(root/"08_stage_analysis/COMPATIBLE_BASELINE_PATH_COMPARISON.csv")
    total_samples=temporal["within_0p5s"]["samples"]+temporal["outside_0p5s"]["samples"]
    low_jerk=sum(row["fraction_total_jerk_cost"] for row in temporal["segment_type_attribution"] if row["category"] in {"negative","low"})
    low_time=sum(row["fraction_total_time_samples"] for row in temporal["segment_type_attribution"] if row["category"] in {"negative","low"})
    per_agent_path=c["MEAN_PATH_LENGTH_SUCCESS"]/3.0
    tacv_path_percent=100*tacv["mean_path_change_m"]/tacv["mean_original_path_m"]

    stage_lines=[]
    for row in stages:
        stage_lines.append(f"| {row['stage']} | {int(row['successful_team_episodes'])} | {float(row['detour_ratio_mean']):.4f} | {float(row['excess_path_m_mean']):.3f} | {100*float(row['backward_ratio_mean']):.3f}% | {100*float(row['lateral_ratio_mean']):.2f}% | {100*float(row['vertical_ratio_mean']):.2f}% | {float(row['smoothness_m2_s6_mean']):.2f} |")
    baseline_lines=[]
    for row in baselines:
        success=int(row['successful_team_episodes'])
        baseline_lines.append(f"| {row['method']} | {success}/400 ({100*success/400:.2f}%) | {float(row['detour_ratio_mean']):.4f} | {100*float(row['backward_ratio_mean']):.3f}% | {100*float(row['lateral_ratio_mean']):.2f}% | {100*float(row['vertical_ratio_mean']):.2f}% | {float(row['smoothness_m2_s6_mean']):.2f} |")

    lines=[
        "# Path-Inefficiency and Recurrent-Correction Root-Cause Audit", "",
        "## Executive result", "",
        "The frozen Proposed method remains unchanged. The audit classifies `PATH_EXCESS_PRIMARY_SOURCE = LATERAL_DETOUR`, but the central hypothesis that long paths and high jerk share a low-efficiency, self-canceling ERR mechanism is not supported: `PATH_AND_JERK_SHARED_CAUSE_EVIDENCE = NONE`.", "",
        f"Across the 381 successful Formal V2 episodes, mean team path length was **{c['MEAN_PATH_LENGTH_SUCCESS']:.3f} m** ({per_agent_path:.3f} m per agent), mean mission distance was **{c['MEAN_MISSION_DISTANCE']:.3f} m per agent**, and mean detour ratio was **{c['MEAN_DETOUR_RATIO']:.4f}**. Thus successful paths were about **{100*(c['MEAN_DETOUR_RATIO']-1):.2f}%** longer than straight-line mission distance, with mean per-agent excess **{c['MEAN_EXCESS_PATH_M']:.3f} m**.", "",
        f"Low-efficiency segments were rare: eta<0.2 represented **{pct(c['LOW_EFFICIENCY_SEGMENT_FRACTION_ETA_LT_0P2'])}** of segments and only **{pct(c['PATH_SHARE_IN_LOW_EFFICIENCY_SEGMENTS'])}** of executed path. Mean adjacent-segment cancellation was **{c['MEAN_SEGMENT_CANCELLATION']:.5f}** and only **{pct(c['HIGH_CANCELLATION_PAIR_FRACTION'])}** of pairs had C>=0.5. The mandatory stop rule therefore sets `FRAMEWORK_PRESERVING_PATH_REPAIR_HEADROOM = LOW` and `NEXT_STEP_RECOMMENDATION = NO CHANGE - KEEP ORIGINAL`.", "",
        "## 1. Exact metric contract and integrity", "",
        "Per-agent path length is the sum of Euclidean distances between every consecutive raw executed position at dt=0.1 s. The reset/start point is retained, no terminal interpolation is applied, and team path is the sum over three agents. Failure paths exist in the raw records, but the paper successful-path statistics use team-success episodes; paired continuous comparisons use the both-success subset.", "",
        f"The exact stored path and smoothness metrics were reproduced over all 381 successes. Maximum absolute reconciliation errors were **{c['metric_reconciliation']['max_team_path_error_m']:.3e} m** for team path and **{c['metric_reconciliation']['max_smoothness_error']:.3e} m^2/s^6** for smoothness. No controller, environment, or simulator was executed.", "",
        "## 2. Path decomposition", "",
        f"The mean explicit backward motion was only **{c['MEAN_BACKWARD_PATH_M']:.3f} m** per agent (**{pct(c['MEAN_BACKWARD_RATIO'])}** of path), so backtracking is not a major contributor. Mean vertical total variation was **{c['MEAN_VERTICAL_VARIATION_M']:.3f} m** (**{pct(c['MEAN_VERTICAL_RATIO'])}**). Mean cumulative lateral-motion magnitude was **{c['MEAN_LATERAL_MOTION_M']:.3f} m** (**{100*float(stages[0]['lateral_ratio_mean']):.2f}%**). Lateral, vertical, and backward terms are diagnostic and non-additive; the lateral value must not be read as an algebraic allocation of total excess path.", "",
        "| Scope | Success n | Mean detour | Mean excess/agent (m) | Backward ratio | Lateral ratio | Vertical ratio | Smoothness |", "|---|---:|---:|---:|---:|---:|---:|---:|",
        *stage_lines, "",
        "The stage-wise signatures are similar. No single difficult stage accounts for the overall detour pattern.", "",
        "## 3. ERR segment efficiency and cancellation", "",
        f"The 1,143 successful agent trajectories contain **{seg['segment_count']:,}** active-reference segments. Median eta was **{c['MEDIAN_ERR_SEGMENT_EFFICIENCY']:.4f}**. Negative-progress segments were **{pct(c['NEGATIVE_PROGRESS_SEGMENT_FRACTION'])}**; eta<0.2 segments were **{pct(c['LOW_EFFICIENCY_SEGMENT_FRACTION_ETA_LT_0P2'])}**. They consumed **{pct(c['PATH_SHARE_IN_LOW_EFFICIENCY_SEGMENTS'])}** of path and **{pct(c['EXCESS_PATH_SHARE_ASSOCIATED_WITH_LOW_EFFICIENCY_SEGMENTS'])}** of the explicitly labeled diagnostic excess burden.", "",
        f"Mean accepted reference changes were **{c['MEAN_ERR_ACCEPTED_CHANGES']:.1f} per team episode** (about {c['MEAN_ERR_ACCEPTED_CHANGES']/3:.1f} per agent). Absolute count correlated with team excess path (rho={err['err_count_vs_excess_path']['rho']:.3f}, 95% CI [{err['err_count_vs_excess_path']['ci95'][0]:.3f}, {err['err_count_vs_excess_path']['ci95'][1]:.3f}]), but changes per simulated second correlated in the opposite direction (rho={err['err_rate_vs_excess_path']['rho']:.3f}). Absolute event count therefore co-varies with task duration/path length and is not evidence that redundant updates caused the extra distance.", "",
        f"Mean cancellation was **{c['MEAN_SEGMENT_CANCELLATION']:.5f}**; the C>=0.5 pair fraction was **{pct(c['HIGH_CANCELLATION_PAIR_FRACTION'])}**. Although episode-average cancellation and excess path co-vary, the absolute magnitude and high-cancellation prevalence are too small to support recurrent reference cancellation as the dominant path source.", "",
        "## 4. Path-jerk relation", "",
        f"The expected shared-cause signature is absent. Detour ratio vs smoothness had rho={smooth['detour_ratio_vs_smoothness']['rho']:.3f}; low-efficiency path share vs smoothness had rho={smooth['low_efficiency_share_vs_smoothness']['rho']:.3f}; cancellation vs smoothness had rho={smooth['cancellation_vs_smoothness']['rho']:.3f}. All are opposite to the prediction that more inefficient/canceling motion should produce higher jerk. Eta<0.2 and negative segments together carried only **{pct(low_jerk)}** of total jerk cost across **{pct(low_time)}** of jerk samples.", "",
        "This does not mean ERR timing is unrelated to jerk. It means the specific low-progress/self-canceling path mechanism is not shared by the two outcomes.", "",
        "## 5. Switch-window and stable-reference jerk", "",
        f"Using per-agent unions of jerk-sample indices, 0.2 s reference-change windows covered **{pct(temporal['within_0p2s']['samples']/total_samples)}** of samples and contained **{pct(temporal['within_0p2s']['cost_fraction'])}** of total jerk cost. The 0.5 s unions covered **{pct(temporal['within_0p5s']['samples']/total_samples)}** of samples and contained **{pct(temporal['within_0p5s']['cost_fraction'])}** of cost. Outside those windows, **{pct(temporal['outside_0p5s']['cost_fraction'])}** of cost remained across **{pct(temporal['outside_0p5s']['samples']/total_samples)}** of samples, at **{temporal['outside_0p5s']['cost_per_step']:.2f} m^2/s^6** per step.", "",
        f"Switch-adjacent jerk is therefore temporally concentrated: 0.5 s-window cost per step is {temporal['within_0p5s']['cost_per_step']/temporal['outside_0p5s']['cost_per_step']:.2f}x the outside-window level. However, CRT reduced reliability and TACV-Medium reduced switch-aligned jerk by only about 0.19%; prior interventions do not justify another switch-only repair branch.", "",
        "## 6. Compatible successful-subset baselines", "",
        "| Method | Team success | Mean detour | Backward ratio | Lateral ratio | Vertical ratio | Smoothness |", "|---|---:|---:|---:|---:|---:|---:|",
        *baseline_lines, "",
        "These rows are unpaired, method-specific successful subsets. DWA-SensingMatched and PPO-Direct have shorter/smoother successful trajectories but substantially lower team success; conditional trajectory quality is not overall method superiority.", "",
        "## 7. TACV-Medium Development intervention", "",
        f"On the 93 paired both-success Development scenes, TACV-Medium changed team path by **{tacv['mean_path_change_m']:.3f} m ({tacv_path_percent:.3f}%)**, detour ratio by **{tacv['mean_detour_ratio_change']:.5f}**, smoothness by **{tacv['mean_smoothness_change']:.3f}**, and accepted reference changes by **{tacv['mean_reference_change_count_change']:+.3f}**. This is Case B: path geometry changed negligibly while smoothness improved modestly, and reference count did not fall.", "",
        "TACV-Medium did not retain executed positions in its NPZ. Backward/lateral/vertical motion, eta path share, and cancellation changes are therefore unavailable and were not reconstructed. These are Development diagnostics, not Formal evidence.", "",
        "## 8. Answers to Q1-Q12", "",
        f"1. Successful paths are about **{100*(c['MEAN_DETOUR_RATIO']-1):.2f}%** longer than straight-line mission distance.",
        "2. The largest diagnostic signature is lateral detour; it is non-additive and not proven unnecessary.",
        f"3. Vertical variation is secondary at **{pct(c['MEAN_VERTICAL_RATIO'])}** of path.",
        f"4. Explicit backtracking is negligible at **{pct(c['MEAN_BACKWARD_RATIO'])}**.",
        f"5. Eta<0.2 segments occupy **{pct(c['PATH_SHARE_IN_LOW_EFFICIENCY_SEGMENTS'])}** of path.",
        f"6. Adjacent-segment cancellation is low: mean C={c['MEAN_SEGMENT_CANCELLATION']:.5f}, high-C fraction={pct(c['HIGH_CANCELLATION_PAIR_FRACTION'])}.",
        f"7. Absolute ERR count correlates with excess (rho={c['ERR_COUNT_VS_EXCESS_PATH_RHO']:.3f}), but rate-normalized evidence reverses sign.",
        f"8. Low-efficiency segments do not contribute disproportionate jerk; their cost share is {pct(low_jerk)}.",
        f"9. **{pct(c['JERK_COST_WITHIN_0P5S_OF_REFERENCE_CHANGE'])}** of jerk cost lies inside 0.5 s change-window unions.",
        f"10. **{pct(c['JERK_COST_OUTSIDE_0P5S_SWITCH_WINDOWS'])}** persists outside those windows.",
        f"11. TACV-Medium produced a {tacv_path_percent:.3f}% path change and did not reduce reference changes; its 4.84% smoothness gain is a small distributed behavior change, not a recovered path-efficiency mechanism.",
        "12. No. The hard-stop conditions are met, so no further framework-preserving runtime repair is justified by this audit.", "",
        "## Final decision", "",
        "- `PATH_EXCESS_PRIMARY_SOURCE = LATERAL_DETOUR`",
        "- `PATH_AND_JERK_SHARED_CAUSE_EVIDENCE = NONE`",
        "- `FRAMEWORK_PRESERVING_PATH_REPAIR_HEADROOM = LOW`",
        "- `NEXT_STEP_RECOMMENDATION = NO CHANGE - KEEP ORIGINAL`",
        "- `NEW_SIMULATION_RUN = NO`",
        "- `METHOD_CHANGED = NO`",
        "- `FORMAL_RESULT_CHANGED = NO`", "",
        "## Scientific boundary", "",
        "The audit identifies associations and temporal concentration. It does not establish that ERR causes long paths, that lateral detour is unnecessary, or that any reference update can be suppressed safely. Obstacle-specific segment attribution is unavailable because the retained active-direction safety margin is untyped.", "",
        "## Artifact index", "",
        "Machine-readable conclusions are in `conclusion.json`; exact metric definitions in `01_metric_contract/PATH_METRIC_CONTRACT.json`; raw decompositions in `02_formal_path_decomposition/`; segment and cancellation tables in `03_err_segment_analysis/` and `04_cancellation/`; path-jerk and stable-reference analyses in `05_path_jerk_relation/` and `06_stable_reference/`; TACV and stage/baseline comparisons in `07_tacv_medium_diagnostic/` and `08_stage_analysis/`; figures A-N and source data in `09_figures/`; decision tables in `10_root_cause/`; paper-facing outputs in `11_paper_ready/`.", "",
    ]
    text="\n".join(lines)
    (root/"10_root_cause/PATH_INEFFICIENCY_ROOT_CAUSE_REPORT.md").write_text(text,encoding="utf-8")
    (root/"FINAL_REPORT.md").write_text(text,encoding="utf-8")
    paper=["# Paper Path-Efficiency Diagnostic","",f"On 381 successful Formal V2 episodes, mean agent detour ratio was {c['MEAN_DETOUR_RATIO']:.4f} and mean agent excess path was {c['MEAN_EXCESS_PATH_M']:.3f} m. The main diagnostic signature was lateral motion, while backward motion was {pct(c['MEAN_BACKWARD_RATIO'])} and eta<0.2 segments occupied only {pct(c['PATH_SHARE_IN_LOW_EFFICIENCY_SEGMENTS'])} of executed path.","",f"Low-efficiency segments did not carry disproportionate jerk. In contrast, the union of 0.5 s post-reference-change windows contained {pct(c['JERK_COST_WITHIN_0P5S_OF_REFERENCE_CHANGE'])} of jerk cost over {pct(temporal['within_0p5s']['samples']/total_samples)} of samples. This temporal concentration did not translate into a safe no-retraining repair in the prior CRT/TACV interventions.","", "Accordingly, path and jerk do not share the hypothesized low-progress/self-canceling ERR cause. The frozen method is retained; no new optimization branch is opened.",""]
    (root/"11_paper_ready/PAPER_PATH_EFFICIENCY_DIAGNOSTIC.md").write_text("\n".join(paper),encoding="utf-8")


if __name__=="__main__": main()
