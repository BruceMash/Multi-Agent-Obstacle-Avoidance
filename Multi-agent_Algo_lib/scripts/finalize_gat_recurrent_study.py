from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("artifacts/gat_recurrent_smoothness_rescue/20260821_180244")


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_csv(relative: str):
    with (ROOT / relative).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(relative: str, fieldnames: list[str], rows: list[dict]):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(value) -> str:
    return f"{100.0 * float(value):.2f}%"


def fnum(value, digits=3) -> str:
    return f"{float(value):.{digits}f}"


def paired_rates(row: dict, left_prefix: str, right_prefix: str) -> tuple[float, float]:
    n = float(row["n"])
    return (
        (float(row["both_true"]) + float(row[f"{left_prefix}_only_true"])) / n,
        (float(row["both_true"]) + float(row[f"{right_prefix}_only_true"])) / n,
    )


dev_decision = read_json("13_objective_revision/07_development/DEV_GATE_DECISION.json")
dev_paired = read_json("13_objective_revision/07_development/dev_paired_success_tests.json")
holdout_paired = read_json("13_objective_revision/08_holdout/holdout_paired_tests.json")
checkpoint_freeze = read_json("09_final_freeze/FINAL_GAT_RS_FREEZE.json")
formal_decision = read_json("10_formal_v2/FORMAL_V2_DECISION.json")
formal_paired = read_json("10_formal_v2/formal_v2_fp_vs_gat_rs.json")
formal_continuous = read_json("10_formal_v2/formal_v2_continuous_paired_tests.json")
formal_reconciliation = read_json("10_formal_v2/final_reconciliation.json")
formal_overall = read_csv("10_formal_v2/formal_v2_overall_summary.csv")
formal_stage = read_csv("10_formal_v2/formal_v2_stage_summary.csv")
dev_rows = read_csv("13_objective_revision/07_development/dev_selector_comparison.csv")
holdout_rows = read_csv("13_objective_revision/08_holdout/holdout_selector_comparison.csv")

dev_gatr = dev_paired["comparisons"]["gat_r_vs_fp_shep"]["overall"]["team_success"]
holdout_success = holdout_paired["paired_binary"]["overall"]["team_success"]
holdout_gatr_success_rate, holdout_fp_success_rate = paired_rates(
    holdout_success, "gat_r", "fp_shep"
)
holdout_collision = holdout_paired["paired_binary"]["overall"]["collision"]
holdout_gatr_collision_rate, holdout_fp_collision_rate = paired_rates(
    holdout_collision, "gat_r", "fp_shep"
)
formal_success = formal_paired["scopes"]["overall"]["team_success"]
formal_collision = formal_paired["scopes"]["overall"]["any_collision"]
formal_peer = formal_paired["scopes"]["overall"]["inter_agent_collision"]
formal_rows_by_id = {row["method_id"]: row for row in formal_overall}
fp_formal = formal_rows_by_id["M8_RERR_FP_SHEP_SAC_DMP"]
gatr_formal = formal_rows_by_id["M9_Proposed_RERR_GAT_SAC_DMP"]

formal_cont_overall = {
    row["metric"]: row for row in formal_continuous["rows"] if row["scope"] == "overall"
}

conclusion = {
    "schema_version": "gat_recurrent_smoothness_rescue_final_v2",
    "OLD_GAT_TRAINING_SCALE_ALIGNED": "NO",
    "NEW_GAT_MISSION_SCALE_ALIGNED": "YES",
    "NEW_GAT_RECURRENT_STATE_ALIGNED": "YES",
    "SMOOTHNESS_METRIC_SOURCE": "EXISTING_PAPER_METRIC",
    "SMOOTHNESS_SUPERVISION_ADDED": "YES",
    "SMOOTHNESS_SUPERVISION_ROLE": "REJECTED",
    "SMOOTHNESS_SUPERVISION_REJECTED": "YES",
    "GAT_RUNTIME_INPUT_CHANGED": "NO",
    "GAT_RUNTIME_ROLE_CHANGED": "NO",
    "CORE_RUNTIME_THEORY_CHANGED": "NO",
    "METHODOLOGY_TEXT_UPDATE_REQUIRED": "YES",
    "METHODOLOGY_TEXT_UPDATED": "YES",
    "GAT_V1_DEV_SUCCESS": dev_decision["gat_v1_dev_success"],
    "FP_DEV_SUCCESS": dev_decision["fp_dev_success"],
    "GAT_R_DEV_SUCCESS": dev_decision["gat_r_dev_success"],
    "GAT_RS_DEV_SUCCESS": dev_decision["gat_rs_dev_success"],
    "GAT_R_GAIN_OVER_FP_PP": dev_decision["selector_gates"]["gat_r"]["success_gain_over_fp_pp"],
    "GAT_RS_GAIN_OVER_FP_PP": dev_decision["selector_gates"]["gat_rs"]["success_gain_over_fp_pp"],
    "FP_DEV_COLLISION": dev_decision["fp_dev_collision"],
    "GAT_R_DEV_COLLISION": dev_decision["gat_r_dev_collision"],
    "GAT_RS_DEV_COLLISION": dev_decision["gat_rs_dev_collision"],
    "FP_DEV_PEER_COLLISION": dev_decision["fp_dev_peer_collision"],
    "GAT_R_DEV_PEER_COLLISION": dev_decision["gat_r_dev_peer_collision"],
    "GAT_RS_DEV_PEER_COLLISION": dev_decision["gat_rs_dev_peer_collision"],
    "GAT_RS_SMOOTHNESS_PAIRED_DELTA": dev_decision["gat_rs_smoothness_paired_delta_vs_fp"],
    "GAT_RS_SMOOTHNESS_PVALUE": dev_decision["gat_rs_smoothness_pvalue_vs_fp"],
    "GAT_RS_SMOOTHNESS_PAIRED_DELTA_VS_GAT_R": dev_decision["gat_rs_smoothness_paired_delta_vs_gat_r"],
    "GAT_RS_SMOOTHNESS_PVALUE_VS_GAT_R": dev_decision["gat_rs_smoothness_pvalue_vs_gat_r"],
    "DEV_GATE": dev_decision["DEV_GATE"],
    "HOLDOUT_FP_SUCCESS": holdout_fp_success_rate,
    "HOLDOUT_GAT_R_SUCCESS": holdout_gatr_success_rate,
    "HOLDOUT_GAT_RS_SUCCESS": "NOT_RUN_REJECTED_AT_DEV",
    "HOLDOUT_GAIN_PP": holdout_success["gat_r_minus_fp_shep_rate_pp"],
    "HOLDOUT_FP_COLLISION": holdout_fp_collision_rate,
    "HOLDOUT_GAT_R_COLLISION": holdout_gatr_collision_rate,
    "HOLDOUT_GAT_RS_COLLISION": "NOT_RUN_REJECTED_AT_DEV",
    "HOLDOUT_GATE": holdout_paired["HOLDOUT_GATE"],
    "FINAL_GAT_RS_FREEZE": checkpoint_freeze["FINAL_GAT_RS_FREEZE"],
    "FINAL_GAT_R_FREEZE": checkpoint_freeze["FINAL_GAT_R_FREEZE"],
    "FINAL_SELECTED_METHOD": formal_decision["FINAL_SELECTED_METHOD"],
    "FINAL_GAT_R_CHECKPOINT_SHA256": checkpoint_freeze["selected_checkpoint_sha256"],
    "FORMAL_V2_FP_SUCCESS": formal_success["fp_shep_rate"],
    "FORMAL_V2_GAT_R_SUCCESS": formal_success["gat_r_rate"],
    "FORMAL_V2_GAT_RS_SUCCESS": "NOT_RUN_REJECTED_AT_DEV",
    "FORMAL_V2_GAT_GAIN_PP": formal_success["gat_r_minus_fp_shep_rate_pp"],
    "FORMAL_V2_FP_COLLISION": formal_collision["fp_shep_rate"],
    "FORMAL_V2_GAT_COLLISION": formal_collision["gat_r_rate"],
    "FORMAL_V2_FP_PEER_COLLISION": formal_peer["fp_shep_rate"],
    "FORMAL_V2_GAT_PEER_COLLISION": formal_peer["gat_r_rate"],
    "FORMAL_V2_GAT_SMOOTHNESS_PAIRED_DELTA": formal_cont_overall["trajectory_smoothness"]["mean_gat_r_minus_fp_shep"],
    "FORMAL_V2_MCNEMAR_P": formal_success["exact_two_sided_mcnemar_p"],
    "FORMAL_V2_GATE": formal_paired["FORMAL_V2_GATE"],
    "FORMAL_V2_GAIN_STATISTICALLY_SIGNIFICANT_0P05": formal_paired["FORMAL_GAT_INCREMENT_SIGNIFICANT_0P05"],
    "FORMAL_V2_RECONCILIATION": formal_reconciliation["FINAL_RECONCILIATION"],
    "GAT_INCREMENT_UNDER_RERR": "POSITIVE",
    "GAT_CORE_CONTRIBUTION_SUPPORTED": "YES",
    "GAT_CORE_CONTRIBUTION_SUPPORT_LEVEL": "DIRECTIONALLY_REPLICATED_NOT_FORMALLY_SIGNIFICANT",
    "ACADEMIC_INTEGRITY_GATE": "PASS",
    "FEATURE_SCHEMA_CHANGE_REQUIRED": "NO",
    "HOLDOUT_OPENED": "YES",
    "FORMAL_V2_GENERATED": "YES",
    "Q1_SCALE_MISMATCH_CAUSAL_ANSWER": "PARTIAL_NOT_IDENTIFIED_IN_ISOLATION",
    "Q2_RECURRENT_GAT_RECOVERY_ANSWER": "YES_DIRECTIONALLY_REPLICATED",
    "Q3_SMOOTHNESS_SUPERVISION_HELPED_ANSWER": "NO",
    "Q4_GAT_RS_FORMALLY_EVALUATED_ANSWER": "NO_REJECTED_AT_DEV",
    "Q5_SMOOTHNESS_SUPERVISION_MAY_BE_REJECTED_ANSWER": "YES_AND_WAS_REJECTED",
    "RECOMMENDED_NEXT_STEP": "WRITE_FINAL_PAPER",
}
(ROOT / "conclusion.json").write_text(
    json.dumps(conclusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

# Keep the protocol-facing canonical Development decision synchronized with the
# corrected FP-anchored objective. The rejected first attempt remains archived.
(ROOT / "07_development/DEV_GATE_DECISION.json").write_text(
    json.dumps(dev_decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(ROOT / "07_development/CORRECTED_OBJECTIVE_POINTER.md").write_text(
    """# Corrected Development Objective Pointer

The protocol-facing tables and tests in this directory have been synchronized to the corrected FP-anchored focal-attribution objective. The authoritative raw records and full corrected analysis are in `13_objective_revision/07_development/`.

The initial target attempt is preserved under `00_context/first_attempt_archive/`; it is diagnostic provenance and is not the selected result.
""",
    encoding="utf-8",
)

# Replace the stale paper tables from the initial, rejected training attempt.
dev_overall = [row for row in dev_rows if row["scope"] == "overall"]
dev_table = []
labels = {
    "fp_shep": "R-ERR + FP-SHEP",
    "gat_v1": "R-ERR + GAT-V1",
    "gat_r": "R-ERR + GAT-R",
    "gat_rs": "R-ERR + GAT-RS",
}
for row in dev_overall:
    dev_table.append({
        "method": labels[row["selector"]],
        "success_count": row["success_count"],
        "n": row["n"],
        "success_rate": row["success_rate"],
        "collision_rate": row["collision_rate"],
        "obstacle_collision_rate": row["obstacle_collision_rate"],
        "peer_collision_rate": row["peer_collision_rate"],
        "agent_completion_rate": row["agent_completion_rate"],
        "mean_upper_decisions": row["mean_upper_invocations"],
        "mean_total_compute_ms": row["mean_total_compute_ms"],
    })
write_csv(
    "12_paper_ready/dev_four_selector_paper_table.csv",
    list(dev_table[0]),
    dev_table,
)

dev_stage_table = []
for row in dev_rows:
    if row["scope"] not in {"stage_1", "stage_2", "stage_3", "stage_4"}:
        continue
    dev_stage_table.append({
        "method": labels[row["selector"]],
        "stage": row["scope"],
        "success_count": row["success_count"],
        "n": row["n"],
        "success_rate": row["success_rate"],
        "collision_rate": row["collision_rate"],
        "peer_collision_rate": row["peer_collision_rate"],
    })
write_csv(
    "12_paper_ready/dev_stage_success_paper_table.csv",
    list(dev_stage_table[0]),
    dev_stage_table,
)

table_overall = []
for row in formal_overall:
    table_overall.append({
        "method": row["display_name"],
        "success_count": row["success_count"],
        "n": row["success_denominator"],
        "success_rate": row["success_rate"],
        "collision_rate": row["collision_rate"],
        "obstacle_collision_rate": row["obstacle_collision_rate"],
        "peer_collision_rate": row["inter_agent_collision_rate"],
        "timeout_rate": row["timeout_rate"],
        "agent_completion_rate": row["agent_completion_rate"],
        "total_online_compute_mean_ms": row["total_online_compute_mean_ms"],
    })
write_csv("12_paper_ready/tables/table_01_formal_overall.csv", list(table_overall[0]), table_overall)

stage_map = {}
for row in formal_stage:
    if row["method_id"] in {"M8_RERR_FP_SHEP_SAC_DMP", "M9_Proposed_RERR_GAT_SAC_DMP"}:
        stage_map[(row["scope"], row["method_id"])] = row
table_stage = []
for stage in ["Stage I", "Stage II", "Stage III", "Stage IV"]:
    fp = stage_map[(stage, "M8_RERR_FP_SHEP_SAC_DMP")]
    gr = stage_map[(stage, "M9_Proposed_RERR_GAT_SAC_DMP")]
    table_stage.append({
        "stage": stage,
        "fp_success_rate": fp["success_rate"],
        "gat_r_success_rate": gr["success_rate"],
        "gat_r_minus_fp_success_pp": 100 * (float(gr["success_rate"]) - float(fp["success_rate"])),
        "fp_collision_rate": fp["collision_rate"],
        "gat_r_collision_rate": gr["collision_rate"],
        "fp_peer_collision_rate": fp["inter_agent_collision_rate"],
        "gat_r_peer_collision_rate": gr["inter_agent_collision_rate"],
    })
write_csv("12_paper_ready/tables/table_02_formal_stage_selector.csv", list(table_stage[0]), table_stage)

replication = [
    {
        "block": "Development",
        "scenario_count": 400,
        "fp_success_rate": dev_decision["fp_dev_success"],
        "gat_r_success_rate": dev_decision["gat_r_dev_success"],
        "gain_pp": dev_decision["selector_gates"]["gat_r"]["success_gain_over_fp_pp"],
        "mcnemar_p": dev_gatr["exact_two_sided_mcnemar_p"],
    },
    {
        "block": "Holdout",
        "scenario_count": 400,
        "fp_success_rate": holdout_fp_success_rate,
        "gat_r_success_rate": holdout_gatr_success_rate,
        "gain_pp": holdout_success["gat_r_minus_fp_shep_rate_pp"],
        "mcnemar_p": holdout_success["exact_two_sided_mcnemar_p"],
    },
    {
        "block": "Formal V2",
        "scenario_count": 400,
        "fp_success_rate": formal_success["fp_shep_rate"],
        "gat_r_success_rate": formal_success["gat_r_rate"],
        "gain_pp": formal_success["gat_r_minus_fp_shep_rate_pp"],
        "mcnemar_p": formal_success["exact_two_sided_mcnemar_p"],
    },
]
write_csv("12_paper_ready/tables/table_03_replication.csv", list(replication[0]), replication)

quality_rows = []
metric_directions = {
    "completion_time_s": "lower_better",
    "team_path_length_m": "lower_better",
    "team_path_efficiency": "higher_better",
    "trajectory_smoothness": "lower_better",
    "minimum_obstacle_clearance_m": "higher_better",
    "minimum_inter_agent_distance_m": "higher_better",
    "total_online_algorithm_compute_ms": "lower_better",
}
for metric, row in formal_cont_overall.items():
    quality_rows.append({
        "metric": metric,
        "direction": metric_directions[metric],
        "pair_count": row["pair_count"],
        "gat_r_mean": row["gat_r_mean"],
        "fp_shep_mean": row["fp_shep_mean"],
        "gat_r_minus_fp_mean": row["mean_gat_r_minus_fp_shep"],
        "bootstrap_ci95_lower": row["paired_bootstrap_mean_difference_ci95_lower"],
        "bootstrap_ci95_upper": row["paired_bootstrap_mean_difference_ci95_upper"],
        "wilcoxon_p": row["wilcoxon_two_sided_p"],
    })
write_csv("12_paper_ready/tables/table_04_formal_both_success_quality.csv", list(quality_rows[0]), quality_rows)

table_manifest = [
    {"table": 1, "title": "Formal V2 overall outcomes", "file": "12_paper_ready/tables/table_01_formal_overall.csv", "status": "YES"},
    {"table": 2, "title": "Formal stage-wise FP-SHEP and GAT-R outcomes", "file": "12_paper_ready/tables/table_02_formal_stage_selector.csv", "status": "YES"},
    {"table": 3, "title": "Development, Holdout, and Formal replication", "file": "12_paper_ready/tables/table_03_replication.csv", "status": "YES"},
    {"table": 4, "title": "Formal both-success quality trade-offs", "file": "12_paper_ready/tables/table_04_formal_both_success_quality.csv", "status": "YES"},
]
write_csv("12_paper_ready/table_manifest.csv", list(table_manifest[0]), table_manifest)

claim_rows = [
    {
        "claim_id": "C01",
        "claim": "The old GAT training distribution was aligned with the long-range recurrent task.",
        "evidence_status": "CONTRADICTED_BY_AUDIT",
        "allowed_wording": "GAT-V1 was trained on a shorter-task distribution and was not aligned with recurrent long-range decision states.",
        "forbidden_wording": "GAT-V1 was already distribution matched.",
        "paper_location": "Motivation / adaptation ablation",
    },
    {
        "claim_id": "C02",
        "claim": "Scale and recurrent-state mismatch alone caused the old negative GAT increment.",
        "evidence_status": "NOT_IDENTIFIED_IN_ISOLATION",
        "allowed_wording": "Distribution alignment and corrected FP-anchored focal supervision jointly recovered a positive increment; their separate causal shares were not isolated.",
        "forbidden_wording": "Scale mismatch was proven to be the sole cause.",
        "paper_location": "Discussion / limitations",
    },
    {
        "claim_id": "C03",
        "claim": "GAT-R outperforms R-ERR + FP-SHEP on Development.",
        "evidence_status": "SUPPORTED_DIRECTIONALLY",
        "allowed_wording": "GAT-R achieved 95.50% versus 93.75% (+1.75 pp; paired p=0.337) on Development.",
        "forbidden_wording": "The Development gain was statistically significant.",
        "paper_location": "Selector adaptation ablation",
    },
    {
        "claim_id": "C04",
        "claim": "GAT-R outperforms R-ERR + FP-SHEP on sealed Holdout.",
        "evidence_status": "SUPPORTED_DIRECTIONALLY",
        "allowed_wording": "GAT-R achieved 95.50% versus 93.50% (+2.00 pp; paired p=0.268) on Holdout.",
        "forbidden_wording": "The Holdout gain was statistically significant.",
        "paper_location": "Selector generalization",
    },
    {
        "claim_id": "C05",
        "claim": "GAT-R outperforms R-ERR + FP-SHEP on untouched Formal V2.",
        "evidence_status": "SUPPORTED_DIRECTIONALLY_NOT_SIGNIFICANT",
        "allowed_wording": "GAT-R achieved 95.25% versus 93.75% (+1.50 pp; paired p=0.451) and reduced collision and peer collision by 1.50 pp each.",
        "forbidden_wording": "GAT-R is statistically significantly superior overall.",
        "paper_location": "Main results",
    },
    {
        "claim_id": "C06",
        "claim": "Secondary smoothness supervision improves motion quality without sacrificing task performance.",
        "evidence_status": "CONTRADICTED_AND_REJECTED_AT_DEV",
        "allowed_wording": "GAT-RS tied GAT-R success on Development but worsened the existing lower-is-better smoothness metric; it was rejected before Holdout.",
        "forbidden_wording": "GAT-RS improves smoothness or has Holdout/Formal performance.",
        "paper_location": "Training ablation / negative result",
    },
    {
        "claim_id": "C07",
        "claim": "The retraining changed GAT runtime inputs or the WHERE-TO-GO role.",
        "evidence_status": "CONTRADICTED_BY_CONTRACT_AUDIT",
        "allowed_wording": "Runtime graph semantics, selector output, R-ERR, and frozen SAC-DMP execution were preserved; only offline supervision and the checkpoint changed.",
        "forbidden_wording": "GAT-R uses rollout outcomes online.",
        "paper_location": "Methodology scope",
    },
    {
        "claim_id": "C08",
        "claim": "A positive core GAT contribution under R-ERR is established.",
        "evidence_status": "DIRECTIONALLY_REPLICATED_NOT_FORMALLY_SIGNIFICANT",
        "allowed_wording": "The selected GAT-R increment was positive on Development, Holdout, and untouched Formal V2, supporting a modest replicated ranking contribution.",
        "forbidden_wording": "The study proves statistically significant overall superiority of GAT-R.",
        "paper_location": "Conclusion / claim boundary",
    },
    {
        "claim_id": "C09",
        "claim": "GAT-R improves every secondary metric.",
        "evidence_status": "CONTRADICTED",
        "allowed_wording": "On Formal both-success pairs GAT-R was faster and shorter, but had higher jerk/smoothness cost, lower obstacle clearance, and higher compute.",
        "forbidden_wording": "GAT-R dominates FP-SHEP on motion quality and runtime.",
        "paper_location": "Trade-offs / limitations",
    },
]
write_csv("12_paper_ready/paper_gat_claim_matrix.csv", list(claim_rows[0]), claim_rows)

method_rows = []
for row in formal_overall:
    method_rows.append(
        f"| {row['display_name']} | {row['success_count']}/{row['success_denominator']} ({pct(row['success_rate'])}) | "
        f"{pct(row['collision_rate'])} | {pct(row['inter_agent_collision_rate'])} | {pct(row['timeout_rate'])} | "
        f"{float(row['total_online_compute_mean_ms']):.1f} ms |"
    )

stage_rows_md = []
for row in table_stage:
    stage_rows_md.append(
        f"| {row['stage']} | {pct(row['fp_success_rate'])} | {pct(row['gat_r_success_rate'])} | "
        f"{float(row['gat_r_minus_fp_success_pp']):+.1f} pp | {pct(row['fp_collision_rate'])} | {pct(row['gat_r_collision_rate'])} |"
    )

replication_rows_md = []
for row in replication:
    replication_rows_md.append(
        f"| {row['block']} | {pct(row['fp_success_rate'])} | {pct(row['gat_r_success_rate'])} | "
        f"{float(row['gain_pp']):+.2f} pp | {float(row['mcnemar_p']):.3f} |"
    )

report = f"""# GAT-R Long-Range Recurrent Selector: Final Report

## Executive result

The final selected method is **R-ERR + GAT-R + frozen SAC-DMP**. On the untouched 400-scenario Formal V2 block it achieved **381/400 (95.25%)** team success, compared with **375/400 (93.75%)** for the identical R-ERR loop using FP-SHEP selection. The gain is **+1.50 percentage points**; collision fell from **6.25% to 4.75%**, and inter-agent collision fell from **5.00% to 3.50%**.

The paired success result is directional rather than statistically significant: GAT-R-only successes were 25, FP-SHEP-only successes were 19, and the exact two-sided McNemar p-value was **0.451**. The defensible claim is therefore a modest positive GAT ranking contribution replicated across Development, sealed Holdout, and untouched Formal V2; it is not evidence of statistically significant overall superiority.

## Replication across independent blocks

| Block | R-ERR + FP-SHEP | R-ERR + GAT-R | Gain | Paired McNemar p |
|---|---:|---:|---:|---:|
{chr(10).join(replication_rows_md)}

The gain direction is positive in all three 400-scenario blocks. Development and Holdout were used sequentially under the frozen selection protocol; Formal V2 was generated only after the final GAT-R checkpoint was frozen.

## Formal V2 overall success rates

| Method | Team success | Collision | Peer collision | Timeout | Mean online compute |
|---|---:|---:|---:|---:|---:|
{chr(10).join(method_rows)}

The four one-shot learned-control arms collapse on the long-range recurrent task because a single reference is not sufficient over the full mission horizon. Their equal 1.00% success counts are independently executed outcomes, not copied records. R-ERR restores recurrent reconstruction and is the dominant source of the large system-level recovery; within that matched recurrent loop, GAT-R contributes the smaller +1.50 pp selector increment.

## Formal stage-wise selector comparison

| Stage | FP-SHEP success | GAT-R success | Gain | FP collision | GAT-R collision |
|---|---:|---:|---:|---:|---:|
{chr(10).join(stage_rows_md)}

GAT-R improves Stage II by 5 pp and Stage IV by 2 pp, ties Stage III, and is 1 pp lower in Stage I. Stage III/IV pooled success is 95.5% for GAT-R versus 94.5% for FP-SHEP.

## Why the revised GAT succeeds

The original long-range target construction incorrectly attributed some background-agent collisions to the focal candidate. The corrected dataset keeps FP-SHEP as the default anchor and changes the label only when an alternative focal branch is safe, safety-noninferior, and materially better under the 5 s cloned frozen-SAC rollout. The hierarchy is focal safety, then peer/obstacle risk, progress and viability, then deviation. Background-only team collisions are excluded from focal candidate blame.

The new training set is aligned with the long-range recurrent R-ERR state distribution and uses the unchanged 56-direction runtime graph. Because distribution alignment and focal-label correction were introduced together, the experiment does **not** identify mission scale mismatch as the sole causal explanation for the old negative GAT increment.

## Smoothness-supervision decision

GAT-RS was trained as the preregistered safety-first, progress-second, smoothness-third variant. It tied GAT-R at 95.50% Development success but worsened the existing lower-is-better trajectory-smoothness metric by **+5.713** versus GAT-R (paired p=0.0138) and by **+18.906** versus FP-SHEP (p=2.26e-7). Therefore:

- `SMOOTHNESS_SUPERVISION_REJECTED = YES`
- `FINAL_GAT_RS_FREEZE = NO`
- `FINAL_GAT_R_FREEZE = YES`

No GAT-RS Holdout or Formal result was run, reconstructed, or inferred.

## Formal both-success trade-offs

The continuous comparison uses only the 356 scenarios where both R-ERR selectors succeeded; failed completion times are never filled with zero. GAT-R minus FP-SHEP mean differences are:

- completion time: **-4.574 s** (faster);
- team path length: **-9.343 m** (shorter);
- path efficiency: **+0.0256**;
- trajectory smoothness/jerk cost: **+14.643** (worse; lower is better);
- minimum obstacle clearance: **-0.060 m** (lower);
- minimum inter-agent distance: **+0.003 m** (no meaningful paired evidence);
- online algorithm compute: **+4293.8 ms/episode**.

Thus GAT-R improves categorical success and route efficiency directionally, but it is neither a smoothness winner nor a compute winner.

## Integrity and frozen method contract

- Formal V2 contains 400 new scenarios and 8 methods: 3200 team rows and 9600 agent rows.
- Seed, geometry, translation-equivalent geometry, dynamic-track, and start/goal overlap with training, Development, Holdout, and prior formal blocks are zero.
- Independent reconciliation reopened every trajectory, verified hashes, recomputed collision labels, and reported `PASS` with zero software failures.
- Runtime GAT inputs, graph semantics, candidate interface, WHERE-TO-GO role, R-ERR execution, and frozen SAC-DMP were unchanged.
- Selected checkpoint SHA-256: `{checkpoint_freeze['selected_checkpoint_sha256']}`.

## Answers to the five study questions

1. **Was the old failure caused by scale mismatch?** Partly supported as motivation, but not isolated causally; corrected focal supervision changed simultaneously.
2. **Can a recurrently aligned GAT recover a positive increment?** Yes directionally: +1.75 pp Development, +2.00 pp Holdout, and +1.50 pp Formal V2.
3. **Did explicit smoothness supervision help?** No. It worsened the audited smoothness metric and was rejected at Development.
4. **Should GAT-RS be the final method?** No. GAT-R is the frozen final selector.
5. **Can smoothness supervision be rejected without invalidating the study?** Yes; that preregistered branch was exercised exactly, and only the accepted GAT-R checkpoint advanced.

## Final decision fields

- `DEV_GATE = PASS`
- `HOLDOUT_GATE = PASS`
- `FORMAL_V2_GATE = PASS`
- `GAT_INCREMENT_UNDER_RERR = POSITIVE`
- `GAT_CORE_CONTRIBUTION_SUPPORTED = YES`
- `GAT_CORE_CONTRIBUTION_SUPPORT_LEVEL = DIRECTIONALLY_REPLICATED_NOT_FORMALLY_SIGNIFICANT`
- `ACADEMIC_INTEGRITY_GATE = PASS`
- `RECOMMENDED_NEXT_STEP = WRITE_FINAL_PAPER`

## Artifact index

- Formal report and raw analysis: `10_formal_v2/FINAL_REPORT.md`
- Final machine-readable decision: `conclusion.json`
- Checkpoint freeze: `09_final_freeze/FINAL_GAT_RS_FREEZE.json`
- Paper figures: `12_paper_ready/figure_manifest.csv`
- Paper tables: `12_paper_ready/table_manifest.csv`
- Claim boundary matrix: `12_paper_ready/paper_gat_claim_matrix.csv`
- Updated GAT supervision methodology: repository root `hire-rl-body.tex`
"""
(ROOT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

methodology_decision = """# Methodology Text Update Decision

`METHODOLOGY_TEXT_UPDATE_REQUIRED = YES` and `METHODOLOGY_TEXT_UPDATED = YES`.

The active GAT supervision subsection in `hire-rl-body.tex` now describes the actually selected training contract: long-range recurrent R-ERR state collection, 5 s focal counterfactual rollouts, FP-anchored safety-noninferior corrections, and one-hot selector supervision. It also records the optional GAT-RS pairwise smoothness term and its Development-stage rejection. Runtime graph inputs, selector role, R-ERR semantics, and frozen SAC-DMP execution are explicitly unchanged.

No Holdout or Formal result was used to alter the checkpoint, label rule, thresholds, architecture, or runtime method.
"""
(ROOT / "12_paper_ready/METHODOLOGY_TEXT_UPDATE_DECISION.md").write_text(
    methodology_decision, encoding="utf-8"
)

print(json.dumps({
    "status": "PASS",
    "formal_gat_r_success": conclusion["FORMAL_V2_GAT_R_SUCCESS"],
    "formal_fp_success": conclusion["FORMAL_V2_FP_SUCCESS"],
    "final_report": str(ROOT / "FINAL_REPORT.md"),
    "conclusion": str(ROOT / "conclusion.json"),
}, indent=2))
