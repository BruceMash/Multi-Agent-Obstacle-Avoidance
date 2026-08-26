#!/usr/bin/env python3
"""Finalize the gated GAT-R/GAT-RS experiment without opening Holdout.

This script is intentionally downstream of the four-selector Dev analysis.  It
creates explicit NOT_RUN artifacts for phases forbidden by the failed Dev gate,
the mandatory conclusion schema, paper claim boundaries, and the final report.
It never imports or evaluates Holdout scene contents.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
DEV = ROOT / "07_development"
HOLDOUT = ROOT / "08_holdout"
FREEZE = ROOT / "09_final_freeze"
FORMAL = ROOT / "10_formal_v2"
STATS = ROOT / "11_statistics"
PAPER = ROOT / "12_paper_ready"
SCHEMA = "gat_recurrent_smoothness_rescue_final_v1"

ARMS = {
    "fp_shep": DEV / "FP_SHEP_RERR_DEV400/development_team_results.csv",
    "gat_v1": DEV / "GAT_V1_RERR_DEV400/development_team_results.csv",
    "gat_r": DEV / "GAT_R_RERR_DEV400/development_team_results.csv",
    "gat_rs": DEV / "GAT_RS_RERR_DEV400/development_team_results.csv",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flag(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(rows: list[dict[str, str]]) -> dict[str, Any]:
    n = len(rows)
    success = sum(flag(row["team_success"]) for row in rows)
    collision = sum(flag(row["collision"]) for row in rows)
    obstacle = sum(flag(row["obstacle_collision"]) for row in rows)
    peer = sum(flag(row["inter_agent_collision"]) for row in rows)
    timeout = sum(flag(row["timeout"]) for row in rows)
    return {
        "n": n,
        "success_count": success,
        "success_rate": success / n,
        "collision_count": collision,
        "collision_rate": collision / n,
        "obstacle_collision_count": obstacle,
        "obstacle_collision_rate": obstacle / n,
        "peer_collision_count": peer,
        "peer_collision_rate": peer / n,
        "timeout_count": timeout,
        "timeout_rate": timeout / n,
        "agent_completion_rate": sum(float(row["agent_completion_rate"]) for row in rows) / n,
        "mean_upper_invocations": sum(float(row["upper_pipeline_invocation_count"]) for row in rows) / n,
        "mean_total_compute_ms": sum(float(row["total_online_algorithm_compute_ms"]) for row in rows) / n,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    for directory in (HOLDOUT, FREEZE, FORMAL, STATS, PAPER):
        directory.mkdir(parents=True, exist_ok=True)

    decision = read_json(DEV / "DEV_GATE_DECISION.json")
    paired = read_json(DEV / "dev_paired_success_tests.json")
    continuous = read_json(DEV / "dev_both_success_continuous_tests.json")
    split = read_json(ROOT / "02_recurrent_dataset/independent_split_reconciliation.json")
    dataset = read_json(ROOT / "02_recurrent_dataset/recurrent_dataset_reconciliation.json")
    metric = read_json(ROOT / "01_smoothness_metric_audit/smoothness_metric_definition.json")
    labels = read_json(ROOT / "03_counterfactual_rollouts/COUNTERFACTUAL_LABEL_FREEZE.json")
    goal_audit = read_json(ROOT / "04_feature_alignment/goal_distance_distribution_audit.json")
    gat_r_manifest = read_json(ROOT / "05_gat_r_training/GAT_R_TRAINING_MANIFEST.json")
    gat_rs_manifest = read_json(ROOT / "06_gat_rs_training/GAT_RS_TRAINING_MANIFEST.json")

    assert decision["status"] == "PASS", "Dev analysis integrity did not pass"
    assert decision["DEV_GATE"] == "FAIL", "Finalizer is only for the failed Dev branch"
    assert decision["holdout_opened"] is False
    assert decision["formal_v2_generated"] is False
    assert split["status"] == "PASS"
    assert dataset["status"] == "PASS"
    assert split["performance_rows_read"] == 0
    assert split["old_formal_v1_episode_rows_read"] == 0
    assert dataset["formal_v1_used_for_training"] is False
    assert dataset["dev_used_for_training_or_label_selection"] is False
    assert metric["source"] == "EXISTING_PAPER_METRIC"
    assert labels["offline_future_information_at_runtime"] is False
    assert goal_audit["frozen_first_round_decision"]["feature_dimension_changed"] is False
    assert gat_r_manifest["gat_runtime_input_changed"] is False
    assert gat_rs_manifest["gat_runtime_input_changed"] is False
    assert gat_r_manifest["gat_runtime_role_changed"] is False
    assert gat_rs_manifest["gat_runtime_role_changed"] is False

    raw = {name: read_csv(path) for name, path in ARMS.items()}
    identities: set[tuple[str, str, str, str]] | None = None
    summaries: dict[str, dict[str, Any]] = {}
    stages: dict[str, dict[str, dict[str, Any]]] = {}
    for name, rows in raw.items():
        assert len(rows) == 400, f"{name} has {len(rows)} rows"
        assert all(not flag(row["software_error"]) for row in rows)
        assert all(flag(row["scene_reconstruction_match"]) for row in rows)
        keys = {(r["scenario_id"], r["seed"], r["stage"], r["family"]) for r in rows}
        assert len(keys) == 400
        if identities is None:
            identities = keys
        else:
            assert keys == identities, f"paired identity mismatch in {name}"
        summaries[name] = aggregate(rows)
        stages[name] = {}
        for stage in ("stage_1", "stage_2", "stage_3", "stage_4"):
            subset = [row for row in rows if row["stage"] == stage]
            assert len(subset) == 100
            stages[name][stage] = aggregate(subset)

    expected = {
        "fp_shep": (375, 25, 21),
        "gat_v1": (358, 42, 32),
        "gat_r": (326, 74, 71),
        "gat_rs": (355, 45, 42),
    }
    for name, (success, collision, peer) in expected.items():
        assert summaries[name]["success_count"] == success
        assert summaries[name]["collision_count"] == collision
        assert summaries[name]["peer_collision_count"] == peer

    # Checkpoint bytes must still equal the selected immutable evidence.
    checkpoint_checks = {}
    for name, manifest in (("gat_r", gat_r_manifest), ("gat_rs", gat_rs_manifest)):
        path = REPO_ROOT / manifest["selected_checkpoint"]
        observed = sha256(path)
        expected_hash = manifest["selected_checkpoint_sha256"]
        checkpoint_checks[name] = {
            "path": manifest["selected_checkpoint"],
            "expected_sha256": expected_hash,
            "observed_sha256": observed,
            "match": observed == expected_hash,
        }
        assert observed == expected_hash

    rs_fp = paired["comparisons"]["gat_rs_vs_fp_shep"]["overall"]
    rs_r = paired["comparisons"]["gat_rs_vs_gat_r"]["overall"]
    smooth_fp = continuous["comparisons"]["gat_rs_vs_fp_shep"]["metrics"]["trajectory_smoothness"]
    smooth_r = continuous["comparisons"]["gat_rs_vs_gat_r"]["metrics"]["trajectory_smoothness"]

    tex = (REPO_ROOT / "hire-rl-body.tex").read_text(encoding="utf-8")
    old_loss_explicit = all(
        token in tex
        for token in (
            "eq:unified_execution_quality",
            "eq:execution_quality_soft_label",
            "eq:candidate_selection_loss",
            "eq:spatiotemporal_overlap_loss",
        )
    )
    methodology_update_required = old_loss_explicit

    not_run_reason = (
        "Dev hard gate failed: GAT-RS success was not greater than FP-SHEP and "
        "both total and peer collision rates were higher."
    )

    # Explicit stop artifacts: these files document absence rather than invent results.
    write_csv(
        HOLDOUT / "holdout_selector_comparison.csv",
        ["status", "performance_row_count", "reason"],
        [{"status": "NOT_RUN", "performance_row_count": 0, "reason": not_run_reason}],
    )
    write_json(
        HOLDOUT / "holdout_paired_tests.json",
        {
            "schema_version": SCHEMA,
            "status": "NOT_RUN",
            "performance_row_count": 0,
            "holdout_manifest_exists_but_performance_was_not_opened": True,
            "reason": not_run_reason,
        },
    )
    write_json(
        FREEZE / "FINAL_GAT_RS_FREEZE.json",
        {
            "schema_version": SCHEMA,
            "status": "NOT_FROZEN",
            "FINAL_GAT_RS_FREEZE": "NO",
            "reason": not_run_reason,
            "rejected_experimental_checkpoint": checkpoint_checks["gat_rs"],
            "holdout_gate": "NOT_RUN",
        },
    )
    write_json(
        FORMAL / "FORMAL_V2_MANIFEST.json",
        {
            "schema_version": SCHEMA,
            "status": "NOT_GENERATED",
            "scenario_count": 0,
            "team_episode_count": 0,
            "agent_result_count": 0,
            "reason": "Holdout was forbidden because the Dev gate failed.",
        },
    )
    write_csv(
        FORMAL / "formal_v2_team_results.csv",
        ["scenario_id", "method", "team_success", "collision", "inter_agent_collision", "timeout"],
        [],
    )
    write_csv(
        FORMAL / "formal_v2_agent_results.csv",
        ["scenario_id", "method", "agent_id", "completion", "collision", "completion_time_s"],
        [],
    )
    write_json(
        FORMAL / "formal_v2_fp_vs_gat_rs.json",
        {"schema_version": SCHEMA, "status": "NOT_RUN", "paired_episode_count": 0, "reason": not_run_reason},
    )
    write_json(
        FORMAL / "formal_v2_continuous_paired_tests.json",
        {"schema_version": SCHEMA, "status": "NOT_RUN", "both_success_count": 0, "reason": not_run_reason},
    )
    write_csv(
        FORMAL / "formal_v2_stage_summary.csv",
        ["status", "stage", "method", "n", "success_rate", "collision_rate"],
        [],
    )
    write_csv(
        FORMAL / "formal_v2_failure_taxonomy.csv",
        ["status", "method", "failure_type", "count", "rate"],
        [],
    )
    write_csv(
        FORMAL / "formal_v2_runtime_summary.csv",
        ["status", "method", "n", "mean_total_compute_ms", "mean_upper_decisions"],
        [],
    )

    conclusion: dict[str, Any] = {
        "schema_version": SCHEMA,
        "OLD_GAT_TRAINING_SCALE_ALIGNED": "NO",
        "NEW_GAT_MISSION_SCALE_ALIGNED": "YES",
        "NEW_GAT_RECURRENT_STATE_ALIGNED": "YES",
        "SMOOTHNESS_METRIC_SOURCE": "EXISTING_PAPER_METRIC",
        "SMOOTHNESS_SUPERVISION_ADDED": "YES",
        "SMOOTHNESS_SUPERVISION_ROLE": "REJECTED",
        "GAT_RUNTIME_INPUT_CHANGED": "NO",
        "GAT_RUNTIME_ROLE_CHANGED": "NO",
        "CORE_RUNTIME_THEORY_CHANGED": "NO",
        "METHODOLOGY_TEXT_UPDATE_REQUIRED": "YES" if methodology_update_required else "NO",
        "GAT_V1_DEV_SUCCESS": summaries["gat_v1"]["success_rate"],
        "FP_DEV_SUCCESS": summaries["fp_shep"]["success_rate"],
        "GAT_R_DEV_SUCCESS": summaries["gat_r"]["success_rate"],
        "GAT_RS_DEV_SUCCESS": summaries["gat_rs"]["success_rate"],
        "GAT_R_GAIN_OVER_FP_PP": 100.0 * (summaries["gat_r"]["success_rate"] - summaries["fp_shep"]["success_rate"]),
        "GAT_RS_GAIN_OVER_FP_PP": 100.0 * (summaries["gat_rs"]["success_rate"] - summaries["fp_shep"]["success_rate"]),
        "FP_DEV_COLLISION": summaries["fp_shep"]["collision_rate"],
        "GAT_RS_DEV_COLLISION": summaries["gat_rs"]["collision_rate"],
        "FP_DEV_PEER_COLLISION": summaries["fp_shep"]["peer_collision_rate"],
        "GAT_RS_DEV_PEER_COLLISION": summaries["gat_rs"]["peer_collision_rate"],
        "GAT_RS_SMOOTHNESS_PAIRED_DELTA": smooth_fp["gat_rs_minus_fp_shep_mean"],
        "GAT_RS_SMOOTHNESS_PVALUE": smooth_fp["wilcoxon_two_sided_p"],
        "GAT_RS_SMOOTHNESS_PAIRED_DELTA_VS_GAT_R": smooth_r["gat_rs_minus_gat_r_mean"],
        "GAT_RS_SMOOTHNESS_PVALUE_VS_GAT_R": smooth_r["wilcoxon_two_sided_p"],
        "SMOOTHNESS_SUPERVISION_REJECTED": "YES",
        "DEV_GATE": "FAIL",
        "HOLDOUT_FP_SUCCESS": "NOT_RUN",
        "HOLDOUT_GAT_RS_SUCCESS": "NOT_RUN",
        "HOLDOUT_GAIN_PP": "NOT_RUN",
        "HOLDOUT_FP_COLLISION": "NOT_RUN",
        "HOLDOUT_GAT_RS_COLLISION": "NOT_RUN",
        "HOLDOUT_GATE": "NOT_RUN",
        "FINAL_GAT_RS_FREEZE": "NO",
        "FORMAL_V2_FP_SUCCESS": "NOT_RUN",
        "FORMAL_V2_GAT_RS_SUCCESS": "NOT_RUN",
        "FORMAL_V2_GAT_GAIN_PP": "NOT_RUN",
        "FORMAL_V2_FP_COLLISION": "NOT_RUN",
        "FORMAL_V2_GAT_COLLISION": "NOT_RUN",
        "FORMAL_V2_FP_PEER_COLLISION": "NOT_RUN",
        "FORMAL_V2_GAT_PEER_COLLISION": "NOT_RUN",
        "FORMAL_V2_GAT_SMOOTHNESS_PAIRED_DELTA": "NOT_RUN",
        "FORMAL_V2_MCNEMAR_P": "NOT_RUN",
        "GAT_INCREMENT_UNDER_RERR": "NEGATIVE",
        "GAT_CORE_CONTRIBUTION_SUPPORTED": "NO",
        "ACADEMIC_INTEGRITY_GATE": "PASS",
        "RECOMMENDED_NEXT_STEP": "RECONSIDER_GAT_TRAINING_OBJECTIVE",
        "FEATURE_SCHEMA_CHANGE_REQUIRED": "NO",
        "HOLDOUT_OPENED": "NO",
        "FORMAL_V2_GENERATED": "NO",
        "GAT_RS_VS_FP_DEV_MCNEMAR_P": rs_fp["team_success"]["exact_two_sided_mcnemar_p"],
        "GAT_RS_VS_FP_PEER_COLLISION_MCNEMAR_P": rs_fp["inter_agent_collision"]["exact_two_sided_mcnemar_p"],
        "GAT_RS_GAIN_OVER_GAT_R_PP": rs_r["team_success"]["gat_rs_minus_gat_r_rate_pp"],
        "GAT_RS_CHECKPOINT_SHA256": gat_rs_manifest["selected_checkpoint_sha256"],
        "GAT_R_CHECKPOINT_SHA256": gat_r_manifest["selected_checkpoint_sha256"],
    }
    write_json(ROOT / "conclusion.json", conclusion)

    claim_rows = [
        {
            "claim_id": "C01",
            "claim": "The old GAT training distribution was aligned with the long-range recurrent task.",
            "evidence_status": "CONTRADICTED_BY_SCALE_AUDIT",
            "allowed_wording": "GAT-V1 was trained on a shorter-task distribution and uses the historical 9 m goal-distance scale.",
            "forbidden_wording": "GAT-V1 was already distribution matched.",
            "paper_location": "Motivation / adaptation ablation",
        },
        {
            "claim_id": "C02",
            "claim": "Mission-scale and recurrent-state mismatch was the primary cause of the negative GAT increment.",
            "evidence_status": "NOT_SUPPORTED",
            "allowed_wording": "The mismatch motivated retraining, but aligned GAT-R did not recover the FP-SHEP baseline.",
            "forbidden_wording": "The prior gap was caused primarily by scale mismatch.",
            "paper_location": "Discussion",
        },
        {
            "claim_id": "C03",
            "claim": "GAT-R outperforms R-ERR + FP-SHEP on Dev.",
            "evidence_status": "CONTRADICTED",
            "allowed_wording": "GAT-R achieved 81.50% versus FP-SHEP 93.75% on the independent Dev block.",
            "forbidden_wording": "Recurrent retraining recovered a positive GAT increment.",
            "paper_location": "Supplementary adaptation ablation",
        },
        {
            "claim_id": "C04",
            "claim": "Secondary smoothness supervision improves motion quality without sacrificing success or safety.",
            "evidence_status": "CONTRADICTED",
            "allowed_wording": "GAT-RS recovered 7.25 pp over GAT-R but remained 5.00 pp below FP-SHEP and had higher collision and peer-collision rates; paired smoothness was worse than GAT-R.",
            "forbidden_wording": "GAT-RS provides a stable safety-preserving smoothness gain.",
            "paper_location": "Supplementary negative result",
        },
        {
            "claim_id": "C05",
            "claim": "GAT-RS outperforms FP-SHEP in high-density stages.",
            "evidence_status": "CONTRADICTED",
            "allowed_wording": "Stage III/IV pooled success was 87.50% for GAT-RS and 93.50% for FP-SHEP.",
            "forbidden_wording": "GAT-RS has a high-density coordination advantage.",
            "paper_location": "Supplementary adaptation ablation",
        },
        {
            "claim_id": "C06",
            "claim": "The retraining changed GAT runtime inputs or its WHERE-TO-GO role.",
            "evidence_status": "CONTRADICTED_BY_CONTRACT_AUDIT",
            "allowed_wording": "Only training supervision and checkpoint-specific goal-distance normalization changed; runtime graph semantics, logits, selection role, SAC-DMP, and R-ERR were preserved.",
            "forbidden_wording": "GAT-RS uses future rollout outcomes online.",
            "paper_location": "Methodology scope",
        },
        {
            "claim_id": "C07",
            "claim": "GAT-RS is validated on Holdout or Formal V2.",
            "evidence_status": "NOT_RUN_BY_PREREGISTERED_STOP_RULE",
            "allowed_wording": "The Dev gate failed, so Holdout remained unopened and Formal V2 was not generated.",
            "forbidden_wording": "GAT-RS generalizes to unseen holdout or formal scenarios.",
            "paper_location": "Limitations / protocol",
        },
        {
            "claim_id": "C08",
            "claim": "A positive core GAT contribution under R-ERR is established.",
            "evidence_status": "NO",
            "allowed_wording": "All three evaluated GAT selectors were below the FP-SHEP selector on Dev; a positive GAT increment is not supported.",
            "forbidden_wording": "GAT is the source of the full method's performance advantage.",
            "paper_location": "Claims boundary",
        },
    ]
    write_csv(
        PAPER / "paper_gat_claim_matrix.csv",
        ["claim_id", "claim", "evidence_status", "allowed_wording", "forbidden_wording", "paper_location"],
        claim_rows,
    )

    paper_table_rows = []
    display_names = {
        "fp_shep": "R-ERR + FP-SHEP",
        "gat_v1": "R-ERR + GAT-V1",
        "gat_r": "R-ERR + GAT-R",
        "gat_rs": "R-ERR + GAT-RS",
    }
    for name in ("fp_shep", "gat_v1", "gat_r", "gat_rs"):
        summary = summaries[name]
        paper_table_rows.append(
            {
                "method": display_names[name],
                "success_count": summary["success_count"],
                "n": summary["n"],
                "success_rate": summary["success_rate"],
                "collision_rate": summary["collision_rate"],
                "obstacle_collision_rate": summary["obstacle_collision_rate"],
                "peer_collision_rate": summary["peer_collision_rate"],
                "agent_completion_rate": summary["agent_completion_rate"],
                "mean_upper_decisions": summary["mean_upper_invocations"],
                "mean_total_compute_ms": summary["mean_total_compute_ms"],
            }
        )
    write_csv(
        PAPER / "dev_four_selector_paper_table.csv",
        list(paper_table_rows[0]),
        paper_table_rows,
    )

    stage_rows = []
    for name in ("fp_shep", "gat_v1", "gat_r", "gat_rs"):
        for stage in ("stage_1", "stage_2", "stage_3", "stage_4"):
            summary = stages[name][stage]
            stage_rows.append(
                {
                    "method": display_names[name],
                    "stage": stage,
                    "success_count": summary["success_count"],
                    "n": summary["n"],
                    "success_rate": summary["success_rate"],
                    "collision_rate": summary["collision_rate"],
                    "peer_collision_rate": summary["peer_collision_rate"],
                }
            )
    write_csv(
        PAPER / "dev_stage_success_paper_table.csv",
        list(stage_rows[0]),
        stage_rows,
    )

    methodology_note = f"""# GAT methodology text decision

`METHODOLOGY_TEXT_UPDATE_REQUIRED = {'YES' if methodology_update_required else 'NO'}`.

The active Methodology explicitly defines the historical weighted execution-quality
soft label and overlap-regularized loss. GAT-R instead used safety-first hierarchical
5 s counterfactual targets, and GAT-RS added a pairwise lower-jerk preference only for
the {dataset['eligible_smoothness_pair_count']:,} safety- and progress-equivalent pairs.
Therefore any paper section reporting these experimental variants must state the
updated supervision exactly.

GAT-RS failed the Dev acceptance gate and is not adopted as the final method. The
active core Methodology must not be silently rewritten to present this rejected
experimental checkpoint as the deployed selector. The appropriate placement is a
clearly labelled adaptation/negative-result subsection or supplement.
"""
    (PAPER / "METHODOLOGY_TEXT_UPDATE_DECISION.md").write_text(methodology_note, encoding="utf-8")

    report = f"""# Long-Range Recurrent GAT-R / GAT-RS Rescue Experiment

## Executive result

The requested positive GAT increment was **not recovered**. On the independent
400-scenario Dev block, R-ERR + FP-SHEP achieved **375/400 ({pct(summaries['fp_shep']['success_rate'])})**
team success. Frozen GAT-V1 achieved **358/400 ({pct(summaries['gat_v1']['success_rate'])})**,
GAT-R achieved **326/400 ({pct(summaries['gat_r']['success_rate'])})**, and GAT-RS achieved
**355/400 ({pct(summaries['gat_rs']['success_rate'])})**.

GAT-RS remained **5.00 pp below FP-SHEP** in success, while collision increased
from **{pct(summaries['fp_shep']['collision_rate'])}** to **{pct(summaries['gat_rs']['collision_rate'])}**
and peer collision increased from **{pct(summaries['fp_shep']['peer_collision_rate'])}**
to **{pct(summaries['gat_rs']['peer_collision_rate'])}**. The preregistered Dev
success, total-safety, peer-safety, high-density, and joint smoothness gates therefore
failed. Holdout was not opened, GAT-RS was not frozen, and Formal V2 was not generated.

## 1. Integrity and scope

- Old Formal V1 is archived as read-only diagnostic provenance and contributed zero
  training, development, validation, or checkpoint-selection rows.
- New scene manifests contain 1,000 Train, 400 Dev, and 400 reserved Holdout scenes.
  Every split is balanced across four stages and five families. Seed, geometry,
  dynamic-track, start-goal, and translation-equivalent overlap are all zero.
- The counterfactual training panel executed 200 balanced recurrent episodes and
  retained {dataset['state_count']:,} recurrent states, {dataset['candidate_branch_count_primary']:,}
  primary 5 s candidate branches, {dataset['ambiguous_state_count']:,} ambiguous states,
  and {dataset['peer_conflict_candidate_row_count']:,} peer-conflict candidate rows.
- All four Dev arms used identical 400 scenarios, frozen R-ERR, Proposal, FP-SHEP H4,
  adapted SAC-DMP execution, physical limits, and outcome definitions. There were no
  software errors and all scene reconstructions matched.

## 2. Smoothness metric and supervision

The reused paper metric is mean squared jerk,
`mean_t sum_xyz(((a[t+1]-a[t])/dt)^2)`, in **m^2/s^6**; lower is better. It is
computed from applied acceleration and is an execution-quality metric, not a runtime
GAT input, primary label, or null-class criterion.

GAT-R used hierarchical 5 s counterfactual supervision ordered by hard safety,
peer/obstacle risk, long-range progress and viability, then execution deviation.
GAT-RS used the same architecture, graphs, split, and primary targets, plus a fixed
`lambda_sm=0.1` pairwise term over {dataset['eligible_smoothness_pair_count']:,} eligible
pairs. Every pair was safety- and progress-equivalent and excluded null; no weight
search was performed.

## 3. Feature and runtime contract

The historical 9 m goal-distance normalization saturated the 65--85 m task scale.
GAT-R and GAT-RS therefore use a pre-performance 100 m normalization while GAT-V1
retains its frozen 9 m scale. Feature dimension and semantic meaning did not change;
the 56-direction canonical graph projection was retained.

Runtime remains current legal graph observation -> candidate logits -> reference
selection. No future trajectory, collision, jerk, or counterfactual outcome is used
online. GAT remains the WHERE-TO-GO selector and frozen SAC-DMP remains HOW-TO-EXECUTE;
R-ERR and all execution semantics are unchanged.

## 4. Training outcomes

GAT-R selected seed {gat_r_manifest['selected_optimization_seed']} at epoch
{gat_r_manifest['selected_best_epoch']} by internal validation loss. Its checkpoint
SHA-256 is `{gat_r_manifest['selected_checkpoint_sha256']}`. Internal validation
Top-1/Top-3/MRR were 49.14%/90.09%/0.6973.

GAT-RS was trained independently from scratch and selected seed
{gat_rs_manifest['selected_optimization_seed']} at epoch {gat_rs_manifest['selected_best_epoch']}.
Its checkpoint SHA-256 is `{gat_rs_manifest['selected_checkpoint_sha256']}`. Internal
validation Top-1/Top-3/MRR were 49.14%/87.07%/0.6962; eligible-pair accuracy was 52.20%.

## 5. Four-selector Dev outcomes

| Selector | Success | Collision | Obstacle | Peer | Agent completion |
|---|---:|---:|---:|---:|---:|
| FP-SHEP | 375/400 (93.75%) | 6.25% | 1.00% | 5.25% | 94.00% |
| GAT-V1 | 358/400 (89.50%) | 10.50% | 2.50% | 8.00% | 89.83% |
| GAT-R | 326/400 (81.50%) | 18.50% | 0.75% | 17.75% | 81.58% |
| GAT-RS | 355/400 (88.75%) | 11.25% | 0.75% | 10.50% | 88.75% |

Stage success rates were:

| Selector | Stage I | Stage II | Stage III | Stage IV |
|---|---:|---:|---:|---:|
| FP-SHEP | 93% | 95% | 96% | 91% |
| GAT-V1 | 89% | 89% | 91% | 89% |
| GAT-R | 78% | 82% | 83% | 83% |
| GAT-RS | 90% | 90% | 90% | 85% |

For GAT-RS versus FP-SHEP, FP-only/GAT-RS-only successes were 42/22 and exact
two-sided McNemar `p={rs_fp['team_success']['exact_two_sided_mcnemar_p']:.6g}`.
Peer-collision discordances were FP-only/GAT-RS-only 19/40,
`p={rs_fp['inter_agent_collision']['exact_two_sided_mcnemar_p']:.6g}`. Stage III/IV pooled
success was 87.50% versus 93.50% (-6.00 pp).

GAT-RS did recover 7.25 pp success and reduce peer collision by 7.25 pp relative to
GAT-R. This is a partial recovery from a poor recurrent target, not a positive GAT
increment over FP-SHEP or GAT-V1.

## 6. Paired smoothness analysis

Continuous metrics use pair-specific both-success episodes only; failed completion
times are never filled with zero.

- GAT-RS versus FP-SHEP: n={smooth_fp['n']}, mean jerk delta
  **{smooth_fp['gat_rs_minus_fp_shep_mean']:+.3f} m^2/s^6**, bootstrap 95% CI
  [{smooth_fp['paired_bootstrap_95_ci'][0]:+.3f}, {smooth_fp['paired_bootstrap_95_ci'][1]:+.3f}],
  two-sided Wilcoxon p={smooth_fp['wilcoxon_two_sided_p']:.6g}. The CI crosses zero;
  a robust smoothness improvement is not established.
- GAT-RS versus GAT-R: n={smooth_r['n']}, mean jerk delta
  **{smooth_r['gat_rs_minus_gat_r_mean']:+.3f} m^2/s^6**, bootstrap 95% CI
  [{smooth_r['paired_bootstrap_95_ci'][0]:+.3f}, {smooth_r['paired_bootstrap_95_ci'][1]:+.3f}],
  p={smooth_r['wilcoxon_two_sided_p']:.6g}. Positive is worse, so GAT-RS is clearly
  less smooth than GAT-R on their paired-success subset.

Because the secondary term did not deliver its stated motion-quality benefit and the
full GAT-RS arm also failed success and safety gates against FP-SHEP,
`SMOOTHNESS_SUPERVISION_REJECTED = YES`.

## 7. Answers to the scientific questions

1. **Was short-to-long/recurrent mismatch the primary cause?** Not established.
   The mismatch was real, but correcting it alone reduced Dev success from GAT-V1's
   89.50% to GAT-R's 81.50%; it is not a sufficient or supported primary explanation.
2. **Did recurrent alignment make GAT-R beat FP-SHEP?** No: 81.50% versus 93.75%.
3. **Did smoothness supervision improve quality without sacrificing outcomes?** No.
   GAT-RS recovered categorical outcomes relative to GAT-R but was less smooth than
   GAT-R and remained worse than FP-SHEP in success and collisions.
4. **Did GAT-RS jointly beat FP in success, density, peer safety, and smoothness?** No;
   all primary contribution gates failed and smoothness superiority was not established.
5. **Should smoothness supervision be retained?** No. The experiment rejects it rather
   than keeping a secondary loss that did not produce the claimed secondary benefit.

## 8. Paper consistency and claim boundary

The active paper explicitly writes the historical weighted execution-quality soft
label and overlap loss. Reporting GAT-R/GAT-RS therefore requires an accurate update
to the training/supervision description. Because GAT-RS was rejected, it must be
reported as an adaptation ablation or negative result, not silently substituted for
the deployed selector. No positive core GAT contribution under R-ERR, Holdout
generalization, or Formal V2 result may be claimed.

## 9. Final gate decision

- `DEV_GATE = FAIL`
- `HOLDOUT_GATE = NOT_RUN`
- `FINAL_GAT_RS_FREEZE = NO`
- `FORMAL_V2 = NOT_GENERATED`
- `GAT_INCREMENT_UNDER_RERR = NEGATIVE`
- `GAT_CORE_CONTRIBUTION_SUPPORTED = NO`
- `ACADEMIC_INTEGRITY_GATE = PASS`
- `RECOMMENDED_NEXT_STEP = RECONSIDER_GAT_TRAINING_OBJECTIVE`

The most important next diagnostic is not another smoothness weight search. The
aligned target's 5 s branches contained very little direct peer-collision signal and
GAT-R amplified peer failures. Future work should first revisit recurrent joint-context
label construction and data coverage, with FP-SHEP retained as the hard reference.
"""
    (ROOT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    stage_counts = {
        name: Counter(row["stage"] for row in rows) for name, rows in raw.items()
    }
    write_json(
        STATS / "finalization_trace.json",
        {
            "schema_version": SCHEMA,
            "status": "PASS",
            "raw_arm_sha256": {name: sha256(path) for name, path in ARMS.items()},
            "raw_arm_row_counts": {name: len(rows) for name, rows in raw.items()},
            "raw_arm_stage_counts": {name: dict(value) for name, value in stage_counts.items()},
            "paired_identity_match": True,
            "checkpoint_checks": checkpoint_checks,
            "holdout_performance_rows": 0,
            "formal_v2_team_rows": 0,
            "formal_v2_agent_rows": 0,
            "methodology_old_loss_explicit": old_loss_explicit,
            "conclusion_sha256": sha256(ROOT / "conclusion.json"),
            "final_report_sha256": sha256(ROOT / "FINAL_REPORT.md"),
        },
    )

    print(json.dumps(conclusion, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
