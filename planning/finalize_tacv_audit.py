#!/usr/bin/env python3
"""Finalize the rejected TACV Development branch without running Holdout/Formal."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "artifacts/transient_aware_candidate_veto/20260824_184551"
VARIANTS = ("tacv_mild", "tacv_medium", "tacv_strong")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_mean(values: Iterable[Any]) -> float | None:
    cleaned = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not cleaned else float(np.mean(cleaned))


def episode_payloads(variant: str) -> list[dict[str, Any]]:
    directory = ROOT / f"06_development/records/{variant}/episode_records"
    return [
        load_json(path)
        for path in sorted(directory.glob("*.json"))
        if not path.stem.endswith("_SOFTWARE_ERROR")
    ]


def main() -> None:
    selection = load_json(ROOT / "06_development/TACV_DEVELOPMENT_SELECTION.json")
    if selection["holdout_authorized"] or selection["selected_variant"] != "NONE":
        raise RuntimeError("this finalizer is only valid for a rejected Development branch")
    predictability = load_json(ROOT / "02_predictability/PREVIEW_TRANSIENT_PREDICTABILITY.json")
    replaceability = load_json(ROOT / "03_replaceability/SAFE_ALTERNATIVE_REPLACEABILITY.json")
    gate = load_json(ROOT / "04_gate_decision/TACV_GATE_DECISION.json")
    paired_rows = read_csv(ROOT / "06_development/TACV_DEVELOPMENT_PAIRED_SUMMARY.csv")
    paired = {row["variant"]: row for row in paired_rows}
    overall_rows = {
        row["method"]: row
        for row in read_csv(ROOT / "06_development/TACV_DEVELOPMENT_RESULTS.csv")
        if row["scope"] == "overall"
    }

    final_freeze = {
        "schema_version": "final_tacv_freeze_v1",
        "status": "DEVELOPMENT_REJECTED_NO_VARIANT_FROZEN",
        "selected_variant": "NONE",
        "holdout_authorized": False,
        "holdout_manifest_created": False,
        "formal_v2_authorized": False,
        "development_selection_sha256": sha256(ROOT / "06_development/TACV_DEVELOPMENT_SELECTION.json"),
        "predevelopment_freeze_sha256": sha256(ROOT / "11_freeze/PRE_DEVELOPMENT_TACV_FREEZE.json"),
        "source_sha256": {
            "planning/transient_aware_candidate_veto.py": sha256(REPO_ROOT / "planning/transient_aware_candidate_veto.py"),
            "Multi-agent_Algo_lib/scripts/run_tacv_development.py": sha256(
                REPO_ROOT / "Multi-agent_Algo_lib/scripts/run_tacv_development.py"
            ),
        },
        "reason": "No Development arm passed the pre-frozen reliability plus 5% smoothness and 5% switch-transient gate.",
    }
    atomic_json(ROOT / "11_freeze/FINAL_TACV_FREEZE.json", final_freeze)

    holdout_reason = "NOT_RUN because no TACV Development arm passed the frozen selection gate"
    write_csv(
        ROOT / "08_holdout/TACV_HOLDOUT_RESULTS.csv",
        [{"status": "NOT_RUN", "scenario_count": 0, "reason": holdout_reason}],
    )
    atomic_json(
        ROOT / "08_holdout/TACV_HOLDOUT_PAIRED_STATISTICS.json",
        {
            "schema_version": "tacv_holdout_paired_statistics_v1",
            "status": "NOT_RUN",
            "scenario_count": 0,
            "reason": holdout_reason,
            "statistics_fabricated": False,
        },
    )

    runtime_variants: dict[str, Any] = {}
    for variant in VARIANTS:
        payloads = episode_payloads(variant)
        episodes = [payload["episode"] for payload in payloads]
        runtime_variants[variant] = {
            "episode_count": len(episodes),
            "tacv_veto_only_mean_ms_per_episode": finite_mean(row["tacv_runtime_ms"] for row in episodes),
            "online_algorithm_plus_tacv_mean_ms_per_episode": finite_mean(
                row["total_online_algorithm_compute_plus_tacv_ms"] for row in episodes
            ),
            "candidate_decision_count": int(sum(int(row["tacv_candidate_decision_count"]) for row in episodes)),
            "tacv_veto_mean_ms_per_candidate_decision": float(
                sum(float(row["tacv_runtime_ms"]) for row in episodes)
                / max(sum(int(row["tacv_candidate_decision_count"]) for row in episodes), 1)
            ),
            "additional_preview_rollout_count": 0,
        }
    runtime = {
        "schema_version": "tacv_runtime_summary_v1",
        "scope": "Development_only_unselected_variants",
        "original_online_compute_mean_ms_per_episode": float(overall_rows["original"]["online_compute_mean_ms"]),
        "variants": runtime_variants,
        "selected_variant": "NONE",
        "holdout_runtime": "NOT_RUN",
    }
    atomic_json(ROOT / "09_runtime/TACV_RUNTIME_SUMMARY.json", runtime)

    go_no_go = {
        "schema_version": "tacv_formal_go_no_go_v1",
        "selected_variant": "NONE",
        "development": {variant: selection["paired_results"][index] for index, variant in enumerate(VARIANTS)},
        "holdout": "NOT_RUN",
        "HOLDOUT_ACCEPTANCE": "NOT_RUN",
        "FORMAL_REEVALUATION_RECOMMENDED": "NO",
        "FORMAL_V2_EXECUTED": "NO",
        "ORIGINAL_FORMAL_SUCCESS": 0.9525,
        "FORMAL_RESULT_MODIFIED": "NO",
        "NO_RETRAIN_SMOOTHNESS_REPAIR_STOP": "YES",
        "stop_reason": "Development reliability/smoothness/switch-transient gate failed for all arms",
    }
    atomic_json(ROOT / "TACV_FORMAL_GO_NO_GO.json", go_no_go)

    paragraph = (
        "### FP-preview transient diagnostic and candidate-veto result\n\n"
        "A read-only audit showed that the existing four-step FP-SHEP preview carried moderate information about "
        "the realized post-switch transient (Development clean-window Spearman $\\rho=0.477$, 95% episode-cluster "
        "CI $[0.455,0.497]$; independent diagnostic Holdout $\\rho=0.485$). In the Development high-preview tail, "
        "97.7% of selected candidates had a safety-noninferior lower-transient alternative. We therefore evaluated "
        "a minimal post-GAT Transient-Aware Candidate Veto (TACV) without changing the graph, logits, preview, or "
        "learned checkpoints. The best reliability arms reached 97/100 Development successes versus 96/100 for "
        "the original method, but their paired both-success smoothness reductions were only 4.84% and 4.24%, and "
        "their switch-aligned jerk reductions were only 0.19% and 0.11%. No arm passed the pre-frozen joint gate, "
        "so no TACV variant was frozen or evaluated on sealed Holdout/Formal V2. The paper therefore retains the "
        "original Proposed method and treats TACV as a negative no-retraining smoothness diagnostic, not a method contribution.\n"
    )
    atomic_text(ROOT / "12_paper_ready/PAPER_TACV_DIAGNOSTIC_PARAGRAPH.md", paragraph)

    dev_clean = predictability["sections"]["development_clean_window"]
    hold_clean = predictability["sections"]["holdout_clean_window"]
    dev_replace = replaceability["blocks"]["development"]
    conclusion = {
        "schema_version": "transient_aware_candidate_veto_conclusion_v1",
        "CRT_BRANCH_STATUS": "CLOSED_REJECTED",
        "SECTOR_DENSIFICATION_STATUS": "CLOSED_INCOMPATIBLE",
        "FP_PREVIEW_ACCELERATION_AVAILABLE": "YES",
        "PREVIEW_TRANSIENT_METRIC": "mean squared discrete jerk over [current applied acceleration, H4 predicted applied accelerations], dt=0.1 s",
        "CLEAN_WINDOW_EVENT_COUNT": {"development": dev_clean["event_count"], "diagnostic_holdout": hold_clean["event_count"]},
        "PREVIEW_REALIZED_SPEARMAN_RHO": {"development": dev_clean["spearman_rho"], "diagnostic_holdout": hold_clean["spearman_rho"]},
        "PREVIEW_REALIZED_RHO_95CI": {"development": dev_clean["episode_cluster_bootstrap_95ci"], "diagnostic_holdout": hold_clean["episode_cluster_bootstrap_95ci"]},
        "PREVIEW_HIGH_JERK_AUROC": {"development": dev_clean["high_jerk_auroc"], "diagnostic_holdout": hold_clean["high_jerk_auroc"]},
        "PREVIEW_TRANSIENT_PREDICTABILITY": predictability["PREVIEW_TRANSIENT_PREDICTABILITY"],
        "HIGH_TRANSIENT_EVENT_COUNT": dev_replace["high_transient_event_count"],
        "SAFE_ALTERNATIVE_REPLACEABLE_RATE": dev_replace["replaceable_rate"],
        "MEDIAN_BEST_TRANSIENT_REDUCTION": dev_replace["median_best_transient_reduction"],
        "SAFE_ALTERNATIVE_HEADROOM": replaceability["SAFE_ALTERNATIVE_HEADROOM"],
        "NECESSARY_HIGH_TRANSIENT_FRACTION": dev_replace["necessary_fraction"],
        "POTENTIALLY_AVOIDABLE_HIGH_TRANSIENT_FRACTION": dev_replace["potentially_avoidable_fraction"],
        "TACV_AUTHORIZED": gate["TACV_AUTHORIZED"],
        "SELECTED_TACV_VARIANT": "NONE",
        "DEV_ORIGINAL_SUCCESS": float(overall_rows["original"]["team_success_rate"]),
        "DEV_TACV_SUCCESS": {variant: float(overall_rows[variant]["team_success_rate"]) for variant in VARIANTS},
        "DEV_ORIGINAL_SMOOTHNESS": {variant: float(paired[variant]["both_success_original_smoothness"]) for variant in VARIANTS},
        "DEV_TACV_SMOOTHNESS": {variant: float(paired[variant]["both_success_tacv_smoothness"]) for variant in VARIANTS},
        "DEV_SMOOTHNESS_REDUCTION_PERCENT": {variant: float(paired[variant]["smoothness_reduction_percent"]) for variant in VARIANTS},
        "DEV_SWITCH_TRANSIENT_REDUCTION_PERCENT": {variant: float(paired[variant]["switch_jerk_peak_reduction_percent"]) for variant in VARIANTS},
        "DEV_PEER_COLLISION_DELTA_PP": {variant: float(paired[variant]["peer_collision_delta_pp"]) for variant in VARIANTS},
        "DEV_REPLACEMENT_RATE": {variant: float(paired[variant]["tacv_replacement_rate"]) for variant in VARIANTS},
        "DEV_GAT_TOP1_RETENTION": {variant: float(paired[variant]["gat_effective_top1_retention_rate"]) for variant in VARIANTS},
        "HOLDOUT_ORIGINAL_SUCCESS": "NOT_RUN",
        "HOLDOUT_TACV_SUCCESS": "NOT_RUN",
        "HOLDOUT_ORIGINAL_SMOOTHNESS": "NOT_RUN",
        "HOLDOUT_TACV_SMOOTHNESS": "NOT_RUN",
        "HOLDOUT_PEER_COLLISION_DELTA": "NOT_RUN",
        "HOLDOUT_REPLACEMENT_RATE": "NOT_RUN",
        "FORMAL_REEVALUATION_RECOMMENDED": "NO",
        "ORIGINAL_FORMAL_SUCCESS": 0.9525,
        "FORMAL_RESULT_MODIFIED": "NO",
        "RETRAINING_PERFORMED": "NO",
        "ACADEMIC_INTEGRITY_GATE": "PASS",
        "NO_RETRAIN_SMOOTHNESS_REPAIR_STOP": "YES",
        "FINAL_RECOMMENDATION": "KEEP_ORIGINAL_PROPOSED; close TACV and all no-retraining smoothness-repair branches.",
    }
    atomic_json(ROOT / "conclusion.json", conclusion)

    figure_names = [
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
    available_figures = [
        name
        for name in figure_names
        if (ROOT / f"12_paper_ready/pdf/{name}.pdf").exists()
    ]
    report = f"""# FP-SHEP Preview Transient Predictability and TACV Audit

## Executive result

The frozen FP-SHEP H4 preview contains a **moderate, replicated** signal for realized post-switch transient, and the high-preview tail contains abundant safety-noninferior lower-transient alternatives. This justified a minimal post-GAT Transient-Aware Candidate Veto (TACV) Development experiment. However, **no TACV arm passed the pre-frozen Development gate**.

Original Development success was **96/100 (96.0%)**. TACV-Mild achieved **94/100 (94.0%)**, while TACV-Medium and TACV-Strong each achieved **97/100 (97.0%)**. The two reliability-preserving arms reduced paired both-success smoothness by only **4.84%** and **4.24%**, below the 5% gate, and reduced switch-aligned 0.5 s jerk by only **0.19%** and **0.11%**, far below the 5% gate. Therefore `SELECTED_TACV_VARIANT = NONE`, sealed Holdout was not run, and Formal V2 was not run.

## 1. Recovered FP-SHEP transient contract

The existing FP-SHEP preview clones the frozen SAC-DMP execution state for H=4 steps at dt=0.1 s. It already retains predicted positions, velocities, applied accelerations, terminal speed, clearance, progress, and execution deviation. The real current applied acceleration is available from the live controller state. The audited preview score is the mean squared discrete jerk over the sequence consisting of the current real applied acceleration followed by the four predicted applied accelerations. No learned predictor, regression model, new rollout, or retraining was introduced.

## 2. Predictability

The primary CLEAN-window analysis excludes events followed by another accepted reference command inside the H4 comparison window.

| Block | CLEAN events | Spearman rho | 95% episode-cluster CI | AUROC for P90 realized high jerk | Q4/Q1 realized jerk |
|---|---:|---:|---:|---:|---:|
| Development | {dev_clean['event_count']:,} | {dev_clean['spearman_rho']:.3f} | [{dev_clean['episode_cluster_bootstrap_95ci'][0]:.3f}, {dev_clean['episode_cluster_bootstrap_95ci'][1]:.3f}] | {dev_clean['high_jerk_auroc']:.3f} | {dev_clean['quartile_q4_q1_ratio']:.2f}x |
| Diagnostic Holdout | {hold_clean['event_count']:,} | {hold_clean['spearman_rho']:.3f} | [{hold_clean['episode_cluster_bootstrap_95ci'][0]:.3f}, {hold_clean['episode_cluster_bootstrap_95ci'][1]:.3f}] | {hold_clean['high_jerk_auroc']:.3f} | {hold_clean['quartile_q4_q1_ratio']:.2f}x |

The rank relationship is stable and replicated, but AUROC is only about 0.59--0.60. `PREVIEW_TRANSIENT_PREDICTABILITY = MODERATE`, not STRONG.

## 3. Safe-alternative replaceability

At the Development CLEAN-P90 activation threshold (`J_preview = 801.7011`), Development contained {dev_replace['high_transient_event_count']:,} high-transient selected events. A safety-noninferior lower-transient alternative existed in **{100.0 * dev_replace['replaceable_rate']:.2f}%**; the median best predicted reduction was **{100.0 * dev_replace['median_best_transient_reduction']:.1f}%**. Only **{100.0 * dev_replace['necessary_fraction']:.2f}%** were classified as `NECESSARY_HIGH_TRANSIENT`. The independent diagnostic Holdout replicated a {100.0 * replaceability['blocks']['holdout']['replaceable_rate']:.2f}% replaceable rate.

Admissibility was conservative and safety-first: an alternative had to be an existing feasible Top-K proposal, have no worse interaction-risk class, no lower FP-SHEP minimum clearance, and, when both candidates were risky, no longer risk duration and no lower predicted peer separation. Static-versus-dynamic dominance was not fabricated because the frozen FP clearance is untyped.

The high-peer stratum was small (22 Development and 16 diagnostic-Holdout events) and less replaceable than ordinary free flight: **77.3%** and **81.3%**, respectively. Thus the overall 97.7--97.8% headroom must not be interpreted as equally strong in peer-critical states.

## 4. TACV implementation

TACV ran after the unchanged Top-K=10 graph, FP-SHEP evaluation, GAT-R logits, and existing interaction mask. It searched only GAT candidate ranks 2--3 and selected the first safety-noninferior candidate meeting the frozen relative transient-reduction condition. Initial selection was excluded. Graph nodes, logits, checkpoints, SAC-DMP, ERR triggers, sensors, dynamics, and previews were unchanged; no additional preview was run.

## 5. Development result

| Arm | Success | Peer collision | Smoothness reduction | Switch-jerk reduction | Replacement | GAT top-1 retained | Compute delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mild | 94.0% | 6.0% | {float(paired['tacv_mild']['smoothness_reduction_percent']):.2f}% | {float(paired['tacv_mild']['switch_jerk_peak_reduction_percent']):+.2f}% | {100*float(paired['tacv_mild']['tacv_replacement_rate']):.2f}% | {100*float(paired['tacv_mild']['gat_effective_top1_retention_rate']):.2f}% | +{float(paired['tacv_mild']['online_compute_delta_ms']):.1f} ms |
| Medium | 97.0% | 3.0% | {float(paired['tacv_medium']['smoothness_reduction_percent']):.2f}% | {float(paired['tacv_medium']['switch_jerk_peak_reduction_percent']):+.2f}% | {100*float(paired['tacv_medium']['tacv_replacement_rate']):.2f}% | {100*float(paired['tacv_medium']['gat_effective_top1_retention_rate']):.2f}% | +{float(paired['tacv_medium']['online_compute_delta_ms']):.1f} ms |
| Strong | 97.0% | 3.0% | {float(paired['tacv_strong']['smoothness_reduction_percent']):.2f}% | {float(paired['tacv_strong']['switch_jerk_peak_reduction_percent']):+.2f}% | {100*float(paired['tacv_strong']['tacv_replacement_rate']):.2f}% | {100*float(paired['tacv_strong']['gat_effective_top1_retention_rate']):.2f}% | +{float(paired['tacv_strong']['online_compute_delta_ms']):.1f} ms |

Original peer collision was 3.0%. Medium and Strong preserved that rate and gained one net success, but neither reached the pre-frozen 5% smoothness requirement, and their true switch-aligned jerk reductions were essentially zero. Mild lost two net successes, increased peer collision by 3 pp, and slightly worsened switch jerk. These results reject the hypothesis that simply vetoing high-preview candidates removes the observed reference-switch transient.

## 6. Paired failure interpretation

Medium and Strong each produced three Original-success/TACV-failure episodes and four TACV-only recoveries; exact paired success p=1.0. Mild produced five Original-only versus three TACV-only successes (p=0.727). The event-level audit retains every replacement preceding an Original-success/TACV-failure outcome, with original/replacement candidate IDs, GAT ranks, FP clearance, interaction risk, predicted reduction, collision type, and veto-to-failure time. Exact peer state was not retained and is explicitly unavailable rather than reconstructed.

The central finding is a mismatch between candidate-level predicted improvement and system-level switch transient: Medium replacements reduced predicted transient by {100*float(paired['tacv_medium']['mean_predicted_transient_reduction']):.1f}% on average, yet aggregate switch-aligned jerk fell only 0.19%. This indicates that a large fraction of observed jerk is governed by recurrent closed-loop switching/execution dynamics rather than the selected candidate's isolated H4 preview transient alone.

## 7. Runtime and stop rule

The Development variants introduced no extra preview calls, but changed trajectories and upper-event counts and added post-GAT candidate checks. Mean total online compute increased by 4.81--5.81 s/episode relative to Original. Because no arm passed the Development gate, no variant was selected or frozen for performance use, no sealed Holdout manifest was created, and no Holdout or Formal V2 episode was run.

- `TACV_AUTHORIZED = YES` (diagnostic gate)
- `SELECTED_TACV_VARIANT = NONE`
- `FORMAL_REEVALUATION_RECOMMENDED = NO`
- `NO_RETRAIN_SMOOTHNESS_REPAIR_STOP = YES`
- `FINAL_RECOMMENDATION = KEEP_ORIGINAL_PROPOSED`

The original Formal V2 result remains **381/400 (95.25%)** and is not modified or combined with Development smoothness statistics.

## 8. Integrity

All three Development arms contain 100 unique paired scenarios and zero software errors. The first integration attempt failed before completing an episode, is preserved under an explicit exclusion marker, and is absent from all statistics. Diagnostic replay independently reproduced 200/200 trajectories and the reported rank/replaceability gates. No network was retrained, no Formal V2 data selected any parameter, and no prior CRT result was reopened.

## Figure index

Generated figures: {', '.join(available_figures) if available_figures else 'pending final render'}.

## Artifact index

- `01_preview_contract/FP_PREVIEW_TRANSIENT_CONTRACT.json`
- `02_predictability/PREVIEW_VS_REALIZED_TRANSIENT.csv`
- `02_predictability/PREVIEW_TRANSIENT_PREDICTABILITY.json`
- `03_replaceability/TACV_SAFETY_ADMISSIBILITY_CONTRACT.json`
- `03_replaceability/SAFE_ALTERNATIVE_REPLACEABILITY.csv`
- `03_replaceability/NECESSARY_VS_AVOIDABLE_TRANSIENTS.csv`
- `04_gate_decision/TACV_GATE_DECISION.json`
- `05_tacv_implementation/TACV_IMPLEMENTATION_CONTRACT.json`
- `06_development/TACV_DEVELOPMENT_RESULTS.csv`
- `06_development/TACV_EVENT_LEVEL_DECISIONS.csv`
- `07_failure_audit/TACV_PAIRED_FAILURE_AUDIT.csv`
- `09_runtime/TACV_RUNTIME_SUMMARY.json`
- `TACV_FORMAL_GO_NO_GO.json`, `conclusion.json`, and `12_paper_ready/`
"""
    atomic_text(ROOT / "FINAL_REPORT.md", report)
    print(json.dumps({"status": "PASS", "selected_variant": "NONE"}, indent=2))


if __name__ == "__main__":
    main()
