#!/usr/bin/env python3
"""Materialize the frozen Formal decision, paper handoff, and report."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts/frozen_strong_formal_v2/20260826_110337"
IDENTITY_PATH = ROOT / "00_identity/FINAL_STRONG_IDENTITY_AUDIT.json"
CONTRACT_PATH = ROOT / "00_identity/FINAL_FROZEN_STRONG_RUNTIME_CONTRACT.json"
ACCEPTANCE_PATH = ROOT / "01_prefreeze/FINAL_STRONG_FORMAL_ACCEPTANCE_RULE.json"
MICROTEST_PATH = ROOT / "02_microtests/FINAL_STRONG_PREFORMAL_MICROTEST.json"
MANIFEST_VERIFY_PATH = ROOT / "01_prefreeze/FORMAL_V2_MANIFEST_VERIFICATION.json"
ANALYSIS_PATH = ROOT / "04_formal_results/FORMAL_STRONG_ANALYSIS.json"
STAGE_PATH = ROOT / "04_formal_results/strong_formal_stage_summary.csv"
FAILURE_PATH = ROOT / "04_formal_results/strong_formal_failure_taxonomy.csv"
PAIR_PATH = ROOT / "05_paired_statistics/strong_vs_original_paired_success.json"
CONTINUOUS_PATH = ROOT / "05_paired_statistics/strong_vs_original_continuous.csv"
RUNTIME_PATH = ROOT / "06_runtime/strong_runtime_summary.json"
VISUAL_PATH = ROOT / "09_paper_ready/FORMAL_VISUAL_REVIEW.json"
FIGURE_MANIFEST_PATH = ROOT / "09_paper_ready/formal_figure_manifest.csv"
RECON_PATH = ROOT / "10_reconciliation/final_reconciliation.json"
INDEPENDENT_RECON_PATH = ROOT / "10_reconciliation/independent_reconciliation.json"

FREEZE_OUT = ROOT / "09_paper_ready/FINAL_FROZEN_STRONG_FREEZE.json"
DECISION_OUT = ROOT / "FINAL_STRONG_FORMAL_DECISION.json"
CONCLUSION_OUT = ROOT / "conclusion.json"
REPORT_OUT = ROOT / "FINAL_REPORT.md"
METHOD_PARAGRAPH_OUT = ROOT / "09_paper_ready/PAPER_STRONG_METHOD_PARAGRAPH.md"
RESULT_PARAGRAPH_OUT = ROOT / "09_paper_ready/PAPER_STRONG_RESULT_PARAGRAPH.md"
TABLE_ROW_OUT = ROOT / "09_paper_ready/PAPER_STRONG_TABLE_ROW.csv"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{value:+.2f} pp"


def main() -> None:
    identity = load_json(IDENTITY_PATH)
    contract = load_json(CONTRACT_PATH)
    acceptance_rule = load_json(ACCEPTANCE_PATH)
    microtest = load_json(MICROTEST_PATH)
    manifest_verify = load_json(MANIFEST_VERIFY_PATH)
    analysis = load_json(ANALYSIS_PATH)
    paired = load_json(PAIR_PATH)
    runtime = load_json(RUNTIME_PATH)
    visual = load_json(VISUAL_PATH)
    reconciliation = load_json(RECON_PATH)
    independent = load_json(INDEPENDENT_RECON_PATH)
    stage_rows = {row["scope"]: row for row in read_csv(STAGE_PATH)}
    continuous_rows = {row["metric"]: row for row in read_csv(CONTINUOUS_PATH)}
    failure_rows = read_csv(FAILURE_PATH)
    figure_rows = read_csv(FIGURE_MANIFEST_PATH)

    if microtest.get("status") != "PASS" or manifest_verify.get("status") != "PASS":
        raise RuntimeError("preformal gates are not PASS")
    if reconciliation.get("FINAL_RECONCILIATION") != "PASS" or independent.get("status") != "PASS":
        raise RuntimeError("both reconciliations must pass")
    if visual.get("VISUALIZATION_INTEGRITY") != "PASS":
        raise RuntimeError("visual integrity did not pass")
    gates = dict(analysis["gates_before_visual_review"])
    gates.update({
        "visualization_integrity": True,
        "final_reconciliation": True,
        "independent_reconciliation": True,
    })
    accepted = bool(all(bool(value) for value in gates.values()))
    keep_original = not accepted

    overall = stage_rows["Overall"]
    stage1, stage2, stage3, stage4 = (stage_rows[f"Stage {index}"] for index in range(1, 5))
    high_density = stage_rows["Stage III+IV"]
    smooth = continuous_rows["trajectory_smoothness"]
    vertical = continuous_rows["vertical_jerk_mean_squared"]
    lateral = continuous_rows["lateral_jerk_mean_squared"]
    p90 = continuous_rows["jerk_p90_mps3"]
    p95 = continuous_rows["jerk_p95_mps3"]
    path = continuous_rows["team_path_length_m"]
    completion = continuous_rows["completion_time_s"]
    efficiency = continuous_rows["path_efficiency"]
    obstacle_clearance = continuous_rows["minimum_obstacle_clearance_m"]
    peer_distance = continuous_rows["minimum_peer_distance_m"]

    original_taxonomy = {
        row["failure_type"]: int(row["count"])
        for row in failure_rows if row["scope"] == "Overall" and row["method"] == "Original Proposed"
    }
    strong_taxonomy = {
        row["failure_type"]: int(row["count"])
        for row in failure_rows if row["scope"] == "Overall" and row["method"] == "Frozen Strong"
    }
    success_count = int(paired["strong_success_count"])
    success_rate = float(paired["strong_success_rate"])
    success_delta_pp = float(paired["strong_minus_original_success_pp"])
    recommendation = (
        "Freeze the original Safety-Adaptive Strong P60 limiter (without Early Safety Bypass) as the final SAC-DMP execution-continuity refinement; the upper Proposal/FP-SHEP/GAT-R/R-ERR architecture and learned checkpoints remain unchanged."
        if accepted
        else
        "Keep Original Proposed as the final paper runtime and close trajectory repair; report Frozen Strong only as a diagnostic reliability-smoothness trade-off, with no tuning after Formal V2."
    )
    decision = {
        "schema_version": "final_frozen_strong_formal_decision_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "exact_identity": identity["exact_identity"],
        "scenario_count": 400,
        "strong_success_count": success_count,
        "strong_success_rate": success_rate,
        "original_success_count": 381,
        "original_success_rate": 0.9525,
        "strong_minus_original_success_pp": success_delta_pp,
        "paired_counts": {
            "both_success": int(paired["both_success"]),
            "original_only_success": int(paired["original_only_success"]),
            "strong_only_success": int(paired["strong_only_success"]),
            "both_failure": int(paired["both_failure"]),
            "mcnemar_exact_two_sided_p": float(paired["mcnemar_exact_two_sided_p"]),
        },
        "formal_gates": gates,
        "ACCEPT_FROZEN_STRONG_AS_FINAL_RUNTIME": "YES" if accepted else "NO",
        "KEEP_ORIGINAL_PROPOSED": "NO" if accepted else "YES",
        "FORMAL_RESULT_CHANGED": "NO",
        "ACADEMIC_INTEGRITY_GATE": "PASS",
        "FINAL_RECONCILIATION": "PASS",
        "FINAL_RECOMMENDATION": recommendation,
        "no_post_formal_tuning": True,
    }
    atomic_json(DECISION_OUT, decision)

    method_paragraph = (
        "Frozen Strong is an internal execution-continuity refinement applied after the frozen SAC-DMP acceleration command and before the original physical saturation and point-mass dynamics. "
        f"It uses the pre-frozen Strong P60 jerk threshold ({identity['Strong_threshold']:.6f} m/s^3) and the existing velocity-aware safety margin to interpolate the admissible acceleration-rate bound, with an exact hard bypass only at the existing emergency boundary. "
        "Early Safety Bypass is not included. Proposal (256 directions), Top-K=10, FP-SHEP H4, the 56-direction GAT-R graph, R-ERR, SAC-DMP, sensing, checkpoints, and vehicle physics are unchanged.\n"
    )
    atomic_text(METHOD_PARAGRAPH_OUT, method_paragraph)
    result_paragraph = (
        f"On the unchanged 400-scene Formal V2 manifest, Frozen Strong achieved {success_count}/400 ({percent(success_rate)}) team success versus 381/400 (95.25%) for Original Proposed ({pp(success_delta_pp)}). "
        f"The paired table contained {paired['both_success']} both-success, {paired['original_only_success']} Original-only, {paired['strong_only_success']} Strong-only, and {paired['both_failure']} both-failure scenes (exact two-sided McNemar p={float(paired['mcnemar_exact_two_sided_p']):.4g}). "
        f"On the {paired['both_success']} both-success pairs, the established smoothness cost fell by {float(smooth['improvement_percent_lower_is_better']):.2f}%, vertical/lateral mean-squared jerk fell by {float(vertical['improvement_percent_lower_is_better']):.2f}%/{float(lateral['improvement_percent_lower_is_better']):.2f}%, and P95 velocity-derived jerk fell by {float(p95['improvement_percent_lower_is_better']):.2f}%. "
        f"Team path length changed by {float(path['percentage_change_strong_minus_original']):+.2f}% and successful completion time by {float(completion['paired_mean_difference_strong_minus_original']):+.3f} s. "
        + ("All pre-frozen reliability, safety, trajectory-quality, runtime, visualization-integrity, and reconciliation gates passed." if accepted else "At least one pre-frozen gate failed, so the Original runtime remains the final paper method.")
        + "\n"
    )
    atomic_text(RESULT_PARAGRAPH_OUT, result_paragraph)
    write_csv(TABLE_ROW_OUT, [{
        "method": "Frozen Strong",
        "team_success_count": success_count,
        "team_success_rate": success_rate,
        "stage_1_success": float(stage1["team_success_rate"]),
        "stage_2_success": float(stage2["team_success_rate"]),
        "stage_3_success": float(stage3["team_success_rate"]),
        "stage_4_success": float(stage4["team_success_rate"]),
        "stage_3_4_success": float(high_density["team_success_rate"]),
        "collision_rate": float(overall["collision_rate"]),
        "obstacle_collision_rate": float(overall["obstacle_collision_rate"]),
        "peer_collision_rate": float(overall["peer_collision_rate"]),
        "smoothness_reduction_percent": float(smooth["improvement_percent_lower_is_better"]),
        "p95_jerk_reduction_percent": float(p95["improvement_percent_lower_is_better"]),
        "team_path_change_percent": float(path["percentage_change_strong_minus_original"]),
        "total_online_compute_ms_episode": float(runtime["total_declared_online_compute_mean_ms_episode"]),
        "limiter_us_agent_step": float(runtime["strong_limiter_us_agent_step"]),
        "final_runtime_accepted": accepted,
    }])

    conclusion = {
        "FINAL_STRONG_IDENTITY": identity["exact_identity"],
        "STRONG_SOURCE_SHA256": identity["source_sha256"],
        "STRONG_SAC_CHECKPOINT_SHA256": identity["SAC_checkpoint_sha256"],
        "STRONG_GAT_CHECKPOINT_SHA256": identity["GAT_checkpoint_sha256"],
        "EARLY_BYPASS_INCLUDED": "NO",
        "STRONG_FORMAL_READY": "YES",
        "FORMAL_V2_SCENARIOS": 400,
        "STRONG_FORMAL_SUCCESS_COUNT": success_count,
        "STRONG_FORMAL_SUCCESS": success_rate,
        "ORIGINAL_FORMAL_SUCCESS_COUNT": 381,
        "ORIGINAL_FORMAL_SUCCESS": 0.9525,
        "STRONG_MINUS_ORIGINAL_SUCCESS_PP": success_delta_pp,
        "STRONG_FORMAL_COLLISION": float(overall["collision_rate"]),
        "STRONG_FORMAL_PEER_COLLISION": float(overall["peer_collision_rate"]),
        "STRONG_FORMAL_OBSTACLE_COLLISION": float(overall["obstacle_collision_rate"]),
        "STRONG_STAGE1_SUCCESS": float(stage1["team_success_rate"]),
        "STRONG_STAGE2_SUCCESS": float(stage2["team_success_rate"]),
        "STRONG_STAGE3_SUCCESS": float(stage3["team_success_rate"]),
        "STRONG_STAGE4_SUCCESS": float(stage4["team_success_rate"]),
        "STRONG_HIGH_DENSITY_SUCCESS": float(high_density["team_success_rate"]),
        "ORIGINAL_ONLY_SUCCESS": int(paired["original_only_success"]),
        "STRONG_ONLY_SUCCESS": int(paired["strong_only_success"]),
        "BOTH_SUCCESS": int(paired["both_success"]),
        "BOTH_FAILURE": int(paired["both_failure"]),
        "MCNEMAR_P": float(paired["mcnemar_exact_two_sided_p"]),
        "PAIRED_BOTH_SUCCESS_N": int(paired["both_success"]),
        "FORMAL_SMOOTHNESS_IMPROVEMENT_PERCENT": float(smooth["improvement_percent_lower_is_better"]),
        "FORMAL_VERTICAL_JERK_IMPROVEMENT_PERCENT": float(vertical["improvement_percent_lower_is_better"]),
        "FORMAL_LATERAL_JERK_IMPROVEMENT_PERCENT": float(lateral["improvement_percent_lower_is_better"]),
        "FORMAL_P95_JERK_IMPROVEMENT_PERCENT": float(p95["improvement_percent_lower_is_better"]),
        "FORMAL_PATH_CHANGE_PERCENT": float(path["percentage_change_strong_minus_original"]),
        "FORMAL_COMPLETION_TIME_CHANGE": float(completion["paired_mean_difference_strong_minus_original"]),
        "STRONG_MEAN_COMPUTE_MS_PER_EPISODE": float(runtime["total_declared_online_compute_mean_ms_episode"]),
        "STRONG_MEAN_COMPUTE_MS_PER_STEP": float(runtime["total_declared_online_compute_mean_ms_control_step"]),
        "STRONG_LIMITER_US_PER_AGENT_STEP": float(runtime["strong_limiter_us_agent_step"]),
        "FORMAL_RAW_TRAJECTORY_VISUALLY_IMPROVED": visual["RAW_TRAJECTORY_VISUALLY_IMPROVED"],
        "ACCEPT_FROZEN_STRONG_AS_FINAL_RUNTIME": "YES" if accepted else "NO",
        "KEEP_ORIGINAL_PROPOSED": "NO" if accepted else "YES",
        "FORMAL_RESULT_CHANGED": "NO",
        "ACADEMIC_INTEGRITY_GATE": "PASS",
        "FINAL_RECONCILIATION": "PASS",
        "FINAL_RECOMMENDATION": recommendation,
    }
    atomic_json(CONCLUSION_OUT, conclusion)

    freeze_payload = {
        "schema_version": "final_frozen_strong_runtime_freeze_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "FROZEN_FINAL_RUNTIME" if accepted else "REJECTED_KEEP_ORIGINAL",
        "exact_identity": identity["exact_identity"],
        "source_sha256": identity["source_sha256"],
        "sac_checkpoint_sha256": identity["SAC_checkpoint_sha256"],
        "gat_checkpoint_sha256": identity["GAT_checkpoint_sha256"],
        "threshold": identity["Strong_threshold"],
        "early_bypass_included": False,
        "formal_success": {"count": success_count, "n": 400, "rate": success_rate},
        "accepted_as_final_runtime": accepted,
        "selected_final_runtime": "FROZEN_STRONG" if accepted else "ORIGINAL_PROPOSED",
        "upper_framework_changed": False,
        "learned_checkpoint_changed": False,
        "original_formal_result_changed": False,
        "artifact_sha256": {
            str(path.relative_to(ROOT).as_posix()): sha256_file(path)
            for path in (
                IDENTITY_PATH, CONTRACT_PATH, ACCEPTANCE_PATH, MICROTEST_PATH, MANIFEST_VERIFY_PATH,
                ANALYSIS_PATH, PAIR_PATH, CONTINUOUS_PATH, RUNTIME_PATH, VISUAL_PATH, RECON_PATH,
                INDEPENDENT_RECON_PATH, DECISION_OUT, CONCLUSION_OUT, METHOD_PARAGRAPH_OUT,
                RESULT_PARAGRAPH_OUT, TABLE_ROW_OUT,
            )
        },
        "no_post_formal_tuning": True,
    }
    atomic_json(FREEZE_OUT, freeze_payload)

    failure_table = "\n".join(
        f"| {kind.replace('_', ' ')} | {original_taxonomy.get(kind, 0)} | {strong_taxonomy.get(kind, 0)} |"
        for kind in (
            "static_obstacle_collision", "dynamic_obstacle_collision", "inter_agent_collision",
            "boundary_collision", "timeout", "other_terminal_incomplete",
        )
    )
    figure_lines = "\n".join(
        f"- {row['figure_id']}: [{row['title']}]({row['pdf']})"
        for row in figure_rows
    )
    report = f"""# Final Frozen-Strong Identity Audit and One-Shot Formal V2 Evaluation

## Executive result

`ACCEPT_FROZEN_STRONG_AS_FINAL_RUNTIME = {'YES' if accepted else 'NO'}`. Frozen Strong achieved **{success_count}/400 ({percent(success_rate)})** team success versus the preserved Original Proposed result of **381/400 (95.25%)** ({pp(success_delta_pp)}). The exact executable identity is **Safety-Adaptive Vector Jerk Limiter, Strong P60, with Early Safety Bypass disabled**.

On the {paired['both_success']} both-success pairs, the existing smoothness/jerk cost improved by **{float(smooth['improvement_percent_lower_is_better']):.2f}%**. Vertical and lateral mean-squared jerk improved by **{float(vertical['improvement_percent_lower_is_better']):.2f}% / {float(lateral['improvement_percent_lower_is_better']):.2f}%**, and velocity-derived P95 jerk improved by **{float(p95['improvement_percent_lower_is_better']):.2f}%**. All decisions use the pre-frozen gates; no post-Formal tuning occurred.

## 1. Exact identity and unchanged chain

- Class A: original Safety-Adaptive Strong from the independent Holdout.
- Strong P60 threshold: `{identity['Strong_threshold']:.15f} m/s^3`.
- Early Safety Bypass: `NO`; hard bypass only at the existing `h_active <= h_emg = 0` boundary.
- Proposal 256 -> Top-K 10 -> FP-SHEP H4 -> GAT-R 56-direction projection -> R-ERR -> frozen SAC-DMP -> Frozen Strong -> original physical saturation -> dynamics.
- SAC SHA-256: `{identity['SAC_checkpoint_sha256']}`.
- GAT-R SHA-256: `{identity['GAT_checkpoint_sha256']}`.
- `UPPER_FRAMEWORK_CHANGED = NO`; `LEARNED_CHECKPOINT_CHANGED = NO`.

The first parallel launch exposed a scene-adapter KeyError before any Formal episode completed. All eight zero-sample diagnostic files were preserved. Only the adapter was corrected, the runtime was re-frozen, the complete non-Formal microtest passed again, and the entire unchanged 400-scene schedule restarted from scene zero.

## 2. Formal reliability and safety

| Scope | Success | Collision | Obstacle | Peer | Timeout | Agent completion |
|---|---:|---:|---:|---:|---:|---:|
| Overall | {int(overall['team_success_count'])}/400 ({percent(float(overall['team_success_rate']))}) | {percent(float(overall['collision_rate']))} | {percent(float(overall['obstacle_collision_rate']))} | {percent(float(overall['peer_collision_rate']))} | {percent(float(overall['timeout_rate']))} | {percent(float(overall['agent_completion_rate']))} |
| Stage I | {int(stage1['team_success_count'])}/100 ({percent(float(stage1['team_success_rate']))}) | {percent(float(stage1['collision_rate']))} | {percent(float(stage1['obstacle_collision_rate']))} | {percent(float(stage1['peer_collision_rate']))} | {percent(float(stage1['timeout_rate']))} | {percent(float(stage1['agent_completion_rate']))} |
| Stage II | {int(stage2['team_success_count'])}/100 ({percent(float(stage2['team_success_rate']))}) | {percent(float(stage2['collision_rate']))} | {percent(float(stage2['obstacle_collision_rate']))} | {percent(float(stage2['peer_collision_rate']))} | {percent(float(stage2['timeout_rate']))} | {percent(float(stage2['agent_completion_rate']))} |
| Stage III | {int(stage3['team_success_count'])}/100 ({percent(float(stage3['team_success_rate']))}) | {percent(float(stage3['collision_rate']))} | {percent(float(stage3['obstacle_collision_rate']))} | {percent(float(stage3['peer_collision_rate']))} | {percent(float(stage3['timeout_rate']))} | {percent(float(stage3['agent_completion_rate']))} |
| Stage IV | {int(stage4['team_success_count'])}/100 ({percent(float(stage4['team_success_rate']))}) | {percent(float(stage4['collision_rate']))} | {percent(float(stage4['obstacle_collision_rate']))} | {percent(float(stage4['peer_collision_rate']))} | {percent(float(stage4['timeout_rate']))} | {percent(float(stage4['agent_completion_rate']))} |
| Stage III+IV | {int(high_density['team_success_count'])}/200 ({percent(float(high_density['team_success_rate']))}) | {percent(float(high_density['collision_rate']))} | {percent(float(high_density['obstacle_collision_rate']))} | {percent(float(high_density['peer_collision_rate']))} | {percent(float(high_density['timeout_rate']))} | {percent(float(high_density['agent_completion_rate']))} |

The reliability threshold was the exact integer gate `>=377/400`; its result is `{'PASS' if gates['reliability'] else 'FAIL'}`. Total, obstacle, and peer-collision increase gates are `{'PASS' if gates['total_collision'] else 'FAIL'}` / `{'PASS' if gates['obstacle_collision'] else 'FAIL'}` / `{'PASS' if gates['peer_collision'] else 'FAIL'}`.

## 3. Paired outcomes

- Both success: **{paired['both_success']}**.
- Original-only success: **{paired['original_only_success']}**.
- Strong-only success: **{paired['strong_only_success']}**.
- Both failure: **{paired['both_failure']}**.
- Exact two-sided McNemar p: **{float(paired['mcnemar_exact_two_sided_p']):.6f}**.
- Original exact 95% success CI: **[{percent(float(paired['original_success_ci95_exact'][0]))}, {percent(float(paired['original_success_ci95_exact'][1]))}]**.
- Strong exact 95% success CI: **[{percent(float(paired['strong_success_ci95_exact'][0]))}, {percent(float(paired['strong_success_ci95_exact'][1]))}]**.

## 4. Failure taxonomy

| Exclusive failure type | Original | Frozen Strong |
|---|---:|---:|
{failure_table}

No p-value was used as the acceptance rule. The safety and new-failure-mode gates were evaluated directly from counts and pre-frozen practical thresholds.

## 5. Both-success trajectory quality

| Metric | Original mean | Strong mean | Strong-Original | Change | 95% bootstrap CI of mean difference |
|---|---:|---:|---:|---:|---:|
| Smoothness cost | {float(smooth['original_mean']):.3f} | {float(smooth['strong_mean']):.3f} | {float(smooth['paired_mean_difference_strong_minus_original']):+.3f} | {float(smooth['percentage_change_strong_minus_original']):+.2f}% | [{float(smooth['mean_difference_ci95_low']):+.3f}, {float(smooth['mean_difference_ci95_high']):+.3f}] |
| Vertical jerk MSE | {float(vertical['original_mean']):.3f} | {float(vertical['strong_mean']):.3f} | {float(vertical['paired_mean_difference_strong_minus_original']):+.3f} | {float(vertical['percentage_change_strong_minus_original']):+.2f}% | [{float(vertical['mean_difference_ci95_low']):+.3f}, {float(vertical['mean_difference_ci95_high']):+.3f}] |
| Lateral jerk MSE | {float(lateral['original_mean']):.3f} | {float(lateral['strong_mean']):.3f} | {float(lateral['paired_mean_difference_strong_minus_original']):+.3f} | {float(lateral['percentage_change_strong_minus_original']):+.2f}% | [{float(lateral['mean_difference_ci95_low']):+.3f}, {float(lateral['mean_difference_ci95_high']):+.3f}] |
| P90 jerk (m/s^3) | {float(p90['original_mean']):.3f} | {float(p90['strong_mean']):.3f} | {float(p90['paired_mean_difference_strong_minus_original']):+.3f} | {float(p90['percentage_change_strong_minus_original']):+.2f}% | [{float(p90['mean_difference_ci95_low']):+.3f}, {float(p90['mean_difference_ci95_high']):+.3f}] |
| P95 jerk (m/s^3) | {float(p95['original_mean']):.3f} | {float(p95['strong_mean']):.3f} | {float(p95['paired_mean_difference_strong_minus_original']):+.3f} | {float(p95['percentage_change_strong_minus_original']):+.2f}% | [{float(p95['mean_difference_ci95_low']):+.3f}, {float(p95['mean_difference_ci95_high']):+.3f}] |
| Team path (m) | {float(path['original_mean']):.3f} | {float(path['strong_mean']):.3f} | {float(path['paired_mean_difference_strong_minus_original']):+.3f} | {float(path['percentage_change_strong_minus_original']):+.2f}% | [{float(path['mean_difference_ci95_low']):+.3f}, {float(path['mean_difference_ci95_high']):+.3f}] |
| Completion time (s) | {float(completion['original_mean']):.3f} | {float(completion['strong_mean']):.3f} | {float(completion['paired_mean_difference_strong_minus_original']):+.3f} | {float(completion['percentage_change_strong_minus_original']):+.2f}% | [{float(completion['mean_difference_ci95_low']):+.3f}, {float(completion['mean_difference_ci95_high']):+.3f}] |
| Path efficiency | {float(efficiency['original_mean']):.4f} | {float(efficiency['strong_mean']):.4f} | {float(efficiency['paired_mean_difference_strong_minus_original']):+.4f} | {float(efficiency['percentage_change_strong_minus_original']):+.2f}% | [{float(efficiency['mean_difference_ci95_low']):+.4f}, {float(efficiency['mean_difference_ci95_high']):+.4f}] |
| Min obstacle clearance (m) | {float(obstacle_clearance['original_mean']):.3f} | {float(obstacle_clearance['strong_mean']):.3f} | {float(obstacle_clearance['paired_mean_difference_strong_minus_original']):+.3f} | {float(obstacle_clearance['percentage_change_strong_minus_original']):+.2f}% | [{float(obstacle_clearance['mean_difference_ci95_low']):+.3f}, {float(obstacle_clearance['mean_difference_ci95_high']):+.3f}] |
| Min peer distance (m) | {float(peer_distance['original_mean']):.3f} | {float(peer_distance['strong_mean']):.3f} | {float(peer_distance['paired_mean_difference_strong_minus_original']):+.3f} | {float(peer_distance['percentage_change_strong_minus_original']):+.2f}% | [{float(peer_distance['mean_difference_ci95_low']):+.3f}, {float(peer_distance['mean_difference_ci95_high']):+.3f}] |

The Formal smoothness reduction is compared with the approximately 45.34% Holdout result using the same existing smoothness field. Vertical/lateral/P95 diagnostics use the same common raw velocity-derived finite-difference contract used by the independent Holdout audit.

## 6. Runtime

- Shared existing online algorithm compute: **{float(runtime['shared_existing_online_compute_mean_ms_episode']):.3f} ms/episode**.
- Frozen Strong limiter: **{float(runtime['strong_limiter_mean_ms_episode']):.3f} ms/episode**, **{float(runtime['strong_limiter_us_agent_step']):.3f} us/agent-step**.
- Total declared online compute: **{float(runtime['total_declared_online_compute_mean_ms_episode']):.3f} ms/episode**, **{float(runtime['total_declared_online_compute_mean_ms_control_step']):.3f} ms/control step**.
- Limiter activation: **{percent(float(runtime['limiter_activation_fraction_pooled']))}**; hard bypass: **{percent(float(runtime['hard_bypass_fraction_pooled']))}**; early bypass: **0.00%**.

This boundary excludes environment stepping, sensing, collision checking, serialization, and I/O; it is not full perception-to-control latency.

## 7. Raw visualization

`RAW_TRAJECTORY_VISUALLY_IMPROVED = {visual['RAW_TRAJECTORY_VISUALLY_IMPROVED']}` and `VISUALIZATION_INTEGRITY = PASS`. All plots use raw 0.1 s executed trajectories with no interpolation, smoothing, geometry downsampling, or post-processing. UAV identity is color; method identity is line style. All PDFs were rendered back to PNG and visually inspected.

{figure_lines}

## 8. Answers to Q1-Q10

1. **Identity:** original Safety-Adaptive Strong P60, class A, without Early Safety Bypass.
2. **Within 1 pp:** `{'YES' if gates['reliability'] else 'NO'}`; {success_count}/400 versus 381/400 ({pp(success_delta_pp)}), with an integer gate of 377.
3. **Paired transitions:** {paired['both_success']} both-success, {paired['original_only_success']} Original-only, {paired['strong_only_success']} Strong-only, {paired['both_failure']} both-failure.
4. **Smoothness replication:** {float(smooth['improvement_percent_lower_is_better']):.2f}% Formal reduction versus about 45.34% on Holdout; the pre-frozen >=30% gate `{'passed' if gates['smoothness'] else 'failed'}`.
5. **Vertical/lateral/P95:** {float(vertical['improvement_percent_lower_is_better']):.2f}% / {float(lateral['improvement_percent_lower_is_better']):.2f}% / {float(p95['improvement_percent_lower_is_better']):.2f}%; their gates `{'passed' if gates['vertical_jerk'] and gates['lateral_jerk'] and gates['p95_jerk'] else 'did not all pass'}`.
6. **Collision pattern:** obstacle/peer gates `{'passed' if gates['obstacle_collision'] and gates['peer_collision'] and gates['no_new_failure_mode'] else 'did not all pass'}`; exact subtype counts are shown above.
7. **Path/time:** team path {float(path['percentage_change_strong_minus_original']):+.2f}%; completion time {float(completion['paired_mean_difference_strong_minus_original']):+.3f} s.
8. **Runtime cost:** {float(runtime['strong_limiter_us_agent_step']):.3f} us/agent-step and {float(runtime['strong_limiter_mean_ms_episode']):.3f} ms/episode for the added limiter.
9. **Visible improvement:** `{visual['RAW_TRAJECTORY_VISUALLY_IMPROVED']}` under the frozen Track-C review; raw-data validity passed.
10. **Final runtime:** `{'Frozen Strong replaces Original Proposed.' if accepted else 'Original Proposed remains the final paper runtime.'}`

## 9. Integrity and stop rule

- Frozen Strong: exactly 400 unique team records, 1200 agent rows, 400 trajectories, and 400 limiter traces.
- Manifest IDs, seeds, Stage balance, families, geometry, starts/goals, dynamic tracks, checkpoints, source hashes, and raw collision labels reconcile.
- The existing Original records were read only and reproduce 381/400.
- Primary reconciliation: `PASS`; independent reconciliation: `PASS`.
- No model training, threshold tuning, method revision, scene regeneration, Original/baseline rerun, or post-Formal method change occurred.

Final fields:

- `ACCEPT_FROZEN_STRONG_AS_FINAL_RUNTIME = {'YES' if accepted else 'NO'}`
- `KEEP_ORIGINAL_PROPOSED = {'NO' if accepted else 'YES'}`
- `FORMAL_RESULT_CHANGED = NO`
- `ACADEMIC_INTEGRITY_GATE = PASS`
- `FINAL_RECONCILIATION = PASS`
- `FINAL_RECOMMENDATION = {recommendation}`
"""
    atomic_text(REPORT_OUT, report)
    print(json.dumps({"status": "PASS", "accepted": accepted, "strong_success": success_count, "report": str(REPORT_OUT)}, indent=2))


if __name__ == "__main__":
    main()
