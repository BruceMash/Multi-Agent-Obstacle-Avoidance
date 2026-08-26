from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "theory_aligned_final_recovery"
    / "20260818_232900"
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = int(left_only) + int(right_only)
    if discordant == 0:
        return 1.0
    smaller = min(int(left_only), int(right_only))
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail)


def _runtime_lookup(rows: Sequence[Mapping[str, str]], method: str) -> Mapping[str, str]:
    return next(row for row in rows if row["method"] == method)


def _development_lookup(
    rows: Sequence[Mapping[str, str]], method: str, scope: str
) -> Mapping[str, str]:
    return next(row for row in rows if row["method"] == method and row["scope"] == scope)


def _method_alignment() -> dict[str, Any]:
    return {
        "paper_authority": "active uncommented Methodology in hire-rl-body.tex",
        "inactive_draft_excluded": "commented SDH/pending-reference/J_switch block",
        "theory_mechanism": "Event-Triggered Reference Reconstruction",
        "theory_trigger": (
            "[I(Phi_rep>=0) I(T_age>=T_rep,min)] OR I(h_active<=h_emg)"
        ),
        "event_priority": [
            "REFERENCE_COMPLETION_HANDOFF",
            "EMERGENCY_REPROPOSAL",
            "NORMAL_REPROPOSAL",
            "NO_UPDATE",
        ],
        "upper_pipeline": "Proposal -> Top-K 10 -> FP-SHEP H4 -> frozen GAT-V1",
        "lower_policy": "deterministic frozen SAC-DMP with historical vector forcing gate",
        "THEORY_MULTIPLE_REPROPOSALS_ALLOWED": "YES",
        "ARTIFICIAL_REPROPOSAL_CAP": "NONE",
        "original_benchmark_execution": {
            "planning_decision_count": 1,
            "replanning_count": 0,
            "replanning_enabled": False,
            "label": "ONE_SHOT_PROPOSED",
            "success": "197/400",
        },
        "theory_aligned_development_execution": {
            "initial_event": "full upper pipeline at t=0",
            "recurrent_event": "full upper pipeline at every theory event",
            "handoff_priority": True,
            "position_velocity_time_phase_continuity": True,
            "same_goal_reproposal_executed": True,
            "formal_status": "development_only",
        },
        "CURRENT_ORIGINAL_BENCHMARK_METHOD_MISMATCH": "YES",
        "ORIGINAL_197_OVER_400_LABEL": "ONE_SHOT_PROPOSED",
        "FULL_THEORY_ALIGNED_METHOD_FORMALLY_TESTED": "NO",
        "new_final_benchmark_run": False,
    }


def main() -> None:
    phase_a = load_json(OUTPUT / "corrected_classical_reconciliation.json")
    phase_b = load_json(OUTPUT / "phase_b_runtime" / "runtime_reconciliation.json")
    phase_c = load_json(
        OUTPUT
        / "phase_c_theoretical_trigger_audit"
        / "theory_observability_gate.json"
    )
    phase_d = load_json(
        OUTPUT / "phase_d_theory_err_development" / "phase_d_conclusion.json"
    )
    runtime_rows = read_csv(OUTPUT / "runtime_tradeoff.csv")
    development_summary = read_csv(
        OUTPUT
        / "phase_d_theory_err_development"
        / "development_method_stage_summary.csv"
    )
    development_episodes = read_csv(OUTPUT / "development_episode_results.csv")
    paired = read_csv(OUTPUT / "paired_recovery.csv")

    dwa_runtime = _runtime_lookup(runtime_rows, "dwa_style")
    rvo_runtime = _runtime_lookup(runtime_rows, "rvo_orca_style")
    proposed_runtime = _runtime_lookup(runtime_rows, "gat_v1_one_shot")
    dev_one_runtime = _runtime_lookup(runtime_rows, "one_shot_proposed_development")
    dev_err_runtime = _runtime_lookup(runtime_rows, "theory_err_development")

    by_pair: dict[tuple[str, int], dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in development_episodes:
        by_pair[(row["scenario_id"], int(row["seed"]))][row["method"]] = row
    err_success_only = one_success_only = 0
    err_collision_only = one_collision_only = 0
    for methods in by_pair.values():
        one = methods["gat_v1_one_shot"]
        err = methods["gat_v1_err"]
        one_success = one["team_success"] == "True"
        err_success = err["team_success"] == "True"
        one_collision = one["collision"] == "True"
        err_collision = err["collision"] == "True"
        err_success_only += int(err_success and not one_success)
        one_success_only += int(one_success and not err_success)
        err_collision_only += int(err_collision and not one_collision)
        one_collision_only += int(one_collision and not err_collision)

    dwa_total = float(dwa_runtime["mean_total_online_algorithm_compute_ms"])
    rvo_total = float(rvo_runtime["mean_total_online_algorithm_compute_ms"])
    err_total = float(dev_err_runtime["mean_total_online_algorithm_compute_ms"])
    vs_dwa = "SIMILAR" if 0.8 <= err_total / dwa_total <= 1.2 else (
        "LOWER" if err_total < dwa_total else "HIGHER"
    )
    vs_rvo = "SIMILAR" if 0.8 <= err_total / rvo_total <= 1.2 else (
        "LOWER" if err_total < rvo_total else "HIGHER"
    )
    conclusion = {
        "FUTURE_DYNAMIC_INFORMATION_LEAKAGE_AFTER": phase_a[
            "FUTURE_DYNAMIC_INFORMATION_LEAKAGE_AFTER"
        ],
        "CORRECTED_DWA_SUCCESS": phase_a["summaries"]["dwa_style"]["success_rate"],
        "CORRECTED_RVO_SUCCESS": phase_a["summaries"]["rvo_orca_style"]["success_rate"],
        "BASELINE_COMPARISON_FAIRNESS_AFTER_FIX": phase_a[
            "BASELINE_COMPARISON_FAIRNESS_AFTER_FIX"
        ],
        "TIMING_REPLAY_BEHAVIOR_MATCH": phase_b["TIMING_REPLAY_BEHAVIOR_MATCH"],
        "PROPOSED_ONE_SHOT_TOTAL_COMPUTE_MS": float(
            proposed_runtime["mean_total_online_algorithm_compute_ms"]
        ),
        "DWA_TOTAL_COMPUTE_MS": dwa_total,
        "RVO_TOTAL_COMPUTE_MS": rvo_total,
        "THEORY_MULTIPLE_REPROPOSALS_ALLOWED": "YES",
        "ARTIFICIAL_REPROPOSAL_CAP": "NONE",
        "THEORY_TRIGGER_OBSERVABILITY": phase_c["THEORY_TRIGGER_OBSERVABILITY"],
        "PHASE_D_EXECUTED": phase_d["PHASE_D_EXECUTED"],
        "ONE_SHOT_DEVELOPMENT_SUCCESS": phase_d["ONE_SHOT_DEVELOPMENT_SUCCESS"],
        "THEORY_ERR_DEVELOPMENT_SUCCESS": phase_d["THEORY_ERR_DEVELOPMENT_SUCCESS"],
        "SUCCESS_GAIN_PP": phase_d["SUCCESS_GAIN_PP"],
        "ONE_SHOT_STAGE3_SUCCESS": phase_d["ONE_SHOT_STAGE3_SUCCESS"],
        "THEORY_ERR_STAGE3_SUCCESS": phase_d["THEORY_ERR_STAGE3_SUCCESS"],
        "ONE_SHOT_STAGE4_SUCCESS": phase_d["ONE_SHOT_STAGE4_SUCCESS"],
        "THEORY_ERR_STAGE4_SUCCESS": phase_d["THEORY_ERR_STAGE4_SUCCESS"],
        "ONE_SHOT_COLLISION_RATE": phase_d["ONE_SHOT_COLLISION_RATE"],
        "THEORY_ERR_COLLISION_RATE": phase_d["THEORY_ERR_COLLISION_RATE"],
        "MEAN_REPROPOSALS_PER_EPISODE": phase_d["MEAN_REPROPOSALS_PER_EPISODE"],
        "MEDIAN_REPROPOSALS_PER_EPISODE": phase_d["MEDIAN_REPROPOSALS_PER_EPISODE"],
        "P90_REPROPOSALS_PER_EPISODE": phase_d["P90_REPROPOSALS_PER_EPISODE"],
        "MAX_REPROPOSALS_PER_EPISODE": phase_d["MAX_REPROPOSALS_PER_EPISODE"],
        "NORMAL_REPROPOSAL_COUNT": phase_d["NORMAL_REPROPOSAL_COUNT"],
        "EMERGENCY_REPROPOSAL_COUNT": phase_d["EMERGENCY_REPROPOSAL_COUNT"],
        "SAME_GOAL_REPROPOSAL_COUNT": phase_d["SAME_GOAL_REPROPOSAL_COUNT"],
        "MEAN_INTER_REPROPOSAL_INTERVAL_S": phase_d[
            "MEAN_INTER_REPROPOSAL_INTERVAL_S"
        ],
        "CHATTERING_PRESENT": phase_d["CHATTERING_PRESENT"],
        "THEORY_ERR_TOTAL_COMPUTE_MS": err_total,
        "THEORY_ERR_TOTAL_COMPUTE_VS_DWA": vs_dwa,
        "THEORY_ERR_TOTAL_COMPUTE_VS_RVO": vs_rvo,
        "THEORY_ERR_PERFORMANCE_GAIN": phase_d["THEORY_ERR_PERFORMANCE_GAIN"],
        "THEORY_ERR_EXECUTION_STABILITY": phase_d[
            "THEORY_ERR_EXECUTION_STABILITY"
        ],
        "CURRENT_ORIGINAL_BENCHMARK_METHOD_MISMATCH": "YES",
        "ORIGINAL_197_OVER_400_LABEL": "ONE_SHOT_PROPOSED",
        "FULL_THEORY_ALIGNED_METHOD_FORMALLY_TESTED": "NO",
        "RECOMMENDED_NEXT_STEP": "THEORY_TRIGGER_NEEDS_REVISION",
        "THEORY_ERR_DEVELOPMENT_GAIN": phase_d["THEORY_ERR_DEVELOPMENT_GAIN"],
        "STAGE3_4_SUCCESS_GAIN_PP": phase_d["STAGE3_4_SUCCESS_GAIN_PP"],
        "COLLISION_REDUCTION_PP": phase_d["COLLISION_REDUCTION_PP"],
        "THEORY_ERR_MEAN_PLANNING_DECISIONS_PER_EPISODE": float(
            dev_err_runtime["mean_decisions_per_episode"]
        ),
        "THEORY_ERR_MEAN_CUMULATIVE_PLANNING_MS": float(
            dev_err_runtime["mean_cumulative_planning_ms"]
        ),
        "THEORY_ERR_MEAN_DECISION_LATENCY_MS": float(
            dev_err_runtime["mean_decision_latency_ms"]
        ),
        "DEVELOPMENT_SUCCESS_MCNEMAR": {
            "ERR_only_success": err_success_only,
            "one_shot_only_success": one_success_only,
            "two_sided_exact_p": exact_mcnemar_p(err_success_only, one_success_only),
        },
        "DEVELOPMENT_COLLISION_MCNEMAR": {
            "ERR_only_collision": err_collision_only,
            "one_shot_only_collision": one_collision_only,
            "two_sided_exact_p": exact_mcnemar_p(
                err_collision_only, one_collision_only
            ),
        },
        "recovery_matrix": phase_d["recovery_matrix"],
        "continuity": phase_d["continuity"],
        "runtime_comparison_label_rule": "SIMILAR means within +/-20%",
    }
    write_json(OUTPUT / "method_alignment.json", _method_alignment())
    write_json(OUTPUT / "conclusion.json", conclusion)

    # Append development runtime scopes to the already reconciled original-400 tables.
    method_summary = read_csv(OUTPUT / "runtime_method_summary.csv")
    stage_summary = read_csv(OUTPUT / "runtime_stage_summary.csv")
    development_runtime_methods = {
        "one_shot_proposed_development",
        "theory_err_development",
    }
    method_summary = [
        row for row in method_summary if row["method"] not in development_runtime_methods
    ]
    stage_summary = [
        row for row in stage_summary if row["method"] not in development_runtime_methods
    ]
    for row in (dev_one_runtime, dev_err_runtime):
        item = dict(row)
        item["scope"] = "overall"
        method_summary.append(item)
    for method_key, display in (
        ("gat_v1_one_shot", "one_shot_proposed_development"),
        ("gat_v1_err", "theory_err_development"),
    ):
        for stage in ("stage_1", "stage_2", "stage_3", "stage_4"):
            members = [
                row
                for row in development_episodes
                if row["method"] == method_key and row["stage"] == stage
            ]
            decisions = np.asarray(
                [float(row["planning_decision_count"]) for row in members]
            )
            planning = np.asarray(
                [float(row["upper_planning_total_ms"]) for row in members]
            )
            total = np.asarray(
                [float(row["total_online_algorithm_compute_ms"]) for row in members]
            )
            latency = planning / decisions
            stage_summary.append(
                {
                    "scope": stage,
                    "method": display,
                    "evaluation_block": "new_40_development",
                    "episode_count": len(members),
                    "mean_decisions_per_episode": float(np.mean(decisions)),
                    "median_decisions_per_episode": float(np.median(decisions)),
                    "p90_decisions_per_episode": float(np.quantile(decisions, 0.9)),
                    "mean_decision_latency_ms": float(np.mean(latency)),
                    "mean_decision_latency_over_100ms": float(np.mean(latency)) / 100.0,
                    "mean_cumulative_planning_ms": float(np.mean(planning)),
                    "p90_cumulative_planning_ms": float(np.quantile(planning, 0.9)),
                    "mean_execution_actor_ms": float(
                        np.mean([float(row["execution_actor_forward_ms"]) for row in members])
                    ),
                    "mean_execution_dmp_ms": float(
                        np.mean([float(row["execution_dmp_ms"]) for row in members])
                    ),
                    "mean_total_online_algorithm_compute_ms": float(np.mean(total)),
                    "median_total_online_algorithm_compute_ms": float(np.median(total)),
                    "p90_total_online_algorithm_compute_ms": float(np.quantile(total, 0.9)),
                }
            )
    write_csv(OUTPUT / "runtime_method_summary.csv", method_summary)
    write_csv(OUTPUT / "runtime_stage_summary.csv", stage_summary)

    paper = OUTPUT / "paper_ready"
    write_csv(paper / "source_data" / "corrected_classical_summary.csv", [
        {"method": method, **phase_a["summaries"][method]}
        for method in ("dwa_style", "rvo_orca_style")
    ])
    write_csv(paper / "source_data" / "runtime_tradeoff.csv", runtime_rows)
    write_csv(paper / "source_data" / "development_method_stage_summary.csv", development_summary)
    write_csv(paper / "source_data" / "paired_recovery.csv", paired)
    (paper / "captions").mkdir(parents=True, exist_ok=True)
    (paper / "captions" / "table_corrected_classics.txt").write_text(
        "Corrected current-state-only classical baseline outcomes on the original frozen 400 scenarios.",
        encoding="utf-8",
    )
    (paper / "captions" / "table_runtime_tradeoff.txt").write_text(
        "Measured online compute. Cross-block performance is not compared; runtime rows retain their evaluation block.",
        encoding="utf-8",
    )
    (paper / "captions" / "table_development_recovery.txt").write_text(
        "Paired development comparison of one-shot and theory-aligned recurrent ERR on 40 new disjoint scenarios.",
        encoding="utf-8",
    )

    one_overall = _development_lookup(
        development_summary, "one_shot_proposed_development", "overall"
    )
    err_overall = _development_lookup(
        development_summary, "theory_err_development", "overall"
    )
    report = f"""# Theory-Aligned Final Benchmark Recovery

## Executive result

The confirmed stochastic-future leakage in the classical planners is fixed. On the unchanged 400-scenario manifest, corrected DWA achieved **391/400 (97.75%)** success and corrected RVO achieved **394/400 (98.50%)**. The old 393/400 and 395/400 values remain preserved only as `ORIGINAL_PRIVILEGED_CLASSICAL_RESULT` provenance.

The active paper theory is not the method that produced the original Proposed result. The original **197/400 (49.25%)** is correctly labelled `ONE_SHOT_PROPOSED`: one upper decision at `t=0`, zero reproposals, and `replanning_enabled=false`. The current Methodology permits repeated state-triggered reconstruction with no artificial episode cap, so `CURRENT_ORIGINAL_BENCHMARK_METHOD_MISMATCH = YES`.

On 40 newly frozen, seed/geometry/translation-disjoint development scenarios, the full theory-aligned ERR raised team success from **17/40 (42.5%)** to **29/40 (72.5%)** (+30 pp), raised Stage III/IV pooled success from **0/20** to **15/20 (75.0%)**, and reduced collision from **15/40 (37.5%)** to **5/40 (12.5%)**. The preregistered development gain gate passes. However, mean agent-level reproposals were **14.3/episode**, 444/572 were emergency events, and 82.1% of emergency events with a predecessor followed another emergency event. Therefore `THEORY_ERR_PERFORMANCE_GAIN = YES` but `THEORY_ERR_EXECUTION_STABILITY = NO` and `CHATTERING_PRESENT = YES`.

## 1. Theory contract and implementation alignment

The primary authority is the active, uncommented ERR Methodology in `hire-rl-body.tex`. It defines `T_age`, sliding-window progress `nu`, active-direction margin `h_active`, normalized state `Phi_rep`, normal reconstruction gated by `T_rep,min`, emergency reconstruction at `h_active <= h_emg`, and handoff priority. The later commented SDH/pending/`J_switch` draft is inactive and excluded.

The implemented loop performs Proposal -> Top-K 10 -> FP-SHEP H4 -> frozen GAT-V1 at `t=0` and at every later theory event. It has no `N_replan_max`, cooldown, new threshold, hysteresis, or same-goal suppression. Handoff wins a same-tick conflict. Every switch preserved position, velocity, environment time, terminal goals, and DMP phase; the maximum observed phase delta was 0.

## 2. Corrected classical baselines

| Method | Corrected success | Collision | Timeout | Mean decisions | Mean total compute |
|---|---:|---:|---:|---:|---:|
| 3D-DWA-style | 391/400 (97.75%) | 1/400 (0.25%) | 8/400 (2.00%) | 44.015 | {dwa_total:.3f} ms |
| RVO/ORCA-style | 394/400 (98.50%) | 5/400 (1.25%) | 1/400 (0.25%) | 49.838 | {rvo_total:.3f} ms |

The sole baseline change replaces live-obstacle copying/private RNG rollout with `p_hat(t+h)=p(t)+h*v(t)`. Same current position/velocity with different private RNG states now gives identical predictions, while different true wandering futures leave the planner prediction unchanged. All horizons, buffers, weights, samples, speed/acceleration bounds, peer logic, static-obstacle logic, and outcome definitions are unchanged.

## 3. Complete online compute accounting

| Method / block | Decision latency | Decisions/episode | Cumulative planning | Actor execution | DMP internal | Total compute |
|---|---:|---:|---:|---:|---:|---:|
| Corrected DWA / original 400 | {float(dwa_runtime['mean_decision_latency_ms']):.3f} ms | {float(dwa_runtime['mean_decisions_per_episode']):.3f} | {float(dwa_runtime['mean_cumulative_planning_ms']):.3f} ms | — | — | {dwa_total:.3f} ms |
| Corrected RVO / original 400 | {float(rvo_runtime['mean_decision_latency_ms']):.3f} ms | {float(rvo_runtime['mean_decisions_per_episode']):.3f} | {float(rvo_runtime['mean_cumulative_planning_ms']):.3f} ms | — | — | {rvo_total:.3f} ms |
| Proposal / original 400 | {float(_runtime_lookup(runtime_rows, 'proposal')['mean_decision_latency_ms']):.3f} ms | 1.000 | {float(_runtime_lookup(runtime_rows, 'proposal')['mean_cumulative_planning_ms']):.3f} ms | {float(_runtime_lookup(runtime_rows, 'proposal')['mean_execution_actor_ms']):.3f} ms | {float(_runtime_lookup(runtime_rows, 'proposal')['mean_execution_dmp_ms']):.3f} ms | {float(_runtime_lookup(runtime_rows, 'proposal')['mean_total_online_algorithm_compute_ms']):.3f} ms |
| FP-SHEP / original 400 | {float(_runtime_lookup(runtime_rows, 'fp_shep')['mean_decision_latency_ms']):.3f} ms | 1.000 | {float(_runtime_lookup(runtime_rows, 'fp_shep')['mean_cumulative_planning_ms']):.3f} ms | {float(_runtime_lookup(runtime_rows, 'fp_shep')['mean_execution_actor_ms']):.3f} ms | {float(_runtime_lookup(runtime_rows, 'fp_shep')['mean_execution_dmp_ms']):.3f} ms | {float(_runtime_lookup(runtime_rows, 'fp_shep')['mean_total_online_algorithm_compute_ms']):.3f} ms |
| One-Shot Proposed / original 400 | {float(proposed_runtime['mean_decision_latency_ms']):.3f} ms | 1.000 | {float(proposed_runtime['mean_cumulative_planning_ms']):.3f} ms | {float(proposed_runtime['mean_execution_actor_ms']):.3f} ms | {float(proposed_runtime['mean_execution_dmp_ms']):.3f} ms | {float(proposed_runtime['mean_total_online_algorithm_compute_ms']):.3f} ms |
| Theory ERR / new dev 40 | {float(dev_err_runtime['mean_decision_latency_ms']):.3f} ms | {float(dev_err_runtime['mean_decisions_per_episode']):.3f} | {float(dev_err_runtime['mean_cumulative_planning_ms']):.3f} ms | {float(dev_err_runtime['mean_execution_actor_ms']):.3f} ms | {float(dev_err_runtime['mean_execution_dmp_ms']):.3f} ms | {err_total:.3f} ms |

Actor and GAT timings are CUDA-synchronized. FP-SHEP preview actor calls belong only to upper planning. Execution actor rows contain only real execution calls. DMP timing contains only the historical `compute_*_dmp_transition` kernel and excludes environment stepping, sensing, collision detection, obstacle simulation, and I/O. All 1600 replayed SAC-DMP episodes exactly matched the original positions, velocities, accelerations, candidate bundle, and categorical outcomes (`TIMING_REPLAY_BEHAVIOR_MATCH = YES`).

The 266.5 ms mean One-Shot high-level update is 2.67 times the 100 ms control period; it is a sparse high-level event, not a synchronous 10 Hz planner. Full ERR measured 14.775 upper decisions/episode (initial plus unique event ticks) and 14.3 agent-level reproposals/episode. Its mean total compute was {err_total:.1f} ms, versus {float(dev_one_runtime['mean_total_online_algorithm_compute_ms']):.1f} ms for paired one-shot development (+{err_total-float(dev_one_runtime['mean_total_online_algorithm_compute_ms']):.1f} ms).

## 4. Temporal observability on the original 400 trajectories

This phase is a read-only, non-interventional replay of the paper trigger; it never changes a stored trajectory or fabricates a counterfactual selected goal. Of 138 original Proposed collision episodes, 136 (98.6%) had a theory trigger before collision. Median collision warning lead was 1.8 s (P25 1.1, P75 2.2, P90 2.65 s); 129 had at least 0.5 s lead and 123 at least 1.0 s. All 173 Stage III/IV failures triggered before outcome, with a median 2.3 s lead. `THEORY_TRIGGER_OBSERVABILITY = STRONG`.

Success trajectories were not trigger-quiet: the 197 successes would have produced a mean 25.3 agent-level theoretical reproposals (median 26, P90 60.4) under the non-interventional replay. This correctly predicted the risk of over-reconstruction later observed in Phase D.

## 5. Paired recurrent-ERR development results

| Scope | One-shot success | Theory ERR success | One-shot collision | Theory ERR collision | One-shot timeout | Theory ERR timeout | ERR agent completion |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 42.5% | 72.5% | 37.5% | 12.5% | 20.0% | 15.0% | 84.2% |
| Stage I | 100.0% | 90.0% | 0.0% | 10.0% | 0.0% | 0.0% | 90.0% |
| Stage II | 70.0% | 50.0% | 20.0% | 30.0% | 10.0% | 20.0% | 70.0% |
| Stage III | 0.0% | 90.0% | 70.0% | 0.0% | 30.0% | 10.0% | 96.7% |
| Stage IV | 0.0% | 60.0% | 60.0% | 10.0% | 40.0% | 30.0% | 80.0% |

Paired recovery: 10 one-shot collisions became ERR successes, 5 one-shot timeouts became ERR successes, 2 one-shot collisions remained collisions, 3 one-shot collisions became timeouts, and 3 one-shot successes became ERR failures. Success discordances were ERR-only 15 versus one-shot-only 3 (exact McNemar p={exact_mcnemar_p(err_success_only, one_success_only):.6f}); collision discordances were one-shot-only {one_collision_only} versus ERR-only {err_collision_only} (p={exact_mcnemar_p(err_collision_only, one_collision_only):.6f}). These are development statistics, not formal claims.

The recovery is concentrated in the difficult stages. Stage II regressed from 70% to 50% success, and Stage I lost one success; the mechanism is not uniformly beneficial across distributions.

## 6. Reproposal and chattering diagnosis

Across 40 ERR episodes, there were 572 agent-level reproposals: 128 normal, 444 emergency, and 163 numerically same-goal. Mean/median/P90/max were 14.3/16.5/23/28. Mean per-agent inter-reproposal interval was 1.694 s; normal dwell violations were zero. Nevertheless, emergency events dominated (77.6%), and 82.1% of emergency events with a same-agent predecessor followed another emergency event. The pre-frozen high-frequency-emergency criterion therefore gives `CHATTERING_PRESENT = YES`.

This is not a software continuity failure: position, velocity, time, terminal goals, and DMP phase remained continuous. It is an execution-stability limitation of the current theory trigger. No cap, suppression, cooldown, or post-hoc threshold was added to hide it.

## 7. Answers to the six recovery questions

1. **Corrected classical performance:** DWA 97.75%; RVO 98.50% on the unchanged 400 scenarios.
2. **True online compute:** corrected DWA {dwa_total:.1f} ms/episode, corrected RVO {rvo_total:.1f} ms, original One-Shot Proposed {float(proposed_runtime['mean_total_online_algorithm_compute_ms']):.1f} ms, and development Theory ERR {err_total:.1f} ms. Component and call-level data are retained in the CSV artifacts.
3. **Cause of the Stage III/IV collapse:** the evidence identifies stale one-shot reference execution as a major contributor, not a proven sole cause. Every original difficult-stage failure was trigger-observable, and recurrent ERR recovered 15/20 difficult-stage successes; remaining timeouts, peer collisions, and easy-stage regressions show other limitations remain.
4. **Recovered capability:** overall +30 pp success, difficult stages +75 pp, collision -25 pp, and agent completion 58.3% -> 84.2% on the new development block.
5. **Reproposal/compute price:** 14.3 agent-level reproposals and 14.775 upper decisions per episode; +{err_total-float(dev_one_runtime['mean_total_online_algorithm_compute_ms']):.1f} ms mean total compute versus paired one-shot.
6. **Sparse stability:** no. The mechanism is recurrent and performance-effective, but emergency-dominated chattering is present.

## 8. Integrity, scope, and decision

- Corrected classics: 800 episodes; paired original 400 scenarios; current-state-only prediction; pre-performance code/config/manifest/metric hashes.
- Runtime replay: 1600 SAC-DMP episodes; 1600/1600 exact behavior matches; 223,392 original-block actor rows and 291,612 original-block DMP rows before development extension.
- Trigger audit: 400 source trajectories verified against their stored SHA-256 and read only.
- Development: 40 new scenarios, 10/stage; zero seed, exact geometry, or translation-equivalent overlap; identical paired initial selections.
- No GAT/SAC training, no FP-SHEP/Top-K/H4/checkpoint/gate/max-step/outcome change, no new safety layer, no V2/IA, and no new final benchmark.

Final decisions:

- `FUTURE_DYNAMIC_INFORMATION_LEAKAGE_AFTER = NO`
- `THEORY_ERR_DEVELOPMENT_GAIN = YES`
- `THEORY_ERR_PERFORMANCE_GAIN = YES`
- `THEORY_ERR_EXECUTION_STABILITY = NO`
- `CHATTERING_PRESENT = YES`
- `FULL_THEORY_ALIGNED_METHOD_FORMALLY_TESTED = NO`
- `RECOMMENDED_NEXT_STEP = THEORY_TRIGGER_NEEDS_REVISION`

The correct next action is a separate theory-revision goal addressing emergency-dominated repeated reconstruction, followed—only after a new development freeze—by a new untouched final benchmark. The current goal stops here.

## Artifact index

- `corrected_classical_results.csv`, `corrected_classical_reconciliation.json`
- `runtime_method_summary.csv`, `runtime_stage_summary.csv`, `actor_forward_timing.csv`, `dmp_timing.csv`
- `counterfactual_event_timeline.csv`, `temporal_observability.csv`
- `development_scenario_manifest.json`, `development_episode_results.csv`
- `reproposal_events.csv`, `reproposal_distribution.csv`, `paired_recovery.csv`, `runtime_tradeoff.csv`
- `theory_execution_contract.md`, `theory_code_alignment.csv`, `method_alignment.json`, `conclusion.json`
- `final_reconciliation.json`, `independent_reconciliation.json`
- `paper_ready/source_data/` and `paper_ready/captions/`

An independent raw-record reconciliation passed all freeze-hash, row-count,
unique-key, paired-scenario, behavior-replay, aggregate-reproduction,
continuity, and mandatory-conclusion-field checks.
"""
    (OUTPUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (paper / "README.md").write_text(
        "# Paper-ready tables\n\nSource tables and concise captions are provided. "
        "The recurrent ERR result is development-only and must not be presented as a formal final benchmark.\n",
        encoding="utf-8",
    )
    print(json.dumps(conclusion, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
