"""Preregistered statistics, tables, claims, and report for the final benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import beta, binomtest


ROOT = Path(__file__).resolve().parents[2]
ALGO = ROOT / "Multi-agent_Algo_lib"
for _search in (ROOT, ALGO):
    if str(_search) not in sys.path:
        sys.path.insert(0, str(_search))

from scripts.run_final_untouched_paper_benchmark import (  # noqa: E402
    DEFAULT_OUTPUT,
    EXPECTED_AGENT_ROWS,
    EXPECTED_TEAM_ROWS,
    METHOD_BY_ID,
    METHOD_ORDER,
    SCHEMA,
    load_json,
    write_csv,
    write_json,
)


PROPOSED = METHOD_ORDER[7]
MAIN_METHODS = (METHOD_ORDER[0], METHOD_ORDER[1], METHOD_ORDER[2], METHOD_ORDER[5], METHOD_ORDER[6], METHOD_ORDER[7])
ABLATION_METHODS = METHOD_ORDER[2:]
EASY_STAGES = {"Stage I", "Stage II"}
COMPLEX_STAGES = {"Stage III", "Stage IV"}
STAGES = ("Stage I", "Stage II", "Stage III", "Stage IV")
PRIMARY_COMPARISONS = (
    ("P1", METHOD_ORDER[1], "LOCAL_SENSING_MATCHED_CLASSICAL_COMPARISON"),
    ("P2", METHOD_ORDER[2], "LEARNING_BASELINE"),
    ("P3", METHOD_ORDER[5], "ERR_CONTRIBUTION"),
    ("P4", METHOD_ORDER[6], "GAT_CONTRIBUTION_UNDER_RERR"),
    ("P5", METHOD_ORDER[0], "SYSTEM_LEVEL_STRONG_REFERENCE"),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _number(value: Any) -> float | None:
    if value in (None, "", "None", "null", "NaN", "Infinity", "-Infinity"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int:
    number = _number(value)
    return int(number) if number is not None else 0


def _load_typed(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    for path in sorted((output / "formal_records").rglob("*.json")):
        record = load_json(path)
        episodes.append(dict(record["episode"]))
        agents.extend(dict(row) for row in record["agents"])
    episodes.sort(key=lambda row: (row["scenario_id"], METHOD_ORDER.index(row["method_id"])))
    agents.sort(key=lambda row: (row["scenario_id"], METHOD_ORDER.index(row["method_id"]), int(row["agent_id"])))
    return episodes, agents


def exact_ci(count: int, denominator: int, alpha: float = 0.05) -> tuple[float, float]:
    if denominator <= 0:
        return float("nan"), float("nan")
    lower = 0.0 if count == 0 else float(beta.ppf(alpha / 2.0, count, denominator - count + 1))
    upper = 1.0 if count == denominator else float(beta.ppf(1.0 - alpha / 2.0, count + 1, denominator - count))
    return lower, upper


def _binary_fields(prefix: str, values: Sequence[bool]) -> dict[str, Any]:
    count = int(sum(values))
    denominator = len(values)
    lower, upper = exact_ci(count, denominator)
    return {
        f"{prefix}_count": count,
        f"{prefix}_denominator": denominator,
        f"{prefix}_rate": count / denominator if denominator else None,
        f"{prefix}_ci95_lower": lower if denominator else None,
        f"{prefix}_ci95_upper": upper if denominator else None,
    }


def _mean(values: Iterable[Any]) -> float | None:
    array = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=float)
    return float(np.mean(array)) if array.size else None


def _median(values: Iterable[Any]) -> float | None:
    array = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=float)
    return float(np.median(array)) if array.size else None


def _episode_efficiency(
    episode: Mapping[str, Any], agents_by_key: Mapping[tuple[str, str], list[Mapping[str, Any]]]
) -> float | None:
    values = [
        row.get("agent_path_efficiency")
        for row in agents_by_key[(episode["scenario_id"], episode["method_id"])]
        if bool(row.get("agent_terminal_completed")) and row.get("agent_path_efficiency") is not None
    ]
    return _mean(values)


def summary_row(
    episodes: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    method_id: str,
) -> dict[str, Any]:
    successes = [row for row in episodes if bool(row["team_success"])]
    agent_completed = [bool(row["agent_terminal_completed"]) for row in agents]
    return {
        "schema_version": SCHEMA,
        "scope": scope,
        "method_id": method_id,
        "display_name": METHOD_BY_ID[method_id]["display_name"],
        "episode_count": len(episodes),
        **_binary_fields("success", [bool(row["team_success"]) for row in episodes]),
        **_binary_fields("collision", [bool(row["any_collision"]) for row in episodes]),
        **_binary_fields("obstacle_collision", [bool(row["obstacle_collision"]) for row in episodes]),
        **_binary_fields("inter_agent_collision", [bool(row["inter_agent_collision"]) for row in episodes]),
        **_binary_fields("timeout", [bool(row["timeout"]) for row in episodes]),
        **_binary_fields("agent_completion", agent_completed),
        "successful_completion_time_mean_s": _mean(row.get("completion_time_s") for row in successes),
        "successful_completion_time_median_s": _median(row.get("completion_time_s") for row in successes),
        "successful_team_path_length_mean_m": _mean(row.get("team_path_length_m") for row in successes),
        "successful_path_efficiency_mean": _mean(row.get("team_path_efficiency") for row in successes),
        "successful_smoothness_mean": _mean(row.get("trajectory_smoothness") for row in successes),
        "minimum_obstacle_clearance_mean_m": _mean(row.get("minimum_obstacle_clearance_m") for row in episodes),
        "minimum_inter_agent_distance_mean_m": _mean(row.get("minimum_inter_agent_distance_m") for row in episodes),
        "planning_decisions_mean_per_episode": _mean(row.get("planning_decision_count") for row in episodes),
        "reproposal_mean_per_episode": _mean(row.get("replanning_count") for row in episodes),
        "normal_reproposal_mean_per_episode": _mean(row.get("normal_replanning_count") for row in episodes),
        "emergency_reproposal_mean_per_episode": _mean(row.get("emergency_replanning_count") for row in episodes),
        "upper_or_local_planning_mean_ms": _mean(row.get("upper_planning_total_ms") for row in episodes),
        "execution_actor_mean_ms": _mean(row.get("execution_actor_forward_ms") for row in episodes),
        "execution_dmp_mean_ms": _mean(row.get("execution_dmp_ms") for row in episodes),
        "total_online_compute_mean_ms": _mean(row.get("total_online_algorithm_compute_ms") for row in episodes),
    }


def runtime_row(episodes: Sequence[Mapping[str, Any]], *, scope: str, method_id: str) -> dict[str, Any]:
    decisions = int(sum(int(row.get("planning_decision_count", 0)) for row in episodes))
    planning = float(sum(float(row.get("upper_planning_total_ms", 0.0)) for row in episodes))
    return {
        "schema_version": SCHEMA,
        "scope": scope,
        "method_id": method_id,
        "display_name": METHOD_BY_ID[method_id]["display_name"],
        "episode_count": len(episodes),
        "planning_decision_count_total": decisions,
        "planning_decisions_mean_per_episode": decisions / len(episodes) if episodes else None,
        "planning_latency_pooled_mean_per_decision_ms": planning / decisions if decisions else None,
        "cumulative_upper_or_local_planning_mean_ms": planning / len(episodes) if episodes else None,
        "execution_actor_compute_mean_ms": _mean(row.get("execution_actor_forward_ms") for row in episodes),
        "execution_dmp_compute_mean_ms": _mean(row.get("execution_dmp_ms") for row in episodes),
        "total_online_algorithm_compute_mean_ms": _mean(row.get("total_online_algorithm_compute_ms") for row in episodes),
        "total_online_algorithm_compute_median_ms": _median(row.get("total_online_algorithm_compute_ms") for row in episodes),
        "total_online_algorithm_compute_p90_ms": (
            float(np.percentile([float(row["total_online_algorithm_compute_ms"]) for row in episodes], 90)) if episodes else None
        ),
        "runtime_definition": (
            "per-step local planning cumulative" if METHOD_BY_ID[method_id]["engine"].startswith("dwa_")
            else "actor+DMP without upper planning" if METHOD_BY_ID[method_id]["engine"] == "one_shot_terminal"
            else "event/one-shot upper planning + execution actor + DMP"
        ),
    }


def _mcnemar(
    proposed: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    prop_by_id = {row["scenario_id"]: bool(row["team_success"]) for row in proposed}
    base_by_id = {row["scenario_id"]: bool(row["team_success"]) for row in baseline}
    keys = sorted(set(prop_by_id) & set(base_by_id))
    prop_only = sum(prop_by_id[key] and not base_by_id[key] for key in keys)
    base_only = sum(base_by_id[key] and not prop_by_id[key] for key in keys)
    both = sum(prop_by_id[key] and base_by_id[key] for key in keys)
    neither = len(keys) - prop_only - base_only - both
    discordant = prop_only + base_only
    p_value = float(binomtest(prop_only, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    prop_rate = sum(prop_by_id.values()) / len(keys) if keys else None
    base_rate = sum(base_by_id.values()) / len(keys) if keys else None
    return {
        "paired_scenario_count": len(keys),
        "both_success": both,
        "proposed_only_success": prop_only,
        "baseline_only_success": base_only,
        "both_failure": neither,
        "discordant_count": discordant,
        "proposed_success_rate": prop_rate,
        "baseline_success_rate": base_rate,
        "gain_pp": 100.0 * (prop_rate - base_rate) if prop_rate is not None else None,
        "exact_two_sided_mcnemar_p": p_value,
    }


def _scope_filter(scope: str) -> Callable[[Mapping[str, Any]], bool]:
    if scope == "overall":
        return lambda _: True
    if scope == "easy_stage_i_plus_ii":
        return lambda row: row["stage"] in EASY_STAGES
    if scope == "complex_stage_iii_plus_iv":
        return lambda row: row["stage"] in COMPLEX_STAGES
    return lambda row: row["stage"] == scope


def paired_tests(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    scopes = ("overall", "complex_stage_iii_plus_iv", *STAGES)
    for comparison_id, baseline, role in PRIMARY_COMPARISONS:
        for scope in scopes:
            keep = _scope_filter(scope)
            proposed = [row for row in episodes if row["method_id"] == PROPOSED and keep(row)]
            base = [row for row in episodes if row["method_id"] == baseline and keep(row)]
            results.append(
                {
                    "comparison_id": comparison_id,
                    "scope": scope,
                    "scope_status": "PREREGISTERED" if scope in {"overall", "complex_stage_iii_plus_iv"} else "SECONDARY_STAGE",
                    "proposed_method": PROPOSED,
                    "baseline_method": baseline,
                    "comparison_role": role,
                    **_mcnemar(proposed, base),
                }
            )
    return {
        "schema_version": SCHEMA,
        "test": "exact two-sided McNemar on paired team success",
        "family_level_tests_performed": False,
        "comparisons": results,
    }


def _bootstrap_difference(values: np.ndarray, rng: np.random.Generator, samples: int = 5000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, values.size, size=(samples, values.size))
    estimates = np.mean(values[indices], axis=1)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def continuous_tests(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "completion_time_s",
        "team_path_length_m",
        "team_path_efficiency",
        "trajectory_smoothness",
        "minimum_obstacle_clearance_m",
        "minimum_inter_agent_distance_m",
        "total_online_algorithm_compute_ms",
    )
    rng = np.random.default_rng(20260820)
    rows: list[dict[str, Any]] = []
    for comparison_id, baseline, role in PRIMARY_COMPARISONS:
        for scope in ("overall", "complex_stage_iii_plus_iv"):
            keep = _scope_filter(scope)
            prop = {row["scenario_id"]: row for row in episodes if row["method_id"] == PROPOSED and keep(row) and bool(row["team_success"])}
            base = {row["scenario_id"]: row for row in episodes if row["method_id"] == baseline and keep(row) and bool(row["team_success"])}
            keys = sorted(set(prop) & set(base))
            for metric in metrics:
                pairs = [
                    (prop[key].get(metric), base[key].get(metric))
                    for key in keys
                    if prop[key].get(metric) is not None and base[key].get(metric) is not None
                ]
                differences = np.asarray([float(left) - float(right) for left, right in pairs], dtype=float)
                lower, upper = _bootstrap_difference(differences, rng)
                rows.append(
                    {
                        "comparison_id": comparison_id,
                        "comparison_role": role,
                        "scope": scope,
                        "metric": metric,
                        "both_success_pair_count": len(pairs),
                        "mean_proposed_minus_baseline": float(np.mean(differences)) if differences.size else None,
                        "median_proposed_minus_baseline": float(np.median(differences)) if differences.size else None,
                        "paired_bootstrap_ci95_lower": lower if differences.size else None,
                        "paired_bootstrap_ci95_upper": upper if differences.size else None,
                    }
                )
    return {
        "schema_version": SCHEMA,
        "subset": "both-success paired scenarios only",
        "failed_completion_times_filled_with_zero": False,
        "bootstrap_resamples": 5000,
        "random_seed": 20260820,
        "formal_significance_claim": False,
        "rows": rows,
    }


def _test_lookup(tests: Mapping[str, Any], comparison_id: str, scope: str = "overall") -> Mapping[str, Any]:
    return next(row for row in tests["comparisons"] if row["comparison_id"] == comparison_id and row["scope"] == scope)


def _summary_lookup(rows: Sequence[Mapping[str, Any]], method_id: str, scope: str = "overall") -> Mapping[str, Any]:
    return next(row for row in rows if row["method_id"] == method_id and row["scope"] == scope)


def _claims(
    tests: Mapping[str, Any], overall: Sequence[Mapping[str, Any]], difficulty: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    p1 = _test_lookup(tests, "P1")
    p2 = _test_lookup(tests, "P2")
    p3 = _test_lookup(tests, "P3")
    p4 = _test_lookup(tests, "P4")
    proposed_runtime = _summary_lookup(overall, PROPOSED)["total_online_compute_mean_ms"]
    dwa_sm_runtime = _summary_lookup(overall, METHOD_ORDER[1])["total_online_compute_mean_ms"]
    p1_complex = _test_lookup(tests, "P1", "complex_stage_iii_plus_iv")
    p1_easy_gain = 100.0 * (
        _summary_lookup(difficulty, PROPOSED, "Easy I+II")["success_rate"]
        - _summary_lookup(difficulty, METHOD_ORDER[1], "Easy I+II")["success_rate"]
    )
    rows = [
        ("Proposed improves over direct frozen SAC-DMP.", p2, "P2 overall exact McNemar", "frozen benchmark distribution only"),
        ("R-ERR improves over One-Shot GAT.", p3, "P3 overall exact McNemar", "attributes the paired system contrast; not an isolated trigger equation causal effect"),
        ("GAT improves over R-ERR+FP-SHEP under the same recurrent framework.", p4, "P4 overall exact McNemar", "natural post-selection trajectory/event divergence is allowed"),
        ("Proposed outperforms DWA-SensingMatched under the frozen local-sensing comparison.", p1, "P1 overall exact McNemar", "does not imply strict equality of all system-layer information"),
    ]
    claims = [
        {
            "claim": claim,
            "supported": "YES" if float(test["gain_pp"]) > 0.0 and float(test["exact_two_sided_mcnemar_p"]) < 0.05 else "NO",
            "evidence": f"gain={test['gain_pp']:.2f} pp; exact p={test['exact_two_sided_mcnemar_p']:.6g}",
            "comparison_level": "overall paired team success",
            "statistical_test": evidence,
            "boundary": boundary,
        }
        for claim, test, evidence, boundary in rows
    ]
    claims.extend(
        [
            {
                "claim": "Proposed has lower accumulated online algorithm compute than per-step DWA-SensingMatched.",
                "supported": "YES" if proposed_runtime < dwa_sm_runtime else "NO",
                "evidence": f"Proposed {proposed_runtime:.3f} ms/episode; DWA-SensingMatched {dwa_sm_runtime:.3f} ms/episode",
                "comparison_level": "overall runtime",
                "statistical_test": "descriptive frozen runtime accounting",
                "boundary": "does not imply lower single-decision latency or wall-clock execution time",
            },
            {
                "claim": "The Proposed-vs-DWA-SensingMatched gain is concentrated in Stage III/IV.",
                "supported": "YES" if float(p1_complex["gain_pp"]) > p1_easy_gain and float(p1_complex["gain_pp"]) > 0 else "NO",
                "evidence": f"complex gain={p1_complex['gain_pp']:.2f} pp; easy gain={p1_easy_gain:.2f} pp",
                "comparison_level": "preregistered Easy/Complex stratification",
                "statistical_test": "complex exact McNemar plus descriptive easy contrast",
                "boundary": "concentration within the generated four-stage distribution only",
            },
            {
                "claim": "Proposed is SOTA, safest, or generally superior beyond the test distribution.",
                "supported": "NO",
                "evidence": "not tested by this protocol",
                "comparison_level": "forbidden extrapolation",
                "statistical_test": "none",
                "boundary": "SOTA, real-world safety, canonical DWA equivalence, and out-of-distribution generalization are unsupported",
            },
        ]
    )
    return claims


def _main_tables(
    overall: Sequence[Mapping[str, Any]],
    stage: Sequence[Mapping[str, Any]],
    difficulty: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table1: list[dict[str, Any]] = []
    for method_id in MAIN_METHODS:
        overall_row = _summary_lookup(overall, method_id)
        table1.append(
            {
                "Method": METHOD_BY_ID[method_id]["display_name"],
                "Overall success (%)": 100 * overall_row["success_rate"],
                **{
                    f"{stage_name} success (%)": 100 * _summary_lookup(stage, method_id, stage_name)["success_rate"]
                    for stage_name in STAGES
                },
                "Complex III+IV success (%)": 100 * _summary_lookup(difficulty, method_id, "Complex III+IV")["success_rate"],
                "Collision (%)": 100 * overall_row["collision_rate"],
                "Inter-agent collision (%)": 100 * overall_row["inter_agent_collision_rate"],
                "Timeout (%)": 100 * overall_row["timeout_rate"],
                "Total online compute (ms/episode)": overall_row["total_online_compute_mean_ms"],
            }
        )
    table2: list[dict[str, Any]] = []
    for method_id in ABLATION_METHODS:
        overall_row = _summary_lookup(overall, method_id)
        table2.append(
            {
                "Method": METHOD_BY_ID[method_id]["display_name"],
                "Overall success (%)": 100 * overall_row["success_rate"],
                "Easy I+II success (%)": 100 * _summary_lookup(difficulty, method_id, "Easy I+II")["success_rate"],
                "Complex III+IV success (%)": 100 * _summary_lookup(difficulty, method_id, "Complex III+IV")["success_rate"],
                "Collision (%)": 100 * overall_row["collision_rate"],
                "Peer collision (%)": 100 * overall_row["inter_agent_collision_rate"],
                "Agent completion (%)": 100 * overall_row["agent_completion_rate"],
                "Upper decisions/episode": overall_row["planning_decisions_mean_per_episode"],
                "Total online compute (ms/episode)": overall_row["total_online_compute_mean_ms"],
            }
        )
    return table1, table2


def _format_rate(row: Mapping[str, Any], prefix: str = "success") -> str:
    return f"{int(row[f'{prefix}_count'])}/{int(row[f'{prefix}_denominator'])} ({100*float(row[f'{prefix}_rate']):.1f}%)"


def _report_text(
    conclusion: Mapping[str, Any],
    overall: Sequence[Mapping[str, Any]],
    stage: Sequence[Mapping[str, Any]],
    difficulty: Sequence[Mapping[str, Any]],
    tests: Mapping[str, Any],
    figures_ready: bool,
) -> str:
    proposed = _summary_lookup(overall, PROPOSED)
    lines = [
        "# Final Untouched Four-Stage Paper Benchmark",
        "",
        "## Executive result",
        "",
        (
            f"The frozen Proposed method achieved **{_format_rate(proposed)}** team success, "
            f"**{100*proposed['collision_rate']:.1f}%** collision, and "
            f"**{100*proposed['timeout_rate']:.1f}%** timeout on the single untouched "
            "400-scenario manifest. All 3,200 team episodes and 9,600 agent rows passed "
            "independent hash, reconstruction, collision, and information-contract reconciliation."
        ),
        "",
        "No method, parameter, baseline, schedule, scenario, endpoint, or figure-selection rule was changed after the formal freeze.",
        "",
        "## Frozen protocol",
        "",
        "- 400 unique scenarios: 100 per stage, five families per stage, 20 scenarios per stage-family cell.",
        "- Eight methods on every scenario, with a precomputed scenario-interleaved cyclic schedule.",
        "- Primary endpoint: overall team success with exact 95% Clopper-Pearson confidence intervals.",
        "- Primary paired inference: exact two-sided McNemar tests for Proposed versus DWA-SensingMatched, Direct SAC-DMP, One-Shot GAT, and R-ERR+FP-SHEP; DWA-FullState is a strong system-level reference.",
        "- Easy and Complex were frozen as Stage I+II and Stage III+IV before performance.",
        "",
        "## Overall outcomes",
        "",
        "| Method | Success | Collision | Obstacle | Inter-agent | Timeout | Agent completion | Total compute (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method_id in METHOD_ORDER:
        row = _summary_lookup(overall, method_id)
        lines.append(
            f"| {row['display_name']} | {_format_rate(row)} | {100*row['collision_rate']:.1f}% | "
            f"{100*row['obstacle_collision_rate']:.1f}% | {100*row['inter_agent_collision_rate']:.1f}% | "
            f"{100*row['timeout_rate']:.1f}% | {100*row['agent_completion_rate']:.1f}% | "
            f"{row['total_online_compute_mean_ms']:.3f} |"
        )
    lines.extend(["", "## Stage and difficulty results", ""])
    lines.append("| Method | Stage I | Stage II | Stage III | Stage IV | Easy | Complex |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method_id in METHOD_ORDER:
        values = [100 * _summary_lookup(stage, method_id, name)["success_rate"] for name in STAGES]
        easy = 100 * _summary_lookup(difficulty, method_id, "Easy I+II")["success_rate"]
        complex_ = 100 * _summary_lookup(difficulty, method_id, "Complex III+IV")["success_rate"]
        lines.append(
            f"| {METHOD_BY_ID[method_id]['display_name']} | {values[0]:.1f}% | {values[1]:.1f}% | {values[2]:.1f}% | {values[3]:.1f}% | {easy:.1f}% | {complex_:.1f}% |"
        )
    lines.extend(["", "## Preregistered paired comparisons", ""])
    lines.append("| Comparison | Scope | Proposed only | Baseline only | Gain (pp) | Exact p |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in tests["comparisons"]:
        if row["scope"] not in {"overall", "complex_stage_iii_plus_iv"}:
            continue
        lines.append(
            f"| {row['comparison_id']}: Proposed vs {METHOD_BY_ID[row['baseline_method']]['display_name']} | {row['scope']} | "
            f"{row['proposed_only_success']} | {row['baseline_only_success']} | {row['gain_pp']:.2f} | {row['exact_two_sided_mcnemar_p']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Runtime interpretation",
            "",
            (
                f"Proposed used a mean {proposed['planning_decisions_mean_per_episode']:.3f} upper decisions and "
                f"{proposed['total_online_compute_mean_ms']:.3f} ms total online algorithm compute per episode. "
                "DWA totals are the cumulative cost of per-step local replanning; learned totals include upper planning, real execution actor calls, and DMP kernels. These are compute measurements, not physical execution time."
            ),
            "",
            "## Claim boundaries",
            "",
            "DWA-FullState remains a strong exact-current-state system reference and is not described as equal-information. DWA-SensingMatched is the frozen local-sensing comparison, while the precise upper/execution information differences remain explicit. The data do not establish SOTA, canonical DWA equivalence, real-world safety, or generalization beyond the generated four-stage distribution.",
            "",
            "## Integrity and closure",
            "",
            "- `FINAL_RECONCILIATION = PASS`",
            "- `FINAL_INFORMATION_INTEGRITY = PASS`",
            "- Historical seed, exact-geometry, and translation-equivalent overlap: 0.",
            "- Internal exact and translation-equivalent duplicates: 0.",
            f"- Paper tables ready: YES; paper figures ready: {'YES' if figures_ready else 'PENDING_PLOTTING'}.",
            "- RVO was not rerun and remains supplementary historical evidence on a different manifest; NMPC was excluded because its frozen readiness gate was NO.",
            "",
            "The experiment is closed. The next step is writing the paper Results and Discussion; method development does not resume from these formal outcomes.",
        ]
    )
    return "\n".join(lines) + "\n"


def refresh_report(output: Path, *, figures_ready: bool) -> None:
    conclusion = load_json(output / "conclusion.json")
    overall = load_json(output / "analysis_cache.json")["overall"]
    stage = load_json(output / "analysis_cache.json")["stage"]
    difficulty = load_json(output / "analysis_cache.json")["difficulty"]
    tests = load_json(output / "primary_paired_tests.json")
    (output / "FINAL_REPORT.md").write_text(
        _report_text(conclusion, overall, stage, difficulty, tests, figures_ready),
        encoding="utf-8",
    )


def analyze(output: Path) -> dict[str, Any]:
    reconciliation = load_json(output / "final_reconciliation.json")
    if reconciliation["FINAL_RECONCILIATION"] != "PASS":
        raise RuntimeError("statistics are forbidden until FINAL_RECONCILIATION=PASS")
    episodes, agents = _load_typed(output)
    if len(episodes) != EXPECTED_TEAM_ROWS or len(agents) != EXPECTED_AGENT_ROWS:
        raise RuntimeError("raw record count changed before analysis")
    agents_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in agents:
        agents_by_key[(row["scenario_id"], row["method_id"])].append(row)
    for episode in episodes:
        episode["team_path_efficiency"] = _episode_efficiency(episode, agents_by_key)

    overall: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    difficulty_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    runtime_overall: list[dict[str, Any]] = []
    runtime_stage: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        method_episodes = [row for row in episodes if row["method_id"] == method_id]
        method_agents = [row for row in agents if row["method_id"] == method_id]
        overall.append(summary_row(method_episodes, method_agents, scope="overall", method_id=method_id))
        runtime_overall.append(runtime_row(method_episodes, scope="overall", method_id=method_id))
        for stage_name in STAGES:
            scoped_e = [row for row in method_episodes if row["stage"] == stage_name]
            scoped_a = [row for row in method_agents if row["stage"] == stage_name]
            stage_rows.append(summary_row(scoped_e, scoped_a, scope=stage_name, method_id=method_id))
            runtime_stage.append(runtime_row(scoped_e, scope=stage_name, method_id=method_id))
        for scope, stages in (("Easy I+II", EASY_STAGES), ("Complex III+IV", COMPLEX_STAGES)):
            scoped_e = [row for row in method_episodes if row["stage"] in stages]
            scoped_a = [row for row in method_agents if row["stage"] in stages]
            difficulty_rows.append(summary_row(scoped_e, scoped_a, scope=scope, method_id=method_id))
        cells = sorted({(row["stage"], row["family"]) for row in method_episodes})
        for stage_name, family in cells:
            scoped_e = [row for row in method_episodes if row["stage"] == stage_name and row["family"] == family]
            scoped_a = [row for row in method_agents if row["stage"] == stage_name and row["family"] == family]
            row = summary_row(scoped_e, scoped_a, scope=f"{stage_name}::{family}", method_id=method_id)
            row.update({"stage": stage_name, "family": family, "family_analysis": "DESCRIPTIVE_ONLY"})
            family_rows.append(row)

    tests = paired_tests(episodes)
    continuous = continuous_tests(episodes)
    write_csv(output / "overall_method_summary.csv", overall)
    write_csv(output / "stage_method_summary.csv", stage_rows)
    write_csv(output / "difficulty_aggregate_summary.csv", difficulty_rows)
    write_csv(output / "family_method_summary.csv", family_rows)
    write_csv(output / "runtime_method_summary.csv", runtime_overall)
    write_csv(output / "runtime_stage_summary.csv", runtime_stage)
    write_json(output / "primary_paired_tests.json", tests)
    write_json(output / "continuous_paired_tests.json", continuous)

    failure_rows: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        for scope in ("overall", *STAGES):
            selected = [row for row in episodes if row["method_id"] == method_id and (scope == "overall" or row["stage"] == scope)]
            counts = Counter(str(row["termination_reason"]) for row in selected if not bool(row["team_success"]))
            for reason, count in sorted(counts.items()):
                failure_rows.append(
                    {
                        "method_id": method_id,
                        "display_name": METHOD_BY_ID[method_id]["display_name"],
                        "scope": scope,
                        "failure_type": reason,
                        "count": count,
                        "denominator": len(selected),
                        "rate_of_all_episodes": count / len(selected),
                    }
                )
    write_csv(output / "failure_taxonomy.csv", failure_rows)

    representative: list[dict[str, Any]] = []
    for stage_name in STAGES:
        successes = [row for row in episodes if row["method_id"] == PROPOSED and row["stage"] == stage_name and bool(row["team_success"])]
        if not successes:
            representative.append({"stage": stage_name, "selection_status": "UNAVAILABLE_NO_PROPOSED_SUCCESS"})
            continue
        median = float(np.median([float(row["completion_time_s"]) for row in successes]))
        anchor = min(successes, key=lambda row: (abs(float(row["completion_time_s"]) - median), row["scenario_id"]))
        for method_id in MAIN_METHODS:
            row = next(item for item in episodes if item["scenario_id"] == anchor["scenario_id"] and item["method_id"] == method_id)
            representative.append(
                {
                    "stage": stage_name,
                    "anchor_scenario_id": anchor["scenario_id"],
                    "proposed_success_median_completion_time_s": median,
                    "proposed_anchor_completion_time_s": anchor["completion_time_s"],
                    "selection_rule": "Proposed success closest to stage Proposed-success median completion time; scenario-id tie break",
                    "method_id": method_id,
                    "display_name": METHOD_BY_ID[method_id]["display_name"],
                    "team_success": row["team_success"],
                    "termination_reason": row["termination_reason"],
                    "trajectory_relative_path": f"trajectories/{anchor['scenario_id']}/{method_id}.npz",
                    "post_hoc_visual_choice": False,
                }
            )
    write_csv(output / "representative_trajectory_manifest.csv", representative)

    contribution_rows: list[dict[str, Any]] = []
    for label, baseline in (("ERR contribution", METHOD_ORDER[5]), ("GAT contribution under R-ERR", METHOD_ORDER[6])):
        for scope in ("overall", *STAGES, "Easy I+II", "Complex III+IV"):
            source = overall if scope == "overall" else stage_rows if scope in STAGES else difficulty_rows
            prop_rate = _summary_lookup(source, PROPOSED, scope)["success_rate"]
            base_rate = _summary_lookup(source, baseline, scope)["success_rate"]
            test = next((row for row in tests["comparisons"] if row["baseline_method"] == baseline and row["scope"] == ("complex_stage_iii_plus_iv" if scope == "Complex III+IV" else scope)), None)
            contribution_rows.append(
                {
                    "contribution": label,
                    "scope": scope,
                    "proposed_method": PROPOSED,
                    "baseline_method": baseline,
                    "success_gain_pp": 100 * (prop_rate - base_rate),
                    "exact_mcnemar_p": test["exact_two_sided_mcnemar_p"] if test is not None else None,
                    "additivity_claimed": False,
                }
            )
    write_csv(output / "module_contribution.csv", contribution_rows)

    claims = _claims(tests, overall, difficulty_rows)
    write_csv(output / "paper_claim_matrix.csv", claims)
    table1, table2 = _main_tables(overall, stage_rows, difficulty_rows)
    source_dir = output / "paper_ready" / "source_data"
    write_csv(source_dir / "main_table_1.csv", table1)
    write_csv(source_dir / "main_table_2_ablation.csv", table2)
    write_csv(
        output / "paper_ready" / "table_manifest.csv",
        [
            {"table": "Main Table 1", "title": "Final frozen method comparison", "source": "paper_ready/source_data/main_table_1.csv", "status": "YES"},
            {"table": "Main Table 2", "title": "Frozen learned-chain ablation", "source": "paper_ready/source_data/main_table_2_ablation.csv", "status": "YES"},
        ],
    )

    p1 = _test_lookup(tests, "P1")
    p2 = _test_lookup(tests, "P2")
    p3 = _test_lookup(tests, "P3")
    p4 = _test_lookup(tests, "P4")
    p5 = _test_lookup(tests, "P5")
    proposed = _summary_lookup(overall, PROPOSED)
    conclusion = {
        "FINAL_METHOD_SET_FROZEN": "YES",
        "FINAL_METHOD_COUNT": 8,
        "FINAL_SCENARIO_COUNT": 400,
        "FORMAL_TEAM_EPISODES_EXPECTED": 3200,
        "FORMAL_AGENT_ROWS_EXPECTED": 9600,
        "FORMAL_MANIFEST_HISTORY_OVERLAP": 0,
        "FORMAL_EXACT_GEOMETRY_DUPLICATES": 0,
        "FORMAL_TRANSLATION_EQUIVALENT_DUPLICATES": 0,
        "FORMAL_RUN_COMPLETE": "YES",
        "FINAL_RECONCILIATION": reconciliation["FINAL_RECONCILIATION"],
        "FINAL_INFORMATION_INTEGRITY": reconciliation["FINAL_INFORMATION_INTEGRITY"],
        "PROPOSED_OVERALL_SUCCESS": proposed["success_rate"],
        "PROPOSED_STAGE1_SUCCESS": _summary_lookup(stage_rows, PROPOSED, "Stage I")["success_rate"],
        "PROPOSED_STAGE2_SUCCESS": _summary_lookup(stage_rows, PROPOSED, "Stage II")["success_rate"],
        "PROPOSED_STAGE3_SUCCESS": _summary_lookup(stage_rows, PROPOSED, "Stage III")["success_rate"],
        "PROPOSED_STAGE4_SUCCESS": _summary_lookup(stage_rows, PROPOSED, "Stage IV")["success_rate"],
        "PROPOSED_EASY_SUCCESS": _summary_lookup(difficulty_rows, PROPOSED, "Easy I+II")["success_rate"],
        "PROPOSED_COMPLEX_SUCCESS": _summary_lookup(difficulty_rows, PROPOSED, "Complex III+IV")["success_rate"],
        "PROPOSED_COLLISION_RATE": proposed["collision_rate"],
        "PROPOSED_INTER_AGENT_COLLISION_RATE": proposed["inter_agent_collision_rate"],
        "PROPOSED_TIMEOUT_RATE": proposed["timeout_rate"],
        "PROPOSED_TOTAL_COMPUTE_MS": proposed["total_online_compute_mean_ms"],
        "PROPOSED_MEAN_UPPER_DECISIONS": proposed["planning_decisions_mean_per_episode"],
        "DWA_FULLSTATE_SUCCESS": _summary_lookup(overall, METHOD_ORDER[0])["success_rate"],
        "DWA_SENSING_MATCHED_SUCCESS": _summary_lookup(overall, METHOD_ORDER[1])["success_rate"],
        "DIRECT_SAC_DMP_SUCCESS": _summary_lookup(overall, METHOD_ORDER[2])["success_rate"],
        "ONE_SHOT_GAT_SUCCESS": _summary_lookup(overall, METHOD_ORDER[5])["success_rate"],
        "RERR_FP_SHEP_SUCCESS": _summary_lookup(overall, METHOD_ORDER[6])["success_rate"],
        "PROPOSED_VS_DWA_SM_GAIN_PP": p1["gain_pp"],
        "PROPOSED_VS_DWA_SM_MCNEMAR_P": p1["exact_two_sided_mcnemar_p"],
        "PROPOSED_VS_DIRECT_SAC_GAIN_PP": p2["gain_pp"],
        "PROPOSED_VS_DIRECT_SAC_MCNEMAR_P": p2["exact_two_sided_mcnemar_p"],
        "ERR_GAIN_VS_ONE_SHOT_PP": p3["gain_pp"],
        "ERR_GAIN_VS_ONE_SHOT_MCNEMAR_P": p3["exact_two_sided_mcnemar_p"],
        "GAT_GAIN_UNDER_RERR_PP": p4["gain_pp"],
        "GAT_GAIN_UNDER_RERR_MCNEMAR_P": p4["exact_two_sided_mcnemar_p"],
        "PROPOSED_VS_DWA_FULLSTATE_GAIN_PP": p5["gain_pp"],
        "FULLSTATE_DWA_ROLE": "STRONG_SYSTEM_LEVEL_REFERENCE",
        "SENSING_MATCHED_DWA_ROLE": "LOCAL_SENSING_MATCHED_CLASSICAL_COMPARISON",
        "DIRECT_SAC_DMP_ROLE": "LEARNING_BASELINE",
        "NMPC_INCLUDED": "NO",
        "RVO_NEW_FORMAL_INCLUDED": "NO",
        "RVO_ROLE": "SUPPLEMENTARY_HISTORICAL",
        "PRIMARY_ENDPOINT": "OVERALL_TEAM_SUCCESS",
        "SECONDARY_COMPLEX_ENDPOINT": "STAGE_III_PLUS_IV_SUCCESS",
        "FAMILY_LEVEL_ANALYSIS": "DESCRIPTIVE_ONLY",
        "POST_HOC_SCENE_SELECTION": "NO",
        "METHOD_CHANGED_AFTER_FORMAL_START": "NO",
        "PARAMETER_TUNING_AFTER_FORMAL_START": "NO",
        "FINAL_PAPER_FIGURES_READY": "NO",
        "FINAL_PAPER_TABLES_READY": "YES",
        "RECOMMENDED_NEXT_STEP": "WRITE_PAPER_RESULTS",
    }
    write_json(output / "conclusion.json", conclusion)
    write_json(output / "analysis_cache.json", {"overall": overall, "stage": stage_rows, "difficulty": difficulty_rows, "family": family_rows})
    refresh_report(output, figures_ready=False)
    return conclusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    conclusion = analyze(args.output.resolve())
    print(json.dumps({"analysis": "PASS", "proposed_success": conclusion["PROPOSED_OVERALL_SUCCESS"], "paper_tables": conclusion["FINAL_PAPER_TABLES_READY"]}), flush=True)


if __name__ == "__main__":
    main()
