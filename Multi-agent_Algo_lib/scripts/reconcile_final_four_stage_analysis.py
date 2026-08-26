"""Independent reconciliation of final tables, statistics, figures, and report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
from scipy.stats import binomtest


METHODS = ("dwa_style", "rvo_orca_style", "terminal", "proposal", "fp_shep", "gat_v1")
STAGES = ("stage_1", "stage_2", "stage_3", "stage_4")
METRICS = ("team_success", "any_collision", "timeout")
CONTINUOUS = ("completion_time_s", "team_path_length_m", "trajectory_smoothness", "minimum_inter_agent_distance_m")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if left in (None, "") and right in (None, ""):
        return True
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left == right


def bootstrap(values: np.ndarray, rng: np.random.Generator, samples: int = 5000) -> tuple[float, float]:
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = np.mean(values[draws], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    failures: list[str] = []

    integrity = load_json(out / "independent_reconciliation.json")
    if integrity.get("status") != "PASSED":
        failures.append("independent_raw_reconciliation_not_passed")
    episodes = read_csv(out / "formal_episode_results.csv")
    agents = read_csv(out / "formal_agent_results.csv")
    runtimes = read_csv(out / "planning_runtime_records.csv")
    method_summary = read_csv(out / "method_summary.csv")
    stage_summary = read_csv(out / "stage_summary.csv")
    conclusion = load_json(out / "conclusion.json")
    statistics = load_json(out / "statistical_tests.json")

    episode_keys = [(row["stage"], row["scenario_id"], row["method"]) for row in episodes]
    agent_keys = [(row["stage"], row["scenario_id"], row["method"], row["agent_id"]) for row in agents]
    if len(episodes) != 2400 or len(set(episode_keys)) != 2400:
        failures.append("formal_episode_csv_count_or_uniqueness")
    if len(agents) != 7200 or len(set(agent_keys)) != 7200:
        failures.append("formal_agent_csv_count_or_uniqueness")
    if not runtimes:
        failures.append("planning_runtime_csv_empty")

    # Reconcile flat CSV fields against all original JSON records.
    flat = {key: row for key, row in zip(episode_keys, episodes)}
    for path in sorted((out / "formal_records").glob("stage_*/*/*.json")):
        record = load_json(path)
        episode = record["episode"]
        key = (record["stage"], record["scenario_id"], record["method"])
        row = flat.get(key)
        if row is None:
            failures.append(f"flat_missing:{key}")
            continue
        comparisons = {
            "team_success": bool(episode["team_success"]),
            "any_collision": bool(episode.get("any_collision", episode.get("collision", False))),
            "obstacle_collision": bool(episode.get("obstacle_collision", False)),
            "inter_agent_collision": bool(episode.get("inter_agent_collision", False)),
            "timeout": bool(episode.get("timeout", False)),
        }
        for field, expected in comparisons.items():
            if boolean(row[field]) != expected:
                failures.append(f"raw_flat_mismatch:{key}:{field}")
        if not close(row["team_path_length_m"], episode["team_path_length_m"]):
            failures.append(f"raw_flat_mismatch:{key}:team_path_length_m")
        if bool(episode["team_success"]) != bool(row["completion_time_s"]):
            failures.append(f"completion_missingness:{key}")

    # Completion efficiency is success-only at both levels.
    episode_success = {(row["stage"], row["scenario_id"], row["method"]): boolean(row["team_success"]) for row in episodes}
    for row in episodes:
        success = boolean(row["team_success"])
        if success != bool(row["team_path_efficiency"]):
            failures.append(f"team_efficiency_missingness:{row['scenario_id']}:{row['method']}")
    for row in agents:
        completed = boolean(row["agent_terminal_completed"])
        if completed != bool(row["agent_path_efficiency"]):
            failures.append(f"agent_efficiency_missingness:{row['scenario_id']}:{row['method']}:{row['agent_id']}")

    method_lookup = {row["method"]: row for row in method_summary}
    stage_lookup = {(row["stage"], row["method"]): row for row in stage_summary}
    for method in METHODS:
        subset = [row for row in episodes if row["method"] == method]
        summary = method_lookup[method]
        if int(summary["n"]) != 400:
            failures.append(f"method_n:{method}")
        for metric in METRICS + ("obstacle_collision", "inter_agent_collision"):
            count = sum(boolean(row[metric]) for row in subset)
            if int(summary[f"{metric}_count"]) != count or not close(summary[f"{metric}_rate"], count / 400):
                failures.append(f"method_aggregate:{method}:{metric}")
        agent_subset = [row for row in agents if row["method"] == method]
        agent_count = sum(boolean(row["agent_terminal_completed"]) for row in agent_subset)
        if int(summary["agent_completion_count"]) != agent_count:
            failures.append(f"method_agent_completion:{method}")
        for stage in STAGES:
            stage_rows = [row for row in subset if row["stage"] == stage]
            stage_row = stage_lookup[(stage, method)]
            if int(stage_row["n"]) != 100:
                failures.append(f"stage_n:{stage}:{method}")
            for metric in METRICS:
                count = sum(boolean(row[metric]) for row in stage_rows)
                if int(stage_row[f"{metric}_count"]) != count or not close(stage_row[f"{metric}_rate"], count / 100):
                    failures.append(f"stage_aggregate:{stage}:{method}:{metric}")

    recomputed_ranking = sorted(METHODS, key=lambda method: (
        -float(method_lookup[method]["team_success_rate"]),
        float(method_lookup[method]["any_collision_rate"]),
        float(method_lookup[method]["timeout_rate"]),
        float(method_lookup[method]["successful_completion_time_s_mean"]),
        float(method_lookup[method]["successful_team_path_length_m_mean"]),
        float(method_lookup[method]["planning_runtime_total_ms_mean"]),
    ))
    ranking_rows = sorted(read_csv(out / "method_ranking.csv"), key=lambda row: int(row["rank"]))
    if recomputed_ranking != [row["method"] for row in ranking_rows]:
        failures.append("ranking_reproduction")
    if conclusion.get("OVERALL_METHOD_RANKING") != recomputed_ranking or conclusion.get("PROPOSED_OVERALL_RANK") != recomputed_ranking.index("gat_v1") + 1:
        failures.append("conclusion_ranking")

    by_key = {(row["stage"], row["scenario_id"], row["method"]): row for row in episodes}
    paired_rows = {(row["baseline"], row["metric"]): row for row in read_csv(out / "paired_outcomes.csv")}
    for baseline in METHODS[:-1]:
        pairs = [(by_key[(stage, f"S{stage[-1]}_{index:03d}", "gat_v1")], by_key[(stage, f"S{stage[-1]}_{index:03d}", baseline)]) for stage in STAGES for index in range(100)]
        for metric in METRICS:
            prop_only = sum(boolean(a[metric]) and not boolean(b[metric]) for a, b in pairs)
            base_only = sum(not boolean(a[metric]) and boolean(b[metric]) for a, b in pairs)
            discordant = prop_only + base_only
            expected_p = float(binomtest(prop_only, discordant, 0.5).pvalue) if discordant else 1.0
            stored = paired_rows[(baseline, metric)]
            if int(stored["proposed_only"]) != prop_only or int(stored["baseline_only"]) != base_only or not close(stored["mcnemar_exact_p"], expected_p):
                failures.append(f"paired_binary:{baseline}:{metric}")

    rng = np.random.default_rng(20260818)
    continuous_lookup = {(row["baseline"], row["metric"]): row for row in statistics["continuous_both_success"]}
    for baseline in METHODS[:-1]:
        pairs = [(by_key[(stage, f"S{stage[-1]}_{index:03d}", "gat_v1")], by_key[(stage, f"S{stage[-1]}_{index:03d}", baseline)]) for stage in STAGES for index in range(100)]
        both_success = [(a, b) for a, b in pairs if boolean(a["team_success"]) and boolean(b["team_success"])]
        for metric in CONTINUOUS:
            values = np.asarray([float(a[metric]) - float(b[metric]) for a, b in both_success if a[metric] and b[metric]], dtype=float)
            low, high = bootstrap(values, rng)
            stored = continuous_lookup[(baseline, metric)]
            if int(stored["n"]) != len(values) or not close(stored["mean_difference"], np.mean(values)) or not close(stored["bootstrap_ci95_low"], low) or not close(stored["bootstrap_ci95_high"], high):
                failures.append(f"paired_continuous:{baseline}:{metric}")

    required_conclusion = {
        "FORMAL_SCENARIOS_PER_STAGE", "FORMAL_STAGE_COUNT", "FORMAL_UNIQUE_SCENARIOS", "FORMAL_METHOD_COUNT",
        "EXPECTED_TEAM_EPISODES", "EXPECTED_AGENT_RECORDS", "ACTUAL_TEAM_EPISODES", "ACTUAL_AGENT_RECORDS",
        "FORMAL_DATA_COMPLETENESS", "SCENARIO_DUPLICATION_RATE", "GEOMETRY_DIVERSITY_VALID",
        "DIFFICULTY_MONOTONICITY_VALID", "ALL_METHODS_SHARE_IDENTICAL_SCENARIOS",
        "ENGINEERING_OPTIMIZATION_ALLOWED", "TECHNICAL_PATH_UNCHANGED", "PROPOSED_ENGINEERING_CONFIGS_TESTED",
        "CLASSIC_A_CONFIGS_TESTED", "CLASSIC_B_CONFIGS_TESTED", "PRE_ENGINEERING_PROPOSED_SUCCESS",
        "POST_ENGINEERING_PROPOSED_SUCCESS", "ENGINEERING_SUCCESS_GAIN_PP", "ENGINEERING_COLLISION_CHANGE_PP",
        "ENGINEERING_TIMEOUT_CHANGE_PP", "ENGINEERING_RUNTIME_CHANGE", "ENGINEERING_FREEZE_VALID",
        "FORMAL_RESULT_USED_FOR_TUNING", "REPRODUCIBILITY_VALID", "CLASSICAL_BASELINE_A", "CLASSICAL_BASELINE_B",
        "PROPOSED_STAGE1_SUCCESS", "PROPOSED_STAGE2_SUCCESS", "PROPOSED_STAGE3_SUCCESS", "PROPOSED_STAGE4_SUCCESS",
        "PROPOSED_OVERALL_SUCCESS", "PROPOSED_OVERALL_COLLISION", "PROPOSED_OVERALL_TIMEOUT",
        "PROPOSED_MEAN_PATH_LENGTH", "PROPOSED_MEAN_COMPLETION_TIME", "PROPOSED_MEAN_PLANNING_RUNTIME_MS",
        "PROPOSED_BEST_STAGE_COUNT", "PROPOSED_OVERALL_RANK", "PROPOSED_VS_DWA_SUCCESS_GAIN_PP",
        "PROPOSED_VS_ORCA_SUCCESS_GAIN_PP", "PROPOSED_VS_TERMINAL_SUCCESS_GAIN_PP",
        "PROPOSED_VS_PROPOSAL_SUCCESS_GAIN_PP", "PROPOSED_VS_FP_SHEP_SUCCESS_GAIN_PP",
        "FULL_METHOD_OVERALL_SUPERIORITY", "FULL_METHOD_SAFETY_SUPERIORITY", "FULL_METHOD_RUNTIME_COMPETITIVENESS",
        "GRACEFUL_DEGRADATION_GAIN", "ENGINEERING_READINESS_TARGET_REACHED", "PAPER_READY_FIGURES_GENERATED",
        "PAPER_READY_FIGURE_COUNT", "ALL_FIGURES_ENGLISH_ONLY", "ALL_PNG_DPI_GE_600",
        "ALL_PRIMARY_FIGURES_HAVE_VECTOR_PDF", "ALL_FIGURES_HAVE_SOURCE_DATA",
        "ALL_FIGURES_HAVE_REPRODUCTION_SCRIPT", "ALL_FIGURES_HAVE_CAPTION_DRAFT", "FINAL_PRINT_SIZE_READABLE",
        "GRAYSCALE_READABILITY_CHECK", "PAPER_FINAL_CANDIDATE_READY", "FINAL_METHOD", "METHOD_CHANGE_REQUIRED",
    }
    missing_fields = sorted(required_conclusion - set(conclusion))
    if missing_fields:
        failures.append(f"conclusion_missing_fields:{missing_fields}")

    figure_validation = load_json(out / "paper_ready" / "figure_validation.json")
    if figure_validation.get("status") != "PASSED" or len(figure_validation.get("figures", [])) != 10:
        failures.append("figure_validation_status")
    for item in figure_validation.get("figures", []):
        stem = item["stem"]
        paths = {
            "SOURCE_DATA_SHA256": out / "paper_ready" / "source_data" / f"{stem}.csv",
            "PLOT_SCRIPT_SHA256": out / "paper_ready" / "scripts" / f"{stem}.py",
            "PDF_SHA256": out / "paper_ready" / "pdf" / f"{stem}.pdf",
            "PNG_SHA256": out / "paper_ready" / "png_600dpi" / f"{stem}.png",
        }
        if item.get("PAPER_READY_VALID") != "YES":
            failures.append(f"figure_not_valid:{stem}")
        for field, path in paths.items():
            if not path.is_file() or item.get(field) != sha256(path):
                failures.append(f"figure_hash:{stem}:{field}")
        caption = out / "paper_ready" / "captions" / f"{stem}_caption.txt"
        text = caption.read_text(encoding="utf-8") if caption.exists() else ""
        if re.search(r"[\u3400-\u9fff]", text):
            failures.append(f"figure_chinese_text:{stem}")
        with Image.open(paths["PNG_SHA256"]) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
        if min(dpi) < 600:
            failures.append(f"figure_dpi:{stem}:{dpi}")

    report = (out / "FINAL_REPORT.md").read_text(encoding="utf-8")
    for number in range(1, 18):
        if not re.search(rf"^## {number}\. ", report, flags=re.MULTILINE):
            failures.append(f"report_section_missing:{number}")

    result = {
        "schema_version": "final_four_stage_final_reconciliation_v1",
        "status": "PASSED" if not failures else "FAILED",
        "checks": {
            "raw_integrity_passed": integrity.get("status") == "PASSED",
            "team_rows": len(episodes), "agent_rows": len(agents), "runtime_rows": len(runtimes),
            "aggregate_tables_reproduced": not any(item.startswith(("method_aggregate", "stage_aggregate", "method_agent")) for item in failures),
            "paired_statistics_reproduced": not any(item.startswith("paired_") for item in failures),
            "ranking_reproduced": "ranking_reproduction" not in failures and "conclusion_ranking" not in failures,
            "completion_missingness_valid": not any("missingness" in item for item in failures),
            "conclusion_schema_complete": not missing_fields,
            "all_ten_figures_hash_and_dpi_valid": not any(item.startswith("figure_") for item in failures),
            "report_has_17_sections": not any(item.startswith("report_section") for item in failures),
        },
        "failure_count": len(failures),
        "failure_examples": failures[:50],
    }
    (out / "final_reconciliation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
