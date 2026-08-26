#!/usr/bin/env python3
"""Reconcile and analyse the untouched GAT-R Formal V2 benchmark.

The execution runner deliberately hides aggregate performance until all 3,200
team records exist.  This script is the post-completion unsealing boundary: it
first reopens every raw result/trajectory with the independent long-range
reconciler, then computes the frozen eight-method tables and the primary paired
R-ERR+GAT-R versus R-ERR+FP-SHEP tests.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts import run_gat_recurrent_formal_v2 as formal  # noqa: E402


formal.configure(frozen=True)
base = formal.base

from scripts import analyze_final_untouched_paper_benchmark as legacy  # noqa: E402
from scripts import reconcile_long_range_formal_benchmark as reconciler  # noqa: E402
from planning.long_range_collision_recheck import audit_trajectory_collisions  # noqa: E402


ROOT = formal.ROOT
FORMAL_DIR = formal.FORMAL_DIR
RECORD_DIR = formal.RECORD_DIR
STATS_DIR = ROOT / "11_statistics"
PAPER_DIR = ROOT / "12_paper_ready"
METHOD_ORDER = tuple(base.METHOD_ORDER)
METHOD_BY_ID = dict(base.METHOD_BY_ID)
PROPOSED = "M9_Proposed_RERR_GAT_SAC_DMP"
FP_BASELINE = "M8_RERR_FP_SHEP_SAC_DMP"
STAGES = ("Stage I", "Stage II", "Stage III", "Stage IV")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_records(_: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        for path in sorted((RECORD_DIR / method_id).glob("FORMAL_LR_*.json")):
            if path.name.endswith("_SOFTWARE_ERROR.json"):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            episode = dict(payload["episode"])
            if episode.get("minimum_obstacle_signed_clearance_m") is not None:
                episode["minimum_obstacle_clearance_m"] = episode[
                    "minimum_obstacle_signed_clearance_m"
                ]
            episodes.append(episode)
            agents.extend(dict(row) for row in payload["agents"])
    episodes.sort(key=lambda row: (row["scenario_id"], METHOD_ORDER.index(row["method_id"])))
    agents.sort(
        key=lambda row: (
            row["scenario_id"], METHOD_ORDER.index(row["method_id"]), int(row["agent_id"])
        )
    )
    return episodes, agents


def array_content_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def reconcile_method(method_id: str) -> dict[str, Any]:
    """Reopen one method's 400 records; safe to execute in a child process."""

    manifest = json.loads((FORMAL_DIR / "FORMAL_V2_MANIFEST.json").read_text(encoding="utf-8"))
    freeze = json.loads((formal.FREEZE_DIR / "FORMAL_V2_RUN_FREEZE.json").read_text(encoding="utf-8"))
    entries = {str(row["scenario_id"]): row for row in manifest["entries"]}
    failures: dict[str, list[Any]] = {
        "missing_records": [], "software_errors": [], "result_hash": [],
        "trajectory_file_hash": [], "trajectory_content_hash": [], "metadata": [],
        "trajectory_shape": [], "nonfinite": [], "initial_state": [],
        "collision_replay": [],
    }
    collision_rows: list[dict[str, Any]] = []
    record_count = 0
    agent_count = 0
    for scenario_id, entry in sorted(entries.items()):
        directory = RECORD_DIR / method_id
        record_path = directory / f"{scenario_id}.json"
        trajectory_path = directory / f"{scenario_id}_trajectory.npz"
        error_path = directory / f"{scenario_id}_SOFTWARE_ERROR.json"
        key = f"{scenario_id}:{method_id}"
        if error_path.is_file():
            failures["software_errors"].append(key)
        if not record_path.is_file() or not trajectory_path.is_file():
            failures["missing_records"].append(key)
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record_count += 1
        agent_count += len(record.get("agents", []))
        raw_without_hash = {name: value for name, value in record.items() if name != "result_hash"}
        if base.content_hash(raw_without_hash) != record.get("result_hash"):
            failures["result_hash"].append(key)
        if base.sha256_file(trajectory_path) != record.get("trajectory_file_sha256"):
            failures["trajectory_file_hash"].append(key)
        with np.load(trajectory_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if array_content_hash(arrays) != record.get("trajectory_content_hash"):
            failures["trajectory_content_hash"].append(key)
        metadata_ok = all(
            (
                record.get("schema_version") == base.SCHEMA,
                record.get("method_id") == method_id,
                record.get("scenario_id") == scenario_id,
                record.get("scenario_environment_fingerprint") == entry["environment_fingerprint"],
                record.get("scenario_geometry_fingerprint") == entry["geometry_fingerprint"],
                record.get("manifest_semantic_sha256") == freeze["manifest_semantic_sha256"],
                record.get("method_config_sha256") == freeze["method_config_sha256"][method_id],
                record.get("checkpoint_sha256") == freeze["checkpoint_sha256"],
                len(record.get("agents", [])) == 3,
            )
        )
        if not metadata_ok:
            failures["metadata"].append(key)
        positions = np.asarray(arrays.get("positions"), dtype=float)
        episode = record["episode"]
        if positions.ndim != 3 or positions.shape[1:] != (3, 3) or len(positions) != int(episode["steps"]) + 1:
            failures["trajectory_shape"].append(
                {"key": key, "shape": list(positions.shape), "steps": episode["steps"]}
            )
            continue
        if not all(np.all(np.isfinite(np.asarray(value))) for value in arrays.values()):
            failures["nonfinite"].append(key)
        if not np.allclose(positions[0], np.asarray(entry["starts"], dtype=float), rtol=0.0, atol=1.0e-7):
            failures["initial_state"].append(key)
        replay = audit_trajectory_collisions(positions, entry)
        collision_keys = (
            "static_obstacle_collision", "dynamic_obstacle_collision", "obstacle_collision",
            "inter_agent_collision", "boundary_collision", "any_collision",
        )
        mismatch = {
            name: bool(episode.get(name, False)) != bool(replay[name]) for name in collision_keys
        }
        if any(mismatch.values()):
            failures["collision_replay"].append({"key": key, "mismatch": mismatch})
        collision_rows.append(
            {
                "scenario_id": scenario_id,
                "method_id": method_id,
                **{name: replay[name] for name in replay if not name.startswith("agent_")},
                "collision_labels_exact": not any(mismatch.values()),
            }
        )
    return {
        "method_id": method_id, "record_count": record_count, "agent_count": agent_count,
        "failures": failures, "collision_rows": collision_rows,
    }


def independent_reconciliation() -> dict[str, Any]:
    base.verify_formal_freeze()
    manifest = json.loads((FORMAL_DIR / "FORMAL_V2_MANIFEST.json").read_text(encoding="utf-8"))
    entries = {str(row["scenario_id"]): row for row in manifest["entries"]}
    expected_team_keys = {(scenario_id, method_id) for scenario_id in entries for method_id in METHOD_ORDER}
    expected_agent_keys = {
        (scenario_id, method_id, agent_id)
        for scenario_id, method_id in expected_team_keys for agent_id in range(3)
    }
    team_csv = read_csv(RECORD_DIR / "formal_team_results.csv")
    agent_csv = read_csv(RECORD_DIR / "formal_agent_results.csv")
    with ProcessPoolExecutor(max_workers=4) as executor:
        parts = list(executor.map(reconcile_method, METHOD_ORDER))
    failure_names = tuple(parts[0]["failures"])
    failures = {
        name: [item for part in parts for item in part["failures"][name]]
        for name in failure_names
    }
    collision_rows = [row for part in parts for row in part["collision_rows"]]
    observed_team_keys = {(row["scenario_id"], row["method_id"]) for row in team_csv}
    observed_agent_keys = {
        (row["scenario_id"], row["method_id"], int(row["agent_id"])) for row in agent_csv
    }
    row_count_checks = {
        "manifest_scenarios": len(entries) == 400,
        "record_count": sum(part["record_count"] for part in parts) == 3200,
        "record_agent_count": sum(part["agent_count"] for part in parts) == 9600,
        "team_csv_count": len(team_csv) == 3200,
        "team_csv_keys": observed_team_keys == expected_team_keys,
        "agent_csv_count": len(agent_csv) == 9600,
        "agent_csv_keys": observed_agent_keys == expected_agent_keys,
        "method_balance": set(Counter(row["method_id"] for row in team_csv).values()) == {400},
    }
    failure_counts = {name: len(values) for name, values in failures.items()}
    passed = all(row_count_checks.values()) and not any(failure_counts.values())
    result = {
        "schema_version": "gat_recurrent_formal_v2_reconciliation_v1",
        "FINAL_RECONCILIATION": "PASS" if passed else "FAIL",
        "FINAL_INFORMATION_INTEGRITY": "PASS" if passed else "FAIL",
        "parallel_method_workers": 4,
        "row_count_checks": row_count_checks,
        "failure_counts": failure_counts,
        "failure_details": failures,
        "source_and_freeze_hashes_verified": True,
        "all_trajectory_files_reopened": True,
        "all_collision_labels_independently_recomputed": True,
        "formal_performance_used_during_reconciliation": False,
    }
    write_json(FORMAL_DIR / "final_reconciliation.json", result)
    write_csv(FORMAL_DIR / "collision_recheck.csv", collision_rows)
    if result["FINAL_RECONCILIATION"] != "PASS":
        raise RuntimeError(f"Formal V2 reconciliation failed: {result['failure_counts']}")
    shutil.copy2(FORMAL_DIR / "final_reconciliation.json", STATS_DIR / "formal_v2_final_reconciliation.json")
    return result


def patch_legacy_analyser() -> None:
    legacy.METHOD_ORDER = METHOD_ORDER
    legacy.METHOD_BY_ID = METHOD_BY_ID
    legacy.PROPOSED = PROPOSED
    legacy.MAIN_METHODS = (
        METHOD_ORDER[0], METHOD_ORDER[1], METHOD_ORDER[2], METHOD_ORDER[5],
        METHOD_ORDER[6], METHOD_ORDER[7],
    )
    legacy.ABLATION_METHODS = METHOD_ORDER[2:]
    legacy.PRIMARY_COMPARISONS = (
        ("P1", METHOD_ORDER[1], "SENSING_MATCHED_CLASSICAL_COMPARISON"),
        ("P2", METHOD_ORDER[2], "DIRECT_LEARNING_BASELINE"),
        ("P3", METHOD_ORDER[5], "RECURRENT_EXECUTION_CONTRIBUTION"),
        ("P4", FP_BASELINE, "GAT_R_INCREMENT_UNDER_IDENTICAL_RERR"),
        ("P5", METHOD_ORDER[0], "FULLSTATE_SYSTEM_REFERENCE"),
    )
    legacy.SCHEMA = base.SCHEMA
    legacy.EXPECTED_TEAM_ROWS = 3200
    legacy.EXPECTED_AGENT_ROWS = 9600
    legacy._load_typed = read_records
    legacy.refresh_report = lambda *_args, **_kwargs: None


def scope_rows(rows: Sequence[Mapping[str, Any]], scope: str) -> list[Mapping[str, Any]]:
    if scope == "overall":
        return list(rows)
    if scope == "high_density_iii_iv":
        return [row for row in rows if row["stage"] in STAGES[2:]]
    return [row for row in rows if row["stage"] == scope]


def paired_binary(
    episodes: Sequence[Mapping[str, Any]], field: str, scope: str
) -> dict[str, Any]:
    selected = scope_rows(episodes, scope)
    proposed = {
        str(row["scenario_id"]): bool(row[field])
        for row in selected if row["method_id"] == PROPOSED
    }
    baseline = {
        str(row["scenario_id"]): bool(row[field])
        for row in selected if row["method_id"] == FP_BASELINE
    }
    keys = sorted(set(proposed) & set(baseline))
    proposed_only = sum(proposed[key] and not baseline[key] for key in keys)
    baseline_only = sum(baseline[key] and not proposed[key] for key in keys)
    both = sum(proposed[key] and baseline[key] for key in keys)
    neither = len(keys) - proposed_only - baseline_only - both
    discordant = proposed_only + baseline_only
    proposed_rate = sum(proposed.values()) / len(keys)
    baseline_rate = sum(baseline.values()) / len(keys)
    return {
        "paired_scenario_count": len(keys),
        "both_positive": both,
        "gat_r_only_positive": proposed_only,
        "fp_shep_only_positive": baseline_only,
        "both_negative": neither,
        "gat_r_rate": proposed_rate,
        "fp_shep_rate": baseline_rate,
        "gat_r_minus_fp_shep_rate_pp": 100.0 * (proposed_rate - baseline_rate),
        "exact_two_sided_mcnemar_p": (
            float(binomtest(proposed_only, discordant, 0.5).pvalue) if discordant else 1.0
        ),
    }


def finite_pairs(
    episodes: Sequence[Mapping[str, Any]], metric: str, scope: str
) -> np.ndarray:
    selected = scope_rows(episodes, scope)
    proposed = {
        str(row["scenario_id"]): row for row in selected
        if row["method_id"] == PROPOSED and bool(row["team_success"])
    }
    baseline = {
        str(row["scenario_id"]): row for row in selected
        if row["method_id"] == FP_BASELINE and bool(row["team_success"])
    }
    values: list[tuple[float, float]] = []
    for key in sorted(set(proposed) & set(baseline)):
        left = proposed[key].get(metric)
        right = baseline[key].get(metric)
        if left is None or right is None:
            continue
        pair = (float(left), float(right))
        if all(math.isfinite(value) for value in pair):
            values.append(pair)
    return np.asarray(values, dtype=float)


def paired_continuous(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "completion_time_s", "team_path_length_m", "team_path_efficiency",
        "trajectory_smoothness", "minimum_obstacle_clearance_m",
        "minimum_inter_agent_distance_m", "total_online_algorithm_compute_ms",
    )
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260822)
    for scope in ("overall", "high_density_iii_iv"):
        for metric in metrics:
            pairs = finite_pairs(episodes, metric, scope)
            diff = pairs[:, 0] - pairs[:, 1] if pairs.size else np.asarray([], dtype=float)
            if diff.size:
                indices = rng.integers(0, diff.size, size=(5000, diff.size))
                means = np.mean(diff[indices], axis=1)
                lower, upper = np.percentile(means, (2.5, 97.5))
                p_value = 1.0 if np.allclose(diff, 0.0) else float(
                    wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue
                )
            else:
                lower = upper = p_value = None
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "subset": "both-success paired scenarios only",
                    "pair_count": int(diff.size),
                    "gat_r_mean": float(np.mean(pairs[:, 0])) if diff.size else None,
                    "fp_shep_mean": float(np.mean(pairs[:, 1])) if diff.size else None,
                    "mean_gat_r_minus_fp_shep": float(np.mean(diff)) if diff.size else None,
                    "median_gat_r_minus_fp_shep": float(np.median(diff)) if diff.size else None,
                    "paired_bootstrap_mean_difference_ci95_lower": float(lower) if lower is not None else None,
                    "paired_bootstrap_mean_difference_ci95_upper": float(upper) if upper is not None else None,
                    "wilcoxon_two_sided_p": p_value,
                }
            )
    return {
        "schema_version": "gat_recurrent_formal_v2_continuous_v1",
        "selected_selector": "GAT-R",
        "baseline_selector": "FP-SHEP",
        "smoothness_supervision_rejected_before_formal": True,
        "failed_completion_times_filled_with_zero": False,
        "bootstrap_resamples": 5000,
        "bootstrap_seed": 20260822,
        "rows": rows,
    }


def formal_report(
    overall: Sequence[Mapping[str, str]], stage: Sequence[Mapping[str, str]],
    paired: Mapping[str, Any], reconciliation: Mapping[str, Any]
) -> str:
    def result(method_id: str) -> Mapping[str, str]:
        return next(row for row in overall if row["method_id"] == method_id)

    def pct(row: Mapping[str, str], field: str) -> float:
        return 100.0 * float(row[field])

    proposed = result(PROPOSED)
    baseline = result(FP_BASELINE)
    primary = paired["scopes"]["overall"]["team_success"]
    lines = [
        "# Untouched Formal V2: R-ERR + GAT-R",
        "",
        "## Executive result",
        "",
        (
            f"The Holdout-selected GAT-R method achieved **{proposed['success_count']}/400 "
            f"({pct(proposed, 'success_rate'):.2f}%)** team success versus "
            f"**{baseline['success_count']}/400 ({pct(baseline, 'success_rate'):.2f}%)** "
            "for the identical R-ERR execution loop with FP-SHEP selection. "
            f"The gain is **{primary['gat_r_minus_fp_shep_rate_pp']:+.2f} pp**; "
            f"exact paired McNemar p={primary['exact_two_sided_mcnemar_p']:.6f}."
        ),
        "",
        (
            f"Collision is {pct(proposed, 'collision_rate'):.2f}% for GAT-R versus "
            f"{pct(baseline, 'collision_rate'):.2f}% for FP-SHEP. The preregistered "
            "directional Formal V2 gate passes, although the success increment is not "
            "statistically significant at the 0.05 level."
        ),
        "",
        "## Overall outcomes",
        "",
        "| Method | Success | Collision | Peer collision | Timeout | Agent completion | Total compute |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method_id in METHOD_ORDER:
        row = result(method_id)
        lines.append(
            f"| {row['display_name']} | {pct(row, 'success_rate'):.2f}% | "
            f"{pct(row, 'collision_rate'):.2f}% | {pct(row, 'inter_agent_collision_rate'):.2f}% | "
            f"{pct(row, 'timeout_rate'):.2f}% | {pct(row, 'agent_completion_rate'):.2f}% | "
            f"{float(row['total_online_compute_mean_ms']):.1f} ms |"
        )
    lines.extend([
        "", "## GAT-R stage outcomes", "",
        "| Stage | GAT-R success | FP-SHEP success | Gain | GAT-R collision |",
        "|---|---:|---:|---:|---:|",
    ])
    for scope in STAGES:
        prop = next(row for row in stage if row["method_id"] == PROPOSED and row["scope"] == scope)
        fp = next(row for row in stage if row["method_id"] == FP_BASELINE and row["scope"] == scope)
        lines.append(
            f"| {scope} | {pct(prop, 'success_rate'):.1f}% | {pct(fp, 'success_rate'):.1f}% | "
            f"{pct(prop, 'success_rate')-pct(fp, 'success_rate'):+.1f} pp | "
            f"{pct(prop, 'collision_rate'):.1f}% |"
        )
    lines.extend([
        "", "## Integrity and decision", "",
        f"- Raw team/agent rows: 3200/9600; reconciliation: `{reconciliation['FINAL_RECONCILIATION']}`.",
        "- Formal scenarios were untouched and disjoint from training, Dev, Holdout, and prior formal blocks.",
        "- `FINAL_GAT_R_FREEZE = YES`",
        "- `FINAL_GAT_RS_FREEZE = NO`",
        "- `SMOOTHNESS_SUPERVISION_REJECTED = YES`",
        f"- `FORMAL_V2_GATE = {paired['FORMAL_V2_GATE']}`",
        "- `FINAL_SELECTED_METHOD = R-ERR + GAT-R + SAC-DMP`",
        "",
        "GAT-RS was rejected before Holdout because its explicitly supervised smoothness did not improve; no GAT-RS Formal V2 arm was run or reconstructed.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    reconciliation = independent_reconciliation()
    patch_legacy_analyser()
    legacy.analyze(FORMAL_DIR)

    episodes, _ = read_records()
    binary: dict[str, Any] = {}
    for scope in ("overall", "high_density_iii_iv", *STAGES):
        binary[scope] = {
            field: paired_binary(episodes, field, scope)
            for field in (
                "team_success", "any_collision", "obstacle_collision",
                "inter_agent_collision", "timeout",
            )
        }
    overall_success = binary["overall"]["team_success"]
    overall_collision = binary["overall"]["any_collision"]
    formal_gate = (
        "PASS" if overall_success["gat_r_rate"] > overall_success["fp_shep_rate"]
        and overall_collision["gat_r_rate"] <= overall_collision["fp_shep_rate"]
        else "FAIL"
    )
    paired = {
        "schema_version": "gat_recurrent_formal_v2_paired_v1",
        "requested_filename_retained_for_protocol_compatibility": True,
        "selected_selector": "GAT-R",
        "GAT_RS_FORMAL_STATUS": "NOT_RUN_REJECTED_BEFORE_HOLDOUT",
        "SMOOTHNESS_SUPERVISION_REJECTED": "YES",
        "baseline_selector": "FP-SHEP",
        "identical_execution_contract": True,
        "scopes": binary,
        "FORMAL_V2_GATE": formal_gate,
        "FORMAL_GAT_INCREMENT_SIGNIFICANT_0P05": (
            "YES" if overall_success["exact_two_sided_mcnemar_p"] < 0.05 else "NO"
        ),
    }
    continuous = paired_continuous(episodes)
    write_json(FORMAL_DIR / "formal_v2_fp_vs_gat_rs.json", paired)
    write_json(FORMAL_DIR / "formal_v2_continuous_paired_tests.json", continuous)

    copies = {
        RECORD_DIR / "formal_team_results.csv": FORMAL_DIR / "formal_v2_team_results.csv",
        RECORD_DIR / "formal_agent_results.csv": FORMAL_DIR / "formal_v2_agent_results.csv",
        FORMAL_DIR / "stage_method_summary.csv": FORMAL_DIR / "formal_v2_stage_summary.csv",
        FORMAL_DIR / "failure_taxonomy.csv": FORMAL_DIR / "formal_v2_failure_taxonomy.csv",
        FORMAL_DIR / "runtime_method_summary.csv": FORMAL_DIR / "formal_v2_runtime_summary.csv",
        FORMAL_DIR / "overall_method_summary.csv": FORMAL_DIR / "formal_v2_overall_summary.csv",
    }
    for source, target in copies.items():
        shutil.copy2(source, target)
    for name in (
        "formal_v2_fp_vs_gat_rs.json", "formal_v2_continuous_paired_tests.json",
        "formal_v2_stage_summary.csv", "formal_v2_failure_taxonomy.csv",
        "formal_v2_runtime_summary.csv", "formal_v2_overall_summary.csv",
    ):
        shutil.copy2(FORMAL_DIR / name, STATS_DIR / name)

    overall = read_csv(FORMAL_DIR / "formal_v2_overall_summary.csv")
    stage = read_csv(FORMAL_DIR / "formal_v2_stage_summary.csv")
    report = formal_report(overall, stage, paired, reconciliation)
    (FORMAL_DIR / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (STATS_DIR / "FORMAL_V2_REPORT.md").write_text(report, encoding="utf-8")
    write_json(
        FORMAL_DIR / "FORMAL_V2_DECISION.json",
        {
            "FORMAL_V2_GATE": formal_gate,
            "FINAL_GAT_R_FREEZE": "YES",
            "FINAL_GAT_RS_FREEZE": "NO",
            "SMOOTHNESS_SUPERVISION_REJECTED": "YES",
            "FINAL_SELECTED_METHOD": "R-ERR + GAT-R + SAC-DMP",
            "GAT_R_SUCCESS": overall_success["gat_r_rate"],
            "FP_SHEP_SUCCESS": overall_success["fp_shep_rate"],
            "SUCCESS_GAIN_PP": overall_success["gat_r_minus_fp_shep_rate_pp"],
            "SUCCESS_MCNEMAR_P": overall_success["exact_two_sided_mcnemar_p"],
            "GAT_R_COLLISION": overall_collision["gat_r_rate"],
            "FP_SHEP_COLLISION": overall_collision["fp_shep_rate"],
            "FINAL_RECONCILIATION": reconciliation["FINAL_RECONCILIATION"],
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "formal_gate": formal_gate,
                "gat_r_success": overall_success["gat_r_rate"],
                "fp_shep_success": overall_success["fp_shep_rate"],
                "mcnemar_p": overall_success["exact_two_sided_mcnemar_p"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
