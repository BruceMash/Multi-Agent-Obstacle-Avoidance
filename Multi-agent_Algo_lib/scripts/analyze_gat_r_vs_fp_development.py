#!/usr/bin/env python3
"""Paired diagnostic for the frozen GAT-R versus FP-SHEP Dev arms."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
SCHEMA_VERSION = "gat_r_vs_fp_development_diagnostic_v1"
BOOTSTRAP_SEED = 20260821
BOOTSTRAP_RESAMPLES = 5000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def flag(value: str) -> bool:
    return str(value).strip().lower() == "true"


def finite(value: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = int(left_only) + int(right_only)
    if discordant == 0:
        return 1.0
    smaller = min(int(left_only), int(right_only))
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.choice(values, size=(BOOTSTRAP_RESAMPLES, values.size), replace=True).mean(axis=1)
    return (float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975)))


def scope_rows(rows: Iterable[dict[str, str]], scope: str) -> list[dict[str, str]]:
    rows = list(rows)
    if scope == "overall":
        return rows
    if scope == "high_density":
        return [row for row in rows if row["stage"] in {"stage_3", "stage_4"}]
    if scope.startswith("stage_"):
        return [row for row in rows if row["stage"] == scope]
    raise ValueError(f"unknown scope: {scope}")


def aggregate(selector: str, scope: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    success = sum(flag(row["team_success"]) for row in rows)
    collision = sum(flag(row["collision"]) for row in rows)
    obstacle = sum(flag(row["obstacle_collision"]) for row in rows)
    peer = sum(flag(row["inter_agent_collision"]) for row in rows)
    timeout = sum(flag(row["timeout"]) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "selector": selector,
        "scope": scope,
        "n": len(rows),
        "success_count": success,
        "success_rate": success / len(rows),
        "collision_count": collision,
        "collision_rate": collision / len(rows),
        "obstacle_collision_count": obstacle,
        "obstacle_collision_rate": obstacle / len(rows),
        "peer_collision_count": peer,
        "peer_collision_rate": peer / len(rows),
        "timeout_count": timeout,
        "timeout_rate": timeout / len(rows),
        "agent_completion_rate": float(np.mean([float(row["agent_completion_rate"]) for row in rows])),
        "mean_upper_invocations": float(np.mean([float(row["upper_pipeline_invocation_count"]) for row in rows])),
        "mean_total_compute_ms": float(np.mean([float(row["total_online_algorithm_compute_ms"]) for row in rows])),
    }


def paired_binary(
    fp: Mapping[str, dict[str, str]],
    gat: Mapping[str, dict[str, str]],
    scope_ids: list[str],
    field: str,
) -> dict[str, Any]:
    both_true = fp_only = gat_only = both_false = 0
    for scenario_id in scope_ids:
        left = flag(fp[scenario_id][field])
        right = flag(gat[scenario_id][field])
        if left and right:
            both_true += 1
        elif left:
            fp_only += 1
        elif right:
            gat_only += 1
        else:
            both_false += 1
    return {
        "field": field,
        "n": len(scope_ids),
        "both_true": both_true,
        "fp_only_true": fp_only,
        "gat_r_only_true": gat_only,
        "both_false": both_false,
        "discordant_count": fp_only + gat_only,
        "exact_two_sided_mcnemar_p": exact_mcnemar(fp_only, gat_only),
        "gat_r_minus_fp_rate_pp": 100.0 * (gat_only - fp_only) / len(scope_ids),
    }


def continuous_tests(
    fp: Mapping[str, dict[str, str]],
    gat: Mapping[str, dict[str, str]],
) -> dict[str, Any]:
    both_success = sorted(
        scenario_id
        for scenario_id in fp
        if flag(fp[scenario_id]["team_success"]) and flag(gat[scenario_id]["team_success"])
    )
    fields = (
        "completion_time_s",
        "team_path_length_m",
        "trajectory_smoothness",
        "minimum_inter_agent_distance_m",
        "minimum_static_obstacle_clearance_m",
        "total_online_algorithm_compute_ms",
    )
    metrics: dict[str, Any] = {}
    for field in fields:
        pairs = [
            (finite(fp[scenario_id][field]), finite(gat[scenario_id][field]))
            for scenario_id in both_success
        ]
        pairs = [(left, right) for left, right in pairs if left is not None and right is not None]
        differences = np.asarray([right - left for left, right in pairs], dtype=np.float64)
        low, high = bootstrap_mean_ci(differences)
        nonzero = differences[np.abs(differences) > 0.0]
        try:
            p_value = float(wilcoxon(differences, alternative="two-sided").pvalue) if nonzero.size else 1.0
        except ValueError:
            p_value = 1.0
        metrics[field] = {
            "subset": "both_success_only",
            "n": int(differences.size),
            "gat_r_minus_fp_mean": float(np.mean(differences)) if differences.size else None,
            "gat_r_minus_fp_median": float(np.median(differences)) if differences.size else None,
            "paired_bootstrap_95_ci": [low, high] if differences.size else None,
            "wilcoxon_two_sided_p": p_value,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "both_success_count": len(both_success),
        "failed_completion_times_filled_with_zero": False,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.artifact_root.resolve() if args.artifact_root.is_absolute() else (REPO_ROOT / args.artifact_root).resolve()
    dev = root / "07_development"
    fp_rows = read_csv(dev / "FP_SHEP_RERR_DEV400/development_team_results.csv")
    gat_rows = read_csv(dev / "GAT_R_RERR_DEV400/development_team_results.csv")
    fp = {row["scenario_id"]: row for row in fp_rows}
    gat = {row["scenario_id"]: row for row in gat_rows}
    if len(fp) != 400 or len(gat) != 400 or fp.keys() != gat.keys():
        raise RuntimeError("paired Dev scenario keys are not exactly 400 and identical")
    for scenario_id in fp:
        for identity in ("seed", "stage", "family", "task_pattern"):
            if fp[scenario_id][identity] != gat[scenario_id][identity]:
                raise RuntimeError(f"paired identity mismatch: {scenario_id}/{identity}")

    scopes = ("overall", "stage_1", "stage_2", "stage_3", "stage_4", "high_density")
    comparison: list[dict[str, Any]] = []
    paired: dict[str, Any] = {}
    for scope in scopes:
        scoped_fp = scope_rows(fp_rows, scope)
        scoped_gat = scope_rows(gat_rows, scope)
        comparison.append(aggregate("fp_shep", scope, scoped_fp))
        comparison.append(aggregate("gat_r", scope, scoped_gat))
        ids = sorted(row["scenario_id"] for row in scoped_fp)
        paired[scope] = {
            field: paired_binary(fp, gat, ids, field)
            for field in ("team_success", "collision", "obstacle_collision", "inter_agent_collision", "timeout")
        }

    family_rows: list[dict[str, Any]] = []
    families = sorted({row["family"] for row in fp_rows})
    for family in families:
        ids = sorted(row["scenario_id"] for row in fp_rows if row["family"] == family)
        for selector, source in (("fp_shep", fp), ("gat_r", gat)):
            success = sum(flag(source[scenario_id]["team_success"]) for scenario_id in ids)
            collision = sum(flag(source[scenario_id]["collision"]) for scenario_id in ids)
            peer = sum(flag(source[scenario_id]["inter_agent_collision"]) for scenario_id in ids)
            family_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "family": family,
                    "selector": selector,
                    "n": len(ids),
                    "success_count": success,
                    "success_rate": success / len(ids),
                    "collision_count": collision,
                    "collision_rate": collision / len(ids),
                    "peer_collision_count": peer,
                    "peer_collision_rate": peer / len(ids),
                }
            )

    continuous = continuous_tests(fp, gat)
    outcome_transitions = Counter()
    for scenario_id in fp:
        fp_outcome = "success" if flag(fp[scenario_id]["team_success"]) else (
            "peer_collision" if flag(fp[scenario_id]["inter_agent_collision"]) else "obstacle_collision"
        )
        gat_outcome = "success" if flag(gat[scenario_id]["team_success"]) else (
            "peer_collision" if flag(gat[scenario_id]["inter_agent_collision"]) else "obstacle_collision"
        )
        outcome_transitions[f"{fp_outcome}__to__{gat_outcome}"] += 1

    overall_fp = next(row for row in comparison if row["selector"] == "fp_shep" and row["scope"] == "overall")
    overall_gat = next(row for row in comparison if row["selector"] == "gat_r" and row["scope"] == "overall")
    high_fp = next(row for row in comparison if row["selector"] == "fp_shep" and row["scope"] == "high_density")
    high_gat = next(row for row in comparison if row["selector"] == "gat_r" and row["scope"] == "high_density")
    diagnostic = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "comparison_scope": "independent development only",
        "scenario_count": 400,
        "paired_identity_match": True,
        "formal_v1_used": False,
        "fp_success": overall_fp["success_rate"],
        "gat_r_success": overall_gat["success_rate"],
        "gat_r_gain_over_fp_pp": 100.0 * (overall_gat["success_rate"] - overall_fp["success_rate"]),
        "fp_collision": overall_fp["collision_rate"],
        "gat_r_collision": overall_gat["collision_rate"],
        "fp_peer_collision": overall_fp["peer_collision_rate"],
        "gat_r_peer_collision": overall_gat["peer_collision_rate"],
        "high_density_fp_success": high_fp["success_rate"],
        "high_density_gat_r_success": high_gat["success_rate"],
        "high_density_gat_r_gain_pp": 100.0 * (high_gat["success_rate"] - high_fp["success_rate"]),
        "paired_success": paired["overall"]["team_success"],
        "paired_collision": paired["overall"]["collision"],
        "paired_peer_collision": paired["overall"]["inter_agent_collision"],
        "outcome_transitions": dict(sorted(outcome_transitions.items())),
        "both_success_count": continuous["both_success_count"],
        "marginal_success_smoothness_not_directly_comparable": True,
        "gat_r_dev_gain": "NO",
        "gat_r_offline_to_closed_loop_transfer": "FAILED",
        "holdout_authorized": False,
        "formal_v2_authorized": False,
        "next_step_under_strict_order": "TRAIN_PREDECLARED_GAT_RS_WITH_SECONDARY_SMOOTHNESS_ONLY",
    }
    write_csv(dev / "gat_r_vs_fp_dev_selector_comparison.csv", comparison)
    write_csv(dev / "gat_r_vs_fp_dev_family_diagnostic.csv", family_rows)
    write_json(
        dev / "gat_r_vs_fp_dev_paired_tests.json",
        {
            "schema_version": SCHEMA_VERSION,
            "test": "exact two-sided McNemar via paired binomial discordances",
            "scopes": paired,
        },
    )
    write_json(dev / "gat_r_vs_fp_dev_both_success_continuous_tests.json", continuous)
    write_json(dev / "GAT_R_DEV_DIAGNOSTIC.json", diagnostic)
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
