#!/usr/bin/env python3
"""Read-only runtime and 10-Hz feasibility audit for the frozen Formal V2.

This script reads the already completed Formal V2 CSV/JSON artifacts.  It does
not import or execute an environment, planner, policy, or checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
FORMAL_ROOT = SOURCE_ROOT / "10_formal_v2"
FREEZE_ROOT = SOURCE_ROOT / "09_final_freeze"
CONTROL_PERIOD_MS = 100.0
DT_S = 0.1

METHOD_ORDER = (
    "M1_DWA_FullState",
    "M2_DWA_SensingMatched",
    "M4_Direct_SAC_DMP",
    "M5_Proposal_SAC_DMP",
    "M6_FP_SHEP_SAC_DMP",
    "M7_OneShot_GAT_SAC_DMP",
    "M8_RERR_FP_SHEP_SAC_DMP",
    "M9_Proposed_RERR_GAT_SAC_DMP",
)
PRIMARY_METHODS = (
    "M1_DWA_FullState",
    "M2_DWA_SensingMatched",
    "M8_RERR_FP_SHEP_SAC_DMP",
    "M9_Proposed_RERR_GAT_SAC_DMP",
)
FP_ID = "M8_RERR_FP_SHEP_SAC_DMP"
GAT_ID = "M9_Proposed_RERR_GAT_SAC_DMP"

SOURCE_FILES = (
    "planning/online_runtime_instrumentation.py",
    "planning/final_four_stage_benchmark.py",
    "planning/sensing_matched_classical.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_formal_benchmark.py",
    "Multi-agent_Algo_lib/scripts/run_gat_recurrent_formal_v2.py",
)

NUMERIC_FIELDS = {
    "steps",
    "termination_time_s",
    "team_path_length_m",
    "team_path_length_mean_agent_m",
    "planning_decision_count",
    "perception_adapter_runtime_ms",
    "planner_core_runtime_ms",
    "proposal_generation_ms",
    "coarse_ranking_ms",
    "fp_shep_total_ms",
    "fp_shep_preview_actor_ms",
    "graph_build_ms",
    "gat_forward_ms",
    "upper_planning_total_ms",
    "execution_actor_forward_ms",
    "execution_actor_call_count",
    "execution_dmp_ms",
    "execution_dmp_call_count",
    "total_online_algorithm_compute_ms",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_team_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for source in csv.DictReader(handle):
            row: dict[str, Any] = dict(source)
            for field in NUMERIC_FIELDS:
                value = row.get(field, "")
                row[field] = float(value) if value not in (None, "") else math.nan
            row["team_success"] = str(row.get("team_success", "")).lower() == "true"
            rows.append(row)
    return rows


def rows_for(team: Sequence[Mapping[str, Any]], method_id: str) -> list[Mapping[str, Any]]:
    return [row for row in team if row["method_id"] == method_id]


def mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = finite(row[field] for row in rows)
    if not values.size:
        return math.nan
    return float(np.mean(values))


def sum_field(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = finite(row[field] for row in rows)
    return float(np.sum(values)) if values.size else 0.0


def means_by_method(team: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        method_id: {field: mean_field(rows_for(team, method_id), field) for field in NUMERIC_FIELDS}
        for method_id in METHOD_ORDER
    }


def finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def q(values: Iterable[Any], percentile: float) -> float | None:
    array = finite(values)
    return float(np.percentile(array, percentile)) if array.size else None


def stats(values: Iterable[Any], *, full: bool = True) -> dict[str, Any]:
    array = finite(values)
    if not array.size:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p05": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "p99_9": None,
            "min": None,
            "max": None,
        }
    result = {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    if not full:
        return {key: result[key] for key in ("n", "mean", "median", "p95", "p99")}
    return result


def method_name_map(team: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        method_id: str(rows_for(team, method_id)[0]["display_name"])
        for method_id in METHOD_ORDER
    }


def load_upper_invocations(
    team: Sequence[Mapping[str, Any]], names: Mapping[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for method_id in (FP_ID, GAT_ID):
        record_dir = FORMAL_ROOT / "formal_records" / method_id
        files = sorted(record_dir.glob("*.json"))
        if len(files) != 400:
            raise RuntimeError(f"{method_id}: expected 400 record JSON files, got {len(files)}")
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            episode = payload["episode"]
            by_step: dict[int, dict[str, Any]] = {}
            event_types: dict[int, set[str]] = defaultdict(set)
            for event in payload.get("events", []):
                component = event.get("runtime_components")
                if not isinstance(component, dict):
                    continue
                step = int(event["step"])
                event_types[step].add(str(event.get("event", "UNKNOWN")))
                candidate = {
                    "method_id": method_id,
                    "display_name": names[method_id],
                    "scenario_id": str(episode["scenario_id"]),
                    "stage": str(episode["stage"]),
                    "team_success": bool(episode["team_success"]),
                    "step": step,
                    **{key: float(value) for key, value in component.items()},
                }
                if step in by_step:
                    for key, value in candidate.items():
                        previous = by_step[step].get(key)
                        if isinstance(value, float):
                            if not math.isclose(float(previous), value, rel_tol=0.0, abs_tol=1e-9):
                                raise RuntimeError(
                                    f"duplicate event timing mismatch: {path.name} step={step} key={key}"
                                )
                        elif previous != value:
                            raise RuntimeError(
                                f"duplicate event identity mismatch: {path.name} step={step} key={key}"
                            )
                else:
                    by_step[step] = candidate
            for step, row in sorted(by_step.items()):
                kinds = event_types[step]
                if step == 0 or "INITIAL_SELECTION" in kinds:
                    event_class = "INITIAL_SELECTION"
                elif "EMERGENCY_REPROPOSAL" in kinds:
                    event_class = "EMERGENCY_RECONSTRUCTION"
                elif "REFERENCE_COMPLETION_REPROPOSAL" in kinds:
                    event_class = "REFERENCE_COMPLETION_RECONSTRUCTION"
                elif "NORMAL_REPROPOSAL" in kinds:
                    event_class = "NORMAL_RECONSTRUCTION"
                else:
                    event_class = "OTHER_RECONSTRUCTION"
                row["event_class"] = event_class
                row["raw_event_types"] = "|".join(sorted(kinds))
                known = (
                    row["proposal_generation_ms"]
                    + row["coarse_ranking_ms"]
                    + row["fp_shep_total_ms"]
                    + row["graph_build_ms"]
                    + row["gat_forward_ms"]
                )
                row["other_upper_overhead_ms"] = row["upper_planning_total_ms"] - known
                rows.append(row)
            expected = int(episode["planning_decision_count"])
            checks.append(
                {
                    "method_id": method_id,
                    "scenario_id": str(episode["scenario_id"]),
                    "expected": expected,
                    "recovered": len(by_step),
                    "match": expected == len(by_step),
                }
            )
    if not all(row["match"] for row in checks):
        failures = [row for row in checks if not row["match"]]
        raise RuntimeError(f"upper invocation recovery failed: {failures[:3]}")
    return rows, {
        "record_json_count": 800,
        "episode_invocation_count_match": True,
        "recovered_invocations": {
            method_id: sum(row["method_id"] == method_id for row in rows)
            for method_id in (FP_ID, GAT_ID)
        },
    }


def accounting_contract() -> list[dict[str, Any]]:
    columns = [
        "Observation acquisition",
        "Observation preprocessing",
        "Sensor projection",
        "Peer-state preparation",
        "Candidate generation",
        "Proposal scoring",
        "FP-SHEP preview",
        "Graph construction",
        "GAT inference",
        "SAC actor inference",
        "DMP transition",
        "R-ERR bookkeeping",
        "DWA candidate generation",
        "DWA rollout",
        "DWA collision checking",
        "Dynamic-obstacle state preparation",
        "Logging",
        "Python serialization",
        "Environment physics",
        "Visualization",
        "Disk I/O",
    ]

    def row(
        method: str,
        start: str,
        end: str,
        scope: str,
        values: Mapping[str, str],
        notes: str,
        evidence: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "Method": method,
            "Timer start": start,
            "Timer end": end,
            "Accounting scope": scope,
        }
        result.update({column: values.get(column, "NOT_APPLICABLE") for column in columns})
        result["Known exclusions / interpretation"] = notes
        result["Frozen source evidence"] = evidence
        return result

    common_excluded = {
        "Logging": "EXCLUDED",
        "Python serialization": "EXCLUDED",
        "Environment physics": "EXCLUDED",
        "Visualization": "EXCLUDED",
        "Disk I/O": "EXCLUDED",
    }
    return [
        row(
            "DWA-FullState",
            "entry to dwa_style_accelerations, before structured state reads",
            "after acceleration clipping and planner result construction",
            "PLANNER_LEVEL_ONLINE_COMPUTE_WITH_ENVIRONMENT_PROVIDED_PRIVILEGED_STATE",
            {
                **common_excluded,
                "Observation acquisition": "INCLUDED",
                "Observation preprocessing": "INCLUDED",
                "Peer-state preparation": "INCLUDED",
                "DWA candidate generation": "INCLUDED",
                "DWA rollout": "INCLUDED",
                "DWA collision checking": "INCLUDED",
                "Dynamic-obstacle state preparation": "INCLUDED",
            },
            "Static geometry, current dynamic-obstacle state/velocity, and peer state are supplied as structured environment objects; upstream sensing/state-estimation is not part of this full-state reference.",
            "planning/final_four_stage_benchmark.py:1002-1065 (planner timer); :1401-1598 (cycle accumulation)",
        ),
        row(
            "DWA-SensingMatched",
            "before reconstruction of local perception from latest sensor packets",
            "after local adapter, DWA core, and acceleration clipping",
            "PLANNER_LEVEL_ONLINE_COMPUTE_INCLUDING_SENSING_MATCHED_ADAPTER",
            {
                **common_excluded,
                "Observation acquisition": "EXCLUDED",
                "Observation preprocessing": "INCLUDED",
                "Sensor projection": "INCLUDED",
                "Peer-state preparation": "INCLUDED",
                "DWA candidate generation": "INCLUDED",
                "DWA rollout": "INCLUDED",
                "DWA collision checking": "INCLUDED",
                "Dynamic-obstacle state preparation": "INCLUDED",
            },
            "Ray-to-surface conversion, local obstacle-set construction, zero-order-hold surface representation, and local anonymous peer extraction are timed. Sensor raycasting/packet acquisition occurs in environment stepping and is excluded.",
            "planning/sensing_matched_classical.py:178-211 (adapter); :334-422 (inclusive timer); :561-762 (cycle accumulation)",
        ),
        row(
            "R-ERR + FP-SHEP + SAC-DMP",
            "upper: before proposal/config setup; execution: immediately around policy.predict and DMP transition",
            "upper: after plan assembly; execution: immediately after each timed call",
            "COMPONENT_SUMMED_ONLINE_ALGORITHM_COMPUTE",
            {
                **common_excluded,
                "Observation acquisition": "EXCLUDED",
                "Observation preprocessing": "UNCLEAR",
                "Sensor projection": "INCLUDED",
                "Peer-state preparation": "UNCLEAR",
                "Candidate generation": "INCLUDED",
                "Proposal scoring": "INCLUDED",
                "FP-SHEP preview": "INCLUDED",
                "SAC actor inference": "INCLUDED",
                "DMP transition": "INCLUDED",
                "R-ERR bookkeeping": "EXCLUDED",
                "Dynamic-obstacle state preparation": "UNCLEAR",
            },
            "Upper-pipeline wall time plus execution actor and DMP compute. Trigger checks, supervisor updates, execution-observation assembly, sensor refresh, and environment step are outside the summed timers.",
            "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py:344-785 (upper timer); :1115-1886 (episode/component accumulation); planning/online_runtime_instrumentation.py",
        ),
        row(
            "R-ERR + GAT-R + SAC-DMP",
            "upper: before proposal/config setup; execution: immediately around policy.predict and DMP transition",
            "upper: after plan assembly; execution: immediately after each timed call",
            "COMPONENT_SUMMED_ONLINE_ALGORITHM_COMPUTE",
            {
                **common_excluded,
                "Observation acquisition": "EXCLUDED",
                "Observation preprocessing": "UNCLEAR",
                "Sensor projection": "INCLUDED",
                "Peer-state preparation": "UNCLEAR",
                "Candidate generation": "INCLUDED",
                "Proposal scoring": "INCLUDED",
                "FP-SHEP preview": "INCLUDED",
                "Graph construction": "INCLUDED",
                "GAT inference": "INCLUDED",
                "SAC actor inference": "INCLUDED",
                "DMP transition": "INCLUDED",
                "R-ERR bookkeeping": "EXCLUDED",
                "Dynamic-obstacle state preparation": "UNCLEAR",
            },
            "Upper-pipeline wall time already contains FP-SHEP preview actor calls, graph construction, and synchronized GAT inference. Execution actor rows are mode-filtered and contain no preview calls.",
            "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py:344-785 (upper timer); :1115-1886 (mode-filtered accumulation); planning/online_runtime_instrumentation.py",
        ),
    ]


def summarize_steps(team: Sequence[Mapping[str, Any]], names: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        group = rows_for(team, method_id)
        for scope, subset in (
            ("ALL_EPISODES", group),
            ("SUCCESSFUL_EPISODES", [row for row in group if row["team_success"]]),
            ("FAILED_EPISODES", [row for row in group if not row["team_success"]]),
        ):
            summary = stats(row["steps"] for row in subset)
            rows.append(
                {
                    "method_id": method_id,
                    "display_name": names[method_id],
                    "scope": scope,
                    "episode_count": summary["n"],
                    "mean_steps": summary["mean"],
                    "std_steps_population": summary["std"],
                    "median_steps": summary["median"],
                    "p05_steps": summary["p05"],
                    "p95_steps": summary["p95"],
                    "p99_steps": summary["p99"],
                    "min_steps": summary["min"],
                    "max_steps": summary["max"],
                    "mean_simulated_duration_s": (
                        summary["mean"] * DT_S if summary["mean"] is not None else None
                    ),
                    "dt_s": DT_S,
                    "max_steps_is_ceiling_not_observation": 1500,
                }
            )
    return rows


def compute_step_rows(team: Sequence[Mapping[str, Any]], names: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        group = rows_for(team, method_id)
        ratio = np.asarray(
            [row["total_online_algorithm_compute_ms"] / row["steps"] for row in group],
            dtype=float,
        )
        utilization = ratio / CONTROL_PERIOD_MS
        summary = stats(ratio, full=False)
        utilization_summary = stats(utilization, full=False)
        rows.append(
            {
                "method_id": method_id,
                "display_name": names[method_id],
                "episode_count": int(len(group)),
                "mean_online_compute_ms_per_episode": mean_field(group, "total_online_algorithm_compute_ms"),
                "mean_executed_steps": mean_field(group, "steps"),
                "ratio_of_means_compute_ms_per_step": float(
                    mean_field(group, "total_online_algorithm_compute_ms") / mean_field(group, "steps")
                ),
                "mean_of_episode_compute_ms_per_step": summary["mean"],
                "median_episode_compute_ms_per_step": summary["median"],
                "p95_episode_average_compute_ms_per_step": summary["p95"],
                "p99_episode_average_compute_ms_per_step": summary["p99"],
                "mean_realtime_utilization": utilization_summary["mean"],
                "median_realtime_utilization": utilization_summary["median"],
                "p95_episode_average_utilization": utilization_summary["p95"],
                "p99_episode_average_utilization": utilization_summary["p99"],
                "control_period_ms": CONTROL_PERIOD_MS,
                "interpretation": "episode-average load only; not a per-step latency distribution",
            }
        )
    return rows


def component_rows(
    team: Sequence[Mapping[str, Any]], invocations: Sequence[Mapping[str, Any]], names: Mapping[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    definitions = (
        ("candidate_generation", "proposal_generation_ms", "upper_event"),
        ("proposal_scoring", "coarse_ranking_ms", "upper_event"),
        ("fp_shep_preview_inclusive", "fp_shep_total_ms", "upper_event"),
        ("graph_build", "graph_build_ms", "upper_event"),
        ("gat_forward", "gat_forward_ms", "upper_event"),
        ("other_upper_overhead", "other_upper_overhead_ms", "upper_event"),
        ("execution_sac_actor", "execution_actor_forward_ms", "execution_actor_call_count"),
        ("execution_dmp_transition", "execution_dmp_ms", "execution_dmp_call_count"),
    )
    for method_id in (FP_ID, GAT_ID):
        group = rows_for(team, method_id)
        event_group = [row for row in invocations if row["method_id"] == method_id]
        other_upper_episode = [
            row["upper_planning_total_ms"]
            - row["proposal_generation_ms"]
            - row["coarse_ranking_ms"]
            - row["fp_shep_total_ms"]
            - row["graph_build_ms"]
            - row["gat_forward_ms"]
            for row in group
        ]
        episode_values: dict[str, Sequence[float]] = {
            "proposal_generation_ms": [row["proposal_generation_ms"] for row in group],
            "coarse_ranking_ms": [row["coarse_ranking_ms"] for row in group],
            "fp_shep_total_ms": [row["fp_shep_total_ms"] for row in group],
            "graph_build_ms": [row["graph_build_ms"] for row in group],
            "gat_forward_ms": [row["gat_forward_ms"] for row in group],
            "other_upper_overhead_ms": other_upper_episode,
            "execution_actor_forward_ms": [row["execution_actor_forward_ms"] for row in group],
            "execution_dmp_ms": [row["execution_dmp_ms"] for row in group],
        }
        total_mean = mean_field(group, "total_online_algorithm_compute_ms")
        for component, field, invocation_field in definitions:
            values = episode_values[field]
            total = float(np.mean(np.asarray(values, dtype=float)))
            if invocation_field == "upper_event":
                calls = mean_field(group, "planning_decision_count")
                per_call_values = [row[field] for row in event_group]
                per_call_mean = float(np.mean(np.asarray(per_call_values, dtype=float)))
                per_call_p95 = q(per_call_values, 95)
                p95_available = "YES"
            else:
                calls = mean_field(group, invocation_field)
                per_call_mean = float(sum_field(group, field) / sum_field(group, invocation_field))
                per_call_p95 = None
                p95_available = "NO_CUMULATIVE_ONLY"
            rows.append(
                {
                    "method_id": method_id,
                    "display_name": names[method_id],
                    "component": component,
                    "accounting_role": "ADDITIVE",
                    "invocations_mean_per_episode": calls,
                    "mean_ms_per_invocation": per_call_mean,
                    "p95_ms_per_invocation": per_call_p95,
                    "p95_invocation_latency_available": p95_available,
                    "mean_total_ms_per_episode": total,
                    "percentage_of_total_compute": 100.0 * total / total_mean,
                }
            )
        preview_total = mean_field(group, "fp_shep_preview_actor_ms")
        rows.append(
            {
                "method_id": method_id,
                "display_name": names[method_id],
                "component": "fp_shep_preview_actor_nested_diagnostic",
                "accounting_role": "NESTED_DO_NOT_ADD",
                "invocations_mean_per_episode": mean_field(group, "planning_decision_count"),
                "mean_ms_per_invocation": float(np.mean([row["fp_shep_preview_actor_ms"] for row in event_group])),
                "p95_ms_per_invocation": q((row["fp_shep_preview_actor_ms"] for row in event_group), 95),
                "p95_invocation_latency_available": "YES",
                "mean_total_ms_per_episode": preview_total,
                "percentage_of_total_compute": 100.0 * preview_total / total_mean,
            }
        )
        rows.append(
            {
                "method_id": method_id,
                "display_name": names[method_id],
                "component": "rerr_trigger_and_bookkeeping",
                "accounting_role": "REQUIRED_BUT_EXCLUDED_UNTIMED",
                "invocations_mean_per_episode": None,
                "mean_ms_per_invocation": None,
                "p95_ms_per_invocation": None,
                "p95_invocation_latency_available": "NO",
                "mean_total_ms_per_episode": None,
                "percentage_of_total_compute": None,
            }
        )
    return rows


def delta_rows(team: Sequence[Mapping[str, Any]], names: Mapping[str, str]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    means = means_by_method(team)
    fp = means[FP_ID]
    gat = means[GAT_ID]
    delta = float(gat["total_online_algorithm_compute_ms"] - fp["total_online_algorithm_compute_ms"])
    graph = float(gat["graph_build_ms"] - fp["graph_build_ms"])
    gat_forward = float(gat["gat_forward_ms"] - fp["gat_forward_ms"])

    fp_shared = float(fp["upper_planning_total_ms"] - fp["graph_build_ms"] - fp["gat_forward_ms"])
    gat_shared = float(gat["upper_planning_total_ms"] - gat["graph_build_ms"] - gat["gat_forward_ms"])
    n_fp = float(fp["planning_decision_count"])
    n_gat = float(gat["planning_decision_count"])
    l_fp = fp_shared / n_fp
    l_gat = gat_shared / n_gat
    event_count_effect = (n_gat - n_fp) * 0.5 * (l_gat + l_fp)
    shared_latency_effect = (l_gat - l_fp) * 0.5 * (n_gat + n_fp)

    fp_exec = float(fp["execution_actor_forward_ms"] + fp["execution_dmp_ms"])
    gat_exec = float(gat["execution_actor_forward_ms"] + gat["execution_dmp_ms"])
    s_fp = float(fp["steps"])
    s_gat = float(gat["steps"])
    e_fp = fp_exec / s_fp
    e_gat = gat_exec / s_gat
    episode_length_effect = (s_gat - s_fp) * 0.5 * (e_gat + e_fp)
    execution_latency_effect = (e_gat - e_fp) * 0.5 * (s_gat + s_fp)

    components = (
        ("graph_construction", graph, "direct mean episode component delta"),
        ("gat_forward", gat_forward, "direct mean episode component delta"),
        ("shared_upper_event_count", event_count_effect, "symmetric count/rate decomposition"),
        ("shared_upper_per_event_latency", shared_latency_effect, "symmetric count/rate decomposition"),
        ("execution_episode_length", episode_length_effect, "symmetric step-count/rate decomposition"),
        ("execution_per_step_latency", execution_latency_effect, "symmetric step-count/rate decomposition"),
    )
    rows = [
        {
            "comparison": f"{names[GAT_ID]} minus {names[FP_ID]}",
            "component": name,
            "contribution_ms_per_episode": value,
            "contribution_percent_of_total_delta": 100.0 * value / delta,
            "decomposition_basis": basis,
        }
        for name, value, basis in components
    ]
    rows.append(
        {
            "comparison": f"{names[GAT_ID]} minus {names[FP_ID]}",
            "component": "TOTAL_OBSERVED_DELTA",
            "contribution_ms_per_episode": delta,
            "contribution_percent_of_total_delta": 100.0,
            "decomposition_basis": "mean total_online_algorithm_compute_ms",
        }
    )
    reconstructed = sum(value for _, value, _ in components)
    diagnostics = {
        "delta_ms": delta,
        "reconstructed_delta_ms": reconstructed,
        "residual_ms": delta - reconstructed,
        "explained_percent": 100.0 * reconstructed / delta,
        "graph_percent": 100.0 * graph / delta,
        "gat_forward_percent": 100.0 * gat_forward / delta,
        "event_count_percent": 100.0 * event_count_effect / delta,
        "graph_ms": graph,
        "gat_forward_ms": gat_forward,
        "event_count_ms": event_count_effect,
        "fp_event_count_mean": n_fp,
        "gat_event_count_mean": n_gat,
        "fp_shared_upper_ms_per_event": l_fp,
        "gat_shared_upper_ms_per_event": l_gat,
        "gat_extra_graph_plus_forward_ms_per_invocation": (graph + gat_forward) / n_gat,
    }
    return rows, diagnostics


def dwa_decomposition(team: Sequence[Mapping[str, Any]], names: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id in ("M1_DWA_FullState", "M2_DWA_SensingMatched"):
        group = rows_for(team, method_id)
        total = mean_field(group, "total_online_algorithm_compute_ms")
        decisions = mean_field(group, "planning_decision_count")
        if method_id == "M1_DWA_FullState":
            available = (
                ("structured_state_access_plus_complete_planner_core", total),
            )
            unavailable = (
                "candidate_sampling",
                "rollout",
                "collision_checks",
                "scoring",
                "other_planner_overhead",
            )
        else:
            adapter = mean_field(group, "perception_adapter_runtime_ms")
            core = mean_field(group, "planner_core_runtime_ms")
            available = (
                ("local_observation_preprocessing_adapter", adapter),
                ("complete_dwa_planner_core", core),
                ("timer_wrapper_residual", total - adapter - core),
            )
            unavailable = (
                "candidate_sampling",
                "rollout",
                "collision_checks",
                "scoring",
            )
        for component, value in available:
            rows.append(
                {
                    "method_id": method_id,
                    "display_name": names[method_id],
                    "component": component,
                    "availability": "EXACT_EPISODE_CUMULATIVE",
                    "mean_ms_per_episode": value,
                    "mean_ms_per_planning_cycle_pooled": value / decisions,
                    "percentage_of_timed_compute": 100.0 * value / total,
                    "true_cycle_p95_ms": None,
                }
            )
        for component in unavailable:
            rows.append(
                {
                    "method_id": method_id,
                    "display_name": names[method_id],
                    "component": component,
                    "availability": "INCLUDED_IN_CORE_NOT_SEPARATELY_TIMED",
                    "mean_ms_per_episode": None,
                    "mean_ms_per_planning_cycle_pooled": None,
                    "percentage_of_timed_compute": None,
                    "true_cycle_p95_ms": None,
                }
            )
    return rows


def distance_rows(
    team: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], names: Mapping[str, str]
) -> list[dict[str, Any]]:
    straight_by_scenario: dict[str, float] = {}
    for entry in manifest["entries"]:
        starts = np.asarray(entry["starts"], dtype=float)
        goals = np.asarray(entry["goals"], dtype=float)
        straight_by_scenario[str(entry["scenario_id"])] = float(
            np.linalg.norm(goals - starts, axis=1).sum()
        )
    rows: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        group = rows_for(team, method_id)
        total = np.asarray([row["total_online_algorithm_compute_ms"] for row in group], dtype=float)
        traveled = np.asarray([row["team_path_length_m"] for row in group], dtype=float)
        mean_agent_traveled = np.asarray(
            [row["team_path_length_mean_agent_m"] for row in group], dtype=float
        )
        nominal = np.asarray([straight_by_scenario[str(row["scenario_id"])] for row in group], dtype=float)
        actual_ratio = 100.0 * total / traveled
        mean_agent_ratio = 100.0 * total / mean_agent_traveled
        nominal_ratio = 100.0 * total / nominal
        rows.append(
            {
                "method_id": method_id,
                "display_name": names[method_id],
                "episode_count": int(len(group)),
                "distance_basis_primary": "actual team traveled path: sum of three UAV paths",
                "mean_team_traveled_distance_m": float(np.mean(traveled)),
                "mean_compute_per_team_traveled_100m_ms": float(np.mean(actual_ratio)),
                "median_compute_per_team_traveled_100m_ms": float(np.median(actual_ratio)),
                "p95_compute_per_team_traveled_100m_ms": q(actual_ratio, 95),
                "mean_compute_per_mean_uav_traveled_100m_ms": float(np.mean(mean_agent_ratio)),
                "mean_nominal_team_straight_line_distance_m": float(np.mean(nominal)),
                "mean_compute_per_nominal_team_straight_100m_ms": float(np.mean(nominal_ratio)),
                "old_compute_per_100m_semantics": "NOT_PREVIOUSLY_REPORTED_IN_FORMAL_V2",
            }
        )
    return rows


def realtime_rows(team: Sequence[Mapping[str, Any]], names: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        group = rows_for(team, method_id)
        rtf = np.asarray(
            [
                row["total_online_algorithm_compute_ms"] / (row["steps"] * DT_S * 1000.0)
                for row in group
            ],
            dtype=float,
        )
        summary = stats(rtf, full=False)
        rows.append(
            {
                "method_id": method_id,
                "display_name": names[method_id],
                "episode_count": int(len(group)),
                "mean_compute_realtime_factor": summary["mean"],
                "median_compute_realtime_factor": summary["median"],
                "p95_episode_compute_realtime_factor": summary["p95"],
                "p99_episode_compute_realtime_factor": summary["p99"],
                "definition": "online compute ms / (executed steps * 0.1 s * 1000 ms/s)",
                "claim_boundary": "cumulative compute load only; does not prove per-step deadline compliance",
            }
        )
    return rows


def event_latency_rows(
    team: Sequence[Mapping[str, Any]], invocations: Sequence[Mapping[str, Any]], names: Mapping[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, float]]]:
    event_rows: list[dict[str, Any]] = []
    deadline_rows: list[dict[str, Any]] = []
    proposed_summary: dict[str, dict[str, float]] = {}
    for method_id in (FP_ID, GAT_ID):
        group = [row for row in invocations if row["method_id"] == method_id]
        team_group = rows_for(team, method_id)
        total_steps = int(sum_field(team_group, "steps"))
        event_step_count = int(len(group))
        event_rows.append(
            {
                "method_id": method_id,
                "display_name": names[method_id],
                "event_class": "ORDINARY_EXECUTION_STEP",
                "latency_scope": "FULL_STEP_LATENCY_NOT_RETAINED",
                "count": total_steps - event_step_count,
                "mean_ms": None,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "max_ms": None,
                "count_upper_latency_over_100ms": None,
                "rate_upper_latency_over_100ms": None,
            }
        )
        for event_class, subset in (
            ("ALL_UPPER_RECONSTRUCTION_EVENTS", group),
            ("INITIAL_SELECTION", [row for row in group if row["event_class"] == "INITIAL_SELECTION"]),
            ("NORMAL_RECONSTRUCTION", [row for row in group if row["event_class"] == "NORMAL_RECONSTRUCTION"]),
            (
                "REFERENCE_COMPLETION_RECONSTRUCTION",
                [row for row in group if row["event_class"] == "REFERENCE_COMPLETION_RECONSTRUCTION"],
            ),
            (
                "EMERGENCY_RECONSTRUCTION",
                [row for row in group if row["event_class"] == "EMERGENCY_RECONSTRUCTION"],
            ),
            ("OTHER_RECONSTRUCTION", [row for row in group if row["event_class"] == "OTHER_RECONSTRUCTION"]),
        ):
            values = [row["upper_planning_total_ms"] for row in subset]
            summary = stats(values)
            misses = sum(row["upper_planning_total_ms"] > CONTROL_PERIOD_MS for row in subset)
            event_rows.append(
                {
                    "method_id": method_id,
                    "display_name": names[method_id],
                    "event_class": event_class,
                    "latency_scope": "UPPER_PIPELINE_ONLY_EXACT_WALL_TIME",
                    "count": summary["n"],
                    "mean_ms": summary["mean"],
                    "p50_ms": summary["p50"],
                    "p95_ms": summary["p95"],
                    "p99_ms": summary["p99"],
                    "max_ms": summary["max"],
                    "count_upper_latency_over_100ms": misses,
                    "rate_upper_latency_over_100ms": (
                        misses / summary["n"] if summary["n"] else None
                    ),
                }
            )
            if method_id == GAT_ID and event_class == "ALL_UPPER_RECONSTRUCTION_EVENTS":
                proposed_summary[event_class] = {
                    "p95": float(summary["p95"]),
                    "p99": float(summary["p99"]),
                    "max": float(summary["max"]),
                    "misses": float(misses),
                    "event_miss_rate": float(misses / summary["n"]),
                    "lower_bound_all_step_miss_rate": float(misses / total_steps),
                }
        definite_misses = sum(row["upper_planning_total_ms"] > CONTROL_PERIOD_MS for row in group)
        deadline_rows.append(
            {
                "method_id": method_id,
                "display_name": names[method_id],
                "evidence_scope": "LOWER_BOUND_FROM_UPPER_RECONSTRUCTION_LATENCY_ONLY",
                "executed_control_steps": total_steps,
                "upper_reconstruction_events": event_step_count,
                "definite_deadline_miss_steps_lower_bound": definite_misses,
                "definite_deadline_miss_rate_lower_bound_all_steps": definite_misses / total_steps,
                "upper_event_miss_rate": definite_misses / event_step_count,
                "true_total_deadline_miss_count": None,
                "true_total_deadline_miss_rate": None,
                "reason_total_is_unavailable": "ordinary full-step latency and per-call execution actor/DMP latency were not persisted",
            }
        )
    for method_id in ("M1_DWA_FullState", "M2_DWA_SensingMatched"):
        group = rows_for(team, method_id)
        deadline_rows.append(
            {
                "method_id": method_id,
                "display_name": names[method_id],
                "evidence_scope": "NOT_AVAILABLE_CUMULATIVE_EPISODE_TIMING_ONLY",
                "executed_control_steps": int(sum_field(group, "steps")),
                "upper_reconstruction_events": None,
                "definite_deadline_miss_steps_lower_bound": None,
                "definite_deadline_miss_rate_lower_bound_all_steps": None,
                "upper_event_miss_rate": None,
                "true_total_deadline_miss_count": None,
                "true_total_deadline_miss_rate": None,
                "reason_total_is_unavailable": "per-cycle runtime rows were returned in memory but not persisted in Formal V2 records",
            }
        )
    return event_rows, deadline_rows, proposed_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    team_path = FORMAL_ROOT / "formal_v2_team_results.csv"
    manifest_path = FORMAL_ROOT / "FORMAL_V2_MANIFEST.json"
    run_freeze_path = FREEZE_ROOT / "FORMAL_V2_RUN_FREEZE.json"
    team = load_team_csv(team_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_freeze = json.loads(run_freeze_path.read_text(encoding="utf-8"))
    names = method_name_map(team)

    if len(team) != 3200 or {row["method_id"] for row in team} != set(METHOD_ORDER):
        raise RuntimeError("Formal V2 team table identity/count mismatch")
    if not all(len(rows_for(team, method_id)) == 400 for method_id in METHOD_ORDER):
        raise RuntimeError("Formal V2 does not contain exactly 400 episodes per method")
    if not all(
        math.isclose(row["steps"], row["termination_time_s"] / DT_S, rel_tol=0.0, abs_tol=1e-8)
        for row in team
    ):
        raise RuntimeError("executed steps disagree with termination time / dt")

    invocations, invocation_check = load_upper_invocations(team, names)
    contracts = accounting_contract()
    step_summary = summarize_steps(team, names)
    per_step = compute_step_rows(team, names)
    component = component_rows(team, invocations, names)
    delta, delta_diag = delta_rows(team, names)
    dwa = dwa_decomposition(team, names)
    distance = distance_rows(team, manifest, names)
    realtime = realtime_rows(team, names)
    event_latency, deadline, event_diag = event_latency_rows(team, invocations, names)

    write_csv(output / "RUNTIME_ACCOUNTING_CONTRACT.csv", contracts)
    write_csv(output / "executed_steps_summary.csv", step_summary)
    write_csv(output / "compute_per_control_step.csv", per_step)
    write_csv(output / "runtime_component_decomposition.csv", component)
    write_csv(output / "fp_vs_gat_runtime_delta.csv", delta)
    write_csv(output / "dwa_runtime_decomposition.csv", dwa)
    write_csv(output / "distance_normalized_compute.csv", distance)
    write_csv(output / "realtime_factor_summary.csv", realtime)
    write_csv(output / "rerr_event_latency_summary.csv", event_latency)
    write_csv(output / "deadline_miss_summary.csv", deadline)

    means = means_by_method(team)
    proposed = rows_for(team, GAT_ID)
    proposed_step = next(row for row in per_step if row["method_id"] == GAT_ID)
    proposed_distance = next(row for row in distance if row["method_id"] == GAT_ID)
    proposed_rtf = next(row for row in realtime if row["method_id"] == GAT_ID)
    proposed_steps_all = next(
        row for row in step_summary if row["method_id"] == GAT_ID and row["scope"] == "ALL_EPISODES"
    )
    proposed_steps_success = next(
        row for row in step_summary if row["method_id"] == GAT_ID and row["scope"] == "SUCCESSFUL_EPISODES"
    )
    proposed_steps_failure = next(
        row for row in step_summary if row["method_id"] == GAT_ID and row["scope"] == "FAILED_EPISODES"
    )
    event_all = event_diag["ALL_UPPER_RECONSTRUCTION_EVENTS"]

    double_count = {
        "schema_version": "formal_v2_runtime_double_count_audit_v1",
        "PROPOSED_RUNTIME_DOUBLE_COUNT": "NO",
        "total_formula": "upper_planning_total_ms + execution_actor_forward_ms + execution_dmp_ms",
        "upper_timer_is_inclusive_wall_time": True,
        "fp_shep_preview_actor_nested_inside_fp_shep_total_ms": True,
        "fp_shep_preview_actor_nested_inside_upper_planning_total_ms": True,
        "fp_shep_preview_actor_added_again_to_total_online_compute": False,
        "execution_actor_rows_filtered_mode": "execution_actor",
        "preview_actor_rows_filtered_mode": "fp_shep_preview_actor",
        "nested_diagnostic_rule": "fp_shep_preview_actor_ms is diagnostic only and must never be added to fp_shep_total_ms or upper_planning_total_ms",
        "episode_formula_max_abs_residual_ms": float(
            np.max(
                np.abs(
                    [
                        row["total_online_algorithm_compute_ms"]
                        - row["upper_planning_total_ms"]
                        - row["execution_actor_forward_ms"]
                        - row["execution_dmp_ms"]
                        for row in proposed
                    ]
                )
            )
        ),
        "known_untimed_required_online_work": [
            "R-ERR trigger evaluation and bookkeeping",
            "execution temporary-checkpoint observation assembly",
            "sensor packet acquisition/raycasting",
        ],
        "source_evidence": [
            "planning/online_runtime_instrumentation.py",
            "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
        ],
    }
    json_dump(output / "PROPOSED_DOUBLE_COUNT_AUDIT.json", double_count)

    success_compute = []
    for method_id in PRIMARY_METHODS:
        group = rows_for(team, method_id)
        distance_row = next(row for row in distance if row["method_id"] == method_id)
        success_compute.append(
            {
                "method_id": method_id,
                "display_name": names[method_id],
                "information_contract": (
                    "PRIVILEGED_FULL_STATE_REFERENCE"
                    if method_id == "M1_DWA_FullState"
                    else "SENSING_MATCHED_LOCAL"
                ),
                "team_success_rate": float(np.mean([row["team_success"] for row in group])),
                "mean_online_compute_ms_per_episode": mean_field(group, "total_online_algorithm_compute_ms"),
                "mean_compute_ms_per_executed_step": next(
                    row["mean_of_episode_compute_ms_per_step"]
                    for row in per_step
                    if row["method_id"] == method_id
                ),
                "mean_compute_per_team_traveled_100m_ms": distance_row[
                    "mean_compute_per_team_traveled_100m_ms"
                ],
            }
        )
    write_csv(output / "success_compute_tradeoff.csv", success_compute)

    paper_table = []
    for row in success_compute:
        method_id = row["method_id"]
        compute = next(item for item in per_step if item["method_id"] == method_id)
        dist = next(item for item in distance if item["method_id"] == method_id)
        paper_table.append(
            {
                "Method": row["display_name"],
                "Success_percent": 100.0 * row["team_success_rate"],
                "Mean_compute_ms_per_episode": row["mean_online_compute_ms_per_episode"],
                "Mean_episode_average_compute_ms_per_control_step": compute[
                    "mean_of_episode_compute_ms_per_step"
                ],
                "P95_true_step_latency_ms": "NOT_AVAILABLE",
                "P99_true_step_latency_ms": "NOT_AVAILABLE",
                "True_deadline_miss_rate": "NOT_AVAILABLE",
                "Mean_compute_per_team_traveled_100m_ms": dist[
                    "mean_compute_per_team_traveled_100m_ms"
                ],
                "Information_contract": row["information_contract"],
            }
        )
    write_csv(output / "paper_runtime_table.csv", paper_table)

    claim_gate = {
        "schema_version": "formal_v2_runtime_claim_gate_v1",
        "control_period_ms": CONTROL_PERIOD_MS,
        "REALTIME_CLAIM_LEVEL": 1,
        "LEVEL_1_AVERAGE_ONLINE_FEASIBILITY": (
            "PASS" if proposed_step["mean_of_episode_compute_ms_per_step"] < CONTROL_PERIOD_MS else "FAIL"
        ),
        "LEVEL_2_SOFT_REALTIME_EVIDENCE": "NOT_ESTABLISHED_TRUE_STEP_LATENCY_UNAVAILABLE",
        "LEVEL_3_STRONG_SOFT_REALTIME_EVIDENCE": "NOT_ESTABLISHED_TRUE_STEP_LATENCY_UNAVAILABLE",
        "TRUE_PER_STEP_LATENCY_AVAILABLE": "NO",
        "TEN_HZ_SOFT_REALTIME_SUPPORTED": "INSUFFICIENT_DATA",
        "HARD_REALTIME_SUPPORTED": "NO",
        "WCET_AVAILABLE": "NO",
        "RTOS_SCHEDULING_PROOF_AVAILABLE": "NO",
        "RERR_UPPER_EVENT_LATENCY_AVAILABLE": "YES",
        "RERR_UPPER_EVENT_P95_MS": event_all["p95"],
        "RERR_UPPER_EVENT_P99_MS": event_all["p99"],
        "RERR_UPPER_EVENT_EXCEEDS_100MS": "YES" if event_all["misses"] else "NO",
        "recommended_claim": "The average recorded online computational load is below the 100-ms control period; true per-step deadline compliance was not logged and is not established.",
        "forbidden_claim": "hard real-time guaranteed",
    }
    json_dump(output / "runtime_claim_gate.json", claim_gate)

    claims_text = f"""# Paper runtime claims

Formal V2 reports cumulative mission-level online algorithm compute, not execution time and not a complete perception-to-control latency. The proposed method averages {means[GAT_ID]['total_online_algorithm_compute_ms']:.1f} ms over {proposed_steps_all['mean_steps']:.1f} executed 0.1-s control steps, equivalent to {proposed_step['mean_of_episode_compute_ms_per_step']:.2f} ms per step ({100.0 * proposed_step['mean_realtime_utilization']:.1f}% of the 100-ms period) when computed as the mean of per-episode ratios.

This supports only: **“The average recorded online computational load is below the 100-ms control period.”** It does not establish 10-Hz soft-real-time deadline compliance because true full-step P95/P99 latencies and the complete deadline-miss rate were not retained. The saved upper-pipeline event timings show that reconstruction is the latency peak (P95 {event_all['p95']:.2f} ms; P99 {event_all['p99']:.2f} ms), and {int(event_all['misses'])} reconstruction events alone exceeded 100 ms.

DWA-FullState is a privileged structured-state reference and its {means['M1_DWA_FullState']['total_online_algorithm_compute_ms']:.1f} ms/episode value is planner-level compute, not full perception-to-control runtime. DWA-SensingMatched includes its ray-to-surface/local-obstacle/anonymous-peer adapter and planner core, while sensor acquisition remains in the excluded environment step.

GAT-R raises success from {100.0 * np.mean([row['team_success'] for row in rows_for(team, FP_ID)]):.2f}% to {100.0 * np.mean([row['team_success'] for row in rows_for(team, GAT_ID)]):.2f}% and reduces collision, but adds {delta_diag['delta_ms']:.1f} ms of cumulative compute per episode. Runtime comparisons are suitable only with the explicit timer and information-contract qualifications in `RUNTIME_ACCOUNTING_CONTRACT.csv`.
"""
    (output / "paper_runtime_claims.md").write_text(claims_text, encoding="utf-8")

    source_hashes = {}
    for relative in SOURCE_FILES:
        observed = sha256_file(REPO_ROOT / relative)
        expected = str(run_freeze["source_sha256"][relative])
        source_hashes[relative] = {
            "expected": expected,
            "observed": observed,
            "match": observed == expected,
        }
    source_hash_pass = all(item["match"] for item in source_hashes.values())

    conclusion = {
        "schema_version": "formal_v2_runtime_realtime_audit_v1",
        "FORMAL_V2_RERUN": "NO",
        "METHOD_CHANGED": "NO",
        "DWA_PARAMETERS_CHANGED": "NO",
        "PROPOSED_PARAMETERS_CHANGED": "NO",
        "RUNTIME_ACCOUNTING_FAIRNESS": "PARTIAL",
        "DWA_FULLSTATE_RUNTIME_SCOPE": "PLANNER_LEVEL_ONLINE_COMPUTE_WITH_ENVIRONMENT_PROVIDED_PRIVILEGED_STATE",
        "DWA_SM_RUNTIME_SCOPE": "PLANNER_LEVEL_ONLINE_COMPUTE_INCLUDING_SENSING_MATCHED_ADAPTER",
        "PROPOSED_RUNTIME_SCOPE": "COMPONENT_SUMMED_ONLINE_ALGORITHM_COMPUTE_EXCLUDING_SENSOR_ACQUISITION_RERR_BOOKKEEPING_OBSERVATION_ASSEMBLY_ENVIRONMENT_STEP_AND_IO",
        "DWA_FULLSTATE_RUNTIME_UNDERCOUNTED": "NO",
        "DWA_SM_RUNTIME_UNDERCOUNTED": "NO",
        "PROPOSED_RUNTIME_DOUBLE_COUNT": "NO",
        "PROPOSED_MEAN_STEPS_PER_EPISODE": proposed_steps_all["mean_steps"],
        "PROPOSED_MEAN_SUCCESS_STEPS": proposed_steps_success["mean_steps"],
        "PROPOSED_MEAN_FAILURE_STEPS": proposed_steps_failure["mean_steps"],
        "PROPOSED_MEAN_COMPUTE_MS_PER_STEP": proposed_step["mean_of_episode_compute_ms_per_step"],
        "PROPOSED_EPISODE_COMPUTE_MS": float(means[GAT_ID]["total_online_algorithm_compute_ms"]),
        "FP_EPISODE_COMPUTE_MS": float(means[FP_ID]["total_online_algorithm_compute_ms"]),
        "DWA_FULLSTATE_EPISODE_COMPUTE_MS": float(means["M1_DWA_FullState"]["total_online_algorithm_compute_ms"]),
        "DWA_SM_EPISODE_COMPUTE_MS": float(means["M2_DWA_SensingMatched"]["total_online_algorithm_compute_ms"]),
        "PROPOSED_MEAN_REALTIME_UTILIZATION": proposed_step["mean_realtime_utilization"],
        "TRUE_PER_STEP_LATENCY_AVAILABLE": "NO",
        "PROPOSED_STEP_LATENCY_P50_MS": "NOT_AVAILABLE",
        "PROPOSED_STEP_LATENCY_P95_MS": "NOT_AVAILABLE",
        "PROPOSED_STEP_LATENCY_P99_MS": "NOT_AVAILABLE",
        "PROPOSED_STEP_LATENCY_MAX_MS": "NOT_AVAILABLE",
        "PROPOSED_DEADLINE_MISS_RATE": "NOT_AVAILABLE",
        "PROPOSED_DEFINITE_DEADLINE_MISS_RATE_LOWER_BOUND_FROM_UPPER_EVENTS": event_all[
            "lower_bound_all_step_miss_rate"
        ],
        "PROPOSED_RERR_EVENT_P95_MS": event_all["p95"],
        "PROPOSED_RERR_EVENT_P99_MS": event_all["p99"],
        "PROPOSED_COMPUTE_PER_TRAVELED_100M_MS": proposed_distance[
            "mean_compute_per_team_traveled_100m_ms"
        ],
        "PROPOSED_COMPUTE_REALTIME_FACTOR": proposed_rtf["mean_compute_realtime_factor"],
        "GAT_EXTRA_COMPUTE_PER_EPISODE_MS": delta_diag["delta_ms"],
        "GAT_EXTRA_COMPUTE_EXPLAINED_BY_GRAPH_PCT": delta_diag["graph_percent"],
        "GAT_EXTRA_COMPUTE_EXPLAINED_BY_GAT_FORWARD_PCT": delta_diag[
            "gat_forward_percent"
        ],
        "GAT_EXTRA_COMPUTE_EXPLAINED_BY_EVENT_COUNT_PCT": delta_diag[
            "event_count_percent"
        ],
        "DELTA_RUNTIME_EXPLAINED_PERCENT": delta_diag["explained_percent"],
        "REALTIME_CLAIM_LEVEL": 1,
        "TEN_HZ_SOFT_REALTIME_SUPPORTED": "INSUFFICIENT_DATA",
        "HARD_REALTIME_SUPPORTED": "NO",
        "LITERATURE_RUNTIME_CONTEXT_REQUIRES_EXTERNAL_VERIFICATION": "YES",
        "PRIVILEGED_STATE_PREPARATION": "ENVIRONMENT_PROVIDED",
        "OLD_COMPUTE_PER_100M_SEMANTICS": "NOT_PREVIOUSLY_REPORTED_IN_FORMAL_V2",
        "MICROBENCHMARK_REQUIRED": "YES_IF_A_SOFT_REALTIME_CLAIM_IS_DESIRED",
        "USER_DECISION_REQUIRED": "NO_FOR_CONSERVATIVE_PAPER_CLAIM__YES_BEFORE_ANY_NEW_TIMING_REPLAY",
        "RECOMMENDED_RUNTIME_CLAIM": claim_gate["recommended_claim"],
        "RECOMMENDED_NEXT_STEP": "WRITE_RUNTIME_AND_DISCUSSION",
    }
    json_dump(output / "conclusion.json", conclusion)

    report = f"""# Formal V2 Runtime Accounting and 10-Hz Feasibility Audit

## Executive result

The proposed method's **{means[GAT_ID]['total_online_algorithm_compute_ms']:.1f} ms/episode** is cumulative mission-level compute, accumulated over a mean **{proposed_steps_all['mean_steps']:.1f} executed control steps** ({proposed_steps_all['mean_simulated_duration_s']:.2f} simulated seconds). The mean of the 400 episode-level ratios is **{proposed_step['mean_of_episode_compute_ms_per_step']:.2f} ms/control step**, or **{100.0 * proposed_step['mean_realtime_utilization']:.1f}% average utilization** of a 100-ms period.

That is Level-1 evidence (average online feasibility), not proof of 10-Hz soft-real-time execution. Formal V2 did not persist complete wall-clock latency for every control step, so true step P50/P95/P99/max and the total deadline-miss rate are unavailable. The exact saved upper-pipeline event timings show a P95/P99 of **{event_all['p95']:.2f}/{event_all['p99']:.2f} ms**; **{int(event_all['misses'])}** reconstruction events exceeded 100 ms before adding the untallied remainder of their control steps. Reconstruction is therefore the known latency peak.

`RUNTIME_ACCOUNTING_FAIRNESS = PARTIAL`. All methods exclude physics, logging, serialization, rendering, and disk I/O, but their timer boundaries are not complete perception-to-control boundaries. DWA-FullState consumes environment-provided privileged structured state; DWA-SensingMatched times its local observation adapter and planner core; recurrent FP/GAT time the upper pipeline, execution actor, and DMP but omit R-ERR bookkeeping and execution-observation assembly.

## Exact answers

1. **Why 16.56 s?** Repeated event-triggered reconstruction causes a mean {means[GAT_ID]['planning_decision_count']:.2f} upper-pipeline invocations per episode. The cumulative total is upper planning ({means[GAT_ID]['upper_planning_total_ms']:.1f} ms) + execution actor ({means[GAT_ID]['execution_actor_forward_ms']:.1f} ms) + DMP ({means[GAT_ID]['execution_dmp_ms']:.1f} ms). It is not physical execution time.
2. **Is it compatible with dt=0.1 s?** The average recorded load is compatible ({proposed_step['mean_of_episode_compute_ms_per_step']:.2f} < 100 ms), but synchronous reconstruction spikes often exceed the budget and complete per-step deadline evidence is missing. Claim only average computational feasibility.
3. **Are boundaries fair?** Partially. The comparison is usable in a paper only when described as qualified planner/online-algorithm compute and paired with the information-contract/timer table; it is not a complete-system latency comparison.

## Steps and average load

| Method | Mean steps | Mean compute/episode (ms) | Mean of episode ratios (ms/step) | Mean utilization |
|---|---:|---:|---:|---:|
"""
    for method_id in PRIMARY_METHODS:
        step_row = next(row for row in step_summary if row["method_id"] == method_id and row["scope"] == "ALL_EPISODES")
        compute_row = next(row for row in per_step if row["method_id"] == method_id)
        report += (
            f"| {names[method_id]} | {step_row['mean_steps']:.2f} | "
            f"{compute_row['mean_online_compute_ms_per_episode']:.1f} | "
            f"{compute_row['mean_of_episode_compute_ms_per_step']:.2f} | "
            f"{100.0 * compute_row['mean_realtime_utilization']:.1f}% |\n"
        )
    report += f"""

## Double-count and component result

`PROPOSED_RUNTIME_DOUBLE_COUNT = NO`. `upper_planning_total_ms` is an inclusive wall timer. FP-SHEP preview actor time is nested within both `fp_shep_total_ms` and the upper total, but is retained only as a diagnostic; it is not added again. The only actor time added outside the upper total is mode-filtered real execution actor inference.

The GAT-R method adds **{delta_diag['delta_ms']:.1f} ms/episode** versus recurrent FP. The additive decomposition closes at **{delta_diag['explained_percent']:.6f}%**. Graph and GAT-forward totals contribute {delta_diag['graph_percent']:.1f}% and {delta_diag['gat_forward_percent']:.1f}% of the net delta, while the event-count effect is {delta_diag['event_count_percent']:.1f}% (negative means GAT-R had fewer reconstruction events and partially offset its module cost). See `fp_vs_gat_runtime_delta.csv` for all positive and negative contributions.

## DWA interpretation

DWA-FullState's {means['M1_DWA_FullState']['total_online_algorithm_compute_ms']:.1f} ms/episode includes state reads, candidate sampling, analytic/geometric rollout, dynamic/peer prediction, collision checks, scoring, and selection inside its planner timer. Its structured static/dynamic/peer inputs are environment-provided, so this is privileged planner-level compute, not full perception-to-control runtime.

DWA-SensingMatched includes ray-to-point projection, local obstacle-set creation, zero-order-hold visible-surface representation, anonymous peer extraction, and the DWA core. These sum to {means['M2_DWA_SensingMatched']['total_online_algorithm_compute_ms']:.1f} ms/episode. Candidate sampling, rollout, collision checks, and scoring were not separately timed within the core. DWA is faster here because of lower-complexity vectorized analytic rollouts and the absence of neural FP-SHEP previews and graph inference—not because it performs no planning.

## Claim gate and next step

- `REALTIME_CLAIM_LEVEL = 1`
- `TEN_HZ_SOFT_REALTIME_SUPPORTED = INSUFFICIENT_DATA`
- `HARD_REALTIME_SUPPORTED = NO`
- No Formal V2 episode was rerun and no method or parameter was changed.

The paper can be written now with the conservative Level-1 wording in `paper_runtime_claims.md`. A minimal read-only timing replay would be required only if a stronger soft-real-time claim is desired; it must not be run without user approval.
"""
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    expected_means = {
        "M1_DWA_FullState": 1264.3249555,
        "M2_DWA_SensingMatched": 5756.2017595,
        FP_ID: 12234.943993,
        GAT_ID: 16561.6856575,
    }
    reconciliation = {
        "schema_version": "formal_v2_runtime_realtime_reconciliation_v1",
        "status": "PASS",
        "source_artifact_root": str(SOURCE_ROOT.relative_to(REPO_ROOT).as_posix()),
        "formal_team_row_count": int(len(team)),
        "method_episode_counts": {
            method_id: len(rows_for(team, method_id)) for method_id in METHOD_ORDER
        },
        "formal_agent_row_count": sum(
            1
            for _ in csv.DictReader(
                (FORMAL_ROOT / "formal_v2_agent_results.csv").open(
                    "r", newline="", encoding="utf-8-sig"
                )
            )
        ),
        "input_sha256": {
            str(team_path.relative_to(REPO_ROOT).as_posix()): sha256_file(team_path),
            str(manifest_path.relative_to(REPO_ROOT).as_posix()): sha256_file(manifest_path),
            str(run_freeze_path.relative_to(REPO_ROOT).as_posix()): sha256_file(run_freeze_path),
        },
        "frozen_source_hashes": source_hashes,
        "frozen_source_hash_gate": "PASS" if source_hash_pass else "FAIL",
        "executed_steps_vs_termination_time_gate": "PASS",
        "upper_invocation_recovery": invocation_check,
        "reported_mean_compute_reproduction": {
            method_id: {
                "expected_ms": expected,
                "recomputed_ms": float(means[method_id]["total_online_algorithm_compute_ms"]),
                "match": math.isclose(
                    expected,
                    float(means[method_id]["total_online_algorithm_compute_ms"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
            }
            for method_id, expected in expected_means.items()
        },
        "proposed_total_formula_max_abs_residual_ms": double_count[
            "episode_formula_max_abs_residual_ms"
        ],
        "delta_decomposition": delta_diag,
        "true_per_step_latency_file_expected": False,
        "true_per_step_latency_file_present": (output / "per_step_latency_summary.csv").exists(),
        "required_outputs_present": {},
    }
    required = (
        "RUNTIME_ACCOUNTING_CONTRACT.csv",
        "PROPOSED_DOUBLE_COUNT_AUDIT.json",
        "executed_steps_summary.csv",
        "compute_per_control_step.csv",
        "runtime_component_decomposition.csv",
        "fp_vs_gat_runtime_delta.csv",
        "dwa_runtime_decomposition.csv",
        "distance_normalized_compute.csv",
        "realtime_factor_summary.csv",
        "rerr_event_latency_summary.csv",
        "deadline_miss_summary.csv",
        "runtime_claim_gate.json",
        "success_compute_tradeoff.csv",
        "paper_runtime_table.csv",
        "paper_runtime_claims.md",
        "conclusion.json",
        "FINAL_REPORT.md",
    )
    reconciliation["required_outputs_present"] = {
        name: (output / name).is_file() for name in required
    }
    gates = [
        source_hash_pass,
        all(item["match"] for item in reconciliation["reported_mean_compute_reproduction"].values()),
        abs(float(delta_diag["residual_ms"])) < 1e-8,
        double_count["episode_formula_max_abs_residual_ms"] < 1e-8,
        all(reconciliation["required_outputs_present"].values()),
        not reconciliation["true_per_step_latency_file_present"],
    ]
    reconciliation["status"] = "PASS" if all(gates) else "FAIL"
    json_dump(output / "final_reconciliation.json", reconciliation)
    if reconciliation["status"] != "PASS":
        raise RuntimeError("final reconciliation failed")


if __name__ == "__main__":
    main()
