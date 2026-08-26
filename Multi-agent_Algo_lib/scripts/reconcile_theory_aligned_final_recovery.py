from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "theory_aligned_final_recovery"
    / "20260818_232900"
)
SOURCE = REPO_ROOT / "artifacts" / "final_four_stage_benchmark" / "20260818_202620"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def verify_file_hashes(
    expected: Mapping[str, str], *, prefix: Path = REPO_ROOT
) -> dict[str, Any]:
    mismatches: list[dict[str, str]] = []
    for name, digest in expected.items():
        actual = file_hash(prefix / name)
        if actual != digest:
            mismatches.append({"path": name, "expected": digest, "actual": actual})
    return {
        "checked_count": len(expected),
        "match": not mismatches,
        "mismatches": mismatches,
    }


def reconcile_freezes() -> dict[str, Any]:
    phase_a = OUTPUT / "phase_a_corrected_classics"
    freeze_a = load_json(phase_a / "corrected_classical_prefreeze.json")
    selected = load_json(SOURCE / "engineering_search" / "selected_configs.json")
    source_config = load_json(SOURCE / "config.json")
    config_payload = {
        "dwa_style": selected["dwa_style"],
        "rvo_orca_style": selected["rvo_orca_style"],
        "execution": source_config["execution"],
    }
    a_checks = {
        "implementation_file": file_hash(
            REPO_ROOT / "planning" / "final_four_stage_benchmark.py"
        )
        == freeze_a["implementation_file_sha256"],
        "scenario_manifest_file": file_hash(SOURCE / "scenario_manifest.json")
        == freeze_a["scenario_manifest_file_sha256"],
        "selected_config": stable_hash(config_payload) == freeze_a["config_sha256"],
        "information_boundary": load_json(
            phase_a / "information_boundary_tests.json"
        )["FUTURE_DYNAMIC_INFORMATION_LEAKAGE_AFTER"]
        == "NO",
    }

    phase_b = OUTPUT / "phase_b_runtime"
    freeze_b = load_json(phase_b / "runtime_prefreeze_v2.json")
    b_static = {
        "source_manifest": file_hash(SOURCE / "scenario_manifest.json")
        == freeze_b["source_manifest_sha256"],
        "selected_configs": file_hash(
            SOURCE / "engineering_search" / "selected_configs.json"
        )
        == freeze_b["selected_configs_sha256"],
        "metric_source": file_hash(SOURCE / "config.json")
        == freeze_b["metric_source_sha256"],
    }
    b_code = verify_file_hashes(freeze_b["code_hashes"])

    phase_c = OUTPUT / "phase_c_theoretical_trigger_audit"
    freeze_c = load_json(phase_c / "counterfactual_prefreeze_v2.json")
    c_checks = {
        "source_manifest": file_hash(SOURCE / "scenario_manifest.json")
        == freeze_c["source_manifest_sha256"],
        "source_results": file_hash(SOURCE / "formal_episode_results.csv")
        == freeze_c["source_results_sha256"],
        "script": file_hash(
            REPO_ROOT
            / "Multi-agent_Algo_lib"
            / "scripts"
            / "run_theory_recovery_counterfactual_audit.py"
        )
        == freeze_c["script_sha256"],
    }

    phase_d = OUTPUT / "phase_d_theory_err_development"
    freeze_d = load_json(phase_d / "development_prefreeze_v2.json")
    d_static = {
        "development_manifest": file_hash(
            OUTPUT / "development_scenario_manifest.json"
        )
        == freeze_d["development_manifest_sha256"],
        "development_config": file_hash(phase_d / "development_eval_config.json")
        == freeze_d["development_config_sha256"],
        "disjointness_audit": file_hash(
            phase_d / "development_disjointness_audit.json"
        )
        == freeze_d["disjointness_audit_sha256"],
    }
    d_code = verify_file_hashes(freeze_d["code_hashes"])
    sources = source_config["sources"]
    d_checkpoints = {
        "sac": file_hash(REPO_ROOT / sources["sac_checkpoint"])
        == freeze_d["sac_checkpoint_sha256"],
        "gat": file_hash(REPO_ROOT / sources["gat_checkpoint"])
        == freeze_d["gat_checkpoint_sha256"],
    }

    all_match = all(a_checks.values()) and all(b_static.values()) and b_code["match"]
    all_match = all_match and all(c_checks.values()) and all(d_static.values())
    all_match = all_match and d_code["match"] and all(d_checkpoints.values())
    return {
        "phase_a": a_checks,
        "phase_b": {"static": b_static, "code": b_code},
        "phase_c": c_checks,
        "phase_d": {
            "static": d_static,
            "code": d_code,
            "checkpoints": d_checkpoints,
        },
        "all_prefreeze_hashes_match": all_match,
    }


def reconcile_phase_a() -> dict[str, Any]:
    records = sorted(
        (OUTPUT / "phase_a_corrected_classics" / "records").rglob("*.json")
    )
    keys: set[tuple[str, str, int, str]] = set()
    methods: Counter[str] = Counter()
    successes: Counter[str] = Counter()
    collisions: Counter[str] = Counter()
    timeouts: Counter[str] = Counter()
    result_hash_match = True
    trajectory_hash_match = True
    agent_count = 0
    scenario_sets: defaultdict[str, set[tuple[str, int]]] = defaultdict(set)
    for path in records:
        payload = load_json(path)
        episode = payload["episode"]
        key = (
            episode["stage"],
            episode["scenario_id"],
            int(episode["seed"]),
            episode["method"],
        )
        keys.add(key)
        method = episode["method"]
        methods[method] += 1
        successes[method] += int(episode["team_success"])
        collisions[method] += int(episode["any_collision"])
        timeouts[method] += int(episode["timeout"])
        scenario_sets[method].add((episode["scenario_id"], int(episode["seed"])))
        agent_count += len(payload["agents"])
        expected_result = payload.pop("result_sha256")
        result_hash_match &= stable_hash(payload) == expected_result
        trajectory_path = OUTPUT / "phase_a_corrected_classics" / payload["trajectory_file"]
        trajectory_hash_match &= file_hash(trajectory_path) == payload["trajectory_sha256"]
    published = load_json(OUTPUT / "corrected_classical_reconciliation.json")
    reproduced = {
        method: {
            "episodes": methods[method],
            "success": successes[method],
            "collision": collisions[method],
            "timeout": timeouts[method],
        }
        for method in sorted(methods)
    }
    published_match = all(
        successes[method] == published["summaries"][method]["success_count"]
        and collisions[method] == published["summaries"][method]["collision_count"]
        and timeouts[method] == published["summaries"][method]["timeout_count"]
        for method in methods
    )
    return {
        "json_record_count": len(records),
        "unique_episode_key_count": len(keys),
        "agent_count": agent_count,
        "method_counts": dict(methods),
        "reproduced": reproduced,
        "paired_scenario_sets_identical": len(scenario_sets) == 2
        and len(next(iter(scenario_sets.values()))) == 400
        and len({frozenset(value) for value in scenario_sets.values()}) == 1,
        "result_hashes_match": result_hash_match,
        "trajectory_hashes_match": trajectory_hash_match,
        "published_aggregate_match": published_match,
        "pass": len(records) == 800
        and len(keys) == 800
        and agent_count == 2400
        and result_hash_match
        and trajectory_hash_match
        and published_match,
    }


def reconcile_phase_b() -> dict[str, Any]:
    records = sorted((OUTPUT / "phase_b_runtime" / "records_v2").rglob("*.json"))
    scenario_keys: set[tuple[str, str, int]] = set()
    episode_keys: set[tuple[str, str, int, str]] = set()
    methods: Counter[str] = Counter()
    all_behavior_match = True
    all_trajectory_match = True
    all_bundle_match = True
    actor_rows = 0
    dmp_rows = 0
    for path in records:
        payload = load_json(path)
        scenario_key = (
            payload["stage"],
            payload["scenario_id"],
            int(payload["seed"]),
        )
        scenario_keys.add(scenario_key)
        # The shared FP-SHEP preview bundle is charged once to each method that
        # actually consumes it: FP-SHEP and GAT-V1 One-Shot.
        actor_rows += 2 * len(payload["upper_actor_rows"])
        for method, record in payload["methods"].items():
            methods[method] += 1
            episode_keys.add((*scenario_key, method))
            match = record["behavior_match"]
            all_behavior_match &= bool(match["all_match"])
            all_trajectory_match &= bool(match["trajectory_exact_match"])
            all_bundle_match &= bool(match["candidate_bundle_hash_match"])
            actor_rows += len(record["execution_actor_rows"])
            dmp_rows += len(record["execution_dmp_rows"])
    published = load_json(OUTPUT / "phase_b_runtime" / "runtime_reconciliation.json")
    published_match = (
        published["matched_episode_count"] == len(episode_keys)
        and published["runtime_scenario_record_count"] == len(scenario_keys)
        and published["actor_timing_row_count"] == actor_rows
        and published["dmp_timing_row_count"] == dmp_rows
    )
    return {
        "scenario_record_count": len(records),
        "unique_scenario_key_count": len(scenario_keys),
        "unique_episode_key_count": len(episode_keys),
        "method_counts": dict(methods),
        "actor_timing_row_count": actor_rows,
        "dmp_timing_row_count": dmp_rows,
        "all_behavior_match": all_behavior_match,
        "all_trajectory_exact_match": all_trajectory_match,
        "all_candidate_bundle_match": all_bundle_match,
        "published_reconciliation_match": published_match,
        "pass": len(records) == 400
        and len(scenario_keys) == 400
        and len(episode_keys) == 1600
        and all_behavior_match
        and all_trajectory_match
        and all_bundle_match
        and published_match,
    }


def reconcile_phase_c() -> dict[str, Any]:
    records = sorted(
        (OUTPUT / "phase_c_theoretical_trigger_audit" / "records_v2").rglob(
            "*.json"
        )
    )
    keys: set[tuple[str, str, int]] = set()
    summaries: list[dict[str, Any]] = []
    for path in records:
        summary = load_json(path)["summary"]
        keys.add((summary["stage"], summary["scenario_id"], int(summary["seed"])))
        summaries.append(summary)
    collisions = [row for row in summaries if row["collision"]]
    detected_collisions = [row for row in collisions if row["detected_before_outcome"]]
    leads = np.asarray(
        [float(row["collision_warning_lead_s"]) for row in detected_collisions],
        dtype=float,
    )
    difficult_failures = [
        row
        for row in summaries
        if row["stage"] in {"stage_3", "stage_4"} and not row["team_success"]
    ]
    difficult_detected = [
        row for row in difficult_failures if row["detected_before_outcome"]
    ]
    difficult_leads = np.asarray(
        [
            (int(row["outcome_step"]) - int(row["first_theoretical_trigger_step"]))
            * 0.1
            for row in difficult_detected
        ],
        dtype=float,
    )
    successes = [row for row in summaries if row["team_success"]]
    success_replans = np.asarray(
        [float(row["theoretical_reproposal_count"]) for row in successes], dtype=float
    )
    reproduced = {
        "collision_count": len(collisions),
        "collision_detected_count": len(detected_collisions),
        "collision_lead_ge_0p5_count": int(np.sum(leads >= 0.5 - 1e-12)),
        "collision_lead_ge_1p0_count": int(np.sum(leads >= 1.0 - 1e-12)),
        "collision_median_lead_s": float(np.median(leads)),
        "stage_3_4_failure_count": len(difficult_failures),
        "stage_3_4_detected_count": len(difficult_detected),
        "stage_3_4_median_lead_s": float(np.median(difficult_leads)),
        "successful_episode_count": len(successes),
        "success_mean_theoretical_replans": float(np.mean(success_replans)),
    }
    published = load_json(
        OUTPUT
        / "phase_c_theoretical_trigger_audit"
        / "theory_observability_gate.json"
    )
    published_match = (
        reproduced["collision_count"] == published["collision_count"]
        and reproduced["collision_detected_count"]
        == published["collision_detected_count"]
        and reproduced["collision_lead_ge_0p5_count"]
        == published["collision_lead_ge_0p5_count"]
        and reproduced["collision_lead_ge_1p0_count"]
        == published["collision_lead_ge_1p0_count"]
        and close(
            reproduced["collision_median_lead_s"],
            published["collision_lead_quantiles_s"]["median"],
        )
        and reproduced["stage_3_4_failure_count"]
        == published["stage_3_4_failure_count"]
        and reproduced["stage_3_4_detected_count"]
        == published["stage_3_4_detected_before_outcome_count"]
        and close(
            reproduced["stage_3_4_median_lead_s"],
            published["stage_3_4_median_warning_lead_s"],
        )
        and reproduced["successful_episode_count"]
        == published["successful_episode_count"]
        and close(
            reproduced["success_mean_theoretical_replans"],
            published["success_mean_theoretical_replans"],
        )
    )
    return {
        "record_count": len(records),
        "unique_key_count": len(keys),
        "all_source_trajectories_read_only": all(
            row["source_trajectory_read_only"] for row in summaries
        ),
        "reproduced": reproduced,
        "published_gate_match": published_match,
        "pass": len(records) == 400
        and len(keys) == 400
        and all(row["source_trajectory_read_only"] for row in summaries)
        and published_match,
    }


def reconcile_phase_d() -> dict[str, Any]:
    records = sorted(
        (OUTPUT / "phase_d_theory_err_development" / "records_v2").rglob(
            "*.json"
        )
    )
    keys: set[tuple[str, str, int, str]] = set()
    episodes: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    event_counts: Counter[str] = Counter()
    same_goal_count = 0
    maximum_phase_delta = 0.0
    initial_hashes: defaultdict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for path in records:
        payload = load_json(path)
        episode = payload["episode"]
        method = episode["method"]
        key = (
            episode["stage"],
            episode["scenario_id"],
            int(episode["seed"]),
            method,
        )
        keys.add(key)
        episodes[method].append(episode)
        initial_hashes[(episode["scenario_id"], int(episode["seed"]))][method] = episode[
            "initial_selection_semantic_hash"
        ]
        maximum_phase_delta = max(
            maximum_phase_delta, abs(float(episode["maximum_phase_switch_delta"]))
        )
        if method == "gat_v1_err":
            for event in payload["events"]:
                if not event["counts_as_reproposal"]:
                    continue
                event_counts[event["event"]] += 1
                same_goal_count += int(not event["goal_changed"])
    reproduced: dict[str, Any] = {}
    for method, rows in episodes.items():
        reproduced[method] = {
            "episodes": len(rows),
            "success": sum(int(row["team_success"]) for row in rows),
            "collision": sum(int(row["collision"]) for row in rows),
            "timeout": sum(int(row["timeout"]) for row in rows),
            "stage_success": {
                stage: sum(
                    int(row["team_success"]) for row in rows if row["stage"] == stage
                )
                for stage in ("stage_1", "stage_2", "stage_3", "stage_4")
            },
            "mean_total_compute_ms": float(
                np.mean([float(row["total_online_algorithm_compute_ms"]) for row in rows])
            ),
        }
    err_rows = episodes["gat_v1_err"]
    replan_counts = np.asarray(
        [float(row["replanning_count"]) for row in err_rows], dtype=float
    )
    published = load_json(
        OUTPUT / "phase_d_theory_err_development" / "phase_d_conclusion.json"
    )
    initial_selection_match = all(
        len(values) == 2 and len(set(values.values())) == 1
        for values in initial_hashes.values()
    )
    raw_match = (
        reproduced["gat_v1_one_shot"]["success"] == 17
        and reproduced["gat_v1_err"]["success"] == 29
        and reproduced["gat_v1_one_shot"]["collision"] == 15
        and reproduced["gat_v1_err"]["collision"] == 5
        and reproduced["gat_v1_err"]["stage_success"]["stage_3"] == 9
        and reproduced["gat_v1_err"]["stage_success"]["stage_4"] == 6
        and close(float(np.mean(replan_counts)), published["MEAN_REPROPOSALS_PER_EPISODE"])
        and close(float(np.median(replan_counts)), published["MEDIAN_REPROPOSALS_PER_EPISODE"])
        and close(float(np.quantile(replan_counts, 0.9)), published["P90_REPROPOSALS_PER_EPISODE"])
        and close(float(np.max(replan_counts)), published["MAX_REPROPOSALS_PER_EPISODE"])
        and event_counts["NORMAL_REPROPOSAL"] == published["NORMAL_REPROPOSAL_COUNT"]
        and event_counts["EMERGENCY_REPROPOSAL"]
        == published["EMERGENCY_REPROPOSAL_COUNT"]
        and same_goal_count == published["SAME_GOAL_REPROPOSAL_COUNT"]
        and close(
            reproduced["gat_v1_err"]["mean_total_compute_ms"],
            published["THEORY_ERR_TOTAL_COMPUTE_MS"],
        )
    )
    return {
        "record_count": len(records),
        "unique_key_count": len(keys),
        "reproduced": reproduced,
        "normal_reproposal_count": event_counts["NORMAL_REPROPOSAL"],
        "emergency_reproposal_count": event_counts["EMERGENCY_REPROPOSAL"],
        "same_goal_reproposal_count": same_goal_count,
        "mean_reproposals_per_episode": float(np.mean(replan_counts)),
        "median_reproposals_per_episode": float(np.median(replan_counts)),
        "p90_reproposals_per_episode": float(np.quantile(replan_counts, 0.9)),
        "max_reproposals_per_episode": float(np.max(replan_counts)),
        "paired_initial_selection_match": initial_selection_match,
        "maximum_absolute_phase_switch_delta": maximum_phase_delta,
        "published_aggregate_match": raw_match,
        "pass": len(records) == 80
        and len(keys) == 80
        and all(len(rows) == 40 for rows in episodes.values())
        and initial_selection_match
        and maximum_phase_delta == 0.0
        and raw_match,
    }


def reconcile_conclusion() -> dict[str, Any]:
    conclusion = load_json(OUTPUT / "conclusion.json")
    expected = {
        "FUTURE_DYNAMIC_INFORMATION_LEAKAGE_AFTER": "NO",
        "CORRECTED_DWA_SUCCESS": 0.9775,
        "CORRECTED_RVO_SUCCESS": 0.985,
        "BASELINE_COMPARISON_FAIRNESS_AFTER_FIX": "YES",
        "TIMING_REPLAY_BEHAVIOR_MATCH": "YES",
        "PROPOSED_ONE_SHOT_TOTAL_COMPUTE_MS": 412.43878475,
        "DWA_TOTAL_COMPUTE_MS": 1028.33987175,
        "RVO_TOTAL_COMPUTE_MS": 3172.641249,
        "THEORY_MULTIPLE_REPROPOSALS_ALLOWED": "YES",
        "ARTIFICIAL_REPROPOSAL_CAP": "NONE",
        "THEORY_TRIGGER_OBSERVABILITY": "STRONG",
        "PHASE_D_EXECUTED": "YES",
        "ONE_SHOT_DEVELOPMENT_SUCCESS": 0.425,
        "THEORY_ERR_DEVELOPMENT_SUCCESS": 0.725,
        "SUCCESS_GAIN_PP": 30.0,
        "ONE_SHOT_STAGE3_SUCCESS": 0.0,
        "THEORY_ERR_STAGE3_SUCCESS": 0.9,
        "ONE_SHOT_STAGE4_SUCCESS": 0.0,
        "THEORY_ERR_STAGE4_SUCCESS": 0.6,
        "ONE_SHOT_COLLISION_RATE": 0.375,
        "THEORY_ERR_COLLISION_RATE": 0.125,
        "MEAN_REPROPOSALS_PER_EPISODE": 14.3,
        "MEDIAN_REPROPOSALS_PER_EPISODE": 16.5,
        "P90_REPROPOSALS_PER_EPISODE": 23.0,
        "MAX_REPROPOSALS_PER_EPISODE": 28.0,
        "NORMAL_REPROPOSAL_COUNT": 128,
        "EMERGENCY_REPROPOSAL_COUNT": 444,
        "SAME_GOAL_REPROPOSAL_COUNT": 163,
        "MEAN_INTER_REPROPOSAL_INTERVAL_S": 1.6942917547568712,
        "CHATTERING_PRESENT": "YES",
        "THEORY_ERR_TOTAL_COMPUTE_MS": 3516.227485,
        "THEORY_ERR_TOTAL_COMPUTE_VS_DWA": "HIGHER",
        "THEORY_ERR_TOTAL_COMPUTE_VS_RVO": "SIMILAR",
        "THEORY_ERR_PERFORMANCE_GAIN": "YES",
        "THEORY_ERR_EXECUTION_STABILITY": "NO",
        "CURRENT_ORIGINAL_BENCHMARK_METHOD_MISMATCH": "YES",
        "ORIGINAL_197_OVER_400_LABEL": "ONE_SHOT_PROPOSED",
        "FULL_THEORY_ALIGNED_METHOD_FORMALLY_TESTED": "NO",
        "RECOMMENDED_NEXT_STEP": "THEORY_TRIGGER_NEEDS_REVISION",
    }
    mismatches: list[dict[str, Any]] = []
    for key, expected_value in expected.items():
        actual = conclusion.get(key)
        matches = (
            close(float(actual), expected_value)
            if isinstance(expected_value, float)
            else actual == expected_value
        )
        if not matches:
            mismatches.append(
                {"field": key, "expected": expected_value, "actual": actual}
            )
    return {
        "checked_field_count": len(expected),
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def main() -> None:
    freeze = reconcile_freezes()
    phase_a = reconcile_phase_a()
    phase_b = reconcile_phase_b()
    phase_c = reconcile_phase_c()
    phase_d = reconcile_phase_d()
    conclusion = reconcile_conclusion()
    passed = (
        freeze["all_prefreeze_hashes_match"]
        and phase_a["pass"]
        and phase_b["pass"]
        and phase_c["pass"]
        and phase_d["pass"]
        and conclusion["pass"]
    )
    payload = {
        "FINAL_RECONCILIATION": "PASS" if passed else "FAIL",
        "independent_of_synthesis_script": True,
        "freeze_integrity": freeze,
        "phase_a_corrected_classics": phase_a,
        "phase_b_runtime_replay": phase_b,
        "phase_c_trigger_observability": phase_c,
        "phase_d_recurrent_err_development": phase_d,
        "conclusion_fields": conclusion,
        "invalidated_preflights_excluded": {
            "phase_b_v1": True,
            "phase_c_v1": True,
            "phase_d_v1": True,
        },
    }
    write_json(OUTPUT / "final_reconciliation.json", payload)
    write_json(OUTPUT / "independent_reconciliation.json", payload)
    print(json.dumps({"FINAL_RECONCILIATION": payload["FINAL_RECONCILIATION"]}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
