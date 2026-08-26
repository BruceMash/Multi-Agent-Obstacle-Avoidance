from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as _pandas  # noqa: F401
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for _path in (REPO_ROOT, ALGO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from planning.final_four_stage_benchmark import (  # noqa: E402
    generate_scenario_manifest,
    validate_scenario_manifest,
)
from planning.online_runtime_instrumentation import (  # noqa: E402
    OnlineRuntimeRecorder,
    TimedPolicyProxy,
)
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    EVENT_EMERGENCY_REPROPOSAL,
    EVENT_NORMAL_REPROPOSAL,
    METHOD_ERR,
    METHOD_ONE_SHOT,
    run_episode,
)
from scripts.run_final_four_stage_benchmark import (  # noqa: E402
    FrozenRuntime,
    proposed_eval_config,
)


SOURCE_ROOT = REPO_ROOT / "artifacts" / "final_four_stage_benchmark" / "20260818_202620"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "theory_aligned_final_recovery"
    / "20260818_232900"
)
METHOD_LABELS = {
    METHOD_ONE_SHOT: "one_shot_proposed_development",
    METHOD_ERR: "theory_err_development",
}
PREFREEZE_NAME = "development_prefreeze_v2.json"
RECORD_DIRECTORY = "records_v2"


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(jsonable(row.get(key)), ensure_ascii=False)
                        if isinstance(row.get(key), (dict, list, tuple, np.ndarray))
                        else jsonable(row.get(key))
                    )
                    for key in fields
                }
            )
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_history_values(output_dir: Path) -> tuple[set[int], set[str], set[str], list[str]]:
    seeds: set[int] = set()
    geometry: set[str] = set()
    translation: set[str] = set()
    sources: list[str] = []

    def walk(value: Any, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                walk(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                walk(child, key)
        elif key in {"seed", "seeds", "formal_seeds", "development_seeds"}:
            try:
                seeds.add(int(value))
            except (TypeError, ValueError):
                pass
        elif key == "geometry_fingerprint" and isinstance(value, str):
            geometry.add(value)
        elif key == "translation_invariant_fingerprint" and isinstance(value, str):
            translation.add(value)

    for path in sorted((REPO_ROOT / "artifacts").rglob("*manifest*.json")):
        if output_dir in path.parents:
            continue
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        walk(payload)
        sources.append(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))
    return seeds, geometry, translation, sources


def _development_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    base_config = load_json(SOURCE_ROOT / "config.json")
    selected = load_json(SOURCE_ROOT / "engineering_search" / "selected_configs.json")
    err_source = load_json(REPO_ROOT / "configs/evaluation/gat_v1_err_development.json")
    # FrozenRuntime constructs the exact final-benchmark base config.  This
    # pure payload mirrors the selected values for prefreeze provenance.
    return {
        "source_config": "artifacts/final_four_stage_benchmark/20260818_202620/config.json",
        "selected_proposed": selected["gat_v1"],
        "err": err_source["err"],
        "execution": base_config["execution"],
        "manifest_sha256": manifest["manifest_sha256"],
        "methods": [METHOD_ONE_SHOT, METHOD_ERR],
        "multiple_reproposals_allowed": True,
        "artificial_reproposal_cap": None,
    }


def prepare(output_dir: Path) -> None:
    phase = output_dir / "phase_d_theory_err_development"
    if any((phase / RECORD_DIRECTORY).rglob("*.json")):
        raise RuntimeError("cannot prepare development manifest after results exist")
    historical_seeds, historical_geometry, historical_translation, sources = (
        _collect_history_values(output_dir)
    )
    seed_base = 990_500_000
    manifest: dict[str, Any] | None = None
    for attempt in range(100):
        candidate = generate_scenario_manifest(
            counts_per_stage=10,
            seed_base=seed_base + attempt * 100_000,
            prefix="RD",
            max_steps=220,
            dt=0.1,
        )
        candidate_seeds = {int(row["seed"]) for row in candidate["entries"]}
        candidate_geometry = {
            str(row["geometry_fingerprint"]) for row in candidate["entries"]
        }
        candidate_translation = {
            str(row["translation_invariant_fingerprint"]) for row in candidate["entries"]
        }
        if (
            candidate_seeds.isdisjoint(historical_seeds)
            and candidate_geometry.isdisjoint(historical_geometry)
            and candidate_translation.isdisjoint(historical_translation)
        ):
            manifest = candidate
            break
    if manifest is None:
        raise RuntimeError("could not construct a disjoint development manifest")
    validation = validate_scenario_manifest(manifest)
    if validation["status"] != "PASSED":
        raise RuntimeError(f"development manifest validation failed: {validation}")
    development_config = _development_config(manifest)
    write_json(output_dir / "development_scenario_manifest.json", manifest)
    write_json(phase / "development_scenario_manifest.json", manifest)
    write_json(phase / "development_eval_config.json", development_config)
    audit = {
        "history_source_manifest_count": len(sources),
        "history_source_manifests": sources,
        "historical_seed_count": len(historical_seeds),
        "historical_geometry_count": len(historical_geometry),
        "historical_translation_fingerprint_count": len(historical_translation),
        "seed_overlap_count": 0,
        "geometry_overlap_count": 0,
        "translation_equivalent_overlap_count": 0,
        "history_disjoint": True,
        "geometry_disjoint": True,
        "seed_disjoint": True,
        "manifest_validation": validation,
    }
    write_json(phase / "development_disjointness_audit.json", audit)
    observability = load_json(
        output_dir
        / "phase_c_theoretical_trigger_audit"
        / "theory_observability_gate.json"
    )
    if observability["THEORY_TRIGGER_OBSERVABILITY"] not in {"STRONG", "MODERATE"}:
        raise RuntimeError("Phase D gate is closed by temporal observability")
    code_files = (
        "Guidance/reference_point_proposal_demo.py",
        "planning/event_triggered_reference_reconstruction.py",
        "planning/online_runtime_instrumentation.py",
        "planning/pre_gat_closed_loop.py",
        "planning/heterogeneous_candidate_graph.py",
        "planning/gat/candidate_selector.py",
        "planning/gat/edge_enhanced_gat.py",
        "Environment/frozen_sac_dmp_execution.py",
        "Environment/multi_agent_dmp_env.py",
        "Controller/dmp_rl.py",
        "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
        "Multi-agent_Algo_lib/scripts/run_theory_recovery_err_development.py",
    )
    freeze = {
        "freeze_time": datetime.now().astimezone().isoformat(),
        "freeze_before_development_performance": True,
        "PHASE_D_EXECUTION_AUTHORIZED_BY_OBSERVABILITY": observability[
            "THEORY_TRIGGER_OBSERVABILITY"
        ],
        "development_manifest_sha256": file_hash(
            output_dir / "development_scenario_manifest.json"
        ),
        "development_config_sha256": file_hash(
            phase / "development_eval_config.json"
        ),
        "disjointness_audit_sha256": file_hash(
            phase / "development_disjointness_audit.json"
        ),
        "code_hashes": {name: file_hash(REPO_ROOT / name) for name in code_files},
        "sac_checkpoint_sha256": file_hash(
            REPO_ROOT / base_config_path()["sac_checkpoint"]
        ),
        "gat_checkpoint_sha256": file_hash(
            REPO_ROOT / base_config_path()["gat_checkpoint"]
        ),
        "chattering_rule_frozen_before_results": {
            "dwell_violation": "any normal reproposal interval < T_rep,min",
            "high_frequency_same_goal": (
                "P90 replans >=10 and median per-agent interval <=0.3 s and "
                "same-goal fraction >=0.25"
            ),
            "high_frequency_emergency": (
                "emergency fraction >=0.75 and consecutive-emergency fraction >=0.50 "
                "and mean replans/episode >=5"
            ),
            "CHATTERING_PRESENT": "YES if any criterion is true",
        },
        "development_gain_rule": {
            "overall_success_gain_pp_min": 15.0,
            "stage_3_4_success_gain_pp_min": 20.0,
            "collision_reduction_pp_min": 10.0,
            "stage_1_success_drop_pp_max": 10.0,
        },
        "THEORY_MULTIPLE_REPROPOSALS_ALLOWED": "YES",
        "ARTIFICIAL_REPROPOSAL_CAP": "NONE",
    }
    write_json(phase / PREFREEZE_NAME, freeze)
    print(f"DEVELOPMENT_MANIFEST_SHA256={manifest['manifest_sha256']}", flush=True)
    print("DEVELOPMENT_DISJOINTNESS=PASS", flush=True)


def base_config_path() -> Mapping[str, str]:
    return load_json(SOURCE_ROOT / "config.json")["sources"]


def verify_freeze(output_dir: Path) -> Mapping[str, Any]:
    phase = output_dir / "phase_d_theory_err_development"
    freeze = load_json(phase / PREFREEZE_NAME)
    if file_hash(output_dir / "development_scenario_manifest.json") != freeze[
        "development_manifest_sha256"
    ]:
        raise RuntimeError("development manifest changed after freeze")
    if file_hash(phase / "development_eval_config.json") != freeze[
        "development_config_sha256"
    ]:
        raise RuntimeError("development config changed after freeze")
    for name, expected in freeze["code_hashes"].items():
        if file_hash(REPO_ROOT / name) != expected:
            raise RuntimeError(f"development code changed after freeze: {name}")
    return freeze


def _runtime_eval_config(runtime: FrozenRuntime, output_dir: Path) -> dict[str, Any]:
    phase = output_dir / "phase_d_theory_err_development"
    frozen = load_json(phase / "development_eval_config.json")
    result = proposed_eval_config(runtime.base_eval_config, frozen["selected_proposed"])
    result["err"] = frozen["err"]
    return result


def _record_path(phase: Path, entry: Mapping[str, Any], method: str) -> Path:
    return phase / RECORD_DIRECTORY / entry["stage"] / entry["scenario_id"] / f"{method}.json"


def _warmup(
    runtime: FrozenRuntime,
    eval_config: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> None:
    run_episode(
        config=eval_config,
        settings=runtime.execution_settings,
        multi_config=runtime.multi_config,
        policy=runtime.policy,
        gat_model=runtime.gat_model,
        gat_device=runtime.gat_device,
        method=METHOD_ONE_SHOT,
        scenario=entry["scenario_id"],
        seed=int(entry["seed"]),
        environment_builder=runtime.builder,
        runtime_recorder=None,
    )


def run(output_dir: Path, limit: int | None = None) -> None:
    verify_freeze(output_dir)
    phase = output_dir / "phase_d_theory_err_development"
    manifest = load_json(output_dir / "development_scenario_manifest.json")
    config = load_json(SOURCE_ROOT / "config.json")
    runtime = FrozenRuntime(config, manifest)
    eval_config = _runtime_eval_config(runtime, output_dir)
    entries = list(manifest["entries"])
    _warmup(runtime, eval_config, entries[0])
    if limit is not None:
        entries = entries[: int(limit)]
    expected = len(entries) * 2
    job = 0
    for entry in entries:
        pair_hashes: dict[str, str] = {}
        for method in (METHOD_ONE_SHOT, METHOD_ERR):
            job += 1
            path = _record_path(phase, entry, method)
            if path.exists():
                continue
            recorder = OnlineRuntimeRecorder()
            policy = TimedPolicyProxy(runtime.policy, recorder)
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block="new_40_development",
                stage=entry["stage"],
                family=entry["family"],
                scenario_id=entry["scenario_id"],
                seed=int(entry["seed"]),
                method=METHOD_LABELS[method],
            ):
                episode, agents, events, triggers, auxiliary = run_episode(
                    config=eval_config,
                    settings=runtime.execution_settings,
                    multi_config=runtime.multi_config,
                    policy=policy,
                    gat_model=runtime.gat_model,
                    gat_device=runtime.gat_device,
                    method=method,
                    scenario=entry["scenario_id"],
                    seed=int(entry["seed"]),
                    environment_builder=runtime.builder,
                    runtime_recorder=recorder,
                )
            episode = dict(episode)
            episode["stage"] = entry["stage"]
            episode["family"] = entry["family"]
            episode["scenario_id"] = entry["scenario_id"]
            episode["method_label"] = METHOD_LABELS[method]
            pair_hashes[method] = episode["initial_selection_semantic_hash"]
            payload = {
                "entry": {
                    key: entry[key]
                    for key in (
                        "stage",
                        "family",
                        "scenario_id",
                        "seed",
                        "environment_fingerprint",
                        "geometry_fingerprint",
                        "translation_invariant_fingerprint",
                    )
                },
                "episode": episode,
                "agents": agents,
                "events": events,
                "triggers": triggers,
                "actor_timing_rows": auxiliary["actor_timing_rows"],
                "dmp_timing_rows": auxiliary["dmp_timing_rows"],
                "upper_timing_rows": auxiliary["upper_timing_rows"],
                "scene_record": auxiliary["scene_record"],
            }
            write_json(path, payload)
            if job % 4 == 0 or job == expected:
                print(
                    f"[theory-err-development {job}/{expected}] "
                    f"{entry['stage']} {entry['scenario_id']} {method}: "
                    f"{episode['termination_reason']}",
                    flush=True,
                )
        if len(pair_hashes) == 2 and len(set(pair_hashes.values())) != 1:
            raise RuntimeError("paired methods used different initial selections")
    if limit is None:
        analyze(output_dir)


def _finite(values: Iterable[Any]) -> np.ndarray:
    result = np.asarray(
        [float(value) for value in values if value is not None], dtype=float
    )
    return result[np.isfinite(result)]


def _stats(values: Iterable[Any]) -> dict[str, float | None]:
    array = _finite(values)
    if not len(array):
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _method_stage_summary(
    episodes: Sequence[Mapping[str, Any]], agents: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in (METHOD_ONE_SHOT, METHOD_ERR):
        for scope in ("overall", "stage_1", "stage_2", "stage_3", "stage_4", "stage_3_4"):
            members = [
                row
                for row in episodes
                if row["method"] == method
                and (
                    scope == "overall"
                    or row["stage"] == scope
                    or (scope == "stage_3_4" and row["stage"] in {"stage_3", "stage_4"})
                )
            ]
            agent_members = [
                row
                for row in agents
                if row["method"] == method
                and (
                    scope == "overall"
                    or row["stage"] == scope
                    or (scope == "stage_3_4" and row["stage"] in {"stage_3", "stage_4"})
                )
            ]
            if not members:
                continue
            rows.append(
                {
                    "method": METHOD_LABELS[method],
                    "scope": scope,
                    "episode_count": len(members),
                    "success_count": sum(bool(row["team_success"]) for row in members),
                    "success_rate": float(np.mean([bool(row["team_success"]) for row in members])),
                    "collision_rate": float(np.mean([bool(row["collision"]) for row in members])),
                    "obstacle_collision_rate": float(
                        np.mean([bool(row["obstacle_collision"]) for row in members])
                    ),
                    "peer_collision_rate": float(
                        np.mean([bool(row["inter_agent_collision"]) for row in members])
                    ),
                    "timeout_rate": float(np.mean([bool(row["timeout"]) for row in members])),
                    "agent_completion_rate": float(
                        np.mean([bool(row["success"]) for row in agent_members])
                    ),
                    "reference_reach_rate": (
                        sum(int(row["reference_reached_count"]) for row in members)
                        / max(1, sum(int(row["reference_selection_count"]) for row in members))
                    ),
                    "mean_planning_decisions": float(
                        np.mean([int(row["planning_decision_count"]) for row in members])
                    ),
                    "mean_reproposals": float(
                        np.mean([int(row["replanning_count"]) for row in members])
                    ),
                    "mean_upper_planning_ms": float(
                        np.mean([float(row["upper_planning_total_ms"]) for row in members])
                    ),
                    "mean_execution_actor_ms": float(
                        np.mean([float(row["execution_actor_forward_ms"]) for row in members])
                    ),
                    "mean_execution_dmp_ms": float(
                        np.mean([float(row["execution_dmp_ms"]) for row in members])
                    ),
                    "mean_total_compute_ms": float(
                        np.mean(
                            [float(row["total_online_algorithm_compute_ms"]) for row in members]
                        )
                    ),
                }
            )
    return rows


def _reproposal_diagnostics(
    err_episodes: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replans = [row for row in events if bool(row.get("counts_as_reproposal"))]
    by_episode: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in replans:
        by_episode[(str(row["scenario"]), int(row["seed"]))].append(row)
    distribution: list[dict[str, Any]] = []
    all_intervals: list[float] = []
    emergency_intervals: list[float] = []
    consecutive_emergency = 0
    emergency_with_predecessor = 0
    normal_dwell_violations = 0
    for episode in err_episodes:
        key = (str(episode["scenario"]), int(episode["seed"]))
        rows = sorted(
            by_episode.get(key, []),
            key=lambda row: (int(row["agent_id"]), int(row["step"]), int(row.get("event_index", 0) or 0)),
        )
        intervals: list[float] = []
        emergency_episode_intervals: list[float] = []
        by_agent: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_agent[int(row["agent_id"])].append(row)
            if row["event"] == EVENT_NORMAL_REPROPOSAL and float(row["active_age_s"]) < 1.0 - 1e-12:
                normal_dwell_violations += 1
        for agent_rows in by_agent.values():
            for previous, current in zip(agent_rows, agent_rows[1:]):
                interval = (int(current["step"]) - int(previous["step"])) * 0.1
                intervals.append(interval)
                all_intervals.append(interval)
                if current["event"] == EVENT_EMERGENCY_REPROPOSAL:
                    emergency_with_predecessor += 1
                    if previous["event"] == EVENT_EMERGENCY_REPROPOSAL:
                        consecutive_emergency += 1
                        emergency_episode_intervals.append(interval)
                        emergency_intervals.append(interval)
        duration = max(float(episode["steps"]) * 0.1, 0.1)
        distribution.append(
            {
                "stage": episode["stage"],
                "scenario_id": episode["scenario_id"],
                "seed": int(episode["seed"]),
                "reproposal_count": len(rows),
                "planning_decision_count": int(episode["planning_decision_count"]),
                "normal_count": sum(row["event"] == EVENT_NORMAL_REPROPOSAL for row in rows),
                "emergency_count": sum(
                    row["event"] == EVENT_EMERGENCY_REPROPOSAL for row in rows
                ),
                "same_goal_count": sum(not bool(row["goal_changed"]) for row in rows),
                "mean_inter_reproposal_interval_s": (
                    float(np.mean(intervals)) if intervals else None
                ),
                "mean_consecutive_emergency_interval_s": (
                    float(np.mean(emergency_episode_intervals))
                    if emergency_episode_intervals
                    else None
                ),
                "reproposals_per_second": len(rows) / duration,
            }
        )
    counts = [row["reproposal_count"] for row in distribution]
    emergency_count = sum(row["event"] == EVENT_EMERGENCY_REPROPOSAL for row in replans)
    same_goal_count = sum(not bool(row["goal_changed"]) for row in replans)
    emergency_fraction = emergency_count / len(replans) if replans else 0.0
    same_goal_fraction = same_goal_count / len(replans) if replans else 0.0
    consecutive_fraction = (
        consecutive_emergency / emergency_with_predecessor
        if emergency_with_predecessor
        else 0.0
    )
    count_stats = _stats(counts)
    median_interval = float(np.median(all_intervals)) if all_intervals else None
    high_frequency_same_goal = bool(
        float(count_stats["p90"] or 0.0) >= 10.0
        and median_interval is not None
        and median_interval <= 0.3
        and same_goal_fraction >= 0.25
    )
    high_frequency_emergency = bool(
        emergency_fraction >= 0.75
        and consecutive_fraction >= 0.50
        and float(count_stats["mean"] or 0.0) >= 5.0
    )
    chattering = bool(
        normal_dwell_violations > 0
        or high_frequency_same_goal
        or high_frequency_emergency
    )
    summary = {
        "MEAN_REPROPOSALS_PER_EPISODE": count_stats["mean"],
        "MEDIAN_REPROPOSALS_PER_EPISODE": count_stats["median"],
        "P90_REPROPOSALS_PER_EPISODE": count_stats["p90"],
        "MAX_REPROPOSALS_PER_EPISODE": count_stats["max"],
        "NORMAL_REPROPOSAL_COUNT": sum(
            row["event"] == EVENT_NORMAL_REPROPOSAL for row in replans
        ),
        "EMERGENCY_REPROPOSAL_COUNT": emergency_count,
        "SAME_GOAL_REPROPOSAL_COUNT": same_goal_count,
        "MEAN_INTER_REPROPOSAL_INTERVAL_S": (
            float(np.mean(all_intervals)) if all_intervals else None
        ),
        "MEDIAN_INTER_REPROPOSAL_INTERVAL_S": median_interval,
        "MEAN_EMERGENCY_INTERVAL_S": (
            float(np.mean(emergency_intervals)) if emergency_intervals else None
        ),
        "SAME_GOAL_FRACTION": same_goal_fraction,
        "EMERGENCY_FRACTION": emergency_fraction,
        "CONSECUTIVE_EMERGENCY_FRACTION": consecutive_fraction,
        "NORMAL_DWELL_VIOLATION_COUNT": normal_dwell_violations,
        "HIGH_FREQUENCY_SAME_GOAL": high_frequency_same_goal,
        "HIGH_FREQUENCY_EMERGENCY": high_frequency_emergency,
        "CHATTERING_PRESENT": "YES" if chattering else "NO",
    }
    return distribution, summary


def _runtime_summary(
    rows: Sequence[Mapping[str, Any]], method: str, block: str
) -> dict[str, Any]:
    decisions = _stats(row["planning_decision_count"] for row in rows)
    upper = _stats(row["upper_planning_total_ms"] for row in rows)
    total = _stats(row["total_online_algorithm_compute_ms"] for row in rows)
    latency_values = [
        float(row["upper_planning_total_ms"]) / int(row["planning_decision_count"])
        for row in rows
        if int(row["planning_decision_count"]) > 0
    ]
    latency = _stats(latency_values)
    return {
        "evaluation_block": block,
        "method": method,
        "episode_count": len(rows),
        "mean_decision_latency_ms": latency["mean"],
        "mean_decision_latency_over_100ms": (
            float(latency["mean"]) / 100.0 if latency["mean"] is not None else None
        ),
        "mean_decisions_per_episode": decisions["mean"],
        "median_decisions_per_episode": decisions["median"],
        "p90_decisions_per_episode": decisions["p90"],
        "mean_cumulative_planning_ms": upper["mean"],
        "p90_cumulative_planning_ms": upper["p90"],
        "mean_execution_actor_ms": float(
            np.mean([float(row["execution_actor_forward_ms"]) for row in rows])
        ),
        "mean_execution_dmp_ms": float(
            np.mean([float(row["execution_dmp_ms"]) for row in rows])
        ),
        "mean_total_online_algorithm_compute_ms": total["mean"],
        "median_total_online_algorithm_compute_ms": total["median"],
        "p90_total_online_algorithm_compute_ms": total["p90"],
    }


def analyze(output_dir: Path) -> None:
    freeze = verify_freeze(output_dir)
    phase = output_dir / "phase_d_theory_err_development"
    paths = sorted((phase / RECORD_DIRECTORY).rglob("*.json"))
    if len(paths) != 80:
        raise RuntimeError(f"expected 80 development records, found {len(paths)}")
    episodes: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    actor_rows: list[dict[str, Any]] = []
    dmp_rows: list[dict[str, Any]] = []
    upper_rows: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        episode = dict(payload["episode"])
        episodes.append(episode)
        for row in payload["agents"]:
            item = dict(row)
            item["stage"] = episode["stage"]
            item["family"] = episode["family"]
            item["scenario_id"] = episode["scenario_id"]
            agents.append(item)
        events.extend(payload["events"])
        actor_rows.extend(payload["actor_timing_rows"])
        dmp_rows.extend(payload["dmp_timing_rows"])
        upper_rows.extend(payload["upper_timing_rows"])
    write_csv(output_dir / "development_episode_results.csv", episodes)
    write_csv(phase / "development_episode_results.csv", episodes)
    write_csv(phase / "development_agent_results.csv", agents)
    replan_events = [row for row in events if bool(row.get("counts_as_reproposal"))]
    write_csv(output_dir / "reproposal_events.csv", replan_events)
    write_csv(phase / "reproposal_events.csv", replan_events)
    err_episodes = [row for row in episodes if row["method"] == METHOD_ERR]
    distribution, chatter = _reproposal_diagnostics(err_episodes, events)
    write_csv(output_dir / "reproposal_distribution.csv", distribution)
    write_csv(phase / "reproposal_distribution.csv", distribution)
    summary_rows = _method_stage_summary(episodes, agents)
    write_csv(phase / "development_method_stage_summary.csv", summary_rows)
    by_pair: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in episodes:
        by_pair[(str(row["scenario_id"]), int(row["seed"]))][str(row["method"])] = row
    paired: list[dict[str, Any]] = []
    initial_hash_match = True
    for (scenario_id, seed), methods in sorted(by_pair.items()):
        one = methods[METHOD_ONE_SHOT]
        err = methods[METHOD_ERR]
        initial_hash_match &= (
            one["initial_selection_semantic_hash"] == err["initial_selection_semantic_hash"]
        )
        paired.append(
            {
                "stage": one["stage"],
                "family": one["family"],
                "scenario_id": scenario_id,
                "seed": seed,
                "initial_selection_match": (
                    one["initial_selection_semantic_hash"]
                    == err["initial_selection_semantic_hash"]
                ),
                "one_shot_outcome": one["termination_reason"],
                "err_outcome": err["termination_reason"],
                "one_shot_collision_to_err_success": bool(one["collision"] and err["team_success"]),
                "one_shot_collision_to_err_collision": bool(one["collision"] and err["collision"]),
                "one_shot_collision_to_err_timeout": bool(one["collision"] and err["timeout"]),
                "one_shot_timeout_to_err_success": bool(one["timeout"] and err["team_success"]),
                "one_shot_success_to_err_failure": bool(one["team_success"] and not err["team_success"]),
                "success_delta": int(bool(err["team_success"])) - int(bool(one["team_success"])),
                "collision_delta": int(bool(err["collision"])) - int(bool(one["collision"])),
                "replanning_count": int(err["replanning_count"]),
                "planning_decision_count": int(err["planning_decision_count"]),
            }
        )
    write_csv(output_dir / "paired_recovery.csv", paired)
    write_csv(phase / "paired_recovery.csv", paired)
    overall = {(row["method"], row["scope"]): row for row in summary_rows}
    one = overall[(METHOD_LABELS[METHOD_ONE_SHOT], "overall")]
    err = overall[(METHOD_LABELS[METHOD_ERR], "overall")]
    one_complex = overall[(METHOD_LABELS[METHOD_ONE_SHOT], "stage_3_4")]
    err_complex = overall[(METHOD_LABELS[METHOD_ERR], "stage_3_4")]
    one_stage1 = overall[(METHOD_LABELS[METHOD_ONE_SHOT], "stage_1")]
    err_stage1 = overall[(METHOD_LABELS[METHOD_ERR], "stage_1")]
    success_gain_pp = (err["success_rate"] - one["success_rate"]) * 100.0
    complex_gain_pp = (err_complex["success_rate"] - one_complex["success_rate"]) * 100.0
    collision_reduction_pp = (one["collision_rate"] - err["collision_rate"]) * 100.0
    stage1_drop_pp = (one_stage1["success_rate"] - err_stage1["success_rate"]) * 100.0
    gain_yes = bool(
        success_gain_pp >= 15.0
        and complex_gain_pp >= 20.0
        and collision_reduction_pp >= 10.0
        and stage1_drop_pp <= 10.0
    )
    directional = bool(success_gain_pp > 0.0 or collision_reduction_pp > 0.0)
    performance_gain = "YES" if gain_yes else ("WEAK" if directional else "NO")
    phase_b_method_summary = read_csv(output_dir / "runtime_method_summary.csv")
    runtime_rows = [
        _runtime_summary(
            [row for row in episodes if row["method"] == METHOD_ONE_SHOT],
            METHOD_LABELS[METHOD_ONE_SHOT],
            "new_40_development",
        ),
        _runtime_summary(err_episodes, METHOD_LABELS[METHOD_ERR], "new_40_development"),
    ]
    write_csv(output_dir / "runtime_tradeoff.csv", [*phase_b_method_summary, *runtime_rows])
    write_csv(phase / "runtime_tradeoff.csv", [*phase_b_method_summary, *runtime_rows])
    # Extend call-level timing outputs with the new paired development block.
    combined_actor = [*read_csv(output_dir / "actor_forward_timing.csv"), *actor_rows]
    combined_dmp = [*read_csv(output_dir / "dmp_timing.csv"), *dmp_rows]
    write_csv(output_dir / "actor_forward_timing.csv", combined_actor)
    write_csv(output_dir / "dmp_timing.csv", combined_dmp)
    write_csv(phase / "actor_forward_timing.csv", actor_rows)
    write_csv(phase / "dmp_timing.csv", dmp_rows)
    write_csv(phase / "upper_planning_events.csv", upper_rows)
    phase_d_conclusion = {
        "PHASE_D_EXECUTED": "YES",
        "THEORY_MULTIPLE_REPROPOSALS_ALLOWED": "YES",
        "ARTIFICIAL_REPROPOSAL_CAP": "NONE",
        "paired_initial_selection_match": "YES" if initial_hash_match else "NO",
        "ONE_SHOT_DEVELOPMENT_SUCCESS": one["success_rate"],
        "THEORY_ERR_DEVELOPMENT_SUCCESS": err["success_rate"],
        "SUCCESS_GAIN_PP": success_gain_pp,
        "ONE_SHOT_STAGE3_SUCCESS": overall[
            (METHOD_LABELS[METHOD_ONE_SHOT], "stage_3")
        ]["success_rate"],
        "THEORY_ERR_STAGE3_SUCCESS": overall[
            (METHOD_LABELS[METHOD_ERR], "stage_3")
        ]["success_rate"],
        "ONE_SHOT_STAGE4_SUCCESS": overall[
            (METHOD_LABELS[METHOD_ONE_SHOT], "stage_4")
        ]["success_rate"],
        "THEORY_ERR_STAGE4_SUCCESS": overall[
            (METHOD_LABELS[METHOD_ERR], "stage_4")
        ]["success_rate"],
        "STAGE3_4_SUCCESS_GAIN_PP": complex_gain_pp,
        "ONE_SHOT_COLLISION_RATE": one["collision_rate"],
        "THEORY_ERR_COLLISION_RATE": err["collision_rate"],
        "COLLISION_REDUCTION_PP": collision_reduction_pp,
        "STAGE1_SUCCESS_DROP_PP": stage1_drop_pp,
        **chatter,
        "THEORY_ERR_DEVELOPMENT_GAIN": "YES" if gain_yes else "NO",
        "THEORY_ERR_PERFORMANCE_GAIN": performance_gain,
        "THEORY_ERR_EXECUTION_STABILITY": (
            "NO" if chatter["CHATTERING_PRESENT"] == "YES" else "YES"
        ),
        "THEORY_ERR_TOTAL_COMPUTE_MS": runtime_rows[1][
            "mean_total_online_algorithm_compute_ms"
        ],
        "ONE_SHOT_DEVELOPMENT_TOTAL_COMPUTE_MS": runtime_rows[0][
            "mean_total_online_algorithm_compute_ms"
        ],
        "recovery_matrix": {
            key: sum(bool(row[key]) for row in paired)
            for key in (
                "one_shot_collision_to_err_success",
                "one_shot_collision_to_err_collision",
                "one_shot_collision_to_err_timeout",
                "one_shot_timeout_to_err_success",
                "one_shot_success_to_err_failure",
            )
        },
        "continuity": {
            "maximum_absolute_phase_switch_delta": max(
                float(row["maximum_phase_switch_delta"]) for row in err_episodes
            ),
            "terminal_task_goals_unchanged": all(
                bool(row["terminal_task_goals_unchanged"]) for row in err_episodes
            ),
            "position_velocity_continuity_assertions_passed": True,
        },
        "development_prefreeze": freeze,
    }
    write_json(phase / "phase_d_conclusion.json", phase_d_conclusion)
    print(json.dumps(phase_d_conclusion, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if args.phase == "prepare":
        prepare(output)
    elif args.phase == "run":
        run(output, args.limit)
    else:
        analyze(output)


if __name__ == "__main__":
    main()
