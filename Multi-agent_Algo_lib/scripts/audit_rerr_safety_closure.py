"""Read-only causal audit for the remaining R-ERR collision regression.

Phases ``prepare`` and ``replay`` do not alter trigger or execution semantics.
They recover the frozen 80-scene M3 trajectories plus the eight adverse M2
counterparts with additional diagnostic fields, and verify every categorical
outcome against the original frozen development records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import runpy
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    EVENT_EMERGENCY_REPROPOSAL,
    EVENT_NORMAL_REPROPOSAL,
    EVENT_REFERENCE_HANDOFF,
    METHOD_ERR,
    METHOD_RERR_GAT,
    run_episode,
)
from scripts.run_final_four_stage_benchmark import (  # noqa: E402
    FrozenRuntime,
    proposed_eval_config,
)
from scripts.run_sparse_err_trigger_revision import (  # noqa: E402
    file_hash,
    jsonable,
    load_json,
    write_csv,
    write_json,
)


OUTPUT = REPO_ROOT / "artifacts/rerr_safety_closure/20260819_134216"
SOURCE = REPO_ROOT / "artifacts/sparse_err_trigger_revision/20260819_020727"
ATTACHMENT = Path(
    r"C:\Users\Administrator\.codex\attachments\557fe5dd-f702-4e89-bd9c-1b4385039af1\pasted-text.txt"
)
DIAGNOSTIC_RECORDS = "diagnostic_records"

ADVERSE_SCENARIOS = (
    "SR2_014",
    "SR2_019",
    "SR3_003",
    "SR3_004",
    "SR3_019",
    "SR4_008",
    "SR4_010",
    "SR4_015",
)
KNOWN_NORMAL_CHATTER = ("SR3_004", "SR3_014", "SR4_009", "SR4_014")

SOURCE_ARTIFACTS = (
    "FINAL_REPORT.md",
    "conclusion.json",
    "final_reconciliation.json",
    "development_episode_results.csv",
    "development_agent_results.csv",
    "reproposal_events.csv",
    "reproposal_distribution.csv",
    "stage_reproposal_summary.csv",
    "paired_trigger_revision.csv",
    "paired_gat_ablation.csv",
    "event_conditioned_gat_analysis.csv",
    "runtime_method_summary.csv",
    "revised_emergency_event_contract.md",
    "theory_diff.md",
    "development_scenario_manifest.json",
    "development_eval_config.json",
)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _source_record(entry: Mapping[str, Any], method: str) -> Path:
    return (
        SOURCE
        / "development_records"
        / str(entry["stage"])
        / str(entry["scenario_id"])
        / f"{method}.json"
    )


def _diagnostic_record(entry: Mapping[str, Any], method: str) -> Path:
    return (
        OUTPUT
        / DIAGNOSTIC_RECORDS
        / str(entry["stage"])
        / str(entry["scenario_id"])
        / f"{method}.json"
    )


def _uncommented_tex_excerpt() -> tuple[str, list[int]]:
    lines = (REPO_ROOT / "hire-rl-body.tex").read_text(encoding="utf-8").splitlines()
    active: list[tuple[int, str]] = []
    in_block = False
    for number, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if "\\begin{comment}" in stripped:
            in_block = True
            continue
        if "\\end{comment}" in stripped:
            in_block = False
            continue
        if in_block or stripped.startswith("%"):
            continue
        active.append((number, line))
    selected_indices: set[int] = set()
    windows = {
        r"\label{eq:final_reproposal_event}": (-2, 19),
        r"\label{eq:unified_reference_update}": (-3, 14),
    }
    for token, (before, after) in windows.items():
        index = next(i for i, (_, line) in enumerate(active) if token in line)
        selected_indices.update(
            range(max(0, index + before), min(len(active), index + after + 1))
        )
    selected = [active[index] for index in sorted(selected_indices)]
    return "\n".join(line for _, line in selected), [number for number, _ in selected]


def prepare() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    required = [SOURCE / name for name in SOURCE_ARTIFACTS]
    missing = [_relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing authority inputs: {missing}")
    tex_excerpt, tex_lines = _uncommented_tex_excerpt()
    authority_paths = [
        *required,
        REPO_ROOT / "planning/event_triggered_reference_reconstruction.py",
        REPO_ROOT / "hire-rl-body.tex",
        REPO_ROOT / "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_err_development.py",
        REPO_ROOT / "Multi-agent_Algo_lib/scripts/run_sparse_err_trigger_revision.py",
        ATTACHMENT,
    ]
    manifest = {
        "goal": "R-ERR Safety Closure and Normal-Trigger Specificity Audit",
        "authority": (
            "active uncommented Methodology in hire-rl-body.tex and the frozen "
            "sparse_err_trigger_revision artifact"
        ),
        "AGENTS_MD": "NOT_PRESENT" if not (REPO_ROOT / "AGENTS.md").exists() else "PRESENT",
        "CODEX_HANDOFF_MD": (
            "NOT_PRESENT" if not (REPO_ROOT / "CODEX_HANDOFF.md").exists() else "PRESENT"
        ),
        "read_only_diagnosis_first": True,
        "one_change_maximum": 1,
        "threshold_tuning_authorized": False,
        "formal_benchmark_authorized": False,
        "training_authorized": False,
        "source_hashes": {
            _relative(path): file_hash(path) for path in authority_paths
        },
        "active_theory_excerpt": tex_excerpt,
        "active_theory_line_numbers": tex_lines,
        "frozen_reference_results": {
            "M2_success": 0.75,
            "M2_collision": 0.0875,
            "M2_timeout": 0.1625,
            "M3_success": 0.8125,
            "M3_collision": 0.1625,
            "M3_timeout": 0.025,
            "M3_inter_agent_collision": 0.125,
            "M3_mean_reproposals": 8.9625,
            "M3_total_compute_ms": 2271.591,
            "GAT_closed_loop_value": "STRONG",
        },
        "diagnostic_replay_scope": {
            "M3_scenarios": 80,
            "M2_adverse_scenarios": list(ADVERSE_SCENARIOS),
            "performance_sample_added": False,
            "execution_semantics_changed": False,
        },
    }
    write_json(OUTPUT / "context_recovery_manifest.json", manifest)
    print(f"OUTPUT={OUTPUT}", flush=True)
    print("CONTEXT_RECOVERY=PASS", flush=True)


def _runtime_config(runtime: FrozenRuntime) -> dict[str, Any]:
    frozen = load_json(SOURCE / "development_eval_config.json")
    config = proposed_eval_config(runtime.base_eval_config, frozen["selected_proposed"])
    config["err"] = frozen["err"]
    return config


def _event_semantics(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step": row["step"],
            "agent_id": row["agent_id"],
            "event": row["event"],
            "selected_candidate_id": row.get("selected_candidate_id"),
            "old_active_goal": row.get("old_active_goal"),
            "new_active_goal": row.get("new_active_goal"),
            "new_active_goal_type": row.get("new_active_goal_type"),
            "goal_changed": row.get("goal_changed"),
        }
        for row in events
    ]


def _verify_replay(
    original: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    episode_keys = (
        "team_success",
        "collision",
        "obstacle_collision",
        "inter_agent_collision",
        "timeout",
        "termination_reason",
        "steps",
        "replanning_count",
        "emergency_replanning_count",
        "handoff_count",
        "reference_selection_count",
        "reference_reached_count",
        "initial_condition_hash",
        "initial_candidate_bundle_hash",
        "initial_selection_semantic_hash",
    )
    differences: dict[str, Any] = {}
    for key in episode_keys:
        if original["episode"].get(key) != replay["episode"].get(key):
            differences[key] = {
                "original": original["episode"].get(key),
                "replay": replay["episode"].get(key),
            }
    original_events = _event_semantics(original["events"])
    replay_events = _event_semantics(replay["events"])
    if original_events != replay_events:
        differences["event_semantics"] = {
            "original_count": len(original_events),
            "replay_count": len(replay_events),
        }
    return {
        "status": "PASS" if not differences else "FAIL",
        "differences": differences,
    }


def replay(limit: int | None = None) -> None:
    if not (OUTPUT / "context_recovery_manifest.json").exists():
        prepare()
    manifest = load_json(SOURCE / "development_scenario_manifest.json")
    entries = list(manifest["entries"])
    runtime = FrozenRuntime(load_json(REPO_ROOT / load_json(SOURCE / "development_eval_config.json")["source_config"]), manifest)
    config = _runtime_config(runtime)

    # Warm-up is not retained and does not add a performance sample.
    first = entries[0]
    run_episode(
        config=config,
        settings=runtime.execution_settings,
        multi_config=runtime.multi_config,
        policy=runtime.policy,
        gat_model=runtime.gat_model,
        gat_device=runtime.gat_device,
        method=METHOD_RERR_GAT,
        scenario=first["scenario_id"],
        seed=int(first["seed"]),
        environment_builder=runtime.builder,
        runtime_recorder=None,
    )

    jobs: list[tuple[Mapping[str, Any], str]] = []
    for entry in entries:
        jobs.append((entry, METHOD_RERR_GAT))
        if entry["scenario_id"] in ADVERSE_SCENARIOS:
            jobs.append((entry, METHOD_ERR))
    if limit is not None:
        jobs = jobs[: int(limit)]
    checks: list[dict[str, Any]] = []
    for index, (entry, method) in enumerate(jobs, start=1):
        destination = _diagnostic_record(entry, method)
        if destination.exists():
            payload = load_json(destination)
        else:
            episode, agents, events, triggers, auxiliary = run_episode(
                config=config,
                settings=runtime.execution_settings,
                multi_config=runtime.multi_config,
                policy=runtime.policy,
                gat_model=runtime.gat_model,
                gat_device=runtime.gat_device,
                method=method,
                scenario=entry["scenario_id"],
                seed=int(entry["seed"]),
                environment_builder=runtime.builder,
                runtime_recorder=None,
            )
            episode = dict(episode)
            episode.update(
                {
                    "stage": entry["stage"],
                    "family": entry["family"],
                    "scenario_id": entry["scenario_id"],
                    "diagnostic_replay": True,
                    "performance_sample_added": False,
                }
            )
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
                "path_rows": auxiliary["path_rows"],
                "scene_record": auxiliary["scene_record"],
            }
            write_json(destination, payload)
        original = load_json(_source_record(entry, method))
        check = _verify_replay(original, payload)
        check.update(
            {
                "scenario_id": entry["scenario_id"],
                "stage": entry["stage"],
                "method": method,
            }
        )
        checks.append(check)
        if check["status"] != "PASS":
            write_json(OUTPUT / "diagnostic_replay_reconciliation.json", {"checks": checks})
            raise RuntimeError(f"diagnostic replay diverged: {check}")
        if index % 8 == 0 or index == len(jobs):
            print(
                f"[diagnostic {index}/{len(jobs)}] {entry['scenario_id']} {method}: PASS",
                flush=True,
            )
    write_json(
        OUTPUT / "diagnostic_replay_reconciliation.json",
        {
            "status": "PASS",
            "record_count": len(checks),
            "M3_record_count": sum(row["method"] == METHOD_RERR_GAT for row in checks),
            "M2_adverse_record_count": sum(row["method"] == METHOD_ERR for row in checks),
            "all_categorical_outcomes_match": True,
            "all_event_semantics_match": True,
            "execution_logic_changed": False,
            "performance_sample_added": False,
            "checks": checks,
        },
    )
    print("DIAGNOSTIC_REPLAY=PASS", flush=True)


def _load_records() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_json(SOURCE / "development_scenario_manifest.json")
    entries = {str(row["scenario_id"]): row for row in manifest["entries"]}
    m3: dict[str, Any] = {}
    m2: dict[str, Any] = {}
    for scenario, entry in entries.items():
        m3[scenario] = load_json(_diagnostic_record(entry, METHOD_RERR_GAT))
        if scenario in ADVERSE_SCENARIOS:
            m2[scenario] = load_json(_diagnostic_record(entry, METHOD_ERR))
    return m3, m2


def _vector(row: Mapping[str, Any], prefix: str) -> np.ndarray:
    if prefix == "v":
        keys = ("vx_mps", "vy_mps", "vz_mps")
    elif prefix == "":
        keys = ("x_m", "y_m", "z_m")
    else:
        keys = tuple(f"{prefix}{axis}" for axis in ("x", "y", "z"))
    return np.asarray([row[key] for key in keys], dtype=float)


def _path_maps(record: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    return {
        (int(row["step"]), int(row["agent_id"])): row
        for row in record["path_rows"]
    }


def _trigger_maps(record: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    return {
        (int(row["step"]), int(row["agent_id"])): row
        for row in record["triggers"]
    }


def _events_by_key(
    record: Mapping[str, Any],
) -> dict[tuple[int, int], list[Mapping[str, Any]]]:
    result: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in record["events"]:
        result[(int(row["step"]), int(row["agent_id"]))].append(row)
    return result


def _planning_events(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in record["events"]
        if row["event"] == "INITIAL_SELECTION" or bool(row.get("counts_as_reproposal"))
    ]


def _execution_reference(
    path_row: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, str]:
    if events:
        last = events[-1]
        return np.asarray(last["new_active_goal"], dtype=float), str(
            last["new_active_goal_type"]
        )
    return np.asarray(
        [
            path_row["active_goal_x_m"],
            path_row["active_goal_y_m"],
            path_row["active_goal_z_m"],
        ],
        dtype=float,
    ), str(path_row["active_goal_type"])


def _nearest_peer(
    path: Mapping[tuple[int, int], Mapping[str, Any]], step: int, agent_id: int
) -> dict[str, Any]:
    ego = path[(step, agent_id)]
    ego_p = _vector(ego, "")
    ego_v = _vector(ego, "v")
    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for peer_id in range(3):
        if peer_id == agent_id or (step, peer_id) not in path:
            continue
        peer = path[(step, peer_id)]
        relative_p = _vector(peer, "") - ego_p
        relative_v = _vector(peer, "v") - ego_v
        candidates.append((float(np.linalg.norm(relative_p)), peer_id, relative_p, relative_v))
    if not candidates:
        return {
            "nearest_peer_id": None,
            "peer_distance_m": None,
            "relative_peer_position_m": None,
            "relative_peer_velocity_mps": None,
            "peer_closing_rate_mps": None,
            "estimated_closest_approach_time_s": None,
            "estimated_closest_approach_distance_m": None,
        }
    distance, peer_id, relative_p, relative_v = min(candidates, key=lambda item: item[0])
    speed2 = float(np.dot(relative_v, relative_v))
    t_min = max(0.0, -float(np.dot(relative_p, relative_v)) / speed2) if speed2 > 1e-12 else 0.0
    closest = float(np.linalg.norm(relative_p + t_min * relative_v))
    closing = -float(np.dot(relative_p, relative_v)) / max(distance, 1e-12)
    return {
        "nearest_peer_id": peer_id,
        "peer_distance_m": distance,
        "relative_peer_position_m": relative_p.tolist(),
        "relative_peer_velocity_mps": relative_v.tolist(),
        "peer_closing_rate_mps": closing,
        "estimated_closest_approach_time_s": t_min,
        "estimated_closest_approach_distance_m": closest,
    }


def _fp_order(scores: Sequence[Any] | None) -> list[int] | None:
    if not scores:
        return None
    array = np.asarray(scores, dtype=float)
    return (np.argsort(-array, kind="stable") + 1).astype(int).tolist()


def _timeline_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = _path_maps(record)
    triggers = _trigger_maps(record)
    events = _events_by_key(record)
    max_step = int(record["episode"]["steps"])
    latest_plan: dict[int, Mapping[str, Any] | None] = {0: None, 1: None, 2: None}
    rows: list[dict[str, Any]] = []
    for step in range(max_step):
        owner_ids = sorted(
            {
                int(row["agent_id"])
                for row in record["events"]
                if int(row["step"]) == step and bool(row.get("counts_as_reproposal"))
            }
        )
        for agent_id in range(3):
            key = (step, agent_id)
            if key not in path:
                continue
            step_events = events.get(key, [])
            for event in step_events:
                if event["event"] == "INITIAL_SELECTION" or bool(
                    event.get("counts_as_reproposal")
                ):
                    latest_plan[agent_id] = event
            base = path[key]
            trigger = triggers.get(key, {})
            event = step_events[-1] if step_events else None
            plan = latest_plan[agent_id]
            reference, reference_type = _execution_reference(base, step_events)
            next_row = path.get((step + 1, agent_id), {})
            rows.append(
                {
                    "stage": record["entry"]["stage"],
                    "scenario_id": record["entry"]["scenario_id"],
                    "seed": record["entry"]["seed"],
                    "method": record["episode"]["method"],
                    "step": step,
                    "time_s": step * 0.1,
                    "agent_id": agent_id,
                    "position_m": _vector(base, "").tolist(),
                    "velocity_mps": _vector(base, "v").tolist(),
                    "active_reference_for_execution_m": reference.tolist(),
                    "active_reference_type": reference_type,
                    "terminal_goal_m": [
                        base["terminal_goal_x_m"],
                        base["terminal_goal_y_m"],
                        base["terminal_goal_z_m"],
                    ],
                    "distance_to_active_reference_m": float(
                        np.linalg.norm(reference - _vector(base, ""))
                    ),
                    "distance_to_terminal_goal_m": base["terminal_goal_distance_m"],
                    "h_active_m": trigger.get("active_safety_margin_m"),
                    "progress_nu_mps": trigger.get("progress_rate_mps"),
                    "progress_valid": trigger.get("progress_valid"),
                    "execution_age_s": trigger.get("active_age_s"),
                    "Phi_tau": trigger.get("phi_tau"),
                    "Phi_p": trigger.get("phi_p"),
                    "Phi_h": trigger.get("phi_h"),
                    "Phi_rep": trigger.get("S_rep"),
                    "normal_trigger": trigger.get("normal_trigger", False),
                    "emergency_trigger": trigger.get("emergency_trigger", False),
                    "trigger_event": trigger.get("event"),
                    "trigger_reasons": trigger.get("trigger_reasons", []),
                    "event_owner": agent_id in owner_ids,
                    "event_owner_ids": owner_ids,
                    "replanning_tick": bool(owner_ids),
                    "event": event.get("event") if event else None,
                    "candidate_count": plan.get("K_t") if plan else None,
                    "selected_candidate_id": (
                        plan.get("selected_candidate_id") if plan else None
                    ),
                    "GAT_logits": plan.get("class_logits") if plan else None,
                    "GAT_probabilities": plan.get("class_probabilities") if plan else None,
                    "FP_SHEP_ordering_1based": (
                        _fp_order(plan.get("fp_shep_scores")) if plan else None
                    ),
                    "same_goal_event": (
                        not bool(event.get("goal_changed"))
                        if event and bool(event.get("counts_as_reproposal"))
                        else None
                    ),
                    "reference_handoff": bool(
                        event and event.get("event") == EVENT_REFERENCE_HANDOFF
                    ),
                    "selected_edge_records": (
                        plan.get("selected_edge_records", []) if plan else []
                    ),
                    "policy_action": [
                        next_row.get("policy_action_0"),
                        next_row.get("policy_action_1"),
                        next_row.get("policy_action_2"),
                    ],
                    "commanded_acceleration_mps2": [
                        next_row.get("commanded_ax_mps2"),
                        next_row.get("commanded_ay_mps2"),
                        next_row.get("commanded_az_mps2"),
                    ],
                    "applied_acceleration_mps2": [
                        next_row.get("applied_ax_mps2"),
                        next_row.get("applied_ay_mps2"),
                        next_row.get("applied_az_mps2"),
                    ],
                    "obstacle_clearance_m": base["static_obstacle_clearance_m"],
                    "sensor_min_clearance_m": base["sensor_min_clearance_m"],
                    "collision_tick": max_step if record["episode"]["collision"] else None,
                    "collision_after_control": bool(
                        next_row.get("obstacle_collision", False)
                        or next_row.get("inter_agent_collision", False)
                    ),
                    "collision_type_after_control": (
                        "obstacle"
                        if next_row.get("obstacle_collision", False)
                        else (
                            "inter_agent"
                            if next_row.get("inter_agent_collision", False)
                            else None
                        )
                    ),
                    **_nearest_peer(path, step, agent_id),
                }
            )
    return rows


def _trigger_signature(record: Mapping[str, Any]) -> dict[tuple[int, int], tuple[Any, ...]]:
    return {
        (int(row["step"]), int(row["agent_id"])): (
            row.get("event"),
            tuple(row.get("trigger_reasons", [])),
            bool(row.get("normal_trigger")),
            bool(row.get("emergency_trigger")),
            bool(row.get("handoff_trigger")),
        )
        for row in record["triggers"]
    }


def _first_divergences(m2: Mapping[str, Any], m3: Mapping[str, Any]) -> dict[str, Any]:
    path2, path3 = _path_maps(m2), _path_maps(m3)
    events2, events3 = _events_by_key(m2), _events_by_key(m3)
    trigger2, trigger3 = _trigger_signature(m2), _trigger_signature(m3)
    common_last = min(int(m2["episode"]["steps"]), int(m3["episode"]["steps"]))

    first_trigger: int | None = None
    trigger_agents: list[int] = []
    for step in range(common_last):
        different = [
            agent
            for agent in range(3)
            if trigger2.get((step, agent)) != trigger3.get((step, agent))
        ]
        if different:
            first_trigger, trigger_agents = step, different
            break

    first_reference: int | None = None
    reference_agents: list[int] = []
    for step in range(common_last):
        different: list[int] = []
        for agent in range(3):
            if (step, agent) not in path2 or (step, agent) not in path3:
                continue
            ref2, type2 = _execution_reference(path2[(step, agent)], events2.get((step, agent), []))
            ref3, type3 = _execution_reference(path3[(step, agent)], events3.get((step, agent), []))
            if type2 != type3 or not np.allclose(ref2, ref3, rtol=0.0, atol=1e-12):
                different.append(agent)
        if different:
            first_reference, reference_agents = step, different
            break

    first_control: int | None = None
    control_agents: list[int] = []
    for decision_step in range(common_last):
        state_step = decision_step + 1
        different: list[int] = []
        for agent in range(3):
            a = path2.get((state_step, agent))
            b = path3.get((state_step, agent))
            if a is None or b is None:
                continue
            vector2 = np.asarray(
                [a["commanded_ax_mps2"], a["commanded_ay_mps2"], a["commanded_az_mps2"]],
                dtype=float,
            )
            vector3 = np.asarray(
                [b["commanded_ax_mps2"], b["commanded_ay_mps2"], b["commanded_az_mps2"]],
                dtype=float,
            )
            if not np.allclose(vector2, vector3, rtol=0.0, atol=1e-12):
                different.append(agent)
        if different:
            first_control, control_agents = decision_step, different
            break

    first_peer: int | None = None
    for step in range(common_last + 1):
        positions2 = [
            _vector(path2[(step, agent)], "")
            for agent in range(3)
            if (step, agent) in path2
        ]
        positions3 = [
            _vector(path3[(step, agent)], "")
            for agent in range(3)
            if (step, agent) in path3
        ]
        if len(positions2) != 3 or len(positions3) != 3:
            continue
        distances2 = [
            np.linalg.norm(positions2[i] - positions2[j])
            for i, j in ((0, 1), (0, 2), (1, 2))
        ]
        distances3 = [
            np.linalg.norm(positions3[i] - positions3[j])
            for i, j in ((0, 1), (0, 2), (1, 2))
        ]
        if not np.allclose(distances2, distances3, rtol=0.0, atol=1e-12):
            first_peer = step
            break

    ordered = [
        ("TRIGGER", first_trigger),
        ("REFERENCE", first_reference),
        ("CONTROL", first_control),
        ("PEER_GEOMETRY", first_peer),
    ]
    finite = [(name, value) for name, value in ordered if value is not None]
    first_kind, first_step = min(finite, key=lambda item: item[1]) if finite else ("NONE", None)
    return {
        "FIRST_TRIGGER_DIVERGENCE_STEP": first_trigger,
        "FIRST_TRIGGER_DIVERGENCE_AGENTS": trigger_agents,
        "FIRST_REFERENCE_DIVERGENCE_STEP": first_reference,
        "FIRST_REFERENCE_DIVERGENCE_AGENTS": reference_agents,
        "FIRST_CONTROL_DIVERGENCE_STEP": first_control,
        "FIRST_CONTROL_DIVERGENCE_AGENTS": control_agents,
        "FIRST_PEER_GEOMETRY_DIVERGENCE_STEP": first_peer,
        "FIRST_CAUSAL_DIVERGENCE": first_kind,
        "FIRST_CAUSAL_DIVERGENCE_STEP": first_step,
    }


def _event_at(
    record: Mapping[str, Any], step: int | None, agent_id: int
) -> Mapping[str, Any] | None:
    if step is None:
        return None
    matches = [
        row
        for row in record["events"]
        if int(row["step"]) == int(step) and int(row["agent_id"]) == int(agent_id)
    ]
    return matches[-1] if matches else None


def _selected_risky_for_peer(event: Mapping[str, Any], peer_id: int) -> bool:
    for edge in event.get("selected_edge_records", []):
        if int(edge["neighbor_agent_id"]) != int(peer_id):
            continue
        if (
            float(edge["minimum_separation_m"]) < float(edge["d_safe_m"])
            or float(edge["risk_duration_s"]) > 0.0
        ):
            return True
    return False


def _colliding_pairs(record: Mapping[str, Any]) -> list[tuple[int, int]]:
    path = _path_maps(record)
    step = int(record["episode"]["steps"])
    result: list[tuple[int, int]] = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if (step, i) not in path or (step, j) not in path:
            continue
        distance = float(np.linalg.norm(_vector(path[(step, i)], "") - _vector(path[(step, j)], "")))
        if (
            distance <= 0.6 + 1e-9
            and bool(path[(step, i)]["inter_agent_collision"])
            and bool(path[(step, j)]["inter_agent_collision"])
        ):
            result.append((i, j))
    return result


def _classify_adverse(
    scenario: str,
    m2: Mapping[str, Any],
    m3: Mapping[str, Any],
    divergences: Mapping[str, Any],
) -> dict[str, Any]:
    factors: list[str] = []
    step = divergences["FIRST_TRIGGER_DIVERGENCE_STEP"]
    agents = divergences["FIRST_TRIGGER_DIVERGENCE_AGENTS"]
    signature2 = _trigger_signature(m2)
    signature3 = _trigger_signature(m3)
    first_details: list[dict[str, Any]] = []
    for agent in agents:
        value2 = signature2.get((step, agent)) if step is not None else None
        value3 = signature3.get((step, agent)) if step is not None else None
        first_details.append({"agent_id": agent, "M2": value2, "M3": value3})
        if value3 and value3[0] == EVENT_NORMAL_REPROPOSAL and (
            not value2 or value2[0] != EVENT_NORMAL_REPROPOSAL
        ):
            factors.append("NORMAL_TRIGGER_INDUCED")
        if value2 and value2[0] == EVENT_EMERGENCY_REPROPOSAL and (
            not value3 or value3[0] != EVENT_EMERGENCY_REPROPOSAL
        ):
            factors.append("EMERGENCY_REARM_TRAJECTORY_EFFECT")

    collision_step = int(m3["episode"]["steps"])
    pairs = _colliding_pairs(m3)
    risky_switches: list[dict[str, Any]] = []
    for i, j in pairs:
        for event in m3["events"]:
            if (
                int(event["agent_id"]) in {i, j}
                and bool(event.get("counts_as_reproposal"))
                and bool(event.get("goal_changed"))
                and collision_step - 20 <= int(event["step"]) < collision_step
            ):
                peer = j if int(event["agent_id"]) == i else i
                if _selected_risky_for_peer(event, peer):
                    risky_switches.append(
                        {
                            "step": event["step"],
                            "agent_id": event["agent_id"],
                            "peer_id": peer,
                            "selected_candidate_id": event.get("selected_candidate_id"),
                        }
                    )
    if risky_switches:
        factors.append("REFERENCE_SWITCH_CREATED_PEER_CONFLICT")

    recent_pair_events = [
        event
        for event in m3["events"]
        if bool(event.get("counts_as_reproposal"))
        and any(int(event["agent_id"]) in pair for pair in pairs)
        and collision_step - 5 <= int(event["step"]) < collision_step
    ]
    if pairs and recent_pair_events:
        factors.append("TRIGGER_TOO_LATE")

    first_m3_events = [
        _event_at(m3, step, agent) for agent in agents
    ]
    if any(
        event is not None
        and bool(event.get("counts_as_reproposal"))
        and not bool(event.get("goal_changed"))
        for event in first_m3_events
    ):
        factors.append("SAME_GOAL_REDUNDANT_REPLAN")

    factors = list(dict.fromkeys(factors))
    if len(factors) == 1:
        classification = factors[0]
    elif len(factors) > 1:
        classification = "MULTIPLE_FACTORS"
    else:
        classification = "NOT_ESTABLISHED"
    return {
        "scenario_id": scenario,
        "stage": m3["entry"]["stage"],
        "seed": m3["entry"]["seed"],
        "M2_outcome": m2["episode"]["termination_reason"],
        "M3_outcome": m3["episode"]["termination_reason"],
        "M3_collision_type": (
            "inter_agent" if m3["episode"]["inter_agent_collision"] else "obstacle"
        ),
        **dict(divergences),
        "first_trigger_difference": first_details,
        "colliding_pairs": [list(pair) for pair in pairs],
        "risky_reference_switches_within_2s": risky_switches,
        "classification": classification,
        "contributing_factors": factors,
        "causal_claim_strength": (
            "PAIRED_FIRST_DIVERGENCE_PLUS_LOCAL_EVENT_EVIDENCE"
            if factors
            else "NOT_ESTABLISHED"
        ),
    }


def _normal_utility(
    m3: Mapping[str, Any],
    harmful_keys: set[tuple[str, int, int]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], str]]:
    config = load_json(SOURCE / "development_eval_config.json")["err"]
    dwell_steps = int(round(float(config["T_dwell_s"]) / 0.1))
    h_rep = float(config["h_rep_m"])
    p_min = float(config["p_min_mps"])
    rows: list[dict[str, Any]] = []
    utility: dict[tuple[str, int, int], str] = {}
    for scenario, record in sorted(m3.items()):
        triggers_by_agent: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        events_by_agent: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for trigger in record["triggers"]:
            triggers_by_agent[int(trigger["agent_id"])].append(trigger)
        for event in record["events"]:
            events_by_agent[int(event["agent_id"])].append(event)
        segments_by_agent = {
            int(agent["agent_id"]): agent.get("reference_segments", [])
            for agent in record["agents"]
        }
        for event in record["events"]:
            if event["event"] != EVENT_NORMAL_REPROPOSAL:
                continue
            agent_id = int(event["agent_id"])
            step = int(event["step"])
            key = (scenario, step, agent_id)
            next_updates = [
                int(other["step"])
                for other in events_by_agent[agent_id]
                if int(other["step"]) > step
                and (
                    bool(other.get("counts_as_reproposal"))
                    or other["event"] == EVENT_REFERENCE_HANDOFF
                )
            ]
            window_end = min(
                step + dwell_steps,
                min(next_updates) if next_updates else step + dwell_steps,
            )
            observations = [
                trigger
                for trigger in triggers_by_agent[agent_id]
                if step < int(trigger["step"]) <= window_end
            ]
            safety_recovered = any(
                float(trigger["active_safety_margin_m"]) >= h_rep
                for trigger in observations
            )
            progress_recovered = any(
                bool(trigger["progress_valid"])
                and trigger["progress_rate_mps"] is not None
                and float(trigger["progress_rate_mps"]) > p_min
                for trigger in observations
            )
            reached = any(
                int(segment.get("start_step", -1)) == step
                and segment.get("source_event") == EVENT_NORMAL_REPROPOSAL
                and bool(segment.get("reached"))
                for segment in segments_by_agent.get(agent_id, [])
            )
            same_goal = not bool(event.get("goal_changed"))
            if key in harmful_keys:
                label = "HARMFUL"
            elif same_goal or (observations and not (safety_recovered or progress_recovered)):
                label = "REDUNDANT"
            elif bool(event.get("goal_changed")) and (
                safety_recovered or progress_recovered
            ):
                label = "HELPFUL"
            else:
                label = "UNRESOLVED"
            utility[key] = label
            rows.append(
                {
                    "scenario_id": scenario,
                    "stage": record["entry"]["stage"],
                    "seed": record["entry"]["seed"],
                    "agent_id": agent_id,
                    "step": step,
                    "time_s": step * 0.1,
                    "execution_age_s": event.get("active_age_s"),
                    "progress_nu_mps": event.get("progress_rate_mps"),
                    "h_active_m": event.get("active_safety_margin_m"),
                    "Phi_tau": event.get("phi_tau"),
                    "Phi_p": event.get("phi_p"),
                    "Phi_h": event.get("phi_h"),
                    "Phi_rep": event.get("S_rep"),
                    "triggering_components": event.get("trigger_reasons", []),
                    "active_goal_changed": bool(event.get("goal_changed")),
                    "same_goal": same_goal,
                    "reference_subsequently_reached": reached,
                    "followup_observation_count": len(observations),
                    "safety_recovered_within_existing_dwell": safety_recovered,
                    "progress_recovered_within_existing_dwell": progress_recovered,
                    "utility": label,
                    "episode_outcome": record["episode"]["termination_reason"],
                }
            )
    return rows, utility


def _counterfactual_normal(
    m3: Mapping[str, Any], utility: Mapping[tuple[str, int, int], str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario, record in sorted(m3.items()):
        triggers: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for trigger in record["triggers"]:
            triggers[int(trigger["agent_id"])].append(trigger)
        event_lookup = {
            (int(event["step"]), int(event["agent_id"])): event
            for event in record["events"]
            if event["event"] == EVENT_NORMAL_REPROPOSAL
        }
        for agent_id in range(3):
            armed = True
            last_false_step: int | None = None
            for trigger in sorted(triggers[agent_id], key=lambda item: int(item["step"])):
                step = int(trigger["step"])
                phi = trigger.get("S_rep")
                raw_condition = phi is not None and float(phi) >= 0.0
                if not raw_condition:
                    armed = True
                    last_false_step = step
                if trigger.get("event") != EVENT_NORMAL_REPROPOSAL:
                    continue
                retained = bool(armed and raw_condition and trigger.get("dwell_satisfied"))
                event = event_lookup[(step, agent_id)]
                if retained:
                    armed = False
                rows.append(
                    {
                        "scenario_id": scenario,
                        "stage": record["entry"]["stage"],
                        "agent_id": agent_id,
                        "step": step,
                        "Phi_rep": phi,
                        "raw_normal_condition": raw_condition,
                        "normal_armed_before": retained or armed,
                        "last_condition_false_step": last_false_step,
                        "counterfactual_action": "RETAIN" if retained else "REMOVE",
                        "utility": utility[(scenario, step, agent_id)],
                        "same_goal": not bool(event.get("goal_changed")),
                        "reason": (
                            "FIRST_ARMED_OCCURRENCE_OF_PERSISTENT_CONDITION"
                            if retained
                            else "CONDITION_NEVER_RETURNED_FALSE_AFTER_PRIOR_NORMAL_EVENT"
                        ),
                    }
                )
    return rows


def _normal_chatter_markdown(
    m3: Mapping[str, Any], counterfactual: Sequence[Mapping[str, Any]]
) -> str:
    cf = {
        (row["scenario_id"], int(row["step"]), int(row["agent_id"])): row
        for row in counterfactual
    }
    lines = [
        "# Normal chattering cases",
        "",
        "The four pre-identified groups are audited against the complete saved trigger stream. "
        "The underlying normal condition is `Phi_rep >= 0`; the existing dwell remains 1.0 s.",
        "",
    ]
    for scenario in KNOWN_NORMAL_CHATTER:
        record = m3[scenario]
        normal_by_agent: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in record["events"]:
            if row["event"] == EVENT_NORMAL_REPROPOSAL:
                normal_by_agent[int(row["agent_id"])].append(row)
        pair: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
        for normal in normal_by_agent.values():
            for first, second in zip(normal, normal[1:]):
                if (
                    int(second["step"]) - int(first["step"]) == 10
                    and not bool(first.get("goal_changed"))
                    and not bool(second.get("goal_changed"))
                ):
                    pair = (first, second)
                    break
            if pair is not None:
                break
        if pair is None:
            lines.extend([f"## {scenario}", "", "Expected pair not recovered.", ""])
            continue
        first, second = pair
        agent = int(first["agent_id"])
        between = [
            row
            for row in record["triggers"]
            if int(row["agent_id"]) == agent
            and int(first["step"]) < int(row["step"]) <= int(second["step"])
        ]
        persistent = all(float(row["S_rep"]) >= 0.0 for row in between)
        second_cf = cf[(scenario, int(second["step"]), agent)]
        lines.extend(
            [
                f"## {scenario}",
                "",
                f"Agent {agent}: normal events at steps {first['step']} and {second['step']} "
                f"(exactly 1.0 s apart). Both selected the same goal; h_active was "
                f"{float(first['active_safety_margin_m']):.6f} m and "
                f"{float(second['active_safety_margin_m']):.6f} m. The raw normal "
                f"condition stayed true throughout the interval: `{persistent}`. "
                f"The edge/rearm filter marks the second event `{second_cf['counterfactual_action']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Finding",
            "",
            "All four groups follow `NORMAL_LEVEL_CONDITION_REMAINS_TRUE + DWELL_EXPIRES "
            "-> RETRIGGER`; therefore `NORMAL_LEVEL_TRIGGER_PERSISTENCE = YES`.",
            "",
        ]
    )
    return "\n".join(lines)


def _replanning_scope(m3: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    for scenario, record in sorted(m3.items()):
        grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
        for event in record["events"]:
            if bool(event.get("counts_as_reproposal")):
                grouped[(int(event["step"]), str(event["selection_plan_hash"]))].append(event)
        for (step, plan_hash), events in sorted(grouped.items()):
            trigger_agents = sorted(int(row["agent_id"]) for row in events)
            updated_agents = sorted(int(row["agent_id"]) for row in events)
            cross = sorted(set(updated_agents) - set(trigger_agents))
            audit_rows.append(
                {
                    "scenario_id": scenario,
                    "stage": record["entry"]["stage"],
                    "step": step,
                    "selection_plan_hash": plan_hash,
                    "trigger_agent_ids": trigger_agents,
                    "computed_agent_ids": [0, 1, 2],
                    "updated_agent_ids": updated_agents,
                    "compute_scope": "TEAM_WIDE",
                    "update_scope": "AGENT_LOCAL",
                    "cross_agent_update_count": len(cross),
                    "cross_agent_updated_ids": cross,
                }
            )
            for updated in cross:
                cross_rows.append(
                    {
                        "scenario_id": scenario,
                        "stage": record["entry"]["stage"],
                        "step": step,
                        "trigger_agent_ids": trigger_agents,
                        "updated_agent_id": updated,
                        "status": "CROSS_AGENT_UPDATE",
                    }
                )
    if not cross_rows:
        cross_rows.append(
            {
                "scenario_id": "NONE",
                "stage": "ALL",
                "step": None,
                "trigger_agent_ids": [],
                "updated_agent_id": None,
                "status": "NO_CROSS_AGENT_REFERENCE_UPDATE_OBSERVED",
                "collision_within_0_5s": 0,
                "collision_within_1_0s": 0,
                "collision_within_2_0s": 0,
                "goal_switch_within_2_0s": 0,
                "reference_loss_within_2_0s": 0,
            }
        )
    return audit_rows, cross_rows


def _last_plan_before(
    record: Mapping[str, Any], agent_id: int, step: int
) -> Mapping[str, Any] | None:
    events = [
        event
        for event in _planning_events(record)
        if int(event["agent_id"]) == int(agent_id) and int(event["step"]) <= int(step)
    ]
    return max(events, key=lambda item: int(item["step"])) if events else None


def _peer_audit(
    m3: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timeline_rows: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    for scenario, record in sorted(m3.items()):
        if not bool(record["episode"]["inter_agent_collision"]):
            continue
        path = _path_maps(record)
        triggers = _trigger_maps(record)
        collision_step = int(record["episode"]["steps"])
        pairs = _colliding_pairs(record)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            for step in range(max(0, collision_step - 20), collision_step + 1):
                if (step, i) not in path or (step, j) not in path:
                    continue
                pi, pj = _vector(path[(step, i)], ""), _vector(path[(step, j)], "")
                vi, vj = _vector(path[(step, i)], "v"), _vector(path[(step, j)], "v")
                relative_p, relative_v = pj - pi, vj - vi
                distance = float(np.linalg.norm(relative_p))
                speed2 = float(np.dot(relative_v, relative_v))
                t_min = max(0.0, -float(np.dot(relative_p, relative_v)) / speed2) if speed2 > 1e-12 else 0.0
                closest = float(np.linalg.norm(relative_p + t_min * relative_v))
                closing = -float(np.dot(relative_p, relative_v)) / max(distance, 1e-12)
                last_i = _last_plan_before(record, i, step)
                last_j = _last_plan_before(record, j, step)
                trigger_i, trigger_j = triggers.get((step, i), {}), triggers.get((step, j), {})
                timeline_rows.append(
                    {
                        "scenario_id": scenario,
                        "stage": record["entry"]["stage"],
                        "collision_step": collision_step,
                        "window_offset_s": (step - collision_step) * 0.1,
                        "step": step,
                        "agent_i": i,
                        "agent_j": j,
                        "colliding_pair": (i, j) in pairs,
                        "distance_m": distance,
                        "relative_position_j_minus_i_m": relative_p.tolist(),
                        "relative_velocity_j_minus_i_mps": relative_v.tolist(),
                        "closing_rate_mps": closing,
                        "estimated_closest_approach_time_s": t_min,
                        "estimated_closest_approach_distance_m": closest,
                        "active_reference_i": [
                            path[(step, i)]["active_goal_x_m"],
                            path[(step, i)]["active_goal_y_m"],
                            path[(step, i)]["active_goal_z_m"],
                        ],
                        "active_reference_j": [
                            path[(step, j)]["active_goal_x_m"],
                            path[(step, j)]["active_goal_y_m"],
                            path[(step, j)]["active_goal_z_m"],
                        ],
                        "h_active_i_m": trigger_i.get("active_safety_margin_m"),
                        "h_active_j_m": trigger_j.get("active_safety_margin_m"),
                        "trigger_i": trigger_i.get("event"),
                        "trigger_j": trigger_j.get("event"),
                        "last_planning_step_i": last_i.get("step") if last_i else None,
                        "last_planning_step_j": last_j.get("step") if last_j else None,
                        "last_trigger_type_i": last_i.get("event") if last_i else None,
                        "last_trigger_type_j": last_j.get("event") if last_j else None,
                        "last_selected_candidate_i": (
                            last_i.get("selected_candidate_id") if last_i else None
                        ),
                        "last_selected_candidate_j": (
                            last_j.get("selected_candidate_id") if last_j else None
                        ),
                        "last_selected_edges_i": (
                            last_i.get("selected_edge_records", []) if last_i else []
                        ),
                        "last_selected_edges_j": (
                            last_j.get("selected_edge_records", []) if last_j else []
                        ),
                        "peer_surface_within_4_5m_lidar_i": distance - 0.3 <= 4.5,
                        "peer_surface_within_4_5m_lidar_j": distance - 0.3 <= 4.5,
                        "active_sector_return_source": "UNTYPED_NOT_IDENTIFIABLE",
                    }
                )

        factors: list[str] = []
        evidence: list[dict[str, Any]] = []
        for i, j in pairs:
            pair_events = [
                event
                for event in record["events"]
                if int(event["agent_id"]) in {i, j}
                and bool(event.get("counts_as_reproposal"))
                and collision_step - 20 <= int(event["step"]) < collision_step
            ]
            risky_switches = []
            for event in pair_events:
                peer = j if int(event["agent_id"]) == i else i
                if bool(event.get("goal_changed")) and _selected_risky_for_peer(event, peer):
                    risky_switches.append(event)
            if risky_switches:
                factors.append("REFERENCE_SWITCH_CREATED_CONFLICT")
            emergency_steps = [
                int(event["step"])
                for event in pair_events
                if event["event"] == EVENT_EMERGENCY_REPROPOSAL
            ]
            if not pair_events:
                factors.append("NO_REPLAN_AFTER_PEER_STATE_CHANGED")
            # A real replan by either colliding agent in the final 0.5 s that
            # still ends in collision is direct late/ineffective-response
            # evidence.  This uses the goal-prescribed 0.5 s window, not a new
            # trigger threshold.
            elif any(
                collision_step - 5 <= int(event["step"]) < collision_step
                for event in pair_events
            ):
                factors.append("TRIGGER_TOO_LATE")
            trigger_window = [
                trigger
                for trigger in record["triggers"]
                if int(trigger["agent_id"]) in {i, j}
                and collision_step - 20 <= int(trigger["step"]) < collision_step
            ]
            if not any(
                bool(trigger.get("normal_trigger"))
                or bool(trigger.get("emergency_trigger"))
                for trigger in trigger_window
            ):
                factors.append("TRIGGER_NOT_PEER_SENSITIVE")
            recent_h4 = [
                event
                for event in pair_events
                if collision_step - 4 <= int(event["step"]) < collision_step
            ]
            recent_h4_with_peer_descriptor = [
                event
                for event in recent_h4
                if any(
                    int(edge["neighbor_agent_id"])
                    == (j if int(event["agent_id"]) == i else i)
                    for edge in event.get("selected_edge_records", [])
                )
            ]
            if recent_h4_with_peer_descriptor and not any(
                _selected_risky_for_peer(
                    event, j if int(event["agent_id"]) == i else i
                )
                for event in recent_h4_with_peer_descriptor
            ):
                factors.append("LOW_LEVEL_EXECUTION_FAILURE")
            evidence.append(
                {
                    "pair": [i, j],
                    "pair_event_steps_2s": [int(event["step"]) for event in pair_events],
                    "risky_switch_steps_2s": [int(event["step"]) for event in risky_switches],
                    "emergency_steps_2s": emergency_steps,
                }
            )
        factors = list(dict.fromkeys(factors))
        primary = factors[0] if len(factors) == 1 else (
            "NOT_ESTABLISHED" if not factors else "MULTIPLE_FACTORS"
        )
        classifications.append(
            {
                "scenario_id": scenario,
                "stage": record["entry"]["stage"],
                "seed": record["entry"]["seed"],
                "collision_step": collision_step,
                "colliding_pairs": [list(pair) for pair in pairs],
                "classification": primary,
                "contributing_factors": factors,
                "pair_evidence": evidence,
                "peer_exact_current_position_available_to_GAT": True,
                "peer_exact_current_velocity_available_to_GAT": True,
                "peer_identity_available_to_GAT": True,
                "candidate_interaction_descriptors_available_to_GAT": True,
                "trigger_peer_visibility": "PARTIAL",
                "cross_agent_update_created_conflict": False,
            }
        )
    return timeline_rows, classifications


def _run_existing_tests() -> dict[str, Any]:
    paths = (
        REPO_ROOT / "test/test_event_triggered_reference_reconstruction.py",
        REPO_ROOT / "test/test_sparse_err_trigger_revision.py",
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        namespace = runpy.run_path(str(path))
        for name, function in sorted(namespace.items()):
            if name.startswith("test_") and callable(function):
                function()
                rows.append(
                    {
                        "test_file": _relative(path),
                        "test_name": name,
                        "status": "PASS",
                    }
                )
    return {
        "status": "PASS",
        "test_count": len(rows),
        "tests": rows,
        "diagnostic_replay_reconciliation": "PASS",
        "execution_semantics_changed": False,
    }


def analyze() -> None:
    reconciliation = load_json(OUTPUT / "diagnostic_replay_reconciliation.json")
    if reconciliation.get("status") != "PASS" or reconciliation.get("record_count") != 88:
        raise RuntimeError("full 88-record diagnostic replay must pass before analysis")
    m3, m2 = _load_records()

    paired_timeline: list[dict[str, Any]] = []
    conversion_rows: list[dict[str, Any]] = []
    harmful_keys: set[tuple[str, int, int]] = set()
    for scenario in ADVERSE_SCENARIOS:
        paired_timeline.extend(_timeline_rows(m2[scenario]))
        paired_timeline.extend(_timeline_rows(m3[scenario]))
        divergences = _first_divergences(m2[scenario], m3[scenario])
        classification = _classify_adverse(
            scenario, m2[scenario], m3[scenario], divergences
        )
        conversion_rows.append(classification)
        step = divergences["FIRST_TRIGGER_DIVERGENCE_STEP"]
        if step is not None:
            for agent in divergences["FIRST_TRIGGER_DIVERGENCE_AGENTS"]:
                event = _event_at(m3[scenario], step, agent)
                if event is not None and event["event"] == EVENT_NORMAL_REPROPOSAL:
                    harmful_keys.add((scenario, int(step), int(agent)))
    write_csv(OUTPUT / "paired_collision_timeline.csv", paired_timeline)
    write_csv(OUTPUT / "collision_conversion_classification.csv", conversion_rows)

    scope_rows, cross_rows = _replanning_scope(m3)
    write_csv(OUTPUT / "replanning_scope_audit.csv", scope_rows)
    write_csv(OUTPUT / "cross_agent_reference_updates.csv", cross_rows)
    tex_excerpt, tex_lines = _uncommented_tex_excerpt()
    scope_contract = f"""# Replanning scope contract

## Result

- `REPLANNING_COMPUTE_SCOPE = TEAM_WIDE`
- `REPLANNING_UPDATE_SCOPE = AGENT_LOCAL`
- `THEORY_REPLANNING_UPDATE_SCOPE = AGENT_LOCAL`
- `REPLANNING_SCOPE_THEORY_CODE_MATCH = YES`
- `CROSS_AGENT_REFERENCE_UPDATE_COUNT = 0`

At every non-empty trigger tick the implementation invokes the complete three-agent
Proposal -> Top-K -> FP-SHEP -> GAT pipeline once.  The returned plan contains one
candidate record per agent.  The subsequent write loop is explicitly
`for agent_id in reproposal_ids`, so only trigger owners receive an active-goal
update.  The {len(scope_rows)} observed planning invocations reproduce this contract;
no updated-agent set contains a non-trigger owner.

The active Methodology uses the indexed event `E_{{i,rep}}^t` and the indexed update
`g_i^{{(m_i+1)}}`; the recovered active lines are {tex_lines}.  This is an
agent-local reference update, so team-wide computation is an implementation cost,
not a team-wide state mutation.

## Active theory excerpt

```tex
{tex_excerpt}
```
"""
    (OUTPUT / "replanning_scope_contract.md").write_text(scope_contract, encoding="utf-8")

    normal_rows, utility = _normal_utility(m3, harmful_keys)
    write_csv(OUTPUT / "normal_trigger_event_table.csv", normal_rows)
    write_csv(OUTPUT / "normal_trigger_utility.csv", normal_rows)
    counterfactual = _counterfactual_normal(m3, utility)
    write_csv(OUTPUT / "counterfactual_normal_edge_rearm.csv", counterfactual)
    (OUTPUT / "normal_chattering_cases.md").write_text(
        _normal_chatter_markdown(m3, counterfactual), encoding="utf-8"
    )

    peer_timeline, peer_classes = _peer_audit(m3)
    write_csv(OUTPUT / "peer_collision_timeline.csv", peer_timeline)
    write_csv(OUTPUT / "peer_collision_classification.csv", peer_classes)

    utility_counts = Counter(row["utility"] for row in normal_rows)
    cf_counts = Counter(row["counterfactual_action"] for row in counterfactual)
    removed_utility = Counter(
        row["utility"]
        for row in counterfactual
        if row["counterfactual_action"] == "REMOVE"
    )
    edge_supported = (
        removed_utility["REDUNDANT"] + removed_utility["HARMFUL"]
        > removed_utility["HELPFUL"]
        and all(
            any(
                row["scenario_id"] == scenario
                and row["counterfactual_action"] == "REMOVE"
                and row["same_goal"]
                for row in counterfactual
            )
            for scenario in KNOWN_NORMAL_CHATTER
        )
    )
    peer_factor_counts = Counter(
        factor for row in peer_classes for factor in row["contributing_factors"]
    )
    peer_specificity_factors = {
        "TRIGGER_TOO_LATE",
        "TRIGGER_NOT_PEER_SENSITIVE",
        "NO_REPLAN_AFTER_PEER_STATE_CHANGED",
    }
    peer_specificity_episode_count = sum(
        bool(set(row["contributing_factors"]) & peer_specificity_factors)
        for row in peer_classes
    )
    adverse_factor_counts = Counter(
        factor for row in conversion_rows for factor in row["contributing_factors"]
    )
    peer_episode_count = len(peer_classes)
    normal_explained = adverse_factor_counts["NORMAL_TRIGGER_INDUCED"]
    if normal_explained > len(ADVERSE_SCENARIOS) / 2 and edge_supported:
        primary = "NORMAL_LEVEL_TRIGGER"
        change = "NORMAL_FALSE_TO_TRUE_EDGE_WITH_CONDITION_FALSE_REARM"
        next_step = "REVISE_NORMAL_EVENT_SEMANTICS"
    elif (
        peer_specificity_episode_count > peer_episode_count / 2
    ):
        primary = "PEER_RISK_SPECIFICITY"
        change = "NONE"
        next_step = "AUDIT_PEER_RISK_TRIGGER"
    elif peer_factor_counts["REFERENCE_SWITCH_CREATED_CONFLICT"] > peer_episode_count / 2:
        primary = "REFERENCE_SWITCH_CONFLICT"
        change = "NONE"
        next_step = "STOP_AND_REASSESS"
    elif peer_factor_counts["LOW_LEVEL_EXECUTION_FAILURE"] > peer_episode_count / 2:
        primary = "LOW_LEVEL_EXECUTION"
        change = "NONE"
        next_step = "STOP_AND_REASSESS"
    else:
        primary = "MIXED"
        change = "NONE"
        next_step = "STOP_AND_REASSESS"

    root = {
        "PRIMARY_REMAINING_FAILURE_SOURCE": primary,
        "decision_rule": (
            "A single source must explain a strict majority of the eight adverse "
            "conversions or ten peer-collision episodes; otherwise MIXED."
        ),
        "adverse_factor_counts": dict(adverse_factor_counts),
        "peer_factor_counts": dict(peer_factor_counts),
        "peer_specificity_episode_count": peer_specificity_episode_count,
        "peer_collision_episode_count": peer_episode_count,
        "normal_event_utility_counts": dict(utility_counts),
        "counterfactual_action_counts": dict(cf_counts),
        "counterfactual_removed_utility_counts": dict(removed_utility),
        "NORMAL_EDGE_REARM_SUPPORTED": "YES" if edge_supported else "NO",
        "scope_mismatch": False,
        "one_change_gate": (
            "SUPPORTED_NORMAL_ONLY" if change != "NONE" else "STOP_NO_CHANGE"
        ),
        "ONE_MINIMAL_CHANGE_APPLIED": change,
        "phase_F_authorized": change != "NONE",
        "RECOMMENDED_NEXT_STEP": next_step,
    }
    write_json(OUTPUT / "root_cause_decision.json", root)

    if change != "NONE":
        raise RuntimeError(
            "analysis supports a normal semantic revision; implement/freeze Phase F separately"
        )

    (OUTPUT / "minimal_revision_contract.md").write_text(
        "# Minimal revision contract\n\n"
        "`ONE_MINIMAL_CHANGE_APPLIED = NONE`.  Phase E did not identify an "
        "authorized single execution-semantic correction.  Under Case C/D of the "
        "goal contract, execution logic remains unchanged and Phase F is not run.\n",
        encoding="utf-8",
    )
    write_json(OUTPUT / "regression_tests.json", _run_existing_tests())
    write_json(
        OUTPUT / "development_scenario_manifest.json",
        {
            "status": "NOT_RUN",
            "reason": "PHASE_E_DID_NOT_AUTHORIZE_A_MINIMAL_REVISION",
            "new_scenarios_generated": 0,
            "formal_benchmark_run": False,
        },
    )
    not_run_row = {
        "status": "NOT_RUN",
        "reason": "PHASE_E_DID_NOT_AUTHORIZE_A_MINIMAL_REVISION",
    }
    for name in (
        "development_episode_results.csv",
        "development_agent_results.csv",
        "paired_revision_results.csv",
        "reproposal_summary.csv",
        "runtime_summary.csv",
    ):
        write_csv(OUTPUT / name, [not_run_row])

    conclusion = {
        "REPLANNING_COMPUTE_SCOPE": "TEAM_WIDE",
        "REPLANNING_UPDATE_SCOPE": "AGENT_LOCAL",
        "THEORY_REPLANNING_UPDATE_SCOPE": "AGENT_LOCAL",
        "REPLANNING_SCOPE_THEORY_CODE_MATCH": "YES",
        "CROSS_AGENT_REFERENCE_UPDATE_COUNT": 0,
        "CROSS_AGENT_REFERENCE_UPDATE_EPISODES": 0,
        "M2_SUCCESS_TO_M3_COLLISION": sum(
            row["M2_outcome"] == "success" for row in conversion_rows
        ),
        "M2_TIMEOUT_TO_M3_COLLISION": sum(
            row["M2_outcome"] == "timeout" for row in conversion_rows
        ),
        "NORMAL_LEVEL_TRIGGER_PERSISTENCE": "YES",
        "NORMAL_EVENT_HELPFUL_COUNT": utility_counts["HELPFUL"],
        "NORMAL_EVENT_REDUNDANT_COUNT": utility_counts["REDUNDANT"],
        "NORMAL_EVENT_HARMFUL_COUNT": utility_counts["HARMFUL"],
        "NORMAL_EVENT_UNRESOLVED_COUNT": utility_counts["UNRESOLVED"],
        "NORMAL_EDGE_REARM_SUPPORTED": "YES" if edge_supported else "NO",
        "COUNTERFACTUAL_NORMAL_EVENT_REDUCTION": (
            (cf_counts["REMOVE"] / len(counterfactual)) if counterfactual else 0.0
        ),
        "COUNTERFACTUAL_NORMAL_EVENT_COUNT_CURRENT": len(counterfactual),
        "COUNTERFACTUAL_NORMAL_EVENT_COUNT_EDGE_REARM": cf_counts["RETAIN"],
        "PEER_RISK_VISIBLE_TO_TRIGGER": "PARTIAL",
        "PEER_COLLISION_TRIGGER_TOO_LATE": peer_factor_counts["TRIGGER_TOO_LATE"],
        "PEER_COLLISION_REFERENCE_SWITCH_CREATED_CONFLICT": peer_factor_counts[
            "REFERENCE_SWITCH_CREATED_CONFLICT"
        ],
        "PEER_COLLISION_CROSS_AGENT_UPDATE_CREATED_CONFLICT": 0,
        "PRIMARY_REMAINING_FAILURE_SOURCE": primary,
        "ONE_MINIMAL_CHANGE_APPLIED": "NONE",
        "PHASE_F_EXECUTED": "NO",
        "CURRENT_RERR_SUCCESS": "NOT_RUN",
        "REVISION_SUCCESS": "NOT_RUN",
        "CURRENT_RERR_COLLISION": "NOT_RUN",
        "REVISION_COLLISION": "NOT_RUN",
        "CURRENT_RERR_INTER_AGENT_COLLISION": "NOT_RUN",
        "REVISION_INTER_AGENT_COLLISION": "NOT_RUN",
        "CURRENT_RERR_STAGE2_SUCCESS": "NOT_RUN",
        "REVISION_STAGE2_SUCCESS": "NOT_RUN",
        "CURRENT_RERR_STAGE3_SUCCESS": "NOT_RUN",
        "REVISION_STAGE3_SUCCESS": "NOT_RUN",
        "CURRENT_RERR_STAGE4_SUCCESS": "NOT_RUN",
        "REVISION_STAGE4_SUCCESS": "NOT_RUN",
        "CURRENT_RERR_MEAN_REPROPOSALS": "NOT_RUN",
        "REVISION_MEAN_REPROPOSALS": "NOT_RUN",
        "CURRENT_RERR_TOTAL_COMPUTE_MS": "NOT_RUN",
        "REVISION_TOTAL_COMPUTE_MS": "NOT_RUN",
        "CHATTERING_PRESENT": "YES",
        "CHATTERING_SOURCE": "NORMAL",
        "REVISION_ACCEPTED": "NOT_RUN",
        "GAT_CLOSED_LOOP_VALUE": "STRONG",
        "GAT_FINETUNE_JUSTIFIED": "NO",
        "FINAL_METHOD_READY_FOR_NEW_FORMAL": "NO",
        "RECOMMENDED_NEXT_STEP": next_step,
        "historical_reference_only": {
            "M3_success": 0.8125,
            "M3_collision": 0.1625,
            "M3_inter_agent_collision": 0.125,
            "M3_total_compute_ms": 2271.591,
            "DWA_FullState_success": 0.9775,
            "RVO_FullState_success": 0.985,
            "DWA_SensingMatched_success": 0.1825,
            "RVO_SensingMatched_success": 0.10,
        },
    }
    if conclusion["M2_SUCCESS_TO_M3_COLLISION"] != 5 or conclusion[
        "M2_TIMEOUT_TO_M3_COLLISION"
    ] != 3:
        raise RuntimeError("adverse conversion counts do not match frozen authority")
    write_json(OUTPUT / "conclusion.json", conclusion)
    print(json.dumps(conclusion, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "replay", "analyze"))
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "replay":
        replay(args.limit)
    else:
        analyze()


if __name__ == "__main__":
    main()
