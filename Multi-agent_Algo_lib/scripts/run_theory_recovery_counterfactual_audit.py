from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as _pandas  # noqa: F401
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for _path in (REPO_ROOT, ALGO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402
from planning.event_triggered_reference_reconstruction import (  # noqa: E402
    ACTIVE_GOAL_REFERENCE,
    ACTIVE_GOAL_TERMINAL,
    ERRConfig,
    EVENT_EMERGENCY_REPROPOSAL,
    EVENT_NORMAL_REPROPOSAL,
    EVENT_REFERENCE_HANDOFF,
    EventTriggeredReferenceSupervisor,
    active_direction_safety_margin,
)
from planning.final_four_stage_benchmark import WORKSPACE_BOUNDS  # noqa: E402
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.run_final_four_stage_benchmark import (  # noqa: E402
    ManifestEnvironmentBuilder,
    proposed_eval_config,
    build_eval_config,
)


SOURCE_ROOT = REPO_ROOT / "artifacts" / "final_four_stage_benchmark" / "20260818_202620"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "theory_aligned_final_recovery"
    / "20260818_232900"
)
PREFREEZE_NAME = "counterfactual_prefreeze_v2.json"
RECORD_DIRECTORY = "records_v2"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prefreeze(output_dir: Path) -> None:
    phase = output_dir / "phase_c_theoretical_trigger_audit"
    if any((phase / RECORD_DIRECTORY).rglob("*.json")):
        raise RuntimeError("cannot freeze counterfactual audit after records exist")
    err_source = load_json(REPO_ROOT / "configs/evaluation/gat_v1_err_development.json")
    payload = {
        "freeze_time": datetime.now().astimezone().isoformat(),
        "paper_authority": "active uncommented ERR Methodology in hire-rl-body.tex",
        "trajectory_mode": "read_only_non_interventional_trigger_replay",
        "counterfactual_commit_semantics": (
            "on a theoretical reproposal, reset age/progress using the unchanged observed "
            "active goal; do not alter the frozen trajectory or fabricate a selected goal"
        ),
        "err_parameters": err_source["err"],
        "source_manifest_sha256": file_hash(SOURCE_ROOT / "scenario_manifest.json"),
        "source_results_sha256": file_hash(SOURCE_ROOT / "formal_episode_results.csv"),
        "script_sha256": file_hash(
            REPO_ROOT
            / "Multi-agent_Algo_lib"
            / "scripts"
            / "run_theory_recovery_counterfactual_audit.py"
        ),
        "observability_rubric_frozen_before_results": {
            "STRONG": "stage_3_4 failure detection >=0.75 and median lead >=0.5 s",
            "MODERATE": "stage_3_4 failure detection >=0.50 and median lead >=0.3 s",
            "WEAK": "at least one stage_3_4 failure detected before outcome",
            "NO": "no stage_3_4 failure detected before outcome",
        },
    }
    write_json(phase / PREFREEZE_NAME, payload)
    print(f"COUNTERFACTUAL_PREFREEZE={phase / PREFREEZE_NAME}", flush=True)


def verify_prefreeze(output_dir: Path) -> Mapping[str, Any]:
    phase = output_dir / "phase_c_theoretical_trigger_audit"
    frozen = load_json(phase / PREFREEZE_NAME)
    current_script = file_hash(
        REPO_ROOT
        / "Multi-agent_Algo_lib"
        / "scripts"
        / "run_theory_recovery_counterfactual_audit.py"
    )
    if current_script != frozen["script_sha256"]:
        raise RuntimeError("counterfactual audit script changed after freeze")
    if file_hash(SOURCE_ROOT / "scenario_manifest.json") != frozen["source_manifest_sha256"]:
        raise RuntimeError("source scenario manifest changed")
    if file_hash(SOURCE_ROOT / "formal_episode_results.csv") != frozen["source_results_sha256"]:
        raise RuntimeError("source episode results changed")
    return frozen


def _build_runtime(
    manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[ManifestEnvironmentBuilder, Any, Mapping[str, Any]]:
    execution = config["execution"]
    base = build_single_distribution_multi_config(
        num_agents=int(execution["num_agents"]), max_steps=int(execution["max_steps"])
    )
    multi_config = replace(
        base,
        workspace_bounds=WORKSPACE_BOUNDS,
        randomize_start_goal=False,
        start_position_bounds=((-0.4, -2.0, -0.9), (0.2, 2.0, 0.9)),
        goal_position_bounds=((7.0, -2.0, -0.9), (11.0, 2.0, 0.9)),
        min_start_goal_distance=5.5,
    )
    selected = load_json(SOURCE_ROOT / "engineering_search" / "selected_configs.json")
    eval_config = proposed_eval_config(build_eval_config(config, manifest), selected["gat_v1"])
    return ManifestEnvironmentBuilder(manifest), multi_config, eval_config


def _source_record(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    return load_json(
        SOURCE_ROOT
        / "formal_records"
        / entry["stage"]
        / entry["scenario_id"]
        / "gat_v1.json"
    )


def _set_frame(
    env: Any,
    *,
    frame: int,
    positions: np.ndarray,
    velocities: np.ndarray,
    dynamic_positions: np.ndarray,
    dt: float,
) -> None:
    for agent_id in range(int(env.num_agents)):
        env.dynamics[agent_id].p = np.asarray(positions[frame, agent_id], dtype=float).copy()
        env.dynamics[agent_id].v = np.asarray(velocities[frame, agent_id], dtype=float).copy()
        env.dynamics[agent_id].state = np.concatenate(
            [env.dynamics[agent_id].p, env.dynamics[agent_id].v]
        )
    for obstacle_id, obstacle in enumerate(env.dynamic_obstacles):
        obstacle.center = np.asarray(dynamic_positions[frame, obstacle_id], dtype=float).copy()
        if frame == 0:
            velocity = np.asarray(obstacle.velocity, dtype=float)
        else:
            velocity = (
                np.asarray(dynamic_positions[frame, obstacle_id], dtype=float)
                - np.asarray(dynamic_positions[frame - 1, obstacle_id], dtype=float)
            ) / float(dt)
        obstacle.velocity = velocity.copy()
    for agent_id in range(int(env.num_agents)):
        env.latest_sensor_packets[agent_id] = env.sensors[agent_id].sense(
            env.dynamics[agent_id].p,
            env.dynamics[agent_id].v,
            env.goals[agent_id],
            env._sensor_static_obstacles(),
            env._sensor_dynamic_obstacles(agent_id),
        )
    env.steps = int(frame)


def audit_episode(
    *,
    entry: Mapping[str, Any],
    builder: ManifestEnvironmentBuilder,
    multi_config: Any,
    eval_config: Mapping[str, Any],
    err_config: ERRConfig,
) -> dict[str, Any]:
    source_record = _source_record(entry)
    trajectory_path = SOURCE_ROOT / str(source_record["trajectory_path"])
    if file_hash(trajectory_path) != source_record["trajectory_sha256"]:
        raise RuntimeError("source trajectory hash mismatch")
    with np.load(trajectory_path) as frozen:
        positions = np.asarray(frozen["positions"], dtype=float)
        velocities = np.asarray(frozen["velocities"], dtype=float)
        temporary_references = np.asarray(frozen["temporary_references"], dtype=float)
        dynamic_positions = np.asarray(frozen["dynamic_obstacle_positions"], dtype=float)
        terminal_goals = np.asarray(frozen["goals"], dtype=float)
    env, _ = builder(
        config=multi_config,
        scenario=entry["scenario_id"],
        seed=int(entry["seed"]),
        peer_radius=float(eval_config["peer_radius"]),
    )
    try:
        selected = [bool(row["reference_selected"]) for row in source_record["agents"]]
        active_goals = np.asarray(
            [temporary_references[i] if selected[i] else terminal_goals[i] for i in range(3)],
            dtype=float,
        )
        goal_types = [
            ACTIVE_GOAL_REFERENCE if value else ACTIVE_GOAL_TERMINAL for value in selected
        ]
        supervisor = EventTriggeredReferenceSupervisor(
            err_config,
            terminal_goals=terminal_goals,
            active_goals=active_goals,
            active_goal_types=goal_types,
            initial_positions=positions[0],
        )
        proposal_config = ProposalConfig(**dict(eval_config["proposal_config"]))
        events: list[dict[str, Any]] = []
        handoff_steps: list[int | None] = [None, None, None]
        outcome_step = int(source_record["episode"]["steps"])
        for step in range(outcome_step):
            _set_frame(
                env,
                frame=step,
                positions=positions,
                velocities=velocities,
                dynamic_positions=dynamic_positions,
                dt=float(eval_config["dt"]),
            )
            for agent_id, state in enumerate(supervisor.states):
                safety = active_direction_safety_margin(
                    env, agent_id, state.active_goal, proposal_config
                )
                decision = supervisor.evaluate(
                    agent_id,
                    current_step=step,
                    position=positions[step, agent_id],
                    active_safety_margin_m=safety.value_m,
                )
                if decision.event == EVENT_REFERENCE_HANDOFF:
                    if handoff_steps[agent_id] is None:
                        handoff_steps[agent_id] = step
                    supervisor.handoff_to_terminal(
                        agent_id,
                        current_step=step,
                        position=positions[step, agent_id],
                    )
                    continue
                if decision.event not in {
                    EVENT_NORMAL_REPROPOSAL,
                    EVENT_EMERGENCY_REPROPOSAL,
                }:
                    continue
                event_index = len(events)
                events.append(
                    {
                        "stage": entry["stage"],
                        "family": entry["family"],
                        "scenario_id": entry["scenario_id"],
                        "seed": int(entry["seed"]),
                        "agent_id": agent_id,
                        "event_index": event_index,
                        "theory_trigger_step": step,
                        "theory_trigger_time_s": step * float(eval_config["dt"]),
                        "trigger_type": (
                            "EMERGENCY"
                            if decision.event == EVENT_EMERGENCY_REPROPOSAL
                            else "NORMAL"
                        ),
                        "trigger_reasons": "|".join(decision.trigger_reasons),
                        "T_age_s": decision.active_age_s,
                        "d_active_m": decision.active_goal_distance_m,
                        "nu_mps": decision.progress_rate_mps,
                        "nu_valid": decision.progress_valid,
                        "h_active_m": decision.active_safety_margin_m,
                        "phi_T": decision.phi_tau,
                        "phi_nu": decision.phi_p,
                        "phi_h": decision.phi_h,
                        "Phi_rep": decision.S_rep,
                        "E_normal": decision.normal_trigger,
                        "E_emergency": decision.emergency_trigger,
                        "E_rep": True,
                        "before_collision": bool(
                            source_record["episode"]["collision"] and step < outcome_step
                        ),
                        "before_timeout": bool(
                            source_record["episode"]["timeout"] and step < outcome_step
                        ),
                        "before_reference_reach": bool(
                            state.active_goal_type == ACTIVE_GOAL_REFERENCE
                        ),
                        "after_terminal_handoff": bool(
                            state.active_goal_type == ACTIVE_GOAL_TERMINAL
                            and handoff_steps[agent_id] is not None
                        ),
                        "episode_team_success": bool(source_record["episode"]["team_success"]),
                        "episode_termination_reason": source_record["episode"][
                            "termination_reason"
                        ],
                        "outcome_step": outcome_step,
                        "lead_to_outcome_s": (outcome_step - step)
                        * float(eval_config["dt"]),
                        "replay_commit": "SAME_OBSERVED_ACTIVE_GOAL",
                    }
                )
                supervisor.update_goal(
                    agent_id,
                    new_goal=state.active_goal.copy(),
                    new_goal_type=state.active_goal_type,
                    current_step=step,
                    position=positions[step, agent_id],
                    event=decision.event,
                )
        if not events:
            events.append(
                {
                    "stage": entry["stage"],
                    "family": entry["family"],
                    "scenario_id": entry["scenario_id"],
                    "seed": int(entry["seed"]),
                    "agent_id": None,
                    "event_index": None,
                    "theory_trigger_step": None,
                    "theory_trigger_time_s": None,
                    "trigger_type": "NO_TRIGGER",
                    "E_rep": False,
                    "episode_team_success": bool(source_record["episode"]["team_success"]),
                    "episode_termination_reason": source_record["episode"][
                        "termination_reason"
                    ],
                    "outcome_step": outcome_step,
                }
            )
        actual_events = [row for row in events if row["trigger_type"] != "NO_TRIGGER"]
        unique_steps = sorted({int(row["theory_trigger_step"]) for row in actual_events})
        first_step = min(unique_steps) if unique_steps else None
        summary = {
            "stage": entry["stage"],
            "family": entry["family"],
            "scenario_id": entry["scenario_id"],
            "seed": int(entry["seed"]),
            "team_success": bool(source_record["episode"]["team_success"]),
            "collision": bool(source_record["episode"]["collision"]),
            "obstacle_collision": bool(source_record["episode"]["obstacle_collision"]),
            "inter_agent_collision": bool(
                source_record["episode"]["inter_agent_collision"]
            ),
            "timeout": bool(source_record["episode"]["timeout"]),
            "termination_reason": source_record["episode"]["termination_reason"],
            "outcome_step": outcome_step,
            "theoretical_reproposal_count": len(actual_events),
            "theoretical_upper_event_count": len(unique_steps),
            "normal_reproposal_count": sum(
                row["trigger_type"] == "NORMAL" for row in actual_events
            ),
            "emergency_reproposal_count": sum(
                row["trigger_type"] == "EMERGENCY" for row in actual_events
            ),
            "first_theoretical_trigger_step": first_step,
            "first_theoretical_trigger_time_s": (
                first_step * float(eval_config["dt"]) if first_step is not None else None
            ),
            "detected_before_outcome": first_step is not None and first_step < outcome_step,
            "collision_warning_lead_s": (
                (outcome_step - first_step) * float(eval_config["dt"])
                if bool(source_record["episode"]["collision"]) and first_step is not None
                else None
            ),
            "timeout_warning_lead_s": (
                (outcome_step - first_step) * float(eval_config["dt"])
                if bool(source_record["episode"]["timeout"]) and first_step is not None
                else None
            ),
            "handoff_steps_reconstructed": handoff_steps,
            "source_trajectory_sha256": source_record["trajectory_sha256"],
            "source_trajectory_read_only": True,
        }
        return {"events": events, "summary": summary}
    finally:
        env.close()


def _record_path(phase: Path, entry: Mapping[str, Any]) -> Path:
    return phase / RECORD_DIRECTORY / entry["stage"] / f"{entry['scenario_id']}.json"


def run(
    output_dir: Path,
    limit: int | None = None,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
) -> None:
    frozen = verify_prefreeze(output_dir)
    phase = output_dir / "phase_c_theoretical_trigger_audit"
    manifest = load_json(SOURCE_ROOT / "scenario_manifest.json")
    config = load_json(SOURCE_ROOT / "config.json")
    builder, multi_config, eval_config = _build_runtime(manifest, config)
    err_config = ERRConfig.from_mapping(
        {"dt": eval_config["dt"], **frozen["err_parameters"]}
    )
    if not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("shard_index must be in [0, shard_count)")
    entries = [
        entry
        for global_index, entry in enumerate(manifest["entries"])
        if global_index % int(shard_count) == int(shard_index)
    ]
    if limit is not None:
        entries = entries[: int(limit)]
    for index, entry in enumerate(entries, start=1):
        path = _record_path(phase, entry)
        if path.exists():
            continue
        write_json(
            path,
            audit_episode(
                entry=entry,
                builder=builder,
                multi_config=multi_config,
                eval_config=eval_config,
                err_config=err_config,
            ),
        )
        if index % 20 == 0 or index == len(entries):
            print(
                f"[counterfactual shard {shard_index}/{shard_count} "
                f"{index}/{len(entries)}] {entry['scenario_id']}",
                flush=True,
            )
    if limit is None and int(shard_count) == 1:
        analyze(output_dir)


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"median": None, "p25": None, "p75": None, "p90": None}
    return {
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def analyze(output_dir: Path) -> None:
    verify_prefreeze(output_dir)
    phase = output_dir / "phase_c_theoretical_trigger_audit"
    paths = sorted((phase / RECORD_DIRECTORY).rglob("*.json"))
    if len(paths) != 400:
        raise RuntimeError(f"expected 400 counterfactual records, found {len(paths)}")
    events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        events.extend(payload["events"])
        row = dict(payload["summary"])
        row["handoff_steps_reconstructed"] = json.dumps(row["handoff_steps_reconstructed"])
        summaries.append(row)
    write_csv(output_dir / "counterfactual_event_timeline.csv", events)
    write_csv(phase / "counterfactual_event_timeline.csv", events)
    write_csv(output_dir / "temporal_observability.csv", summaries)
    write_csv(phase / "temporal_observability.csv", summaries)
    collisions = [row for row in summaries if row["collision"]]
    collision_leads = [
        float(row["collision_warning_lead_s"])
        for row in collisions
        if row["collision_warning_lead_s"] is not None
    ]
    complex_failures = [
        row
        for row in summaries
        if row["stage"] in {"stage_3", "stage_4"} and not row["team_success"]
    ]
    complex_detected = [row for row in complex_failures if row["detected_before_outcome"]]
    complex_leads = [
        float(
            row["collision_warning_lead_s"]
            if row["collision_warning_lead_s"] is not None
            else row["timeout_warning_lead_s"]
        )
        for row in complex_detected
    ]
    detection_rate = len(complex_detected) / len(complex_failures) if complex_failures else 0.0
    median_complex_lead = float(np.median(complex_leads)) if complex_leads else None
    if detection_rate >= 0.75 and median_complex_lead is not None and median_complex_lead >= 0.5:
        gate = "STRONG"
    elif detection_rate >= 0.50 and median_complex_lead is not None and median_complex_lead >= 0.3:
        gate = "MODERATE"
    elif complex_detected:
        gate = "WEAK"
    else:
        gate = "NO"
    successes = [row for row in summaries if row["team_success"]]
    success_counts = [float(row["theoretical_reproposal_count"]) for row in successes]
    gate_payload = {
        "THEORY_TRIGGER_OBSERVABILITY": gate,
        "stage_3_4_failure_count": len(complex_failures),
        "stage_3_4_detected_before_outcome_count": len(complex_detected),
        "stage_3_4_detection_rate": detection_rate,
        "stage_3_4_median_warning_lead_s": median_complex_lead,
        "collision_count": len(collisions),
        "collision_detected_count": len(collision_leads),
        "collision_detected_rate": len(collision_leads) / len(collisions) if collisions else None,
        "collision_lead_ge_0p3_count": sum(value >= 0.3 for value in collision_leads),
        "collision_lead_ge_0p5_count": sum(value >= 0.5 for value in collision_leads),
        "collision_lead_ge_1p0_count": sum(value >= 1.0 for value in collision_leads),
        "collision_lead_ge_2p0_count": sum(value >= 2.0 for value in collision_leads),
        "collision_lead_quantiles_s": _quantiles(collision_leads),
        "successful_episode_count": len(successes),
        "success_mean_theoretical_replans": float(np.mean(success_counts)),
        "success_median_theoretical_replans": float(np.median(success_counts)),
        "success_p90_theoretical_replans": float(np.quantile(success_counts, 0.9)),
        "read_only_source_trajectory_count": len(summaries),
        "non_interventional_replay": True,
    }
    write_json(phase / "theory_observability_gate.json", gate_payload)
    print(json.dumps(gate_payload, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prefreeze", "run", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if args.phase == "prefreeze":
        prefreeze(output)
    elif args.phase == "run":
        run(
            output,
            args.limit,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    else:
        analyze(output)


if __name__ == "__main__":
    main()
