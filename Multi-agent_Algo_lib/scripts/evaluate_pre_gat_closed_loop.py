"""Run paired Pre-GAT closed-loop baselines with a frozen SAC-DMP policy.

The deployed methods are Frozen SAC-DMP, Proposal + SAC-DMP, and FP-SHEP +
SAC-DMP.  No GAT forward, optimizer, supervision target, oracle action, safety
override, SDH, or pending-goal handoff is used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Environment.frozen_sac_dmp_execution import freeze_policy  # noqa: E402
from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402
from experiment_config import EXPERIMENT_CONFIG as SINGLE_AGENT_CONFIG  # noqa: E402
from planning.policy_preview import point_to_segment_distance  # noqa: E402
from planning.pre_gat_closed_loop import (  # noqa: E402
    CLOSED_LOOP_SCHEMA_VERSION,
    FORMAL_PREVIEW_HORIZON,
    METHOD_DISPLAY_NAMES,
    METHOD_FP_SHEP,
    METHOD_FROZEN,
    METHOD_ORDER,
    METHOD_PROPOSAL,
    SELECTION_FP_SHEP,
    SELECTION_NO_CANDIDATE_FALLBACK,
    SELECTION_PROPOSAL,
    FPSHEPOnlineScoreSpec,
    FixedPeriodProtocol,
    choose_execution_reference,
    generate_candidate_set,
    goal_switch,
    termination_reason,
    trajectory_metrics,
)
from planning.temporary_reference_diagnosis import (  # noqa: E402
    candidate_safety_diagnostic,
    dmp_switch_diagnostic,
    observation_switch_diagnostic,
)
from runner_sac import build_env as build_single_env  # noqa: E402
from runner_sac import build_model as build_single_model  # noqa: E402
from runner_sac import load_checkpoint  # noqa: E402
from scripts.evaluate_frozen_policy_waypoint_guidance import (  # noqa: E402
    build_active_goal_observations,
    predict_actions_without_postprocessing,
    set_dmp_active_goal_preserve_phase,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    STAGE_SPECS,
    build_single_distribution_multi_config,
    build_stage_scenario,
)
from scripts.evaluate_single_policy_multi_agent import _build_environment  # noqa: E402
from scripts.validate_policy_preview import (  # noqa: E402
    SCENE_ADAPTERS,
    _scene_snapshot,
    build_validation_environment,
)


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "evaluation" / "pre_gat_closed_loop.json"
CRITICAL_SOURCE_PATHS = (
    "Environment/multi_agent_dmp_env.py",
    "Environment/frozen_sac_dmp_execution.py",
    "Controller/dmp_rl.py",
    "Entity/KinematicModel.py",
    "Guidance/reference_point_proposal_demo.py",
    "planning/policy_preview.py",
    "planning/heterogeneous_candidate_graph.py",
    "planning/candidate_supervision.py",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded: dict[str, Any] = {}
            for key in fields:
                value = _jsonable(row.get(key))
                encoded[key] = (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict))
                    else value
                )
            writer.writerow(encoded)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _policy_parameter_sha256(policy: Any) -> str:
    hasher = hashlib.sha256()
    for key, value in sorted(policy.actor.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(str(array.shape).encode("ascii"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _critical_hashes(checkpoint: Path) -> dict[str, str]:
    result = {"checkpoint": _sha256(checkpoint)}
    for relative in CRITICAL_SOURCE_PATHS:
        result[relative] = _sha256(REPO_ROOT / relative)
    return result


def _scenario_hash(snapshot: dict[str, Any]) -> str:
    payload = json.dumps(_jsonable(snapshot), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stage(name: str) -> dict[str, Any]:
    return next(item for item in STAGE_SPECS if item["name"] == name)


def build_closed_loop_environment(
    *,
    config: Any,
    scenario: str,
    seed: int,
    peer_radius: float,
) -> tuple[Any, dict[str, Any]]:
    """Reuse exactly the seven scenario adapters from the label audit."""

    if scenario in SCENE_ADAPTERS:
        return build_validation_environment(
            config=config,
            scene_type=scenario,
            seed=int(seed),
            peer_radius=float(peer_radius),
        )
    if scenario != "narrow_head_on":
        raise ValueError(f"unknown closed-loop scenario: {scenario}")
    stage = _stage("E_head_on_narrow_peer_spheres")
    options = build_stage_scenario(config, stage, seed=int(seed))
    env = _build_environment(
        config,
        observation_mode="peer_spheres",
        peer_radius=float(peer_radius),
        training_distribution=False,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )
    env.reset(seed=int(seed), options=copy.deepcopy(options))
    return env, {
        "scene_type": scenario,
        "source_stage": stage["name"],
        "observation_mode": "peer_spheres",
        "include_boundaries_in_sensor": False,
        "terminate_on_boundary_collision": False,
        "static_obstacle_count": len(options["static_obstacles"]),
        "dynamic_obstacle_count": len(options["dynamic_obstacles"]),
    }


def _new_window_state(
    *,
    timestep: int,
    start_position: np.ndarray,
    terminal_goal: np.ndarray,
    execution_reference: np.ndarray,
    next_replan_step: int,
    reached_tolerance: float,
) -> dict[str, Any]:
    start_position = np.asarray(start_position, dtype=float).copy()
    terminal_goal = np.asarray(terminal_goal, dtype=float).copy()
    execution_reference = np.asarray(execution_reference, dtype=float).copy()
    initial_reference_distance = float(np.linalg.norm(execution_reference - start_position))
    already_reached = initial_reference_distance <= float(reached_tolerance)
    return {
        "window_start_timestep": int(timestep),
        "window_end_timestep": int(timestep),
        "window_start_position": start_position,
        "window_start_terminal_distance": float(np.linalg.norm(terminal_goal - start_position)),
        "window_reference": execution_reference,
        "window_next_replan_step": int(next_replan_step),
        "window_steps": 0,
        "window_min_clearance": float("inf"),
        "window_min_inter_agent_distance": float("inf"),
        "window_collision": False,
        "window_max_execution_deviation": 0.0,
        "reference_reached_early": bool(already_reached),
        "reference_reached_step": int(timestep) if already_reached else None,
        "reference_hold_steps_after_reached": 0,
        "distance_from_reference_during_hold": [],
        "progress_during_hold": [],
        "clearance_during_hold": [],
        "initial_reference_distance": initial_reference_distance,
    }


def _update_window_state(
    state: dict[str, Any],
    *,
    completed_step: int,
    position: np.ndarray,
    min_clearance: float,
    min_inter_agent_distance: float,
    collision: bool,
    reached_tolerance: float,
    terminal_goal: np.ndarray,
) -> None:
    position = np.asarray(position, dtype=float)
    state["window_steps"] += 1
    state["window_end_timestep"] = int(completed_step)
    state["window_min_clearance"] = min(
        float(state["window_min_clearance"]), float(min_clearance)
    )
    state["window_min_inter_agent_distance"] = min(
        float(state["window_min_inter_agent_distance"]),
        float(min_inter_agent_distance),
    )
    state["window_collision"] = bool(state["window_collision"] or collision)
    state["window_max_execution_deviation"] = max(
        float(state["window_max_execution_deviation"]),
        float(
            point_to_segment_distance(
                position,
                state["window_start_position"],
                state["window_reference"],
            )
        ),
    )
    reference_distance = float(np.linalg.norm(state["window_reference"] - position))
    if state["reference_reached_step"] is None and reference_distance <= reached_tolerance:
        state["reference_reached_step"] = int(completed_step)
        state["reference_reached_early"] = bool(
            int(completed_step) < int(state["window_next_replan_step"])
        )
    elif state["reference_reached_step"] is not None:
        state["reference_hold_steps_after_reached"] += 1
        state["distance_from_reference_during_hold"].append(reference_distance)
        terminal_distance = float(
            np.linalg.norm(np.asarray(terminal_goal, dtype=float) - position)
        )
        state["progress_during_hold"].append(
            float(state["window_start_terminal_distance"] - terminal_distance)
        )
        state["clearance_during_hold"].append(float(min_clearance))


def _finalize_window_state(
    event: dict[str, Any],
    state: dict[str, Any],
    *,
    final_position: np.ndarray,
    terminal_goal: np.ndarray,
    end_reason: str,
) -> None:
    terminal_distance = float(
        np.linalg.norm(np.asarray(terminal_goal, dtype=float) - np.asarray(final_position, dtype=float))
    )
    event.update(
        execution_window_start_timestep=int(state["window_start_timestep"]),
        execution_window_end_timestep=int(state["window_end_timestep"]),
        execution_window_steps=int(state["window_steps"]),
        execution_window_task_progress=float(
            state["window_start_terminal_distance"] - terminal_distance
        ),
        execution_window_minimum_clearance=float(state["window_min_clearance"]),
        execution_window_minimum_inter_agent_distance=float(
            state["window_min_inter_agent_distance"]
        ),
        execution_window_collision=bool(state["window_collision"]),
        execution_window_max_deviation=float(state["window_max_execution_deviation"]),
        reference_reached_early=bool(state["reference_reached_early"]),
        reference_reached_step=state["reference_reached_step"],
        reference_hold_steps_after_reached=int(
            state["reference_hold_steps_after_reached"]
        ),
        distance_from_reference_during_hold=list(
            state["distance_from_reference_during_hold"]
        ),
        progress_during_hold=list(state["progress_during_hold"]),
        clearance_during_hold=list(state["clearance_during_hold"]),
        execution_window_end_reason=str(end_reason),
    )


def run_closed_loop_episode(
    *,
    policy: Any,
    multi_config: Any,
    settings: dict[str, Any],
    phase: str,
    scenario: str,
    seed: int,
    method: str,
    m_upper: int,
    episode_index: int = 0,
    diagnostic_step_sink: list[dict[str, Any]] | None = None,
    diagnostic_switch_sink: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Run one method from one deterministic paired initial condition."""

    if method not in METHOD_ORDER:
        raise ValueError(f"unknown method: {method}")
    protocol = FixedPeriodProtocol(
        m_upper=int(m_upper),
        consumer_top_k=int(settings["candidate_semantics"]["K_requested"]),
        goal_switch_epsilon=float(settings["fixed_period_protocol"]["goal_switch_epsilon"]),
    )
    score_spec = FPSHEPOnlineScoreSpec.from_mapping(settings["fp_shep_online_selector"])
    proposal_config = ProposalConfig(**settings.get("proposal_config", {}))
    env, scenario_metadata = build_closed_loop_environment(
        config=multi_config,
        scenario=scenario,
        seed=int(seed),
        peer_radius=float(settings["peer_radius"]),
    )
    pair_id = f"{phase}__{scenario}__seed{int(seed):03d}__episode{int(episode_index):03d}"
    method_episode_id = f"{pair_id}__{method}__M{protocol.m_upper}"
    try:
        initial_snapshot = _scene_snapshot(env)
        initial_hash = _scenario_hash(initial_snapshot)
        terminal_goals = np.asarray(env.goals, dtype=float).copy()
        terminal_goals.setflags(write=False)
        starts = np.asarray(env.starts, dtype=float).copy()
        active_references = terminal_goals.copy()
        previous_references = terminal_goals.copy()
        for agent_index in range(int(env.num_agents)):
            set_dmp_active_goal_preserve_phase(env.dmps[agent_index], active_references[agent_index])

        position_history = [env._positions().copy()]
        velocity_history = [env._velocities().copy()]
        acceleration_history: list[np.ndarray] = []
        reference_history: list[np.ndarray] = []
        replan_mask_history: list[np.ndarray] = []
        min_inter_agent_history: list[float] = [
            float(env._check_collision()["min_inter_agent_distance"])
        ]
        min_clearance_history: list[float] = [
            float(
                np.min(
                    [
                        packet.min_clearance
                        for packet in env.latest_sensor_packets
                        if packet is not None
                    ]
                )
            )
        ]
        events: list[dict[str, Any]] = []
        active_event_indices: list[int | None] = [None] * int(env.num_agents)
        active_window_states: list[dict[str, Any] | None] = [None] * int(env.num_agents)
        upper_cycle_count = 0
        goal_switch_count = 0
        same_goal_retention_count = 0
        goal_jumps: list[float] = []
        fallback_count = 0
        proposal_selection_count = 0
        fp_selection_count = 0
        candidate_generation_times_ms: list[float] = []
        fp_preview_times_ms: list[float] = []
        selector_times_ms: list[float] = []
        full_replan_times_ms: list[float] = []
        inference_times_ms: list[float] = []
        low_level_step_times_ms: list[float] = []
        unsafe_step_count = 0
        d_safe = float(settings["safety"]["d_safe"])
        collision = obstacle_collision = inter_agent_collision = boundary_collision = False
        terminated = truncated = False
        info: dict[str, Any] = {}
        last_applied_accelerations = np.zeros(
            (int(env.num_agents), 3), dtype=float
        )
        episode_started = time.perf_counter_ns()

        while not (terminated or truncated):
            timestep = int(env.steps)
            replan_mask = np.zeros(int(env.num_agents), dtype=bool)
            if method != METHOD_FROZEN and protocol.is_replanning_step(timestep):
                upper_cycle_count += 1
                cycle_started = time.perf_counter_ns()
                for agent_index in range(int(env.num_agents)):
                    if bool(env.success_rewarded_mask[agent_index]):
                        active_references[agent_index] = terminal_goals[agent_index]
                        set_dmp_active_goal_preserve_phase(
                            env.dmps[agent_index], terminal_goals[agent_index]
                        )
                        continue
                    replan_mask[agent_index] = True
                    previous_event_index = active_event_indices[agent_index]
                    if previous_event_index is not None:
                        _finalize_window_state(
                            events[previous_event_index],
                            active_window_states[agent_index],
                            final_position=env.dynamics[agent_index].p,
                            terminal_goal=terminal_goals[agent_index],
                            end_reason="scheduled_replan",
                        )
                    generation_started = time.perf_counter_ns()
                    proposals, proposal_count_before_consumer = generate_candidate_set(
                        env,
                        agent_index,
                        proposal_config,
                        consumer_top_k=protocol.consumer_top_k,
                    )
                    generation_ms = (time.perf_counter_ns() - generation_started) / 1.0e6
                    candidate_generation_times_ms.append(generation_ms)
                    selector_started = time.perf_counter_ns()
                    decision = choose_execution_reference(
                        method=method,
                        terminal_task_goal=terminal_goals[agent_index],
                        proposals=proposals,
                        policy=policy,
                        env=env,
                        agent_index=agent_index,
                        score_spec=score_spec,
                    )
                    selector_ms = (time.perf_counter_ns() - selector_started) / 1.0e6
                    selector_times_ms.append(selector_ms)
                    fp_runtime = float(sum(item.runtime_ms for item in decision.fp_shep_records))
                    if fp_runtime > 0.0:
                        fp_preview_times_ms.append(fp_runtime)
                    selected_reference = decision.execution_reference.copy()
                    previous_reference = previous_references[agent_index].copy()
                    changed, jump = goal_switch(
                        previous_reference,
                        selected_reference,
                        protocol.goal_switch_epsilon,
                    )
                    goal_switch_count += int(changed)
                    same_goal_retention_count += int(not changed)
                    goal_jumps.append(jump)
                    previous_phase = float(env.dmps[agent_index].phase)
                    terminal_before = np.asarray(env.goals, dtype=float).copy()
                    switch_diagnostic = observation_switch_diagnostic(
                        env,
                        agent_index,
                        previous_active_goal=previous_reference,
                        new_active_goal=selected_reference,
                    )
                    switch_diagnostic.update(
                        dmp_switch_diagnostic(
                            env,
                            agent_index,
                            previous_active_goal=previous_reference,
                            new_active_goal=selected_reference,
                            terminal_goal=terminal_goals[agent_index],
                            policy=policy,
                        )
                    )
                    switch_diagnostic.update(
                        candidate_safety_diagnostic(
                            env,
                            agent_index,
                            candidate=selected_reference,
                            boundary_margin=float(
                                settings.get("diagnostics", {}).get(
                                    "boundary_margin_m", 0.4
                                )
                            ),
                            collision_clearance=float(
                                settings.get("diagnostics", {}).get(
                                    "collision_clearance_m", 0.0
                                )
                            ),
                            segment_samples=int(
                                settings.get("diagnostics", {}).get(
                                    "segment_samples", 16
                                )
                            ),
                        )
                    )
                    switch_diagnostic.update(
                        {
                            "phase": phase,
                            "pair_id": pair_id,
                            "method": method,
                            "scenario": scenario,
                            "seed": int(seed),
                            "timestep": timestep,
                            "agent_id": int(agent_index),
                            "current_acceleration": last_applied_accelerations[
                                agent_index
                            ].tolist(),
                            "boundary_validity_used_for_selection": False,
                        }
                    )
                    if diagnostic_switch_sink is not None:
                        diagnostic_switch_sink.append(copy.deepcopy(switch_diagnostic))
                    set_dmp_active_goal_preserve_phase(
                        env.dmps[agent_index], selected_reference
                    )
                    if not np.array_equal(env.goals, terminal_before):
                        raise RuntimeError("temporary reference overwrote terminal task goal")
                    if float(env.dmps[agent_index].phase) != previous_phase:
                        raise RuntimeError("reference switch reset DMP phase")
                    active_references[agent_index] = selected_reference
                    previous_references[agent_index] = selected_reference
                    fallback_count += int(decision.no_candidate_fallback)
                    proposal_selection_count += int(
                        decision.selection_kind == SELECTION_PROPOSAL
                    )
                    fp_selection_count += int(decision.selection_kind == SELECTION_FP_SHEP)

                    proposal_scores = [float(item.score) for item in proposals]
                    fp_records = [item.to_record() for item in decision.fp_shep_records]
                    fp_scores = [float(item.score) for item in decision.fp_shep_records]
                    fp_order = (
                        np.argsort(-np.asarray(fp_scores), kind="mergesort").astype(int).tolist()
                        if fp_scores
                        else []
                    )
                    preview_gap = (
                        float(np.sort(np.asarray(fp_scores))[-1] - np.sort(np.asarray(fp_scores))[-2])
                        if len(fp_scores) >= 2
                        else None
                    )
                    packet = env.latest_sensor_packets[agent_index]
                    pairwise = env._compute_pairwise_distances()
                    other_distances = np.delete(pairwise[agent_index], agent_index)
                    event = {
                        "schema_version": CLOSED_LOOP_SCHEMA_VERSION,
                        "phase": phase,
                        "pair_id": pair_id,
                        "method_episode_id": method_episode_id,
                        "method": method,
                        "method_display_name": METHOD_DISPLAY_NAMES[method],
                        "scenario": scenario,
                        "seed": int(seed),
                        "episode": int(episode_index),
                        "timestep": timestep,
                        "agent_id": agent_index,
                        "M_upper": protocol.m_upper,
                        "replanning_rule_satisfied": protocol.is_replanning_step(timestep),
                        "candidate_count_before_consumer": proposal_count_before_consumer,
                        "candidate_count": len(proposals),
                        "K_requested": protocol.consumer_top_k,
                        "K_t": len(proposals),
                        "candidate_ids": list(range(len(proposals))),
                        "candidate_world_positions": [
                            np.asarray(item.point, dtype=float).tolist() for item in proposals
                        ],
                        "proposal_scores": proposal_scores,
                        "proposal_top1_candidate_id": 0 if proposals else None,
                        "fp_shep_candidate_records": fp_records,
                        "fp_shep_scores": fp_scores,
                        "fp_shep_ranking_candidate_ids": fp_order,
                        "fp_shep_top1_candidate_id": fp_order[0] if fp_order else None,
                        "selected_fp_shep_rank": (
                            fp_order.index(int(decision.selected_candidate_id))
                            if fp_order and decision.selected_candidate_id is not None
                            else None
                        ),
                        "proposal_fp_shep_disagreement": bool(
                            fp_order and int(fp_order[0]) != 0
                        ),
                        "top1_top2_preview_gap": preview_gap,
                        "selected_candidate_id": decision.selected_candidate_id,
                        "selected_candidate_original_index": decision.selected_candidate_original_index,
                        "selected_candidate_world_position": (
                            selected_reference.tolist()
                            if decision.selected_candidate_id is not None
                            else None
                        ),
                        "selection_kind": decision.selection_kind,
                        "no_candidate_fallback": decision.no_candidate_fallback,
                        "fallback_counted_as_null": False,
                        "fallback_counted_as_proposal_selection": False,
                        "fallback_counted_as_fp_shep_selection": False,
                        "previous_execution_reference": previous_reference.tolist(),
                        "new_execution_reference": selected_reference.tolist(),
                        "goal_jump": jump,
                        "selection_changed": changed,
                        "current_terminal_task_goal": terminal_goals[agent_index].tolist(),
                        "terminal_goal_unchanged": bool(
                            np.array_equal(env.goals, terminal_goals)
                        ),
                        "phase_preserved_on_switch": True,
                        "current_minimum_inter_agent_distance": (
                            float(np.min(other_distances)) if other_distances.size else float("inf")
                        ),
                        "current_minimum_clearance": float(packet.min_clearance),
                        "candidate_generation_runtime_ms": generation_ms,
                        "fp_shep_total_preview_runtime_ms": fp_runtime,
                        "candidate_selection_runtime_ms": selector_ms,
                        "selector_score_specification": score_spec.metadata(),
                        "terminal_speed_used_for_online_ranking": False,
                        "supervision_target_used_online": False,
                        "diagnostic_oracle_used_for_selection": False,
                        "GAT_used_for_selection": False,
                        "episode_status_at_selection": "running",
                        "observation_switch_diagnostic": {
                            key: value
                            for key, value in switch_diagnostic.items()
                            if key
                            in {
                                "goal_direction_delta_l2",
                                "normalized_goal_distance_delta",
                                "observation_l2_change",
                                "non_goal_feature_l2_change",
                            }
                        },
                        "closed_loop_commanded_acceleration_jump_mps2": switch_diagnostic[
                            "closed_loop_commanded_acceleration_jump_mps2"
                        ],
                        "zero_action_nominal_acceleration_jump_mps2": switch_diagnostic[
                            "zero_action_nominal_acceleration_jump_mps2"
                        ],
                        "segment_clearance_m": switch_diagnostic["segment_clearance_m"],
                        "segment_validity": switch_diagnostic["segment_validity"],
                        "point_safety": switch_diagnostic["point_safety"],
                        "boundary_validity": switch_diagnostic["boundary_validity"],
                        "boundary_validity_used_for_selection": False,
                    }
                    events.append(event)
                    active_event_indices[agent_index] = len(events) - 1
                    active_window_states[agent_index] = _new_window_state(
                        timestep=timestep,
                        start_position=env.dynamics[agent_index].p,
                        terminal_goal=terminal_goals[agent_index],
                        execution_reference=selected_reference,
                        next_replan_step=timestep + protocol.m_upper,
                        reached_tolerance=float(env.env_config.goal_tolerance),
                    )
                full_replan_times_ms.append(
                    (time.perf_counter_ns() - cycle_started) / 1.0e6
                )

            if not np.array_equal(env.goals, terminal_goals):
                raise RuntimeError("terminal task goal changed during closed-loop episode")
            for agent_index in range(int(env.num_agents)):
                if not np.array_equal(env.dmps[agent_index].goal, active_references[agent_index]):
                    raise RuntimeError("DMP goal differs from tracked execution reference")
            observations = build_active_goal_observations(env, active_references)
            inference_started = time.perf_counter_ns()
            actions = predict_actions_without_postprocessing(
                policy, observations, tuple(env.action_shape)
            )
            inference_times_ms.append(
                (time.perf_counter_ns() - inference_started) / 1.0e6
            )
            reference_history.append(active_references.copy())
            replan_mask_history.append(replan_mask.copy())
            step_started = time.perf_counter_ns()
            _, _, terminated, truncated, info = env.step(actions)
            low_level_step_times_ms.append(
                (time.perf_counter_ns() - step_started) / 1.0e6
            )
            if not np.array_equal(env.goals, terminal_goals):
                raise RuntimeError("environment step changed terminal task goal")
            positions = env._positions().copy()
            velocities = env._velocities().copy()
            accelerations = np.asarray(info["applied_accelerations"], dtype=float).copy()
            last_applied_accelerations = accelerations.copy()
            position_history.append(positions)
            velocity_history.append(velocities)
            acceleration_history.append(accelerations)
            min_inter = float(info["min_inter_agent_distance"])
            min_clearance = float(np.min(np.asarray(info["min_clearances"], dtype=float)))
            min_inter_agent_history.append(min_inter)
            min_clearance_history.append(min_clearance)
            unsafe_step_count += int(min_inter < d_safe)
            collision |= bool(info["collision"])
            obstacle_collision |= bool(np.any(info["obstacle_collision_mask"]))
            inter_agent_collision |= bool(np.any(info["inter_agent_collision_mask"]))
            boundary_collision |= bool(np.any(info["boundary_collision_mask"]))
            for agent_index, event_index in enumerate(active_event_indices):
                agent_clearance = float(np.asarray(info["min_clearances"])[agent_index])
                agent_collision = bool(np.asarray(info["collision_mask"])[agent_index])
                window = active_window_states[agent_index]
                if event_index is not None and window is not None:
                    _update_window_state(
                        window,
                        completed_step=int(env.steps),
                        position=positions[agent_index],
                        min_clearance=agent_clearance,
                        min_inter_agent_distance=min_inter,
                        collision=agent_collision,
                        reached_tolerance=float(env.env_config.goal_tolerance),
                        terminal_goal=terminal_goals[agent_index],
                    )
                if diagnostic_step_sink is not None:
                    controller_info = env.latest_controller_infos[agent_index]
                    reached_step = (
                        window.get("reference_reached_step")
                        if window is not None
                        else None
                    )
                    diagnostic_step_sink.append(
                        {
                            "phase": phase,
                            "pair_id": pair_id,
                            "method": method,
                            "scenario": scenario,
                            "seed": int(seed),
                            "timestep": timestep,
                            "completed_step": int(env.steps),
                            "agent_id": int(agent_index),
                            "position": positions[agent_index].tolist(),
                            "velocity": velocities[agent_index].tolist(),
                            "active_goal": active_references[agent_index].tolist(),
                            "terminal_goal": terminal_goals[agent_index].tolist(),
                            "distance_to_active_goal": float(
                                np.linalg.norm(
                                    active_references[agent_index]
                                    - positions[agent_index]
                                )
                            ),
                            "distance_to_terminal_goal": float(
                                np.linalg.norm(
                                    terminal_goals[agent_index]
                                    - positions[agent_index]
                                )
                            ),
                            "forcing_gate_value": float(
                                controller_info.get("forcing_gate_scalar", 0.0)
                            ),
                            "forcing_norm": float(
                                np.linalg.norm(controller_info.get("forcing", np.zeros(3)))
                            ),
                            "commanded_acceleration": np.asarray(
                                info["commanded_accelerations"][agent_index], dtype=float
                            ).tolist(),
                            "applied_acceleration": np.asarray(
                                info["applied_accelerations"][agent_index], dtype=float
                            ).tolist(),
                            "minimum_clearance": agent_clearance,
                            "minimum_inter_agent_distance": min_inter,
                            "collision": agent_collision,
                            "obstacle_collision": bool(
                                info["obstacle_collision_mask"][agent_index]
                            ),
                            "inter_agent_collision": bool(
                                info["inter_agent_collision_mask"][agent_index]
                            ),
                            "boundary_collision": bool(
                                info["boundary_collision_mask"][agent_index]
                            ),
                            "reference_reached_step": reached_step,
                            "fixed_period_hold_after_reached": bool(
                                reached_step is not None
                                and int(env.steps) > int(reached_step)
                                and not bool(replan_mask[agent_index])
                            ),
                            "terminal_return_active": False,
                        }
                    )

        status = termination_reason(
            success=bool(info.get("success", False)),
            collision=collision,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        for agent_index, event_index in enumerate(active_event_indices):
            if event_index is not None and active_window_states[agent_index] is not None:
                if "execution_window_steps" not in events[event_index]:
                    _finalize_window_state(
                        events[event_index],
                        active_window_states[agent_index],
                        final_position=env.dynamics[agent_index].p,
                        terminal_goal=terminal_goals[agent_index],
                        end_reason=status,
                    )
                events[event_index]["episode_status_at_window_end"] = status
        positions_array = np.stack(position_history)
        velocities_array = np.stack(velocity_history)
        accelerations_array = (
            np.stack(acceleration_history)
            if acceleration_history
            else np.zeros((0, int(env.num_agents), 3), dtype=float)
        )
        metrics = trajectory_metrics(
            positions_array,
            velocities_array,
            accelerations_array,
            dt=float(settings["dt"]),
        )
        success = bool(info.get("success", False))
        agent_events = len(events)
        episode_runtime_ms = (time.perf_counter_ns() - episode_started) / 1.0e6
        episode = {
            "schema_version": CLOSED_LOOP_SCHEMA_VERSION,
            "phase": phase,
            "pair_id": pair_id,
            "method_episode_id": method_episode_id,
            "method": method,
            "method_display_name": METHOD_DISPLAY_NAMES[method],
            "scenario": scenario,
            "seed": int(seed),
            "episode": int(episode_index),
            "M_upper": protocol.m_upper,
            "initial_condition_hash": initial_hash,
            "terminal_task_goals": terminal_goals.tolist(),
            "terminal_task_goals_unchanged": bool(np.array_equal(env.goals, terminal_goals)),
            "team_success": success,
            "success": success,
            "collision": collision,
            "inter_agent_collision": inter_agent_collision,
            "obstacle_collision": obstacle_collision,
            "boundary_collision": boundary_collision,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "timeout": status == "timeout",
            "termination_reason": status,
            "steps": int(env.steps),
            "completion_time_all_s": float(env.steps * settings["dt"]),
            "completion_time_success_s": float(env.steps * settings["dt"]) if success else None,
            "path_lengths_per_agent": metrics["path_lengths"].tolist(),
            "path_length_team_mean_m": metrics["path_length_team_mean"],
            "path_length_team_sum_m": metrics["path_length_team_sum"],
            "path_length_success_team_mean_m": metrics["path_length_team_mean"] if success else None,
            "minimum_inter_agent_distance_m": float(np.min(min_inter_agent_history)),
            "minimum_obstacle_clearance_m": float(np.min(min_clearance_history)),
            "minimum_obstacle_clearance_source": "real_environment_sensor_packet",
            "minimum_obstacle_clearance_is_exact_global_distance": False,
            "unsafe_step_count": int(unsafe_step_count),
            "unsafe_duration_s": float(unsafe_step_count * settings["dt"]),
            "d_safe_m": d_safe,
            "upper_replanning_cycle_count": int(upper_cycle_count),
            "upper_replanning_agent_event_count": int(agent_events),
            "goal_switch_count": int(goal_switch_count),
            "same_goal_retention_count": int(same_goal_retention_count),
            "goal_switch_rate": float(goal_switch_count / max(1, agent_events)),
            "mean_goal_jump_m": float(np.mean(goal_jumps)) if goal_jumps else 0.0,
            "maximum_goal_jump_m": float(np.max(goal_jumps)) if goal_jumps else 0.0,
            "velocity_variation_team_mean": metrics["velocity_variation_team_mean"],
            "acceleration_variation_team_mean": metrics["acceleration_variation_team_mean"],
            "trajectory_smoothness_team_mean": metrics["trajectory_smoothness_team_mean"],
            "trajectory_smoothness_success_team_mean": (
                metrics["trajectory_smoothness_team_mean"] if success else None
            ),
            "jerk_rms_team_mean": metrics["jerk_rms_team_mean"],
            "smoothness_definition": metrics["smoothness_definition"],
            "K_t_zero_count": int(fallback_count),
            "no_candidate_fallback_count": int(fallback_count),
            "no_candidate_fallback_rate": float(fallback_count / max(1, agent_events)),
            "proposal_selection_count": int(proposal_selection_count),
            "fp_shep_selection_count": int(fp_selection_count),
            "fallback_counted_as_selection": False,
            "reference_reached_early_count": int(
                sum(bool(row.get("reference_reached_early", False)) for row in events)
            ),
            "reference_hold_steps_after_reached_sum": int(
                sum(int(row.get("reference_hold_steps_after_reached", 0)) for row in events)
            ),
            "proposal_fp_shep_disagreement_count": int(
                sum(bool(row.get("proposal_fp_shep_disagreement", False)) for row in events)
            ),
            "proposal_fp_shep_disagreement_rate": float(
                sum(bool(row.get("proposal_fp_shep_disagreement", False)) for row in events)
                / max(1, sum(bool(row.get("fp_shep_scores")) for row in events))
            ),
            "candidate_generation_runtime_mean_ms": (
                float(np.mean(candidate_generation_times_ms)) if candidate_generation_times_ms else 0.0
            ),
            "fp_shep_preview_runtime_mean_ms": (
                float(np.mean(fp_preview_times_ms)) if fp_preview_times_ms else 0.0
            ),
            "candidate_selection_runtime_mean_ms": (
                float(np.mean(selector_times_ms)) if selector_times_ms else 0.0
            ),
            "upper_replanning_runtime_mean_ms": (
                float(np.mean(full_replan_times_ms)) if full_replan_times_ms else 0.0
            ),
            "frozen_sac_inference_runtime_mean_ms": float(np.mean(inference_times_ms)),
            "low_level_step_runtime_mean_ms": float(np.mean(low_level_step_times_ms)),
            "episode_runtime_ms": float(episode_runtime_ms),
            "formal_preview_horizon": FORMAL_PREVIEW_HORIZON,
            "selector_score_specification": score_spec.metadata(),
            "fixed_period_protocol": protocol.metadata(),
            "null_candidate_included": False,
            "supervision_target_used_online": False,
            "diagnostic_oracle_used_for_selection": False,
            "GAT_used_for_selection": False,
            "GAT_optimizer_created": False,
            "SAC_training_performed": False,
            **scenario_metadata,
        }
        trajectory = {
            "positions": positions_array,
            "velocities": velocities_array,
            "accelerations": accelerations_array,
            "execution_references": np.stack(reference_history),
            "replanning_mask": np.stack(replan_mask_history),
            "minimum_inter_agent_distance": np.asarray(min_inter_agent_history, dtype=float),
            "minimum_obstacle_clearance": np.asarray(min_clearance_history, dtype=float),
            "starts": starts,
            "terminal_goals": terminal_goals.copy(),
            "initial_condition_hash": np.asarray(initial_hash),
            "scene_snapshot_json": np.asarray(
                json.dumps(_jsonable(initial_snapshot), ensure_ascii=False)
            ),
        }
        return episode, events, trajectory
    finally:
        env.close()


def _save_episode_artifacts(
    output_dir: Path,
    episode: dict[str, Any],
    events: list[dict[str, Any]],
    trajectory: dict[str, np.ndarray],
) -> None:
    identifier = str(episode["method_episode_id"])
    _write_json(output_dir / "per_episode" / f"{identifier}.json", episode)
    _write_json(output_dir / "per_replanning_event" / f"{identifier}.json", events)
    trajectory_path = output_dir / "trajectories" / f"{identifier}.npz"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trajectory_path, **trajectory)


def _development_key(rows: list[dict[str, Any]], m_upper: int) -> tuple[float, ...]:
    selected = [row for row in rows if int(row["M_upper"]) == int(m_upper)]
    if not selected:
        raise ValueError(f"no development rows for M_upper={m_upper}")
    return (
        -float(np.mean([float(row["success"]) for row in selected])),
        float(np.mean([float(row["collision"]) for row in selected])),
        float(np.mean([float(row["goal_switch_rate"]) for row in selected])),
        float(np.mean([float(row["trajectory_smoothness_team_mean"]) for row in selected])),
        float(np.mean([float(row["upper_replanning_runtime_mean_ms"]) for row in selected])),
        float(m_upper),
    )


def choose_development_m_upper(
    rows: list[dict[str, Any]], values: Iterable[int]
) -> tuple[int, list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    for value in [int(item) for item in values]:
        selected = [row for row in rows if int(row["M_upper"]) == value]
        successful = [row for row in selected if bool(row["success"])]

        def mean(field: str, source: list[dict[str, Any]] = selected) -> float | None:
            values_for_field = [
                float(row[field])
                for row in source
                if row.get(field) is not None and np.isfinite(float(row[field]))
            ]
            return float(np.mean(values_for_field)) if values_for_field else None

        summaries.append({
            "M_upper": value,
            "episode_count": len(selected),
            "successful_episode_count": len(successful),
            "success_rate": mean("success"),
            "collision_rate": mean("collision"),
            "inter_agent_collision_rate": mean("inter_agent_collision"),
            "obstacle_collision_rate": mean("obstacle_collision"),
            "path_length_success_team_mean_m": mean(
                "path_length_success_team_mean_m", successful
            ),
            "completion_time_success_s": mean("completion_time_success_s", successful),
            "goal_switch_count_mean": mean("goal_switch_count"),
            "goal_switch_rate_mean": mean("goal_switch_rate"),
            "mean_goal_jump_m": mean("mean_goal_jump_m"),
            "trajectory_smoothness_mean": mean("trajectory_smoothness_team_mean"),
            "minimum_inter_agent_distance_mean_m": mean(
                "minimum_inter_agent_distance_m"
            ),
            "candidate_generation_runtime_mean_ms": mean(
                "candidate_generation_runtime_mean_ms"
            ),
            "fp_shep_preview_runtime_mean_ms": mean(
                "fp_shep_preview_runtime_mean_ms"
            ),
            "upper_replanning_runtime_mean_ms": mean(
                "upper_replanning_runtime_mean_ms"
            ),
            "selection_key": list(_development_key(rows, value)),
        })
    chosen = min([int(value) for value in values], key=lambda value: _development_key(rows, value))
    for row in summaries:
        row["selected_as_development_default"] = int(row["M_upper"]) == chosen
        row["selection_uses_formal_seeds"] = False
    return chosen, summaries


def run_phase(
    *,
    policy: Any,
    multi_config: Any,
    settings: dict[str, Any],
    output_dir: Path,
    phase: str,
    seeds: Iterable[int],
    methods: Iterable[str],
    m_upper_values: Iterable[int],
    scenarios: Iterable[str],
    max_episodes: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    jobs = [
        (scenario, int(seed), method, int(m_upper))
        for scenario in scenarios
        for seed in seeds
        for m_upper in m_upper_values
        for method in methods
    ]
    if max_episodes is not None:
        jobs = jobs[: int(max_episodes)]
    for index, (scenario, seed, method, m_upper) in enumerate(jobs, start=1):
        episode, event_rows, trajectory = run_closed_loop_episode(
            policy=policy,
            multi_config=multi_config,
            settings=settings,
            phase=phase,
            scenario=scenario,
            seed=seed,
            method=method,
            m_upper=m_upper,
        )
        _save_episode_artifacts(output_dir, episode, event_rows, trajectory)
        episodes.append(episode)
        events.extend(event_rows)
        print(
            f"[{phase} {index}/{len(jobs)}] {scenario} seed={seed} "
            f"method={method} M={m_upper}: {episode['termination_reason']}",
            flush=True,
        )
        _write_csv(output_dir / "per_episode" / f"{phase}_episodes.csv", episodes)
        _write_csv(output_dir / "per_replanning_event" / f"{phase}_events.csv", events)
    return episodes, events


def run_experiment(
    settings: dict[str, Any],
    output_dir: Path,
    *,
    phases: tuple[str, ...] = ("smoke", "development", "formal"),
    max_episodes_per_phase: int | None = None,
    run_analysis: bool = True,
) -> dict[str, Any]:
    checkpoint = Path(settings["checkpoint"])
    checkpoint = checkpoint if checkpoint.is_absolute() else REPO_ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if settings.get("deterministic_policy") is not True:
        raise ValueError("closed-loop baseline requires deterministic_policy=true")
    smoke = set(int(item) for item in settings["seeds"]["smoke"])
    development = set(int(item) for item in settings["seeds"]["development"])
    formal = set(int(item) for item in settings["seeds"]["formal"])
    if development & formal:
        raise ValueError("development and formal seeds must be disjoint")
    methods = tuple(settings["methods"])
    if methods != METHOD_ORDER:
        raise ValueError("method ordering/naming differs from the fixed baseline protocol")
    multi_config = build_single_distribution_multi_config(
        num_agents=int(settings["num_agents"]),
        max_steps=int(settings["max_steps"]),
    )
    if not np.isclose(float(multi_config.time_step), float(settings["dt"])):
        raise ValueError("configured dt differs from checkpoint-aligned environment")
    if not np.isclose(
        float(multi_config.inter_agent_safe_distance),
        float(settings["safety"]["d_safe"]),
    ):
        raise ValueError("d_safe must reuse the current environment safety distance")

    output_dir.mkdir(parents=True, exist_ok=False)
    for name in (
        "summary", "per_episode", "per_replanning_event", "trajectories",
        "switching_diagnostics", "upper_period_sensitivity", "representative_cases",
        "runtime", "paper_ready",
    ):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    reference_env = build_single_env(config=SINGLE_AGENT_CONFIG, action_guidance_enabled=False)
    policy = build_single_model(reference_env, config=SINGLE_AGENT_CONFIG, verbose=0)
    load_checkpoint(policy, checkpoint)
    freeze_policy(policy)
    parameter_hash_before = _policy_parameter_sha256(policy)
    critical_before = _critical_hashes(checkpoint)
    resolved = copy.deepcopy(settings)
    resolved.update({
        "created_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": critical_before["checkpoint"],
        "policy_parameter_sha256_before": parameter_hash_before,
        "output_dir": str(output_dir.resolve()),
        "phases_requested": list(phases),
        "selector_score_specification_resolved": FPSHEPOnlineScoreSpec.from_mapping(
            settings["fp_shep_online_selector"]
        ).metadata(),
        "GAT_training_started": False,
    })
    _write_json(output_dir / "config.json", resolved)
    all_episode_rows: list[dict[str, Any]] = []
    all_event_rows: list[dict[str, Any]] = []
    development_m = settings.get("formal_M_upper")
    started = time.perf_counter()
    try:
        if "smoke" in phases:
            rows, event_rows = run_phase(
                policy=policy,
                multi_config=multi_config,
                settings=settings,
                output_dir=output_dir,
                phase="smoke",
                seeds=sorted(smoke),
                methods=methods,
                m_upper_values=[int(settings["development_M_upper_values"][0])],
                scenarios=settings["scenario_types"][:1],
                max_episodes=max_episodes_per_phase,
            )
            all_episode_rows.extend(rows)
            all_event_rows.extend(event_rows)
        if "development" in phases:
            rows, event_rows = run_phase(
                policy=policy,
                multi_config=multi_config,
                settings=settings,
                output_dir=output_dir,
                phase="development",
                seeds=sorted(development),
                methods=(METHOD_PROPOSAL, METHOD_FP_SHEP),
                m_upper_values=settings["development_M_upper_values"],
                scenarios=settings["scenario_types"],
                max_episodes=max_episodes_per_phase,
            )
            all_episode_rows.extend(rows)
            all_event_rows.extend(event_rows)
            if max_episodes_per_phase is None:
                development_m, sensitivity = choose_development_m_upper(
                    rows, settings["development_M_upper_values"]
                )
                _write_csv(
                    output_dir / "upper_period_sensitivity" / "m_upper_sensitivity.csv",
                    sensitivity,
                )
                _write_json(
                    output_dir / "upper_period_sensitivity" / "m_upper_selection.json",
                    {
                        "selected_M_upper": development_m,
                        "selection_rule": settings["development_selection_rule"],
                        "development_seeds": sorted(development),
                        "formal_seeds_used": False,
                        "claim": "development_default_not_global_optimum",
                    },
                )
        if "formal" in phases:
            if development_m is None:
                raise RuntimeError(
                    "formal evaluation requires a configured or fully evaluated development M_upper"
                )
            rows, event_rows = run_phase(
                policy=policy,
                multi_config=multi_config,
                settings=settings,
                output_dir=output_dir,
                phase="formal",
                seeds=sorted(formal),
                methods=methods,
                m_upper_values=[int(development_m)],
                scenarios=settings["scenario_types"],
                max_episodes=max_episodes_per_phase,
            )
            all_episode_rows.extend(rows)
            all_event_rows.extend(event_rows)
    finally:
        reference_env.close()

    parameter_hash_after = _policy_parameter_sha256(policy)
    critical_after = _critical_hashes(checkpoint)
    if parameter_hash_before != parameter_hash_after:
        raise RuntimeError("closed-loop evaluation modified frozen policy parameters")
    if critical_before != critical_after:
        raise RuntimeError("closed-loop evaluation modified critical execution semantics")
    _write_csv(output_dir / "per_episode" / "all_episode_records.csv", all_episode_rows)
    _write_csv(
        output_dir / "per_replanning_event" / "all_replanning_events.csv",
        all_event_rows,
    )
    generation_summary = {
        "schema_version": CLOSED_LOOP_SCHEMA_VERSION,
        "elapsed_seconds": time.perf_counter() - started,
        "phase_episode_counts": {
            phase: sum(row["phase"] == phase for row in all_episode_rows)
            for phase in ("smoke", "development", "formal")
        },
        "replanning_event_count": len(all_event_rows),
        "formal_M_upper": development_m,
        "checkpoint_sha256_before": critical_before["checkpoint"],
        "checkpoint_sha256_after": critical_after["checkpoint"],
        "checkpoint_unchanged": critical_before["checkpoint"] == critical_after["checkpoint"],
        "policy_parameter_sha256_before": parameter_hash_before,
        "policy_parameter_sha256_after": parameter_hash_after,
        "policy_parameters_unchanged": parameter_hash_before == parameter_hash_after,
        "critical_source_hashes_before": critical_before,
        "critical_source_hashes_after": critical_after,
        "critical_execution_semantics_unchanged": critical_before == critical_after,
        "GAT_training_started": False,
        "GAT_forward_used_for_selection": False,
        "supervision_target_used_online": False,
        "diagnostic_oracle_used_for_selection": False,
    }
    _write_json(output_dir / "summary" / "generation_summary.json", generation_summary)
    if run_analysis and any(row["phase"] == "formal" for row in all_episode_rows):
        from scripts.analyze_pre_gat_closed_loop import analyze_run
        from scripts.validate_paper_ready_artifacts import validate_run

        analyze_run(output_dir)
        validation = validate_run(
            output_dir,
            promote=bool(settings["paper_ready"].get("promote_after_validation", True)),
        )
        if not validation["validation_passed"]:
            raise RuntimeError(
                "paper-ready validation failed: " + "; ".join(validation["failures"])
            )
    return generation_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("smoke", "development", "formal"),
        default=["smoke", "development", "formal"],
    )
    parser.add_argument("--formal-m-upper", type=int, default=None)
    parser.add_argument("--max-episodes-per-phase", type=int, default=None)
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


def main() -> Path:
    args = _parse_args()
    settings = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if args.formal_m_upper is not None:
        settings["formal_M_upper"] = int(args.formal_m_upper)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        base = Path(settings["output_dir"])
        base = base if base.is_absolute() else REPO_ROOT / base
        output_dir = base / timestamp
    else:
        output_dir = args.output_dir.resolve()
    run_experiment(
        settings,
        output_dir,
        phases=tuple(args.phases),
        max_episodes_per_phase=args.max_episodes_per_phase,
        run_analysis=not args.skip_analysis,
    )
    print(f"Artifacts written to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
