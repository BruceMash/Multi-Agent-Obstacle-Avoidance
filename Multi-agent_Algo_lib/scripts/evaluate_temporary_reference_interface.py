"""Evaluate temporary-reference execution protocols without changing SAC-DMP.

Stage A/B isolates execution-interface effects with Proposal top-1.  Stage C is
deliberately gated and is not run by this entry point until the development
artifacts show that a boundary-free protocol is usable.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Environment.frozen_sac_dmp_execution import freeze_policy  # noqa: E402
from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402
from experiment_config import EXPERIMENT_CONFIG as SINGLE_AGENT_CONFIG  # noqa: E402
from planning.pre_gat_closed_loop import (  # noqa: E402
    METHOD_FROZEN,
    METHOD_PROPOSAL,
    FPSHEPOnlineScoreSpec,
    choose_execution_reference,
    generate_candidate_set,
    termination_reason,
    trajectory_metrics,
)
from planning.temporary_reference_diagnosis import (  # noqa: E402
    PROTOCOL_DISPLAY_NAMES,
    PROTOCOL_EXISTING_BOUNDARY_FREE,
    PROTOCOL_FIXED_PERIOD,
    PROTOCOL_ONE_SHOT,
    PROTOCOL_TERMINAL,
    OneShotReferenceState,
    candidate_safety_diagnostic,
    dmp_switch_diagnostic,
    failure_attribution,
    observation_switch_diagnostic,
)
from runner_sac import build_env as build_single_env  # noqa: E402
from runner_sac import build_model as build_single_model  # noqa: E402
from runner_sac import load_checkpoint  # noqa: E402
from scripts.evaluate_frozen_policy_waypoint_guidance import (  # noqa: E402
    build_active_goal_observations,
    predict_actions_without_postprocessing,
    run_waypoint_episode,
    set_dmp_active_goal_preserve_phase,
)
from scripts.evaluate_pre_gat_closed_loop import (  # noqa: E402
    _critical_hashes,
    _jsonable,
    _policy_parameter_sha256,
    _scenario_hash,
    _scene_snapshot,
    build_closed_loop_environment,
    run_closed_loop_episode,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)


DEFAULT_CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "evaluation"
    / "temporary_reference_interface_diagnosis.json"
)
PRE_GAT_CONFIG_PATH = REPO_ROOT / "configs" / "evaluation" / "pre_gat_closed_loop.json"
SCHEMA_VERSION = "temporary_reference_interface_diagnosis_v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
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
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(row.get(key)), ensure_ascii=False)
                        if isinstance(row.get(key), (list, tuple, dict, np.ndarray))
                        else _jsonable(row.get(key))
                    )
                    for key in fields
                }
            )


def _protocol_pair_id(scenario: str, seed: int, protocol: str) -> str:
    return f"development__{scenario}__seed{int(seed):03d}__{protocol}"


def _runtime_pre_gat_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    base = json.loads(PRE_GAT_CONFIG_PATH.read_text(encoding="utf-8"))
    base["num_agents"] = int(settings["num_agents"])
    base["max_steps"] = int(settings["max_steps"])
    base["dt"] = float(settings["dt"])
    base["peer_radius"] = float(settings["peer_radius"])
    base["candidate_semantics"] = copy.deepcopy(settings["candidate_semantics"])
    base["proposal_config"] = copy.deepcopy(settings["proposal_config"])
    base["fixed_period_protocol"]["reference_reached_tolerance_source"] = (
        "MultiAgentEnvConfig.goal_tolerance"
    )
    base["diagnostics"] = {
        "boundary_margin_m": float(
            settings["existing_waypoint"]["boundary_margin_m"]
        ),
        "collision_clearance_m": float(
            settings["existing_waypoint"]["collision_clearance_m"]
        ),
        "segment_samples": int(settings["existing_waypoint"]["segment_samples"]),
    }
    return base


def _load_policy(settings: Mapping[str, Any], multi_config: Any) -> tuple[Any, Path]:
    checkpoint = Path(settings["checkpoint"])
    checkpoint = checkpoint if checkpoint.is_absolute() else REPO_ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    reference_env = build_single_env(
        config=SINGLE_AGENT_CONFIG, action_guidance_enabled=False
    )
    policy = build_single_model(reference_env, config=SINGLE_AGENT_CONFIG, verbose=0)
    load_checkpoint(policy, checkpoint)
    freeze_policy(policy)
    return policy, checkpoint.resolve()


def _trajectory_from_waypoint_trace(
    trace: Sequence[Mapping[str, Any]], num_agents: int
) -> dict[str, np.ndarray]:
    if not trace:
        return {
            "positions": np.zeros((0, num_agents, 3), dtype=float),
            "velocities": np.zeros((0, num_agents, 3), dtype=float),
            "accelerations": np.zeros((0, num_agents, 3), dtype=float),
            "execution_references": np.zeros((0, num_agents, 3), dtype=float),
        }
    by_step: dict[int, list[Mapping[str, Any]]] = {}
    for row in trace:
        by_step.setdefault(int(row["timestep"]), []).append(row)
    steps = sorted(by_step)
    first = sorted(by_step[steps[0]], key=lambda row: int(row["agent_id"]))
    positions = [np.asarray([row["position_before"] for row in first], dtype=float)]
    velocities = [np.asarray([row["velocity_before"] for row in first], dtype=float)]
    accelerations: list[np.ndarray] = []
    references: list[np.ndarray] = []
    for step in steps:
        rows = sorted(by_step[step], key=lambda row: int(row["agent_id"]))
        if len(rows) != int(num_agents):
            raise RuntimeError("waypoint trace is missing an agent row")
        positions.append(np.asarray([row["position_after"] for row in rows], dtype=float))
        velocities.append(np.asarray([row["velocity_after"] for row in rows], dtype=float))
        accelerations.append(
            np.asarray([row["applied_acceleration"] for row in rows], dtype=float)
        )
        references.append(np.asarray([row["active_goal"] for row in rows], dtype=float))
    return {
        "positions": np.stack(positions),
        "velocities": np.stack(velocities),
        "accelerations": np.stack(accelerations),
        "execution_references": np.stack(references),
    }


def _normalize_fixed_events(
    events: Sequence[Mapping[str, Any]], episode_steps: int
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in events:
        end_reason = str(row.get("execution_window_end_reason", "terminal_episode"))
        normalized.append(
            {
                **dict(row),
                "reference_created_step": int(row["timestep"]),
                "reference_activated_step": int(row["timestep"]),
                "reference_reached": row.get("reference_reached_step") is not None,
                "reference_reached_time_s": (
                    0.1 * float(row["reference_reached_step"])
                    if row.get("reference_reached_step") is not None
                    else None
                ),
                "reference_released_step": int(
                    row.get("execution_window_end_timestep", episode_steps)
                ),
                "release_reason": (
                    "fixed_period_replace"
                    if end_reason == "scheduled_replan"
                    else end_reason
                ),
            }
        )
    return normalized


def _normalize_waypoint_lifecycle(
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    episode_steps: int,
    dt: float,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    by_agent: dict[int, list[Mapping[str, Any]]] = {}
    for row in reference_rows:
        by_agent.setdefault(int(row["agent_id"]), []).append(row)
    for agent_id, rows in by_agent.items():
        rows = sorted(rows, key=lambda row: int(row["timestep"]))
        for index, row in enumerate(rows):
            next_row = rows[index + 1] if index + 1 < len(rows) else None
            next_reason = str(next_row["reason"]) if next_row is not None else "terminal_episode"
            reached = next_reason == "reached"
            released_step = (
                int(next_row["timestep"]) if next_row is not None else int(episode_steps)
            )
            normalized.append(
                {
                    **dict(row),
                    "agent_id": agent_id,
                    "reference_created_step": int(row["timestep"]),
                    "reference_activated_step": int(row["timestep"]),
                    "reference_reached": reached,
                    "reference_reached_step": released_step if reached else None,
                    "reference_reached_time_s": released_step * float(dt) if reached else None,
                    "reference_released_step": released_step,
                    "release_reason": next_reason,
                }
            )
    return normalized


def _augment_step_identity(
    rows: list[dict[str, Any]], *, protocol: str, pair_id: str
) -> None:
    for row in rows:
        row["schema_version"] = SCHEMA_VERSION
        row["protocol"] = protocol
        row["protocol_display_name"] = PROTOCOL_DISPLAY_NAMES[protocol]
        row["pair_id"] = pair_id


def run_pre_gat_protocol(
    *,
    policy: Any,
    multi_config: Any,
    settings: Mapping[str, Any],
    scenario: str,
    seed: int,
    protocol: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    method = METHOD_FROZEN if protocol == PROTOCOL_TERMINAL else METHOD_PROPOSAL
    step_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    runtime_settings = _runtime_pre_gat_settings(settings)
    episode, raw_events, trajectory = run_closed_loop_episode(
        policy=policy,
        multi_config=multi_config,
        settings=runtime_settings,
        phase="development",
        scenario=scenario,
        seed=int(seed),
        method=method,
        m_upper=8,
        diagnostic_step_sink=step_rows,
        diagnostic_switch_sink=switch_rows,
    )
    pair_id = _protocol_pair_id(scenario, seed, protocol)
    episode.update(
        {
            "schema_version": SCHEMA_VERSION,
            "pair_id": pair_id,
            "protocol": protocol,
            "protocol_display_name": PROTOCOL_DISPLAY_NAMES[protocol],
            "selector": "none" if protocol == PROTOCOL_TERMINAL else "Proposal top-1",
            "reference_reached_tolerance_m": (
                None
                if protocol == PROTOCOL_TERMINAL
                else float(multi_config.goal_tolerance)
            ),
            "reference_reached_tolerance_source": (
                "not_applicable"
                if protocol == PROTOCOL_TERMINAL
                else "MultiAgentEnvConfig.goal_tolerance"
            ),
            "workspace_boundary_filter_enabled": False,
            "boundary_validity_diagnostic_only": True,
        }
    )
    events = (
        []
        if protocol == PROTOCOL_TERMINAL
        else _normalize_fixed_events(raw_events, int(episode["steps"]))
    )
    for row in events:
        row["protocol"] = protocol
        row["protocol_display_name"] = PROTOCOL_DISPLAY_NAMES[protocol]
        row["pair_id"] = pair_id
    _augment_step_identity(step_rows, protocol=protocol, pair_id=pair_id)
    for row in switch_rows:
        row["protocol"] = protocol
        row["protocol_display_name"] = PROTOCOL_DISPLAY_NAMES[protocol]
        row["pair_id"] = pair_id
    return episode, events, step_rows, trajectory, switch_rows


def run_one_shot_episode(
    *,
    policy: Any,
    multi_config: Any,
    settings: Mapping[str, Any],
    scenario: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    protocol = PROTOCOL_ONE_SHOT
    pair_id = _protocol_pair_id(scenario, seed, protocol)
    env, scenario_metadata = build_closed_loop_environment(
        config=multi_config,
        scenario=scenario,
        seed=int(seed),
        peer_radius=float(settings["peer_radius"]),
    )
    proposal_config = ProposalConfig(**dict(settings["proposal_config"]))
    reached_tolerance = float(
        settings["protocol_metadata"][protocol]["reached_tolerance_m"]
    )
    boundary_margin = float(settings["existing_waypoint"]["boundary_margin_m"])
    collision_clearance = float(
        settings["existing_waypoint"]["collision_clearance_m"]
    )
    segment_samples = int(settings["existing_waypoint"]["segment_samples"])
    states = [OneShotReferenceState() for _ in range(int(env.num_agents))]
    events: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    try:
        initial_snapshot = _scene_snapshot(env)
        initial_hash = _scenario_hash(initial_snapshot)
        terminal_goals = np.asarray(env.goals, dtype=float).copy()
        starts = np.asarray(env.starts, dtype=float).copy()
        active_goals = terminal_goals.copy()
        fallback_count = 0
        candidate_generation_ms: list[float] = []
        selector_ms: list[float] = []
        goal_jumps: list[float] = []
        tracking_path = np.zeros(int(env.num_agents), dtype=float)
        reference_start_positions: list[np.ndarray | None] = [None] * int(env.num_agents)
        for agent_index in range(int(env.num_agents)):
            generation_started = time.perf_counter_ns()
            proposals, count_before = generate_candidate_set(
                env,
                agent_index,
                proposal_config,
                consumer_top_k=int(settings["candidate_semantics"]["K_requested"]),
            )
            candidate_generation_ms.append(
                (time.perf_counter_ns() - generation_started) / 1.0e6
            )
            selector_started = time.perf_counter_ns()
            decision = choose_execution_reference(
                method=METHOD_PROPOSAL,
                terminal_task_goal=terminal_goals[agent_index],
                proposals=proposals,
            )
            selector_ms.append((time.perf_counter_ns() - selector_started) / 1.0e6)
            fallback_count += int(decision.no_candidate_fallback)
            if decision.selected_candidate_id is None:
                set_dmp_active_goal_preserve_phase(
                    env.dmps[agent_index], terminal_goals[agent_index]
                )
                continue
            selected = decision.execution_reference.copy()
            previous = np.asarray(env.dmps[agent_index].goal, dtype=float).copy()
            diagnostic = observation_switch_diagnostic(
                env,
                agent_index,
                previous_active_goal=previous,
                new_active_goal=selected,
            )
            diagnostic.update(
                dmp_switch_diagnostic(
                    env,
                    agent_index,
                    previous_active_goal=previous,
                    new_active_goal=selected,
                    terminal_goal=terminal_goals[agent_index],
                    policy=policy,
                )
            )
            diagnostic.update(
                candidate_safety_diagnostic(
                    env,
                    agent_index,
                    candidate=selected,
                    boundary_margin=boundary_margin,
                    collision_clearance=collision_clearance,
                    segment_samples=segment_samples,
                )
            )
            diagnostic.update(
                {
                    "protocol": protocol,
                    "pair_id": pair_id,
                    "scenario": scenario,
                    "seed": int(seed),
                    "timestep": 0,
                    "agent_id": int(agent_index),
                    "switch_reason": "one_shot_activation",
                    "boundary_validity_used_for_selection": False,
                }
            )
            switch_rows.append(diagnostic)
            states[agent_index].activate(selected, timestep=0)
            active_goals[agent_index] = selected
            reference_start_positions[agent_index] = starts[agent_index].copy()
            goal_jumps.append(float(np.linalg.norm(selected - previous)))
            set_dmp_active_goal_preserve_phase(env.dmps[agent_index], selected)
            events.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol": protocol,
                    "protocol_display_name": PROTOCOL_DISPLAY_NAMES[protocol],
                    "pair_id": pair_id,
                    "scenario": scenario,
                    "seed": int(seed),
                    "agent_id": int(agent_index),
                    "reference_created_step": 0,
                    "reference_activated_step": 0,
                    "reference_point": selected.tolist(),
                    "proposal_rank": int(decision.selected_candidate_id),
                    "proposal_count_before_consumer": int(count_before),
                    "K_t": int(len(proposals)),
                    "reference_reached": False,
                    "reference_reached_step": None,
                    "reference_reached_time_s": None,
                    "distance_to_reference_at_switch_m": float(
                        np.linalg.norm(selected - starts[agent_index])
                    ),
                    "reference_released_step": None,
                    "release_reason": None,
                    "goal_jump_m": float(np.linalg.norm(selected - previous)),
                    "segment_validity": diagnostic["segment_validity"],
                    "segment_clearance_m": diagnostic["segment_clearance_m"],
                    "point_safety": diagnostic["point_safety"],
                    "boundary_validity": diagnostic["boundary_validity"],
                    "boundary_validity_used_for_selection": False,
                    "closed_loop_commanded_acceleration_jump_mps2": diagnostic[
                        "closed_loop_commanded_acceleration_jump_mps2"
                    ],
                }
            )

        position_history = [env._positions().copy()]
        velocity_history = [env._velocities().copy()]
        acceleration_history: list[np.ndarray] = []
        reference_history: list[np.ndarray] = []
        min_inter_history = [float(env._check_collision()["min_inter_agent_distance"])]
        min_clearance_history = [
            float(min(packet.min_clearance for packet in env.latest_sensor_packets))
        ]
        ever_success = np.zeros(int(env.num_agents), dtype=bool)
        collision = obstacle_collision = inter_agent_collision = boundary_collision = False
        terminated = truncated = False
        info: dict[str, Any] = {}
        inference_times_ms: list[float] = []
        low_level_times_ms: list[float] = []
        episode_started = time.perf_counter_ns()
        while not (terminated or truncated):
            timestep = int(env.steps)
            observations = build_active_goal_observations(env, active_goals)
            inference_started = time.perf_counter_ns()
            actions = predict_actions_without_postprocessing(
                policy, observations, tuple(env.action_shape)
            )
            inference_times_ms.append(
                (time.perf_counter_ns() - inference_started) / 1.0e6
            )
            positions_before = env._positions().copy()
            tracking_before = np.asarray(
                [
                    state.temporary_reference is not None
                    and not state.returned_to_terminal
                    for state in states
                ],
                dtype=bool,
            )
            reference_history.append(active_goals.copy())
            step_started = time.perf_counter_ns()
            _, _, terminated, truncated, info = env.step(actions)
            low_level_times_ms.append((time.perf_counter_ns() - step_started) / 1.0e6)
            positions = env._positions().copy()
            velocities = env._velocities().copy()
            accelerations = np.asarray(info["applied_accelerations"], dtype=float)
            tracking_path[tracking_before] += np.linalg.norm(
                positions[tracking_before] - positions_before[tracking_before], axis=1
            )
            position_history.append(positions)
            velocity_history.append(velocities)
            acceleration_history.append(accelerations.copy())
            min_inter = float(info["min_inter_agent_distance"])
            min_inter_history.append(min_inter)
            min_clearance_history.append(float(np.min(info["min_clearances"])))
            ever_success |= np.asarray(info["success_mask"], dtype=bool)
            collision |= bool(info["collision"])
            obstacle_collision |= bool(np.any(info["obstacle_collision_mask"]))
            inter_agent_collision |= bool(np.any(info["inter_agent_collision_mask"]))
            boundary_collision |= bool(np.any(info["boundary_collision_mask"]))

            for agent_index, state in enumerate(states):
                active_used = active_goals[agent_index].copy()
                controller_info = env.latest_controller_infos[agent_index]
                event = next(
                    (row for row in events if int(row["agent_id"]) == agent_index),
                    None,
                )
                step_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "protocol": protocol,
                        "protocol_display_name": PROTOCOL_DISPLAY_NAMES[protocol],
                        "pair_id": pair_id,
                        "scenario": scenario,
                        "seed": int(seed),
                        "timestep": timestep,
                        "completed_step": int(env.steps),
                        "agent_id": int(agent_index),
                        "position": positions[agent_index].tolist(),
                        "velocity": velocities[agent_index].tolist(),
                        "active_goal": active_used.tolist(),
                        "terminal_goal": terminal_goals[agent_index].tolist(),
                        "distance_to_active_goal": float(
                            np.linalg.norm(active_used - positions[agent_index])
                        ),
                        "distance_to_terminal_goal": float(
                            np.linalg.norm(terminal_goals[agent_index] - positions[agent_index])
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
                        "applied_acceleration": accelerations[agent_index].tolist(),
                        "minimum_clearance": float(info["min_clearances"][agent_index]),
                        "minimum_inter_agent_distance": min_inter,
                        "collision": bool(info["collision_mask"][agent_index]),
                        "obstacle_collision": bool(
                            info["obstacle_collision_mask"][agent_index]
                        ),
                        "inter_agent_collision": bool(
                            info["inter_agent_collision_mask"][agent_index]
                        ),
                        "boundary_collision": bool(
                            info["boundary_collision_mask"][agent_index]
                        ),
                        "reference_reached_step": (
                            event.get("reference_reached_step") if event else None
                        ),
                        "fixed_period_hold_after_reached": False,
                        "terminal_return_active": bool(state.returned_to_terminal),
                    }
                )

            for agent_index, state in enumerate(states):
                if terminated or truncated:
                    continue
                if state.temporary_reference is None or state.returned_to_terminal:
                    continue
                next_goal, returned = state.update_reached(
                    position=positions[agent_index],
                    terminal_goal=terminal_goals[agent_index],
                    completed_step=int(env.steps),
                    reached_tolerance=reached_tolerance,
                )
                if not returned:
                    continue
                previous = active_goals[agent_index].copy()
                diagnostic = observation_switch_diagnostic(
                    env,
                    agent_index,
                    previous_active_goal=previous,
                    new_active_goal=next_goal,
                )
                diagnostic.update(
                    dmp_switch_diagnostic(
                        env,
                        agent_index,
                        previous_active_goal=previous,
                        new_active_goal=next_goal,
                        terminal_goal=terminal_goals[agent_index],
                        policy=policy,
                    )
                )
                diagnostic.update(
                    {
                        "protocol": protocol,
                        "pair_id": pair_id,
                        "scenario": scenario,
                        "seed": int(seed),
                        "timestep": int(env.steps),
                        "agent_id": int(agent_index),
                        "switch_reason": "reference_reached_terminal_return",
                        "current_acceleration": accelerations[agent_index].tolist(),
                        "boundary_validity_used_for_selection": False,
                    }
                )
                switch_rows.append(diagnostic)
                goal_jumps.append(float(np.linalg.norm(next_goal - previous)))
                active_goals[agent_index] = next_goal
                set_dmp_active_goal_preserve_phase(env.dmps[agent_index], next_goal)
                event = next(row for row in events if int(row["agent_id"]) == agent_index)
                event.update(
                    {
                        "reference_reached": True,
                        "reference_reached_step": int(env.steps),
                        "reference_reached_time_s": float(env.steps * settings["dt"]),
                        "distance_to_reference_at_switch_m": float(
                            np.linalg.norm(previous - positions[agent_index])
                        ),
                        "reference_released_step": int(env.steps),
                        "release_reason": "reached",
                        "steps_tracking_reference": int(env.steps),
                    }
                )

        status = termination_reason(
            success=bool(info.get("success", False)),
            collision=collision,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        for event in events:
            if event["reference_released_step"] is None:
                event["reference_released_step"] = int(env.steps)
                event["release_reason"] = status
            event["steps_after_return_to_terminal_goal"] = (
                int(env.steps - event["reference_released_step"])
                if event["reference_reached"]
                else 0
            )
            event["path_to_reference_m"] = float(
                tracking_path[int(event["agent_id"])]
            )
            event["terminal_completion_after_return"] = bool(
                event["reference_reached"] and info.get("success", False)
            )

        positions_array = np.stack(position_history)
        velocities_array = np.stack(velocity_history)
        accelerations_array = np.stack(acceleration_history)
        metrics = trajectory_metrics(
            positions_array,
            velocities_array,
            accelerations_array,
            dt=float(settings["dt"]),
        )
        success = bool(info.get("success", False))
        reached_count = sum(bool(event["reference_reached"]) for event in events)
        collision_before = sum(
            bool(row["collision"])
            and not bool(row["terminal_return_active"])
            for row in step_rows
        )
        collision_after = sum(
            bool(row["collision"]) and bool(row["terminal_return_active"])
            for row in step_rows
        )
        episode = {
            "schema_version": SCHEMA_VERSION,
            "pair_id": pair_id,
            "protocol": protocol,
            "protocol_display_name": PROTOCOL_DISPLAY_NAMES[protocol],
            "selector": "Proposal top-1",
            "scenario": scenario,
            "seed": int(seed),
            "initial_condition_hash": initial_hash,
            "terminal_task_goals": terminal_goals.tolist(),
            "terminal_task_goals_unchanged": bool(np.array_equal(env.goals, terminal_goals)),
            "team_success": success,
            "success": success,
            "collision": collision,
            "obstacle_collision": obstacle_collision,
            "inter_agent_collision": inter_agent_collision,
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
            "trajectory_smoothness_team_mean": metrics["trajectory_smoothness_team_mean"],
            "jerk_rms_team_mean": metrics["jerk_rms_team_mean"],
            "smoothness_definition": metrics["smoothness_definition"],
            "minimum_inter_agent_distance_m": float(np.min(min_inter_history)),
            "minimum_obstacle_clearance_m": float(np.min(min_clearance_history)),
            "minimum_obstacle_clearance_source": "real_environment_sensor_packet",
            "goal_switch_count": int(len(switch_rows)),
            "mean_goal_jump_m": float(np.mean(goal_jumps)) if goal_jumps else 0.0,
            "maximum_goal_jump_m": float(np.max(goal_jumps)) if goal_jumps else 0.0,
            "reference_reached_tolerance_m": reached_tolerance,
            "reference_reached_tolerance_source": "existing waypoint consumer config",
            "temporary_reference_count": int(len(events)),
            "temporary_reference_reached_count": int(reached_count),
            "temporary_reference_reached_rate": float(reached_count / max(1, len(events))),
            "not_reached_count": int(len(events) - reached_count),
            "reached_then_terminal_success_rate": float(success and reached_count > 0),
            "collision_before_reference_count": int(collision_before),
            "collision_after_reference_count": int(collision_after),
            "mean_path_to_reference_m": float(np.mean(tracking_path)),
            "mean_time_to_reference_s": float(
                np.mean(
                    [event["reference_reached_time_s"] for event in events if event["reference_reached"]]
                )
            )
            if reached_count
            else None,
            "terminal_completion_after_return": bool(success and reached_count > 0),
            "K_t_zero_count": int(fallback_count),
            "no_candidate_fallback_rate": float(fallback_count / max(1, int(env.num_agents))),
            "candidate_generation_runtime_mean_ms": float(np.mean(candidate_generation_ms)),
            "candidate_selection_runtime_mean_ms": float(np.mean(selector_ms)),
            "frozen_sac_inference_runtime_mean_ms": float(np.mean(inference_times_ms)),
            "low_level_step_runtime_mean_ms": float(np.mean(low_level_times_ms)),
            "episode_runtime_ms": float((time.perf_counter_ns() - episode_started) / 1.0e6),
            "workspace_boundary_filter_enabled": False,
            "boundary_validity_diagnostic_only": True,
            "phase_reset_on_switch": False,
            "GAT_used": False,
            "SAC_training_performed": False,
            **scenario_metadata,
        }
        trajectory = {
            "positions": positions_array,
            "velocities": velocities_array,
            "accelerations": accelerations_array,
            "execution_references": np.stack(reference_history),
            "starts": starts,
            "terminal_goals": terminal_goals.copy(),
            "initial_condition_hash": np.asarray(initial_hash),
            "scene_snapshot_json": np.asarray(
                json.dumps(_jsonable(initial_snapshot), ensure_ascii=False)
            ),
        }
        return episode, events, step_rows, trajectory, switch_rows
    finally:
        env.close()


def _scenario_options(
    multi_config: Any, scenario: str, seed: int, peer_radius: float
) -> tuple[dict[str, Any], dict[str, Any], str]:
    env, metadata = build_closed_loop_environment(
        config=multi_config,
        scenario=scenario,
        seed=int(seed),
        peer_radius=float(peer_radius),
    )
    try:
        snapshot = _scene_snapshot(env)
        initial_hash = _scenario_hash(snapshot)
        options = {
            "starts": np.asarray(env.starts, dtype=float).copy(),
            "goals": np.asarray(env.goals, dtype=float).copy(),
            "static_obstacles": copy.deepcopy(env.static_obstacles),
            "dynamic_obstacles": copy.deepcopy(env.dynamic_obstacles),
        }
        return options, metadata, initial_hash
    finally:
        env.close()


def run_existing_boundary_free_episode(
    *,
    policy: Any,
    multi_config: Any,
    settings: Mapping[str, Any],
    scenario: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    protocol = PROTOCOL_EXISTING_BOUNDARY_FREE
    pair_id = _protocol_pair_id(scenario, seed, protocol)
    options, scenario_metadata, initial_hash = _scenario_options(
        multi_config, scenario, seed, float(settings["peer_radius"])
    )
    waypoint = settings["existing_waypoint"]
    priority = waypoint["priority_hold"]
    step_rows: list[dict[str, Any]] = []
    raw_reference_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    episode_started = time.perf_counter_ns()
    row = run_waypoint_episode(
        model=policy,
        config=multi_config,
        scenario_name=scenario,
        scenario_label=settings["scenario_display_names"][scenario],
        seed=int(seed),
        episode_index=0,
        peer_radius=float(settings["peer_radius"]),
        scenario_options=options,
        requested_depth=int(waypoint["lookahead_depth"]),
        proposal_config=ProposalConfig(**dict(settings["proposal_config"])),
        reached_tolerance=float(waypoint["reached_tolerance_m"]),
        boundary_margin=float(waypoint["boundary_margin_m"]),
        collision_clearance=float(waypoint["collision_clearance_m"]),
        segment_samples=int(waypoint["segment_samples"]),
        stagnation_steps=int(waypoint["stagnation_steps"]),
        stagnation_progress_epsilon=float(waypoint["stagnation_progress_epsilon_m"]),
        priority_hold_enabled=True,
        priority_order=[int(value) for value in priority["priority_order"]],
        scheduler_nominal_speed=float(priority["nominal_speed"]),
        scheduler_prediction_horizon=float(priority["prediction_horizon_s"]),
        scheduler_conflict_separation=float(priority["conflict_separation_m"]),
        scheduler_release_clear_steps=int(priority["release_clear_steps"]),
        scheduler_deadlock_steps=int(priority["deadlock_steps"]),
        scheduler_yield_distance=float(priority["yield_distance_m"]),
        scheduler_retreat_distance=float(priority["retreat_distance_m"]),
        boundary_filter_enabled=False,
        diagnostic_step_sink=step_rows,
        diagnostic_reference_sink=raw_reference_rows,
        diagnostic_switch_sink=switch_rows,
    )
    episode_runtime_ms = float((time.perf_counter_ns() - episode_started) / 1.0e6)
    trajectory = _trajectory_from_waypoint_trace(step_rows, int(settings["num_agents"]))
    metrics = trajectory_metrics(
        trajectory["positions"],
        trajectory["velocities"],
        trajectory["accelerations"],
        dt=float(settings["dt"]),
    )
    events = _normalize_waypoint_lifecycle(
        raw_reference_rows, episode_steps=int(row["steps"]), dt=float(settings["dt"])
    )
    for step in step_rows:
        step["position"] = copy.deepcopy(step["position_after"])
        step["velocity"] = copy.deepcopy(step["velocity_after"])
        step["distance_to_active_goal"] = float(
            step["distance_to_active_goal_before"]
        )
        step["distance_to_terminal_goal"] = float(
            step["distance_to_terminal_goal_before"]
        )
        step["minimum_clearance"] = float(step["min_clearance"])
        step["minimum_inter_agent_distance"] = float(row["min_pairwise_distance"])
        step["forcing_norm"] = None
        step["fixed_period_hold_after_reached"] = False
        step["terminal_return_active"] = False
    _augment_step_identity(step_rows, protocol=protocol, pair_id=pair_id)
    for event in events:
        event.update(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": protocol,
                "protocol_display_name": PROTOCOL_DISPLAY_NAMES[protocol],
                "pair_id": pair_id,
            }
        )
    for diagnostic in switch_rows:
        diagnostic.update(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": protocol,
                "protocol_display_name": PROTOCOL_DISPLAY_NAMES[protocol],
                "pair_id": pair_id,
            }
        )
    episode = {
        "schema_version": SCHEMA_VERSION,
        "pair_id": pair_id,
        "protocol": protocol,
        "protocol_display_name": PROTOCOL_DISPLAY_NAMES[protocol],
        "selector": "Proposal score + existing filtering/lookahead",
        "scenario": scenario,
        "seed": int(seed),
        "initial_condition_hash": initial_hash,
        "team_success": bool(row["team_success"]),
        "success": bool(row["team_success"]),
        "collision": bool(row["collision"]),
        "obstacle_collision": bool(row["obstacle_collision"]),
        "inter_agent_collision": bool(row["inter_agent_collision"]),
        "boundary_collision": bool(row["boundary_collision"]),
        "terminated": bool(row["team_success"] or row["collision"]),
        "truncated": bool(row["timeout"]),
        "timeout": bool(row["timeout"]),
        "termination_reason": str(row["status"]),
        "steps": int(row["steps"]),
        "completion_time_all_s": float(row["flight_time"]),
        "completion_time_success_s": float(row["flight_time"]) if row["team_success"] else None,
        "path_lengths_per_agent": metrics["path_lengths"].tolist(),
        "path_length_team_mean_m": metrics["path_length_team_mean"],
        "path_length_team_sum_m": metrics["path_length_team_sum"],
        "trajectory_smoothness_team_mean": metrics["trajectory_smoothness_team_mean"],
        "jerk_rms_team_mean": metrics["jerk_rms_team_mean"],
        "smoothness_definition": metrics["smoothness_definition"],
        "minimum_inter_agent_distance_m": float(row["min_pairwise_distance"]),
        "minimum_obstacle_clearance_m": float(row["min_sensor_clearance"]),
        "minimum_obstacle_clearance_source": "real_environment_sensor_packet",
        "goal_switch_count": int(len(switch_rows)),
        "mean_goal_jump_m": float(
            np.mean([diagnostic["goal_jump_m"] for diagnostic in switch_rows])
        )
        if switch_rows
        else 0.0,
        "maximum_goal_jump_m": float(
            np.max([diagnostic["goal_jump_m"] for diagnostic in switch_rows])
        )
        if switch_rows
        else 0.0,
        "reference_reached_tolerance_m": float(waypoint["reached_tolerance_m"]),
        "reference_reached_tolerance_source": "existing waypoint consumer config",
        "temporary_reference_count": int(len(events)),
        "temporary_reference_reached_count": int(
            sum(bool(event["reference_reached"]) for event in events)
        ),
        "temporary_reference_reached_rate": float(
            sum(bool(event["reference_reached"]) for event in events) / max(1, len(events))
        ),
        "waypoint_invalid_replan_count": int(row["waypoint_invalid_replan_count"]),
        "waypoint_stagnation_replan_count": int(row["waypoint_stagnation_replan_count"]),
        "waypoint_segment_fallback_count": int(row["waypoint_segment_fallback_count"]),
        "coordination_hold_event_count": int(row["coordination_hold_event_count"]),
        "coordination_release_count": int(row["coordination_release_count"]),
        "coordination_deadlock_event_count": int(row["coordination_deadlock_event_count"]),
        "episode_runtime_ms": episode_runtime_ms,
        "runtime_component_profiling": "not_instrumented_by_existing_waypoint_consumer",
        "workspace_boundary_filter_enabled": False,
        "boundary_validity_diagnostic_only": True,
        "boundary_rejection_count": int(row["waypoint_boundary_rejection_count"]),
        "phase_reset_on_switch": False,
        "GAT_used": False,
        "SAC_training_performed": False,
        **scenario_metadata,
    }
    trajectory.update(
        {
            "starts": np.asarray(options["starts"], dtype=float),
            "terminal_goals": np.asarray(options["goals"], dtype=float),
            "initial_condition_hash": np.asarray(initial_hash),
        }
    )
    return episode, events, step_rows, trajectory, switch_rows


def aggregate_protocols(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    protocols = list(dict.fromkeys(str(row["protocol"]) for row in episodes))
    for protocol in protocols:
        for scenario in ["overall"] + sorted(
            {str(row["scenario"]) for row in episodes if row["protocol"] == protocol}
        ):
            members = [
                row
                for row in episodes
                if row["protocol"] == protocol
                and (scenario == "overall" or row["scenario"] == scenario)
            ]
            if not members:
                continue
            mean = lambda key: float(np.mean([float(row.get(key, 0.0)) for row in members]))
            rows.append(
                {
                    "protocol": protocol,
                    "protocol_display_name": PROTOCOL_DISPLAY_NAMES[protocol],
                    "scenario": scenario,
                    "episodes": len(members),
                    "success_rate": mean("success"),
                    "collision_rate": mean("collision"),
                    "obstacle_collision_rate": mean("obstacle_collision"),
                    "inter_agent_collision_rate": mean("inter_agent_collision"),
                    "path_length_team_mean_m": mean("path_length_team_mean_m"),
                    "completion_time_all_mean_s": mean("completion_time_all_s"),
                    "trajectory_smoothness_mean": mean("trajectory_smoothness_team_mean"),
                    "minimum_inter_agent_distance_mean_m": mean(
                        "minimum_inter_agent_distance_m"
                    ),
                    "minimum_obstacle_clearance_mean_m": mean(
                        "minimum_obstacle_clearance_m"
                    ),
                    "goal_switch_count_mean": mean("goal_switch_count"),
                    "mean_goal_jump_m": mean("mean_goal_jump_m"),
                    "reference_reached_rate": mean("temporary_reference_reached_rate"),
                    "reference_reached_tolerance_m": members[0].get(
                        "reference_reached_tolerance_m"
                    ),
                    "reference_reached_tolerance_source": members[0].get(
                        "reference_reached_tolerance_source"
                    ),
                    "workspace_boundary_filter_enabled": members[0].get(
                        "workspace_boundary_filter_enabled", False
                    ),
                }
            )
    return rows


def _save_trajectory(path: Path, trajectory: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in trajectory.items()})


def _create_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in (
        "protocol_summary",
        "per_episode",
        "per_step",
        "reference_events",
        "dmp_switch_diagnostics",
        "observation_diagnostics",
        "failure_attribution",
        "representative_cases",
        "trajectories",
        "figures",
        "figure_data",
        "tests",
    ):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def run_stage_ab(
    settings: dict[str, Any],
    output_dir: Path,
    *,
    seeds: Sequence[int] | None = None,
    scenarios: Sequence[str] | None = None,
    max_jobs: int | None = None,
) -> dict[str, Any]:
    seeds = list(settings["development_seeds"] if seeds is None else seeds)
    scenarios = list(settings["scenarios"] if scenarios is None else scenarios)
    formal = set(int(value) for value in settings["formal_seeds_excluded_during_interface_design"])
    if formal.intersection(int(value) for value in seeds):
        raise ValueError("formal seeds 10-29 cannot be used during interface diagnosis")
    multi_config = build_single_distribution_multi_config(
        num_agents=int(settings["num_agents"]),
        max_steps=int(settings["max_steps"]),
    )
    policy, checkpoint = _load_policy(settings, multi_config)
    policy_hash_before = _policy_parameter_sha256(policy)
    critical_before = _critical_hashes(checkpoint)
    _create_output_dirs(output_dir)
    resolved = copy.deepcopy(settings)
    resolved.update(
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "output_dir": str(output_dir.resolve()),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": critical_before["checkpoint"],
            "policy_parameter_sha256_before": policy_hash_before,
            "seeds_executed": [int(value) for value in seeds],
            "scenarios_executed": list(scenarios),
            "stage_c_executed": False,
            "GAT_training_started": False,
        }
    )
    write_json(output_dir / "config.json", resolved)
    episodes: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    switches: list[dict[str, Any]] = []
    protocols = (
        PROTOCOL_TERMINAL,
        PROTOCOL_ONE_SHOT,
        PROTOCOL_FIXED_PERIOD,
        PROTOCOL_EXISTING_BOUNDARY_FREE,
    )
    jobs = [(scenario, int(seed), protocol) for scenario in scenarios for seed in seeds for protocol in protocols]
    if max_jobs is not None:
        jobs = jobs[: int(max_jobs)]
    started = time.perf_counter()
    for job_index, (scenario, seed, protocol) in enumerate(jobs, start=1):
        if protocol in {PROTOCOL_TERMINAL, PROTOCOL_FIXED_PERIOD}:
            result = run_pre_gat_protocol(
                policy=policy,
                multi_config=multi_config,
                settings=settings,
                scenario=scenario,
                seed=seed,
                protocol=protocol,
            )
        elif protocol == PROTOCOL_ONE_SHOT:
            result = run_one_shot_episode(
                policy=policy,
                multi_config=multi_config,
                settings=settings,
                scenario=scenario,
                seed=seed,
            )
        else:
            result = run_existing_boundary_free_episode(
                policy=policy,
                multi_config=multi_config,
                settings=settings,
                scenario=scenario,
                seed=seed,
            )
        episode, event_rows, step_rows, trajectory, switch_rows = result
        attribution = failure_attribution(
            episode=episode,
            step_rows=step_rows,
            reference_events=event_rows,
            shortly_after_switch_steps=int(
                settings["failure_attribution"]["shortly_after_switch_low_level_steps"]
            ),
        )
        episode["failure_attribution"] = attribution
        episodes.append(episode)
        events.extend(event_rows)
        steps.extend(step_rows)
        switches.extend(switch_rows)
        _save_trajectory(
            output_dir / "trajectories" / f"{episode['pair_id']}.npz", trajectory
        )
        write_json(output_dir / "per_episode" / f"{episode['pair_id']}.json", episode)
        print(
            f"[{job_index}/{len(jobs)}] {scenario} seed={seed} "
            f"{PROTOCOL_DISPLAY_NAMES[protocol]}: {episode['termination_reason']}",
            flush=True,
        )
        write_csv(output_dir / "per_episode" / "episodes.csv", episodes)

    summaries = aggregate_protocols(episodes)
    fixed_row = next(
        (
            row
            for row in summaries
            if row["scenario"] == "overall"
            and row["protocol"] == PROTOCOL_FIXED_PERIOD
        ),
        None,
    )
    fixed_success = float(fixed_row["success_rate"]) if fixed_row is not None else None
    gate = settings["stage_c_gate"]
    eligible: list[dict[str, Any]] = []
    for protocol in gate["eligible_protocols"]:
        row = next(
            (
                value
                for value in summaries
                if value["scenario"] == "overall" and value["protocol"] == protocol
            ),
            None,
        )
        if row is None or fixed_success is None:
            eligible.append(
                {
                    "protocol": protocol,
                    "success_rate": row["success_rate"] if row is not None else None,
                    "fixed_period_success_rate": fixed_success,
                    "absolute_improvement": None,
                    "stage_c_eligible": False,
                    "gate_status": "NOT_EVALUATED_IN_PARTIAL_RUN",
                }
            )
            continue
        improvement = float(row["success_rate"] - fixed_success)
        passed = bool(
            float(row["success_rate"]) > float(gate["minimum_success_rate_exclusive"])
            and improvement
            >= float(gate["minimum_absolute_success_improvement_over_fixed_period"])
        )
        eligible.append(
            {
                "protocol": protocol,
                "success_rate": row["success_rate"],
                "fixed_period_success_rate": fixed_success,
                "absolute_improvement": improvement,
                "stage_c_eligible": passed,
                "gate_status": "PASSED" if passed else "FAILED",
            }
        )
    policy_hash_after = _policy_parameter_sha256(policy)
    critical_after = _critical_hashes(checkpoint)
    integrity = {
        "policy_parameter_sha256_before": policy_hash_before,
        "policy_parameter_sha256_after": policy_hash_after,
        "policy_parameters_unchanged": policy_hash_before == policy_hash_after,
        "critical_hashes_before": critical_before,
        "critical_hashes_after": critical_after,
        "critical_files_unchanged": critical_before == critical_after,
        "GAT_training_started": False,
        "SAC_training_performed": False,
    }
    write_csv(output_dir / "protocol_summary" / "protocol_summary.csv", summaries)
    write_json(output_dir / "protocol_summary" / "protocol_summary.json", summaries)
    write_csv(output_dir / "per_step" / "per_step.csv", steps)
    write_json(output_dir / "per_step" / "per_step.json", steps)
    write_csv(output_dir / "reference_events" / "reference_events.csv", events)
    write_json(output_dir / "reference_events" / "reference_events.json", events)
    write_csv(output_dir / "dmp_switch_diagnostics" / "switch_diagnostics.csv", switches)
    write_json(output_dir / "dmp_switch_diagnostics" / "switch_diagnostics.json", switches)
    observation_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "protocol",
                "pair_id",
                "scenario",
                "seed",
                "timestep",
                "agent_id",
                "switch_reason",
                "previous_goal_direction",
                "new_goal_direction",
                "goal_direction_delta",
                "goal_direction_delta_l2",
                "previous_normalized_goal_distance",
                "new_normalized_goal_distance",
                "normalized_goal_distance_delta",
                "phase",
                "K_alpha",
                "K_beta",
                "observation_l2_change",
                "non_goal_feature_l2_change",
                "observation_dimension",
                "same_state_snapshot",
            }
        }
        for row in switches
    ]
    write_csv(output_dir / "observation_diagnostics" / "observation_diagnostics.csv", observation_rows)
    write_json(output_dir / "observation_diagnostics" / "observation_diagnostics.json", observation_rows)
    attribution_rows = [
        {
            "pair_id": row["pair_id"],
            "protocol": row["protocol"],
            "scenario": row["scenario"],
            "seed": row["seed"],
            "success": row["success"],
            "failure_attribution": row["failure_attribution"],
        }
        for row in episodes
    ]
    write_csv(output_dir / "failure_attribution" / "failure_attribution.csv", attribution_rows)
    write_json(output_dir / "failure_attribution" / "failure_attribution.json", attribution_rows)
    write_csv(output_dir / "protocol_summary" / "stage_c_gate.csv", eligible)
    write_json(output_dir / "protocol_summary" / "stage_c_gate.json", eligible)
    write_json(output_dir / "tests" / "integrity.json", integrity)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episode_count": len(episodes),
        "reference_event_count": len(events),
        "step_row_count": len(steps),
        "switch_diagnostic_count": len(switches),
        "runtime_seconds": float(time.perf_counter() - started),
        "stage_c_gate": eligible,
        "stage_c_executed": False,
        "integrity": integrity,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--scenarios", type=str, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    settings = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPO_ROOT / settings["output_dir"] / timestamp
    )
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else None
    )
    scenarios = (
        [value.strip() for value in args.scenarios.split(",") if value.strip()]
        if args.scenarios
        else None
    )
    run_stage_ab(
        settings,
        output_dir,
        seeds=seeds,
        scenarios=scenarios,
        max_jobs=args.max_jobs,
    )
    print(output_dir, flush=True)
    return output_dir


if __name__ == "__main__":
    main()
