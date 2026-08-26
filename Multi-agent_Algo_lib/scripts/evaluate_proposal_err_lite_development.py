"""Paired development-only evaluation of Proposal one-shot and ERR-lite.

This evaluator reuses the frozen 40-scene development manifest, Proposal Top-1,
historical 122-D SAC input, DMP transition, and environment semantics from the
final four-stage benchmark.  It does not read the formal 400-scene results.
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
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# The pinned Windows environment has an incompatible optional pyarrow build.
# This evaluator does not use pyarrow; preventing pandas' optional import avoids
# a native access violation without changing any control/evaluation semantics.
sys.modules.setdefault("pyarrow", None)

import numpy as np
import torch  # Load Torch DLLs before the legacy runner imports pandas on Windows.


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Guidance.reference_point_proposal_demo import (  # noqa: E402
    ProposalConfig,
    compute_sector_safety_field,
)
from planning.final_four_stage_benchmark import WORKSPACE_BOUNDS  # noqa: E402
from planning.goal_semantics_diagnosis import (  # noqa: E402
    temporary_checkpoint_observations,
)
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.pre_gat_closed_loop import (  # noqa: E402
    generate_candidate_set,
    termination_reason,
    trajectory_metrics,
)
from planning.proposal_err_lite import (  # noqa: E402
    ProposalERRLiteConfig,
    ProposalERRLiteSupervisor,
    TRIGGER_REFERENCE_REACHED,
    TRIGGER_REFERENCE_STAGNATION,
    TRIGGER_TERMINAL_RETRY,
)
from scripts.evaluate_frozen_policy_waypoint_guidance import (  # noqa: E402
    set_dmp_active_goal_preserve_phase,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402
from scripts.run_final_four_stage_benchmark import (  # noqa: E402
    ManifestEnvironmentBuilder,
)


SCHEMA_VERSION = "proposal_err_lite_development_v1"
METHOD_ONE_SHOT = "proposal_one_shot"
METHOD_ERR_LITE = "proposal_err_lite"
METHODS = (METHOD_ONE_SHOT, METHOD_ERR_LITE)
DEFAULT_CONFIG = REPO_ROOT / "configs/evaluation/proposal_err_lite_development.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    records = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in records:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(row.get(key)), ensure_ascii=False)
                        if isinstance(row.get(key), (dict, list, tuple, np.ndarray))
                        else _jsonable(row.get(key))
                    )
                    for key in fields
                }
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_runtime(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    execution = config["execution"]
    base = build_single_distribution_multi_config(
        num_agents=int(execution["num_agents"]),
        max_steps=int(execution["max_steps"]),
    )
    multi_config = replace(
        base,
        workspace_bounds=WORKSPACE_BOUNDS,
        randomize_start_goal=False,
        start_position_bounds=((-0.4, -2.0, -0.9), (0.2, 2.0, 0.9)),
        goal_position_bounds=((7.0, -2.0, -0.9), (11.0, 2.0, 0.9)),
        min_start_goal_distance=5.5,
    )
    settings = load_json(REPO_ROOT / config["sources"]["base_execution_config"])
    settings.update(
        {
            "checkpoint": config["sources"]["sac_checkpoint"],
            "checkpoint_sha256_expected": config["sources"][
                "sac_checkpoint_sha256_expected"
            ],
            "deterministic_policy": True,
            "num_agents": int(execution["num_agents"]),
            "max_steps": int(execution["max_steps"]),
            "dt": float(execution["dt"]),
            "peer_radius": float(execution["peer_radius"]),
            "proposal_config": copy.deepcopy(config["proposal_config"]),
        }
    )
    checkpoint = REPO_ROOT / config["sources"]["sac_checkpoint"]
    observed_hash = sha256_file(checkpoint)
    if observed_hash != config["sources"]["sac_checkpoint_sha256_expected"]:
        raise RuntimeError("frozen SAC checkpoint hash mismatch")
    policy, loaded = _load_policy(settings, multi_config)
    if loaded.resolve() != checkpoint.resolve():
        raise RuntimeError("SAC loader resolved a different checkpoint")
    return {
        "multi_config": multi_config,
        "settings": settings,
        "policy": policy,
        "builder": ManifestEnvironmentBuilder(manifest),
        "checkpoint_hash": observed_hash,
    }


def select_proposal(
    env: Any,
    *,
    agent_id: int,
    terminal_goal: np.ndarray,
    proposal_config: ProposalConfig,
    top_k: int,
) -> tuple[np.ndarray, bool, dict[str, Any], float]:
    started = time.perf_counter_ns()
    proposals, count_before_top_k = generate_candidate_set(
        env,
        int(agent_id),
        proposal_config,
        consumer_top_k=int(top_k),
    )
    elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
    if proposals:
        selected = proposals[0]
        return (
            np.asarray(selected.point, dtype=float).copy(),
            True,
            {
                "K_before_top_k": int(count_before_top_k),
                "K_t": int(len(proposals)),
                "selected_rank": 1,
                "selected_score": float(selected.score),
                "selected_distance_m": float(selected.distance),
                "selection_type": "proposal_top1",
            },
            float(elapsed_ms),
        )
    return (
        np.asarray(terminal_goal, dtype=float).copy(),
        False,
        {
            "K_before_top_k": int(count_before_top_k),
            "K_t": 0,
            "selected_rank": None,
            "selected_score": None,
            "selected_distance_m": None,
            "selection_type": "no_candidate_terminal_fallback",
        },
        float(elapsed_ms),
    )


def direct_terminal_feasible(
    env: Any,
    *,
    agent_id: int,
    terminal_goal: np.ndarray,
    proposal_config: ProposalConfig,
) -> tuple[bool, dict[str, Any], float]:
    """Use the existing Proposal safety field to gate a direct terminal handoff."""

    started = time.perf_counter_ns()
    index = int(agent_id)
    position = np.asarray(env.dynamics[index].p, dtype=float)
    goal = np.asarray(terminal_goal, dtype=float)
    packet = env.latest_sensor_packets[index]
    sensor = env.sensors[index]
    field = compute_sector_safety_field(
        position,
        goal,
        env.dynamics[index].v,
        packet,
        sensor,
        proposal_config,
        float(env.env_config.goal_tolerance),
    )
    displacement = goal - position
    goal_distance = float(np.linalg.norm(displacement))
    rays = np.asarray(sensor.ray_directions, dtype=float).reshape(-1, 3)
    if goal_distance > 1.0e-12:
        sector_flat = int(np.argmax(rays @ (displacement / goal_distance)))
    else:
        sector_flat = int(np.argmax(field.normalized_margin.reshape(-1)))
    azimuth, elevation = np.unravel_index(
        sector_flat, field.obstacle_distance.shape
    )
    guarded_distance = float(field.obstacle_distance[azimuth, elevation])
    travel_budget = max(
        0.0,
        min(
            guarded_distance - float(field.effective_safe_radius),
            float(sensor.sensing_radius),
        ),
    )
    feasible = bool(goal_distance <= travel_budget + 1.0e-9)
    elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
    return (
        feasible,
        {
            "direct_terminal_goal_distance_m": goal_distance,
            "direct_terminal_travel_budget_m": travel_budget,
            "direct_terminal_sector_flat": sector_flat,
            "direct_terminal_guarded_obstacle_distance_m": guarded_distance,
            "direct_terminal_effective_safe_radius_m": float(
                field.effective_safe_radius
            ),
            "direct_terminal_feasible": feasible,
        },
        float(elapsed_ms),
    )


def run_episode(
    *,
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    entry: Mapping[str, Any],
    method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    execution = config["execution"]
    env, metadata = runtime["builder"](
        config=runtime["multi_config"],
        scenario=str(entry["scenario_id"]),
        seed=int(entry["seed"]),
        peer_radius=float(execution["peer_radius"]),
    )
    wall_started = time.perf_counter_ns()
    proposal_config = ProposalConfig(**dict(config["proposal_config"]))
    err_config = ProposalERRLiteConfig(
        dt=float(execution["dt"]),
        **{
            key: value
            for key, value in config["err_lite"].items()
            if key in ProposalERRLiteConfig.__dataclass_fields__ and key != "dt"
        },
    )
    event_rows: list[dict[str, Any]] = []
    try:
        starts = np.asarray(env.starts, dtype=float).copy()
        terminal_goals = np.asarray(env.goals, dtype=float).copy()
        active_goals = terminal_goals.copy()
        reference_flags = np.zeros(int(env.num_agents), dtype=bool)
        planning_runtime_ms = 0.0
        per_agent_planning_calls = 0
        planning_cycle_count = 1
        phase_deltas: list[float] = []
        for agent_id in range(int(env.num_agents)):
            goal, is_reference, record, runtime_ms = select_proposal(
                env,
                agent_id=agent_id,
                terminal_goal=terminal_goals[agent_id],
                proposal_config=proposal_config,
                top_k=int(execution["top_k"]),
            )
            active_goals[agent_id] = goal
            reference_flags[agent_id] = is_reference
            planning_runtime_ms += runtime_ms
            per_agent_planning_calls += 1
            phase_before = float(env.dmps[agent_id].phase)
            set_dmp_active_goal_preserve_phase(env.dmps[agent_id], goal)
            phase_deltas.append(float(env.dmps[agent_id].phase) - phase_before)
            event_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "method": method,
                    "stage": entry["stage"],
                    "scenario_id": entry["scenario_id"],
                    "seed": int(entry["seed"]),
                    "step": 0,
                    "agent_id": agent_id,
                    "event": "INITIAL_SELECTION",
                    "goal_changed": bool(is_reference),
                    "is_reference": bool(is_reference),
                    "planning_runtime_ms": runtime_ms,
                    **record,
                }
            )
        supervisor = ProposalERRLiteSupervisor(
            err_config,
            active_goals=active_goals,
            is_reference=reference_flags,
            terminal_retry_enabled=np.logical_not(reference_flags),
            positions=starts,
        )
        completed = np.zeros(int(env.num_agents), dtype=bool)
        completion_steps: list[int | None] = [None] * int(env.num_agents)
        handoff_counts = np.zeros(int(env.num_agents), dtype=int)
        reference_selection_counts = reference_flags.astype(int)
        reference_reached_counts = np.zeros(int(env.num_agents), dtype=int)
        trigger_counts: Counter[str] = Counter()
        direct_safe_handoff_count = 0
        positions = [env._positions().copy()]
        velocities = [env._velocities().copy()]
        accelerations: list[np.ndarray] = []
        lower_level_runtime_ms = 0.0
        environment_step_runtime_ms = 0.0
        minimum_clearance = float(
            min(packet.min_clearance for packet in env.latest_sensor_packets)
        )
        minimum_peer_distance = float(
            env._check_collision()["min_inter_agent_distance"]
        )
        obstacle_collision = False
        inter_agent_collision = False
        collision = False
        obstacle_masks = np.zeros(int(env.num_agents), dtype=bool)
        inter_agent_masks = np.zeros(int(env.num_agents), dtype=bool)
        terminated = truncated = False
        info: dict[str, Any] = {}
        gate_names: list[str | None] = []

        def observe_execution(_: dict[str, Any], transition: Any) -> None:
            gate_names.append(transition.controller_info.get("forcing_gate_semantics"))

        with scoped_historical_preview_and_multi_agent_transition(
            execution_observer=observe_execution
        ):
            while not (terminated or truncated):
                current_step = int(env.steps)
                triggered_agents: list[tuple[int, Any]] = []
                if current_step > 0:
                    for agent_id in range(int(env.num_agents)):
                        if completed[agent_id]:
                            continue
                        state = supervisor.states[agent_id]
                        if method == METHOD_ONE_SHOT:
                            if (
                                state.is_reference
                                and np.linalg.norm(
                                    env.dynamics[agent_id].p - state.active_goal
                                )
                                <= float(execution["handoff_threshold_m"])
                            ):
                                reference_reached_counts[agent_id] += 1
                                handoff_counts[agent_id] += 1
                                old_goal = state.active_goal.copy()
                                state.active_goal = terminal_goals[agent_id].copy()
                                state.is_reference = False
                                phase_before = float(env.dmps[agent_id].phase)
                                set_dmp_active_goal_preserve_phase(
                                    env.dmps[agent_id], state.active_goal
                                )
                                phase_deltas.append(
                                    float(env.dmps[agent_id].phase) - phase_before
                                )
                                event_rows.append(
                                    {
                                        "schema_version": SCHEMA_VERSION,
                                        "method": method,
                                        "stage": entry["stage"],
                                        "scenario_id": entry["scenario_id"],
                                        "seed": int(entry["seed"]),
                                        "step": current_step,
                                        "agent_id": agent_id,
                                        "event": "ONE_SHOT_HANDOFF_TO_TERMINAL",
                                        "goal_changed": not np.array_equal(
                                            old_goal, state.active_goal
                                        ),
                                        "is_reference": False,
                                        "planning_runtime_ms": 0.0,
                                    }
                                )
                            continue
                        decision = supervisor.evaluate(
                            agent_id,
                            current_step=current_step,
                            position=env.dynamics[agent_id].p,
                        )
                        if decision.trigger is not None:
                            triggered_agents.append((agent_id, decision))
                if triggered_agents:
                    planning_cycle_count += 1
                    for agent_id, decision in triggered_agents:
                        old_goal = supervisor.states[agent_id].active_goal.copy()
                        old_was_reference = supervisor.states[agent_id].is_reference
                        trigger_counts[decision.trigger] += 1
                        gate_record: dict[str, Any] = {}
                        gate_runtime_ms = 0.0
                        direct_handoff = False
                        if decision.trigger == TRIGGER_REFERENCE_REACHED:
                            reference_reached_counts[agent_id] += 1
                            handoff_counts[agent_id] += 1
                            direct_handoff, gate_record, gate_runtime_ms = (
                                direct_terminal_feasible(
                                    env,
                                    agent_id=agent_id,
                                    terminal_goal=terminal_goals[agent_id],
                                    proposal_config=proposal_config,
                                )
                            )
                            planning_runtime_ms += gate_runtime_ms
                        if direct_handoff:
                            direct_safe_handoff_count += 1
                            goal = terminal_goals[agent_id].copy()
                            is_reference = False
                            record = {
                                "K_before_top_k": None,
                                "K_t": None,
                                "selected_rank": None,
                                "selected_score": None,
                                "selected_distance_m": None,
                                "selection_type": "direct_safe_terminal_handoff",
                            }
                            runtime_ms = 0.0
                        else:
                            goal, is_reference, record, runtime_ms = select_proposal(
                                env,
                                agent_id=agent_id,
                                terminal_goal=terminal_goals[agent_id],
                                proposal_config=proposal_config,
                                top_k=int(execution["top_k"]),
                            )
                            per_agent_planning_calls += 1
                        planning_runtime_ms += runtime_ms
                        reference_selection_counts[agent_id] += int(is_reference)
                        changed = supervisor.update(
                            agent_id,
                            new_goal=goal,
                            is_reference=is_reference,
                            terminal_retry_enabled=bool(
                                not is_reference and not direct_handoff
                            ),
                            current_step=current_step,
                            position=env.dynamics[agent_id].p,
                            count_as_replan=not direct_handoff,
                        )
                        active_goals[agent_id] = goal
                        phase_before = float(env.dmps[agent_id].phase)
                        if changed:
                            set_dmp_active_goal_preserve_phase(
                                env.dmps[agent_id], goal
                            )
                        phase_deltas.append(
                            float(env.dmps[agent_id].phase) - phase_before
                        )
                        event_rows.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "method": method,
                                "stage": entry["stage"],
                                "scenario_id": entry["scenario_id"],
                                "seed": int(entry["seed"]),
                                "step": current_step,
                                "agent_id": agent_id,
                                "event": decision.trigger,
                                "old_was_reference": bool(old_was_reference),
                                "goal_changed": bool(changed),
                                "is_reference": bool(is_reference),
                                "active_goal_distance_m": decision.active_goal_distance_m,
                                "active_goal_age_s": decision.active_goal_age_s,
                                "progress_rate_mps": decision.progress_rate_mps,
                                "planning_runtime_ms": runtime_ms
                                + gate_runtime_ms,
                                "old_goal": old_goal.tolist(),
                                "new_goal": goal.tolist(),
                                **gate_record,
                                **record,
                            }
                        )
                active_goals = np.stack(
                    [state.active_goal for state in supervisor.states]
                )
                inference_started = time.perf_counter_ns()
                observations = temporary_checkpoint_observations(env, active_goals)
                actions, _ = runtime["policy"].predict(
                    observations, deterministic=True
                )
                lower_level_runtime_ms += (
                    time.perf_counter_ns() - inference_started
                ) / 1.0e6
                actions = np.asarray(actions, dtype=np.float32)
                if actions.shape != tuple(env.action_shape):
                    raise RuntimeError("frozen SAC returned an unexpected action shape")
                step_started = time.perf_counter_ns()
                _, _, terminated, truncated, info = env.step(actions)
                environment_step_runtime_ms += (
                    time.perf_counter_ns() - step_started
                ) / 1.0e6
                positions.append(env._positions().copy())
                velocities.append(env._velocities().copy())
                accelerations.append(
                    np.asarray(info["applied_accelerations"], dtype=float).copy()
                )
                minimum_clearance = min(
                    minimum_clearance, float(np.min(info["min_clearances"]))
                )
                minimum_peer_distance = min(
                    minimum_peer_distance,
                    float(info["min_inter_agent_distance"]),
                )
                collision |= bool(info["collision"])
                obstacle_collision |= bool(np.any(info["obstacle_collision_mask"]))
                inter_agent_collision |= bool(
                    np.any(info["inter_agent_collision_mask"])
                )
                obstacle_masks |= np.asarray(
                    info["obstacle_collision_mask"], dtype=bool
                )
                inter_agent_masks |= np.asarray(
                    info["inter_agent_collision_mask"], dtype=bool
                )
                success_mask = np.asarray(info["success_mask"], dtype=bool)
                for agent_id in np.flatnonzero(success_mask):
                    completed[int(agent_id)] = True
                    if completion_steps[int(agent_id)] is None:
                        completion_steps[int(agent_id)] = int(env.steps)

        if not np.array_equal(env.goals, terminal_goals):
            raise RuntimeError("terminal task goals changed")
        if phase_deltas and not np.allclose(
            phase_deltas, 0.0, rtol=0.0, atol=0.0
        ):
            raise RuntimeError("active-goal update changed DMP phase")
        if gate_names and not all(name == HISTORICAL_GATE_NAME for name in gate_names):
            raise RuntimeError("execution did not use the historical forcing gate")
        position_array = np.stack(positions)
        velocity_array = np.stack(velocities)
        acceleration_array = np.stack(accelerations)
        metrics = trajectory_metrics(
            position_array,
            velocity_array,
            acceleration_array,
            dt=float(execution["dt"]),
        )
        team_success = bool(info.get("success", False))
        reason = termination_reason(
            success=team_success,
            collision=collision,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        final_distances = np.linalg.norm(
            terminal_goals - position_array[-1], axis=1
        )
        initial_distances = np.linalg.norm(terminal_goals - starts, axis=1)
        agent_rows: list[dict[str, Any]] = []
        for agent_id in range(int(env.num_agents)):
            agent_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "method": method,
                    "stage": entry["stage"],
                    "family": entry["family"],
                    "scenario_id": entry["scenario_id"],
                    "seed": int(entry["seed"]),
                    "agent_id": agent_id,
                    "completed": completion_steps[agent_id] is not None,
                    "completion_step": completion_steps[agent_id],
                    "obstacle_collision": bool(obstacle_masks[agent_id]),
                    "inter_agent_collision": bool(inter_agent_masks[agent_id]),
                    "terminal_progress_m": float(
                        initial_distances[agent_id] - final_distances[agent_id]
                    ),
                    "final_terminal_distance_m": float(final_distances[agent_id]),
                    "path_length_m": float(metrics["path_lengths"][agent_id]),
                    "reference_selection_count": int(
                        reference_selection_counts[agent_id]
                    ),
                    "reference_reached_count": int(
                        reference_reached_counts[agent_id]
                    ),
                    "replan_count": int(supervisor.states[agent_id].replan_count),
                    "goal_change_count": int(
                        supervisor.states[agent_id].goal_change_count
                    ),
                    "unchanged_replan_count": int(
                        supervisor.states[agent_id].unchanged_replan_count
                    ),
                }
            )
        episode = {
            "schema_version": SCHEMA_VERSION,
            "split": "development",
            "method": method,
            "stage": entry["stage"],
            "family": entry["family"],
            "scenario_id": entry["scenario_id"],
            "seed": int(entry["seed"]),
            "environment_fingerprint": entry["environment_fingerprint"],
            "team_success": team_success,
            "agent_completion_rate": float(
                np.mean([row["completed"] for row in agent_rows])
            ),
            "any_collision": collision,
            "obstacle_collision": obstacle_collision,
            "inter_agent_collision": inter_agent_collision,
            "timeout": bool(truncated),
            "termination_reason": reason,
            "steps": int(env.steps),
            "completion_time_s": (
                float(env.steps) * float(execution["dt"])
                if team_success
                else None
            ),
            "team_path_length_m": float(metrics["path_length_team_sum"]),
            "trajectory_smoothness": float(
                metrics["trajectory_smoothness_team_mean"]
            ),
            "minimum_obstacle_clearance_m": minimum_clearance,
            "minimum_inter_agent_distance_m": minimum_peer_distance,
            "reference_selection_count": int(np.sum(reference_selection_counts)),
            "reference_reached_count": int(np.sum(reference_reached_counts)),
            "reference_executability_rate": (
                float(
                    np.sum(reference_reached_counts)
                    / np.sum(reference_selection_counts)
                )
                if np.sum(reference_selection_counts)
                else None
            ),
            "replan_event_count": int(
                sum(state.replan_count for state in supervisor.states)
            ),
            "goal_change_count": int(
                sum(state.goal_change_count for state in supervisor.states)
            ),
            "unchanged_replan_count": int(
                sum(state.unchanged_replan_count for state in supervisor.states)
            ),
            "handoff_count": int(np.sum(handoff_counts)),
            "reference_reached_trigger_count": int(
                trigger_counts[TRIGGER_REFERENCE_REACHED]
            ),
            "stagnation_trigger_count": int(
                trigger_counts[TRIGGER_REFERENCE_STAGNATION]
            ),
            "terminal_retry_trigger_count": int(
                trigger_counts[TRIGGER_TERMINAL_RETRY]
            ),
            "direct_safe_terminal_handoff_count": int(
                direct_safe_handoff_count
            ),
            "planning_cycle_count": int(planning_cycle_count),
            "per_agent_proposal_call_count": int(per_agent_planning_calls),
            "planning_runtime_ms": float(planning_runtime_ms),
            "planning_runtime_per_cycle_ms": float(
                planning_runtime_ms / planning_cycle_count
            ),
            "lower_level_runtime_ms": float(lower_level_runtime_ms),
            "environment_step_runtime_ms": float(environment_step_runtime_ms),
            "wall_runtime_ms": float(
                (time.perf_counter_ns() - wall_started) / 1.0e6
            ),
            "historical_gate_verified": bool(gate_names)
            and all(name == HISTORICAL_GATE_NAME for name in gate_names),
            "terminal_task_goals_unchanged": True,
            "maximum_phase_switch_delta": float(
                np.max(np.abs(phase_deltas)) if phase_deltas else 0.0
            ),
            "emergency_trigger_count": 0,
            "safety_trigger_count": 0,
            **metadata,
        }
        return episode, agent_rows, event_rows
    finally:
        env.close()


def _summary_row(rows: Sequence[Mapping[str, Any]], method: str, scope: str) -> dict[str, Any]:
    members = [
        row
        for row in rows
        if row["method"] == method
        and (scope == "overall" or row["stage"] == scope)
    ]
    successful = [row for row in members if row["team_success"]]
    return {
        "method": method,
        "scope": scope,
        "episode_count": len(members),
        "team_success_count": sum(bool(row["team_success"]) for row in members),
        "team_success_rate": float(np.mean([row["team_success"] for row in members])),
        "any_collision_count": sum(bool(row["any_collision"]) for row in members),
        "any_collision_rate": float(np.mean([row["any_collision"] for row in members])),
        "obstacle_collision_count": sum(
            bool(row["obstacle_collision"]) for row in members
        ),
        "inter_agent_collision_count": sum(
            bool(row["inter_agent_collision"]) for row in members
        ),
        "timeout_count": sum(bool(row["timeout"]) for row in members),
        "timeout_rate": float(np.mean([row["timeout"] for row in members])),
        "agent_completion_rate": float(
            np.mean([row["agent_completion_rate"] for row in members])
        ),
        "successful_completion_time_s": (
            float(np.mean([row["completion_time_s"] for row in successful]))
            if successful
            else None
        ),
        "successful_team_path_length_m": (
            float(np.mean([row["team_path_length_m"] for row in successful]))
            if successful
            else None
        ),
        "planning_runtime_ms": float(
            np.mean([row["planning_runtime_ms"] for row in members])
        ),
        "planning_cycle_count": int(
            sum(row["planning_cycle_count"] for row in members)
        ),
        "replan_event_count": int(
            sum(row["replan_event_count"] for row in members)
        ),
        "reference_reached_trigger_count": int(
            sum(row["reference_reached_trigger_count"] for row in members)
        ),
        "stagnation_trigger_count": int(
            sum(row["stagnation_trigger_count"] for row in members)
        ),
        "terminal_retry_trigger_count": int(
            sum(row["terminal_retry_trigger_count"] for row in members)
        ),
        "direct_safe_terminal_handoff_count": int(
            sum(row["direct_safe_terminal_handoff_count"] for row in members)
        ),
        "unchanged_replan_count": int(
            sum(row["unchanged_replan_count"] for row in members)
        ),
    }


def build_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    stages = ["stage_1", "stage_2", "stage_3", "stage_4"]
    return [
        _summary_row(rows, method, scope)
        for method in METHODS
        for scope in [*stages, "overall"]
    ]


def failure_category(row: Mapping[str, Any]) -> str:
    if row["team_success"]:
        return "SUCCESS"
    if row["obstacle_collision"]:
        return "OBSTACLE_COLLISION"
    if row["inter_agent_collision"]:
        return "INTER_AGENT_COLLISION"
    if row["timeout"]:
        return "TIMEOUT"
    return "OTHER"


def paired_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["scenario_id"])][str(row["method"])] = row
    both_success = err_only = one_only = both_fail = 0
    paired_rows: list[dict[str, Any]] = []
    for scenario_id in sorted(grouped):
        methods = grouped[scenario_id]
        one = methods[METHOD_ONE_SHOT]
        err = methods[METHOD_ERR_LITE]
        if one["team_success"] and err["team_success"]:
            outcome = "BOTH_SUCCESS"
            both_success += 1
        elif err["team_success"]:
            outcome = "ERR_LITE_ONLY_SUCCESS"
            err_only += 1
        elif one["team_success"]:
            outcome = "ONE_SHOT_ONLY_SUCCESS"
            one_only += 1
        else:
            outcome = "BOTH_FAIL"
            both_fail += 1
        paired_rows.append(
            {
                "scenario_id": scenario_id,
                "stage": one["stage"],
                "outcome": outcome,
                "one_shot_failure": failure_category(one),
                "err_lite_failure": failure_category(err),
                "err_minus_one_success": int(err["team_success"])
                - int(one["team_success"]),
                "err_minus_one_collision": int(err["any_collision"])
                - int(one["any_collision"]),
                "err_minus_one_timeout": int(err["timeout"])
                - int(one["timeout"]),
            }
        )
    discordant = err_only + one_only
    exact_p = (
        min(
            1.0,
            2.0
            * sum(
                math.comb(discordant, index) * (0.5**discordant)
                for index in range(0, min(err_only, one_only) + 1)
            ),
        )
        if discordant
        else 1.0
    )
    return {
        "both_success": both_success,
        "err_lite_only_success": err_only,
        "one_shot_only_success": one_only,
        "both_fail": both_fail,
        "mcnemar_two_sided_exact_p": exact_p,
        "rows": paired_rows,
    }


def render_report(
    config: Mapping[str, Any],
    summary: Sequence[Mapping[str, Any]],
    paired: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
) -> str:
    index = {(row["method"], row["scope"]): row for row in summary}
    one = index[(METHOD_ONE_SHOT, "overall")]
    err = index[(METHOD_ERR_LITE, "overall")]
    success_gain_pp = 100.0 * (
        err["team_success_rate"] - one["team_success_rate"]
    )
    collision_change_pp = 100.0 * (
        err["any_collision_rate"] - one["any_collision_rate"]
    )
    timeout_change_pp = 100.0 * (err["timeout_rate"] - one["timeout_rate"])
    decision = (
        "PROMISING"
        if success_gain_pp > 0.0 and collision_change_pp <= 0.0
        else "NO_CLEAR_GAIN"
    )
    lines = [
        "# Proposal ERR-lite Development Validation",
        "",
        "## Executive result",
        "",
        f"`ERR_LITE_DEVELOPMENT_RESULT = {decision}`. This is a paired development-only mechanism check on the frozen 40-scene manifest; it is not a formal result.",
        "",
        f"Proposal one-shot achieved {one['team_success_count']}/{one['episode_count']} ({one['team_success_rate']:.1%}) team success, while Proposal ERR-lite achieved {err['team_success_count']}/{err['episode_count']} ({err['team_success_rate']:.1%}), a {success_gain_pp:+.1f} pp change. Collision changed by {collision_change_pp:+.1f} pp and timeout by {timeout_change_pp:+.1f} pp.",
        "",
        "## Frozen protocol",
        "",
        "- 40 development scenes only: 10 per stage, same manifest, starts/goals, obstacles, dynamic tracks, SAC checkpoint, 122-D historical gate, DMP and 220-step limit.",
        "- Both methods use Proposal Top-1; FP-SHEP, GAT, retraining, emergency triggers and safety triggers are disabled.",
        "- ERR-lite replans only after reference reach, 30-step stagnation below 0.08 m total progress, or 5 s terminal-fallback age; the cooldown is 1 s.",
        "- A reached reference hands off to the terminal when the existing Proposal sector field certifies the direct segment; otherwise Proposal Top-1 is regenerated. Safe terminal handoffs are not treated as Null/K=0 retries.",
        "- Planning time is Proposal computation only. SAC inference and environment execution are reported separately in the raw episode table.",
        "",
        "## Pilot correction",
        "",
        "The earlier C0 pilot artifact (`20260818_235144`) is rejected as an implementation diagnostic: reference-reached events bypassed the configured dwell and there was no direct-terminal gate, producing a short-waypoint asymptotic loop. The current C1 result enforces both pre-stated conditions; no C0 outcome is used below.",
        "",
        "## Stage-wise outcomes",
        "",
        "| Stage | Method | Success | Collision | Timeout | Agent completion | Planning / episode (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "stage_1": "Stage I",
        "stage_2": "Stage II",
        "stage_3": "Stage III",
        "stage_4": "Stage IV",
        "overall": "Overall",
    }
    names = {METHOD_ONE_SHOT: "Proposal one-shot", METHOD_ERR_LITE: "Proposal ERR-lite"}
    for scope in ["stage_1", "stage_2", "stage_3", "stage_4", "overall"]:
        for method in METHODS:
            row = index[(method, scope)]
            lines.append(
                f"| {labels[scope]} | {names[method]} | {row['team_success_count']}/{row['episode_count']} ({row['team_success_rate']:.1%}) | {row['any_collision_rate']:.1%} | {row['timeout_rate']:.1%} | {row['agent_completion_rate']:.1%} | {row['planning_runtime_ms']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Trigger and paired analysis",
            "",
            f"ERR-lite executed {err['replan_event_count']} per-agent Proposal replans: {err['reference_reached_trigger_count']} reference-reached triggers ({err['direct_safe_terminal_handoff_count']} handed directly to the terminal through the existing Proposal safety-field gate), {err['stagnation_trigger_count']} stagnation triggers, and {err['terminal_retry_trigger_count']} terminal-fallback retries. {err['unchanged_replan_count']} replans returned an unchanged active goal.",
            "",
            f"Paired outcomes were: both success {paired['both_success']}, ERR-lite only {paired['err_lite_only_success']}, one-shot only {paired['one_shot_only_success']}, both fail {paired['both_fail']}. Exact two-sided McNemar p={paired['mcnemar_two_sided_exact_p']:.6f}.",
            "",
            "## Failure causes",
            "",
            "| Method | Success | Obstacle collision | Inter-agent collision | Timeout | Other |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        counts = Counter(
            failure_category(row) for row in episodes if row["method"] == method
        )
        lines.append(
            f"| {names[method]} | {counts['SUCCESS']} | {counts['OBSTACLE_COLLISION']} | {counts['INTER_AGENT_COLLISION']} | {counts['TIMEOUT']} | {counts['OTHER']} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This experiment may select a lightweight mechanism for further study, but it cannot replace or amend the frozen 400-scene formal benchmark. No formal result file was read or modified.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(config_path: Path, output_dir: Path) -> Path:
    config = load_json(config_path)
    manifest_path = REPO_ROOT / config["sources"]["development_manifest"]
    manifest = load_json(manifest_path)
    if int(manifest["unique_scenario_count"]) != 40:
        raise RuntimeError("development manifest must contain exactly 40 scenes")
    if any(not str(row["scenario_id"]).startswith("D") for row in manifest["entries"]):
        raise RuntimeError("non-development scenario detected")
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_config = copy.deepcopy(config)
    frozen_config["created_at"] = datetime.now().astimezone().isoformat()
    frozen_config["resolved_output_dir"] = str(output_dir.resolve())
    frozen_config["development_manifest_sha256"] = sha256_file(manifest_path)
    write_json(output_dir / "config.json", frozen_config)
    runtime = build_runtime(config, manifest)
    episodes: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    total = len(manifest["entries"]) * len(METHODS)
    completed = 0
    for entry in manifest["entries"]:
        for method in METHODS:
            record_path = (
                output_dir / "records" / method / f"{entry['scenario_id']}.json"
            )
            if record_path.exists():
                payload = load_json(record_path)
                episode = payload["episode"]
                agent_rows = payload["agents"]
                event_rows = payload["events"]
            else:
                episode, agent_rows, event_rows = run_episode(
                    config=config,
                    runtime=runtime,
                    entry=entry,
                    method=method,
                )
                write_json(
                    record_path,
                    {"episode": episode, "agents": agent_rows, "events": event_rows},
                )
            episodes.append(episode)
            agents.extend(agent_rows)
            events.extend(event_rows)
            completed += 1
            print(
                f"[{completed}/{total}] {entry['scenario_id']} {method}: "
                f"{episode['termination_reason']} replans={episode['replan_event_count']}",
                flush=True,
            )
    summary = build_summary(episodes)
    paired = paired_analysis(episodes)
    write_csv(output_dir / "episode_results.csv", episodes)
    write_csv(output_dir / "agent_results.csv", agents)
    write_csv(output_dir / "event_log.csv", events)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "paired_results.csv", paired["rows"])
    write_json(
        output_dir / "summary.json",
        {"summary": summary, "paired": {k: v for k, v in paired.items() if k != "rows"}},
    )
    overall = {(row["method"]): row for row in summary if row["scope"] == "overall"}
    one = overall[METHOD_ONE_SHOT]
    err = overall[METHOD_ERR_LITE]
    conclusion = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": config.get("candidate_id"),
        "split": "development_only",
        "scene_count": 40,
        "formal_scene_count_read": 0,
        "one_shot_success_rate": one["team_success_rate"],
        "err_lite_success_rate": err["team_success_rate"],
        "success_gain_pp": 100.0
        * (err["team_success_rate"] - one["team_success_rate"]),
        "collision_change_pp": 100.0
        * (err["any_collision_rate"] - one["any_collision_rate"]),
        "timeout_change_pp": 100.0
        * (err["timeout_rate"] - one["timeout_rate"]),
        "replan_event_count": err["replan_event_count"],
        "ERR_LITE_DEVELOPMENT_RESULT": (
            "PROMISING"
            if err["team_success_rate"] > one["team_success_rate"]
            and err["any_collision_rate"] <= one["any_collision_rate"]
            else "NO_CLEAR_GAIN"
        ),
        "FORMAL_CONCLUSION_CHANGED": "NO",
    }
    write_json(output_dir / "conclusion.json", conclusion)
    (output_dir / "FINAL_REPORT.md").write_text(
        render_report(config, summary, paired, episodes), encoding="utf-8"
    )
    write_json(
        output_dir / "integrity.json",
        {
            "schema_version": SCHEMA_VERSION,
            "episode_count": len(episodes),
            "agent_count": len(agents),
            "unique_scenario_count": len({row["scenario_id"] for row in episodes}),
            "development_ids_only": all(
                str(row["scenario_id"]).startswith("D") for row in episodes
            ),
            "pair_complete": all(
                sum(row["scenario_id"] == scenario for row in episodes) == 2
                for scenario in {row["scenario_id"] for row in episodes}
            ),
            "checkpoint_sha256": runtime["checkpoint_hash"],
            "checkpoint_hash_match": runtime["checkpoint_hash"]
            == config["sources"]["sac_checkpoint_sha256_expected"],
            "historical_gate_all_verified": all(
                row["historical_gate_verified"] for row in episodes
            ),
            "terminal_goals_all_unchanged": all(
                row["terminal_task_goals_unchanged"] for row in episodes
            ),
            "max_phase_switch_delta": max(
                row["maximum_phase_switch_delta"] for row in episodes
            ),
            "formal_artifact_read": False,
        },
    )
    grouped = defaultdict(dict)
    for row in episodes:
        grouped[str(row["scenario_id"])][str(row["method"])] = row
    reconciliation = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": config.get("candidate_id"),
        "status": "PASSED",
        "checks": {
            "episode_rows_80": len(episodes) == 80,
            "agent_rows_240": len(agents) == 240,
            "scenario_pairs_40": len(grouped) == 40
            and all(set(value) == set(METHODS) for value in grouped.values()),
            "development_prefix_only": all(
                str(row["scenario_id"]).startswith("D") for row in episodes
            ),
            "one_shot_success_reproduced": sum(
                row["method"] == METHOD_ONE_SHOT and row["team_success"]
                for row in episodes
            )
            == one["team_success_count"],
            "err_lite_success_reproduced": sum(
                row["method"] == METHOD_ERR_LITE and row["team_success"]
                for row in episodes
            )
            == err["team_success_count"],
            "checkpoint_hash_match": runtime["checkpoint_hash"]
            == config["sources"]["sac_checkpoint_sha256_expected"],
            "historical_gate_verified": all(
                row["historical_gate_verified"] for row in episodes
            ),
            "phase_preserved": all(
                float(row["maximum_phase_switch_delta"]) == 0.0
                for row in episodes
            ),
            "terminal_goals_immutable": all(
                row["terminal_task_goals_unchanged"] for row in episodes
            ),
        },
        "formal_result_rows_read": 0,
    }
    if not all(reconciliation["checks"].values()):
        reconciliation["status"] = "FAILED"
    write_json(output_dir / "reconciliation.json", reconciliation)
    return output_dir / "FINAL_REPORT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / config["output_root"] / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    report = run_experiment(config_path, output_dir.resolve())
    print(json.dumps({"status": "complete", "report": str(report)}), flush=True)


if __name__ == "__main__":
    main()
