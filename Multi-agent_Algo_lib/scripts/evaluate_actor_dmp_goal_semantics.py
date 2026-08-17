"""Run the paired Actor--DMP goal-semantics 2x2 diagnosis.

The evaluator changes neither the frozen Actor nor SAC-DMP dynamics.  Each
variant differs only in which goal is used to construct the historical 122-D
Actor input and which goal is stored in ``dmp.goal``.
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
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402
from planning.horizon_agent_setting_diagnosis import stable_hash  # noqa: E402
from planning.goal_semantics_diagnosis import (  # noqa: E402
    VARIANT_A,
    VARIANT_B,
    VARIANT_C,
    VARIANT_D,
    VARIANT_DISPLAY_NAMES,
    VARIANT_ORDER,
    VARIANT_SEMANTICS,
    action_saturation_mask,
    actor_goal_shift_diagnostic,
    dmp_attractor_shift_diagnostic,
    temporary_checkpoint_observations,
    terminal_checkpoint_observations,
)
from planning.pre_gat_closed_loop import (  # noqa: E402
    METHOD_PROPOSAL,
    FPSHEPOnlineScoreSpec,
    choose_execution_reference,
    generate_candidate_set,
    termination_reason,
    trajectory_metrics,
)
from planning.policy_preview import point_to_segment_distance  # noqa: E402
from planning.multi_agent_obstacle_scenario_audit import (  # noqa: E402
    minimum_static_surface_clearances,
)
from scripts.evaluate_frozen_policy_waypoint_guidance import (  # noqa: E402
    predict_actions_without_postprocessing,
    set_dmp_active_goal_preserve_phase,
)
from scripts.evaluate_pre_gat_closed_loop import (  # noqa: E402
    _critical_hashes,
    _jsonable,
    _policy_parameter_sha256,
    _scenario_hash,
    _scene_snapshot,
    build_closed_loop_environment,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


DEFAULT_CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "evaluation"
    / "actor_dmp_goal_semantics_diagnosis.json"
)
SCHEMA_VERSION = "actor_dmp_goal_semantics_diagnosis_v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    encoded_rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not encoded_rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in encoded_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in encoded_rows:
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


def _reference_hash(references: np.ndarray, available: np.ndarray) -> str:
    payload = {
        "references": np.asarray(references, dtype=float).round(12).tolist(),
        "available": np.asarray(available, dtype=bool).tolist(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _select_one_shot_references(
    env: Any,
    *,
    proposal_config: ProposalConfig,
    requested_k: int,
    candidate_sets: Sequence[Sequence[Any]] | None = None,
    selection_method: str = METHOD_PROPOSAL,
    policy: Any | None = None,
    score_spec: FPSHEPOnlineScoreSpec | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Select one-shot references from generated or injected t=0 candidates."""

    terminal_goals = np.asarray(env.goals, dtype=float)
    references = terminal_goals.copy()
    available = np.zeros(int(env.num_agents), dtype=bool)
    records: list[dict[str, Any]] = []
    for agent_index in range(int(env.num_agents)):
        if candidate_sets is None:
            proposals, proposal_count_before_consumer = generate_candidate_set(
                env,
                agent_index,
                proposal_config,
                consumer_top_k=int(requested_k),
            )
        else:
            if len(candidate_sets) != int(env.num_agents):
                raise ValueError("candidate_sets must contain one sequence per agent")
            proposals = tuple(candidate_sets[agent_index])
            proposal_count_before_consumer = len(proposals)
        decision = choose_execution_reference(
            method=selection_method,
            terminal_task_goal=terminal_goals[agent_index],
            proposals=proposals,
            policy=policy,
            env=env,
            agent_index=agent_index,
            score_spec=score_spec,
        )
        if decision.selected_candidate_id is not None:
            references[agent_index] = np.asarray(
                decision.execution_reference, dtype=float
            )
            available[agent_index] = True
        records.append(
            {
                "agent_id": int(agent_index),
                "candidate_available": bool(available[agent_index]),
                "temporary_reference": references[agent_index].tolist(),
                "selected_candidate_id": decision.selected_candidate_id,
                "selection_source": decision.selection_kind,
                "no_candidate_fallback": bool(decision.no_candidate_fallback),
                "proposal_count_before_consumer": int(
                    proposal_count_before_consumer
                ),
                "K_t": int(len(proposals)),
                "candidate_world_points": [
                    np.asarray(item.point, dtype=float).tolist() for item in proposals
                ],
                "proposal_scores": [float(item.score) for item in proposals],
                "candidate_metadata": [
                    getattr(item, "metadata", {}) for item in proposals
                ],
                "selected_proposal_rank": decision.selected_candidate_original_index,
                "selected_proposal_score": (
                    float(proposals[decision.selected_candidate_id].score)
                    if decision.selected_candidate_id is not None
                    else None
                ),
                "fp_shep_candidate_records": [
                    item.to_record() for item in decision.fp_shep_records
                ],
                "selected_fp_shep_score": (
                    float(decision.fp_shep_records[decision.selected_candidate_id].score)
                    if decision.selected_candidate_id is not None
                    and decision.fp_shep_records
                    else None
                ),
            }
        )
    return references, available, records


def _terminal_progress(
    starts: np.ndarray,
    final_positions: np.ndarray,
    terminal_goals: np.ndarray,
) -> tuple[float, list[float]]:
    initial = np.linalg.norm(terminal_goals - starts, axis=1)
    final = np.linalg.norm(terminal_goals - final_positions, axis=1)
    values = initial - final
    return float(np.mean(values)), values.astype(float).tolist()


def _variant_pair_id(scenario: str, seed: int, variant: str) -> str:
    return f"diagnostic__{scenario}__seed{int(seed):03d}__{variant}"


def _multi_physical_state_hash(env: Any) -> str:
    """Hash the horizon-independent physical/sensor state of a multi-agent run."""

    return stable_hash(
        {
            "steps": int(env.steps),
            "positions": env._positions(),
            "velocities": env._velocities(),
            "phases": [float(dmp.phase) for dmp in env.dmps],
            "current_scans": [
                packet.current_scan for packet in env.latest_sensor_packets
            ],
            "previous_scans": [
                packet.previous_scan for packet in env.latest_sensor_packets
            ],
            "success_rewarded_mask": env.success_rewarded_mask,
        }
    )


def run_variant_episode(
    *,
    policy: Any,
    multi_config: Any,
    settings: Mapping[str, Any],
    scenario: str,
    seed: int,
    variant: str,
    candidate_sets: Sequence[Sequence[Any]] | None = None,
    selection_method: str = METHOD_PROPOSAL,
    score_spec: FPSHEPOnlineScoreSpec | None = None,
    environment_builder: Callable[..., tuple[Any, dict[str, Any]]] | None = None,
    selection_plan: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one 2x2 variant from a deterministic fresh environment."""

    if variant not in VARIANT_SEMANTICS:
        raise ValueError(f"unknown goal-semantics variant: {variant}")
    semantics = VARIANT_SEMANTICS[variant]
    builder = environment_builder or build_closed_loop_environment
    env, scene_metadata = builder(
        config=multi_config,
        scenario=scenario,
        seed=int(seed),
        peer_radius=float(settings["peer_radius"]),
    )
    pair_id = _variant_pair_id(scenario, seed, variant)
    actor_shift_rows: list[dict[str, Any]] = []
    dmp_shift_rows: list[dict[str, Any]] = []
    episode_started = time.perf_counter_ns()
    try:
        initial_snapshot = _scene_snapshot(env)
        initial_condition_hash = _scenario_hash(initial_snapshot)
        starts = np.asarray(env.starts, dtype=float).copy()
        terminal_goals = np.asarray(env.goals, dtype=float).copy()
        initial_phases = np.asarray([dmp.phase for dmp in env.dmps], dtype=float)
        if env.get_observation().shape[1] != int(
            settings["checkpoint_observation"]["environment_native_observation_dimension"]
        ):
            raise RuntimeError("native environment observation dimension audit changed")

        temporary_references = terminal_goals.copy()
        temporary_available = np.zeros(int(env.num_agents), dtype=bool)
        candidate_records: list[dict[str, Any]] = []
        if variant != VARIANT_A:
            if selection_plan is None:
                temporary_references, temporary_available, candidate_records = (
                    _select_one_shot_references(
                        env,
                        proposal_config=ProposalConfig(**dict(settings["proposal_config"])),
                        requested_k=int(settings["temporary_reference"]["K_requested"]),
                        candidate_sets=candidate_sets,
                        selection_method=selection_method,
                        policy=policy,
                        score_spec=score_spec,
                    )
                )
            else:
                temporary_references = np.asarray(
                    selection_plan["references"], dtype=float
                ).copy()
                temporary_available = np.asarray(
                    selection_plan["available"], dtype=bool
                ).copy()
                candidate_records = copy.deepcopy(
                    list(selection_plan["candidate_records"])
                )
                expected_reference_shape = (int(env.num_agents), 3)
                if temporary_references.shape != expected_reference_shape:
                    raise ValueError(
                        "selection plan references must have shape "
                        f"{expected_reference_shape}"
                    )
                if temporary_available.shape != (int(env.num_agents),):
                    raise ValueError(
                        "selection plan availability must have one value per agent"
                    )
                if len(candidate_records) != int(env.num_agents):
                    raise ValueError(
                        "selection plan must contain one candidate record per agent"
                    )
                if not np.all(np.isfinite(temporary_references)):
                    raise ValueError("selection plan references must be finite")
                if not np.allclose(
                    temporary_references[~temporary_available],
                    terminal_goals[~temporary_available],
                    rtol=0.0,
                    atol=0.0,
                ):
                    raise ValueError(
                        "selection plan unavailable branches must use terminal goals"
                    )
        temporary_reference_hash = (
            _reference_hash(temporary_references, temporary_available)
            if variant != VARIANT_A
            else None
        )

        returned_to_terminal = np.logical_not(temporary_available)
        reference_reached = np.zeros(int(env.num_agents), dtype=bool)
        reference_reached_step: list[int | None] = [None] * int(env.num_agents)
        terminal_success_step: list[int | None] = [None] * int(env.num_agents)
        phase_switch_deltas: list[float] = []
        if semantics.dmp_goal == "temporary":
            for agent_index in range(int(env.num_agents)):
                if not temporary_available[agent_index]:
                    continue
                phase_before = float(env.dmps[agent_index].phase)
                set_dmp_active_goal_preserve_phase(
                    env.dmps[agent_index], temporary_references[agent_index]
                )
                phase_switch_deltas.append(
                    float(env.dmps[agent_index].phase) - phase_before
                )

        positions = [env._positions().copy()]
        velocities = [env._velocities().copy()]
        accelerations: list[np.ndarray] = []
        applied_acceleration_norms: list[float] = []
        commanded_acceleration_norms: list[float] = []
        action_saturation_values: list[float] = []
        min_clearances = [
            float(min(packet.min_clearance for packet in env.latest_sensor_packets))
        ]
        min_inter_agent_distances = [
            float(env._check_collision()["min_inter_agent_distance"])
        ]
        collision = False
        obstacle_collision = False
        inter_agent_collision = False
        boundary_collision = False
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        state_at_step_50_hash: str | None = None
        final_obstacle_collision_mask = np.zeros(int(env.num_agents), dtype=bool)
        final_inter_agent_collision_mask = np.zeros(int(env.num_agents), dtype=bool)
        obstacle_collision_before_reference = np.zeros(int(env.num_agents), dtype=bool)
        inter_agent_collision_before_reference = np.zeros(int(env.num_agents), dtype=bool)
        collision_after_reference = np.zeros(int(env.num_agents), dtype=bool)
        stage1_min_clearance = np.asarray(
            [packet.min_clearance for packet in env.latest_sensor_packets], dtype=float
        )
        initial_pairwise = env._compute_pairwise_distances()
        stage1_min_inter_agent = np.asarray(
            [
                float(np.min(np.delete(initial_pairwise[index], index)))
                if int(env.num_agents) > 1
                else float("inf")
                for index in range(int(env.num_agents))
            ],
            dtype=float,
        )
        stage1_max_deviation = np.zeros(int(env.num_agents), dtype=float)
        minimum_static_clearance_by_agent = minimum_static_surface_clearances(
            env._positions(), env.static_obstacles
        )
        maximum_terminal_line_deviation = np.zeros(int(env.num_agents), dtype=float)
        obstacle_pressure_path_deviation = np.zeros(int(env.num_agents), dtype=float)
        initial_reference_distances = np.linalg.norm(
            temporary_references - starts, axis=1
        )
        reached_tolerance = float(
            settings["temporary_reference"]["reached_tolerance_m"]
        )
        saturation_tolerance = float(
            settings["action_saturation"]["relative_tolerance"]
        )

        while not (terminated or truncated):
            timestep = int(env.steps)
            terminal_observations = terminal_checkpoint_observations(env)
            if semantics.actor_goal == "temporary":
                actor_references = temporary_references.copy()
                if variant == VARIANT_D:
                    actor_references[returned_to_terminal] = terminal_goals[
                        returned_to_terminal
                    ]
                observations = temporary_checkpoint_observations(
                    env, actor_references
                )
            else:
                observations = terminal_observations
            actions = predict_actions_without_postprocessing(
                policy, observations, tuple(env.action_shape)
            )

            if variant == VARIANT_B:
                for agent_index in range(int(env.num_agents)):
                    if not temporary_available[agent_index]:
                        continue
                    diagnostic = actor_goal_shift_diagnostic(
                        env,
                        agent_index,
                        temporary_reference=temporary_references[agent_index],
                        policy=policy,
                        saturation_relative_tolerance=saturation_tolerance,
                    )
                    actor_shift_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "pair_id": pair_id,
                            "variant": variant,
                            "scenario": scenario,
                            "seed": int(seed),
                            "timestep": timestep,
                            "agent_id": int(agent_index),
                            **diagnostic,
                        }
                    )
            if variant == VARIANT_C:
                for agent_index in range(int(env.num_agents)):
                    if returned_to_terminal[agent_index]:
                        continue
                    diagnostic = dmp_attractor_shift_diagnostic(
                        env,
                        agent_index,
                        temporary_reference=temporary_references[agent_index],
                        shared_terminal_action=actions[agent_index],
                    )
                    dmp_shift_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "pair_id": pair_id,
                            "variant": variant,
                            "scenario": scenario,
                            "seed": int(seed),
                            "timestep": timestep,
                            "agent_id": int(agent_index),
                            **diagnostic,
                        }
                    )

            saturation_mask = action_saturation_mask(
                actions,
                env.action_space.low,
                env.action_space.high,
                relative_tolerance=saturation_tolerance,
            )
            action_saturation_values.extend(
                saturation_mask.astype(float).reshape(-1).tolist()
            )
            _, _, terminated, truncated, info = env.step(actions)
            positions.append(env._positions().copy())
            velocities.append(env._velocities().copy())
            applied = np.asarray(info["applied_accelerations"], dtype=float)
            commanded = np.asarray(info["commanded_accelerations"], dtype=float)
            accelerations.append(applied.copy())
            applied_acceleration_norms.extend(
                np.linalg.norm(applied, axis=1).astype(float).tolist()
            )
            commanded_acceleration_norms.extend(
                np.linalg.norm(commanded, axis=1).astype(float).tolist()
            )
            min_clearances.append(float(np.min(info["min_clearances"])))
            min_inter_agent_distances.append(
                float(info["min_inter_agent_distance"])
            )
            collision |= bool(info["collision"])
            obstacle_collision |= bool(np.any(info["obstacle_collision_mask"]))
            inter_agent_collision |= bool(
                np.any(info["inter_agent_collision_mask"])
            )
            final_obstacle_collision_mask = np.asarray(
                info["obstacle_collision_mask"], dtype=bool
            ).copy()
            final_inter_agent_collision_mask = np.asarray(
                info["inter_agent_collision_mask"], dtype=bool
            ).copy()
            step_clearances = np.asarray(info["min_clearances"], dtype=float)
            pairwise = env._compute_pairwise_distances()
            pure_static_clearance = minimum_static_surface_clearances(
                env._positions(), env.static_obstacles
            )
            minimum_static_clearance_by_agent = np.minimum(
                minimum_static_clearance_by_agent, pure_static_clearance
            )
            terminal_line_deviation = np.asarray(
                [
                    point_to_segment_distance(
                        env.dynamics[index].p,
                        starts[index],
                        terminal_goals[index],
                    )
                    for index in range(int(env.num_agents))
                ],
                dtype=float,
            )
            maximum_terminal_line_deviation = np.maximum(
                maximum_terminal_line_deviation, terminal_line_deviation
            )
            obstacle_pressure_mask = pure_static_clearance <= float(
                env.env_config.obstacle_influence_distance
            )
            obstacle_pressure_path_deviation[obstacle_pressure_mask] = np.maximum(
                obstacle_pressure_path_deviation[obstacle_pressure_mask],
                terminal_line_deviation[obstacle_pressure_mask],
            )
            for agent_index in range(int(env.num_agents)):
                before_reference = bool(
                    temporary_available[agent_index]
                    and not reference_reached[agent_index]
                )
                if before_reference:
                    stage1_min_clearance[agent_index] = min(
                        stage1_min_clearance[agent_index],
                        float(step_clearances[agent_index]),
                    )
                    if int(env.num_agents) > 1:
                        stage1_min_inter_agent[agent_index] = min(
                            stage1_min_inter_agent[agent_index],
                            float(np.min(np.delete(pairwise[agent_index], agent_index))),
                        )
                    stage1_max_deviation[agent_index] = max(
                        stage1_max_deviation[agent_index],
                        point_to_segment_distance(
                            env.dynamics[agent_index].p,
                            starts[agent_index],
                            temporary_references[agent_index],
                        ),
                    )
                    obstacle_collision_before_reference[agent_index] |= bool(
                        final_obstacle_collision_mask[agent_index]
                    )
                    inter_agent_collision_before_reference[agent_index] |= bool(
                        final_inter_agent_collision_mask[agent_index]
                    )
                elif reference_reached[agent_index]:
                    collision_after_reference[agent_index] |= bool(
                        final_obstacle_collision_mask[agent_index]
                        or final_inter_agent_collision_mask[agent_index]
                    )
            boundary_collision |= bool(np.any(info["boundary_collision_mask"]))
            success_mask = np.asarray(info["success_mask"], dtype=bool)
            for agent_index in np.flatnonzero(success_mask):
                if terminal_success_step[int(agent_index)] is None:
                    terminal_success_step[int(agent_index)] = int(env.steps)

            if variant == VARIANT_B:
                for agent_index in range(int(env.num_agents)):
                    if (
                        not temporary_available[agent_index]
                        or reference_reached[agent_index]
                    ):
                        continue
                    distance = float(
                        np.linalg.norm(
                            temporary_references[agent_index]
                            - env.dynamics[agent_index].p
                        )
                    )
                    if distance <= reached_tolerance:
                        reference_reached[agent_index] = True
                        reference_reached_step[agent_index] = int(env.steps)

            if (
                variant in {VARIANT_C, VARIANT_D}
                and not (terminated or truncated)
            ):
                for agent_index in range(int(env.num_agents)):
                    if returned_to_terminal[agent_index]:
                        continue
                    distance = float(
                        np.linalg.norm(
                            temporary_references[agent_index]
                            - env.dynamics[agent_index].p
                        )
                    )
                    if distance > reached_tolerance:
                        continue
                    reference_reached[agent_index] = True
                    reference_reached_step[agent_index] = int(env.steps)
                    phase_before = float(env.dmps[agent_index].phase)
                    set_dmp_active_goal_preserve_phase(
                        env.dmps[agent_index], terminal_goals[agent_index]
                    )
                    phase_switch_deltas.append(
                        float(env.dmps[agent_index].phase) - phase_before
                    )
                    returned_to_terminal[agent_index] = True

            if int(env.steps) == 50:
                state_at_step_50_hash = _multi_physical_state_hash(env)

        position_array = np.stack(positions)
        velocity_array = np.stack(velocities)
        acceleration_array = np.stack(accelerations)
        trajectory = trajectory_metrics(
            position_array,
            velocity_array,
            acceleration_array,
            dt=float(settings["dt"]),
        )
        speed_norms = np.linalg.norm(velocity_array, axis=2)
        acceleration_norms = np.linalg.norm(acceleration_array, axis=2)
        final_positions = position_array[-1]
        terminal_progress, per_agent_progress = _terminal_progress(
            starts, final_positions, terminal_goals
        )
        success = bool(info.get("success", False))
        final_state_hash = _multi_physical_state_hash(env)
        status = termination_reason(
            success=success,
            collision=collision,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        final_dmp_goals = np.stack(
            [np.asarray(dmp.goal, dtype=float) for dmp in env.dmps]
        )
        terminal_goals_unchanged = bool(np.array_equal(env.goals, terminal_goals))
        if not terminal_goals_unchanged:
            raise RuntimeError("variant modified env.goals")
        if phase_switch_deltas and not np.allclose(
            phase_switch_deltas, 0.0, rtol=0.0, atol=0.0
        ):
            raise RuntimeError("goal switch reset or changed DMP phase")
        expected_terminal_dmp = variant in {VARIANT_A, VARIANT_B}
        if expected_terminal_dmp and not np.allclose(
            final_dmp_goals, terminal_goals, rtol=0.0, atol=0.0
        ):
            raise RuntimeError("terminal-attractor variant modified dmp.goal")

        reached_then_terminal_mask = np.asarray(
            [
                reached_step is not None
                and completion_step is not None
                and int(completion_step) >= int(reached_step)
                for reached_step, completion_step in zip(
                    reference_reached_step, terminal_success_step, strict=True
                )
            ],
            dtype=bool,
        )
        collided_mask = np.logical_or(
            final_obstacle_collision_mask, final_inter_agent_collision_mask
        )
        collision_before_reference_count = int(
            np.sum(
                [
                    bool(collided_mask[index])
                    and reference_reached_step[index] is None
                    for index in range(int(env.num_agents))
                ]
            )
        )
        collision_after_reference_count = int(
            np.sum(
                [
                    bool(collided_mask[index])
                    and reference_reached_step[index] is not None
                    for index in range(int(env.num_agents))
                ]
            )
        )
        remaining_steps_after_reference = [
            int(env.env_config.max_steps) - int(step)
            for step in reference_reached_step
            if step is not None
        ]
        final_reference_distances = np.linalg.norm(
            temporary_references - final_positions, axis=1
        )
        agent_stage_records: list[dict[str, Any]] = []
        for agent_index in range(int(env.num_agents)):
            candidate_record = (
                candidate_records[agent_index]
                if agent_index < len(candidate_records)
                else {}
            )
            reached_step = reference_reached_step[agent_index]
            completion_step = terminal_success_step[agent_index]
            completed_after_reference = bool(
                reached_step is not None
                and completion_step is not None
                and int(completion_step) >= int(reached_step)
            )
            stage1_end_distance = (
                0.0
                if reached_step is not None
                else float(final_reference_distances[agent_index])
            )
            agent_stage_records.append(
                {
                    "agent_id": int(agent_index),
                    "reference_available": bool(temporary_available[agent_index]),
                    "no_candidate_fallback": bool(
                        candidate_record.get("no_candidate_fallback", False)
                    ),
                    "reference_reached": bool(reference_reached[agent_index]),
                    "reference_reached_step": reached_step,
                    "initial_distance_to_reference_m": float(
                        initial_reference_distances[agent_index]
                    ),
                    "distance_to_reference_at_termination_m": float(
                        final_reference_distances[agent_index]
                    ),
                    "stage1_real_progress_m": float(
                        initial_reference_distances[agent_index] - stage1_end_distance
                    ),
                    "stage1_minimum_obstacle_clearance_m": float(
                        stage1_min_clearance[agent_index]
                    ),
                    "stage1_minimum_inter_agent_distance_m": float(
                        stage1_min_inter_agent[agent_index]
                    ),
                    "stage1_max_execution_deviation_m": float(
                        stage1_max_deviation[agent_index]
                    ),
                    "minimum_static_obstacle_clearance_m": float(
                        minimum_static_clearance_by_agent[agent_index]
                    ),
                    "static_obstacle_interaction": bool(
                        minimum_static_clearance_by_agent[agent_index]
                        <= float(env.env_config.obstacle_influence_distance)
                    ),
                    "maximum_terminal_line_deviation_m": float(
                        maximum_terminal_line_deviation[agent_index]
                    ),
                    "obstacle_pressure_path_deviation_m": float(
                        obstacle_pressure_path_deviation[agent_index]
                    ),
                    "obstacle_collision_before_reference": bool(
                        obstacle_collision_before_reference[agent_index]
                    ),
                    "inter_agent_collision_before_reference": bool(
                        inter_agent_collision_before_reference[agent_index]
                    ),
                    "collision_before_reference": bool(
                        obstacle_collision_before_reference[agent_index]
                        or inter_agent_collision_before_reference[agent_index]
                    ),
                    "reference_timeout": bool(
                        truncated
                        and temporary_available[agent_index]
                        and not reference_reached[agent_index]
                    ),
                    "terminal_completed_step": completion_step,
                    "terminal_completed_after_reference": completed_after_reference,
                    "stage2_completion_steps": (
                        int(completion_step) - int(reached_step)
                        if completed_after_reference
                        else None
                    ),
                    "remaining_steps_after_reference": (
                        int(env.env_config.max_steps) - int(reached_step)
                        if reached_step is not None
                        else None
                    ),
                    "collision_after_reference": bool(
                        collision_after_reference[agent_index]
                    ),
                    "timeout_after_reference": bool(
                        truncated
                        and reached_step is not None
                        and not completed_after_reference
                    ),
                    "team_success": bool(success),
                    "team_timeout": bool(truncated),
                    **candidate_record,
                }
            )

        episode = {
            "schema_version": SCHEMA_VERSION,
            "pair_id": pair_id,
            "variant": variant,
            "variant_display_name": VARIANT_DISPLAY_NAMES[variant],
            "actor_goal_semantics": semantics.actor_goal,
            "dmp_goal_semantics": semantics.dmp_goal,
            "scenario": scenario,
            "seed": int(seed),
            "initial_condition_hash": initial_condition_hash,
            "temporary_reference_hash": temporary_reference_hash,
            "temporary_references": temporary_references.tolist()
            if variant != VARIANT_A
            else None,
            "terminal_task_goals": terminal_goals.tolist(),
            "temporary_reference_available": temporary_available.tolist()
            if variant != VARIANT_A
            else None,
            "candidate_records": candidate_records,
            "agent_stage_records": agent_stage_records,
            "selection_method": selection_method if variant != VARIANT_A else None,
            "temporary_reference_count": int(np.sum(temporary_available)),
            "temporary_reference_reached_count": int(np.sum(reference_reached)),
            "temporary_reference_reached_rate": float(
                np.sum(reference_reached) / max(1, np.sum(temporary_available))
            )
            if variant != VARIANT_A
            else None,
            "reference_reached_steps": reference_reached_step,
            "terminal_success_steps": terminal_success_step,
            "terminal_success_count": int(
                sum(step is not None for step in terminal_success_step)
            ),
            "reached_then_terminal_completion_count": int(
                np.sum(reached_then_terminal_mask)
            ),
            "reached_then_terminal_completion_rate": float(
                np.sum(reached_then_terminal_mask)
                / max(1, np.sum(reference_reached))
            ),
            "team_stage1_success": bool(
                np.sum(temporary_available) > 0
                and np.all(reference_reached[temporary_available])
            ),
            "team_terminal_completion_step": int(env.steps) if success else None,
            "remaining_steps_after_reference": remaining_steps_after_reference,
            "success": success,
            "collision": collision,
            "obstacle_collision": obstacle_collision,
            "inter_agent_collision": inter_agent_collision,
            "boundary_collision": boundary_collision,
            "obstacle_collision_mask": final_obstacle_collision_mask.tolist(),
            "inter_agent_collision_mask": final_inter_agent_collision_mask.tolist(),
            "collision_before_reference_reached_count": (
                collision_before_reference_count
            ),
            "collision_after_reference_reached_count": (
                collision_after_reference_count
            ),
            "timeout_after_reference_reached": bool(
                truncated and np.any(reference_reached)
            ),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": status,
            "steps": int(env.steps),
            "path_length_team_mean_m": trajectory["path_length_team_mean"],
            "path_length_team_sum_m": trajectory["path_length_team_sum"],
            "path_length_per_agent_m": trajectory["path_lengths"].tolist(),
            "mean_speed_team_mps": float(np.mean(speed_norms)),
            "peak_speed_team_mps": float(np.max(speed_norms)),
            "mean_speed_per_agent_mps": np.mean(speed_norms, axis=0).tolist(),
            "peak_speed_per_agent_mps": np.max(speed_norms, axis=0).tolist(),
            "mean_acceleration_per_agent_mps2": np.mean(
                acceleration_norms, axis=0
            ).tolist(),
            "peak_acceleration_per_agent_mps2": np.max(
                acceleration_norms, axis=0
            ).tolist(),
            "heading_yaw_change_available": False,
            "heading_yaw_change_unavailable_reason": (
                "point_mass_model_has_no_yaw_or_body_heading_state"
            ),
            "trajectory_smoothness": trajectory[
                "trajectory_smoothness_team_mean"
            ],
            "terminal_progress_team_mean_m": terminal_progress,
            "terminal_progress_per_agent_m": per_agent_progress,
            "minimum_obstacle_clearance_m": float(np.min(min_clearances)),
            "minimum_obstacle_clearance_source": "composite_lidar_nearest_hit",
            "minimum_static_obstacle_clearance_m": float(
                np.min(minimum_static_clearance_by_agent)
            ),
            "minimum_static_obstacle_clearance_source": (
                "ground_truth_static_geometry_evaluation_only"
            ),
            "obstacle_interaction_agent_count": int(
                np.sum(
                    minimum_static_clearance_by_agent
                    <= float(env.env_config.obstacle_influence_distance)
                )
            ),
            "obstacle_interaction_episode": bool(
                np.any(
                    minimum_static_clearance_by_agent
                    <= float(env.env_config.obstacle_influence_distance)
                )
            ),
            "minimum_inter_agent_distance_m": float(
                np.min(min_inter_agent_distances)
            ),
            "inter_agent_interaction_episode": bool(
                np.min(min_inter_agent_distances)
                <= float(env.env_config.inter_agent_influence_distance)
            ),
            "combined_conflict_episode": bool(
                np.any(
                    minimum_static_clearance_by_agent
                    <= float(env.env_config.obstacle_influence_distance)
                )
                and np.min(min_inter_agent_distances)
                <= float(env.env_config.inter_agent_influence_distance)
            ),
            "maximum_terminal_line_deviation_team_mean_m": float(
                np.mean(maximum_terminal_line_deviation)
            ),
            "maximum_terminal_line_deviation_team_max_m": float(
                np.max(maximum_terminal_line_deviation)
            ),
            "obstacle_pressure_path_deviation_team_mean_m": float(
                np.mean(obstacle_pressure_path_deviation)
            ),
            "obstacle_pressure_path_deviation_team_max_m": float(
                np.max(obstacle_pressure_path_deviation)
            ),
            "mean_applied_acceleration_mps2": float(
                np.mean(applied_acceleration_norms)
            ),
            "max_applied_acceleration_mps2": float(
                np.max(applied_acceleration_norms)
            ),
            "mean_commanded_acceleration_mps2": float(
                np.mean(commanded_acceleration_norms)
            ),
            "max_commanded_acceleration_mps2": float(
                np.max(commanded_acceleration_norms)
            ),
            "action_saturation_rate": float(np.mean(action_saturation_values)),
            "terminal_task_goals_unchanged": terminal_goals_unchanged,
            "phase_reset_on_switch": False,
            "maximum_phase_switch_delta": float(
                np.max(np.abs(phase_switch_deltas))
            )
            if phase_switch_deltas
            else 0.0,
            "initial_phases": initial_phases.tolist(),
            "final_phases": [float(dmp.phase) for dmp in env.dmps],
            "final_dmp_goals": final_dmp_goals.tolist(),
            "forcing_gate_distance_source": "terminal_goal",
            "actor_observation_dimension": 122,
            "native_environment_observation_dimension": int(
                env.single_agent_observation_dim
            ),
            "native_environment_observation_used_for_actor": False,
            "proposal_selector": (
                str(selection_plan.get("selection_source", "preselected"))
                if selection_plan is not None
                else "Proposal score top-1"
                if variant != VARIANT_A
                else "none"
            ),
            "selection_plan_used": bool(selection_plan is not None),
            "boundary_filter_enabled": False,
            "episode_runtime_ms": float(
                (time.perf_counter_ns() - episode_started) / 1.0e6
            ),
            "GAT_used": bool(
                selection_plan is not None
                and selection_plan.get("selection_source") == "gat_stage1"
            ),
            "FP_SHEP_selector_used": bool(
                selection_plan is not None
                and selection_plan.get("selection_source") == "fp_shep_h4"
            ),
            "training_performed": False,
            "state_at_step_50_hash": state_at_step_50_hash,
            "final_state_hash": final_state_hash,
            **scene_metadata,
        }
        return episode, actor_shift_rows, dmp_shift_rows
    finally:
        env.close()


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def aggregate_protocol_summary(
    episodes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = (
        "path_length_team_mean_m",
        "trajectory_smoothness",
        "terminal_progress_team_mean_m",
        "minimum_obstacle_clearance_m",
        "minimum_inter_agent_distance_m",
        "mean_applied_acceleration_mps2",
        "max_applied_acceleration_mps2",
        "mean_commanded_acceleration_mps2",
        "max_commanded_acceleration_mps2",
        "action_saturation_rate",
        "episode_runtime_ms",
    )
    for variant in VARIANT_ORDER:
        variant_rows = [row for row in episodes if row["variant"] == variant]
        scenarios = ["overall"] + sorted(
            {str(row["scenario"]) for row in variant_rows}
        )
        for scenario in scenarios:
            members = (
                variant_rows
                if scenario == "overall"
                else [row for row in variant_rows if row["scenario"] == scenario]
            )
            row: dict[str, Any] = {
                "variant": variant,
                "variant_display_name": VARIANT_DISPLAY_NAMES[variant],
                "actor_goal_semantics": VARIANT_SEMANTICS[variant].actor_goal,
                "dmp_goal_semantics": VARIANT_SEMANTICS[variant].dmp_goal,
                "scenario": scenario,
                "episodes": len(members),
                "success_rate": _mean(members, "success"),
                "collision_rate": _mean(members, "collision"),
                "obstacle_collision_rate": _mean(members, "obstacle_collision"),
                "inter_agent_collision_rate": _mean(
                    members, "inter_agent_collision"
                ),
            }
            for metric in metrics:
                row[metric] = _mean(members, metric)
            rows.append(row)
    return rows


def validate_pairing(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in episodes:
        groups.setdefault((str(row["scenario"]), int(row["seed"])), []).append(row)
    for (scenario, seed), rows in groups.items():
        if {str(row["variant"]) for row in rows} != set(VARIANT_ORDER):
            errors.append(f"missing_variant:{scenario}:{seed}")
            continue
        initial_hashes = {str(row["initial_condition_hash"]) for row in rows}
        if len(initial_hashes) != 1:
            errors.append(f"initial_hash_mismatch:{scenario}:{seed}")
        reference_hashes = {
            str(row["temporary_reference_hash"])
            for row in rows
            if row["variant"] != VARIANT_A
        }
        if len(reference_hashes) != 1:
            errors.append(f"temporary_reference_mismatch:{scenario}:{seed}")
        for row in rows:
            if not bool(row["terminal_task_goals_unchanged"]):
                errors.append(f"terminal_goal_changed:{row['pair_id']}")
            if float(row["maximum_phase_switch_delta"]) != 0.0:
                errors.append(f"phase_switch_delta:{row['pair_id']}")
            if bool(row["include_boundaries_in_sensor"]):
                errors.append(f"boundary_sensor_enabled:{row['pair_id']}")
            if bool(row["terminate_on_boundary_collision"]):
                errors.append(f"boundary_termination_enabled:{row['pair_id']}")
    return {
        "status": "PASSED" if not errors else "FAILED",
        "errors": errors,
        "paired_state_count": len(groups),
        "episode_count": len(episodes),
    }


def run_experiment(
    settings: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_artifact = copy.deepcopy(dict(settings))
    config_artifact["output_dir_resolved"] = str(output_dir.resolve())
    write_json(output_dir / "config.json", config_artifact)
    multi_config = build_single_distribution_multi_config(
        num_agents=int(settings["num_agents"]),
        max_steps=int(settings["max_steps"]),
    )
    policy, checkpoint = _load_policy(settings, multi_config)
    policy_hash_before = _policy_parameter_sha256(policy)
    critical_hashes_before = _critical_hashes(checkpoint)
    episodes: list[dict[str, Any]] = []
    actor_shift_rows: list[dict[str, Any]] = []
    dmp_shift_rows: list[dict[str, Any]] = []
    jobs = [
        (scenario, int(seed), variant)
        for scenario in settings["scenarios"]
        for seed in settings["seeds"]
        for variant in VARIANT_ORDER
    ]
    started = time.perf_counter()
    for index, (scenario, seed, variant) in enumerate(jobs, start=1):
        episode, actor_rows, dmp_rows = run_variant_episode(
            policy=policy,
            multi_config=multi_config,
            settings=settings,
            scenario=str(scenario),
            seed=int(seed),
            variant=variant,
        )
        episodes.append(episode)
        actor_shift_rows.extend(actor_rows)
        dmp_shift_rows.extend(dmp_rows)
        print(
            f"[{index}/{len(jobs)}] {scenario} seed={seed} "
            f"{VARIANT_DISPLAY_NAMES[variant]}: {episode['termination_reason']}",
            flush=True,
        )

    summary = aggregate_protocol_summary(episodes)
    pairing = validate_pairing(episodes)
    policy_hash_after = _policy_parameter_sha256(policy)
    critical_hashes_after = _critical_hashes(checkpoint)
    integrity = {
        "policy_parameter_sha256_before": policy_hash_before,
        "policy_parameter_sha256_after": policy_hash_after,
        "policy_parameters_unchanged": policy_hash_before == policy_hash_after,
        "critical_hashes_before": critical_hashes_before,
        "critical_hashes_after": critical_hashes_after,
        "critical_files_unchanged": critical_hashes_before == critical_hashes_after,
        "training_performed": False,
        "GAT_used": False,
        "FP_SHEP_selector_used": False,
    }
    if pairing["status"] != "PASSED":
        raise RuntimeError(f"paired evaluation validation failed: {pairing['errors']}")
    if not integrity["policy_parameters_unchanged"] or not integrity[
        "critical_files_unchanged"
    ]:
        raise RuntimeError("diagnosis changed the frozen policy or a critical source")

    write_csv(output_dir / "per_episode.csv", episodes)
    write_json(output_dir / "per_episode.json", episodes)
    write_csv(output_dir / "protocol_summary.csv", summary)
    write_json(output_dir / "protocol_summary.json", summary)
    write_csv(output_dir / "actor_shift.csv", actor_shift_rows)
    write_json(output_dir / "actor_shift.json", actor_shift_rows)
    write_csv(output_dir / "dmp_shift.csv", dmp_shift_rows)
    write_json(output_dir / "dmp_shift.json", dmp_shift_rows)
    write_json(output_dir / "pairing_validation.json", pairing)
    write_json(output_dir / "integrity.json", integrity)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episode_count": len(episodes),
        "actor_shift_record_count": len(actor_shift_rows),
        "dmp_shift_record_count": len(dmp_shift_rows),
        "runtime_seconds": float(time.perf_counter() - started),
        "pairing": pairing,
        "integrity": integrity,
        "analysis_complete": False,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    config_path = args.config.resolve()
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPO_ROOT / str(settings["output_dir"]) / timestamp
    )
    run_experiment(settings, output_dir)
    print(output_dir, flush=True)
    return output_dir


if __name__ == "__main__":
    main()
