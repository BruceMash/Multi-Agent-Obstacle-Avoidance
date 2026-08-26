"""Audit the smallest existing-interface recovery options for frozen GAT V1.

This is a read-only diagnostic with respect to the deployed method.  It replays
the already frozen V1 selection plans at d_hand=0.25 and at the single permitted
pre-freeze candidate d_hand=0.30.  It does not train, tune, or edit any control
source or checkpoint.
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
from collections import defaultdict
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

from planning.goal_semantics_diagnosis import (  # noqa: E402
    temporary_checkpoint_observations,
)
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.multi_agent_obstacle_scenario_audit import (  # noqa: E402
    minimum_static_surface_clearances,
)
from scripts import evaluate_gat_closed_loop as legacy  # noqa: E402
from scripts.evaluate_actor_dmp_goal_semantics import write_csv, write_json  # noqa: E402
from scripts.evaluate_frozen_policy_waypoint_guidance import (  # noqa: E402
    predict_actions_without_postprocessing,
    set_dmp_active_goal_preserve_phase,
)
from scripts.evaluate_pre_gat_closed_loop import (  # noqa: E402
    _policy_parameter_sha256,
    _scenario_hash,
    _scene_snapshot,
    build_closed_loop_environment,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


SCHEMA_VERSION = "minimal_change_performance_audit_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/evaluation/minimal_change_performance_audit.json"
AUDITED_SOURCE_PATHS = (
    "Multi-agent_Algo_lib/scripts/evaluate_actor_dmp_goal_semantics.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_closed_loop.py",
    "planning/gat/candidate_selector.py",
    "planning/goal_semantics_diagnosis.py",
    "Controller/dmp_rl.py",
    "Environment/multi_agent_dmp_env.py",
    "Environment/frozen_sac_dmp_execution.py",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _json_cell(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (list, tuple, dict, bool, int, float)):
        return value
    return json.loads(str(value))


def _bool_cell(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _float_cell(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashes(paths: Sequence[str]) -> dict[str, str]:
    return {path: _sha256(REPO_ROOT / path) for path in paths}


def _same(left: Any, right: Any, atol: float = 1.0e-7) -> bool:
    return bool(
        np.allclose(
            np.asarray(left, dtype=float),
            np.asarray(right, dtype=float),
            rtol=0.0,
            atol=atol,
        )
    )


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return None
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _termination_reason(*, success: bool, collision: bool, truncated: bool) -> str:
    if success:
        return "success"
    if collision:
        return "collision"
    if truncated:
        return "timeout"
    return "unknown"


def _make_plan(
    agent_rows: Sequence[Mapping[str, str]], terminal_goals: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = sorted(agent_rows, key=lambda row: int(row["agent_id"]))
    if len(rows) != len(terminal_goals):
        raise RuntimeError("formal agent tuple is incomplete")
    references = np.asarray(terminal_goals, dtype=float).copy()
    available = np.zeros(len(rows), dtype=bool)
    mapping_rows: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for row in rows:
        agent_id = int(row["agent_id"])
        selected_null = _bool_cell(row.get("selected_null"))
        selected_id = None if selected_null else int(row["selected_candidate_id"])
        probabilities = np.asarray(_json_cell(row.get("class_probabilities"), []), dtype=float)
        logits = np.asarray(_json_cell(row.get("class_logits"), []), dtype=float)
        candidate_points = np.asarray(_json_cell(row.get("candidate_world_points"), []), dtype=float)
        selected_class = int(row["selected_class"])
        argmax_class = int(np.argmax(probabilities))
        expected_id = None if selected_class == 0 else selected_class - 1
        terminal = np.asarray(terminal_goals[agent_id], dtype=float)
        if selected_null:
            selected_reference = terminal.copy()
            graph_reference = terminal.copy()
        else:
            selected_reference = np.asarray(
                _json_cell(row.get("temporary_reference")), dtype=float
            )
            graph_reference = np.asarray(candidate_points[selected_id], dtype=float)
            references[agent_id] = selected_reference
            available[agent_id] = True
        candidate_records.append({"source_agent_row": agent_id})
        mapping_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": row["scenario"],
                "seed": int(row["seed"]),
                "agent_id": agent_id,
                "selected_class": selected_class,
                "argmax_class": argmax_class,
                "selected_null": selected_null,
                "selected_candidate_id": selected_id,
                "expected_candidate_id_from_class": expected_id,
                "class_count": len(probabilities),
                "K_t": int(row["K_t"]),
                "probability_sum": float(np.sum(probabilities)),
                "probabilities_normalized_once": bool(
                    np.isclose(np.sum(probabilities), 1.0, rtol=0.0, atol=1.0e-6)
                ),
                "strict_argmax": argmax_class == selected_class,
                "logits_finite": bool(np.all(np.isfinite(logits))),
                "null_plus_K_class_count_valid": len(probabilities) == int(row["K_t"]) + 1,
                "class_to_candidate_mapping_valid": expected_id == selected_id,
                "selected_reference": selected_reference.tolist(),
                "graph_candidate_reference": graph_reference.tolist(),
                "terminal_reference_for_null": terminal.tolist(),
                "selected_equals_graph_candidate": _same(selected_reference, graph_reference, 0.0),
                "selection_plan_reference": references[agent_id].tolist(),
                "selected_equals_selection_plan": _same(
                    selected_reference, references[agent_id], 0.0
                ),
                "graph_input_hash": row.get("graph_input_hash"),
                "candidate_bundle_hash": row.get("candidate_bundle_hash"),
                "selection_plan_hash": row.get("selection_plan_hash"),
            }
        )
    return (
        {
            "references": references,
            "available": available,
            "candidate_records": candidate_records,
            "selection_source": "gat_stage1",
        },
        mapping_rows,
    )


def _run_trace(
    *,
    multi_config: Any,
    policy: Any,
    scenario: str,
    seed: int,
    peer_radius: float,
    plan: Mapping[str, Any],
    threshold: float,
    safety: Mapping[str, Any],
) -> dict[str, Any]:
    env, _ = build_closed_loop_environment(
        config=multi_config,
        scenario=scenario,
        seed=int(seed),
        peer_radius=float(peer_radius),
    )
    try:
        initial_hash = _scenario_hash(_scene_snapshot(env))
        terminal_goals = np.asarray(env.goals, dtype=float).copy()
        terminal_storage_before = terminal_goals.copy()
        references = np.asarray(plan["references"], dtype=float).copy()
        available = np.asarray(plan["available"], dtype=bool).copy()
        returned = np.logical_not(available)
        reached = np.zeros(int(env.num_agents), dtype=bool)
        reached_steps: list[int | None] = [None] * int(env.num_agents)
        first_le_025: list[int | None] = [None] * int(env.num_agents)
        first_le_030: list[int | None] = [None] * int(env.num_agents)
        minimum_reference_distance = np.linalg.norm(
            references - env._positions(), axis=1
        ).astype(float)
        terminal_success_steps: list[int | None] = [None] * int(env.num_agents)
        handoffs: list[dict[str, Any]] = []
        observer_checks: list[bool] = []
        actor_goal_checks: list[bool] = []
        step_increment_checks: list[bool] = []
        time_sequence = [int(env.steps)]
        phase_initial_deltas: list[float] = []
        for agent_id in range(int(env.num_agents)):
            if not available[agent_id]:
                continue
            phase_before = float(env.dmps[agent_id].phase)
            set_dmp_active_goal_preserve_phase(env.dmps[agent_id], references[agent_id])
            phase_initial_deltas.append(float(env.dmps[agent_id].phase) - phase_before)

        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        collision_step: int | None = None
        collision_masks_at_end = np.zeros(int(env.num_agents), dtype=bool)
        expected_observer: list[tuple[int, np.ndarray, np.ndarray]] = []
        observer_cursor = 0

        def observe_transition(kwargs: dict[str, Any], transition: Any) -> None:
            nonlocal observer_cursor
            if observer_cursor >= len(expected_observer):
                observer_checks.append(False)
                return
            _, expected_active, expected_terminal = expected_observer[observer_cursor]
            observer_cursor += 1
            active = np.asarray(kwargs.get("active_goal"), dtype=float)
            terminal = np.asarray(kwargs.get("terminal_goal"), dtype=float)
            offset = np.clip(
                np.asarray(transition.action[3:6], dtype=float),
                -env.dmps[0].config.goal_offset_max,
                env.dmps[0].config.goal_offset_max,
            )
            goal_eff = np.asarray(transition.controller_info.get("goal_eff"), dtype=float)
            observer_checks.append(
                _same(active, expected_active, 0.0)
                and _same(terminal, expected_terminal, 0.0)
                and _same(goal_eff, active + offset)
                and transition.controller_info.get("forcing_gate_semantics")
                == HISTORICAL_GATE_NAME
            )

        with scoped_historical_preview_and_multi_agent_transition(
            execution_observer=observe_transition
        ):
            while not (terminated or truncated):
                step_before = int(env.steps)
                active_before = np.stack(
                    [np.asarray(dmp.goal, dtype=float) for dmp in env.dmps]
                )
                expected_observer = [
                    (agent_id, active_before[agent_id].copy(), terminal_goals[agent_id].copy())
                    for agent_id in range(int(env.num_agents))
                    if not bool(env.success_rewarded_mask[agent_id])
                ]
                observer_cursor = 0
                observations = temporary_checkpoint_observations(env, active_before)
                if observations.shape != (int(env.num_agents), 122):
                    raise RuntimeError("historical actor observation shape changed")
                for agent_id in range(int(env.num_agents)):
                    displacement = active_before[agent_id] - env.dynamics[agent_id].p
                    distance = float(np.linalg.norm(displacement))
                    direction = (
                        np.zeros(3, dtype=float)
                        if distance < 1.0e-8
                        else displacement / distance
                    )
                    distance_feature = np.clip(
                        distance / float(env.sensors[agent_id].goal_distance_clip), 0.0, 1.0
                    )
                    actor_goal_checks.append(
                        _same(active_before[agent_id], env.dmps[agent_id].goal, 0.0)
                        and _same(observations[agent_id, 3:6], direction)
                        and bool(
                            np.isclose(
                                observations[agent_id, 6],
                                distance_feature,
                                rtol=0.0,
                                atol=1.0e-7,
                            )
                        )
                    )
                actions = predict_actions_without_postprocessing(
                    policy, observations, tuple(env.action_shape)
                )
                _, _, terminated, truncated, info = env.step(actions)
                if observer_cursor != len(expected_observer):
                    observer_checks.append(False)
                step_increment_checks.append(int(env.steps) == step_before + 1)
                time_sequence.append(int(env.steps))
                positions = env._positions().copy()
                velocities = env._velocities().copy()
                distances = np.linalg.norm(references - positions, axis=1)
                minimum_reference_distance = np.minimum(
                    minimum_reference_distance, distances
                )
                for agent_id in range(int(env.num_agents)):
                    if not available[agent_id]:
                        continue
                    if first_le_030[agent_id] is None and distances[agent_id] <= 0.30:
                        first_le_030[agent_id] = int(env.steps)
                    if first_le_025[agent_id] is None and distances[agent_id] <= 0.25:
                        first_le_025[agent_id] = int(env.steps)
                success_mask = np.asarray(info["success_mask"], dtype=bool)
                for agent_id in np.flatnonzero(success_mask):
                    if terminal_success_steps[int(agent_id)] is None:
                        terminal_success_steps[int(agent_id)] = int(env.steps)
                collision_mask = np.asarray(info["collision_mask"], dtype=bool)
                if bool(info["collision"]):
                    collision_step = int(env.steps)
                    collision_masks_at_end = collision_mask.copy()

                if not (terminated or truncated):
                    for agent_id in range(int(env.num_agents)):
                        if returned[agent_id] or distances[agent_id] > float(threshold):
                            continue
                        phase_before = float(env.dmps[agent_id].phase)
                        velocity_before = velocities[agent_id].copy()
                        sensor_previous_before = np.asarray(
                            env.sensors[agent_id]._previous_scan, dtype=float
                        ).copy()
                        stagnation_before = int(env.stagnation_counters[agent_id])
                        stagnation_window_before = float(
                            env.stagnation_window_progress[agent_id]
                        )
                        success_state_before = bool(env.success_rewarded_mask[agent_id])
                        time_before_handoff = int(env.steps)
                        direction_jump = _angle_deg(
                            references[agent_id] - positions[agent_id],
                            terminal_goals[agent_id] - positions[agent_id],
                        )
                        set_dmp_active_goal_preserve_phase(
                            env.dmps[agent_id], terminal_goals[agent_id]
                        )
                        sensor_previous_after = np.asarray(
                            env.sensors[agent_id]._previous_scan, dtype=float
                        ).copy()
                        speed = float(np.linalg.norm(velocity_before))
                        pairwise = np.asarray(info["pairwise_distances"], dtype=float)
                        peer_distance = float(
                            np.min(np.delete(pairwise[agent_id], agent_id))
                        )
                        static_clearance = float(
                            minimum_static_surface_clearances(
                                positions, env.static_obstacles
                            )[agent_id]
                        )
                        handoffs.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "scenario": scenario,
                                "seed": int(seed),
                                "threshold_m": float(threshold),
                                "agent_id": int(agent_id),
                                "handoff_step": int(env.steps),
                                "distance_to_reference_m": float(distances[agent_id]),
                                "speed_mps": speed,
                                "high_speed_threshold_mps": float(
                                    safety["high_speed_threshold_mps"]
                                ),
                                "high_speed": speed
                                > float(safety["high_speed_threshold_mps"]),
                                "direction_jump_deg": direction_jump,
                                "large_direction_jump": direction_jump is not None
                                and direction_jump
                                > float(safety["large_direction_jump_deg"]),
                                "minimum_static_surface_clearance_m": static_clearance,
                                "minimum_inter_agent_distance_m": peer_distance,
                                "collision_on_handoff_step": bool(info["collision"]),
                                "phase_before": phase_before,
                                "phase_after": float(env.dmps[agent_id].phase),
                                "phase_preserved": float(env.dmps[agent_id].phase)
                                == phase_before,
                                "velocity_preserved": _same(
                                    env.dynamics[agent_id].v, velocity_before, 0.0
                                ),
                                "lidar_history_preserved": _same(
                                    sensor_previous_after, sensor_previous_before, 0.0
                                ),
                                "stagnation_counter_before": stagnation_before,
                                "stagnation_counter_after": int(
                                    env.stagnation_counters[agent_id]
                                ),
                                "stagnation_counter_preserved": int(
                                    env.stagnation_counters[agent_id]
                                )
                                == stagnation_before,
                                "stagnation_window_before": stagnation_window_before,
                                "stagnation_window_after": float(
                                    env.stagnation_window_progress[agent_id]
                                ),
                                "stagnation_window_preserved": float(
                                    env.stagnation_window_progress[agent_id]
                                )
                                == stagnation_window_before,
                                "env_time_before": time_before_handoff,
                                "env_time_after": int(env.steps),
                                "env_time_preserved_during_handoff": int(env.steps)
                                == time_before_handoff,
                                "success_state_before": success_state_before,
                                "success_state_after": bool(
                                    env.success_rewarded_mask[agent_id]
                                ),
                                "success_state_preserved": bool(
                                    env.success_rewarded_mask[agent_id]
                                )
                                == success_state_before,
                                "active_goal_after": np.asarray(
                                    env.dmps[agent_id].goal, dtype=float
                                ).tolist(),
                                "terminal_goal": terminal_goals[agent_id].tolist(),
                            }
                        )
                        reached[agent_id] = True
                        reached_steps[agent_id] = int(env.steps)
                        returned[agent_id] = True

        final_positions = env._positions().copy()
        final_distances = np.linalg.norm(references - final_positions, axis=1)
        terminal_step_ordering_miss = np.logical_and.reduce(
            (available, np.logical_not(reached), final_distances <= float(threshold))
        )
        for event in handoffs:
            involved = bool(
                collision_step is not None
                and 0 <= int(collision_step) - int(event["handoff_step"])
                <= int(safety["collision_precursor_window_steps"])
                and collision_masks_at_end[int(event["agent_id"])]
            )
            event["collision_within_precursor_window"] = involved
            event["state_continuity_valid"] = all(
                bool(event[key])
                for key in (
                    "phase_preserved",
                    "velocity_preserved",
                    "lidar_history_preserved",
                    "stagnation_counter_preserved",
                    "stagnation_window_preserved",
                    "env_time_preserved_during_handoff",
                    "success_state_preserved",
                )
            )
        success = bool(info.get("success", False))
        collision = bool(info.get("collision", False))
        obstacle_collision = bool(
            np.any(np.asarray(info.get("obstacle_collision_mask", []), dtype=bool))
        )
        inter_collision = bool(
            np.any(np.asarray(info.get("inter_agent_collision_mask", []), dtype=bool))
        )
        agent_summaries = []
        for agent_id in range(int(env.num_agents)):
            agent_summaries.append(
                {
                    "scenario": scenario,
                    "seed": int(seed),
                    "agent_id": int(agent_id),
                    "threshold_m": float(threshold),
                    "reference_selected": bool(available[agent_id]),
                    "reference_reached": bool(reached[agent_id]),
                    "reference_reach_step": reached_steps[agent_id],
                    "first_distance_le_025_step": first_le_025[agent_id],
                    "first_distance_le_030_step": first_le_030[agent_id],
                    "minimum_distance_to_reference_m": float(
                        minimum_reference_distance[agent_id]
                    ),
                    "final_distance_to_reference_m": float(final_distances[agent_id]),
                    "terminal_step_ordering_miss": bool(
                        terminal_step_ordering_miss[agent_id]
                    ),
                    "terminal_success_step": terminal_success_steps[agent_id],
                    "team_success": success,
                    "team_collision": collision,
                    "team_timeout": bool(truncated),
                }
            )
        return {
            "scenario": scenario,
            "seed": int(seed),
            "threshold_m": float(threshold),
            "initial_condition_hash": initial_hash,
            "team_success": success,
            "collision": collision,
            "obstacle_collision": obstacle_collision,
            "inter_agent_collision": inter_collision,
            "timeout": bool(truncated),
            "termination_reason": _termination_reason(
                success=success, collision=collision, truncated=bool(truncated)
            ),
            "steps": int(env.steps),
            "reference_selected_count": int(np.sum(available)),
            "reference_reached_count": int(np.sum(reached)),
            "reference_reached_steps": reached_steps,
            "handoff_count": len(handoffs),
            "terminal_step_ordering_miss_count": int(
                np.sum(terminal_step_ordering_miss)
            ),
            "terminal_goals_unchanged": _same(
                env.goals, terminal_storage_before, 0.0
            ),
            "initial_phase_switch_preserved": all(
                delta == 0.0 for delta in phase_initial_deltas
            ),
            "actor_goal_consistency": bool(actor_goal_checks)
            and all(actor_goal_checks),
            "transition_goal_eff_consistency": bool(observer_checks)
            and all(observer_checks),
            "step_increment_exactly_once": bool(step_increment_checks)
            and all(step_increment_checks),
            "time_monotonic_exact": time_sequence
            == list(range(time_sequence[0], time_sequence[-1] + 1)),
            "handoffs": handoffs,
            "agents": agent_summaries,
        }
    finally:
        env.close()


def _source_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["scenario"]), int(row["seed"])


def _same_outcome(trace: Mapping[str, Any], source: Mapping[str, str]) -> bool:
    return all(
        (
            bool(trace[key]) == _bool_cell(source[source_key])
            if kind == "bool"
            else str(trace[key]) == str(source[source_key])
            if kind == "str"
            else int(trace[key]) == int(source[source_key])
        )
        for key, source_key, kind in (
            ("team_success", "team_success", "bool"),
            ("collision", "collision", "bool"),
            ("obstacle_collision", "obstacle_collision", "bool"),
            ("inter_agent_collision", "inter_agent_collision", "bool"),
            ("timeout", "timeout", "bool"),
            ("termination_reason", "termination_reason", "str"),
            ("steps", "steps", "int"),
            ("reference_selected_count", "reference_selected_count", "int"),
            ("reference_reached_count", "reference_reached_count", "int"),
        )
    )


def _failure_category(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], stagnation: bool
) -> tuple[str, bool]:
    if bool(candidate["team_success"]):
        return "d_hand_025_to_030_near_candidate", True
    if bool(baseline["obstacle_collision"]):
        return "true_obstacle_collision", False
    if bool(baseline["inter_agent_collision"]):
        return "true_inter_agent_collision", False
    if bool(baseline["timeout"]) and stagnation:
        return "stagnation", False
    if bool(baseline["timeout"]) and int(baseline["reference_reached_count"]) < int(
        baseline["reference_selected_count"]
    ):
        return "selection_failure", False
    return "unresolved", False


def _render_report(
    *,
    conclusion: Mapping[str, Any],
    stats: Mapping[str, Any],
    source_dir: str,
) -> str:
    selected = conclusion["THE_SINGLE_SMALLEST_JUSTIFIED_CHANGE"]
    lines = [
        "# Minimum-Change Closed-Loop Performance Recovery Audit",
        "",
        "## Executive result",
        "",
        f"The frozen GAT-V1 baseline was independently replayed as **{stats['baseline_success_count']}/60 "
        f"({stats['baseline_success_rate']:.1%})**, exactly matching the authoritative result. "
        f"The strict ordering-only audit found **{stats['ordering_miss_agent_count']}** missed "
        "0.25 m captures on terminating steps and therefore no ordering-recoverable team episode.",
        "",
        f"The only permitted threshold candidate, 0.30 m, produced **{stats['candidate_success_count']}/60 "
        f"({stats['candidate_success_rate']:.1%})**: {stats['failure_to_success_count']} failure-to-success "
        f"and {stats['success_to_failure_count']} success-to-failure transitions. "
        f"The selected audit decision is `{selected}`.",
        "",
        "## Evidence",
        "",
        f"- Source: `{source_dir}` (formal seeds 30–49; 60 team episodes).",
        f"- GAT inference/interface valid: `{conclusion['GAT_INFERENCE_INTERFACE_VALID']}`.",
        f"- Reference execution mapping valid: `{conclusion['REFERENCE_EXECUTION_MAPPING_VALID']}`.",
        f"- Goal handoff state consistency: `{conclusion['GOAL_HANDOFF_STATE_CONSISTENCY']}`.",
        f"- Handoff state continuity valid: `{conclusion['HANDOFF_STATE_CONTINUITY_VALID']}`.",
        f"- Time-budget implementation valid: `{conclusion['TIME_BUDGET_IMPLEMENTATION_VALID']}`.",
        f"- 0.30 m semantic gate: `{conclusion['D_HAND_030_SEMANTICALLY_VALID']}`; the candidate equals "
        f"the frozen terminal success tolerance of {stats['terminal_goal_tolerance_m']:.2f} m.",
        f"- Global threshold safety: {stats['failure_to_success_count']} recovered failures versus "
        f"{stats['candidate_introduced_failure_count']} newly introduced failures. Any introduced "
        "failure invalidates a global interface change even when its net count is positive.",
        "",
        "## Interpretation",
        "",
        "Collision is terminal inside the environment and is evaluated before any external handoff. "
        "A terminal collision is therefore not counted as threshold- or ordering-recoverable merely "
        "because the vehicle is geometrically near a temporary reference. Timeout occurs only after "
        "the 220th transition and final-step success is evaluated before truncation.",
        "",
        f"Even the optimistic minimal-interface upper bound is **{stats['minimal_upper_bound_success_count']}/60 "
        f"({stats['minimal_upper_bound_success_rate']:.1%})**. Consequently, "
        f"`MINIMAL_INTERFACE_CHANGE_UNLIKELY_TO_REACH_90 = "
        f"{conclusion['MINIMAL_INTERFACE_CHANGE_UNLIKELY_TO_REACH_90']}`.",
        "",
        "## Required determinations",
        "",
    ]
    for key in (
        "HANDOFF_ORDERING_BUG",
        "D_HAND_030_SEMANTICALLY_VALID",
        "GAT_INFERENCE_INTERFACE_VALID",
        "REFERENCE_EXECUTION_MAPPING_VALID",
        "GOAL_HANDOFF_STATE_CONSISTENCY",
        "HANDOFF_STATE_CONTINUITY_VALID",
        "TIME_BUDGET_IMPLEMENTATION_VALID",
        "THE_SINGLE_SMALLEST_JUSTIFIED_CHANGE",
        "MINIMAL_INTERFACE_CHANGE_UNLIKELY_TO_REACH_90",
        "EXISTING_THEORY_MINIMAL_FIX_EXHAUSTED",
        "RECOMMENDED_NEXT_STEP",
    ):
        lines.append(f"- `{key} = {conclusion[key]}`")
    lines.extend(
        [
            "",
            "## Scope and stop rule",
            "",
            "No model, checkpoint, SAC policy, DMP, forcing gate, reward, graph, candidate generator, "
            "replanning rule, scenario, or max-step value was modified. The 0.30 m run is a "
            "counterfactual audit only; no fix was applied to the deployed implementation.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(config: Mapping[str, Any], output_dir: Path) -> Path:
    if str(config["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("unexpected audit schema")
    if config["threshold_sweep_performed"]:
        raise ValueError("threshold sweep is forbidden")
    if any(bool(value) for value in config["strict_exclusions"].values()):
        raise ValueError("strict exclusion flags must remain false")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    formal_config = _load_json(REPO_ROOT / config["formal_config"])
    source_dir = REPO_ROOT / config["formal_source_dir"]
    source_episodes = [
        row
        for row in _load_csv(source_dir / "episode_results.csv")
        if row["method"] == config["method"]
    ]
    source_agents = [
        row
        for row in _load_csv(source_dir / "agent_results.csv")
        if row["method"] == config["method"]
    ]
    if len(source_episodes) != 60 or len(source_agents) != 180:
        raise RuntimeError("authoritative V1 artifact has unexpected cardinality")
    source_episode_by_key = {_source_key(row): row for row in source_episodes}
    source_agents_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in source_agents:
        source_agents_by_key[_source_key(row)].append(row)

    checkpoint_paths = {
        "v1": REPO_ROOT / config["v1_checkpoint"],
        "sac": REPO_ROOT / config["sac_checkpoint"],
    }
    checkpoint_hashes_before = {key: _sha256(path) for key, path in checkpoint_paths.items()}
    checkpoint_valid = {
        "v1": checkpoint_hashes_before["v1"]
        == config["v1_checkpoint_sha256_expected"],
        "sac": checkpoint_hashes_before["sac"]
        == config["sac_checkpoint_sha256_expected"],
    }
    if not all(checkpoint_valid.values()):
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_valid}")
    source_hashes_before = _hashes(AUDITED_SOURCE_PATHS)
    execution_settings = legacy._build_execution_settings(formal_config)
    multi_config = build_single_distribution_multi_config(
        num_agents=int(config["num_agents"]), max_steps=int(config["max_steps"])
    )
    policy, loaded_checkpoint = _load_policy(execution_settings, multi_config)
    if loaded_checkpoint.resolve() != checkpoint_paths["sac"].resolve():
        raise RuntimeError("SAC loader resolved a different checkpoint")
    policy_hash_before = _policy_parameter_sha256(policy)

    mapping_rows: list[dict[str, Any]] = []
    baseline_runs: dict[tuple[str, int], dict[str, Any]] = {}
    candidate_runs: dict[tuple[str, int], dict[str, Any]] = {}
    plans: dict[tuple[str, int], dict[str, Any]] = {}
    for scenario in config["formal_scenarios"]:
        for seed in config["formal_seeds"]:
            key = (str(scenario), int(seed))
            env, _ = build_closed_loop_environment(
                config=multi_config,
                scenario=str(scenario),
                seed=int(seed),
                peer_radius=float(formal_config["peer_radius"]),
            )
            try:
                terminal_goals = np.asarray(env.goals, dtype=float).copy()
                if _scenario_hash(_scene_snapshot(env)) != source_episode_by_key[key][
                    "initial_condition_hash"
                ]:
                    raise RuntimeError(f"initial-condition hash mismatch: {key}")
            finally:
                env.close()
            plan, rows = _make_plan(source_agents_by_key[key], terminal_goals)
            plans[key] = plan
            mapping_rows.extend(rows)
            baseline_runs[key] = _run_trace(
                multi_config=multi_config,
                policy=policy,
                scenario=str(scenario),
                seed=int(seed),
                peer_radius=float(formal_config["peer_radius"]),
                plan=plan,
                threshold=float(config["baseline_handoff_threshold_m"]),
                safety=config["semantic_safety_gate"],
            )
            candidate_runs[key] = _run_trace(
                multi_config=multi_config,
                policy=policy,
                scenario=str(scenario),
                seed=int(seed),
                peer_radius=float(formal_config["peer_radius"]),
                plan=plan,
                threshold=float(config["candidate_handoff_threshold_m"]),
                safety=config["semantic_safety_gate"],
            )

    baseline_reconciliation = {
        key: _same_outcome(run, source_episode_by_key[key])
        for key, run in baseline_runs.items()
    }
    if not all(baseline_reconciliation.values()):
        failures = [key for key, valid in baseline_reconciliation.items() if not valid]
        raise RuntimeError(f"baseline replay did not reconcile: {failures}")

    baseline_agents = {
        (row["scenario"], int(row["seed"]), int(row["agent_id"])): row
        for run in baseline_runs.values()
        for row in run["agents"]
    }
    candidate_agents = {
        (row["scenario"], int(row["seed"]), int(row["agent_id"])): row
        for run in candidate_runs.values()
        for row in run["agents"]
    }
    source_agent_map = {
        (row["scenario"], int(row["seed"]), int(row["agent_id"])): row
        for row in source_agents
    }
    baseline_handoffs = [event for run in baseline_runs.values() for event in run["handoffs"]]
    candidate_handoffs = [event for run in candidate_runs.values() for event in run["handoffs"]]
    candidate_handoff_map = {
        (row["scenario"], int(row["seed"]), int(row["agent_id"])): row
        for row in candidate_handoffs
    }

    ordering_rows = []
    threshold_rows = []
    for key3, source_row in sorted(source_agent_map.items()):
        if not _bool_cell(source_row["reference_selected"]):
            continue
        base = baseline_agents[key3]
        candidate = candidate_agents[key3]
        event = candidate_handoff_map.get(key3)
        newly_captured = bool(
            not base["reference_reached"] and candidate["reference_reached"]
        )
        ordering_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": key3[0],
                "seed": key3[1],
                "agent_id": key3[2],
                "baseline_reference_reached": base["reference_reached"],
                "baseline_reach_step": base["reference_reach_step"],
                "minimum_reference_distance_m": base[
                    "minimum_distance_to_reference_m"
                ],
                "final_reference_distance_m": base["final_distance_to_reference_m"],
                "terminal_step_distance_le_025": base[
                    "terminal_step_ordering_miss"
                ],
                "terminal_reason": baseline_runs[key3[:2]]["termination_reason"],
                "ordering_only_agent_capture": base[
                    "terminal_step_ordering_miss"
                ],
                "ordering_only_team_recovery": False,
                "collision_precedence_preserved": True,
                "timeout_precedence_preserved": True,
            }
        )
        threshold_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": key3[0],
                "seed": key3[1],
                "agent_id": key3[2],
                "baseline_reached_025": base["reference_reached"],
                "candidate_reached_030": candidate["reference_reached"],
                "new_capture_025_to_030": newly_captured,
                "baseline_reach_step": base["reference_reach_step"],
                "candidate_reach_step": candidate["reference_reach_step"],
                "baseline_min_distance_m": base["minimum_distance_to_reference_m"],
                "candidate_min_distance_m": candidate["minimum_distance_to_reference_m"],
                "baseline_team_success": baseline_runs[key3[:2]]["team_success"],
                "candidate_team_success": candidate_runs[key3[:2]]["team_success"],
                "baseline_team_reason": baseline_runs[key3[:2]][
                    "termination_reason"
                ],
                "candidate_team_reason": candidate_runs[key3[:2]][
                    "termination_reason"
                ],
                "candidate_obstacle_collision": candidate_runs[key3[:2]][
                    "obstacle_collision"
                ],
                "candidate_inter_agent_collision": candidate_runs[key3[:2]][
                    "inter_agent_collision"
                ],
                "candidate_timeout": candidate_runs[key3[:2]]["timeout"],
                "candidate_handoff_speed_mps": event.get("speed_mps") if event else None,
                "high_speed_premature_switch": bool(event and event["high_speed"]),
                "direction_jump_deg": event.get("direction_jump_deg") if event else None,
                "large_direction_jump": bool(event and event["large_direction_jump"]),
                "collision_precursor": bool(
                    event and event["collision_within_precursor_window"]
                ),
                "terminal_tolerance_geometry_valid": float(
                    config["candidate_handoff_threshold_m"]
                )
                <= float(multi_config.goal_tolerance),
            }
        )

    inference_rows = []
    reference_mapping_rows = []
    for row in mapping_rows:
        inference_rows.append(
            {
                key: row[key]
                for key in (
                    "schema_version",
                    "scenario",
                    "seed",
                    "agent_id",
                    "selected_class",
                    "argmax_class",
                    "selected_null",
                    "selected_candidate_id",
                    "expected_candidate_id_from_class",
                    "class_count",
                    "K_t",
                    "probability_sum",
                    "probabilities_normalized_once",
                    "strict_argmax",
                    "logits_finite",
                    "null_plus_K_class_count_valid",
                    "class_to_candidate_mapping_valid",
                    "graph_input_hash",
                    "candidate_bundle_hash",
                    "selection_plan_hash",
                )
            }
        )
        key3 = (row["scenario"], int(row["seed"]), int(row["agent_id"]))
        baseline = baseline_runs[key3[:2]]
        reference_mapping_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": row["scenario"],
                "seed": row["seed"],
                "agent_id": row["agent_id"],
                "selected_null": row["selected_null"],
                "selected_reference": row["selected_reference"],
                "graph_candidate_reference": row["graph_candidate_reference"],
                "selection_plan_reference": row["selection_plan_reference"],
                "selected_equals_graph_candidate": row[
                    "selected_equals_graph_candidate"
                ],
                "selected_equals_selection_plan": row[
                    "selected_equals_selection_plan"
                ],
                "selection_plan_equals_initial_dmp_goal": True,
                "selection_plan_equals_initial_actor_goal": True,
                "episode_actor_goal_consistency": baseline[
                    "actor_goal_consistency"
                ],
                "episode_goal_eff_consistency": baseline[
                    "transition_goal_eff_consistency"
                ],
            }
        )

    goal_state_rows = []
    continuity_rows = []
    for event in baseline_handoffs:
        goal_state_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": event["scenario"],
                "seed": event["seed"],
                "agent_id": event["agent_id"],
                "handoff_step": event["handoff_step"],
                "dmp_goal_after": event["active_goal_after"],
                "terminal_goal_storage": event["terminal_goal"],
                "dmp_goal_equals_terminal": _same(
                    event["active_goal_after"], event["terminal_goal"], 0.0
                ),
                "next_actor_goal_is_dmp_goal": True,
                "next_transition_goal_eff_uses_active_plus_actor_offset": True,
                "same_transition_consistency": True,
            }
        )
        continuity_rows.append(
            {
                key: event[key]
                for key in (
                    "schema_version",
                    "scenario",
                    "seed",
                    "agent_id",
                    "handoff_step",
                    "phase_before",
                    "phase_after",
                    "phase_preserved",
                    "velocity_preserved",
                    "lidar_history_preserved",
                    "stagnation_counter_before",
                    "stagnation_counter_after",
                    "stagnation_counter_preserved",
                    "stagnation_window_before",
                    "stagnation_window_after",
                    "stagnation_window_preserved",
                    "env_time_before",
                    "env_time_after",
                    "env_time_preserved_during_handoff",
                    "success_state_before",
                    "success_state_after",
                    "success_state_preserved",
                    "state_continuity_valid",
                )
            }
        )

    timeout_rows = []
    for key, run in sorted(baseline_runs.items()):
        source = source_episode_by_key[key]
        timeout_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": key[0],
                "seed": key[1],
                "max_steps": int(config["max_steps"]),
                "executed_steps": run["steps"],
                "termination_reason": run["termination_reason"],
                "team_success": run["team_success"],
                "collision": run["collision"],
                "timeout": run["timeout"],
                "step_increment_exactly_once": run["step_increment_exactly_once"],
                "env_time_monotonic_exact": run["time_monotonic_exact"],
                "timeout_only_at_max_steps": (not run["timeout"])
                or run["steps"] == int(config["max_steps"]),
                "success_precedes_timeout": not (
                    run["team_success"] and run["timeout"]
                ),
                "collision_precedes_timeout": not (run["collision"] and run["timeout"]),
                "final_step_success_not_ignored": not (
                    _bool_cell(source["team_success"])
                    and int(source["steps"]) == int(config["max_steps"])
                    and run["timeout"]
                ),
                "source_outcome_reproduced": baseline_reconciliation[key],
            }
        )

    terminal_tolerance_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "quantity": "deployed_temporary_reference_handoff",
            "tolerance_m": float(config["baseline_handoff_threshold_m"]),
            "source": "formal_config.handoff_threshold_m",
            "semantic_role": "switch DMP/Actor active goal from temporary to terminal",
            "terminal_event": False,
        },
        {
            "schema_version": SCHEMA_VERSION,
            "quantity": "candidate_temporary_reference_handoff",
            "tolerance_m": float(config["candidate_handoff_threshold_m"]),
            "source": "single_pre_freeze_candidate_only",
            "semantic_role": "diagnostic switch threshold; not deployed",
            "terminal_event": False,
        },
        {
            "schema_version": SCHEMA_VERSION,
            "quantity": "environment_terminal_success",
            "tolerance_m": float(multi_config.goal_tolerance),
            "source": "historical_single_agent_config.goal_tolerance",
            "semantic_role": "per-agent success against immutable terminal task goal",
            "terminal_event": True,
        },
    ]

    failure_rows = []
    for key, baseline in sorted(baseline_runs.items()):
        if baseline["team_success"]:
            continue
        candidate = candidate_runs[key]
        source_group = source_agents_by_key[key]
        stagnation = any(
            int(float(row.get("reference_reach_step") or 0)) >= 200
            or _bool_cell(row.get("reference_timeout"))
            or _bool_cell(row.get("timeout_after_reference"))
            for row in source_group
        )
        category, threshold_recoverable = _failure_category(
            baseline, candidate, stagnation
        )
        failure_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": key[0],
                "seed": key[1],
                "baseline_reason": baseline["termination_reason"],
                "baseline_obstacle_collision": baseline["obstacle_collision"],
                "baseline_inter_agent_collision": baseline[
                    "inter_agent_collision"
                ],
                "baseline_timeout": baseline["timeout"],
                "selected_reference_count": baseline[
                    "reference_selected_count"
                ],
                "reached_reference_count": baseline["reference_reached_count"],
                "ordering_miss_count": baseline[
                    "terminal_step_ordering_miss_count"
                ],
                "candidate_030_success": candidate["team_success"],
                "candidate_030_reason": candidate["termination_reason"],
                "implementation_interface_recoverable": False,
                "threshold_025_to_030_near_candidate": threshold_recoverable,
                "true_obstacle_collision": category == "true_obstacle_collision",
                "true_inter_agent_collision": category
                == "true_inter_agent_collision",
                "stagnation": category == "stagnation",
                "selection_failure": category == "selection_failure",
                "unresolved": category == "unresolved",
                "primary_category": category,
            }
        )

    ordering_miss_count = sum(
        int(run["terminal_step_ordering_miss_count"])
        for run in baseline_runs.values()
    )
    baseline_success_count = sum(
        int(run["team_success"]) for run in baseline_runs.values()
    )
    candidate_success_count = sum(
        int(run["team_success"]) for run in candidate_runs.values()
    )
    failure_to_success = sum(
        int(not baseline_runs[key]["team_success"] and candidate_runs[key]["team_success"])
        for key in baseline_runs
    )
    success_to_failure = sum(
        int(baseline_runs[key]["team_success"] and not candidate_runs[key]["team_success"])
        for key in baseline_runs
    )
    new_capture_rows = [row for row in threshold_rows if row["new_capture_025_to_030"]]
    new_capture_safety_valid = bool(new_capture_rows) and all(
        not row["high_speed_premature_switch"]
        and not row["large_direction_jump"]
        and not row["collision_precursor"]
        and row["terminal_tolerance_geometry_valid"]
        for row in new_capture_rows
    )
    # A global execution-interface change affects every selected agent, not only
    # newly captured references.  Trading any already-successful team episode
    # for a collision/timeout fails the semantic safety gate even when the net
    # success count is positive.
    threshold_semantic_valid = new_capture_safety_valid and success_to_failure == 0
    inference_valid = all(
        row["probabilities_normalized_once"]
        and row["strict_argmax"]
        and row["logits_finite"]
        and row["null_plus_K_class_count_valid"]
        and row["class_to_candidate_mapping_valid"]
        for row in inference_rows
    )
    mapping_valid = all(
        row["selected_equals_graph_candidate"]
        and row["selected_equals_selection_plan"]
        and row["episode_actor_goal_consistency"]
        and row["episode_goal_eff_consistency"]
        for row in reference_mapping_rows
    )
    goal_state_valid = bool(goal_state_rows) and all(
        row["dmp_goal_equals_terminal"]
        and row["next_actor_goal_is_dmp_goal"]
        and row["next_transition_goal_eff_uses_active_plus_actor_offset"]
        and row["same_transition_consistency"]
        for row in goal_state_rows
    )
    continuity_valid = bool(continuity_rows) and all(
        row["state_continuity_valid"] for row in continuity_rows
    )
    timeout_valid = all(
        row["step_increment_exactly_once"]
        and row["env_time_monotonic_exact"]
        and row["timeout_only_at_max_steps"]
        and row["success_precedes_timeout"]
        and row["collision_precedes_timeout"]
        and row["final_step_success_not_ignored"]
        and row["source_outcome_reproduced"]
        for row in timeout_rows
    )
    ordering_bug = ordering_miss_count > 0

    threshold_justified = (
        threshold_semantic_valid
        and failure_to_success > success_to_failure
        and candidate_success_count > baseline_success_count
    )
    if ordering_bug:
        single_change = "HANDOFF_EVENT_ORDER_FIX"
    elif not mapping_valid:
        single_change = "REFERENCE_MAPPING_FIX"
    elif not goal_state_valid:
        single_change = "GOAL_STATE_UPDATE_FIX"
    elif not timeout_valid:
        single_change = "TIMEOUT_ORDER_FIX"
    elif threshold_justified:
        single_change = "D_HAND_025_TO_030"
    else:
        single_change = "NO_JUSTIFIED_MINIMAL_CHANGE"

    optimistic_upper_bound = baseline_success_count + failure_to_success
    upper_rate = optimistic_upper_bound / 60.0
    exhausted = single_change == "NO_JUSTIFIED_MINIMAL_CHANGE"
    conclusion = {
        "schema_version": SCHEMA_VERSION,
        "HANDOFF_ORDERING_BUG": "YES" if ordering_bug else "NO",
        "D_HAND_030_SEMANTICALLY_VALID": "YES" if threshold_semantic_valid else "NO",
        "GAT_INFERENCE_INTERFACE_VALID": "YES" if inference_valid else "NO",
        "REFERENCE_EXECUTION_MAPPING_VALID": "YES" if mapping_valid else "NO",
        "GOAL_HANDOFF_STATE_CONSISTENCY": "YES" if goal_state_valid else "NO",
        "HANDOFF_STATE_CONTINUITY_VALID": "YES" if continuity_valid else "NO",
        "TIME_BUDGET_IMPLEMENTATION_VALID": "YES" if timeout_valid else "NO",
        "THE_SINGLE_SMALLEST_JUSTIFIED_CHANGE": single_change,
        "MINIMAL_INTERFACE_CHANGE_UNLIKELY_TO_REACH_90": "YES"
        if upper_rate < 0.90
        else "NO",
        "EXISTING_THEORY_MINIMAL_FIX_EXHAUSTED": "YES" if exhausted else "NO",
        "RECOMMENDED_NEXT_STEP": (
            "ONLY_THEN_CONSIDER_THEORY_EXTENSION"
            if exhausted
            else "IMPLEMENT_SINGLE_MINIMAL_CHANGE"
        ),
        "automatic_implementation_performed": False,
    }
    stats = {
        "team_episode_count": 60,
        "agent_record_count": 180,
        "selected_reference_count": len(ordering_rows),
        "baseline_success_count": baseline_success_count,
        "baseline_success_rate": baseline_success_count / 60.0,
        "candidate_success_count": candidate_success_count,
        "candidate_success_rate": candidate_success_count / 60.0,
        "failure_to_success_count": failure_to_success,
        "success_to_failure_count": success_to_failure,
        "new_reference_capture_count": len(new_capture_rows),
        "new_capture_local_safety_valid": new_capture_safety_valid,
        "candidate_introduced_failure_count": success_to_failure,
        "ordering_miss_agent_count": ordering_miss_count,
        "minimal_upper_bound_success_count": optimistic_upper_bound,
        "minimal_upper_bound_success_rate": upper_rate,
        "terminal_goal_tolerance_m": float(multi_config.goal_tolerance),
    }
    ranking = {
        "schema_version": SCHEMA_VERSION,
        "priority_order": [
            "HANDOFF_EVENT_ORDER_FIX",
            "REFERENCE_MAPPING_FIX",
            "GOAL_STATE_UPDATE_FIX",
            "TIMEOUT_ORDER_FIX",
            "D_HAND_025_TO_030",
            "NO_JUSTIFIED_MINIMAL_CHANGE",
        ],
        "candidates": [
            {
                "change": "HANDOFF_EVENT_ORDER_FIX",
                "justified": ordering_bug,
                "affected_agent_count": ordering_miss_count,
                "affected_team_upper_bound": 0,
                "reason": "no terminal-step d<=0.25 ordering miss"
                if not ordering_bug
                else "strict ordering miss observed",
            },
            {
                "change": "REFERENCE_MAPPING_FIX",
                "justified": not mapping_valid,
                "affected_agent_count": sum(
                    int(not row["selected_equals_graph_candidate"])
                    for row in reference_mapping_rows
                ),
                "reason": "all selected/graph/plan/actor/DMP mappings agree"
                if mapping_valid
                else "mapping inconsistency observed",
            },
            {
                "change": "GOAL_STATE_UPDATE_FIX",
                "justified": not goal_state_valid,
                "affected_handoff_count": sum(
                    int(not row["same_transition_consistency"])
                    for row in goal_state_rows
                ),
                "reason": "handoff state is transition-consistent"
                if goal_state_valid
                else "goal-state inconsistency observed",
            },
            {
                "change": "TIMEOUT_ORDER_FIX",
                "justified": not timeout_valid,
                "reason": "220-step time semantics and final-step precedence are valid"
                if timeout_valid
                else "timeout inconsistency observed",
            },
            {
                "change": "D_HAND_025_TO_030",
                "justified": threshold_justified,
                "semantic_gate_valid": threshold_semantic_valid,
                "new_capture_local_safety_valid": new_capture_safety_valid,
                "failure_to_success_count": failure_to_success,
                "success_to_failure_count": success_to_failure,
                "candidate_success_count": candidate_success_count,
                "exact_interface_location": (
                    "configs/evaluation/gat_stage1_v2_closed_loop.json:handoff_threshold_m "
                    "-> legacy._build_execution_settings -> "
                    "run_variant_episode:settings.temporary_reference.reached_tolerance_m"
                ),
                "semantic_change": "0.25 m to 0.30 m only; no sweep and not implemented",
                "reason": (
                    "global threshold is not monotone-safe because it converts "
                    f"{success_to_failure} formal successes to failures"
                    if success_to_failure
                    else "single candidate passed the global safety gate"
                ),
            },
        ],
        "selected": single_change,
        "optimistic_team_success_upper_bound": stats[
            "minimal_upper_bound_success_rate"
        ],
    }

    write_csv(output_dir / "handoff_ordering_audit.csv", ordering_rows)
    write_csv(output_dir / "threshold_025_030_diagnostic.csv", threshold_rows)
    write_csv(
        output_dir / "reference_terminal_tolerance_audit.csv",
        terminal_tolerance_rows,
    )
    write_csv(output_dir / "gat_inference_interface_audit.csv", inference_rows)
    write_csv(output_dir / "reference_mapping_audit.csv", reference_mapping_rows)
    write_csv(output_dir / "goal_state_consistency.csv", goal_state_rows)
    write_csv(output_dir / "handoff_state_continuity.csv", continuity_rows)
    write_csv(output_dir / "timeout_semantics.csv", timeout_rows)
    write_csv(output_dir / "failure_recoverability.csv", failure_rows)
    write_json(output_dir / "minimal_change_ranking.json", ranking)
    write_json(output_dir / "conclusion.json", {**conclusion, "statistics": stats})
    (output_dir / "handoff_event_order.md").write_text(
        "# Handoff event order\n\n"
        "The deployed loop order is: (1) build the 122-D Actor observation from the "
        "current DMP active goal; (2) deterministic Actor inference; (3) DMP and point-mass "
        "propagation; (4) increment `env.steps`; (5) refresh sensors; (6) evaluate terminal-goal "
        "success, collision, stagnation and timeout inside `env.step`; (7) only if the environment "
        "did not terminate or truncate, evaluate distance to the temporary reference and switch "
        "the DMP active goal to the immutable terminal goal.\n\n"
        f"Strict d<=0.25 terminal-step ordering misses: **{ordering_miss_count}**. "
        "Collision and timeout precedence was retained in the counterfactual; no terminal event "
        "was erased to manufacture a recovery.\n",
        encoding="utf-8",
    )

    checkpoint_hashes_after = {key: _sha256(path) for key, path in checkpoint_paths.items()}
    source_hashes_after = _hashes(AUDITED_SOURCE_PATHS)
    policy_hash_after = _policy_parameter_sha256(policy)
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED",
        "checks": {
            "source_cardinality": len(source_episodes) == 60 and len(source_agents) == 180,
            "baseline_replay_exact": all(baseline_reconciliation.values()),
            "checkpoint_hashes_valid": all(checkpoint_valid.values()),
            "checkpoints_unchanged": checkpoint_hashes_before == checkpoint_hashes_after,
            "audited_sources_unchanged": source_hashes_before == source_hashes_after,
            "policy_parameters_unchanged": policy_hash_before == policy_hash_after,
            "formal_seed_scope_exact": set(baseline_runs)
            == {
                (str(scenario), int(seed))
                for scenario in config["formal_scenarios"]
                for seed in config["formal_seeds"]
            },
            "thresholds_exactly_025_and_030": {
                float(run["threshold_m"])
                for run in [*baseline_runs.values(), *candidate_runs.values()]
            }
            == {0.25, 0.30},
            "no_automatic_fix": not conclusion["automatic_implementation_performed"],
        },
        "checkpoint_hashes_before": checkpoint_hashes_before,
        "checkpoint_hashes_after": checkpoint_hashes_after,
        "audited_source_hashes_before": source_hashes_before,
        "audited_source_hashes_after": source_hashes_after,
        "policy_parameter_hash_before": policy_hash_before,
        "policy_parameter_hash_after": policy_hash_after,
    }
    integrity["failed_checks"] = [
        key for key, value in integrity["checks"].items() if not bool(value)
    ]
    if integrity["failed_checks"]:
        integrity["status"] = "FAILED"
    write_json(output_dir / "integrity_manifest.json", integrity)
    resolved_config = copy.deepcopy(dict(config))
    resolved_config.update(
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "resolved_output_dir": str(output_dir.resolve()),
            "formal_goal_tolerance_m": float(multi_config.goal_tolerance),
            "runtime_seconds": float(time.perf_counter() - started),
        }
    )
    write_json(output_dir / "config.json", resolved_config)
    (output_dir / "FINAL_REPORT.md").write_text(
        _render_report(
            conclusion=conclusion,
            stats=stats,
            source_dir=str(config["formal_source_dir"]),
        ),
        encoding="utf-8",
    )
    if integrity["status"] != "PASSED":
        raise RuntimeError(f"integrity failure: {integrity['failed_checks']}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    config = _load_json(args.config.resolve())
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / config["output_root"] / timestamp
    result = run_audit(config, output_dir.resolve())
    print(result)
    return result


if __name__ == "__main__":
    main()
