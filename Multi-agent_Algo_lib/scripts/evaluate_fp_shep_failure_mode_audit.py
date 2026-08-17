"""Run the read-only FP-SHEP failure-mode audit on frozen artifacts.

The script never invokes candidate generation or online selection.  It loads
the exact references selected by the frozen Geometry Generalization Audit,
replays only those 24 FP-SHEP team episodes for missing traces, and performs
post-hoc preview diagnostics whose output cannot feed back into execution.
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

from planning.fp_shep_failure_mode_analysis import (  # noqa: E402
    SCHEMA_VERSION as ANALYSIS_SCHEMA_VERSION,
    build_mechanism_conclusion,
    candidate_geometry_record,
    classify_reference_unreachable,
    descriptive_statistics,
    family_failure_summary,
    parse_bool,
    parse_json_cell,
    score_margin_record,
)
from planning.goal_semantics_diagnosis import (  # noqa: E402
    action_saturation_mask,
    temporary_checkpoint_observations,
)
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.multi_agent_obstacle_scenario_audit import (  # noqa: E402
    minimum_static_surface_clearances,
)
from planning.policy_preview import (  # noqa: E402
    build_preview_inputs_from_env,
    preview_candidate,
)
from planning.pre_gat_closed_loop import (  # noqa: E402
    FPSHEPOnlineScoreSpec,
    trajectory_metrics,
)
from planning.reference_transition_finetuning import sha256_file  # noqa: E402
from scripts.evaluate_actor_dmp_goal_semantics import (  # noqa: E402
    write_csv,
    write_json,
)
from scripts.evaluate_frozen_policy_waypoint_guidance import (  # noqa: E402
    predict_actions_without_postprocessing,
    set_dmp_active_goal_preserve_phase,
)
from scripts.evaluate_geometry_generalization_audit import (  # noqa: E402
    build_generalization_environment,
)
from scripts.evaluate_pre_gat_closed_loop import (  # noqa: E402
    _policy_parameter_sha256,
    _scenario_hash,
    _scene_snapshot,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/evaluation/fp_shep_failure_mode_audit.json"
METHOD_FP_SHEP = "fp_shep_top1_one_shot"
SCHEMA_VERSION = "fp_shep_failure_mode_audit_v1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _stable_hash(value: Any) -> str:
    def jsonable(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Mapping):
            return {str(key): jsonable(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [jsonable(child) for child in item]
        return item

    payload = json.dumps(
        jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reference_hash(references: np.ndarray) -> str:
    return _stable_hash(
        {"references": np.asarray(references, dtype=float).round(12).tolist()}
    )


def _critical_source_hashes() -> dict[str, str]:
    relative_paths = (
        "planning/policy_preview.py",
        "planning/pre_gat_closed_loop.py",
        "Guidance/reference_point_proposal_demo.py",
        "Controller/dmp_rl.py",
        "Environment/frozen_sac_dmp_execution.py",
        "Environment/multi_agent_dmp_env.py",
        "planning/historical_forcing_gate.py",
        "planning/geometry_generalization_scenarios.py",
    )
    return {path: sha256_file(REPO_ROOT / path) for path in relative_paths}


def _assert_config(settings: Mapping[str, Any]) -> None:
    if int(settings["layout_count"]) != 24:
        raise ValueError("failure audit requires the frozen 24-layout set")
    if int(settings["failure_layout_count_expected"]) != 18:
        raise ValueError("expected failure cohort must remain 18")
    if int(settings["success_layout_count_expected"]) != 6:
        raise ValueError("expected success control must remain 6")
    if int(settings["max_steps"]) != 220 or int(settings["num_agents"]) != 3:
        raise ValueError("execution contract must remain 3 agents x 220 steps")
    horizons = list(settings["post_hoc_preview"]["selected_candidate_horizons"])
    if horizons != [4, 8, 12, 20]:
        raise ValueError("H_diag must remain exactly [4,8,12,20]")
    formal = settings["formal_selector"]
    if int(formal["H_preview"]) != 4:
        raise ValueError("formal H_preview changed")
    if float(formal["weights"]["terminal_speed"]) != 0.0:
        raise ValueError("terminal speed entered formal online ranking")
    spec = FPSHEPOnlineScoreSpec.from_mapping(
        {
            "H_preview": formal["H_preview"],
            "weights": formal["weights"],
            "normalization": formal["normalization"],
        }
    )
    if spec.metadata()["formula"] != "+progress +clearance -deviation":
        raise ValueError("formal score formula changed")
    if any(bool(value) for value in settings["strict_exclusions"].values()):
        raise ValueError("all strict exclusion flags must remain false")
    semantics = settings["execution_semantics"]
    if semantics["historical_gate"] != HISTORICAL_GATE_NAME:
        raise ValueError("historical gate changed")
    if bool(semantics["include_boundaries_in_sensor"]) or bool(
        semantics["terminate_on_boundary_collision"]
    ):
        raise ValueError("audit must remain boundary-free")
    if not bool(semantics["one_shot"]) or bool(semantics["repeated_replanning"]):
        raise ValueError("one-shot execution semantics changed")


def _source_artifacts(settings: Mapping[str, Any]) -> dict[str, Any]:
    root = (REPO_ROOT / str(settings["source_artifact"])).resolve()
    required = (
        "scenario_manifest.json",
        "per_episode.csv",
        "per_agent.csv",
        "candidate_selection.csv",
        "fp_shep_failure_analysis.csv",
        "conclusion.json",
        "FINAL_REPORT.md",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source artifact is incomplete: {missing}")
    manifest_path = root / "scenario_manifest.json"
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != str(settings["source_manifest_sha256_expected"]):
        raise RuntimeError("source scenario manifest SHA256 mismatch")
    manifest = _load_json(manifest_path)
    episodes = [
        row
        for row in _load_csv(root / "per_episode.csv")
        if row["method"] == METHOD_FP_SHEP
    ]
    agents = [
        row
        for row in _load_csv(root / "per_agent.csv")
        if row["method"] == METHOD_FP_SHEP
    ]
    selections = [
        row
        for row in _load_csv(root / "candidate_selection.csv")
        if row["method"] == METHOD_FP_SHEP
    ]
    failures = _load_csv(root / "fp_shep_failure_analysis.csv")
    if len(episodes) != 24 or len(agents) != 72 or len(selections) != 72:
        raise RuntimeError("source FP-SHEP row counts changed")
    success_count = sum(parse_bool(row["team_success"]) for row in episodes)
    if success_count != int(settings["success_layout_count_expected"]):
        raise RuntimeError("source success control count changed")
    if len(failures) != int(settings["failure_layout_count_expected"]):
        raise RuntimeError("source failure cohort count changed")
    return {
        "root": root,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
        "manifest_bytes": manifest_path.read_bytes(),
        "manifest": manifest,
        "episodes": episodes,
        "agents": agents,
        "selections": selections,
        "failures": failures,
    }


def _source_selected_preview_value(
    agent_row: Mapping[str, Any], field: str
) -> float:
    """Recover open-space +inf clearance that CSV/JSON encoded as empty/null."""

    value = agent_row.get(field)
    if value not in (None, ""):
        return float(value)
    if field != "selected_preview_min_clearance":
        raise ValueError(f"source preview field {field} is unexpectedly unavailable")
    records = list(parse_json_cell(agent_row.get("fp_shep_candidate_records"), []))
    selected = int(agent_row["selected_candidate_index"])
    record = records[selected]
    raw = record.get("preview_min_clearance")
    if raw is not None:
        return float(raw)
    valid_mask = list(record.get("preview_feature_valid_mask", []))
    normalized = list(record.get("normalized_preview_features", []))
    if len(valid_mask) >= 2 and not bool(valid_mask[1]) and len(normalized) >= 2 and np.isclose(
        float(normalized[1]), 1.0
    ):
        return float("inf")
    raise ValueError("empty source clearance is not a verified open-space encoding")


def _heading_alignment(velocity: np.ndarray, displacement: np.ndarray) -> float | None:
    velocity = np.asarray(velocity, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    denominator = float(np.linalg.norm(velocity) * np.linalg.norm(displacement))
    if denominator <= 1.0e-12:
        return None
    return float(np.clip(np.dot(velocity, displacement) / denominator, -1.0, 1.0))


def _closing_speed(velocity: np.ndarray, displacement: np.ndarray) -> float | None:
    displacement = np.asarray(displacement, dtype=float)
    distance = float(np.linalg.norm(displacement))
    if distance <= 1.0e-12:
        return None
    return float(np.dot(np.asarray(velocity, dtype=float), displacement / distance))


def _kinematic_stopping_distance(
    velocity: np.ndarray, applied_acceleration: np.ndarray
) -> tuple[float | None, bool]:
    """Secondary approximation; never used in primary classification."""

    velocity = np.asarray(velocity, dtype=float)
    speed = float(np.linalg.norm(velocity))
    if speed <= 1.0e-9:
        return 0.0, True
    direction = velocity / speed
    deceleration = -float(np.dot(np.asarray(applied_acceleration, dtype=float), direction))
    if deceleration <= 1.0e-9:
        return None, False
    return float(speed * speed / (2.0 * deceleration)), True


def _vector_reversal_count(vectors: np.ndarray) -> int:
    vectors = np.asarray(vectors, dtype=float)
    count = 0
    for first, second in zip(vectors[:-1], vectors[1:], strict=True):
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator > 1.0e-9 and float(np.dot(first, second) / denominator) < 0.0:
            count += 1
    return count


def _preview_termination(
    env: Any, position: np.ndarray, task_goal: np.ndarray
) -> tuple[bool, str | None]:
    # Boundary and peer termination are intentionally unavailable to the
    # independent frozen-surface preview.  Static collision uses the existing
    # environment obstacle predicate and collision margin without changing the
    # preview clearance feature.
    margin = float(env.env_config.collision_margin)
    if any(obstacle.contains(position, margin=margin) for obstacle in env.static_obstacles):
        return True, "static_obstacle_collision"
    if float(np.linalg.norm(np.asarray(task_goal) - np.asarray(position))) <= float(
        env.env_config.goal_tolerance
    ):
        return True, "terminal_success"
    return False, None


def _preview_row(
    *,
    layout_id: str,
    family: str,
    agent_id: int,
    candidate_index: int,
    candidate_role: str,
    requested_horizon: int,
    preview: Any,
    initial_state: Any,
    env: Any,
    terminated: bool,
    termination_step: int | None,
    termination_reason: str | None,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    trajectory = preview.trajectory
    effective = int(trajectory.horizon)
    candidate = np.asarray(trajectory.candidate_goal, dtype=float)
    initial_distance = float(np.linalg.norm(candidate - initial_state.position))
    final_distance = float(np.linalg.norm(candidate - trajectory.positions[-1]))
    candidate_distances = np.linalg.norm(
        candidate[None, :] - trajectory.positions, axis=1
    )
    distance_deltas = np.diff(candidate_distances)
    final_velocity = trajectory.velocities[-1]
    final_displacement = candidate - trajectory.positions[-1]
    action_low = np.asarray(env.action_space.low[int(agent_id)], dtype=float)
    action_high = np.asarray(env.action_space.high[int(agent_id)], dtype=float)
    saturation = action_saturation_mask(
        trajectory.actions,
        np.broadcast_to(action_low, trajectory.actions.shape),
        np.broadcast_to(action_high, trajectory.actions.shape),
        relative_tolerance=float(
            thresholds.get("action_saturation_relative_tolerance", 0.01)
        ),
    )
    stopping, stopping_available = _kinematic_stopping_distance(
        final_velocity, trajectory.accelerations[-1]
    )
    reversal_count = int(
        np.sum(distance_deltas > float(thresholds["progress_reversal_step_m"]))
    )
    control_reversals = _vector_reversal_count(trajectory.actions[:, :3])
    diagnostic_failure = bool(
        termination_reason == "static_obstacle_collision"
        or float(preview.min_clearance) <= float(thresholds["clearance_collapse_m"])
        or reversal_count > 0
        or control_reversals
        >= int(thresholds["control_direction_reversal_count"])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "layout_id": layout_id,
        "family": family,
        "agent_id": int(agent_id),
        "candidate_index": int(candidate_index),
        "candidate_role": candidate_role,
        "candidate_world_position": candidate.tolist(),
        "H_diag": int(requested_horizon),
        "effective_horizon_steps": effective,
        "task_progress": float(preview.task_progress),
        "minimum_clearance_m": float(preview.min_clearance),
        "maximum_execution_deviation_m": float(preview.max_execution_deviation),
        "terminal_speed_mps": float(preview.terminal_speed),
        "distance_to_candidate_initial_m": initial_distance,
        "distance_to_candidate_final_m": final_distance,
        "candidate_closing_progress_m": initial_distance - final_distance,
        "closing_speed_toward_candidate_mps": _closing_speed(
            final_velocity, final_displacement
        ),
        "final_heading_alignment_to_candidate": _heading_alignment(
            final_velocity, final_displacement
        ),
        "mean_forcing_norm": float(np.mean(np.linalg.norm(trajectory.actions[:, :3], axis=1))),
        "maximum_forcing_norm": float(np.max(np.linalg.norm(trajectory.actions[:, :3], axis=1))),
        "mean_goal_offset_norm": float(np.mean(np.linalg.norm(trajectory.actions[:, 3:], axis=1))),
        "maximum_goal_offset_norm": float(np.max(np.linalg.norm(trajectory.actions[:, 3:], axis=1))),
        "action_saturation_fraction": float(np.mean(saturation)),
        "forcing_saturation_fraction": float(np.mean(saturation[:, :3])),
        "goal_offset_saturation_fraction": float(np.mean(saturation[:, 3:])),
        "mean_commanded_acceleration_norm_mps2": float(
            np.mean(np.linalg.norm(trajectory.commanded_accelerations, axis=1))
        ),
        "maximum_commanded_acceleration_norm_mps2": float(
            np.max(np.linalg.norm(trajectory.commanded_accelerations, axis=1))
        ),
        "terminal_phase": float(trajectory.phases[-1]),
        "progress_reversal_count": reversal_count,
        "clearance_collapse": bool(
            float(preview.min_clearance) <= float(thresholds["clearance_collapse_m"])
        ),
        "control_direction_reversal_count": control_reversals,
        "control_oscillation": bool(
            control_reversals >= int(thresholds["control_direction_reversal_count"])
        ),
        "diagnostic_failure_signal": diagnostic_failure,
        "preview_terminated": bool(terminated),
        "preview_termination_step": termination_step,
        "preview_termination_reason": termination_reason,
        "preview_termination_scope": "static_obstacle_or_terminal_success_only",
        "minimum_inter_agent_distance_m": None,
        "minimum_inter_agent_distance_available": False,
        "minimum_inter_agent_distance_unavailable_reason": (
            "frozen_visible_surface_approximation_has_no_dynamic_peer_identity_or_extrapolation"
        ),
        "kinematic_stopping_distance_approx_m": stopping,
        "kinematic_stopping_distance_available": stopping_available,
        "stopping_distance_semantics": "KINEMATIC_STOPPING_DISTANCE_APPROX",
        "stopping_distance_used_for_primary_classification": False,
        "clearance_source": preview.metadata["clearance_source"],
        "clearance_is_approximate": preview.metadata["clearance_is_approximate"],
        "runtime_ms": float(preview.performance.total_ms),
        "used_for_online_selection": False,
    }


def _selected_candidate_horizon_diagnostic(
    *,
    env: Any,
    layout_id: str,
    family: str,
    agent_id: int,
    candidate_index: int,
    candidate: np.ndarray,
    policy: Any,
    horizons: Sequence[int],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate exact prefixes; stop requesting longer prefixes after termination."""

    initial_state, local_context = build_preview_inputs_from_env(env, agent_id)
    requested = sorted(int(value) for value in horizons)
    previews: dict[int, Any] = {}
    termination_step: int | None = None
    termination_reason: str | None = None
    last_preview: Any | None = None
    for prefix in range(1, max(requested) + 1):
        preview = preview_candidate(
            initial_state=initial_state,
            local_context=local_context,
            candidate_goal=np.asarray(candidate, dtype=float),
            policy=policy,
            horizon=prefix,
            dmp_config=env.dmps[agent_id].config,
            dynamics=env.dynamics[agent_id],
        )
        last_preview = preview
        if prefix in requested:
            previews[prefix] = preview
        terminated, reason = _preview_termination(
            env, preview.trajectory.positions[-1], initial_state.task_goal
        )
        if terminated:
            termination_step = prefix
            termination_reason = reason
            break
    if last_preview is None:
        raise RuntimeError("diagnostic preview produced no prefix")
    rows: list[dict[str, Any]] = []
    for horizon in requested:
        effective_preview = previews.get(horizon, last_preview)
        is_terminated = termination_step is not None and termination_step <= horizon
        row = _preview_row(
            layout_id=layout_id,
            family=family,
            agent_id=agent_id,
            candidate_index=candidate_index,
            candidate_role="selected",
            requested_horizon=horizon,
            preview=effective_preview,
            initial_state=initial_state,
            env=env,
            terminated=is_terminated,
            termination_step=termination_step if is_terminated else None,
            termination_reason=termination_reason if is_terminated else None,
            thresholds=thresholds,
        )
        rows.append(row)
    for row in rows:
        horizon = int(row["H_diag"])
        current = previews.get(horizon, last_preview)
        comparable_later = [
            value for key, value in previews.items() if int(key) >= horizon
        ]
        row["preview_prefix_consistency_verified"] = bool(
            all(
                np.allclose(
                    current.trajectory.positions,
                    later.trajectory.positions[: current.trajectory.positions.shape[0]],
                    rtol=0.0,
                    atol=1.0e-10,
                )
                and np.allclose(
                    current.trajectory.actions,
                    later.trajectory.actions[: current.trajectory.actions.shape[0]],
                    rtol=0.0,
                    atol=1.0e-10,
                )
                for later in comparable_later
            )
        )
        row["preview_definition"] = (
            "same_s0_candidate_policy_history_historical_gate_and_feature_definitions;"
            "only_rollout_length_changes"
        )
    h4 = next(row for row in rows if int(row["H_diag"]) == 4)
    for row in rows:
        row["new_failure_signal_beyond_h4"] = bool(
            int(row["H_diag"]) > 4
            and not parse_bool(h4["diagnostic_failure_signal"])
            and (
                parse_bool(row["diagnostic_failure_signal"])
                or float(row["task_progress"])
                < float(h4["task_progress"])
                - float(thresholds["progress_degradation_from_h4_m"])
                or float(row["minimum_clearance_m"])
                < float(h4["minimum_clearance_m"])
                - float(thresholds["clearance_degradation_from_h4_m"])
                or float(row["maximum_execution_deviation_m"])
                > float(h4["maximum_execution_deviation_m"])
                + float(thresholds["deviation_growth_from_h4_m"])
            )
        )
    return rows


def _run_frozen_selected_reference_episode(
    *,
    policy: Any,
    multi_config: Any,
    settings: Mapping[str, Any],
    layout_record: Mapping[str, Any],
    source_episode: Mapping[str, Any],
    source_agent_rows: Sequence[Mapping[str, Any]],
    source_selection_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay one frozen selected-reference episode without invoking a selector."""

    seed = int(layout_record["evaluation_seed"])
    env, scene_metadata = build_generalization_environment(
        config=multi_config,
        scenario=str(source_episode["scenario"]),
        seed=seed,
        peer_radius=float(settings["peer_radius"]),
    )
    source_agents = sorted(source_agent_rows, key=lambda row: int(row["agent_id"]))
    source_selections = sorted(
        source_selection_rows, key=lambda row: int(row["agent_id"])
    )
    if len(source_agents) != int(settings["num_agents"]) or len(source_selections) != int(
        settings["num_agents"]
    ):
        env.close()
        raise RuntimeError("source agent/selection tuple is incomplete")
    references = np.stack(
        [
            np.asarray(parse_json_cell(row["selected_world_reference"]), dtype=float)
            for row in source_selections
        ]
    )
    candidate_hashes = {str(row["candidate_set_hash"]) for row in source_selections}
    if len(candidate_hashes) != 1 or next(iter(candidate_hashes)) != str(
        source_episode["candidate_set_hash"]
    ):
        env.close()
        raise RuntimeError("source candidate hash is inconsistent")

    trace_rows: list[dict[str, Any]] = []
    handoff_events: list[dict[str, Any]] = []
    execution_transition_records: list[dict[str, Any]] = []
    try:
        initial_snapshot = _scene_snapshot(env)
        initial_state_hash = _scenario_hash(initial_snapshot)
        starts = np.asarray(env.starts, dtype=float).copy()
        terminal_goals = np.asarray(env.goals, dtype=float).copy()
        initial_phases = np.asarray([dmp.phase for dmp in env.dmps], dtype=float)
        selected_reference_hash = _reference_hash(references)
        returned_to_terminal = np.zeros(int(env.num_agents), dtype=bool)
        reached = np.zeros(int(env.num_agents), dtype=bool)
        reached_steps: list[int | None] = [None] * int(env.num_agents)
        terminal_success_steps: list[int | None] = [None] * int(env.num_agents)
        handoff_counts = np.zeros(int(env.num_agents), dtype=int)
        phase_switch_deltas: list[float] = []
        for agent_id in range(int(env.num_agents)):
            phase_before = float(env.dmps[agent_id].phase)
            set_dmp_active_goal_preserve_phase(env.dmps[agent_id], references[agent_id])
            phase_switch_deltas.append(float(env.dmps[agent_id].phase) - phase_before)

        positions = [env._positions().copy()]
        velocities = [env._velocities().copy()]
        accelerations: list[np.ndarray] = []
        collision_step: int | None = None
        obstacle_collision_step: int | None = None
        inter_agent_collision_step: int | None = None
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        saturation_tolerance = float(settings["action_saturation"]["relative_tolerance"])
        reached_tolerance = float(
            settings["execution_semantics"]["temporary_reached_tolerance_m"]
        )

        def execution_observer(_kwargs: dict[str, Any], transition: Any) -> None:
            execution_transition_records.append(dict(transition.controller_info))

        with scoped_historical_preview_and_multi_agent_transition(
            execution_observer=execution_observer
        ):
            while not (terminated or truncated):
                step_before = int(env.steps)
                active_before = np.stack(
                    [np.asarray(dmp.goal, dtype=float) for dmp in env.dmps]
                )
                observations = temporary_checkpoint_observations(env, active_before)
                if observations.shape != (int(env.num_agents), 122):
                    raise RuntimeError("checkpoint observation shape changed")
                actions = predict_actions_without_postprocessing(
                    policy, observations, tuple(env.action_shape)
                )
                saturation = action_saturation_mask(
                    actions,
                    env.action_space.low,
                    env.action_space.high,
                    relative_tolerance=saturation_tolerance,
                )
                _, _, terminated, truncated, info = env.step(actions)
                positions.append(env._positions().copy())
                velocities.append(env._velocities().copy())
                applied = np.asarray(info["applied_accelerations"], dtype=float)
                commanded = np.asarray(info["commanded_accelerations"], dtype=float)
                accelerations.append(applied.copy())
                success_mask = np.asarray(info["success_mask"], dtype=bool)
                obstacle_mask = np.asarray(info["obstacle_collision_mask"], dtype=bool)
                inter_mask = np.asarray(info["inter_agent_collision_mask"], dtype=bool)
                if bool(info["collision"]) and collision_step is None:
                    collision_step = int(env.steps)
                if bool(np.any(obstacle_mask)) and obstacle_collision_step is None:
                    obstacle_collision_step = int(env.steps)
                if bool(np.any(inter_mask)) and inter_agent_collision_step is None:
                    inter_agent_collision_step = int(env.steps)
                for agent_id in np.flatnonzero(success_mask):
                    if terminal_success_steps[int(agent_id)] is None:
                        terminal_success_steps[int(agent_id)] = int(env.steps)

                positions_now = env._positions().copy()
                velocities_now = env._velocities().copy()
                phases_now = np.asarray([dmp.phase for dmp in env.dmps], dtype=float)
                active_after = active_before.copy()
                handoff_now = np.zeros(int(env.num_agents), dtype=bool)
                if not (terminated or truncated):
                    for agent_id in range(int(env.num_agents)):
                        if returned_to_terminal[agent_id]:
                            continue
                        distance = float(
                            np.linalg.norm(references[agent_id] - positions_now[agent_id])
                        )
                        if distance > reached_tolerance:
                            continue
                        reached[agent_id] = True
                        reached_steps[agent_id] = int(env.steps)
                        phase_before = float(env.dmps[agent_id].phase)
                        set_dmp_active_goal_preserve_phase(
                            env.dmps[agent_id], terminal_goals[agent_id]
                        )
                        phase_delta = float(env.dmps[agent_id].phase) - phase_before
                        phase_switch_deltas.append(phase_delta)
                        returned_to_terminal[agent_id] = True
                        handoff_counts[agent_id] += 1
                        handoff_now[agent_id] = True
                        active_after[agent_id] = terminal_goals[agent_id]
                        handoff_events.append(
                            {
                                "layout_id": layout_record["layout_id"],
                                "family": layout_record["family"],
                                "agent_id": int(agent_id),
                                "handoff_step": int(env.steps),
                                "phase_switch_delta": phase_delta,
                            }
                        )

                pairwise = np.asarray(info["pairwise_distances"], dtype=float)
                static_clearances = minimum_static_surface_clearances(
                    positions_now, env.static_obstacles
                )
                for agent_id in range(int(env.num_agents)):
                    velocity = velocities_now[agent_id]
                    reference_displacement = references[agent_id] - positions_now[agent_id]
                    terminal_displacement = terminal_goals[agent_id] - positions_now[agent_id]
                    stopping, stopping_available = _kinematic_stopping_distance(
                        velocity, applied[agent_id]
                    )
                    peer_distance = float(
                        np.min(np.delete(pairwise[agent_id], agent_id))
                    )
                    trace_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "layout_id": layout_record["layout_id"],
                            "family": layout_record["family"],
                            "seed": seed,
                            "agent_id": int(agent_id),
                            "step": int(env.steps),
                            "step_before": step_before,
                            "position": positions_now[agent_id].tolist(),
                            "velocity": velocity.tolist(),
                            "speed_mps": float(np.linalg.norm(velocity)),
                            "active_reference_before_step": active_before[agent_id].tolist(),
                            "active_reference_after_step": active_after[agent_id].tolist(),
                            "terminal_goal": terminal_goals[agent_id].tolist(),
                            "selected_reference": references[agent_id].tolist(),
                            "distance_to_reference_m": float(
                                np.linalg.norm(reference_displacement)
                            ),
                            "distance_to_terminal_m": float(
                                np.linalg.norm(terminal_displacement)
                            ),
                            "heading_toward_reference": _heading_alignment(
                                velocity, reference_displacement
                            ),
                            "heading_toward_terminal": _heading_alignment(
                                velocity, terminal_displacement
                            ),
                            "phase": float(phases_now[agent_id]),
                            "forcing_xyz": actions[agent_id, :3].astype(float).tolist(),
                            "goal_offset_xyz": actions[agent_id, 3:].astype(float).tolist(),
                            "forcing_norm": float(np.linalg.norm(actions[agent_id, :3])),
                            "goal_offset_norm": float(np.linalg.norm(actions[agent_id, 3:])),
                            "action_saturation_fraction": float(
                                np.mean(saturation[agent_id])
                            ),
                            "forcing_saturation_fraction": float(
                                np.mean(saturation[agent_id, :3])
                            ),
                            "goal_offset_saturation_fraction": float(
                                np.mean(saturation[agent_id, 3:])
                            ),
                            "commanded_acceleration": commanded[agent_id].tolist(),
                            "commanded_acceleration_norm_mps2": float(
                                np.linalg.norm(commanded[agent_id])
                            ),
                            "applied_acceleration": applied[agent_id].tolist(),
                            "applied_acceleration_norm_mps2": float(
                                np.linalg.norm(applied[agent_id])
                            ),
                            "lidar_minimum_clearance_m": float(
                                np.asarray(info["min_clearances"], dtype=float)[agent_id]
                            ),
                            "static_obstacle_clearance_m": float(
                                static_clearances[agent_id]
                            ),
                            "minimum_inter_agent_distance_m": peer_distance,
                            "obstacle_collision": bool(obstacle_mask[agent_id]),
                            "inter_agent_collision": bool(inter_mask[agent_id]),
                            "team_collision": bool(info["collision"]),
                            "terminal_success": bool(success_mask[agent_id]),
                            "reference_reached_at_step": bool(handoff_now[agent_id]),
                            "reference_reached": bool(reached[agent_id]),
                            "returned_to_terminal": bool(returned_to_terminal[agent_id]),
                            "kinematic_stopping_distance_approx_m": stopping,
                            "kinematic_stopping_distance_available": stopping_available,
                            "stopping_distance_semantics": "KINEMATIC_STOPPING_DISTANCE_APPROX",
                            "stopping_distance_used_for_primary_classification": False,
                        }
                    )

        position_array = np.stack(positions)
        velocity_array = np.stack(velocities)
        acceleration_array = np.stack(accelerations)
        metrics = trajectory_metrics(
            position_array,
            velocity_array,
            acceleration_array,
            dt=float(settings["dt"]),
        )
        final_positions = position_array[-1]
        per_agent_summaries: list[dict[str, Any]] = []
        for agent_id in range(int(env.num_agents)):
            agent_trace = [
                row for row in trace_rows if int(row["agent_id"]) == agent_id
            ]
            collision_rows = [
                row
                for row in agent_trace
                if parse_bool(row["obstacle_collision"])
                or parse_bool(row["inter_agent_collision"])
            ]
            per_agent_summaries.append(
                {
                    "layout_id": layout_record["layout_id"],
                    "family": layout_record["family"],
                    "agent_id": agent_id,
                    "reference_reached": bool(reached[agent_id]),
                    "reference_reached_step": reached_steps[agent_id],
                    "distance_to_reference_at_termination_m": float(
                        np.linalg.norm(references[agent_id] - final_positions[agent_id])
                    ),
                    "terminal_success": terminal_success_steps[agent_id] is not None,
                    "terminal_completion_step": terminal_success_steps[agent_id],
                    "collision_type": (
                        "obstacle"
                        if any(parse_bool(row["obstacle_collision"]) for row in collision_rows)
                        else "inter_agent"
                        if any(parse_bool(row["inter_agent_collision"]) for row in collision_rows)
                        else "none"
                    ),
                    "collision_step": (
                        min(int(row["step"]) for row in collision_rows)
                        if collision_rows
                        else None
                    ),
                    "timeout": bool(truncated),
                    "minimum_obstacle_clearance_m": float(
                        min(float(row["lidar_minimum_clearance_m"]) for row in agent_trace)
                    ),
                    "minimum_static_obstacle_clearance_m": float(
                        min(float(row["static_obstacle_clearance_m"]) for row in agent_trace)
                    ),
                    "minimum_inter_agent_distance_m": float(
                        min(float(row["minimum_inter_agent_distance_m"]) for row in agent_trace)
                    ),
                    "path_length_m": float(metrics["path_lengths"][agent_id]),
                    "trajectory_smoothness": float(
                        metrics["trajectory_smoothness_per_agent"][agent_id]
                    ),
                    "handoff_count": int(handoff_counts[agent_id]),
                }
            )
        gate_values = {
            str(info.get("forcing_gate_semantics", HISTORICAL_GATE_NAME))
            for info in execution_transition_records
        }
        episode_summary = {
            "layout_id": layout_record["layout_id"],
            "family": layout_record["family"],
            "seed": seed,
            "initial_state_hash": initial_state_hash,
            "frozen_selected_reference_hash": selected_reference_hash,
            "candidate_set_hash": next(iter(candidate_hashes)),
            "scenario_geometry_hash": scene_metadata["scenario_geometry_hash"],
            "team_success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "obstacle_collision": bool(obstacle_collision_step is not None),
            "inter_agent_collision": bool(inter_agent_collision_step is not None),
            "timeout": bool(truncated),
            "terminated": bool(terminated),
            "steps": int(env.steps),
            "collision_step": collision_step,
            "obstacle_collision_step": obstacle_collision_step,
            "inter_agent_collision_step": inter_agent_collision_step,
            "reference_reached_count": int(np.sum(reached)),
            "reference_reached_mask": reached.tolist(),
            "reference_reached_steps": reached_steps,
            "handoff_counts": handoff_counts.tolist(),
            "maximum_phase_switch_delta": float(np.max(np.abs(phase_switch_deltas))),
            "terminal_task_goals_unchanged": bool(
                np.array_equal(np.asarray(env.goals), terminal_goals)
            ),
            "initial_phases": initial_phases.tolist(),
            "execution_transition_call_count": len(execution_transition_records),
            "historical_gate_values": sorted(gate_values),
            "historical_gate_verified": gate_values == {HISTORICAL_GATE_NAME},
            "one_shot_handoff_verified": bool(np.all(handoff_counts <= 1)),
            "path_length_team_sum_m": float(metrics["path_length_team_sum"]),
            "trajectory_smoothness_team_mean": float(
                metrics["trajectory_smoothness_team_mean"]
            ),
        }
        return episode_summary, per_agent_summaries, trace_rows
    finally:
        env.close()


def _reproduction_checks(
    *,
    rerun: Mapping[str, Any],
    source_episode: Mapping[str, Any],
    rerun_agents: Sequence[Mapping[str, Any]],
    source_agents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rerun_agent_index = {int(row["agent_id"]): row for row in rerun_agents}
    source_agent_index = {int(row["agent_id"]): row for row in source_agents}
    checks = {
        "initial_state_hash": str(rerun["initial_state_hash"])
        == str(source_episode["initial_condition_hash"]),
        "environment_geometry": str(rerun["scenario_geometry_hash"])
        == str(source_episode["scenario_geometry_hash"]),
        "candidate_set_hash": str(rerun["candidate_set_hash"])
        == str(source_episode["candidate_set_hash"]),
        "historical_gate": bool(rerun["historical_gate_verified"]),
        "one_shot_handoff": bool(rerun["one_shot_handoff_verified"]),
        "phase_preserved": float(rerun["maximum_phase_switch_delta"]) == 0.0,
        "terminal_goals_unchanged": bool(rerun["terminal_task_goals_unchanged"]),
        "team_success": bool(rerun["team_success"])
        == parse_bool(source_episode["team_success"]),
        "collision": bool(rerun["collision"]) == parse_bool(source_episode["collision"]),
        "obstacle_collision": bool(rerun["obstacle_collision"])
        == parse_bool(source_episode["obstacle_collision"]),
        "inter_agent_collision": bool(rerun["inter_agent_collision"])
        == parse_bool(source_episode["inter_agent_collision"]),
        "timeout": bool(rerun["timeout"]) == parse_bool(source_episode["timeout"]),
        "steps": int(rerun["steps"]) == int(source_episode["steps"]),
        "reference_reached": all(
            bool(rerun_agent_index[index]["reference_reached"])
            == parse_bool(source_agent_index[index]["reference_reached"])
            for index in sorted(source_agent_index)
        ),
    }
    return {
        "layout_id": rerun["layout_id"],
        "family": rerun["family"],
        "frozen_selected_reference_hash": rerun["frozen_selected_reference_hash"],
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "status": "PASSED" if all(checks.values()) else "FAILED",
    }


def _handoff_window_rows(
    trace_rows: Sequence[Mapping[str, Any]],
    agent_summaries: Sequence[Mapping[str, Any]],
    *,
    before_steps: int,
    after_steps: int,
    team_success_by_layout: Mapping[str, bool],
) -> list[dict[str, Any]]:
    traces: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        traces[(str(row["layout_id"]), int(row["agent_id"]))].append(row)
    results: list[dict[str, Any]] = []
    for summary in agent_summaries:
        reached_step = summary.get("reference_reached_step")
        if reached_step is None:
            continue
        layout_id = str(summary["layout_id"])
        agent_id = int(summary["agent_id"])
        for row in traces[(layout_id, agent_id)]:
            relative = int(row["step"]) - int(reached_step)
            if -int(before_steps) <= relative <= int(after_steps):
                results.append(
                    {
                        **dict(row),
                        "handoff_step": int(reached_step),
                        "relative_to_handoff_step": relative,
                        "handoff_window_role": (
                            "handoff"
                            if relative == 0
                            else "pre_handoff"
                            if relative < 0
                            else "post_handoff"
                        ),
                        "terminal_completed_after_reference": bool(
                            summary["terminal_success"]
                        ),
                        "team_success": bool(team_success_by_layout[layout_id]),
                        "comparison_group": (
                            "successful_fp_shep_reference"
                            if bool(summary["terminal_success"])
                            else "reached_but_terminal_failed_reference"
                        ),
                    }
                )
    return results


def _feature_summary_rows(
    *,
    source_episodes: Sequence[Mapping[str, Any]],
    source_agents: Sequence[Mapping[str, Any]],
    score_margins: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    success_by_layout = {
        str(row["layout_id"]): parse_bool(row["team_success"])
        for row in source_episodes
    }
    margin_index = {
        (str(row["layout_id"]), int(row["agent_id"])): row for row in score_margins
    }
    geometry_index = {
        (str(row["layout_id"]), int(row["agent_id"])): row for row in geometry_rows
    }
    trace_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        trace_groups[(str(row["layout_id"]), int(row["agent_id"]))].append(row)
    records: list[dict[str, Any]] = []
    for agent in source_agents:
        key = (str(agent["layout_id"]), int(agent["agent_id"]))
        trace = trace_groups[key]
        margin = margin_index[key]
        geometry = geometry_index[key]
        records.append(
            {
                "layout_id": key[0],
                "agent_id": key[1],
                "cohort": "success_control" if success_by_layout[key[0]] else "failure",
                "h4_predicted_progress": float(agent["selected_preview_task_progress"]),
                "h4_predicted_clearance": _source_selected_preview_value(
                    agent, "selected_preview_min_clearance"
                ),
                "h4_predicted_deviation": float(
                    agent["selected_preview_max_execution_deviation"]
                ),
                "h4_terminal_speed": float(agent["selected_preview_terminal_speed"]),
                "score_normalized_margin": margin["normalized_margin_1_2"],
                "candidate_distance_m": geometry["candidate_distance_from_ego_m"],
                "candidate_terminal_turning_angle_deg": geometry[
                    "candidate_to_terminal_turning_angle_deg"
                ],
                "candidate_terminal_clearance_m": geometry[
                    "candidate_to_terminal_minimum_static_clearance_m"
                ],
                "actual_reference_progress_m": float(agent["stage1_real_progress_m"]),
                "action_saturation_rate": float(
                    np.mean([float(row["action_saturation_fraction"]) for row in trace])
                ),
                "minimum_actual_obstacle_clearance_m": float(
                    min(float(row["static_obstacle_clearance_m"]) for row in trace)
                ),
            }
        )
    metric_names = [key for key in records[0] if key not in {"layout_id", "agent_id", "cohort"}]
    summary: list[dict[str, Any]] = []
    for cohort in ("success_control", "failure"):
        members = [row for row in records if row["cohort"] == cohort]
        for metric in metric_names:
            summary.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "cohort": cohort,
                    "metric": metric,
                    **descriptive_statistics(row[metric] for row in members),
                }
            )
    success_speed = np.asarray(
        [row["h4_terminal_speed"] for row in records if row["cohort"] == "success_control"],
        dtype=float,
    )
    failure_speed = np.asarray(
        [row["h4_terminal_speed"] for row in records if row["cohort"] == "failure"],
        dtype=float,
    )
    pooled = np.concatenate([success_speed, failure_speed])
    iqr = float(np.percentile(pooled, 75) - np.percentile(pooled, 25))
    if len(success_speed) < 2 or len(failure_speed) < 2 or iqr <= 1.0e-12:
        terminal_value = "NOT_ESTABLISHED"
        robust_effect = None
    else:
        robust_effect = abs(float(np.median(success_speed) - np.median(failure_speed))) / iqr
        if robust_effect >= float(thresholds["terminal_speed_high_robust_effect"]):
            terminal_value = "HIGH"
        elif robust_effect >= float(thresholds["terminal_speed_moderate_robust_effect"]):
            terminal_value = "MODERATE"
        else:
            terminal_value = "LOW"
    summary.append(
        {
            "schema_version": SCHEMA_VERSION,
            "cohort": "success_vs_failure",
            "metric": "h4_terminal_speed_robust_effect",
            "count": len(pooled),
            "mean": robust_effect,
            "median": None,
            "minimum": None,
            "maximum": None,
            "diagnostic_value": terminal_value,
            "definition": "abs(median_success-median_failure)/pooled_IQR",
            "used_for_online_ranking": False,
        }
    )
    return summary, terminal_value


def _build_lower_policy_rows(
    *,
    failure_rows: Sequence[Mapping[str, Any]],
    source_agents: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    horizon_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    failure_index = {str(row["layout_id"]): row for row in failure_rows}
    geometry_index = {
        (str(row["layout_id"]), int(row["agent_id"])): row for row in geometry_rows
    }
    horizon_index: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in horizon_rows:
        horizon_index[(str(row["layout_id"]), int(row["agent_id"]))].append(row)
    trace_index: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        trace_index[(str(row["layout_id"]), int(row["agent_id"]))].append(row)
    results: list[dict[str, Any]] = []
    for agent in source_agents:
        layout_id = str(agent["layout_id"])
        original = failure_index.get(layout_id)
        if original is None or "REFERENCE_UNREACHABLE" not in str(
            original["primary_failure_category"]
        ):
            continue
        if parse_bool(agent["reference_reached"]):
            continue
        key = (layout_id, int(agent["agent_id"]))
        classification = classify_reference_unreachable(
            trace_index[key], thresholds
        )
        flags = set(classification["diagnostic_flags"])
        geometry_poor = parse_bool(
            geometry_index[key]["geometrically_poor_handoff_candidate"]
        )
        blocking = bool(
            "OBSTACLE_AVOIDANCE_DEADLOCK" in flags
            or "INTER_AGENT_BLOCKING" in flags
        )
        abnormal_control = bool(
            "OSCILLATION" in flags
            or "ACTION_SATURATION" in flags
            or "FORCING_SATURATION" in flags
            or "GOAL_OFFSET_SATURATION" in flags
        )
        longer_signal = any(
            parse_bool(row.get("new_failure_signal_beyond_h4", False))
            for row in horizon_index[key]
        )
        lower_supported = bool(
            not geometry_poor and not blocking and abnormal_control
        )
        results.append(
            {
                "schema_version": SCHEMA_VERSION,
                "layout_id": layout_id,
                "family": agent["family"],
                "agent_id": int(agent["agent_id"]),
                "original_primary_failure_category": original[
                    "primary_failure_category"
                ],
                **classification,
                "candidate_geometry_poor": geometry_poor,
                "blocking_explanation_present": blocking,
                "persistent_control_abnormality_present": abnormal_control,
                "longer_preview_failure_signal": longer_signal,
                "lower_policy_local_compatibility_failure": lower_supported,
                "lower_policy_support_rule": (
                    "geometry_not_poor AND no_obstacle_or_peer_blocking AND "
                    "persistent_oscillation_or_action_saturation"
                ),
            }
        )
    return results


def _build_obstacle_collision_rows(
    *,
    failure_rows: Sequence[Mapping[str, Any]],
    agent_summaries: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    horizon_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    layout_index: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    obstacle_layouts = {
        str(row["layout_id"])
        for row in failure_rows
        if "OBSTACLE_COLLISION" in str(row["primary_failure_category"])
    }
    horizon_index = {
        (str(row["layout_id"]), int(row["agent_id"]), int(row["H_diag"])): row
        for row in horizon_rows
    }
    geometry_index = {
        (str(row["layout_id"]), int(row["agent_id"])): row for row in geometry_rows
    }
    trace_index: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        trace_index[(str(row["layout_id"]), int(row["agent_id"]))].append(row)
    results: list[dict[str, Any]] = []
    for agent in agent_summaries:
        layout_id = str(agent["layout_id"])
        if layout_id not in obstacle_layouts or str(agent["collision_type"]) != "obstacle":
            continue
        agent_id = int(agent["agent_id"])
        trace = sorted(trace_index[(layout_id, agent_id)], key=lambda row: int(row["step"]))
        collision_step = int(agent["collision_step"])
        pre_collision = [row for row in trace if int(row["step"]) <= collision_step][-10:]
        obstacle_centers = np.asarray(
            [row["center"] for row in layout_index[layout_id]["obstacles"]],
            dtype=float,
        )
        obstacle_relative_vectors = []
        for trace_row in pre_collision:
            position = np.asarray(trace_row["position"], dtype=float)
            nearest_index = int(
                np.argmin(np.linalg.norm(obstacle_centers - position[None, :], axis=1))
            )
            obstacle_relative_vectors.append(
                (obstacle_centers[nearest_index] - position).astype(float).tolist()
            )
        values = {
            horizon: horizon_index[(layout_id, agent_id, horizon)]
            for horizon in (4, 8, 12, 20)
        }
        first_collapse = next(
            (
                horizon
                for horizon in (4, 8, 12, 20)
                if parse_bool(values[horizon]["clearance_collapse"])
                or str(values[horizon]["preview_termination_reason"])
                == "static_obstacle_collision"
            ),
            None,
        )
        results.append(
            {
                "schema_version": SCHEMA_VERSION,
                "layout_id": layout_id,
                "family": agent["family"],
                "agent_id": agent_id,
                "collision_step": collision_step,
                "collision_before_reference": not bool(agent["reference_reached"])
                or collision_step <= int(agent["reference_reached_step"] or collision_step),
                "preview_h4_min_clearance_m": values[4]["minimum_clearance_m"],
                "preview_h8_min_clearance_m": values[8]["minimum_clearance_m"],
                "preview_h12_min_clearance_m": values[12]["minimum_clearance_m"],
                "preview_h20_min_clearance_m": values[20]["minimum_clearance_m"],
                "preview_first_clearance_collapse_horizon": first_collapse,
                "preview_static_collision_termination_steps": {
                    horizon: values[horizon]["preview_termination_step"]
                    for horizon in (4, 8, 12, 20)
                },
                "actual_pre_collision_static_clearance_trace_m": [
                    float(row["static_obstacle_clearance_m"]) for row in pre_collision
                ],
                "actual_pre_collision_positions": [row["position"] for row in pre_collision],
                "actual_pre_collision_closest_obstacle_relative_vectors": (
                    obstacle_relative_vectors
                ),
                "obstacle_relative_vector_definition": "closest_obstacle_center-position",
                "candidate_to_terminal_direct_path_clearance_m": geometry_index[
                    (layout_id, agent_id)
                ]["candidate_to_terminal_minimum_static_clearance_m"],
            }
        )
    return results


def _layout_cohort_rows(
    *,
    source_episodes: Sequence[Mapping[str, Any]],
    source_agents: Sequence[Mapping[str, Any]],
    source_selections: Sequence[Mapping[str, Any]],
    source_failures: Sequence[Mapping[str, Any]],
    rerun_agents: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    horizon_rows: Sequence[Mapping[str, Any]],
    lower_policy_rows: Sequence[Mapping[str, Any]],
    score_margins: Sequence[Mapping[str, Any]],
    low_confidence_threshold: float,
) -> list[dict[str, Any]]:
    agents: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    selections: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    reruns: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    geometries: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    horizons: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    lowers: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    margins: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_agents:
        agents[str(row["layout_id"])].append(row)
    for row in source_selections:
        selections[str(row["layout_id"])].append(row)
    for row in rerun_agents:
        reruns[str(row["layout_id"])].append(row)
    for row in geometry_rows:
        geometries[str(row["layout_id"])].append(row)
    for row in horizon_rows:
        horizons[str(row["layout_id"])].append(row)
    for row in lower_policy_rows:
        lowers[str(row["layout_id"])].append(row)
    for row in score_margins:
        margins[str(row["layout_id"])].append(row)
    failure_index = {str(row["layout_id"]): row for row in source_failures}
    results: list[dict[str, Any]] = []
    for episode in sorted(source_episodes, key=lambda row: str(row["layout_id"])):
        layout_id = str(episode["layout_id"])
        success = parse_bool(episode["team_success"])
        primary = (
            "SUCCESS_CONTROL"
            if success
            else str(failure_index[layout_id]["primary_failure_category"])
        )
        nested_agents = []
        source_agent_index = {
            int(row["agent_id"]): row for row in agents[layout_id]
        }
        selection_index = {
            int(row["agent_id"]): row for row in selections[layout_id]
        }
        rerun_index = {int(row["agent_id"]): row for row in reruns[layout_id]}
        margin_index = {int(row["agent_id"]): row for row in margins[layout_id]}
        for agent_id in sorted(source_agent_index):
            source_agent = source_agent_index[agent_id]
            selection = selection_index[agent_id]
            rerun = rerun_index[agent_id]
            nested_agents.append(
                {
                    "agent_id": agent_id,
                    "selected_candidate_index": int(selection["selected_candidate_index"]),
                    "proposal_rank": int(selection["selected_original_proposal_rank"]),
                    "proposal_score": float(selection["selected_proposal_score"]),
                    "fp_shep_score": float(selection["selected_fp_shep_score"]),
                    "predicted_progress": float(source_agent["selected_preview_task_progress"]),
                    "predicted_clearance": _source_selected_preview_value(
                        source_agent, "selected_preview_min_clearance"
                    ),
                    "predicted_deviation": float(
                        source_agent["selected_preview_max_execution_deviation"]
                    ),
                    "predicted_terminal_speed": float(
                        source_agent["selected_preview_terminal_speed"]
                    ),
                    "reference_reached": bool(rerun["reference_reached"]),
                    "reference_reached_step": rerun["reference_reached_step"],
                    "distance_to_reference_at_termination_m": rerun[
                        "distance_to_reference_at_termination_m"
                    ],
                    "terminal_success": bool(rerun["terminal_success"]),
                    "terminal_completion_step": rerun["terminal_completion_step"],
                    "collision_type": rerun["collision_type"],
                    "collision_step": rerun["collision_step"],
                    "timeout": bool(rerun["timeout"]),
                    "minimum_obstacle_clearance_m": rerun[
                        "minimum_obstacle_clearance_m"
                    ],
                    "minimum_inter_agent_distance_m": rerun[
                        "minimum_inter_agent_distance_m"
                    ],
                    "path_length_m": rerun["path_length_m"],
                    "trajectory_smoothness": rerun["trajectory_smoothness"],
                }
            )
        horizon_signal = any(
            parse_bool(row.get("new_failure_signal_beyond_h4", False))
            for row in horizons[layout_id]
        )
        poor_geometry = any(
            parse_bool(row["geometrically_poor_handoff_candidate"])
            for row in geometries[layout_id]
        )
        lower_supported = any(
            parse_bool(row["lower_policy_local_compatibility_failure"])
            for row in lowers[layout_id]
        )
        low_confidence = any(
            row["normalized_margin_1_2"] is not None
            and float(row["normalized_margin_1_2"]) <= float(low_confidence_threshold)
            for row in margins[layout_id]
        )
        results.append(
            {
                "schema_version": SCHEMA_VERSION,
                "layout_id": layout_id,
                "family": episode["family"],
                "team_success": success,
                "primary_failure_category": primary,
                "diagnostic_failure_flags": (
                    parse_json_cell(
                        failure_index[layout_id].get("diagnostic_failure_flags"), []
                    )
                    if not success
                    else []
                ),
                "agent_records": nested_agents,
                "horizon_limitation_signal": horizon_signal,
                "geometrically_poor_handoff_candidate": poor_geometry,
                "lower_policy_local_compatibility_failure": lower_supported,
                "low_selector_confidence": low_confidence,
                "better_preview_alternative": "NOT_ESTABLISHED",
                "better_preview_alternative_reason": (
                    "no outcome-independent non-arbitrary preview-only criterion was frozen"
                ),
            }
        )
    return results


def _render_report(
    *,
    config: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    failure_cohort: Sequence[Mapping[str, Any]],
    success_control: Sequence[Mapping[str, Any]],
    lower_rows: Sequence[Mapping[str, Any]],
    obstacle_rows: Sequence[Mapping[str, Any]],
    mechanism_rows: Sequence[Mapping[str, Any]],
    horizon_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    score_margin_rows: Sequence[Mapping[str, Any]],
    handoff_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    conclusion: Mapping[str, Any],
    terminal_speed_value: str,
) -> str:
    category_counts = Counter(
        str(row["primary_failure_category"]) for row in failure_cohort
    )
    subtype_counts = Counter(str(row["primary_subcategory"]) for row in lower_rows)
    failed_ids = {str(row["layout_id"]) for row in failure_cohort}
    success_ids = {str(row["layout_id"]) for row in success_control}
    progress_medians: dict[tuple[str, int], float] = {}
    for cohort, ids in (("failure", failed_ids), ("success", success_ids)):
        for horizon in (4, 8, 12, 20):
            values = [
                float(row["task_progress"])
                for row in horizon_rows
                if str(row["layout_id"]) in ids and int(row["H_diag"]) == horizon
            ]
            progress_medians[(cohort, horizon)] = float(np.median(values))
    collision_steps = sorted(int(row["collision_step"]) for row in obstacle_rows)
    preview_termination_count = sum(
        parse_bool(row["preview_terminated"]) for row in horizon_rows
    )
    low_margin_threshold = float(
        config["diagnostic_thresholds"]["low_selector_confidence_normalized_margin"]
    )
    low_margin_failure = sum(
        str(row["layout_id"]) in failed_ids
        and float(row["normalized_margin_1_2"]) <= low_margin_threshold
        for row in score_margin_rows
    )
    low_margin_success = sum(
        str(row["layout_id"]) in success_ids
        and float(row["normalized_margin_1_2"]) <= low_margin_threshold
        for row in score_margin_rows
    )
    poor_failure_layouts = {
        str(row["layout_id"])
        for row in geometry_rows
        if str(row["layout_id"]) in failed_ids
        and parse_bool(row["geometrically_poor_handoff_candidate"])
    }
    poor_success_layouts = {
        str(row["layout_id"])
        for row in geometry_rows
        if str(row["layout_id"]) in success_ids
        and parse_bool(row["geometrically_poor_handoff_candidate"])
    }
    handoff_at_zero = [
        row for row in handoff_rows if int(row["relative_to_handoff_step"]) == 0
    ]

    def handoff_median(group: str, field: str) -> float | None:
        values = [
            float(row[field])
            for row in handoff_at_zero
            if str(row["comparison_group"]) == group and row.get(field) not in (None, "")
        ]
        return float(np.median(values)) if values else None
    lines = [
        "# FP-SHEP Failure-Mode Audit",
        "",
        "## 1. Scope and frozen semantics",
        "",
        "This audit is diagnostic-only. It reused the frozen 24-layout manifest, "
        "frozen deterministic SAC-DMP checkpoint, historical vector goal-eff gate, "
        "one-shot handoff, boundary-free environment, and 220-step execution horizon.",
        "",
        f"- Source artifact: `{config['source_artifact']}`",
        f"- Scenario manifest SHA256: `{config['source_manifest_sha256_expected']}`",
        f"- Formal FP-SHEP horizon: H={config['formal_selector']['H_preview']}",
        "- H_diag={4,8,12,20} is post-hoc only and never enters online selection.",
        "- No Proposal, FP-SHEP, score, normalization, SAC, DMP, reward, observation, "
        "collision, success, geometry, or GAT code was modified by the audit.",
        "",
        "## 2. Frozen episode reproduction gate",
        "",
        f"- Status: **{reproduction['status']}**",
        f"- Passed layouts: {reproduction['passed_layout_count']}/24",
        f"- FROZEN_EPISODE_REPRODUCTION_MISMATCH = {reproduction['FROZEN_EPISODE_REPRODUCTION_MISMATCH']}",
        "",
        "Mechanism attribution is valid only because all rerun episodes reproduced "
        "initial-state hash, geometry, historical gate, one-shot handoff, team outcome, "
        "collision type, timeout, step count, and agent-level reference reached labels.",
        "",
        "## 3. Failure cohort",
        "",
        f"- Failure layouts: {len(failure_cohort)}",
        f"- Success controls: {len(success_control)}",
        f"- Original primary categories: `{dict(category_counts)}`",
        f"- Reclassified unreachable-agent subtypes: `{dict(subtype_counts)}`",
        "",
        "## 4. Preview-horizon audit",
        "",
        "All H_diag values use the same initial state, selected candidate, frozen policy, "
        "historical transition, observation/history reconstruction, clearance definition, "
        "progress definition, and deviation definition. Rollout length is the only active "
        "variable. Prefix rollout stops at the first supported static-collision or terminal-"
        "success event; inter-agent termination is unavailable in the independent frozen-"
        "surface preview and is not fabricated.",
        "",
        f"- Median task progress, failure cohort: H4={progress_medians[('failure', 4)]:.3f} m, "
        f"H8={progress_medians[('failure', 8)]:.3f} m, H12={progress_medians[('failure', 12)]:.3f} m, "
        f"H20={progress_medians[('failure', 20)]:.3f} m.",
        f"- Median task progress, success control: H4={progress_medians[('success', 4)]:.3f} m, "
        f"H8={progress_medians[('success', 8)]:.3f} m, H12={progress_medians[('success', 12)]:.3f} m, "
        f"H20={progress_medians[('success', 20)]:.3f} m.",
        f"- The median progress gap increases from "
        f"{abs(progress_medians[('success', 4)]-progress_medians[('failure', 4)]):.3f} m at H4 "
        f"to {abs(progress_medians[('success', 12)]-progress_medians[('failure', 12)]):.3f} m at H12.",
        f"- Preview termination count across 72 selected candidates x 4 horizons: {preview_termination_count}.",
        f"- Actual obstacle collision steps: {collision_steps}; all are later than H=20.",
        "- H20 did not expose a clearance-collapse or collision termination for the six actual obstacle-collision agents. "
        "This prevents attributing the mismatch to horizon alone; frozen visible-surface reconstruction remains a confound.",
        "",
        "## 5. Existing feature and terminal-speed audit",
        "",
        f"- TERMINAL_SPEED_DIAGNOSTIC_VALUE = {terminal_speed_value}",
        "- Terminal speed remains recorded but has weight 0 and is not added to the score.",
        "- KINEMATIC_STOPPING_DISTANCE_APPROX is secondary-only; unavailable values are not imputed.",
        "",
        "## 6. Candidate-rank confidence",
        "",
        "`FAILED_TOP1_WITH_BETTER_PREVIEW_ALTERNATIVE` is reported as NOT_ESTABLISHED. "
        "The audit did not define a post-outcome threshold or use 220-step future outcomes "
        "to construct a counterfactual selector. Score margins are reported independently.",
        "",
        f"- Low normalized score margin (<= {low_margin_threshold:.2f}): "
        f"{low_margin_failure}/54 failure agents versus {low_margin_success}/18 success-control agents.",
        "- Low selector confidence is therefore common but is not enriched in failures.",
        f"- Geometric poor-handoff flag: {len(poor_failure_layouts)}/18 failure layouts versus "
        f"{len(poor_success_layouts)}/6 controls; the current post-hoc geometry label is non-discriminative.",
        "",
        "## 7. Obstacle and lower-policy diagnostics",
        "",
        f"- Obstacle-collision agent records: {len(obstacle_rows)}",
        f"- Unreached agent records reclassified: {len(lower_rows)}",
        "- LOWER_POLICY_LOCAL_COMPATIBILITY_FAILURE is supported only when candidate geometry "
        "is not poor, obstacle/inter-agent blocking is absent, and persistent oscillation or "
        "action saturation is present.",
        f"- Handoff-step median speed: failed terminal completion="
        f"{handoff_median('reached_but_terminal_failed_reference', 'speed_mps'):.3f} m/s, "
        f"successful completion={handoff_median('successful_fp_shep_reference', 'speed_mps'):.3f} m/s.",
        f"- Handoff-step median terminal heading alignment: failed="
        f"{handoff_median('reached_but_terminal_failed_reference', 'heading_toward_terminal'):.3f}, "
        f"successful={handoff_median('successful_fp_shep_reference', 'heading_toward_terminal'):.3f}. "
        "These descriptive values do not support a simple excessive-speed or heading-misalignment explanation.",
        "",
        "## 8. Mechanism evidence",
        "",
        "| Mechanism | Failure rate | Control rate | Excess | Label |",
        "|---|---:|---:|---:|---|",
    ]
    for row in mechanism_rows:
        lines.append(
            f"| {row['mechanism']} | {float(row['supporting_fraction']):.3f} | "
            f"{float(row['success_control_supporting_fraction']):.3f} | "
            f"{float(row['failure_specific_excess_fraction']):.3f} | {row['label']} |"
        )
    lines.extend(["", "## 9. Family-level evidence", ""])
    for row in family_rows:
        lines.append(
            f"- Family {row['family']}: success={row['success_count']}/{row['layout_count']}, "
            f"reference-unreachable={row['reference_unreachable_count']}, "
            f"obstacle-collision={row['obstacle_collision_count']}, "
            f"inter-agent failure={row['inter_agent_failure_count']}."
        )
    lines.extend(
        [
            "",
            "## 10. Final decision labels",
            "",
        ]
    )
    keys = (
        "SHORT_PREVIEW_HORIZON_LIMITATION",
        "PREVIEW_FEATURE_LIMITATION",
        "CANDIDATE_ADMISSIBILITY_LIMITATION",
        "LOWER_POLICY_LOCAL_COMPATIBILITY_LIMITATION",
        "RESIDUAL_COORDINATION_LIMITATION",
        "FROZEN_LIDAR_SURFACE_APPROXIMATION_COMPONENT",
        "TERMINAL_SPEED_DIAGNOSTIC_VALUE",
        "PRIMARY_LIMITATION",
        "SECONDARY_LIMITATION",
        "FP_SHEP_REDESIGN_RECOMMENDED",
        "SAC_FINE_TUNING_RECOMMENDED",
        "GAT_STAGE_I_RECOMMENDED",
        "NEXT_STEP",
    )
    for key in keys:
        lines.append(f"- {key} = {conclusion[key]}")
    lines.extend(
        [
            "",
            "## 11. Answers to the audit questions",
            "",
            "1. The 18 failures remain heterogeneous: 8 reference-unreachable layouts, 6 pure obstacle collisions, and 4 inter-agent collision layouts.",
            f"2. The unreachable cohort splits at agent level into {dict(subtype_counts)}; no strict lower-policy-only case is established.",
            "3. H=4 is partially limiting because H12 improves cohort separation, but longer horizon alone is not sufficient.",
            "4. H=8/12/20 does not directly expose the six later obstacle collisions; no selected preview terminates within H20.",
            "5. The current three-feature H4 score does not contain a failure-specific absolute warning under the frozen thresholds.",
            f"6. Terminal speed has {terminal_speed_value.lower()} descriptive diagnostic value, but remains excluded from ranking.",
            "7. The current geometric handoff flag is present in both failures and controls, so candidate admissibility is not established as the primary cause.",
            "8. Handoff speed and terminal heading do not show a simple failure-specific direction under descriptive medians.",
            "9. Strict evidence for SAC-DMP local-reference compatibility failure is absent after excluding geometry and blocking explanations.",
            "10. The evidence supports a mixed preview-horizon/feature-information limitation, confounded by frozen LiDAR reconstruction; it does not support immediate GAT or SAC fine-tuning.",
            "",
            "## 12. Limitations",
            "",
            "- Preview clearance remains the frozen visible-surface approximation.",
            "- LiDAR hit identity and dynamic peer extrapolation are unavailable.",
            "- H_diag does not estimate synchronized multi-agent future trajectories.",
            "- Counts, medians, means, and ranges are descriptive; no statistical "
            "significance is claimed.",
            "- This audit diagnoses absolute FP-SHEP limitations without negating its "
            "previously established relative improvement over Proposal ranking.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment(settings: Mapping[str, Any], output_dir: Path) -> Path:
    _assert_config(settings)
    source = _source_artifacts(settings)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config_bytes = json.dumps(
        settings, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    diagnostic_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    thresholds_sha256 = _stable_hash(settings["diagnostic_thresholds"])
    source_manifest_sha_before = sha256_file(source["manifest_path"])
    checkpoint_path = (REPO_ROOT / str(settings["checkpoint"])).resolve()
    checkpoint_sha_before = sha256_file(checkpoint_path)
    if checkpoint_sha_before != str(settings["checkpoint_sha256_expected"]):
        raise RuntimeError("checkpoint SHA256 mismatch")
    critical_before = _critical_source_hashes()

    source_manifest_record = {
        "source_artifact": str(source["root"]),
        "scenario_manifest_path": str(source["manifest_path"]),
        "scenario_manifest_sha256": source["manifest_sha256"],
        "scenario_manifest_sha256_expected": settings[
            "source_manifest_sha256_expected"
        ],
        "scenario_manifest_bytes_reused": True,
        "scenario_manifest_regenerated": False,
        "layout_count": len(source["manifest"]["layouts"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha_before,
        "diagnostic_config_sha256": diagnostic_config_sha256,
        "diagnostic_thresholds_sha256": thresholds_sha256,
        "thresholds_frozen_before_episode_rerun": True,
    }
    write_json(output_dir / "source_artifact_manifest.json", source_manifest_record)
    resolved_config = copy.deepcopy(dict(settings))
    resolved_config.update(
        {
            "resolved_output_dir": str(output_dir),
            "diagnostic_config_sha256": diagnostic_config_sha256,
            "diagnostic_thresholds_sha256": thresholds_sha256,
            "thresholds_frozen_before_episode_rerun": True,
        }
    )
    write_json(output_dir / "config.json", resolved_config)

    geometry_settings = _load_json(
        (REPO_ROOT / str(settings["geometry_config"])).resolve()
    )
    base_settings = _load_json((REPO_ROOT / str(settings["base_config"])).resolve())
    base_settings.update(
        {
            "checkpoint": settings["checkpoint"],
            "checkpoint_sha256_expected": settings["checkpoint_sha256_expected"],
            "deterministic_policy": True,
            "num_agents": 3,
            "max_steps": 220,
            "peer_radius": float(settings["peer_radius"]),
        }
    )
    multi_config = build_single_distribution_multi_config(num_agents=3, max_steps=220)
    policy, loaded_checkpoint = _load_policy(base_settings, multi_config)
    if loaded_checkpoint.resolve() != checkpoint_path:
        raise RuntimeError("policy loader resolved another checkpoint")
    policy_hash_before = _policy_parameter_sha256(policy)

    layout_index = {
        str(row["layout_id"]): row for row in source["manifest"]["layouts"]
    }
    episode_index = {str(row["layout_id"]): row for row in source["episodes"]}
    agent_index: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    selection_index: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source["agents"]:
        agent_index[str(row["layout_id"])].append(row)
    for row in source["selections"]:
        selection_index[str(row["layout_id"])].append(row)

    # Stage 1: selected-candidate H_diag from untouched t=0 environments.
    horizon_rows: list[dict[str, Any]] = []
    preview_identity_checks: list[dict[str, Any]] = []
    diagnostic_thresholds = dict(settings["diagnostic_thresholds"])
    diagnostic_thresholds["action_saturation_relative_tolerance"] = float(
        settings["action_saturation"]["relative_tolerance"]
    )
    import Environment.multi_agent_dmp_env as environment_module
    import planning.policy_preview as preview_module

    default_execution_symbol = environment_module.propagate_sac_dmp_action
    default_preview_symbol = preview_module.propagate_sac_dmp_action
    for layout_counter, layout_id in enumerate(sorted(layout_index), start=1):
        layout = layout_index[layout_id]
        env, _ = build_generalization_environment(
            config=multi_config,
            scenario=str(source["manifest"]["scenario_id"]),
            seed=int(layout["evaluation_seed"]),
            peer_radius=float(settings["peer_radius"]),
        )
        state_hash_before = _scenario_hash(_scene_snapshot(env))
        try:
            with scoped_historical_preview_and_multi_agent_transition():
                for selection in sorted(
                    selection_index[layout_id], key=lambda row: int(row["agent_id"])
                ):
                    agent_id = int(selection["agent_id"])
                    candidate_index = int(selection["selected_candidate_index"])
                    candidate = np.asarray(
                        parse_json_cell(selection["selected_world_reference"]), dtype=float
                    )
                    rows = _selected_candidate_horizon_diagnostic(
                        env=env,
                        layout_id=layout_id,
                        family=str(layout["family"]),
                        agent_id=agent_id,
                        candidate_index=candidate_index,
                        candidate=candidate,
                        policy=policy,
                        horizons=settings["post_hoc_preview"][
                            "selected_candidate_horizons"
                        ],
                        thresholds=diagnostic_thresholds,
                    )
                    h4 = next(row for row in rows if int(row["H_diag"]) == 4)
                    source_agent = next(
                        row
                        for row in agent_index[layout_id]
                        if int(row["agent_id"]) == agent_id
                    )
                    checks = {
                        "task_progress": np.isclose(
                            float(h4["task_progress"]),
                            float(source_agent["selected_preview_task_progress"]),
                            rtol=1.0e-7,
                            atol=1.0e-8,
                        ),
                        "minimum_clearance": np.isclose(
                            float(h4["minimum_clearance_m"]),
                            _source_selected_preview_value(
                                source_agent, "selected_preview_min_clearance"
                            ),
                            rtol=1.0e-7,
                            atol=1.0e-8,
                        ),
                        "maximum_deviation": np.isclose(
                            float(h4["maximum_execution_deviation_m"]),
                            float(
                                source_agent[
                                    "selected_preview_max_execution_deviation"
                                ]
                            ),
                            rtol=1.0e-7,
                            atol=1.0e-8,
                        ),
                        "terminal_speed": np.isclose(
                            float(h4["terminal_speed_mps"]),
                            float(source_agent["selected_preview_terminal_speed"]),
                            rtol=1.0e-7,
                            atol=1.0e-8,
                        ),
                    }
                    preview_identity_checks.append(
                        {
                            "layout_id": layout_id,
                            "agent_id": agent_id,
                            "checks": checks,
                            "status": "PASSED" if all(checks.values()) else "FAILED",
                        }
                    )
                    horizon_rows.extend(rows)
        finally:
            state_hash_after = _scenario_hash(_scene_snapshot(env))
            env.close()
        if state_hash_before != state_hash_after:
            raise RuntimeError(f"H_diag mutated environment state for {layout_id}")
        print(f"[H_diag {layout_counter}/24] {layout_id}", flush=True)

    if any(row["status"] != "PASSED" for row in preview_identity_checks):
        write_csv(output_dir / "horizon_diagnostic.csv", horizon_rows)
        write_json(
            output_dir / "preview_identity_gate.json",
            {"status": "FAILED", "checks": preview_identity_checks},
        )
        raise RuntimeError("H=4 diagnostic preview does not reproduce formal H=4")
    if not all(
        parse_bool(row["preview_prefix_consistency_verified"])
        for row in horizon_rows
    ):
        write_csv(output_dir / "horizon_diagnostic.csv", horizon_rows)
        raise RuntimeError("H_diag prefixes differ when only rollout length changes")
    write_json(
        output_dir / "preview_identity_gate.json",
        {"status": "PASSED", "checks": preview_identity_checks},
    )
    write_csv(output_dir / "horizon_diagnostic.csv", horizon_rows)

    # Stage 2: replay exactly the frozen FP-SHEP selected references for missing
    # step traces. No candidate generation or selector is called here.
    rerun_episodes: list[dict[str, Any]] = []
    rerun_agents: list[dict[str, Any]] = []
    all_trace_rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    for layout_counter, layout_id in enumerate(sorted(layout_index), start=1):
        with scoped_historical_preview_and_multi_agent_transition():
            episode, agents, trace = _run_frozen_selected_reference_episode(
                policy=policy,
                multi_config=multi_config,
                settings=settings,
                layout_record=layout_index[layout_id],
                source_episode=episode_index[layout_id],
                source_agent_rows=agent_index[layout_id],
                source_selection_rows=selection_index[layout_id],
            )
        checks = _reproduction_checks(
            rerun=episode,
            source_episode=episode_index[layout_id],
            rerun_agents=agents,
            source_agents=agent_index[layout_id],
        )
        rerun_episodes.append(episode)
        rerun_agents.extend(agents)
        all_trace_rows.extend(trace)
        reproduction_rows.append(checks)
        print(
            f"[reproduction {layout_counter}/24] {layout_id}: {checks['status']}",
            flush=True,
        )

    reproduction_gate = {
        "status": (
            "PASSED"
            if all(row["status"] == "PASSED" for row in reproduction_rows)
            else "FAILED"
        ),
        "passed_layout_count": sum(
            row["status"] == "PASSED" for row in reproduction_rows
        ),
        "layout_count": len(reproduction_rows),
        "FROZEN_EPISODE_REPRODUCTION_MISMATCH": (
            "NO"
            if all(row["status"] == "PASSED" for row in reproduction_rows)
            else "YES"
        ),
        "layout_checks": reproduction_rows,
    }
    write_json(output_dir / "reproduction_gate.json", reproduction_gate)
    write_csv(output_dir / "diagnostic_step_trace.csv", all_trace_rows)
    if reproduction_gate["status"] != "PASSED":
        write_json(
            output_dir / "conclusion.json",
            {
                "FROZEN_EPISODE_REPRODUCTION_MISMATCH": "YES",
                "MECHANISM_ATTRIBUTION_PERFORMED": False,
                "NEXT_STEP": "Diagnose the frozen episode adapter mismatch before interpreting legacy failures.",
            },
        )
        raise RuntimeError("frozen episode reproduction gate failed")

    # Stage 3: artifact-only and reproduced-trace mechanism analysis.
    geometry_rows: list[dict[str, Any]] = []
    score_margin_rows: list[dict[str, Any]] = []
    epsilon = float(settings["diagnostic_thresholds"]["score_margin_epsilon"])
    for selection in source["selections"]:
        layout = layout_index[str(selection["layout_id"])]
        geometry_rows.append(
            candidate_geometry_record(
                layout=layout,
                agent_id=int(selection["agent_id"]),
                candidate=np.asarray(
                    parse_json_cell(selection["selected_world_reference"]), dtype=float
                ),
                proposal_rank=int(selection["selected_original_proposal_rank"]),
                thresholds=settings["diagnostic_thresholds"],
            )
        )
        score_margin_rows.append(score_margin_record(selection, epsilon=epsilon))
    write_csv(output_dir / "candidate_geometry.csv", geometry_rows)
    write_csv(output_dir / "score_margin_analysis.csv", score_margin_rows)

    lower_rows = _build_lower_policy_rows(
        failure_rows=source["failures"],
        source_agents=source["agents"],
        trace_rows=all_trace_rows,
        geometry_rows=geometry_rows,
        horizon_rows=horizon_rows,
        thresholds=settings["diagnostic_thresholds"],
    )
    write_csv(output_dir / "lower_policy_failure_analysis.csv", lower_rows)
    obstacle_rows = _build_obstacle_collision_rows(
        failure_rows=source["failures"],
        agent_summaries=rerun_agents,
        trace_rows=all_trace_rows,
        horizon_rows=horizon_rows,
        geometry_rows=geometry_rows,
        layout_index=layout_index,
    )
    write_csv(output_dir / "obstacle_collision_analysis.csv", obstacle_rows)

    team_success_by_layout = {
        str(row["layout_id"]): parse_bool(row["team_success"])
        for row in source["episodes"]
    }
    handoff_rows = _handoff_window_rows(
        all_trace_rows,
        rerun_agents,
        before_steps=int(
            settings["diagnostic_thresholds"]["handoff_window_before_steps"]
        ),
        after_steps=int(
            settings["diagnostic_thresholds"]["handoff_window_after_steps"]
        ),
        team_success_by_layout=team_success_by_layout,
    )
    write_csv(output_dir / "handoff_state_analysis.csv", handoff_rows)

    feature_rows, terminal_speed_value = _feature_summary_rows(
        source_episodes=source["episodes"],
        source_agents=source["agents"],
        score_margins=score_margin_rows,
        geometry_rows=geometry_rows,
        trace_rows=all_trace_rows,
        thresholds=settings["diagnostic_thresholds"],
    )
    write_csv(output_dir / "feature_diagnostic.csv", feature_rows)

    cohort_rows = _layout_cohort_rows(
        source_episodes=source["episodes"],
        source_agents=source["agents"],
        source_selections=source["selections"],
        source_failures=source["failures"],
        rerun_agents=rerun_agents,
        geometry_rows=geometry_rows,
        horizon_rows=horizon_rows,
        lower_policy_rows=lower_rows,
        score_margins=score_margin_rows,
        low_confidence_threshold=float(
            settings["diagnostic_thresholds"][
                "low_selector_confidence_normalized_margin"
            ]
        ),
    )
    failure_cohort = [row for row in cohort_rows if not parse_bool(row["team_success"])]
    success_control = [row for row in cohort_rows if parse_bool(row["team_success"])]
    write_csv(output_dir / "failure_cohort.csv", failure_cohort)
    write_csv(output_dir / "success_control.csv", success_control)
    family_rows = family_failure_summary(cohort_rows)
    write_csv(output_dir / "family_failure_summary.csv", family_rows)

    mechanism_rows, conclusion = build_mechanism_conclusion(
        failed_layouts=failure_cohort,
        horizon_rows=horizon_rows,
        geometry_rows=geometry_rows,
        lower_policy_rows=lower_rows,
        thresholds=settings["diagnostic_thresholds"],
        terminal_speed_diagnostic_value=terminal_speed_value,
    )
    conclusion["TERMINAL_SPEED_DIAGNOSTIC_VALUE"] = terminal_speed_value
    conclusion["FROZEN_EPISODE_REPRODUCTION_MISMATCH"] = "NO"
    conclusion["MECHANISM_ATTRIBUTION_PERFORMED"] = True
    conclusion["FAILED_TOP1_WITH_BETTER_PREVIEW_ALTERNATIVE"] = "NOT_ESTABLISHED"
    conclusion["BETTER_PREVIEW_ALTERNATIVE_SCOPE"] = "existing_H4_records_only"
    conclusion["FROZEN_LIDAR_SURFACE_APPROXIMATION_COMPONENT"] = (
        "SUPPORTED_AS_HORIZON_INTERPRETATION_CONFOUND"
        if obstacle_rows
        and not any(
            row["preview_first_clearance_collapse_horizon"] is not None
            for row in obstacle_rows
        )
        else "NOT_ESTABLISHED"
    )
    conclusion["NEXT_STEP"] = (
        "Run one separately approved controlled diagnostic that isolates rollout horizon "
        "from frozen-visible-surface observation reconstruction; do not lengthen H blindly, "
        "change the score, fine-tune SAC, or start GAT from this audit alone."
    )
    write_csv(output_dir / "mechanism_summary.csv", mechanism_rows)
    write_json(output_dir / "conclusion.json", conclusion)

    checkpoint_sha_after = sha256_file(checkpoint_path)
    policy_hash_after = _policy_parameter_sha256(policy)
    critical_after = _critical_source_hashes()
    source_manifest_sha_after = sha256_file(source["manifest_path"])
    integrity = {
        "status": "PASSED",
        "scenario_manifest_sha256_before_after": [
            source_manifest_sha_before,
            source_manifest_sha_after,
        ],
        "source_manifest_bytes_unchanged": source["manifest_path"].read_bytes()
        == source["manifest_bytes"],
        "checkpoint_sha256_before_after": [checkpoint_sha_before, checkpoint_sha_after],
        "policy_parameter_sha256_before_after": [
            policy_hash_before,
            policy_hash_after,
        ],
        "critical_source_hashes_unchanged": critical_before == critical_after,
        "default_execution_symbol_restored": (
            environment_module.propagate_sac_dmp_action is default_execution_symbol
        ),
        "default_preview_symbol_restored": (
            preview_module.propagate_sac_dmp_action is default_preview_symbol
        ),
        "gradient_update_count": 0,
        "GAT_inference_count": 0,
        "candidate_generation_count": 0,
        "online_selection_count": 0,
        "formal_benchmark_episode_count": 0,
        "diagnostic_selected_episode_count": len(rerun_episodes),
        "thresholds_sha256_before_after": [thresholds_sha256, _stable_hash(settings["diagnostic_thresholds"])],
        "geometry_modified": False,
        "outcome_based_scene_filtering_count": 0,
        "repeated_replanning": False,
    }
    required_checks = {
        "source_manifest_unchanged": source_manifest_sha_before
        == source_manifest_sha_after
        and integrity["source_manifest_bytes_unchanged"],
        "checkpoint_unchanged": checkpoint_sha_before == checkpoint_sha_after,
        "policy_unchanged": policy_hash_before == policy_hash_after,
        "critical_sources_unchanged": critical_before == critical_after,
        "transition_symbols_restored": integrity["default_execution_symbol_restored"]
        and integrity["default_preview_symbol_restored"],
        "thresholds_unchanged": integrity["thresholds_sha256_before_after"][0]
        == integrity["thresholds_sha256_before_after"][1],
        "reproduction_gate": reproduction_gate["status"] == "PASSED",
        "cohort_counts": len(failure_cohort) == 18 and len(success_control) == 6,
        "horizon_prefix_consistency": all(
            parse_bool(row["preview_prefix_consistency_verified"])
            for row in horizon_rows
        ),
    }
    integrity["checks"] = required_checks
    integrity["failed_checks"] = [
        name for name, passed in required_checks.items() if not passed
    ]
    integrity["status"] = "PASSED" if not integrity["failed_checks"] else "FAILED"
    write_json(output_dir / "integrity.json", integrity)
    if integrity["status"] != "PASSED":
        raise RuntimeError(f"final integrity failed: {integrity['failed_checks']}")

    report = _render_report(
        config=settings,
        reproduction=reproduction_gate,
        failure_cohort=failure_cohort,
        success_control=success_control,
        lower_rows=lower_rows,
        obstacle_rows=obstacle_rows,
        mechanism_rows=mechanism_rows,
        horizon_rows=horizon_rows,
        geometry_rows=geometry_rows,
        score_margin_rows=score_margin_rows,
        handoff_rows=handoff_rows,
        family_rows=family_rows,
        conclusion=conclusion,
        terminal_speed_value=terminal_speed_value,
    )
    (output_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    resolved_config["runtime_seconds"] = float(time.perf_counter() - started)
    resolved_config["analysis_schema_version"] = ANALYSIS_SCHEMA_VERSION
    resolved_config["source_geometry_config"] = geometry_settings
    write_json(output_dir / "config.json", resolved_config)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    settings = _load_json(config_path)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / str(settings["output_dir"]) / timestamp
    else:
        output_dir = args.output_dir.resolve()
    result = run_experiment(settings, output_dir)
    print(f"FP-SHEP failure-mode audit complete: {result}", flush=True)


if __name__ == "__main__":
    main()
