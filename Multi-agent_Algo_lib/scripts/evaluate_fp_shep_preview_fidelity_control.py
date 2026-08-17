"""Run the post-hoc FP-SHEP 2x2 preview-fidelity control diagnosis.

The runner consumes frozen selected candidates and never invokes Proposal or
formal online selection.  Stage A must reproduce all prior H=4 preview feature
records before B/C/D are permitted to run.
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
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.fp_shep_failure_mode_analysis import parse_bool, parse_json_cell  # noqa: E402
from planning.fp_shep_preview_fidelity_analysis import (  # noqa: E402
    H4_TERMINATION_RULE,
    H20_TERMINATION_RULE,
    REFRESHED_SENSING_SCOPE,
    SCHEMA_VERSION,
    SENSING_FROZEN,
    SENSING_REFRESHED_STATIC,
    build_refreshed_static_sensing_context,
    classify_directional_fraction,
    descriptive,
    diagnostic_preview_rollout,
    post_hoc_three_feature_score,
    rank_biserial,
)
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.policy_preview import build_preview_inputs_from_env  # noqa: E402
from planning.reference_transition_finetuning import sha256_file  # noqa: E402
from scripts.evaluate_actor_dmp_goal_semantics import write_csv, write_json  # noqa: E402
from scripts.evaluate_fp_shep_failure_mode_audit import (  # noqa: E402
    _critical_source_hashes,
    _policy_parameter_sha256,
    _scenario_hash,
    _scene_snapshot,
    _source_selected_preview_value,
    _stable_hash,
)
from scripts.evaluate_geometry_generalization_audit import (  # noqa: E402
    build_generalization_environment,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs/evaluation/fp_shep_preview_fidelity_control.json"
)
METHOD_FP_SHEP = "fp_shep_top1_one_shot"
CELL_ORDER = ("A", "B", "C", "D")
FEATURE_FIELDS = {
    "task_progress": "task_progress",
    "minimum_clearance_m": "min_clearance",
    "maximum_execution_deviation_m": "max_execution_deviation",
    "terminal_speed_mps": "terminal_speed",
    "distance_to_reference_end_m": "distance_to_reference_end_m",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _numpy_global_rng_hash() -> str:
    state = np.random.get_state()
    payload = {
        "algorithm": state[0],
        "keys": state[1].tolist(),
        "position": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }
    return _stable_hash(payload)


def _torch_rng_hash() -> str:
    return _hash_bytes(torch.random.get_rng_state().cpu().numpy().tobytes())


def _environment_rng_hash(env: Any) -> str:
    return _stable_hash(copy.deepcopy(env.np_random.bit_generator.state))


def _full_environment_state_hash(env: Any) -> str:
    packet_rows = []
    for packet in env.latest_sensor_packets:
        packet_rows.append(
            None
            if packet is None
            else {
                "observation": np.asarray(packet.observation),
                "current_scan": np.asarray(packet.current_scan),
                "previous_scan": np.asarray(packet.previous_scan),
                "min_clearance": float(packet.min_clearance),
                "collision": bool(packet.collision),
            }
        )
    return _stable_hash(
        {
            "scene": _scene_snapshot(env),
            "steps": int(env.steps),
            "action_guidance_step": int(env.action_guidance_step),
            "dmp_goals": [np.asarray(item.goal) for item in env.dmps],
            "dmp_phases": [float(item.phase) for item in env.dmps],
            "sensor_previous": [
                None
                if item._previous_scan is None
                else np.asarray(item._previous_scan)
                for item in env.sensors
            ],
            "latest_packets": packet_rows,
            "latest_controller_infos": env.latest_controller_infos,
            "latest_observation": env.latest_observation,
            "latest_collision_info": env.latest_collision_info,
            "previous_velocities": env.previous_velocities,
            "success_rewarded_mask": env.success_rewarded_mask,
            "stagnation_window_progress": env.stagnation_window_progress,
            "stagnation_counters": env.stagnation_counters,
            "env_rng": copy.deepcopy(env.np_random.bit_generator.state),
        }
    )


def _assert_config(settings: Mapping[str, Any]) -> None:
    if int(settings["layout_count"]) != 24 or int(settings["selected_candidate_count"]) != 72:
        raise ValueError("diagnosis requires exactly 24 layouts and 72 selected candidates")
    cells = settings["cells"]
    expected = {
        "A": (4, SENSING_FROZEN, False),
        "B": (4, SENSING_REFRESHED_STATIC, False),
        "C": (20, SENSING_FROZEN, True),
        "D": (20, SENSING_REFRESHED_STATIC, True),
    }
    for cell, (horizon, sensing, early_stop) in expected.items():
        actual = cells[cell]
        if (
            int(actual["horizon"]) != horizon
            or str(actual["sensing_mode"]) != sensing
            or bool(actual["diagnostic_early_stop"]) != early_stop
        ):
            raise ValueError(f"2x2 cell {cell} contract changed")
    if settings["factor_contract"]["H4_termination_rule"] != H4_TERMINATION_RULE:
        raise ValueError("H4 anchor termination contract changed")
    if settings["factor_contract"]["H20_termination_rule"] != H20_TERMINATION_RULE:
        raise ValueError("H20 diagnostic termination contract changed")
    if settings["refreshed_sensing"]["scope"] != REFRESHED_SENSING_SCOPE:
        raise ValueError("refreshed sensing scope must remain STATIC_GEOMETRY_ONLY")
    formal = settings["formal_selector_unchanged"]
    if int(formal["H_preview"]) != 4 or float(formal["weights"]["terminal_speed"]) != 0.0:
        raise ValueError("formal FP-SHEP contract changed")
    if str(formal["formula"]) != "+progress +clearance -deviation":
        raise ValueError("formal three-feature score changed")
    if any(bool(value) for value in settings["strict_exclusions"].values()):
        raise ValueError("all strict exclusion flags must remain false")
    if settings["execution_semantics"]["historical_gate"] != HISTORICAL_GATE_NAME:
        raise ValueError("historical transition contract changed")


def _load_sources(settings: Mapping[str, Any]) -> dict[str, Any]:
    failure_root = (REPO_ROOT / str(settings["source_failure_audit"])).resolve()
    geometry_root = (REPO_ROOT / str(settings["source_geometry_artifact"])).resolve()
    failure_required = (
        "config.json",
        "source_artifact_manifest.json",
        "reproduction_gate.json",
        "horizon_diagnostic.csv",
        "failure_cohort.csv",
        "success_control.csv",
        "lower_policy_failure_analysis.csv",
        "obstacle_collision_analysis.csv",
        "diagnostic_step_trace.csv",
        "conclusion.json",
    )
    geometry_required = (
        "scenario_manifest.json",
        "candidate_selection.csv",
        "per_agent.csv",
        "per_episode.csv",
    )
    missing = [
        str(root / name)
        for root, names in (
            (failure_root, failure_required),
            (geometry_root, geometry_required),
        )
        for name in names
        if not (root / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"frozen source artifacts are incomplete: {missing}")
    manifest_path = geometry_root / "scenario_manifest.json"
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != str(settings["source_manifest_sha256_expected"]):
        raise RuntimeError("frozen scenario manifest SHA256 mismatch")
    reproduction = _load_json(failure_root / "reproduction_gate.json")
    if reproduction.get("status") != "PASSED":
        raise RuntimeError("source failure-mode reproduction gate is not passed")
    selections = [
        row
        for row in _load_csv(geometry_root / "candidate_selection.csv")
        if row["method"] == METHOD_FP_SHEP
    ]
    agents = [
        row
        for row in _load_csv(geometry_root / "per_agent.csv")
        if row["method"] == METHOD_FP_SHEP
    ]
    episodes = [
        row
        for row in _load_csv(geometry_root / "per_episode.csv")
        if row["method"] == METHOD_FP_SHEP
    ]
    prior_horizon = _load_csv(failure_root / "horizon_diagnostic.csv")
    prior_h4 = [row for row in prior_horizon if int(row["H_diag"]) == 4]
    if not (len(selections) == len(agents) == len(prior_h4) == 72 and len(episodes) == 24):
        raise RuntimeError("frozen source row counts changed")
    return {
        "failure_root": failure_root,
        "geometry_root": geometry_root,
        "manifest_path": manifest_path,
        "manifest_bytes": manifest_path.read_bytes(),
        "manifest_sha256": manifest_sha,
        "manifest": _load_json(manifest_path),
        "failure_config": _load_json(failure_root / "config.json"),
        "failure_conclusion": _load_json(failure_root / "conclusion.json"),
        "selections": selections,
        "agents": agents,
        "episodes": episodes,
        "prior_h4": prior_h4,
        "failures": _load_csv(failure_root / "failure_cohort.csv"),
        "successes": _load_csv(failure_root / "success_control.csv"),
        "reference_agents": _load_csv(
            failure_root / "lower_policy_failure_analysis.csv"
        ),
        "obstacle_agents": _load_csv(
            failure_root / "obstacle_collision_analysis.csv"
        ),
        "step_trace": _load_csv(failure_root / "diagnostic_step_trace.csv"),
    }


def _candidate_record(selection: Mapping[str, Any]) -> dict[str, Any]:
    index = int(selection["selected_candidate_index"])
    points = list(parse_json_cell(selection["candidate_world_points"], []))
    scores = list(parse_json_cell(selection["proposal_scores"], []))
    metadata = list(parse_json_cell(selection["candidate_metadata"], []))
    selected = np.asarray(parse_json_cell(selection["selected_world_reference"]), dtype=float)
    if index < 0 or index >= len(points) or index >= len(scores) or index >= len(metadata):
        raise RuntimeError("selected candidate index is outside frozen candidate bundle")
    if not np.allclose(selected, np.asarray(points[index], dtype=float), rtol=0.0, atol=1e-12):
        raise RuntimeError("selected reference differs from frozen candidate bundle")
    payload = {
        "candidate_set_hash": selection["candidate_set_hash"],
        "candidate_index": index,
        "candidate_world_point": selected.round(12).tolist(),
        "proposal_score": float(scores[index]),
        "metadata": metadata[index],
    }
    return {
        **payload,
        "candidate_record_hash": _stable_hash(payload),
        "candidate_world_point": selected,
        "K_t": int(selection["K_t"]),
    }


def _cohort_indices(source: Mapping[str, Any]) -> dict[str, Any]:
    failure_category = {
        str(row["layout_id"]): str(row["primary_failure_category"])
        for row in source["failures"]
    }
    success_layouts = {str(row["layout_id"]) for row in source["successes"]}
    reference = {
        (str(row["layout_id"]), int(row["agent_id"])): str(row["primary_subcategory"])
        for row in source["reference_agents"]
    }
    obstacle = {
        (str(row["layout_id"]), int(row["agent_id"])): row
        for row in source["obstacle_agents"]
    }
    if len(success_layouts) != 6 or len(reference) != 11 or len(obstacle) != 6:
        raise RuntimeError("frozen primary cohort counts changed")
    return {
        "failure_category": failure_category,
        "success_layouts": success_layouts,
        "reference": reference,
        "obstacle": obstacle,
    }


def _cell_row(
    *,
    cell: str,
    layout: Mapping[str, Any],
    selection: Mapping[str, Any],
    candidate: Mapping[str, Any],
    preview: Any,
    initial_state: Any,
    source_agent: Mapping[str, Any],
    source_episode: Mapping[str, Any],
    cohort: Mapping[str, Any],
    normalization: Mapping[str, float],
    gate_values: set[str],
    state_hash_before: str,
    state_hash_after: str,
    env_rng_before: str,
    env_rng_after: str,
    numpy_rng_before: str,
    numpy_rng_after: str,
    torch_rng_before: str,
    torch_rng_after: str,
    refreshed_identity: bool | None,
    peer_visible_point_count: int | None,
) -> dict[str, Any]:
    trajectory = preview.trajectory
    clearances = np.asarray(trajectory.clearances, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(clearances))
    first_min_step = None
    if finite_indices.size:
        finite_values = clearances[finite_indices]
        first_min_step = int(finite_indices[int(np.argmin(finite_values))] + 1)
    candidate_point = np.asarray(candidate["candidate_world_point"], dtype=float)
    initial_ref_distance = float(np.linalg.norm(candidate_point - initial_state.position))
    final_ref_distance = float(np.linalg.norm(candidate_point - trajectory.positions[-1]))
    layout_id = str(layout["layout_id"])
    agent_id = int(selection["agent_id"])
    reference_key = (layout_id, agent_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "layout_id": layout_id,
        "family": str(layout["family"]),
        "evaluation_seed": int(layout["evaluation_seed"]),
        "agent_id": agent_id,
        "candidate_index": int(candidate["candidate_index"]),
        "candidate_world_point": candidate_point.tolist(),
        "candidate_record_hash": candidate["candidate_record_hash"],
        "candidate_set_hash": candidate["candidate_set_hash"],
        "candidate_metadata": candidate["metadata"],
        "proposal_score": float(candidate["proposal_score"]),
        "K_t": int(candidate["K_t"]),
        "horizon_requested": int(preview.metadata["requested_horizon_steps"]),
        "horizon_effective": int(preview.metadata["effective_horizon_steps"]),
        "sensing_mode": preview.metadata["sensing_mode"],
        "REFRESHED_SENSING_SCOPE": preview.metadata["REFRESHED_SENSING_SCOPE"],
        "diagnostic_termination_enabled": bool(
            preview.metadata["diagnostic_termination_enabled"]
        ),
        "termination_reason": preview.metadata["termination_reason"],
        "preview_termination_step": preview.metadata["preview_termination_step"],
        "task_progress": float(preview.task_progress),
        "min_clearance": float(preview.min_clearance),
        "max_execution_deviation": float(preview.max_execution_deviation),
        "terminal_speed": float(preview.terminal_speed),
        "sensor_derived_clearance": float(preview.min_clearance),
        "sensor_derived_clearance_finite": bool(np.isfinite(preview.min_clearance)),
        "sensor_open_space_or_no_modeled_hit": bool(
            not np.isfinite(preview.min_clearance)
        ),
        "clearance_source": preview.metadata["clearance_source"],
        "ground_truth_collision_used_in_clearance": False,
        "preview_ground_truth_collision": bool(
            preview.metadata["preview_ground_truth_collision"]
        ),
        "preview_collision_step": preview.metadata["preview_collision_step"],
        "first_minimum_clearance_step": first_min_step,
        "distance_to_reference_initial_m": initial_ref_distance,
        "distance_to_reference_end_m": final_ref_distance,
        "candidate_closing_progress_m": initial_ref_distance - final_ref_distance,
        "candidate_closing_progress_per_step_m": (
            initial_ref_distance - final_ref_distance
        )
        / float(trajectory.horizon),
        "POST_HOC_SCORE_ONLY": float(
            post_hoc_three_feature_score(preview, normalization)
        ),
        "used_for_candidate_selection": False,
        "trajectory_positions": trajectory.positions.tolist(),
        "trajectory_velocities": trajectory.velocities.tolist(),
        "trajectory_actions": trajectory.actions.tolist(),
        "current_scans_sha256": _stable_hash(trajectory.current_scans),
        "previous_scans_sha256": _stable_hash(trajectory.previous_scans),
        "history_shift_verified": bool(
            trajectory.horizon <= 1
            or np.array_equal(
                trajectory.previous_scans[1:], trajectory.current_scans[:-1]
            )
        ),
        "runtime_total_ms": float(preview.performance.total_ms),
        "runtime_observation_ms": float(preview.performance.observation_ms),
        "runtime_policy_ms": float(preview.performance.policy_ms),
        "runtime_transition_ms": float(preview.performance.transition_ms),
        "runtime_sensing_ms": float(preview.performance.sensor_reconstruction_ms),
        "policy_calls": int(preview.performance.policy_calls),
        "historical_gate_values": sorted(gate_values),
        "historical_gate_verified": gate_values == {HISTORICAL_GATE_NAME},
        "team_success": parse_bool(source_episode["team_success"]),
        "team_failure_category": cohort["failure_category"].get(
            layout_id, "SUCCESS_CONTROL"
        ),
        "success_control": layout_id in cohort["success_layouts"],
        "reference_failure_agent": reference_key in cohort["reference"],
        "reference_failure_subcategory": cohort["reference"].get(reference_key),
        "actual_obstacle_collision_agent": reference_key in cohort["obstacle"],
        "actual_reference_reached": parse_bool(source_agent["reference_reached"]),
        "actual_distance_to_reference_at_termination_m": (
            float(source_agent["distance_to_reference_at_termination_m"])
            if source_agent["distance_to_reference_at_termination_m"] not in (None, "")
            else None
        ),
        "environment_state_hash_before": state_hash_before,
        "environment_state_hash_after": state_hash_after,
        "environment_state_unchanged": state_hash_before == state_hash_after,
        "environment_rng_hash_before": env_rng_before,
        "environment_rng_hash_after": env_rng_after,
        "environment_rng_unchanged": env_rng_before == env_rng_after,
        "numpy_rng_hash_before": numpy_rng_before,
        "numpy_rng_hash_after": numpy_rng_after,
        "numpy_rng_unchanged": numpy_rng_before == numpy_rng_after,
        "torch_rng_hash_before": torch_rng_before,
        "torch_rng_hash_after": torch_rng_after,
        "torch_rng_unchanged": torch_rng_before == torch_rng_after,
        "refreshed_t0_source_identity_verified": refreshed_identity,
        "frozen_peer_visible_surface_point_count": peer_visible_point_count,
        "peer_future_motion_prediction": False,
        "joint_multi_agent_rollout": False,
        "real_env_step_used": False,
    }


def _run_cell(
    *,
    env: Any,
    cell: str,
    layout: Mapping[str, Any],
    selection: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: Any,
    source_agent: Mapping[str, Any],
    source_episode: Mapping[str, Any],
    cohort: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    agent_id = int(selection["agent_id"])
    cell_spec = settings["cells"][cell]
    initial_state, local_context = build_preview_inputs_from_env(env, agent_id)
    refreshed = None
    if str(cell_spec["sensing_mode"]) == SENSING_REFRESHED_STATIC:
        refreshed = build_refreshed_static_sensing_context(
            env,
            agent_id,
            identity_atol=float(
                settings["refreshed_sensing"]["initial_typed_scan_identity_atol"]
            ),
            hit_epsilon=float(settings["refreshed_sensing"]["hit_epsilon"]),
        )
    state_before = _full_environment_state_hash(env)
    env_rng_before = _environment_rng_hash(env)
    numpy_before = _numpy_global_rng_hash()
    torch_before = _torch_rng_hash()
    gate_values: set[str] = set()

    def observer(_kwargs: dict[str, Any], transition: Any) -> None:
        gate_values.add(str(transition.controller_info["forcing_gate_semantics"]))

    with scoped_historical_preview_and_multi_agent_transition(
        preview_observer=observer
    ):
        preview = diagnostic_preview_rollout(
            initial_state=initial_state,
            local_context=local_context,
            candidate_goal=np.asarray(candidate["candidate_world_point"], dtype=float),
            policy=policy,
            horizon=int(cell_spec["horizon"]),
            dmp_config=env.dmps[agent_id].config,
            dynamics=env.dynamics[agent_id],
            static_obstacles=tuple(copy.deepcopy(env.static_obstacles)),
            collision_margin=float(env.env_config.collision_margin),
            terminal_tolerance=float(env.env_config.goal_tolerance),
            sensing_mode=str(cell_spec["sensing_mode"]),
            refreshed_context=refreshed,
            stop_on_diagnostic_termination=bool(cell_spec["diagnostic_early_stop"]),
        )
    state_after = _full_environment_state_hash(env)
    env_rng_after = _environment_rng_hash(env)
    numpy_after = _numpy_global_rng_hash()
    torch_after = _torch_rng_hash()
    return _cell_row(
        cell=cell,
        layout=layout,
        selection=selection,
        candidate=candidate,
        preview=preview,
        initial_state=initial_state,
        source_agent=source_agent,
        source_episode=source_episode,
        cohort=cohort,
        normalization=settings["formal_selector_unchanged"]["normalization"],
        gate_values=gate_values,
        state_hash_before=state_before,
        state_hash_after=state_after,
        env_rng_before=env_rng_before,
        env_rng_after=env_rng_after,
        numpy_rng_before=numpy_before,
        numpy_rng_after=numpy_after,
        torch_rng_before=torch_before,
        torch_rng_after=torch_after,
        refreshed_identity=(refreshed.source_identity_verified if refreshed else None),
        peer_visible_point_count=(
            int(refreshed.peer_visible_surface_points.shape[0]) if refreshed else None
        ),
    )


def _reproduction_checks(
    rows: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    prior_index = {
        (str(row["layout_id"]), int(row["agent_id"])): row
        for row in source["prior_h4"]
    }
    agent_index = {
        (str(row["layout_id"]), int(row["agent_id"])): row
        for row in source["agents"]
    }
    rtol = float(settings["reproduction_tolerance"]["rtol"])
    atol = float(settings["reproduction_tolerance"]["atol"])
    checks = []
    for row in rows:
        key = (str(row["layout_id"]), int(row["agent_id"]))
        prior = prior_index[key]
        source_agent = agent_index[key]
        metric_checks = {
            "task_progress": bool(
                np.isclose(float(row["task_progress"]), float(prior["task_progress"]), rtol=rtol, atol=atol)
            ),
            "minimum_clearance": bool(
                np.isclose(
                    float(row["min_clearance"]),
                    _source_selected_preview_value(source_agent, "selected_preview_min_clearance"),
                    rtol=rtol,
                    atol=atol,
                )
            ),
            "maximum_execution_deviation": bool(
                np.isclose(
                    float(row["max_execution_deviation"]),
                    float(prior["maximum_execution_deviation_m"]),
                    rtol=rtol,
                    atol=atol,
                )
            ),
            "terminal_speed": bool(
                np.isclose(float(row["terminal_speed"]), float(prior["terminal_speed_mps"]), rtol=rtol, atol=atol)
            ),
            "candidate_index": int(row["candidate_index"]) == int(prior["candidate_index"]),
            "candidate_world_point": bool(
                np.allclose(
                    np.asarray(row["candidate_world_point"], dtype=float),
                    np.asarray(parse_json_cell(prior["candidate_world_position"]), dtype=float),
                    rtol=0.0,
                    atol=1e-12,
                )
            ),
            "formal_h4_no_early_stop": (
                int(row["horizon_effective"]) == 4
                and not parse_bool(row["diagnostic_termination_enabled"])
            ),
            "historical_gate": parse_bool(row["historical_gate_verified"]),
            "environment_state": parse_bool(row["environment_state_unchanged"]),
            "environment_rng": parse_bool(row["environment_rng_unchanged"]),
            "numpy_rng": parse_bool(row["numpy_rng_unchanged"]),
            "torch_rng": parse_bool(row["torch_rng_unchanged"]),
        }
        checks.append(
            {
                "layout_id": key[0],
                "agent_id": key[1],
                "checks": metric_checks,
                "status": "PASSED" if all(metric_checks.values()) else "FAILED",
            }
        )
    passed = all(row["status"] == "PASSED" for row in checks) and len(checks) == 72
    return {
        "PREVIEW_BASELINE_REPRODUCTION": "PASS" if passed else "FAIL",
        "PREVIEW_BASELINE_REPRODUCTION_MISMATCH": "NO" if passed else "YES",
        "status": "PASSED" if passed else "FAILED",
        "passed_candidate_count": sum(row["status"] == "PASSED" for row in checks),
        "candidate_count": len(checks),
        "rtol": rtol,
        "atol": atol,
        "A_early_stop_disabled": True,
        "checks": checks,
    }


def _verify_factor_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    index = {
        (str(row["layout_id"]), int(row["agent_id"]), str(row["cell"])): row
        for row in rows
    }
    checks = []
    for layout_id, agent_id in sorted({(key[0], key[1]) for key in index}):
        cells = {cell: index[(layout_id, agent_id, cell)] for cell in CELL_ORDER}
        invariant_fields = (
            "candidate_record_hash",
            "candidate_set_hash",
            "candidate_index",
            "candidate_world_point",
            "proposal_score",
            "K_t",
        )
        invariant = all(cells[cell][field] == cells["A"][field] for cell in CELL_ORDER for field in invariant_fields)
        a_positions = np.asarray(cells["A"]["trajectory_positions"], dtype=float)
        c_positions = np.asarray(cells["C"]["trajectory_positions"], dtype=float)
        a_actions = np.asarray(cells["A"]["trajectory_actions"], dtype=float)
        c_actions = np.asarray(cells["C"]["trajectory_actions"], dtype=float)
        b_positions = np.asarray(cells["B"]["trajectory_positions"], dtype=float)
        d_positions = np.asarray(cells["D"]["trajectory_positions"], dtype=float)
        b_actions = np.asarray(cells["B"]["trajectory_actions"], dtype=float)
        d_actions = np.asarray(cells["D"]["trajectory_actions"], dtype=float)
        ac_prefix = bool(
            c_positions.shape[0] >= a_positions.shape[0]
            and c_actions.shape[0] >= a_actions.shape[0]
            and np.allclose(a_positions, c_positions[: a_positions.shape[0]], rtol=0.0, atol=1e-10)
            and np.allclose(a_actions, c_actions[: a_actions.shape[0]], rtol=0.0, atol=1e-10)
        )
        bd_length = min(b_positions.shape[0], 5)
        bd_action_length = min(b_actions.shape[0], 4)
        bd_prefix = bool(
            d_positions.shape[0] >= bd_length
            and d_actions.shape[0] >= bd_action_length
            and np.allclose(b_positions[:bd_length], d_positions[:bd_length], rtol=0.0, atol=1e-10)
            and np.allclose(b_actions[:bd_action_length], d_actions[:bd_action_length], rtol=0.0, atol=1e-10)
        )
        checks.append(
            {
                "layout_id": layout_id,
                "agent_id": agent_id,
                "candidate_invariants": invariant,
                "A_C_first_four_prefix_identity": ac_prefix,
                "B_D_first_four_prefix_identity": bd_prefix,
                "A_B_only_sensing_factor": (
                    cells["A"]["horizon_requested"] == cells["B"]["horizon_requested"] == 4
                    and cells["A"]["sensing_mode"] != cells["B"]["sensing_mode"]
                ),
                "C_D_only_sensing_factor": (
                    cells["C"]["horizon_requested"] == cells["D"]["horizon_requested"] == 20
                    and cells["C"]["sensing_mode"] != cells["D"]["sensing_mode"]
                    and cells["C"]["diagnostic_termination_enabled"]
                    and cells["D"]["diagnostic_termination_enabled"]
                ),
            }
        )
    passed = all(all(value for key, value in row.items() if key not in ("layout_id", "agent_id")) for row in checks)
    return {"status": "PASSED" if passed else "FAILED", "checks": checks}


def _paired_rows(
    rows: Sequence[Mapping[str, Any]],
    first_cell: str,
    second_cell: str,
    comparison: str,
) -> list[dict[str, Any]]:
    index = {
        (str(row["layout_id"]), int(row["agent_id"]), str(row["cell"])): row
        for row in rows
    }
    output = []
    keys = sorted({(key[0], key[1]) for key in index})
    for layout_id, agent_id in keys:
        first = index[(layout_id, agent_id, first_cell)]
        second = index[(layout_id, agent_id, second_cell)]
        item = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "candidate_pair",
            "comparison": comparison,
            "from_cell": first_cell,
            "to_cell": second_cell,
            "layout_id": layout_id,
            "family": first["family"],
            "agent_id": agent_id,
            "success_control": first["success_control"],
            "reference_failure_agent": first["reference_failure_agent"],
            "reference_failure_subcategory": first["reference_failure_subcategory"],
            "actual_obstacle_collision_agent": first["actual_obstacle_collision_agent"],
            "candidate_record_hash": first["candidate_record_hash"],
            "ground_truth_collision_from": first["preview_ground_truth_collision"],
            "ground_truth_collision_to": second["preview_ground_truth_collision"],
            "new_ground_truth_collision_detected": (
                not parse_bool(first["preview_ground_truth_collision"])
                and parse_bool(second["preview_ground_truth_collision"])
            ),
        }
        for output_name, field in FEATURE_FIELDS.items():
            first_value = float(first[field])
            second_value = float(second[field])
            item[f"{output_name}_from"] = first_value
            item[f"{output_name}_to"] = second_value
            item[f"{output_name}_delta"] = second_value - first_value
        output.append(item)
    return output


def _success_failure_separation(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for cell in CELL_ORDER:
        cell_rows = [row for row in rows if row["cell"] == cell]
        success = [row for row in cell_rows if parse_bool(row["success_control"])]
        comparison_cohorts = {
            "ALL_FAILURE_AGENTS": [
                row for row in cell_rows if not parse_bool(row["success_control"])
            ],
            "REFERENCE_FAILURE_AGENTS": [
                row for row in cell_rows if parse_bool(row["reference_failure_agent"])
            ],
            "OBSTACLE_COLLISION_AGENTS": [
                row
                for row in cell_rows
                if parse_bool(row["actual_obstacle_collision_agent"])
            ],
            "INTER_AGENT_FAILURE_CONTEXT": [
                row
                for row in cell_rows
                if "INTER_AGENT" in str(row["team_failure_category"])
            ],
        }
        for comparison_cohort, failure in comparison_cohorts.items():
            for output_name, field in FEATURE_FIELDS.items():
                success_values = [float(row[field]) for row in success]
                failure_values = [float(row[field]) for row in failure]
                success_stats = descriptive(success_values)
                failure_stats = descriptive(failure_values)
                success_median = success_stats["median"]
                failure_median = failure_stats["median"]
                output.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "cell": cell,
                        "comparison_cohort": comparison_cohort,
                        "metric": output_name,
                        "success_count": success_stats["count"],
                        "failure_count": failure_stats["count"],
                        "success_median": success_median,
                        "failure_median": failure_median,
                        "success_minus_failure_median_gap": (
                            None
                            if success_median is None or failure_median is None
                            else float(success_median) - float(failure_median)
                        ),
                        "rank_biserial_success_vs_failure": rank_biserial(
                            success_values, failure_values
                        ),
                        "classifier_trained": False,
                    }
                )
    return output


def _obstacle_rows(
    rows: Sequence[Mapping[str, Any]], source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    index = {
        (str(row["layout_id"]), int(row["agent_id"]), str(row["cell"])): row
        for row in rows
    }
    output = []
    for actual in source["obstacle_agents"]:
        key = (str(actual["layout_id"]), int(actual["agent_id"]))
        item = {
            "schema_version": SCHEMA_VERSION,
            "layout_id": key[0],
            "family": actual["family"],
            "agent_id": key[1],
            "real_collision_step": int(actual["collision_step"]),
            "real_pre_collision_clearance_trace_m": parse_json_cell(
                actual["actual_pre_collision_static_clearance_trace_m"], []
            ),
            "real_trace_usage": "RETROSPECTIVE_LABEL_ONLY_NOT_PREVIEW_INPUT",
        }
        for cell in CELL_ORDER:
            row = index[(key[0], key[1], cell)]
            item[f"{cell}_sensor_derived_min_clearance_m"] = row["min_clearance"]
            item[f"{cell}_sensor_derived_clearance_finite"] = row[
                "sensor_derived_clearance_finite"
            ]
            item[f"{cell}_preview_ground_truth_collision"] = row[
                "preview_ground_truth_collision"
            ]
            item[f"{cell}_preview_collision_step"] = row["preview_collision_step"]
            item[f"{cell}_first_minimum_clearance_step"] = row[
                "first_minimum_clearance_step"
            ]
            item[f"{cell}_task_progress"] = row["task_progress"]
            item[f"{cell}_max_execution_deviation"] = row[
                "max_execution_deviation"
            ]
        output.append(item)
    return output


def _reference_rows(
    rows: Sequence[Mapping[str, Any]], source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    index = {
        (str(row["layout_id"]), int(row["agent_id"]), str(row["cell"])): row
        for row in rows
    }
    output = []
    for actual in source["reference_agents"]:
        key = (str(actual["layout_id"]), int(actual["agent_id"]))
        item = {
            "schema_version": SCHEMA_VERSION,
            "layout_id": key[0],
            "family": actual["family"],
            "agent_id": key[1],
            "primary_subcategory": actual["primary_subcategory"],
            "actual_window_reference_progress_m": float(
                actual["window_reference_progress_m"]
            ),
            "actual_window_mean_speed_mps": float(actual["window_mean_speed_mps"]),
            "actual_diagnostic_flags": parse_json_cell(actual["diagnostic_flags"], []),
        }
        for cell in CELL_ORDER:
            row = index[(key[0], key[1], cell)]
            for output_name, field in FEATURE_FIELDS.items():
                item[f"{cell}_{output_name}"] = row[field]
            item[f"{cell}_candidate_closing_progress_m"] = row[
                "candidate_closing_progress_m"
            ]
            item[f"{cell}_candidate_closing_progress_per_step_m"] = row[
                "candidate_closing_progress_per_step_m"
            ]
            item[f"{cell}_preview_ground_truth_collision"] = row[
                "preview_ground_truth_collision"
            ]
        output.append(item)
    return output


def _runtime_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    a_mean = float(
        descriptive([float(row["runtime_total_ms"]) for row in rows if row["cell"] == "A"])["mean"]
    )
    output = []
    for cell in CELL_ORDER:
        cell_rows = [row for row in rows if row["cell"] == cell]
        total = descriptive([float(row["runtime_total_ms"]) for row in cell_rows])
        sensing = descriptive([float(row["runtime_sensing_ms"]) for row in cell_rows])
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "cell": cell,
                "candidate_count": len(cell_rows),
                "mean_runtime_ms_per_candidate": total["mean"],
                "p50_runtime_ms_per_candidate": total["p50"],
                "p95_runtime_ms_per_candidate": total["p95"],
                "mean_sensing_runtime_ms": sensing["mean"],
                "relative_total_cost_vs_A": (
                    float(total["mean"]) / a_mean if a_mean > 0.0 else None
                ),
            }
        )
    return output


def _effect_conclusion(
    *,
    rows: Sequence[Mapping[str, Any]],
    obstacle_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    separation_rows: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    thresholds = settings["effect_classification"]

    def collision_improvement(first: str, second: str) -> tuple[int, int]:
        count = 0
        for row in obstacle_rows:
            new_event = (
                not parse_bool(row[f"{first}_preview_ground_truth_collision"])
                and parse_bool(row[f"{second}_preview_ground_truth_collision"])
            )
            first_clearance = float(row[f"{first}_sensor_derived_min_clearance_m"])
            second_clearance = float(row[f"{second}_sensor_derived_min_clearance_m"])
            lower_clearance = second_clearance < first_clearance - float(
                thresholds["directional_improvement_epsilon"]
            )
            count += int(new_event or lower_clearance)
        return count, len(obstacle_rows)

    ac_count, obstacle_total = collision_improvement("A", "C")
    ab_count, _ = collision_improvement("A", "B")
    cd_count, _ = collision_improvement("C", "D")
    effect_h = classify_directional_fraction(ac_count, obstacle_total, thresholds)
    effect_s = classify_directional_fraction(ab_count, obstacle_total, thresholds)
    effect_long_s = classify_directional_fraction(cd_count, obstacle_total, thresholds)

    detection_counts = {
        cell: sum(
            parse_bool(row[f"{cell}_preview_ground_truth_collision"])
            for row in obstacle_rows
        )
        for cell in CELL_ORDER
    }
    d_strictly_better = detection_counts["D"] > max(
        detection_counts["B"], detection_counts["C"]
    )
    d_clearance_better_b = sum(
        float(row["D_sensor_derived_min_clearance_m"])
        < float(row["B_sensor_derived_min_clearance_m"])
        - float(thresholds["directional_improvement_epsilon"])
        for row in obstacle_rows
    )
    d_clearance_better_c = sum(
        float(row["D_sensor_derived_min_clearance_m"])
        < float(row["C_sensor_derived_min_clearance_m"])
        - float(thresholds["directional_improvement_epsilon"])
        for row in obstacle_rows
    )
    if d_strictly_better and d_clearance_better_b > 0 and d_clearance_better_c > 0:
        interaction = "SUPPORTED"
    elif d_clearance_better_b > 0 and d_clearance_better_c > 0:
        interaction = "PARTIAL"
    else:
        interaction = "NO"

    sep_index: dict[str, float] = {}
    overall_sep_index: dict[str, float] = {}
    for cell in CELL_ORDER:
        values = [
            abs(float(row["rank_biserial_success_vs_failure"]))
            for row in separation_rows
            if row["cell"] == cell
            and row["comparison_cohort"] == "REFERENCE_FAILURE_AGENTS"
            and row["rank_biserial_success_vs_failure"] is not None
        ]
        sep_index[cell] = float(np.mean(values)) if values else 0.0
        overall_values = [
            abs(float(row["rank_biserial_success_vs_failure"]))
            for row in separation_rows
            if row["cell"] == cell
            and row["comparison_cohort"] == "ALL_FAILURE_AGENTS"
            and row["rank_biserial_success_vs_failure"] is not None
        ]
        overall_sep_index[cell] = (
            float(np.mean(overall_values)) if overall_values else 0.0
        )
    reference_delta = sep_index["D"] - sep_index["A"]
    if reference_delta >= 0.1:
        reference_predictability = "IMPROVED"
    elif reference_delta > float(thresholds["directional_improvement_epsilon"]):
        reference_predictability = "PARTIAL"
    else:
        reference_predictability = "NOT_IMPROVED"
    if detection_counts["D"] > detection_counts["A"]:
        obstacle_predictability = "IMPROVED"
    elif cd_count > 0 or ab_count > 0 or ac_count > 0:
        obstacle_predictability = "PARTIAL"
    else:
        obstacle_predictability = "NOT_IMPROVED"

    frozen_limitation = (
        "YES"
        if effect_s in ("STRONG", "MODERATE") or effect_long_s in ("STRONG", "MODERATE")
        else "PARTIAL"
        if effect_s == "WEAK" or effect_long_s == "WEAK"
        else "NO"
    )
    horizon_limitation = (
        "YES"
        if effect_h in ("STRONG", "MODERATE")
        else "PARTIAL"
        if effect_h == "WEAK"
        else "NO"
    )
    major_failure_exposure_improved = bool(
        detection_counts["D"] > detection_counts["A"]
        or reference_predictability == "IMPROVED"
    )
    if not major_failure_exposure_improved:
        primary = "NOT_RESOLVED_BY_HORIZON_OR_SENSING"
        direction = "NEITHER_ESTABLISHED"
    elif horizon_limitation == "YES" and frozen_limitation == "YES":
        primary = "HORIZON_AND_SENSING_INTERACTION" if interaction != "NO" else "HORIZON_AND_SENSING"
        direction = "BOTH"
    elif frozen_limitation == "YES":
        primary = "FROZEN_SENSING_RECONSTRUCTION"
        direction = "REFRESHED_SENSING"
    elif horizon_limitation == "YES":
        primary = "SHORT_HORIZON"
        direction = "LONGER_HORIZON"
    else:
        primary = "NOT_RESOLVED_BY_HORIZON_OR_SENSING"
        direction = "NEITHER_ESTABLISHED"
    formal_change = bool(
        direction != "NEITHER_ESTABLISHED"
        and obstacle_predictability == "IMPROVED"
        and effect_long_s in ("STRONG", "MODERATE")
    )
    conclusion = {
        "PREVIEW_BASELINE_REPRODUCTION": "PASS",
        "HORIZON_ONLY_EFFECT": effect_h,
        "SENSING_ONLY_EFFECT": effect_s,
        "LONG_HORIZON_SENSING_EFFECT": effect_long_s,
        "HORIZON_SENSING_INTERACTION": interaction,
        "FROZEN_SURFACE_RECONSTRUCTION_LIMITATION": frozen_limitation,
        "SHORT_HORIZON_LIMITATION": horizon_limitation,
        "OBSTACLE_FAILURE_PREDICTABILITY": obstacle_predictability,
        "REFERENCE_FAILURE_PREDICTABILITY": reference_predictability,
        "PRIMARY_PREVIEW_LIMITATION": primary,
        "FINAL_FP_SHEP_REDESIGN_DIRECTION": direction,
        "FORMAL_ALGORITHM_CHANGE_RECOMMENDED": "YES" if formal_change else "NO",
        "REFRESHED_SENSING_SCOPE": REFRESHED_SENSING_SCOPE,
        "COLLISION_DETECTED_A": f"{detection_counts['A']}/{obstacle_total}",
        "COLLISION_DETECTED_B": f"{detection_counts['B']}/{obstacle_total}",
        "COLLISION_DETECTED_C": f"{detection_counts['C']}/{obstacle_total}",
        "COLLISION_DETECTED_D": f"{detection_counts['D']}/{obstacle_total}",
        "STATIC_SURFACE_VISIBILITY_IMPROVED_D_VS_C": f"{cd_count}/{obstacle_total}",
        "MAJOR_FAILURE_WARNING_GAIN_D_VS_A": f"{max(detection_counts['D'] - detection_counts['A'], 0)}/{obstacle_total}",
        "FP_SHEP_SCOPE_LIMITATION": (
            "SUPPORTED" if not major_failure_exposure_improved else "NOT_ESTABLISHED"
        ),
        "reference_failure_separation_index": sep_index,
        "overall_failure_separation_index": overall_sep_index,
        "effect_labels_are_statistical_significance": False,
        "NEXT_STEP": (
            "Await human review before any formal FP-SHEP change; do not enable refreshed sensing or alter H automatically."
        ),
    }
    mechanism_rows = [
        {
            "mechanism": "HORIZON_ONLY_A_TO_C",
            "directionally_improved_obstacle_cases": ac_count,
            "obstacle_case_count": obstacle_total,
            "effect_label": effect_h,
        },
        {
            "mechanism": "SENSING_ONLY_A_TO_B",
            "directionally_improved_obstacle_cases": ab_count,
            "obstacle_case_count": obstacle_total,
            "effect_label": effect_s,
        },
        {
            "mechanism": "LONG_HORIZON_SENSING_C_TO_D",
            "directionally_improved_obstacle_cases": cd_count,
            "obstacle_case_count": obstacle_total,
            "effect_label": effect_long_s,
        },
        {
            "mechanism": "HORIZON_SENSING_INTERACTION",
            "directionally_improved_obstacle_cases": None,
            "obstacle_case_count": obstacle_total,
            "effect_label": interaction,
        },
        {
            "mechanism": "REFERENCE_FAILURE_SEPARATION",
            "directionally_improved_obstacle_cases": None,
            "obstacle_case_count": len(reference_rows),
            "effect_label": reference_predictability,
        },
    ]
    return conclusion, mechanism_rows


def _render_report(
    *,
    conclusion: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    runtime_rows: Sequence[Mapping[str, Any]],
    obstacle_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    factor_checks: Mapping[str, Any],
) -> str:
    runtime = {row["cell"]: row for row in runtime_rows}
    lines = [
        "# FP-SHEP 2×2 Preview-Fidelity Control Diagnosis",
        "",
        "## 1. Scope and semantic controls",
        "",
        "This is a post-hoc preview-only control experiment over the frozen 24-layout manifest and 72 already-selected candidates. No Proposal call, candidate reselection, real environment step, training, GAT, joint rollout, or formal benchmark rerun was performed.",
        "",
        f"- REFRESHED_SENSING_SCOPE = **{REFRESHED_SENSING_SCOPE}**",
        "- A/B retain the complete H=4 trajectory without diagnostic early stopping.",
        "- C/D share the same H=20 static-collision/terminal-success diagnostic termination rule.",
        "- Sensor-derived clearance is separate from ground-truth collision geometry.",
        "- Peer-visible t=0 surface samples remain frozen; peer future motion is not predicted.",
        "",
        "## 2. Baseline reproduction gate",
        "",
        f"- PREVIEW_BASELINE_REPRODUCTION = **{conclusion['PREVIEW_BASELINE_REPRODUCTION']}**",
        f"- Passed: {reproduction['passed_candidate_count']}/{reproduction['candidate_count']} selected candidates.",
        f"- Factor-integrity gate: **{factor_checks['status']}**.",
        "",
        "## 3. Primary results",
        "",
        f"- HORIZON_ONLY_EFFECT = **{conclusion['HORIZON_ONLY_EFFECT']}**",
        f"- SENSING_ONLY_EFFECT = **{conclusion['SENSING_ONLY_EFFECT']}**",
        f"- LONG_HORIZON_SENSING_EFFECT = **{conclusion['LONG_HORIZON_SENSING_EFFECT']}**",
        f"- HORIZON_SENSING_INTERACTION = **{conclusion['HORIZON_SENSING_INTERACTION']}**",
        f"- Collision detection A/B/C/D: {conclusion['COLLISION_DETECTED_A']}, {conclusion['COLLISION_DETECTED_B']}, {conclusion['COLLISION_DETECTED_C']}, {conclusion['COLLISION_DETECTED_D']}.",
        f"- Static-surface visibility improved from C to D in {conclusion['STATIC_SURFACE_VISIBILITY_IMPROVED_D_VS_C']} real collision cases, but major warning gain was {conclusion['MAJOR_FAILURE_WARNING_GAIN_D_VS_A']}.",
        "",
        "The six real obstacle-collision traces are used only as retrospective labels. They never enter preview propagation or feature computation.",
        "",
        "## 4. Failure-cohort interpretation",
        "",
        f"- OBSTACLE_FAILURE_PREDICTABILITY = **{conclusion['OBSTACLE_FAILURE_PREDICTABILITY']}**",
        f"- REFERENCE_FAILURE_PREDICTABILITY = **{conclusion['REFERENCE_FAILURE_PREDICTABILITY']}**",
        f"- Obstacle-collision agents: {len(obstacle_rows)}.",
        f"- Reference-unreachable affected agents: {len(reference_rows)}.",
        "- TIME_BUDGET_ONLY remains an execution-budget label and is not reinterpreted as a preview failure.",
        "",
        "## 5. Runtime",
        "",
        "| Cell | Mean ms/candidate | P50 | P95 | Relative to A |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in CELL_ORDER:
        row = runtime[cell]
        lines.append(
            f"| {cell} | {row['mean_runtime_ms_per_candidate']:.3f} | "
            f"{row['p50_runtime_ms_per_candidate']:.3f} | "
            f"{row['p95_runtime_ms_per_candidate']:.3f} | "
            f"{row['relative_total_cost_vs_A']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 6. Decision",
            "",
            f"- FROZEN_SURFACE_RECONSTRUCTION_LIMITATION = **{conclusion['FROZEN_SURFACE_RECONSTRUCTION_LIMITATION']}**",
            f"- SHORT_HORIZON_LIMITATION = **{conclusion['SHORT_HORIZON_LIMITATION']}**",
            f"- PRIMARY_PREVIEW_LIMITATION = **{conclusion['PRIMARY_PREVIEW_LIMITATION']}**",
            f"- FINAL_FP_SHEP_REDESIGN_DIRECTION = **{conclusion['FINAL_FP_SHEP_REDESIGN_DIRECTION']}**",
            f"- FP_SHEP_SCOPE_LIMITATION = **{conclusion['FP_SHEP_SCOPE_LIMITATION']}**",
            f"- FORMAL_ALGORITHM_CHANGE_RECOMMENDED = **{conclusion['FORMAL_ALGORITHM_CHANGE_RECOMMENDED']}**",
            f"- NEXT_STEP: {conclusion['NEXT_STEP']}",
            "",
            "The refreshed result is attributable only to static-obstacle sensing fidelity. It must not be generalized to dynamic-obstacle or multi-agent future-sensing reconstruction.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment(settings: Mapping[str, Any], output_dir: Path) -> Path:
    _assert_config(settings)
    source = _load_sources(settings)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checkpoint_path = (REPO_ROOT / str(settings["checkpoint"])).resolve()
    checkpoint_sha_before = sha256_file(checkpoint_path)
    if checkpoint_sha_before != str(settings["checkpoint_sha256_expected"]):
        raise RuntimeError("checkpoint SHA256 mismatch")
    manifest_sha_before = sha256_file(source["manifest_path"])
    critical_before = _critical_source_hashes()
    source_failure_hashes = {
        name: sha256_file(source["failure_root"] / name)
        for name in (
            "horizon_diagnostic.csv",
            "failure_cohort.csv",
            "lower_policy_failure_analysis.csv",
            "obstacle_collision_analysis.csv",
        )
    }
    config_hash = _stable_hash(settings)
    write_json(
        output_dir / "source_artifact_manifest.json",
        {
            "source_failure_audit": str(source["failure_root"]),
            "source_geometry_artifact": str(source["geometry_root"]),
            "scenario_manifest_path": str(source["manifest_path"]),
            "scenario_manifest_sha256": source["manifest_sha256"],
            "scenario_manifest_bytes_reused": True,
            "scenario_manifest_regenerated": False,
            "layout_count": len(source["manifest"]["layouts"]),
            "selected_candidate_count": len(source["selections"]),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha_before,
            "source_failure_artifact_hashes": source_failure_hashes,
            "diagnostic_config_sha256": config_hash,
        },
    )
    resolved_config = copy.deepcopy(dict(settings))
    resolved_config.update(
        {
            "resolved_output_dir": str(output_dir),
            "diagnostic_config_sha256": config_hash,
            "warning_criteria_frozen_before_run": True,
            "REFRESHED_SENSING_SCOPE": REFRESHED_SENSING_SCOPE,
        }
    )
    write_json(output_dir / "config.json", resolved_config)

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

    layout_index = {str(row["layout_id"]): row for row in source["manifest"]["layouts"]}
    selection_index: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    agent_index = {
        (str(row["layout_id"]), int(row["agent_id"])): row for row in source["agents"]
    }
    episode_index = {str(row["layout_id"]): row for row in source["episodes"]}
    for row in source["selections"]:
        selection_index[str(row["layout_id"])].append(row)
    cohort = _cohort_indices(source)

    # Stage A: strict formal H=4 numerical anchor. No B/C/D code runs before
    # every selected candidate passes this gate.
    all_rows: list[dict[str, Any]] = []
    initial_state_checks: list[dict[str, Any]] = []
    for layout_counter, layout_id in enumerate(sorted(layout_index), start=1):
        layout = layout_index[layout_id]
        env, _ = build_generalization_environment(
            config=multi_config,
            scenario=str(source["manifest"]["scenario_id"]),
            seed=int(layout["evaluation_seed"]),
            peer_radius=float(settings["peer_radius"]),
        )
        scene_hash = _scenario_hash(_scene_snapshot(env))
        source_scene_hash = str(episode_index[layout_id]["initial_condition_hash"])
        initial_state_checks.append(
            {
                "layout_id": layout_id,
                "computed_initial_state_hash": scene_hash,
                "source_initial_state_hash": source_scene_hash,
                "match": scene_hash == source_scene_hash,
            }
        )
        if scene_hash != source_scene_hash:
            env.close()
            raise RuntimeError(f"frozen initial state mismatch for {layout_id}")
        for selection in sorted(selection_index[layout_id], key=lambda row: int(row["agent_id"])):
            candidate = _candidate_record(selection)
            key = (layout_id, int(selection["agent_id"]))
            all_rows.append(
                _run_cell(
                    env=env,
                    cell="A",
                    layout=layout,
                    selection=selection,
                    candidate=candidate,
                    policy=policy,
                    source_agent=agent_index[key],
                    source_episode=episode_index[layout_id],
                    cohort=cohort,
                    settings=settings,
                )
            )
        env.close()
        print(f"[A reproduction {layout_counter}/24] {layout_id}", flush=True)

    reproduction = _reproduction_checks(all_rows, source, settings)
    reproduction["initial_state_checks"] = initial_state_checks
    write_json(output_dir / "reproduction_check.json", reproduction)
    write_csv(output_dir / "per_candidate_2x2.csv", all_rows)
    if reproduction["status"] != "PASSED":
        raise RuntimeError("A formal H=4 preview reproduction mismatch")

    # Stage B/C/D: same frozen candidate and initial state, no selection call.
    for layout_counter, layout_id in enumerate(sorted(layout_index), start=1):
        layout = layout_index[layout_id]
        env, _ = build_generalization_environment(
            config=multi_config,
            scenario=str(source["manifest"]["scenario_id"]),
            seed=int(layout["evaluation_seed"]),
            peer_radius=float(settings["peer_radius"]),
        )
        for selection in sorted(selection_index[layout_id], key=lambda row: int(row["agent_id"])):
            candidate = _candidate_record(selection)
            key = (layout_id, int(selection["agent_id"]))
            for cell in ("B", "C", "D"):
                all_rows.append(
                    _run_cell(
                        env=env,
                        cell=cell,
                        layout=layout,
                        selection=selection,
                        candidate=candidate,
                        policy=policy,
                        source_agent=agent_index[key],
                        source_episode=episode_index[layout_id],
                        cohort=cohort,
                        settings=settings,
                    )
                )
        env.close()
        print(f"[B/C/D {layout_counter}/24] {layout_id}", flush=True)

    all_rows.sort(key=lambda row: (row["layout_id"], int(row["agent_id"]), CELL_ORDER.index(row["cell"])))
    if len(all_rows) != 288:
        raise RuntimeError("2x2 diagnosis did not produce 72 x 4 records")
    write_csv(output_dir / "per_candidate_2x2.csv", all_rows)

    factor_checks = _verify_factor_pairs(all_rows)
    if factor_checks["status"] != "PASSED":
        write_json(output_dir / "factor_integrity.json", factor_checks)
        raise RuntimeError("2x2 factor isolation check failed")
    write_json(output_dir / "factor_integrity.json", factor_checks)

    obstacle_rows = _obstacle_rows(all_rows, source)
    reference_rows = _reference_rows(all_rows, source)
    separation_rows = _success_failure_separation(all_rows)
    horizon_rows = _paired_rows(all_rows, "A", "C", "HORIZON_ONLY")
    sensing_rows = _paired_rows(all_rows, "A", "B", "SENSING_ONLY_H4")
    sensing_rows.extend(_paired_rows(all_rows, "C", "D", "SENSING_ONLY_H20"))
    interaction_rows = []
    row_index = {
        (str(row["layout_id"]), int(row["agent_id"]), str(row["cell"])): row
        for row in all_rows
    }
    for layout_id, agent_id in sorted({(key[0], key[1]) for key in row_index}):
        cells = {cell: row_index[(layout_id, agent_id, cell)] for cell in CELL_ORDER}
        item = {
            "schema_version": SCHEMA_VERSION,
            "layout_id": layout_id,
            "family": cells["A"]["family"],
            "agent_id": agent_id,
            "actual_obstacle_collision_agent": cells["A"]["actual_obstacle_collision_agent"],
        }
        for output_name, field in FEATURE_FIELDS.items():
            a, b, c, d = (float(cells[cell][field]) for cell in CELL_ORDER)
            item[f"{output_name}_A"] = a
            item[f"{output_name}_B"] = b
            item[f"{output_name}_C"] = c
            item[f"{output_name}_D"] = d
            item[f"{output_name}_difference_in_differences"] = (d - c) - (b - a)
        interaction_rows.append(item)
    runtime_rows = _runtime_rows(all_rows)
    conclusion, mechanism_rows = _effect_conclusion(
        rows=all_rows,
        obstacle_rows=obstacle_rows,
        reference_rows=reference_rows,
        separation_rows=separation_rows,
        settings=settings,
    )

    write_csv(output_dir / "obstacle_collision_2x2.csv", obstacle_rows)
    write_csv(output_dir / "reference_failure_2x2.csv", reference_rows)
    write_csv(output_dir / "success_failure_separation.csv", separation_rows)
    write_csv(output_dir / "horizon_effect.csv", horizon_rows)
    write_csv(output_dir / "sensing_effect.csv", sensing_rows)
    write_csv(output_dir / "interaction_effect.csv", interaction_rows)
    write_csv(output_dir / "runtime_cost.csv", runtime_rows)
    write_csv(output_dir / "mechanism_summary.csv", mechanism_rows)

    checkpoint_sha_after = sha256_file(checkpoint_path)
    manifest_sha_after = sha256_file(source["manifest_path"])
    policy_hash_after = _policy_parameter_sha256(policy)
    critical_after = _critical_source_hashes()
    source_failure_hashes_after = {
        name: sha256_file(source["failure_root"] / name) for name in source_failure_hashes
    }
    integrity = {
        "manifest_hash_unchanged": manifest_sha_before == manifest_sha_after,
        "manifest_bytes_unchanged": source["manifest_path"].read_bytes() == source["manifest_bytes"],
        "checkpoint_hash_unchanged": checkpoint_sha_before == checkpoint_sha_after,
        "actor_parameters_unchanged": policy_hash_before == policy_hash_after,
        "critical_source_hashes_unchanged": critical_before == critical_after,
        "source_failure_artifacts_unchanged": source_failure_hashes == source_failure_hashes_after,
        "candidate_generation_count": 0,
        "candidate_reselection_count": 0,
        "real_environment_step_count": 0,
        "GAT_call_count": 0,
        "gradient_update_count": 0,
        "formal_benchmark_rerun": False,
        "formal_H_preview": 4,
        "formal_score": "+progress +clearance -deviation",
        "formal_terminal_speed_weight": 0.0,
        "REFRESHED_SENSING_SCOPE": REFRESHED_SENSING_SCOPE,
        "all_record_state_and_rng_checks_pass": all(
            parse_bool(row[field])
            for row in all_rows
            for field in (
                "environment_state_unchanged",
                "environment_rng_unchanged",
                "numpy_rng_unchanged",
                "torch_rng_unchanged",
                "history_shift_verified",
                "historical_gate_verified",
            )
        ),
    }
    required_true_integrity_fields = (
        "manifest_hash_unchanged",
        "manifest_bytes_unchanged",
        "checkpoint_hash_unchanged",
        "actor_parameters_unchanged",
        "critical_source_hashes_unchanged",
        "source_failure_artifacts_unchanged",
        "all_record_state_and_rng_checks_pass",
    )
    if not all(bool(integrity[key]) for key in required_true_integrity_fields):
        write_json(output_dir / "integrity.json", integrity)
        raise RuntimeError("final integrity gate failed")
    write_json(output_dir / "integrity.json", integrity)
    conclusion["integrity"] = integrity
    conclusion["runtime_seconds"] = time.perf_counter() - started
    conclusion["candidate_records"] = len(all_rows)
    conclusion["formal_algorithm_was_modified"] = False
    write_json(output_dir / "conclusion.json", conclusion)
    report = _render_report(
        conclusion=conclusion,
        reproduction=reproduction,
        runtime_rows=runtime_rows,
        obstacle_rows=obstacle_rows,
        reference_rows=reference_rows,
        factor_checks=factor_checks,
    )
    (output_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    resolved_config["runtime_seconds"] = conclusion["runtime_seconds"]
    resolved_config["completed_candidate_records"] = len(all_rows)
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
        output_dir = (REPO_ROOT / str(settings["output_dir"]) / timestamp).resolve()
    else:
        output_dir = args.output_dir.resolve()
    result = run_experiment(settings, output_dir)
    print(result)


if __name__ == "__main__":
    main()
