"""Minimal Go/No-Go evaluation for a frozen SAC-DMP corridor navigator.

The upper controller in this module is restricted to selecting an active DMP
waypoint.  The deterministic network action is passed to ``env.step`` without
projection, blending, residuals, forced braking, or any other post-processing.

Execution is deliberately gated.  E0 and E1 are always evaluated first.  The
runner writes their evidence and stops with a No-Go decision when either route
replay or staggered execution is not viable.  Later stages are added only after
their prerequisite gate has passed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
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

from experiment_config import EXPERIMENT_CONFIG as SINGLE_AGENT_CONFIG  # noqa: E402
from runner_sac import build_env as build_single_env  # noqa: E402
from runner_sac import build_model as build_single_model  # noqa: E402
from runner_sac import load_checkpoint  # noqa: E402
from scripts import frozen_policy_corridor_reservation as corridor  # noqa: E402
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


DEFAULT_CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "evaluation"
    / "frozen_policy_corridor_reservation_minimal.json"
)
E_SCENARIO_NAME = "E_head_on_narrow_peer_spheres"
E0_GROUP = "E0_single_agent_route_replay"
E1_GROUP = "E1_three_agent_staggered_oracle"
FIRST_FAILURE_TYPES = {
    "INTER_AGENT_COLLISION",
    "OBSTACLE_COLLISION",
    "OUT_OF_BOUNDS",
    "TIMEOUT",
    "DEADLOCK",
    "HOLD_TRACKING_FAILURE",
    "ENTRY_TRACKING_FAILURE",
    "PREMATURE_ENTRY",
    "PREMATURE_RELEASE",
    "OTHER",
}

ROLLOUT_PREDICTION_FIELDS = [
    "group",
    "seed",
    "episode_index",
    "agent_id",
    "prediction_step",
    "candidate_waypoint_type",
    "candidate_waypoint",
    "predicted_entry_time",
    "actual_entry_time",
    "prediction_entry_error",
    "predicted_exit_time",
    "actual_exit_time",
    "prediction_exit_error",
    "predicted_stopping_distance",
    "predicted_min_obstacle_clearance",
    "predicted_min_peer_distance",
    "predicted_feasible",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> None:
    resolved_fields = list(fieldnames or ())
    for row in rows:
        for key in row:
            if key not in resolved_fields:
                resolved_fields.append(key)
    if not resolved_fields:
        raise ValueError(f"CSV schema required for empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(value), ensure_ascii=False)
                        if isinstance(value, (dict, list, tuple, np.ndarray))
                        else _jsonable(value)
                    )
                    for key, value in row.items()
                }
            )


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _actor_snapshot(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.actor.state_dict().items()
    }


def _actor_unchanged(
    before: dict[str, torch.Tensor],
    model: Any,
) -> bool:
    after = model.actor.state_dict()
    return before.keys() == after.keys() and all(
        torch.equal(value, after[name].detach().cpu())
        for name, value in before.items()
    )


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=REPO_ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _waypoint_type(candidate: Any) -> str:
    value = getattr(candidate, "waypoint_type", getattr(candidate, "type", "OTHER"))
    return str(getattr(value, "value", value)).upper()


def _candidate_position(candidate: Any) -> np.ndarray:
    return np.asarray(candidate.position, dtype=float)


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def _minimum_obstacle_clearance(env: Any, agent_index: int) -> float:
    point = np.asarray(env.dynamics[agent_index].p, dtype=float)
    obstacles = list(env.static_obstacles) + list(env.dynamic_obstacles)
    if not obstacles:
        return float("inf")
    return min(float(obstacle.signed_distance(point)) for obstacle in obstacles)


def _minimum_peer_clearance(env: Any, agent_index: int) -> float:
    positions = np.asarray(env._positions(), dtype=float)
    if len(positions) <= 1:
        return float("inf")
    distances = np.linalg.norm(positions - positions[agent_index], axis=1)
    distances[agent_index] = float("inf")
    return float(np.min(distances))


def _out_of_bounds_mask(env: Any) -> np.ndarray:
    lower, upper = np.asarray(env.env_config.workspace_bounds, dtype=float)
    positions = np.asarray(env._positions(), dtype=float)
    return np.any(np.logical_or(positions < lower, positions > upper), axis=1)


def _first_present_failure(candidates: Iterable[str]) -> str | None:
    priority = (
        "INTER_AGENT_COLLISION",
        "OBSTACLE_COLLISION",
        "OUT_OF_BOUNDS",
        "PREMATURE_ENTRY",
        "PREMATURE_RELEASE",
        "HOLD_TRACKING_FAILURE",
        "ENTRY_TRACKING_FAILURE",
        "DEADLOCK",
        "TIMEOUT",
        "OTHER",
    )
    observed = set(candidates)
    return next((name for name in priority if name in observed), None)


@dataclass
class AgentRuntime:
    route_agent_id: int
    environment_agent_id: int
    candidates: list[Any]
    candidate_index: int = 0
    hold_release_step: int | None = None
    upper_state: str = "CRUISE"
    stable_hold_steps: int = 0
    first_hold_entry_speed: float = float("nan")
    hold_min_speed: float = float("inf")
    hold_exit_speed: float = float("nan")
    path_length: float = 0.0
    waiting_steps: int = 0
    switches: int = 0
    max_speed: float = 0.0
    max_acceleration: float = 0.0
    min_obstacle_clearance: float = float("inf")
    min_peer_clearance: float = float("inf")
    max_lateral_deviation: float = 0.0
    entry_time: float = float("nan")
    exit_time: float = float("nan")
    time_to_goal: float = float("nan")
    request_count: int = 0
    authorization_count: int = 0
    hold_timed_out: bool = False
    entry_timed_out: bool = False
    state_enter_step: int = 0
    waypoint_switch_rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def candidate(self) -> Any:
        return self.candidates[self.candidate_index]

    @property
    def waypoint_type(self) -> str:
        return _waypoint_type(self.candidate)


def _state_for_waypoint(waypoint_type: str) -> str:
    return {
        "PROGRESS": "CRUISE",
        "DECELERATION": "DECELERATE",
        "HOLD": "HOLD",
        "ENTRY": "ENTER",
        "EXIT": "OCCUPY",
        "TERMINAL": "CRUISE",
    }.get(str(waypoint_type).upper(), "CRUISE")


def _find_candidate_index(runtime: AgentRuntime, waypoint_type: str) -> int | None:
    requested = str(waypoint_type).upper()
    return next(
        (
            index
            for index, candidate in enumerate(runtime.candidates)
            if _waypoint_type(candidate) == requested
        ),
        None,
    )


def _switch_candidate(
    runtime: AgentRuntime,
    new_index: int,
    *,
    env: Any,
    step: int,
    dt: float,
    reason: str,
    reservation_owner: int | None,
) -> None:
    if int(new_index) == int(runtime.candidate_index):
        return
    if not 0 <= int(new_index) < len(runtime.candidates):
        raise IndexError("candidate waypoint index is out of range")
    previous = runtime.candidate
    new = runtime.candidates[int(new_index)]
    position = np.asarray(env.dynamics[runtime.environment_agent_id].p, dtype=float)
    velocity = np.asarray(env.dynamics[runtime.environment_agent_id].v, dtype=float)
    dmp = env.dmps[runtime.environment_agent_id]
    runtime.waypoint_switch_rows.append(
        {
            "group": "",
            "seed": -1,
            "episode_index": -1,
            "agent_id": runtime.route_agent_id,
            "step": int(step),
            "time": float(step * dt),
            "previous_type": _waypoint_type(previous),
            "new_type": _waypoint_type(new),
            "previous_waypoint": _candidate_position(previous),
            "new_waypoint": _candidate_position(new),
            "switch_reason": str(reason),
            "agent_state": runtime.upper_state,
            "speed": float(np.linalg.norm(velocity)),
            "distance_to_previous": float(np.linalg.norm(position - _candidate_position(previous))),
            "distance_to_new": float(np.linalg.norm(position - _candidate_position(new))),
            "dmp_phase": float(dmp.phase),
            "reservation_owner": reservation_owner,
            "predicted_entry_time": float("nan"),
            "predicted_exit_time": float("nan"),
        }
    )
    if _waypoint_type(previous) == "HOLD":
        runtime.hold_exit_speed = float(np.linalg.norm(velocity))
    runtime.candidate_index = int(new_index)
    runtime.upper_state = _state_for_waypoint(_waypoint_type(new))
    runtime.state_enter_step = int(step)
    runtime.switches += 1
    set_dmp_active_goal_preserve_phase(dmp, _candidate_position(new))


def _initialize_runtime_waypoint(
    runtime: AgentRuntime,
    *,
    env: Any,
    group: str,
    seed: int,
    episode_index: int,
    initial_type: str | None,
    dt: float,
) -> None:
    if initial_type is not None:
        selected = _find_candidate_index(runtime, initial_type)
        if selected is None:
            raise ValueError(
                f"route {runtime.route_agent_id} has no {initial_type} candidate"
            )
        runtime.candidate_index = int(selected)
    runtime.upper_state = _state_for_waypoint(runtime.waypoint_type)
    runtime.state_enter_step = 0
    set_dmp_active_goal_preserve_phase(
        env.dmps[runtime.environment_agent_id],
        _candidate_position(runtime.candidate),
    )
    runtime.waypoint_switch_rows.append(
        {
            "group": group,
            "seed": int(seed),
            "episode_index": int(episode_index),
            "agent_id": runtime.route_agent_id,
            "step": 0,
            "time": 0.0,
            "previous_type": "NONE",
            "new_type": runtime.waypoint_type,
            "previous_waypoint": None,
            "new_waypoint": _candidate_position(runtime.candidate),
            "switch_reason": "episode_initialization",
            "agent_state": runtime.upper_state,
            "speed": 0.0,
            "distance_to_previous": float("nan"),
            "distance_to_new": float(
                np.linalg.norm(
                    env.dynamics[runtime.environment_agent_id].p
                    - _candidate_position(runtime.candidate)
                )
            ),
            "dmp_phase": float(env.dmps[runtime.environment_agent_id].phase),
            "reservation_owner": None,
            "predicted_entry_time": float("nan"),
            "predicted_exit_time": float("nan"),
        }
    )


def _route_direction(runtime: AgentRuntime) -> str:
    value = getattr(runtime.candidate, "direction", "BYPASS")
    return str(getattr(value, "value", value))


def _lateral_deviation(metadata: Any, point: np.ndarray) -> float:
    point = np.asarray(point, dtype=float)
    projection = (
        metadata.centerline_point
        + metadata.longitudinal_coordinate(point) * metadata.centerline_direction
    )
    delta = point - projection
    delta[int(metadata.longitudinal_axis)] = 0.0
    return float(np.linalg.norm(delta))


def _advance_runtime(
    runtime: AgentRuntime,
    *,
    env: Any,
    metadata: Any,
    step: int,
    dt: float,
    execution_config: dict[str, Any],
    reservation_owner: int | None,
) -> None:
    agent_index = runtime.environment_agent_id
    point = np.asarray(env.dynamics[agent_index].p, dtype=float)
    velocity = np.asarray(env.dynamics[agent_index].v, dtype=float)
    speed = float(np.linalg.norm(velocity))
    waypoint_type = runtime.waypoint_type
    waypoint_distance = float(np.linalg.norm(point - _candidate_position(runtime.candidate)))
    waypoint_tolerance = float(execution_config["waypoint_tolerance"])

    if waypoint_type == "HOLD":
        runtime.waiting_steps += 1
        if not np.isfinite(runtime.first_hold_entry_speed):
            runtime.first_hold_entry_speed = speed
        runtime.hold_min_speed = min(runtime.hold_min_speed, speed)
        within_zone = waypoint_distance <= float(execution_config["hold_zone_radius"])
        settled = within_zone and speed <= float(execution_config["hold_speed_threshold"])
        runtime.stable_hold_steps = runtime.stable_hold_steps + 1 if settled else 0
        if step - runtime.state_enter_step >= int(execution_config["hold_tracking_timeout_steps"]):
            if runtime.stable_hold_steps < int(execution_config["hold_stable_steps"]):
                runtime.hold_timed_out = True
        oracle_release_due = (
            runtime.hold_release_step is not None
            and step >= int(runtime.hold_release_step)
        )
        settled_release_due = (
            runtime.hold_release_step is None
            and runtime.stable_hold_steps >= int(execution_config["hold_stable_steps"])
        )
        if oracle_release_due or settled_release_due:
            entry_index = _find_candidate_index(runtime, "ENTRY")
            if entry_index is not None:
                runtime.authorization_count += 1
                _switch_candidate(
                    runtime,
                    entry_index,
                    env=env,
                    step=step,
                    dt=dt,
                    reason=(
                        "staggered_oracle_release"
                        if oracle_release_due
                        else "hold_stable_and_authorized"
                    ),
                    reservation_owner=reservation_owner,
                )
        return

    direction = _route_direction(runtime)
    if waypoint_type == "ENTRY":
        if step - runtime.state_enter_step >= int(execution_config["entry_tracking_timeout_steps"]):
            runtime.entry_timed_out = True
        if metadata.has_crossed_entry(point, direction):
            if not np.isfinite(runtime.entry_time):
                runtime.entry_time = float(step * dt)
            exit_index = _find_candidate_index(runtime, "EXIT")
            if exit_index is not None:
                _switch_candidate(
                    runtime,
                    exit_index,
                    env=env,
                    step=step,
                    dt=dt,
                    reason="crossed_entry_gate",
                    reservation_owner=reservation_owner,
                )
        return

    if waypoint_type == "EXIT":
        if metadata.has_crossed_exit(point, direction):
            if not np.isfinite(runtime.exit_time):
                runtime.exit_time = float(step * dt)
            terminal_index = _find_candidate_index(runtime, "TERMINAL")
            if terminal_index is not None:
                _switch_candidate(
                    runtime,
                    terminal_index,
                    env=env,
                    step=step,
                    dt=dt,
                    reason="crossed_exit_gate",
                    reservation_owner=reservation_owner,
                )
        return

    if waypoint_type != "TERMINAL" and waypoint_distance <= waypoint_tolerance:
        next_index = runtime.candidate_index + 1
        if next_index < len(runtime.candidates):
            _switch_candidate(
                runtime,
                next_index,
                env=env,
                step=step,
                dt=dt,
                reason="candidate_reached",
                reservation_owner=reservation_owner,
            )


def _oracle_owner(
    group: str,
    runtimes: list[AgentRuntime],
    step: int,
) -> int | None:
    crossing = [runtime for runtime in runtimes if _find_candidate_index(runtime, "ENTRY") is not None]
    if not crossing:
        return None
    if group == E0_GROUP:
        runtime = crossing[0]
        return None if np.isfinite(runtime.exit_time) else runtime.route_agent_id
    eligible = [
        runtime
        for runtime in crossing
        if step >= int(runtime.hold_release_step or 0) and not np.isfinite(runtime.exit_time)
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda item: (int(item.hold_release_step or 0), item.route_agent_id)).route_agent_id


def run_execution_episode(
    *,
    model: Any,
    env: Any,
    metadata: Any,
    runtimes: list[AgentRuntime],
    group: str,
    seed: int,
    episode_index: int,
    execution_config: dict[str, Any],
    dt: float,
) -> dict[str, list[dict[str, Any]]]:
    """Execute one episode while changing only finite-set active waypoints."""

    if len(runtimes) != int(env.num_agents):
        raise ValueError("one runtime is required for each environment agent")
    task_goals = np.asarray(env.goals, dtype=float).copy()
    starts = np.asarray(env.starts, dtype=float).copy()
    previous_positions = np.asarray(env._positions(), dtype=float).copy()
    ever_success = np.zeros(int(env.num_agents), dtype=bool)
    goal_steps = np.full(int(env.num_agents), -1, dtype=int)
    trajectories: list[dict[str, Any]] = []
    corridor_events: list[dict[str, Any]] = []
    failure_events: list[tuple[int, str]] = []
    observed_failure_names: set[str] = set()
    occupancy_overlap_steps = 0
    maximum_queue_length = 0
    action_contract_unchanged = True
    terminated = False
    truncated = False
    latest_info: dict[str, Any] = {}

    for runtime in runtimes:
        if _find_candidate_index(runtime, "ENTRY") is not None:
            runtime.request_count = 1
            corridor_events.append(
                {
                    "group": group,
                    "seed": int(seed),
                    "episode_index": int(episode_index),
                    "step": 0,
                    "time": 0.0,
                    "agent_id": runtime.route_agent_id,
                    "event": "REQUEST",
                    "direction": _route_direction(runtime),
                    "reservation_owner": None,
                    "predicted_entry_time": float("nan"),
                    "predicted_exit_time": float("nan"),
                    "actual_entry_time": float("nan"),
                    "actual_exit_time": float("nan"),
                }
            )

    while not (terminated or truncated):
        step_before = int(env.steps)
        owner = _oracle_owner(group, runtimes, step_before)
        active_goals = np.stack(
            [_candidate_position(runtime.candidate) for runtime in runtimes],
            axis=0,
        )
        for runtime in runtimes:
            set_dmp_active_goal_preserve_phase(
                env.dmps[runtime.environment_agent_id],
                active_goals[runtime.environment_agent_id],
            )
        observations = build_active_goal_observations(env, active_goals)
        actions = predict_actions_without_postprocessing(
            model,
            observations,
            expected_shape=env.action_shape,
        ).astype(np.float32, copy=False)
        submitted_actions = actions.copy()
        if not (
            np.all(actions >= env.action_space.low)
            and np.all(actions <= env.action_space.high)
        ):
            raise RuntimeError("deterministic policy returned an out-of-space action")

        _, _, terminated, truncated, latest_info = env.step(actions)
        action_contract_unchanged &= bool(np.array_equal(actions, submitted_actions))
        step = int(env.steps)
        positions = np.asarray(env._positions(), dtype=float)
        velocities = np.asarray(env._velocities(), dtype=float)
        commanded_accelerations = np.asarray(
            latest_info.get(
                "commanded_accelerations",
                np.zeros_like(positions, dtype=np.float32),
            ),
            dtype=float,
        )
        obstacle_collision_mask = np.asarray(
            latest_info.get("obstacle_collision_mask", np.zeros(env.num_agents, dtype=bool)),
            dtype=bool,
        )
        inter_collision_mask = np.asarray(
            latest_info.get("inter_agent_collision_mask", np.zeros(env.num_agents, dtype=bool)),
            dtype=bool,
        )
        out_of_bounds_mask = _out_of_bounds_mask(env)
        success_mask = np.asarray(
            latest_info.get("success_mask", np.zeros(env.num_agents, dtype=bool)),
            dtype=bool,
        )
        newly_reached = np.logical_and(success_mask, goal_steps < 0)
        goal_steps[newly_reached] = step
        ever_success |= success_mask

        step_failure_names: list[str] = []
        if np.any(inter_collision_mask):
            step_failure_names.append("INTER_AGENT_COLLISION")
        if np.any(obstacle_collision_mask):
            step_failure_names.append("OBSTACLE_COLLISION")
        if np.any(out_of_bounds_mask):
            step_failure_names.append("OUT_OF_BOUNDS")
        for failure_name in step_failure_names:
            if failure_name not in observed_failure_names:
                failure_events.append((step, failure_name))
                observed_failure_names.add(failure_name)

        inside_agents: list[int] = []
        previous_entries = [runtime.entry_time for runtime in runtimes]
        previous_exits = [runtime.exit_time for runtime in runtimes]
        for runtime in runtimes:
            env_index = runtime.environment_agent_id
            displacement = float(np.linalg.norm(positions[env_index] - previous_positions[env_index]))
            speed = float(np.linalg.norm(velocities[env_index]))
            acceleration = float(np.linalg.norm(commanded_accelerations[env_index]))
            runtime.path_length += displacement
            runtime.max_speed = max(runtime.max_speed, speed)
            runtime.max_acceleration = max(runtime.max_acceleration, acceleration)
            runtime.min_obstacle_clearance = min(
                runtime.min_obstacle_clearance,
                _minimum_obstacle_clearance(env, env_index),
            )
            runtime.min_peer_clearance = min(
                runtime.min_peer_clearance,
                _minimum_peer_clearance(env, env_index),
            )
            runtime.max_lateral_deviation = max(
                runtime.max_lateral_deviation,
                _lateral_deviation(metadata, positions[env_index]),
            )
            if metadata.inside_conflict_region(positions[env_index]):
                inside_agents.append(runtime.route_agent_id)

            _advance_runtime(
                runtime,
                env=env,
                metadata=metadata,
                step=step,
                dt=dt,
                execution_config=execution_config,
                reservation_owner=owner,
            )
            if goal_steps[env_index] >= 0 and not np.isfinite(runtime.time_to_goal):
                runtime.time_to_goal = float(goal_steps[env_index] * dt)

            candidate_point = _candidate_position(runtime.candidate)
            trajectories.append(
                {
                    "group": group,
                    "seed": int(seed),
                    "episode_index": int(episode_index),
                    "route_agent_id": runtime.route_agent_id,
                    "environment_agent_id": env_index,
                    "step": step,
                    "time": float(step * dt),
                    "x": positions[env_index, 0],
                    "y": positions[env_index, 1],
                    "z": positions[env_index, 2],
                    "vx": velocities[env_index, 0],
                    "vy": velocities[env_index, 1],
                    "vz": velocities[env_index, 2],
                    "speed": speed,
                    "ax": commanded_accelerations[env_index, 0],
                    "ay": commanded_accelerations[env_index, 1],
                    "az": commanded_accelerations[env_index, 2],
                    "active_waypoint_type": runtime.waypoint_type,
                    "active_waypoint_x": candidate_point[0],
                    "active_waypoint_y": candidate_point[1],
                    "active_waypoint_z": candidate_point[2],
                    "upper_state": runtime.upper_state,
                    "reservation_owner": owner,
                    "inter_agent_collision": bool(inter_collision_mask[env_index]),
                    "obstacle_collision": bool(obstacle_collision_mask[env_index]),
                    "out_of_bounds": bool(out_of_bounds_mask[env_index]),
                    "longitudinal_position": float(
                        metadata.longitudinal_coordinate(positions[env_index])
                    ),
                }
            )

        queue_length = sum(runtime.waypoint_type == "HOLD" for runtime in runtimes)
        maximum_queue_length = max(maximum_queue_length, int(queue_length))
        if len(inside_agents) > 1:
            occupancy_overlap_steps += 1
        previous_positions = positions.copy()

        for runtime, prior_entry, prior_exit in zip(runtimes, previous_entries, previous_exits):
            if not np.isfinite(prior_entry) and np.isfinite(runtime.entry_time):
                corridor_events.append(
                    {
                        "group": group,
                        "seed": int(seed),
                        "episode_index": int(episode_index),
                        "step": step,
                        "time": float(step * dt),
                        "agent_id": runtime.route_agent_id,
                        "event": "ACTUAL_ENTRY",
                        "direction": _route_direction(runtime),
                        "reservation_owner": owner,
                        "predicted_entry_time": float("nan"),
                        "predicted_exit_time": float("nan"),
                        "actual_entry_time": runtime.entry_time,
                        "actual_exit_time": float("nan"),
                    }
                )
            if not np.isfinite(prior_exit) and np.isfinite(runtime.exit_time):
                corridor_events.append(
                    {
                        "group": group,
                        "seed": int(seed),
                        "episode_index": int(episode_index),
                        "step": step,
                        "time": float(step * dt),
                        "agent_id": runtime.route_agent_id,
                        "event": "ACTUAL_EXIT",
                        "direction": _route_direction(runtime),
                        "reservation_owner": owner,
                        "predicted_entry_time": float("nan"),
                        "predicted_exit_time": float("nan"),
                        "actual_entry_time": runtime.entry_time,
                        "actual_exit_time": runtime.exit_time,
                    }
                )

    for runtime in runtimes:
        if runtime.hold_timed_out:
            failure_events.append((int(env.steps), "HOLD_TRACKING_FAILURE"))
            observed_failure_names.add("HOLD_TRACKING_FAILURE")
        if runtime.entry_timed_out:
            failure_events.append((int(env.steps), "ENTRY_TRACKING_FAILURE"))
            observed_failure_names.add("ENTRY_TRACKING_FAILURE")
    if bool(truncated):
        timeout_type = "TIMEOUT"
        if any(runtime.hold_timed_out for runtime in runtimes):
            timeout_type = "HOLD_TRACKING_FAILURE"
        elif any(runtime.entry_timed_out for runtime in runtimes):
            timeout_type = "ENTRY_TRACKING_FAILURE"
        failure_events.append((int(env.steps), timeout_type))
        observed_failure_names.add(timeout_type)

    failure_events.sort(key=lambda item: (item[0], item[1]))
    if failure_events:
        first_step = failure_events[0][0]
        first_failure = _first_present_failure(
            name for event_step, name in failure_events if event_step == first_step
        )
    else:
        first_failure = None
    if first_failure is not None and first_failure not in FIRST_FAILURE_TYPES:
        raise RuntimeError(f"unregistered first failure type: {first_failure}")
    secondary_failures = sorted(observed_failure_names - ({first_failure} if first_failure else set()))
    any_inter_collision = bool(
        "INTER_AGENT_COLLISION" in observed_failure_names
    )
    any_obstacle_collision = bool("OBSTACLE_COLLISION" in observed_failure_names)
    any_out_of_bounds = bool("OUT_OF_BOUNDS" in observed_failure_names)
    team_success = bool(
        np.all(ever_success)
        and not any_inter_collision
        and not any_obstacle_collision
        and not any_out_of_bounds
        and not bool(truncated)
    )
    episode_row = {
        "group": group,
        "scenario": E_SCENARIO_NAME,
        "seed": int(seed),
        "episode_index": int(episode_index),
        "team_success": team_success,
        "timeout": bool(truncated),
        "any_inter_agent_collision": any_inter_collision,
        "any_obstacle_collision": any_obstacle_collision,
        "any_out_of_bounds": any_out_of_bounds,
        "first_failure_type": first_failure or "NONE",
        "secondary_failures": secondary_failures,
        "episode_steps": int(env.steps),
        "makespan": float(env.steps * dt),
        "agent_success_rate": float(np.mean(ever_success)),
        "occupancy_overlap_time": float(occupancy_overlap_steps * dt),
        "action_contract_unchanged": bool(action_contract_unchanged),
    }

    agent_rows: list[dict[str, Any]] = []
    for runtime in runtimes:
        env_index = runtime.environment_agent_id
        hold_min_speed = (
            runtime.hold_min_speed
            if np.isfinite(runtime.hold_min_speed)
            else float("nan")
        )
        agent_rows.append(
            {
                "group": group,
                "seed": int(seed),
                "episode_index": int(episode_index),
                "agent_id": runtime.route_agent_id,
                "environment_agent_id": env_index,
                "start": starts[env_index],
                "goal": task_goals[env_index],
                "goal_reached": bool(ever_success[env_index]),
                "time_to_goal": runtime.time_to_goal,
                "path_length": runtime.path_length,
                "waiting_time": float(runtime.waiting_steps * dt),
                "hold_fraction": runtime.waiting_steps / max(1, int(env.steps)),
                "number_of_waypoint_switches": runtime.switches,
                "maximum_speed": runtime.max_speed,
                "maximum_acceleration": runtime.max_acceleration,
                "minimum_obstacle_clearance": runtime.min_obstacle_clearance,
                "minimum_peer_clearance": runtime.min_peer_clearance,
                "maximum_lateral_deviation": runtime.max_lateral_deviation,
                "hold_entry_speed": runtime.first_hold_entry_speed,
                "hold_min_speed": hold_min_speed,
                "hold_exit_speed": runtime.hold_exit_speed,
                "final_waypoint_type": runtime.waypoint_type,
                "hold_tracking_failure": runtime.hold_timed_out,
                "entry_tracking_failure": runtime.entry_timed_out,
            }
        )

    crossing_runtimes = [
        runtime for runtime in runtimes if _find_candidate_index(runtime, "ENTRY") is not None
    ]
    corridor_row = {
        "group": group,
        "seed": int(seed),
        "episode_index": int(episode_index),
        "number_of_requests": int(sum(item.request_count for item in crossing_runtimes)),
        "number_of_authorizations": int(
            sum(item.authorization_count for item in crossing_runtimes)
        ),
        "corridor_entry_times": [item.entry_time for item in crossing_runtimes],
        "corridor_exit_times": [item.exit_time for item in crossing_runtimes],
        "occupancy_durations": [
            item.exit_time - item.entry_time
            if np.isfinite(item.entry_time) and np.isfinite(item.exit_time)
            else float("nan")
            for item in crossing_runtimes
        ],
        "occupancy_overlap_time": float(occupancy_overlap_steps * dt),
        "reservation_violation": False,
        "unauthorized_entry": False,
        "premature_release": False,
        "deadlock": bool(
            truncated and all(item.waypoint_type == "HOLD" for item in crossing_runtimes)
        ),
        "maximum_queue_length": int(maximum_queue_length),
        "mean_waiting_time": float(
            np.mean([item.waiting_steps * dt for item in crossing_runtimes])
        )
        if crossing_runtimes
        else 0.0,
        "prediction_entry_error": float("nan"),
        "prediction_exit_error": float("nan"),
    }
    switch_rows: list[dict[str, Any]] = []
    for runtime in runtimes:
        for row in runtime.waypoint_switch_rows:
            row["group"] = group
            row["seed"] = int(seed)
            row["episode_index"] = int(episode_index)
            switch_rows.append(row)
    return {
        "episodes": [episode_row],
        "agents": agent_rows,
        "corridor_events": corridor_events,
        "waypoint_switches": switch_rows,
        "trajectories": trajectories,
        "corridor_summary": [corridor_row],
    }


def _merge_result_tables(
    destination: dict[str, list[dict[str, Any]]],
    source: dict[str, list[dict[str, Any]]],
) -> None:
    for key, rows in source.items():
        destination.setdefault(key, []).extend(rows)


def _build_candidates(
    metadata: Any,
    *,
    start: np.ndarray,
    goal: np.ndarray,
    static_obstacles: list[Any],
    geometry_config: dict[str, Any],
    workspace_bounds: Any,
) -> list[Any]:
    candidates = corridor.build_route_candidates(
        metadata,
        np.asarray(start, dtype=float),
        np.asarray(goal, dtype=float),
        static_obstacles,
        geometry_config,
        workspace_bounds,
    )
    if not candidates:
        raise RuntimeError("every route must have at least one active-waypoint candidate")
    positions = [_candidate_position(candidate) for candidate in candidates]
    if any(position.shape != (3,) for position in positions):
        raise ValueError("corridor candidates must be three-dimensional")
    unique = {tuple(np.round(position, decimals=10)) for position in positions}
    if len(unique) != len(positions):
        raise ValueError("candidate set contains duplicate coordinates")
    if _waypoint_type(candidates[-1]) != "TERMINAL":
        raise ValueError("the last route candidate must be TERMINAL")
    return candidates


def _build_environment_for_options(
    config: Any,
    *,
    settings: dict[str, Any],
    options: dict[str, Any],
    observation_mode: str,
) -> Any:
    env = _build_environment(
        config,
        observation_mode=observation_mode,
        peer_radius=float(settings["peer_radius"]),
        training_distribution=False,
        include_boundaries_in_sensor=bool(settings["include_boundaries_in_sensor"]),
        terminate_on_boundary_collision=bool(settings["terminate_on_boundary_collision"]),
    )
    env.reset(seed=int(options["seed"]), options=options["scenario_options"])
    return env


def _metadata_payload(metadata: Any) -> dict[str, Any]:
    if hasattr(metadata, "to_dict"):
        return _jsonable(metadata.to_dict())
    if hasattr(metadata, "__dataclass_fields__"):
        return _jsonable(asdict(metadata))
    raise TypeError("CorridorMetadata must be serializable")


def _e_stage() -> dict[str, Any]:
    return next(stage for stage in STAGE_SPECS if stage["name"] == E_SCENARIO_NAME)


def run_step1(
    *,
    model: Any,
    settings: dict[str, Any],
    seed_count: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Run E0 route replay followed by the E1 staggered oracle."""

    max_steps = int(settings["max_steps"])
    dt = float(settings["dt"])
    geometry_config = dict(settings["corridor_geometry"])
    execution_config = dict(settings["waypoint_execution"])
    config_three = build_single_distribution_multi_config(num_agents=3, max_steps=max_steps)
    config_single = build_single_distribution_multi_config(num_agents=1, max_steps=max_steps)
    stage = _e_stage()
    tables: dict[str, list[dict[str, Any]]] = {
        "episodes": [],
        "agents": [],
        "corridor_events": [],
        "waypoint_switches": [],
        "trajectories": [],
        "corridor_summary": [],
        "rollout_predictions": [],
    }
    resolved_metadata: dict[str, Any] | None = None
    episode_index = 0

    for seed_offset in range(int(seed_count)):
        seed = int(settings["seed_base"]) + seed_offset
        original_options = build_stage_scenario(config_three, stage, seed=seed)
        metadata = corridor.infer_corridor_metadata(
            original_options["static_obstacles"],
            config_three.workspace_bounds,
            geometry_config,
        )
        payload = _metadata_payload(metadata)
        if resolved_metadata is None:
            resolved_metadata = payload
        elif payload != resolved_metadata:
            raise RuntimeError("E corridor geometry changed across paired seeds")

        for route_agent_id in range(3):
            single_options = {
                "starts": np.asarray(original_options["starts"][[route_agent_id]], dtype=float),
                "goals": np.asarray(original_options["goals"][[route_agent_id]], dtype=float),
                "static_obstacles": copy.deepcopy(original_options["static_obstacles"]),
                "dynamic_obstacles": copy.deepcopy(original_options["dynamic_obstacles"]),
            }
            env = _build_environment_for_options(
                config_single,
                settings=settings,
                options={
                    "seed": seed,
                    "scenario_options": single_options,
                },
                observation_mode="blind",
            )
            try:
                candidates = _build_candidates(
                    metadata,
                    start=single_options["starts"][0],
                    goal=single_options["goals"][0],
                    static_obstacles=single_options["static_obstacles"],
                    geometry_config=geometry_config,
                    workspace_bounds=config_single.workspace_bounds,
                )
                runtime = AgentRuntime(
                    route_agent_id=route_agent_id,
                    environment_agent_id=0,
                    candidates=candidates,
                )
                initial_type = (
                    "HOLD"
                    if _find_candidate_index(runtime, "ENTRY") is not None
                    else "TERMINAL"
                )
                _initialize_runtime_waypoint(
                    runtime,
                    env=env,
                    group=E0_GROUP,
                    seed=seed,
                    episode_index=episode_index,
                    initial_type=initial_type,
                    dt=dt,
                )
                result = run_execution_episode(
                    model=model,
                    env=env,
                    metadata=metadata,
                    runtimes=[runtime],
                    group=E0_GROUP,
                    seed=seed,
                    episode_index=episode_index,
                    execution_config=execution_config,
                    dt=dt,
                )
                _merge_result_tables(tables, result)
            finally:
                env.close()
            episode_index += 1

        stagger_steps = [
            int(value) for value in execution_config["stagger_release_steps"]
        ]
        if len(stagger_steps) != 3:
            raise ValueError("stagger_release_steps must contain one value per agent")
        multi_options = {
            "starts": np.asarray(original_options["starts"], dtype=float).copy(),
            "goals": np.asarray(original_options["goals"], dtype=float).copy(),
            "static_obstacles": copy.deepcopy(original_options["static_obstacles"]),
            "dynamic_obstacles": copy.deepcopy(original_options["dynamic_obstacles"]),
        }
        env = _build_environment_for_options(
            config_three,
            settings=settings,
            options={"seed": seed, "scenario_options": multi_options},
            observation_mode=str(settings["observation_mode"]),
        )
        try:
            runtimes: list[AgentRuntime] = []
            for agent_index in range(3):
                candidates = _build_candidates(
                    metadata,
                    start=multi_options["starts"][agent_index],
                    goal=multi_options["goals"][agent_index],
                    static_obstacles=multi_options["static_obstacles"],
                    geometry_config=geometry_config,
                    workspace_bounds=config_three.workspace_bounds,
                )
                runtime = AgentRuntime(
                    route_agent_id=agent_index,
                    environment_agent_id=agent_index,
                    candidates=candidates,
                    hold_release_step=stagger_steps[agent_index],
                )
                crossing_route = _find_candidate_index(runtime, "ENTRY") is not None
                initial_type = None
                if crossing_route:
                    initial_type = "HOLD" if stagger_steps[agent_index] > 0 else "ENTRY"
                    if initial_type == "ENTRY":
                        runtime.authorization_count = 1
                else:
                    initial_type = "TERMINAL"
                _initialize_runtime_waypoint(
                    runtime,
                    env=env,
                    group=E1_GROUP,
                    seed=seed,
                    episode_index=episode_index,
                    initial_type=initial_type,
                    dt=dt,
                )
                runtimes.append(runtime)
            result = run_execution_episode(
                model=model,
                env=env,
                metadata=metadata,
                runtimes=runtimes,
                group=E1_GROUP,
                seed=seed,
                episode_index=episode_index,
                execution_config=execution_config,
                dt=dt,
            )
            _merge_result_tables(tables, result)
        finally:
            env.close()
        episode_index += 1

    if resolved_metadata is None:
        raise RuntimeError("no E scenario metadata was generated")
    return tables, resolved_metadata


def aggregate_tables(tables: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = sorted({str(row["group"]) for row in tables["episodes"]})
    for group in groups:
        episodes = [row for row in tables["episodes"] if row["group"] == group]
        agents = [row for row in tables["agents"] if row["group"] == group]
        corridor_rows = [
            row for row in tables["corridor_summary"] if row["group"] == group
        ]
        rows.append(
            {
                "group": group,
                "episode_count": len(episodes),
                "agent_count": len(agents),
                "team_success_rate": float(np.mean([row["team_success"] for row in episodes])),
                "agent_success_rate": float(np.mean([row["goal_reached"] for row in agents])),
                "inter_agent_collision_rate": float(
                    np.mean([row["any_inter_agent_collision"] for row in episodes])
                ),
                "obstacle_collision_rate": float(
                    np.mean([row["any_obstacle_collision"] for row in episodes])
                ),
                "out_of_bounds_rate": float(
                    np.mean([row["any_out_of_bounds"] for row in episodes])
                ),
                "timeout_rate": float(np.mean([row["timeout"] for row in episodes])),
                "hold_tracking_failure_rate": float(
                    np.mean([row["hold_tracking_failure"] for row in agents])
                ),
                "entry_tracking_failure_rate": float(
                    np.mean([row["entry_tracking_failure"] for row in agents])
                ),
                "mean_path_length": float(np.mean([row["path_length"] for row in agents])),
                "mean_waiting_time": float(np.mean([row["waiting_time"] for row in agents])),
                "mean_occupancy_overlap_time": float(
                    np.mean([row["occupancy_overlap_time"] for row in corridor_rows])
                ),
                "action_contract_pass_rate": float(
                    np.mean([row["action_contract_unchanged"] for row in episodes])
                ),
            }
        )
    return rows


def decide_step1(
    summary: list[dict[str, Any]],
    gates: dict[str, Any],
) -> tuple[str, list[str]]:
    by_group = {str(row["group"]): row for row in summary}
    e0 = by_group[E0_GROUP]
    e1 = by_group[E1_GROUP]
    reasons: list[str] = []
    if float(e0["agent_success_rate"]) < float(gates["e0_min_agent_success_rate"]):
        reasons.append(
            "E0 agent-level success rate below "
            f"{float(gates['e0_min_agent_success_rate']):.0%}"
        )
    if float(e0["hold_tracking_failure_rate"]) > 0.0:
        reasons.append("E0 contains hold waypoint tracking failures")
    if float(e0["obstacle_collision_rate"]) > 0.0:
        reasons.append("E0 contains entry/exit or wall collisions")
    if float(e0["out_of_bounds_rate"]) > 0.0:
        reasons.append(
            f"E0 contains out-of-bounds trajectories ({float(e0['out_of_bounds_rate']):.1%})"
        )
    if float(e1["team_success_rate"]) < float(gates["e1_min_team_success_rate"]):
        reasons.append(
            "E1 team success rate below "
            f"{float(gates['e1_min_team_success_rate']):.0%}"
        )
    if float(e1["out_of_bounds_rate"]) > 0.0:
        reasons.append(
            f"E1 contains out-of-bounds trajectories ({float(e1['out_of_bounds_rate']):.1%})"
        )
    return ("NO_GO" if reasons else "STEP1_GO", reasons)


def _failure_counts(episodes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in episodes:
        name = str(row["first_failure_type"])
        counts[name] = counts.get(name, 0) + 1
    return counts


def _format_rate(value: Any) -> str:
    return f"{float(value):.1%}"


def _write_group_report(
    path: Path,
    *,
    group: str,
    summary: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> None:
    failure_counts = _failure_counts(episodes)
    lines = [
        f"# {group}",
        "",
        "本组仅通过有限候选 active waypoint 控制冻结 SAC-DMP；策略 action 未执行任何后处理。",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| Episodes | {summary['episode_count']} |",
        f"| Team success | {_format_rate(summary['team_success_rate'])} |",
        f"| Agent success | {_format_rate(summary['agent_success_rate'])} |",
        f"| Inter-agent collision | {_format_rate(summary['inter_agent_collision_rate'])} |",
        f"| Obstacle collision | {_format_rate(summary['obstacle_collision_rate'])} |",
        f"| Out of bounds | {_format_rate(summary['out_of_bounds_rate'])} |",
        f"| Timeout | {_format_rate(summary['timeout_rate'])} |",
        f"| Hold tracking failure | {_format_rate(summary['hold_tracking_failure_rate'])} |",
        "",
        "## 第一失败原因",
        "",
    ]
    for name, count in sorted(failure_counts.items()):
        lines.append(f"- `{name}`：{count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_report(
    path: Path,
    *,
    summary: list[dict[str, Any]],
    decision: str,
    reasons: list[str],
    checkpoint: Path,
    checkpoint_sha256: str,
    actor_unchanged: bool,
    tables: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# 冻结单机 SAC-DMP 走廊协调最小可行性实验",
        "",
        f"- Checkpoint：`{checkpoint}`",
        f"- SHA256：`{checkpoint_sha256}`",
        f"- Actor 参数保持不变：`{actor_unchanged}`",
        "- 策略 action 原样进入现有环境/DMP；无 projection、blending、residual 或人工制动。",
        "- 本报告当前覆盖 Step 1：E0 单机路线重放与 E1 三机错峰 oracle。",
        "",
        "## 10-seed smoke test",
        "",
        "| 方案 | Episodes | Team success | Agent success | Inter collision | Obstacle collision | OOB | Timeout | Hold failure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {group} | {episodes} | {team} | {agent} | {inter} | {obstacle} | {oob} | {timeout} | {hold} |".format(
                group=row["group"],
                episodes=row["episode_count"],
                team=_format_rate(row["team_success_rate"]),
                agent=_format_rate(row["agent_success_rate"]),
                inter=_format_rate(row["inter_agent_collision_rate"]),
                obstacle=_format_rate(row["obstacle_collision_rate"]),
                oob=_format_rate(row["out_of_bounds_rate"]),
                timeout=_format_rate(row["timeout_rate"]),
                hold=_format_rate(row["hold_tracking_failure_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## 路线级结果",
            "",
            "| 方案 | Agent | 成功数/样本数 | HOLD failure | 平均路径/m | 平均等待/s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    route_keys = sorted(
        {(str(row["group"]), int(row["agent_id"])) for row in tables["agents"]}
    )
    for group, agent_id in route_keys:
        agent_rows = [
            row
            for row in tables["agents"]
            if row["group"] == group and int(row["agent_id"]) == agent_id
        ]
        lines.append(
            "| {group} | {agent} | {success}/{count} | {hold}/{count} | {path:.3f} | {wait:.3f} |".format(
                group=group,
                agent=agent_id,
                success=sum(bool(row["goal_reached"]) for row in agent_rows),
                hold=sum(bool(row["hold_tracking_failure"]) for row in agent_rows),
                count=len(agent_rows),
                path=float(np.mean([row["path_length"] for row in agent_rows])),
                wait=float(np.mean([row["waiting_time"] for row in agent_rows])),
            )
        )
    lines.extend(["", "## 第一失败原因", ""])
    for group in (E0_GROUP, E1_GROUP):
        episodes = [row for row in tables["episodes"] if row["group"] == group]
        counts = _failure_counts(episodes)
        lines.append(
            f"- `{group}`："
            + "，".join(f"{name}={count}" for name, count in sorted(counts.items()))
        )
    lines.extend(["", "## Go / No-Go", "", f"结论：**{decision}**。", ""])
    if reasons:
        lines.append("触发原因：")
        lines.append("")
        lines.extend(f"- {reason}" for reason in reasons)
    elif decision == "STEP1_GO":
        lines.append("E0/E1 执行门槛通过，可以进入 E2/E3 规则 reservation 验证。")
    lines.extend(
        [
            "",
            "## 当前开发门控",
            "",
            (
                "由于 Step 1 触发 No-Go，按实验协议停止实现完整 E3/E4 和 PCO-GAT。"
                if decision == "NO_GO"
                else "Step 1 已通过；下一步实现 E2/E3，E4 仍需等待 E3 晋级。"
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_artifacts(
    output_dir: Path,
    *,
    tables: dict[str, list[dict[str, Any]]],
    summary: list[dict[str, Any]],
    config_payload: dict[str, Any],
    decision: str,
    reasons: list[str],
    checkpoint: Path,
    checkpoint_sha256: str,
    actor_unchanged: bool,
) -> None:
    group_dirs = {
        E0_GROUP: output_dir / "single_agent",
        E1_GROUP: output_dir / "staggered",
    }
    summary_by_group = {row["group"]: row for row in summary}
    for group, group_dir in group_dirs.items():
        group_dir.mkdir(parents=True, exist_ok=False)
        group_tables = {
            key: [row for row in rows if row.get("group") == group]
            for key, rows in tables.items()
        }
        _write_csv(group_dir / "summary.csv", [summary_by_group[group]])
        _write_csv(group_dir / "episodes.csv", group_tables["episodes"])
        _write_csv(group_dir / "agents.csv", group_tables["agents"])
        _write_csv(group_dir / "corridor_events.csv", group_tables["corridor_events"])
        _write_csv(group_dir / "waypoint_switches.csv", group_tables["waypoint_switches"])
        _write_csv(
            group_dir / "rollout_predictions.csv",
            group_tables["rollout_predictions"],
            fieldnames=ROLLOUT_PREDICTION_FIELDS,
        )
        _write_csv(group_dir / "trajectories.csv", group_tables["trajectories"])
        _write_json(group_dir / "config.json", {**config_payload, "group": group})
        _write_group_report(
            group_dir / "report.md",
            group=group,
            summary=summary_by_group[group],
            episodes=group_tables["episodes"],
        )

    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(comparison_dir / "summary.csv", summary)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "episodes.csv", tables["episodes"])
    _write_csv(output_dir / "agents.csv", tables["agents"])
    _write_csv(output_dir / "corridor_events.csv", tables["corridor_events"])
    _write_csv(output_dir / "waypoint_switches.csv", tables["waypoint_switches"])
    _write_csv(output_dir / "trajectories.csv", tables["trajectories"])
    _write_csv(
        output_dir / "rollout_predictions.csv",
        tables["rollout_predictions"],
        fieldnames=ROLLOUT_PREDICTION_FIELDS,
    )
    _write_json(output_dir / "config.json", config_payload)
    _write_json(
        output_dir / "decision.json",
        {"decision": decision, "reasons": reasons},
    )
    _write_final_report(
        output_dir / "report.md",
        summary=summary,
        decision=decision,
        reasons=reasons,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        actor_unchanged=actor_unchanged,
        tables=tables,
    )
    (comparison_dir / "report.md").write_text(
        (output_dir / "report.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--skip-visualization", action="store_true")
    return parser.parse_args()


def main() -> Path:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = _resolve_path(settings["checkpoint"]).resolve()
    expected_sha256 = str(settings["checkpoint_sha256"]).lower()
    actual_sha256 = _sha256(checkpoint)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"checkpoint SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    seed_count = int(args.seed_count or settings["smoke_seed_count"])
    if seed_count <= 0:
        raise ValueError("seed count must be positive")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _resolve_path(settings["artifact_dir"]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    reference_env = build_single_env(
        config=SINGLE_AGENT_CONFIG,
        action_guidance_enabled=False,
    )
    model = build_single_model(reference_env, config=SINGLE_AGENT_CONFIG, verbose=0)
    load_checkpoint(model, checkpoint)
    model.actor.train(False)
    before_actor = _actor_snapshot(model)
    try:
        tables, resolved_metadata = run_step1(
            model=model,
            settings=settings,
            seed_count=seed_count,
        )
    finally:
        reference_env.close()
    actor_unchanged = _actor_unchanged(before_actor, model)
    if not actor_unchanged:
        raise RuntimeError("frozen Actor parameters changed during evaluation")
    if not all(row["action_contract_unchanged"] for row in tables["episodes"]):
        raise RuntimeError("at least one episode mutated the submitted policy action")

    summary = aggregate_tables(tables)
    decision, reasons = decide_step1(summary, dict(settings["gates"]))
    launch_command = " ".join([str(Path(sys.executable).resolve()), *sys.argv])
    config_payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "evaluation_config": str(config_path),
        "launch_command": launch_command,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha256,
        "actor_parameters_unchanged": actor_unchanged,
        "python_version": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "git": _git_metadata(),
        "settings": settings,
        "resolved_corridor_metadata": resolved_metadata,
        "frozen_policy_contract": {
            "deterministic": True,
            "network_action_passed_to_env_step_unchanged": True,
            "action_projection": False,
            "action_residual": False,
            "action_blending": False,
            "forced_braking": False,
            "manual_goal_offset": False,
            "terminal_goal_forcing_gate_unchanged": True,
        },
        "decision": decision,
        "decision_reasons": reasons,
    }
    _write_artifacts(
        output_dir,
        tables=tables,
        summary=summary,
        config_payload=config_payload,
        decision=decision,
        reasons=reasons,
        checkpoint=checkpoint,
        checkpoint_sha256=actual_sha256,
        actor_unchanged=actor_unchanged,
    )

    if not args.skip_visualization:
        visualizer = SCRIPTS_ROOT / "visualize_frozen_policy_corridor_reservation.py"
        if visualizer.is_file():
            subprocess.run(
                [sys.executable, str(visualizer), "--artifact-dir", str(output_dir)],
                cwd=REPO_ROOT,
                check=False,
            )
    print(json.dumps({"output_dir": str(output_dir), "decision": decision}, ensure_ascii=False))
    return output_dir


if __name__ == "__main__":
    main()
