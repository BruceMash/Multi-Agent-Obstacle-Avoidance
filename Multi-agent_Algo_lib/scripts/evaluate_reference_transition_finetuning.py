"""Evaluation utilities for targeted reference-transition SAC fine-tuning."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiment_config import EXPERIMENT_CONFIG  # noqa: E402
from planning.horizon_agent_setting_diagnosis import stable_hash  # noqa: E402
from planning.reference_transition_finetuning import (  # noqa: E402
    TASK_REFERENCE,
    TASK_TERMINAL,
    build_reference_transition_env,
    load_checkpoint_weights_only,
    sha256_file,
    smoke_training_config,
    state_dict_sha256,
)
from runner_sac import build_model  # noqa: E402


DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "training" / "reference_transition_finetuning_smoke.json"
)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    encoded = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not encoded:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in encoded:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in encoded:
            writer.writerow(
                {
                    key: json.dumps(jsonable(value), ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict, np.ndarray))
                    else value
                    for key, value in row.items()
                }
            )


def _scenario_options(config: Any, scene: str, seed: int) -> dict[str, Any]:
    """Construct deterministic single-agent evaluation scenes."""

    rng = np.random.default_rng(int(seed))
    start, goal = config.build_start_goal_generator()(rng)
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    options: dict[str, Any] = {
        "start": start,
        "goal": goal,
        "static_obstacles": [],
        "dynamic_obstacles": [],
    }
    if scene == "open":
        return options
    if scene not in {"sparse_static", "historical_base", "historical_fixed_base"}:
        raise ValueError(f"unknown evaluation scene: {scene}")

    base = replace(
        config,
        training_scene_mixture_enabled=False,
        curriculum_enabled=False,
    )
    fixed_box = base.build_fixed_box()
    static_generator = base.build_static_obstacle_generator(fixed_box)
    static = static_generator(start.copy(), goal.copy(), int(seed) + 100_003)
    options["static_obstacles"] = copy.deepcopy(static)
    if scene in {"historical_base", "historical_fixed_base"}:
        dynamic_generator = base.build_dynamic_obstacle_generator()
        dynamic = dynamic_generator(
            start.copy(),
            goal.copy(),
            int(seed) + 200_003,
            copy.deepcopy(static),
        )
        options["dynamic_obstacles"] = copy.deepcopy(dynamic)
    return options


def _smoothness(accelerations: list[np.ndarray]) -> float:
    if len(accelerations) < 2:
        return 0.0
    values = np.asarray(accelerations, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(values, axis=0), axis=1)))


def _action_saturation(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    action = np.asarray(action, dtype=float)
    scale = np.maximum(np.asarray(high, dtype=float) - np.asarray(low, dtype=float), 1e-8)
    tolerance = 0.01 * scale
    saturated = (action <= np.asarray(low) + tolerance) | (
        action >= np.asarray(high) - tolerance
    )
    return float(np.mean(saturated))


def _single_evaluation_state_hash(env: Any, observation: np.ndarray) -> str:
    """Hash all controller-relevant single-agent state without mutating it."""

    packet = env.latest_sensor_packet
    if packet is None:
        raise RuntimeError("single-agent sensor packet is unavailable")
    return stable_hash(
        {
            "steps": int(env.steps),
            "position": env.dynamics.p,
            "velocity": env.dynamics.v,
            "phase": float(env.dmp.phase),
            "terminal_goal": env.terminal_goal,
            "active_goal": env.active_goal,
            "reference_reached": bool(env.reference_reached),
            "reference_handoff_count": int(env.reference_handoff_count),
            "current_scan": packet.current_scan,
            "previous_scan": packet.previous_scan,
            "actor_observation": np.asarray(observation, dtype=np.float32),
        }
    )


def _single_physical_state_hash(env: Any) -> str:
    """Hash the trajectory-prefix state, excluding post-step handoff bookkeeping."""

    packet = env.latest_sensor_packet
    if packet is None:
        raise RuntimeError("single-agent sensor packet is unavailable")
    return stable_hash(
        {
            "steps": int(env.steps),
            "position": env.dynamics.p,
            "velocity": env.dynamics.v,
            "phase": float(env.dmp.phase),
            "current_scan": packet.current_scan,
            "previous_scan": packet.previous_scan,
        }
    )


def run_evaluation_episode(
    *,
    model: Any,
    config: Any,
    evaluation_step: int,
    evaluation_kind: str,
    scene: str,
    seed: int,
    task: str,
    proposal_top_k: int,
    reference_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_config = (
        replace(
            config,
            training_scene_mixture_enabled=False,
            curriculum_enabled=False,
        )
        if scene == "historical_fixed_base"
        else config
    )
    env = build_reference_transition_env(
        episode_config,
        forced_task=task,
        proposal_top_k=proposal_top_k,
        reference_tolerance=reference_tolerance,
    )
    step_records: list[dict[str, Any]] = []
    handoff_rows: list[dict[str, Any]] = []
    try:
        options = (
            None
            if scene == "historical_fixed_base"
            else _scenario_options(episode_config, scene, int(seed))
        )
        observation, reset_info = env.reset(seed=int(seed), options=options)
        start = env.dynamics.p.copy()
        terminal_goal = env.terminal_goal.copy()
        initial_condition_hash = _single_evaluation_state_hash(env, observation)
        temporary_reference_hash = stable_hash(
            {
                "available": bool(not env.reference_generation_fallback),
                "temporary_reference": env.temporary_reference,
            }
        )
        initial_terminal_distance = float(np.linalg.norm(terminal_goal - start))
        episode_return = 0.0
        accelerations: list[np.ndarray] = []
        positions: list[np.ndarray] = [start.copy()]
        saturation_values: list[float] = []
        terminated = False
        truncated = False
        last_info = reset_info
        state_at_step_50_hash: str | None = None

        while not (terminated or truncated):
            active_before = env.active_goal.copy()
            position_before = env.dynamics.p.copy()
            velocity_before = env.dynamics.v.copy()
            phase_before = float(env.dmp.phase)
            action, _ = model.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            acceleration = np.asarray(info["commanded_acceleration"], dtype=float)
            accelerations.append(acceleration.copy())
            positions.append(env.dynamics.p.copy())
            saturation = _action_saturation(action, env.action_space.low, env.action_space.high)
            saturation_values.append(saturation)
            step_records.append(
                {
                    "evaluation_step": int(evaluation_step),
                    "evaluation_kind": evaluation_kind,
                    "scene": scene,
                    "seed": int(seed),
                    "episode_step": int(env.steps),
                    "active_goal_before": active_before.copy(),
                    "active_goal_after": env.active_goal.copy(),
                    "terminal_goal": terminal_goal.copy(),
                    "position_before": position_before,
                    "position_after": env.dynamics.p.copy(),
                    "velocity_before": velocity_before,
                    "velocity_after": env.dynamics.v.copy(),
                    "phase_before": phase_before,
                    "phase_after": float(env.dmp.phase),
                    "forcing_xyz": action[:3].copy(),
                    "goal_offset_xyz": action[3:].copy(),
                    "commanded_acceleration": acceleration.copy(),
                    "distance_to_active_before": float(
                        np.linalg.norm(active_before - position_before)
                    ),
                    "distance_to_active_after": float(
                        np.linalg.norm(active_before - env.dynamics.p)
                    ),
                    "distance_to_terminal_after": float(
                        np.linalg.norm(terminal_goal - env.dynamics.p)
                    ),
                    "reward": float(reward),
                    "progress_reward": float(info["reward_progress"]),
                    "action_saturation": saturation,
                    "handoff_event": bool(info["reference_handoff_event"]),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            last_info = info
            if int(env.steps) == 50:
                state_at_step_50_hash = _single_physical_state_hash(env)

        handoff_indices = [
            index for index, row in enumerate(step_records) if bool(row["handoff_event"])
        ]
        for handoff_index in handoff_indices:
            first = max(0, handoff_index - 3)
            last = min(len(step_records), handoff_index + 6)
            for index in range(first, last):
                row = dict(step_records[index])
                row["handoff_episode_step"] = int(step_records[handoff_index]["episode_step"])
                row["relative_step"] = int(index - handoff_index)
                handoff_rows.append(row)

        terminal_distance = float(np.linalg.norm(terminal_goal - env.dynamics.p))
        reference_available = bool(
            task == TASK_REFERENCE and not env.reference_generation_fallback
        )
        reference_reached_step = (
            int(env.reference_reached_step)
            if env.reference_reached_step is not None
            else None
        )
        terminal_success = bool(last_info["success"])
        terminal_completion_step = int(env.steps) if terminal_success else None
        position_array = np.stack(positions)
        path_length = float(
            np.sum(np.linalg.norm(np.diff(position_array, axis=0), axis=1))
        )
        final_state_hash = _single_physical_state_hash(env)
        reference_reached_steps = (
            [reference_reached_step] if reference_reached_step is not None else []
        )
        terminal_completion_steps = (
            [terminal_completion_step] if terminal_completion_step is not None else []
        )
        remaining_steps_after_reference = (
            [int(env.env_config.max_steps) - reference_reached_step]
            if reference_reached_step is not None
            else []
        )
        row = {
            "evaluation_step": int(evaluation_step),
            "evaluation_kind": evaluation_kind,
            "scene": scene,
            "seed": int(seed),
            "requested_task": env.requested_task,
            "episode_task": env.episode_task,
            "reference_generation_fallback": bool(env.reference_generation_fallback),
            "reference_available": reference_available,
            "temporary_reference": (
                None if env.temporary_reference is None else env.temporary_reference.copy()
            ),
            "temporary_reference_reached": bool(env.reference_reached),
            "reference_reached_step": reference_reached_step,
            "reference_reached_steps": reference_reached_steps,
            "reference_handoff_count": int(env.reference_handoff_count),
            "reached_then_terminal_success": bool(
                reference_available and env.reference_reached and last_info["success"]
            ),
            "terminal_success": terminal_success,
            "terminal_completion_step": terminal_completion_step,
            "terminal_completion_steps": terminal_completion_steps,
            "remaining_steps_after_reference": remaining_steps_after_reference,
            "collision": bool(last_info["collision"]),
            "timeout": bool(truncated),
            "termination_reason": (
                "collision"
                if bool(last_info["collision"])
                else "terminal_success"
                if bool(last_info["success"])
                else "timeout"
            ),
            "episode_length": int(env.steps),
            "episode_return": float(episode_return),
            "initial_terminal_distance_m": initial_terminal_distance,
            "final_terminal_distance_m": terminal_distance,
            "terminal_progress_m": initial_terminal_distance - terminal_distance,
            "path_length_m": path_length,
            "trajectory_smoothness": _smoothness(accelerations),
            "mean_action_saturation": float(np.mean(saturation_values)),
            "observation_dimension": int(observation.shape[0]),
            "forcing_gate_semantics": last_info["forcing_gate_semantics"],
            "terminal_goal_unchanged": bool(np.array_equal(env.goal, terminal_goal)),
            "GAT_used": False,
            "FP_SHEP_selector_used": False,
            "repeated_waypoint_used": False,
            "hard_boundary_used": False,
            "initial_condition_hash": initial_condition_hash,
            "temporary_reference_hash": temporary_reference_hash,
            "state_at_step_50_hash": state_at_step_50_hash,
            "final_state_hash": final_state_hash,
            "reference_generation_count": int(reference_available),
            "collision_before_reference_reached_count": int(
                bool(last_info["collision"]) and not bool(env.reference_reached)
            ),
            "collision_after_reference_reached_count": int(
                bool(last_info["collision"]) and bool(env.reference_reached)
            ),
            "timeout_after_reference_reached": bool(
                truncated and bool(env.reference_reached)
            ),
        }
        return row, handoff_rows
    finally:
        env.close()


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(bool(row[key])) for row in rows])) if rows else 0.0


def aggregate_retention(rows: list[dict[str, Any]], evaluation_step: int) -> dict[str, Any]:
    return {
        "evaluation_step": int(evaluation_step),
        "episode_count": len(rows),
        "terminal_success_count": sum(bool(row["terminal_success"]) for row in rows),
        "terminal_success_total": len(rows),
        "terminal_success_rate": _rate(rows, "terminal_success"),
        "collision_count": sum(bool(row["collision"]) for row in rows),
        "collision_total": len(rows),
        "collision_rate": _rate(rows, "collision"),
        "timeout_count": sum(bool(row["timeout"]) for row in rows),
        "timeout_total": len(rows),
        "timeout_rate": _rate(rows, "timeout"),
        "mean_episode_length": float(np.mean([row["episode_length"] for row in rows])),
        "mean_return": float(np.mean([row["episode_return"] for row in rows])),
    }


def aggregate_adaptation(rows: list[dict[str, Any]], evaluation_step: int) -> dict[str, Any]:
    valid = [row for row in rows if bool(row["reference_available"])]
    reached = [row for row in valid if bool(row["temporary_reference_reached"])]
    return {
        "evaluation_step": int(evaluation_step),
        "episode_count": len(rows),
        "reference_valid_count": len(valid),
        "reference_fallback_count": len(rows) - len(valid),
        "temporary_reference_reached_count": sum(
            bool(row["temporary_reference_reached"]) for row in valid
        ),
        "temporary_reference_reached_total": len(valid),
        "temporary_reference_reached_rate": _rate(valid, "temporary_reference_reached"),
        "reached_then_terminal_success_count": sum(
            bool(row["reached_then_terminal_success"]) for row in reached
        ),
        "reached_then_terminal_success_total": len(reached),
        "reached_then_terminal_success_rate": _rate(
            reached, "reached_then_terminal_success"
        ),
        "overall_terminal_success_count": sum(bool(row["terminal_success"]) for row in rows),
        "overall_terminal_success_total": len(rows),
        "overall_terminal_success_rate": _rate(rows, "terminal_success"),
        "collision_count": sum(bool(row["collision"]) for row in rows),
        "collision_total": len(rows),
        "collision_rate": _rate(rows, "collision"),
        "timeout_count": sum(bool(row["timeout"]) for row in rows),
        "timeout_total": len(rows),
        "timeout_rate": _rate(rows, "timeout"),
        "mean_terminal_progress_m": float(
            np.mean([row["terminal_progress_m"] for row in rows])
        ),
        "mean_trajectory_smoothness": float(
            np.mean([row["trajectory_smoothness"] for row in rows])
        ),
    }


def evaluate_model(
    *,
    model: Any,
    config: Any,
    settings: Mapping[str, Any],
    evaluation_step: int,
) -> dict[str, Any]:
    actor_before = state_dict_sha256(model.actor.state_dict())
    critic_before = state_dict_sha256(model.critic.state_dict())
    replay_before = int(model.replay_buffer.size())
    retention_rows: list[dict[str, Any]] = []
    adaptation_rows: list[dict[str, Any]] = []
    handoff_rows: list[dict[str, Any]] = []

    retention = settings["retention_evaluation"]
    for seed in retention["seeds"]:
        row, windows = run_evaluation_episode(
            model=model,
            config=config,
            evaluation_step=int(evaluation_step),
            evaluation_kind="retention",
            scene=str(retention["scene"]),
            seed=int(seed),
            task=TASK_TERMINAL,
            proposal_top_k=int(settings["proposal"]["consumer_top_k"]),
            reference_tolerance=float(settings["episode"]["temporary_reference_tolerance_m"]),
        )
        retention_rows.append(row)
        handoff_rows.extend(windows)

    adaptation = settings["adaptation_evaluation"]
    for scene in adaptation["scenes"]:
        for seed in adaptation["seeds"]:
            row, windows = run_evaluation_episode(
                model=model,
                config=config,
                evaluation_step=int(evaluation_step),
                evaluation_kind="adaptation",
                scene=str(scene),
                seed=int(seed),
                task=TASK_REFERENCE,
                proposal_top_k=int(settings["proposal"]["consumer_top_k"]),
                reference_tolerance=float(
                    settings["episode"]["temporary_reference_tolerance_m"]
                ),
            )
            adaptation_rows.append(row)
            handoff_rows.extend(windows)

    integrity = {
        "actor_unchanged": actor_before == state_dict_sha256(model.actor.state_dict()),
        "critic_unchanged": critic_before == state_dict_sha256(model.critic.state_dict()),
        "replay_size_before": replay_before,
        "replay_size_after": int(model.replay_buffer.size()),
    }
    integrity["replay_unchanged"] = (
        integrity["replay_size_before"] == integrity["replay_size_after"]
    )
    if not all(
        integrity[key]
        for key in ("actor_unchanged", "critic_unchanged", "replay_unchanged")
    ):
        raise RuntimeError("evaluation modified model parameters or replay buffer")
    return {
        "retention_rows": retention_rows,
        "adaptation_rows": adaptation_rows,
        "handoff_rows": handoff_rows,
        "retention_summary": aggregate_retention(retention_rows, int(evaluation_step)),
        "adaptation_summary": aggregate_adaptation(adaptation_rows, int(evaluation_step)),
        "integrity": integrity,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0)
    return parser.parse_args()


def main() -> Path:
    args = _parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    config = smoke_training_config(EXPERIMENT_CONFIG)
    checkpoint = args.checkpoint or (REPO_ROOT / settings["checkpoint"])
    env = build_reference_transition_env(config, forced_task=TASK_TERMINAL)
    try:
        model = build_model(env, config=config, tensorboard_log=None, verbose=0)
        audit = load_checkpoint_weights_only(
            model,
            checkpoint,
            learning_rate=float(settings["learning"]["actor_lr"]),
        )
        result = evaluate_model(
            model=model,
            config=config,
            settings=settings,
            evaluation_step=int(args.step),
        )
    finally:
        env.close()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "warm_start_audit.json", audit)
    write_csv(args.output / "retention_eval.csv", result["retention_rows"])
    write_csv(args.output / "adaptation_eval.csv", result["adaptation_rows"])
    write_csv(args.output / "handoff_diagnostics.csv", result["handoff_rows"])
    write_json(args.output / "evaluation_summary.json", result)
    if sha256_file(checkpoint) != audit.checkpoint_sha256_before:
        raise RuntimeError("source checkpoint hash changed during evaluation")
    return args.output


if __name__ == "__main__":
    print(main())
