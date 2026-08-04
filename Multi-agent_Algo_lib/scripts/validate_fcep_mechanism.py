from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = PROJECT_ROOT / "Multi-agent_Algo_lib"
for path in (PROJECT_ROOT, ALGO_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from Entity.dynamic_obstacles import MovingSphereObstacle
from Entity.static_obstacles import AxisAlignedBoxObstacle, StaticSphereObstacle
from Environment.multi_agent_dmp_env import MultiAgentDMPEnv
from validate_masac_multi_agent_scenarios import (
    _find_run_config,
    _load_configs,
    _load_policy,
    _resolve_checkpoint,
    _resolve_device,
)


SCENARIO_NAMES = (
    "open_flight",
    "static_detour",
    "dynamic_crossing",
    "two_agent_crossing",
    "narrow_yielding",
)


def matrix_to_agent_dict(values: np.ndarray, agent_ids: list[str]) -> dict[str, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    return {
        agent_id: array[index].astype(np.float32, copy=True)
        for index, agent_id in enumerate(agent_ids)
    }


def agent_dict_to_matrix(values: dict[str, np.ndarray], agent_ids: list[str]) -> np.ndarray:
    return np.stack(
        [np.asarray(values[agent_id], dtype=np.float32) for agent_id in agent_ids],
        axis=0,
    )


class AgentObservationHistory:
    def __init__(self, temporal_steps: int, agent_ids: list[str]):
        self.temporal_steps = int(temporal_steps)
        self.agent_ids = tuple(agent_ids)
        self._histories = {
            agent_id: deque(maxlen=self.temporal_steps)
            for agent_id in self.agent_ids
        }

    def reset(self, observations: dict[str, np.ndarray]) -> None:
        for history in self._histories.values():
            history.clear()
        self.append(observations)

    def append(self, observations: dict[str, np.ndarray]) -> None:
        for agent_id in self.agent_ids:
            self._histories[agent_id].append(
                np.asarray(observations[agent_id], dtype=np.float32).copy()
            )

    def policy_inputs(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        observations, masks = {}, {}
        for agent_id, history in self._histories.items():
            frames = list(history)
            observations[agent_id] = np.stack(frames, axis=0).astype(np.float32)
            masks[agent_id] = np.ones(len(frames), dtype=bool)
        return observations, masks


def _base_starts_goals(config, scenario_name: str) -> tuple[np.ndarray, np.ndarray]:
    count = int(config.num_agents)
    if count < 2:
        raise ValueError("FCEP mechanism validation requires at least two agents")
    lower, upper = np.asarray(config.workspace_bounds, dtype=float)
    x_start = lower[0] + 0.8
    x_goal = upper[0] - 0.8
    y_values = np.linspace(lower[1] + 0.8, upper[1] - 0.8, count)
    z_value = float(np.clip(1.2, lower[2] + 0.3, upper[2] - 0.3))
    starts = np.column_stack(
        [np.full(count, x_start), y_values, np.full(count, z_value)]
    )
    goals = np.column_stack(
        [np.full(count, x_goal), y_values, np.full(count, z_value)]
    )

    if scenario_name == "two_agent_crossing":
        starts[0, 1], starts[1, 1] = y_values[0], y_values[-1]
        goals[0, 1], goals[1, 1] = y_values[-1], y_values[0]
        if count > 2:
            starts[2:, 1] = 0.5 * (lower[1] + upper[1])
            goals[2:, 1] = 0.5 * (lower[1] + upper[1])
            starts[2:, 2] = lower[2] + 0.35
            goals[2:, 2] = lower[2] + 0.35
    elif scenario_name == "narrow_yielding":
        corridor_center = 0.5 * (lower[1] + upper[1])
        starts[0] = [x_start, corridor_center - 0.18, z_value]
        goals[0] = [x_goal, corridor_center - 0.18, z_value]
        starts[1] = [x_goal, corridor_center + 0.18, z_value]
        goals[1] = [x_start, corridor_center + 0.18, z_value]
        if count > 2:
            starts[2:, 1] = lower[1] + 0.55
            goals[2:, 1] = lower[1] + 0.55
            starts[2:, 2] = lower[2] + 0.35
            goals[2:, 2] = lower[2] + 0.35
    return starts.astype(float), goals.astype(float)


def build_scenario_options(config, scenario_name: str) -> dict[str, Any]:
    starts, goals = _base_starts_goals(config, scenario_name)
    lower, upper = np.asarray(config.workspace_bounds, dtype=float)
    middle_x = 0.5 * (lower[0] + upper[0])
    middle_y = 0.5 * (lower[1] + upper[1])
    middle_z = float(starts[0, 2])
    static_obstacles: list[Any] = []
    dynamic_obstacles: list[Any] = []

    if scenario_name == "static_detour":
        static_obstacles.append(
            StaticSphereObstacle(
                center=np.array([middle_x, starts[0, 1], middle_z]),
                radius=0.42,
                safety_margin=0.08,
            )
        )
    elif scenario_name == "dynamic_crossing":
        dynamic_obstacles.append(
            MovingSphereObstacle(
                center=np.array([middle_x, lower[1] + 0.4, middle_z]),
                radius=0.30,
                safety_margin=0.05,
                velocity=np.array([0.0, 0.65, 0.0]),
                bounds=(lower, upper),
            )
        )
    elif scenario_name == "narrow_yielding":
        half_length = max(0.8, 0.5 * (upper[0] - lower[0]) - 1.6)
        wall_half_width = 0.34
        corridor_half_width = 0.48
        wall_height = max(0.25, 0.5 * (upper[2] - lower[2]) - 0.15)
        for sign in (-1.0, 1.0):
            static_obstacles.append(
                AxisAlignedBoxObstacle(
                    center=np.array(
                        [
                            middle_x,
                            middle_y + sign * (corridor_half_width + wall_half_width),
                            0.5 * (lower[2] + upper[2]),
                        ]
                    ),
                    half_extents=np.array([half_length, wall_half_width, wall_height]),
                    safety_margin=0.02,
                )
            )

    return {
        "starts": starts,
        "goals": goals,
        "static_obstacles": static_obstacles,
        "dynamic_obstacles": dynamic_obstacles,
    }


def build_env(config, phase_mode: str, max_steps: int) -> MultiAgentDMPEnv:
    run_config = replace(
        config,
        randomize_start_goal=False,
        max_steps=int(max_steps),
        phase_mode=str(phase_mode),
        phase_integrator="exponential",
        phase_min=1.0e-4,
    )
    return MultiAgentDMPEnv(**copy.deepcopy(run_config.build_core_env_kwargs()))


def generate_frozen_actions(
    policy,
    agent_ids: list[str],
    config,
    options: dict[str, Any],
    *,
    seed: int,
    max_steps: int,
) -> list[np.ndarray]:
    env = build_env(config, "classic", max_steps)
    actions: list[np.ndarray] = []
    try:
        obs_matrix, _ = env.reset(seed=int(seed), options=copy.deepcopy(options))
        obs = matrix_to_agent_dict(obs_matrix, agent_ids)
        history = AgentObservationHistory(policy.temporal_steps, agent_ids)
        history.reset(obs)
        for _ in range(int(max_steps)):
            policy_obs, masks = history.policy_inputs()
            action_dict = policy.evaluate_action(policy_obs, temporal_masks=masks)
            action = np.clip(
                agent_dict_to_matrix(action_dict, agent_ids),
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)
            actions.append(action.copy())
            next_obs, _, terminated, truncated, _ = env.step(action)
            obs = matrix_to_agent_dict(next_obs, agent_ids)
            history.append(obs)
            if bool(terminated or truncated):
                break
    finally:
        env.close()
    return actions


def replay_actions(
    config,
    options: dict[str, Any],
    actions: list[np.ndarray],
    *,
    scenario_name: str,
    phase_mode: str,
    seed: int,
    max_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = build_env(config, phase_mode, max_steps)
    rows: list[dict[str, Any]] = []
    positions_by_step: list[np.ndarray] = []
    accelerations_by_step: list[np.ndarray] = []
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}
    try:
        env.reset(seed=int(seed), options=copy.deepcopy(options))
        positions_by_step.append(env._positions().copy())
        for step_index, action in enumerate(actions):
            _, rewards, terminated, truncated, info = env.step(action.copy())
            final_info = info
            positions = env._positions().copy()
            velocities = env._velocities().copy()
            positions_by_step.append(positions)
            accelerations = np.asarray(info["applied_accelerations"], dtype=float)
            accelerations_by_step.append(accelerations)
            for agent_index in range(int(env.num_agents)):
                rows.append(
                    {
                        "scenario": scenario_name,
                        "phase_mode": phase_mode,
                        "seed": int(seed),
                        "step": int(step_index + 1),
                        "time": float((step_index + 1) * env.dynamics[0].dt),
                        "agent": int(agent_index),
                        "phase": float(info["phases"][agent_index]),
                        "flow_consistency": float(info["flow_consistencies"][agent_index]),
                        "phase_rate": float(info["phase_rates"][agent_index]),
                        "phase_paused": bool(info["phase_paused_mask"][agent_index]),
                        "agent_active": bool(
                            not info["success_rewarded_mask"][agent_index]
                        ),
                        "forcing_norm": float(info["residual_forcing_norms"][agent_index]),
                        "nominal_drive_norm": float(info["nominal_drive_norms"][agent_index]),
                        "closed_loop_drive_norm": float(info["closed_loop_drive_norms"][agent_index]),
                        "goal_error": float(info["distance_to_goals"][agent_index]),
                        "min_clearance": float(info["min_clearances"][agent_index]),
                        "min_inter_agent_separation": float(info["min_inter_agent_distance"]),
                        "reward": float(rewards[agent_index]),
                        "x": float(positions[agent_index, 0]),
                        "y": float(positions[agent_index, 1]),
                        "z": float(positions[agent_index, 2]),
                        "speed": float(np.linalg.norm(velocities[agent_index])),
                    }
                )
            if bool(terminated or truncated):
                break
    finally:
        env.close()

    position_array = np.asarray(positions_by_step, dtype=float)
    acceleration_array = np.asarray(accelerations_by_step, dtype=float)
    if len(position_array) > 1:
        path_length = np.linalg.norm(np.diff(position_array, axis=0), axis=-1).sum(axis=0)
    else:
        path_length = np.zeros(int(config.num_agents), dtype=float)
    if len(acceleration_array) > 1:
        jerk = np.diff(acceleration_array, axis=0) / float(config.time_step)
        jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))
    else:
        jerk_rms = 0.0

    active_rows = [row for row in rows if row["agent_active"]] or rows
    consistency_values = np.array(
        [row["flow_consistency"] for row in active_rows], dtype=float
    )
    phase_rate_values = np.array([row["phase_rate"] for row in active_rows], dtype=float)
    forcing_values = np.array([row["forcing_norm"] for row in active_rows], dtype=float)
    summary = {
        "scenario": scenario_name,
        "phase_mode": phase_mode,
        "seed": int(seed),
        "steps": int(len(positions_by_step) - 1),
        "flight_time": float((len(positions_by_step) - 1) * config.time_step),
        "success": bool(final_info.get("success", False)),
        "collision": bool(final_info.get("collision", False)),
        "timeout": bool(truncated),
        "flow_consistency_mean": float(np.mean(consistency_values)),
        "flow_consistency_min": float(np.min(consistency_values)),
        "negative_consistency_rate": float(np.mean(consistency_values < 0.0)),
        "phase_rate_mean": float(np.mean(phase_rate_values)),
        "phase_pause_fraction": float(np.mean(phase_rate_values <= 0.0)),
        "residual_forcing_norm_mean": float(np.mean(forcing_values)),
        "residual_forcing_norm_max": float(np.max(forcing_values)),
        "path_length_mean": float(np.mean(path_length)),
        "path_length_max": float(np.max(path_length)),
        "minimum_obstacle_clearance": float(
            np.nanmin([row["min_clearance"] for row in rows])
        ),
        "inter_agent_minimum_separation": float(
            np.min([row["min_inter_agent_separation"] for row in rows])
        ),
        "acceleration_peak": float(np.max(np.abs(acceleration_array))) if acceleration_array.size else 0.0,
        "jerk_rms": jerk_rms,
        "final_goal_error_mean": float(
            np.mean(final_info.get("distance_to_goals", np.full(int(config.num_agents), np.nan)))
        ),
        "final_phase_mean": float(
            np.mean(final_info.get("phases", np.full(int(config.num_agents), np.nan)))
        ),
        "positions": position_array,
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    serializable_rows = [
        {key: value for key, value in row.items() if key != "positions"}
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(serializable_rows[0]))
        writer.writeheader()
        writer.writerows(serializable_rows)


def plot_scenario(output_path: Path, scenario_name: str, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    colors = {"classic": "#2563eb", "fcep": "#dc2626"}
    for phase_mode in ("classic", "fcep"):
        selected = [row for row in rows if row["phase_mode"] == phase_mode]
        steps = sorted({int(row["step"]) for row in selected})
        times = np.array([step for step in steps], dtype=float)
        for axis, key, label in (
            (axes[0, 0], "phase", "phase"),
            (axes[0, 1], "flow_consistency", "flow consistency"),
            (axes[1, 0], "phase_rate", "phase rate"),
            (axes[1, 1], "forcing_norm", "residual forcing norm"),
            (axes[2, 0], "goal_error", "active goal error"),
        ):
            values = [
                np.mean([row[key] for row in selected if int(row["step"]) == step])
                for step in steps
            ]
            axis.plot(times, values, label=phase_mode, color=colors[phase_mode])
            axis.set_xlabel("step")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)

        for agent_index in sorted({int(row["agent"]) for row in selected}):
            trajectory = [row for row in selected if int(row["agent"]) == agent_index]
            axes[2, 1].plot(
                [row["x"] for row in trajectory],
                [row["y"] for row in trajectory],
                color=colors[phase_mode],
                linestyle="-" if phase_mode == "classic" else "--",
                alpha=0.8,
                label=f"{phase_mode}/agent_{agent_index}",
            )
    for axis in axes.flat[:5]:
        axis.legend()
    axes[2, 1].set_xlabel("x / m")
    axes[2, 1].set_ylabel("y / m")
    axes[2, 1].set_title("trajectory projection")
    axes[2, 1].grid(alpha=0.25)
    axes[2, 1].legend(fontsize=8, ncol=2)
    figure.suptitle(f"Frozen-policy FCEP mechanism: {scenario_name}")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-action classic/FCEP mechanism validation")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-output-root", type=Path, default=Path("artifacts/masac"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/fcep_mechanism"))
    parser.add_argument("--scenario", choices=("all", *SCENARIO_NAMES), default="all")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = _resolve_checkpoint(args.checkpoint, args.model_output_root)
    config_path = _find_run_config(checkpoint, args.config)
    experiment_config, network_config, _ = _load_configs(config_path)
    device = _resolve_device(args.device)
    probe = build_env(experiment_config, "classic", args.max_steps)
    try:
        policy, agent_ids = _load_policy(checkpoint, probe, network_config, device)
    finally:
        probe.close()

    selected_scenarios = SCENARIO_NAMES if args.scenario == "all" else (args.scenario,)
    output_dir = args.output_root / time.strftime("validation_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    criteria: dict[str, Any] = {}

    for scenario_index, scenario_name in enumerate(selected_scenarios):
        scenario_seed = int(args.seed) + scenario_index
        options = build_scenario_options(experiment_config, scenario_name)
        actions = generate_frozen_actions(
            policy,
            agent_ids,
            experiment_config,
            options,
            seed=scenario_seed,
            max_steps=args.max_steps,
        )
        scenario_rows: list[dict[str, Any]] = []
        scenario_summaries = []
        for phase_mode in ("classic", "fcep"):
            rows, summary = replay_actions(
                experiment_config,
                options,
                actions,
                scenario_name=scenario_name,
                phase_mode=phase_mode,
                seed=scenario_seed,
                max_steps=args.max_steps,
            )
            scenario_rows.extend(rows)
            scenario_summaries.append(summary)
        classic_summary, fcep_summary = scenario_summaries
        common_steps = min(
            len(classic_summary["positions"]),
            len(fcep_summary["positions"]),
        )
        trajectory_deviation = float(np.max(np.abs(
            classic_summary["positions"][:common_steps]
            - fcep_summary["positions"][:common_steps]
        )))
        for summary in scenario_summaries:
            summary["trajectory_max_abs_deviation"] = trajectory_deviation
            summary.pop("positions", None)
            summaries.append(summary)
        criteria[scenario_name] = {
            "same_action_trajectory_max_abs_deviation": trajectory_deviation,
            "same_action_trajectory_preserved": trajectory_deviation <= 1.0e-6,
            "classic_flow_consistency_mean": classic_summary["flow_consistency_mean"],
            "fcep_flow_consistency_mean": fcep_summary["flow_consistency_mean"],
            "fcep_phase_pause_fraction": fcep_summary["phase_pause_fraction"],
            "fcep_phase_slower_than_classic": (
                fcep_summary["final_phase_mean"] >= classic_summary["final_phase_mean"] - 1.0e-7
            ),
        }
        all_rows.extend(scenario_rows)
    _write_csv(output_dir / "traces.csv", all_rows)
    _write_csv(output_dir / "summary.csv", summaries)
    report = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "device": str(device),
        "seed": int(args.seed),
        "scenarios": list(selected_scenarios),
        "frozen_components": ["Actor", "Critic", "observation encoder"],
        "action_replay": "classic deterministic actor actions replayed unchanged in both modes",
        "active_goal_switch_metrics": "not_applicable: current environment has no active/pending goal switch",
        "criteria": criteria,
        "summaries": summaries,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plotting_script = Path(__file__).with_name("plot_fcep_mechanism.py")
    subprocess.run(
        [
            sys.executable,
            str(plotting_script),
            "--trace-csv",
            str(output_dir / "traces.csv"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    print(f"FCEP mechanism validation output: {output_dir}")


if __name__ == "__main__":
    main()
