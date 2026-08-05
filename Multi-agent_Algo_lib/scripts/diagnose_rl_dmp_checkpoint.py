from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ALGO_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
for path in (PROJECT_ROOT, ALGO_ROOT, SCRIPT_PATH.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validate_fcep_mechanism import SCENARIO_NAMES, build_scenario_options
from validate_masac_multi_agent_scenarios import (
    _find_run_config,
    _load_configs,
    _load_policy,
    _resolve_checkpoint,
    _resolve_device,
)
from Environment.multi_agent_dmp_env import MultiAgentDMPEnv


REWARD_KEYS = (
    "reward_progress",
    "reward_obstacle_potential_penalty",
    "reward_boundary_potential_penalty",
    "reward_inter_agent_potential_penalty",
    "reward_stagnation_penalty",
    "reward_acceleration_penalty",
    "reward_acceleration_clip_penalty",
    "reward_individual_success_bonus",
    "reward_team_success_bonus",
    "reward_team_collision_penalty",
    "reward_local_collision_penalty",
    "reward_team_timeout_penalty",
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
        observations: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        for agent_id, history in self._histories.items():
            frames = list(history)
            feature_dim = frames[-1].shape[-1]
            sequence = np.zeros((self.temporal_steps, feature_dim), dtype=np.float32)
            mask = np.zeros(self.temporal_steps, dtype=bool)
            sequence[: len(frames)] = np.asarray(frames, dtype=np.float32)
            mask[: len(frames)] = True
            observations[agent_id] = sequence
            masks[agent_id] = mask
        return observations, masks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def matrix_to_agent_dict(values: np.ndarray, agent_ids: list[str]) -> dict[str, np.ndarray]:
    return {agent_id: np.asarray(values[index], dtype=np.float32) for index, agent_id in enumerate(agent_ids)}


def agent_dict_to_matrix(values: dict[str, np.ndarray], agent_ids: list[str]) -> np.ndarray:
    return np.stack([np.asarray(values[agent_id], dtype=np.float32) for agent_id in agent_ids])


def build_env(config, max_steps: int) -> MultiAgentDMPEnv:
    run_config = replace(
        config,
        randomize_start_goal=False,
        max_steps=int(max_steps),
        phase_mode="classic",
        enable_flow_consistency_loss=False,
    )
    return MultiAgentDMPEnv(**copy.deepcopy(run_config.build_core_env_kwargs()))


def deterministic_actions_with_diagnostics(
    policy: Any,
    policy_obs: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    agent_ids: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    import torch

    actions: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    obs_tensors: dict[str, Any] = {}
    mask_tensors: dict[str, Any] = {}
    with torch.no_grad():
        for agent_id in agent_ids:
            obs_tensor, mask_tensor = policy._prepare_actor_input(
                policy_obs[agent_id], masks.get(agent_id)
            )
            action, _, actor_info = policy.agents[agent_id].actor(
                obs_tensor,
                deterministic=True,
                temporal_mask=mask_tensor,
                return_diagnostics=True,
            )
            actions[agent_id] = action.cpu().numpy().squeeze(0)
            diagnostics[agent_id] = {
                key: value.detach().cpu().numpy().squeeze(0)
                for key, value in actor_info.items()
            }
            obs_tensors[agent_id] = obs_tensor
            mask_tensors[agent_id] = mask_tensor
    return actions, diagnostics, obs_tensors, mask_tensors


def evaluate_q_scales(
    policy: Any,
    obs_tensors: dict[str, Any],
    mask_tensors: dict[str, Any],
    physical_actions: dict[str, np.ndarray],
    lambdas: list[float],
    agent_ids: list[str],
) -> list[dict[str, float]]:
    import torch

    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for scale in lambdas:
            critic_actions: dict[str, Any] = {}
            for agent_id in agent_ids:
                physical = torch.as_tensor(
                    physical_actions[agent_id] * float(scale),
                    dtype=torch.float32,
                    device=policy.device,
                ).reshape(1, -1)
                critic_actions[agent_id] = policy.actor_action_to_critic_action(
                    obs_tensors[agent_id], physical, temporal_mask=mask_tensors[agent_id]
                )
            for agent_id in agent_ids:
                q1, q2 = policy.agents[agent_id].critic(
                    obs_tensors,
                    critic_actions,
                    temporal_masks=mask_tensors,
                )
                q1_value = float(q1.item())
                q2_value = float(q2.item())
                rows.append(
                    {
                        "agent": int(agent_id.rsplit("_", 1)[-1]),
                        "lambda": float(scale),
                        "q1": q1_value,
                        "q2": q2_value,
                        "q_min": min(q1_value, q2_value),
                        "q_gap": abs(q1_value - q2_value),
                    }
                )
    return rows


def clearance_bin(value: float) -> str:
    if value < 0.5:
        return "lt_0.5"
    if value < 1.0:
        return "0.5_to_1.0"
    return "ge_1.0"


def run_online_episode(
    policy: Any,
    agent_ids: list[str],
    config,
    options: dict[str, Any],
    scenario: str,
    seed: int,
    max_steps: int,
    lambdas: list[float],
    has_critics: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray], dict[str, Any], dict[str, list[float]]]:
    env = build_env(config, max_steps)
    action_rows: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    actions_by_step: list[np.ndarray] = []
    reward_values = {key: [] for key in REWARD_KEYS}
    reward_values["total_reward"] = []
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    try:
        obs_matrix, reset_info = env.reset(seed=seed, options=copy.deepcopy(options))
        initial_goal_errors = np.asarray(reset_info["distance_to_goals"], dtype=float)
        obs = matrix_to_agent_dict(obs_matrix, agent_ids)
        history = AgentObservationHistory(policy.temporal_steps, agent_ids)
        history.reset(obs)
        for step_index in range(max_steps):
            policy_obs, masks = history.policy_inputs()
            action_dict, actor_diagnostics, obs_tensors, mask_tensors = deterministic_actions_with_diagnostics(
                policy, policy_obs, masks, agent_ids
            )
            action_matrix = np.clip(
                agent_dict_to_matrix(action_dict, agent_ids),
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)
            actions_by_step.append(action_matrix.copy())
            if has_critics:
                step_q_rows = evaluate_q_scales(
                    policy, obs_tensors, mask_tensors, action_dict, lambdas, agent_ids
                )
                for row in step_q_rows:
                    row.update({"scenario": scenario, "step": step_index + 1})
                q_rows.extend(step_q_rows)

            next_obs_matrix, rewards, terminated, truncated, info = env.step(action_matrix)
            final_info = info
            for key in REWARD_KEYS:
                values = np.asarray(info.get(key, np.zeros(env.num_agents)), dtype=float)
                reward_values[key].extend(values.tolist())
            reward_values["total_reward"].extend(np.asarray(rewards, dtype=float).tolist())

            for agent_index, agent_id in enumerate(agent_ids):
                nominal_norm = float(info["nominal_drive_norms"][agent_index])
                raw_forcing = np.asarray(info["raw_residual_forcing"][agent_index], dtype=float)
                effective_forcing = np.asarray(info["effective_residual_forcing"][agent_index], dtype=float)
                goal_delta = np.asarray(info["goal_deltas"][agent_index], dtype=float)
                raw_norm = float(np.linalg.norm(raw_forcing))
                effective_norm = float(np.linalg.norm(effective_forcing))
                minimum_clearance = float(info["min_clearances"][agent_index])
                goal_error = float(info["distance_to_goals"][agent_index])
                remaining_fraction = goal_error / max(float(initial_goal_errors[agent_index]), 1.0e-8)
                actor_info = actor_diagnostics[agent_id]
                raw_limit = float(env.dmp_config.forcing_term_max)
                action_rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "step": step_index + 1,
                        "agent": agent_index,
                        "agent_active": bool(not info["success_rewarded_mask"][agent_index]),
                        "flight_stage": ("early" if remaining_fraction > 2.0 / 3.0 else "middle" if remaining_fraction > 1.0 / 3.0 else "late"),
                        "clearance_bin": clearance_bin(minimum_clearance),
                        "goal_error": goal_error,
                        "minimum_clearance": minimum_clearance,
                        "raw_forcing_norm": raw_norm,
                        "effective_forcing_norm": effective_norm,
                        "nominal_drive_norm": nominal_norm,
                        "raw_residual_ratio": raw_norm / (nominal_norm + 1.0e-8),
                        "effective_residual_ratio": effective_norm / (nominal_norm + 1.0e-8),
                        "forcing_saturation_rate": float(np.mean(np.abs(raw_forcing) >= 0.99 * raw_limit)),
                        "goal_offset_norm": float(np.linalg.norm(info["goal_offsets"][agent_index])),
                        "axis_gate_mean": float(np.mean(info["forcing_gates"][agent_index])),
                        "scalar_gate": float(np.tanh(env.dmp_config.forcing_gate_kappa * np.linalg.norm(goal_delta))),
                        "acceleration_clipped": bool(np.any(info["acceleration_clip_mask"][agent_index])),
                        "velocity_clipped": bool(np.any(info["velocity_clip_mask"][agent_index])),
                        "pre_tanh_mean": float(np.mean(actor_info["pre_tanh_action"])),
                        "pre_tanh_std": float(np.std(actor_info["pre_tanh_action"])),
                        "post_tanh_abs_mean": float(np.mean(np.abs(actor_info["post_tanh_action"]))),
                    }
                )
            obs = matrix_to_agent_dict(next_obs_matrix, agent_ids)
            history.append(obs)
            if bool(terminated or truncated):
                break
    finally:
        env.close()
    outcome = {
        "scenario": scenario,
        "seed": seed,
        "lambda": 1.0,
        "steps": len(actions_by_step),
        "success": bool(final_info.get("success", False)),
        "collision": bool(final_info.get("collision", False)),
        "timeout": bool(truncated),
        "total_reward_mean": float(np.sum(reward_values["total_reward"]) / max(len(agent_ids), 1)),
    }
    return action_rows, q_rows, actions_by_step, outcome, reward_values


def replay_scaled_actions(
    actions: list[np.ndarray],
    scale: float,
    config,
    options: dict[str, Any],
    scenario: str,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    env = build_env(config, max_steps)
    total_reward = np.zeros(env.num_agents, dtype=float)
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    try:
        env.reset(seed=seed, options=copy.deepcopy(options))
        for action in actions:
            scaled = np.clip(action * float(scale), env.action_space.low, env.action_space.high)
            _, rewards, terminated, truncated, info = env.step(scaled)
            total_reward += np.asarray(rewards, dtype=float)
            final_info = info
            if bool(terminated or truncated):
                break
    finally:
        env.close()
    return {
        "scenario": scenario,
        "seed": seed,
        "lambda": float(scale),
        "steps": int(final_info.get("steps", 0)),
        "success": bool(final_info.get("success", False)),
        "collision": bool(final_info.get("collision", False)),
        "timeout": bool(truncated),
        "total_reward_mean": float(np.mean(total_reward)),
        "final_goal_error_mean": float(np.mean(final_info.get("distance_to_goals", [np.nan]))),
    }


def quantile_stats(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p75": float(np.quantile(array, 0.75)),
        f"{prefix}_p90": float(np.quantile(array, 0.90)),
        f"{prefix}_p95": float(np.quantile(array, 0.95)),
        f"{prefix}_p99": float(np.quantile(array, 0.99)),
    }


def grouped_action_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active_rows = [row for row in rows if row.get("agent_active", True)]
    if active_rows:
        rows = active_rows
    definitions = {
        "overall": lambda row: "all",
        "scenario": lambda row: row["scenario"],
        "agent": lambda row: str(row["agent"]),
        "flight_stage": lambda row: row["flight_stage"],
        "clearance_bin": lambda row: row["clearance_bin"],
    }
    output = []
    for group_type, key_fn in definitions.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(key_fn(row))].append(row)
        for group_value, group_rows in groups.items():
            summary = {
                "group_type": group_type,
                "group_value": group_value,
                "sample_count": len(group_rows),
                "forcing_saturation_rate": float(np.mean([row["forcing_saturation_rate"] for row in group_rows])),
                "acceleration_clip_rate": float(np.mean([row["acceleration_clipped"] for row in group_rows])),
                "velocity_clip_rate": float(np.mean([row["velocity_clipped"] for row in group_rows])),
                "axis_gate_mean": float(np.mean([row["axis_gate_mean"] for row in group_rows])),
                "scalar_gate_mean": float(np.mean([row["scalar_gate"] for row in group_rows])),
            }
            for key in ("raw_forcing_norm", "effective_forcing_norm", "raw_residual_ratio", "effective_residual_ratio"):
                summary.update(quantile_stats([row[key] for row in group_rows], key))
            output.append(summary)
    return output


def reward_summary(scenario_values: dict[str, dict[str, list[float]]]) -> list[dict[str, Any]]:
    rows = []
    for scenario, values_by_key in scenario_values.items():
        total_abs = sum(float(np.sum(np.abs(values))) for key, values in values_by_key.items() if key != "total_reward")
        for key, values in values_by_key.items():
            array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "scenario": scenario,
                    "component": key,
                    "sample_count": len(array),
                    "episode_sum": float(np.sum(array)),
                    "step_mean": float(np.mean(array)),
                    "absolute_mean": float(np.mean(np.abs(array))),
                    "p90_absolute": float(np.quantile(np.abs(array), 0.9)),
                    "absolute_component_ratio": float(np.sum(np.abs(array)) / (total_abs + 1.0e-8)) if key != "total_reward" else np.nan,
                }
            )
    return rows


def plot_results(output_dir: Path, action_rows, q_rows, outcomes, reward_rows) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].hist([row["raw_forcing_norm"] for row in action_rows], bins=40, alpha=0.65, label="raw")
    axes[0, 0].hist([row["effective_forcing_norm"] for row in action_rows], bins=40, alpha=0.65, label="effective")
    axes[0, 0].set_title("Forcing norm distribution")
    axes[0, 0].legend()
    axes[0, 1].hist([row["effective_residual_ratio"] for row in action_rows], bins=40)
    axes[0, 1].set_title("Effective residual/nominal ratio")
    scenarios = list(dict.fromkeys(row["scenario"] for row in action_rows))
    axes[1, 0].bar(scenarios, [np.mean([row["forcing_saturation_rate"] for row in action_rows if row["scenario"] == name]) for name in scenarios])
    axes[1, 0].set_title("Forcing saturation rate")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 1].bar(scenarios, [np.mean([row["acceleration_clipped"] for row in action_rows if row["scenario"] == name]) for name in scenarios])
    axes[1, 1].set_title("Acceleration clipping rate")
    axes[1, 1].tick_params(axis="x", rotation=20)
    fig.savefig(output_dir / "action_diagnostics.png", dpi=160)
    plt.close(fig)

    if q_rows:
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        lambdas = sorted(set(row["lambda"] for row in q_rows))
        axis.plot(lambdas, [np.mean([row["q_min"] for row in q_rows if row["lambda"] == value]) for value in lambdas], marker="o")
        axis.set_title("Qmin versus action scale")
        axis.set_xlabel("lambda")
        axis.set_ylabel("Qmin")
        axis.grid(True, alpha=0.3)
        fig.savefig(output_dir / "q_action_scale.png", dpi=160)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for scenario in sorted(set(row["scenario"] for row in outcomes)):
        rows = sorted([row for row in outcomes if row["scenario"] == scenario], key=lambda row: row["lambda"])
        axis.plot([row["lambda"] for row in rows], [row["total_reward_mean"] for row in rows], marker="o", label=scenario)
    axis.set_title("Frozen-action replay return versus action scale")
    axis.set_xlabel("lambda")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.savefig(output_dir / "scaled_action_outcomes.png", dpi=160)
    plt.close(fig)

    components = sorted(set(row["component"] for row in reward_rows if row["component"] != "total_reward"))
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    values = [np.mean([row["absolute_mean"] for row in reward_rows if row["component"] == component]) for component in components]
    axis.barh(components, values)
    axis.set_title("Mean absolute reward component scale")
    fig.savefig(output_dir / "reward_components.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose action scale, reward scale, and Critic boundary preference.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-output-root", type=Path, default=Path("artifacts/masac"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/calibration"))
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenario", choices=("all", *SCENARIO_NAMES), default="all")
    parser.add_argument("--lambdas", default="0,0.25,0.5,0.75,1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = _resolve_checkpoint(args.checkpoint, args.model_output_root)
    config_path = _find_run_config(checkpoint, args.config)
    experiment_config, network_config, _ = _load_configs(config_path)
    device = _resolve_device(args.device)
    probe = build_env(experiment_config, args.max_steps)
    try:
        policy, agent_ids = _load_policy(checkpoint, probe, network_config, device)
    finally:
        probe.close()
    has_critics = bool(getattr(policy, "checkpoint_has_critics", False))
    lambdas = [float(value) for value in args.lambdas.split(",")]
    scenarios = SCENARIO_NAMES if args.scenario == "all" else (args.scenario,)
    output_dir = args.output_root / time.strftime("checkpoint_diagnostic_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    action_rows: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    rewards_by_scenario: dict[str, dict[str, list[float]]] = {}
    for scenario_index, scenario in enumerate(scenarios):
        seed = int(args.seed) + scenario_index
        options = build_scenario_options(experiment_config, scenario)
        rows, episode_q_rows, actions, base_outcome, reward_values = run_online_episode(
            policy,
            agent_ids,
            experiment_config,
            options,
            scenario,
            seed,
            args.max_steps,
            lambdas,
            has_critics,
        )
        action_rows.extend(rows)
        q_rows.extend(episode_q_rows)
        rewards_by_scenario[scenario] = reward_values
        for scale in lambdas:
            if np.isclose(scale, 1.0):
                outcomes.append(base_outcome)
            else:
                outcomes.append(
                    replay_scaled_actions(actions, scale, experiment_config, options, scenario, seed, args.max_steps)
                )

    action_summary = grouped_action_summary(action_rows)
    rewards = reward_summary(rewards_by_scenario)
    write_csv(output_dir / "action_traces.csv", action_rows)
    write_csv(output_dir / "action_summary.csv", action_summary)
    write_csv(output_dir / "reward_summary.csv", rewards)
    write_csv(output_dir / "scaled_action_outcomes.csv", outcomes)
    write_csv(output_dir / "q_action_scan.csv", q_rows)
    overall = next(row for row in action_summary if row["group_type"] == "overall")
    report = {
        "launch_command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "checkpoint": str(checkpoint.resolve()),
        "config": str(config_path.resolve()),
        "device": str(device),
        "critic_checkpoint_available": has_critics,
        "critic_action_scan_status": "completed" if has_critics else "not_available: legacy checkpoint stores Actor only",
        "scenarios": list(scenarios),
        "overall_action_statistics": overall,
        "scaled_action_outcomes": outcomes,
        "artifacts": {
            "action_traces": "action_traces.csv",
            "action_summary": "action_summary.csv",
            "reward_summary": "reward_summary.csv",
            "scaled_action_outcomes": "scaled_action_outcomes.csv",
            "q_action_scan": "q_action_scan.csv" if q_rows else None,
            "plots": ["action_diagnostics.png", "scaled_action_outcomes.png", "reward_components.png"] + (["q_action_scale.png"] if q_rows else []),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH.with_name("plot_checkpoint_diagnostics.py")),
            "--input-dir",
            str(output_dir),
        ],
        check=True,
    )
    print(json.dumps({"output_dir": str(output_dir), "critic_available": has_critics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
