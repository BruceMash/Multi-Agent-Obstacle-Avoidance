from __future__ import annotations

import copy
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks

from Environment.multi_agent_dmp_env import MultiAgentDMPEnv


class MultiAgentDMPMetricsCallback(DefaultCallbacks):
    """Collect episode-level diagnostics from MultiAgentDMPEnv info fields."""

    @staticmethod
    def _last_info(episode) -> dict:
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if isinstance(info, dict) and info:
                return info
        return {}

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs) -> None:
        info = self._last_info(episode)
        if not info:
            return

        success_mask = np.asarray(info.get("success_mask", []), dtype=bool)
        success_rewarded_mask = np.asarray(info.get("success_rewarded_mask", success_mask), dtype=bool)
        obstacle_collision_mask = np.asarray(info.get("obstacle_collision_mask", []), dtype=bool)
        inter_agent_collision_mask = np.asarray(info.get("inter_agent_collision_mask", []), dtype=bool)
        boundary_collision_mask = np.asarray(info.get("boundary_collision_mask", []), dtype=bool)
        distance_to_goals = np.asarray(info.get("distance_to_goals", []), dtype=float)

        if success_mask.size:
            episode.custom_metrics["instant_success_rate"] = float(np.mean(success_mask))
            episode.custom_metrics["full_success_rate"] = float(np.all(success_mask))
        if success_rewarded_mask.size:
            episode.custom_metrics["success_rate"] = float(np.mean(success_rewarded_mask))
            episode.custom_metrics["reached_agent_count"] = float(np.sum(success_rewarded_mask))

        episode.custom_metrics["timeout_rate"] = float(bool(info.get("truncated", False)))
        episode.custom_metrics["collision_rate"] = float(bool(info.get("collision", False)))

        if obstacle_collision_mask.size:
            episode.custom_metrics["obstacle_collision_rate"] = float(np.mean(obstacle_collision_mask))
        if inter_agent_collision_mask.size:
            episode.custom_metrics["inter_agent_collision_rate"] = float(np.mean(inter_agent_collision_mask))
        if boundary_collision_mask.size:
            episode.custom_metrics["boundary_collision_rate"] = float(np.mean(boundary_collision_mask))

        if distance_to_goals.size:
            episode.custom_metrics["final_distance_mean"] = float(np.mean(distance_to_goals))
            episode.custom_metrics["final_distance_min"] = float(np.min(distance_to_goals))
            episode.custom_metrics["final_distance_max"] = float(np.max(distance_to_goals))

    def on_train_result(self, *, trainer=None, algorithm=None, result: dict | None = None, **kwargs) -> None:
        trainer = trainer or algorithm
        if trainer is None or result is None:
            return

        fixed_eval_config = self._get_fixed_eval_config(trainer)
        if not fixed_eval_config.get("enabled", False):
            return

        iteration = int(result.get("training_iteration", 0))
        checkpoint_freq = max(1, int(fixed_eval_config.get("checkpoint_freq", 1)))
        should_eval = iteration > 0 and iteration % checkpoint_freq == 0
        should_eval = should_eval or bool(result.get("done", False))
        if not should_eval:
            return
        if getattr(self, "_last_fixed_eval_iteration", None) == iteration:
            return
        self._last_fixed_eval_iteration = iteration

        try:
            eval_record = self._run_fixed_eval(
                trainer=trainer,
                result=result,
                fixed_eval_config=fixed_eval_config,
            )
            self._attach_fixed_eval_metrics(result, eval_record["summary"])
            self._write_fixed_eval_record(trainer, fixed_eval_config, eval_record)
        except Exception as exc:
            error_record = {
                "training_iteration": iteration,
                "timesteps_total": int(result.get("timesteps_total", 0)),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            result.setdefault("custom_metrics", {})["fixed_eval_error"] = 1.0
            self._write_fixed_eval_record(trainer, fixed_eval_config, error_record)

    @staticmethod
    def _get_fixed_eval_config(trainer) -> dict[str, Any]:
        model_config = trainer.config.get("model", {})
        custom_config = model_config.get("custom_model_config", {})
        return dict(custom_config.get("fixed_eval_config", {}))

    @staticmethod
    def _get_core_env_kwargs(trainer) -> dict[str, Any]:
        model_config = trainer.config.get("model", {})
        custom_config = model_config.get("custom_model_config", {})
        env_args = custom_config.get("env_args", {})
        return copy.deepcopy(env_args.get("core_env_kwargs", {}))

    @staticmethod
    def _get_policy_id(trainer) -> str:
        try:
            policy_map = trainer.workers.local_worker().policy_map
            if "shared_policy" in policy_map:
                return "shared_policy"
            if "default_policy" in policy_map:
                return "default_policy"
            return next(iter(policy_map.keys()))
        except Exception:
            return "shared_policy"

    @staticmethod
    def _compute_policy_action(trainer, observation: np.ndarray, policy_id: str, deterministic: bool) -> np.ndarray:
        action = trainer.compute_single_action(
            {"obs": observation.astype(np.float32, copy=True)},
            policy_id=policy_id,
            explore=not deterministic,
        )
        if isinstance(action, tuple):
            action = action[0]
        return np.asarray(action, dtype=np.float32)

    def _run_fixed_eval(self, *, trainer, result: dict, fixed_eval_config: dict[str, Any]) -> dict[str, Any]:
        core_env_kwargs = self._get_core_env_kwargs(trainer)
        scenarios = list(fixed_eval_config.get("scenarios", []))
        num_scenarios = max(0, int(fixed_eval_config.get("num_scenarios", len(scenarios))))
        scenarios = scenarios[:num_scenarios]
        deterministic = bool(fixed_eval_config.get("deterministic", True))
        policy_id = self._get_policy_id(trainer)

        scenario_results = []
        for scenario_index, scenario in enumerate(scenarios):
            env = MultiAgentDMPEnv(**copy.deepcopy(core_env_kwargs))
            try:
                scenario_result = self._run_single_fixed_scenario(
                    trainer=trainer,
                    env=env,
                    scenario=scenario,
                    scenario_index=scenario_index,
                    policy_id=policy_id,
                    deterministic=deterministic,
                )
                scenario_results.append(scenario_result)
            finally:
                env.close()

        summary = self._summarize_fixed_eval(scenario_results)
        return {
            "training_iteration": int(result.get("training_iteration", 0)),
            "timesteps_total": int(result.get("timesteps_total", 0)),
            "deterministic": deterministic,
            "policy_id": policy_id,
            "scenario_results": scenario_results,
            "summary": summary,
        }

    def _run_single_fixed_scenario(
        self,
        *,
        trainer,
        env: MultiAgentDMPEnv,
        scenario: dict[str, Any],
        scenario_index: int,
        policy_id: str,
        deterministic: bool,
    ) -> dict[str, Any]:
        starts = np.asarray(scenario["starts"], dtype=float)
        goals = np.asarray(scenario["goals"], dtype=float)
        observation, info = env.reset(
            seed=10_000 + scenario_index,
            options={"starts": starts, "goals": goals},
        )

        total_reward = 0.0
        terminated = False
        truncated = False
        last_info = info

        while not (terminated or truncated):
            actions = np.stack(
                [
                    self._compute_policy_action(
                        trainer,
                        observation[agent_index],
                        policy_id,
                        deterministic,
                    )
                    for agent_index in range(env.num_agents)
                ],
                axis=0,
            )
            observation, rewards, terminated, truncated, last_info = env.step(actions)
            total_reward += float(np.sum(rewards))

        distance_to_goals = np.asarray(last_info.get("distance_to_goals", []), dtype=float)
        success_mask = np.asarray(last_info.get("success_mask", []), dtype=bool)
        success_rewarded_mask = np.asarray(last_info.get("success_rewarded_mask", success_mask), dtype=bool)
        obstacle_collision_mask = np.asarray(last_info.get("obstacle_collision_mask", []), dtype=bool)
        inter_agent_collision_mask = np.asarray(last_info.get("inter_agent_collision_mask", []), dtype=bool)
        boundary_collision_mask = np.asarray(last_info.get("boundary_collision_mask", []), dtype=bool)

        return {
            "name": str(scenario.get("name", f"scenario_{scenario_index}")),
            "episode_reward": float(total_reward),
            "mean_agent_reward": float(total_reward / max(1, env.num_agents)),
            "episode_length": int(env.steps),
            "success": bool(last_info.get("success", False)),
            "instant_success_rate": float(np.mean(success_mask)) if success_mask.size else 0.0,
            "reached_agent_count": float(np.sum(success_rewarded_mask)) if success_rewarded_mask.size else 0.0,
            "collision": bool(last_info.get("collision", False)),
            "timeout": bool(last_info.get("truncated", False)),
            "obstacle_collision_rate": float(np.mean(obstacle_collision_mask)) if obstacle_collision_mask.size else 0.0,
            "inter_agent_collision_rate": float(np.mean(inter_agent_collision_mask)) if inter_agent_collision_mask.size else 0.0,
            "boundary_collision_rate": float(np.mean(boundary_collision_mask)) if boundary_collision_mask.size else 0.0,
            "final_distance_mean": float(np.mean(distance_to_goals)) if distance_to_goals.size else float("nan"),
            "final_distance_min": float(np.min(distance_to_goals)) if distance_to_goals.size else float("nan"),
            "final_distance_max": float(np.max(distance_to_goals)) if distance_to_goals.size else float("nan"),
        }

    @staticmethod
    def _mean_metric(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
        values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        return float(np.mean(values)) if values else float(default)

    def _summarize_fixed_eval(self, scenario_results: list[dict[str, Any]]) -> dict[str, float]:
        rows = scenario_results
        return {
            "scenario_count": float(len(rows)),
            "mean_reward": self._mean_metric(rows, "episode_reward"),
            "mean_agent_reward": self._mean_metric(rows, "mean_agent_reward"),
            "episode_len_mean": self._mean_metric(rows, "episode_length"),
            "success_rate": self._mean_metric(rows, "success"),
            "instant_success_rate": self._mean_metric(rows, "instant_success_rate"),
            "collision_rate": self._mean_metric(rows, "collision"),
            "timeout_rate": self._mean_metric(rows, "timeout"),
            "reached_agent_count": self._mean_metric(rows, "reached_agent_count"),
            "obstacle_collision_rate": self._mean_metric(rows, "obstacle_collision_rate"),
            "inter_agent_collision_rate": self._mean_metric(rows, "inter_agent_collision_rate"),
            "boundary_collision_rate": self._mean_metric(rows, "boundary_collision_rate"),
            "final_distance_mean": self._mean_metric(rows, "final_distance_mean", float("nan")),
            "final_distance_min": self._mean_metric(rows, "final_distance_min", float("nan")),
            "final_distance_max": self._mean_metric(rows, "final_distance_max", float("nan")),
        }

    @staticmethod
    def _attach_fixed_eval_metrics(result: dict, summary: dict[str, float]) -> None:
        custom_metrics = result.setdefault("custom_metrics", {})
        for key, value in summary.items():
            custom_metrics[f"fixed_eval_{key}"] = float(value)

    @staticmethod
    def _write_fixed_eval_record(trainer, fixed_eval_config: dict[str, Any], record: dict[str, Any]) -> None:
        logdir = Path(getattr(trainer, "logdir", "") or ".")
        output_dir = logdir / str(fixed_eval_config.get("output_dirname", "fixed_eval"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "fixed_eval_results.jsonl"
        with output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")
