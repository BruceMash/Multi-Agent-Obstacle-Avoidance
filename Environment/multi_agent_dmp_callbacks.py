from __future__ import annotations

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks


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

        if distance_to_goals.size:
            episode.custom_metrics["final_distance_mean"] = float(np.mean(distance_to_goals))
            episode.custom_metrics["final_distance_min"] = float(np.min(distance_to_goals))
            episode.custom_metrics["final_distance_max"] = float(np.max(distance_to_goals))
