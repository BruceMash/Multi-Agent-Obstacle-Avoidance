from __future__ import annotations

import numpy as np
import ray

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from Controller.dmp_rl import DMPConfig
from Environment.multi_agent_dmp_env import MultiAgentDMPEnv, MultiAgentEnvConfig


class MultiAgentDMPRllibEnv(MultiAgentEnv):
    """
    RLlib adapter for MultiAgentDMPEnv.

    The core environment keeps matrix-style multi-agent simulation:
        obs:     (num_agents, obs_dim)
        action:  (num_agents, action_dim)
        reward:  (num_agents,)

    This adapter converts it to RLlib's dictionary-style MultiAgentEnv API.
    """

    def __init__(self, env_config: dict | None = None):
        super().__init__()
        env_config = env_config or {}
        self._new_step_api = int(ray.__version__.split(".", 1)[0]) >= 2

        self.core_env = self._build_core_env(env_config)
        self._num_agents = int(self.core_env.num_agents)
        self.agents = [f"agent_{index}" for index in range(self._num_agents)]
        self.possible_agents = list(self.agents)
        self._agent_ids = set(self.agents)

        self.single_observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=self.core_env.observation_space.low[0],
                    high=self.core_env.observation_space.high[0],
                    dtype=np.float32,
                )
            }
        )
        self.single_action_space = spaces.Box(
            low=self.core_env.action_space.low[0],
            high=self.core_env.action_space.high[0],
            dtype=np.float32,
        )

        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

    @staticmethod
    def _build_core_env(env_config: dict) -> MultiAgentDMPEnv:
        core_kwargs = dict(env_config.get("core_env_kwargs", env_config.get("core_kwargs", {})))

        env_config_value = core_kwargs.get("env_config")
        if isinstance(env_config_value, dict):
            core_kwargs["env_config"] = MultiAgentEnvConfig(**env_config_value)

        dmp_config_value = core_kwargs.get("dmp_config")
        if isinstance(dmp_config_value, dict):
            core_kwargs["dmp_config"] = DMPConfig(**dmp_config_value)

        return MultiAgentDMPEnv(**core_kwargs)

    def _pack_obs(self, observation: np.ndarray) -> dict:
        return {
            agent_id: {"obs": observation[index].astype(np.float32, copy=True)}
            for index, agent_id in enumerate(self.agents)
        }

    def reset(self, *, seed=None, options=None):
        observation, info = self.core_env.reset(seed=seed, options=options)
        obs_dict = self._pack_obs(observation)
        info_dict = {
            agent_id: dict(info)
            for agent_id in self.agents
        }
        if self._new_step_api:
            return obs_dict, info_dict
        return obs_dict

    def step(self, action_dict: dict):
        action_matrix = np.stack(
            [
                np.asarray(action_dict[agent_id], dtype=np.float32)
                for agent_id in self.agents
            ],
            axis=0,
        )

        observation, rewards, terminated, truncated, info = self.core_env.step(action_matrix)

        obs_dict = self._pack_obs(observation)
        reward_dict = {
            agent_id: float(rewards[index])
            for index, agent_id in enumerate(self.agents)
        }

        terminated_dict = {agent_id: bool(terminated) for agent_id in self.agents}
        terminated_dict["__all__"] = bool(terminated)

        truncated_dict = {agent_id: bool(truncated) for agent_id in self.agents}
        truncated_dict["__all__"] = bool(truncated)

        info_dict = {
            agent_id: dict(info)
            for agent_id in self.agents
        }

        if not self._new_step_api:
            done = bool(terminated or truncated)
            done_dict = {agent_id: done for agent_id in self.agents}
            done_dict["__all__"] = done
            return obs_dict, reward_dict, done_dict, info_dict

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def get_env_info(self) -> dict:
        return {
            "num_agents": self._num_agents,
            "episode_limit": int(self.core_env.env_config.max_steps),
            "space_obs": self.single_observation_space,
            "space_act": self.single_action_space,
            "policy_mapping_info": {
                "all_scenario": {
                    "team_prefix": ("agent",),
                    "all_agents_one_policy": True,
                    "one_agent_one_policy": True,
                }
            },
        }

    def close(self):
        if hasattr(self.core_env, "close"):
            return self.core_env.close()
        return None
