import unittest
from collections import deque

import numpy as np

from Environment.multi_agent_dmp_env import MultiAgentDMPEnv, MultiAgentEnvConfig


class TestMASACRewardStructure(unittest.TestCase):
    def build_env(self, **overrides):
        config_values = {
            "num_agents": 3,
            "max_steps": 20,
            "goal_tolerance": 0.3,
            "randomize_start_goal": False,
            "min_start_distance": 0.6,
            "min_goal_distance": 0.0,
            "min_start_goal_distance": 0.0,
            "obstacle_potential_weight": 0.0,
            "boundary_potential_weight": 0.0,
            "inter_agent_potential_weight": 0.0,
            "step_reward_weight": 0.0,
            "individual_success_bonus": 50.0,
            "team_success_bonus": 150.0,
            "team_collision_penalty": 40.0,
            "local_obstacle_collision_penalty": 40.0,
            "local_boundary_collision_penalty": 40.0,
            "local_inter_agent_collision_penalty": 20.0,
            "team_timeout_penalty": 50.0,
            "stagnation_window": 20,
            "stagnation_progress_threshold": 0.08,
            "stagnation_patience": 5,
            "stagnation_penalty_start": 0.5,
            "stagnation_penalty_growth": 0.05,
            "stagnation_penalty_max": 1.5,
        }
        config_values.update(overrides)
        return MultiAgentDMPEnv(
            dynamics_config={
                "velocity_clip": (-4.0, 4.0),
                "accelerate_clip": (-4.0, 4.0),
                "time_step": 0.1,
            },
            sensor_config={
                "sensing_radius": 5.0,
                "azimuth_bins": 4,
                "elevation_bins": 4,
                "include_previous_scan": False,
            },
            dmp_config={"dt": 0.1, "dims": 3},
            env_config=MultiAgentEnvConfig(**config_values),
        )

    @staticmethod
    def separated_points(x_values):
        return np.array(
            [[x_values[index], 0.7 + 1.3 * index, 1.0] for index in range(3)],
            dtype=float,
        )

    def test_observation_exposes_stagnation_state(self):
        env = self.build_env()
        starts = self.separated_points([1.0, 1.0, 1.0])
        goals = self.separated_points([7.0, 7.0, 7.0])
        observation, _ = env.reset(options={"starts": starts, "goals": goals})

        self.assertEqual(env.extra_observation_dim, 5)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertEqual(observation.shape[1], env.single_agent_observation_dim)

    def test_stagnation_penalty_is_delayed_and_resets_on_success(self):
        env = self.build_env()
        starts = self.separated_points([1.0, 1.0, 1.0])
        goals = self.separated_points([7.0, 7.0, 7.0])
        env.reset(options={"starts": starts, "goals": goals})
        env.stagnation_distance_histories = [
            deque([6.0], maxlen=env.env_config.stagnation_window + 1)
            for _ in range(env.num_agents)
        ]

        penalties = np.zeros(env.num_agents, dtype=np.float32)
        for _ in range(env.env_config.stagnation_window + env.env_config.stagnation_patience - 1):
            penalties = env._compute_stagnation_penalties(
                np.full(env.num_agents, 6.0),
                np.zeros(env.num_agents, dtype=bool),
            )

        self.assertTrue(np.allclose(penalties, 0.5))
        success_mask = np.array([True, False, False])
        penalties = env._compute_stagnation_penalties(np.full(env.num_agents, 6.0), success_mask)
        self.assertEqual(float(penalties[0]), 0.0)
        self.assertEqual(int(env.stagnation_counters[0]), 0)

    def test_full_success_combines_individual_and_team_bonuses(self):
        env = self.build_env()
        starts = self.separated_points([1.0, 1.0, 1.0])
        observation, _ = env.reset(options={"starts": starts, "goals": starts.copy()})
        self.assertEqual(observation.shape, env.observation_space.shape)

        _, rewards, terminated, truncated, info = env.step(np.zeros(env.action_shape, dtype=np.float32))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        self.assertTrue(np.allclose(info["reward_individual_success_bonus"], 50.0))
        self.assertTrue(np.allclose(info["reward_team_success_bonus"], 150.0))
        self.assertTrue(np.allclose(rewards, 200.0))

    def test_inter_agent_collision_has_team_and_local_penalties(self):
        env = self.build_env()
        starts = self.separated_points([1.0, 1.0, 1.0])
        goals = self.separated_points([7.0, 7.0, 7.0])
        env.reset(options={"starts": starts, "goals": goals})
        env.dynamics[0].p = np.array([2.0, 1.0, 1.0])
        env.dynamics[1].p = np.array([2.5, 1.0, 1.0])
        env.dynamics[2].p = np.array([2.0, 3.0, 1.0])

        _, _, terminated, _, info = env.step(np.zeros(env.action_shape, dtype=np.float32))

        self.assertTrue(terminated)
        self.assertFalse(info["success"])
        self.assertTrue(np.allclose(info["reward_team_collision_penalty"], 40.0))
        self.assertTrue(np.allclose(info["reward_local_collision_penalty"], [20.0, 20.0, 0.0]))

    def test_timeout_penalty_is_shared(self):
        env = self.build_env(max_steps=1)
        starts = self.separated_points([1.0, 1.0, 1.0])
        goals = self.separated_points([7.0, 7.0, 7.0])
        env.reset(options={"starts": starts, "goals": goals})

        _, _, terminated, truncated, info = env.step(np.zeros(env.action_shape, dtype=np.float32))

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(np.allclose(info["reward_team_timeout_penalty"], 50.0))

    def test_inter_agent_potential_is_capped(self):
        env = self.build_env(
            inter_agent_potential_weight=1.0,
            inter_agent_potential_penalty_max=2.0,
        )
        distances = np.array(
            [[0.0, 0.01, 2.0], [0.01, 0.0, 2.0], [2.0, 2.0, 0.0]],
            dtype=np.float32,
        )

        penalties = env._compute_inter_agent_potential_penalties(distances)

        self.assertTrue(np.all(penalties <= 2.0))
        self.assertTrue(np.allclose(penalties[:2], 2.0))


if __name__ == "__main__":
    unittest.main()
