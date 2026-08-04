import unittest

import numpy as np

from Environment.multi_agent_continuous_env import (
    MultiAgentContinuousEnv,
    MultiAgentContinuousEnvConfig,
)
from Environment.multi_agent_dmp_env import (
    MultiAgentDMPEnv,
    MultiAgentEnvConfig,
)


class TestMultiAgentContinuousEnv(unittest.TestCase):
    @staticmethod
    def separated_points(x_value: float) -> np.ndarray:
        return np.array(
            [
                [x_value, 0.7, 1.0],
                [x_value, 2.0, 1.0],
                [x_value, 3.3, 1.0],
            ],
            dtype=float,
        )

    def build_env(
        self,
        *,
        azimuth_bins: int = 4,
        elevation_bins: int = 4,
        **overrides,
    ) -> MultiAgentContinuousEnv:
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
            "step_reward_weight": 4.0,
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
        return MultiAgentContinuousEnv(
            dynamics_config={
                "velocity_clip": (-4.0, 4.0),
                "accelerate_clip": (-4.0, 4.0),
                "time_step": 0.1,
            },
            sensor_config={
                "sensing_radius": 5.0,
                "azimuth_bins": azimuth_bins,
                "elevation_bins": elevation_bins,
                "include_previous_scan": False,
            },
            env_config=MultiAgentContinuousEnvConfig(**config_values),
        )

    def reset_to_standard_scene(
        self,
        env: MultiAgentContinuousEnv,
    ) -> tuple[np.ndarray, dict]:
        return env.reset(
            options={
                "starts": self.separated_points(1.0),
                "goals": self.separated_points(7.0),
            }
        )

    def test_action_and_training_observation_shapes(self):
        env = self.build_env(azimuth_bins=16, elevation_bins=16)
        observation, _ = self.reset_to_standard_scene(env)

        self.assertEqual(env.action_space.shape, (3, 3))
        self.assertEqual(env.action_shape, (3, 3))
        self.assertEqual(env.sensor_observation_dim, 263)
        self.assertEqual(env.extra_observation_dim, 2)
        self.assertEqual(env.inter_agent_observation_dim, 14)
        self.assertEqual(env.observation_space.shape, (3, 279))
        self.assertEqual(observation.shape, (3, 279))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(observation))

    def test_environment_contains_no_dmp_runtime_state_or_info(self):
        env = self.build_env()
        self.assertFalse(hasattr(env, "dmps"))
        self.assertFalse(hasattr(env, "dmp_config"))
        self.assertFalse(hasattr(env, "latest_controller_infos"))

        observation, reset_info = self.reset_to_standard_scene(env)
        self.assertEqual(observation.shape, env.observation_space.shape)
        _, _, _, _, step_info = env.step(
            np.zeros(env.action_shape, dtype=np.float32)
        )

        dmp_info_fields = {
            "phases",
            "taus",
            "guided_action",
            "action_guidance_weights",
            "action_guidance_weight",
        }
        self.assertTrue(dmp_info_fields.isdisjoint(reset_info))
        self.assertTrue(dmp_info_fields.isdisjoint(step_info))

    def test_zero_acceleration_preserves_stationary_state(self):
        env = self.build_env()
        self.reset_to_standard_scene(env)
        positions_before = env._positions().copy()

        _, _, terminated, truncated, info = env.step(
            np.zeros(env.action_shape, dtype=np.float32)
        )

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        np.testing.assert_allclose(env._positions(), positions_before)
        np.testing.assert_allclose(env._velocities(), 0.0)
        np.testing.assert_allclose(info["commanded_accelerations"], 0.0)
        np.testing.assert_allclose(info["applied_accelerations"], 0.0)
        self.assertFalse(np.any(info["acceleration_clip_mask"]))

    def test_acceleration_is_clipped_before_dynamics_step(self):
        env = self.build_env()
        self.reset_to_standard_scene(env)
        action = np.array(
            [
                [10.0, -10.0, 2.0],
                [0.0, 0.0, 0.0],
                [-4.0, 4.0, -5.0],
            ],
            dtype=np.float32,
        )

        _, _, _, _, info = env.step(action)

        expected_applied = np.array(
            [
                [4.0, -4.0, 2.0],
                [0.0, 0.0, 0.0],
                [-4.0, 4.0, -4.0],
            ],
            dtype=np.float32,
        )
        expected_mask = np.array(
            [
                [True, True, False],
                [False, False, False],
                [False, False, True],
            ]
        )
        np.testing.assert_allclose(info["commanded_accelerations"], action)
        np.testing.assert_allclose(
            info["applied_accelerations"],
            expected_applied,
        )
        np.testing.assert_array_equal(
            info["acceleration_clip_mask"],
            expected_mask,
        )
        np.testing.assert_allclose(
            env._velocities(),
            0.1 * expected_applied,
        )

    def test_reset_seed_is_reproducible(self):
        env_1 = self.build_env(randomize_start_goal=True)
        env_2 = self.build_env(randomize_start_goal=True)

        observation_1, info_1 = env_1.reset(seed=20260728)
        observation_2, info_2 = env_2.reset(seed=20260728)

        np.testing.assert_allclose(info_1["starts"], info_2["starts"])
        np.testing.assert_allclose(info_1["goals"], info_2["goals"])
        np.testing.assert_allclose(observation_1, observation_2)

    def test_full_success_uses_existing_bonus_and_termination_logic(self):
        env = self.build_env(step_reward_weight=0.0)
        starts = self.separated_points(1.0)
        env.reset(options={"starts": starts, "goals": starts.copy()})

        _, rewards, terminated, truncated, info = env.step(
            np.zeros(env.action_shape, dtype=np.float32)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        np.testing.assert_allclose(
            info["reward_individual_success_bonus"],
            50.0,
        )
        np.testing.assert_allclose(
            info["reward_team_success_bonus"],
            150.0,
        )
        np.testing.assert_allclose(rewards, 200.0)

    def test_collision_uses_existing_team_and_local_penalty_logic(self):
        env = self.build_env(step_reward_weight=0.0)
        self.reset_to_standard_scene(env)
        env.dynamics[0].p = np.array([2.0, 1.0, 1.0])
        env.dynamics[1].p = np.array([2.5, 1.0, 1.0])
        env.dynamics[2].p = np.array([2.0, 3.0, 1.0])

        _, rewards, terminated, truncated, info = env.step(
            np.zeros(env.action_shape, dtype=np.float32)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["success"])
        np.testing.assert_allclose(
            info["reward_team_collision_penalty"],
            40.0,
        )
        np.testing.assert_allclose(
            info["reward_local_collision_penalty"],
            [20.0, 20.0, 0.0],
        )
        np.testing.assert_allclose(rewards, [-60.0, -60.0, -40.0])

    def test_timeout_uses_existing_shared_penalty_logic(self):
        env = self.build_env(max_steps=1, step_reward_weight=0.0)
        self.reset_to_standard_scene(env)

        _, rewards, terminated, truncated, info = env.step(
            np.zeros(env.action_shape, dtype=np.float32)
        )

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        np.testing.assert_allclose(
            info["reward_team_timeout_penalty"],
            50.0,
        )
        np.testing.assert_allclose(rewards, -50.0)

    def test_progress_and_acceleration_reward_diagnostics_are_available(self):
        env = self.build_env(
            step_reward_weight=4.0,
            acceleration_penalty_weight=0.01,
            acceleration_clip_penalty_weight=0.05,
        )
        self.reset_to_standard_scene(env)
        action = np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        _, rewards, terminated, truncated, info = env.step(action)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(np.all(info["progress"] > 0.0))
        np.testing.assert_allclose(
            info["reward_progress"],
            4.0 * info["progress"],
        )
        np.testing.assert_allclose(
            info["reward_acceleration_penalty"],
            [0.01, 0.04, 0.16],
        )
        np.testing.assert_allclose(
            info["reward_acceleration_clip_penalty"],
            [0.0, 0.0, 0.05],
        )
        np.testing.assert_allclose(rewards, info["reward_progress"])

    def test_existing_dmp_environment_interface_is_unchanged(self):
        env = MultiAgentDMPEnv(
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
            env_config=MultiAgentEnvConfig(
                num_agents=3,
                randomize_start_goal=False,
                min_start_goal_distance=0.0,
            ),
        )
        observation, _ = env.reset(
            options={
                "starts": self.separated_points(1.0),
                "goals": self.separated_points(7.0),
            }
        )

        self.assertEqual(env.action_shape, (3, 6))
        self.assertEqual(env.extra_observation_dim, 5)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertTrue(hasattr(env, "dmps"))
        self.assertTrue(hasattr(env, "dmp_config"))


if __name__ == "__main__":
    unittest.main()
