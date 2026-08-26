from __future__ import annotations

import unittest

import numpy as np

from Controller.dmp_rl import DMPConfig
from Entity.KinematicModel import PartialDynamic, propagate_point_mass
from Environment.frozen_sac_dmp_execution import propagate_sac_dmp_action
from Environment.single_agent_dmp_env import EnvConfig, SingleAgentDMPEnv
from planning.historical_forcing_gate import propagate_historical_sac_dmp_action


class OperationalSpeedNormContractTest(unittest.TestCase):
    def test_disabled_norm_cap_preserves_legacy_equations(self) -> None:
        position = np.asarray([1.0, 2.0, 3.0])
        velocity = np.asarray([3.0, 2.0, 0.0])
        acceleration = np.asarray([4.0, -1.0, 0.5])
        result = propagate_point_mass(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            dt=0.1,
            acceleration_min=-4.0,
            acceleration_max=4.0,
            velocity_min=-4.0,
            velocity_max=4.0,
        )
        np.testing.assert_allclose(result["velocity"], [3.4, 1.9, 0.05])
        np.testing.assert_allclose(
            result["position"], position + velocity * 0.1 + 0.5 * acceleration * 0.1**2
        )
        self.assertFalse(result["speed_norm_clipped"])

    def test_enabled_norm_cap_is_isotropic(self) -> None:
        position = np.zeros(3)
        velocity = np.asarray([3.0, 0.0, 0.0])
        result = propagate_point_mass(
            position=position,
            velocity=velocity,
            acceleration=np.asarray([4.0, 4.0, 0.0]),
            dt=0.1,
            acceleration_min=-4.0,
            acceleration_max=4.0,
            velocity_min=-4.0,
            velocity_max=4.0,
            maximum_speed_norm=3.2,
        )
        self.assertAlmostEqual(float(np.linalg.norm(result["velocity"])), 3.2, places=12)
        expected_position = 0.5 * (velocity + result["velocity"]) * 0.1
        np.testing.assert_allclose(result["position"], expected_position)
        self.assertTrue(result["speed_norm_clipped"])

    def test_partial_dynamic_and_sac_preview_share_the_same_cap(self) -> None:
        config = {
            "velocity_clip": [-4.0, 4.0],
            "accelerate_clip": [-4.0, 4.0],
            "maximum_speed_norm": 3.2,
            "time_step": 0.1,
        }
        real = PartialDynamic(config)
        real.reset({"position": [1.0, 1.0, 2.0], "velocity": [3.0, 0.0, 0.0]})
        dmp = DMPConfig(dt=0.1, dims=3)
        preview = propagate_sac_dmp_action(
            position=real.p.copy(),
            velocity=real.v.copy(),
            phase=1.0,
            active_goal=np.asarray([80.0, 20.0, 2.0]),
            terminal_goal=np.asarray([80.0, 20.0, 2.0]),
            action=np.zeros(6, dtype=np.float32),
            dmp_config=dmp,
            dynamics=real,
        )
        real.step(preview.commanded_acceleration)
        np.testing.assert_allclose(real.p, preview.position)
        np.testing.assert_allclose(real.v, preview.velocity)
        self.assertLessEqual(float(np.linalg.norm(real.v)), 3.2 + 1.0e-12)

    def test_historical_checkpoint_transition_uses_the_same_optional_cap(self) -> None:
        dynamics = PartialDynamic(
            {
                "velocity_clip": [-4.0, 4.0],
                "accelerate_clip": [-4.0, 4.0],
                "maximum_speed_norm": 3.2,
                "time_step": 0.1,
            }
        )
        dynamics.reset({"position": [1.0, 1.0, 2.0], "velocity": [3.1, 0.0, 0.0]})
        transition = propagate_historical_sac_dmp_action(
            position=dynamics.p.copy(),
            velocity=dynamics.v.copy(),
            phase=1.0,
            active_goal=np.asarray([80.0, 20.0, 2.0]),
            terminal_goal=np.asarray([80.0, 20.0, 2.0]),
            action=np.zeros(6, dtype=np.float32),
            dmp_config=DMPConfig(dt=0.1, dims=3),
            dynamics=dynamics,
        )
        self.assertLessEqual(float(np.linalg.norm(transition.velocity)), 3.2 + 1.0e-12)

    def test_single_agent_optional_transition_uses_historical_gate_and_cap(self) -> None:
        env = SingleAgentDMPEnv(
            dynamics_config={
                "velocity_clip": [-4.0, 4.0],
                "accelerate_clip": [-4.0, 4.0],
                "maximum_speed_norm": 3.2,
                "time_step": 0.1,
            },
            sensor_config={
                "sensing_radius": 4.5,
                "azimuth_bins": 8,
                "elevation_bins": 7,
                "include_previous_scan": True,
            },
            dmp_config=DMPConfig(dt=0.1, dims=3),
            env_config=EnvConfig(
                max_steps=5,
                workspace_bounds=((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0)),
            ),
            transition_function=propagate_historical_sac_dmp_action,
        )
        env.reset(options={"start": [0.0, 0.0, 0.0], "goal": [8.0, 0.0, 0.0]})
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        self.assertEqual(info["forcing_gate"].shape, (3,))
        self.assertLessEqual(float(np.linalg.norm(env.dynamics.v)), 3.2 + 1.0e-12)
        self.assertEqual(
            env.latest_controller_info["forcing_gate_semantics"],
            "historical_vector_goal_eff_gate",
        )
        env.close()

    def test_invalid_cap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            PartialDynamic(
                {
                    "velocity_clip": [-4.0, 4.0],
                    "accelerate_clip": [-4.0, 4.0],
                    "maximum_speed_norm": 0.0,
                    "time_step": 0.1,
                }
            )


if __name__ == "__main__":
    unittest.main()
