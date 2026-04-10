import unittest

import numpy as np

from Controller.dmp_rl import DMPConfig, HeuristicDMPPolicy, SecondOrderDMPController
from Entity.dynamic_obstacles import MovingSphereObstacle
from Entity.sensors import LocalObstacleSensor
from Entity.static_obstacles import AxisAlignedBoxObstacle, StaticSphereObstacle
from Environment.single_agent_dmp_env import SingleAgentDMPEnv


class TestDMPRLPipeline(unittest.TestCase):
    def setUp(self):
        self.static_obstacles = [
            StaticSphereObstacle(center=[2.5, 0.4, 0.0], radius=0.6, safety_margin=0.1),
            AxisAlignedBoxObstacle(center=[4.5, -0.3, 0.0], half_extents=[0.5, 0.5, 0.3], safety_margin=0.1),
        ]
        self.dynamic_obstacles = [
            MovingSphereObstacle(
                center=[1.8, -1.2, 0.0],
                radius=0.3,
                velocity=[0.0, 0.5, 0.0],
                safety_margin=0.1,
            )
        ]
        self.sensor = LocalObstacleSensor(sensing_radius=5.0, max_obstacles=4)
        self.controller = SecondOrderDMPController(DMPConfig(dt=0.1))
        self.controller.reset(start=[0.0, 0.0, 0.0], goal=[6.0, 0.0, 0.0])

    def test_sensor_packet_shape(self):
        packet = self.sensor.sense(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            goal=np.array([6.0, 0.0, 0.0]),
            static_obstacles=self.static_obstacles,
            dynamic_obstacles=self.dynamic_obstacles,
        )
        self.assertEqual(packet.obstacle_features.shape, (4, 9))
        self.assertEqual(packet.observation.shape[0], self.sensor.observation_dim)

    def test_controller_output(self):
        packet = self.sensor.sense(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            goal=np.array([6.0, 0.0, 0.0]),
            static_obstacles=self.static_obstacles,
            dynamic_obstacles=self.dynamic_obstacles,
        )
        action = np.zeros(6, dtype=float)
        acceleration, info = self.controller.compute_acceleration(
            position=np.zeros(3),
            velocity=np.zeros(3),
            rl_action=action,
            sensor_packet=packet,
        )
        self.assertEqual(acceleration.shape, (3,))
        self.assertIn("phase", info)
        self.assertIn("forcing", info)
        self.assertIn("tau", info)

    def test_heuristic_policy_shape(self):
        packet = self.sensor.sense(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            goal=np.array([6.0, 0.0, 0.0]),
            static_obstacles=self.static_obstacles,
            dynamic_obstacles=self.dynamic_obstacles,
        )
        policy = HeuristicDMPPolicy(goal_offset_max=0.8)
        action = policy.act(packet)
        self.assertEqual(action.shape, (6,))
        self.assertTrue(np.all(np.isfinite(action)))

    def test_single_agent_env_step(self):
        env = SingleAgentDMPEnv(
            dynamics_config={
                "velocity_clip": (-2.0, 2.0),
                "accelerate_clip": (-4.0, 4.0),
                "time_step": 0.1,
            },
            sensor_config={"sensing_radius": 5.0, "max_obstacles": 4},
            dmp_config=DMPConfig(dt=0.1),
            static_obstacles=self.static_obstacles,
            dynamic_obstacles=self.dynamic_obstacles,
        )
        observation = env.reset(start=np.array([0.0, 0.0, 0.0]), goal=np.array([6.0, 0.0, 0.0]))
        self.assertEqual(observation.shape[0], env.observation_dim)
        self.assertEqual(env.get_sensor_observation().shape[0], env.sensor_observation_dim)

        action = np.zeros(env.action_dim, dtype=float)
        next_observation, reward, done, info = env.step(action)
        self.assertEqual(next_observation.shape[0], env.observation_dim)
        self.assertEqual(info["sensor_observation"].shape[0], env.sensor_observation_dim)
        self.assertIsInstance(reward, float)
        self.assertIn("distance_to_goal", info)
        self.assertFalse(np.isnan(reward))
        self.assertIsInstance(done, bool)


if __name__ == "__main__":
    unittest.main()
