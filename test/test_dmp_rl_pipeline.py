import unittest

import numpy as np

from Controller.dmp_rl import DMPConfig, HeuristicDMPPolicy, SecondOrderDMPController
from Entity.dynamic_obstacles import MovingSphereObstacle
from Entity.sensors import LocalObstacleSensor
from Entity.static_obstacles import (
    AxisAlignedBoxObstacle,
    StaticCylinderObstacle,
    StaticSphereObstacle,
    WorkspaceBoundaryPlaneObstacle,
)
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
        self.sensor = LocalObstacleSensor(sensing_radius=5.0)
        self.controller = SecondOrderDMPController(DMPConfig(dt=0.1))
        self.controller.reset(start=[0.0, 0.0, 0.0], goal=[6.0, 0.0, 0.0])

    def test_sensor_packet_shape(self):
        self.sensor.reset()
        packet = self.sensor.sense(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            goal=np.array([6.0, 0.0, 0.0]),
            static_obstacles=self.static_obstacles,
            dynamic_obstacles=self.dynamic_obstacles,
        )
        self.assertEqual(packet.current_scan.shape, (24, 9))
        self.assertEqual(packet.previous_scan.shape, (24, 9))
        self.assertEqual(packet.observation.shape[0], 439)
        self.assertEqual(packet.observation.shape[0], self.sensor.observation_dim)
        self.assertTrue(np.all(packet.current_scan >= 0.0))
        self.assertTrue(np.all(packet.current_scan <= 1.0))
        self.assertTrue(np.all(packet.previous_scan >= 0.0))
        self.assertTrue(np.all(packet.previous_scan <= 1.0))
        self.assertTrue(np.allclose(packet.current_scan, packet.previous_scan))

    def test_sensor_temporal_scan_update(self):
        moving_obstacle = MovingSphereObstacle(
            center=[4.0, 0.0, 0.0],
            radius=0.4,
            velocity=[-1.0, 0.0, 0.0],
            safety_margin=0.0,
        )
        self.sensor.reset()
        packet_first = self.sensor.sense(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            goal=np.array([6.0, 0.0, 0.0]),
            dynamic_obstacles=[moving_obstacle],
        )
        moving_obstacle.step(1.0)
        packet_second = self.sensor.sense(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            goal=np.array([6.0, 0.0, 0.0]),
            dynamic_obstacles=[moving_obstacle],
        )
        self.assertTrue(np.allclose(packet_second.previous_scan, packet_first.current_scan))
        self.assertLess(packet_second.min_clearance, packet_first.min_clearance)
        self.assertFalse(np.allclose(packet_second.current_scan, packet_first.current_scan))

    def test_sensor_current_only_observation_keeps_internal_history(self):
        sensor = LocalObstacleSensor(
            sensing_radius=5.0,
            azimuth_bins=16,
            elevation_bins=16,
            include_previous_scan=False,
        )
        sensor.reset()
        packet_first = sensor.sense(
            position=np.zeros(3),
            velocity=np.zeros(3),
            goal=np.array([6.0, 0.0, 0.0]),
            static_obstacles=self.static_obstacles,
        )
        packet_second = sensor.sense(
            position=np.array([0.1, 0.0, 0.0]),
            velocity=np.zeros(3),
            goal=np.array([6.0, 0.0, 0.0]),
            static_obstacles=self.static_obstacles,
        )

        self.assertEqual(packet_first.current_scan.shape, (16, 16))
        self.assertEqual(packet_first.previous_scan.shape, (16, 16))
        self.assertEqual(sensor.n_rays, 256)
        self.assertEqual(sensor.observation_dim, 263)
        self.assertEqual(packet_first.observation.shape, (263,))
        self.assertTrue(np.allclose(packet_second.previous_scan, packet_first.current_scan))

    def test_broad_phase_preserves_exact_scan_values(self):
        lower = np.array([-1.0, -2.0, -1.5])
        upper = np.array([7.0, 2.0, 1.5])
        boundaries = [
            WorkspaceBoundaryPlaneObstacle(axis, bound, lower, upper, is_lower)
            for axis in range(3)
            for is_lower, bound in ((True, lower[axis]), (False, upper[axis]))
        ]
        obstacles = [
            StaticSphereObstacle(center=[2.2, 0.4, 0.1], radius=0.45, safety_margin=0.1),
            AxisAlignedBoxObstacle(
                center=[4.0, -0.5, 0.0],
                half_extents=[0.5, 0.35, 0.4],
                safety_margin=0.05,
            ),
            StaticCylinderObstacle(
                center=[5.2, 0.8, 0.0],
                radius=0.35,
                half_height=0.7,
                safety_margin=0.05,
            ),
            MovingSphereObstacle(
                center=[3.2, -0.8, 0.4],
                radius=0.3,
                velocity=[0.0, 0.2, 0.0],
                safety_margin=0.05,
            ),
            *boundaries,
        ]
        sensor_kwargs = dict(
            sensing_radius=5.0,
            azimuth_bins=16,
            elevation_bins=16,
            include_previous_scan=False,
        )
        exact_sensor = LocalObstacleSensor(**sensor_kwargs, broad_phase_enabled=False)
        broad_sensor = LocalObstacleSensor(**sensor_kwargs, broad_phase_enabled=True)

        for position in (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.1, -0.4, 0.3]),
            np.array([3.0, 0.2, -0.2]),
            np.array([-1.2, 0.0, 0.0]),
        ):
            exact_packet = exact_sensor.sense(
                position=position,
                velocity=np.zeros(3),
                goal=np.array([6.0, 0.0, 0.0]),
                static_obstacles=obstacles,
            )
            broad_packet = broad_sensor.sense(
                position=position,
                velocity=np.zeros(3),
                goal=np.array([6.0, 0.0, 0.0]),
                static_obstacles=obstacles,
            )
            self.assertTrue(np.array_equal(broad_packet.current_scan, exact_packet.current_scan))
            self.assertTrue(np.array_equal(broad_packet.observation, exact_packet.observation))

    def test_broad_phase_reduces_exact_intersection_calls(self):
        class CountingSphere(StaticSphereObstacle):
            def __init__(self):
                super().__init__(center=[3.0, 0.0, 0.0], radius=0.8)
                self.calls = 0

            def ray_intersection(self, origin, direction, max_distance):
                self.calls += 1
                return super().ray_intersection(origin, direction, max_distance)

        obstacle = CountingSphere()
        sensor = LocalObstacleSensor(
            sensing_radius=5.0,
            azimuth_bins=16,
            elevation_bins=16,
            broad_phase_enabled=True,
        )
        sensor.sense(
            position=np.zeros(3),
            velocity=np.zeros(3),
            goal=np.array([6.0, 0.0, 0.0]),
            static_obstacles=[obstacle],
        )

        self.assertGreater(obstacle.calls, 0)
        self.assertLess(obstacle.calls, sensor.n_rays)

    def test_broad_phase_falls_back_for_unknown_obstacle_type(self):
        class UnknownObstacle:
            def __init__(self):
                self.calls = 0

            def ray_intersection(self, origin, direction, max_distance):
                self.calls += 1
                return None

        obstacle = UnknownObstacle()
        sensor = LocalObstacleSensor(
            sensing_radius=5.0,
            azimuth_bins=16,
            elevation_bins=16,
            broad_phase_enabled=True,
        )
        sensor._scan_obstacles(np.zeros(3), [obstacle])

        self.assertEqual(obstacle.calls, sensor.n_rays)

    def test_sphere_ray_intersection(self):
        obstacle = StaticSphereObstacle(center=[3.0, 0.0, 0.0], radius=0.5)
        distance = obstacle.ray_intersection(
            origin=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            max_distance=10.0,
        )
        self.assertAlmostEqual(distance, 2.5, places=6)

    def test_box_ray_intersection(self):
        obstacle = AxisAlignedBoxObstacle(center=[4.0, 0.0, 0.0], half_extents=[1.0, 0.5, 0.5])
        distance = obstacle.ray_intersection(
            origin=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            max_distance=10.0,
        )
        self.assertAlmostEqual(distance, 3.0, places=6)

    def test_lidar_occlusion_prefers_nearest_hit(self):
        sensor = LocalObstacleSensor(sensing_radius=5.0)
        packet = sensor.sense(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            goal=np.array([6.0, 0.0, 0.0]),
            static_obstacles=[
                StaticSphereObstacle(center=[2.0, 0.0, 0.0], radius=0.3),
                StaticSphereObstacle(center=[4.0, 0.0, 0.0], radius=0.3),
            ],
        )
        front_azimuth_index = 12
        center_elevation_index = 4
        expected_normalized_distance = 1.7 / 5.0
        self.assertAlmostEqual(
            float(packet.current_scan[front_azimuth_index, center_elevation_index]),
            expected_normalized_distance,
            places=5,
        )

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
            sensor_config={"sensing_radius": 5.0},
            dmp_config=DMPConfig(dt=0.1),
            static_obstacles=self.static_obstacles,
            dynamic_obstacles=self.dynamic_obstacles,
        )
        observation, info = env.reset(
            options={
                "start": np.array([0.0, 0.0, 0.0]),
                "goal": np.array([6.0, 0.0, 0.0]),
            }
        )
        self.assertEqual(env.sensor_observation_dim, 439)
        self.assertEqual(env.extra_observation_dim, 3)
        self.assertEqual(env.observation_dim, 442)
        self.assertEqual(observation.shape[0], env.observation_dim)
        self.assertEqual(env.get_sensor_observation().shape[0], env.sensor_observation_dim)
        self.assertEqual(env.observation_space.shape[0], env.observation_dim)
        self.assertEqual(env.action_space.shape[0], env.action_dim)
        self.assertIn("distance_to_goal", info)

        action = np.zeros(env.action_dim, dtype=np.float32)
        next_observation, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(next_observation.shape[0], env.observation_dim)
        self.assertEqual(info["sensor_observation"].shape[0], env.sensor_observation_dim)
        self.assertIsInstance(reward, float)
        self.assertIn("distance_to_goal", info)
        self.assertFalse(np.isnan(reward))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)


if __name__ == "__main__":
    unittest.main()
