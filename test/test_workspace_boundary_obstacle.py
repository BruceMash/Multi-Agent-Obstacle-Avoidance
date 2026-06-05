import unittest

import numpy as np

from Entity.static_obstacles import WorkspaceBoundaryPlaneObstacle


class TestWorkspaceBoundaryPlaneObstacle(unittest.TestCase):
    def setUp(self):
        self.lower = np.array([-0.5, -2.5, -1.2], dtype=float)
        self.upper = np.array([8.5, 2.0, 1.2], dtype=float)

    def test_lower_boundary_signed_distance(self):
        obstacle = WorkspaceBoundaryPlaneObstacle(
            axis=1,
            bound=self.lower[1],
            lower_bounds=self.lower,
            upper_bounds=self.upper,
            is_lower=True,
        )

        self.assertGreater(obstacle.signed_distance([0.0, -2.0, 0.0]), 0.0)
        self.assertAlmostEqual(obstacle.signed_distance([0.0, -2.5, 0.0]), 0.0)
        self.assertLess(obstacle.signed_distance([0.0, -2.6, 0.0]), 0.0)

    def test_upper_boundary_signed_distance(self):
        obstacle = WorkspaceBoundaryPlaneObstacle(
            axis=2,
            bound=self.upper[2],
            lower_bounds=self.lower,
            upper_bounds=self.upper,
            is_lower=False,
        )

        self.assertGreater(obstacle.signed_distance([0.0, 0.0, 0.8]), 0.0)
        self.assertAlmostEqual(obstacle.signed_distance([0.0, 0.0, 1.2]), 0.0)
        self.assertLess(obstacle.signed_distance([0.0, 0.0, 1.3]), 0.0)

    def test_ray_hits_boundary_when_pointing_toward_face(self):
        obstacle = WorkspaceBoundaryPlaneObstacle(
            axis=0,
            bound=self.upper[0],
            lower_bounds=self.lower,
            upper_bounds=self.upper,
            is_lower=False,
        )

        distance = obstacle.ray_intersection(
            origin=np.array([8.0, 0.0, 0.0], dtype=float),
            direction=np.array([1.0, 0.0, 0.0], dtype=float),
            max_distance=6.0,
        )

        self.assertAlmostEqual(distance, 0.5)

    def test_ray_misses_boundary_when_pointing_away_from_face(self):
        obstacle = WorkspaceBoundaryPlaneObstacle(
            axis=0,
            bound=self.upper[0],
            lower_bounds=self.lower,
            upper_bounds=self.upper,
            is_lower=False,
        )

        distance = obstacle.ray_intersection(
            origin=np.array([8.0, 0.0, 0.0], dtype=float),
            direction=np.array([-1.0, 0.0, 0.0], dtype=float),
            max_distance=6.0,
        )

        self.assertIsNone(distance)

    def test_ray_misses_finite_face_outside_other_axes(self):
        obstacle = WorkspaceBoundaryPlaneObstacle(
            axis=0,
            bound=self.upper[0],
            lower_bounds=self.lower,
            upper_bounds=self.upper,
            is_lower=False,
        )

        distance = obstacle.ray_intersection(
            origin=np.array([8.0, self.upper[1] + 0.2, 0.0], dtype=float),
            direction=np.array([1.0, 0.0, 0.0], dtype=float),
            max_distance=6.0,
        )

        self.assertIsNone(distance)


if __name__ == "__main__":
    unittest.main()
