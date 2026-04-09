from dataclasses import dataclass

import numpy as np

from Entity.static_obstacles import _to_vector3


@dataclass
class MovingSphereObstacle:
    """
    动态球形障碍物。
    """

    center: np.ndarray
    radius: float
    velocity: np.ndarray
    safety_margin: float = 0.0
    bounds: tuple | None = None

    def __post_init__(self):
        self.center = _to_vector3(self.center)
        self.velocity = _to_vector3(self.velocity)
        self.radius = float(self.radius)
        self.safety_margin = float(self.safety_margin)
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")

        if self.bounds is not None:
            lower, upper = self.bounds
            self.bounds = (_to_vector3(lower), _to_vector3(upper))

    @property
    def effective_radius(self):
        return self.radius + self.safety_margin

    def signed_distance(self, point):
        point = _to_vector3(point)
        return np.linalg.norm(point - self.center) - self.effective_radius

    def contains(self, point, margin=0.0):
        return self.signed_distance(point) <= float(margin)

    def closest_point(self, point):
        point = _to_vector3(point)
        direction = point - self.center
        distance = np.linalg.norm(direction)
        if distance < 1e-8:
            direction = np.array([1.0, 0.0, 0.0], dtype=float)
            distance = 1.0
        return self.center + direction / distance * self.effective_radius

    def step(self, dt):
        self.center = self.center + self.velocity * float(dt)

        if self.bounds is None:
            return self.center.copy()

        lower, upper = self.bounds
        for dim in range(3):
            min_bound = lower[dim] + self.effective_radius
            max_bound = upper[dim] - self.effective_radius
            if self.center[dim] < min_bound:
                self.center[dim] = min_bound
                self.velocity[dim] *= -1.0
            elif self.center[dim] > max_bound:
                self.center[dim] = max_bound
                self.velocity[dim] *= -1.0

        return self.center.copy()

    def to_feature(self, point):
        closest = self.closest_point(point)
        return {
            "closest_point": closest,
            "center": self.center.copy(),
            "velocity": self.velocity.copy(),
            "clearance": self.signed_distance(point),
            "size": self.effective_radius,
        }
