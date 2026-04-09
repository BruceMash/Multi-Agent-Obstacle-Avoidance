from dataclasses import dataclass

import numpy as np


def _to_vector3(value):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError("value must have shape (3,)")
    return vector


@dataclass
class StaticSphereObstacle:
    """
    静态球形障碍物。
    """

    center: np.ndarray
    radius: float
    safety_margin: float = 0.0

    def __post_init__(self):
        self.center = _to_vector3(self.center)
        self.radius = float(self.radius)
        self.safety_margin = float(self.safety_margin)
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")

    @property
    def velocity(self):
        return np.zeros(3, dtype=float)

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

    def to_feature(self, point):
        closest = self.closest_point(point)
        return {
            "closest_point": closest,
            "center": self.center.copy(),
            "velocity": self.velocity.copy(),
            "clearance": self.signed_distance(point),
            "size": self.effective_radius,
        }


@dataclass
class AxisAlignedBoxObstacle:
    """
    静态长方体障碍物，使用中心点和半边长描述。
    """

    center: np.ndarray
    half_extents: np.ndarray
    safety_margin: float = 0.0

    def __post_init__(self):
        self.center = _to_vector3(self.center)
        self.half_extents = _to_vector3(self.half_extents)
        self.safety_margin = float(self.safety_margin)
        if np.any(self.half_extents <= 0.0):
            raise ValueError("half_extents must be positive")

    @property
    def velocity(self):
        return np.zeros(3, dtype=float)

    @property
    def expanded_half_extents(self):
        return self.half_extents + self.safety_margin

    @property
    def effective_radius(self):
        return float(np.linalg.norm(self.expanded_half_extents))

    def signed_distance(self, point):
        point = _to_vector3(point)
        q = np.abs(point - self.center) - self.expanded_half_extents
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(np.max(q), 0.0)
        return outside + inside

    def contains(self, point, margin=0.0):
        return self.signed_distance(point) <= float(margin)

    def closest_point(self, point):
        point = _to_vector3(point)
        lower = self.center - self.expanded_half_extents
        upper = self.center + self.expanded_half_extents
        return np.clip(point, lower, upper)

    def to_feature(self, point):
        closest = self.closest_point(point)
        return {
            "closest_point": closest,
            "center": self.center.copy(),
            "velocity": self.velocity.copy(),
            "clearance": self.signed_distance(point),
            "size": self.effective_radius,
        }
