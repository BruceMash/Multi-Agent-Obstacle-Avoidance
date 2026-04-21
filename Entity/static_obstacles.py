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

    def ray_intersection(self, origin, direction, max_distance):
        """
        计算射线与球形障碍物的最近正向交点距离。
        若无交点则返回 None。
        """
        origin = _to_vector3(origin)
        direction = _to_vector3(direction)
        max_distance = float(max_distance)

        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-8:
            raise ValueError("direction must be non-zero")
        direction = direction / direction_norm

        if self.contains(origin):
            return 0.0

        offset = origin - self.center
        b = float(np.dot(direction, offset))
        c = float(np.dot(offset, offset) - self.effective_radius ** 2)
        discriminant = b * b - c
        if discriminant < 0.0:
            return None

        sqrt_discriminant = np.sqrt(discriminant)
        candidates = [-b - sqrt_discriminant, -b + sqrt_discriminant]
        positive_candidates = [distance for distance in candidates if distance >= 0.0]
        if not positive_candidates:
            return None

        hit_distance = min(positive_candidates)
        if hit_distance > max_distance:
            return None
        return float(hit_distance)

    def to_feature(self, point):    # 转化为特征
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

    def ray_intersection(self, origin, direction, max_distance):
        """
        计算射线与轴对齐长方体的最近正向交点距离。
        使用 slab 法求交。
        """
        origin = _to_vector3(origin)
        direction = _to_vector3(direction)
        max_distance = float(max_distance)

        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-8:
            raise ValueError("direction must be non-zero")
        direction = direction / direction_norm

        if self.contains(origin):
            return 0.0

        lower = self.center - self.expanded_half_extents
        upper = self.center + self.expanded_half_extents
        t_min = -np.inf
        t_max = np.inf

        for dim in range(3):
            if abs(direction[dim]) < 1e-8:
                if origin[dim] < lower[dim] or origin[dim] > upper[dim]:
                    return None
                continue

            t1 = (lower[dim] - origin[dim]) / direction[dim]
            t2 = (upper[dim] - origin[dim]) / direction[dim]
            near = min(t1, t2)
            far = max(t1, t2)
            t_min = max(t_min, near)
            t_max = min(t_max, far)

            if t_min > t_max:
                return None

        if t_max < 0.0:
            return None

        hit_distance = max(t_min, 0.0)
        if hit_distance > max_distance:
            return None
        return float(hit_distance)

    def to_feature(self, point):
        closest = self.closest_point(point)
        return {
            "closest_point": closest,
            "center": self.center.copy(),
            "velocity": self.velocity.copy(),
            "clearance": self.signed_distance(point),
            "size": self.effective_radius,
        }
