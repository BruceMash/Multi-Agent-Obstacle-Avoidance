from dataclasses import dataclass

import numpy as np


def _to_vector3(value):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError("value must have shape (3,)")
    return vector


@dataclass
class WorkspaceBoundaryPlaneObstacle:
    """
    工作空间边界面障碍物。

    该类将轴对齐 workspace 的单个矩形边界面封装为静态障碍物，
    用于让局部激光雷达能够观测到边界约束。
    """

    axis: int
    bound: float
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    is_lower: bool

    def __post_init__(self):
        self.axis = int(self.axis)
        self.bound = float(self.bound)
        self.lower_bounds = _to_vector3(self.lower_bounds)
        self.upper_bounds = _to_vector3(self.upper_bounds)
        self.is_lower = bool(self.is_lower)
        if self.axis < 0 or self.axis >= 3:
            raise ValueError("axis must be in [0, 2]")
        if np.any(self.lower_bounds >= self.upper_bounds):
            raise ValueError("lower_bounds must be smaller than upper_bounds")

    @property
    def velocity(self):
        return np.zeros(3, dtype=float)

    @property
    def center(self):
        center = 0.5 * (self.lower_bounds + self.upper_bounds)
        center[self.axis] = self.bound
        return center

    @property
    def effective_radius(self):
        face_extents = self.upper_bounds - self.lower_bounds
        face_extents[self.axis] = 0.0
        return float(0.5 * np.linalg.norm(face_extents))

    def signed_distance(self, point):
        point = _to_vector3(point)
        if self.is_lower:
            return float(point[self.axis] - self.bound)
        return float(self.bound - point[self.axis])

    def contains(self, point, margin=0.0):
        return self.signed_distance(point) <= float(margin)

    def closest_point(self, point):
        point = _to_vector3(point)
        closest = np.clip(point, self.lower_bounds, self.upper_bounds)
        closest[self.axis] = self.bound
        return closest

    def ray_intersection(self, origin, direction, max_distance):
        """
        计算射线与有限矩形边界面的最近正向交点距离。
        若射线不朝向该边界面，或交点落在边界面矩形范围外，则返回 None。
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

        axis_direction = float(direction[self.axis])
        if abs(axis_direction) < 1e-8:
            return None

        hit_distance = (self.bound - origin[self.axis]) / axis_direction
        if hit_distance < 0.0 or hit_distance > max_distance:
            return None

        hit_point = origin + hit_distance * direction
        tolerance = 1e-8
        for dim in range(3):
            if dim == self.axis:
                continue
            if hit_point[dim] < self.lower_bounds[dim] - tolerance:
                return None
            if hit_point[dim] > self.upper_bounds[dim] + tolerance:
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
        closest = np.clip(point, lower, upper)
        if np.any(point < lower) or np.any(point > upper):
            return closest

        # 点在盒子内部时，最近点应落在距离它最近的表面上，而不是返回点本身。
        distance_to_lower = point - lower
        distance_to_upper = upper - point
        axis = int(np.argmin(np.minimum(distance_to_lower, distance_to_upper)))
        if distance_to_lower[axis] <= distance_to_upper[axis]:
            closest[axis] = lower[axis]
        else:
            closest[axis] = upper[axis]
        return closest

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


@dataclass
class StaticCylinderObstacle:
    """
    Z 轴对齐的静态圆柱体障碍物，使用中心点、半径和半高描述。
    """

    center: np.ndarray
    radius: float
    half_height: float
    safety_margin: float = 0.0

    def __post_init__(self):
        self.center = _to_vector3(self.center)
        self.radius = float(self.radius)
        self.half_height = float(self.half_height)
        self.safety_margin = float(self.safety_margin)
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.half_height <= 0.0:
            raise ValueError("half_height must be positive")

    @property
    def velocity(self):
        return np.zeros(3, dtype=float)

    @property
    def effective_radius(self):
        r = self.radius + self.safety_margin
        h = self.half_height + self.safety_margin
        return float(np.sqrt(r * r + h * h))

    @property
    def expanded_radius(self):
        return self.radius + self.safety_margin

    @property
    def expanded_half_height(self):
        return self.half_height + self.safety_margin

    def signed_distance(self, point):
        point = _to_vector3(point)
        offset = point - self.center
        d_xy = np.linalg.norm(offset[:2]) - self.expanded_radius
        d_z = abs(offset[2]) - self.expanded_half_height
        outside_xy = max(d_xy, 0.0)
        outside_z = max(d_z, 0.0)
        inside = min(max(d_xy, d_z), 0.0)
        return float(np.sqrt(outside_xy * outside_xy + outside_z * outside_z) + inside)

    def contains(self, point, margin=0.0):
        return self.signed_distance(point) <= float(margin)

    def closest_point(self, point):
        point = _to_vector3(point)
        offset = point - self.center
        xy_offset = offset[:2]
        z_offset = offset[2]
        R = self.expanded_radius
        H = self.expanded_half_height

        xy_len = float(np.linalg.norm(xy_offset))
        if xy_len < 1e-8:
            direction_xy = np.array([1.0, 0.0], dtype=float)
        else:
            direction_xy = xy_offset / xy_len

        if xy_len <= R:
            if abs(z_offset) <= H:
                d_side = R - xy_len
                d_top = H - z_offset
                d_bottom = H + z_offset
                if d_side <= d_top and d_side <= d_bottom:
                    return np.array([
                        self.center[0] + direction_xy[0] * R,
                        self.center[1] + direction_xy[1] * R,
                        point[2],
                    ], dtype=float)
                elif d_top <= d_bottom:
                    return np.array([point[0], point[1], self.center[2] + H], dtype=float)
                else:
                    return np.array([point[0], point[1], self.center[2] - H], dtype=float)
            else:
                z_sign = 1.0 if z_offset > 0.0 else -1.0
                return np.array([point[0], point[1], self.center[2] + z_sign * H], dtype=float)
        else:
            if abs(z_offset) <= H:
                return np.array([
                    self.center[0] + direction_xy[0] * R,
                    self.center[1] + direction_xy[1] * R,
                    point[2],
                ], dtype=float)
            else:
                z_sign = 1.0 if z_offset > 0.0 else -1.0
                return np.array([
                    self.center[0] + direction_xy[0] * R,
                    self.center[1] + direction_xy[1] * R,
                    self.center[2] + z_sign * H,
                ], dtype=float)

    def ray_intersection(self, origin, direction, max_distance):
        origin = _to_vector3(origin)
        direction = _to_vector3(direction)
        max_distance = float(max_distance)

        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-8:
            raise ValueError("direction must be non-zero")
        direction = direction / direction_norm

        if self.contains(origin):
            return 0.0

        R = self.expanded_radius
        H = self.expanded_half_height

        candidates = []

        a = direction[0] * direction[0] + direction[1] * direction[1]
        if a > 1e-8:
            o_xy = origin[:2] - self.center[:2]
            b = 2.0 * (o_xy[0] * direction[0] + o_xy[1] * direction[1])
            c = o_xy[0] * o_xy[0] + o_xy[1] * o_xy[1] - R * R
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                sqrt_d = np.sqrt(discriminant)
                t0 = (-b - sqrt_d) / (2.0 * a)
                t1 = (-b + sqrt_d) / (2.0 * a)
                for t in (t0, t1):
                    if t > 0.0:
                        z = origin[2] + t * direction[2]
                        if abs(z - self.center[2]) <= H:
                            candidates.append(float(t))

        if abs(direction[2]) > 1e-8:
            for sign in (1.0, -1.0):
                cap_z = self.center[2] + sign * H
                t = (cap_z - origin[2]) / direction[2]
                if t > 0.0:
                    p_xy = origin[:2] + t * direction[:2]
                    if float(np.linalg.norm(p_xy - self.center[:2])) <= R:
                        candidates.append(float(t))

        if not candidates:
            return None

        hit_distance = min(candidates)
        if hit_distance > max_distance:
            return None
        return hit_distance

    def to_feature(self, point):
        closest = self.closest_point(point)
        return {
            "closest_point": closest,
            "center": self.center.copy(),
            "velocity": self.velocity.copy(),
            "clearance": self.signed_distance(point),
            "size": self.effective_radius,
        }
