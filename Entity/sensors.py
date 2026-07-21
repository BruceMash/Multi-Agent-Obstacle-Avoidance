from dataclasses import dataclass

import numpy as np

from Entity.dynamic_obstacles import MovingSphereObstacle
from Entity.static_obstacles import (
    AxisAlignedBoxObstacle,
    StaticCylinderObstacle,
    StaticSphereObstacle,
    WorkspaceBoundaryPlaneObstacle,
)


@dataclass
class SensorPacket:
    """
    本地传感器打包结果。目前这里封装的observation是用于单机训练的
    """

    observation: np.ndarray
    current_scan: np.ndarray
    previous_scan: np.ndarray
    min_clearance: float
    collision: bool


class LocalObstacleSensor:
    """
    使用 3D 多束激光雷达生成局部观测。

    主观测顺序固定为：
    1. 当前速度 3 维
    2. 目标方向单位向量 3 维
    3. 目标距离归一化标量 1 维
    4. 当前帧雷达扫描
    5. 可选的上一帧雷达扫描
    """

    def __init__(
        self,
        sensing_radius=6.0,
        azimuth_bins=24,
        elevation_bins=9,
        elevation_range_deg=(-80.0, 80.0),
        goal_distance_clip=None,
        include_previous_scan=True,
        broad_phase_enabled=True,
    ):
        self.sensing_radius = float(sensing_radius)
        self.azimuth_bins = int(azimuth_bins)
        self.elevation_bins = int(elevation_bins)
        self.include_previous_scan = bool(include_previous_scan)
        self.broad_phase_enabled = bool(broad_phase_enabled)
        if self.sensing_radius <= 0.0:
            raise ValueError("sensing_radius must be positive")
        if self.azimuth_bins <= 0 or self.elevation_bins <= 0:
            raise ValueError("azimuth_bins and elevation_bins must be positive")

        self.elevation_range_deg = tuple(float(value) for value in elevation_range_deg)
        if len(self.elevation_range_deg) != 2 or self.elevation_range_deg[0] >= self.elevation_range_deg[1]:
            raise ValueError("elevation_range_deg must contain increasing lower and upper bounds")

        if goal_distance_clip is None:  # 不进行归一化，则默认为感知半径的两倍
            goal_distance_clip = 2.0 * self.sensing_radius
        self.goal_distance_clip = float(goal_distance_clip)

        if self.goal_distance_clip <= 0.0:
            raise ValueError("goal_distance_clip must be positive")

        self.azimuth_angles = np.linspace(-np.pi, np.pi, self.azimuth_bins, endpoint=False, dtype=float)
        self.elevation_angles = np.deg2rad(
            np.linspace(self.elevation_range_deg[0], self.elevation_range_deg[1], self.elevation_bins, dtype=float)
        )
        self.ray_directions = self._build_ray_directions()
        self._previous_scan = None

    @property
    def scan_shape(self):
        return (self.azimuth_bins, self.elevation_bins)

    @property
    def n_rays(self):
        return self.azimuth_bins * self.elevation_bins

    @property
    def observation_dim(self):
        scan_count = 2 if self.include_previous_scan else 1
        return 3 + 3 + 1 + scan_count * self.n_rays

    def reset(self):
        """
        清空上一帧扫描缓存。
        """
        self._previous_scan = None

    def _build_ray_directions(self):
        directions = np.zeros((self.azimuth_bins, self.elevation_bins, 3), dtype=float)
        for azimuth_index, azimuth in enumerate(self.azimuth_angles):
            for elevation_index, elevation in enumerate(self.elevation_angles):
                cos_elevation = np.cos(elevation)
                directions[azimuth_index, elevation_index] = np.array(
                    [
                        cos_elevation * np.cos(azimuth),
                        cos_elevation * np.sin(azimuth),
                        np.sin(elevation),
                    ],
                    dtype=float,
                )
        return directions

    @staticmethod
    def _obstacle_aabb(obstacle):
        """Return a conservative AABB, or None for an unsupported obstacle."""
        if isinstance(obstacle, WorkspaceBoundaryPlaneObstacle):
            lower = np.asarray(obstacle.lower_bounds, dtype=float).copy()
            upper = np.asarray(obstacle.upper_bounds, dtype=float).copy()
            lower[int(obstacle.axis)] = float(obstacle.bound)
            upper[int(obstacle.axis)] = float(obstacle.bound)
            return lower, upper

        if isinstance(obstacle, AxisAlignedBoxObstacle):
            center = np.asarray(obstacle.center, dtype=float)
            half_extents = np.asarray(obstacle.expanded_half_extents, dtype=float)
            return center - half_extents, center + half_extents

        if isinstance(obstacle, StaticCylinderObstacle):
            center = np.asarray(obstacle.center, dtype=float)
            half_extents = np.array(
                [
                    float(obstacle.expanded_radius),
                    float(obstacle.expanded_radius),
                    float(obstacle.expanded_half_height),
                ],
                dtype=float,
            )
            return center - half_extents, center + half_extents

        if isinstance(obstacle, (StaticSphereObstacle, MovingSphereObstacle)):
            center = np.asarray(obstacle.center, dtype=float)
            radius = float(obstacle.effective_radius)
            return center - radius, center + radius

        return None

    def _ray_aabb_candidates(self, position, directions, lower, upper):
        """Conservatively select rays that can reach an AABB within sensing range."""
        position = np.asarray(position, dtype=float)
        directions = np.asarray(directions, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        scale = max(
            1.0,
            self.sensing_radius,
            float(np.max(np.abs(position))),
            float(np.max(np.abs(lower))),
            float(np.max(np.abs(upper))),
        )
        tolerance = 1e-7 * scale
        padded_lower = lower - tolerance
        padded_upper = upper + tolerance

        ray_count = directions.shape[0]
        t_near = np.full(ray_count, -np.inf, dtype=float)
        t_far = np.full(ray_count, np.inf, dtype=float)
        candidates = np.ones(ray_count, dtype=bool)

        for axis in range(3):
            axis_direction = directions[:, axis]
            parallel = np.abs(axis_direction) < 1e-12
            outside = np.logical_or(
                position[axis] < padded_lower[axis],
                position[axis] > padded_upper[axis],
            )
            if outside:
                candidates[parallel] = False

            non_parallel = np.logical_not(parallel)
            if not np.any(non_parallel):
                continue
            t1 = (padded_lower[axis] - position[axis]) / axis_direction[non_parallel]
            t2 = (padded_upper[axis] - position[axis]) / axis_direction[non_parallel]
            near = np.minimum(t1, t2)
            far = np.maximum(t1, t2)
            t_near[non_parallel] = np.maximum(t_near[non_parallel], near)
            t_far[non_parallel] = np.minimum(t_far[non_parallel], far)

        candidates &= t_near <= t_far + tolerance
        candidates &= t_far >= -tolerance
        candidates &= np.maximum(t_near, 0.0) <= self.sensing_radius + tolerance
        return candidates

    def _build_candidate_masks(self, position, obstacles):
        directions = self.ray_directions.reshape(-1, 3)
        masks = []
        for obstacle in obstacles:
            bounds = self._obstacle_aabb(obstacle)
            if bounds is None:
                masks.append(np.ones(self.n_rays, dtype=bool))
            elif obstacle.contains(position):
                masks.append(np.ones(self.n_rays, dtype=bool))
            else:
                masks.append(
                    self._ray_aabb_candidates(
                        position,
                        directions,
                        bounds[0],
                        bounds[1],
                    )
                )
        if not masks:
            return np.zeros((0, self.n_rays), dtype=bool)
        return np.stack(masks, axis=0)

    def _scan_obstacles(self, position, obstacles):
        """
        对每一束射线求最近交点距离；若无交点则返回感知半径。
        """
        distances = np.full(self.scan_shape, self.sensing_radius, dtype=float)
        candidate_masks = None
        if self.broad_phase_enabled:
            candidate_masks = self._build_candidate_masks(position, obstacles)
        for azimuth_index in range(self.azimuth_bins):
            for elevation_index in range(self.elevation_bins):
                direction = self.ray_directions[azimuth_index, elevation_index]
                nearest_distance = self.sensing_radius
                ray_index = azimuth_index * self.elevation_bins + elevation_index
                for obstacle_index, obstacle in enumerate(obstacles):
                    if candidate_masks is not None and not candidate_masks[obstacle_index, ray_index]:
                        continue
                    hit_distance = obstacle.ray_intersection(position, direction, self.sensing_radius)
                    if hit_distance is not None and hit_distance < nearest_distance:
                        nearest_distance = hit_distance
                distances[azimuth_index, elevation_index] = nearest_distance
        return distances

    def _build_goal_features(self, position, goal): 
        goal_vector = goal - position
        goal_distance = float(np.linalg.norm(goal_vector))
        if goal_distance < 1e-8:
            goal_direction = np.zeros(3, dtype=float)
        else:
            goal_direction = goal_vector / goal_distance
        goal_distance_norm = np.array(
            [np.clip(goal_distance / self.goal_distance_clip, 0.0, 1.0)],   # 归一化并裁剪到[0, 1]
            dtype=float,
        )
        return goal_direction, goal_distance_norm

    def sense(self, position, velocity, goal, static_obstacles=None, dynamic_obstacles=None):
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        goal = np.asarray(goal, dtype=float)
        if position.shape != (3,) or velocity.shape != (3,) or goal.shape != (3,):
            raise ValueError("position, velocity and goal must have shape (3,)")

        static_obstacles = list(static_obstacles or [])
        dynamic_obstacles = list(dynamic_obstacles or [])
        obstacles = static_obstacles + dynamic_obstacles

        raw_scan = self._scan_obstacles(position, obstacles)
        current_scan = np.clip(raw_scan / self.sensing_radius, 0.0, 1.0).astype(np.float32)
        if self._previous_scan is None:
            previous_scan = current_scan.copy()
        else:
            previous_scan = self._previous_scan.copy()

        min_clearance = min((obstacle.signed_distance(position) for obstacle in obstacles), default=np.inf)
        collision = bool(min_clearance <= 0.0)
        goal_direction, goal_distance_norm = self._build_goal_features(position, goal)

        observation_parts = [
            velocity.astype(np.float32),
            goal_direction.astype(np.float32),
            goal_distance_norm.astype(np.float32),
            current_scan.reshape(-1),
        ]
        if self.include_previous_scan:
            observation_parts.append(previous_scan.reshape(-1))
        observation = np.concatenate(observation_parts, axis=0).astype(np.float32)

        self._previous_scan = current_scan.copy()
        return SensorPacket(
            observation=observation,
            current_scan=current_scan.copy(),
            previous_scan=previous_scan.copy(),
            min_clearance=float(min_clearance),
            collision=collision,
        )
