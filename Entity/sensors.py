from dataclasses import dataclass

import numpy as np


@dataclass
class SensorPacket:
    """
    本地传感器打包结果。
    """

    observation: np.ndarray
    obstacle_features: np.ndarray
    min_clearance: float
    collision: bool


class LocalObstacleSensor:
    """
    简化的局部障碍物传感器。

    每个障碍物编码为:
    [rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz, clearance, size, valid]
    """

    def __init__(self, sensing_radius=6.0, max_obstacles=4):
        self.sensing_radius = float(sensing_radius)
        self.max_obstacles = int(max_obstacles)
        if self.max_obstacles <= 0:
            raise ValueError("max_obstacles must be positive")

    @property
    def feature_dim(self):
        return 9

    @property
    def observation_dim(self):
        return 6 + 3 + self.max_obstacles * self.feature_dim

    def sense(self, position, velocity, goal, static_obstacles=None, dynamic_obstacles=None):
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        goal = np.asarray(goal, dtype=float)

        if position.shape != (3,) or velocity.shape != (3,) or goal.shape != (3,):
            raise ValueError("position, velocity and goal must have shape (3,)")

        static_obstacles = static_obstacles or []
        dynamic_obstacles = dynamic_obstacles or []
        raw_features = []

        for obstacle in list(static_obstacles) + list(dynamic_obstacles):
            feature = obstacle.to_feature(position)
            relative_position = feature["closest_point"] - position
            relative_velocity = feature["velocity"] - velocity
            clearance = float(feature["clearance"])
            if clearance <= self.sensing_radius:
                raw_features.append(
                    np.concatenate(
                        [
                            relative_position,
                            relative_velocity,
                            np.array([clearance, feature["size"], 1.0], dtype=float),
                        ]
                    )
                )

        raw_features.sort(key=lambda item: item[6])
        padded = np.zeros((self.max_obstacles, self.feature_dim), dtype=float)
        for index, feature in enumerate(raw_features[: self.max_obstacles]):
            padded[index] = feature

        min_clearance = min([feature[6] for feature in raw_features], default=np.inf)
        collision = bool(min_clearance <= 0.0)
        observation = np.concatenate([position, velocity, goal - position, padded.reshape(-1)])
        return SensorPacket(
            observation=observation,
            obstacle_features=padded,
            min_clearance=float(min_clearance),
            collision=collision,
        )
