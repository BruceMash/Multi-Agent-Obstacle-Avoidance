import copy
from dataclasses import dataclass

import numpy as np

from Controller.dmp_rl import DMPConfig, SecondOrderDMPController
from Entity.KinematicModel import PartialDynamic
from Entity.sensors import LocalObstacleSensor


@dataclass
class EnvConfig:
    """
    单机 DMP-RL 环境配置。
    """

    max_steps: int = 250
    goal_tolerance: float = 0.35
    collision_penalty: float = 80.0
    success_bonus: float = 80.0
    progress_weight: float = 4.0
    clearance_weight: float = 0.8
    action_penalty: float = 0.02
    living_penalty: float = 0.01
    collision_margin: float = 0.0


class SingleAgentDMPEnv:
    """
    面向质点模型的单机局部规划 Demo 环境。

    action 不是直接加速度，而是 DMP-RL 的调制量:
    [obstacle_gain, tau_scale, goal_offset_x, goal_offset_y, goal_offset_z]

    用forcing_term控制避障，goal_offset控制局部目标残差。
    [forcing_term, goal_offset_x, goal_offset_y, goal_offset_z]
    """

    def __init__(
        self,
        dynamics_config,
        sensor_config=None,
        dmp_config=None,
        env_config=None,
        static_obstacles=None,
        dynamic_obstacles=None,
    ):
        self.dynamics = PartialDynamic(dynamics_config)
        self.sensor = LocalObstacleSensor(**(sensor_config or {}))
        self.dmp = SecondOrderDMPController(dmp_config or DMPConfig(dt=self.dynamics.dt))
        self.env_config = env_config or EnvConfig()

        self._initial_static_obstacles = copy.deepcopy(static_obstacles or [])
        self._initial_dynamic_obstacles = copy.deepcopy(dynamic_obstacles or [])
        self.static_obstacles = copy.deepcopy(self._initial_static_obstacles)
        self.dynamic_obstacles = copy.deepcopy(self._initial_dynamic_obstacles)

        self.goal = np.zeros(3, dtype=float)
        self.steps = 0
        self.latest_sensor_packet = None
        self.latest_controller_info = {}

    @property
    def observation_dim(self):
        return self.sensor.observation_dim + 1

    @property
    def action_dim(self):
        # *联动修改：
        # 如果 Controller/dmp_rl.py 最终固定为
        # [forcing_term, goal_offset_x, goal_offset_y, goal_offset_z]，
        # 这里应同步改成 1 + dims，而不是 2 + dims。
        return 2 + self.dmp.config.dims

    def reset(self, start=None, goal=None, static_obstacles=None, dynamic_obstacles=None):
        start = np.asarray(start if start is not None else np.zeros(3, dtype=float), dtype=float)
        goal = np.asarray(goal if goal is not None else np.array([8.0, 0.0, 0.0], dtype=float), dtype=float)

        if start.shape != (3,) or goal.shape != (3,):
            raise ValueError("start and goal must have shape (3,)")

        self.goal = goal
        self.steps = 0
        self.static_obstacles = copy.deepcopy(static_obstacles) if static_obstacles is not None else copy.deepcopy(self._initial_static_obstacles)
        self.dynamic_obstacles = copy.deepcopy(dynamic_obstacles) if dynamic_obstacles is not None else copy.deepcopy(self._initial_dynamic_obstacles)

        self.dynamics.reset({"position": start, "velocity": np.zeros(3, dtype=float)})
        self.dmp.reset(start, goal)
        self.latest_sensor_packet = self.sensor.sense(
            self.dynamics.p,
            self.dynamics.v,
            self.goal,
            self.static_obstacles,
            self.dynamic_obstacles,
        )
        return self._compose_observation(self.latest_sensor_packet)

    def step(self, action):
        if self.latest_sensor_packet is None:
            raise RuntimeError("reset must be called before step")

        previous_distance = np.linalg.norm(self.goal - self.dynamics.p)
        acceleration, controller_info = self.dmp.compute_acceleration(
            self.dynamics.p,
            self.dynamics.v,
            action,
            self.latest_sensor_packet,
        )
        applied_acceleration = np.clip(
            acceleration,
            self.dynamics.accelerate_min,
            self.dynamics.accelerate_max,
        )
        self.latest_controller_info = controller_info
        next_state = self.dynamics.step(acceleration)

        for obstacle in self.dynamic_obstacles:
            obstacle.step(self.dynamics.dt)

        self.steps += 1
        self.latest_sensor_packet = self.sensor.sense(
            self.dynamics.p,
            self.dynamics.v,
            self.goal,
            self.static_obstacles,
            self.dynamic_obstacles,
        )
        observation = self._compose_observation(self.latest_sensor_packet)

        current_distance = np.linalg.norm(self.goal - self.dynamics.p)
        progress = previous_distance - current_distance
        reward = self.env_config.progress_weight * progress - self.env_config.action_penalty * np.linalg.norm(applied_acceleration)
        reward -= self.env_config.living_penalty

        if np.isfinite(self.latest_sensor_packet.min_clearance):
            reward -= self.env_config.clearance_weight * np.exp(-max(self.latest_sensor_packet.min_clearance, 0.0))

        done = False
        success = current_distance <= self.env_config.goal_tolerance
        collision = self._check_collision()
        truncated = self.steps >= self.env_config.max_steps

        if collision:
            reward -= self.env_config.collision_penalty
            done = True
        elif success:
            reward += self.env_config.success_bonus
            done = True
        elif truncated:
            done = True

        info = {
            "success": success,
            "collision": collision,
            "truncated": truncated,
            "distance_to_goal": current_distance,
            "min_clearance": self.latest_sensor_packet.min_clearance,
            # *联动修改：
            # 当前 dmp_rl.py 的 controller_info 尚未稳定返回 tau；
            # 如果控制器不再显式维护 tau，这个字段也需要同步删除或改名。
            "phase": controller_info["phase"],
            "tau": controller_info["tau"],
            "commanded_acceleration": acceleration.copy(),
            "applied_acceleration": applied_acceleration.copy(),
            "next_state": next_state.copy(),
        }
        return observation, float(reward), done, info

    def _check_collision(self):
        point = self.dynamics.p
        for obstacle in list(self.static_obstacles) + list(self.dynamic_obstacles):
            if obstacle.contains(point, margin=self.env_config.collision_margin):
                return True
        return False

    def _compose_observation(self, sensor_packet):
        return np.concatenate([sensor_packet.observation, np.array([self.dmp.phase], dtype=float)])
