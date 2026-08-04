from dataclasses import dataclass

import numpy as np

from Controller.dmp_flow import (
    compute_dmp_drives,
    compute_dmp_flow_consistency,
    update_dmp_phase,
)


@dataclass
class DMPConfig:
    """
    二阶离散 DMP 配置。
    """

    dt: float
    dims: int = 3
    K_alpha: float = 20.0
    K_beta: float = 5.0
    alpha_s: float = 4.0
    tau: float = 1.2
    forcing_term_max: float = 10.0
    forcing_term_min: float = -10.0
    goal_offset_max: float = 1.5
    phase_mode: str = "classic"
    phase_integrator: str = "legacy_euler"
    phase_min: float = 0.0
    flow_zero_threshold: float = 1.0e-4

    def __post_init__(self):
        self.dt = float(self.dt)
        self.dims = int(self.dims)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.dims <= 0:
            raise ValueError("dims must be positive")
        self.phase_mode = str(self.phase_mode).lower()
        self.phase_integrator = str(self.phase_integrator).lower()
        self.phase_min = float(self.phase_min)
        self.flow_zero_threshold = float(self.flow_zero_threshold)
        if self.phase_mode not in {"classic", "fcep"}:
            raise ValueError("phase_mode must be 'classic' or 'fcep'")
        if self.phase_integrator not in {"legacy_euler", "exponential"}:
            raise ValueError("phase_integrator must be 'legacy_euler' or 'exponential'")
        if not 0.0 <= self.phase_min <= 1.0:
            raise ValueError("phase_min must lie in [0, 1]")
        if self.flow_zero_threshold < 0.0:
            raise ValueError("flow_zero_threshold must be non-negative")


class SecondOrderDMPController:
    """
    面向质点模型的二阶 DMP 控制器。

    RL 不直接输出加速度，而是输出：
    1. 每个维度对应的 forcing term
    2. 局部目标偏移
    """

    def __init__(self, config: DMPConfig):
        self.config = config
        self.start = np.zeros(self.config.dims, dtype=float)
        self.goal = np.zeros(self.config.dims, dtype=float)
        self.phase = 1.0
        self.K_alpha = self.config.K_alpha
        self.K_beta = self.config.K_beta

    def reset(self, start, goal):
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        if self.start.shape != (self.config.dims,) or self.goal.shape != (self.config.dims,):
            raise ValueError("start and goal must match DMP dims")
        self.phase = 1.0

    def compute_acceleration(self, position, velocity, rl_action, sensor_packet=None):
        """
        计算当前 DMP 加速度。

        单机action 的组织方式为：
        [forcing_x, forcing_y, forcing_z, goal_offset_x, goal_offset_y, goal_offset_z]
        """
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        rl_action = np.asarray(rl_action, dtype=float)

        if position.shape != (self.config.dims,) or velocity.shape != (self.config.dims,):
            raise ValueError("position and velocity must match DMP dims")
        if rl_action.shape != (2 * self.config.dims,):
            raise ValueError("rl_action must have shape (2 * dims,)")

        parsed = self._parse_action(rl_action)
        goal_eff = self.goal + parsed["goal_offset"]
        forcing = np.clip(
            parsed["forcing_term"],
            self.config.forcing_term_min,
            self.config.forcing_term_max,
        )
        nominal_drive, residual_drive, closed_loop_drive = compute_dmp_drives(
            goal_eff - position,
            velocity,
            forcing,
            k_alpha=self.config.K_alpha,
            k_beta=self.config.K_beta,
            tau=self.config.tau,
        )
        acceleration = closed_loop_drive / (self.config.tau ** 2)
        flow_consistency = float(compute_dmp_flow_consistency(
            nominal_drive,
            closed_loop_drive,
            zero_threshold=self.config.flow_zero_threshold,
        ))

        previous_phase = float(self.phase)
        next_phase, phase_rate = update_dmp_phase(
            previous_phase,
            flow_consistency,
            alpha_s=self.config.alpha_s,
            dt=self.config.dt,
            tau=self.config.tau,
            phase_min=self.config.phase_min,
            phase_mode=self.config.phase_mode,
            phase_integrator=self.config.phase_integrator,
        )
        self.phase = float(next_phase)

        return acceleration, {
            "phase": float(self.phase),
            "previous_phase": previous_phase,
            "phase_rate": float(phase_rate),
            "phase_paused": bool(float(phase_rate) <= 0.0),
            "phase_mode": self.config.phase_mode,
            "tau": float(self.config.tau),
            "goal_eff": goal_eff.copy(),
            "goal_offset": parsed["goal_offset"].copy(),
            "forcing": forcing.copy(),
            "nominal_drive": np.asarray(nominal_drive, dtype=float).copy(),
            "residual_drive": np.asarray(residual_drive, dtype=float).copy(),
            "closed_loop_drive": np.asarray(closed_loop_drive, dtype=float).copy(),
            "flow_consistency": flow_consistency,
        }

    def _parse_action(self, rl_action):
        """
        解析强化学习动作。
        """
        forcing_term = rl_action[0:self.config.dims]
        goal_offset = np.clip(
            rl_action[self.config.dims:2 * self.config.dims],
            -self.config.goal_offset_max,
            self.config.goal_offset_max,
        )
        return {"forcing_term": forcing_term, "goal_offset": goal_offset}

    @staticmethod
    def calc_distance(goal, current):
        """
        输出逐维距离门控。
        """
        return np.abs(goal - current)


class HeuristicDMPPolicy:       # 这部分仅作测试用
    """
    用于 Demo 的启发式策略。

    该策略只负责构造与 DMP 接口一致的伪动作，
    后续可以直接替换成神经网络策略。
    """

    def __init__(self, goal_offset_max=1.0, forcing_scale=1.0, elevation_range_deg=(-80.0, 80.0), dims=3):
        self.goal_offset_max = float(goal_offset_max)
        self.forcing_scale = float(forcing_scale)
        self.elevation_range_deg = tuple(float(value) for value in elevation_range_deg)
        self.dims = int(dims)
        if self.dims <= 0:
            raise ValueError("dims must be positive")
        if self.dims > 3:
            raise ValueError("HeuristicDMPPolicy only supports dims <= 3")

    @staticmethod
    def _beam_direction(azimuth, elevation):
        cos_elevation = np.cos(elevation)
        return np.array(
            [
                cos_elevation * np.cos(azimuth),
                cos_elevation * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=float,
        )

    def act(self, sensor_packet):
        current_scan = np.asarray(sensor_packet.current_scan, dtype=float)
        previous_scan = np.asarray(sensor_packet.previous_scan, dtype=float)
        if current_scan.shape != previous_scan.shape:
            raise ValueError("current_scan and previous_scan must have the same shape")

        danger = np.clip(1.0 - current_scan, 0.0, 1.0)
        dynamic_change = np.clip(previous_scan - current_scan, 0.0, 1.0)
        weights = danger ** 2 + 0.5 * dynamic_change
        if not np.any(weights > 1e-6):
            return np.zeros(2 * self.dims, dtype=float)

        avoid_direction = np.zeros(self.dims, dtype=float)
        azimuth_bins, elevation_bins = current_scan.shape
        azimuth_angles = np.linspace(-np.pi, np.pi, azimuth_bins, endpoint=False, dtype=float)
        elevation_angles = np.deg2rad(
            np.linspace(self.elevation_range_deg[0], self.elevation_range_deg[1], elevation_bins, dtype=float)
        )

        for azimuth_index, azimuth in enumerate(azimuth_angles):
            for elevation_index, elevation in enumerate(elevation_angles):
                beam_weight = weights[azimuth_index, elevation_index]
                if beam_weight <= 1e-6:
                    continue
                avoid_direction -= self._beam_direction(azimuth, elevation)[: self.dims] * beam_weight

        norm = np.linalg.norm(avoid_direction)
        if norm > 1e-8:
            avoid_direction = avoid_direction / norm
        proximity_score = float(np.mean(weights[weights > 1e-6]))

        forcing_term = np.clip(
            avoid_direction * min(proximity_score, self.forcing_scale),
            -self.forcing_scale,
            self.forcing_scale,
        )
        goal_offset = np.clip(
            avoid_direction * self.goal_offset_max,
            -self.goal_offset_max,
            self.goal_offset_max,
        )
        return np.concatenate([forcing_term, goal_offset]).astype(float)
