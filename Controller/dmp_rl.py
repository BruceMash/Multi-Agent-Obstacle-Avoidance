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
    forcing_gate_kappa: float = 1.0
    goal_offset_max: float = 1.5
    phase_mode: str = "classic"
    phase_integrator: str = "legacy_euler"
    phase_min: float = 0.0
    phase_end_threshold: float = 1.0e-4
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
        self.phase_end_threshold = float(self.phase_end_threshold)
        self.forcing_gate_kappa = float(self.forcing_gate_kappa)
        self.flow_zero_threshold = float(self.flow_zero_threshold)
        if self.phase_mode not in {"classic", "fcep"}:
            raise ValueError("phase_mode must be 'classic' or 'fcep'")
        if self.phase_integrator not in {"legacy_euler", "exponential"}:
            raise ValueError("phase_integrator must be 'legacy_euler' or 'exponential'")
        if not 0.0 <= self.phase_min <= 1.0:
            raise ValueError("phase_min must lie in [0, 1]")
        if not 0.0 <= self.phase_end_threshold <= 1.0:
            raise ValueError("phase_end_threshold must lie in [0, 1]")
        if self.forcing_gate_kappa < 0.0:
            raise ValueError("forcing_gate_kappa must be non-negative")
        if self.flow_zero_threshold < 0.0:
            raise ValueError("flow_zero_threshold must be non-negative")


def compute_dmp_transition(
    *,
    config: DMPConfig,
    position,
    velocity,
    rl_action,
    active_goal,
    terminal_goal,
    phase,
):
    """计算一次无副作用的 DMP 状态转移。

    ``active_goal`` 决定弹簧项，``terminal_goal`` 仅决定 forcing gate。
    该区分与当前历史 checkpoint 的真实执行语义保持一致。
    """
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    rl_action = np.asarray(rl_action, dtype=float)
    active_goal = np.asarray(active_goal, dtype=float)
    terminal_goal = np.asarray(terminal_goal, dtype=float)
    expected_state_shape = (config.dims,)
    if position.shape != expected_state_shape or velocity.shape != expected_state_shape:
        raise ValueError("position and velocity must match DMP dims")
    if active_goal.shape != expected_state_shape:
        raise ValueError("active_goal must match DMP dims")
    if terminal_goal.shape != expected_state_shape:
        raise ValueError("terminal_goal must match DMP dims")
    if rl_action.shape != (2 * config.dims,):
        raise ValueError("rl_action must have shape (2 * dims,)")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError("position and velocity must be finite")
    if not np.all(np.isfinite(rl_action)):
        raise ValueError("rl_action must be finite")
    if not np.all(np.isfinite(active_goal)) or not np.all(np.isfinite(terminal_goal)):
        raise ValueError("active_goal and terminal_goal must be finite")
    phase = float(phase)
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")

    raw_forcing = rl_action[: config.dims].copy()
    forcing = np.clip(raw_forcing, config.forcing_term_min, config.forcing_term_max)
    goal_offset = np.clip(
        rl_action[config.dims : 2 * config.dims],
        -config.goal_offset_max,
        config.goal_offset_max,
    )
    goal_eff = active_goal + goal_offset
    goal_delta = goal_eff - position
    terminal_goal_delta = terminal_goal - position
    terminal_goal_distance = float(np.linalg.norm(terminal_goal_delta))
    forcing_gate_scalar = float(
        np.tanh(config.forcing_gate_kappa * terminal_goal_distance)
    )
    forcing_gate = np.full(config.dims, forcing_gate_scalar, dtype=float)
    spring_drive = float(config.K_alpha) * float(config.K_beta) * goal_delta
    damping_drive = -float(config.K_alpha) * float(config.tau) * velocity
    nominal_drive, residual_drive, closed_loop_drive = compute_dmp_drives(
        goal_delta,
        velocity,
        forcing,
        k_alpha=config.K_alpha,
        k_beta=config.K_beta,
        tau=config.tau,
        forcing_gate_kappa=config.forcing_gate_kappa,
        forcing_gate_distance=terminal_goal_distance,
    )
    acceleration = closed_loop_drive / (config.tau ** 2)
    spring_norm = float(np.linalg.norm(spring_drive))
    damping_norm = float(np.linalg.norm(damping_drive))
    flow_consistency = float(
        compute_dmp_flow_consistency(
            nominal_drive,
            closed_loop_drive,
            zero_threshold=config.flow_zero_threshold,
        )
    )
    next_phase, phase_rate = update_dmp_phase(
        phase,
        flow_consistency,
        alpha_s=config.alpha_s,
        dt=config.dt,
        tau=config.tau,
        phase_min=config.phase_min,
        phase_mode=config.phase_mode,
        phase_integrator=config.phase_integrator,
    )
    info = {
        "phase": float(next_phase),
        "previous_phase": phase,
        "phase_rate": float(phase_rate),
        "phase_paused": bool(float(phase_rate) <= 0.0),
        "phase_end_reached": bool(float(next_phase) <= config.phase_end_threshold),
        "phase_mode": config.phase_mode,
        "tau": float(config.tau),
        "goal_eff": goal_eff.copy(),
        "goal_delta": goal_delta.copy(),
        "terminal_goal": terminal_goal.copy(),
        "terminal_goal_delta": terminal_goal_delta.copy(),
        "terminal_goal_distance": terminal_goal_distance,
        "goal_offset": goal_offset.copy(),
        "raw_forcing": raw_forcing,
        "forcing": forcing.copy(),
        "forcing_gate": forcing_gate,
        "forcing_gate_scalar": forcing_gate_scalar,
        "spring_drive": np.asarray(spring_drive, dtype=float).copy(),
        "damping_drive": np.asarray(damping_drive, dtype=float).copy(),
        "spring_drive_norm": spring_norm,
        "damping_drive_norm": damping_norm,
        "damping_spring_ratio": damping_norm / (spring_norm + 1.0e-8),
        "nominal_drive": np.asarray(nominal_drive, dtype=float).copy(),
        "residual_drive": np.asarray(residual_drive, dtype=float).copy(),
        "closed_loop_drive": np.asarray(closed_loop_drive, dtype=float).copy(),
        "flow_consistency": flow_consistency,
    }
    return np.asarray(acceleration, dtype=float), float(next_phase), info


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

    def compute_acceleration(
        self,
        position,
        velocity,
        rl_action,
        sensor_packet=None,
        terminal_goal=None,
    ):
        """
        计算当前 DMP 加速度。

        单机action 的组织方式为：
        [forcing_x, forcing_y, forcing_z, goal_offset_x, goal_offset_y, goal_offset_z]
        """
        terminal_goal = (
            self.goal.copy()
            if terminal_goal is None
            else np.asarray(terminal_goal, dtype=float)
        )
        acceleration, next_phase, info = compute_dmp_transition(
            config=self.config,
            position=position,
            velocity=velocity,
            rl_action=rl_action,
            active_goal=self.goal,
            terminal_goal=terminal_goal,
            phase=self.phase,
        )
        self.phase = next_phase
        return acceleration, info

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
