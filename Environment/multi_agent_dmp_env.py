from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from Controller.dmp_rl import DMPConfig, SecondOrderDMPController
from Entity.KinematicModel import PartialDynamic
from Entity.sensors import LocalObstacleSensor
from Entity.static_obstacles import WorkspaceBoundaryPlaneObstacle
from Environment.single_agent_dmp_env import EnvConfig


@dataclass  # 简化数据类的表示
class MultiAgentEnvConfig(EnvConfig):
    """
    Multi-agent DMP-RL environment configuration.

    The base waypoint, obstacle, boundary, and reward fields are inherited from
    EnvConfig. The fields below define inter-agent safety and control penalties.
    """

    # 默认参数
    num_agents: int = 4
    inter_agent_safe_distance: float = 0.6
    inter_agent_collision_penalty: float = 20.0
    inter_agent_potential_weight: float = 1.0
    inter_agent_influence_distance: float = 1.2
    # None means observing all other agents; a non-negative integer limits the
    # number of nearest allies kept in the explicit ally observation block.
    nearest_agent_observation_count: int | None = None
    acceleration_penalty_weight: float = 0.01
    acceleration_clip_penalty_weight: float = 0.05
    randomize_start_goal: bool = True
    start_position_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = (
        (0.0, -2.0, -0.8),
        (0.8, 1.5, 0.8),
    )
    goal_position_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = (
        (7.2, -2.0, -0.8),
        (8.0, 1.5, 0.8),
    )
    min_start_distance: float = 0.6
    min_goal_distance: float = 0.0
    min_start_goal_distance: float = 6.0
    start_goal_max_attempts: int = 1000
    near_goal_bonus_radius_1: float = 1.0
    near_goal_bonus_1: float = 0.2
    near_goal_bonus_radius_2: float = 0.6
    near_goal_bonus_2: float = 0.6

    def __post_init__(self) -> None:    # 初始化场景元素的初始位置
        self.num_agents = int(self.num_agents)
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")

        # 奖励参数
        self.inter_agent_safe_distance = float(self.inter_agent_safe_distance)
        self.inter_agent_collision_penalty = float(self.inter_agent_collision_penalty)
        self.inter_agent_potential_weight = float(self.inter_agent_potential_weight)
        self.inter_agent_influence_distance = float(self.inter_agent_influence_distance)
        if self.nearest_agent_observation_count is not None:
            self.nearest_agent_observation_count = int(self.nearest_agent_observation_count)
        self.acceleration_penalty_weight = float(self.acceleration_penalty_weight)
        self.acceleration_clip_penalty_weight = float(self.acceleration_clip_penalty_weight)
        self.randomize_start_goal = bool(self.randomize_start_goal)
        self.min_start_distance = float(self.min_start_distance)
        self.min_goal_distance = float(self.min_goal_distance)
        self.min_start_goal_distance = float(self.min_start_goal_distance)
        self.start_goal_max_attempts = int(self.start_goal_max_attempts)
        self.near_goal_bonus_radius_1 = float(self.near_goal_bonus_radius_1)
        self.near_goal_bonus_1 = float(self.near_goal_bonus_1)
        self.near_goal_bonus_radius_2 = float(self.near_goal_bonus_radius_2)
        self.near_goal_bonus_2 = float(self.near_goal_bonus_2)

        # 安全检查，确保生成的元素位置都合理
        if self.inter_agent_safe_distance <= 0.0:
            raise ValueError("inter_agent_safe_distance must be positive")
        if self.inter_agent_influence_distance <= 0.0:
            raise ValueError("inter_agent_influence_distance must be positive")
        if (
            self.nearest_agent_observation_count is not None
            and self.nearest_agent_observation_count < 0
        ):
            raise ValueError("nearest_agent_observation_count must be non-negative")
        if self.acceleration_penalty_weight < 0.0:
            raise ValueError("acceleration_penalty_weight must be non-negative")
        if self.acceleration_clip_penalty_weight < 0.0:
            raise ValueError("acceleration_clip_penalty_weight must be non-negative")
        if self.min_start_distance < 0.0:
            raise ValueError("min_start_distance must be non-negative")
        if self.min_goal_distance < 0.0:
            raise ValueError("min_goal_distance must be non-negative")
        if self.min_start_goal_distance < 0.0:
            raise ValueError("min_start_goal_distance must be non-negative")
        if self.start_goal_max_attempts <= 0:
            raise ValueError("start_goal_max_attempts must be positive")
        if self.near_goal_bonus_radius_1 < 0.0 or self.near_goal_bonus_radius_2 < 0.0:
            raise ValueError("near-goal bonus radii must be non-negative")
        if self.near_goal_bonus_1 < 0.0 or self.near_goal_bonus_2 < 0.0:
            raise ValueError("near-goal bonuses must be non-negative")


@dataclass
class _AgentAsDynamicObstacle:
    center: np.ndarray
    velocity: np.ndarray
    radius: float
    safety_margin: float = 0.0

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.radius = float(self.radius)
        self.safety_margin = float(self.safety_margin)
        if self.center.shape != (3,) or self.velocity.shape != (3,):
            raise ValueError("agent obstacle center and velocity must have shape (3,)")
        if self.radius <= 0.0:
            raise ValueError("agent obstacle radius must be positive")

    @property
    def effective_radius(self):
        return self.radius + self.safety_margin

    def signed_distance(self, point):
        point = np.asarray(point, dtype=float)
        return np.linalg.norm(point - self.center) - self.effective_radius

    def contains(self, point, margin=0.0):
        return self.signed_distance(point) <= float(margin)

    def closest_point(self, point):
        point = np.asarray(point, dtype=float)
        direction = point - self.center
        distance = np.linalg.norm(direction)
        if distance < 1e-8:
            direction = np.array([1.0, 0.0, 0.0], dtype=float)
            distance = 1.0
        return self.center + direction / distance * self.effective_radius

    def ray_intersection(self, origin, direction, max_distance):
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
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

    def to_feature(self, point):
        closest = self.closest_point(point)
        return {
            "closest_point": closest,
            "center": self.center.copy(),
            "velocity": self.velocity.copy(),
            "clearance": self.signed_distance(point),
            "size": self.effective_radius,
        }


class MultiAgentDMPEnv(gym.Env):
    """
    Matrix-style core environment for multi-agent DMP-RL.

    External training adapters may convert this interface to RLlib, HARL, or
    other multi-agent formats. This class keeps the physical simulation,
    observation construction, reward calculation, and termination logic.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        dynamics_config,    # 动力学参数
        sensor_config=None, # 传感器参数
        dmp_config=None,    # DMP参数
        env_config=None,    # 环境参数
        start_goal_generator: Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]] | None = None, # 随机生成起始点、目标点
        static_obstacles=None,  # 静态障碍物指定
        static_obstacle_generator=None, # 随机生成静态障碍物
        dynamic_obstacles=None,  # 动态障碍物
        dynamic_obstacle_generator=None, # 随机生成动态障碍物
        render_mode=None,
    ):
        if render_mode not in {None, "human"}:  
            raise ValueError("render_mode must be None or 'human'")

        self.dynamics_config = copy.deepcopy(dynamics_config)
        self.sensor_config = copy.deepcopy(sensor_config or {})

        if env_config is None:
            self.env_config = MultiAgentEnvConfig()
        elif isinstance(env_config, MultiAgentEnvConfig):
            self.env_config = env_config
        elif isinstance(env_config, dict):
            self.env_config = MultiAgentEnvConfig(**env_config)
        else:
            self.env_config = MultiAgentEnvConfig(**vars(env_config))

        self.num_agents = int(self.env_config.num_agents)

        if dmp_config is None:
            dt = self.dynamics_config.get("time_step", self.dynamics_config.get("timestep"))
            if dt is None:
                raise ValueError("dynamics_config must contain 'time_step' or 'timestep'")
            self.dmp_config = DMPConfig(dt=float(dt))
        elif isinstance(dmp_config, dict):
            self.dmp_config = DMPConfig(**dmp_config)
        else:
            self.dmp_config = dmp_config

        self.dynamics = [PartialDynamic(copy.deepcopy(self.dynamics_config)) for _ in range(self.num_agents)]
        self.sensors = [LocalObstacleSensor(**copy.deepcopy(self.sensor_config)) for _ in range(self.num_agents)]
        self.dmps = [SecondOrderDMPController(copy.deepcopy(self.dmp_config)) for _ in range(self.num_agents)]

        self.state_dim = int(self.dynamics[0].p.shape[0])
        for agent_index, dmp in enumerate(self.dmps):
            if dmp.config.dims != self.state_dim:
                raise ValueError(
                    f"agent {agent_index} dmp dims ({dmp.config.dims}) must match dynamics dimension ({self.state_dim})"
                )

        # 获取初始场景元素 指定位置优先考虑
        self._initial_static_obstacles = copy.deepcopy(static_obstacles or [])
        self._static_obstacle_generator = static_obstacle_generator
        self._initial_dynamic_obstacles = copy.deepcopy(dynamic_obstacles or [])
        self._dynamic_obstacle_generator = dynamic_obstacle_generator
        self._start_goal_generator = start_goal_generator

        # 默认起始点、目标点
        self._default_starts = np.zeros((self.num_agents, self.state_dim), dtype=float)
        self._default_goals = np.zeros((self.num_agents, self.state_dim), dtype=float)
        for agent_index in range(self.num_agents):
            self._default_starts[agent_index] = np.array([0.0, float(agent_index), 0.0], dtype=float)
            self._default_goals[agent_index] = np.array([8.0, float(agent_index), 0.0], dtype=float)
        
        # 初始场景元素 确保可复现
        self.static_obstacles = copy.deepcopy(self._initial_static_obstacles)
        self.workspace_boundary_obstacles = self._build_workspace_boundary_obstacles()
        self.dynamic_obstacles = copy.deepcopy(self._initial_dynamic_obstacles)
        self.starts = self._default_starts.copy()
        self.goals = self._default_goals.copy()
        self.steps = 0
        self.action_guidance_step = 0
        self.render_mode = render_mode

        # 缓存最新的传感器数据、控制器信息、观测值、碰撞信息等，供观察构建、奖励计算和信息输出使用
        self.latest_sensor_packets = [None for _ in range(self.num_agents)]
        self.latest_controller_infos = [{} for _ in range(self.num_agents)]
        self.latest_observation = None
        self.latest_collision_info = None
        self.success_rewarded_mask = np.zeros(self.num_agents, dtype=bool)

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    @property
    def single_agent_action_dim(self) -> int:
        return 2 * int(self.dmp_config.dims)

    @property
    def action_shape(self) -> tuple[int, int]:
        return self.num_agents, self.single_agent_action_dim

    @property
    def sensor_observation_dim(self) -> int:
        return int(self.sensors[0].observation_dim)

    @property
    def extra_observation_dim(self) -> int:
        return 3

    @property
    def single_pair_observation_dim(self) -> int:
        return 2 * self.state_dim + 1

    @property
    def nearest_agent_observation_count(self) -> int:
        max_count = max(self.num_agents - 1, 0)
        configured_count = self.env_config.nearest_agent_observation_count
        if configured_count is None:
            return max_count
        return min(int(configured_count), max_count)

    @property
    def inter_agent_observation_dim(self) -> int:
        return self.nearest_agent_observation_count * self.single_pair_observation_dim

    @property
    def single_agent_observation_dim(self) -> int:
        return self.sensor_observation_dim + self.extra_observation_dim + self.inter_agent_observation_dim

    @property
    def observation_shape(self) -> tuple[int, int]:
        return self.num_agents, self.single_agent_observation_dim

    @staticmethod
    def _as_state_vector(value, state_dim: int) -> np.ndarray:
        return np.broadcast_to(np.asarray(value, dtype=np.float32), (state_dim,)).astype(np.float32, copy=True)
    
    # 构建动作上下界
    def _build_single_agent_action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        dims = int(self.dmp_config.dims)
        low = np.concatenate(
            [
                np.full(dims, self.dmp_config.forcing_term_min, dtype=np.float32),
                np.full(dims, -self.dmp_config.goal_offset_max, dtype=np.float32),
            ],
            axis=0,
        )
        high = np.concatenate(
            [
                np.full(dims, self.dmp_config.forcing_term_max, dtype=np.float32),
                np.full(dims, self.dmp_config.goal_offset_max, dtype=np.float32),
            ],
            axis=0,
        )
        return low, high

    # 构建动作上下界
    def _build_action_space(self) -> spaces.Box:
        single_low, single_high = self._build_single_agent_action_bounds()  # 获取动作上下界
        return spaces.Box(
            low=np.tile(single_low, (self.num_agents, 1)),
            high=np.tile(single_high, (self.num_agents, 1)),
            dtype=np.float32,
        )

    def _build_observation_space(self) -> spaces.Box:   # 构建观测空间
        
        # 获取观测空间元素的上下界
        velocity_low = self._as_state_vector(self.dynamics[0].velocity_min, self.state_dim)
        velocity_high = self._as_state_vector(self.dynamics[0].velocity_max, self.state_dim)
        goal_direction_low = np.full(self.state_dim, -1.0, dtype=np.float32)
        goal_direction_high = np.full(self.state_dim, 1.0, dtype=np.float32)
        goal_distance_low = np.zeros(1, dtype=np.float32)
        goal_distance_high = np.ones(1, dtype=np.float32)
        scan_low = np.zeros(2 * self.sensors[0].n_rays, dtype=np.float32)
        scan_high = np.ones(2 * self.sensors[0].n_rays, dtype=np.float32)

        sensor_low = np.concatenate([velocity_low, goal_direction_low, goal_distance_low, scan_low], axis=0)
        sensor_high = np.concatenate([velocity_high, goal_direction_high, goal_distance_high, scan_high], axis=0)
        inter_agent_low = np.tile(
            np.concatenate(
                [
                    np.full(self.state_dim, -1.0, dtype=np.float32),
                    np.full(self.state_dim, -1.0, dtype=np.float32),
                    np.zeros(1, dtype=np.float32),
                ],
                axis=0,
            ),
            self.nearest_agent_observation_count,
        )
        inter_agent_high = np.tile(
            np.concatenate(
                [
                    np.full(self.state_dim, 1.0, dtype=np.float32),
                    np.full(self.state_dim, 1.0, dtype=np.float32),
                    np.ones(1, dtype=np.float32),
                ],
                axis=0,
            ),
            self.nearest_agent_observation_count,
        )
        extra_low = np.array([0.0, self.dmp_config.K_alpha, self.dmp_config.K_beta], dtype=np.float32)
        extra_high = np.array([1.0, self.dmp_config.K_alpha, self.dmp_config.K_beta], dtype=np.float32)

        # 封装单个智能体的观测空间边界
        single_low = np.concatenate([sensor_low, extra_low, inter_agent_low], axis=0)
        single_high = np.concatenate([sensor_high, extra_high, inter_agent_high], axis=0)
        
        # 封装多个智能体的观测空间边界
        return spaces.Box(
            low=np.tile(single_low, (self.num_agents, 1)),
            high=np.tile(single_high, (self.num_agents, 1)),
            dtype=np.float32,
        )

    def _validate_agent_points(self, name: str, value) -> np.ndarray:   # 验证智能体点的组织形式
        points = np.asarray(value, dtype=float)
        expected_shape = (self.num_agents, self.state_dim)
        if points.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        return points.copy()

    def _workspace_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        bounds = np.asarray(self.env_config.workspace_bounds, dtype=float)
        if bounds.shape != (2, self.state_dim):
            raise ValueError(f"workspace_bounds must have shape (2, {self.state_dim})")
        lower, upper = bounds
        if np.any(lower >= upper):
            raise ValueError("workspace lower bounds must be smaller than upper bounds")
        return lower.copy(), upper.copy()

    def _build_workspace_boundary_obstacles(self) -> list:
        lower, upper = self._workspace_bounds()
        if self.state_dim != 3:
            raise ValueError("workspace boundary obstacles require 3D dynamics")

        boundary_obstacles = []
        for axis in range(self.state_dim):
            boundary_obstacles.append(
                WorkspaceBoundaryPlaneObstacle(
                    axis=axis,
                    bound=lower[axis],
                    lower_bounds=lower,
                    upper_bounds=upper,
                    is_lower=True,
                )
            )
            boundary_obstacles.append(
                WorkspaceBoundaryPlaneObstacle(
                    axis=axis,
                    bound=upper[axis],
                    lower_bounds=lower,
                    upper_bounds=upper,
                    is_lower=False,
                )
            )
        return boundary_obstacles

    def _sensor_static_obstacles(self) -> list:
        return list(self.static_obstacles) + list(self.workspace_boundary_obstacles)

    def _configured_point_bounds(self, name: str, configured_bounds) -> tuple[np.ndarray, np.ndarray]:
        workspace_lower, workspace_upper = self._workspace_bounds()
        if configured_bounds is None:
            margin = max(float(self.env_config.goal_tolerance), 0.5 * float(self.env_config.inter_agent_safe_distance))
            lower = workspace_lower + margin
            upper = workspace_upper - margin
        else:
            bounds = np.asarray(configured_bounds, dtype=float)
            if bounds.shape != (2, self.state_dim):
                raise ValueError(f"{name} must have shape (2, {self.state_dim})")
            lower, upper = bounds

        if np.any(lower >= upper):
            raise ValueError(f"{name} lower bounds must be smaller than upper bounds")
        if np.any(lower < workspace_lower) or np.any(upper > workspace_upper):
            raise ValueError(f"{name} must stay inside workspace_bounds")
        return lower.astype(float, copy=True), upper.astype(float, copy=True)

    def _effective_start_spacing(self) -> float:
        return max(float(self.env_config.min_start_distance), float(self.env_config.inter_agent_safe_distance))

    def _effective_goal_spacing(self) -> float:
        required = float(self.env_config.inter_agent_safe_distance) + 2.0 * float(self.env_config.goal_tolerance)
        return max(float(self.env_config.min_goal_distance), required)

    @staticmethod
    def _pairwise_min_distance(points: np.ndarray) -> float:
        if len(points) <= 1:
            return float("inf")
        deltas = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(deltas, axis=-1)
        upper = distances[np.triu_indices(len(points), k=1)]
        return float(np.min(upper)) if upper.size else float("inf")

    def _points_inside_workspace(self, points: np.ndarray) -> bool:
        lower, upper = self._workspace_bounds()
        return bool(np.all(points >= lower) and np.all(points <= upper))

    def _sample_spaced_points(
        self,
        rng: np.random.Generator,
        lower: np.ndarray,
        upper: np.ndarray,
        min_distance: float,
    ) -> np.ndarray | None:
        points: list[np.ndarray] = []
        max_attempts = int(self.env_config.start_goal_max_attempts)
        for _ in range(self.num_agents):
            accepted = None
            for _ in range(max_attempts):
                candidate = rng.uniform(lower, upper).astype(float, copy=False)
                if all(float(np.linalg.norm(candidate - point)) >= min_distance for point in points):
                    accepted = candidate.copy()
                    break
            if accepted is None:
                return None
            points.append(accepted)
        return np.stack(points, axis=0)

    def _sample_goals_for_starts(self, rng: np.random.Generator, starts: np.ndarray) -> np.ndarray | None:
        goal_lower, goal_upper = self._configured_point_bounds(
            "goal_position_bounds",
            self.env_config.goal_position_bounds,
        )
        min_goal_distance = self._effective_goal_spacing()
        min_start_goal_distance = float(self.env_config.min_start_goal_distance)
        max_attempts = int(self.env_config.start_goal_max_attempts)
        goals: list[np.ndarray] = []
        for agent_index in range(self.num_agents):
            accepted = None
            for _ in range(max_attempts):
                candidate = rng.uniform(goal_lower, goal_upper).astype(float, copy=False)
                if float(np.linalg.norm(candidate - starts[agent_index])) < min_start_goal_distance:
                    continue
                if any(float(np.linalg.norm(candidate - goal)) < min_goal_distance for goal in goals):
                    continue
                accepted = candidate.copy()
                break
            if accepted is None:
                return None
            goals.append(accepted)
        return np.stack(goals, axis=0)

    def _sample_starts_goals(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        start_lower, start_upper = self._configured_point_bounds(
            "start_position_bounds",
            self.env_config.start_position_bounds,
        )
        max_attempts = int(self.env_config.start_goal_max_attempts)
        for _ in range(max_attempts):
            starts = self._sample_spaced_points(rng, start_lower, start_upper, self._effective_start_spacing())
            if starts is None:
                continue
            goals = self._sample_goals_for_starts(rng, starts)
            if goals is not None:
                return starts.astype(float, copy=True), goals.astype(float, copy=True)
        raise RuntimeError(
            "failed to sample multi-agent starts/goals under workspace, spacing, and start-goal constraints"
        )

    def _validate_start_goal_constraints(self, starts: np.ndarray, goals: np.ndarray) -> None:
        if not self._points_inside_workspace(starts):
            raise ValueError("starts must stay inside workspace_bounds")
        if not self._points_inside_workspace(goals):
            raise ValueError("goals must stay inside workspace_bounds")

        min_start_distance = self._effective_start_spacing()
        if self._pairwise_min_distance(starts) < min_start_distance:
            raise ValueError(f"starts must be at least {min_start_distance:.3f} apart")

        min_goal_distance = self._effective_goal_spacing()
        if self._pairwise_min_distance(goals) < min_goal_distance:
            raise ValueError(f"goals must be at least {min_goal_distance:.3f} apart")

        pair_distances = np.linalg.norm(goals - starts, axis=1)
        min_start_goal_distance = float(self.env_config.min_start_goal_distance)
        if np.any(pair_distances < min_start_goal_distance):
            raise ValueError(f"each start-goal pair must be at least {min_start_goal_distance:.3f} apart")

    def _resolve_starts_goals(self, options: dict) -> tuple[np.ndarray, np.ndarray]:    # 解析起始点与目标点
        has_starts = "starts" in options
        has_goals = "goals" in options
        if self._start_goal_generator is not None and not has_starts and not has_goals:
            max_attempts = int(self.env_config.start_goal_max_attempts)
            last_error: Exception | None = None
            for _ in range(max_attempts):
                starts, goals = self._start_goal_generator(self.np_random)
                starts = self._validate_agent_points("starts", starts)
                goals = self._validate_agent_points("goals", goals)
                try:
                    self._validate_start_goal_constraints(starts, goals)
                    return starts, goals
                except ValueError as exc:
                    last_error = exc
            raise RuntimeError("start_goal_generator failed to produce a feasible multi-agent task") from last_error
        elif bool(self.env_config.randomize_start_goal) and not has_starts and not has_goals:
            starts, goals = self._sample_starts_goals(self.np_random)
        else:
            starts = options.get("starts", self._default_starts)
            goals = options.get("goals", self._default_goals)
        starts = self._validate_agent_points("starts", starts)
        goals = self._validate_agent_points("goals", goals)
        self._validate_start_goal_constraints(starts, goals)
        return starts, goals

    def _generate_static_obstacles(self, starts: np.ndarray, goals: np.ndarray) -> list:    # 生成静态障碍物
        if self._static_obstacle_generator is None:
            return copy.deepcopy(self._initial_static_obstacles)

        generator_seed = int(self.np_random.integers(0, np.iinfo(np.uint32).max))
        try:
            return copy.deepcopy(
                self._static_obstacle_generator(
                    starts=starts.copy(),
                    goals=goals.copy(),
                    seed=generator_seed,
                )
            )
        except TypeError:
            return copy.deepcopy(
                self._static_obstacle_generator(
                    start=starts[0].copy(),
                    goal=goals[0].copy(),
                    seed=generator_seed,
                )
            )

    def _generate_dynamic_obstacles(self, starts: np.ndarray, goals: np.ndarray) -> list:   # 生成动态障碍物
        if self._dynamic_obstacle_generator is None:
            return copy.deepcopy(self._initial_dynamic_obstacles)

        generator_seed = int(self.np_random.integers(0, np.iinfo(np.uint32).max))   # 生成动态障碍物生成器的种子
        
        try:
            return copy.deepcopy(
                self._dynamic_obstacle_generator(
                    starts=starts.copy(),
                    goals=goals.copy(),
                    seed=generator_seed,
                    static_obstacles=copy.deepcopy(self.static_obstacles),
                )
            )
        except TypeError:
            return copy.deepcopy(
                self._dynamic_obstacle_generator(
                    start=starts[0].copy(),
                    goal=goals[0].copy(),
                    seed=generator_seed,
                    static_obstacles=copy.deepcopy(self.static_obstacles),
                )
            )

    def _compose_sensor_observation(self, sensor_packet) -> np.ndarray:     # 组合传感器观测
        return sensor_packet.observation.astype(np.float32, copy=True)

    def _sensor_dynamic_obstacles(self, agent_index: int) -> list:
        return list(self.dynamic_obstacles)

    def _compose_extra_observation(self, agent_index: int) -> np.ndarray:   # 组合额外观测
        dmp = self.dmps[agent_index]
        return np.array([dmp.phase, dmp.config.K_alpha, dmp.config.K_beta], dtype=np.float32)

    def _compose_inter_agent_observation(self, agent_index: int) -> np.ndarray: # 组合智能体之间的观测
        nearest_count = self.nearest_agent_observation_count
        if nearest_count <= 0:
            return np.zeros(0, dtype=np.float32)

        position = self.dynamics[agent_index].p
        velocity = self.dynamics[agent_index].v
        influence_distance = max(float(self.env_config.inter_agent_influence_distance), 1e-5)
        velocity_scale = max(float(np.max(np.abs(np.asarray(self.dynamics[agent_index].velocity_max, dtype=float)))), 1e-5)

        neighbor_entries = []
        for other_index in range(self.num_agents):
            if other_index == agent_index:
                continue
            other_position = self.dynamics[other_index].p
            distance = float(np.linalg.norm(other_position - position))
            neighbor_entries.append((distance, other_index))
        neighbor_entries.sort(key=lambda item: item[0])

        features = []
        for _, other_index in neighbor_entries[:nearest_count]:
            other_position = self.dynamics[other_index].p
            other_velocity = self.dynamics[other_index].v
            relative_position = (other_position - position) / influence_distance
            relative_velocity = (other_velocity - velocity) / velocity_scale
            distance = np.linalg.norm(other_position - position) / influence_distance
            features.append(
                np.concatenate(
                    [
                        np.clip(relative_position, -1.0, 1.0),
                        np.clip(relative_velocity, -1.0, 1.0),
                        np.array([np.clip(distance, 0.0, 1.0)], dtype=float),
                    ],
                    axis=0,
                )
            )
        return np.concatenate(features, axis=0).astype(np.float32)

    def get_observation(self) -> np.ndarray:    # 获取观测
        if any(packet is None for packet in self.latest_sensor_packets):
            raise RuntimeError("reset must be called before reading observation")

        observations = []
        for agent_index in range(self.num_agents):
            observations.append(
                np.concatenate(
                    [
                        self._compose_sensor_observation(self.latest_sensor_packets[agent_index]),
                        self._compose_extra_observation(agent_index),
                        self._compose_inter_agent_observation(agent_index),
                    ],
                    axis=0,
                ).astype(np.float32)
            )
        observation = np.stack(observations, axis=0).astype(np.float32)
        self.latest_observation = observation.copy()
        return observation

    def _positions(self) -> np.ndarray:
        return np.stack([dynamic.p.copy() for dynamic in self.dynamics], axis=0)

    def _velocities(self) -> np.ndarray:
        return np.stack([dynamic.v.copy() for dynamic in self.dynamics], axis=0)

    # 获取智能体之间的距离
    def _compute_pairwise_distances(self) -> np.ndarray:
        positions = self._positions()
        deltas = positions[:, None, :] - positions[None, :, :]
        return np.linalg.norm(deltas, axis=-1).astype(np.float32)

    # 获取智能体之间的碰撞矩阵和碰撞掩码 数据组织形式：一张agent_n * agent_n 的布尔值矩阵
    def _compute_inter_agent_collision_mask(self, pairwise_distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        collision_matrix = np.zeros((self.num_agents, self.num_agents), dtype=bool)
        if self.num_agents <= 1:
            return np.zeros(self.num_agents, dtype=bool), collision_matrix

        threshold = float(self.env_config.inter_agent_safe_distance)
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if pairwise_distances[i, j] <= threshold:
                    collision_matrix[i, j] = True
                    collision_matrix[j, i] = True
        return np.any(collision_matrix, axis=1), collision_matrix

    def _compute_obstacle_collision_mask(self) -> np.ndarray:   # 计算障碍物碰撞矩阵
        collision_mask = np.zeros(self.num_agents, dtype=bool)
        margin = float(self.env_config.collision_margin)
        obstacles = self.static_obstacles + self.dynamic_obstacles
        for agent_index, dynamic in enumerate(self.dynamics):
            for obstacle in obstacles:
                if obstacle.contains(dynamic.p, margin=margin):
                    collision_mask[agent_index] = True
                    break
        return collision_mask

    def _check_collision(self) -> dict: # 检查碰撞
        pairwise_distances = self._compute_pairwise_distances()
        inter_agent_mask, inter_agent_matrix = self._compute_inter_agent_collision_mask(pairwise_distances)
        obstacle_mask = self._compute_obstacle_collision_mask()
        boundary_mask = self._compute_boundary_collision_mask()
        collision_mask = np.logical_or.reduce((obstacle_mask, inter_agent_mask, boundary_mask))
        min_inter_agent_distance = float("inf")
        if self.num_agents > 1:
            upper = pairwise_distances[np.triu_indices(self.num_agents, k=1)]
            min_inter_agent_distance = float(np.min(upper)) if upper.size else float("inf")
        return {
            "obstacle_collision_mask": obstacle_mask,
            "inter_agent_collision_mask": inter_agent_mask,
            "boundary_collision_mask": boundary_mask,
            "inter_agent_collision_matrix": inter_agent_matrix,
            "collision_mask": collision_mask,
            "pairwise_distances": pairwise_distances,
            "min_inter_agent_distance": min_inter_agent_distance,
            "collision": bool(np.any(collision_mask)),
        }

    def _compute_min_boundary_distances(self) -> np.ndarray:   # 计算边界最小距离，用于给出APF惩罚项的值
        bounds = np.asarray(self.env_config.workspace_bounds, dtype=float)
        if bounds.shape != (2, self.state_dim):
            raise ValueError(f"workspace_bounds must have shape (2, {self.state_dim})")
        lower, upper = bounds
        return np.array(
            [
                np.min(np.concatenate([dynamic.p - lower, upper - dynamic.p]))
                for dynamic in self.dynamics
            ],
            dtype=np.float32,
        )

    def _compute_boundary_collision_mask(self) -> np.ndarray:
        return self._compute_min_boundary_distances() < 0.0
    
    def _compute_policy_action_direction(self, agent_index: int, policy_action: np.ndarray) -> np.ndarray | None:
        forcing_component = np.asarray(policy_action[: self.state_dim], dtype=float)
        goal_offset = np.asarray(policy_action[self.state_dim : 2 * self.state_dim], dtype=float)
        dynamic = self.dynamics[agent_index]
        dmp = self.dmps[agent_index]

        goal_eff = self.goals[agent_index] + goal_offset
        forcing_gate = np.tanh(np.abs(goal_eff - dynamic.p))
        action_component = (
            dmp.config.K_alpha * dmp.config.K_beta * goal_offset
            + forcing_component * forcing_gate
        )
        action_norm = float(np.linalg.norm(action_component))
        if action_norm < 1e-8:
            return None
        return action_component / action_norm

    def _apply_action_guidance(
        self,
        actions: np.ndarray,
        distances_to_goals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        weights = np.zeros(self.num_agents, dtype=np.float32)
        if not self.env_config.action_guidance_enabled:
            return actions, weights

        radius = float(self.env_config.action_guidance_radius)
        if radius <= 0.0:
            return actions, weights

        decay_steps = max(1, int(self.env_config.action_guidance_decay_steps))
        training_gate = max(0.0, 1.0 - float(self.action_guidance_step) / float(decay_steps))
        if training_gate <= 0.0:
            return actions, weights

        guided_actions = actions.astype(np.float32, copy=True)
        initial_weight = float(np.clip(self.env_config.action_guidance_initial_weight, 0.0, 1.0))
        for agent_index, distance in enumerate(distances_to_goals):
            if self.success_rewarded_mask[agent_index] or float(distance) >= radius:
                continue
            distance_gate = 1.0 - max(0.0, float(distance)) / radius
            guidance_weight = float(np.clip(initial_weight * distance_gate * training_gate, 0.0, 1.0))
            if guidance_weight <= 0.0:
                continue
            guided_actions[agent_index] = (1.0 - guidance_weight) * guided_actions[agent_index]
            weights[agent_index] = guidance_weight

        guided_actions = np.clip(guided_actions, self.action_space.low, self.action_space.high)
        return guided_actions.astype(np.float32, copy=False), weights

    def _freeze_agent(self, agent_index: int) -> np.ndarray:
        dynamic = self.dynamics[agent_index]
        return dynamic.reset(
            {
                "position": dynamic.p.copy(),
                "velocity": np.zeros(self.state_dim, dtype=float),
            }
        )

    def _compute_obstacle_potential_penalties(self, policy_actions: np.ndarray) -> np.ndarray:  # 计算障碍物势场
        penalties = np.zeros(self.num_agents, dtype=np.float32)
        influence_distance = float(self.env_config.obstacle_influence_distance)
        if influence_distance <= 0.0:
            return penalties

        obstacles = self.static_obstacles + self.dynamic_obstacles  # 获取障碍物集合
        if not obstacles:
            return penalties

        for agent_index in range(self.num_agents):
            action_dir = self._compute_policy_action_direction(agent_index, policy_actions[agent_index])
            if action_dir is None:
                continue
            dynamic = self.dynamics[agent_index]
            velocity_norm = float(np.linalg.norm(dynamic.v))
            velocity_dir = dynamic.v / velocity_norm if velocity_norm >= 1e-8 else None
            penalty_sum = 0.0

            for obstacle in obstacles:
                closest_point = obstacle.closest_point(dynamic.p)   # 当前位置到障碍物的最近点
                to_obstacle = closest_point - dynamic.p # 当前位置到障碍物最近点的向量
                distance_to_surface = float(np.linalg.norm(to_obstacle))    # 当前位置到障碍物最近点的距离
                if distance_to_surface < 1e-8 or distance_to_surface >= influence_distance: # 足够小已经碰撞，足够大不参与计算
                    continue

                obstacle_dir = to_obstacle / distance_to_surface    # 计算方向向量
                action_gate = max(0.0, float(np.dot(action_dir, obstacle_dir))) # 计算动作方向与障碍物方向的门控量
                # 这里action_dir和obstacle_dir都是单位向量，直接得到余弦值
                if action_gate <= 0.0:  # 余弦值为0，则动作方向与障碍物方向平行或相反，不参与计算
                    continue
                velocity_gate = 0.0 if velocity_dir is None else max(0.0, float(np.dot(velocity_dir, obstacle_dir)))
                directional_gate = action_gate * (0.5 + 0.5 * velocity_gate)
                d = max(distance_to_surface, 1e-3)
                penalty_sum += ((1.0 / d - 1.0 / influence_distance) ** 2) * directional_gate   # 改进的APF值

            penalty = float(self.env_config.obstacle_potential_weight) * penalty_sum    # 计算障碍物市场的值
            penalties[agent_index] = min(float(penalty), float(self.env_config.obstacle_potential_penalty_max)) # 截断
        return penalties

    def _compute_boundary_potential_penalties(self, policy_actions: np.ndarray) -> np.ndarray:  # 计算环境边界惩罚

        """
        计算智能体靠近环境边界时的势场惩罚。

        基于改进的人工势场法(APF)，当智能体的策略动作方向朝向边界且距离边界小于影响距离时，
        施加方向性感知的势场惩罚。惩罚值考虑了动作方向和速度方向的组合门控效应，确保只有
        真正朝向边界运动的智能体才会受到惩罚。

        Args:
            policy_actions: 策略网络输出的动作数组，形状为 (num_agents, action_dim)。
                           每个智能体的动作包含 forcing term 和 goal offset 两部分。

        Returns:
            每个智能体的边界势场惩罚值数组，形状为 (num_agents,)，数据类型为 float32。
            惩罚值已根据配置参数进行加权并限制在最大值范围内。

        Raises:
            ValueError: 当 workspace_bounds 的形状不符合 (2, state_dim) 要求时抛出。
        """
         
        penalties = np.zeros(self.num_agents, dtype=np.float32)
        bounds = np.asarray(self.env_config.workspace_bounds, dtype=float)
        if bounds.shape != (2, self.state_dim):
            raise ValueError(f"workspace_bounds must have shape (2, {self.state_dim})")

        influence_distance = float(self.env_config.boundary_influence_distance)
        if influence_distance <= 0.0:
            return penalties

        lower, upper = bounds
        epsilon = max(float(self.env_config.boundary_distance_epsilon), 1e-8)
        for agent_index in range(self.num_agents):
            action_dir = self._compute_policy_action_direction(agent_index, policy_actions[agent_index])
            if action_dir is None:
                continue
            dynamic = self.dynamics[agent_index]
            velocity_norm = float(np.linalg.norm(dynamic.v))
            velocity_dir = dynamic.v / velocity_norm if velocity_norm >= 1e-8 else None
            penalty_sum = 0.0

            for axis in range(self.state_dim):
                boundary_cases = (
                    (float(dynamic.p[axis] - lower[axis]), -1.0),
                    (float(upper[axis] - dynamic.p[axis]), 1.0),
                )

                for signed_distance, direction_sign in boundary_cases:  # 遍历边界情况
                    if signed_distance >= influence_distance:
                        continue
                    boundary_dir = np.zeros(self.state_dim, dtype=float)
                    boundary_dir[axis] = direction_sign
                    action_gate = max(0.0, float(np.dot(action_dir, boundary_dir)))
                    if action_gate <= 0.0:
                        continue
                    velocity_gate = 0.0 if velocity_dir is None else max(0.0, float(np.dot(velocity_dir, boundary_dir)))
                    directional_gate = action_gate * (0.5 + 0.5 * velocity_gate)
                    d = max(signed_distance, epsilon)
                    penalty_sum += ((1.0 / d - 1.0 / influence_distance) ** 2) * directional_gate

            penalty = float(self.env_config.boundary_potential_weight) * penalty_sum
            penalties[agent_index] = min(float(penalty), float(self.env_config.boundary_potential_penalty_max))
        return penalties

    def _compute_inter_agent_potential_penalties(self, pairwise_distances: np.ndarray) -> np.ndarray:   # 智能体间避障惩罚
        
        """
        计算智能体间基于人工势场法的避障惩罚值。

        使用改进的人工势场法计算智能体之间的碰撞惩罚。当两个智能体之间的距离
        小于设定的影响距离时，会产生惩罚值，且距离越近惩罚越大。惩罚值会同时
        分配给相互作用的两个智能体。

        Args:
            pairwise_distances: 智能体间的两两距离矩阵，形状为 (num_agents, num_agents)，
                              其中 pairwise_distances[i, j] 表示智能体 i 和 j 之间的距离。

        Returns:
            每个智能体的避障惩罚值数组，形状为 (num_agents,)，索引 i 对应第 i 个智能体的总惩罚值。
        """

        penalties = np.zeros(self.num_agents, dtype=np.float32)
        if self.num_agents <= 1:
            return penalties

        influence_distance = float(self.env_config.inter_agent_influence_distance)
        if influence_distance <= 0.0:
            return penalties

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                distance = float(pairwise_distances[i, j])
                if distance >= influence_distance:
                    continue
                d = max(distance, 1e-3) # 防止除0

                # 平方反比加入惩罚项
                penalty = float(self.env_config.inter_agent_potential_weight) * (
                    1.0 / d - 1.0 / influence_distance
                ) ** 2
                # 同时作用于双方无人机
                penalties[i] += penalty
                penalties[j] += penalty

        return penalties

    def _compute_near_goal_bonuses(self, distances_to_goals: np.ndarray) -> np.ndarray:
        bonuses = np.zeros(self.num_agents, dtype=np.float32)
        active_mask = np.logical_and(
            np.logical_not(self.success_rewarded_mask),
            distances_to_goals > float(self.env_config.goal_tolerance),
        )

        radius_1 = float(self.env_config.near_goal_bonus_radius_1)
        if radius_1 > 0.0 and float(self.env_config.near_goal_bonus_1) > 0.0:
            mask = np.logical_and(active_mask, distances_to_goals < radius_1)
            bonuses[mask] += float(self.env_config.near_goal_bonus_1)

        radius_2 = float(self.env_config.near_goal_bonus_radius_2)
        if radius_2 > 0.0 and float(self.env_config.near_goal_bonus_2) > 0.0:
            mask = np.logical_and(active_mask, distances_to_goals < radius_2)
            bonuses[mask] += float(self.env_config.near_goal_bonus_2)

        return bonuses

    def _build_info(    # 构建信息字典
        self,
        *,
        success_mask,
        new_success_mask,
        success_reward_bonus,
        collision_info,
        truncated,
        distances_to_goals,
        progress,
        commanded_accelerations,
        applied_accelerations,
        next_states,
        raw_action,
        guided_action,
        action_guidance_weights,
        step_rewards,
        near_goal_bonuses,
        obstacle_potential_penalties,
        boundary_potential_penalties,
        inter_agent_potential_penalties,
        acceleration_penalties,
        acceleration_clip_penalties,
        collision_penalties,
        timeout_penalties,
    ) -> dict:
        min_clearances = np.array(
            [
                float(packet.min_clearance) if packet is not None else np.nan
                for packet in self.latest_sensor_packets
            ],
            dtype=np.float32,
        )
        phases = np.array(
            [
                float(info.get("phase", self.dmps[index].phase))
                for index, info in enumerate(self.latest_controller_infos)
            ],
            dtype=np.float32,
        )
        taus = np.array(
            [
                float(info.get("tau", self.dmps[index].config.tau))
                for index, info in enumerate(self.latest_controller_infos)
            ],
            dtype=np.float32,
        )
        return {
            "success": bool(np.all(success_mask)),
            "success_mask": success_mask.astype(bool).copy(),
            "new_success_mask": new_success_mask.astype(bool).copy(),
            "success_rewarded_mask": self.success_rewarded_mask.astype(bool).copy(),
            "per_agent_success_bonus": float(success_reward_bonus),
            "collision": bool(collision_info["collision"]),
            "collision_mask": collision_info["collision_mask"].astype(bool).copy(),
            "obstacle_collision_mask": collision_info["obstacle_collision_mask"].astype(bool).copy(),
            "inter_agent_collision_mask": collision_info["inter_agent_collision_mask"].astype(bool).copy(),
            "boundary_collision_mask": collision_info["boundary_collision_mask"].astype(bool).copy(),
            "inter_agent_collision_matrix": collision_info["inter_agent_collision_matrix"].astype(bool).copy(),
            "truncated": bool(truncated),
            "steps": int(self.steps),
            "distance_to_goals": distances_to_goals.astype(np.float32).copy(),
            "progress": progress.astype(np.float32).copy(),
            "pairwise_distances": collision_info["pairwise_distances"].astype(np.float32).copy(),
            "min_inter_agent_distance": float(collision_info["min_inter_agent_distance"]),
            "min_boundary_distances": self._compute_min_boundary_distances(),
            "min_clearances": min_clearances,
            "phases": phases,
            "taus": taus,
            "commanded_accelerations": commanded_accelerations.astype(np.float32).copy(),
            "applied_accelerations": applied_accelerations.astype(np.float32).copy(),
            "next_states": next_states.astype(np.float32).copy(),
            "raw_action": raw_action.astype(np.float32).copy(),
            "guided_action": guided_action.astype(np.float32).copy(),
            "action_guidance_weights": action_guidance_weights.astype(np.float32).copy(),
            "action_guidance_weight": float(np.mean(action_guidance_weights)),
            "reward_step": step_rewards.astype(np.float32).copy(),
            "reward_near_goal_bonus": near_goal_bonuses.astype(np.float32).copy(),
            "reward_obstacle_potential_penalty": obstacle_potential_penalties.astype(np.float32).copy(),
            "reward_boundary_potential_penalty": boundary_potential_penalties.astype(np.float32).copy(),
            "reward_inter_agent_potential_penalty": inter_agent_potential_penalties.astype(np.float32).copy(),
            "reward_acceleration_penalty": acceleration_penalties.astype(np.float32).copy(),
            "reward_acceleration_clip_penalty": acceleration_clip_penalties.astype(np.float32).copy(),
            "reward_collision_penalty": collision_penalties.astype(np.float32).copy(),
            "reward_timeout_penalty": timeout_penalties.astype(np.float32).copy(),
        }

    def reset(self, *, seed=None, options=None):    # 重置环境
        try:
            super().reset(seed=seed)
        except TypeError:
            if seed is not None or not hasattr(self, "np_random"):
                self.np_random = np.random.default_rng(seed)
        options = options or {}

        starts, goals = self._resolve_starts_goals(options)
        self.starts = starts.copy()
        self.goals = goals.copy()
        self.steps = 0
        self.success_rewarded_mask = np.zeros(self.num_agents, dtype=bool)

        if "static_obstacles" in options:
            self.static_obstacles = copy.deepcopy(options["static_obstacles"])
        else:
            self.static_obstacles = self._generate_static_obstacles(starts, goals)

        if "dynamic_obstacles" in options:
            self.dynamic_obstacles = copy.deepcopy(options["dynamic_obstacles"])
        else:
            self.dynamic_obstacles = self._generate_dynamic_obstacles(starts, goals)

        zero_velocity = np.zeros(self.state_dim, dtype=float)
        for agent_index in range(self.num_agents):
            self.dynamics[agent_index].reset(
                {
                    "position": starts[agent_index].copy(),
                    "velocity": zero_velocity.copy(),
                }
            )
            self.dmps[agent_index].reset(starts[agent_index], goals[agent_index])
            self.sensors[agent_index].reset()
            self.latest_controller_infos[agent_index] = {
                "phase": float(self.dmps[agent_index].phase),
                "tau": float(self.dmps[agent_index].config.tau),
            }

        for agent_index in range(self.num_agents):
            self.latest_sensor_packets[agent_index] = self.sensors[agent_index].sense(
                self.dynamics[agent_index].p,
                self.dynamics[agent_index].v,
                goals[agent_index],
                self._sensor_static_obstacles(),
                self._sensor_dynamic_obstacles(agent_index),
            )

        observation = self.get_observation()
        collision_info = self._check_collision()
        self.latest_collision_info = collision_info
        distances_to_goals = np.array(
            [np.linalg.norm(self.goals[index] - self.dynamics[index].p) for index in range(self.num_agents)],
            dtype=np.float32,
        )
        info = {
            "starts": starts.copy(),
            "goals": goals.copy(),
            "num_agents": int(self.num_agents),
            "static_obstacle_count": len(self.static_obstacles),
            "dynamic_obstacle_count": len(self.dynamic_obstacles),
            "distance_to_goals": distances_to_goals,
            "collision": bool(collision_info["collision"]),
            "collision_mask": collision_info["collision_mask"].copy(),
            "obstacle_collision_mask": collision_info["obstacle_collision_mask"].copy(),
            "inter_agent_collision_mask": collision_info["inter_agent_collision_mask"].copy(),
            "boundary_collision_mask": collision_info["boundary_collision_mask"].copy(),
            "pairwise_distances": collision_info["pairwise_distances"].copy(),
            "min_inter_agent_distance": float(collision_info["min_inter_agent_distance"]),
            "min_boundary_distances": self._compute_min_boundary_distances(),
        }
        return observation, info

    def step(self, action):
        if any(packet is None for packet in self.latest_sensor_packets):    # 确保传感器数据已更新
            raise RuntimeError("reset must be called before step")

        action = np.asarray(action, dtype=np.float32)   # 构建动作形状并检查
        if action.shape != self.action_shape:
            raise ValueError(f"action must have shape {self.action_shape}")
        
        action = np.clip(action, self.action_space.low, self.action_space.high) # 裁剪动作
        raw_action = action.copy()

        previous_distances = np.array(
            [np.linalg.norm(self.goals[index] - self.dynamics[index].p) for index in range(self.num_agents)],
            dtype=float,
        )   # 获取当前到目标点的距离
        action, action_guidance_weights = self._apply_action_guidance(action, previous_distances)
        guided_action = action.copy()

        # 计算命令加速度和实际加速度
        commanded_accelerations = np.zeros((self.num_agents, self.state_dim), dtype=np.float32) 
        applied_accelerations = np.zeros((self.num_agents, self.state_dim), dtype=np.float32)
        next_states = np.zeros((self.num_agents, 2 * self.state_dim), dtype=np.float32)

        for agent_index in range(self.num_agents):
            if self.success_rewarded_mask[agent_index]:
                action[agent_index] = 0.0
                guided_action[agent_index] = 0.0
                next_states[agent_index] = self._freeze_agent(agent_index)
                self.latest_controller_infos[agent_index] = {
                    "phase": float(self.dmps[agent_index].phase),
                    "tau": float(self.dmps[agent_index].config.tau),
                }
                continue

            acceleration, controller_info = self.dmps[agent_index].compute_acceleration(
                self.dynamics[agent_index].p,
                self.dynamics[agent_index].v,
                action[agent_index],
                sensor_packet=self.latest_sensor_packets[agent_index],
            )   # 由动作输出获取命令加速度
            applied_acceleration = np.clip(
                acceleration,
                self.dynamics[agent_index].accelerate_min,
                self.dynamics[agent_index].accelerate_max,
            )   # 应用加速度
            self.latest_controller_infos[agent_index] = controller_info     # 更新控制器信息
            commanded_accelerations[agent_index] = np.asarray(acceleration, dtype=np.float32)   # 记录命令加速度
            applied_accelerations[agent_index] = np.asarray(applied_acceleration, dtype=np.float32) # 应用加速度
            next_states[agent_index] = self.dynamics[agent_index].step(applied_acceleration)    # 更新动力学并记录下一状态

        for obstacle in self.dynamic_obstacles: # 更新动态障碍物状态
            obstacle.step(self.dynamics[0].dt)

        self.steps += 1 # 步数加1
        self.action_guidance_step += 1
        for agent_index in range(self.num_agents):  # 更新传感器数据
            self.latest_sensor_packets[agent_index] = self.sensors[agent_index].sense(
                self.dynamics[agent_index].p,
                self.dynamics[agent_index].v,
                self.goals[agent_index],
                self._sensor_static_obstacles(),
                self._sensor_dynamic_obstacles(agent_index),
            )   # 获取传感器数据

        observation = self.get_observation()    # 获取传感器数据
        current_distances = np.array(
            [np.linalg.norm(self.goals[index] - self.dynamics[index].p) for index in range(self.num_agents)],
            dtype=float,
        )   # 获取当前到目标点的距离
        progress = previous_distances - current_distances   # 获取进度
        step_rewards = float(self.env_config.step_reward_weight) * progress # 获取奖励
        near_goal_bonuses = self._compute_near_goal_bonuses(current_distances)

        collision_info = self._check_collision()    # 检查碰撞
        self.latest_collision_info = collision_info # 更新碰撞信息
        obstacle_potential_penalties = self._compute_obstacle_potential_penalties(raw_action)   # 计算障碍物势能惩罚
        boundary_potential_penalties = self._compute_boundary_potential_penalties(raw_action)   # 计算边界势能惩罚
        inter_agent_potential_penalties = self._compute_inter_agent_potential_penalties(
            collision_info["pairwise_distances"]
        )   # 获取智能体间的势能惩罚
        acceleration_penalties = (
            float(self.env_config.acceleration_penalty_weight)
            * np.sum(applied_accelerations.astype(float) ** 2, axis=1)  # 加速度越大惩罚越大
        )   # 获取加速度惩罚
        acceleration_clip_penalties = (
            float(self.env_config.acceleration_clip_penalty_weight)
            * np.sum((commanded_accelerations.astype(float) - applied_accelerations.astype(float)) ** 2, axis=1)
        )   # 获取加速度裁剪惩罚

        rewards = (
            step_rewards
            + near_goal_bonuses
            - obstacle_potential_penalties
            - boundary_potential_penalties
            - inter_agent_potential_penalties
            # - acceleration_penalties
            # - acceleration_clip_penalties
            - float(self.env_config.step_penalty)
        ).astype(np.float32)

        success_mask = current_distances <= float(self.env_config.goal_tolerance)
        new_success_mask = np.logical_and(success_mask, np.logical_not(self.success_rewarded_mask))
        full_success = bool(np.all(success_mask))
        collision = bool(collision_info["collision"])
        terminated = bool(full_success or collision)
        truncated = bool((not terminated) and (self.steps >= int(self.env_config.max_steps)))

        collision_penalties = np.zeros(self.num_agents, dtype=np.float32)
        if np.any(collision_info["obstacle_collision_mask"]):
            obstacle_mask = collision_info["obstacle_collision_mask"]
            collision_penalties[obstacle_mask] += float(self.env_config.collision_penalty)
        if np.any(collision_info["inter_agent_collision_mask"]):
            inter_mask = collision_info["inter_agent_collision_mask"]
            collision_penalties[inter_mask] += float(self.env_config.inter_agent_collision_penalty)
        if np.any(collision_info["boundary_collision_mask"]):
            boundary_mask = collision_info["boundary_collision_mask"]
            collision_penalties[boundary_mask] += float(self.env_config.collision_penalty)
        rewards -= collision_penalties

        success_reward_bonus = float(self.env_config.success_bonus) / float(self.num_agents)
        if not collision:
            rewards[new_success_mask] += success_reward_bonus
            self.success_rewarded_mask = np.logical_or(self.success_rewarded_mask, new_success_mask)
            for agent_index in np.flatnonzero(new_success_mask):
                next_states[agent_index] = self._freeze_agent(int(agent_index))

        timeout_penalties = np.zeros(self.num_agents, dtype=np.float32)
        if truncated:
            bounds = np.asarray(self.env_config.workspace_bounds, dtype=float)
            workspace_diag = float(np.linalg.norm(bounds[1] - bounds[0]))
            distance_scale = max(workspace_diag, 1e-6)
            unfinished_mask = np.logical_not(self.success_rewarded_mask)
            timeout_penalties[unfinished_mask] = (
                float(self.env_config.timeout_penalty)
                * current_distances[unfinished_mask]
                / distance_scale
            )
            rewards -= timeout_penalties

        info = self._build_info(
            success_mask=success_mask,
            new_success_mask=new_success_mask,
            success_reward_bonus=success_reward_bonus,
            collision_info=collision_info,
            truncated=truncated,
            distances_to_goals=current_distances.astype(np.float32),
            progress=progress.astype(np.float32),
            commanded_accelerations=commanded_accelerations,
            applied_accelerations=applied_accelerations,
            next_states=next_states,
            raw_action=raw_action,
            guided_action=guided_action,
            action_guidance_weights=action_guidance_weights,
            step_rewards=step_rewards.astype(np.float32),
            near_goal_bonuses=near_goal_bonuses,
            obstacle_potential_penalties=obstacle_potential_penalties,
            boundary_potential_penalties=boundary_potential_penalties,
            inter_agent_potential_penalties=inter_agent_potential_penalties,
            acceleration_penalties=acceleration_penalties.astype(np.float32),
            acceleration_clip_penalties=acceleration_clip_penalties.astype(np.float32),
            collision_penalties=collision_penalties,
            timeout_penalties=timeout_penalties,
        )
        return observation, rewards.astype(np.float32), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        collision_info = self.latest_collision_info or self._check_collision()
        return {
            "positions": self._positions(),
            "velocities": self._velocities(),
            "goals": self.goals.copy(),
            "steps": int(self.steps),
            "collision": bool(collision_info["collision"]),
            "collision_mask": collision_info["collision_mask"].copy(),
            "pairwise_distances": collision_info["pairwise_distances"].copy(),
        }

    def close(self):
        return None
