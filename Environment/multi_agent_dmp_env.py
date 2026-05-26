import copy
from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# 复用dmp框架和实体框架
from Controller.dmp_rl import DMPConfig, SecondOrderDMPController
from Controller.dmp_rl import DMPConfig, SecondOrderDMPController
from Entity.KinematicModel import PartialDynamic
from Entity.sensors import LocalObstacleSensor
from Environment.single_agent_dmp_env import EnvConfig


@dataclass
class MultiAgentEnvConfig(EnvConfig):
    """
    Multi-agent environment-level configuration.

    The base task and reward fields are inherited from the single-agent
    environment. The fields below only describe inter-agent constraints.
    """

    num_agents: int = 4

    # 机间奖励值参数
    inter_agent_safe_distance: float = 0.6
    inter_agent_collision_penalty: float = 20.0
    inter_agent_potential_weight: float = 1.0
    inter_agent_influence_distance: float = 1.2

    def __post_init__(self) -> None:        # 检查传入参数合法性
        self.num_agents = int(self.num_agents)
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        self.inter_agent_safe_distance = float(self.inter_agent_safe_distance)
        self.inter_agent_collision_penalty = float(self.inter_agent_collision_penalty)
        self.inter_agent_potential_weight = float(self.inter_agent_potential_weight)
        self.inter_agent_influence_distance = float(self.inter_agent_influence_distance)
        if self.inter_agent_safe_distance <= 0.0:
            raise ValueError("inter_agent_safe_distance must be positive")
        if self.inter_agent_influence_distance <= 0.0:
            raise ValueError("inter_agent_influence_distance must be positive")


class MultiAgentDMPEnv(gym.Env):    # 复用gym
    """
    Multi-agent DMP-RL environment skeleton.

    Module 1 only defines construction-time ownership:
    - one point-mass dynamics module per agent;
    - one DMP controller per agent;
    - one local obstacle sensor per agent;
    - per-agent action matrix with shape (num_agents, single_agent_action_dim).

    reset(), step(), observation construction, reward and collision logic will be
    implemented in later reviewed modules.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        dynamics_config,    # 动力学模型的参数设置
        sensor_config=None, # 传感器设置
        dmp_config=None,    # dmp设置
        env_config=None,    # 环境设置
        start_goal_generator: Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]] | None = None, # 训练采样策略
        static_obstacles=None,  # 静态障碍物字典传入
        static_obstacle_generator=None, # 静态障碍物生成
        dynamic_obstacles=None, # 同静态障碍物
        dynamic_obstacle_generator=None,
        render_mode=None,
    ):
        if render_mode not in {None, "human"}:
            raise ValueError("render_mode must be None or 'human'")     # gym.env只允许两个值

        self.env_config = env_config or MultiAgentEnvConfig()   # 
        if not isinstance(self.env_config, MultiAgentEnvConfig):    # 判断是否为单机类型，若是，则转化为多机参数类
            self.env_config = MultiAgentEnvConfig(**vars(self.env_config))  # 

        self.num_agents = int(self.env_config.num_agents)   # 获取智能体数量

        # 构建self，得到对应的变量关系
        self.dynamics_config = copy.deepcopy(dynamics_config)   
        self.sensor_config = copy.deepcopy(sensor_config or {})
        self.dmp_config = dmp_config or DMPConfig(dt=float(self.dynamics_config["time_step"]))  # 入参指定值或默认值

        self.dynamics = [PartialDynamic(copy.deepcopy(self.dynamics_config)) for _ in range(self.num_agents)]
        self.sensors = [LocalObstacleSensor(**copy.deepcopy(self.sensor_config)) for _ in range(self.num_agents)]
        self.dmps = [SecondOrderDMPController(copy.deepcopy(self.dmp_config)) for _ in range(self.num_agents)]

        self.state_dim = int(self.dynamics[0].p.shape[0])
        for agent_index, dmp in enumerate(self.dmps):

            if dmp.config.dims != self.state_dim:   # 检查每个agent的dmp维度和动力学维度是否吻合
                raise ValueError(
                    f"agent {agent_index} dmp dims ({dmp.config.dims}) must match dynamics dimension ({self.state_dim})"
                )

        # 初始化障碍物信息
        self._initial_static_obstacles = copy.deepcopy(static_obstacles or [])
        self._static_obstacle_generator = static_obstacle_generator
        self._initial_dynamic_obstacles = copy.deepcopy(dynamic_obstacles or [])
        self._dynamic_obstacle_generator = dynamic_obstacle_generator
        self._start_goal_generator = start_goal_generator

        # 默认起点和目标点构建
        self._default_starts = np.zeros((self.num_agents, self.state_dim), dtype=float)
        self._default_goals = np.zeros((self.num_agents, self.state_dim), dtype=float)
        for agent_index in range(self.num_agents):
            self._default_starts[agent_index] = np.array([0.0, float(agent_index), 0.0], dtype=float)
            self._default_goals[agent_index] = np.array([8.0, float(agent_index), 0.0], dtype=float)

        self.static_obstacles = copy.deepcopy(self._initial_static_obstacles)
        self.dynamic_obstacles = copy.deepcopy(self._initial_dynamic_obstacles)
        self.starts = self._default_starts.copy()
        self.goals = self._default_goals.copy()
        self.steps = 0
        self.render_mode = render_mode

        self.latest_sensor_packets = [None for _ in range(self.num_agents)]
        self.latest_controller_infos = [{} for _ in range(self.num_agents)]
        self.latest_observation = None

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    @property   # property：类似成员的调用方式，而非类似
    def single_agent_action_dim(self) -> int:   # 解包action_dim 
        return 2 * int(self.dmp_config.dims)    # 当前输出：三维度DMP动作 + 三维度

    @property
    def action_shape(self) -> tuple[int, int]:  # 解包动作形状
        return self.num_agents, self.single_agent_action_dim
    
    @property
    def sensor_observation_dim(self) -> int:    # 解包传感器观测形状
        return int(self.sensors[0].observation_dim)
    
    @property
    def inter_agent_observation_dim(self) -> int:   # 解包机间编队维度
        return (self.num_agents - 1) * self.single_pair_observation_dim

    @property
    def single_agent_observation_dim(self) -> int:  # 解包单智能体观测维度
        return self.sensor_observation_dim + self.extra_observation_dim + self.inter_agent_observation_dim

    @property
    def observation_shape(self) -> tuple[int, int]:
        return self.num_agents, self.single_agent_observation_dim   # 获取所有智能体的观测维度形状

    def _build_single_agent_action_bounds(self) -> tuple[np.ndarray, np.ndarray]:   # 构建动作上下界
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

    def _build_action_space(self) -> spaces.Box:    # 构建动作空间
        single_low, single_high = self._build_single_agent_action_bounds()
        low = np.tile(single_low, (self.num_agents, 1))
        high = np.tile(single_high, (self.num_agents, 1))
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _build_observation_space(self) -> spaces.Box:   # 构建观测空间
        velocity_low = np.broadcast_to(
            np.asarray(self.dynamics[0].velocity_min, dtype=np.float32),
            (self.state_dim,),
        ).astype(np.float32, copy=True)
        velocity_high = np.broadcast_to(
            np.asarray(self.dynamics[0].velocity_max, dtype=np.float32),
            (self.state_dim,),
        ).astype(np.float32, copy=True)

        goal_direction_low = np.full(self.state_dim, -1.0, dtype=np.float32)
        goal_direction_high = np.full(self.state_dim, 1.0, dtype=np.float32)

        goal_distance_low = np.zeros(1, dtype=np.float32)
        goal_distance_high = np.ones(1, dtype=np.float32)

        scan_low = np.zeros(2 * self.sensors[0].n_rays, dtype=np.float32)
        scan_high = np.ones(2 * self.sensors[0].n_rays, dtype=np.float32)

        sensor_low = np.concatenate(
            [velocity_low, goal_direction_low, goal_distance_low, scan_low],
            axis=0,
        )
        sensor_high = np.concatenate(
            [velocity_high, goal_direction_high, goal_distance_high, scan_high],
            axis=0,
        )

        phase_low = np.zeros(1, dtype=np.float32)
        phase_high = np.ones(1, dtype=np.float32)

        k_alpha = self.dmp_config.K_alpha
        k_beta = self.dmp_config.K_beta

        extra_low = np.array([phase_low[0], k_alpha, k_beta], dtype=np.float32)
        extra_high = np.array([phase_high[0], k_alpha, k_beta], dtype=np.float32)

        pair_low = np.concatenate(
        [
            np.full(self.state_dim, -1.0, dtype=np.float32),
            np.full(self.state_dim, -1.0, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
        ],
        axis=0,
        )
        pair_high = np.concatenate(
            [
                np.full(self.state_dim, 1.0, dtype=np.float32),
                np.full(self.state_dim, 1.0, dtype=np.float32),
                np.ones(1, dtype=np.float32),
            ],
            axis=0,
        )

        inter_low = np.tile(pair_low, (self.num_agents - 1))    # np.tile: 按指定次数重复数组
        inter_high = np.tile(pair_high, (self.num_agents - 1))

        single_low = np.concatenate([sensor_low, extra_low, inter_low], axis=0)
        single_high = np.concatenate([sensor_high, extra_high, inter_high], axis=0)

        low = np.tile(single_low, (self.num_agents, 1))
        high = np.tile(single_high, (self.num_agents, 1))
        
        return spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def _validate_agent_points(self, name: str, value) -> np.ndarray:   # 验证智能体点是否为期望的形状
        points = np.asarray(value, dtype=float)
        expected_shape = (self.num_agents, self.state_dim)
        if points.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        return points.copy()

    @property
    def sensor_observation_dim(self) -> int:
        return int(self.sensors[0].observation_dim)

    @property
    def extra_observation_dim(self) -> int:
        return 3

    @property
    def single_pair_observation_dim(self) -> int:
        return 2 * self.state_dim + 1

    def _resolve_starts_goals(self, options: dict) -> tuple[np.ndarray, np.ndarray]:
        has_starts = "starts" in options
        has_goals = "goals" in options

        if self._start_goal_generator is not None and not has_starts and not has_goals: # 不指定就随机生成
            starts, goals = self._start_goal_generator(self.np_random)
        else:
            starts = options.get("starts", self._default_starts)
            goals = options.get("goals", self._default_goals)
        starts = self._validate_agent_points("starts", starts)
        goals = self._validate_agent_points("goals", goals)
        return starts, goals

    def _generate_static_obstacles(self, starts: np.ndarray, goals: np.ndarray) -> list:

        if self._static_obstacle_generator is None: # 不存在生成器
            return copy.deepcopy(self._initial_static_obstacles)    # 采用固定障碍物
        generator_seed = int(self.np_random.integers(0, np.iinfo(np.uint32).max))   # 生成随机种子
        try:    # 多智能体版本的
            return copy.deepcopy(
                self._static_obstacle_generator(
                    starts=starts.copy(),
                    goals=goals.copy(),
                    seed=generator_seed,
                )
            )
        except TypeError:   # 单智能体版本的
            return copy.deepcopy(
                self._static_obstacle_generator(
                    start=starts[0].copy(),
                    goal=goals[0].copy(),
                    seed=generator_seed,
                )
            )

    def _generate_dynamic_obstacles(self, starts: np.ndarray, goals: np.ndarray) -> list:   # 生成动态障碍物，同静态障碍物逻辑
        if self._dynamic_obstacle_generator is None:
            return copy.deepcopy(self._initial_dynamic_obstacles)
        generator_seed = int(self.np_random.integers(0, np.iinfo(np.uint32).max))
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

    def _compose_sensor_observations(self, sensor_packet: dict) -> np.ndarray:
        return sensor_packet.observation.astype(np.float32, copy=True)
    
    def _compose_extra_observations(self, agent_index: int) -> np.ndarray:  # 引入dmp参数作为观测
        dmp = self.dmps[agent_index]
        return np.array(
            [
                float(dmp.phase),
                float(dmp.config.K_alpha),
                float(dmp.config.K_beta),
            ],
            dtype=np.float32,
        )
    
    def _compose_inter_agent_observations(self, agent_index: int) -> np.ndarray:    # 观测其他智能体的动作参数
        '''
        包含其他无人机相对位置、相对速度和友机距离三部分
        '''
        
        position = self.dynamics[agent_index].p
        velocity = self.dynamics[agent_index].v

        influence_distance = max(float(self.env_config.inter_agent_influence_distance), 1e-5)
        velocity_max = np.asarray(self.dynamics[agent_index].velocity_max, dtype=float)
        velocity_scale = max(float(np.max(np.abs(velocity_max))), 1e-5)

        features = []
        for other_index in range(self.num_agents):
            if other_index == agent_index:
                continue

            other_position = self.dynamics[other_index].p
            other_velocity = self.dynamics[other_index].v

            ralative_position = (other_position - position) / influence_distance    # 友机相对位置
            ralative_velocity = (other_velocity - velocity) / velocity_scale    # 友机相对速度
            distance = np.linalg.norm(other_position - position) / influence_distance   # 友机距离

            pair_features = np.concatenate(
                [
                    np.clip(ralative_position, -1.0, 1.0),
                    np.clip(ralative_velocity, -1.0, 1.0),
                    np.array([np.clip(distance, 0.0, 1.0)], dtype=float),
                ],
                axis=0,
            )
            features.append(pair_features)
        
        if not features:   # 没有其他智能体
            return np.zeros((0,), dtype=np.float32)
        
        return np.concatenate(features, axis=0).astype(np.float32)
    
    def get_observation(self) -> np.ndarray:
        if any(packet is None for packet in self.latest_sensor_packets):    # 不存在packet
            raise ValueError("sensor packets not available, cannot construct observation")
        
        observations = []

        for agent_index in range(self.num_agents):
            sensor_observation = self._compose_sensor_observations(
                self.latest_sensor_packets[agent_index]
                )
            extra_observation = self._compose_extra_observations(agent_index)
            inter_agent_observation = self._compose_inter_agent_observations(agent_index)            
            agent_observation = np.concatenate(   # 组合观测
                [
                    sensor_observation,
                    extra_observation,
                    inter_agent_observation,
                ],
                axis = 0,
            ).astype(np.float32)

            observations.append(agent_observation)

        observations = np.stack(observations, axis=0).astype(np.float32)
        self.latest_observation = observations.copy()
        return observations
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        # 生成起点与目标点
        starts, goals = self._resolve_starts_goals(options)
        self.starts = starts.copy()
        self.goals = goals.copy()
        self.steps = 0

        # 索引生成障碍物
        if "static_obstacles" in options:
            self.static_obstacles = copy.deepcopy(options["static_obstacles"])
        else:
            self.static_obstacles = self._generate_static_obstacles(starts, goals)

        if "dynamic_obstacles" in options:
            self.dynamic_obstacles = copy.deepcopy(options["dynamic_obstacles"])
        else:
            self.dynamic_obstacles = self._generate_dynamic_obstacles(starts, goals)

        # 初始化agents
        zero_velocity = np.zeros(self.state_dim, dtype=float)
        for agent_index in range(self.num_agents):
            self.dynamics[agent_index].reset(   # 重置位置，生成0速度
                {
                    "position": starts[agent_index].copy(),
                    "velocity": zero_velocity.copy(),
                }
            )

            # 初始化dmps和传感器
            self.dmps[agent_index].reset(starts[agent_index], goals[agent_index])
            self.sensors[agent_index].reset()

            # 记录传感器观测
            self.latest_sensor_packets[agent_index] = self.sensors[agent_index].sense(
                self.dynamics[agent_index].p,
                self.dynamics[agent_index].v,
                goals[agent_index],
                self.static_obstacles,
                self.dynamic_obstacles,
            )
            # 记录上一时刻的dmps参数
            self.latest_controller_infos[agent_index] = {
                "phase": float(self.dmps[agent_index].phase),
                "tau": float(self.dmps[agent_index].config.tau),
            }

        # 返回observation
        observation = self.get_observation()
 
        info = {
            "starts": starts.copy(),
            "goals": goals.copy(),
            "num_agents": int(self.num_agents),
            "static_obstacle_count": len(self.static_obstacles),
            "dynamic_obstacle_count": len(self.dynamic_obstacles),
        }
        return observation, info


    def step(self, action):
        if any(packet is None for packet in self.latest_sensor_packets):   # 不存在传感器观测，说明环境未重置
            raise RuntimeError("reset must be called before step")

        action = np.asarray(action, dtype=np.float32)   # 转换动作向量为float32
        if action.shape != self.action_shape:   # 动作向量大小错误
            raise ValueError(f"action must have shape {self.action_shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)  # 将动作向量裁剪到动作空间范围内

        previous_distances = np.array(  # 获取当前距离目标点的距离
            [
                np.linalg.norm(self.goals[agent_index] - self.dynamics[agent_index].p)
                for agent_index in range(self.num_agents)
            ],
            dtype=float,
        )

        # 获取命令加速度和实际加速度，并推进动力学
        commanded_accelerations = np.zeros((self.num_agents, self.state_dim), dtype=np.float32)
        applied_accelerations = np.zeros((self.num_agents, self.state_dim), dtype=np.float32)
        next_states = np.zeros((self.num_agents, 2 * self.state_dim), dtype=np.float32)

        for agent_index in range(self.num_agents):  # 迭代所有智能体
            acceleration, controller_info = self.dmps[agent_index].compute_acceleration(
                self.dynamics[agent_index].p,
                self.dynamics[agent_index].v,
                action[agent_index],
                sensor_packet=self.latest_sensor_packets[agent_index],
            )
            applied_acceleration = np.clip(
                acceleration,
                self.dynamics[agent_index].accelerate_min,
                self.dynamics[agent_index].accelerate_max,
            )

            self.latest_controller_infos[agent_index] = controller_info
            commanded_accelerations[agent_index] = np.asarray(acceleration, dtype=np.float32)
            applied_accelerations[agent_index] = np.asarray(applied_acceleration, dtype=np.float32)
            next_states[agent_index] = self.dynamics[agent_index].step(applied_acceleration)

        for obstacle in self.dynamic_obstacles:
            obstacle.step(self.dynamics[0].dt)

        self.steps += 1
        for agent_index in range(self.num_agents):
            self.latest_sensor_packets[agent_index] = self.sensors[agent_index].sense(
                self.dynamics[agent_index].p,
                self.dynamics[agent_index].v,
                self.goals[agent_index],
                self.static_obstacles,
                self.dynamic_obstacles,
            )

        observation = self.get_observation()
        current_distances = np.array(
            [
                np.linalg.norm(self.goals[agent_index] - self.dynamics[agent_index].p)
                for agent_index in range(self.num_agents)
            ],
            dtype=float,
        )
        progress = previous_distances - current_distances
        rewards = (
            float(self.env_config.step_reward_weight) * progress
            - float(self.env_config.step_penalty)
        ).astype(np.float32)

        success_mask = current_distances <= float(self.env_config.goal_tolerance)
        terminated = bool(np.all(success_mask))
        truncated = bool((not terminated) and (self.steps >= self.env_config.max_steps))
        if terminated:
            rewards += float(self.env_config.success_bonus)
        if truncated:
            rewards -= float(self.env_config.timeout_penalty)

        info = {
            "success": bool(terminated),
            "success_mask": success_mask.copy(),
            "truncated": bool(truncated),
            "steps": int(self.steps),
            "distance_to_goals": current_distances.astype(np.float32),
            "progress": progress.astype(np.float32),
            "commanded_accelerations": commanded_accelerations.copy(),
            "applied_accelerations": applied_accelerations.copy(),
            "next_states": next_states.copy(),
            "raw_action": action.copy(),
        }
        return observation, rewards, terminated, truncated, info
