import copy
from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from Controller.dmp_rl import DMPConfig, SecondOrderDMPController
from Entity.KinematicModel import PartialDynamic
from Entity.sensors import LocalObstacleSensor


@dataclass
class Observation:
    """
    环境内部使用的观测容器。

    这里故意把观测拆成两部分保存：
    1. sensor_observation: 由传感器模块直接生成的局部感知观测
    2. extra_observation: 环境额外补充的信息，例如 DMP 相位和超参数

    这样做的好处是：
    - 以后如果你想单独替换传感器观测，不会影响额外观测的组织方式
    - 以后如果你想给 baseline 提供不同版本的观测，也能更方便地拆改
    """

    sensor_observation: np.ndarray
    extra_observation: np.ndarray


@dataclass
class EnvConfig:
    """
    单机 DMP-RL 环境配置。

    这部分参数只负责“环境层面的任务定义”，不负责动力学和控制器本身：
    - max_steps: 单回合最大步数，超出后记为 truncated
    - goal_tolerance: 到达目标的距离阈值
    奖励目前由三部分组成：
    1. obstacle_potential_weight: 障碍物方向的人工势场惩罚权重
    2. step_reward_weight: 步进奖励权重（离目标越近奖励越大）
    3. step_penalty: 单步固定惩罚

    额外保留：
    - collision_penalty: 碰撞终止惩罚
    - success_bonus: 到达目标终止奖励
    - collision_margin: 碰撞判定时的额外安全边界
    """

    max_steps: int = 250
    goal_tolerance: float = 0.35
    obstacle_potential_weight: float = 2.0
    obstacle_influence_distance: float = 1.5
    obstacle_potential_penalty_max: float = 20.0
    step_reward_weight: float = 4.0
    step_penalty: float = 0.01
    collision_penalty: float = 80.0
    timeout_penalty: float = 50.0
    success_bonus: float = 300.0
    collision_margin: float = 0.0
    action_guidance_enabled: bool = False
    action_guidance_radius: float = 1.0
    action_guidance_initial_weight: float = 0.7
    action_guidance_decay_steps: int = 500000
    workspace_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-0.5, -2.5, -1.2),
        (8.5, 2.0, 1.2),
    )
    boundary_influence_distance: float = 0.6
    boundary_potential_weight: float = 0.3
    boundary_distance_epsilon: float = 1e-3


class SingleAgentDMPEnv(gym.Env):
    """
    面向质点模型的单机局部规划 Demo 环境。

    这个环境的职责是把 4 个模块组织起来：
    1. 质点动力学模型
    2. 传感器模型
    3. DMP 控制器
    4. 强化学习任务接口

    强化学习看到的是“环境观测”，输出的是“DMP 调制量”：
    [forcing_x, forcing_y, forcing_z, goal_offset_x, goal_offset_y, goal_offset_z]

    然后环境再把这 6 维动作送给 DMP 控制器，得到实际加速度，
    最终用动力学模型推进系统状态。
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        dynamics_config,
        sensor_config=None,
        dmp_config=None,
        env_config=None,
        start_goal_generator: Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]] | None = None,
        static_obstacles=None,
        static_obstacle_generator=None,
        dynamic_obstacles=None,
        dynamic_obstacle_generator=None,
        render_mode=None,
    ):
        # 当前这个 Demo 只支持最小的 human 渲染占位接口。
        if render_mode not in {None, "human"}:
            raise ValueError("render_mode must be None or 'human'")

        # 1. 构建底层模块
        # dynamics: 负责位置速度更新
        # sensor:   负责局部感知
        # dmp:      负责把 RL 动作变成实际控制量
        self.dynamics = PartialDynamic(dynamics_config)
        self.sensor = LocalObstacleSensor(**(sensor_config or {}))
        self.dmp = SecondOrderDMPController(dmp_config or DMPConfig(dt=self.dynamics.dt))
        self.env_config = env_config or EnvConfig()
        self.state_dim = int(self.dynamics.p.shape[0])
        if self.dmp.config.dims != self.state_dim:
            raise ValueError(
                f"dmp dims ({self.dmp.config.dims}) must match dynamics dimension ({self.state_dim})"
            )

        # deepcopy 会递归复制对象，保证每次 reset 都拿到独立场景。
        self._initial_static_obstacles = copy.deepcopy(static_obstacles or [])
        self._static_obstacle_generator = static_obstacle_generator
        self._initial_dynamic_obstacles = copy.deepcopy(dynamic_obstacles or [])
        self._dynamic_obstacle_generator = dynamic_obstacle_generator
        self._start_goal_generator = start_goal_generator
        self._default_start = np.zeros(self.state_dim, dtype=float)
        self._default_goal = np.zeros(self.state_dim, dtype=float)
        self._default_goal[0] = 8.0

        # 2. 运行时状态
        self.static_obstacles = copy.deepcopy(self._initial_static_obstacles)
        self.dynamic_obstacles = copy.deepcopy(self._initial_dynamic_obstacles)
        self.goal = self._default_goal.copy()
        self.steps = 0
        self.action_guidance_step = 0
        self.latest_sensor_packet = None
        self.latest_observation = None
        self.latest_controller_info = {}
        self.render_mode = render_mode

        # 3. 对外声明 Gym 风格接口需要的空间定义
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    @property
    def sensor_observation_dim(self):
        # 传感器模块直接输出的观测维度
        return self.sensor.observation_dim

    @property
    def extra_observation_dim(self):
        # 环境额外附加 3 维：
        # [phase, K_alpha, K_beta]
        return 3

    @property
    def observation_dim(self):
        # 最终给策略网络的总观测维度 = 传感器观测 + 额外观测
        return self.sensor_observation_dim + self.extra_observation_dim

    @property
    def action_dim(self):
        # 每个维度一个 forcing_term + 每个维度一个 goal_offset
        return 2 * self.dmp.config.dims

    def _build_observation_space(self):
        """
        构造完整观测空间。

        这里定义的是“最终交给策略网络的完整观测范围”，
        不是单独 sensor 模块内部的观测范围。

        当前完整观测顺序为：
        1. 速度 3 维
        2. 目标方向 3 维
        3. 目标距离归一化 1 维
        4. 当前帧雷达扫描 216 维
        5. 上一帧雷达扫描 216 维
        6. phase 1 维
        7. K_alpha 1 维
        8. K_beta 1 维
        """
        # 当前速度范围直接使用动力学模型中定义的上下限
        velocity_low = np.full(self.state_dim, self.dynamics.velocity_min, dtype=np.float32)
        velocity_high = np.full(self.state_dim, self.dynamics.velocity_max, dtype=np.float32)

        # 目标方向是单位向量，每一维都不可能超过 [-1, 1]
        goal_direction_low = np.full(self.state_dim, -1.0, dtype=np.float32)
        goal_direction_high = np.full(self.state_dim, 1.0, dtype=np.float32)

        # 目标距离已经在 sensor 中做了归一化和裁剪，所以范围固定在 [0, 1]
        goal_distance_low = np.zeros(1, dtype=np.float32)
        goal_distance_high = np.ones(1, dtype=np.float32)

        # 两帧雷达扫描都已经归一化，所以也在 [0, 1]
        scan_low = np.zeros(2 * self.sensor.n_rays, dtype=np.float32)
        scan_high = np.ones(2 * self.sensor.n_rays, dtype=np.float32)

        # phase 采用衰减相位变量，当前实现始终在 [0, 1]
        phase_low = np.zeros(1, dtype=np.float32)
        phase_high = np.ones(1, dtype=np.float32)

        # K_alpha / K_beta 在当前实验中是固定常数。
        # 这里仍把它们放入观测，是为了让完整观测的语义更显式。
        K_alpha = self.dmp.config.K_alpha
        K_beta = self.dmp.config.K_beta
        k_alpha_low = np.array([K_alpha], dtype=np.float32)
        k_alpha_high = np.array([K_alpha], dtype=np.float32)
        k_beta_low = np.array([K_beta], dtype=np.float32)
        k_beta_high = np.array([K_beta], dtype=np.float32)

        # 按最终观测顺序拼接 low / high，供 Gym 和 RL 算法检查接口合法性
        low = np.concatenate(
            [velocity_low, goal_direction_low, goal_distance_low, scan_low, phase_low, k_alpha_low, k_beta_low],
            axis=0,
        )
        high = np.concatenate(
            [velocity_high, goal_direction_high, goal_distance_high, scan_high, phase_high, k_alpha_high, k_beta_high], 
            axis=0,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _build_action_space(self):
        """
        构造动作空间。

        动作由两部分组成：
        1. forcing_term: 每个维度一个
        2. goal_offset:  每个维度一个
        """
        dims = self.dmp.config.dims

        # forcing_term 的范围来自 DMP 控制器配置
        # goal_offset 的范围也来自 DMP 控制器配置
        low = np.concatenate(
            [
                np.full(dims, self.dmp.config.forcing_term_min, dtype=np.float32),
                np.full(dims, -self.dmp.config.goal_offset_max, dtype=np.float32),
            ],
            axis=0,
        )
        high = np.concatenate(
            [
                np.full(dims, self.dmp.config.forcing_term_max, dtype=np.float32),
                np.full(dims, self.dmp.config.goal_offset_max, dtype=np.float32),
            ],
            axis=0,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32) 

    def get_sensor_observation(self):
        """
        返回传感器直接产生的观测。

        这里返回的是“局部感知结果”，尚未拼接环境额外信息。
        """
        if self.latest_sensor_packet is None:
            raise RuntimeError("reset must be called before reading sensor observation")
        return self._compose_sensor_observation(self.latest_sensor_packet)

    def get_extra_observation(self):
        """
        返回环境额外补充的观测。

        当前版本只补了 DMP 内部状态和超参数；
        后续如果要继续加历史动作、任务阶段等信息，可以优先改这里。
        """
        if self.latest_sensor_packet is None:
            raise RuntimeError("reset must be called before reading extra observation")
        return self._compose_extra_observation()

    def get_observation(self):
        """
        返回最终完整观测。

        这一步会先拿到 sensor 观测和 extra 观测，
        再统一拼成给策略网络使用的一维向量。
        """
        if self.latest_sensor_packet is None:
            raise RuntimeError("reset must be called before reading observation")

        self.latest_observation = Observation(
            sensor_observation=self.get_sensor_observation(),
            extra_observation=self.get_extra_observation(),
        )
        return self._compose_observation(self.latest_observation)

    def reset(self, *, seed=None, options=None):
        """
        按 Gymnasium 规范重置环境。

        options 可覆盖本回合的场景设置，包括：
        - start
        - goal
        - static_obstacles
        - dynamic_obstacles
        """
        # 先让父类处理随机种子，保证接口合法
        super().reset(seed=seed)
        options = options or {}

        # 读取本回合场景配置；训练模式下未显式指定起终点时按配置随机采样。
        has_start = "start" in options
        has_goal = "goal" in options
        if self._start_goal_generator is not None and not has_start and not has_goal:
            start, goal = self._start_goal_generator(self.np_random)
            start = np.asarray(start, dtype=float)
            goal = np.asarray(goal, dtype=float)
        else:
            start = np.asarray(options.get("start", self._default_start), dtype=float)
            goal = np.asarray(options.get("goal", self._default_goal), dtype=float)
        if start.shape != (self.state_dim,) or goal.shape != (self.state_dim,):
            raise ValueError(f"start and goal must have shape ({self.state_dim},)")

        # 1. 回合级状态重置
        self.goal = goal.copy()
        self.steps = 0
        if "static_obstacles" in options:
            self.static_obstacles = copy.deepcopy(options["static_obstacles"])
        elif self._static_obstacle_generator is not None:
            # 训练时可在每个回合重置时重新生成静态障碍物，让策略看到更多场景。
            generator_seed = int(self.np_random.integers(0, np.iinfo(np.uint32).max))
            self.static_obstacles = copy.deepcopy(
                self._static_obstacle_generator(
                    start=start.copy(),
                    goal=goal.copy(),
                    seed=generator_seed,
                )
            )
        else:
            self.static_obstacles = copy.deepcopy(self._initial_static_obstacles)
        if "dynamic_obstacles" in options:
            self.dynamic_obstacles = copy.deepcopy(options["dynamic_obstacles"])
        elif self._dynamic_obstacle_generator is not None:
            generator_seed = int(self.np_random.integers(0, np.iinfo(np.uint32).max))
            self.dynamic_obstacles = copy.deepcopy(
                self._dynamic_obstacle_generator(
                    start=start.copy(),
                    goal=goal.copy(),
                    seed=generator_seed,
                    static_obstacles=copy.deepcopy(self.static_obstacles),
                )
            )
        else:
            self.dynamic_obstacles = copy.deepcopy(self._initial_dynamic_obstacles)

        # 底层模块重置
        self.dynamics.reset({"position": start, "velocity": np.zeros(self.state_dim, dtype=float)})
        self.dmp.reset(start, goal)
        self.sensor.reset()

        # 用重置后的状态生成第一帧传感器读数
        self.latest_sensor_packet = self.sensor.sense(
            self.dynamics.p,
            self.dynamics.v,
            self.goal,
            self.static_obstacles,
            self.dynamic_obstacles,
        )

        # 保存控制器当前内部状态，便于后续 info 直接读取
        self.latest_controller_info = {"phase": float(self.dmp.phase), "tau": float(self.dmp.config.tau)}
        self.latest_observation = None

        # 生成 reset 后的完整观测和信息字典
        observation = self.get_observation()
        info = self._build_info(
            success=False,
            collision=self._check_collision(),
            truncated=False,
            commanded_acceleration=np.zeros(self.state_dim, dtype=np.float32),
            applied_acceleration=np.zeros(self.state_dim, dtype=np.float32),
            next_state=self.dynamics.state.copy(),
            step_reward=0.0,
            obstacle_potential_penalty=0.0,
            boundary_potential_penalty=0.0,
            step_penalty=float(self.env_config.step_penalty),
            timeout_penalty=0.0,
            raw_action=np.zeros(self.action_dim, dtype=np.float32),
            guided_action=np.zeros(self.action_dim, dtype=np.float32),
            action_guidance_weight=0.0,
        )
        return observation, info

    def step(self, action):
        """
        执行一步环境推进。

        这一函数的主流程是：
        1. 检查动作格式
        2. 用 DMP 把动作变成加速度
        3. 用动力学推进无人机状态
        4. 推进动态障碍物
        5. 刷新传感器观测
        6. 计算奖励、终止条件和 info
        """
        if self.latest_sensor_packet is None:
            raise RuntimeError("reset must be called before step")

        # 检查动作维度，并裁剪到动作空间范围内
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},)")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 记录执行前到目标的距离，用于后面计算推进奖励
        previous_distance = float(np.linalg.norm(self.goal - self.dynamics.p))
        raw_action = action.copy()
        action, action_guidance_weight = self._apply_action_guidance(action, previous_distance)

        # 让 DMP 控制器根据当前状态和 RL 动作计算期望加速度
        acceleration, controller_info = self.dmp.compute_acceleration(
            self.dynamics.p,
            self.dynamics.v,
            action,
            sensor_packet=self.latest_sensor_packet,
        )

        # 再用动力学模型允许的加速度范围做一次裁剪
        applied_acceleration = np.clip(
            acceleration,
            self.dynamics.accelerate_min,
            self.dynamics.accelerate_max,
        )

        # 推进无人机动力学
        self.latest_controller_info = controller_info
        next_state = self.dynamics.step(applied_acceleration)

        # 推进所有动态障碍物
        for obstacle in self.dynamic_obstacles:
            obstacle.step(self.dynamics.dt)

        # 更新步数和最新传感器观测
        self.steps += 1
        self.action_guidance_step += 1
        self.latest_sensor_packet = self.sensor.sense(
            self.dynamics.p,
            self.dynamics.v,
            self.goal,
            self.static_obstacles,
            self.dynamic_obstacles,
        )

        # 刷新完整观测
        observation = self.get_observation()
        current_distance = float(np.linalg.norm(self.goal - self.dynamics.p))
        progress = previous_distance - current_distance

        # 奖励重构为三部分：
        # 1) obstacle_potential_penalty：障碍物方向人工势场惩罚
        # 2) step_reward：步进奖励（朝目标前进）
        # 3) step_penalty：固定单步惩罚

        # 两部分势场惩罚
        obstacle_potential_penalty = self._compute_obstacle_potential_penalty(raw_action)
        boundary_potential_penalty = self._compute_boundary_potential_penalty()

        step_reward = self.env_config.step_reward_weight * progress
        step_penalty = self.env_config.step_penalty

        reward = step_reward - obstacle_potential_penalty - boundary_potential_penalty - step_penalty

        # 判断本步结束类型
        # success / collision 算 terminated
        # 超步数算 truncated
        success = current_distance <= self.env_config.goal_tolerance
        collision = self._check_collision()
        terminated = bool(success or collision)
        truncated = bool((not terminated) and (self.steps >= self.env_config.max_steps))

        # 成功和碰撞分别叠加终止奖励/惩罚
        if collision:
            reward -= self.env_config.collision_penalty
        elif success:
            reward += self.env_config.success_bonus

        timeout_penalty = float(self.env_config.timeout_penalty) if truncated else 0.0
        reward -= timeout_penalty

        # 组装当前步的附加信息
        info = self._build_info(
            success=success,
            collision=collision,
            truncated=truncated,
            commanded_acceleration=acceleration,
            applied_acceleration=applied_acceleration,
            next_state=next_state,
            step_reward=step_reward,
            obstacle_potential_penalty=obstacle_potential_penalty,
            boundary_potential_penalty=boundary_potential_penalty,
            step_penalty=step_penalty,
            timeout_penalty=timeout_penalty,
            raw_action=raw_action,
            guided_action=action,
            action_guidance_weight=action_guidance_weight,
        )
        return observation, float(reward), terminated, truncated, info

    def _apply_action_guidance(self, action: np.ndarray, distance_to_goal: float) -> tuple[np.ndarray, float]:
        """
        Mix the policy action toward the zero DMP action near the goal during early training.
        """
        if not self.env_config.action_guidance_enabled:
            return action, 0.0

        radius = float(self.env_config.action_guidance_radius)
        if radius <= 0.0 or distance_to_goal >= radius:
            return action, 0.0

        decay_steps = max(1, int(self.env_config.action_guidance_decay_steps))
        training_gate = max(0.0, 1.0 - float(self.action_guidance_step) / float(decay_steps))
        if training_gate <= 0.0:
            return action, 0.0

        distance_gate = 1.0 - max(0.0, distance_to_goal) / radius
        initial_weight = float(np.clip(self.env_config.action_guidance_initial_weight, 0.0, 1.0))
        guidance_weight = float(np.clip(initial_weight * distance_gate * training_gate, 0.0, 1.0))
        if guidance_weight <= 0.0:
            return action, 0.0

        guided_action = (1.0 - guidance_weight) * action
        guided_action = np.clip(guided_action, self.action_space.low, self.action_space.high)
        return guided_action.astype(np.float32, copy=False), guidance_weight

    def _compute_obstacle_potential_penalty(self, policy_action: np.ndarray) -> float:
        """
        计算“障碍物方向的人工势场惩罚”。

        设计要点：
        1. 只在“策略动作方向朝向障碍物”时产生惩罚（方向门控）
        2. 距障碍物越近，惩罚越大（势场强度随距离增大而衰减）
        3. 多障碍物惩罚累加
        """
        # 使用策略输出诱导的控制方向进行门控，避免 DMP 基础吸引项替策略承担避障惩罚。
        forcing_component = np.asarray(policy_action[: self.state_dim], dtype=float)
        goal_offset = np.asarray(policy_action[self.state_dim: 2 * self.state_dim], dtype=float)
        goal_eff = self.goal + goal_offset
        forcing_gate = np.tanh(np.abs(goal_eff - self.dynamics.p))
        action_component = (
            self.dmp.config.K_alpha * self.dmp.config.K_beta * goal_offset
            + forcing_component * forcing_gate
        )
        action_norm = float(np.linalg.norm(action_component))
        if action_norm < 1e-8:
            return 0.0

        action_dir = action_component / action_norm
        velocity_norm = float(np.linalg.norm(self.dynamics.v))
        velocity_dir = self.dynamics.v / velocity_norm if velocity_norm >= 1e-8 else None
        position = self.dynamics.p
        influence_distance = float(self.env_config.obstacle_influence_distance)
        if influence_distance <= 0.0:
            return 0.0

        penalty_sum = 0.0
        for obstacle in self.static_obstacles + self.dynamic_obstacles:
            # 用障碍物表面最近点构造“指向障碍物”的方向向量。
            closest_point = obstacle.closest_point(position)
            to_obstacle = closest_point - position
            distance_to_surface = float(np.linalg.norm(to_obstacle))
            if distance_to_surface < 1e-8:
                continue

            # 超出势场影响半径则不惩罚。
            if distance_to_surface >= influence_distance:
                continue

            # 在障碍物内部或极近处时给一个稳定下界，避免除零。
            d = max(distance_to_surface, 1e-3)

            # 势场基础强度：常见人工势场形式 (1/d - 1/d0)^2
            # d0 使用独立势场半径，使惩罚范围与传感器感知范围解耦。
            base_field = (1.0 / d - 1.0 / influence_distance) ** 2

            # 策略动作必须指向障碍物才惩罚；速度也指向障碍物时，说明风险正在累积，惩罚增强。
            obstacle_dir = to_obstacle / distance_to_surface
            action_gate = max(0.0, float(np.dot(action_dir, obstacle_dir)))
            if action_gate <= 0.0:
                continue
            velocity_gate = 0.0 if velocity_dir is None else max(0.0, float(np.dot(velocity_dir, obstacle_dir)))
            directional_gate = action_gate * (0.5 + 0.5 * velocity_gate)

            penalty_sum += base_field * directional_gate

        penalty = float(self.env_config.obstacle_potential_weight * penalty_sum)
        return float(min(penalty, self.env_config.obstacle_potential_penalty_max))

    def _compute_min_boundary_distance(self) -> float:
        bounds = np.asarray(self.env_config.workspace_bounds, dtype=float)
        if bounds.shape != (2, self.state_dim):
            raise ValueError(f"workspace_bounds must have shape (2, {self.state_dim})")

        lower, upper = bounds
        distances = np.concatenate([self.dynamics.p - lower, upper - self.dynamics.p])
        return float(np.min(distances))

    def _compute_boundary_potential_penalty(self) -> float:
        bounds = np.asarray(self.env_config.workspace_bounds, dtype=float)
        if bounds.shape != (2, self.state_dim):
            raise ValueError(f"workspace_bounds must have shape (2, {self.state_dim})")

        influence_distance = float(self.env_config.boundary_influence_distance)
        if influence_distance <= 0.0:
            return 0.0

        lower, upper = bounds
        signed_distances = np.concatenate([self.dynamics.p - lower, upper - self.dynamics.p])
        epsilon = max(float(self.env_config.boundary_distance_epsilon), 1e-8)

        penalty_sum = 0.0
        for signed_distance in signed_distances:
            if signed_distance >= influence_distance:
                continue
            d = max(float(signed_distance), epsilon)
            penalty_sum += (1.0 / d - 1.0 / influence_distance) ** 2

        return float(self.env_config.boundary_potential_weight * penalty_sum)

    def _build_info(
        self,
        success,
        collision,
        truncated,
        commanded_acceleration,
        applied_acceleration,
        next_state,
        step_reward,
        obstacle_potential_penalty,
        boundary_potential_penalty,
        step_penalty,
        timeout_penalty,
        raw_action,
        guided_action,
        action_guidance_weight,
    ):
        """
        组装 info 字典。

        info 主要服务三类用途：
        1. 调试训练过程
        2. 离线分析每一步发生了什么
        3. 后面画图或做实验统计
        """
        return {
            "success": bool(success),
            "collision": bool(collision),
            "truncated": bool(truncated),
            "distance_to_goal": float(np.linalg.norm(self.goal - self.dynamics.p)),
            "min_boundary_distance": self._compute_min_boundary_distance(),
            "min_clearance": float(self.latest_sensor_packet.min_clearance),
            "phase": float(self.latest_controller_info.get("phase", self.dmp.phase)),
            "tau": float(self.latest_controller_info.get("tau", self.dmp.config.tau)),
            "sensor_observation": self.get_sensor_observation().copy(),
            "commanded_acceleration": np.asarray(commanded_acceleration, dtype=np.float32).copy(),
            "applied_acceleration": np.asarray(applied_acceleration, dtype=np.float32).copy(),
            "next_state": np.asarray(next_state, dtype=np.float32).copy(),
            "raw_action": np.asarray(raw_action, dtype=np.float32).copy(),
            "guided_action": np.asarray(guided_action, dtype=np.float32).copy(),
            "action_guidance_weight": float(action_guidance_weight),
            "reward_step_reward": float(step_reward),
            "reward_obstacle_potential_penalty": float(obstacle_potential_penalty),
            "reward_boundary_potential_penalty": float(boundary_potential_penalty),
            "reward_step_penalty": float(step_penalty),
            "reward_timeout_penalty": float(timeout_penalty),
        }

    def _check_collision(self):
        """
        用障碍物几何模型判断当前位置是否碰撞。

        注意这里用的是几何判定，不依赖传感器读数。
        也就是说，就算传感器设计以后改了，碰撞判定仍然独立成立。
        """
        point = self.dynamics.p
        for obstacle in self.static_obstacles + self.dynamic_obstacles:
            if obstacle.contains(point, margin=self.env_config.collision_margin):
                return True
        return False

    def _compose_sensor_observation(self, observation):
        """
        从不同类型的输入里统一提取“sensor 那部分观测”。

        两种输入都支持：
        - Observation: 环境内部封装后的完整观测容器
        - SensorPacket: 传感器直接输出
        """
        if isinstance(observation, Observation):
            return observation.sensor_observation.astype(np.float32, copy=True)
        return observation.observation.astype(np.float32, copy=True)

    def _compose_extra_observation(self):
        """
        构造环境额外观测。

        当前额外观测包含：
        - phase: DMP 当前全局相位
        - K_alpha: DMP 弹簧项系数
        - K_beta: DMP 阻尼项系数
        """
        return np.array(
            [self.dmp.phase, self.dmp.config.K_alpha, self.dmp.config.K_beta],
            dtype=np.float32,
        )

    def _compose_observation(self, observation):
        """
        把 sensor 观测和 extra 观测拼成最终一维向量。

        这一步的输出就是策略网络真正看到的输入。
        """
        sensor_observation = self._compose_sensor_observation(observation)
        if isinstance(observation, Observation):
            extra_observation = observation.extra_observation.astype(np.float32, copy=True)
        else:
            extra_observation = self._compose_extra_observation()
        return np.concatenate([sensor_observation, extra_observation], axis=0).astype(np.float32)

    def render(self):
        """
        最小渲染接口。

        当前不做图形显示，只在 human 模式下返回一份可读状态。
        """
        if self.render_mode != "human":
            return None
        return {
            "position": self.dynamics.p.copy(),
            "goal": self.goal.copy(),
            "steps": self.steps,
        }

    def close(self):
        """
        关闭环境。

        当前环境没有外部图形或文件句柄，因此这里是空实现。
        """
        return None
