import copy
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
    - collision_penalty: 碰撞惩罚
    - success_bonus: 到达目标奖励
    - progress_weight: 向目标推进的奖励系数
    - clearance_weight: 贴近障碍物时的惩罚系数
    - action_penalty: 控制过猛时的惩罚系数
    - living_penalty: 每一步的生存惩罚，防止策略原地拖时间
    - collision_margin: 碰撞判定时的额外安全边界
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
        static_obstacles=None,
        dynamic_obstacles=None,
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
        self._initial_dynamic_obstacles = copy.deepcopy(dynamic_obstacles or [])
        self._default_start = np.zeros(self.state_dim, dtype=float)
        self._default_goal = np.zeros(self.state_dim, dtype=float)
        self._default_goal[0] = 8.0

        # 2. 运行时状态
        self.static_obstacles = copy.deepcopy(self._initial_static_obstacles)
        self.dynamic_obstacles = copy.deepcopy(self._initial_dynamic_obstacles)
        self.goal = self._default_goal.copy()
        self.steps = 0
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

        # 读取本回合场景配置；若不传则使用默认值
        start = np.asarray(options.get("start", self._default_start), dtype=float)
        goal = np.asarray(options.get("goal", self._default_goal), dtype=float)
        if start.shape != (self.state_dim,) or goal.shape != (self.state_dim,):
            raise ValueError(f"start and goal must have shape ({self.state_dim},)")

        # 1. 回合级状态重置
        self.goal = goal.copy()
        self.steps = 0
        self.static_obstacles = copy.deepcopy(options.get("static_obstacles", self._initial_static_obstacles))
        self.dynamic_obstacles = copy.deepcopy(options.get("dynamic_obstacles", self._initial_dynamic_obstacles))

        # 2. 底层模块重置
        self.dynamics.reset({"position": start, "velocity": np.zeros(self.state_dim, dtype=float)})
        self.dmp.reset(start, goal)
        self.sensor.reset()

        # 3. 用重置后的状态生成第一帧传感器读数
        self.latest_sensor_packet = self.sensor.sense(
            self.dynamics.p,
            self.dynamics.v,
            self.goal,
            self.static_obstacles,
            self.dynamic_obstacles,
        )

        # 4. 保存控制器当前内部状态，便于后续 info 直接读取
        self.latest_controller_info = {"phase": float(self.dmp.phase), "tau": float(self.dmp.config.tau)}
        self.latest_observation = None

        # 5. 生成 reset 后的完整观测和信息字典
        observation = self.get_observation()
        info = self._build_info(
            success=False,
            collision=self._check_collision(),
            truncated=False,
            commanded_acceleration=np.zeros(self.state_dim, dtype=np.float32),
            applied_acceleration=np.zeros(self.state_dim, dtype=np.float32),
            next_state=self.dynamics.state.copy(),
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

        # 1. 检查动作维度，并裁剪到动作空间范围内
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},)")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 2. 记录执行前到目标的距离，用于后面计算推进奖励
        previous_distance = float(np.linalg.norm(self.goal - self.dynamics.p))

        # 3. 让 DMP 控制器根据当前状态和 RL 动作计算期望加速度
        acceleration, controller_info = self.dmp.compute_acceleration(
            self.dynamics.p,
            self.dynamics.v,
            action,
            sensor_packet=self.latest_sensor_packet,
        )

        # 4. 再用动力学模型允许的加速度范围做一次裁剪
        applied_acceleration = np.clip(
            acceleration,
            self.dynamics.accelerate_min,
            self.dynamics.accelerate_max,
        )

        # 5. 推进无人机动力学
        self.latest_controller_info = controller_info
        next_state = self.dynamics.step(applied_acceleration)

        # 6. 推进所有动态障碍物
        for obstacle in self.dynamic_obstacles:
            obstacle.step(self.dynamics.dt)

        # 7. 更新步数和最新传感器观测
        self.steps += 1
        self.latest_sensor_packet = self.sensor.sense(
            self.dynamics.p,
            self.dynamics.v,
            self.goal,
            self.static_obstacles,
            self.dynamic_obstacles,
        )

        # 8. 刷新完整观测
        observation = self.get_observation()
        current_distance = float(np.linalg.norm(self.goal - self.dynamics.p))
        progress = previous_distance - current_distance

        # 9. 奖励由四部分组成：
        #    - 向目标推进的正奖励
        #    - 控制过猛惩罚
        #    - 每步生存惩罚
        #    - 贴近障碍物惩罚
        reward = self.env_config.progress_weight * progress
        reward -= self.env_config.action_penalty * float(np.linalg.norm(applied_acceleration))
        reward -= self.env_config.living_penalty
        if np.isfinite(self.latest_sensor_packet.min_clearance):
            reward -= self.env_config.clearance_weight * np.exp(-max(self.latest_sensor_packet.min_clearance, 0.0))

        # 10. 判断本步结束类型
        # success / collision 算 terminated
        # 超步数算 truncated
        success = current_distance <= self.env_config.goal_tolerance
        collision = self._check_collision()
        terminated = bool(success or collision)
        truncated = bool((not terminated) and (self.steps >= self.env_config.max_steps))

        # 11. 成功和碰撞分别叠加终止奖励/惩罚
        if collision:
            reward -= self.env_config.collision_penalty
        elif success:
            reward += self.env_config.success_bonus

        # 12. 组装当前步的附加信息
        info = self._build_info(
            success=success,
            collision=collision,
            truncated=truncated,
            commanded_acceleration=acceleration,
            applied_acceleration=applied_acceleration,
            next_state=next_state,
        )
        return observation, float(reward), terminated, truncated, info

    def _build_info(self, success, collision, truncated, commanded_acceleration, applied_acceleration, next_state):
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
            "min_clearance": float(self.latest_sensor_packet.min_clearance),
            "phase": float(self.latest_controller_info.get("phase", self.dmp.phase)),
            "tau": float(self.latest_controller_info.get("tau", self.dmp.config.tau)),
            "sensor_observation": self.get_sensor_observation().copy(),
            "commanded_acceleration": np.asarray(commanded_acceleration, dtype=np.float32).copy(),
            "applied_acceleration": np.asarray(applied_acceleration, dtype=np.float32).copy(),
            "next_state": np.asarray(next_state, dtype=np.float32).copy(),
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
