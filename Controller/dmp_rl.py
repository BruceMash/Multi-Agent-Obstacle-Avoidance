from dataclasses import dataclass, field

import numpy as np

import torch.nn as nn


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
    obstacle_gain_max: float = 8.0
    goal_offset_max: float = 1.5

    def __post_init__(self):
        self.dt = float(self.dt)
        self.dims = int(self.dims)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.dims <= 0:
            raise ValueError("dims must be positive")

class SecondOrderDMPController:
    """
    面向质点模型的二阶 DMP 控制器。

    RL 不直接输出加速度，而是调制：
    1、forcing term
    2、局部目标残差
    """

    def __init__(self, config: DMPConfig):
        '''
        config: dim
                start:[x, y, z]
                goal:[x, y, z]
                K_alpha --> Hyperparameter
                K_beta --> Hyperparameter
                tau --> Hyperparameter
                
        '''
        #DMP参数
        self.config = config

        self.start = np.zeros(self.config.dims, dtype=float)
        self.goal = np.zeros(self.config.dims, dtype=float)
        self.phase = 1.0    #相位

        # 问题标注：
        # 这里只是为了做 tanh 门控，但使用了 torch.nn.Tanh()。
        # 当前项目其余管线是 numpy 风格，这会额外引入 torch 依赖；
        # 如果本地没有安装 torch，模块导入就会失败。
        # 若无特殊需要，这里更适合统一改成 np.tanh。
        self.tanh = nn.Tanh()

    def reset(self, start, goal):
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        if self.start.shape != (self.config.dims,) or self.goal.shape != (self.config.dims,):
            raise ValueError("start and goal must match DMP dims")
        self.phase = 1.0

    def compute_acceleration(self, position, velocity, rl_action, sensor_packet):
        # 将输入参数转换为numpy数组，确保数据类型为浮点数
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        rl_action = np.asarray(rl_action, dtype=float)

        # 验证位置和速度的维度是否与DMP（动态运动原语）配置的维度匹配
        if position.shape != (self.config.dims,) or velocity.shape != (self.config.dims,):
            raise ValueError("position and velocity must match DMP dims")
        # 验证强化学习动作的维度是否正确（应为2加上DMP的维度）
        # *联动修改：
        # 这里假定 action 维度是 1 + dims，即：
        # [forcing_term, goal_offset_x, goal_offset_y, goal_offset_z]
        # Environment/single_agent_dmp_env.py、demo_single_agent_dmp_rl.py、
        # test_dmp_rl_pipeline.py 以及本文件中的 HeuristicDMPPolicy
        # 都需要同步到这一套动作定义。
        if rl_action.shape != (1 + self.config.dims,):
            raise ValueError("rl_action must have shape (1 + dims,)")

        # 解析强化学习动作，提取其中的参数
        parsed = self._parse_action(rl_action)

        # 计算有效目标位置（原始目标位置加上动作中的目标偏移）
        goal_eff = self.goal + parsed["goal_offset"]
        # 计算强迫项（forcing term），用于生成期望的运动轨迹
        forcing = parsed["forcing_term"]
        # 问题标注：
        # sensor_packet 现在完全没有参与计算。
        # 如果你的设计就是“RL 直接学习 forcing 完成避障”，这是可以的；
        # 但需要明确：此时控制器层不再显式使用障碍物几何信息，
        # 避障能力完全依赖策略网络从 observation 中学出来。
        # 计算系统加速度，考虑位置误差、速度、强迫项和障碍物耦合项
        acceleration = (
            self.config.K_alpha * (self.config.K_beta* (goal_eff - position) - self.config.tau * velocity)
            + np.clip(forcing, self.config.forcing_term_min, self.config.forcing_term_max) * self.tanh(self.calc_distance(goal_eff, position))
        ) / (self.config.tau ** 2)

        # 问题标注：
        # 分母仍然使用 self.tau，但当前类里并没有定义 self.tau，
        # 应与上面的 self.config.tau 保持一致，否则运行时会报 AttributeError。

        # 问题标注：
        # 当前 forcing_term 被解析为单个标量。
        # 这意味着 RL 只能整体调节 forcing 强度，不能分别控制 x/y/z 三个方向。
        # 如果后面发现轨迹表达能力不足，可以把 forcing 扩成 dims 维向量。

        # 返回计算得到的加速度和相关信息字典
        return acceleration, {
            "phase": self.phase,           # 当前系统相位
            "goal_eff": goal_eff,          # 有效目标位置
            "goal_offset": parsed["goal_offset"].copy(),  # 目标位置偏移
            # 问题标注：
            # 这里返回的 forcing 是一个标量/0维量，而不是向量 forcing。
            # 论文和代码里都需要统一说明 forcing 的形状语义。
            "forcing": parsed['forcing_term'].copy()      # 强迫项
        }

    def _parse_action(self, rl_action): # 解析强化学习动作，forcing term和局部目标残差
        # 问题标注：
        # 当前 action 只拆成“1个 forcing 标量 + dims 维 goal_offset”。
        # 这版设计足够做最小 Demo，但表达能力有限；
        # 如果后面要做更复杂避障，可能需要扩成向量 forcing 或增加额外调制项。
        forcing_term = rl_action[0]
        goal_offset = np.clip(rl_action[1:], -1.0, 1.0) * self.config.goal_offset_max
        return {"forcing_term": forcing_term, "goal_offset": goal_offset}
    
    @staticmethod
    def calc_distance(goal, current):  
        return np.linalg.norm(goal - current)



class HeuristicDMPPolicy:
    """
    用于 Demo 的启发式策略。

    该策略只负责生成 RL 风格调制量，后续可以直接替换成神经网络策略。
    """

    def __init__(self, goal_offset_max=1.0):    # 初始化
        self.goal_offset_max = float(goal_offset_max)

    def act(self, sensor_packet):
        obstacle_features = sensor_packet.obstacle_features
        valid_mask = obstacle_features[:, -1] > 0.0

        if not np.any(valid_mask):
            # *联动修改：
            # 当前启发式策略仍返回旧的 5 维动作：
            # [obstacle_gain, tau_scale, goal_offset_x, goal_offset_y, goal_offset_z]
            # 若控制器最终固定为 forcing_term 方案，这里需要一起改成 4 维。
            return np.array([-0.8, -0.6, 0.0, 0.0, 0.0], dtype=float)

        close_features = obstacle_features[valid_mask]
        avoid_direction = np.zeros(3, dtype=float)
        proximity_score = 0.0

        for feature in close_features:
            relative_position = feature[0:3]
            clearance = max(feature[6], 1e-3)
            distance = np.linalg.norm(relative_position)
            if distance < 1e-8:
                continue
            proximity_score += np.exp(-clearance)
            avoid_direction -= relative_position / distance * np.exp(-clearance)

        norm = np.linalg.norm(avoid_direction)
        if norm > 1e-8:
            avoid_direction = avoid_direction / norm

        obstacle_gain = np.clip(proximity_score / max(len(close_features), 1), 0.0, 1.0)
        tau_scale = np.clip(proximity_score - 0.5, -1.0, 1.0)
        goal_offset = np.clip(avoid_direction * self.goal_offset_max, -self.goal_offset_max, self.goal_offset_max)

        return np.concatenate(
            [
                # *联动修改：
                # 这里前两个量仍是 obstacle_gain / tau_scale 的旧接口，
                # 与当前 dmp_rl.py 里“forcing_term + goal_offset”的新接口不一致。
                # 后续需要和 _parse_action() 一起统一。
                np.array([obstacle_gain * 2.0 - 1.0, tau_scale], dtype=float),
                goal_offset / max(self.goal_offset_max, 1e-6),
            ]
        )
