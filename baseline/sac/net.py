# coding: utf-8

'''
当前脚本定义一套面向本项目的 SAC 网络。

这里的设计不是直接复用 stable-baselines 里那套通用 actor/critic，
而是围绕当前环境的观测结构单独组织：
1. sensor observation 单独编码
2. extra observation 作为额外信息直接拼接
3. actor 输出 forcing / offside 两组动作
4. critic 对拼接后的状态动作对输出双 Q 值
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MAX = 5


class ActionMLP(nn.Module):   # 定义基础的 MLP 层
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2):    # 当前输出：[forcing_term std & mu] [offside std & mu]
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # forcing 分支输出 3 维均值和 3 维方差参数
        self.forcing_mu = nn.Linear(hidden_dim, output_dim)
        self.forcing_log_std = nn.Linear(hidden_dim, output_dim)

        # offside 分支输出 3 维均值和 3 维方差参数
        self.offside_mu = nn.Linear(hidden_dim, output_dim)
        self.offside_log_std = nn.Linear(hidden_dim, output_dim)

        # 共享主干，先把输入特征变换到统一隐藏空间
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        # 共享主干特征提取
        for layer in self.layers:
            x = F.relu(layer(x))

        '''
        双分支，分别输出 forcing_term 和 offside 的均值和方差
        '''
        # forcing_term 分支
        f_mu = self.forcing_mu(x)
        f_log_std = self.forcing_log_std(x)

        # offside 分支
        off_mu = self.offside_mu(x)
        off_log_std = self.offside_log_std(x)

        # 通过 softplus 保证方差为正，再做上界截断，防止策略发散
        f_log_std_softplus = torch.min(F.softplus(f_log_std, dim=-1), torch.tensor(LOG_STD_MAX).to(f_log_std.device))
        off_log_std_softplus = torch.min(F.softplus(off_log_std, dim=-1), torch.tensor(LOG_STD_MAX).to(off_log_std.device))

        # 拼接，包含两部分
        mu = torch.cat([f_mu, off_mu], dim=-1)
        f_log_std_softplus = torch.cat([f_log_std_softplus, off_log_std_softplus], dim=-1)

        return mu, f_log_std_softplus   # 返回经过拼接的 mu 和又经过 softplus 的 std


class ObservationEncoder(nn.Module):   # 映射 LiDAR 信息，构建更加密集的障碍物感知
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2):
        super(ObservationEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 这里不区分传感器具体来源，只把输入当作高维观测做编码
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):   # 处理 LiDAR 观测
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x


class Actor(nn.Module):
    def __init__(self, sensor_observation_dim, extra_observation_dim, ActionConfig):
        super(Actor, self).__init__()
        '''
        ActionConfig:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_dim: 隐藏层维度
            sensor_output_dim: sensor 的输出维度
            num_sensor_layers: sensor 的隐藏层数
            num_observation_layers: observation 的隐藏层数
        '''

        # 当前实现把观测拆成两部分：
        # 1. sensor_observation: 传感器主观测
        # 2. extra_observation: 额外状态量，例如 phase / 超参数等
        self.sensor_observation_dim = int(sensor_observation_dim)
        self.extra_observation_dim = int(extra_observation_dim)

        # 这里通过 ActionConfig 统一控制网络宽度和层数
        self.hidden_dim = int(ActionConfig.hidden_dim)
        self.output_dim = int(ActionConfig.output_dim)
        self.sensor_output_dim = int(ActionConfig.sensor_output_dim)
        self.num_sensor_layers = int(ActionConfig.num_sensor_layers)
        self.num_observation_layers = int(ActionConfig.num_observation_layers)

        # 先把传感器观测编码到低维特征空间
        self.sensor_encoder = ObservationEncoder(
            input_dim=self.sensor_observation_dim,
            output_dim=self.sensor_output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_sensor_layers,
        )

        # actor 的共享主干。
        # 输入不是原始传感器，而是编码后的 sensor_feature 与 extra_observation 的拼接结果。
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.sensor_output_dim + self.extra_observation_dim, self.hidden_dim))
        for _ in range(self.num_observation_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        # 这里保留双分支设计：
        # forcing 对应 DMP 强迫项，offside 对应局部目标偏移
        self.forcing_mu = nn.Linear(self.hidden_dim, self.output_dim)
        self.forcing_log_std = nn.Linear(self.hidden_dim, self.output_dim)
        self.offside_mu = nn.Linear(self.hidden_dim, self.output_dim)
        self.offside_log_std = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, sensor_observation, extra_observation, deterministic=False):
        # 编码传感器观测
        sensor_feature = self.sensor_encoder(sensor_observation)

        # 拼接额外观测，得到 actor 真正使用的输入
        observation_feature = torch.cat([sensor_feature, extra_observation], dim=-1)

        # 共享主干提取策略特征
        x = observation_feature
        for layer in self.layers:
            x = F.relu(layer(x))

        # 分别得到 forcing / offside 两组均值
        f_mu = self.forcing_mu(x)
        off_mu = self.offside_mu(x)

        # 分别得到 forcing / offside 两组标准差
        f_std = torch.clamp(F.softplus(self.forcing_log_std(x)), max=LOG_STD_MAX)
        off_std = torch.clamp(F.softplus(self.offside_log_std(x)), max=LOG_STD_MAX)

        # 构造两个独立高斯分布
        f_dist = Normal(f_mu, f_std)
        off_dist = Normal(off_mu, off_std)

        # 训练时重参数采样，评估时直接取均值
        if deterministic:
            f_pre_tanh = f_mu
            off_pre_tanh = off_mu
        else:
            f_pre_tanh = f_dist.rsample()
            off_pre_tanh = off_dist.rsample()

        # 通过 tanh 把动作压到有限范围
        f_action = torch.tanh(f_pre_tanh)
        off_action = torch.tanh(off_pre_tanh)

        # 分别计算两组动作的对数概率，并加上 tanh 修正项
        f_log_prob = f_dist.log_prob(f_pre_tanh)
        f_log_prob = f_log_prob - torch.log(1.0 - f_action.pow(2) + 1e-6)
        f_log_prob = f_log_prob.sum(dim=-1, keepdim=True)

        off_log_prob = off_dist.log_prob(off_pre_tanh)
        off_log_prob = off_log_prob - torch.log(1.0 - off_action.pow(2) + 1e-6)
        off_log_prob = off_log_prob.sum(dim=-1, keepdim=True)

        return f_action, off_action, f_log_prob, off_log_prob  # 得到的是归一化动作(tanh)，对 forcing_term 需要乘以限幅值，对 off_side 则乘以当前的目标点，意为修正的比重

class CriticMLP(nn.Module):
    """
    单路 Q 网络。

    输入为拼接后的观测特征和动作，输出一个标量 Q 值。
    """

    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super(CriticMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # critic 每一路都是一个普通 MLP，只是最后输出一个标量
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


class Critic(nn.Module):
    """
    SAC 使用的双 Q critic。

    编码方式保持和 Action 一致：
    1. 先单独编码 sensor observation
    2. 再和 extra observation、action 拼接
    3. 输出两路 Q 值
    """

    def __init__(self, sensor_observation, extra_observation, action_dim, CriticConfig):
        super(Critic, self).__init__()
        # critic 的输入同样拆成三部分：
        # 1. sensor observation
        # 2. extra observation
        # 3. action
        self.sensor_observation_dim = int(sensor_observation)
        self.extra_observation_dim = int(extra_observation)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(CriticConfig.hidden_dim)
        self.sensor_output_dim = int(CriticConfig.sensor_output_dim)
        self.num_sensor_layers = int(CriticConfig.num_sensor_layers)
        self.num_observation_layers = int(CriticConfig.num_observation_layers)

        # 为了和 actor 输入语义一致，这里同样先编码 sensor 观测
        self.sensor_encoder = ObservationEncoder(
            input_dim=self.sensor_observation_dim,
            output_dim=self.sensor_output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_sensor_layers,
        )

        # SAC 使用双 Q，避免 Q 值高估
        critic_input_dim = self.sensor_output_dim + self.extra_observation_dim + self.action_dim
        self.q1_net = CriticMLP(
            input_dim=critic_input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_observation_layers,
        )
        self.q2_net = CriticMLP(
            input_dim=critic_input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_observation_layers,
        )

    def forward(self, sensor_observation, extra_observation, action):
        # critic 输入由编码后的观测和动作组成
        sensor_feature = self.sensor_encoder(sensor_observation)
        critic_input = torch.cat([sensor_feature, extra_observation, action], dim=-1)
        q1 = self.q1_net(critic_input)
        q2 = self.q2_net(critic_input)
        return q1, q2

    def q1_forward(self, sensor_observation, extra_observation, action):
        # 某些场景只需要第一路 Q 值，例如 actor 更新时的简化调用
        sensor_feature = self.sensor_encoder(sensor_observation)
        critic_input = torch.cat([sensor_feature, extra_observation, action], dim=-1)
        return self.q1_net(critic_input)
