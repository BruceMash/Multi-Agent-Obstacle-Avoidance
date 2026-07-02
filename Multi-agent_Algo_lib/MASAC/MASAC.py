import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import numpy as np

_ALGO_LIB_ROOT = Path(__file__).resolve().parents[1]
if str(_ALGO_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGO_LIB_ROOT))

from net.masac import MASACAgentNetworks

try:
    from .Buffer import Buffer
    from .config import MASACNetworkConfig
except ImportError:
    from Buffer import Buffer
    from config import MASACNetworkConfig

'''
这里实现了8种MASAC的写法，
与论文mSAC不同：mSAC 还加入了分解值算法和反事实曲线 mSAC论文链接：https://arxiv.org/pdf/2104.06655

实验发现：效果 
random_steps = 0时 > random_steps = 500
log_std 法1 > log_std 法2 见note中结果
这里选用上述两者较好的结果
'''

## 第一部分：定义Agent类
class Agent:
    def __init__(
        self,
        agent_id,
        obs_dim,
        action_dim,
        dim_info,
        actor_lr,
        critic_lr,
        device,
        network_config,
    ):
        networks = MASACAgentNetworks(
            obs_dim=obs_dim,
            action_dim=action_dim,
            dim_info=dim_info,
            focal_agent_id=agent_id,
            device=device,
            actor_kwargs=network_config.actor_kwargs(),
            critic_kwargs=network_config.critic_kwargs(),
        )
        self.actor = networks.actor
        self.critic = networks.critic
        self.actor_target = networks.actor_target
        self.critic_target = networks.critic_target

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

## 第二部分：定义DQN算法类
class Alpha:    # 自适应调节熵系数
    def __init__(self, action_dim, alpha_lr=0.0001, alpha=0.01,
                 requires_grad=False, is_continue=True, device="cpu"):

        self.log_alpha = torch.tensor(
            np.log(alpha),
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        ) # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        if is_continue:
            self.target_entropy = -action_dim # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper(SAC) 参考原sac论文
        else:
            self.target_entropy = 0.6 * (
                -torch.log(torch.tensor(1.0 / action_dim, device=device))
            ) # 参考:https://zhuanlan.zhihu.com/p/566722896
        self.alpha = self.log_alpha.exp() # 更新actor时无detach会报错,是因为这里只有一个计算图 

    def update_alpha(self, loss):
        self.log_alpha_optimizer.zero_grad()
        loss.backward()
        self.log_alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

class MASAC: #先无attention 再加入
    """
    MASAC 主体

    """
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, buffer_size,
                 device, trick=None, network_config=None):

        self.device = torch.device(device)
        if network_config is None:
            self.network_config = MASACNetworkConfig()
        elif isinstance(network_config, MASACNetworkConfig):
            self.network_config = network_config
        elif isinstance(network_config, dict):
            self.network_config = MASACNetworkConfig(**network_config)
        else:
            raise TypeError("network_config must be MASACNetworkConfig, dict, or None")
        self.temporal_steps = self.network_config.temporal_steps
        action_dims = {int(dims[1]) for dims in dim_info.values()}
        if len(action_dims) != 1:
            raise ValueError("MASAC currently requires homogeneous action dimensions")
        self.action_dim = next(iter(action_dims))
        if self.action_dim % 2 != 0:
            raise ValueError("DMP action dimension must be even: [forcing, goal_offset]")
        self.control_dim = self.action_dim // 2
        self.goal_distance_clip = float(self.network_config.goal_distance_clip)
        self.dmp_k_alpha = float(self.network_config.dmp_k_alpha)
        self.dmp_k_beta = float(self.network_config.dmp_k_beta)
        self.dmp_tau = float(self.network_config.dmp_tau)
        self.forcing_term_min = float(self.network_config.forcing_term_min)
        self.forcing_term_max = float(self.network_config.forcing_term_max)
        self.acceleration_low = self._state_bound_tensor(
            self.network_config.acceleration_low,
            "acceleration_low",
        )
        self.acceleration_high = self._state_bound_tensor(
            self.network_config.acceleration_high,
            "acceleration_high",
        )

        ## dim_info的组织形式: {agent_id: [obs_dim, action_dim]}
        # 根据每个智能体的观测维度与动作维度构建策略网络等
        self.attention = False

        # 为每个agent模块创建空列表
        self.agents  = {}

        # 每个agent的经验回放池
        self.buffers = {}

        # 初始化每个智能体的actor、critic与target网络，用于构建TD Error
        for agent_id, (obs_dim, action_dim) in dim_info.items():
            self.agents[agent_id] = Agent(
                agent_id,
                obs_dim,
                action_dim,
                dim_info,
                actor_lr,
                critic_lr,
                device=self.device,
                network_config=self.network_config,
            )

            # 连续动作时，buffer中保存完整的动作向量，离散动作仅保存动作索引，因此act_dim = action_dim if is_continue else 1作标志位记录
            self.buffers[agent_id] = Buffer(
                buffer_size,
                obs_dim,
                act_dim=action_dim if is_continue else 1,
                device=self.device,
            )
        
        # 是否使用自适应alpha 为true时表示通过梯度反向传播更新alpha
        self.adaptive_alpha = True
        self.alphas = {} 

        for agent_id, (obs_dim, action_dim) in dim_info.items():
            if self.adaptive_alpha:
                # 每个agent维护一个alpha
                self.alphas[agent_id] = Alpha(
                    action_dim,
                    alpha=0.01,
                    requires_grad=True,
                    is_continue=is_continue,
                    device=self.device,
                ) # Alpha(action_dim).alpha 才是值
            else:   # 固定alpha
                self.alphas[agent_id] = Alpha(
                    action_dim,
                    alpha=0.1,
                    requires_grad=False,
                    is_continue=is_continue,
                    device=self.device,
                )
        '''
        更新critic时 熵的采用方式
        '0' (参考github)中的方式, https://github.com/ffelten/MASAC/blob/main/masac/masac.py#L285  
        '1' MAAC论文中的方式, https://github.com/shariqiqbal2810/MAAC/blob/master/algorithms/attention_sac.py#L99
        '''
        # actor，critic更新熵项采用的方式，使用所有agent的alpha和log_pi，还是只使用当前agent的alpha和log_pi，使用0，1索引
        self.entropy_way_c = '1'  # 按w_c,w_a,a_w 顺序'111' > '110' = '001' > '101'  #'111' 则为MAAC更新模式的MASAC 'x10'为MADDPG更新模式的MASAC '001' 则为参考github的MASAC更新模式
        self.entropy_way_a = '1' 
        '''
        更新actor时 动作的采用方式
        '0' MADDPG论文中的方式,  参考 https://github.com/Git-123-Hub/maddpg-pettingzoo-pytorch/blob/master/MADDPG.py#L105 将此方式推广的话, self.entropy_way_a = '1'
        '1' MAAC论文中的方式/(参考github)中的方式 两者 在动作采取一致,但在计算log_pi时有区别
        https://github.com/shariqiqbal2810/MAAC/blob/master/algorithms/attention_sac.py#L139/https://github.com/ffelten/MASAC/blob/main/masac/masac.py#L319
        '''
        # 即只替换当前agent的动作，还是替换所有agent的动作，使用0，1索引
        self.action_way = '1' 
        self.is_continue = is_continue
        self.agent_x = list(self.agents.keys())[0] #sample 用
    
    def select_action(self, obs):
        actions = {}
        with torch.no_grad():
            for agent_id, agent_obs in obs.items():
                agent_obs = torch.as_tensor(
                    agent_obs,
                    dtype=torch.float32,
                    device=self.device,
                )
                if agent_obs.dim() == 1:
                    agent_obs = agent_obs.reshape(1, -1)
                elif agent_obs.dim() == 2:
                    agent_obs = agent_obs.unsqueeze(0)
                else:
                    raise ValueError(
                        "agent observation must have shape [obs_dim] or "
                        f"[temporal_steps, obs_dim], got {tuple(agent_obs.shape)}"
                    )
                if self.is_continue:
                    action, _ = self.agents[agent_id].actor(agent_obs)
                    actions[agent_id] = action.cpu().numpy().squeeze(0)
                else:
                    raise NotImplementedError("discrete MASAC action selection is not implemented")
        return actions
    
    def evaluate_action(self, obs):
        actions = {}
        with torch.no_grad():
            for agent_id, agent_obs in obs.items():
                agent_obs = torch.as_tensor(
                    agent_obs,
                    dtype=torch.float32,
                    device=self.device,
                )
                if agent_obs.dim() == 1:
                    agent_obs = agent_obs.reshape(1, -1)
                elif agent_obs.dim() == 2:
                    agent_obs = agent_obs.unsqueeze(0)
                else:
                    raise ValueError(
                        "agent observation must have shape [obs_dim] or "
                        f"[temporal_steps, obs_dim], got {tuple(agent_obs.shape)}"
                    )
                if self.is_continue:
                    action, _ = self.agents[agent_id].actor(
                        agent_obs,
                        deterministic=True,
                    )
                    actions[agent_id] = action.cpu().numpy().squeeze(0)
                else:
                    raise NotImplementedError("discrete MASAC action evaluation is not implemented")
        return actions
    
    def add(self, obs, action, reward, next_obs, done):
        for agent_id, buffer in self.buffers.items():
            buffer.add(obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id])

    def _state_bound_tensor(self, values, name):
        if values is None:
            return None
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.dim() == 0:
            tensor = tensor.repeat(self.control_dim)
        tensor = tensor.reshape(-1)
        if tensor.shape != (self.control_dim,):
            raise ValueError(
                f"{name} must have shape ({self.control_dim},), "
                f"got {tuple(tensor.shape)}"
            )
        return tensor

    def _current_observation_frame(self, obs, temporal_mask=None):
        if obs.dim() == 2:
            return obs
        if obs.dim() != 3:
            raise ValueError(
                "observation must have shape [batch, obs_dim] or "
                f"[batch, temporal_steps, obs_dim], got {tuple(obs.shape)}"
            )

        batch_size, temporal_steps, _ = obs.shape
        if temporal_mask is None:
            return obs[:, temporal_steps - 1]

        valid_steps = temporal_mask.to(device=obs.device, dtype=torch.bool)
        if valid_steps.shape != (batch_size, temporal_steps):
            raise ValueError(
                "temporal_mask must have shape [batch, temporal_steps], "
                f"got {tuple(valid_steps.shape)}"
            )
        lengths = valid_steps.sum(dim=1).clamp_min(1)
        indices = lengths - 1
        batch_indices = torch.arange(batch_size, device=obs.device)
        return obs[batch_indices, indices]

    def actor_action_to_critic_action(self, obs, actor_action, temporal_mask=None):
        current_obs = self._current_observation_frame(obs, temporal_mask)
        velocity = current_obs[..., : self.control_dim]
        goal_direction = current_obs[..., self.control_dim : 2 * self.control_dim]
        goal_distance = current_obs[
            ...,
            2 * self.control_dim : 2 * self.control_dim + 1,
        ] * self.goal_distance_clip

        forcing = actor_action[..., : self.control_dim].clamp(
            self.forcing_term_min,
            self.forcing_term_max,
        )
        goal_offset = actor_action[..., self.control_dim : 2 * self.control_dim]
        goal_delta = goal_direction * goal_distance
        effective_goal_delta = goal_delta + goal_offset

        sensor_dim = int(self.network_config.sensor_observation_dim or 0)
        extra_dim = int(self.network_config.extra_observation_dim)
        if extra_dim >= 3 and sensor_dim + 3 <= current_obs.shape[-1]:
            k_alpha = current_obs[..., sensor_dim + 1 : sensor_dim + 2]
            k_beta = current_obs[..., sensor_dim + 2 : sensor_dim + 3]
        else:
            k_alpha = torch.as_tensor(
                self.dmp_k_alpha,
                dtype=current_obs.dtype,
                device=current_obs.device,
            )
            k_beta = torch.as_tensor(
                self.dmp_k_beta,
                dtype=current_obs.dtype,
                device=current_obs.device,
            )

        tau = torch.as_tensor(
            self.dmp_tau,
            dtype=current_obs.dtype,
            device=current_obs.device,
        )
        gate = torch.tanh(torch.abs(effective_goal_delta))
        acceleration = (
            k_alpha * (k_beta * effective_goal_delta - tau * velocity)
            + forcing * gate
        ) / (tau ** 2)

        if self.acceleration_low is not None and self.acceleration_high is not None:
            low = self.acceleration_low.to(
                device=acceleration.device,
                dtype=acceleration.dtype,
            )
            high = self.acceleration_high.to(
                device=acceleration.device,
                dtype=acceleration.dtype,
            )
            acceleration = torch.max(torch.min(acceleration, high), low)

        return torch.cat([acceleration, goal_offset], dim=-1)

    def sample(self, batch_size):
        total_size = len(self.buffers[self.agent_x])
        indices = np.random.choice(total_size, batch_size, replace=False)

        obs, action, reward, next_obs, done = {}, {}, {}, {}, {}
        obs_mask, next_obs_mask = {}, {}
        next_action = {}
        next_log_pi = {}
        for agent_id, buffer in self.buffers.items():
            sampled = buffer.sample(indices, sequence_length=self.temporal_steps)
            if self.temporal_steps > 1:
                (
                    obs[agent_id],
                    action[agent_id],
                    reward[agent_id],
                    next_obs[agent_id],
                    done[agent_id],
                    obs_mask[agent_id],
                    next_obs_mask[agent_id],
                ) = sampled
            else:
                obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id] = sampled
                obs_mask[agent_id] = None
                next_obs_mask[agent_id] = None
            with torch.no_grad():
                next_actor_action, next_log_pi[agent_id] = self.agents[agent_id].actor_target(
                    next_obs[agent_id],
                    temporal_mask=next_obs_mask[agent_id],
                )
                next_action[agent_id] = self.actor_action_to_critic_action(
                    next_obs[agent_id],
                    next_actor_action,
                    temporal_mask=next_obs_mask[agent_id],
                )

        return (
            obs,
            action,
            reward,
            next_obs,
            done,
            obs_mask,
            next_obs_mask,
            next_action,
            next_log_pi,
        ) #包含所有智能体的数据

    ## SAC算法相关
    def learn(self, batch_size ,gamma , tau):
        (
            obs,
            action,
            reward,
            next_obs,
            done,
            obs_mask,
            next_obs_mask,
            next_action,
            next_log_pi,
        ) = self.sample(batch_size)
        # Reuse one joint replay batch for all focal-agent updates. The replay
        # tensors and target actions are detached, while policy actions are
        # rebuilt inside the loop so each actor update keeps its own graph.
        # 多智能体特有-- 集中式训练critic:计算next_q值时,要用到所有智能体next状态和动作
        for agent_id, agent in self.agents.items():
            ## 更新前准备
            ''' 这一部分原理和MADDPG 一样'''
            # 必须放for里，否则报二次传播错，原因是原来的数据在计算图中已经被释放了

            with torch.no_grad():
                q1_next_target, q2_next_target = agent.critic_target(
                    next_obs,
                    next_action,
                    temporal_masks=next_obs_mask,
                )
                q_next_target = torch.min(q1_next_target, q2_next_target)

                ''' SAC 特有 '0' 参考github 将next_log_pi 求和 来更新critic , '1' MAAC论文 将当前的next_log_pi 用于更新critic '''
                if self.entropy_way_c == '0':
                    stacked_next_log_pi = torch.stack(
                        [next_log_pi[other_id] for other_id in self.agents.keys()],
                        dim=1,
                    ).sum(dim=1)
                    entropy_next = -stacked_next_log_pi
                elif self.entropy_way_c == '1':
                    entropy_next = -next_log_pi[agent_id]
                else:
                    raise ValueError(f"unsupported entropy_way_c: {self.entropy_way_c}")

                # 先更新critic
                ''' 公式: LQ_w = E_{s,a,r,s',d}[(Q_w(s,a) - (r + gamma * (1 - d) * (Q_w'(s',a') - alpha * log_pi_a(s',a')))^2] '''
                q_target = reward[agent_id] + gamma * (1 - done[agent_id]) * (
                    q_next_target + self.alphas[agent_id].alpha.detach() * entropy_next
                )

            q1, q2 = agent.critic(obs, action, temporal_masks=obs_mask)
            critic_loss = F.mse_loss(q1, q_target.detach()) + F.mse_loss(q2, q_target.detach())
            agent.update_critic(critic_loss)

            ## 再更新actor
            '''公式: Lpi_θ = E_{s,a ~ D}[-Q_w(s,a) + alpha * log_pi_a(s,a)]  
            理解为 最大化函数V,V = Q + alpha * H
            '''
            new_action = {}
            new_log_pi = {}
            for other_id, other_agent in self.agents.items():
                if other_id == agent_id:
                    sampled_action, sampled_log_pi = other_agent.actor(
                        obs[other_id],
                        temporal_mask=obs_mask[other_id],
                    )
                else:
                    with torch.no_grad():
                        sampled_action, sampled_log_pi = other_agent.actor(
                            obs[other_id],
                            temporal_mask=obs_mask[other_id],
                        )
                new_action[other_id] = self.actor_action_to_critic_action(
                    obs[other_id],
                    sampled_action,
                    temporal_mask=obs_mask[other_id],
                )
                new_log_pi[other_id] = sampled_log_pi

            critic_requires_grad = [
                critic_param.requires_grad
                for critic_param in agent.critic.parameters()
            ] # 获取critic参数的 requires_grad字段

            for critic_param in agent.critic.parameters(): # 冻结critic参数
                critic_param.requires_grad_(False)
            try:
                if self.action_way == '0':
                    mixed_action = dict(action)
                    mixed_action[agent_id] = new_action[agent_id]
                    q1_pi, q2_pi = agent.critic(
                        obs,
                        mixed_action,
                        temporal_masks=obs_mask,
                    )
                elif self.action_way == '1':
                    q1_pi, q2_pi = agent.critic(
                        obs,
                        new_action,
                        temporal_masks=obs_mask,
                    )
                else:
                    raise ValueError(f"unsupported action_way: {self.action_way}")
                
                if self.entropy_way_a == '0':
                    stacked_log_pi = torch.stack(
                        [new_log_pi[other_id] for other_id in self.agents.keys()],
                        dim=1,
                    ).sum(dim=1)
                    entropy = -stacked_log_pi
                elif self.entropy_way_a == '1':
                    entropy = -new_log_pi[agent_id]
                else:
                    raise ValueError(f"unsupported entropy_way_a: {self.entropy_way_a}")

                q_pi = torch.min(q1_pi, q2_pi)

                # Actor update only needs dQ/da, not critic parameter gradients.
                actor_loss = (- q_pi - self.alphas[agent_id].alpha.detach() * entropy).mean()
                agent.update_actor(actor_loss)
            finally:
                for critic_param, requires_grad in zip(
                    agent.critic.parameters(),
                    critic_requires_grad,
                ):
                    critic_param.requires_grad_(requires_grad)

            ## 更新alpha
            '''公式: Lα = E_{s,a ~ D} [-α * log_pi_a(s,a) - α * H] = E_{s,a ~ D} [α * (-log_pi_a(s,a) - H)]'''
            if self.adaptive_alpha:
                alpha_loss = (self.alphas[agent_id].alpha * (entropy - self.alphas[agent_id].target_entropy).detach()).mean()
                self.alphas[agent_id].update_alpha(alpha_loss)


        ## 更新所有target网络
        self.update_target(tau)

    def update_target(self, tau):
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        for agent in self.agents.values():
            soft_update(agent.actor_target, agent.actor, tau)
            soft_update(agent.critic_target, agent.critic, tau)

    def save(self, model_path):
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            os.path.join(model_path, f'MASAC.pth')
        )

    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue, model_dir, network_config=None, device='cpu'):
        policy = MASAC(
            dim_info,
            is_continue=is_continue,
            actor_lr=0,
            critic_lr=0,
            buffer_size=0,
            device=device,
            network_config=network_config,
        )
        data = torch.load(os.path.join(model_dir, f'MASAC.pth'), map_location=device)
        for agent_id, agent in policy.agents.items():
            agent.actor.load_state_dict(data[agent_id])

        return policy
    

## 第三部分 main函数
## 环境配置
def get_env(env_name,env_agent_n = None):
    import importlib

    try:
        import gymnasium as gym
    except ImportError:
        import gym

    # 动态导入环境
    module = importlib.import_module(f'pettingzoo.mpe.{env_name}')
    print('env_agent_n or num_good:',env_agent_n) 
    if env_agent_n is None: #默认环境
        env = module.parallel_env(max_cycles=25, continuous_actions=True)
    elif env_name == 'simple_spread_v3' or 'simple_adversary_v3': 
        env = module.parallel_env(max_cycles=25, continuous_actions=True, N = env_agent_n)
    elif env_name == 'simple_tag_v3': 
        env = module.parallel_env(max_cycles=25, continuous_actions=True, num_good= env_agent_n, num_adversaries=3)
    elif env_name == 'simple_world_comm_v3':
        env = module.parallel_env(max_cycles=25, continuous_actions=True, num_good= env_agent_n, num_adversaries=4)
    env.reset()
    dim_info = {}
    for agent_id in env.agents:
        dim_info[agent_id] = []
        if isinstance(env.observation_space(agent_id), gym.spaces.Box):
            dim_info[agent_id].append(env.observation_space(agent_id).shape[0])
        else:
            dim_info[agent_id].append(1)
        if isinstance(env.action_space(agent_id), gym.spaces.Box):
            dim_info[agent_id].append(env.action_space(agent_id).shape[0])
        else:
            dim_info[agent_id].append(env.action_space(agent_id).n)

    return env,dim_info, 1, True # pettingzoo.mpe 环境中，max_action均为1 , 选取连续环境is_continue = True

## make_dir 与DQN.py 里一样
def make_dir(env_name,policy_name = 'DQN',trick = None):
    script_dir = os.path.dirname(os.path.abspath(__file__)) # 当前脚本文件夹
    env_dir = os.path.join(script_dir,'./results', env_name)
    os.makedirs(env_dir) if not os.path.exists(env_dir) else None
    print('trick:',trick)
    # 确定前缀
    if trick is None or not any(trick.values()):
        prefix = policy_name + '_'
    else:
        prefix = policy_name + '_'
        for key in trick.keys():
            if trick[key]:
                prefix += key + '_'
    # 查找现有的文件夹并确定下一个编号
    existing_dirs = [d for d in os.listdir(env_dir) if d.startswith(prefix) and d[len(prefix):].isdigit()]
    max_number = 0 if not existing_dirs else max([int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()])
    model_dir = os.path.join(env_dir, prefix + str(max_number + 1))
    os.makedirs(model_dir)
    return model_dir

''' 
环境见:simple_adversary_v3,simple_crypto_v3,simple_push_v3,simple_reference_v3,simple_speaker_listener_v3,simple_spread_v3,simple_tag_v3
具体见:https://pettingzoo.farama.org/environments/mpe
注意：环境中N个智能体的设置
'''
if __name__ == '__main__':
    import argparse
    import time

    from torch.utils.tensorboard import SummaryWriter

    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="simple_spread_v3") 
    parser.add_argument("--N", type=int, default=None) # 环境中智能体数量 默认None 这里用来对比设置
    # 共有参数
    parser.add_argument("--seed", type=int, default=100) # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(40000))
    parser.add_argument("--save_freq", type=int, default=int(600//4))
    parser.add_argument("--start_steps", type=int, default=500) # 满足此开始更新
    parser.add_argument("--random_steps", type=int, default=0)  #dqn 无此参数 满足此开始自己探索
    parser.add_argument("--learn_steps_interval", type=int, default=1)
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    ## AC参数
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    ## buffer参数   
    parser.add_argument("--buffer_size", type=int, default=1e6) #1e6默认是float,在bufffer中有int强制转换
    parser.add_argument("--batch_size", type=int, default=256)  #保证比start_steps小
    # trick参数
    parser.add_argument("--policy_name", type=str, default='MASAC')
    parser.add_argument("--trick", type=dict, default=None)  
    # device参数   
    parser.add_argument("--device", type=str, default='cpu') # cpu/cuda

    args = parser.parse_args()

    print(args)
    print('-' * 50)
    print('Algorithm:',args.policy_name)

    ## 环境配置
    env,dim_info,max_action,is_continue = get_env(args.env_name, env_agent_n = args.N)
    print(f'Env:{args.env_name}  dim_info:{dim_info}  max_action:{max_action}  max_episodes:{args.max_episodes}')

    ## 随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ### cuda
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random Seed:',args.seed)

    ## 保存model文件夹
    model_dir = make_dir(args.env_name,policy_name = args.policy_name,trick=args.trick)
    print(f'model_dir: {model_dir}')
    writer = SummaryWriter(model_dir)

    ## device参数
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    ## 算法配置
    policy = MASAC(dim_info, is_continue, args.actor_lr, args.critic_lr, args.buffer_size, device, args.trick)

    time_ = time.time()
    ## 训练
    episode_num = 0
    step = 0
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}
    obs,info = env.reset() # 改成env.reset()后增加其鲁棒性，但会导致seed失效
    {agent: env.action_space(agent).seed(seed = args.seed) for agent in env_agents}  # 针对action复现:env.action_space.sample()
    
    while episode_num < args.max_episodes:
        step +=1

        # 获取动作
        if step < args.random_steps: # 区分环境里的action_ 和训练的action 
            action_ = {agent: env.action_space(agent).sample() for agent in env_agents}  # [0,1]
            action = {agent_id: (action_[agent_id] * 2 - 1)* max_action for agent_id in env_agents} # [0,1] -> [-1,1]
        else:
            action = policy.select_action(obs)   # [-1,1]
            action_ = {agent_id: (action[agent_id] + 1) / 2 * max_action for agent_id in env_agents} #[-1,1] -> [0,1] 

        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
        done_bool = {agent_id: done[agent_id] if not truncated[agent_id] else False  for agent_id in env_agents} ### truncated 为超过最大步数
        policy.add(obs, action, reward, next_obs, done_bool)
        episode_reward = {agent_id: episode_reward[agent_id] + reward[agent_id] for agent_id in env_agents}
        obs = next_obs
        
        # episode 结束 ### 在pettingzoo中,env.agents 为空时  一个episode结束
        if any(done.values()):
            ## 显示
            if  (episode_num + 1) % 100 == 0:
                print("episode: {}, reward: {}".format(episode_num + 1, episode_reward))
                for agent_id in env_agents:
                    writer.add_scalar(f'reward_{agent_id}', episode_reward[agent_id], episode_num + 1)
                    train_return[agent_id].append(episode_reward[agent_id])

            episode_num += 1
            obs,info = env.reset() # 改成env.reset()后增加其鲁棒性，但会导致seed失效
            episode_reward = {agent_id: 0 for agent_id in env_agents}
        
        # 满足step,更新网络
        if step > args.start_steps and step % args.learn_steps_interval == 0:
            policy.learn(args.batch_size, args.gamma, args.tau)
        
        # 保存模型
        if episode_num % args.save_freq == 0:
            policy.save(model_dir)

    print('total_time:',time.time()-time_)
    policy.save(model_dir)
    ## 保存数据
    train_return_ = np.array([train_return[agent_id] for agent_id in env.agents])
    if args.N is None:
        np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),train_return_)
    else:
        np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}_N_{len(env_agents)}.npy"),train_return_)
        
