from copy import deepcopy
from types import SimpleNamespace
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from baseline.common.buffers import ReplayBuffer
from baseline.common.noise import ActionNoise
from baseline.common.off_policy_algorithm import OffPolicyAlgorithm
from baseline.common.policies import BasePolicy
from baseline.common.type_aliases import GymEnv, MaybeCallback, Schedule
from baseline.common.utils import get_parameters_by_name, polyak_update
from baseline.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy

from .net import Actor as NetActor, Critic as NetCritic

SelfSAC = TypeVar("SelfSAC", bound="SAC")


class SAC(OffPolicyAlgorithm):
    
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use.
        If ``None``, it will be selected automatically.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param n_steps: When n_step > 1, uses n-step return (with the NStepReplayBuffer) when updating the Q-value network.
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`sac_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: NetActor
    critic: NetCritic
    critic_target: NetCritic

    def __init__(
        self,
        policy: str | type[SACPolicy],                    # 使用哪一种策略网络结构
        env: GymEnv | str,                                # 训练环境
        learning_rate: float | Schedule = 3e-4,           # 学习率，可以是固定值，也可以是随训练进度变化的函数
        buffer_size: int = 1_000_000,  # 1e6              # 回放缓冲区容量
        learning_starts: int = 100,                       # 先收集多少步数据，再开始训练
        batch_size: int = 256,                            # 每次更新使用多少条样本
        tau: float = 0.005,                               # target 网络软更新系数
        gamma: float = 0.99,                              # 奖励折扣因子
        train_freq: int | tuple[int, str] = 1,            # 每隔多少步或多少回合进行一次训练
        gradient_steps: int = 1,                          # 每次触发训练后，做多少次梯度更新
        action_noise: ActionNoise | None = None,          # 是否给动作额外加噪声
        replay_buffer_class: type[ReplayBuffer] | None = None,    # 使用哪种回放缓冲区实现
        replay_buffer_kwargs: dict[str, Any] | None = None,       # 回放缓冲区的额外参数
        optimize_memory_usage: bool = False,              # 是否启用更省内存的缓冲区实现
        n_steps: int = 1,                                 # 是否使用 n-step return
        ent_coef: str | float = "auto",                   # 熵系数，可以固定，也可以自动学习
        target_update_interval: int = 1,                  # 每隔多少次梯度更新同步一次 target 网络
        target_entropy: str | float = "auto",             # 自动学习熵系数时的目标熵
        use_sde: bool = False,                            # 是否使用状态相关探索
        sde_sample_freq: int = -1,                        # 使用状态相关探索时，多长时间重采样一次噪声
        use_sde_at_warmup: bool = False,                  # 预热阶段是否也使用状态相关探索
        stats_window_size: int = 100,                     # 日志统计窗口长度
        tensorboard_log: str | None = None,               # tensorboard 日志目录
        policy_kwargs: dict[str, Any] | None = None,      # 传给策略网络构造函数的额外参数
        verbose: int = 0,                                 # 输出详细程度
        seed: int | None = None,                          # 随机种子
        device: th.device | str = "auto",                 # 训练设备，CPU 或 GPU
        _init_setup_model: bool = True,                   # 初始化对象时是否立刻构建模型
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: th.Tensor | None
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: th.optim.Adam | None = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # 这版 SAC 不再依赖 policies.py 里的通用 actor/critic，
        # 而是直接构建 net.py 里的专用网络。
        # 所以这里手动保留 OffPolicyAlgorithm._setup_model() 中真正需要的部分：
        # 1. 学习率调度器
        # 2. 随机种子
        # 3. replay buffer
        # 4. train_freq 转换
        self._setup_lr_schedule()
        self.set_random_seed(self.seed) #

        if self.use_sde:
            raise NotImplementedError("Current net.py design does not support SDE exploration.")

        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                raise NotImplementedError("Current net.py design only supports flat vector observations.")
            self.replay_buffer_class = ReplayBuffer

        if self.replay_buffer is None:
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **replay_buffer_kwargs,
            )

        assert self.env is not None, "Environment must be available before building SAC networks."
        if not isinstance(self.action_space, spaces.Box):
            raise ValueError("Current SAC implementation only supports continuous Box action spaces.")

        # 当前环境观测被设计成：
        # [sensor_observation | extra_observation]
        # 这里直接从环境属性取出两个维度，避免在训练过程中硬编码切分位置。
        self.sensor_observation_dim = int(self.env.get_attr("sensor_observation_dim")[0])
        self.extra_observation_dim = int(self.env.get_attr("extra_observation_dim")[0])
        self.action_dim = int(np.prod(self.action_space.shape))

        if self.action_dim % 2 != 0:
            raise ValueError("Current net.py actor expects the action dimension to be split into two equal parts.")

        # policy_kwargs 在这版里不再传给通用 policy，
        # 而是直接当作 net.py 的网络结构配置使用。
        hidden_dim = int(self.policy_kwargs.get("hidden_dim", 256))
        sensor_output_dim = int(self.policy_kwargs.get("sensor_output_dim", 128))
        num_sensor_layers = int(self.policy_kwargs.get("num_sensor_layers", 2))
        num_observation_layers = int(self.policy_kwargs.get("num_observation_layers", 2))

        actor_config = SimpleNamespace(
            hidden_dim=hidden_dim,
            output_dim=self.action_dim // 2,
            sensor_output_dim=sensor_output_dim,
            num_sensor_layers=num_sensor_layers,
            num_observation_layers=num_observation_layers,
        )
        critic_config = SimpleNamespace(
            hidden_dim=hidden_dim,
            sensor_output_dim=sensor_output_dim,
            num_sensor_layers=num_sensor_layers,
            num_observation_layers=num_observation_layers,
        )

        self.actor = NetActor(
            self.sensor_observation_dim,
            self.extra_observation_dim,
            actor_config,
        ).to(self.device)
        self.critic = NetCritic(
            self.sensor_observation_dim,
            self.extra_observation_dim,
            self.action_dim,
            critic_config,
        ).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)

        # target critic 不参与梯度更新，只通过软更新同步参数
        for parameter in self.critic_target.parameters():
            parameter.requires_grad = False

        self.actor.optimizer = th.optim.Adam(self.actor.parameters(), lr=self.lr_schedule(1))
        self.critic.optimizer = th.optim.Adam(self.critic.parameters(), lr=self.lr_schedule(1))

        # 当前实现不再使用 BasePolicy 风格的 self.policy，
        # 但为了兼容父类接口，保留一个空占位。
        self.policy = None  # type: ignore[assignment]

        # 训练频率需要从 int / tuple 转成内部统一对象
        self._convert_train_freq()

        # 下面这两组参数是给带 BatchNorm 的网络准备的。
        # 软更新 target critic 时，不仅要更新权重，也要同步 running mean / var。
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

        # target entropy 用来控制策略的随机程度。
        # 如果设为 auto，就按动作维度自动给一个默认值。
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)
        
        # 熵系数 ent_coef 决定“探索”和“利用”的平衡。
        # 这里支持两种模式：
        # 1. 固定常数
        # 2. auto，自适应学习
        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # 这里优化的是 log(alpha) 而不是 alpha 本身。
            # 这么做数值上更稳定，也是很多 SAC 实现的常见写法。
            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        # 这版实现直接持有 actor / critic / critic_target，
        # 不再通过 self.policy 间接访问，因此这里不需要额外工作。
        return None

    def _split_observations(self, observations: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        按当前环境的观测组织方式切分 observation。

        observation = [sensor_observation | extra_observation]
        """
        sensor_observation = observations[..., : self.sensor_observation_dim]
        extra_observation = observations[..., self.sensor_observation_dim : self.sensor_observation_dim + self.extra_observation_dim]
        return sensor_observation, extra_observation

    def _actor_action_log_prob(
        self, observations: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        调用当前 net.py 的 actor，并把两组动作重新拼成完整动作。

        当前 actor 输出：
        - forcing_action: 3 维
        - offside_action: 3 维
        - forcing_log_prob: 标量
        - offside_log_prob: 标量

        SAC 训练主链要的是：
        - 完整 action: 6 维
        - 完整 log_prob: 1 维
        """
        sensor_observation, extra_observation = self._split_observations(observations)
        forcing_action, offside_action, forcing_log_prob, offside_log_prob = self.actor(
            sensor_observation,
            extra_observation,
            deterministic=deterministic,
        )
        action = th.cat([forcing_action, offside_action], dim=-1)
        log_prob = forcing_log_prob + offside_log_prob
        return action, log_prob

    def _critic_forward(
        self,
        observations: th.Tensor,
        actions: th.Tensor,
        critic: NetCritic | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        调用 net.py 的 critic。
        """
        if critic is None:
            critic = self.critic
        sensor_observation, extra_observation = self._split_observations(observations)
        return critic(sensor_observation, extra_observation, actions)

    def _q1_forward(
        self,
        observations: th.Tensor,
        actions: th.Tensor,
        critic: NetCritic | None = None,
    ) -> th.Tensor:
        """
        只取第一路 Q 值，给某些训练步骤使用。
        """
        if critic is None:
            critic = self.critic
        sensor_observation, extra_observation = self._split_observations(observations)
        return critic.q1_forward(sensor_observation, extra_observation, actions)

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        把环境动作从原始物理范围缩放到 [-1, 1]。
        """
        assert isinstance(self.action_space, spaces.Box)
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def _unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        把 [-1, 1] 动作还原回环境动作空间。
        """
        assert isinstance(self.action_space, spaces.Box)
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def predict(
        self,
        observation: np.ndarray,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """
        用当前 net.py actor 直接做动作预测。

        返回值保持和 BaseAlgorithm.predict() 一致：
        - 返回给环境的是原始动作空间下的动作
        - 第二项是 recurrent policy 才会用到的状态，这里恒为 None
        """
        obs_array = np.asarray(observation, dtype=np.float32)
        vectorized = obs_array.ndim > 1
        if not vectorized:
            obs_array = obs_array[None, :]

        obs_tensor = th.as_tensor(obs_array, device=self.device, dtype=th.float32)
        with th.no_grad():
            scaled_action, _ = self._actor_action_log_prob(obs_tensor, deterministic=deterministic)

        scaled_action = scaled_action.cpu().numpy()
        action = self._unscale_action(scaled_action)
        if not vectorized:
            action = action.squeeze(axis=0)
        return action, state

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        按当前自定义 actor 的方式采样动作。

        这里直接复写父类逻辑，避免继续依赖 BasePolicy 的 scale/unscale/predict 接口。
        """
        if self.num_timesteps < learning_starts:
            action = np.array([self.action_space.sample() for _ in range(n_envs)])
            buffer_action = self._scale_action(action)
        else:
            assert self._last_obs is not None, "self._last_obs was not set"
            action, _ = self.predict(self._last_obs, deterministic=False)
            buffer_action = self._scale_action(action)

            if action_noise is not None:
                buffer_action = np.clip(buffer_action + action_noise(), -1, 1)
                action = self._unscale_action(buffer_action)

        return action, buffer_action

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # 进入训练模式，影响到 BatchNorm / Dropout 这类模块的行为
        self.actor.train(True)
        self.critic.train(True)
        self.critic_target.train(False)

        # 需要更新学习率的优化器：
        # 1. actor
        # 2. critic
        # 3. 可选的熵系数优化器
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # 按当前训练进度更新学习率
        self._update_learning_rate(optimizers)

        # 记录训练过程中的统计量，最后写入 logger
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # 从 replay buffer 取一个 batch
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # 如果启用了 n-step return，这里折扣系数不一定是 gamma，而是 gamma^n
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # 使用 gSDE 时，每一步梯度更新前都需要重新采样噪声
            # 用当前 actor 在当前状态上重新采样动作。
            # 这里不是直接拿 replay buffer 里的动作，因为更新 actor 时需要当前策略的动作分布。
            actions_pi, log_prob = self._actor_action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # 更新熵系数时，要把 log_prob 从当前图中摘出来，
                # 避免熵系数损失反向影响 actor 本身。
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # 如果熵系数是自动学习的，这里单独更新一次 alpha
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # 用当前 actor 给 next state 采样动作
                next_actions, next_log_prob = self._actor_action_log_prob(replay_data.next_observations)

                # target critic 有两路 Q，先都算出来，再取较小的一路，抑制 Q 值高估
                next_q_values = th.cat(self._critic_forward(replay_data.next_observations, next_actions, self.critic_target), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

                # SAC 的目标值不是纯 Q，还要减去熵项，鼓励策略保持随机性
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)

                # 组装 TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # 用 replay buffer 里的真实动作，计算当前 critic 的 Q 值
            current_q_values = self._critic_forward(replay_data.observations, replay_data.actions)

            # critic loss = 两路 Q 对同一个 target 的均方误差之和
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # 先更新 critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # 再更新 actor。
            # 先用当前策略采样的动作，经过 critic 得到两路 Q，
            # 然后仍然取较小的一路参与 actor loss。
            q_values_pi = th.cat(self._critic_forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # 更新 actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # 每隔若干步，对 target critic 做一次软更新
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # 如果 critic 里有 BatchNorm，这里把运行统计量也同步过去
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        # 记录训练轮次
        self._n_updates += gradient_steps

        # 写入训练日志
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params()  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["actor", "critic", "critic_target", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables


