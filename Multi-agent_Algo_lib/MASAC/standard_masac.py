from __future__ import annotations

import os

import torch

from .Buffer import Buffer
from .MASAC import Agent, Alpha, MASAC
from .config import MASACNetworkConfig


class StandardMASAC(MASAC):
    """
    面向直接连续动作环境的标准 MASAC 适配层。

    Actor、centralized twin-Q Critic、target networks、经验回放和学习更新
    均复用现有 MASAC 实现。该类仅建立“策略动作等于 Critic 动作”的直接
    动作语义，并允许奇数维连续动作。
    """

    def __init__(
        self,
        dim_info,
        is_continue,
        actor_lr,
        critic_lr,
        buffer_size,
        device,
        trick=None,
        network_config=None,
    ):
        del trick
        if not dim_info:
            raise ValueError("dim_info cannot be empty")
        if not is_continue:
            raise ValueError("StandardMASAC only supports continuous actions")

        self.device = torch.device(device)
        if network_config is None:
            self.network_config = MASACNetworkConfig()
        elif isinstance(network_config, MASACNetworkConfig):
            self.network_config = network_config
        elif isinstance(network_config, dict):
            self.network_config = MASACNetworkConfig(**network_config)
        else:
            raise TypeError(
                "network_config must be MASACNetworkConfig, dict, or None"
            )

        self.temporal_steps = self.network_config.temporal_steps
        # StandardMASAC复用MASAC.learn；保持与基础类相同的默认裁剪阈值。
        self.actor_gradient_clip = 0.5
        self.critic_gradient_clip = 0.5
        action_dims = {int(dims[1]) for dims in dim_info.values()}
        if len(action_dims) != 1:
            raise ValueError(
                "StandardMASAC requires homogeneous action dimensions"
            )
        self.action_dim = next(iter(action_dims))
        if self.action_dim <= 0:
            raise ValueError("action dimension must be positive")

        self.attention = False
        self.agents = {}
        self.buffers = {}
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
            self.buffers[agent_id] = Buffer(
                buffer_size,
                obs_dim,
                act_dim=action_dim,
                device=self.device,
            )

        self.adaptive_alpha = True
        self.alphas = {
            agent_id: Alpha(
                action_dim,
                alpha=0.01,
                requires_grad=True,
                is_continue=True,
                device=self.device,
            )
            for agent_id, (_, action_dim) in dim_info.items()
        }

        self.entropy_way_c = "1"
        self.entropy_way_a = "1"
        self.action_way = "1"
        self.is_continue = True
        self.agent_x = next(iter(self.agents))

    def actor_action_to_critic_action(
        self,
        obs,
        actor_action,
        temporal_mask=None,
    ):
        """直接动作环境中，Actor 动作与 Critic 动作保持完全一致。"""
        del obs, temporal_mask
        return actor_action

    def learn(self, *args, **kwargs):
        """保持标准连续动作适配层原有的诊断返回接口。"""
        diagnostics = super().learn(*args, **kwargs)
        legacy_keys = (
            "critic_loss",
            "actor_loss",
            "q_replay",
            "q_policy",
            "q_target",
            "entropy",
            "alpha",
            "alpha_loss",
        )
        return {key: diagnostics[key] for key in legacy_keys}

    @classmethod
    def load(
        cls,
        dim_info,
        is_continue,
        model_dir,
        network_config=None,
        device="cpu",
    ) -> "StandardMASAC":
        policy = cls(
            dim_info,
            is_continue=is_continue,
            actor_lr=0.0,
            critic_lr=0.0,
            buffer_size=0,
            device=device,
            network_config=network_config,
        )
        data = torch.load(
            os.path.join(model_dir, "MASAC.pth"),
            map_location=device,
            weights_only=False,
        )
        actor_states = data.get("actors", data)
        policy.checkpoint_has_critics = bool("critics" in data)
        policy.checkpoint_format_version = int(data.get("format_version", 1))
        for agent_id, agent in policy.agents.items():
            agent.actor.load_state_dict(actor_states[agent_id])
            if "critics" in data:
                agent.critic.load_state_dict(data["critics"][agent_id])
            if "actor_targets" in data:
                agent.actor_target.load_state_dict(data["actor_targets"][agent_id])
            else:
                agent.actor_target.load_state_dict(actor_states[agent_id])
            if "critic_targets" in data:
                agent.critic_target.load_state_dict(data["critic_targets"][agent_id])
            elif "critics" in data:
                agent.critic_target.load_state_dict(data["critics"][agent_id])
            if "log_alphas" in data:
                policy.alphas[agent_id].log_alpha.data.copy_(
                    data["log_alphas"][agent_id].to(policy.device)
                )
                policy.alphas[agent_id].alpha = policy.alphas[agent_id].log_alpha.exp()
        return policy


__all__ = ["StandardMASAC"]
