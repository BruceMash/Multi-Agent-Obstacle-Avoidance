# coding: utf-8

from __future__ import annotations

from functools import reduce
from typing import Any

import numpy as np
from gym.spaces import Box
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, List, TensorType

if "object" not in np.__dict__:
    setattr(np, "object", object)
if "bool" not in np.__dict__:
    setattr(np, "bool", bool)

torch, nn = try_import_torch()
import torch.nn.functional as F


def _space_dim(space: Any) -> int:
    return int(np.prod(space.shape))


def _mlp(input_dim: int, hidden_dim: int, num_layers: int) -> nn.ModuleList:
    layers = nn.ModuleList()
    layers.append(nn.Linear(input_dim, hidden_dim))
    for _ in range(max(0, num_layers - 1)):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
    return layers


def _forward_layers(layers: nn.ModuleList, x: TensorType) -> TensorType:
    for layer in layers:
        x = F.relu(layer(x))
    return x


class ObservationEncoder(nn.Module):
    """Encode high-dimensional sensor observations before policy/value heads."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.layers = _mlp(input_dim, hidden_dim, num_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: TensorType) -> TensorType:
        x = _forward_layers(self.layers, x)
        return F.relu(self.output_layer(x))


class DMPMAPPOModel(TorchModelV2, nn.Module):
    """RLlib MAPPO model with separated sensor encoding and DMP action heads."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.custom_config = model_config["custom_model_config"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = int(self.custom_config["num_agents"])
        self.action_space = action_space
        self.num_outputs = int(num_outputs)

        if not isinstance(action_space, Box):
            raise TypeError("DMPMAPPOModel only supports continuous Box action spaces.")

        self.obs_dim = _space_dim(self.full_obs_space["obs"])
        self.action_dim = _space_dim(action_space)
        expected_outputs = 2 * self.action_dim
        if self.num_outputs != expected_outputs:
            raise ValueError(
                f"continuous Gaussian policy expects {expected_outputs} outputs, got {self.num_outputs}"
            )

        arch_args = self.custom_config.get("model_arch_args", {})
        self.extra_observation_dim = int(arch_args.get("extra_observation_dim", 3))
        self.sensor_observation_dim = int(arch_args.get("sensor_observation_dim", self.obs_dim - self.extra_observation_dim))
        if self.sensor_observation_dim + self.extra_observation_dim != self.obs_dim:
            raise ValueError(
                "sensor_observation_dim + extra_observation_dim must equal single-agent observation dim"
            )

        self.hidden_dim = int(arch_args.get("hidden_dim", arch_args.get("hidden_state_size", 256)))
        self.sensor_output_dim = int(arch_args.get("sensor_output_dim", 128))
        self.num_sensor_layers = int(arch_args.get("num_sensor_layers", 2))
        self.num_observation_layers = int(arch_args.get("num_observation_layers", 2))
        self.log_std_min = float(arch_args.get("actor_log_std_min", -5.0))
        self.log_std_max = float(arch_args.get("actor_log_std_max", 1.0))

        self.forcing_action_dim = self.action_dim // 2
        self.offside_action_dim = self.action_dim - self.forcing_action_dim

        actor_input_dim = self.sensor_output_dim + self.extra_observation_dim
        self.actor_sensor_encoder = ObservationEncoder(
            self.sensor_observation_dim,
            self.sensor_output_dim,
            self.hidden_dim,
            self.num_sensor_layers,
        )
        self.actor_layers = _mlp(actor_input_dim, self.hidden_dim, self.num_observation_layers)
        self.forcing_mu = nn.Linear(self.hidden_dim, self.forcing_action_dim)
        self.forcing_log_std = nn.Linear(self.hidden_dim, self.forcing_action_dim)
        self.offside_mu = nn.Linear(self.hidden_dim, self.offside_action_dim)
        self.offside_log_std = nn.Linear(self.hidden_dim, self.offside_action_dim)

        self.vf_sensor_encoder = ObservationEncoder(
            self.sensor_observation_dim,
            self.sensor_output_dim,
            self.hidden_dim,
            self.num_sensor_layers,
        )
        self.vf_layers = _mlp(actor_input_dim, self.hidden_dim, self.num_observation_layers)
        self.vf_branch = nn.Linear(self.hidden_dim, 1)

        self.cc_sensor_encoder = ObservationEncoder(
            self.sensor_observation_dim,
            self.sensor_output_dim,
            self.hidden_dim,
            self.num_sensor_layers,
        )
        central_input_dim = self.n_agents * actor_input_dim
        if self.custom_config.get("opp_action_in_cc", True):
            central_input_dim += (self.n_agents - 1) * self.action_dim
        self.cc_vf_layers = _mlp(central_input_dim, self.hidden_dim, self.num_observation_layers)
        self.cc_vf_branch = nn.Linear(self.hidden_dim, 1)

        self._features = None
        self._last_obs = None

        self.actors = [
            self.actor_sensor_encoder,
            self.actor_layers,
            self.forcing_mu,
            self.forcing_log_std,
            self.offside_mu,
            self.offside_log_std,
        ]
        self.critics = [
            self.vf_sensor_encoder,
            self.vf_layers,
            self.vf_branch,
            self.cc_sensor_encoder,
            self.cc_vf_layers,
            self.cc_vf_branch,
        ]
        self.actor_initialized_parameters = self.actor_parameters()

    def _split_observation(self, obs: TensorType) -> tuple[TensorType, TensorType]:
        sensor = obs[..., : self.sensor_observation_dim]
        extra = obs[..., self.sensor_observation_dim : self.sensor_observation_dim + self.extra_observation_dim]
        return sensor, extra

    def _local_features(self, obs: TensorType, encoder: ObservationEncoder) -> TensorType:
        sensor_obs, extra_obs = self._split_observation(obs)
        sensor_features = encoder(sensor_obs)
        return torch.cat([sensor_features, extra_obs], dim=-1)

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> tuple[TensorType, List[TensorType]]:
        flat_obs = input_dict["obs"]["obs"].float()
        self._last_obs = flat_obs

        policy_input = self._local_features(flat_obs, self.actor_sensor_encoder)
        self._features = _forward_layers(self.actor_layers, policy_input)

        forcing_mu = self.forcing_mu(self._features)
        offside_mu = self.offside_mu(self._features)
        forcing_log_std = self.forcing_log_std(self._features)
        offside_log_std = self.offside_log_std(self._features)

        mu = torch.cat([forcing_mu, offside_mu], dim=-1)
        log_std = torch.cat([forcing_log_std, offside_log_std], dim=-1)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return torch.cat([mu, log_std], dim=-1), state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._last_obs is not None, "must call forward() first"
        value_input = self._local_features(self._last_obs, self.vf_sensor_encoder)
        value_features = _forward_layers(self.vf_layers, value_input)
        return torch.reshape(self.vf_branch(value_features), [-1])

    def central_value_function(self, state: TensorType, opponent_actions: TensorType | None = None) -> TensorType:
        batch_size = state.shape[0]
        if state.dim() == 2:
            state = state.reshape(batch_size, self.n_agents, self.obs_dim)
        elif state.dim() != 3:
            raise ValueError(f"central state must be rank 2 or 3, got shape {tuple(state.shape)}")

        sensor_obs, extra_obs = self._split_observation(state.float())
        sensor_obs = sensor_obs.reshape(batch_size * self.n_agents, self.sensor_observation_dim)
        sensor_features = self.cc_sensor_encoder(sensor_obs)
        sensor_features = sensor_features.reshape(batch_size, self.n_agents, self.sensor_output_dim)

        central_features = torch.cat([sensor_features, extra_obs], dim=-1).reshape(batch_size, -1)
        if opponent_actions is not None:
            central_features = torch.cat([central_features, opponent_actions.float().reshape(batch_size, -1)], dim=-1)

        value_features = _forward_layers(self.cc_vf_layers, central_features)
        return torch.reshape(self.cc_vf_branch(value_features), [-1])

    def actor_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda module: list(module.parameters()), self.actors))

    def critic_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda module: list(module.parameters()), self.critics))
