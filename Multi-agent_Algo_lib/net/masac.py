# coding: utf-8

from __future__ import annotations

import math
from copy import deepcopy
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


TensorGroup = Mapping[str, torch.Tensor] | Iterable[torch.Tensor]


def _space_dim(shape_or_space) -> int:
    """Return the flattened dimension of a gym-like space or raw shape."""
    shape = getattr(shape_or_space, "shape", shape_or_space)
    return int(np.prod(shape))


def _mlp(input_dim: int, hidden_dim: int, num_layers: int) -> nn.ModuleList:
    if input_dim <= 0 or hidden_dim <= 0:
        raise ValueError("MLP dimensions must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
    return layers


def _forward_layers(layers: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    for layer in layers:
        x = F.relu(layer(x))
    return x


def _ordered_tensor_list(
    tensors: TensorGroup,
    agent_ids: Sequence[str] | None = None,
) -> list[torch.Tensor]:
    if isinstance(tensors, Mapping):
        if agent_ids is None:
            return list(tensors.values())
        return [tensors[agent_id] for agent_id in agent_ids]
    return list(tensors)


def _masked_pool(
    embeddings: torch.Tensor,
    mask: torch.Tensor | None,
    pooling_method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool [batch, entities, hidden] embeddings and return a validity flag."""
    if embeddings.dim() != 3:
        raise ValueError(
            "embeddings must have shape [batch, entities, hidden], "
            f"got {tuple(embeddings.shape)}"
        )

    batch_size, entity_count, hidden_dim = embeddings.shape
    output_dim = hidden_dim * (2 if pooling_method == "mean_max" else 1)
    if entity_count == 0:
        return (
            embeddings.new_zeros((batch_size, output_dim)),
            torch.zeros((batch_size, 1), dtype=torch.bool, device=embeddings.device),
        )

    if mask is None:
        mask = torch.ones(
            (batch_size, entity_count),
            dtype=torch.bool,
            device=embeddings.device,
        )
    else:
        if mask.shape != (batch_size, entity_count):
            raise ValueError(
                f"mask must have shape {(batch_size, entity_count)}, got {tuple(mask.shape)}"
            )
        mask = mask.to(device=embeddings.device, dtype=torch.bool)

    valid = mask.any(dim=1, keepdim=True)
    expanded_mask = mask.unsqueeze(-1)

    pooled_parts = []
    if pooling_method in {"mean", "mean_max"}:
        masked_sum = (embeddings * expanded_mask.to(embeddings.dtype)).sum(dim=1)
        count = expanded_mask.sum(dim=1).clamp_min(1).to(embeddings.dtype)
        pooled_parts.append(masked_sum / count)

    if pooling_method in {"max", "mean_max"}:
        fill_value = torch.finfo(embeddings.dtype).min
        masked_embeddings = embeddings.masked_fill(~expanded_mask, fill_value)
        max_values = masked_embeddings.max(dim=1).values
        max_values = torch.where(valid, max_values, torch.zeros_like(max_values))
        pooled_parts.append(max_values)

    if not pooled_parts:
        raise ValueError(f"unsupported pooling method: {pooling_method}")
    return torch.cat(pooled_parts, dim=-1), valid

class ObservationEncoder(nn.Module):
    """Encode sensor observations with ray-level attention and temporal context."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,

        hidden_dim: int = 256,
        num_layers: int = 2,
        rnn_layers: int = 1,
        sensor_azimuth_bins: int = 24,
        sensor_elevation_bins: int = 9,
        sensor_elevation_range_deg: tuple[float, float] = (-80.0, 80.0),
        sensor_include_previous_scan: bool = True,
        use_temporal_rnn: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.ego_dim = 7
        self.sensor_include_previous_scan = bool(sensor_include_previous_scan)
        self.ray_measurement_dim = 4 if self.sensor_include_previous_scan else 2
        self.sensor_azimuth_bins = int(sensor_azimuth_bins)
        self.sensor_elevation_bins = int(sensor_elevation_bins)
        self.sensor_elevation_range_deg = tuple(
            float(value) for value in sensor_elevation_range_deg
        )
        self.use_temporal_rnn = bool(use_temporal_rnn)
        if self.sensor_azimuth_bins <= 0 or self.sensor_elevation_bins <= 0:
            raise ValueError("sensor ray bins must be positive")
        if (
            len(self.sensor_elevation_range_deg) != 2
            or self.sensor_elevation_range_deg[0] >= self.sensor_elevation_range_deg[1]
        ):
            raise ValueError("sensor_elevation_range_deg must be increasing")

        self.n_rays = self.sensor_azimuth_bins * self.sensor_elevation_bins
        ray_encoding = self.build_ray_encoding(
            self.sensor_azimuth_bins,
            self.sensor_elevation_bins,
            self.sensor_elevation_range_deg,
        )
        self.register_buffer("ray_encoding", ray_encoding, persistent=False)

        scan_count = 2 if self.sensor_include_previous_scan else 1
        expected_input_dim = self.ego_dim + scan_count * self.n_rays
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"sensor input_dim must be {expected_input_dim} for "
                f"{self.sensor_azimuth_bins}x{self.sensor_elevation_bins} rays, "
                f"got {self.input_dim}"
            )

        self.ray_measurement_layers = _mlp(
            self.ray_measurement_dim,
            hidden_dim,
            num_layers,
        )
        self.ego_layers = _mlp(self.ego_dim, hidden_dim, num_layers)
        self.attn_head = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)
        self.rnn_layers = None
        if self.use_temporal_rnn:
            self.rnn_layers = nn.GRU(
                hidden_dim,
                hidden_dim,
                rnn_layers,
                batch_first=True,
            )
        self.ray_position_projection = nn.Linear(7, hidden_dim)
        self.output_layer = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    @staticmethod
    def build_ray_encoding(
        azimuth_bins: int,
        elevation_bins: int,
        elevation_range_deg: tuple[float, float],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Build a positional encoding for the sensor rays.

        Return:
        ray_encoding: [n_rays, 7]
        每条射线包含:
        [dir_x, dir_y, dir_z, sin_az, cos_az, sin_el, cos_el]
        """
        azimuth_bins = int(azimuth_bins)
        elevation_bins = int(elevation_bins)
        if azimuth_bins <= 0 or elevation_bins <= 0:
            raise ValueError("azimuth_bins and elevation_bins must be positive")

        azimuth_index = torch.arange(azimuth_bins, dtype=dtype)
        azimuth_angles = -torch.pi + 2.0 * torch.pi * azimuth_index / azimuth_bins

        elevation_angles = torch.deg2rad(
            torch.linspace(
                elevation_range_deg[0],
                elevation_range_deg[1],
                elevation_bins,
                dtype=dtype,
            )
        )

        # Create a grid of azimuth/elevation angles
        azimuth_grid = azimuth_angles[:,None].expand(azimuth_bins, elevation_bins)
        elevation_grid = elevation_angles[None,:].expand(azimuth_bins, elevation_bins)

        cos_el = torch.cos(elevation_grid)

        dir_x = cos_el * torch.cos(azimuth_grid)
        dir_y = cos_el * torch.sin(azimuth_grid)
        dir_z = torch.sin(elevation_grid)

        ray_encoding = torch.stack([
            dir_x,
            dir_y,
            dir_z,
            torch.sin(azimuth_grid),
            torch.cos(azimuth_grid),
            torch.sin(elevation_grid),
            torch.cos(elevation_grid)   
            ], dim=-1)
        
        return ray_encoding.reshape(-1, 7).contiguous()  # [n_rays, 7]
        
    def depack_obs(
        self,
        inputs: torch.Tensor,
        temporal_mask: torch.Tensor | None = None,
    ):
        """
        Depack temporal sensor observations.

        Args:
            inputs:
                [batch, temporal_steps, obs_dim]
                或兼容单帧输入 [batch, obs_dim]

            temporal_mask:
                [batch, temporal_steps]
                True 表示该时间步有效，False 表示 padding。

        Returns:
            dict:
                ego_features:     [B, T, 7]
                current_scan:     [B, T, R]
                previous_scan:    [B, T, R] or None
                scan_delta:       [B, T, R] or None
                hit_flag:         [B, T, R]
                ray_measurements: [B, T, R, 4]
                temporal_mask:    [B, T]
        """
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)

        if inputs.dim() != 3:
            raise ValueError(
                "inputs must have shape [batch, temporal_steps, obs_dim] "
                f"or [batch, obs_dim], got {tuple(inputs.shape)}"
            )

        batch_size, temporal_steps, obs_dim = inputs.shape

        ego_dim = self.ego_dim
        scan_count = 2 if self.sensor_include_previous_scan else 1
        expected_dim = ego_dim + scan_count * self.n_rays

        if obs_dim != expected_dim:
            raise ValueError(
                f"expected obs_dim={expected_dim}, got {obs_dim}. "
                "Observation layout should be "
                "[velocity(3), goal_direction(3), goal_distance(1), "
                "current_scan(n_rays), optional previous_scan(n_rays)]."
            )

        if temporal_mask is None:
            temporal_mask = torch.ones(
                batch_size,
                temporal_steps,
                dtype=torch.bool,
                device=inputs.device,
            )
        else:
            temporal_mask = temporal_mask.to(device=inputs.device, dtype=torch.bool)

        if temporal_mask.shape != (batch_size, temporal_steps):
            raise ValueError(
                "temporal_mask must have shape [batch, temporal_steps], "
                f"got {tuple(temporal_mask.shape)}"
            )

        ego_features = inputs[..., :ego_dim]

        scan_start = ego_dim
        scan_mid = scan_start + self.n_rays
        scan_end = scan_mid + self.n_rays

        current_scan = inputs[..., scan_start:scan_mid] # 当前扫描输入
        previous_scan = None
        scan_delta = None
        if self.sensor_include_previous_scan:
            previous_scan = inputs[..., scan_mid:scan_end]
            scan_delta = current_scan - previous_scan

        # current_scan 已经被归一化到 [0, 1]，1.0 通常表示未命中或达到最大探测距离。
        hit_flag = (current_scan < 1.0).to(inputs.dtype)

        measurement_parts = [current_scan]
        if self.sensor_include_previous_scan:
            measurement_parts.extend([previous_scan, scan_delta])
        measurement_parts.append(hit_flag)
        ray_measurements = torch.stack(measurement_parts, dim=-1)

        return {
            "ego_features": ego_features,
            "current_scan": current_scan,
            "previous_scan": previous_scan,
            "scan_delta": scan_delta,
            "hit_flag": hit_flag,
            "ray_measurements": ray_measurements,
            "temporal_mask": temporal_mask,
        }

    def positional_encoding(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Add fixed ray-geometry encoding to sensor ray tokens.

        inputs:
            [batch, n_rays, hidden_dim]
            or [batch, temporal_steps, n_rays, hidden_dim]
        """
        if inputs.shape[-2] != self.n_rays:
            raise ValueError(
                "inputs must have shape [..., n_rays, hidden_dim], "
                f"got {tuple(inputs.shape)}"
            )
        if inputs.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"input hidden dim must be {self.hidden_dim}, got {inputs.shape[-1]}"
            )

        position = self.ray_position_projection(
            self.ray_encoding.to(dtype=inputs.dtype, device=inputs.device)
        )
    
        view_shape = (
            *([1] * (inputs.dim() - 2)),
            self.n_rays,
            inputs.shape[-1],
        )
        return inputs + position.view(view_shape)

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        depacked = self.depack_obs(x, temporal_mask)
        ego_features = depacked["ego_features"]
        ray_measurements = depacked["ray_measurements"]
        temporal_mask = depacked["temporal_mask"]

        batch_size, temporal_steps, _, _ = ray_measurements.shape
        ray_tokens = _forward_layers(self.ray_measurement_layers, ray_measurements)
        ray_tokens = self.positional_encoding(ray_tokens)

        ego_tokens = _forward_layers(self.ego_layers, ego_features)
        query = ego_tokens.reshape(batch_size * temporal_steps, 1, self.hidden_dim)
        key_value = ray_tokens.reshape(
            batch_size * temporal_steps,
            self.n_rays,
            self.hidden_dim,
        )
        frame_context, _ = self.attn_head(
            query,
            key_value,
            key_value,
            need_weights=False,
        )
        frame_context = frame_context.squeeze(1).reshape(
            batch_size,
            temporal_steps,
            self.hidden_dim,
        )

        valid_steps = temporal_mask.to(device=x.device, dtype=torch.bool)
        frame_context = frame_context * valid_steps.unsqueeze(-1).to(frame_context.dtype)
        sequence_lengths = valid_steps.sum(dim=1)
        safe_lengths = sequence_lengths.clamp_min(1)
        last_indices = safe_lengths - 1
        batch_indices = torch.arange(batch_size, device=x.device)
        current_context = frame_context[batch_indices, last_indices]

        if self.rnn_layers is None:
            temporal_context = current_context
        else:
            packed_context = nn.utils.rnn.pack_padded_sequence(
                frame_context,
                safe_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, temporal_hidden = self.rnn_layers(packed_context)
            temporal_context = temporal_hidden[-1]

        valid_sequences = sequence_lengths.gt(0).unsqueeze(-1).to(frame_context.dtype)
        current_context = current_context * valid_sequences
        temporal_context = temporal_context * valid_sequences

        fused_context = torch.cat([current_context, temporal_context], dim=-1)
        return self.output_layer(fused_context) * valid_sequences


class AllyObservationEncoder(nn.Module):
    """Encode a variable-sized set of explicitly identified allied agents."""

    VALID_POOLING_METHODS = {"mean", "max", "mean_max"}

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pooling_method: str = "mean_max",
    ):
        super().__init__()
        if feature_dim <= 0 or output_dim <= 0:
            raise ValueError("ally feature and output dimensions must be positive")
        if pooling_method not in self.VALID_POOLING_METHODS:
            raise ValueError(
                f"invalid pooling method {pooling_method!r}; "
                f"expected one of {sorted(self.VALID_POOLING_METHODS)}"
            )

        self.feature_dim = int(feature_dim)
        self.output_dim = int(output_dim)
        self.pooling_method = pooling_method
        self.layers = _mlp(self.feature_dim, hidden_dim, num_layers)
        pooled_dim = hidden_dim * (2 if pooling_method == "mean_max" else 1)
        self.output_layer = nn.Linear(pooled_dim, self.output_dim)

    def forward(
        self,
        ally_observations: torch.Tensor,
        ally_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squeeze_batch = ally_observations.dim() == 2
        if squeeze_batch:
            ally_observations = ally_observations.unsqueeze(0)
            if ally_mask is not None and ally_mask.dim() == 1:
                ally_mask = ally_mask.unsqueeze(0)

        if ally_observations.dim() != 3:
            raise ValueError(
                "ally_observations must have shape [batch, allies, features], "
                f"got {tuple(ally_observations.shape)}"
            )
        if ally_observations.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected ally feature dimension {self.feature_dim}, "
                f"got {ally_observations.shape[-1]}"
            )

        batch_size, ally_count, _ = ally_observations.shape
        if ally_count == 0:
            output = ally_observations.new_zeros((batch_size, self.output_dim))
        else:
            embeddings = _forward_layers(self.layers, ally_observations)
            pooled, valid = _masked_pool(embeddings, ally_mask, self.pooling_method)
            output = F.relu(self.output_layer(pooled))
            output = output * valid.to(output.dtype)

        return output.squeeze(0) if squeeze_batch else output


class MASACObservationEncoder(nn.Module):
    """Encode sensor, DMP-extra, and variable-sized ally observation blocks."""

    def __init__(
        self,
        obs_dim: int,
        sensor_observation_dim: int | None = None,
        extra_observation_dim: int = 0,
        ally_feature_dim: int = 0,
        sensor_output_dim: int = 128,
        ally_output_dim: int = 64,
        sensor_hidden_dim: int = 128,
        ally_hidden_dim: int = 128,
        hidden_dim: int = 256,
        num_sensor_layers: int = 2,
        num_ally_layers: int = 2,
        ally_pooling: str = "mean_max",
        sensor_azimuth_bins: int = 24,
        sensor_elevation_bins: int = 9,
        sensor_elevation_range_deg: tuple[float, float] = (-80.0, 80.0),
        sensor_include_previous_scan: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.extra_observation_dim = int(extra_observation_dim)
        self.ally_feature_dim = int(ally_feature_dim)

        if self.obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if self.extra_observation_dim < 0 or self.ally_feature_dim < 0:
            raise ValueError("observation block dimensions must be non-negative")

        if sensor_observation_dim is None:
            if self.ally_feature_dim > 0:
                raise ValueError(
                    "sensor_observation_dim is required when ally_feature_dim is enabled"
                )
            sensor_observation_dim = self.obs_dim - self.extra_observation_dim

        self.sensor_observation_dim = int(sensor_observation_dim)
        self.fixed_observation_dim = (
            self.sensor_observation_dim + self.extra_observation_dim
        )
        if self.sensor_observation_dim <= 0:
            raise ValueError("sensor_observation_dim must be positive")
        if self.fixed_observation_dim > self.obs_dim:
            raise ValueError(
                "sensor_observation_dim + extra_observation_dim exceeds obs_dim"
            )

        initial_ally_dim = self.obs_dim - self.fixed_observation_dim
        if self.ally_feature_dim == 0 and initial_ally_dim != 0:
            raise ValueError(
                "observation contains an unassigned trailing block; set ally_feature_dim"
            )
        if self.ally_feature_dim > 0 and initial_ally_dim % self.ally_feature_dim != 0:
            raise ValueError(
                f"ally observation dimension {initial_ally_dim} is not divisible by "
                f"ally_feature_dim={self.ally_feature_dim}"
            )

        self.sensor_encoder = ObservationEncoder(
            self.sensor_observation_dim,
            sensor_output_dim,
            sensor_hidden_dim,
            num_sensor_layers,
            sensor_azimuth_bins=sensor_azimuth_bins,
            sensor_elevation_bins=sensor_elevation_bins,
            sensor_elevation_range_deg=sensor_elevation_range_deg,
            sensor_include_previous_scan=sensor_include_previous_scan,
        )
        self.ally_encoder = None
        if self.ally_feature_dim > 0:
            self.ally_encoder = AllyObservationEncoder(
                self.ally_feature_dim,
                ally_output_dim,
                hidden_dim=ally_hidden_dim,
                num_layers=num_ally_layers,
                pooling_method=ally_pooling,
            )

        self.output_dim = int(sensor_output_dim) + self.extra_observation_dim
        if self.ally_encoder is not None:
            self.output_dim += int(ally_output_dim)

    def split_observation(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if obs.shape[-1] < self.fixed_observation_dim:
            raise ValueError(
                f"observation has dimension {obs.shape[-1]}, expected at least "
                f"{self.fixed_observation_dim}"
            )

        sensor_end = self.sensor_observation_dim
        extra_end = sensor_end + self.extra_observation_dim
        sensor_obs = obs[..., :sensor_end]
        extra_obs = obs[..., sensor_end:extra_end]
        ally_flat = obs[..., extra_end:]

        if self.ally_encoder is None:
            if ally_flat.shape[-1] != 0:
                raise ValueError("received ally observations while ally encoder is disabled")
            return sensor_obs, extra_obs, None

        if ally_flat.shape[-1] % self.ally_feature_dim != 0:
            raise ValueError(
                f"ally observation dimension {ally_flat.shape[-1]} is not divisible by "
                f"ally_feature_dim={self.ally_feature_dim}"
            )
        ally_count = ally_flat.shape[-1] // self.ally_feature_dim
        ally_obs = ally_flat.reshape(*ally_flat.shape[:-1], ally_count, self.ally_feature_dim)
        return sensor_obs, extra_obs, ally_obs

    def forward(
        self,
        obs: torch.Tensor,
        ally_mask: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sensor_obs, extra_obs, ally_obs = self.split_observation(obs.float())

        if obs.dim() == 2:
            parts = [self.sensor_encoder(sensor_obs, temporal_mask), extra_obs]
            if self.ally_encoder is not None:
                parts.append(self.ally_encoder(ally_obs, ally_mask))
            return torch.cat(parts, dim=-1)

        if obs.dim() != 3:
            raise ValueError(
                "obs must have shape [batch, obs_dim] or [batch, temporal_steps, obs_dim], "
                f"got {tuple(obs.shape)}"
            )

        batch_size, temporal_steps, _ = obs.shape
        if temporal_mask is None:
            valid_steps = torch.ones(
                batch_size,
                temporal_steps,
                dtype=torch.bool,
                device=obs.device,
            )
        else:
            valid_steps = temporal_mask.to(device=obs.device, dtype=torch.bool)
            if valid_steps.shape != (batch_size, temporal_steps):
                raise ValueError(
                    f"temporal_mask must have shape {(batch_size, temporal_steps)}, "
                    f"got {tuple(valid_steps.shape)}"
                )

        sequence_lengths = valid_steps.sum(dim=1).clamp_min(1)
        last_indices = sequence_lengths - 1
        batch_indices = torch.arange(batch_size, device=obs.device)

        sensor_features = self.sensor_encoder(sensor_obs, valid_steps)
        extra_current = extra_obs[batch_indices, last_indices]
        parts = [sensor_features, extra_current]

        if self.ally_encoder is not None:
            ally_current = ally_obs[batch_indices, last_indices]
            ally_mask_current = ally_mask
            if ally_mask is not None:
                if ally_mask.dim() == 3:
                    ally_mask_current = ally_mask.to(
                        device=obs.device,
                        dtype=torch.bool,
                    )[batch_indices, last_indices]
                elif ally_mask.dim() == 2:
                    ally_mask_current = ally_mask.to(device=obs.device, dtype=torch.bool)
                else:
                    raise ValueError(
                        "ally_mask must have shape [batch, allies] or "
                        f"[batch, temporal_steps, allies], got {tuple(ally_mask.shape)}"
                    )
            parts.append(self.ally_encoder(ally_current, ally_mask_current))
        return torch.cat(parts, dim=-1)


class MASACActor(nn.Module):
    """Tanh-Gaussian MASAC actor with DMP-specific action heads."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        sensor_observation_dim: int | None = None,
        extra_observation_dim: int = 0,
        ally_feature_dim: int = 0,
        sensor_output_dim: int = 128,
        ally_output_dim: int = 64,
        sensor_hidden_dim: int = 128,
        ally_hidden_dim: int = 128,
        hidden_dim: int = 256,
        num_sensor_layers: int = 2,
        num_ally_layers: int = 2,
        num_observation_layers: int = 2,
        ally_pooling: str = "mean_max",
        sensor_azimuth_bins: int = 24,
        sensor_elevation_bins: int = 9,
        sensor_elevation_range_deg: tuple[float, float] = (-80.0, 80.0),
        sensor_include_previous_scan: bool = True,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        action_low: Sequence[float] | None = None,
        action_high: Sequence[float] | None = None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")
        if (action_low is None) != (action_high is None):
            raise ValueError("action_low and action_high must be both set or both None")
        if action_low is None:
            action_scale = torch.ones(self.action_dim, dtype=torch.float32)
            action_bias = torch.zeros(self.action_dim, dtype=torch.float32)
        else:
            action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32)
            action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32)
            if action_low_tensor.shape != (self.action_dim,):
                raise ValueError(
                    f"action_low must have shape ({self.action_dim},), "
                    f"got {tuple(action_low_tensor.shape)}"
                )
            if action_high_tensor.shape != (self.action_dim,):
                raise ValueError(
                    f"action_high must have shape ({self.action_dim},), "
                    f"got {tuple(action_high_tensor.shape)}"
                )
            if torch.any(action_high_tensor <= action_low_tensor):
                raise ValueError("action_high must be larger than action_low")
            action_scale = 0.5 * (action_high_tensor - action_low_tensor)
            action_bias = 0.5 * (action_high_tensor + action_low_tensor)
        self.register_buffer("action_scale", action_scale, persistent=False)
        self.register_buffer("action_bias", action_bias, persistent=False)

        self.observation_encoder = MASACObservationEncoder(
            obs_dim=self.obs_dim,
            sensor_observation_dim=sensor_observation_dim,
            extra_observation_dim=extra_observation_dim,
            ally_feature_dim=ally_feature_dim,
            sensor_output_dim=sensor_output_dim,
            ally_output_dim=ally_output_dim,
            sensor_hidden_dim=sensor_hidden_dim,
            ally_hidden_dim=ally_hidden_dim,
            hidden_dim=hidden_dim,
            num_sensor_layers=num_sensor_layers,
            num_ally_layers=num_ally_layers,
            ally_pooling=ally_pooling,
            sensor_azimuth_bins=sensor_azimuth_bins,
            sensor_elevation_bins=sensor_elevation_bins,
            sensor_elevation_range_deg=sensor_elevation_range_deg,
            sensor_include_previous_scan=sensor_include_previous_scan,
        )
        self.policy_layers = _mlp(
            self.observation_encoder.output_dim,
            hidden_dim,
            num_observation_layers,
        )

        self.forcing_action_dim = self.action_dim // 2

        # 定义forcing过程 两个MLP
        self.goal_offset_action_dim = self.action_dim - self.forcing_action_dim
        self.forcing_mu = nn.Linear(hidden_dim, self.forcing_action_dim)
        self.forcing_log_std = nn.Linear(hidden_dim, self.forcing_action_dim)

        # 定义偏移过程 两个MLP
        self.goal_offset_mu = nn.Linear(hidden_dim, self.goal_offset_action_dim)
        self.goal_offset_log_std = nn.Linear(hidden_dim, self.goal_offset_action_dim)

        # DMP Actor 学习的是基础吸引轨迹上的残差。新策略若以默认线性层
        # 初始化，会立即产生大幅 forcing/goal offset，并在训练早期频繁触发
        # 加速度裁剪。均值头从零残差开始，较低初始标准差保留局部探索，
        # 同时不改变参数形状，因此仍兼容既有 checkpoint。
        self._initialize_action_heads()

    def _initialize_action_heads(self) -> None:
        for mean_head in (self.forcing_mu, self.goal_offset_mu):
            nn.init.zeros_(mean_head.weight)
            nn.init.zeros_(mean_head.bias)
        for log_std_head in (
            self.forcing_log_std,
            self.goal_offset_log_std,
        ):
            nn.init.normal_(log_std_head.weight, mean=0.0, std=1e-3)
            nn.init.constant_(log_std_head.bias, -2.0)

    def encode_observation(
        self,
        obs: torch.Tensor,
        ally_mask: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.observation_encoder(obs, ally_mask, temporal_mask)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = True,
        ally_mask: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.encode_observation(obs, ally_mask, temporal_mask)
        features = _forward_layers(self.policy_layers, features)

        mean = torch.cat(
            [self.forcing_mu(features), self.goal_offset_mu(features)], dim=-1
        )
        log_std = torch.cat(
            [
                self.forcing_log_std(features),
                self.goal_offset_log_std(features),
            ],
            dim=-1,
        ).clamp(self.log_std_min, self.log_std_max)

        distribution = Normal(mean, log_std.exp())
        pre_tanh_action = mean if deterministic else distribution.rsample()

        log_prob = None
        if with_logprob:
            log_prob = distribution.log_prob(pre_tanh_action).sum(dim=-1, keepdim=True)
            correction = 2.0 * (
                math.log(2.0)
                - pre_tanh_action
                - F.softplus(-2.0 * pre_tanh_action)
            )
            log_prob = log_prob - correction.sum(dim=-1, keepdim=True)

        squashed_action = torch.tanh(pre_tanh_action)
        action = squashed_action * self.action_scale + self.action_bias
        if log_prob is not None:
            log_prob = log_prob - self.action_scale.log().sum()

        return action, log_prob


class _CentralizedQBranch(nn.Module):
    """Centralized Q estimator with focal LiDAR and compact neighbor context."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        focal_agent_index: int,
        *,
        sensor_observation_dim: int | None,
        extra_observation_dim: int,
        ally_feature_dim: int,
        sensor_output_dim: int,
        ally_output_dim: int,
        sensor_hidden_dim: int,
        ally_hidden_dim: int,
        hidden_dim: int,
        num_sensor_layers: int,
        num_ally_layers: int,
        num_observation_layers: int,
        ally_pooling: str,
        sensor_azimuth_bins: int,
        sensor_elevation_bins: int,
        sensor_elevation_range_deg: tuple[float, float],
        sensor_include_previous_scan: bool,
        agent_pooling: str,
    ):
        super().__init__()
        self.focal_agent_index = int(focal_agent_index)
        self.agent_pooling = agent_pooling
        if agent_pooling not in AllyObservationEncoder.VALID_POOLING_METHODS:
            raise ValueError(f"unsupported agent pooling method: {agent_pooling}")

        if sensor_observation_dim is None:
            raise ValueError("compact centralized critic requires sensor_observation_dim")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.sensor_observation_dim = int(sensor_observation_dim)
        self.extra_observation_dim = int(extra_observation_dim)
        self.ally_feature_dim = int(ally_feature_dim)
        self.fixed_observation_dim = (
            self.sensor_observation_dim + self.extra_observation_dim
        )
        trailing_dim = self.obs_dim - self.fixed_observation_dim
        if trailing_dim < 0:
            raise ValueError("fixed critic observation blocks exceed obs_dim")
        if self.ally_feature_dim <= 0 and trailing_dim != 0:
            raise ValueError("critic observation has an unassigned trailing block")
        if self.ally_feature_dim > 0 and trailing_dim % self.ally_feature_dim != 0:
            raise ValueError("critic ally block is not divisible by ally_feature_dim")
        self.ally_count = (
            trailing_dim // self.ally_feature_dim
            if self.ally_feature_dim > 0
            else 0
        )

        self.focal_sensor_encoder = ObservationEncoder(
            self.sensor_observation_dim,
            sensor_output_dim,
            hidden_dim=sensor_hidden_dim,
            num_layers=num_sensor_layers,
            sensor_azimuth_bins=sensor_azimuth_bins,
            sensor_elevation_bins=sensor_elevation_bins,
            sensor_elevation_range_deg=sensor_elevation_range_deg,
            sensor_include_previous_scan=sensor_include_previous_scan,
            use_temporal_rnn=False,
        )

        self.relation_encoder = None
        relation_output_dim = 0
        if self.ally_feature_dim > 0:
            self.relation_encoder = AllyObservationEncoder(
                self.ally_feature_dim,
                ally_output_dim,
                hidden_dim=ally_hidden_dim,
                num_layers=num_ally_layers,
                pooling_method=ally_pooling,
            )
            relation_output_dim = int(ally_output_dim)

        # Neighbor tokens intentionally exclude LiDAR, duplicated ally blocks,
        # and constant DMP gains. They retain ego motion/goal, phase, and action.
        compact_ego_dim = 7
        compact_phase_dim = 1 if self.extra_observation_dim > 0 else 0
        self.neighbor_token_dim = compact_ego_dim + compact_phase_dim + self.action_dim
        self.neighbor_layers = _mlp(
            self.neighbor_token_dim,
            hidden_dim,
            num_observation_layers,
        )
        own_input_dim = int(sensor_output_dim) + compact_phase_dim + self.action_dim
        self.own_layers = _mlp(own_input_dim, hidden_dim, num_observation_layers)
        pooled_dim = hidden_dim * (2 if agent_pooling == "mean_max" else 1)
        self.q_layers = _mlp(
            hidden_dim + pooled_dim + relation_output_dim,
            hidden_dim,
            num_observation_layers,
        )
        self.q_output = nn.Linear(hidden_dim, 1)

    def _current_blocks(
        self,
        observation: torch.Tensor,
        temporal_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = _select_current_observation(observation, temporal_mask)
        if current.shape[-1] != self.obs_dim:
            raise ValueError(
                f"expected critic obs_dim={self.obs_dim}, got {current.shape[-1]}"
            )
        sensor_end = self.sensor_observation_dim
        extra_end = sensor_end + self.extra_observation_dim
        return (
            current[..., :sensor_end],
            current[..., sensor_end:extra_end],
            current[..., extra_end:],
        )

    def forward(
        self,
        observations: Sequence[torch.Tensor],
        actions: Sequence[torch.Tensor],
        agent_mask: torch.Tensor | None = None,
        temporal_masks: Sequence[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        if len(observations) != len(actions):
            raise ValueError("observation and action agent counts must match")
        if not observations:
            raise ValueError("centralized critic requires at least one agent")

        if temporal_masks is None:
            temporal_masks = [None] * len(observations)
        elif len(temporal_masks) != len(observations):
            raise ValueError("temporal mask agent count must match observations")

        current_blocks = [
            self._current_blocks(observation, temporal_mask)
            for observation, temporal_mask in zip(observations, temporal_masks)
        ]
        batch_size = actions[0].shape[0]
        agent_count = len(observations)
        if not 0 <= self.focal_agent_index < agent_count:
            raise ValueError("focal agent index is outside the current agent set")

        focal_sensor, focal_extra, focal_relations = current_blocks[
            self.focal_agent_index
        ]
        focal_sensor_features = self.focal_sensor_encoder(focal_sensor)
        focal_parts = [focal_sensor_features]
        if self.extra_observation_dim > 0:
            focal_parts.append(focal_extra[..., :1])
        focal_parts.append(actions[self.focal_agent_index])
        own_token = _forward_layers(self.own_layers, torch.cat(focal_parts, dim=-1))

        relation_context = None
        if self.relation_encoder is not None:
            relation_features = focal_relations.reshape(
                batch_size,
                self.ally_count,
                self.ally_feature_dim,
            )
            relation_context = self.relation_encoder(relation_features)

        neighbor_tokens = []
        neighbor_indices = []
        for index, ((sensor_obs, extra_obs, _), action) in enumerate(
            zip(current_blocks, actions)
        ):
            if index == self.focal_agent_index:
                continue
            token_parts = [sensor_obs[..., :7]]
            if self.extra_observation_dim > 0:
                token_parts.append(extra_obs[..., :1])
            token_parts.append(action)
            neighbor_tokens.append(torch.cat(token_parts, dim=-1))
            neighbor_indices.append(index)

        if neighbor_tokens:
            stacked_neighbors = torch.stack(neighbor_tokens, dim=1)
            stacked_neighbors = _forward_layers(self.neighbor_layers, stacked_neighbors)
        else:
            stacked_neighbors = own_token.new_zeros((batch_size, 0, own_token.shape[-1]))

        if agent_mask is None:
            neighbor_mask = torch.ones(
                (batch_size, len(neighbor_indices)),
                dtype=torch.bool,
                device=own_token.device,
            )
        else:
            if agent_mask.shape != (batch_size, agent_count):
                raise ValueError(
                    f"agent_mask must have shape {(batch_size, agent_count)}, "
                    f"got {tuple(agent_mask.shape)}"
                )
            full_mask = agent_mask.to(device=own_token.device, dtype=torch.bool)
            neighbor_mask = full_mask[:, neighbor_indices]

        other_context, _ = _masked_pool(
            stacked_neighbors,
            neighbor_mask,
            self.agent_pooling,
        )
        q_parts = [own_token, other_context]
        if relation_context is not None:
            q_parts.append(relation_context)
        q_features = torch.cat(q_parts, dim=-1)
        q_features = _forward_layers(self.q_layers, q_features)
        return self.q_output(q_features)


def _select_current_observation(
    observation: torch.Tensor,
    temporal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if observation.dim() == 2:
        return observation
    if observation.dim() != 3:
        raise ValueError(
            "observation must have shape [batch, obs_dim] or "
            f"[batch, temporal_steps, obs_dim], got {tuple(observation.shape)}"
        )

    if temporal_mask is None:
        return observation[:, -1]

    if temporal_mask.shape != observation.shape[:2]:
        raise ValueError(
            f"temporal_mask must have shape {tuple(observation.shape[:2])}, "
            f"got {tuple(temporal_mask.shape)}"
        )
    valid_steps = temporal_mask.to(device=observation.device, dtype=torch.bool)
    sequence_lengths = valid_steps.sum(dim=1)
    safe_lengths = sequence_lengths.clamp_min(1)
    last_indices = safe_lengths - 1
    batch_indices = torch.arange(observation.shape[0], device=observation.device)
    current = observation[batch_indices, last_indices]
    valid_sequences = sequence_lengths.gt(0).unsqueeze(-1).to(observation.dtype)
    return current * valid_sequences


class _CentralizedMLPQBranch(nn.Module):
    """Lightweight centralized Q estimator over raw current observations/actions."""

    def __init__(
        self,
        dim_info: Mapping[str, Sequence[int]],
        *,
        hidden_dim: int,
        num_observation_layers: int,
    ):
        super().__init__()
        if not dim_info:
            raise ValueError("dim_info cannot be empty")
        self.agent_ids = list(dim_info.keys())
        self.dim_info = {
            agent_id: (int(dims[0]), int(dims[1]))
            for agent_id, dims in dim_info.items()
        }
        input_dim = sum(obs_dim + action_dim for obs_dim, action_dim in self.dim_info.values())
        self.q_layers = _mlp(input_dim, hidden_dim, num_observation_layers)
        self.q_output = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        observations: Sequence[torch.Tensor],
        actions: Sequence[torch.Tensor],
        agent_mask: torch.Tensor | None = None,
        temporal_masks: Sequence[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        if len(observations) != len(actions):
            raise ValueError("observation and action agent counts must match")
        if len(observations) != len(self.agent_ids):
            raise ValueError("agent count must match dim_info")

        if temporal_masks is None:
            temporal_masks = [None] * len(observations)
        elif len(temporal_masks) != len(observations):
            raise ValueError("temporal mask agent count must match observations")

        batch_size = actions[0].shape[0]
        if agent_mask is not None:
            if agent_mask.shape != (batch_size, len(observations)):
                raise ValueError(
                    f"agent_mask must have shape {(batch_size, len(observations))}, "
                    f"got {tuple(agent_mask.shape)}"
                )
            agent_mask = agent_mask.to(device=actions[0].device, dtype=torch.float32)

        features = []
        for index, (observation, action, temporal_mask) in enumerate(
            zip(observations, actions, temporal_masks)
        ):
            current_observation = _select_current_observation(
                observation,
                temporal_mask=temporal_mask,
            )
            current_observation = current_observation.reshape(batch_size, -1)
            action = action.reshape(batch_size, -1)
            if agent_mask is not None:
                mask = agent_mask[:, index].unsqueeze(-1)
                current_observation = current_observation * mask
                action = action * mask
            features.extend([current_observation, action])

        q_features = torch.cat(features, dim=-1)
        q_features = _forward_layers(self.q_layers, q_features)
        return self.q_output(q_features)


def _select_current_observation(
    observation: torch.Tensor,
    temporal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if observation.dim() == 2:
        return observation
    if observation.dim() != 3:
        raise ValueError(
            "observation must have shape [batch, obs_dim] or "
            f"[batch, temporal_steps, obs_dim], got {tuple(observation.shape)}"
        )

    if temporal_mask is None:
        return observation[:, -1]

    if temporal_mask.shape != observation.shape[:2]:
        raise ValueError(
            f"temporal_mask must have shape {tuple(observation.shape[:2])}, "
            f"got {tuple(temporal_mask.shape)}"
        )
    valid_steps = temporal_mask.to(device=observation.device, dtype=torch.bool)
    sequence_lengths = valid_steps.sum(dim=1)
    safe_lengths = sequence_lengths.clamp_min(1)
    last_indices = safe_lengths - 1
    batch_indices = torch.arange(observation.shape[0], device=observation.device)
    current = observation[batch_indices, last_indices]
    valid_sequences = sequence_lengths.gt(0).unsqueeze(-1).to(observation.dtype)
    return current * valid_sequences


class _CentralizedMLPQBranch(nn.Module):
    """Lightweight centralized Q estimator over raw current observations/actions."""

    def __init__(
        self,
        dim_info: Mapping[str, Sequence[int]],
        *,
        hidden_dim: int,
        num_observation_layers: int,
    ):
        super().__init__()
        if not dim_info:
            raise ValueError("dim_info cannot be empty")
        self.agent_ids = list(dim_info.keys())
        self.dim_info = {
            agent_id: (int(dims[0]), int(dims[1]))
            for agent_id, dims in dim_info.items()
        }
        input_dim = sum(obs_dim + action_dim for obs_dim, action_dim in self.dim_info.values())
        self.q_layers = _mlp(input_dim, hidden_dim, num_observation_layers)
        self.q_output = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        observations: Sequence[torch.Tensor],
        actions: Sequence[torch.Tensor],
        agent_mask: torch.Tensor | None = None,
        temporal_masks: Sequence[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        if len(observations) != len(actions):
            raise ValueError("observation and action agent counts must match")
        if len(observations) != len(self.agent_ids):
            raise ValueError("agent count must match dim_info")

        if temporal_masks is None:
            temporal_masks = [None] * len(observations)
        elif len(temporal_masks) != len(observations):
            raise ValueError("temporal mask agent count must match observations")

        batch_size = actions[0].shape[0]
        if agent_mask is not None:
            if agent_mask.shape != (batch_size, len(observations)):
                raise ValueError(
                    f"agent_mask must have shape {(batch_size, len(observations))}, "
                    f"got {tuple(agent_mask.shape)}"
                )
            agent_mask = agent_mask.to(device=actions[0].device, dtype=torch.float32)

        features = []
        for index, (observation, action, temporal_mask) in enumerate(
            zip(observations, actions, temporal_masks)
        ):
            current_observation = _select_current_observation(
                observation,
                temporal_mask=temporal_mask,
            )
            current_observation = current_observation.reshape(batch_size, -1)
            action = action.reshape(batch_size, -1)
            if agent_mask is not None:
                mask = agent_mask[:, index].unsqueeze(-1)
                current_observation = current_observation * mask
                action = action * mask
            features.extend([current_observation, action])

        q_features = torch.cat(features, dim=-1)
        q_features = _forward_layers(self.q_layers, q_features)
        return self.q_output(q_features)


class MASACCritic(nn.Module):
    """Scalable centralized double-Q critic without graph message passing."""

    def __init__(
        self,
        dim_info: Mapping[str, Sequence[int]],
        focal_agent_id: str | None = None,
        *,
        sensor_observation_dim: int | None = None,
        extra_observation_dim: int = 0,
        ally_feature_dim: int = 0,
        sensor_output_dim: int = 128,
        ally_output_dim: int = 64,
        sensor_hidden_dim: int = 128,
        ally_hidden_dim: int = 128,
        hidden_dim: int = 256,
        num_sensor_layers: int = 2,
        num_ally_layers: int = 2,
        num_observation_layers: int = 2,
        ally_pooling: str = "mean_max",
        sensor_azimuth_bins: int = 24,
        sensor_elevation_bins: int = 9,
        sensor_elevation_range_deg: tuple[float, float] = (-80.0, 80.0),
        sensor_include_previous_scan: bool = True,
        agent_pooling: str = "mean_max",
        critic_encoder: str = "attention",
    ):
        super().__init__()
        if not dim_info:
            raise ValueError("dim_info cannot be empty")

        self.agent_ids = list(dim_info.keys())
        self.dim_info = {
            agent_id: (int(dims[0]), int(dims[1]))
            for agent_id, dims in dim_info.items()
        }
        obs_dims = {dims[0] for dims in self.dim_info.values()}
        action_dims = {dims[1] for dims in self.dim_info.values()}
        if len(obs_dims) != 1 or len(action_dims) != 1:
            raise ValueError(
                "set-based MASAC critic currently requires homogeneous agents"
            )

        self.focal_agent_id = focal_agent_id or self.agent_ids[0]
        if self.focal_agent_id not in self.dim_info:
            raise KeyError(f"unknown focal agent id: {self.focal_agent_id}")
        self.critic_encoder = str(critic_encoder)
        if self.critic_encoder not in {"attention", "mlp"}:
            raise ValueError("critic_encoder must be 'attention' or 'mlp'")
        focal_agent_index = self.agent_ids.index(self.focal_agent_id)
        obs_dim = next(iter(obs_dims))
        action_dim = next(iter(action_dims))

        if self.critic_encoder == "attention":
            branch_kwargs = dict(
                obs_dim=obs_dim,
                action_dim=action_dim,
                focal_agent_index=focal_agent_index,
                sensor_observation_dim=sensor_observation_dim,
                extra_observation_dim=extra_observation_dim,
                ally_feature_dim=ally_feature_dim,
                sensor_output_dim=sensor_output_dim,
                ally_output_dim=ally_output_dim,
                sensor_hidden_dim=sensor_hidden_dim,
                ally_hidden_dim=ally_hidden_dim,
                hidden_dim=hidden_dim,
                num_sensor_layers=num_sensor_layers,
                num_ally_layers=num_ally_layers,
                num_observation_layers=num_observation_layers,
                ally_pooling=ally_pooling,
                sensor_azimuth_bins=sensor_azimuth_bins,
                sensor_elevation_bins=sensor_elevation_bins,
                sensor_elevation_range_deg=sensor_elevation_range_deg,
                sensor_include_previous_scan=sensor_include_previous_scan,
                agent_pooling=agent_pooling,
            )
            self.q1 = _CentralizedQBranch(**branch_kwargs)
            self.q2 = _CentralizedQBranch(**branch_kwargs)
        else:
            branch_kwargs = dict(
                dim_info=self.dim_info,
                hidden_dim=hidden_dim,
                num_observation_layers=num_observation_layers,
            )
            self.q1 = _CentralizedMLPQBranch(**branch_kwargs)
            self.q2 = _CentralizedMLPQBranch(**branch_kwargs)

    def _ordered_inputs(
        self,
        obs: TensorGroup,
        action: TensorGroup,
        temporal_masks: TensorGroup | None = None,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor | None] | None,
    ]:
        observations = _ordered_tensor_list(obs, self.agent_ids)
        actions = _ordered_tensor_list(action, self.agent_ids)
        ordered_temporal_masks = None
        if temporal_masks is not None:
            ordered_temporal_masks = _ordered_tensor_list(
                temporal_masks,
                self.agent_ids,
            )
        return observations, actions, ordered_temporal_masks

    def forward_q1(
        self,
        obs: TensorGroup,
        action: TensorGroup,
        agent_mask: torch.Tensor | None = None,
        temporal_masks: TensorGroup | None = None,
    ) -> torch.Tensor:
        observations, actions, ordered_temporal_masks = self._ordered_inputs(
            obs,
            action,
            temporal_masks,
        )
        return self.q1(observations, actions, agent_mask, ordered_temporal_masks)

    def forward_q2(
        self,
        obs: TensorGroup,
        action: TensorGroup,
        agent_mask: torch.Tensor | None = None,
        temporal_masks: TensorGroup | None = None,
    ) -> torch.Tensor:
        observations, actions, ordered_temporal_masks = self._ordered_inputs(
            obs,
            action,
            temporal_masks,
        )
        return self.q2(observations, actions, agent_mask, ordered_temporal_masks)

    def forward(
        self,
        obs: TensorGroup,
        action: TensorGroup,
        agent_mask: torch.Tensor | None = None,
        temporal_masks: TensorGroup | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observations, actions, ordered_temporal_masks = self._ordered_inputs(
            obs,
            action,
            temporal_masks,
        )
        return (
            self.q1(observations, actions, agent_mask, ordered_temporal_masks),
            self.q2(observations, actions, agent_mask, ordered_temporal_masks),
        )


class MASACAgentNetworks:
    """Online and target networks owned by one MASAC agent."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        dim_info: Mapping[str, Sequence[int]],
        focal_agent_id: str | None = None,
        device: torch.device | str | None = None,
        actor_kwargs: Mapping | None = None,
        critic_kwargs: Mapping | None = None,
    ):
        self.actor = MASACActor(obs_dim, action_dim, **dict(actor_kwargs or {}))
        self.critic = MASACCritic(
            dim_info,
            focal_agent_id=focal_agent_id,
            **dict(critic_kwargs or {}),
        )
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        if device is not None:
            self.to(device)

    def to(self, device: torch.device | str) -> "MASACAgentNetworks":
        self.actor.to(device)
        self.critic.to(device)
        self.actor_target.to(device)
        self.critic_target.to(device)
        return self


Actor = MASACActor
Critic = MASACCritic


__all__ = [
    "Actor",
    "AllyObservationEncoder",
    "Critic",
    "MASACActor",
    "MASACCritic",
    "MASACAgentNetworks",
    "MASACObservationEncoder",
    "ObservationEncoder",
    "_forward_layers",
    "_mlp",
    "_space_dim",
]
