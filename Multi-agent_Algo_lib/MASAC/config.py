from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MASACNetworkConfig:
    """Architecture parameters shared by MASAC actors and critics."""

    sensor_observation_dim: int | None = 439
    extra_observation_dim: int = 3
    ally_feature_dim: int = 7
    sensor_output_dim: int = 128
    ally_output_dim: int = 64
    hidden_dim: int = 256
    num_sensor_layers: int = 2
    num_ally_layers: int = 2
    num_observation_layers: int = 2
    sensor_azimuth_bins: int = 24
    sensor_elevation_bins: int = 9
    sensor_elevation_range_deg: tuple[float, float] = (-80.0, 80.0)
    ally_pooling: str = "mean_max"
    agent_pooling: str = "mean_max"
    actor_log_std_min: float = -20.0
    actor_log_std_max: float = 2.0
    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None
    temporal_steps: int = 4

    def __post_init__(self) -> None:
        if self.sensor_observation_dim is not None:
            self.sensor_observation_dim = int(self.sensor_observation_dim)
            if self.sensor_observation_dim <= 0:
                raise ValueError("sensor_observation_dim must be positive")

        integer_fields = (
            "extra_observation_dim",
            "ally_feature_dim",
            "sensor_output_dim",
            "ally_output_dim",
            "hidden_dim",
            "num_sensor_layers",
            "num_ally_layers",
            "num_observation_layers",
            "sensor_azimuth_bins",
            "sensor_elevation_bins",
            "temporal_steps",
        )
        for field_name in integer_fields:
            setattr(self, field_name, int(getattr(self, field_name)))

        if self.extra_observation_dim < 0 or self.ally_feature_dim < 0:
            raise ValueError("observation block dimensions must be non-negative")
        if self.sensor_output_dim <= 0 or self.ally_output_dim <= 0:
            raise ValueError("encoder output dimensions must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.sensor_azimuth_bins <= 0 or self.sensor_elevation_bins <= 0:
            raise ValueError("sensor ray bins must be positive")
        if self.temporal_steps <= 0 or self.temporal_steps > 5:
            raise ValueError("temporal_steps must be in [1, 5]")
        if min(
            self.num_sensor_layers,
            self.num_ally_layers,
            self.num_observation_layers,
        ) <= 0:
            raise ValueError("network layer counts must be positive")

        self.sensor_elevation_range_deg = tuple(
            float(value) for value in self.sensor_elevation_range_deg
        )
        if (
            len(self.sensor_elevation_range_deg) != 2
            or self.sensor_elevation_range_deg[0] >= self.sensor_elevation_range_deg[1]
        ):
            raise ValueError("sensor_elevation_range_deg must be increasing")

        valid_pooling = {"mean", "max", "mean_max"}
        if self.ally_pooling not in valid_pooling:
            raise ValueError(f"unsupported ally_pooling: {self.ally_pooling}")
        if self.agent_pooling not in valid_pooling:
            raise ValueError(f"unsupported agent_pooling: {self.agent_pooling}")

        self.actor_log_std_min = float(self.actor_log_std_min)
        self.actor_log_std_max = float(self.actor_log_std_max)
        if self.actor_log_std_min >= self.actor_log_std_max:
            raise ValueError("actor_log_std_min must be smaller than actor_log_std_max")

        if (self.action_low is None) != (self.action_high is None):
            raise ValueError("action_low and action_high must be both set or both None")
        if self.action_low is not None:
            self.action_low = tuple(float(value) for value in self.action_low)
            self.action_high = tuple(float(value) for value in self.action_high)
            if len(self.action_low) != len(self.action_high):
                raise ValueError("action_low and action_high must have the same length")
            if any(low >= high for low, high in zip(self.action_low, self.action_high)):
                raise ValueError("each action_low value must be smaller than action_high")

    def encoder_kwargs(self) -> dict:
        return {
            "sensor_observation_dim": self.sensor_observation_dim,
            "extra_observation_dim": self.extra_observation_dim,
            "ally_feature_dim": self.ally_feature_dim,
            "sensor_output_dim": self.sensor_output_dim,
            "ally_output_dim": self.ally_output_dim,
            "hidden_dim": self.hidden_dim,
            "num_sensor_layers": self.num_sensor_layers,
            "num_ally_layers": self.num_ally_layers,
            "num_observation_layers": self.num_observation_layers,
            "sensor_azimuth_bins": self.sensor_azimuth_bins,
            "sensor_elevation_bins": self.sensor_elevation_bins,
            "sensor_elevation_range_deg": self.sensor_elevation_range_deg,
            "ally_pooling": self.ally_pooling,
        }

    def actor_kwargs(self) -> dict:
        kwargs = self.encoder_kwargs()
        kwargs.update(
            log_std_min=self.actor_log_std_min,
            log_std_max=self.actor_log_std_max,
            action_low=self.action_low,
            action_high=self.action_high,
        )
        return kwargs

    def critic_kwargs(self) -> dict:
        kwargs = self.encoder_kwargs()
        kwargs["agent_pooling"] = self.agent_pooling
        return kwargs
