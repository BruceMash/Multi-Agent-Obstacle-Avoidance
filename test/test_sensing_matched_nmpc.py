from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from Entity.KinematicModel import propagate_point_mass
from planning.sensing_matched_nmpc import (
    NMPCStyleConfig,
    SensingMatchedNMPCStyle,
    exact_point_mass_rollout,
)


@dataclass
class _Packet:
    current_scan: np.ndarray
    previous_scan: np.ndarray
    min_clearance: float = 4.5


@dataclass
class _Sensor:
    ray_directions: np.ndarray
    sensing_radius: float = 4.5


@dataclass
class _Dynamic:
    p: np.ndarray
    v: np.ndarray
    dt: float = 0.1
    accelerate_min: float = -4.0
    accelerate_max: float = 4.0
    velocity_min: float = -2.0
    velocity_max: float = 2.0


class _GuardEnv:
    def __init__(self, num_agents: int = 3, hit_distance: float | None = None) -> None:
        self.num_agents = num_agents
        self.dynamics = [
            _Dynamic(
                p=np.array([0.0, (index - 1) * 1.5, 0.0]),
                v=np.zeros(3),
            )
            for index in range(num_agents)
        ]
        self.goals = np.asarray(
            [[6.0, (index - 1) * 1.5, 0.0] for index in range(num_agents)], dtype=float
        )
        directions = np.zeros((8, 7, 3), dtype=float)
        directions[..., 0] = 1.0
        directions[..., 1] = np.linspace(-0.6, 0.6, 8)[:, None]
        directions /= np.linalg.norm(directions, axis=2, keepdims=True)
        scan = np.ones((8, 7), dtype=float)
        if hit_distance is not None:
            scan[4, 3] = float(hit_distance) / 4.5
        self.sensors = [_Sensor(directions.copy()) for _ in range(num_agents)]
        self.latest_sensor_packets = [
            _Packet(scan.copy(), scan.copy()) for _ in range(num_agents)
        ]
        self.success_rewarded_mask = np.zeros(num_agents, dtype=bool)

    def _positions(self) -> np.ndarray:
        return np.stack([item.p for item in self.dynamics])

    def _velocities(self) -> np.ndarray:
        return np.stack([item.v for item in self.dynamics])

    @property
    def static_obstacles(self):
        raise AssertionError("hidden static geometry read")

    @property
    def dynamic_obstacles(self):
        raise AssertionError("hidden dynamic state read")


def _config(**overrides) -> NMPCStyleConfig:
    values = dict(
        horizon_steps=6,
        iterations=8,
        goal_weight=8.0,
        control_weight=0.05,
        smoothness_weight=0.1,
        safety_weight=30.0,
    )
    values.update(overrides)
    return NMPCStyleConfig(**values)


def test_exact_dynamics_matches_repository_point_mass() -> None:
    p0 = np.array([0.2, -0.3, 0.5])
    v0 = np.array([1.8, -1.9, 0.1])
    u = np.array([4.8, -4.7, 1.2])
    torch_p, torch_v, torch_a = exact_point_mass_rollout(
        torch.as_tensor(p0[None], dtype=torch.float64),
        torch.as_tensor(v0[None], dtype=torch.float64),
        torch.as_tensor(u[None, None], dtype=torch.float64),
        dt=0.1,
        acceleration_min=torch.full((1, 3), -4.0, dtype=torch.float64),
        acceleration_max=torch.full((1, 3), 4.0, dtype=torch.float64),
        velocity_min=torch.full((1, 3), -2.0, dtype=torch.float64),
        velocity_max=torch.full((1, 3), 2.0, dtype=torch.float64),
    )
    expected = propagate_point_mass(
        position=p0,
        velocity=v0,
        acceleration=u,
        dt=0.1,
        acceleration_min=-4.0,
        acceleration_max=4.0,
        velocity_min=-2.0,
        velocity_max=2.0,
    )
    assert np.array_equal(torch_p.numpy()[0, 0], expected["position"])
    assert np.array_equal(torch_v.numpy()[0, 0], expected["velocity"])
    assert np.array_equal(torch_a.numpy()[0, 0], expected["applied_acceleration"])


def test_zero_control_preserves_zero_state() -> None:
    zeros = torch.zeros((1, 3), dtype=torch.float64)
    positions, velocities, accelerations = exact_point_mass_rollout(
        zeros,
        zeros,
        torch.zeros((1, 4, 3), dtype=torch.float64),
        dt=0.1,
        acceleration_min=torch.full((1, 3), -4.0, dtype=torch.float64),
        acceleration_max=torch.full((1, 3), 4.0, dtype=torch.float64),
        velocity_min=torch.full((1, 3), -2.0, dtype=torch.float64),
        velocity_max=torch.full((1, 3), 2.0, dtype=torch.float64),
    )
    assert torch.count_nonzero(positions) == 0
    assert torch.count_nonzero(velocities) == 0
    assert torch.count_nonzero(accelerations) == 0


def test_constant_acceleration_sequence() -> None:
    zeros = torch.zeros((1, 3), dtype=torch.float64)
    controls = torch.zeros((1, 2, 3), dtype=torch.float64)
    controls[..., 0] = 1.0
    positions, velocities, _ = exact_point_mass_rollout(
        zeros,
        zeros,
        controls,
        dt=0.1,
        acceleration_min=torch.full((1, 3), -4.0, dtype=torch.float64),
        acceleration_max=torch.full((1, 3), 4.0, dtype=torch.float64),
        velocity_min=torch.full((1, 3), -2.0, dtype=torch.float64),
        velocity_max=torch.full((1, 3), 2.0, dtype=torch.float64),
    )
    assert np.allclose(velocities.numpy()[0, :, 0], [0.1, 0.2])
    assert np.allclose(positions.numpy()[0, :, 0], [0.005, 0.02])


def test_velocity_and_acceleration_limits_are_hard() -> None:
    position = torch.zeros((1, 3), dtype=torch.float64)
    velocity = torch.full((1, 3), 1.95, dtype=torch.float64)
    controls = torch.full((1, 2, 3), 99.0, dtype=torch.float64)
    _, velocities, accelerations = exact_point_mass_rollout(
        position,
        velocity,
        controls,
        dt=0.1,
        acceleration_min=torch.full((1, 3), -4.0, dtype=torch.float64),
        acceleration_max=torch.full((1, 3), 4.0, dtype=torch.float64),
        velocity_min=torch.full((1, 3), -2.0, dtype=torch.float64),
        velocity_max=torch.full((1, 3), 2.0, dtype=torch.float64),
    )
    assert torch.max(accelerations) == 4.0
    assert torch.max(velocities) == 2.0


def test_limits_and_finite_output() -> None:
    acceleration, info = SensingMatchedNMPCStyle(_config()).plan(_GuardEnv())
    assert acceleration.shape == (3, 3)
    assert np.all(np.isfinite(acceleration))
    assert np.all(acceleration >= -4.0) and np.all(acceleration <= 4.0)
    assert not info["fallback_used"]


def test_deterministic_fresh_controller() -> None:
    first, first_info = SensingMatchedNMPCStyle(_config()).plan(_GuardEnv())
    second, second_info = SensingMatchedNMPCStyle(_config()).plan(_GuardEnv())
    assert np.array_equal(first, second)
    assert first_info["objective_final"] == second_info["objective_final"]


def test_warm_start_is_shifted_and_reported() -> None:
    controller = SensingMatchedNMPCStyle(_config())
    env = _GuardEnv()
    controller.plan(env)
    _, info = controller.plan(env)
    assert info["warm_started"]


def test_empty_scan_nominal_goal_command() -> None:
    acceleration, info = SensingMatchedNMPCStyle(_config()).plan(_GuardEnv())
    assert np.all(acceleration[:, 0] > 0.0)
    assert info["visible_hit_count"] == 0
    assert info["objective_components_unweighted"]["safety"] == 0.0


def test_goal_direction_changes_with_terminal_goal() -> None:
    forward_env = _GuardEnv()
    reverse_env = _GuardEnv()
    reverse_env.goals[:, 0] = -6.0
    forward, _ = SensingMatchedNMPCStyle(_config()).plan(forward_env)
    reverse, _ = SensingMatchedNMPCStyle(_config()).plan(reverse_env)
    assert np.all(forward[:, 0] > 0.0)
    assert np.all(reverse[:, 0] < 0.0)


def test_single_visible_surface_activates_soft_safety_cost() -> None:
    _, clear = SensingMatchedNMPCStyle(_config()).plan(_GuardEnv())
    _, blocked = SensingMatchedNMPCStyle(_config()).plan(_GuardEnv(hit_distance=0.35))
    assert blocked["visible_hit_count"] == 3
    assert blocked["objective_components_unweighted"]["safety"] > clear["objective_components_unweighted"]["safety"]


def test_simple_visible_detour_generates_transverse_control() -> None:
    clear_acceleration, _ = SensingMatchedNMPCStyle(_config()).plan(_GuardEnv())
    blocked_acceleration, _ = SensingMatchedNMPCStyle(_config()).plan(
        _GuardEnv(hit_distance=0.35)
    )
    assert np.all(np.abs(blocked_acceleration[:, 1]) > np.abs(clear_acceleration[:, 1]))


def test_previous_scan_does_not_create_tracker_or_change_control() -> None:
    first_env = _GuardEnv(hit_distance=1.0)
    second_env = _GuardEnv(hit_distance=1.0)
    second_env.latest_sensor_packets[0].previous_scan[:] = 0.2
    first, _ = SensingMatchedNMPCStyle(_config()).plan(first_env)
    second, _ = SensingMatchedNMPCStyle(_config()).plan(second_env)
    assert np.array_equal(first, second)


def test_hidden_geometry_properties_are_never_read() -> None:
    SensingMatchedNMPCStyle(_config()).plan(_GuardEnv(hit_distance=0.8))


def test_two_agent_shape_and_determinism() -> None:
    env = _GuardEnv(num_agents=2, hit_distance=0.7)
    first, _ = SensingMatchedNMPCStyle(_config(), num_agents=2).plan(env)
    second, _ = SensingMatchedNMPCStyle(_config(), num_agents=2).plan(env)
    assert first.shape == (2, 3)
    assert np.array_equal(first, second)


def test_nonfinite_solver_state_uses_frozen_fallback() -> None:
    env = _GuardEnv()
    controller = SensingMatchedNMPCStyle(_config())
    controller._previous_controls[:] = 0.25
    env.goals[0, 0] = np.nan
    acceleration, info = controller.plan(env)
    assert info["fallback_used"]
    assert np.allclose(acceleration, 0.25)
