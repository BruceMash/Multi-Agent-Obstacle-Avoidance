import numpy as np
import torch
from pathlib import Path
import sys

ALGO_ROOT = Path(__file__).resolve().parents[1] / "Multi-agent_Algo_lib"
if str(ALGO_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGO_ROOT))

from Controller.dmp_flow import compute_dmp_drives
from Controller.dmp_rl import DMPConfig, SecondOrderDMPController
from Environment.multi_agent_dmp_env import MultiAgentDMPEnv, MultiAgentEnvConfig
from net.masac import MASACActor


def test_actor_tanh_mapping_and_slice_order_are_exact():
    actor = MASACActor(
        obs_dim=11,
        action_dim=6,
        sensor_observation_dim=11,
        sensor_azimuth_bins=2,
        sensor_elevation_bins=2,
        sensor_include_previous_scan=False,
        sensor_hidden_dim=16,
        sensor_output_dim=16,
        hidden_dim=16,
        action_low=(-10.0, -10.0, -10.0, -1.0, -1.0, -1.0),
        action_high=(10.0, 10.0, 10.0, 1.0, 1.0, 1.0),
    )
    with torch.no_grad():
        actor.forcing_mu.bias.copy_(torch.tensor([0.0, 0.5, -0.5]))
        actor.goal_offset_mu.bias.copy_(torch.tensor([0.25, -0.25, 0.0]))

    observation = torch.zeros(1, 1, 11)
    action, log_prob, diagnostics = actor(
        observation,
        deterministic=True,
        return_diagnostics=True,
    )
    expected = torch.cat(
        [
            10.0 * torch.tanh(torch.tensor([0.0, 0.5, -0.5])),
            torch.tanh(torch.tensor([0.25, -0.25, 0.0])),
        ]
    ).reshape(1, 6)
    assert torch.allclose(action, expected, atol=1.0e-6)
    assert torch.allclose(diagnostics["post_tanh_action"], torch.tanh(diagnostics["pre_tanh_action"]))
    assert torch.isfinite(log_prob).all()


def test_fixed_dmp_chain_matches_manual_calculation():
    config = DMPConfig(
        dt=0.1,
        dims=3,
        K_alpha=2.0,
        K_beta=0.5,
        tau=2.0,
        forcing_term_min=-3.0,
        forcing_term_max=3.0,
        forcing_gate_kappa=0.5,
        goal_offset_max=0.5,
    )
    controller = SecondOrderDMPController(config)
    controller.reset(np.zeros(3), np.array([2.0, -1.0, 0.5]))
    position = np.array([0.25, -0.25, 0.0])
    velocity = np.array([0.4, -0.2, 0.1])
    raw_action = np.array([4.0, -2.0, 1.0, 0.75, -0.25, 0.1])

    acceleration, info = controller.compute_acceleration(position, velocity, raw_action)
    forcing = np.array([3.0, -2.0, 1.0])
    offset = np.array([0.5, -0.25, 0.1])
    goal_delta = np.array([2.0, -1.0, 0.5]) + offset - position
    spring = config.K_alpha * config.K_beta * goal_delta
    damping = -config.K_alpha * config.tau * velocity
    terminal_distance = np.linalg.norm(np.array([2.0, -1.0, 0.5]) - position)
    gate = np.full(3, np.tanh(config.forcing_gate_kappa * terminal_distance))
    residual = forcing * gate
    expected_preclip_drive = spring + damping + residual
    expected_acceleration = expected_preclip_drive / config.tau**2

    assert np.allclose(info["raw_forcing"], raw_action[:3])
    assert np.allclose(info["forcing"], forcing)
    assert np.allclose(info["goal_offset"], offset)
    assert np.allclose(info["forcing_gate"], gate)
    assert np.isclose(info["terminal_goal_distance"], terminal_distance)
    assert np.allclose(info["spring_drive"], spring)
    assert np.allclose(info["damping_drive"], damping)
    assert np.allclose(info["residual_drive"], residual)
    assert np.allclose(info["closed_loop_drive"], expected_preclip_drive)
    assert np.allclose(acceleration, expected_acceleration)


def test_terminal_distance_gate_is_applied_once_for_numpy_and_torch_batches():
    goal_delta = np.array([[1.0, 0.5], [0.25, 2.0]], dtype=np.float32)
    terminal_distance = np.array([[1.25], [0.4]], dtype=np.float32)
    velocity = np.zeros_like(goal_delta)
    forcing = np.full_like(goal_delta, 2.0)
    expected = forcing * np.tanh(0.75 * terminal_distance)

    _, numpy_residual, _ = compute_dmp_drives(
        goal_delta,
        velocity,
        forcing,
        k_alpha=1.0,
        k_beta=1.0,
        tau=1.0,
        forcing_gate_kappa=0.75,
        forcing_gate_distance=terminal_distance,
    )
    _, torch_residual, _ = compute_dmp_drives(
        torch.from_numpy(goal_delta),
        torch.from_numpy(velocity),
        torch.from_numpy(forcing),
        k_alpha=1.0,
        k_beta=1.0,
        tau=1.0,
        forcing_gate_kappa=0.75,
        forcing_gate_distance=torch.from_numpy(terminal_distance),
    )
    assert np.allclose(numpy_residual, expected)
    assert np.allclose(torch_residual.numpy(), expected)


def test_near_active_waypoint_keeps_forcing_gate_open_when_terminal_is_far():
    config = DMPConfig(dt=0.1, dims=3, forcing_gate_kappa=1.0)
    controller = SecondOrderDMPController(config)
    controller.reset(np.zeros(3), np.array([0.2, 0.0, 0.0]))
    action = np.array([1.0, -2.0, 3.0, 0.0, 0.0, 0.0])

    _, far_info = controller.compute_acceleration(
        np.zeros(3),
        np.zeros(3),
        action,
        terminal_goal=np.array([6.0, 0.0, 0.0]),
    )
    _, near_info = controller.compute_acceleration(
        np.zeros(3),
        np.zeros(3),
        action,
        terminal_goal=np.array([0.1, 0.0, 0.0]),
    )

    assert far_info["terminal_goal_distance"] == 6.0
    assert near_info["terminal_goal_distance"] == 0.1
    np.testing.assert_allclose(
        far_info["forcing_gate"],
        np.full(3, np.tanh(6.0)),
    )
    np.testing.assert_allclose(
        near_info["forcing_gate"],
        np.full(3, np.tanh(0.1)),
    )
    assert np.linalg.norm(far_info["residual_drive"]) > np.linalg.norm(
        near_info["residual_drive"]
    )


def test_environment_reports_acceleration_and_velocity_clipping_separately():
    env = MultiAgentDMPEnv(
        dynamics_config={
            "velocity_clip": (-0.005, 0.005),
            "accelerate_clip": (-0.1, 0.1),
            "time_step": 0.1,
        },
        sensor_config={
            "sensing_radius": 20.0,
            "azimuth_bins": 2,
            "elevation_bins": 2,
            "include_previous_scan": False,
        },
        dmp_config={
            "dt": 0.1,
            "dims": 3,
            "K_alpha": 20.0,
            "K_beta": 5.0,
            "tau": 1.0,
        },
        env_config=MultiAgentEnvConfig(
            num_agents=1,
            randomize_start_goal=False,
            workspace_bounds=((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0)),
            goal_tolerance=0.01,
            min_start_goal_distance=0.0,
        ),
    )
    env.reset(
        seed=3,
        options={
            "starts": np.array([[0.0, 0.0, 0.0]]),
            "goals": np.array([[5.0, 0.0, 0.0]]),
            "static_obstacles": [],
            "dynamic_obstacles": [],
        },
    )
    _, _, _, _, info = env.step(np.zeros((1, 6), dtype=np.float32))
    assert info["acceleration_clip_mask"].shape == (1, 3)
    assert info["velocity_clip_mask"].shape == (1, 3)
    assert bool(info["acceleration_clip_mask"][0, 0])
    assert bool(info["velocity_clip_mask"][0, 0])
    assert np.isclose(info["applied_accelerations"][0, 0], 0.1)
