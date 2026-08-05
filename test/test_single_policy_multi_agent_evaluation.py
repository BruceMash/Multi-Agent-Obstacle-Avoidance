from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.evaluate_single_policy_multi_agent import (
    SinglePolicyMultiAgentEnv,
    build_aligned_config,
    build_policy_observations,
)


def _make_env(mode: str) -> SinglePolicyMultiAgentEnv:
    config = build_aligned_config(num_agents=3, max_steps=20)
    return SinglePolicyMultiAgentEnv(
        **config.build_core_env_kwargs(),
        observation_mode=mode,
        peer_radius=0.3,
    )


def _reset_open_scene(env: SinglePolicyMultiAgentEnv):
    starts = np.asarray(
        [[1.0, 2.0, 1.2], [2.0, 2.0, 1.2], [1.0, 3.5, 1.2]],
        dtype=float,
    )
    goals = np.asarray(
        [[8.0, 0.7, 1.2], [8.0, 2.25, 1.2], [8.0, 3.8, 1.2]],
        dtype=float,
    )
    return env.reset(
        seed=123,
        options={
            "starts": starts,
            "goals": goals,
            "static_obstacles": [],
            "dynamic_obstacles": [],
        },
    )


def test_aligned_config_matches_historical_checkpoint_interface():
    config = build_aligned_config(num_agents=3, max_steps=200)
    assert config.sensor_azimuth_bins == 8
    assert config.sensor_elevation_bins == 7
    assert config.sensor_include_previous_scan is True
    assert config.sensing_radius == 4.5
    assert config.k_alpha == 3.0
    assert config.k_beta == 0.8
    assert config.dmp_tau == 2.5
    assert config.max_steps == 200
    assert config.workspace_bounds == ((0.0, 0.0, 0.0), (9.0, 4.5, 2.4))


def test_policy_observation_is_exactly_single_agent_122_features():
    env = _make_env("blind")
    try:
        _reset_open_scene(env)
        observations = build_policy_observations(env)
        assert observations.shape == (3, 122)
        np.testing.assert_allclose(observations[:, -3], 1.0)
        np.testing.assert_allclose(observations[:, -2], 3.0)
        np.testing.assert_allclose(observations[:, -1], 0.8)
    finally:
        env.close()


def test_peer_sphere_mode_registers_only_other_agents_for_each_sensor():
    env = _make_env("peer_spheres")
    try:
        _reset_open_scene(env)
        obstacles = env._sensor_dynamic_obstacles(0)
        assert len(obstacles) == 2
        assert all(obstacle.radius == 0.3 for obstacle in obstacles)
        centers = np.stack([obstacle.center for obstacle in obstacles])
        np.testing.assert_allclose(centers, env._positions()[1:])
    finally:
        env.close()


def test_peer_spheres_change_lidar_but_not_policy_dimension():
    blind = _make_env("blind")
    peer = _make_env("peer_spheres")
    try:
        _reset_open_scene(blind)
        _reset_open_scene(peer)
        blind_scan = blind.latest_sensor_packets[0].current_scan
        peer_scan = peer.latest_sensor_packets[0].current_scan
        assert float(np.min(peer_scan)) < float(np.min(blind_scan))
        assert build_policy_observations(blind).shape == build_policy_observations(peer).shape
        np.testing.assert_allclose(blind._positions(), peer._positions())
        np.testing.assert_allclose(blind.goals, peer.goals)
    finally:
        blind.close()
        peer.close()
