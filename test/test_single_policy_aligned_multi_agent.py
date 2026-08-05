from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
SCRIPTS_ROOT = ALGO_ROOT / "scripts"
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.evaluate_single_policy_aligned_multi_agent import (
    STAGE_SPECS,
    build_paired_transition_rows,
    build_single_distribution_multi_config,
    build_stage_scenario,
    sample_historical_starts_goals,
)
from scripts.evaluate_single_policy_multi_agent import SinglePolicyMultiAgentEnv


def test_single_distribution_config_restores_historical_workspace_and_horizon():
    config = build_single_distribution_multi_config(num_agents=3, max_steps=50)
    assert config.workspace_bounds == ((-0.5, -2.5, -1.2), (8.5, 2.0, 1.2))
    assert config.start_position_bounds == ((0.0, -1.0, -0.4), (0.8, 1.0, 0.4))
    assert config.goal_position_bounds == ((7.2, -1.0, -0.4), (8.0, 1.0, 0.4))
    assert config.max_steps == 50


def test_single_semantics_exclude_boundary_lidar_and_boundary_termination():
    config = build_single_distribution_multi_config(num_agents=3, max_steps=50)
    env = SinglePolicyMultiAgentEnv(
        **config.build_core_env_kwargs(),
        observation_mode="blind",
        peer_radius=0.3,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )
    try:
        assert env._sensor_static_obstacles() == []
        env.dynamics[0].p[:] = np.asarray([9.0, 0.0, 0.0])
        assert not np.any(env._compute_boundary_collision_mask())
        assert env._compute_min_boundary_distances()[0] < 0.0
    finally:
        env.close()


def test_matched_and_permuted_scenarios_use_identical_endpoint_sets():
    config = build_single_distribution_multi_config(num_agents=3, max_steps=50)
    starts_a, goals_a = sample_historical_starts_goals(
        config,
        seed=202608050,
        assignment="matched",
    )
    starts_b, goals_b = sample_historical_starts_goals(
        config,
        seed=202608050,
        assignment="permuted",
    )
    np.testing.assert_allclose(starts_a, starts_b)
    np.testing.assert_allclose(np.sort(goals_a, axis=0), np.sort(goals_b, axis=0))
    assert not np.allclose(goals_a, goals_b)


def test_stage_scenarios_respect_multi_agent_endpoint_spacing():
    config = build_single_distribution_multi_config(num_agents=3, max_steps=50)
    for index, stage in enumerate(STAGE_SPECS[:-1]):
        options = build_stage_scenario(config, stage, seed=202608050 + index)
        starts = np.asarray(options["starts"])
        goals = np.asarray(options["goals"])
        start_distances = np.linalg.norm(starts[:, None] - starts[None, :], axis=-1)
        goal_distances = np.linalg.norm(goals[:, None] - goals[None, :], axis=-1)
        upper = np.triu_indices(3, k=1)
        assert np.min(start_distances[upper]) >= 0.6
        assert np.min(goal_distances[upper]) >= 1.2
        assert np.all(np.linalg.norm(goals - starts, axis=1) >= 6.0)


def test_stages_a_to_d_are_paired_except_for_the_intended_factor():
    config = build_single_distribution_multi_config(num_agents=3, max_steps=50)
    seed = 202608050
    options = [build_stage_scenario(config, stage, seed=seed) for stage in STAGE_SPECS[:4]]
    for candidate in options[1:]:
        np.testing.assert_allclose(candidate["starts"], options[0]["starts"])
        np.testing.assert_allclose(
            np.sort(candidate["goals"], axis=0),
            np.sort(options[0]["goals"], axis=0),
        )
    np.testing.assert_allclose(options[1]["goals"], options[2]["goals"])
    assert not np.allclose(options[2]["goals"], options[3]["goals"])
    for obstacle_group in ("static_obstacles", "dynamic_obstacles"):
        centers_b = np.stack([obstacle.center for obstacle in options[1][obstacle_group]])
        centers_c = np.stack([obstacle.center for obstacle in options[2][obstacle_group]])
        centers_d = np.stack([obstacle.center for obstacle in options[3][obstacle_group]])
        np.testing.assert_allclose(centers_b, centers_c)
        np.testing.assert_allclose(centers_b, centers_d)


def test_paired_transition_counting():
    rows = [
        {"scenario": stage, "seed": seed, "team_success": success, "inter_agent_collision": collision}
        for stage, values in (
            ("A_parallel_open", ((1, True, False), (2, True, True))),
            ("B_parallel_training_obstacles", ((1, True, False), (2, False, False))),
            ("C_parallel_peer_spheres", ((1, False, True), (2, True, False))),
            ("D_permuted_peer_spheres", ((1, False, True), (2, False, True))),
        )
        for seed, success, collision in values
    ]
    transitions = build_paired_transition_rows(rows)
    assert transitions[0]["both_success"] == 1
    assert transitions[0]["source_success_only"] == 1
    assert transitions[1]["source_success_only"] == 1
    assert transitions[1]["target_success_only"] == 1
    assert transitions[2]["source_success_only"] == 1
