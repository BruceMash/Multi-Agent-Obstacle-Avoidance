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

from scripts.evaluate_frozen_policy_waypoint_guidance import (
    PriorityHoldState,
    build_active_goal_observations,
    point_inside_guidance_bounds,
    predict_constant_velocity_conflicts,
    predict_actions_without_postprocessing,
    segment_is_clear,
    set_dmp_active_goal_preserve_phase,
    update_priority_hold_targets,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (
    build_single_distribution_multi_config,
)
from scripts.evaluate_single_policy_multi_agent import SinglePolicyMultiAgentEnv


def _make_env() -> SinglePolicyMultiAgentEnv:
    config = build_single_distribution_multi_config(num_agents=3, max_steps=20)
    env = SinglePolicyMultiAgentEnv(
        **config.build_core_env_kwargs(),
        observation_mode="peer_spheres",
        peer_radius=0.3,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )
    starts = np.asarray(
        [[0.0, -0.8, -0.3], [0.0, 0.0, 0.3], [0.0, 0.8, -0.3]],
        dtype=float,
    )
    goals = np.asarray(
        [[7.5, -1.0, -0.4], [7.5, 0.0, 0.4], [7.5, 1.0, -0.4]],
        dtype=float,
    )
    env.reset(
        seed=1,
        options={
            "starts": starts,
            "goals": goals,
            "static_obstacles": [],
            "dynamic_obstacles": [],
        },
    )
    return env


def test_policy_action_is_returned_without_postprocessing():
    expected = np.asarray([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)

    class DummyModel:
        def predict(self, observations, deterministic):
            assert deterministic is True
            return expected, None

    actual = predict_actions_without_postprocessing(
        DummyModel(),
        np.zeros((2, 3), dtype=np.float32),
        expected.shape,
    )
    assert actual is expected
    np.testing.assert_array_equal(actual, expected)


def test_active_goal_update_preserves_dmp_phase():
    env = _make_env()
    try:
        dmp = env.dmps[0]
        dmp.phase = 0.37
        set_dmp_active_goal_preserve_phase(dmp, np.asarray([2.0, -0.5, 0.0]))
        assert dmp.phase == 0.37
        np.testing.assert_allclose(dmp.goal, [2.0, -0.5, 0.0])
    finally:
        env.close()


def test_environment_gates_forcing_by_task_terminal_not_active_waypoint():
    env = _make_env()
    try:
        active_goal = env.dynamics[0].p + np.asarray([0.2, 0.0, 0.0])
        set_dmp_active_goal_preserve_phase(env.dmps[0], active_goal)
        position_before = env.dynamics[0].p.copy()
        terminal_distance = np.linalg.norm(env.goals[0] - position_before)
        actions = np.zeros(env.action_shape, dtype=np.float32)
        actions[0, :3] = np.asarray([1.0, 2.0, 3.0])
        _, _, _, _, info = env.step(actions)

        expected_gate = np.tanh(
            env.dmps[0].config.forcing_gate_kappa * terminal_distance
        )
        np.testing.assert_allclose(
            info["forcing_gates"][0],
            np.full(3, expected_gate),
        )
        assert np.isclose(info["terminal_goal_distances"][0], terminal_distance)
        assert terminal_distance > np.linalg.norm(active_goal - position_before)
    finally:
        env.close()


def test_active_goal_observation_uses_real_waypoint_distance():
    env = _make_env()
    try:
        active = np.asarray(env.goals, dtype=float).copy()
        active[0] = env.dynamics[0].p + np.asarray([0.0, 1.0, 0.0])
        observations = build_active_goal_observations(env, active)
        np.testing.assert_allclose(observations[0, 3:6], [0.0, 1.0, 0.0], atol=1e-6)
        assert np.isclose(
            observations[0, 6],
            1.0 / env.sensors[0].goal_distance_clip,
        )
    finally:
        env.close()


def test_boundary_filter_and_segment_check_reject_outside_waypoint():
    env = _make_env()
    try:
        bounds = env.env_config.workspace_bounds
        assert point_inside_guidance_bounds([1.0, 0.0, 0.0], bounds, 0.4)
        assert not point_inside_guidance_bounds([1.0, -2.4, 0.0], bounds, 0.4)
        assert segment_is_clear(
            env,
            0,
            env.dynamics[0].p,
            np.asarray([1.0, -0.8, -0.3]),
            boundary_margin=0.4,
            collision_clearance=0.0,
            samples=8,
        )
        assert not segment_is_clear(
            env,
            0,
            env.dynamics[0].p,
            np.asarray([1.0, -2.4, -0.3]),
            boundary_margin=0.4,
            collision_clearance=0.0,
            samples=8,
        )
    finally:
        env.close()


def test_conflict_predictor_distinguishes_parallel_and_head_on_intents():
    positions = np.asarray([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    parallel_goals = np.asarray([[4.0, -1.0, 0.0], [4.0, 1.0, 0.0]])
    active = np.asarray([True, True])
    assert predict_constant_velocity_conflicts(
        positions,
        parallel_goals,
        active,
        nominal_speed=1.0,
        horizon=3.0,
        separation=0.85,
    ) == []

    head_on_positions = np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    head_on_goals = np.asarray([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    conflicts = predict_constant_velocity_conflicts(
        head_on_positions,
        head_on_goals,
        active,
        nominal_speed=1.0,
        horizon=3.0,
        separation=0.85,
    )
    assert len(conflicts) == 1
    assert conflicts[0][0:2] == (0, 1)
    assert np.isclose(conflicts[0][2], 2.0)
    assert np.isclose(conflicts[0][3], 0.0)


def test_priority_hold_changes_only_lower_priority_active_goal_and_releases():
    env = _make_env()
    try:
        positions = np.asarray(
            [[0.2, 0.0, 0.0], [7.8, 0.0, 0.0], [0.2, -1.8, -0.6]],
            dtype=float,
        )
        goals = np.asarray(
            [[7.8, 0.0, 0.0], [0.2, 0.0, 0.0], [7.8, -1.8, -0.6]],
            dtype=float,
        )
        env.reset(
            seed=7,
            options={
                "starts": positions,
                "goals": goals,
                "static_obstacles": [],
                "dynamic_obstacles": [],
            },
        )
        states = [PriorityHoldState() for _ in range(3)]
        scheduled, hold_mask, diagnostics = update_priority_hold_targets(
            env,
            goals,
            np.asarray([True, True, False]),
            states,
            priority_order=[0, 1, 2],
            nominal_speed=1.0,
            prediction_horizon=5.0,
            conflict_separation=0.85,
            release_clear_steps=1,
            deadlock_steps=10,
            boundary_margin=0.4,
            collision_clearance=0.0,
            segment_samples=8,
            yield_distance=0.9,
            retreat_distance=0.45,
        )
        np.testing.assert_allclose(scheduled[0], goals[0])
        assert not hold_mask[0]
        assert hold_mask[1]
        assert diagnostics["hold_events"] == 1
        assert not np.allclose(scheduled[1], goals[1])
        held_point = scheduled[1].copy()

        released, hold_mask, diagnostics = update_priority_hold_targets(
            env,
            goals,
            np.asarray([False, True, False]),
            states,
            priority_order=[0, 1, 2],
            nominal_speed=1.0,
            prediction_horizon=5.0,
            conflict_separation=0.85,
            release_clear_steps=1,
            deadlock_steps=10,
            boundary_margin=0.4,
            collision_clearance=0.0,
            segment_samples=8,
            yield_distance=0.9,
            retreat_distance=0.45,
        )
        assert np.linalg.norm(held_point - goals[1]) > 1.0e-6
        assert not np.any(hold_mask)
        assert diagnostics["releases"] == 1
        np.testing.assert_allclose(released, goals)
    finally:
        env.close()
