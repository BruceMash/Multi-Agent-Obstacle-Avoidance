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

from Entity.dynamic_obstacles import MovingSphereObstacle
from scripts.evaluate_single_policy_guidance_multi_agent import (
    WaypointState,
    build_direction_proxy_goal,
    build_guided_policy_observations,
    reference_point_has_collision,
    waypoint_refresh_reason,
)
from scripts.evaluate_single_policy_multi_agent import SinglePolicyMultiAgentEnv
from scripts.evaluate_single_policy_aligned_multi_agent import (
    build_single_distribution_multi_config,
)


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


def test_direction_proxy_preserves_task_goal_distance():
    position = np.asarray([0.0, 0.0, 0.0])
    task_goal = np.asarray([7.0, 0.0, 0.0])
    waypoint = np.asarray([0.0, 1.0, 0.0])
    proxy = build_direction_proxy_goal(position, task_goal, waypoint)
    np.testing.assert_allclose(proxy, [0.0, 7.0, 0.0])
    assert np.isclose(np.linalg.norm(proxy - position), np.linalg.norm(task_goal - position))


def test_guided_observation_replaces_direction_but_preserves_distance_feature():
    env = _make_env()
    try:
        task_goals = np.asarray(env.goals, dtype=float).copy()
        proxy_goals = task_goals.copy()
        proxy_goals[0] = env.dynamics[0].p + np.asarray([0.0, 7.5, 0.0])
        observations = build_guided_policy_observations(env, proxy_goals, task_goals)
        np.testing.assert_allclose(observations[0, 3:6], [0.0, 1.0, 0.0], atol=1e-6)
        expected_distance = np.clip(
            np.linalg.norm(task_goals[0] - env.dynamics[0].p)
            / env.sensors[0].goal_distance_clip,
            0.0,
            1.0,
        )
        assert np.isclose(observations[0, 6], expected_distance)
    finally:
        env.close()


def test_occupied_reference_point_triggers_immediate_collision_replan():
    env = _make_env()
    try:
        point = np.asarray([1.0, -0.8, -0.3])
        env.dynamic_obstacles = [
            MovingSphereObstacle(
                center=point.copy(),
                velocity=np.zeros(3),
                radius=0.2,
                safety_margin=0.0,
            )
        ]
        state = WaypointState(point=point.copy(), generated_step=0)
        assert reference_point_has_collision(env, 0, point)
        assert waypoint_refresh_reason(
            env,
            0,
            state,
            step=1,
            reached_tolerance=0.25,
            refresh_steps=5,
            collision_clearance=0.0,
        ) == "collision"
    finally:
        env.close()


def test_refresh_reason_prioritizes_reached_then_periodic_for_safe_point():
    env = _make_env()
    try:
        reached = WaypointState(point=env.dynamics[0].p + [0.1, 0.0, 0.0], generated_step=0)
        assert waypoint_refresh_reason(
            env,
            0,
            reached,
            step=5,
            reached_tolerance=0.25,
            refresh_steps=5,
            collision_clearance=0.0,
        ) == "reached"
        periodic = WaypointState(point=env.dynamics[0].p + [1.0, 0.0, 0.0], generated_step=0)
        assert waypoint_refresh_reason(
            env,
            0,
            periodic,
            step=5,
            reached_tolerance=0.25,
            refresh_steps=5,
            collision_clearance=0.0,
        ) == "periodic"
    finally:
        env.close()
