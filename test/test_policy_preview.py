from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
SCRIPTS_ROOT = ALGO_ROOT / "scripts"
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Environment.frozen_sac_dmp_execution import freeze_policy
from Entity.static_obstacles import StaticSphereObstacle
from experiment_config import EXPERIMENT_CONFIG
from planning.policy_preview import (
    CLEARANCE_SOURCE,
    OBSERVATION_MODEL,
    PreviewInitialState,
    PreviewLocalContext,
    adapt_candidate_proposals,
    build_preview_inputs_from_env,
    point_to_segment_distance,
    preview_candidate,
    preview_candidates,
)
from runner_sac import build_env as build_single_env
from runner_sac import build_model as build_single_model
from runner_sac import load_checkpoint
from scripts.evaluate_frozen_policy_waypoint_guidance import build_active_goal_observations
from scripts.evaluate_single_policy_aligned_multi_agent import build_single_distribution_multi_config
from scripts.evaluate_single_policy_multi_agent import SinglePolicyMultiAgentEnv
from scripts.visualize_policy_preview import plot_policy_preview


CHECKPOINT = REPO_ROOT / "artifacts" / "20260520_201912" / "best_eval_model.pt"


class ObservationDrivenPolicy:
    """A deterministic test policy whose action changes with every observation."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, observation, deterministic):
        assert deterministic is True
        self.calls += 1
        observation = np.asarray(observation, dtype=np.float32)
        action = np.asarray(
            [
                2.0 * observation[3],
                2.0 * observation[4],
                2.0 * observation[5],
                0.2 * observation[0],
                0.2 * observation[1],
                0.2 * observation[2],
            ],
            dtype=np.float32,
        )
        return action, None


def _make_env(num_agents: int = 3) -> SinglePolicyMultiAgentEnv:
    config = build_single_distribution_multi_config(num_agents=num_agents, max_steps=20)
    env = SinglePolicyMultiAgentEnv(
        **config.build_core_env_kwargs(),
        observation_mode="peer_spheres",
        peer_radius=0.3,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )
    starts = np.asarray(
        [[0.2, -1.4, 0.0], [0.2, 0.0, 0.3], [0.2, 1.4, -0.3]][:num_agents],
        dtype=float,
    )
    goals = np.asarray(
        [[7.0, -1.4, 0.0], [7.0, 0.0, 0.3], [7.0, 1.4, -0.3]][:num_agents],
        dtype=float,
    )
    env.reset(
        seed=7,
        options={
            "starts": starts,
            "goals": goals,
            "static_obstacles": [],
            "dynamic_obstacles": [],
        },
    )
    return env


@pytest.fixture(scope="module")
def frozen_checkpoint_policy():
    assert CHECKPOINT.is_file()
    reference_env = build_single_env(config=EXPERIMENT_CONFIG, action_guidance_enabled=False)
    model = build_single_model(reference_env, config=EXPERIMENT_CONFIG, verbose=0)
    load_checkpoint(model, CHECKPOINT)
    freeze_policy(model)
    yield model
    reference_env.close()


def _preview(env, policy, candidate, horizon=4):
    initial, local = build_preview_inputs_from_env(env, 0)
    return preview_candidate(
        initial_state=initial,
        local_context=local,
        candidate_goal=np.asarray(candidate, dtype=float),
        policy=policy,
        horizon=horizon,
        dmp_config=env.dmps[0].config,
        dynamics=env.dynamics[0],
    )


def _snapshot_env(env, model=None):
    return {
        "positions": env._positions().copy(),
        "velocities": env._velocities().copy(),
        "dynamic_obstacles": copy.deepcopy(env.dynamic_obstacles),
        "phases": np.asarray([dmp.phase for dmp in env.dmps]),
        "active_goals": np.stack([dmp.goal.copy() for dmp in env.dmps]),
        "steps": env.steps,
        "action_guidance_step": env.action_guidance_step,
        "sensor_previous": [sensor._previous_scan.copy() for sensor in env.sensors],
        "packets": copy.deepcopy(env.latest_sensor_packets),
        "controller_infos": copy.deepcopy(env.latest_controller_infos),
        "collision": copy.deepcopy(env.latest_collision_info),
        "success": env.success_rewarded_mask.copy(),
        "stagnation_progress": env.stagnation_window_progress.copy(),
        "stagnation_counters": env.stagnation_counters.copy(),
        "replay_size": None if model is None or model.replay_buffer is None else model.replay_buffer.size(),
    }


def _assert_snapshot_equal(before, after):
    for key in ("positions", "velocities", "phases", "active_goals", "success", "stagnation_progress", "stagnation_counters"):
        np.testing.assert_array_equal(before[key], after[key])
    assert before["steps"] == after["steps"]
    assert before["action_guidance_step"] == after["action_guidance_step"]
    assert before["replay_size"] == after["replay_size"]
    for left, right in zip(before["sensor_previous"], after["sensor_previous"], strict=True):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(before["packets"], after["packets"], strict=True):
        np.testing.assert_array_equal(left.observation, right.observation)
        np.testing.assert_array_equal(left.current_scan, right.current_scan)
        np.testing.assert_array_equal(left.previous_scan, right.previous_scan)
    assert repr(before["dynamic_obstacles"]) == repr(after["dynamic_obstacles"])
    assert repr(before["controller_infos"]) == repr(after["controller_infos"])
    assert repr(before["collision"]) == repr(after["collision"])


def test_checkpoint_loads_and_actor_is_frozen(frozen_checkpoint_policy):
    model = frozen_checkpoint_policy
    assert model.actor.training is False
    assert all(parameter.requires_grad is False for parameter in model.actor.parameters())
    assert sum(parameter.numel() for parameter in model.actor.parameters()) == 232076


def test_real_and_preview_first_step_are_consistent(frozen_checkpoint_policy):
    env = _make_env()
    try:
        initial, local = build_preview_inputs_from_env(env, 0)
        active_goals = np.stack([dmp.goal.copy() for dmp in env.dmps])
        observations = build_active_goal_observations(env, active_goals)
        real_actions = np.stack(
            [
                frozen_checkpoint_policy.predict(row, deterministic=True)[0]
                for row in observations
            ]
        )
        preview = preview_candidate(
            initial_state=initial,
            local_context=local,
            candidate_goal=active_goals[0],
            policy=frozen_checkpoint_policy,
            horizon=1,
            dmp_config=env.dmps[0].config,
            dynamics=env.dynamics[0],
        )
        _, _, _, _, info = env.step(real_actions)

        np.testing.assert_allclose(preview.trajectory.observations[0], observations[0], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(preview.trajectory.actions[0], real_actions[0], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(preview.trajectory.commanded_accelerations[0], info["commanded_accelerations"][0], atol=2e-7, rtol=2e-7)
        np.testing.assert_allclose(preview.trajectory.accelerations[0], info["applied_accelerations"][0], atol=2e-7, rtol=2e-7)
        np.testing.assert_allclose(preview.trajectory.positions[1], env.dynamics[0].p, atol=2e-7, rtol=2e-7)
        np.testing.assert_allclose(preview.trajectory.velocities[1], env.dynamics[0].v, atol=2e-7, rtol=2e-7)
        np.testing.assert_allclose(preview.trajectory.phases[1], env.dmps[0].phase, atol=2e-12, rtol=2e-12)
        assert preview.trajectory.controller_infos[0]["terminal_goal_distance"] == pytest.approx(
            np.linalg.norm(initial.task_goal - initial.position)
        )
    finally:
        env.close()


def test_preview_has_no_side_effects(frozen_checkpoint_policy):
    env = _make_env()
    try:
        initial, local = build_preview_inputs_from_env(env, 0)
        candidates = [initial.position + [1.0, y, z] for y, z in ((0.0, 0.0), (0.4, 0.2), (-0.4, -0.2))]
        before = _snapshot_env(env, frozen_checkpoint_policy)
        results = preview_candidates(
            initial_state=initial,
            local_context=local,
            candidates=candidates,
            policy=frozen_checkpoint_policy,
            horizon=3,
            dmp_config=env.dmps[0].config,
            dynamics=env.dynamics[0],
        )
        after = _snapshot_env(env, frozen_checkpoint_policy)
        assert len(results) == 3
        _assert_snapshot_equal(before, after)
    finally:
        env.close()


def test_history_is_rolled_locally_and_candidate_order_is_irrelevant():
    env = _make_env()
    try:
        initial, local = build_preview_inputs_from_env(env, 0)
        a = initial.position + np.asarray([1.0, 0.4, 0.1])
        b = initial.position + np.asarray([1.0, -0.4, -0.1])
        policy_ab = ObservationDrivenPolicy()
        ab = preview_candidates(
            initial_state=initial,
            local_context=local,
            candidates=[a, b],
            policy=policy_ab,
            horizon=4,
            dmp_config=env.dmps[0].config,
            dynamics=env.dynamics[0],
        )
        policy_ba = ObservationDrivenPolicy()
        ba = preview_candidates(
            initial_state=initial,
            local_context=local,
            candidates=[b, a],
            policy=policy_ba,
            horizon=4,
            dmp_config=env.dmps[0].config,
            dynamics=env.dynamics[0],
        )
        policy_b = ObservationDrivenPolicy()
        b_only = preview_candidates(
            initial_state=initial,
            local_context=local,
            candidates=[b],
            policy=policy_b,
            horizon=4,
            dmp_config=env.dmps[0].config,
            dynamics=env.dynamics[0],
        )[0]
        np.testing.assert_allclose(ab[0].trajectory.positions, ba[1].trajectory.positions)
        np.testing.assert_allclose(ab[1].trajectory.positions, ba[0].trajectory.positions)
        np.testing.assert_allclose(ab[1].trajectory.positions, b_only.trajectory.positions)
        np.testing.assert_allclose(ab[1].trajectory.velocities, b_only.trajectory.velocities)
        np.testing.assert_allclose(ab[1].trajectory.actions, b_only.trajectory.actions)
        assert ab[1].task_progress == pytest.approx(b_only.task_progress)
        assert ab[1].min_clearance == pytest.approx(b_only.min_clearance)
        assert ab[1].max_execution_deviation == pytest.approx(
            b_only.max_execution_deviation
        )
        assert ab[1].terminal_speed == pytest.approx(b_only.terminal_speed)
        assert ab[1].metadata == ba[0].metadata == b_only.metadata
        np.testing.assert_array_equal(ab[0].trajectory.previous_scans[1], ab[0].trajectory.current_scans[0])
        np.testing.assert_array_equal(ab[1].trajectory.previous_scans[0], local.previous_scan)
        assert policy_ab.calls == 8
        assert policy_ba.calls == 8
    finally:
        env.close()


def test_preview_is_closed_loop_and_features_are_well_formed():
    env = _make_env()
    try:
        initial, _ = build_preview_inputs_from_env(env, 0)
        result = _preview(env, ObservationDrivenPolicy(), initial.position + [1.2, 0.5, 0.2], horizon=5)
        trajectory = result.trajectory
        assert trajectory.observations.shape == (5, 122)
        assert trajectory.actions.shape == (5, 6)
        assert trajectory.positions.shape == (6, 3)
        assert trajectory.velocities.shape == (6, 3)
        assert trajectory.accelerations.shape == (5, 3)
        assert trajectory.phases.shape == (6,)
        assert not np.array_equal(trajectory.observations[0], trajectory.observations[1])
        assert not np.array_equal(trajectory.actions[0], trajectory.actions[1])
        expected_progress = np.linalg.norm(initial.task_goal - initial.position) - np.linalg.norm(
            initial.task_goal - trajectory.positions[-1]
        )
        assert result.task_progress == pytest.approx(expected_progress)
        assert result.max_execution_deviation >= 0.0
        assert result.terminal_speed == pytest.approx(np.linalg.norm(trajectory.velocities[-1]))
        assert result.performance.policy_calls == 5
        assert result.metadata["requested_horizon_steps"] == 5
        assert result.metadata["effective_horizon_steps"] == 5
        assert result.metadata["effective_horizon_ratio"] == pytest.approx(1.0)
        assert result.metadata["preview_completed"] is True
        assert result.metadata["termination_reason"] == "completed_horizon"
        assert result.metadata["feature_valid_mask"] == {
            "task_progress": True,
            "min_clearance": True,
            "max_execution_deviation": True,
            "terminal_speed": True,
        }
        assert result.metadata["feature_full_horizon_mask"] == result.metadata[
            "feature_valid_mask"
        ]
        assert result.metadata["forcing_gate_distance_source"] == "terminal_task_goal"
        assert result.metadata["boundary_constraint_added"] is False
    finally:
        env.close()


def test_local_context_records_lidar_limit_and_empty_scan_clearance():
    env = _make_env(num_agents=1)
    try:
        initial, local = build_preview_inputs_from_env(env, 0)
        assert local.lidar_hit_source_available is False
        assert local.dynamic_entity_extrapolation is False
        assert local.clearance_source == CLEARANCE_SOURCE
        assert local.observation_model == OBSERVATION_MODEL
        result = _preview(env, ObservationDrivenPolicy(), initial.position + [1.0, 0.0, 0.0], horizon=2)
        assert np.isinf(result.min_clearance)
        assert result.metadata["clearance_is_approximate"] is True
        assert result.metadata["obstacle_clearance_source"] == CLEARANCE_SOURCE
        assert result.metadata["obstacle_clearance_is_approximate"] is True
        assert result.metadata["boundary_clearance_source"] == (
            "not_separately_available_from_untyped_lidar"
        )
        assert result.metadata["boundary_clearance_is_approximate"] is True
        assert result.metadata["clearance_finite_mask"] is False
        assert result.metadata["open_space_flag"] is True
        assert result.metadata["feature_valid_mask"]["min_clearance"] is False
    finally:
        env.close()


def test_candidate_adapter_uses_explicit_consumer_semantics():
    proposals = list(range(4))
    assert adapt_candidate_proposals(proposals) == [0, 1, 2, 3]
    assert adapt_candidate_proposals(proposals, consumer_top_k=10) == [0, 1, 2, 3]
    assert adapt_candidate_proposals(proposals, consumer_top_k=2) == [0, 1]
    with pytest.raises(ValueError):
        adapt_candidate_proposals(proposals, consumer_top_k=0)


def test_unknown_obstacle_outside_lidar_is_not_leaked_into_preview():
    env = _make_env(num_agents=1)
    try:
        hidden = StaticSphereObstacle(
            center=env.dynamics[0].p + np.asarray([20.0, 0.0, 0.0]),
            radius=1.0,
        )
        env.static_obstacles = [hidden]
        env.sensors[0].reset()
        env.latest_sensor_packets[0] = env.sensors[0].sense(
            env.dynamics[0].p,
            env.dynamics[0].v,
            env.goals[0],
            env._sensor_static_obstacles(),
            env._sensor_dynamic_obstacles(0),
        )
        initial, local = build_preview_inputs_from_env(env, 0)
        assert local.visible_surface_points.shape == (0, 3)
        result = _preview(
            env,
            ObservationDrivenPolicy(),
            initial.position + np.asarray([1.0, 0.0, 0.0]),
            horizon=2,
        )
        assert np.isinf(result.min_clearance)
    finally:
        env.close()


def test_degenerate_candidate_and_invalid_inputs_are_handled():
    env = _make_env(num_agents=1)
    try:
        initial, _ = build_preview_inputs_from_env(env, 0)
        result = _preview(env, ObservationDrivenPolicy(), initial.position, horizon=2)
        assert np.isfinite(result.max_execution_deviation)
        assert point_to_segment_distance(initial.position + [0.0, 1.0, 0.0], initial.position, initial.position) == pytest.approx(1.0)
        with pytest.raises(ValueError, match="finite"):
            PreviewInitialState(
                position=[np.nan, 0.0, 0.0],
                velocity=[0.0, 0.0, 0.0],
                acceleration=[0.0, 0.0, 0.0],
                phase=1.0,
                active_goal=[1.0, 0.0, 0.0],
                task_goal=[2.0, 0.0, 0.0],
            )
        with pytest.raises(ValueError, match="positive"):
            _preview(env, ObservationDrivenPolicy(), initial.position, horizon=0)
    finally:
        env.close()


def test_visualization_smoke(tmp_path):
    env = _make_env(num_agents=1)
    try:
        initial, local = build_preview_inputs_from_env(env, 0)
        candidate = initial.position + np.asarray([1.0, 0.2, 0.1])
        result = _preview(env, ObservationDrivenPolicy(), candidate, horizon=2)
        output = plot_policy_preview(
            initial_state=initial,
            local_context=local,
            candidates=[candidate],
            previews=[result],
            output_path=tmp_path / "preview.png",
        )
        assert output.is_file()
        assert output.stat().st_size > 1000
    finally:
        env.close()
