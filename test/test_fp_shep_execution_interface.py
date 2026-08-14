from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
SCRIPTS_ROOT = ALGO_ROOT / "scripts"
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.candidate_execution_interface import (
    EXECUTION_FEATURE_ORDER,
    ExecutionNormalizationSpec,
    GraphReadyCandidateExecution,
    constant_velocity_conflict_diagnostic,
    graph_ready_candidate_execution,
    normalize_execution_features,
)
from planning.candidate_execution_benchmark import real_candidate_rollout
from planning.policy_preview import (
    CandidatePreview,
    PreviewPerformance,
    PreviewTrajectory,
    build_preview_inputs_from_env,
    preview_candidate,
)
from scripts.analyze_fp_shep_execution_interface import analyze_dataset
from scripts.evaluate_single_policy_aligned_multi_agent import (
    build_single_distribution_multi_config,
)
from scripts.evaluate_single_policy_multi_agent import SinglePolicyMultiAgentEnv


def _candidate_preview(
    *,
    horizon: int = 4,
    progress: float = -0.16,
    clearance: float = 2.25,
    deviation: float = 0.525,
    speed: float = 0.6,
    open_space: bool = False,
) -> CandidatePreview:
    positions = np.column_stack(
        [np.linspace(0.0, 0.4, horizon + 1), np.zeros(horizon + 1), np.zeros(horizon + 1)]
    )
    velocities = np.column_stack(
        [np.linspace(0.0, speed, horizon + 1), np.zeros(horizon + 1), np.zeros(horizon + 1)]
    )
    trajectory = PreviewTrajectory(
        candidate_goal=np.asarray([1.0, 0.0, 0.0]),
        positions=positions,
        velocities=velocities,
        accelerations=np.zeros((horizon, 3)),
        commanded_accelerations=np.zeros((horizon, 3)),
        phases=np.linspace(1.0, 0.5, horizon + 1),
        observations=np.zeros((horizon, 122), dtype=np.float32),
        actions=np.zeros((horizon, 6), dtype=np.float32),
        current_scans=np.ones((horizon, 2, 2), dtype=np.float32),
        previous_scans=np.ones((horizon, 2, 2), dtype=np.float32),
        clearances=np.full(horizon, clearance),
        controller_infos=tuple({} for _ in range(horizon)),
    )
    clearance_finite = bool(np.isfinite(clearance))
    valid = {
        "task_progress": bool(np.isfinite(progress)),
        "min_clearance": clearance_finite,
        "max_execution_deviation": bool(np.isfinite(deviation)),
        "terminal_speed": bool(np.isfinite(speed)),
    }
    return CandidatePreview(
        trajectory=trajectory,
        task_progress=progress,
        min_clearance=clearance,
        max_execution_deviation=deviation,
        terminal_speed=speed,
        performance=PreviewPerformance(0.0, 0.0, 0.0, 0.0, 0.0, horizon),
        metadata={
            "requested_horizon_steps": horizon,
            "effective_horizon_steps": horizon,
            "effective_horizon_ratio": 1.0,
            "preview_completed": True,
            "termination_reason": "completed_horizon",
            "feature_valid_mask": valid,
            "feature_full_horizon_mask": valid.copy(),
            "obstacle_clearance_source": "frozen_lidar_surface_samples",
            "obstacle_clearance_is_approximate": True,
            "boundary_clearance_source": "not_separately_available_from_untyped_lidar",
            "boundary_clearance_is_approximate": True,
            "clearance_finite_mask": clearance_finite,
            "open_space_flag": open_space,
        },
    )


def _execution(**kwargs) -> GraphReadyCandidateExecution:
    return graph_ready_candidate_execution(7, _candidate_preview(**kwargs))


class _ZeroPolicy:
    def predict(self, observation, deterministic):
        assert deterministic is True
        observation = np.asarray(observation)
        shape = (6,) if observation.ndim == 1 else (observation.shape[0], 6)
        return np.zeros(shape, dtype=np.float32), None


def _timeout_env() -> SinglePolicyMultiAgentEnv:
    config = build_single_distribution_multi_config(num_agents=1, max_steps=1)
    env = SinglePolicyMultiAgentEnv(
        **config.build_core_env_kwargs(),
        observation_mode="blind",
        peer_radius=0.3,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )
    env.reset(
        seed=3,
        options={
            "starts": np.asarray([[0.2, 0.0, 0.0]], dtype=float),
            "goals": np.asarray([[7.0, 0.0, 0.0]], dtype=float),
            "static_obstacles": [],
            "dynamic_obstacles": [],
        },
    )
    return env


def test_graph_ready_schema_uses_h1_as_array_index_zero_and_preserves_raw_values():
    preview = _candidate_preview()
    execution = graph_ready_candidate_execution(7, preview)
    assert execution.candidate_id == 7
    assert execution.preview_positions.shape == (4, 3)
    assert execution.preview_velocities.shape == (4, 3)
    np.testing.assert_array_equal(
        execution.preview_positions[0], preview.trajectory.positions[1]
    )
    np.testing.assert_array_equal(
        execution.preview_velocities[0], preview.trajectory.velocities[1]
    )
    np.testing.assert_array_equal(
        execution.raw_feature_vector,
        [-0.16, 2.25, 0.525, 0.6],
    )
    assert execution.requested_horizon_steps == 4
    assert execution.effective_horizon_steps == 4
    assert execution.effective_horizon_ratio == pytest.approx(1.0)
    assert execution.preview_completed is True
    assert execution.termination_reason == "completed_horizon"
    assert execution.obstacle_clearance_is_approximate is True
    assert execution.boundary_clearance_source == (
        "not_separately_available_from_untyped_lidar"
    )


def test_graph_ready_schema_rejects_nonfinite_candidate_position():
    preview = _candidate_preview()
    invalid_trajectory = replace(
        preview.trajectory,
        candidate_goal=np.asarray([np.nan, 0.0, 0.0]),
    )
    with pytest.raises(ValueError, match="candidate_world_position"):
        graph_ready_candidate_execution(
            7,
            replace(preview, trajectory=invalid_trajectory),
        )


def test_normalization_handles_negative_progress_and_zero_speed_without_overwriting_raw():
    execution = _execution(speed=0.0)
    before = execution.raw_feature_vector.copy()
    normalized = normalize_execution_features(execution)
    np.testing.assert_allclose(normalized.values, [-0.5, 0.5, 0.5, 0.0])
    np.testing.assert_array_equal(execution.raw_feature_vector, before)
    assert np.all(np.isfinite(normalized.values))
    assert normalized.valid_mask.tolist() == [True, True, True, True]


def test_open_space_clearance_has_finite_value_and_explicit_semantic_masks():
    execution = _execution(clearance=float("inf"), open_space=True)
    normalized = normalize_execution_features(execution)
    clearance_index = EXECUTION_FEATURE_ORDER.index("min_clearance")
    assert np.isinf(execution.min_clearance_raw)
    assert normalized.values[clearance_index] == pytest.approx(1.0)
    assert normalized.valid_mask[clearance_index] == np.bool_(False)
    assert normalized.clearance_finite_mask is False
    assert normalized.open_space_flag is True
    assert np.all(np.isfinite(normalized.values))


def test_very_small_clearance_and_large_deviation_are_finite_and_clipped_explicitly():
    execution = _execution(clearance=1.0e-6, deviation=5.0)
    normalized = normalize_execution_features(execution)
    assert normalized.values[1] == pytest.approx(1.0e-6 / 4.5)
    assert normalized.values[2] == pytest.approx(1.0)
    assert normalized.clipped_mask.tolist() == [False, False, True, False]
    assert np.all(np.isfinite(normalized.values))


def test_invalid_numeric_feature_uses_mask_instead_of_hidden_zero_semantics():
    preview = _candidate_preview(progress=float("nan"))
    execution = graph_ready_candidate_execution(1, preview)
    normalized = normalize_execution_features(execution)
    assert np.isnan(execution.task_progress_raw)
    assert normalized.values[0] == pytest.approx(0.0)
    assert normalized.valid_mask[0] == np.bool_(False)
    assert np.all(np.isfinite(normalized.values))


def test_truncated_execution_retains_effective_horizon_and_full_horizon_masks():
    full = _candidate_preview()
    effective = 2
    trajectory = replace(
        full.trajectory,
        positions=full.trajectory.positions[: effective + 1],
        velocities=full.trajectory.velocities[: effective + 1],
        accelerations=full.trajectory.accelerations[:effective],
        commanded_accelerations=full.trajectory.commanded_accelerations[:effective],
        phases=full.trajectory.phases[: effective + 1],
        observations=full.trajectory.observations[:effective],
        actions=full.trajectory.actions[:effective],
        current_scans=full.trajectory.current_scans[:effective],
        previous_scans=full.trajectory.previous_scans[:effective],
        clearances=full.trajectory.clearances[:effective],
        controller_infos=full.trajectory.controller_infos[:effective],
    )
    metadata = dict(full.metadata)
    metadata.update(
        requested_horizon_steps=4,
        effective_horizon_steps=effective,
        effective_horizon_ratio=0.5,
        preview_completed=False,
        termination_reason="numerical_guard",
        feature_full_horizon_mask={name: False for name in EXECUTION_FEATURE_ORDER},
    )
    truncated = replace(full, trajectory=trajectory, metadata=metadata)
    execution = graph_ready_candidate_execution(2, truncated)
    normalized = normalize_execution_features(execution)
    assert execution.preview_positions.shape == (2, 3)
    assert execution.requested_horizon_steps == 4
    assert execution.effective_horizon_steps == 2
    assert execution.effective_horizon_ratio == pytest.approx(0.5)
    assert execution.preview_completed is False
    assert execution.termination_reason == "numerical_guard"
    assert not np.any(normalized.full_horizon_mask)


def test_preview_completion_is_independent_of_real_rollout_timeout():
    env = _timeout_env()
    try:
        initial, local = build_preview_inputs_from_env(env, 0)
        candidate = initial.position + np.asarray([1.0, 0.0, 0.0])
        policy = _ZeroPolicy()
        preview = preview_candidate(
            initial_state=initial,
            local_context=local,
            candidate_goal=candidate,
            policy=policy,
            horizon=4,
            dmp_config=env.dmps[0].config,
            dynamics=env.dynamics[0],
        )
        real = real_candidate_rollout(
            initial_env=env,
            agent_index=0,
            candidate_goal=candidate,
            policy=policy,
            horizon=4,
            preview=preview,
        )
        assert preview.metadata["preview_completed"] is True
        assert preview.metadata["effective_horizon_steps"] == 4
        assert preview.metadata["termination_reason"] == "completed_horizon"
        assert real.truncated is True
        assert real.effective_steps == 1
        assert real.trajectory_error_is_full_horizon is False
    finally:
        env.close()


def _diagnostic(candidate, neighbor_position, neighbor_velocity, threshold=0.5, dt=1.0):
    return constant_velocity_conflict_diagnostic(
        candidate_preview_positions=np.asarray(candidate, dtype=float),
        neighbor_current_position=np.asarray(neighbor_position, dtype=float),
        neighbor_current_velocity=np.asarray(neighbor_velocity, dtype=float),
        dt=dt,
        risk_separation_threshold=threshold,
    )


def test_conflict_case_a_far_now_but_future_intersection():
    candidate = [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]
    result = _diagnostic(candidate, [3, 3, 0], [0, -1, 0])
    assert result.minimum_separation == pytest.approx(0.0)
    assert result.time_to_minimum_separation == pytest.approx(3.0)
    assert result.risk_duration == pytest.approx(1.0)


def test_conflict_case_b_close_now_but_rapidly_separating():
    candidate = [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]
    result = _diagnostic(candidate, [0.1, 0, 0], [-1, 0, 0])
    assert result.minimum_separation == pytest.approx(1.9)
    assert result.time_to_minimum_separation == pytest.approx(1.0)
    assert result.risk_duration == pytest.approx(0.0)


def test_conflict_case_c_sustained_parallel_proximity_and_off_by_one_time():
    candidate = [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]
    result = _diagnostic(candidate, [0, 0.4, 0], [1, 0, 0])
    np.testing.assert_allclose(result.per_step_distance, 0.4)
    assert result.minimum_separation == pytest.approx(0.4)
    assert result.time_to_minimum_separation == pytest.approx(1.0)
    assert result.risk_duration == pytest.approx(4.0)


def test_conflict_case_d_transient_crossing_then_separation():
    candidate = [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]
    result = _diagnostic(candidate, [2, -2, 0], [0, 1, 0])
    assert result.minimum_separation == pytest.approx(0.0)
    assert result.time_to_minimum_separation == pytest.approx(2.0)
    assert result.risk_step_count == 1
    assert result.risk_duration == pytest.approx(1.0)


def test_conflict_time_index_zero_corresponds_to_h1_dt():
    result = _diagnostic(
        [[0, 0, 0], [10, 0, 0]],
        [0, 0, 0],
        [0, 0, 0],
        threshold=0.5,
        dt=0.1,
    )
    assert result.time_to_minimum_separation == pytest.approx(0.1)


def test_conflict_risk_duration_uses_strict_safe_distance_boundary():
    result = _diagnostic(
        [[1, 0, 0]],
        [1, 0.6, 0],
        [0, 0, 0],
        threshold=0.6,
        dt=0.1,
    )
    assert result.minimum_separation == pytest.approx(0.6)
    assert result.risk_step_count == 0
    assert result.risk_duration == pytest.approx(0.0)


def test_h4_validation_distribution_audit_is_read_only_and_records_inf_semantics():
    dataset = (
        REPO_ROOT
        / "artifacts"
        / "fp_shep_validation"
        / "20260812_203321"
        / "candidate_level_results.csv"
    )
    before = dataset.stat().st_mtime_ns
    report = analyze_dataset(dataset, horizon=4)
    after = dataset.stat().st_mtime_ns
    assert before == after
    assert report["candidate_count"] == 242
    overall = {
        row["feature"]: row
        for row in report["feature_statistics"]
        if row["scene_type"] == "overall"
    }
    assert overall["min_clearance"]["positive_inf_count"] == 90
    assert overall["task_progress"]["min"] < 0.0
    assert overall["max_execution_deviation"]["min"] >= 0.0
    assert report["audit_findings"]["contains_nan"] is False
    assert ExecutionNormalizationSpec().version == "fp_shep_h4_v1"
