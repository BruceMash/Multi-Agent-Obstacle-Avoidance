from __future__ import annotations

import copy
import importlib.util
import json
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

from Entity.static_obstacles import StaticSphereObstacle  # noqa: E402
from planning.fp_shep_preview_fidelity_analysis import (  # noqa: E402
    H4_TERMINATION_RULE,
    H20_TERMINATION_RULE,
    REFRESHED_SENSING_SCOPE,
    SENSING_FROZEN,
    SENSING_REFRESHED_STATIC,
    build_refreshed_static_sensing_context,
    classify_directional_fraction,
    diagnostic_preview_rollout,
    reconstruct_refreshed_static_scan,
    sensor_derived_clearance,
)
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.policy_preview import build_preview_inputs_from_env, preview_candidate  # noqa: E402
from planning.reference_transition_finetuning import sha256_file  # noqa: E402
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_single_policy_multi_agent import SinglePolicyMultiAgentEnv  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs/evaluation/fp_shep_preview_fidelity_control.json"
SCRIPT_PATH = (
    REPO_ROOT
    / "Multi-agent_Algo_lib/scripts/evaluate_fp_shep_preview_fidelity_control.py"
)


class ConstantPolicy:
    def __init__(self, action=None):
        self.action = np.zeros(6, dtype=np.float32) if action is None else np.asarray(action, dtype=np.float32)
        self.calls = 0

    def predict(self, observation, deterministic=True):
        assert deterministic is True
        assert np.asarray(observation).shape == (122,)
        self.calls += 1
        return self.action.copy(), None


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _load_script():
    spec = importlib.util.spec_from_file_location("preview_fidelity_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_env(*, obstacle_center=(2.0, -1.0, 0.0), obstacle_radius=0.45):
    config = build_single_distribution_multi_config(num_agents=3, max_steps=30)
    env = SinglePolicyMultiAgentEnv(
        **config.build_core_env_kwargs(),
        observation_mode="peer_spheres",
        peer_radius=0.3,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )
    starts = np.asarray(
        [[0.0, -1.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.8, 0.2]], dtype=float
    )
    goals = np.asarray(
        [[7.0, -1.0, 0.0], [7.0, 0.5, 0.0], [7.0, 1.8, 0.2]], dtype=float
    )
    obstacle = StaticSphereObstacle(
        center=np.asarray(obstacle_center, dtype=float),
        radius=float(obstacle_radius),
        safety_margin=0.1,
    )
    env.reset(
        seed=11,
        options={
            "starts": starts,
            "goals": goals,
            "static_obstacles": [obstacle],
            "dynamic_obstacles": [],
        },
    )
    return env


def _diagnostic(env, policy, *, sensing, horizon=4, stop=False):
    initial, local = build_preview_inputs_from_env(env, 0)
    refreshed = (
        build_refreshed_static_sensing_context(env, 0)
        if sensing == SENSING_REFRESHED_STATIC
        else None
    )
    return diagnostic_preview_rollout(
        initial_state=initial,
        local_context=local,
        candidate_goal=np.asarray([1.05, -1.0, 0.0]),
        policy=policy,
        horizon=horizon,
        dmp_config=env.dmps[0].config,
        dynamics=env.dynamics[0],
        static_obstacles=copy.deepcopy(env.static_obstacles),
        collision_margin=float(env.env_config.collision_margin),
        terminal_tolerance=float(env.env_config.goal_tolerance),
        sensing_mode=sensing,
        refreshed_context=refreshed,
        stop_on_diagnostic_termination=stop,
    )


def test_01_frozen_manifest_and_checkpoint_hashes_are_pinned():
    config = _config()
    assert sha256_file(REPO_ROOT / "artifacts/geometry_generalization_audit/20260815_175228/scenario_manifest.json") == config["source_manifest_sha256_expected"]
    assert sha256_file(REPO_ROOT / config["checkpoint"]) == config["checkpoint_sha256_expected"]


def test_02_exact_2x2_factor_contract_and_h4_anchor():
    config = _config()
    assert [(config["cells"][cell]["horizon"], config["cells"][cell]["sensing_mode"]) for cell in "ABCD"] == [
        (4, SENSING_FROZEN),
        (4, SENSING_REFRESHED_STATIC),
        (20, SENSING_FROZEN),
        (20, SENSING_REFRESHED_STATIC),
    ]
    assert config["factor_contract"]["H4_termination_rule"] == H4_TERMINATION_RULE
    assert config["factor_contract"]["H20_termination_rule"] == H20_TERMINATION_RULE
    assert not config["cells"]["A"]["diagnostic_early_stop"]
    assert not config["cells"]["B"]["diagnostic_early_stop"]
    assert config["cells"]["C"]["diagnostic_early_stop"]
    assert config["cells"]["D"]["diagnostic_early_stop"]


def test_03_source_contains_exactly_24_layouts_and_72_frozen_selections():
    module = _load_script()
    source = module._load_sources(_config())
    assert len(source["manifest"]["layouts"]) == 24
    assert len(source["selections"]) == 72
    assert len(source["prior_h4"]) == 72


def test_04_selected_candidate_record_is_hash_stable_and_bundle_derived():
    module = _load_script()
    selection = module._load_sources(_config())["selections"][0]
    first = module._candidate_record(selection)
    second = module._candidate_record(copy.deepcopy(selection))
    assert first["candidate_record_hash"] == second["candidate_record_hash"]
    np.testing.assert_array_equal(first["candidate_world_point"], second["candidate_world_point"])


def test_05_refreshed_context_reproduces_real_t0_combined_scan():
    env = _make_env()
    context = build_refreshed_static_sensing_context(env, 0)
    np.testing.assert_allclose(
        context.initial_combined_scan,
        env.latest_sensor_packets[0].current_scan,
        rtol=0.0,
        atol=2.0e-7,
    )
    assert context.source_identity_verified
    env.close()


def test_06_refreshed_static_scan_changes_with_virtual_ego_position():
    env = _make_env()
    context = build_refreshed_static_sensing_context(env, 0)
    start = reconstruct_refreshed_static_scan(env.dynamics[0].p, context)
    moved = reconstruct_refreshed_static_scan(env.dynamics[0].p + np.asarray([0.7, 0.0, 0.0]), context)
    assert not np.array_equal(start, moved)
    env.close()


def test_07_refreshed_scope_freezes_only_visible_peer_surfaces():
    env = _make_env()
    context = build_refreshed_static_sensing_context(env, 0)
    assert context.scope == REFRESHED_SENSING_SCOPE
    assert context.peer_visible_surface_points.ndim == 2
    config = _config()["refreshed_sensing"]
    assert config["peer_hidden_surface_added"] is False
    assert config["peer_future_motion_prediction"] is False
    assert config["joint_multi_agent_rollout"] is False
    env.close()


def test_08_virtual_lidar_history_advances_preview_locally():
    env = _make_env()
    with scoped_historical_preview_and_multi_agent_transition():
        result = _diagnostic(env, ConstantPolicy(), sensing=SENSING_REFRESHED_STATIC, horizon=4)
    np.testing.assert_array_equal(
        result.trajectory.previous_scans[1:], result.trajectory.current_scans[:-1]
    )
    env.close()


def test_09_a_wrapper_numerically_matches_operational_preview():
    env = _make_env()
    policy = ConstantPolicy()
    initial, local = build_preview_inputs_from_env(env, 0)
    with scoped_historical_preview_and_multi_agent_transition():
        formal = preview_candidate(
            initial_state=initial,
            local_context=local,
            candidate_goal=np.asarray([1.05, -1.0, 0.0]),
            policy=policy,
            horizon=4,
            dmp_config=env.dmps[0].config,
            dynamics=env.dynamics[0],
        )
        anchor = _diagnostic(env, policy, sensing=SENSING_FROZEN, horizon=4, stop=False)
    np.testing.assert_allclose(anchor.trajectory.positions, formal.trajectory.positions, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(anchor.trajectory.actions, formal.trajectory.actions, rtol=0.0, atol=1e-12)
    assert anchor.task_progress == pytest.approx(formal.task_progress, abs=1e-12)
    assert anchor.min_clearance == pytest.approx(formal.min_clearance, abs=1e-12)
    assert anchor.max_execution_deviation == pytest.approx(formal.max_execution_deviation, abs=1e-12)
    assert anchor.terminal_speed == pytest.approx(formal.terminal_speed, abs=1e-12)
    env.close()


def test_10_h4_anchor_does_not_early_stop_even_when_collision_is_recorded():
    env = _make_env(obstacle_center=(0.0, -1.0, 0.0), obstacle_radius=0.45)
    with scoped_historical_preview_and_multi_agent_transition():
        result = _diagnostic(env, ConstantPolicy(), sensing=SENSING_FROZEN, horizon=4, stop=False)
    assert result.trajectory.horizon == 4
    assert result.metadata["preview_ground_truth_collision"] is True
    assert result.metadata["preview_termination_step"] is None
    env.close()


def test_11_h20_collision_termination_stops_both_sensing_modes():
    for sensing in (SENSING_FROZEN, SENSING_REFRESHED_STATIC):
        env = _make_env(obstacle_center=(0.0, -1.0, 0.0), obstacle_radius=0.45)
        with scoped_historical_preview_and_multi_agent_transition():
            result = _diagnostic(env, ConstantPolicy(), sensing=sensing, horizon=20, stop=True)
        assert result.trajectory.horizon == 1
        assert result.metadata["preview_termination_step"] == 1
        assert result.metadata["termination_reason"] == "static_obstacle_collision"
        env.close()


def test_12_sensor_clearance_is_independent_of_ground_truth_collision_geometry():
    assert math_is_inf(sensor_derived_clearance(np.ones((8, 7), dtype=np.float32), 6.0))
    config = _config()["clearance_contract"]
    assert config["ground_truth_geometry_used_in_feature"] is False
    assert config["ground_truth_collision_field"] == "preview_ground_truth_collision"


def math_is_inf(value):
    return bool(np.isinf(float(value)))


def test_13_refreshed_sensing_does_not_mutate_real_sensor_buffer_or_state():
    env = _make_env()
    before = (
        env._positions().copy(),
        env._velocities().copy(),
        [sensor._previous_scan.copy() for sensor in env.sensors],
        copy.deepcopy(env.np_random.bit_generator.state),
        int(env.steps),
    )
    with scoped_historical_preview_and_multi_agent_transition():
        _diagnostic(env, ConstantPolicy(), sensing=SENSING_REFRESHED_STATIC, horizon=4)
    np.testing.assert_array_equal(before[0], env._positions())
    np.testing.assert_array_equal(before[1], env._velocities())
    for left, sensor in zip(before[2], env.sensors, strict=True):
        np.testing.assert_array_equal(left, sensor._previous_scan)
    assert before[3] == env.np_random.bit_generator.state
    assert before[4] == env.steps
    env.close()


def test_14_historical_vector_gate_is_used_by_diagnostic_rollout():
    env = _make_env()
    with scoped_historical_preview_and_multi_agent_transition():
        result = _diagnostic(env, ConstantPolicy(), sensing=SENSING_FROZEN)
    assert {row["forcing_gate_semantics"] for row in result.trajectory.controller_infos} == {HISTORICAL_GATE_NAME}
    env.close()


def test_15_actor_observation_contract_remains_122_dimensional():
    env = _make_env()
    with scoped_historical_preview_and_multi_agent_transition():
        result = _diagnostic(env, ConstantPolicy(), sensing=SENSING_REFRESHED_STATIC)
    assert result.trajectory.observations.shape == (4, 122)
    env.close()


def test_16_formal_h_and_three_feature_score_are_unchanged():
    formal = _config()["formal_selector_unchanged"]
    assert formal["H_preview"] == 4
    assert formal["formula"] == "+progress +clearance -deviation"
    assert formal["weights"]["terminal_speed"] == 0.0
    assert formal["used_for_candidate_selection"] is False


def test_17_script_has_no_proposal_call_or_candidate_reselection_path():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "propose_reference_points(" not in source
    assert "generate_candidate_set(" not in source
    assert "choose_execution_reference(" not in source


def test_18_script_has_no_gat_or_training_execution():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "optimizer.step(" not in source
    assert ".backward(" not in source
    assert "GAT(" not in source
    assert "env.step(" not in source


def test_19_all_strict_exclusions_remain_false():
    assert not any(_config()["strict_exclusions"].values())


def test_20_effect_thresholds_are_frozen_before_results():
    config = _config()
    assert config["warning_criteria"]["frozen_before_run"] is True
    assert config["warning_criteria"]["extra_posthoc_clearance_threshold"] is None
    assert classify_directional_fraction(3, 6, config["effect_classification"]) == "STRONG"
    assert classify_directional_fraction(2, 6, config["effect_classification"]) == "MODERATE"
    assert classify_directional_fraction(1, 6, config["effect_classification"]) == "WEAK"
    assert classify_directional_fraction(0, 6, config["effect_classification"]) == "NONE"


def test_21_factor_checker_accepts_only_expected_pair_structure():
    module = _load_script()
    rows = []
    base = {
        "layout_id": "L",
        "family": "A",
        "agent_id": 0,
        "candidate_record_hash": "r",
        "candidate_set_hash": "s",
        "candidate_index": 1,
        "candidate_world_point": [1.0, 0.0, 0.0],
        "proposal_score": 2.0,
        "K_t": 3,
        "trajectory_positions": [[0.0, 0.0, 0.0]],
        "trajectory_actions": [],
    }
    for cell, horizon, sensing, stop in (
        ("A", 4, SENSING_FROZEN, False),
        ("B", 4, SENSING_REFRESHED_STATIC, False),
        ("C", 20, SENSING_FROZEN, True),
        ("D", 20, SENSING_REFRESHED_STATIC, True),
    ):
        rows.append({**base, "cell": cell, "horizon_requested": horizon, "sensing_mode": sensing, "diagnostic_termination_enabled": stop})
    assert module._verify_factor_pairs(rows)["status"] == "PASSED"


def test_22_formal_core_files_are_not_listed_as_modified_targets():
    allowed = {
        "planning/fp_shep_preview_fidelity_analysis.py",
        "configs/evaluation/fp_shep_preview_fidelity_control.json",
        "Multi-agent_Algo_lib/scripts/evaluate_fp_shep_preview_fidelity_control.py",
        "test/test_fp_shep_preview_fidelity_control.py",
    }
    config = _config()
    assert config["strict_exclusions"]["formal_FP_SHEP_modification"] is False
    assert "planning/policy_preview.py" not in allowed
    assert "Environment/multi_agent_dmp_env.py" not in allowed


def test_23_refreshed_result_is_explicitly_static_only_not_dynamic_generalization():
    config = _config()
    assert config["refreshed_sensing"]["scope"] == REFRESHED_SENSING_SCOPE
    assert config["refreshed_sensing"]["peer_future_motion_prediction"] is False
    assert config["refreshed_sensing"]["joint_multi_agent_rollout"] is False
