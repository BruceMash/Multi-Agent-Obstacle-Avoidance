from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for path in (REPO_ROOT, ALGO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from planning.candidate_execution_benchmark import environment_state_fingerprint
from planning.pre_gat_closed_loop import FixedPeriodProtocol
from planning.temporary_reference_diagnosis import (
    PROTOCOL_EXISTING_BOUNDARY_FREE,
    PROTOCOL_FIXED_PERIOD,
    PROTOCOL_ONE_SHOT,
    PROTOCOL_TERMINAL,
    OneShotReferenceState,
    candidate_safety_diagnostic,
    dmp_switch_diagnostic,
    failure_attribution,
    observation_switch_diagnostic,
)
from scripts.evaluate_frozen_policy_waypoint_guidance import segment_is_clear
from scripts.evaluate_pre_gat_closed_loop import _policy_parameter_sha256
from scripts.evaluate_single_policy_aligned_multi_agent import (
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import (
    run_one_shot_episode,
    run_pre_gat_protocol,
)
from scripts.validate_policy_preview import build_validation_environment


class DeterministicPolicy:
    def __init__(self) -> None:
        self.actor = torch.nn.Linear(1, 1)

    def predict(self, observation, deterministic=True):
        assert deterministic is True
        observation = np.asarray(observation, dtype=np.float32)
        action = np.zeros(observation.shape[:-1] + (6,), dtype=np.float32)
        action[..., :3] = 0.2 * observation[..., 3:6]
        return action, None


def _settings(max_steps: int = 4) -> dict:
    values = json.loads(
        (
            REPO_ROOT
            / "configs"
            / "evaluation"
            / "temporary_reference_interface_diagnosis.json"
        ).read_text(encoding="utf-8")
    )
    values["max_steps"] = int(max_steps)
    return values


def _env(max_steps: int = 4):
    config = build_single_distribution_multi_config(num_agents=3, max_steps=max_steps)
    env, _ = build_validation_environment(
        config=config,
        scene_type="open",
        seed=3,
        peer_radius=0.3,
    )
    return env


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_separates_waypoint_and_fixed_period_tolerances():
    settings = _settings()
    assert settings["protocol_metadata"][PROTOCOL_ONE_SHOT]["reached_tolerance_m"] == 0.25
    assert settings["protocol_metadata"][PROTOCOL_EXISTING_BOUNDARY_FREE]["reached_tolerance_m"] == 0.25
    assert settings["protocol_metadata"][PROTOCOL_FIXED_PERIOD]["reached_tolerance_m"] == 0.30
    assert settings["existing_waypoint"]["boundary_filter_enabled"] is False


def test_boundary_free_segment_check_ignores_workspace_only():
    env = _env()
    try:
        outside = np.asarray([1.0, -2.5, 0.0])
        assert not segment_is_clear(
            env,
            0,
            env.dynamics[0].p,
            outside,
            boundary_margin=0.4,
            collision_clearance=0.0,
            samples=8,
            boundary_filter_enabled=True,
        )
        assert segment_is_clear(
            env,
            0,
            env.dynamics[0].p,
            outside,
            boundary_margin=0.4,
            collision_clearance=0.0,
            samples=8,
            boundary_filter_enabled=False,
        )
    finally:
        env.close()


def test_candidate_boundary_validity_is_diagnostic_only():
    env = _env()
    try:
        result = candidate_safety_diagnostic(
            env,
            0,
            candidate=np.asarray([1.0, -2.5, 0.0]),
            boundary_margin=0.4,
            collision_clearance=0.0,
            segment_samples=8,
        )
        assert result["boundary_validity"] is False
        assert result["boundary_validity_used_for_selection"] is False
        assert result["point_safety"] is True
        assert result["segment_validity"] is True
    finally:
        env.close()


def test_observation_delta_changes_only_goal_dependent_features_and_not_env():
    env = _env()
    try:
        fingerprint = environment_state_fingerprint(env)
        result = observation_switch_diagnostic(
            env,
            0,
            previous_active_goal=env.goals[0],
            new_active_goal=env.dynamics[0].p + np.asarray([0.2, 0.4, 0.0]),
        )
        assert result["observation_dimension"] == 122
        assert result["observation_l2_change"] > 0.0
        assert result["non_goal_feature_l2_change"] == 0.0
        assert environment_state_fingerprint(env) == fingerprint
    finally:
        env.close()


def test_dmp_jump_diagnostic_is_side_effect_free_and_terminal_gated():
    env = _env()
    policy = DeterministicPolicy()
    try:
        fingerprint = environment_state_fingerprint(env)
        terminal = env.goals[0].copy()
        temporary = env.dynamics[0].p + np.asarray([0.3, 0.2, 0.0])
        before_hash = _policy_parameter_sha256(policy)
        result = dmp_switch_diagnostic(
            env,
            0,
            previous_active_goal=terminal,
            new_active_goal=temporary,
            terminal_goal=terminal,
            policy=policy,
        )
        assert result["zero_action_nominal_acceleration_jump_mps2"] > 0.0
        assert result["closed_loop_commanded_acceleration_jump_mps2"] > 0.0
        assert result["forcing_gate_before"] == pytest.approx(result["forcing_gate_after"])
        assert result["state_mutated"] is False
        assert environment_state_fingerprint(env) == fingerprint
        assert _policy_parameter_sha256(policy) == before_hash
    finally:
        env.close()


def test_one_shot_state_creates_only_once_and_returns_once():
    state = OneShotReferenceState()
    terminal = np.asarray([5.0, 0.0, 0.0])
    temporary = np.asarray([1.0, 0.0, 0.0])
    state.activate(temporary, timestep=0)
    with pytest.raises(RuntimeError, match="only one"):
        state.activate(temporary, timestep=1)
    goal, switched = state.update_reached(
        position=np.asarray([0.8, 0.0, 0.0]),
        terminal_goal=terminal,
        completed_step=7,
        reached_tolerance=0.25,
    )
    assert switched is True
    np.testing.assert_array_equal(goal, terminal)
    assert state.candidate_created_count == 1
    assert state.returned_to_terminal is True
    goal, switched = state.update_reached(
        position=temporary,
        terminal_goal=terminal,
        completed_step=8,
        reached_tolerance=0.25,
    )
    assert switched is False
    np.testing.assert_array_equal(goal, terminal)


def test_fixed_period_remains_strict_m8_without_event_replanning():
    protocol = FixedPeriodProtocol(8)
    assert [step for step in range(25) if protocol.is_replanning_step(step)] == [0, 8, 16, 24]
    metadata = protocol.metadata()
    assert metadata["waypoint_reached_triggers_replan"] is False
    assert metadata["stagnation_triggers_replan"] is False
    assert metadata["candidate_invalidation_triggers_replan"] is False


@pytest.fixture(scope="module")
def short_protocol_runs():
    settings = _settings(max_steps=4)
    config = build_single_distribution_multi_config(num_agents=3, max_steps=4)
    policy = DeterministicPolicy()
    terminal = run_pre_gat_protocol(
        policy=policy,
        multi_config=config,
        settings=settings,
        scenario="open",
        seed=3,
        protocol=PROTOCOL_TERMINAL,
    )
    fixed = run_pre_gat_protocol(
        policy=policy,
        multi_config=config,
        settings=settings,
        scenario="open",
        seed=3,
        protocol=PROTOCOL_FIXED_PERIOD,
    )
    one_shot = run_one_shot_episode(
        policy=policy,
        multi_config=config,
        settings=settings,
        scenario="open",
        seed=3,
    )
    return settings, policy, terminal, fixed, one_shot


def test_protocols_share_identical_initial_condition(short_protocol_runs):
    _, _, terminal, fixed, one_shot = short_protocol_runs
    hashes = {terminal[0]["initial_condition_hash"], fixed[0]["initial_condition_hash"], one_shot[0]["initial_condition_hash"]}
    assert len(hashes) == 1


def test_terminal_goals_and_phase_contract_hold(short_protocol_runs):
    _, _, terminal, fixed, one_shot = short_protocol_runs
    assert terminal[0]["terminal_task_goals_unchanged"] is True
    assert fixed[0]["terminal_task_goals_unchanged"] is True
    assert one_shot[0]["terminal_task_goals_unchanged"] is True
    assert one_shot[0]["phase_reset_on_switch"] is False
    assert all(row.get("phase_preserved_on_switch", True) for row in fixed[1])


def test_fixed_events_only_exist_on_legal_period(short_protocol_runs):
    _, _, _, fixed, _ = short_protocol_runs
    assert all(int(row["timestep"]) % 8 == 0 for row in fixed[1])
    assert all(row["release_reason"] != "reached_triggered_replan" for row in fixed[1])


def test_one_shot_candidate_created_once_and_no_periodic_replanning(short_protocol_runs):
    settings, _, _, _, one_shot = short_protocol_runs
    episode, events, steps, _, _ = one_shot
    assert len(events) <= settings["num_agents"]
    assert episode["temporary_reference_count"] == len(events)
    assert all(int(row["reference_created_step"]) == 0 for row in events)
    assert len({(row["agent_id"], row["reference_created_step"]) for row in events}) == len(events)
    assert len(steps) == episode["steps"] * settings["num_agents"]


def test_temporary_actor_input_is_explicitly_rebuilt_not_native_step_observation():
    import scripts.evaluate_temporary_reference_interface as evaluator

    source = inspect.getsource(evaluator.run_one_shot_episode)
    assert "build_active_goal_observations(env, active_goals)" in source
    assert "_, _, terminated, truncated, info = env.step(actions)" in source


def test_existing_boundary_free_protocol_calls_real_consumer():
    import scripts.evaluate_temporary_reference_interface as evaluator

    source = inspect.getsource(evaluator.run_existing_boundary_free_episode)
    assert "run_waypoint_episode(" in source
    assert "boundary_filter_enabled=False" in source
    assert "priority_hold_enabled=True" in source


def test_failure_attribution_uses_predeclared_switch_window():
    episode = {"success": False, "collision": True, "truncated": False}
    events = [{"reference_activated_step": 8, "reference_reached_step": 4}]
    steps = [{"completed_step": 10, "collision": True, "inter_agent_collision": False}]
    assert failure_attribution(
        episode=episode,
        step_rows=steps,
        reference_events=events,
        shortly_after_switch_steps=3,
    ) == "collision_shortly_after_reference_switch"


def test_failure_attribution_distinguishes_fixed_hold_collision():
    episode = {"success": False, "collision": True, "truncated": False}
    events = [{"reference_activated_step": 0, "reference_reached_step": 2}]
    steps = [
        {
            "completed_step": 5,
            "collision": True,
            "fixed_period_hold_after_reached": True,
            "inter_agent_collision": False,
        }
    ]
    assert failure_attribution(
        episode=episode,
        step_rows=steps,
        reference_events=events,
        shortly_after_switch_steps=3,
    ) == "collision_during_fixed_period_hold_after_reference_reached"


def test_no_gat_sdh_or_supervision_online_dependencies():
    import planning.temporary_reference_diagnosis as core
    import scripts.evaluate_temporary_reference_interface as evaluator

    source = inspect.getsource(core) + inspect.getsource(evaluator)
    assert "torch.optim" not in source
    assert "planning.heterogeneous_candidate_graph" not in source
    assert "planning.candidate_supervision" not in source
    assert "SDH" not in source
    assert "J_target" not in source


def test_critical_core_files_are_not_modified_by_diagnostics(short_protocol_runs):
    paths = [
        REPO_ROOT / "Controller" / "dmp_rl.py",
        REPO_ROOT / "Environment" / "multi_agent_dmp_env.py",
        REPO_ROOT / "Guidance" / "reference_point_proposal_demo.py",
        REPO_ROOT / "planning" / "policy_preview.py",
    ]
    before = {path: _sha256(path) for path in paths}
    _ = short_protocol_runs
    after = {path: _sha256(path) for path in paths}
    assert before == after


def test_stage_c_gate_is_development_only_and_pre_registered():
    settings = _settings()
    gate = settings["stage_c_gate"]
    assert gate["selection_data"] == "development seeds 0-9 only"
    assert gate["minimum_absolute_success_improvement_over_fixed_period"] == 0.05
    assert set(gate["eligible_protocols"]) == {
        PROTOCOL_ONE_SHOT,
        PROTOCOL_EXISTING_BOUNDARY_FREE,
    }
    assert not set(settings["development_seeds"]) & set(
        settings["formal_seeds_excluded_during_interface_design"]
    )

