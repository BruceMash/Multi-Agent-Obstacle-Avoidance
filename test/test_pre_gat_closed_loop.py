from __future__ import annotations

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
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.candidate_execution_benchmark import environment_state_fingerprint
from planning.pre_gat_closed_loop import (
    FORMAL_PREVIEW_HORIZON,
    METHOD_FP_SHEP,
    METHOD_FROZEN,
    METHOD_PROPOSAL,
    SELECTION_BASELINE_TERMINAL,
    SELECTION_FP_SHEP,
    SELECTION_NO_CANDIDATE_FALLBACK,
    SELECTION_PROPOSAL,
    FPSHEPOnlineScoreSpec,
    FixedPeriodProtocol,
    choose_execution_reference,
    generate_candidate_set,
    goal_switch,
    minimum_pairwise_distance,
    score_fp_shep_candidates,
    termination_reason,
    trajectory_metrics,
)
from scripts.analyze_pre_gat_closed_loop import (
    aggregate_group,
    build_disagreement_summary,
    build_paired_analysis,
)
from scripts.evaluate_frozen_policy_waypoint_guidance import (
    build_active_goal_observations,
    set_dmp_active_goal_preserve_phase,
)
from scripts.evaluate_pre_gat_closed_loop import (
    _policy_parameter_sha256,
    choose_development_m_upper,
    run_closed_loop_episode,
)
from scripts.validate_paper_ready_artifacts import _language_check
from scripts.evaluate_single_policy_aligned_multi_agent import (
    build_single_distribution_multi_config,
)


class DeterministicPolicy:
    def __init__(self) -> None:
        self.actor = torch.nn.Linear(1, 1)

    def predict(self, observation, deterministic=True):
        assert deterministic is True
        observation = np.asarray(observation, dtype=np.float32)
        action = np.zeros(observation.shape[:-1] + (6,), dtype=np.float32)
        action[..., :3] = 0.25 * observation[..., 3:6]
        return action, None


def _settings(max_steps: int = 6) -> dict:
    values = json.loads(
        (REPO_ROOT / "configs" / "evaluation" / "pre_gat_closed_loop.json").read_text(
            encoding="utf-8"
        )
    )
    values["max_steps"] = int(max_steps)
    return values


def _env(max_steps: int = 6):
    from scripts.validate_policy_preview import build_validation_environment

    config = build_single_distribution_multi_config(num_agents=3, max_steps=max_steps)
    env, _ = build_validation_environment(
        config=config,
        scene_type="open",
        seed=3,
        peer_radius=0.3,
    )
    return env


def test_online_score_spec_is_explicit_three_feature_but_records_terminal_speed():
    metadata = FPSHEPOnlineScoreSpec().metadata()
    assert metadata["H_preview"] == 4
    assert metadata["all_four_preview_features_recorded"] is True
    assert metadata["used_for_online_ranking"]["terminal_speed"] is False
    assert metadata["weights"]["terminal_speed"] == 0.0
    assert metadata["definition_status"] != "final_fp_shep_definition"


def test_online_score_rejects_non_h4_and_terminal_speed_weight():
    with pytest.raises(ValueError):
        FPSHEPOnlineScoreSpec(horizon=6)
    with pytest.raises(ValueError):
        FPSHEPOnlineScoreSpec(terminal_speed_weight=1.0)


def test_fixed_period_schedule_has_no_event_triggered_replanning():
    protocol = FixedPeriodProtocol(4)
    assert [step for step in range(13) if protocol.is_replanning_step(step)] == [0, 4, 8, 12]
    metadata = protocol.metadata()
    assert metadata["waypoint_reached_triggers_replan"] is False
    assert metadata["stagnation_triggers_replan"] is False
    assert metadata["candidate_invalidation_triggers_replan"] is False


def test_baseline_uses_terminal_goal_without_candidate_selection():
    goal = np.asarray([3.0, 0.0, 0.0])
    decision = choose_execution_reference(
        method=METHOD_FROZEN, terminal_task_goal=goal, proposals=[]
    )
    assert decision.selection_kind == SELECTION_BASELINE_TERMINAL
    assert decision.selected_candidate_id is None
    np.testing.assert_array_equal(decision.execution_reference, goal)


def test_k_zero_fallback_is_not_null_or_selection():
    goal = np.asarray([3.0, 0.0, 0.0])
    for method in (METHOD_PROPOSAL, METHOD_FP_SHEP):
        decision = choose_execution_reference(
            method=method, terminal_task_goal=goal, proposals=[]
        )
        assert decision.selection_kind == SELECTION_NO_CANDIDATE_FALLBACK
        assert decision.no_candidate_fallback is True
        assert decision.selected_candidate_id is None
        assert len(decision.proposals) == 0


def test_proposal_selector_preserves_existing_order():
    class Proposal:
        def __init__(self, point):
            self.point = np.asarray(point, dtype=float)

    proposals = [Proposal([1, 0, 0]), Proposal([2, 0, 0])]
    decision = choose_execution_reference(
        method=METHOD_PROPOSAL,
        terminal_task_goal=np.asarray([5, 0, 0]),
        proposals=proposals,
    )
    assert decision.selection_kind == SELECTION_PROPOSAL
    assert decision.selected_candidate_id == 0
    np.testing.assert_array_equal(decision.execution_reference, proposals[0].point)


def test_candidate_generation_uses_actual_k_without_padding():
    env = _env()
    try:
        from Guidance.reference_point_proposal_demo import ProposalConfig

        proposals, full_count = generate_candidate_set(
            env, 0, ProposalConfig(top_k=100), consumer_top_k=100
        )
        assert len(proposals) == full_count
        assert len({tuple(item.point) for item in proposals}) == len(proposals)
    finally:
        env.close()


def test_fp_shep_scores_same_proposal_set_and_records_all_features():
    env = _env()
    policy = DeterministicPolicy()
    try:
        from Guidance.reference_point_proposal_demo import ProposalConfig

        proposals, _ = generate_candidate_set(env, 0, ProposalConfig(), consumer_top_k=4)
        fingerprint = environment_state_fingerprint(env)
        records = score_fp_shep_candidates(
            env=env,
            agent_index=0,
            proposals=proposals,
            policy=policy,
        )
        assert environment_state_fingerprint(env) == fingerprint
        assert len(records) == len(proposals)
        for record in records:
            payload = record.to_record()
            assert payload["preview_terminal_speed"] is not None
            assert payload["terminal_speed_used_for_online_ranking"] is False
            expected = (
                record.normalized_features[0]
                + record.normalized_features[1]
                - record.normalized_features[2]
            )
            assert np.isclose(record.score, expected)
    finally:
        env.close()


def test_fp_shep_and_proposal_receive_identical_candidate_sequence():
    env = _env()
    policy = DeterministicPolicy()
    try:
        from Guidance.reference_point_proposal_demo import ProposalConfig

        proposals, _ = generate_candidate_set(env, 0, ProposalConfig(), consumer_top_k=5)
        proposal = choose_execution_reference(
            method=METHOD_PROPOSAL,
            terminal_task_goal=env.goals[0],
            proposals=proposals,
        )
        preview = choose_execution_reference(
            method=METHOD_FP_SHEP,
            terminal_task_goal=env.goals[0],
            proposals=proposals,
            policy=policy,
            env=env,
            agent_index=0,
        )
        assert [tuple(item.point) for item in proposal.proposals] == [
            tuple(item.point) for item in preview.proposals
        ]
        assert preview.selection_kind == SELECTION_FP_SHEP
    finally:
        env.close()


def test_temporary_reference_changes_only_dmp_goal_and_preserves_phase():
    env = _env()
    try:
        terminal = env.goals.copy()
        phase = float(env.dmps[0].phase)
        active = env.dynamics[0].p + np.asarray([0.4, 0.1, 0.0])
        set_dmp_active_goal_preserve_phase(env.dmps[0], active)
        np.testing.assert_array_equal(env.goals, terminal)
        np.testing.assert_array_equal(env.dmps[0].goal, active)
        assert env.dmps[0].phase == phase
    finally:
        env.close()


def test_active_goal_observation_builder_outputs_122d_and_tracks_reference():
    env = _env()
    try:
        active = env.goals.copy()
        active[0] = env.dynamics[0].p + np.asarray([0.0, 0.5, 0.0])
        observation = build_active_goal_observations(env, active)
        assert observation.shape == (3, 122)
        np.testing.assert_allclose(observation[0, 3:6], [0.0, 1.0, 0.0], atol=1e-6)
    finally:
        env.close()


def test_goal_switch_count_and_jump_are_exact():
    changed, jump = goal_switch(np.zeros(3), np.asarray([3.0, 4.0, 0.0]), 1e-8)
    assert changed is True
    assert jump == 5.0
    retained, zero = goal_switch(np.ones(3), np.ones(3), 1e-8)
    assert retained is False
    assert zero == 0.0


def test_pairwise_distance_uses_all_uav_pairs():
    points = np.asarray([[0, 0, 0], [3, 0, 0], [0, 4, 0]], dtype=float)
    assert minimum_pairwise_distance(points) == 3.0


def test_path_length_variation_and_existing_jerk_smoothness_definition():
    positions = np.asarray([
        [[0, 0, 0]], [[1, 0, 0]], [[1, 1, 0]],
    ], dtype=float)
    velocities = np.asarray([
        [[0, 0, 0]], [[1, 0, 0]], [[0, 1, 0]],
    ], dtype=float)
    accelerations = np.asarray([[[1, 0, 0]], [[0, 1, 0]]], dtype=float)
    result = trajectory_metrics(positions, velocities, accelerations, dt=0.5)
    assert result["path_lengths"][0] == 2.0
    assert result["trajectory_smoothness_team_mean"] == 8.0
    assert "delta_acceleration/dt" in result["smoothness_definition"]


def test_termination_reason_preserves_environment_outcomes():
    assert termination_reason(success=True, collision=False, terminated=True, truncated=False) == "success"
    assert termination_reason(success=False, collision=True, terminated=True, truncated=False) == "collision"
    assert termination_reason(success=False, collision=False, terminated=False, truncated=True) == "timeout"


@pytest.fixture(scope="module")
def short_closed_loop_runs():
    settings = _settings(max_steps=6)
    config = build_single_distribution_multi_config(num_agents=3, max_steps=6)
    policy = DeterministicPolicy()
    outputs = {}
    for method in (METHOD_FROZEN, METHOD_PROPOSAL, METHOD_FP_SHEP):
        outputs[method] = run_closed_loop_episode(
            policy=policy,
            multi_config=config,
            settings=settings,
            phase="test",
            scenario="open",
            seed=3,
            method=method,
            m_upper=4,
        )
    return settings, policy, outputs


def test_same_seed_has_identical_initial_state_across_methods(short_closed_loop_runs):
    _, _, outputs = short_closed_loop_runs
    hashes = {outputs[method][0]["initial_condition_hash"] for method in outputs}
    assert len(hashes) == 1


def test_methods_use_same_m_upper_and_only_scheduled_switches(short_closed_loop_runs):
    _, _, outputs = short_closed_loop_runs
    for method in (METHOD_PROPOSAL, METHOD_FP_SHEP):
        episode, events, _ = outputs[method]
        assert episode["M_upper"] == 4
        assert all(int(row["timestep"]) % 4 == 0 for row in events)
        assert all(row["replanning_rule_satisfied"] is True for row in events)


def test_episode_termination_stops_low_level_execution(short_closed_loop_runs):
    settings, _, outputs = short_closed_loop_runs
    for episode, _, trajectory in outputs.values():
        assert episode["steps"] <= settings["max_steps"]
        assert trajectory["positions"].shape[0] == episode["steps"] + 1


def test_terminal_goal_and_phase_contract_hold_in_closed_loop(short_closed_loop_runs):
    _, _, outputs = short_closed_loop_runs
    for episode, events, _ in outputs.values():
        assert episode["terminal_task_goals_unchanged"] is True
        assert episode["formal_preview_horizon"] == FORMAL_PREVIEW_HORIZON
        assert all(row.get("phase_preserved_on_switch", True) for row in events)


def test_reference_reached_fields_are_read_only_diagnostics(short_closed_loop_runs):
    _, _, outputs = short_closed_loop_runs
    for method in (METHOD_PROPOSAL, METHOD_FP_SHEP):
        _, events, _ = outputs[method]
        for row in events:
            assert "reference_reached_early" in row
            assert "reference_reached_step" in row
            assert "reference_hold_steps_after_reached" in row
            assert int(row["timestep"]) % int(row["M_upper"]) == 0


def test_policy_parameter_hash_is_unchanged_by_evaluation(short_closed_loop_runs):
    _, policy, _ = short_closed_loop_runs
    before = _policy_parameter_sha256(policy)
    after = _policy_parameter_sha256(policy)
    assert before == after


def test_no_gat_optimizer_or_supervision_target_is_imported_online():
    import planning.pre_gat_closed_loop as core
    import scripts.evaluate_pre_gat_closed_loop as evaluator

    source = inspect.getsource(core) + inspect.getsource(evaluator)
    assert "planning.gat" not in source
    assert "torch.optim" not in source
    assert "from planning.candidate_supervision" not in source
    assert "import planning.candidate_supervision" not in source
    assert '"GAT_used_for_selection": False' in source


def test_development_m_selection_is_lexicographic_and_formal_independent():
    rows = []
    for value, success, collision in ((4, 0.5, 0.4), (8, 0.7, 0.2), (12, 0.7, 0.3)):
        for _ in range(2):
            rows.append({
                "M_upper": value,
                "success": success,
                "collision": collision,
                "goal_switch_rate": 0.5,
                "trajectory_smoothness_team_mean": 1.0,
                "upper_replanning_runtime_mean_ms": 1.0,
            })
    selected, summary = choose_development_m_upper(rows, [4, 8, 12])
    assert selected == 8
    assert all(row["selection_uses_formal_seeds"] is False for row in summary)


def test_paired_transition_mapping_and_differences_are_recomputable():
    rows = []
    for method, success, path in (
        (METHOD_FROZEN, False, 5.0),
        (METHOD_PROPOSAL, False, 4.0),
        (METHOD_FP_SHEP, True, 3.0),
    ):
        rows.append({
            "pair_id": "p", "method": method, "scenario": "open", "seed": 1,
            "success": success, "collision": not success,
            "path_length_team_mean_m": path, "completion_time_all_s": 1.0,
            "minimum_inter_agent_distance_m": 1.0,
            "trajectory_smoothness_team_mean": 1.0, "goal_switch_count": 1,
        })
    result = build_paired_analysis(rows)
    assert result["transitions"]["proposal_failure_to_fp_shep_success"] == 1
    assert result["difference_rows"][0]["delta_fp_shep_minus_proposal__path_length_team_mean_m"] == -1.0


def test_disagreement_rate_uses_only_eligible_nonempty_fp_shep_events():
    events = [
        {"phase": "formal", "method": METHOD_FP_SHEP, "K_t": 3},
        {"phase": "formal", "method": METHOD_FP_SHEP, "K_t": 2},
        {"phase": "formal", "method": METHOD_FP_SHEP, "K_t": 0},
        {"phase": "formal", "method": METHOD_PROPOSAL, "K_t": 3},
    ]
    disagreements = [{
        "fp_shep_window_task_progress": 1.0,
        "proposal_window_task_progress": 0.5,
        "fp_shep_window_minimum_clearance": 1.0,
        "proposal_window_minimum_clearance": 0.5,
        "fp_shep_window_minimum_inter_agent_distance": 1.0,
        "proposal_window_minimum_inter_agent_distance": 0.5,
        "fp_shep_window_max_deviation": 0.2,
        "proposal_window_max_deviation": 0.3,
        "fp_shep_goal_jump": 0.4,
        "proposal_goal_jump": 0.5,
        "fp_shep_window_collision": False,
        "proposal_window_collision": False,
    }]
    summary = build_disagreement_summary(disagreements, events)
    assert summary["eligible_fp_shep_event_count"] == 2
    assert summary["proposal_fp_shep_disagreement_rate"] == 0.5


def test_successful_only_aggregation_excludes_failed_short_path():
    rows = [
        {
            "success": True, "collision": False, "inter_agent_collision": False,
            "obstacle_collision": False, "path_length_success_team_mean_m": 10.0,
            "completion_time_success_s": 5.0, "minimum_inter_agent_distance_m": 1.0,
            "minimum_obstacle_clearance_m": 1.0, "trajectory_smoothness_team_mean": 1.0,
            "goal_switch_count": 1, "mean_goal_jump_m": 0.2,
        },
        {
            "success": False, "collision": True, "inter_agent_collision": True,
            "obstacle_collision": False, "path_length_success_team_mean_m": None,
            "completion_time_success_s": None, "minimum_inter_agent_distance_m": 0.4,
            "minimum_obstacle_clearance_m": 1.0, "trajectory_smoothness_team_mean": 2.0,
            "goal_switch_count": 2, "mean_goal_jump_m": 0.3,
        },
    ]
    summary = aggregate_group(rows)
    assert summary["path_length_success_team_mean_m_mean"] == 10.0
    assert summary["path_length_success_team_mean_m_n"] == 1


def test_publication_language_check_rejects_chinese_and_accepts_english(tmp_path):
    good = tmp_path / "table.csv"
    good.write_text("Method,Success Rate (%)\nFrozen SAC-DMP,50\n", encoding="utf-8")
    assert _language_check(tmp_path) == []
    bad = tmp_path / "bad.json"
    bad.write_text('{"title": "成功率"}', encoding="utf-8")
    assert _language_check(tmp_path)
