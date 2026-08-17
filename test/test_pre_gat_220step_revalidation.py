from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import Environment.multi_agent_dmp_env as environment_module
import planning.policy_preview as preview_module
from Controller.dmp_rl import DMPConfig
from planning.historical_forcing_gate import (
    HISTORICAL_GATE_NAME,
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.policy_preview import PreviewInitialState, PreviewLocalContext, preview_candidate
from planning.pre_gat_220step_revalidation import (
    METHOD_FP_SHEP,
    METHOD_PROPOSAL,
    ImmutableCandidate,
    ImmutableCandidateBundle,
    build_selector_pairing,
    classify_failure,
)
from planning.pre_gat_closed_loop import FPSHEPOnlineScoreSpec


REPO_ROOT = Path(__file__).resolve().parents[1]


class _Dynamics:
    dt = 0.1
    accelerate_min = -20.0
    accelerate_max = 20.0
    velocity_min = -5.0
    velocity_max = 5.0


class _Policy:
    def predict(self, observation, deterministic=True):
        assert deterministic is True
        return np.asarray([1.0, -2.0, 3.0, 0.2, -0.3, 0.4], dtype=np.float32), None


def _dmp_config() -> DMPConfig:
    return DMPConfig(
        dt=0.1,
        K_alpha=3.0,
        K_beta=0.8,
        tau=2.5,
        forcing_term_min=-10.0,
        forcing_term_max=10.0,
        goal_offset_max=1.0,
        phase_mode="classic",
        phase_integrator="legacy_euler",
    )


def _preview_inputs():
    initial = PreviewInitialState(
        position=np.asarray([1.0, 2.0, 3.0]),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        phase=1.0,
        active_goal=np.asarray([4.0, 4.0, 4.0]),
        task_goal=np.asarray([8.0, 8.0, 8.0]),
    )
    directions = np.zeros((56, 3), dtype=float)
    directions[:, 0] = 1.0
    context = PreviewLocalContext(
        current_scan=np.ones(56, dtype=np.float32),
        previous_scan=np.ones(56, dtype=np.float32),
        ray_directions=directions,
        sensing_radius=5.0,
        goal_distance_clip=10.0,
        visible_surface_points=np.empty((0, 3), dtype=float),
    )
    return initial, context


def test_historical_scope_routes_actual_preview_and_execution_symbols() -> None:
    default_preview = preview_module.propagate_sac_dmp_action
    default_execution = environment_module.propagate_sac_dmp_action
    preview_trace = []
    execution_trace = []
    initial, context = _preview_inputs()
    with scoped_historical_preview_and_multi_agent_transition(
        preview_observer=lambda _kwargs, transition: preview_trace.append(transition),
        execution_observer=lambda _kwargs, transition: execution_trace.append(transition),
    ):
        assert preview_module.propagate_sac_dmp_action is not default_preview
        assert environment_module.propagate_sac_dmp_action is not default_execution
        result = preview_candidate(
            initial_state=initial,
            local_context=context,
            candidate_goal=np.asarray([4.0, 5.0, 6.0]),
            policy=_Policy(),
            horizon=1,
            dmp_config=_dmp_config(),
            dynamics=_Dynamics(),
        )
        environment_module.propagate_sac_dmp_action(
            position=initial.position,
            velocity=initial.velocity,
            phase=initial.phase,
            active_goal=np.asarray([4.0, 5.0, 6.0]),
            terminal_goal=initial.task_goal,
            action=np.asarray([1.0, -2.0, 3.0, 0.2, -0.3, 0.4]),
            dmp_config=_dmp_config(),
            dynamics=_Dynamics(),
        )
    assert len(preview_trace) == 1
    assert len(execution_trace) == 1
    info = result.trajectory.controller_infos[0]
    expected_goal_eff = np.asarray([4.2, 4.7, 6.4])
    np.testing.assert_allclose(info["goal_eff"], expected_goal_eff)
    np.testing.assert_allclose(
        info["forcing_gate"], np.tanh(np.abs(expected_goal_eff - initial.position))
    )
    assert info["forcing_gate_semantics"] == HISTORICAL_GATE_NAME
    assert preview_module.propagate_sac_dmp_action is default_preview
    assert environment_module.propagate_sac_dmp_action is default_execution


def test_historical_scope_restores_both_symbols_after_exception() -> None:
    default_preview = preview_module.propagate_sac_dmp_action
    default_execution = environment_module.propagate_sac_dmp_action
    with pytest.raises(RuntimeError, match="preview failed"):
        with scoped_historical_preview_and_multi_agent_transition():
            raise RuntimeError("preview failed")
    assert preview_module.propagate_sac_dmp_action is default_preview
    assert environment_module.propagate_sac_dmp_action is default_execution


def test_candidate_bundle_is_immutable_and_defensively_exposed() -> None:
    candidate = ImmutableCandidate(
        original_index=0,
        world_point=(1.0, 2.0, 3.0),
        proposal_score=0.75,
        metadata_json='{"alignment":0.5,"direction":[1,0,0]}',
    )
    bundle = ImmutableCandidateBundle(
        scenario="open", seed=0, per_agent=((candidate,),), count_before_consumer=(1,)
    )
    before = bundle.candidate_set_hash
    point = candidate.point
    point[0] = 99.0
    metadata = candidate.metadata
    metadata["alignment"] = -1.0
    assert candidate.point.tolist() == [1.0, 2.0, 3.0]
    assert candidate.metadata["alignment"] == 0.5
    assert bundle.candidate_set_hash == before


def test_online_score_contract_remains_h4_three_feature() -> None:
    config = json.loads(
        (REPO_ROOT / "configs/evaluation/pre_gat_220step_revalidation.json").read_text(
            encoding="utf-8"
        )
    )
    spec = FPSHEPOnlineScoreSpec.from_mapping(config["fp_shep_online_selector"])
    metadata = spec.metadata()
    assert metadata["H_preview"] == 4
    assert metadata["formula"] == "+progress +clearance -deviation"
    assert metadata["weights"]["terminal_speed"] == 0.0
    assert metadata["used_for_online_ranking"]["terminal_speed"] is False
    assert config["fp_shep_online_selector"]["normalization_recalibrated"] is False


def _agent(method: str, *, no_candidate: bool, reached: bool) -> dict:
    return {
        "method": method,
        "scenario": "open",
        "seed": 0,
        "agent_id": 0,
        "candidate_set_hash": "same",
        "candidate_order_hash": "same-order",
        "no_candidate_fallback": no_candidate,
        "selected_candidate_index": None if no_candidate else 0,
        "reference_reached": reached,
    }


def test_k0_pair_is_excluded_from_selector_superiority_denominator() -> None:
    agents = [
        _agent(METHOD_PROPOSAL, no_candidate=True, reached=False),
        _agent(METHOD_FP_SHEP, no_candidate=True, reached=False),
    ]
    episodes = [
        {"method": method, "scenario": "open", "seed": 0, "team_success": True}
        for method in (METHOD_PROPOSAL, METHOD_FP_SHEP)
    ]
    rows, summary = build_selector_pairing(episodes, agents)
    assert rows[0]["selector_pair_status"] == "NO_CANDIDATE_PAIR"
    assert rows[0]["selector_disagreement"] is None
    assert summary["eligible_selector_pair_count"] == 0
    assert summary["no_candidate_pair_count"] == 1
    assert summary["selector_disagreement_rate"] is None


def test_team_pair_classification_is_counted_once_per_episode() -> None:
    agents = []
    for agent_id in range(3):
        for method in (METHOD_PROPOSAL, METHOD_FP_SHEP):
            row = _agent(method, no_candidate=False, reached=True)
            row["agent_id"] = agent_id
            agents.append(row)
    episodes = [
        {"method": METHOD_PROPOSAL, "scenario": "open", "seed": 0, "team_success": False},
        {"method": METHOD_FP_SHEP, "scenario": "open", "seed": 0, "team_success": True},
    ]
    _, summary = build_selector_pairing(episodes, agents)
    assert summary["eligible_selector_pair_count"] == 3
    assert summary["eligible_team_pair_count"] == 1
    assert summary["team_pair_class_counts"]["FP_SHEP_ONLY_SUCCESS"] == 1


def test_disagreement_reference_counts_only_include_disagreement_pairs() -> None:
    agents = []
    for agent_id, fp_index in ((0, 1), (1, 0)):
        proposal = _agent(METHOD_PROPOSAL, no_candidate=False, reached=False)
        proposal["agent_id"] = agent_id
        fp = _agent(METHOD_FP_SHEP, no_candidate=False, reached=True)
        fp["agent_id"] = agent_id
        fp["selected_candidate_index"] = fp_index
        agents.extend((proposal, fp))
    episodes = [
        {"method": method, "scenario": "open", "seed": 0, "team_success": False}
        for method in (METHOD_PROPOSAL, METHOD_FP_SHEP)
    ]
    _, summary = build_selector_pairing(episodes, agents)
    assert summary["selector_disagreement_count"] == 1
    assert summary["disagreement_reference_pair_class_counts"] == {
        "BOTH_REACH": 0,
        "PROPOSAL_ONLY_REACH": 0,
        "FP_SHEP_ONLY_REACH": 1,
        "BOTH_FAIL": 0,
    }


def test_failure_attribution_is_primary_exclusive_with_overlapping_flags() -> None:
    row = {
        "no_candidate_fallback": False,
        "reference_available": True,
        "reference_reached": False,
        "team_timeout": True,
        "obstacle_collision_before_reference": True,
        "inter_agent_collision_before_reference": False,
        "collision_after_reference": False,
        "terminal_completed_after_reference": False,
    }
    primary, flags = classify_failure([row])
    assert primary == "C_obstacle_collision_before_reference"
    assert flags["B_reference_not_reached_before_timeout"] is True
    assert flags["C_obstacle_collision_before_reference"] is True


def test_revalidation_config_freezes_220step_no_training_scope() -> None:
    config = json.loads(
        (REPO_ROOT / "configs/evaluation/pre_gat_220step_revalidation.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["max_steps"] == 220
    assert config["scenarios"] == ["open", "sparse_static", "multi_agent"]
    assert config["stage_a_seeds"] == list(range(5))
    assert config["stage_b_seeds"] == list(range(5, 10))
    assert not any(config["strict_exclusions"].values())
    assert config["graph_builder_post_hoc"]["enabled"] is False
