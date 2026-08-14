from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
SCRIPTS_ROOT = ALGO_ROOT / "scripts"
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Guidance.reference_point_proposal_demo import Proposal, ProposalConfig
from planning.candidate_supervision import (
    CandidateQualitySpec,
    assign_seed_split,
    branch_is_failure,
    candidate_set_ranking_metrics,
    compute_candidate_quality,
    soft_target,
    state_group_id,
    target_bundle,
    termination_reason,
    validate_sample_timesteps,
    validate_split_seeds,
)
from planning.candidate_supervision_dataset import (
    CandidateSupervisionConfig,
    evaluate_ego_supervision,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (
    build_single_distribution_multi_config,
)
from scripts.evaluate_single_policy_multi_agent import SinglePolicyMultiAgentEnv


CHECKPOINT = REPO_ROOT / "artifacts" / "20260520_201912" / "best_eval_model.pt"


class ParameterizedDeterministicPolicy:
    def __init__(self) -> None:
        self.actor = torch.nn.Linear(1, 1)

    def predict(self, observation, deterministic=True):
        assert deterministic is True
        observation = np.asarray(observation, dtype=np.float32)
        action = np.zeros(observation.shape[:-1] + (6,), dtype=np.float32)
        action[..., :3] = 1.5 * observation[..., 3:6]
        return action, None


def _make_env(max_steps: int = 20) -> SinglePolicyMultiAgentEnv:
    config = build_single_distribution_multi_config(num_agents=3, max_steps=max_steps)
    env = SinglePolicyMultiAgentEnv(
        **config.build_core_env_kwargs(),
        observation_mode="peer_spheres",
        peer_radius=0.3,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )
    starts = np.asarray(
        [[0.2, -1.4, 0.0], [0.2, 0.0, 0.3], [0.2, 1.4, -0.3]],
        dtype=float,
    )
    goals = np.asarray(
        [[7.0, -1.4, 0.0], [7.0, 0.0, 0.3], [7.0, 1.4, -0.3]],
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


def _proposal(index: int, point: np.ndarray, score: float) -> Proposal:
    point = np.asarray(point, dtype=float)
    origin = np.asarray([0.2, -1.4, 0.0], dtype=float)
    delta = point - origin
    distance = float(np.linalg.norm(delta))
    direction = delta / distance
    return Proposal(
        azimuth_index=index,
        elevation_index=0,
        direction=direction,
        point=point,
        distance=distance,
        raw_obstacle_distance=4.5,
        obstacle_distance=4.5,
        effective_safe_radius=1.15,
        braking_distance=0.0,
        safety_margin=3.35,
        normalized_margin=0.9,
        distance_progress=0.8,
        normalized_progress=0.8,
        alignment=0.9,
        smoothness=0.0,
        usable_length=0.9,
        score=score,
    )


@dataclass
class _RolloutFlags:
    collision: bool = False
    obstacle_collision: bool = False
    inter_agent_collision: bool = False
    boundary_collision: bool = False
    success: bool = False
    agent_success: bool = False
    terminated: bool = False
    truncated: bool = False
    effective_steps: int = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_physical_normalization_and_open_space_encoding():
    rows = [
        {
            "task_progress": 0.16,
            "min_clearance": float("inf"),
            "max_execution_deviation": 0.525,
            "terminal_speed": 0.6,
        }
    ]
    result = compute_candidate_quality(rows)
    np.testing.assert_allclose(result.normalized.values[0], [0.5, 1.0, 0.5, 0.5])
    assert result.normalized.clearance_positive_inf_flag[0]
    assert not result.normalized.clearance_finite_mask[0]
    assert np.all(result.normalized.valid_mask[0])


def test_collision_failure_tier_is_strictly_below_every_safe_branch():
    rows = [
        {"task_progress": -0.32, "min_clearance": 0.0, "max_execution_deviation": 1.05, "terminal_speed": 1.2},
        {"task_progress": 0.32, "min_clearance": float("inf"), "max_execution_deviation": 0.0, "terminal_speed": 0.0},
    ]
    quality = compute_candidate_quality(rows, failure_mask=[False, True])
    assert quality.target_three_feature[1] < quality.target_three_feature[0]
    assert quality.target_four_feature[1] < quality.target_four_feature[0]
    assert quality.nominal_three_feature[1] > quality.nominal_three_feature[0]


def test_success_and_timeout_are_not_implicitly_collision_failures():
    success = _RolloutFlags(success=True, terminated=True)
    timeout = _RolloutFlags(truncated=True, effective_steps=3)
    collision = _RolloutFlags(collision=True, inter_agent_collision=True, terminated=True, effective_steps=2)
    assert not branch_is_failure(success)
    assert not branch_is_failure(timeout)
    assert branch_is_failure(collision)
    assert termination_reason(success, 4) == "team_success"
    assert termination_reason(timeout, 4) == "timeout"
    assert termination_reason(collision, 4) == "collision:inter_agent"


def test_three_and_four_feature_targets_are_both_present_but_not_final():
    rows = [
        {"task_progress": 0.1, "min_clearance": 1.0, "max_execution_deviation": 0.1, "terminal_speed": 0.1},
        {"task_progress": 0.2, "min_clearance": 1.2, "max_execution_deviation": 0.2, "terminal_speed": 0.9},
    ]
    quality = compute_candidate_quality(rows, spec=CandidateQualitySpec())
    bundle = target_bundle(quality, (0.1, 0.25, 0.5))
    assert bundle["provisional_primary_target"].shape == (2,)
    assert bundle["companion_target"].shape == (2,)
    assert not np.array_equal(
        bundle["provisional_primary_target"], bundle["companion_target"]
    )
    assert CandidateQualitySpec().metadata()["final_target_frozen"] is False


def test_soft_target_is_finite_normalized_and_temperature_sensitive():
    cold = soft_target([1.0, 0.9, 0.0], 0.1)
    warm = soft_target([1.0, 0.9, 0.0], 0.5)
    assert np.sum(cold) == pytest.approx(1.0)
    assert np.sum(warm) == pytest.approx(1.0)
    assert np.all(np.isfinite(cold)) and np.all(np.isfinite(warm))
    assert cold.max() > warm.max()


def test_state_sampling_gap_and_seed_split_are_leakage_free():
    assert validate_sample_timesteps((0, 12, 24), maximum_label_horizon=8) == (0, 12, 24)
    with pytest.raises(ValueError, match="at least H_label_max"):
        validate_sample_timesteps((0, 4, 8), maximum_label_horizon=8)
    split = {"train": list(range(6)), "validation": [6, 7], "test": [8, 9]}
    validate_split_seeds(split)
    assert assign_seed_split(5, split) == "train"
    assert assign_seed_split(7, split) == "validation"
    assert assign_seed_split(9, split) == "test"
    assert state_group_id(scenario="open", seed=2, episode=0, timestep=12).endswith("t0012")


def test_ranking_metrics_treat_null_as_unavailable_to_proposal_baseline():
    metrics = candidate_set_ranking_metrics(
        proposal_scores_with_null=[float("nan"), 3.0, 2.0],
        formal_preview_quality=[2.0, 1.0, 0.0],
        real_target_quality=[2.0, 1.0, 0.0],
    )
    assert metrics["oracle_top1_class_index"] == 0
    assert metrics["proposal_top1_hit_full_target"] == 0.0
    assert metrics["fp_shep_top1_hit_full_target"] == 1.0
    assert metrics["proposal_null_score_available"] is False


def test_supervision_config_rejects_non_operational_graph_preview_horizon():
    with pytest.raises(ValueError, match="H_preview=4"):
        CandidateSupervisionConfig(h_preview=6)


@pytest.fixture(scope="module")
def supervision_evaluation():
    env = _make_env()
    policy = ParameterizedDeterministicPolicy()
    parameters_before = {
        key: value.detach().clone() for key, value in policy.actor.state_dict().items()
    }
    dmp_configs_before = [repr(item.config) for item in env.dmps]
    checkpoint_hash_before = _sha256(CHECKPOINT)
    proposals = [
        _proposal(0, np.asarray([1.05, -1.35, 0.0]), 2.0),
        _proposal(1, np.asarray([0.95, -1.10, 0.1]), 1.0),
    ]
    result = evaluate_ego_supervision(
        env=env,
        agent_index=0,
        proposals=proposals,
        policy=policy,
        proposal_config=ProposalConfig(top_k=2),
        config=CandidateSupervisionConfig(h_labels=(1,), consumer_top_k=2),
    )
    yield env, policy, parameters_before, dmp_configs_before, checkpoint_hash_before, result, proposals
    env.close()


def test_null_class_mapping_and_graph_schema_are_unchanged(supervision_evaluation):
    _, _, _, _, _, result, _ = supervision_evaluation
    np.testing.assert_allclose(
        result.formal_previews[0].trajectory.candidate_goal,
        result.graph["agent"].task_goal[0].cpu().numpy(),
    )
    assert result.graph["null"].x.shape == (1, 5)
    assert result.graph["proposal"].num_nodes == 2
    assert tuple(result.graph.proposal_node_to_candidate_id) == (0, 1)
    assert not hasattr(result.graph["null"], "J_target")


def test_formal_graph_always_uses_h4_and_diagnostic_is_excluded(supervision_evaluation):
    _, _, _, _, _, result, _ = supervision_evaluation
    assert result.graph.graph_metadata["H"] == 4
    assert all(item.trajectory.horizon == 4 for item in result.formal_previews)
    horizon = result.horizon_evaluations[1]
    assert all(item.trajectory.horizon == 1 for item in horizon.diagnostic_previews)
    assert all(record["diagnostic_preview_excluded_from_graph"] for record in horizon.class_records)
    assert all(record["diagnostic_preview_excluded_from_formal_J_preview"] for record in horizon.class_records)


def test_supervision_is_side_effect_free_for_environment_policy_dmp_and_checkpoint(
    supervision_evaluation,
):
    env, policy, parameters_before, dmp_configs_before, checkpoint_hash_before, result, _ = supervision_evaluation
    assert result.initial_state_fingerprint == result.final_state_fingerprint
    for key, value in policy.actor.state_dict().items():
        torch.testing.assert_close(value, parameters_before[key])
    assert [repr(item.config) for item in env.dmps] == dmp_configs_before
    assert _sha256(CHECKPOINT) == checkpoint_hash_before


def test_k_larger_than_available_candidates_uses_actual_k_without_padding():
    env = _make_env()
    try:
        proposal = _proposal(0, np.asarray([1.05, -1.35, 0.0]), 1.0)
        result = evaluate_ego_supervision(
            env=env,
            agent_index=0,
            proposals=[proposal],
            policy=ParameterizedDeterministicPolicy(),
            proposal_config=ProposalConfig(top_k=10),
            config=CandidateSupervisionConfig(h_labels=(1,), consumer_top_k=10),
        )
        assert len(result.selected_proposals) == 1
        assert result.graph["proposal"].num_nodes == 1
        assert len(result.formal_previews) == 2
    finally:
        env.close()


def test_candidate_order_does_not_change_goal_keyed_supervision_results():
    env = _make_env()
    try:
        proposals = [
            _proposal(0, np.asarray([1.05, -1.35, 0.0]), 2.0),
            _proposal(1, np.asarray([0.95, -1.10, 0.1]), 1.0),
        ]
        config = CandidateSupervisionConfig(h_labels=(1,), consumer_top_k=2)
        forward = evaluate_ego_supervision(
            env=env, agent_index=0, proposals=proposals,
            policy=ParameterizedDeterministicPolicy(), proposal_config=ProposalConfig(top_k=2),
            config=config,
        )
        reverse = evaluate_ego_supervision(
            env=env, agent_index=0, proposals=list(reversed(proposals)),
            policy=ParameterizedDeterministicPolicy(), proposal_config=ProposalConfig(top_k=2),
            config=config,
        )
        def keyed(result):
            records = result.horizon_evaluations[1].class_records
            return {
                tuple(np.round(record["candidate_xyz"], 7)): np.asarray([
                    record["formal_J_preview_3"], record["J_target_3"],
                    record["real_task_progress"], record["real_terminal_speed"],
                ])
                for record in records
            }
        left, right = keyed(forward), keyed(reverse)
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_allclose(left[key], right[key], rtol=0.0, atol=1.0e-7)
    finally:
        env.close()
