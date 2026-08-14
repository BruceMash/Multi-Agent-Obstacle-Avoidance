from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
SCRIPTS_ROOT = ALGO_ROOT / "scripts"
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Guidance.reference_point_proposal_demo import ProposalConfig, propose_reference_points
from Environment.frozen_sac_dmp_execution import freeze_policy
from experiment_config import EXPERIMENT_CONFIG as SINGLE_AGENT_CONFIG
from planning.candidate_execution_benchmark import (
    correlation_rows,
    environment_state_fingerprint,
    evaluate_candidate_set,
    real_candidate_rollout,
    score_and_summarize_candidate_set,
)
from runner_sac import build_env as build_single_env
from runner_sac import build_model as build_single_model
from runner_sac import load_checkpoint
from scripts.evaluate_single_policy_aligned_multi_agent import (
    build_single_distribution_multi_config,
)
from scripts.evaluate_single_policy_multi_agent import SinglePolicyMultiAgentEnv


CHECKPOINT = REPO_ROOT / "artifacts" / "20260520_201912" / "best_eval_model.pt"


class DeterministicDummyPolicy:
    def __init__(self):
        self.actor = torch.nn.Linear(1, 1)

    def predict(self, observation, deterministic):
        assert deterministic is True
        observation = np.asarray(observation)
        shape = (6,) if observation.ndim == 1 else (observation.shape[0], 6)
        action = np.zeros(shape, dtype=np.float32)
        direction = observation[..., 3:6]
        action[..., 3:6] = 0.25 * direction
        return action, None


def _make_env() -> SinglePolicyMultiAgentEnv:
    config = build_single_distribution_multi_config(num_agents=3, max_steps=30)
    env = SinglePolicyMultiAgentEnv(
        **config.build_core_env_kwargs(),
        observation_mode="peer_spheres",
        peer_radius=0.3,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )
    starts = np.asarray(
        [[0.2, -1.5, -0.4], [0.2, 0.0, 0.4], [0.2, 1.5, -0.4]], dtype=float
    )
    goals = np.asarray(
        [[7.5, -1.5, -0.3], [7.5, 0.0, 0.3], [7.5, 1.5, -0.3]], dtype=float
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


def _proposals(env, *, limit=4):
    proposals = propose_reference_points(
        env.dynamics[0].p,
        env.goals[0],
        env.dynamics[0].v,
        env.latest_sensor_packets[0],
        env.sensors[0],
        ProposalConfig(top_k=10),
        env.env_config.goal_tolerance,
    )
    assert len(proposals) >= limit
    return proposals[:limit]


def _keyed(rows):
    return {
        tuple(np.round(row["candidate_xyz"], 10)): np.asarray([
            row["preview_task_progress"], row["real_task_progress"],
            row["preview_execution_deviation"], row["real_execution_deviation"],
            row["preview_terminal_speed"], row["real_terminal_speed"],
        ])
        for row in rows
    }


def test_candidate_evaluation_has_no_environment_side_effects_and_uses_actual_k_t():
    env = _make_env()
    try:
        policy = DeterministicDummyPolicy()
        proposals = _proposals(env, limit=3)
        before = environment_state_fingerprint(env)
        rows, _ = evaluate_candidate_set(
            env=env,
            agent_index=0,
            proposals=proposals,
            policy=policy,
            horizon=3,
            consumer_top_k=10,
        )
        assert environment_state_fingerprint(env) == before
        assert len(rows) == 3
        assert {row["K_t"] for row in rows} == {3}
        assert all(row["K_requested"] == 10 for row in rows)
    finally:
        env.close()


def test_candidate_order_does_not_change_preview_or_real_outcomes():
    env = _make_env()
    try:
        policy = DeterministicDummyPolicy()
        proposals = _proposals(env, limit=4)
        forward, _ = evaluate_candidate_set(
            env=env, agent_index=0, proposals=proposals, policy=policy,
            horizon=4, consumer_top_k=None,
        )
        reverse, _ = evaluate_candidate_set(
            env=env, agent_index=0, proposals=list(reversed(proposals)), policy=policy,
            horizon=4, consumer_top_k=None,
        )
        left, right = _keyed(forward), _keyed(reverse)
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_allclose(left[key], right[key], rtol=0.0, atol=1e-10)
    finally:
        env.close()


def test_real_rollout_calls_real_environment_step(monkeypatch):
    env = _make_env()
    calls = {"count": 0}
    original = type(env).step

    def counted(self, action):
        calls["count"] += 1
        return original(self, action)

    monkeypatch.setattr(type(env), "step", counted)
    try:
        rollout = real_candidate_rollout(
            initial_env=env,
            agent_index=0,
            candidate_goal=_proposals(env, limit=1)[0].point,
            policy=DeterministicDummyPolicy(),
            horizon=3,
        )
        assert calls["count"] == rollout.effective_steps == 3
        assert rollout.metadata["execution_semantics"] == "MultiAgentDMPEnv.step"
        assert rollout.metadata["sensor_refresh"] == "real_each_step"
        assert rollout.trajectory_error_is_full_horizon is True
    finally:
        env.close()


def test_horizon_sweep_keeps_initial_state_and_dmp_parameters():
    env = _make_env()
    try:
        policy = DeterministicDummyPolicy()
        proposals = _proposals(env, limit=2)
        before = environment_state_fingerprint(env)
        dmp_configs = [copy.deepcopy(dmp.config) for dmp in env.dmps]
        for horizon in (2, 4, 6):
            evaluate_candidate_set(
                env=env, agent_index=0, proposals=proposals, policy=policy,
                horizon=horizon, consumer_top_k=2,
            )
            assert environment_state_fingerprint(env) == before
            assert [dmp.config for dmp in env.dmps] == dmp_configs
    finally:
        env.close()


def test_score_metrics_and_correlations_are_candidate_set_local():
    rows = [
        {
            "proposal_score": proposal,
            "preview_task_progress": progress,
            "real_task_progress": progress + 0.01,
            "preview_min_clearance": clearance,
            "real_min_clearance": clearance + 0.1,
            "preview_execution_deviation": deviation,
            "real_execution_deviation": deviation + 0.02,
            "preview_terminal_speed": speed,
            "real_terminal_speed": speed + 0.03,
            "preview_runtime_ms": 1.0,
            "mean_trajectory_error": 0.1,
            "terminal_position_error": 0.2,
            "mean_trajectory_error_at_effective_horizon": 0.1,
            "terminal_position_error_at_effective_horizon": 0.2,
            "trajectory_error_is_full_horizon": True,
        }
        for proposal, progress, clearance, deviation, speed in (
            (3.0, 0.3, 1.0, 0.1, 0.2),
            (2.0, 0.2, 0.8, 0.2, 0.3),
            (1.0, 0.1, 0.6, 0.3, 0.4),
        )
    ]
    summary = score_and_summarize_candidate_set(
        rows,
        weights={"progress": 1.0, "clearance": 1.0, "deviation": 1.0, "terminal_speed": 0.0},
        clearance_cap=10.0,
    )
    assert summary["preview_real_spearman"] == 1.0
    assert summary["preview_top1_match"] == 1
    assert summary["preview_top3_hit"] == 1
    assert {row["preview_rank"] for row in rows} == {1, 2, 3}
    correlations = correlation_rows(rows)
    assert len(correlations) == 4
    assert all(row["sample_count"] == 3 for row in correlations)
    assert next(row for row in correlations if row["feature"] == "min_clearance")["clearance_semantics_comparable"] is False


def test_benchmark_does_not_modify_real_checkpoint_actor_parameters():
    reference_env = build_single_env(config=SINGLE_AGENT_CONFIG, action_guidance_enabled=False)
    env = _make_env()
    try:
        model = build_single_model(reference_env, config=SINGLE_AGENT_CONFIG, verbose=0)
        load_checkpoint(model, CHECKPOINT)
        freeze_policy(model)
        before = [parameter.detach().cpu().clone() for parameter in model.actor.parameters()]
        evaluate_candidate_set(
            env=env,
            agent_index=0,
            proposals=_proposals(env, limit=1),
            policy=model,
            horizon=1,
            consumer_top_k=1,
        )
        after = [parameter.detach().cpu() for parameter in model.actor.parameters()]
        assert all(torch.equal(left, right) for left, right in zip(before, after, strict=True))
        assert all(not parameter.requires_grad for parameter in model.actor.parameters())
    finally:
        env.close()
        reference_env.close()
