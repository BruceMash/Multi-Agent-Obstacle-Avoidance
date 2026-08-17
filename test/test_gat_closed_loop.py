from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch


pytest.importorskip("torch_geometric")

from Guidance.reference_point_proposal_demo import Proposal, ProposalConfig
from planning.gat.candidate_selector import batch_candidate_graphs
from planning.gat.stage1_training import load_model_checkpoint
from planning.heterogeneous_candidate_graph import (
    HeterogeneousCandidateGraphConfig,
    build_heterogeneous_candidate_graph_from_env,
)
from planning.pre_gat_220step_revalidation import ImmutableCandidate
from scripts.evaluate_actor_dmp_goal_semantics import run_variant_episode
from scripts.evaluate_gat_closed_loop import (
    METHOD_FP_SHEP,
    METHOD_GAT,
    METHOD_ORDER,
    METHOD_PROPOSAL,
    _assert_frozen_config,
    _make_method_plan,
    _mcnemar_exact,
    aggregate_method_summary,
    build_gat_fp_pairing,
    build_smoke_gate,
    classify_rate_improvement,
    reconstruct_proposals,
)
from scripts.evaluate_pre_gat_closed_loop import build_closed_loop_environment
from scripts.evaluate_single_policy_aligned_multi_agent import (
    build_single_distribution_multi_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/evaluation/gat_closed_loop.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _proposal(index: int = 0) -> Proposal:
    return Proposal(
        azimuth_index=index,
        elevation_index=1,
        direction=np.asarray([1.0, 0.0, 0.0]),
        point=np.asarray([1.0 + index, 2.0, 3.0]),
        distance=1.0,
        raw_obstacle_distance=4.0,
        obstacle_distance=4.0,
        effective_safe_radius=1.0,
        braking_distance=0.1,
        safety_margin=3.0,
        normalized_margin=0.75,
        distance_progress=0.8,
        normalized_progress=0.7,
        alignment=0.9,
        smoothness=0.5,
        usable_length=1.0,
        score=2.0 - index,
    )


def test_frozen_config_contract() -> None:
    config = _config()
    _assert_frozen_config(config)
    assert tuple(config["methods"]) == METHOD_ORDER
    assert config["formal_seeds"] == list(range(10, 30))
    assert config["statistics"]["mcnemar_exact_unit"] == "scenario_seed_team_episode"
    assert config["selection_plan"]["candidate_reordering_allowed"] is False


def test_run_variant_exposes_selection_plan_without_replacing_execution_loop() -> None:
    signature = inspect.signature(run_variant_episode)
    assert "selection_plan" in signature.parameters
    source = inspect.getsource(run_variant_episode)
    assert "env.step(actions)" in source
    assert "selection_plan[\"references\"]" in source
    assert "_select_one_shot_references" in source


def test_proposal_reconstruction_is_field_exact_and_order_preserving() -> None:
    original = (_proposal(0), _proposal(1))
    frozen = tuple(
        ImmutableCandidate.from_proposal(proposal, index)
        for index, proposal in enumerate(original)
    )
    recovered, rows, equivalent = reconstruct_proposals(frozen)
    assert equivalent is True
    assert [item.azimuth_index for item in recovered] == [0, 1]
    assert [item.score for item in recovered] == [2.0, 1.0]
    assert all(row["equivalent"] for row in rows)
    for source, target in zip(original, recovered, strict=True):
        assert np.array_equal(source.point, target.point)
        assert np.array_equal(source.direction, target.direction)
        assert vars(source).keys() == vars(target).keys()


def test_shared_method_plan_preserves_candidate_ids() -> None:
    proposals = ((_proposal(0), _proposal(1)),)

    class Preview:
        def __init__(self, candidate_id: int, score: float) -> None:
            self.candidate_id = candidate_id
            self.score = score

        def to_record(self) -> dict:
            return {"candidate_id": self.candidate_id, "score": self.score}

    previews = ((Preview(0, 0.1), Preview(1, 0.9)),)
    plan = _make_method_plan(
        source="fp_shep_h4",
        terminal_goals=np.asarray([[9.0, 9.0, 9.0]]),
        proposals_by_agent=proposals,
        previews_by_agent=previews,
        selected_ids=[1],
    )
    assert plan["candidate_records"][0]["selected_candidate_id"] == 1
    assert plan["candidate_records"][0]["selected_proposal_rank_1based"] == 2
    assert plan["candidate_records"][0]["selected_fp_shep_rank_1based"] == 1
    assert plan["references"][0] == proposals[0][1].point.tolist()


def test_rate_classification_uses_positive_is_better_and_ten_pp() -> None:
    config = _config()
    assert classify_rate_improvement(0.10, config) == "YES"
    assert classify_rate_improvement(0.099, config) == "WEAK"
    assert classify_rate_improvement(0.0, config) == "NO"
    assert classify_rate_improvement(-0.01, config) == "NO"


def test_mcnemar_uses_paired_team_episode_counts() -> None:
    result = _mcnemar_exact(
        [True, True, False, False],
        [True, False, True, False],
    )
    assert result["left_only"] == 1
    assert result["right_only"] == 1
    assert result["statistical_unit"] == "scenario_seed_team_episode"
    assert result["exact_two_sided_p_value"] == 1.0


def _episode(method: str, success: bool, seed: int = 10) -> dict:
    return {
        "method": method,
        "scenario": "open",
        "seed": seed,
        "team_success": success,
        "obstacle_collision": False,
        "inter_agent_collision": False,
        "collision": False,
        "timeout": not success,
        "completion_step": 20 if success else None,
        "completion_time_s": 2.0 if success else None,
        "team_path_length_m": 4.0,
        "trajectory_smoothness": 0.5,
        "mean_speed_mps": 0.8,
        "peak_speed_mps": 1.0,
        "mean_acceleration_mps2": 0.2,
        "peak_acceleration_mps2": 0.4,
        "minimum_obstacle_clearance_m": 3.0,
        "minimum_inter_agent_distance_m": 2.0,
        "candidate_bundle_hash": "same",
        "preview_bundle_hash": "preview",
    }


def test_continuous_completion_summary_excludes_failed_episode() -> None:
    rows = []
    for method in METHOD_ORDER:
        rows.extend([_episode(method, True, 10), _episode(method, False, 11)])
    summary = aggregate_method_summary(rows)
    gat = next(
        row
        for row in summary
        if row["method"] == METHOD_GAT and row["scenario"] == "overall"
    )
    assert gat["success_completion_time_s_count"] == 1
    assert gat["success_completion_time_s_mean"] == 2.0


def test_gat_fp_pairing_counts_team_outcome_once() -> None:
    episodes = [
        _episode(METHOD_FP_SHEP, False),
        _episode(METHOD_GAT, True),
    ]
    agents = []
    for method, reached in ((METHOD_FP_SHEP, False), (METHOD_GAT, True)):
        for agent_id in range(3):
            agents.append(
                {
                    "method": method,
                    "scenario": "open",
                    "seed": 10,
                    "agent_id": agent_id,
                    "reference_selected_type": "proposal",
                    "selected_candidate_id": 0,
                    "reference_reached": reached,
                    "terminal_completed_after_reference": reached,
                    "candidate_bundle_hash": "same",
                    "preview_bundle_hash": "preview",
                }
            )
    paired, disagreements = build_gat_fp_pairing(episodes, agents)
    assert len(paired) == 1
    assert paired[0]["pair_outcome"] == "GAT_ONLY_SUCCESS"
    assert len(disagreements) == 3


def test_stage1_checkpoint_accepts_null_only_online_graphs() -> None:
    config = _config()
    env, _ = build_closed_loop_environment(
        config=build_single_distribution_multi_config(num_agents=3, max_steps=220),
        scenario="open",
        seed=10,
        peer_radius=0.3,
    )
    try:
        graphs = [
            build_heterogeneous_candidate_graph_from_env(
                env=env,
                agent_index=agent_id,
                proposals=(),
                executions=(),
                proposal_config=ProposalConfig(top_k=10),
                config=HeterogeneousCandidateGraphConfig(horizon_steps=4),
            )
            for agent_id in range(3)
        ]
        stage1 = json.loads(
            (REPO_ROOT / config["stage1_config"]).read_text(encoding="utf-8")
        )
        model = load_model_checkpoint(
            REPO_ROOT / config["gat_checkpoint"], stage1, torch.device("cpu")
        )
        with torch.inference_mode():
            output = model(batch_candidate_graphs(graphs))
        assert output.selected_class_index.tolist() == [0, 0, 0]
        assert output.selected_candidate_id == (None, None, None)
        assert torch.isfinite(output.candidate_logits).all()
    finally:
        env.close()


def test_smoke_gate_accepts_explicit_unbounded_open_space_clearance() -> None:
    config = _config()
    episodes = [
        {
            "scenario": "open",
            "seed": 10,
            "initial_condition_hash": "same",
            "maximum_handoff_count_per_agent": 0,
            "replanning_count": 0,
            "team_path_length_m": 1.0,
            "trajectory_smoothness": 0.0,
            "minimum_obstacle_clearance_m": None,
            "minimum_inter_agent_distance_m": 2.0,
            "execution_historical_gate_verified": True,
            "selection_plan_unchanged": True,
        }
        for _ in range(36)
    ]
    agents = [
        {
            "method": METHOD_GAT,
            "selected_candidate_id": None,
            "K_t": 0,
            "selected_null": True,
            "reference_selected_type": "null",
        }
    ]
    shared = {
        ("open", 10): {
            "gat_diagnostics": [
                {"logits_finite": True, "class_mapping_valid": True}
            ],
            "candidate_hashes_by_method": {
                METHOD_PROPOSAL: "same",
                METHOD_FP_SHEP: "same",
                METHOD_GAT: "same",
            },
            "preview_hashes_by_method": {
                METHOD_FP_SHEP: "preview",
                METHOD_GAT: "preview",
            },
            "proposal_reconstruction_equivalent": True,
            "graph_schema_match": True,
            "preview_historical_gate_verified": True,
        }
    }
    gate = build_smoke_gate(
        config=config,
        episodes=episodes,
        agents=agents,
        shared_cache=shared,
        gat_checkpoint_valid=True,
        code_hash_before={"same": "hash"},
        code_hash_after={"same": "hash"},
    )
    assert gate["status"] == "PASSED"
    assert gate["checks"]["no_nan"] is True
