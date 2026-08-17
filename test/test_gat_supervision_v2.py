from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ALGO = ROOT / "Multi-agent_Algo_lib"
if str(ALGO) not in sys.path:
    sys.path.insert(0, str(ALGO))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from planning.gat_supervision_v2 import (  # noqa: E402
    BACKGROUND_NULL,
    FrozenBackgroundPlan,
    build_tiered_target,
    classify_attribution_stability,
    information_density,
    outcome_tier,
    pairwise_ordering_agreement,
    run_long_horizon_branch,
    stable_hash,
)


CONFIG_PATH = ROOT / "configs/evaluation/gat_supervision_v2.json"


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _outcome(**updates):
    result = {
        "any_collision": False,
        "team_success": False,
        "ego_terminal_reached": False,
        "ego_reference_applicable": True,
        "ego_reference_reached": False,
        "ego_terminal_progress": 1.0,
        "ego_terminal_distance_final": 2.0,
        "reference_to_terminal_progress": None,
        "min_obstacle_clearance": 1.0,
        "min_inter_agent_distance": 1.0,
        "path_length": 3.0,
        "completion_step": None,
        "collision_step": None,
        "reference_handoff_direction_alignment": None,
        "reference_handoff_speed": None,
    }
    result.update(updates)
    return result


def _target(outcomes):
    config = _config()
    tier = config["tier_definition"]
    return build_tiered_target(
        outcomes,
        progress_epsilon=tier["positive_terminal_progress_epsilon_m"],
        within_tier_scale=tier["within_tier_scale"],
        round_decimals=tier["float_comparison_round_decimals"],
        temperature=config["soft_target_temperature"],
    )


def test_final_split_and_absolute_horizon_are_frozen():
    config = _config()
    assert config["effective_split"] == {
        "train": list(range(7)),
        "validation": [7],
        "test": [8, 9],
        "role": "formal_v2_training_consumption",
    }
    assert config["source_split"]["train"] == list(range(6))
    assert config["source_split"]["validation"] == [6, 7]
    assert config["max_absolute_episode_steps"] == 220
    assert config["branch_budget"] == "max_0_220_minus_source_env_steps"
    assert config["fixed_future_220_sensitivity_enabled"] is False


def test_all_post_source_transitions_require_historical_gate():
    gate = _config()["forcing_gate"]
    assert gate["name"] == "historical_vector_goal_eff_gate"
    assert gate["formula"] == "tanh(abs(goal_eff-position))"
    assert all(value is True for key, value in gate.items() if key.startswith("required_for_"))


def test_strict_exclusions_and_stress_isolation():
    config = _config()
    assert not any(config["strict_exclusions"].values())
    assert not any(
        config["stress_isolation"][key]
        for key in (
            "artifact_may_be_read_by_generator",
            "used_for_target_tuning",
            "used_for_tier_tuning",
            "used_for_threshold_tuning",
            "used_for_tau_tuning",
            "included_in_dataset",
            "rerun_allowed",
        )
    )


def test_candidate_reproduction_tolerance_is_strict_and_frozen():
    tolerance = _config()["reproduction_tolerance"]
    assert tolerance["candidate_position_atol"] == 1.0e-10
    assert tolerance["candidate_position_rtol"] == 0.0
    assert tolerance["discrete_exact"] is True


def test_tier_five_to_zero_is_exhaustive_and_collision_has_precedence():
    eps = 1.0e-6
    cases = [
        (_outcome(team_success=True), 5),
        (_outcome(ego_terminal_reached=True), 4),
        (_outcome(ego_reference_reached=True), 3),
        (_outcome(ego_terminal_progress=0.1), 2),
        (_outcome(ego_terminal_progress=0.0), 1),
        (_outcome(any_collision=True, team_success=True), 0),
    ]
    assert [outcome_tier(row, progress_epsilon=eps) for row, _ in cases] == [
        expected for _, expected in cases
    ]


def test_null_branch_can_receive_progress_tier_without_fake_reference():
    result = _outcome(
        ego_reference_applicable=False,
        ego_reference_reached=False,
        ego_terminal_progress=0.5,
    )
    assert outcome_tier(result, progress_epsilon=1.0e-6) == 2


def test_continuous_tie_break_never_crosses_tier():
    worst_safe = _outcome(
        ego_terminal_progress=1.0e-5,
        min_obstacle_clearance=-100.0,
        min_inter_agent_distance=-100.0,
        path_length=1.0e6,
    )
    best_lower = _outcome(
        ego_terminal_progress=0.0,
        min_obstacle_clearance=float("inf"),
        min_inter_agent_distance=float("inf"),
        path_length=0.0,
    )
    target = _target([best_lower, worst_safe])
    assert target.tiers.tolist() == [1, 2]
    assert target.utilities[1] > target.utilities[0]
    assert target.hard_target == 1


def test_collision_branch_is_strictly_below_every_collision_free_branch():
    collision = _outcome(
        any_collision=True,
        collision_step=219,
        ego_terminal_progress=100.0,
        min_obstacle_clearance=float("inf"),
    )
    safe = _outcome(ego_terminal_progress=-100.0)
    target = _target([collision, safe])
    assert target.tiers.tolist() == [0, 1]
    assert target.utilities[1] > target.utilities[0]
    assert target.probabilities[1] > target.probabilities[0]


def test_equal_outcomes_keep_equal_soft_probability_and_variable_k_normalizes():
    equal = _outcome(ego_terminal_progress=0.5)
    target = _target([equal, dict(equal), _outcome(ego_reference_reached=True)])
    assert target.probabilities.shape == (3,)
    assert np.isclose(np.sum(target.probabilities), 1.0)
    assert target.probabilities[0] == target.probabilities[1]


def test_information_density_and_attribution_thresholds_are_pre_result_rules():
    config = _config()
    targets = [
        _target([_outcome(ego_terminal_progress=0.0), _outcome(team_success=True)]),
        _target([_outcome(any_collision=True), _outcome(ego_reference_reached=True)]),
    ]
    density = information_density(targets, config["information_density_gate"])
    assert density["classification"] in {"STRONG", "ADEQUATE", "WEAK", "FAILED"}
    assert classify_attribution_stability(
        {
            "top1_agreement": 0.59,
            "spearman": 1.0,
            "pairwise_agreement": 1.0,
            "tier_agreement": 1.0,
        },
        config["attribution_stability_audit"],
    ) == "WEAK"


def test_information_density_marks_single_class_margin_unavailable():
    config = _config()
    target = _target([_outcome(team_success=True, ego_terminal_reached=True)])

    density = information_density([target], config["information_density_gate"])

    assert density["single_class_graph_count"] == 1
    assert density["top1_top2_margin_graph_count"] == 0
    assert density["mean_top1_top2_utility_margin"] is None
    assert density["median_top1_top2_utility_margin"] is None


def test_pairwise_ordering_agreement_handles_ties_explicitly():
    assert pairwise_ordering_agreement([3, 2, 1], [30, 20, 10]) == 1.0
    assert pairwise_ordering_agreement([1, 1], [2, 2]) == 1.0
    assert pairwise_ordering_agreement([2, 1], [1, 2]) == 0.0


def test_absolute_horizon_branch_preserves_source_and_uses_remaining_budget():
    from planning.historical_forcing_gate import (
        scoped_historical_preview_and_multi_agent_transition,
    )
    from scripts.generate_candidate_supervision_dataset import (
        _build_supervision_environment,
    )
    from scripts.evaluate_single_policy_aligned_multi_agent import (
        build_single_distribution_multi_config,
    )

    class ZeroPolicy:
        def predict(self, observations, deterministic=True):
            batch = np.asarray(observations)
            shape = (6,) if batch.ndim == 1 else (batch.shape[0], 6)
            return np.zeros(shape, dtype=np.float32), None

    multi_config = build_single_distribution_multi_config(num_agents=3, max_steps=2)
    env, _ = _build_supervision_environment(
        config=multi_config,
        scene_type="open",
        seed=0,
        peer_radius=0.3,
    )
    try:
        source_hash = stable_hash(
            {
                "steps": env.steps,
                "positions": env._positions(),
                "velocities": env._velocities(),
            }
        )
        references = np.asarray(env.goals, dtype=float).copy()
        plan = FrozenBackgroundPlan(
            selector=BACKGROUND_NULL,
            references=references,
            available=np.zeros(3, dtype=bool),
            selected_candidate_ids=(None, None, None),
            scores=((), (), ()),
            plan_hash=stable_hash(references),
        )
        calls = []
        with scoped_historical_preview_and_multi_agent_transition(
            execution_observer=lambda kwargs, transition: calls.append(
                transition.controller_info["forcing_gate_semantics"]
            )
        ):
            rollout = run_long_horizon_branch(
                initial_env=env,
                ego_agent_id=0,
                ego_candidate_goal=None,
                background_plan=plan,
                policy=ZeroPolicy(),
                max_absolute_episode_steps=2,
                reference_reached_tolerance_m=0.25,
                class_index=0,
                candidate_id=None,
            )
        assert rollout.outcome["source_env_steps"] == 0
        assert rollout.outcome["remaining_budget_at_source"] == 2
        assert rollout.outcome["final_absolute_env_steps"] <= 2
        assert rollout.outcome["ego_reference_applicable"] is False
        assert calls and set(calls) == {"historical_vector_goal_eff_gate"}
        assert env.steps == 0
        assert source_hash == stable_hash(
            {
                "steps": env.steps,
                "positions": env._positions(),
                "velocities": env._velocities(),
            }
        )
    finally:
        env.close()


def test_formal_generator_declares_every_required_artifact_and_no_stress_read():
    from scripts import generate_gat_supervision_v2 as generator

    source = Path(generator.__file__).read_text(encoding="utf-8")
    required = [
        "config.json",
        "integrity_manifest.json",
        "semantic_contract.json",
        "candidate_reproduction.csv",
        "historical_h4_graph_audit.csv",
        "branch_rollout_manifest.csv",
        "branch_raw_outcomes.parquet",
        "supervision_v2_dataset.pt",
        "supervision_v2_metadata.json",
        "target_v1_scalar_h6.csv",
        "target_v1_historical_h6.csv",
        "target_v2_long_horizon.csv",
        "target_agreement.csv",
        "label_information_density.json",
        "attribution_stability.csv",
        "validity_gate.json",
        "conclusion.json",
        "FINAL_REPORT.md",
    ]
    assert all(name in source for name in required)
    assert "prohibited_stress_dir" not in inspect.getsource(generator.run_generation)


def test_generator_has_no_training_or_optimizer_call():
    from scripts import generate_gat_supervision_v2 as generator

    source = inspect.getsource(generator)
    assert ".backward(" not in source
    assert "Adam(" not in source
    assert "AdamW(" not in source
    assert "train_stage1(" not in source
