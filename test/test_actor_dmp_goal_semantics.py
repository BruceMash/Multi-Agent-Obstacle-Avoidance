from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.goal_semantics_diagnosis import (  # noqa: E402
    VARIANT_A,
    VARIANT_B,
    VARIANT_C,
    VARIANT_D,
    VARIANT_ORDER,
    temporary_checkpoint_observations,
    terminal_checkpoint_observations,
)
from planning.temporary_reference_diagnosis import (  # noqa: E402
    PROTOCOL_ONE_SHOT,
    PROTOCOL_TERMINAL,
)
from scripts.evaluate_actor_dmp_goal_semantics import (  # noqa: E402
    _load_policy,
    run_variant_episode,
)
from scripts.evaluate_frozen_policy_waypoint_guidance import (  # noqa: E402
    build_active_goal_observations,
)
from scripts.evaluate_pre_gat_closed_loop import (  # noqa: E402
    _policy_parameter_sha256,
    build_closed_loop_environment,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import (  # noqa: E402
    run_one_shot_episode,
    run_pre_gat_protocol,
)


CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "evaluation"
    / "actor_dmp_goal_semantics_diagnosis.json"
)
LEGACY_CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "evaluation"
    / "temporary_reference_interface_diagnosis.json"
)
EVALUATOR_PATH = (
    ALGO_ROOT / "scripts" / "evaluate_actor_dmp_goal_semantics.py"
)


@pytest.fixture(scope="module")
def mini_diagnosis() -> dict:
    settings = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    settings = copy.deepcopy(settings)
    settings["max_steps"] = 3
    multi_config = build_single_distribution_multi_config(
        num_agents=int(settings["num_agents"]), max_steps=3
    )
    policy, _ = _load_policy(settings, multi_config)
    policy_hash_before = _policy_parameter_sha256(policy)
    records = {}
    actor_rows = {}
    dmp_rows = {}
    for variant in VARIANT_ORDER:
        episode, actor_shift, dmp_shift = run_variant_episode(
            policy=policy,
            multi_config=multi_config,
            settings=settings,
            scenario="open",
            seed=0,
            variant=variant,
        )
        records[variant] = episode
        actor_rows[variant] = actor_shift
        dmp_rows[variant] = dmp_shift

    legacy_settings = json.loads(LEGACY_CONFIG_PATH.read_text(encoding="utf-8"))
    legacy_settings["max_steps"] = 3
    legacy_a, _, _, _, _ = run_pre_gat_protocol(
        policy=policy,
        multi_config=multi_config,
        settings=legacy_settings,
        scenario="open",
        seed=0,
        protocol=PROTOCOL_TERMINAL,
    )
    legacy_d, legacy_events, _, _, _ = run_one_shot_episode(
        policy=policy,
        multi_config=multi_config,
        settings=legacy_settings,
        scenario="open",
        seed=0,
    )
    return {
        "settings": settings,
        "multi_config": multi_config,
        "policy": policy,
        "records": records,
        "actor_rows": actor_rows,
        "dmp_rows": dmp_rows,
        "legacy_a": legacy_a,
        "legacy_d": legacy_d,
        "legacy_events": legacy_events,
        "policy_hash_before": policy_hash_before,
        "policy_hash_after": _policy_parameter_sha256(policy),
    }


def test_variant_a_uses_terminal_actor_and_terminal_dmp(mini_diagnosis):
    row = mini_diagnosis["records"][VARIANT_A]
    legacy = mini_diagnosis["legacy_a"]
    assert row["actor_goal_semantics"] == "terminal"
    assert row["dmp_goal_semantics"] == "terminal"
    assert row["temporary_references"] is None
    assert row["steps"] == legacy["steps"]
    assert row["collision"] == legacy["collision"]
    assert row["success"] == legacy["success"]
    assert row["path_length_team_mean_m"] == pytest.approx(
        legacy["path_length_team_mean_m"], abs=1.0e-10
    )


def test_variant_b_temporary_goal_changes_actor_observation_only(mini_diagnosis):
    row = mini_diagnosis["records"][VARIANT_B]
    shifts = mini_diagnosis["actor_rows"][VARIANT_B]
    assert row["actor_goal_semantics"] == "temporary"
    assert row["dmp_goal_semantics"] == "terminal"
    assert shifts
    assert max(item["non_goal_feature_l2_delta"] for item in shifts) == 0.0
    assert any(item["observation_l2_delta"] > 0.0 for item in shifts)


def test_variant_b_never_writes_temporary_reference_to_dmp_goal(mini_diagnosis):
    row = mini_diagnosis["records"][VARIANT_B]
    assert np.array_equal(
        np.asarray(row["final_dmp_goals"]),
        np.asarray(row["temporary_references"])
    ) is False
    env, _ = build_closed_loop_environment(
        config=mini_diagnosis["multi_config"],
        scenario="open",
        seed=0,
        peer_radius=0.3,
    )
    try:
        assert np.array_equal(np.asarray(row["final_dmp_goals"]), env.goals)
    finally:
        env.close()


def test_variant_c_actor_input_is_terminal_conditioned_122d(mini_diagnosis):
    env, _ = build_closed_loop_environment(
        config=mini_diagnosis["multi_config"],
        scenario="open",
        seed=0,
        peer_radius=0.3,
    )
    try:
        explicit_terminal = terminal_checkpoint_observations(env)
        established_terminal = build_active_goal_observations(env, np.asarray(env.goals))
        temporary = temporary_checkpoint_observations(
            env, np.asarray(mini_diagnosis["records"][VARIANT_C]["temporary_references"])
        )
        assert explicit_terminal.shape == (3, 122)
        assert np.array_equal(explicit_terminal, established_terminal)
        assert not np.array_equal(explicit_terminal, temporary)
        assert env.get_observation().shape == (3, 138)
    finally:
        env.close()


def test_variant_c_temporary_reference_affects_only_dmp_attractor(mini_diagnosis):
    row = mini_diagnosis["records"][VARIANT_C]
    assert row["actor_goal_semantics"] == "terminal"
    assert row["dmp_goal_semantics"] == "temporary"
    assert not mini_diagnosis["actor_rows"][VARIANT_C]
    assert mini_diagnosis["dmp_rows"][VARIANT_C]


def test_variant_d_matches_existing_one_shot_semantics(mini_diagnosis):
    row = mini_diagnosis["records"][VARIANT_D]
    legacy = mini_diagnosis["legacy_d"]
    assert row["actor_goal_semantics"] == "temporary"
    assert row["dmp_goal_semantics"] == "temporary"
    assert row["steps"] == legacy["steps"]
    assert row["collision"] == legacy["collision"]
    assert row["success"] == legacy["success"]
    assert row["path_length_team_mean_m"] == pytest.approx(
        legacy["path_length_team_mean_m"], abs=1.0e-10
    )
    legacy_points = {
        int(event["agent_id"]): np.asarray(event["reference_point"])
        for event in mini_diagnosis["legacy_events"]
    }
    for agent_id, available in enumerate(row["temporary_reference_available"]):
        if available:
            assert np.array_equal(
                np.asarray(row["temporary_references"])[agent_id],
                legacy_points[agent_id],
            )


def test_env_terminal_goals_remain_unchanged_for_all_variants(mini_diagnosis):
    assert all(
        mini_diagnosis["records"][variant]["terminal_task_goals_unchanged"]
        for variant in VARIANT_ORDER
    )


def test_reference_switches_preserve_phase(mini_diagnosis):
    assert all(
        mini_diagnosis["records"][variant]["phase_reset_on_switch"] is False
        and mini_diagnosis["records"][variant]["maximum_phase_switch_delta"] == 0.0
        for variant in VARIANT_ORDER
    )


def test_forcing_gate_remains_terminal_goal_based(mini_diagnosis):
    rows = mini_diagnosis["dmp_rows"][VARIANT_C]
    assert rows
    assert all(
        item["forcing_gate_terminal_nominal"]
        == item["forcing_gate_temporary_nominal"]
        and item["forcing_gate_terminal_commanded"]
        == item["forcing_gate_temporary_commanded"]
        for item in rows
    )


def test_actor_and_checkpoint_state_are_unchanged(mini_diagnosis):
    assert mini_diagnosis["policy_hash_before"] == mini_diagnosis["policy_hash_after"]
    assert all(
        parameter.requires_grad is False
        for parameter in mini_diagnosis["policy"].actor.parameters()
    )


def test_evaluator_contains_no_training_path():
    source = EVALUATOR_PATH.read_text(encoding="utf-8")
    assert ".learn(" not in source
    assert ".fit(" not in source
    assert "optimizer" not in source.lower()
    assert "backward(" not in source


def test_evaluator_imports_no_gat_or_graph_builder():
    tree = ast.parse(EVALUATOR_PATH.read_text(encoding="utf-8"))
    modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    names = modules | imported
    forbidden_fragments = (
        "edge_enhanced_gat",
        "gat_forward",
        "heterogeneous_candidate_graph",
    )
    assert not any(
        fragment in name.lower()
        for name in names
        for fragment in forbidden_fragments
    )


def test_evaluator_uses_no_fp_shep_selector():
    tree = ast.parse(EVALUATOR_PATH.read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "METHOD_FP_SHEP" not in imported_names
    assert "score_fp_shep_candidates" not in imported_names
    assert all(
        mini is False
        for mini in json.loads(CONFIG_PATH.read_text(encoding="utf-8"))[
            "strict_exclusions"
        ].values()
    )
