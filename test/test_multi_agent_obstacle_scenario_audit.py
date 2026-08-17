from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
SCRIPTS_ROOT = ALGO_ROOT / "scripts"
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.multi_agent_obstacle_scenario_audit import (  # noqa: E402
    FROZEN_LAYOUT_HASH,
    FROZEN_LAYOUT_SPEC,
    SCENARIO_ID,
    SCENARIO_ROLE,
    SEED_SET_ROLE,
    build_multi_agent_obstacle_environment,
    build_multi_agent_obstacle_options,
    geometry_record,
    stable_hash,
    summarize_geometry,
)
from scripts.evaluate_actor_dmp_goal_semantics import run_variant_episode  # noqa: E402
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.validate_policy_preview import SCENE_ADAPTERS  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs/evaluation/multi_agent_obstacle_scenario_audit.json"
EXPECTED_LAYOUT_HASH = "8deca19622f9578adfc68e167ba0e1dc4aacee5943333a502853a5cea78fb1e1"


def _payload(options):
    return {
        "starts": np.asarray(options["starts"]).tolist(),
        "goals": np.asarray(options["goals"]).tolist(),
        "obstacles": [
            {
                "center": np.asarray(item.center).tolist(),
                "radius": float(item.radius),
                "safety_margin": float(item.safety_margin),
            }
            for item in options["static_obstacles"]
        ],
    }


def test_frozen_layout_hash_is_explicit_and_stable() -> None:
    assert FROZEN_LAYOUT_HASH == EXPECTED_LAYOUT_HASH
    assert stable_hash(FROZEN_LAYOUT_SPEC) == EXPECTED_LAYOUT_HASH
    assert FROZEN_LAYOUT_SPEC["geometry_frozen_before_method_results"] is True
    assert FROZEN_LAYOUT_SPEC["closed_loop_result_tuning_forbidden"] is True


def test_scenario_is_seed_deterministic_and_has_two_static_obstacles() -> None:
    for seed in range(10):
        first = build_multi_agent_obstacle_options(seed)
        second = build_multi_agent_obstacle_options(seed)
        assert stable_hash(_payload(first)) == stable_hash(_payload(second))
        assert len(first["static_obstacles"]) == 2
        assert first["dynamic_obstacles"] == []


def test_geometry_gate_passes_for_confirmed_development_seeds() -> None:
    config = build_single_distribution_multi_config(num_agents=3, max_steps=220)
    records = []
    for seed in range(10):
        env, metadata = build_multi_agent_obstacle_environment(
            config=config,
            scenario=SCENARIO_ID,
            seed=seed,
            peer_radius=0.3,
        )
        try:
            record = geometry_record(
                env,
                scenario=SCENARIO_ID,
                seed=seed,
                obstacle_influence_distance=1.5,
                inter_agent_risk_distance=0.6,
            )
            assert metadata["scenario_role"] == SCENARIO_ROLE
            assert metadata["seed_set_role"] == SEED_SET_ROLE
            assert record["checkpoint_actor_observation_shape"] == [3, 122]
            assert record["native_observation_shape"] == [3, 138]
            assert record["peer_spheres_enabled"] is True
            assert record["include_boundaries_in_sensor"] is False
            assert record["terminate_on_boundary_collision"] is False
            records.append(record)
        finally:
            env.close()
    gate = summarize_geometry(records, required_static_obstacle_count=2)
    assert gate["status"] == "PASSED"
    assert gate["checks"]["no_initial_obstacle_collision"] is True
    assert gate["checks"]["no_initial_inter_agent_collision"] is True
    assert gate["checks"]["goals_feasible"] is True
    assert gate["pressure_episode_count"] == 10
    assert gate["predicted_conflict_episode_count"] == 10


def test_new_diagnostic_does_not_replace_existing_scene_adapters() -> None:
    assert all(name in SCENE_ADAPTERS for name in ("open", "sparse_static", "multi_agent"))
    assert SCENARIO_ID not in SCENE_ADAPTERS


def test_shared_episode_runner_keeps_default_builder_and_adds_optional_factory() -> None:
    parameter = inspect.signature(run_variant_episode).parameters["environment_builder"]
    assert parameter.default is None


def test_config_freezes_diagnostic_scope_and_strict_exclusions() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["scenario"] == SCENARIO_ID
    assert config["scenario_role"] == SCENARIO_ROLE
    assert config["frozen_layout_hash_expected"] == EXPECTED_LAYOUT_HASH
    assert config["seed_set_role"] == SEED_SET_ROLE
    assert config["held_out_required_for_future_paper_comparison"] is True
    assert config["stage_a_seeds"] == list(range(5))
    assert config["stage_b_seeds"] == list(range(5, 10))
    assert config["max_steps"] == 220
    assert config["execution_semantics"]["actor_observation_shape"] == [3, 122]
    assert config["execution_semantics"]["historical_gate"] == "historical_vector_goal_eff_gate"
    assert config["execution_semantics"]["one_shot"] is True
    assert config["execution_semantics"]["repeated_replanning"] is False
    assert config["execution_semantics"]["include_boundaries_in_sensor"] is False
    assert config["execution_semantics"]["terminate_on_boundary_collision"] is False
    assert config["fp_shep_online_selector"]["H_preview"] == 4
    assert config["fp_shep_online_selector"]["weights"]["terminal_speed"] == 0.0
    assert not any(config["strict_exclusions"].values())


def test_geometry_has_clear_crossing_but_safe_initial_spacing() -> None:
    options = build_multi_agent_obstacle_options(0)
    starts = np.asarray(options["starts"], dtype=float)
    goals = np.asarray(options["goals"], dtype=float)
    crossing_positions = starts + 0.5 * (goals - starts)
    np.testing.assert_allclose(crossing_positions[0], crossing_positions[1])
    initial_distances = np.linalg.norm(starts[:, None] - starts[None, :], axis=-1)
    assert np.min(initial_distances[np.triu_indices(3, k=1)]) > 0.6


def test_confirmed_seed_jitter_is_common_translation_only() -> None:
    relative_hashes = []
    for seed in range(10):
        options = build_multi_agent_obstacle_options(seed)
        starts = np.asarray(options["starts"], dtype=float)
        goals = np.asarray(options["goals"], dtype=float)
        anchor = 0.5 * (starts[0] + goals[0])
        relative_starts = np.round(starts - anchor, 12)
        relative_goals = np.round(goals - anchor, 12)
        relative_starts[np.abs(relative_starts) < 1e-12] = 0.0
        relative_goals[np.abs(relative_goals) < 1e-12] = 0.0
        relative_obstacles = []
        for item in options["static_obstacles"]:
            relative = np.round(np.asarray(item.center) - anchor, 12)
            relative[np.abs(relative) < 1e-12] = 0.0
            relative_obstacles.append(relative.tolist())
        relative_hashes.append(
            stable_hash(
                {
                    "starts": relative_starts.tolist(),
                    "goals": relative_goals.tolist(),
                    "obstacles": relative_obstacles,
                }
            )
        )
    assert len(set(relative_hashes)) == 1
