from __future__ import annotations

import ast
import hashlib
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

from planning.geometry_generalization_scenarios import (  # noqa: E402
    FROZEN_HELD_OUT_LAYOUTS,
    FROZEN_LAYOUT_TABLE_SHA256,
    LAYOUT_SET_ROLE,
    SCENARIO_ID,
    SCENARIO_ROLE,
    analytic_geometry_gate,
    build_scenario_manifest,
    relative_geometry_record,
    stable_hash,
    validate_manifest_geometry,
)
from planning.multi_agent_obstacle_scenario_audit import (  # noqa: E402
    build_multi_agent_obstacle_options,
)
from planning.pre_gat_220step_revalidation import (  # noqa: E402
    METHOD_ORDER,
)
from planning.pre_gat_closed_loop import FPSHEPOnlineScoreSpec  # noqa: E402
from scripts.evaluate_geometry_generalization_audit import (  # noqa: E402
    _assert_manifest_frozen,
    build_generalization_environment,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_single_policy_multi_agent import build_policy_observations  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs/evaluation/geometry_generalization_audit.json"
SCENARIO_MODULE = REPO_ROOT / "planning/geometry_generalization_scenarios.py"
RUNNER_PATH = (
    REPO_ROOT
    / "Multi-agent_Algo_lib/scripts/evaluate_geometry_generalization_audit.py"
)
EXPECTED_TABLE_HASH = "4567cd11511edb3bdeceaa8c79ec8109e6da996fc5545ed0b216c5d8f5a0c3c7"


def _legacy_relative_hash() -> str:
    options = build_multi_agent_obstacle_options(0)
    return stable_hash(
        relative_geometry_record(
            np.asarray(options["starts"], dtype=float),
            np.asarray(options["goals"], dtype=float),
            options["static_obstacles"],
        )
    )


def test_01_frozen_table_has_24_explicit_family_balanced_layouts() -> None:
    assert len(FROZEN_HELD_OUT_LAYOUTS) == 24
    assert FROZEN_LAYOUT_TABLE_SHA256 == EXPECTED_TABLE_HASH
    assert [item.layout_id for item in FROZEN_HELD_OUT_LAYOUTS] == [
        f"HOLDOUT_{family}_{index:02d}"
        for family in "ABCDEF"
        for index in range(4)
    ]
    assert all(
        sum(item.family == family for item in FROZEN_HELD_OUT_LAYOUTS) == 4
        for family in "ABCDEF"
    )


def test_02_manifest_is_deterministic_complete_and_non_translation_equivalent() -> None:
    first = build_scenario_manifest()
    second = build_scenario_manifest()
    assert stable_hash(first) == stable_hash(second)
    audit = validate_manifest_geometry(
        first, legacy_relative_hashes=[_legacy_relative_hash()]
    )
    assert audit["status"] == "PASSED", audit["failed_checks"]
    assert audit["layout_count"] == 24
    assert audit["relative_geometry_variant_count"] == 24
    assert audit["checks"]["none_matches_legacy_relative_geometry"] is True
    for row in first["layouts"]:
        assert row["layout_id"]
        assert row["family"] in "ABCDEF"
        assert np.asarray(row["starts"]).shape == (3, 3)
        assert np.asarray(row["terminal_goals"]).shape == (3, 3)
        assert len(row["obstacles"]) == 2
        assert row["geometry_descriptors"]
        assert len(row["layout_hash"]) == 64
        assert len(row["relative_geometry_hash"]) == 64


def test_03_all_geometry_gates_use_only_analytic_initial_geometry_and_pass() -> None:
    for layout in FROZEN_HELD_OUT_LAYOUTS:
        gate = analytic_geometry_gate(layout)
        assert gate["status"] == "PASSED", (layout.layout_id, gate["failed_checks"])
        assert gate["method_outcome_used"] is False
        assert gate["checks"]["exactly_two_static_obstacles"] is True
        assert gate["checks"]["zero_dynamic_obstacles"] is True
        assert gate["checks"]["peer_spheres_enabled"] is True
        assert gate["checks"]["boundary_free"] is True
        assert gate["checks"]["analytic_unbounded_free_space_exists"] is True
        assert "no planner, policy, rollout, or method outcome" in gate[
            "solvability_statement"
        ]


def test_04_family_c_has_real_3d_spatiotemporal_pressure() -> None:
    family_c = [item for item in FROZEN_HELD_OUT_LAYOUTS if item.family == "C"]
    assert len(family_c) == 4
    for layout in family_c:
        gate = analytic_geometry_gate(layout)
        descriptor = gate["descriptor"]
        assert gate["checks"]["family_c_actual_3d_spatiotemporal_pressure"] is True
        assert abs(float(descriptor["near_orthogonal_route_angle_3d_deg"]) - 90.0) <= 15.0
        assert float(descriptor["near_orthogonal_pair_predicted_minimum_distance_m"]) <= 0.6
        assert float(descriptor["near_orthogonal_pair_time_difference_fraction"]) <= 0.08
        assert float(descriptor["near_orthogonal_pair_vertical_separation_m"]) <= 0.6
        assert "near_orthogonal_route_angle_xy_deg" in descriptor
        pair_rows = descriptor["route_pair_descriptors"]
        assert all("route_angle_3d_deg" in row for row in pair_rows)
        assert all("route_angle_xy_deg" in row for row in pair_rows)
        assert all(
            "predicted_closest_approach_time_fraction" in row for row in pair_rows
        )
        assert all(
            "predicted_closest_approach_time_difference_fraction" in row
            for row in pair_rows
        )


def test_05_scenario_module_has_no_policy_or_outcome_dependency() -> None:
    tree = ast.parse(SCENARIO_MODULE.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    forbidden = (
        "Guidance.reference_point_proposal_demo",
        "planning.policy_preview",
        "baseline.sac",
        "runner_sac",
        "heterogeneous",
        "GAT",
    )
    assert not any(
        name == item or name.startswith(item + ".")
        for name in imported
        for item in forbidden
    )
    source = SCENARIO_MODULE.read_text(encoding="utf-8")
    assert "method_outcome_used\": False" in source
    assert "analytic_unbounded_free_space_exists" in source


def test_06_manifest_freeze_detects_any_byte_change(tmp_path: Path) -> None:
    path = tmp_path / "scenario_manifest.json"
    payload = json.dumps(build_scenario_manifest(), ensure_ascii=False, indent=2)
    path.write_text(payload, encoding="utf-8")
    expected_bytes = path.read_bytes()
    expected_hash = hashlib.sha256(expected_bytes).hexdigest()
    _assert_manifest_frozen(path, expected_hash, expected_bytes)
    path.write_text(payload + "\n", encoding="utf-8")
    try:
        _assert_manifest_frozen(path, expected_hash, expected_bytes)
    except RuntimeError as exc:
        assert "manifest" in str(exc)
    else:
        raise AssertionError("manifest mutation was not detected")


def test_07_all_environment_builds_preserve_peer_and_observation_contracts() -> None:
    config = build_single_distribution_multi_config(num_agents=3, max_steps=220)
    for index, layout in enumerate(FROZEN_HELD_OUT_LAYOUTS):
        env, metadata = build_generalization_environment(
            config=config,
            scenario=SCENARIO_ID,
            seed=1000 + index,
            peer_radius=0.3,
        )
        try:
            assert metadata["layout_id"] == layout.layout_id
            assert metadata["family"] == layout.family
            assert metadata["scenario_role"] == SCENARIO_ROLE
            assert metadata["peer_spheres_enabled"] is True
            assert metadata["include_boundaries_in_sensor"] is False
            assert metadata["terminate_on_boundary_collision"] is False
            assert metadata["static_obstacle_count"] == 2
            assert metadata["dynamic_obstacle_count"] == 0
            assert build_policy_observations(env).shape == (3, 122)
            assert env.get_observation().shape == (3, 138)
            assert len(env.static_obstacles) == 2
            assert len(env.dynamic_obstacles) == 0
        finally:
            env.close()


def test_08_config_freezes_execution_and_exclusion_contracts() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["scenario"] == SCENARIO_ID
    assert config["scenario_role"] == SCENARIO_ROLE
    assert config["layout_set_role"] == LAYOUT_SET_ROLE
    assert config["frozen_layout_table_sha256_expected"] == EXPECTED_TABLE_HASH
    assert config["layout_count"] == 24
    assert config["families"] == list("ABCDEF")
    assert tuple(config["methods"]) == METHOD_ORDER
    assert len(config["stage_2_integrity_layouts"]) <= 4
    assert config["max_steps"] == 220
    assert config["temporary_reference"]["reached_tolerance_m"] == 0.25
    assert config["execution_semantics"]["historical_gate"] == (
        "historical_vector_goal_eff_gate"
    )
    assert config["execution_semantics"]["one_shot"] is True
    assert config["execution_semantics"]["repeated_replanning"] is False
    assert config["execution_semantics"]["include_boundaries_in_sensor"] is False
    assert config["execution_semantics"]["terminate_on_boundary_collision"] is False
    assert config["geometry_gate"]["method_outcome_allowed"] is False
    assert config["manifest_freeze"]["freeze_before_checkpoint_load"] is True
    assert config["manifest_freeze"]["outcome_based_geometry_modification"] is False
    assert not any(config["strict_exclusions"].values())
    score = FPSHEPOnlineScoreSpec.from_mapping(config["fp_shep_online_selector"])
    assert score.horizon == 4
    assert score.terminal_speed_weight == 0.0


def test_09_runner_orders_manifest_freeze_before_checkpoint_load() -> None:
    source = inspect.getsource(
        __import__(
            "scripts.evaluate_geometry_generalization_audit",
            fromlist=["run_experiment"],
        ).run_experiment
    )
    assert source.index("build_scenario_manifest") < source.index("_load_policy")
    assert source.index("write_json(manifest_path") < source.index("_load_policy")
    assert source.index("_assert_manifest_frozen") < source.index("_load_policy")
    assert "outcome_based_layout_rejection_count\": 0" in source
    assert "GAT_inference_count\": 0" in source
    assert "SAC_fine_tuning_performed\": False" in source


def test_10_stage2_outcomes_are_not_integrity_or_geometry_criteria() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"outcome_not_used_for_integrity_gate": True' in source
    assert '"method_outcomes_used_in_integrity_gate": False' in source
    assert '"method_outcomes_used_to_modify_geometry": False' in source
    stage2_position = source.index('run_layout_block(stage_2_ids, "stage_2_integrity_subset")')
    stage3_position = source.index('run_layout_block(remaining_ids, "stage_3_full_frozen_set")')
    assert stage2_position < stage3_position
    integrity_function = source[
        source.index("def _stage_integrity(") : source.index("def _render_report(")
    ]
    assert "team_success" not in integrity_function
    assert "obstacle_collision" not in integrity_function
    assert "inter_agent_collision" not in integrity_function


def test_11_layout_table_and_manifest_are_held_out_not_training_registered() -> None:
    manifest = build_scenario_manifest()
    assert manifest["layout_set_role"] == LAYOUT_SET_ROLE
    assert manifest["geometry_frozen_before_method_evaluation"] is True
    assert manifest["outcome_based_rejection_forbidden"] is True
    assert all(row["layout_set_role"] == LAYOUT_SET_ROLE for row in manifest["layouts"])
    assert all(row["evaluation_seed"] >= 1000 for row in manifest["layouts"])
    from scripts.validate_policy_preview import SCENE_ADAPTERS

    assert SCENARIO_ID not in SCENE_ADAPTERS
