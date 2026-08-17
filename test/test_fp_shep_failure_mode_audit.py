from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "Multi-agent_Algo_lib"
    / "scripts"
    / "evaluate_fp_shep_failure_mode_audit.py"
)
CONFIG_PATH = REPO_ROOT / "configs/evaluation/fp_shep_failure_mode_audit.json"
SOURCE_ROOT = REPO_ROOT / "artifacts/geometry_generalization_audit/20260815_175228"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.fp_shep_failure_mode_analysis import (  # noqa: E402
    UNREACHABLE_PRIMARY_CATEGORIES,
    build_mechanism_conclusion,
    candidate_geometry_record,
    classify_reference_unreachable,
    score_margin_record,
)
from planning.reference_transition_finetuning import sha256_file  # noqa: E402


def _load_script():
    spec = importlib.util.spec_from_file_location("failure_audit_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _thresholds():
    return _config()["diagnostic_thresholds"]


def _trace(
    distances,
    *,
    saturation=0.0,
    static_clearance=3.0,
    peer_distance=3.0,
    speed=0.5,
):
    rows = []
    for step, distance in enumerate(distances, start=1):
        rows.append(
            {
                "step": step,
                "distance_to_reference_m": float(distance),
                "action_saturation_fraction": float(saturation),
                "forcing_saturation_fraction": float(saturation),
                "goal_offset_saturation_fraction": float(saturation),
                "speed_mps": float(speed),
                "static_obstacle_clearance_m": float(static_clearance),
                "minimum_inter_agent_distance_m": float(peer_distance),
            }
        )
    return rows


def test_frozen_manifest_checkpoint_and_formal_h4_contract():
    config = _config()
    assert sha256_file(SOURCE_ROOT / "scenario_manifest.json") == config[
        "source_manifest_sha256_expected"
    ]
    assert sha256_file(REPO_ROOT / config["checkpoint"]) == config[
        "checkpoint_sha256_expected"
    ]
    assert config["formal_selector"]["H_preview"] == 4
    assert config["post_hoc_preview"]["selected_candidate_horizons"] == [4, 8, 12, 20]
    assert config["formal_selector"]["weights"]["terminal_speed"] == 0.0
    assert config["post_hoc_preview"]["used_for_online_selection"] is False


def test_config_freezes_all_diagnostic_thresholds_and_better_alternative_is_not_established():
    config = _config()
    assert config["post_hoc_preview"]["better_preview_alternative_criterion"].startswith(
        "NOT_ESTABLISHED"
    )
    assert config["post_hoc_preview"]["termination_check"][
        "stop_after_first_termination"
    ]
    assert config["diagnostic_thresholds"]["trace_window_steps"] == 20
    assert not any(config["strict_exclusions"].values())


def test_score_margin_uses_frozen_range_normalization():
    row = {
        "layout_id": "L",
        "family": "A",
        "agent_id": "0",
        "fp_shep_candidate_records": json.dumps(
            [
                {"fp_shep_online_score": 2.0},
                {"fp_shep_online_score": 1.8},
                {"fp_shep_online_score": 1.0},
            ]
        ),
    }
    result = score_margin_record(row, epsilon=1.0e-12)
    assert result["margin_1_2"] == pytest.approx(0.2)
    assert result["margin_1_3"] == pytest.approx(1.0)
    assert result["normalized_margin_1_2"] == pytest.approx(0.2)


def test_unreachable_time_budget_only_requires_continuing_progress():
    result = classify_reference_unreachable(
        _trace(np.linspace(3.0, 2.0, 20)), _thresholds()
    )
    assert result["primary_subcategory"] == "TIME_BUDGET_ONLY"
    assert "STAGNATION" not in result["diagnostic_flags"]


def test_unreachable_stagnation_is_mutually_exclusive_primary():
    result = classify_reference_unreachable(
        _trace(np.linspace(3.0, 2.98, 20)), _thresholds()
    )
    assert result["primary_subcategory"] == "STAGNATION"
    assert result["primary_subcategory"] in UNREACHABLE_PRIMARY_CATEGORIES


def test_obstacle_deadlock_and_inter_agent_blocking_use_explicit_pressure():
    obstacle = classify_reference_unreachable(
        _trace(np.linspace(3.0, 2.99, 20), static_clearance=0.4), _thresholds()
    )
    peer = classify_reference_unreachable(
        _trace(
            np.linspace(3.0, 2.99, 20),
            static_clearance=3.0,
            peer_distance=0.8,
        ),
        _thresholds(),
    )
    assert obstacle["primary_subcategory"] == "OBSTACLE_AVOIDANCE_DEADLOCK"
    assert peer["primary_subcategory"] == "INTER_AGENT_BLOCKING"


def test_action_saturation_is_not_called_lower_policy_when_blocking_dominates():
    result = classify_reference_unreachable(
        _trace(
            np.linspace(3.0, 2.99, 20),
            saturation=0.9,
            static_clearance=0.4,
        ),
        _thresholds(),
    )
    assert result["primary_subcategory"] == "OBSTACLE_AVOIDANCE_DEADLOCK"
    assert "ACTION_SATURATION" in result["diagnostic_flags"]


def test_candidate_geometry_is_post_hoc_and_detects_blocked_handoff_segment():
    layout = {
        "layout_id": "L",
        "family": "A",
        "starts": [[0.0, 0.0, 0.0]],
        "terminal_goals": [[4.0, 0.0, 0.0]],
        "obstacles": [
            {
                "center": [2.5, 0.0, 0.0],
                "radius_m": 0.45,
                "safety_margin_m": 0.1,
                "effective_radius_m": 0.55,
            },
            {
                "center": [2.5, 2.0, 0.0],
                "radius_m": 0.45,
                "safety_margin_m": 0.1,
                "effective_radius_m": 0.55,
            },
        ],
    }
    result = candidate_geometry_record(
        layout=layout,
        agent_id=0,
        candidate=np.asarray([2.0, 0.0, 0.0]),
        proposal_rank=0,
        thresholds=_thresholds(),
    )
    assert result["candidate_to_terminal_minimum_static_clearance_m"] < 0.0
    assert result["geometrically_poor_handoff_candidate"]
    assert result["geometry_used_for_selection"] is False


def test_stopping_distance_is_secondary_and_unavailable_without_deceleration():
    module = _load_script()
    value, available = module._kinematic_stopping_distance(
        np.asarray([1.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0])
    )
    assert value is None
    assert available is False


def test_preview_termination_uses_static_collision_and_terminal_only():
    module = _load_script()

    class Obstacle:
        def contains(self, position, margin=0.0):
            return float(position[0]) <= 0.0

    class Config:
        collision_margin = 0.1
        goal_tolerance = 0.25

    class Env:
        env_config = Config()
        static_obstacles = [Obstacle()]

    assert module._preview_termination(Env(), np.zeros(3), np.ones(3)) == (
        True,
        "static_obstacle_collision",
    )
    assert module._preview_termination(
        Env(), np.asarray([2.0, 0.0, 0.0]), np.asarray([2.1, 0.0, 0.0])
    ) == (True, "terminal_success")


def test_reproduction_gate_detects_substantive_outcome_mismatch():
    module = _load_script()
    rerun = {
        "layout_id": "L",
        "family": "A",
        "initial_state_hash": "a",
        "scenario_geometry_hash": "g",
        "candidate_set_hash": "c",
        "historical_gate_verified": True,
        "one_shot_handoff_verified": True,
        "maximum_phase_switch_delta": 0.0,
        "terminal_task_goals_unchanged": True,
        "team_success": False,
        "collision": True,
        "obstacle_collision": True,
        "inter_agent_collision": False,
        "timeout": False,
        "steps": 20,
        "frozen_selected_reference_hash": "r",
    }
    source = {
        "initial_condition_hash": "a",
        "scenario_geometry_hash": "g",
        "candidate_set_hash": "c",
        "team_success": "False",
        "collision": "False",
        "obstacle_collision": "False",
        "inter_agent_collision": "False",
        "timeout": "True",
        "steps": "220",
    }
    rerun_agents = [{"agent_id": 0, "reference_reached": False}]
    source_agents = [{"agent_id": 0, "reference_reached": "False"}]
    result = module._reproduction_checks(
        rerun=rerun,
        source_episode=source,
        rerun_agents=rerun_agents,
        source_agents=source_agents,
    )
    assert result["status"] == "FAILED"
    assert "collision" in result["failed_checks"]
    assert "steps" in result["failed_checks"]


def test_audit_source_does_not_import_candidate_generation_or_gat():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "propose_reference_points" not in source
    assert "generate_candidate_set" not in source
    assert "torch_geometric" not in source
    assert "edge_enhanced_gat" not in source
    assert "gat_training" not in source.lower()


def test_mechanism_attribution_rejects_signal_shared_by_all_controls():
    failed = [{"layout_id": "F", "primary_failure_category": "X"}]
    horizon = []
    for layout_id in ("F", "S"):
        for h in (4, 8, 12, 20):
            horizon.append(
                {
                    "layout_id": layout_id,
                    "candidate_role": "selected",
                    "H_diag": h,
                    "task_progress": 0.1 * h,
                    "diagnostic_failure_signal": False,
                    "new_failure_signal_beyond_h4": h > 4,
                }
            )
    geometry = [
        {"layout_id": "F", "geometrically_poor_handoff_candidate": True},
        {"layout_id": "S", "geometrically_poor_handoff_candidate": True},
    ]
    rows, conclusion = build_mechanism_conclusion(
        failed_layouts=failed,
        horizon_rows=horizon,
        geometry_rows=geometry,
        lower_policy_rows=[],
        thresholds=_thresholds(),
    )
    index = {row["mechanism"]: row for row in rows}
    assert index["CANDIDATE_ADMISSIBILITY_LIMITATION"][
        "failure_specific_excess_fraction"
    ] == pytest.approx(0.0)
    assert index["CANDIDATE_ADMISSIBILITY_LIMITATION"]["label"] == "NO"
    assert index["SHORT_PREVIEW_HORIZON_LIMITATION"][
        "success_control_supporting_fraction"
    ] == pytest.approx(1.0)
    assert index["SHORT_PREVIEW_HORIZON_LIMITATION"]["label"] == "NO"
    assert conclusion["CANDIDATE_ADMISSIBILITY_LIMITATION"] == "NO"
