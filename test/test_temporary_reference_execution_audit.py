from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for path in (REPO_ROOT, ALGO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import audit_temporary_reference_execution as module  # noqa: E402


def _config() -> dict:
    return json.loads(module.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def _saved_row() -> dict:
    metadata = {
        "safety_margin": 0.5,
        "distance_progress": 0.8,
    }
    fp = {
        "fp_shep_online_score": 1.2,
        "preview_task_progress": 0.1,
        "preview_min_clearance": None,
        "preview_max_execution_deviation": 0.2,
        "preview_terminal_speed": 0.4,
        "normalized_preview_features": [0.3, 1.0, 0.2, 0.4],
        "preview_feature_valid_mask": [True, False, True, True],
    }
    return {
        "scenario": "open",
        "seed": "10",
        "agent_id": "0",
        "selected_candidate_id": "0",
        "temporary_reference": "[1.0, 0.0, 0.0]",
        "candidate_world_points": "[[1.0, 0.0, 0.0]]",
        "candidate_metadata": json.dumps([metadata]),
        "fp_shep_candidate_records": json.dumps([fp]),
        "K_t": "1",
        "candidate_available": "True",
        "selected_null": "False",
    }


def test_frozen_config_and_strict_read_only_contract() -> None:
    config = _config()
    module._assert_config(config)
    assert config["formal_seeds"] == list(range(10, 30))
    assert config["handoff_threshold_m"] == 0.25
    assert config["capture_radius_diagnostic_m"] == [0.20, 0.25, 0.30, 0.40]
    assert not any(config["strict_exclusions"].values())


def test_admissibility_uses_saved_descriptors_and_accepts_explicit_unbounded_clearance() -> None:
    result = module.assess_reference_admissibility(
        _saved_row(), _config(), source_consistent=True
    )
    assert result["reference_reasonably_admissible"]
    assert result["fp_shep_unbounded_clearance_explicit"]
    assert result["outcome_fields_used"] is False


def test_admissibility_rejects_unsafe_or_nonprogressing_reference() -> None:
    row = _saved_row()
    metadata = json.loads(row["candidate_metadata"])
    metadata[0]["safety_margin"] = -0.01
    metadata[0]["distance_progress"] = 0.0
    row["candidate_metadata"] = json.dumps(metadata)
    result = module.assess_reference_admissibility(
        row, _config(), source_consistent=True
    )
    assert not result["reference_reasonably_admissible"]
    assert "immediately_invalid_safety_margin" in result["exclusion_reasons"]
    assert "nonpositive_or_pathological_progress_descriptor" in result["exclusion_reasons"]


def test_gap_classification_matches_frozen_15_and_5_pp_rule() -> None:
    rules = _config()["classification"]
    assert module.classify_gap(0.15, rules) == "YES"
    assert module.classify_gap(0.149, rules) == "WEAK"
    assert module.classify_gap(0.05, rules) == "WEAK"
    assert module.classify_gap(0.049, rules) == "NO"


def test_b_failure_taxonomy_is_mutually_exclusive_and_collision_precedes_timeout() -> None:
    row = {
        "reference_reached": False,
        "ego_obstacle_collision_before_reference": True,
        "ego_inter_agent_collision_before_reference": False,
        "peer_only_collision_before_reference": False,
        "stagnation_before_reference": True,
        "timeout": True,
        "target_distance_reduction_m": -1.0,
        "environment_terminated": True,
    }
    assert module.classify_b_failure(row) == "OBSTACLE_COLLISION_BEFORE_REFERENCE"


def test_c_failure_taxonomy_requires_reference_before_handoff_attribution() -> None:
    row = {
        "reference_reached": False,
        "terminal_success": False,
        "post_reference_ego_obstacle_collision": True,
        "post_reference_ego_inter_agent_collision": False,
        "post_reference_stagnation": False,
        "post_reference_timeout": False,
    }
    assert module.classify_c_failure(row) == "REFERENCE_NOT_REACHED"


def test_config_rejects_handoff_or_training_changes() -> None:
    changed = copy.deepcopy(_config())
    changed["handoff_threshold_m"] = 0.30
    try:
        module._assert_config(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("changed handoff threshold must be rejected")
    changed = copy.deepcopy(_config())
    changed["strict_exclusions"]["sac_training"] = True
    try:
        module._assert_config(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("training flag must be rejected")


def test_saved_formal_artifact_contains_expected_122_selected_references() -> None:
    config = _config()
    source = REPO_ROOT / config["source_gat_closed_loop_dir"] / "per_agent_results.csv"
    rows = [
        row
        for row in module.read_csv(source)
        if row["method"] == "gat_stage1"
        and not module._boolean(row.get("selected_null"))
        and module._boolean(row.get("candidate_available"))
    ]
    assert len(rows) == 122


def test_err_same_goal_subset_recovers_exactly_70_saved_events() -> None:
    rows, classification = module.err_same_goal_rows(_config())
    assert len(rows) == 70
    assert classification in {"YES", "WEAK", "NO"}


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"{len(tests)} temporary-reference audit tests passed")
