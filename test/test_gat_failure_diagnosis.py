from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ALGO = ROOT / "Multi-agent_Algo_lib"
if str(ALGO) not in sys.path:
    sys.path.insert(0, str(ALGO))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))
sys.modules.setdefault("pyarrow", None)

from scripts import audit_gat_failure_diagnosis as audit


CONFIG_PATH = ROOT / "configs/evaluation/gat_failure_diagnosis.json"


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_read_only_contract_and_gate_roles():
    config = _config()
    assert config["H_preview"] == 4
    assert config["H_label"] == 6
    assert config["target_feature_count"] == 3
    assert config["soft_target_temperature"] == 0.25
    assert "scalar" in config["formal_h6_target_gate"]
    assert "historical_vector" in config["companion_h6_target_gate"]
    assert config["companion_target_used_for_ranking_conclusion"] is False
    assert config["strict_read_only"]["training_allowed"] is False
    assert config["strict_read_only"]["new_closed_loop_allowed"] is False
    assert config["strict_read_only"]["short_h6_supervision_rollout_allowed"] is True
    assert config["stress_set_status"] == "DIAGNOSTIC_ONLY_AFTER_OBSERVATION"


def test_authoritative_artifacts_and_checkpoint_hashes_exist():
    config = _config()
    for key in (
        "stage1_training_dir",
        "stage1_dataset_dir",
        "main_closed_loop_dir",
        "stress_dir",
    ):
        assert (ROOT / config[key]).is_dir()
    for path_key, hash_key in (
        ("stress_manifest", "stress_manifest_sha256"),
        ("stage1_checkpoint", "stage1_checkpoint_sha256"),
        ("sac_checkpoint", "sac_checkpoint_sha256"),
    ):
        path = ROOT / config[path_key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config[hash_key]


def test_diagnostic_thresholds_are_frozen_before_results():
    config = _config()
    thresholds = config["decision_thresholds"]
    assert config["strict_read_only"]["threshold_adjustment_after_results_allowed"] is False
    assert thresholds["ranking_clear_difference_pp"] == 0.05
    assert thresholds["feature_strong_ood_outside_fraction"] == 0.25
    assert thresholds["feature_weak_ood_outside_fraction"] == 0.15
    assert thresholds["null_strong_probability_shift"] == 0.10
    assert thresholds["null_weak_shift"] == 0.05
    assert thresholds["harmful_override_clear_pattern_fraction"] == 0.75


def test_distribution_reports_finite_quantiles_and_inf_rate():
    result = audit._distribution([1.0, 2.0, 3.0, float("inf")])
    assert result["count"] == 4
    assert result["finite_count"] == 3
    assert result["median"] == 2.0
    assert result["positive_infinity_rate"] == 0.25


def test_rank_percentile_maps_best_to_one_and_worst_to_zero():
    values = [3.0, 1.0, 2.0]
    assert audit._rank_percentile(values, 0) == 1.0
    assert audit._rank_percentile(values, 1) == 0.0
    assert audit._rank_percentile(values, 2) == 0.5


def test_scalar_historical_top1_agreement_uses_graph_not_class_rows():
    rows = [
        {
            "layout_id": layout,
            "agent_id": 0,
            "class_index": class_index,
            "formal_h6_top1": class_index == formal,
            "companion_h6_top1": class_index == companion,
        }
        for layout, formal, companion in (("A", 0, 0), ("B", 0, 1))
        for class_index in range(3)
    ]
    agreement, changed, graph_count = audit._graph_level_top1_agreement(rows)
    assert agreement == 0.5
    assert changed == 1
    assert graph_count == 2


def test_null_shift_and_harm_rules_use_unique_layouts():
    config = _config()
    distributions = [
        {
            "dataset": "stage1_test",
            "subset": "overall",
            "mean": 0.20,
            "median": 0.18,
            "null_top1_rate": 0.20,
        },
        {
            "dataset": "stress",
            "subset": "overall",
            "mean": 0.35,
            "median": 0.30,
            "null_top1_rate": 0.35,
        },
    ]
    null_rows = [
        {
            "layout_id": f"L{i}",
            "paired_actual_category": "GAT_NULL_FAILURE_FP_SUCCESS"
            if i < 4
            else "BOTH_FAIL",
        }
        for i in range(6)
    ]
    override_rows = [
        {
            "layout_id": f"H{i}",
            "harmful_layout_context": True,
            "gat_selected_null": True,
            "fp_rank_of_gat_selection_1based": None,
            "fp_score_gap_top1_minus_gat_selection": 0.2,
            "gat_confidence": 0.6,
            "gat_top1_top2_margin": 0.2,
            "gat_collision": True,
            "gat_timeout": False,
        }
        for i in range(3)
    ]
    result = audit._null_and_override_conclusions(
        config=config,
        null_rows=null_rows,
        null_distribution_rows=distributions,
        override_rows=override_rows,
    )
    assert result["NULL_OOD_SHIFT"] == "YES"
    assert result["NULL_SELECTION_HARM_SIGNAL"] == "YES"
    assert result["HARMFUL_OVERRIDE_PATTERN"] == "CLEAR"
    assert result["null_selected_unique_layout_count"] == 6


def test_safe_spearman_refuses_constant_or_insufficient_labels():
    assert audit._safe_spearman([1.0], [0.0]) is None
    assert audit._safe_spearman([1.0, 2.0], [1.0, 1.0]) is None
    assert np.isclose(audit._safe_spearman([1, 2, 3], [0, 1, 2]), 1.0)


def test_required_artifact_names_are_emitted_by_adapter_source():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    required = [
        "config.json",
        "integrity_manifest.json",
        "training_sufficiency.csv",
        "checkpoint_comparison.csv",
        "learning_curve_diagnosis.json",
        "train_val_test_gap.json",
        "stress_offline_ranking.csv",
        "stress_target_alignment.csv",
        "null_audit.csv",
        "null_distribution_shift.csv",
        "selector_override_analysis.csv",
        "post_reference_failure.csv",
        "feature_distribution_shift.csv",
        "family_diagnosis.csv",
        "conclusion.json",
        "FINAL_REPORT.md",
    ]
    assert all(name in source for name in required)


def test_adapter_contains_no_optimizer_or_training_invocation():
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    called_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
    assert "train_stage1" not in called_names
    assert "Adam" not in called_names
    assert "AdamW" not in called_names
    assert "backward" not in called_names
    assert "step" not in called_names


def test_feature_groups_use_only_existing_graph_inputs():
    config = _config()
    features = {
        value for values in config["feature_groups"].values() for value in values
    }
    assert features == {
        "candidate_distance",
        "sector_safety",
        "preview_progress",
        "preview_clearance",
        "preview_deviation",
        "preview_terminal_speed",
        "t_min",
        "d_min",
        "T_risk",
        "proposal_count",
        "null_goal_distance",
        "null_goal_sector_safety",
    }


def test_gate_confound_downgrades_pure_horizon_claim_but_keeps_supervision_scope():
    config = _config()
    metric_index = {
        "stage1_test:gat_stage1": {"top1_accuracy": 0.70},
        "stage1_test:fp_shep": {"top1_accuracy": 0.60},
        "stress_h6_scalar_target:gat_stage1": {"top1_accuracy": 0.45},
        "stress_h6_scalar_target:fp_shep": {"top1_accuracy": 0.30},
        "stress_h6_historical_companion:gat_stage1": {"top1_accuracy": 0.39},
        "stress_h6_historical_companion:fp_shep": {"top1_accuracy": 0.38},
    }
    conclusion = audit._final_conclusion(
        config=config,
        curve={"CHECKPOINT_SELECTION_ISSUE": "NO", "TRAINING_CURVE_STATE": "MIXED"},
        metric_index=metric_index,
        feature_summary={"strong_shift_groups": [], "weak_shift_groups": []},
        alignment_summary={"H6_TARGET_LONG_HORIZON_ALIGNMENT": "WEAK"},
        null_summary={
            "NULL_OOD_SHIFT": "NO",
            "NULL_SELECTION_HARM_SIGNAL": "NO",
            "HARMFUL_OVERRIDE_PATTERN": "CLEAR",
        },
        semantic_audit={
            "formal_vs_companion_top1_agreement": 0.70,
            "formal_vs_companion_top1_change_count": 21,
        },
    )
    assert conclusion["SUPERVISION_HORIZON_MISMATCH"] == "WEAK"
    assert conclusion["HORIZON_ONLY_ATTRIBUTION"].startswith("CONFOUNDED")
    assert conclusion["PRIMARY_GAT_LIMITATION"] == "SUPERVISION_MISMATCH"
    assert conclusion["RECOMMENDED_NEXT_DIRECTION"] == "SUPERVISION_TARGET_REDESIGN"
