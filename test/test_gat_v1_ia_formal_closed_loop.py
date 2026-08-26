from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for path in (REPO_ROOT, ALGO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.evaluate_gat_v1_ia_formal_closed_loop import (  # noqa: E402
    METHOD_IA,
    METHOD_V1,
    choose_seed_block,
    classify_selection,
    formal_decision,
    mcnemar_rows,
)


def test_choose_minimum_contiguous_unused_seed_block() -> None:
    used = set(range(60)) | {80, 81}
    assert choose_seed_block(used, block_size=20, minimum=0, maximum=1000) == list(range(60, 80))


def test_frozen_risk_contract_classification() -> None:
    assert classify_selection({"selected_null": True}, 0.6) == "NULL"
    assert classify_selection({"selected_null": False, "selected_candidate_id": None}, 0.6) == "NO_CANDIDATE"
    assert classify_selection(
        {
            "selected_null": False,
            "selected_candidate_id": 2,
            "selected_align_edge_count": 0,
            "selected_minimum_d_min": None,
            "selected_maximum_T_risk": None,
        },
        0.6,
    ) == "NEUTRAL"
    assert classify_selection(
        {
            "selected_null": False,
            "selected_candidate_id": 2,
            "selected_align_edge_count": 1,
            "selected_minimum_d_min": 0.59,
            "selected_maximum_T_risk": 0.0,
        },
        0.6,
    ) == "RISKY"
    assert classify_selection(
        {
            "selected_null": False,
            "selected_candidate_id": 2,
            "selected_align_edge_count": 1,
            "selected_minimum_d_min": 0.6,
            "selected_maximum_T_risk": 0.0,
        },
        0.6,
    ) == "SAFE"


def test_mcnemar_exact_counts_and_p_value() -> None:
    episodes = []
    pairs = [(True, True), (True, False), (False, True), (False, True)]
    for seed, (v1_success, ia_success) in enumerate(pairs):
        for method, success in ((METHOD_V1, v1_success), (METHOD_IA, ia_success)):
            episodes.append(
                {
                    "scenario": "multi_agent",
                    "seed": seed,
                    "method": method,
                    "team_success": success,
                    "any_collision": not success,
                    "inter_agent_collision": False,
                    "timeout": False,
                }
            )
    row = next(
        item
        for item in mcnemar_rows(episodes)
        if item["scope"] == "overall" and item["metric"] == "team_success"
    )
    assert row["v1_only_true_count"] == 1
    assert row["ia_only_true_count"] == 2
    assert row["discordant_count"] == 3
    assert row["exact_two_sided_p_value"] == 1.0


def _decision_inputs(overall_v1: float, overall_ia: float, multi_v1: float, multi_ia: float):
    methods = [
        {"method": METHOD_V1, "scope": "overall", "team_success_rate": overall_v1, "any_collision_rate": 0.3, "inter_agent_collision_rate": 0.1, "timeout_rate": 0.0},
        {"method": METHOD_IA, "scope": "overall", "team_success_rate": overall_ia, "any_collision_rate": 0.3, "inter_agent_collision_rate": 0.1, "timeout_rate": 0.0},
    ]
    scenarios = [
        {"method": METHOD_V1, "scope": "multi_agent", "team_success_rate": multi_v1},
        {"method": METHOD_IA, "scope": "multi_agent", "team_success_rate": multi_ia},
    ]
    risky = [
        {"method": METHOD_V1, "scope_type": "scenario", "scope": "overall", "risky_selection_rate_all_agents": 0.2},
        {"method": METHOD_IA, "scope_type": "scenario", "scope": "overall", "risky_selection_rate_all_agents": 0.1},
    ]
    nulls = [
        {"method": METHOD_V1, "scope": "overall", "null_selection_rate": 0.2},
        {"method": METHOD_IA, "scope": "overall", "null_selection_rate": 0.3},
    ]
    references = [
        {"method": METHOD_V1, "scope": "overall", "reference_reach_rate": 0.8, "reached_to_terminal_rate": 0.8},
        {"method": METHOD_IA, "scope": "overall", "reference_reach_rate": 0.8, "reached_to_terminal_rate": 0.8},
    ]
    tests = [
        {"scope": "overall", "metric": "team_success", "exact_two_sided_p_value": 0.5},
        {"scope": "overall", "metric": "any_collision", "exact_two_sided_p_value": 1.0},
    ]
    context = {
        "status": "PASSED",
        "checks": {"v1_checkpoint_hash": True, "ia_checkpoint_hash": True},
        "IA_CHECKPOINT_SHA256": "hash",
    }
    manifest = {"formal_seed_block": "60-79", "FINAL_TEST_SEED_OVERLAP": 0}
    integrity = {
        "status": "PASSED",
        "fairness": {
            "candidate_hash_mismatch_count": 0,
            "preview_hash_mismatch_count": 0,
            "graph_hash_mismatch_count": 0,
        },
    }
    return methods, scenarios, risky, nulls, references, tests, context, manifest, integrity


def _config():
    return {
        "formal_gain_decision": {
            "yes_minimum_overall_team_success_gain": 0.05,
            "yes_maximum_overall_collision_worsening": 0.05,
            "material_multi_agent_regression": -0.10,
            "weak_minimum_multi_agent_success_gain": 0.15,
            "weak_final_ia_maximum_reference_reach_decline": 0.05,
            "weak_final_ia_maximum_reached_to_terminal_decline": 0.05,
            "team_success_90_target": 0.90,
        }
    }


def test_formal_yes_and_weak_decisions_are_pre_registered() -> None:
    values = _decision_inputs(0.60, 0.66, 0.30, 0.30)
    yes = formal_decision(
        config=_config(), methods=values[0], scenarios=values[1], risky=values[2],
        nulls=values[3], references=values[4], tests=values[5], context=values[6],
        formal_manifest=values[7], integrity=values[8],
    )
    assert yes["IA_FORMAL_GAIN"] == "YES"
    assert yes["FINAL_GAT_CHECKPOINT"] == "IA"

    values = _decision_inputs(0.60, 0.60, 0.20, 0.40)
    weak = formal_decision(
        config=_config(), methods=values[0], scenarios=values[1], risky=values[2],
        nulls=values[3], references=values[4], tests=values[5], context=values[6],
        formal_manifest=values[7], integrity=values[8],
    )
    assert weak["IA_FORMAL_GAIN"] == "WEAK"
    assert weak["FINAL_GAT_CHECKPOINT"] == "IA"
