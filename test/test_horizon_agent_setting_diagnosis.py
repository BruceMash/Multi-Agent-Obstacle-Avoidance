from __future__ import annotations

import json
from pathlib import Path

import pytest

from planning.horizon_agent_setting_diagnosis import (
    analyze_diagnosis,
    build_paired_comparisons,
    classify_effect,
    validate_horizon_only_pair,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/evaluation/horizon_agent_setting_diagnosis.json"


def _record(
    *,
    protocol: str,
    setting: str,
    scenario: str,
    horizon: int,
    success: bool,
    reached: int = 1,
    available: int = 1,
    completion_step: int | None = None,
) -> dict[str, object]:
    long = horizon == 220
    completion = completion_step if success else None
    return {
        "protocol": protocol,
        "agent_setting": setting,
        "scenario": scenario,
        "seed": 0,
        "max_steps": horizon,
        "episode_steps": completion or horizon,
        "team_terminal_success": success,
        "team_terminal_completion_step": completion,
        "collision": False,
        "obstacle_collision": False,
        "inter_agent_collision": False,
        "timeout": not success,
        "termination_reason": "terminal_success" if success else "timeout",
        "temporary_reference_available_count": available,
        "temporary_reference_reached_count": reached,
        "reached_then_terminal_completion_count": reached if success else 0,
        "team_stage1_success": bool(available and reached == available),
        "reference_reached_steps": [20] if reached else [],
        "terminal_completion_steps": [completion] if completion is not None else [],
        "remaining_steps_after_reference": [horizon - 20] if reached else [],
        "collision_before_reference_reached_count": 0,
        "collision_after_reference_reached_count": 0,
        "timeout_after_reference_reached": bool(not success and reached),
        "terminal_progress_m": 1.0,
        "path_length_m": 2.0,
        "trajectory_smoothness": 3.0,
        "initial_condition_hash": f"initial:{setting}:{scenario}",
        "temporary_reference_hash": f"reference:{setting}:{scenario}",
        "horizon_independent_contract_hash": f"contract:{setting}",
        "state_at_step_50_hash": f"prefix:{setting}:{scenario}" if long else None,
        "final_state_hash": (
            f"prefix:{setting}:{scenario}"
            if horizon == 50
            else f"final:{setting}:{scenario}"
        ),
    }


def _synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for setting, prefix, scenarios in (
        ("single_agent", "S", ("open", "sparse_static", "historical_base")),
        ("multi_agent", "M", ("open", "sparse_static", "multi_agent")),
    ):
        for scenario in scenarios:
            shared = scenario in {"open", "sparse_static"}
            for horizon in (50, 220):
                # Supplementary scenarios intentionally have the opposite result.
                success = shared and setting == "single_agent"
                if not shared:
                    success = setting == "multi_agent"
                rows.append(
                    _record(
                        protocol=f"{prefix}{horizon}",
                        setting=setting,
                        scenario=scenario,
                        horizon=horizon,
                        success=success,
                        completion_step=40 if success else None,
                    )
                )
    return rows


def test_horizon_only_pair_accepts_max_steps_as_only_difference() -> None:
    result = validate_horizon_only_pair(
        {"max_steps": 50, "scene": "open", "seed": 0},
        {"max_steps": 220, "scene": "open", "seed": 0},
    )
    assert result["status"] == "PASSED"
    assert result["only_active_difference"] == "max_steps"


def test_horizon_only_pair_rejects_any_other_difference() -> None:
    with pytest.raises(ValueError, match="other than max_steps"):
        validate_horizon_only_pair(
            {"max_steps": 50, "scene": "open"},
            {"max_steps": 220, "scene": "sparse_static"},
        )


@pytest.mark.parametrize(
    ("difference", "expected"),
    [(0.099, "NONE"), (0.10, "WEAK"), (0.299, "WEAK"), (0.30, "STRONG")],
)
def test_effect_classification_is_descriptive_percentage_point_rule(
    difference: float, expected: str
) -> None:
    assert classify_effect(difference) == expected
    assert classify_effect(-difference) == expected


def test_config_freezes_scenes_seeds_horizons_and_exclusions() -> None:
    settings = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert settings["horizons"] == [50, 220]
    assert settings["single_agent"]["scenarios"] == [
        "open",
        "sparse_static",
        "historical_base",
    ]
    assert settings["multi_agent"]["scenarios"] == [
        "open",
        "sparse_static",
        "multi_agent",
    ]
    assert settings["single_agent"]["seeds"] == list(range(5))
    assert settings["multi_agent"]["seeds"] == list(range(5))
    assert not any(settings["strict_exclusions"].values())


def test_config_records_historical_gate_and_no_counterfactual_rerun() -> None:
    settings = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    transition = settings["historical_transition"]
    assert transition["goal_eff"] == "goal_base + clip(goal_offset,-1,1)"
    assert transition["forcing_gate"] == "tanh(abs(goal_eff-position))"
    assert settings["counterfactual"]["cutoff_step"] == 50
    assert settings["counterfactual"]["controller_rerun"] is False


def test_shared_scenario_agent_effect_excludes_setting_specific_scenes() -> None:
    analysis = analyze_diagnosis(_synthetic_rows())
    effect = analysis["conclusion"]["multi_agent_effect_shared_at_220"]
    assert effect["baseline_episode_count"] == 2
    assert effect["comparison_episode_count"] == 2
    assert effect["baseline_success_count"] == 0
    assert effect["comparison_success_count"] == 2
    assert analysis["conclusion"]["MULTI_AGENT_EFFECT_SHARED_SCENARIOS"] == "STRONG"


def test_agent_reference_and_team_terminal_metrics_are_not_conflated() -> None:
    rows = _synthetic_rows()
    for row in rows:
        if row["protocol"] == "M50" and row["scenario"] == "open":
            row["temporary_reference_available_count"] = 3
            row["temporary_reference_reached_count"] = 1
            row["team_stage1_success"] = False
            row["team_terminal_success"] = False
    analysis = analyze_diagnosis(rows)
    m50_open = next(
        row
        for row in analysis["stage_rows"]
        if row["protocol"] == "M50" and row["scenario_scope"] == "open"
    )
    assert m50_open["agent_reference_reached_count"] == 1
    assert m50_open["agent_reference_reached_rate"] == pytest.approx(1 / 3)
    assert m50_open["team_stage1_success_episode_count"] == 0
    assert m50_open["terminal_success_count"] == 0


def test_counterfactual_success_uses_observed_long_completion_step() -> None:
    short = _record(
        protocol="S50",
        setting="single_agent",
        scenario="open",
        horizon=50,
        success=False,
    )
    long = _record(
        protocol="S220",
        setting="single_agent",
        scenario="open",
        horizon=220,
        success=True,
        completion_step=80,
    )
    comparisons, integrity = build_paired_comparisons([short, long])
    assert integrity["status"] == "PASSED"
    assert comparisons[0]["counterfactual_success_at_50"] is False
    assert comparisons[0]["success_censored_by_50_step_horizon"] is True


def test_pairing_rejects_step_50_trajectory_prefix_mismatch() -> None:
    short = _record(
        protocol="M50",
        setting="multi_agent",
        scenario="open",
        horizon=50,
        success=False,
    )
    long = _record(
        protocol="M220",
        setting="multi_agent",
        scenario="open",
        horizon=220,
        success=False,
    )
    long["state_at_step_50_hash"] = "different"
    _, integrity = build_paired_comparisons([short, long])
    assert integrity["status"] == "FAILED"
    assert "prefix=False" in integrity["errors"][0]


def test_analysis_emits_required_effect_conclusions() -> None:
    conclusion = analyze_diagnosis(_synthetic_rows())["conclusion"]
    assert "HORIZON_EFFECT_SINGLE" in conclusion
    assert "HORIZON_EFFECT_MULTI" in conclusion
    assert "MULTI_AGENT_EFFECT_SHARED_SCENARIOS" in conclusion
    assert "PRIMARY_LIMITATION" in conclusion
    assert conclusion["effect_classification_is_statistical_significance"] is False


def test_mixed_horizon_and_agent_effect_does_not_recommend_fine_tuning() -> None:
    rows = _synthetic_rows()
    for row in rows:
        protocol = str(row["protocol"])
        scenario = str(row["scenario"])
        success = protocol == "S220" or (
            protocol == "M220" and scenario == "open"
        )
        row["team_terminal_success"] = success
        row["team_terminal_completion_step"] = 80 if success else None
        row["episode_steps"] = 80 if success else int(row["max_steps"])
        row["termination_reason"] = "terminal_success" if success else "timeout"
    conclusion = analyze_diagnosis(rows)["conclusion"]
    assert conclusion["PRIMARY_LIMITATION"] == "HORIZON_PLUS_MULTI_AGENT_INTERACTION"
    assert conclusion["SAC_FINE_TUNING_RECOMMENDED"] == "NO"
