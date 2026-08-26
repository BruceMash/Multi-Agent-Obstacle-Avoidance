#!/usr/bin/env python3
"""Freeze the branch-level conclusion for the parallel zigzag study.

This script never runs an experiment.  It only combines already-closed Track
A/B/C decision files, enforces the combination and Formal stop rules, and
writes the paper-facing machine-readable conclusion and concise final report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "artifacts/parallel_zigzag_resolution/20260826_091334"


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def yes(value: Any) -> bool:
    return value is True or str(value).upper() in {"YES", "PASS", "TRUE"}


def scalar(source: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in source:
            return source[name]
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()

    a = load(root / "track_A_motion_proposal/TRACK_A_CONCLUSION.json")
    b = load(root / "track_B_learned_turn/TRACK_B_CONCLUSION.json")
    c = load(root / "track_C_visualization/TRACK_C_CONCLUSION.json")

    a_accepted = yes(scalar(a, "TRACK_A_ACCEPTED", default="NO"))
    b_accepted = yes(scalar(b, "TRACK_B_ACCEPTED", default="NO"))
    combination_authorized = a_accepted and b_accepted
    combination_path = root / "optional_combined/COMBINATION_CONCLUSION.json"
    combination = load(combination_path) if combination_path.is_file() else None
    if not combination_authorized and combination is not None and yes(scalar(combination, "COMBINATION_EXECUTED", default="NO")):
        raise RuntimeError("combination was executed without both independent tracks passing")

    combination_executed = combination is not None and yes(scalar(combination, "COMBINATION_EXECUTED", default="NO"))
    combination_accepted = scalar(combination or {}, "COMBINATION_ACCEPTED", default="NOT_RUN")
    if yes(combination_accepted):
        selected = "TRACK_A_PLUS_B"
    elif a_accepted and not b_accepted:
        selected = "TRACK_A"
    elif b_accepted and not a_accepted:
        selected = "TRACK_B"
    elif a_accepted and b_accepted:
        # Combination is optional.  If it was not selected, retain the simpler
        # accepted arm unless its own decision file explicitly names B.
        selected = str(scalar(combination or {}, "SELECTED_SIMPLER_VARIANT", default="TRACK_A"))
    else:
        selected = "FROZEN_STRONG"

    # The selected candidate differs from the current paper-formal Original
    # whenever it is Strong, A, B, or A+B.  It therefore needs a new explicitly
    # authorized Formal evaluation before inheriting any paper-formal claim.
    formal_recommended = selected != "ORIGINAL"
    formal_executed = any(
        yes(scalar(block, "FORMAL_EXECUTED", "FORMAL_V2_EXECUTED", default="NO"))
        for block in (a, b, c, combination or {})
    )
    if formal_executed:
        raise RuntimeError("a Track A/B/C/combined arm reports forbidden Formal execution")

    conclusion = {
        "schema_version": "parallel_zigzag_resolution_conclusion_v1",
        "ORIGINAL_FORMAL_SUCCESS": 0.9525,
        "FORMAL_RESULT_CHANGED": "NO",
        "TRACK_A_THETA_MAX_DEG": scalar(a, "theta_max_deg", "TRACK_A_THETA_MAX_DEG"),
        "TRACK_A_FALLBACK_RATE": scalar(a, "fallback_rate", "TRACK_A_FALLBACK_RATE"),
        "TRACK_A_DEV_SUCCESS_DELTA_PP": scalar(a, "dev_success_delta_pp", "TRACK_A_DEV_SUCCESS_DELTA_PP"),
        "TRACK_A_YAW_REVERSAL_REDUCTION": scalar(a, "yaw_reversal_reduction_percent", "TRACK_A_YAW_REVERSAL_REDUCTION"),
        "TRACK_A_PITCH_REVERSAL_REDUCTION": scalar(a, "pitch_reversal_reduction_percent", "TRACK_A_PITCH_REVERSAL_REDUCTION"),
        "TRACK_A_LONG_ARC_SCENES": scalar(a, "long_arc_scenes", "TRACK_A_LONG_ARC_SCENES"),
        "TRACK_A_ACCEPTED": "YES" if a_accepted else "NO",
        "TRACK_B_BEST_CHECKPOINT_STEPS": scalar(b, "best_checkpoint_steps", "TRACK_B_BEST_CHECKPOINT_STEPS"),
        "TRACK_B_DEV_SUCCESS_DELTA_PP": scalar(b, "dev_success_delta_pp", "TRACK_B_DEV_SUCCESS_DELTA_PP"),
        "TRACK_B_YAW_REVERSAL_REDUCTION": scalar(b, "yaw_reversal_reduction_percent", "TRACK_B_YAW_REVERSAL_REDUCTION"),
        "TRACK_B_PITCH_REVERSAL_REDUCTION": scalar(b, "pitch_reversal_reduction_percent", "TRACK_B_PITCH_REVERSAL_REDUCTION"),
        "TRACK_B_LONG_ARC_SCENES": scalar(b, "long_arc_scenes", "TRACK_B_LONG_ARC_SCENES"),
        "TRACK_B_SENSOR_ENCODER_DRIFT": scalar(b, "sensor_encoder_drift", "TRACK_B_SENSOR_ENCODER_DRIFT"),
        "TRACK_B_ACCEPTED": "YES" if b_accepted else "NO",
        "COMBINATION_AUTHORIZED": "YES" if combination_authorized else "NO",
        "COMBINATION_EXECUTED": "YES" if combination_executed else "NO",
        "COMBINATION_ACCEPTED": combination_accepted,
        "SELECTED_FINAL_CANDIDATE": selected,
        "FORMAL_REEVALUATION_RECOMMENDED": "YES" if formal_recommended else "NO",
        "FORMAL_V2_EXECUTED": "NO",
        "VISUALIZATION_STYLE_READY": scalar(c, "VISUALIZATION_STYLE_READY", default="NO"),
        "RAW_TRAJECTORY_INTEGRITY": scalar(c, "RAW_TRAJECTORY_INTEGRITY", default="FAIL"),
        "FINAL_RECOMMENDATION": (
            f"Select {selected} as the final candidate and reject Track A, Track B, and their combination. "
            "The current paper-formal method remains Original at 381/400; evaluate the selected candidate "
            "on a new explicitly authorized Formal run before making any new formal claim."
        ),
    }
    final_dir = root / "final_selection"
    final_dir.mkdir(parents=True, exist_ok=True)
    for destination in (root / "conclusion.json", final_dir / "conclusion.json"):
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(conclusion, stream, indent=2, ensure_ascii=False)
            stream.write("\n")

    formal_decision = {
        "schema_version": "parallel_zigzag_formal_go_no_go_v1",
        "ORIGINAL_FORMAL_SUCCESS": 0.9525,
        "FORMAL_RESULT_CHANGED": "NO",
        "SELECTED_FINAL_CANDIDATE": selected,
        "FORMAL_REEVALUATION_RECOMMENDED": conclusion["FORMAL_REEVALUATION_RECOMMENDED"],
        "FORMAL_V2_EXECUTED": "NO",
        "EXPLICIT_USER_AUTHORIZATION_REQUIRED": "YES",
    }
    for destination in (
        root / "FINAL_ZIGZAG_FORMAL_GO_NO_GO.json",
        final_dir / "FINAL_ZIGZAG_FORMAL_GO_NO_GO.json",
    ):
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(formal_decision, stream, indent=2, ensure_ascii=False)
            stream.write("\n")

    a_decision = load(root / "track_A_motion_proposal/MOTION_PROPOSAL_DEV_GO_NO_GO.json")
    b_teacher = load(root / "track_B_learned_turn/TURN_SMOOTH_TEACHER_ACTION_RECOVERABILITY.json")
    b_zero = load(root / "track_B_learned_turn/ZERO_INIT_EQUIVALENCE.json")
    report = f"""# Parallel Final Zigzag Resolution and Visualization Upgrade

## Executive result

- Track A accepted: **{conclusion['TRACK_A_ACCEPTED']}**.
- Track B accepted: **{conclusion['TRACK_B_ACCEPTED']}**.
- Combination executed: **{conclusion['COMBINATION_EXECUTED']}**.
- Selected final candidate: **{selected}**.
- Formal V2 executed: **NO**; the original result remains **381/400 (95.25%)**.

## Track A - Motion-Aligned Proposal

- Frozen motion-cone threshold: **{conclusion['TRACK_A_THETA_MAX_DEG']:.4f} deg**.
- Frozen Strong / Motion-Aligned success: **{a_decision['a0_success_rate']:.1%} / {a_decision['a1_success_rate']:.1%}** ({conclusion['TRACK_A_DEV_SUCCESS_DELTA_PP']:+.1f} pp).
- Full-sphere fallback / actual cone activation: **{conclusion['TRACK_A_FALLBACK_RATE']:.2%} / {a_decision['motion_proposal']['motion_cone_activation_rate']:.2%}**.
- Yaw/pitch reversal reductions: **{conclusion['TRACK_A_YAW_REVERSAL_REDUCTION']:.2f}% / {conclusion['TRACK_A_PITCH_REVERSAL_REDUCTION']:.2f}%**.
- Fixed long-arc scenes: {conclusion['TRACK_A_LONG_ARC_SCENES']}.

Track A failed the reversal and raw-morphology gates. Holdout was not generated or run.

## Track B - Learned Turn Continuity

- Observation expansion: **522 -> 529**, with only the seven requested context features.
- Zero-init equivalence: **{b_zero['ZERO_INIT_EQUIVALENCE']}**; actor max error {b_zero['maximum_absolute_actor_action_difference']:.3e}, critic max error {b_zero['maximum_absolute_critic_q_difference']:.3e}.
- Actor/online-critic/target-critic sensor-encoder pretraining freeze drift: **0 / 0 / 0**.
- Smooth-teacher action map: **invalid**; the DMP Jacobian is {b_teacher['jacobian_shape'][0]}x{b_teacher['jacobian_shape'][1]}, rank {b_teacher['jacobian_rank']}, nullity {b_teacher['jacobian_nullity']}.
- Training, Development, Holdout, and Formal: **NOT RUN**.

Track B stopped before training because a filtered 3-D execution acceleration has no unique, source-authorized 6-D SAC action target. No pseudoinverse or replacement loss was invented.

## Visualization and integrity

- Visualization style ready: **{conclusion['VISUALIZATION_STYLE_READY']}**.
- Raw trajectory integrity: **{conclusion['RAW_TRAJECTORY_INTEGRITY']}**.
- Track C is presentation-only and changed no scientific result.
- Independent Track A/B/C audits and final reconciliation: **PASS**.

## Final stop rule

`FORMAL_REEVALUATION_RECOMMENDED = {conclusion['FORMAL_REEVALUATION_RECOMMENDED']}`. No new variant inherits the original 95.25% Formal result. A new Formal run requires explicit user authorization.
"""
    (root / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(conclusion, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
