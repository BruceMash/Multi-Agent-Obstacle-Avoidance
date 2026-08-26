#!/usr/bin/env python3
"""Deterministic microtests for the frozen turn-sign persistence rule."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, ALGO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from planning.turn_sign_persistence import (  # noqa: E402
    TurnSignPersistenceConfig,
    TurnSignPersistenceExecutionFilter,
)
from planning.safety_adaptive_jerk_limiter import (  # noqa: E402
    SafetyAdaptiveJerkLimiterConfig,
    SafetyAdaptiveVectorJerkLimiter,
)


ARTIFACT_ROOT = REPO_ROOT / "artifacts/turn_sign_persistence_zigzag_repair/20260825_222130"
THRESHOLD_PATH = ARTIFACT_ROOT / "TURN_PERSISTENCE_THRESHOLDS.json"
OUTPUT = ARTIFACT_ROOT / "TURN_PERSISTENCE_MICROTEST.json"
STRONG_THRESHOLD_PATH = (
    REPO_ROOT
    / "artifacts/safety_adaptive_jerk_limiter/20260825_115704/JERK_LIMITER_THRESHOLDS.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def limiter_for_strong(*, enabled: bool = True, early: bool = False) -> SafetyAdaptiveVectorJerkLimiter:
    thresholds = load_json(STRONG_THRESHOLD_PATH)
    return SafetyAdaptiveVectorJerkLimiter(
        SafetyAdaptiveJerkLimiterConfig(
            dt=0.1,
            j_smooth_mps3=float(thresholds["variants"]["strong"]["j_smooth_mps3"]),
            j_free_mps3=float(thresholds["j_free_mps3"]),
            emergency_margin_m=0.0,
            comfortable_margin_m=0.35,
            enabled=bool(enabled),
            early_bypass_enabled=bool(early),
            warning_margin_m=0.35,
        ),
        num_agents=3,
    )


def wrapper(*, persistence: bool = True, early: bool = True) -> TurnSignPersistenceExecutionFilter:
    thresholds = load_json(THRESHOLD_PATH)
    strong = limiter_for_strong(early=early)
    return TurnSignPersistenceExecutionFilter(
        strong,
        TurnSignPersistenceConfig(
            n_persist=2,
            a_rev_lat_mps2=float(thresholds["A_REV_LAT"]),
            a_rev_vert_mps2=float(thresholds["A_REV_VERT"]),
            warning_margin_m=0.35,
            velocity_epsilon_mps=1.0e-9,
            command_epsilon_mps2=1.0e-9,
            enabled=persistence,
        ),
    )


def apply(filter_: Any, step: int, raw: list[float], *, margin: float = 1.0, velocity: list[float] | None = None) -> np.ndarray:
    velocity = [1.0, 0.0, 0.0] if velocity is None else velocity
    filter_.set_execution_velocity(0, np.asarray(velocity, dtype=float))
    filter_.set_context(0, step=step, safety_margin_m=margin, time_since_reference_change_s=step * 0.1)
    output = np.asarray(filter_.limit(0, np.asarray(raw, dtype=float)), dtype=float)
    filter_.observe_executed(0, output)
    return output


def sequence() -> tuple[dict[str, bool], dict[str, Any]]:
    same = wrapper()
    first = apply(same, 0, [0.4, 0.4, 0.3])
    same_output = apply(same, 1, [0.6, 0.5, 0.4])

    weak = wrapper()
    apply(weak, 0, [0.0, 0.5, 0.4])
    weak_first = apply(weak, 1, [0.0, -0.5, -0.4])
    weak_second = apply(weak, 2, [0.0, -0.5, -0.4])

    large = wrapper()
    apply(large, 0, [0.0, 0.5, 0.4])
    large_output = apply(large, 1, [0.0, -1.2, -1.0])

    warning = wrapper()
    apply(warning, 0, [0.0, 0.5, 0.4])
    warning_raw = np.asarray([0.8, -1.7, -1.4])
    warning_output = apply(warning, 1, warning_raw.tolist(), margin=0.3)

    hard = wrapper()
    hard_raw = np.asarray([1.1, -1.4, -1.2])
    hard_output = apply(hard, 0, hard_raw.tolist(), margin=0.0)

    tangent = wrapper()
    tangent_output = apply(tangent, 0, [0.8, 0.5, 0.0])

    baseline = limiter_for_strong()
    disabled = wrapper(persistence=False, early=False)
    baseline.reset()
    disabled.reset()
    disabled_equal = True
    for step, raw in enumerate(([0.2, 0.4, 0.1], [0.7, -0.4, -0.2], [-0.5, 0.3, 0.6])):
        baseline.set_context(0, step=step, safety_margin_m=1.0, time_since_reference_change_s=step * 0.1)
        base_output = baseline.limit(0, np.asarray(raw, dtype=float))
        disabled.set_execution_velocity(0, np.asarray([1.0, 0.0, 0.0]))
        disabled.set_context(0, step=step, safety_margin_m=1.0, time_since_reference_change_s=step * 0.1)
        disabled_output = disabled.limit(0, np.asarray(raw, dtype=float))
        disabled_equal &= bool(np.array_equal(base_output, disabled_output))
        baseline.observe_executed(0, base_output)
        disabled.observe_executed(0, disabled_output)

    deterministic_a = wrapper()
    deterministic_b = wrapper()
    deterministic_equal = True
    finite = True
    for step, (raw, margin) in enumerate((([0.3, 0.5, 0.2], 1.0), ([0.1, -0.4, -0.3], 1.0), ([1.0, -1.3, 1.1], 0.2))):
        left = apply(deterministic_a, step, raw, margin=margin)
        right = apply(deterministic_b, step, raw, margin=margin)
        deterministic_equal &= bool(np.array_equal(left, right))
        finite &= bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right)))

    checks = {
        "same_sign_turn_passes_exactly": bool(np.array_equal(same_output, np.asarray([0.6, 0.5, 0.4]))),
        "single_weak_opposite_is_neutralized": bool(np.array_equal(weak_first, np.zeros(3))),
        "two_weak_opposites_confirm_reversal": bool(np.array_equal(weak_second, np.asarray([0.0, -0.5, -0.4]))),
        "large_opposite_bypasses_persistence": bool(np.array_equal(large_output, np.asarray([0.0, -1.2, -1.0]))),
        "warning_bypass_passes_raw": bool(np.array_equal(warning_output, warning_raw)),
        "hard_bypass_passes_raw": bool(np.array_equal(hard_output, hard_raw)),
        "tangential_acceleration_preserved": bool(tangent_output[0] == 0.8),
        "disabled_persistence_reproduces_frozen_strong": bool(disabled_equal),
        "all_outputs_finite": bool(finite),
        "deterministic_repeated_execution": bool(deterministic_equal),
    }
    traces = {
        "same_sign": same.trace_rows(),
        "weak_opposite": weak.trace_rows(),
        "large_opposite": large.trace_rows(),
        "warning": warning.trace_rows(),
        "hard": hard.trace_rows(),
        "tangent": tangent.trace_rows(),
    }
    return checks, traces


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("microtest artifact already exists; refusing overwrite")
    checks, traces = sequence()
    payload = {
        "schema_version": "turn_persistence_microtest_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "traces": traces,
        "performance_episodes_executed": 0,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": checks}, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
