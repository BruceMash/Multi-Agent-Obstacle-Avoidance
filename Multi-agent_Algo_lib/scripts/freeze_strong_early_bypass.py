"""Freeze and microtest the source-supported Strong early safety bypass."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.safety_adaptive_jerk_limiter import (  # noqa: E402
    SafetyAdaptiveJerkLimiterConfig,
    SafetyAdaptiveVectorJerkLimiter,
)


METHOD_CONFIG = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/09_final_freeze/method_configs/M9_Proposed_RERR_GAT_SAC_DMP.json"
THRESHOLDS = REPO_ROOT / "artifacts/safety_adaptive_jerk_limiter/20260825_115704/JERK_LIMITER_THRESHOLDS.json"
DT = 0.1


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_limiter(*, early: bool) -> SafetyAdaptiveVectorJerkLimiter:
    config = load(METHOD_CONFIG)
    thresholds = load(THRESHOLDS)
    return SafetyAdaptiveVectorJerkLimiter(
        SafetyAdaptiveJerkLimiterConfig(
            dt=DT,
            j_smooth_mps3=float(thresholds["variants"]["strong"]["j_smooth_mps3"]),
            j_free_mps3=float(thresholds["j_free_mps3"]),
            emergency_margin_m=float(config["err"]["h_emg_m"]),
            comfortable_margin_m=float(config["err"]["h_rep_m"]),
            warning_margin_m=float(config["err"]["h_rep_m"]),
            enabled=True,
            early_bypass_enabled=early,
        ),
        num_agents=int(config["num_agents"]),
    )


def run_sequence(limiter: SafetyAdaptiveVectorJerkLimiter, margins: list[float], commands: list[np.ndarray]) -> tuple[list[np.ndarray], list[dict]]:
    outputs: list[np.ndarray] = []
    for step, (margin, command) in enumerate(zip(margins, commands, strict=True)):
        limiter.set_context(
            0,
            step=step,
            safety_margin_m=margin,
            time_since_reference_change_s=step * DT,
        )
        value = limiter.limit(0, command)
        limiter.observe_executed(0, value)
        outputs.append(value)
    return outputs, limiter.trace_rows()


def behavior_rows(rows: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in row.items() if key != "limiter_runtime_ns"}
        for row in rows
    ]


def freeze(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    config = load(METHOD_CONFIG)
    thresholds = load(THRESHOLDS)
    err = config["err"]
    graph = config["graph"]
    proposal = config["proposal_config"]
    contract = {
        "schema_version": "strong_early_bypass_contract_v1",
        "status": "FROZEN_BEFORE_LAG_AUDIT_AND_ANY_NEW_PERFORMANCE",
        "online_margin": {
            "symbol": "m_t",
            "source": "existing active_direction_safety_margin.value_m nearest the active-reference direction",
            "current_and_past_only": True,
            "hard_emergency_margin_m": float(err["h_emg_m"]),
            "warning_margin_m": float(err["h_rep_m"]),
            "q_safe_formula": "clip((m_t-h_emg)/(h_rep-h_emg),0,1)",
            "q_safe_scale_endpoints": [float(err["h_emg_m"]), float(err["h_rep_m"])],
        },
        "existing_safety_thresholds": {
            "ERR_h_emg_m": float(err["h_emg_m"]),
            "ERR_h_rep_m": float(err["h_rep_m"]),
            "ERR_h_scale_m": float(err["h_scale_m"]),
            "peer_d_safe_m": float(graph["d_safe"]),
            "proposal_safe_radius_m": float(proposal["safe_radius"]),
            "proposal_obstacle_motion_allowance_m": float(proposal["obstacle_motion_allowance"]),
            "interaction_risk": "existing H4 d_min<d_safe or T_risk>0 mask",
        },
        "rule": {
            "hard_bypass": "m_t <= h_emg",
            "early_warning_bypass": "m_t <= h_rep",
            "deteriorating_warning_flag": "m_t <= h_rep AND previous m available AND m_t-m_(t-1)<0",
            "combined": "hard_bypass OR early_warning_bypass",
            "action_during_bypass": "pass raw SAC-DMP acceleration immediately to original physical acceleration/velocity saturation",
            "no_gradual_transition": True,
            "no_smoothing_during_bypass": True,
            "threshold_origin": "both bypass boundaries are existing R-ERR thresholds; no failure-derived or post-performance threshold",
        },
        "frozen_strong": {
            "j_smooth_mps3": float(thresholds["variants"]["strong"]["j_smooth_mps3"]),
            "j_free_mps3": float(thresholds["j_free_mps3"]),
            "jerk_threshold_changed": False,
        },
        "source_sha256": {
            str(METHOD_CONFIG.relative_to(REPO_ROOT)).replace("\\", "/"): sha256(METHOD_CONFIG),
            str(THRESHOLDS.relative_to(REPO_ROOT)).replace("\\", "/"): sha256(THRESHOLDS),
            "planning/safety_adaptive_jerk_limiter.py": sha256(REPO_ROOT / "planning/safety_adaptive_jerk_limiter.py"),
        },
        "performance_episodes_observed": 0,
    }
    (output_root / "STRONG_EARLY_BYPASS_CONTRACT.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )

    raw = np.asarray([4.0, -4.0, 3.0])
    safe = make_limiter(early=True)
    safe_outputs, safe_rows = run_sequence(safe, [1.0, 1.0], [np.zeros(3), raw])
    frozen = make_limiter(early=False)
    frozen_outputs, _ = run_sequence(frozen, [1.0, 1.0], [np.zeros(3), raw])

    critical = make_limiter(early=True)
    critical_outputs, critical_rows = run_sequence(critical, [0.0], [raw])
    warning = make_limiter(early=True)
    warning_outputs, warning_rows = run_sequence(
        warning,
        [0.40, 0.30],
        [np.zeros(3), raw],
    )

    disabled = make_limiter(early=False)
    reference = SafetyAdaptiveVectorJerkLimiter(
        SafetyAdaptiveJerkLimiterConfig(
            dt=disabled.config.dt,
            j_smooth_mps3=disabled.config.j_smooth_mps3,
            j_free_mps3=disabled.config.j_free_mps3,
            emergency_margin_m=disabled.config.emergency_margin_m,
            comfortable_margin_m=disabled.config.comfortable_margin_m,
            enabled=True,
        ),
        num_agents=disabled.num_agents,
    )
    commands = [
        np.asarray([math.sin(i), math.cos(0.7 * i), math.sin(0.3 * i)]) * 4.0
        for i in range(50)
    ]
    margins = [0.05 + 0.09 * (i % 10) for i in range(50)]
    disabled_outputs, disabled_rows = run_sequence(disabled, margins, commands)
    reference_outputs, reference_rows = run_sequence(reference, margins, commands)
    exact_disabled = all(
        np.array_equal(left, right)
        for left, right in zip(disabled_outputs, reference_outputs, strict=True)
    ) and behavior_rows(disabled_rows) == behavior_rows(reference_rows)

    deterministic_a = make_limiter(early=True)
    deterministic_b = make_limiter(early=True)
    out_a, rows_a = run_sequence(deterministic_a, margins, commands)
    out_b, rows_b = run_sequence(deterministic_b, margins, commands)
    deterministic = all(
        np.array_equal(left, right)
        for left, right in zip(out_a, out_b, strict=True)
    ) and behavior_rows(rows_a) == behavior_rows(rows_b)
    finite = all(np.all(np.isfinite(value)) for value in out_a)
    checks = {
        "safe_state_matches_frozen_strong_exactly": all(
            np.array_equal(left, right)
            for left, right in zip(safe_outputs, frozen_outputs, strict=True)
        ),
        "critical_state_passes_raw_immediately": np.array_equal(critical_outputs[0], raw),
        "critical_state_hard_bypass_recorded": bool(critical_rows[0]["hard_bypass"]),
        "deteriorating_warning_activates_early_bypass": bool(warning_rows[1]["early_bypass"]),
        "deteriorating_warning_passes_raw_immediately": np.array_equal(warning_outputs[1], raw),
        "disabled_guard_reproduces_frozen_strong_exactly": exact_disabled,
        "deterministic_replay_exact": deterministic,
        "all_outputs_finite": finite,
    }
    microtest = {
        "schema_version": "strong_early_bypass_microtest_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "warning_trace": warning_rows,
        "critical_trace": critical_rows,
        "safe_trace": safe_rows,
        "contract_sha256": sha256(output_root / "STRONG_EARLY_BYPASS_CONTRACT.json"),
        "performance_episodes_executed": 0,
    }
    (output_root / "STRONG_EARLY_BYPASS_MICROTEST.json").write_text(
        json.dumps(microtest, indent=2) + "\n", encoding="utf-8"
    )
    if microtest["status"] != "PASS":
        raise RuntimeError(json.dumps(microtest, indent=2))
    print(json.dumps({"status": "PASS", "checks": checks}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.output_root)


if __name__ == "__main__":
    main()
