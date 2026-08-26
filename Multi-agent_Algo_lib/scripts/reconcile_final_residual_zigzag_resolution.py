#!/usr/bin/env python3
"""Independent raw-record reconciliation for the final zigzag Development block."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "artifacts/final_residual_zigzag_resolution/20260825_183146"
ARMS = ("strong", "repaired")
EPS_SPEED = 1.0e-9
EPS_ANGLE = 1.0e-9


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct_reduction(reference: float, variant: float) -> float:
    return 100.0 * (reference - variant) / reference


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def direction_metrics(payload: dict[str, Any], record_root: Path) -> dict[str, float | int]:
    trajectory = np.load(record_root / payload["trajectory_file"])
    velocity = np.asarray(trajectory["velocities"], dtype=float)
    dt = float(trajectory["dt"])
    totals: dict[str, float | int] = {
        "yaw_reversal_count": 0,
        "yaw_eligible_count": 0,
        "yaw_directional_tv": 0.0,
        "pitch_reversal_count": 0,
        "pitch_eligible_count": 0,
        "pitch_directional_tv": 0.0,
    }
    agents = {int(row["agent_id"]): row for row in payload["agents"]}
    for agent_id in range(velocity.shape[1]):
        stop = agents[agent_id].get("terminal_completion_step")
        end = len(velocity) if stop is None else min(len(velocity), int(stop) + 1)
        agent_velocity = velocity[:end, agent_id]
        speed = np.linalg.norm(agent_velocity, axis=1)
        valid = speed > EPS_SPEED
        yaw = np.full(len(speed), np.nan)
        pitch = np.full(len(speed), np.nan)
        yaw[valid] = np.arctan2(agent_velocity[valid, 1], agent_velocity[valid, 0])
        horizontal = np.linalg.norm(agent_velocity[:, :2], axis=1)
        pitch[valid] = np.arctan2(agent_velocity[valid, 2], horizontal[valid])
        consecutive = valid[1:] & valid[:-1]
        indexes = np.flatnonzero(consecutive) + 1
        yaw_rate = np.full(len(speed), np.nan)
        pitch_rate = np.full(len(speed), np.nan)
        yaw_rate[indexes] = wrap_angle(yaw[indexes] - yaw[indexes - 1]) / dt
        pitch_rate[indexes] = (pitch[indexes] - pitch[indexes - 1]) / dt
        for axis, rate in (("yaw", yaw_rate), ("pitch", pitch_rate)):
            eligible_pair = (
                np.isfinite(rate[1:])
                & np.isfinite(rate[:-1])
                & (np.abs(rate[1:]) > EPS_ANGLE)
                & (np.abs(rate[:-1]) > EPS_ANGLE)
            )
            eligible_indexes = np.flatnonzero(eligible_pair) + 1
            reversals = rate[eligible_indexes] * rate[eligible_indexes - 1] < 0.0
            finite = rate[np.isfinite(rate)]
            totals[f"{axis}_eligible_count"] += int(len(eligible_indexes))
            totals[f"{axis}_reversal_count"] += int(np.sum(reversals))
            totals[f"{axis}_directional_tv"] += (
                float(np.sum(np.abs(np.diff(finite)))) if len(finite) > 1 else 0.0
            )
    return totals


def close(left: float, right: float, tolerance: float = 1.0e-9) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance * max(1.0, abs(float(right))))


def main() -> None:
    contract_path = ARTIFACT_ROOT / "FINAL_ZIGZAG_RUNTIME_CONTRACT.json"
    manifest_path = ARTIFACT_ROOT / "FINAL_ZIGZAG_DEV100_MANIFEST.json"
    decision_path = ARTIFACT_ROOT / "FINAL_ZIGZAG_DEV_GO_NO_GO.json"
    contract = load_json(contract_path)
    manifest = load_json(manifest_path)
    decision = load_json(decision_path)
    entries = manifest["entries"]
    ids = [str(row["scenario_id"]) for row in entries]
    id_set = set(ids)

    checks: dict[str, bool] = {}
    checks["runtime_frozen_before_performance"] = (
        contract.get("status") == "FROZEN_BEFORE_NEW_DEVELOPMENT_PERFORMANCE"
        and int(contract.get("performance_episodes_observed_before_freeze", -1)) == 0
    )
    checks["manifest_exact_hash"] = sha256(manifest_path) == contract["development"]["manifest_sha256"]
    checks["manifest_unique_100"] = len(ids) == 100 and len(id_set) == 100
    stage_counts = Counter(str(row["stage"]) for row in entries)
    stage_family_counts = Counter((str(row["stage"]), str(row["family"])) for row in entries)
    checks["manifest_balanced_4x5x5"] = (
        sorted(stage_counts.values()) == [25, 25, 25, 25]
        and len(stage_family_counts) == 20
        and set(stage_family_counts.values()) == {5}
    )
    checks["scene_hashes_match_manifest"] = all(
        sha256(REPO_ROOT / row["scenario_file"]) == row["scenario_file_sha256"] for row in entries
    )
    checks["source_hashes_match_runtime_freeze"] = all(
        sha256(REPO_ROOT / relative) == expected
        for relative, expected in contract["source_sha256"].items()
    )
    checks["checkpoint_hashes_match_runtime_freeze"] = all(
        sha256(REPO_ROOT / contract["frozen_method"][path_key])
        == contract["frozen_method"][hash_key]
        for path_key, hash_key in (
            ("config", "config_sha256"),
            ("gat_checkpoint", "gat_checkpoint_sha256"),
            ("sac_checkpoint", "sac_checkpoint_sha256"),
        )
    )

    records: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    record_integrity: dict[str, Any] = {}
    morphology: dict[str, dict[str, float | int]] = {}
    for arm in ARMS:
        root = ARTIFACT_ROOT / "development" / arm / "episode_records"
        json_paths = sorted(root.glob("FZR_DEV_*.json"))
        trajectory_hash_ok = True
        limiter_hash_ok = True
        identity_ok = True
        no_software_failure = True
        pooled: defaultdict[str, float] = defaultdict(float)
        for path in json_paths:
            payload = load_json(path)
            sid = str(payload["entry_identity"]["scenario_id"])
            records[arm][sid] = payload
            identity_ok &= sid == path.stem and sid in id_set
            trajectory_hash_ok &= (
                sha256(root / payload["trajectory_file"]) == payload["trajectory_sha256"]
            )
            limiter_hash_ok &= (
                sha256(root / payload["limiter_trace_file"]) == payload["limiter_trace_sha256"]
            )
            reason = str(payload["episode"].get("termination_reason", "")).lower()
            no_software_failure &= "software" not in reason and "numerical" not in reason
            for key, value in direction_metrics(payload, root).items():
                pooled[key] += float(value)
        morphology[arm] = dict(pooled)
        record_integrity[arm] = {
            "json_record_count": len(json_paths),
            "unique_scenario_count": len(records[arm]),
            "identity_match": bool(identity_ok),
            "trajectory_hash_match": bool(trajectory_hash_ok),
            "limiter_trace_hash_match": bool(limiter_hash_ok),
            "software_or_numerical_failure_count": sum(
                any(token in str(row["episode"].get("termination_reason", "")).lower() for token in ("software", "numerical"))
                for row in records[arm].values()
            ),
        }
        checks[f"{arm}_record_set_exact"] = set(records[arm]) == id_set and len(json_paths) == 100
        checks[f"{arm}_record_hashes_valid"] = bool(trajectory_hash_ok and limiter_hash_ok)
        checks[f"{arm}_no_software_failure"] = bool(no_software_failure)

    aggregate: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        episodes = [records[arm][sid]["episode"] for sid in ids]
        aggregate[arm] = {
            "team_success_count": sum(bool(row["team_success"]) for row in episodes),
            "collision_count": sum(bool(row["collision"]) for row in episodes),
            "obstacle_collision_count": sum(bool(row["obstacle_collision"]) for row in episodes),
            "peer_collision_count": sum(bool(row["inter_agent_collision"]) for row in episodes),
            "timeout_count": sum(bool(row["timeout"]) for row in episodes),
            "stage_success_count": {
                stage: sum(
                    bool(records[arm][sid]["episode"]["team_success"])
                    for sid in ids
                    if records[arm][sid]["entry_identity"]["stage"] == stage
                )
                for stage in sorted(stage_counts)
            },
        }
    both_success = [
        sid
        for sid in ids
        if records["strong"][sid]["episode"]["team_success"]
        and records["repaired"][sid]["episode"]["team_success"]
    ]
    strong_only = [
        sid
        for sid in ids
        if records["strong"][sid]["episode"]["team_success"]
        and not records["repaired"][sid]["episode"]["team_success"]
    ]
    repaired_only = [
        sid
        for sid in ids
        if records["repaired"][sid]["episode"]["team_success"]
        and not records["strong"][sid]["episode"]["team_success"]
    ]

    for arm in ARMS:
        morphology[arm]["yaw_reversal_rate"] = (
            morphology[arm]["yaw_reversal_count"] / morphology[arm]["yaw_eligible_count"]
        )
        morphology[arm]["pitch_reversal_rate"] = (
            morphology[arm]["pitch_reversal_count"] / morphology[arm]["pitch_eligible_count"]
        )
    yaw_reduction = pct_reduction(
        morphology["strong"]["yaw_reversal_rate"], morphology["repaired"]["yaw_reversal_rate"]
    )
    pitch_reduction = pct_reduction(
        morphology["strong"]["pitch_reversal_rate"], morphology["repaired"]["pitch_reversal_rate"]
    )
    yaw_tv_reduction = pct_reduction(
        morphology["strong"]["yaw_directional_tv"], morphology["repaired"]["yaw_directional_tv"]
    )
    pitch_tv_reduction = pct_reduction(
        morphology["strong"]["pitch_directional_tv"], morphology["repaired"]["pitch_directional_tv"]
    )
    smooth_strong = np.asarray(
        [records["strong"][sid]["episode"]["trajectory_smoothness"] for sid in both_success], dtype=float
    )
    smooth_repaired = np.asarray(
        [records["repaired"][sid]["episode"]["trajectory_smoothness"] for sid in both_success], dtype=float
    )
    smooth_reduction = pct_reduction(float(np.mean(smooth_strong)), float(np.mean(smooth_repaired)))

    dctb_trace = [
        row
        for sid in ids
        for row in records["repaired"][sid].get("direction_continuity_trace", [])
    ]
    limiter_summaries = [records["repaired"][sid]["limiter_summary"] for sid in ids]
    dctb = {
        "trace_count": len(dctb_trace),
        "activation_count": sum(bool(row.get("activated")) for row in dctb_trace),
        "replacement_count": sum(bool(row.get("replaced")) for row in dctb_trace),
        "rank2_replacement_count": sum(row.get("replacement_gat_rank") == 2 for row in dctb_trace),
        "rank3_replacement_count": sum(row.get("replacement_gat_rank") == 3 for row in dctb_trace),
        "safety_critical_bypass_count": sum(
            row.get("reason") == "EXISTING_SAFETY_CRITICAL_BYPASS" for row in dctb_trace
        ),
    }
    early_bypass = {
        "early_bypass_count": sum(int(row.get("early_bypass_count", 0)) for row in limiter_summaries),
        "hard_bypass_count": sum(int(row.get("hard_bypass_count", 0)) for row in limiter_summaries),
        "combined_bypass_count": sum(int(row.get("combined_bypass_count", 0)) for row in limiter_summaries),
    }

    checks["decision_counts_reproduced"] = (
        aggregate["strong"]["team_success_count"] == int(round(100 * decision["team_success"]["strong"]))
        and aggregate["repaired"]["team_success_count"] == int(round(100 * decision["team_success"]["repaired"]))
        and len(both_success) == int(decision["both_success_count"])
    )
    checks["decision_morphology_reproduced"] = all(
        close(recomputed, float(decision[key]), 1.0e-10)
        for recomputed, key in (
            (yaw_reduction, "yaw_reversal_reduction_percent"),
            (pitch_reduction, "pitch_reversal_reduction_percent"),
            (yaw_tv_reduction, "yaw_directional_tv_reduction_percent"),
            (pitch_tv_reduction, "pitch_directional_tv_reduction_percent"),
            (smooth_reduction, "smoothness_reduction_percent_both_success"),
        )
    )
    checks["decision_interventions_reproduced"] = dctb == {
        key: decision["dctb"][key] for key in dctb
    } and early_bypass == decision["early_bypass"]
    checks["development_gate_and_stop_rule_consistent"] = (
        decision["DEV_GATE"] == "FAIL"
        and decision["HOLDOUT_AUTHORIZED"] is False
        and decision["FORMAL_V2_EXECUTED"] is False
        and not decision["gates"]["executed_reversal"]
        and not decision["gates"]["same_axis_directional_tv"]
    )
    holdout_candidates = list(ARTIFACT_ROOT.glob("*HOLDOUT*")) + list(
        (ARTIFACT_ROOT / "holdout").glob("**/*") if (ARTIFACT_ROOT / "holdout").exists() else []
    )
    formal_candidates = list(ARTIFACT_ROOT.glob("*FORMAL*")) + list(
        (ARTIFACT_ROOT / "formal_v2").glob("**/*") if (ARTIFACT_ROOT / "formal_v2").exists() else []
    )
    checks["holdout_not_generated_or_run"] = not any(path.is_file() for path in holdout_candidates)
    checks["formal_v2_not_run"] = not any(path.is_file() for path in formal_candidates)

    result = {
        "schema_version": "final_zigzag_independent_reconciliation_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "manifest": {
            "scenario_count": len(ids),
            "stage_counts": dict(stage_counts),
            "stage_family_cell_count": len(stage_family_counts),
            "scenes_hash_checked": len(entries),
        },
        "record_integrity": record_integrity,
        "aggregate_reproduction": aggregate,
        "paired_outcomes": {
            "both_success": len(both_success),
            "strong_only_success": len(strong_only),
            "repaired_only_success": len(repaired_only),
            "strong_only_scenarios": strong_only,
            "repaired_only_scenarios": repaired_only,
        },
        "raw_trajectory_morphology_reproduction": {
            "yaw_reversal_reduction_percent": yaw_reduction,
            "pitch_reversal_reduction_percent": pitch_reduction,
            "yaw_directional_tv_reduction_percent": yaw_tv_reduction,
            "pitch_directional_tv_reduction_percent": pitch_tv_reduction,
            "smoothness_reduction_percent_both_success": smooth_reduction,
        },
        "intervention_reproduction": {"dctb": dctb, "early_bypass": early_bypass},
        "stop_rule": {
            "development_gate": decision["DEV_GATE"],
            "holdout_file_count": sum(path.is_file() for path in holdout_candidates),
            "formal_v2_file_count": sum(path.is_file() for path in formal_candidates),
        },
    }
    output = ARTIFACT_ROOT / "FINAL_ZIGZAG_INDEPENDENT_RECONCILIATION.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(output)
    print(result["status"])
    if result["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"independent reconciliation failed: {failed}")


if __name__ == "__main__":
    main()
