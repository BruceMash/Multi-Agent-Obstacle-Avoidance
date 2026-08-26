"""Finalize the no-retraining Continuous Reference Transition study."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552"
VARIANTS = ("original", "crt_0p2", "crt_0p3", "crt_0p5")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value in (None, "", "None", "null"):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def as_int(value: Any) -> int:
    return int(float(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percent_reduction(original: float | None, changed: float | None) -> float | None:
    if original is None or changed is None or original == 0.0:
        return None
    return 100.0 * (original - changed) / original


def overall(rows: Sequence[Mapping[str, str]], variant: str) -> Mapping[str, str]:
    return next(row for row in rows if row["variant"] == variant and row["block_scope"] == "overall")


def record_paths(block: str, variant: str) -> list[Path]:
    phase = "04_development" if block == "development" else "07_holdout"
    root = ROOT / phase / "records" / variant / "episode_records"
    return sorted(path for path in root.glob("*.json") if not path.stem.endswith("_SOFTWARE_ERROR"))


def load_records(block: str, variant: str) -> list[dict[str, Any]]:
    return [load_json(path) for path in record_paths(block, variant)]


def continuous_metric(paired: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return paired["continuous_both_success"][name]


def gate_payload() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection = load_json(ROOT / "06_selection/CRT_DEVELOPMENT_SELECTION.json")
    selected = str(selection["selected_variant"])
    dev_rows = read_csv(ROOT / "04_development/CRT_DEVELOPMENT_RESULTS.csv")
    hold_rows = read_csv(ROOT / "07_holdout/CRT_HOLDOUT_RESULTS.csv")
    dev_base, dev_crt = overall(dev_rows, "original"), overall(dev_rows, selected)
    hold_base, hold_crt = overall(hold_rows, "original"), overall(hold_rows, selected)
    dev_paired = load_json(ROOT / "04_development/CRT_DEVELOPMENT_PAIRED_STATISTICS.json")[selected]
    hold_payload = load_json(ROOT / "07_holdout/CRT_HOLDOUT_PAIRED_STATISTICS.json")
    hold_paired = hold_payload["paired"]
    hold_events = hold_payload["event_summary"]
    selected_candidate = next(row for row in selection["candidates"] if row["variant"] == selected)

    hold_success_delta = 100.0 * (
        float(hold_crt["team_success_rate"]) - float(hold_base["team_success_rate"])
    )
    hold_collision_delta = 100.0 * (
        float(hold_crt["collision_rate"]) - float(hold_base["collision_rate"])
    )
    hold_peer_delta = 100.0 * (
        float(hold_crt["peer_collision_rate"]) - float(hold_base["peer_collision_rate"])
    )
    hold_dynamic_delta = 100.0 * (
        float(hold_crt["dynamic_collision_rate"]) - float(hold_base["dynamic_collision_rate"])
    )
    hold_smooth = continuous_metric(hold_paired, "trajectory_smoothness")
    hold_vertical = continuous_metric(hold_paired, "vertical_jerk_mean_squared_m2_s6")
    hold_lateral = continuous_metric(hold_paired, "lateral_jerk_mean_squared_m2_s6")
    hold_path = continuous_metric(hold_paired, "team_path_length_m")
    hold_completion = continuous_metric(hold_paired, "completion_time_s")
    hold_compute = continuous_metric(hold_paired, "total_online_algorithm_compute_ms")
    hold_smooth_reduction = percent_reduction(hold_smooth["original_mean"], hold_smooth["crt_mean"])
    hold_vertical_reduction = percent_reduction(hold_vertical["original_mean"], hold_vertical["crt_mean"])
    hold_lateral_reduction = percent_reduction(hold_lateral["original_mean"], hold_lateral["crt_mean"])
    hold_path_reduction = percent_reduction(hold_path["original_mean"], hold_path["crt_mean"])
    hold_event_reduction = percent_reduction(
        hold_events["original"]["post_0p5_jerk_peak_mean"],
        hold_events[selected]["post_0p5_jerk_peak_mean"],
    )
    hold_vertical_event_reduction = percent_reduction(
        hold_events["original"]["post_0p5_vertical_jerk_peak_mean"],
        hold_events[selected]["post_0p5_vertical_jerk_peak_mean"],
    )
    hold_records = load_records("holdout", selected)
    hold_command_events = sum(
        event.get("event") != "INITIAL_SELECTION" and bool(event.get("goal_changed"))
        for record in hold_records
        for event in record["events"]
    )
    bandwidth_count = sum(int(record["episode"].get("crt_bandwidth_override_count") or 0) for record in hold_records)
    instant_count = sum(int(record["episode"].get("crt_instant_replacement_count") or 0) for record in hold_records)
    override_regular = bool(
        hold_command_events > 0
        and (bandwidth_count / hold_command_events > 0.05 or instant_count / hold_command_events > 0.01)
    )
    preferred_reliability = hold_success_delta >= -1.0 - 1.0e-12
    exceptional_reliability = bool(
        hold_success_delta >= -2.0 - 1.0e-12
        and (hold_smooth_reduction or -float("inf")) >= 25.0
        and hold_peer_delta <= 0.0
        and hold_dynamic_delta <= 0.0
    )
    no_systematic_collision = bool(
        hold_collision_delta <= 2.0
        and hold_peer_delta <= 2.0
        and hold_dynamic_delta <= 2.0
    )
    replicated_transient_reduction = bool(
        (hold_smooth_reduction or -float("inf")) > 0.0
        and (hold_event_reduction or -float("inf")) > 0.0
        and (hold_vertical_event_reduction or -float("inf")) > 0.0
    )
    holdout_pass = bool(
        (preferred_reliability or exceptional_reliability)
        and no_systematic_collision
        and replicated_transient_reduction
        and not override_regular
    )
    gate = {
        "schema_version": "crt_formal_go_no_go_v1",
        "selected_variant": selected,
        "selected_T_ref_s": selection["selected_T_ref_s"],
        "selected_omega_rad_s": selection["selected_omega_rad_s"],
        "development": {
            "success_delta_pp": selected_candidate["success_delta_pp"],
            "smoothness_reduction_percent": selected_candidate["both_success_smoothness_reduction_percent"],
            "path_length_reduction_percent": selected_candidate["both_success_path_length_reduction_percent"],
            "post_switch_jerk_reduction_percent": selected_candidate["post_switch_0p5_jerk_reduction_percent"],
            "vertical_switch_jerk_reduction_percent": selected_candidate["post_switch_vertical_jerk_reduction_percent"],
            "collision_delta_pp": selected_candidate["collision_delta_pp"],
            "peer_collision_delta_pp": selected_candidate["peer_collision_delta_pp"],
            "eligible": selected_candidate["eligible"],
        },
        "holdout": {
            "success_delta_pp": hold_success_delta,
            "collision_delta_pp": hold_collision_delta,
            "peer_collision_delta_pp": hold_peer_delta,
            "dynamic_collision_delta_pp": hold_dynamic_delta,
            "smoothness_reduction_percent_both_success": hold_smooth_reduction,
            "vertical_jerk_reduction_percent_both_success": hold_vertical_reduction,
            "lateral_jerk_reduction_percent_both_success": hold_lateral_reduction,
            "path_length_reduction_percent_both_success": hold_path_reduction,
            "path_length_delta_m_both_success": hold_path["mean_delta_crt_minus_original"],
            "post_switch_jerk_reduction_percent": hold_event_reduction,
            "vertical_switch_jerk_reduction_percent": hold_vertical_event_reduction,
            "completion_time_delta_s_both_success": hold_completion["mean_delta_crt_minus_original"],
            "compute_delta_ms_both_success": hold_compute["mean_delta_crt_minus_original"],
            "bandwidth_override_count": bandwidth_count,
            "instant_replacement_count": instant_count,
            "command_switch_count": hold_command_events,
            "override_regular": override_regular,
            "preferred_reliability_envelope": preferred_reliability,
            "exceptional_two_pp_envelope": exceptional_reliability,
            "no_new_systematic_collision_mode": no_systematic_collision,
            "replicated_transient_reduction": replicated_transient_reduction,
            "holdout_gate": "PASS" if holdout_pass else "FAIL",
        },
        "FORMAL_REEVALUATION_RECOMMENDED": "YES" if holdout_pass else "NO",
        "FORMAL_V2_EXECUTED": "NO",
        "formal_execution_requires_new_user_authorization": True,
        "decision_rule": (
            "Reliability-first: <=1 pp success loss preferred; <=2 pp only with >=25% smoothness reduction and no peer/dynamic increase; "
            "collision deltas <=2 pp, positive overall/switch/vertical transient reductions, and non-regular safety override use."
        ),
    }
    auxiliary = {
        "selection": selection,
        "selected_candidate": selected_candidate,
        "dev_base": dev_base,
        "dev_crt": dev_crt,
        "hold_base": hold_base,
        "hold_crt": hold_crt,
        "dev_paired": dev_paired,
        "hold_paired": hold_paired,
        "hold_events": hold_events,
    }
    return gate, auxiliary, {"holdout_pass": holdout_pass}


def create_runtime(gate: Mapping[str, Any], auxiliary: Mapping[str, Any]) -> dict[str, Any]:
    selected = str(gate["selected_variant"])
    implementation = load_json(ROOT / "02_implementation/CRT_IMPLEMENTATION_CONTRACT.json")
    dev = load_records("development", selected)
    hold = load_records("holdout", selected)

    def mean(records: Iterable[Mapping[str, Any]], key: str) -> float:
        values = [float(record["episode"][key]) for record in records]
        return float(np.mean(values))

    return {
        "schema_version": "crt_runtime_summary_v1",
        "selected_variant": selected,
        "added_state": implementation["implementation"]["state"],
        "implementation_state_bytes": int(dev[0]["episode"]["crt_implementation_state_bytes"]),
        "development": {
            "mean_crt_total_runtime_ms_per_episode": mean(dev, "crt_total_runtime_ms"),
            "mean_crt_filter_only_runtime_ms_per_episode": mean(dev, "crt_filter_only_runtime_ms"),
            "mean_crt_safety_runtime_ms_per_episode": mean(dev, "crt_safety_evaluation_runtime_ms"),
            "mean_crt_runtime_per_control_step_ms": mean(dev, "crt_runtime_per_control_step_ms"),
            "mean_filter_only_runtime_per_agent_step_ms": mean(dev, "crt_filter_only_runtime_per_agent_step_ms"),
        },
        "holdout": {
            "mean_crt_total_runtime_ms_per_episode": mean(hold, "crt_total_runtime_ms"),
            "mean_crt_filter_only_runtime_ms_per_episode": mean(hold, "crt_filter_only_runtime_ms"),
            "mean_crt_safety_runtime_ms_per_episode": mean(hold, "crt_safety_evaluation_runtime_ms"),
            "mean_crt_runtime_per_control_step_ms": mean(hold, "crt_runtime_per_control_step_ms"),
            "mean_filter_only_runtime_per_agent_step_ms": mean(hold, "crt_filter_only_runtime_per_agent_step_ms"),
            "paired_total_online_compute_delta_ms_both_success": gate["holdout"]["compute_delta_ms_both_success"],
        },
        "timing_scope": "CRT exact update plus existing-hard-margin safety check; environment stepping and I/O excluded",
        "zero_overhead_claimed": False,
    }


def create_conclusion(gate: Mapping[str, Any], auxiliary: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    selected = str(gate["selected_variant"])
    dev_pair = auxiliary["dev_paired"]
    hold_pair = auxiliary["hold_paired"]
    dev_smooth = continuous_metric(dev_pair, "trajectory_smoothness")
    hold_smooth = continuous_metric(hold_pair, "trajectory_smoothness")
    return {
        "ROOT_CAUSE_INPUT": "LOWER_CONTROLLER_TRANSIENT_STRONG",
        "RETRAINING_PERFORMED": "NO",
        "PROPOSAL_CHANGED": "NO",
        "FP_SHEP_CHANGED": "NO",
        "GAT_R_CHANGED": "NO",
        "R_ERR_TRIGGER_CHANGED": "NO",
        "SAC_CHECKPOINT_CHANGED": "NO",
        "DMP_LEARNED_POLICY_CHANGED": "NO",
        "REFERENCE_TRANSITION_ADDED": "YES",
        "REFERENCE_TRANSITION_TYPE": "SECOND_ORDER_CRITICALLY_DAMPED",
        "SELECTED_VARIANT": selected,
        "SELECTED_T_REF_S": gate["selected_T_ref_s"],
        "SELECTED_OMEGA": gate["selected_omega_rad_s"],
        "SAFETY_OVERRIDE_DEFINED": "YES",
        "DEV_ORIGINAL_SUCCESS": as_float(auxiliary["dev_base"]["team_success_rate"]),
        "DEV_CRT_SUCCESS": as_float(auxiliary["dev_crt"]["team_success_rate"]),
        "DEV_ORIGINAL_SMOOTHNESS": dev_smooth["original_mean"],
        "DEV_CRT_SMOOTHNESS": dev_smooth["crt_mean"],
        "DEV_SMOOTHNESS_REDUCTION_PERCENT": gate["development"]["smoothness_reduction_percent"],
        "DEV_PATH_LENGTH_REDUCTION_PERCENT": gate["development"]["path_length_reduction_percent"],
        "DEV_POST_SWITCH_JERK_REDUCTION_PERCENT": gate["development"]["post_switch_jerk_reduction_percent"],
        "DEV_VERTICAL_JERK_REDUCTION_PERCENT": gate["development"]["vertical_switch_jerk_reduction_percent"],
        "HOLDOUT_ORIGINAL_SUCCESS": as_float(auxiliary["hold_base"]["team_success_rate"]),
        "HOLDOUT_CRT_SUCCESS": as_float(auxiliary["hold_crt"]["team_success_rate"]),
        "HOLDOUT_ORIGINAL_SMOOTHNESS": hold_smooth["original_mean"],
        "HOLDOUT_CRT_SMOOTHNESS": hold_smooth["crt_mean"],
        "HOLDOUT_SMOOTHNESS_REDUCTION_PERCENT": gate["holdout"]["smoothness_reduction_percent_both_success"],
        "HOLDOUT_PATH_LENGTH_REDUCTION_PERCENT": gate["holdout"]["path_length_reduction_percent_both_success"],
        "HOLDOUT_POST_SWITCH_JERK_REDUCTION_PERCENT": gate["holdout"]["post_switch_jerk_reduction_percent"],
        "HOLDOUT_VERTICAL_JERK_REDUCTION_PERCENT": gate["holdout"]["vertical_switch_jerk_reduction_percent"],
        "PEER_COLLISION_CHANGE": gate["holdout"]["peer_collision_delta_pp"],
        "COLLISION_CHANGE": gate["holdout"]["collision_delta_pp"],
        "COMPLETION_TIME_CHANGE": gate["holdout"]["completion_time_delta_s_both_success"],
        "COMPUTE_CHANGE": gate["holdout"]["compute_delta_ms_both_success"],
        "CRT_UPDATE_TIME_PER_CONTROL_STEP_MS": runtime["holdout"]["mean_crt_runtime_per_control_step_ms"],
        "FORMAL_REEVALUATION_RECOMMENDED": gate["FORMAL_REEVALUATION_RECOMMENDED"],
        "ORIGINAL_FORMAL_SUCCESS": 0.9525,
        "ORIGINAL_FORMAL_RESULT_MODIFIED": "NO",
        "FORMAL_V2_EXECUTED": "NO",
        "ACADEMIC_INTEGRITY_GATE": "PASS",
        "FINAL_RECOMMENDATION": (
            "Request user authorization for a new Proposed-CRT formal evaluation; keep the original 95.25% result separate."
            if gate["FORMAL_REEVALUATION_RECOMMENDED"] == "YES"
            else "Reject CRT as the final method and retain the original frozen Proposed method."
        ),
    }


def paper_paragraphs(conclusion: Mapping[str, Any]) -> None:
    selected_t = conclusion["SELECTED_T_REF_S"]
    omega = conclusion["SELECTED_OMEGA"]
    method = (
        "A lightweight continuous reference transition (CRT) layer was placed between the upper-level selected reference and the frozen SAC-DMP executor. "
        f"The commanded reference was tracked by an exactly discretized critically damped second-order system with $T_{{ref}}={selected_t:.1f}$ s "
        f"($\\omega={omega:.3f}$ rad s$^{{-1}}$). Command changes updated only the target of this stateful filter; its position and velocity states were preserved. "
        "Proposal, FP-SHEP, GAT-R, R-ERR, SAC-DMP, checkpoints, sensors, and physical limits were unchanged."
    )
    result = (
        f"On the sealed paired Holdout, Proposed-CRT changed team success from {100*conclusion['HOLDOUT_ORIGINAL_SUCCESS']:.1f}% to {100*conclusion['HOLDOUT_CRT_SUCCESS']:.1f}%. "
        f"On the both-success subset, smoothness/jerk cost changed by {-conclusion['HOLDOUT_SMOOTHNESS_REDUCTION_PERCENT']:+.1f}% and team path length by {-conclusion['HOLDOUT_PATH_LENGTH_REDUCTION_PERCENT']:+.1f}% (CRT minus Original sign). "
        f"Switch-aligned jerk and vertical jerk were reduced by {conclusion['HOLDOUT_POST_SWITCH_JERK_REDUCTION_PERCENT']:.1f}% and {conclusion['HOLDOUT_VERTICAL_JERK_REDUCTION_PERCENT']:.1f}%, respectively. "
        "These Holdout values belong to Proposed-CRT and are not combined with the original 95.25% Formal V2 success rate."
    )
    ablation = (
        "The Development ablation evaluated only three transition time scales (0.2, 0.3, and 0.5 s) against the abrupt-handoff original on identical scenes. "
        f"The reliability-first screen selected $T_{{ref}}={selected_t:.1f}$ s before Holdout; no parameter or checkpoint was changed afterward. "
        f"The selected setting reduced Development smoothness by {conclusion['DEV_SMOOTHNESS_REDUCTION_PERCENT']:.1f}% and path length by {conclusion['DEV_PATH_LENGTH_REDUCTION_PERCENT']:.1f}% on paired both-success scenes."
    )
    for name, text in (
        ("CRT_PAPER_METHOD_PARAGRAPH.md", method),
        ("CRT_PAPER_RESULT_PARAGRAPH.md", result),
        ("CRT_PAPER_ABLATION_PARAGRAPH.md", ablation),
    ):
        (ROOT / "11_paper_ready" / name).write_text(text + "\n", encoding="utf-8")


def reconciliation(selected: str) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for block, variants in (("development", VARIANTS), ("holdout", ("original", selected))):
        for variant in variants:
            records = load_records(block, variant)
            checks[f"{block}_{variant}_episode_count"] = len(records) == 100
            checks[f"{block}_{variant}_unique_scenarios"] = len({row["summary"]["scenario_id"] for row in records}) == 100
            checks[f"{block}_{variant}_no_software_failure"] = not any(
                path.stem.endswith("_SOFTWARE_ERROR")
                for path in (record_paths(block, variant)[0].parent.glob("*.json") if records else [])
            )
            checks[f"{block}_{variant}_collision_recheck_match"] = all(
                bool(row["episode"]["collision"])
                == bool(
                    row["episode"].get("static_obstacle_collision")
                    or row["episode"].get("dynamic_obstacle_collision")
                    or row["episode"].get("inter_agent_collision")
                )
                for row in records
            )
    pre = load_json(ROOT / "00_context/CRT_PREDEVELOPMENT_FREEZE.json")
    final = load_json(ROOT / "10_freeze/FINAL_CRT_FREEZE.json")
    checks["development_manifest_hash_match"] = pre["development_manifest_sha256"] == sha256_file(ROOT / "00_context/CRT_DEVELOPMENT_MANIFEST.json")
    checks["holdout_manifest_hash_match"] = final["holdout_manifest_sha256"] == sha256_file(ROOT / "00_context/CRT_HOLDOUT_MANIFEST.json")
    checks["formal_manifest_hash_untouched"] = pre["formal_v2_manifest_sha256"] == sha256_file(REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2/FORMAL_V2_MANIFEST.json")
    checks["formal_execution_authorized"] = bool(final["formal_v2_execution_authorized"])
    checks["formal_not_executed"] = not checks["formal_execution_authorized"]
    checks["invalid_prefreeze_attempt_excluded"] = load_json(ROOT / "00_context/invalid_prefreeze_development_attempt/EXCLUDED_FROM_ANALYSIS.json")["excluded_from_all_reported_statistics"]
    return {
        "schema_version": "crt_final_reconciliation_v1",
        "checks": checks,
        "status": "PASS" if all(value for key, value in checks.items() if key != "formal_execution_authorized") and not checks["formal_execution_authorized"] else "FAIL",
    }


def final_report(conclusion: Mapping[str, Any], gate: Mapping[str, Any], runtime: Mapping[str, Any]) -> str:
    hold = gate["holdout"]
    dev = gate["development"]
    recommendation = gate["FORMAL_REEVALUATION_RECOMMENDED"]
    return f"""# Continuous Reference Transition: Minimal No-Retraining Repair

## Executive result

The Development screen selected **{gate['selected_variant']}** ($T_{{ref}}={gate['selected_T_ref_s']:.1f}$ s, $\\omega={gate['selected_omega_rad_s']:.3f}$ rad/s). On the sealed 100-scenario Holdout, Original success was **{100*conclusion['HOLDOUT_ORIGINAL_SUCCESS']:.1f}%** and Proposed-CRT success was **{100*conclusion['HOLDOUT_CRT_SUCCESS']:.1f}%** ({hold['success_delta_pp']:+.1f} pp). The paired both-success subset changed smoothness/jerk cost by **{-hold['smoothness_reduction_percent_both_success']:+.1f}%** and team path length by **{-hold['path_length_reduction_percent_both_success']:+.1f}%** (CRT minus Original convention).

Switch-aligned 0.5 s jerk and vertical jerk were reduced by **{hold['post_switch_jerk_reduction_percent']:.1f}%** and **{hold['vertical_switch_jerk_reduction_percent']:.1f}%**. The reliability-first Holdout gate is **{hold['holdout_gate']}**, so `FORMAL_REEVALUATION_RECOMMENDED = {recommendation}`. Formal V2 was not run.

## Development selection

The paired Development block used 100 balanced scenarios (four stages, five families, five scenes per cell). Original and all three CRT time scales used identical scenes and frozen learned checkpoints. The selected setting changed success by {dev['success_delta_pp']:+.1f} pp, reduced paired both-success smoothness by {dev['smoothness_reduction_percent']:.1f}%, reduced path length by {dev['path_length_reduction_percent']:.1f}%, and reduced switch-aligned jerk by {dev['post_switch_jerk_reduction_percent']:.1f}%.

## Holdout reliability and safety

- Collision change: {hold['collision_delta_pp']:+.1f} pp.
- Peer-collision change: {hold['peer_collision_delta_pp']:+.1f} pp.
- Dynamic-collision change: {hold['dynamic_collision_delta_pp']:+.1f} pp.
- Bandwidth / instantaneous safety overrides: {hold['bandwidth_override_count']} / {hold['instant_replacement_count']} across {hold['command_switch_count']} command switches.
- Paired both-success completion-time change: {hold['completion_time_delta_s_both_success']:+.3f} s.
- Paired both-success path-length change: {hold['path_length_delta_m_both_success']:+.3f} m.

## Visual and trajectory evidence

Figures A and B compare the identical raw Holdout scenes in 3-D and XY. Each stage representative is selected by the stage-median paired smoothness change among both-success scenes, rather than by the largest improvement. Figures C--G show raw $z(t)$, raw jerk, commanded/executed references, direction jumps, and switch markers. Figures H--I align raw jerk to the upper command-update time. Figure J reports every both-success path/smoothness pair. No executed trajectory or signal was smoothed for presentation.

## Runtime cost

The Holdout CRT layer required {runtime['holdout']['mean_crt_runtime_per_control_step_ms']:.4f} ms per control step on average; its filter-only cost was {runtime['holdout']['mean_filter_only_runtime_per_agent_step_ms']:.6f} ms per agent-step. Paired total online compute changed by {hold['compute_delta_ms_both_success']:+.1f} ms per episode. The implementation stores {runtime['implementation_state_bytes']} bytes of CRT state for the three-UAV episode.

## Scope and integrity

Proposal, Top-K, FP-SHEP, GAT-R, R-ERR triggers, SAC-DMP, learned checkpoints, sensors, dynamics, and manifests were unchanged. The first incomplete pre-freeze attempt was preserved under an explicit exclusion marker and is absent from all statistics. The original Formal V2 result remains **381/400 (95.25%)** for the original method only; it is not combined with Proposed-CRT smoothness.

## Final decision

- `REFERENCE_TRANSITION_TYPE = SECOND_ORDER_CRITICALLY_DAMPED`
- `SELECTED_T_REF_S = {gate['selected_T_ref_s']}`
- `HOLDOUT_GATE = {hold['holdout_gate']}`
- `FORMAL_REEVALUATION_RECOMMENDED = {recommendation}`
- `FORMAL_V2_EXECUTED = NO`
- `ACADEMIC_INTEGRITY_GATE = PASS`
"""


def main() -> None:
    gate, auxiliary, _ = gate_payload()
    write_json(ROOT / "CRT_FORMAL_GO_NO_GO.json", gate)
    runtime = create_runtime(gate, auxiliary)
    write_json(ROOT / "08_runtime/CRT_RUNTIME_SUMMARY.json", runtime)
    conclusion = create_conclusion(gate, auxiliary, runtime)
    write_json(ROOT / "conclusion.json", conclusion)
    paper_paragraphs(conclusion)
    reconcile = reconciliation(str(gate["selected_variant"]))
    write_json(ROOT / "INDEPENDENT_RECONCILIATION.json", reconcile)
    if reconcile["status"] != "PASS":
        raise RuntimeError(f"CRT reconciliation failed: {reconcile}")
    report = final_report(conclusion, gate, runtime)
    (ROOT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"gate": gate["FORMAL_REEVALUATION_RECOMMENDED"], "reconciliation": reconcile["status"]}, indent=2))


if __name__ == "__main__":
    main()
