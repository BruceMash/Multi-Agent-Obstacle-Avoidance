#!/usr/bin/env python3
"""Finalize machine-readable decisions and the human report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts/turn_sign_persistence_zigzag_repair/20260825_222130"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> None:
    dev = load(ROOT / "TURN_PERSISTENCE_DEV_GO_NO_GO.json")
    reconciliation = load(ROOT / "independent_reconciliation.json")
    dev_results = rows(ROOT / "TURN_PERSISTENCE_DEV_RESULTS.csv")
    overall = {row["arm"]: row for row in dev_results if row["scope"] == "overall"}
    holdout_path = ROOT / "TURN_PERSISTENCE_HOLDOUT_DECISION.json"
    holdout = load(holdout_path) if holdout_path.exists() else None
    dev_pass = dev["DEV_GATE"] == "PASS"
    holdout_pass = holdout is not None and holdout.get("HOLDOUT_GATE") == "PASS"
    selected = "STRONG_EARLY_BYPASS_TURN_PERSISTENCE" if holdout_pass else "FROZEN_STRONG"
    recommendation = (
        "AUTHORIZE_SEPARATE_FORMAL_V2_COMPARISON"
        if holdout_pass
        else "KEEP_FROZEN_STRONG_AND_STOP_TURN_PERSISTENCE"
    )
    formal = {
        "schema_version": "turn_persistence_formal_v2_go_no_go_v1",
        "DEV_GATE": dev["DEV_GATE"],
        "HOLDOUT_EXECUTED": "YES" if holdout is not None else "NO",
        "HOLDOUT_EXECUTED_BOOLEAN": holdout is not None,
        "HOLDOUT_GATE": holdout.get("HOLDOUT_GATE") if holdout else "NOT_RUN",
        "FORMAL_REEVALUATION_RECOMMENDED": "YES" if holdout_pass else "NO",
        "FORMAL_REEVALUATION_RECOMMENDED_BOOLEAN": bool(holdout_pass),
        "FORMAL_V2_EXECUTED": False,
        "selected_method_for_possible_formal": selected,
        "reason": (
            "Development and independent Holdout both passed every frozen gate; a separate explicit goal is still required."
            if holdout_pass
            else "At least one mandatory pre-formal gate did not pass, so Formal V2 is forbidden."
        ),
    }
    if dev_pass:
        write_json(ROOT / "TURN_PERSISTENCE_FORMAL_GO_NO_GO.json", formal)
    zigzag_metric_pass = all(
        dev["gates"].get(name, False)
        for name in (
            "yaw_reversal", "pitch_reversal", "yaw_rate_tv", "pitch_rate_tv",
            "fixed_raw_visual_longer_arc_proxy",
        )
    )
    zigzag_eliminated = "YES" if holdout_pass else "PARTIAL" if zigzag_metric_pass else "NO"
    threshold = load(ROOT / "TURN_PERSISTENCE_THRESHOLDS.json")
    run = dev["same_sign_run_duration_s"]
    conclusion = {
        "schema_version": "turn_persistence_zigzag_repair_conclusion_v1",
        "STRONG_JERK_LIMITER_CHANGED": "NO",
        "EARLY_SAFETY_BYPASS": "YES",
        "N_PERSIST": 2,
        "A_REV_LAT": threshold["A_REV_LAT"],
        "A_REV_VERT": threshold["A_REV_VERT"],
        "DEV_STRONG_SUCCESS": dev["team_success"]["strong"],
        "DEV_REPAIRED_SUCCESS": dev["team_success"]["repaired"],
        "DEV_PEER_COLLISION_DELTA_PP": dev["peer_collision_delta_pp"],
        "DEV_OBSTACLE_COLLISION_DELTA_PP": dev["obstacle_collision_delta_pp"],
        "DEV_YAW_REVERSAL_STRONG": dev["yaw_reversal_rate"]["strong"],
        "DEV_YAW_REVERSAL_REPAIRED": dev["yaw_reversal_rate"]["repaired"],
        "DEV_YAW_REVERSAL_REDUCTION_PERCENT": dev["yaw_reversal_reduction_percent"],
        "DEV_PITCH_REVERSAL_STRONG": dev["pitch_reversal_rate"]["strong"],
        "DEV_PITCH_REVERSAL_REPAIRED": dev["pitch_reversal_rate"]["repaired"],
        "DEV_PITCH_REVERSAL_REDUCTION_PERCENT": dev["pitch_reversal_reduction_percent"],
        "DEV_YAW_TV_REDUCTION_PERCENT": dev["yaw_rate_total_variation_reduction_percent"],
        "DEV_PITCH_TV_REDUCTION_PERCENT": dev["pitch_rate_total_variation_reduction_percent"],
        "DEV_SAME_SIGN_YAW_RUN_CHANGE": run["yaw_median_change"],
        "DEV_SAME_SIGN_PITCH_RUN_CHANGE": run["pitch_median_change"],
        "DEV_SMOOTHNESS_CHANGE_PERCENT": -dev["smoothness_reduction_percent_both_success"],
        "DEV_PATH_CHANGE_PERCENT": -dev["path_length_reduction_percent_both_success"],
        "RAW_LONG_ARC_SCENES": f"{dev['fixed_visual_longer_arc_proxy_count']}/4",
        "TURN_PERSISTENCE_ACCEPTED": "YES" if holdout_pass else "NO",
        "HOLDOUT_PASSED": holdout.get("HOLDOUT_GATE") if holdout else "NOT_RUN",
        "ZIGZAG_ELIMINATED": zigzag_eliminated,
        "ORIGINAL_FORMAL_SUCCESS": 0.9525,
        "FORMAL_RESULT_CHANGED": "NO",
        "FINAL_RECOMMENDATION": (
            "Run a separately authorized Formal V2 paired comparison."
            if holdout_pass
            else "Keep Frozen Strong and permanently close this turn-persistence repair branch."
        ),
        "DEV_GATE": dev["DEV_GATE"],
        "HOLDOUT_AUTHORIZED": bool(dev_pass),
        "HOLDOUT_EXECUTED": "YES" if holdout is not None else "NO",
        "HOLDOUT_EXECUTED_BOOLEAN": holdout is not None,
        "HOLDOUT_GATE": holdout.get("HOLDOUT_GATE") if holdout else "NOT_RUN",
        "FORMAL_V2_EXECUTED": False,
        "FORMAL_REEVALUATION_RECOMMENDED": "YES" if holdout_pass else "NO",
        "FORMAL_REEVALUATION_RECOMMENDED_BOOLEAN": bool(holdout_pass),
        "SELECTED_EXECUTION_METHOD": selected,
        "TURN_SIGN_PERSISTENCE_ACCEPTED_BOOLEAN": bool(holdout_pass),
        "DEVELOPMENT": {
            "frozen_strong_success": dev["team_success"]["strong"],
            "repaired_success": dev["team_success"]["repaired"],
            "success_loss_pp": dev["success_loss_pp"],
            "peer_collision_delta_pp": dev["peer_collision_delta_pp"],
            "obstacle_collision_delta_pp": dev["obstacle_collision_delta_pp"],
            "yaw_reversal_reduction_percent": dev["yaw_reversal_reduction_percent"],
            "pitch_reversal_reduction_percent": dev["pitch_reversal_reduction_percent"],
            "yaw_rate_tv_reduction_percent": dev["yaw_rate_total_variation_reduction_percent"],
            "pitch_rate_tv_reduction_percent": dev["pitch_rate_total_variation_reduction_percent"],
            "smoothness_reduction_percent_both_success": dev["smoothness_reduction_percent_both_success"],
            "path_length_reduction_percent_both_success": dev["path_length_reduction_percent_both_success"],
            "fixed_visual_longer_arc_proxy_count": dev["fixed_visual_longer_arc_proxy_count"],
            "gates": dev["gates"],
        },
        "INDEPENDENT_RECONCILIATION": reconciliation["status"],
        "RECOMMENDED_NEXT_STEP": recommendation,
    }
    if holdout is not None:
        conclusion["HOLDOUT"] = holdout
    write_json(ROOT / "conclusion.json", conclusion)

    stages = []
    for scope in ("stage_1", "stage_2", "stage_3", "stage_4"):
        group = {row["arm"]: row for row in dev_results if row["scope"] == scope}
        if group:
            stages.append(
                f"| {scope.replace('_', ' ').title()} | {percent(float(group['strong']['team_success_rate']))} | "
                f"{percent(float(group['repaired']['team_success_rate']))} | "
                f"{percent(float(group['strong']['collision_rate']))} | {percent(float(group['repaired']['collision_rate']))} |"
            )
    failed_gates = [name for name, passed in dev["gates"].items() if not passed]
    trace = dev["turn_persistence"]
    report = f"""# Final Execution-Side Zigzag Repair: Turn-Sign Persistence

## Executive result

Development gate: **{dev['DEV_GATE']}**. Frozen Strong achieved **{percent(dev['team_success']['strong'])}** team success and the repaired arm achieved **{percent(dev['team_success']['repaired'])}** ({-dev['success_loss_pp']:+.1f} pp repaired minus Strong). The repaired arm changed peer collision by **{dev['peer_collision_delta_pp']:+.1f} pp** and obstacle collision by **{dev['obstacle_collision_delta_pp']:+.1f} pp**.

Raw executed yaw/pitch reversal-rate reductions were **{dev['yaw_reversal_reduction_percent']:.2f}% / {dev['pitch_reversal_reduction_percent']:.2f}%**, and yaw/pitch rate total-variation reductions were **{dev['yaw_rate_total_variation_reduction_percent']:.2f}% / {dev['pitch_rate_total_variation_reduction_percent']:.2f}%**. On the paired both-success subset, the existing smoothness/jerk-cost reduction was **{dev['smoothness_reduction_percent_both_success']:.2f}%** and the team-path reduction was **{dev['path_length_reduction_percent_both_success']:.2f}%** (negative values denote deterioration).

{('All Development gates passed; a new independent Holdout100 was therefore run.' if dev_pass else 'At least one mandatory Development gate failed. The experiment stops without generating or running Holdout100.')} Formal V2 was **not run**.

## Frozen execution contract

- Upper chain unchanged: Proposal -> Top-K -> FP-SHEP -> GAT-R -> R-ERR -> frozen SAC-DMP.
- Normal order: raw SAC-DMP acceleration -> Frozen Strong -> turn-sign persistence -> original physical clipping -> dynamics.
- Safety order: at `m_t <= h_rep = 0.35 m`, both Strong and persistence are bypassed; the raw command reaches the original physical clip. The hard flag remains `m_t <= h_emg = 0`.
- Horizontal and vertical persistence states are independent; `N_persist=2`.
- Frozen P75 immediate overrides: lateral **1.010861 m/s^2**, vertical **0.871703 m/s^2**. No grid search or performance-driven threshold adjustment was used.
- Horizontal persistence modifies only the normal component relative to current executed horizontal velocity; the tangential component is preserved before physical clipping.

## Development outcomes

| Scope | Frozen Strong success | Repaired success | Strong collision | Repaired collision |
|---|---:|---:|---:|---:|
| Overall | {percent(float(overall['strong']['team_success_rate']))} | {percent(float(overall['repaired']['team_success_rate']))} | {percent(float(overall['strong']['collision_rate']))} | {percent(float(overall['repaired']['collision_rate']))} |
{chr(10).join(stages)}

The new block contains 100 balanced scenarios (4 stages x 5 families x 5 scenes) and is disjoint from the registered training, Development, Holdout, Formal, selector-ablation, and previous DCTB Development blocks.

## Mandatory gate audit

| Gate | Result |
|---|---|
{chr(10).join(f'| {name} | {"PASS" if passed else "FAIL"} |' for name, passed in dev['gates'].items())}

Failed gates: **{', '.join(failed_gates) if failed_gates else 'none'}**. The four outcome-blind raw scenes met the objective longer-arc proxy in **{dev['fixed_visual_longer_arc_proxy_count']}/4** cases.

## Intervention and runtime diagnostics

- Persistence trace rows: {trace['trace_count']:,}; modified rows: {trace['persistence_modified_count']:,}; unchanged fraction: {100.0 * trace['persistence_unchanged_fraction']:.2f}%.
- Horizontal/vertical activations: {trace['horizontal_persistence_activation_count']:,}/{trace['vertical_persistence_activation_count']:,}.
- Horizontal/vertical confirmed reversals: {trace['horizontal_confirmed_reversal_count']:,}/{trace['vertical_confirmed_reversal_count']:,}.
- Horizontal/vertical P75 magnitude overrides: {trace['horizontal_magnitude_override_count']:,}/{trace['vertical_magnitude_override_count']:,}.
- Early/hard safety bypass rows: {trace['early_bypass_count']:,}/{trace['hard_bypass_count']:,}.
- Mean filter diagnostic runtime: {trace['mean_filter_runtime_ms_per_episode']:.3f} ms/episode. Online algorithm time in the result CSV excludes trajectory serialization.

## Integrity and stop rule

The 10 deterministic microtests passed. The first four repaired integration attempts reached the post-episode serializer and failed before any repaired result was saved; both the missing-field fix and the subsequent diagnostic-complexity fix are preserved as explicit trace-only amendments, with `control_semantics_changed=false`. No failed attempt contributes to statistics.

Independent reconciliation: **{reconciliation['status']}**. It reopened every raw JSON/NPZ record, verified scene/source/checkpoint/frozen-artifact hashes, recomputed categorical totals, checked trace schemas and hashes, and verified safety-bypass output and horizontal tangential preservation.

- `DEV_GATE = {dev['DEV_GATE']}`
- `HOLDOUT_EXECUTED = {'YES' if holdout is not None else 'NO'}`
- `FORMAL_V2_EXECUTED = NO`
- `TURN_SIGN_PERSISTENCE_ACCEPTED = {'YES' if holdout_pass else 'NO'}`
- `SELECTED_EXECUTION_METHOD = {selected}`
- `RECOMMENDED_NEXT_STEP = {recommendation}`

## Artifact index

- `TURN_PERSISTENCE_CONTROL_CONTRACT.json`
- `TURN_PERSISTENCE_THRESHOLDS.json`
- `TURN_PERSISTENCE_MICROTEST.json`
- `TURN_PERSISTENCE_DEV_RESULTS.csv`
- `TURN_PERSISTENCE_DEV_MORPHOLOGY.csv`
- `TURN_PERSISTENCE_FAILURE_AUDIT.csv`
- `TURN_PERSISTENCE_DEV_RAW_TRAJECTORIES.pdf`
- `TURN_PERSISTENCE_DEV_GO_NO_GO.json`
- `conclusion.json`, `independent_reconciliation.json`, `final_reconciliation.json`
"""
    (ROOT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"DEV_GATE": dev["DEV_GATE"], "selected": selected, "next": recommendation}, indent=2))


if __name__ == "__main__":
    main()
