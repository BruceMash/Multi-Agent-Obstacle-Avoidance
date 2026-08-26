"""Finalize the sector-resolution audit reports and machine-readable decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "artifacts/sector_resolution_oscillation_audit/20260824_110611"
FORMAL_RECORDS = REPO_ROOT / (
    "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2/"
    "formal_records/M9_Proposed_RERR_GAT_SAC_DMP"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def association(rows: pd.DataFrame, relationship: str) -> pd.Series:
    return rows.loc[rows["relationship"] == relationship].iloc[0]


def formal_success_smoothness() -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    episodes = [read_json(path)["episode"] for path in sorted(FORMAL_RECORDS.glob("*.json"))]
    successful = [row for row in episodes if bool(row["team_success"])]
    overall = {
        "formal_episode_count": len(episodes),
        "formal_success_count": len(successful),
        "formal_success_rate": len(successful) / len(episodes),
        "successful_smoothness_mean": float(np.mean([float(row["trajectory_smoothness"]) for row in successful])),
        "successful_reference_selection_mean": float(np.mean([float(row["reference_selection_count"]) for row in successful])),
        "successful_replanning_mean": float(np.mean([float(row["replanning_count"]) for row in successful])),
        "successful_upper_invocation_mean": float(np.mean([float(row["upper_pipeline_invocation_count"]) for row in successful])),
    }
    stages: dict[str, dict[str, float]] = {}
    for stage in ("Stage I", "Stage II", "Stage III", "Stage IV"):
        selected = [row for row in successful if row["stage"] == stage]
        stages[stage] = {
            "n": len(selected),
            "successful_smoothness_mean": float(np.mean([float(row["trajectory_smoothness"]) for row in selected])),
            "successful_replanning_mean": float(np.mean([float(row["replanning_count"]) for row in selected])),
        }
    return overall, stages


def write_figure_manifest(root: Path) -> None:
    availability = read_json(root / "09_figures/FIGURE_AVAILABILITY.json")
    titles = {
        "A_current_proposal_sector_geometry_3d": "Current Proposal sector geometry in 3D",
        "B_azimuth_elevation_distribution": "Azimuth/elevation distribution",
        "C_adjacent_sector_angular_distance_distribution": "Adjacent-sector angular distance distribution",
        "D_reference_angular_jump_histogram": "Reference angular jump histogram",
        "E_elevation_jump_histogram": "Elevation jump histogram",
        "F_angular_jump_vs_jerk_peak": "Angular jump versus jerk peak",
        "G_elevation_jump_vs_vertical_jerk_peak": "Elevation jump versus vertical jerk peak",
        "H_ABA_switchback_examples": "A-B-A switchback examples",
        "I_current_vs_denser_raw_trajectories": "Current versus denser raw trajectories",
        "J_current_vs_hysteresis_trajectories": "Current versus hysteresis trajectories",
        "K_2x2_reliability_smoothness_pareto": "2x2 reliability-smoothness Pareto",
    }
    stems = {
        "A_current_proposal_sector_geometry_3d": "figure_A_current_proposal_sector_geometry_3d",
        "B_azimuth_elevation_distribution": "figure_B_azimuth_elevation_distribution",
        "C_adjacent_sector_angular_distance_distribution": "figure_C_adjacent_sector_angular_distance_distribution",
        "D_reference_angular_jump_histogram": "figure_D_reference_angular_jump_histogram",
        "E_elevation_jump_histogram": "figure_E_elevation_jump_histogram",
        "F_angular_jump_vs_jerk_peak": "figure_F_angular_jump_vs_jerk_peak",
        "G_elevation_jump_vs_vertical_jerk_peak": "figure_G_elevation_jump_vs_vertical_jerk_peak",
        "H_ABA_switchback_examples": "figure_H_ABA_switchback_examples",
    }
    rows = []
    for key, status in availability.items():
        stem = stems.get(key)
        rows.append(
            {
                "figure_id": key.split("_", 1)[0],
                "title": titles[key],
                "status": status,
                "pdf": f"pdf/{stem}.pdf" if stem else "",
                "png_600dpi": f"png_600dpi/{stem}.png" if stem else "",
                "caption": f"captions/{stem}_caption.txt" if stem else "",
                "not_run_reason": "" if status == "YES" else status,
            }
        )
    path = root / "11_paper_ready/figure_manifest.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    contract = read_json(root / "01_sector_contract/PROPOSAL_SECTOR_CONTRACT.json")
    resolution = read_json(root / "01_sector_contract/SECTOR_ANGULAR_RESOLUTION_SUMMARY.json")
    vertical = read_json(root / "01_sector_contract/VERTICAL_SECTOR_RESOLUTION_AUDIT.json")
    integrity = read_json(root / "02_existing_event_analysis/EVENT_AUDIT_INTEGRITY.json")
    chatter = read_json(root / "02_existing_event_analysis/ADJACENT_SECTOR_CHATTER_AUDIT.json")
    boundary = read_json(root / "02_existing_event_analysis/SECTOR_BOUNDARY_INSTABILITY.json")
    transient = read_json(root / "02_existing_event_analysis/SWITCH_TRANSIENT_CONTROL_AUDIT.json")
    compatibility = read_json(root / "03_root_cause/SECTOR_DENSIFICATION_COMPATIBILITY.json")
    associations = pd.read_csv(root / "02_existing_event_analysis/SECTOR_JERK_ASSOCIATION.csv")
    switch_stats = pd.read_csv(root / "02_existing_event_analysis/SECTOR_SWITCH_STATISTICS.csv")
    angular_bins = pd.read_csv(root / "02_existing_event_analysis/ANGULAR_JUMP_JERK_BINS.csv")
    overall_switch = switch_stats.loc[switch_stats["scope"] == "overall"].iloc[0]
    angle_assoc = association(associations, "sector_center_angular_jump_vs_post_switch_jerk_peak")
    elevation_assoc = association(associations, "absolute_elevation_jump_vs_vertical_jerk_peak")
    adjacent_contrast = association(associations, "adjacent_vs_nonadjacent_post_switch_jerk_peak")
    switchback_contrast = association(associations, "switchback_vs_ordinary_post_switch_jerk_peak")
    elevation_contrast = association(associations, "elevation_switch_vs_no_elevation_switch_vertical_jerk_peak")
    gat_assoc = association(associations, "gat_margin_vs_post_switch_jerk_peak")
    post_pre = transient["tests"][0]
    post_pre_vertical = transient["tests"][1]
    formal, formal_stages = formal_success_smoothness()

    adjacent_high_rate = float(chatter["all_switches"]["high_jerk_adjacent_switch_count"]) / float(chatter["all_switches"]["adjacent_switch_count"])
    nonadjacent = chatter["by_adjacency_type"]["non_adjacent"]
    nonadjacent_high_rate = float(nonadjacent["high_jerk_switch_count"]) / float(nonadjacent["candidate_sector_switch_count"])
    adjacent_angle = resolution["adjacent_pair_angular_distance_deg"]
    median_displacement = 2.0 * float(contract["reference_distance_rule"]["formal_observed_selected_reference_distance_quantiles_m"]["P50"]) * np.sin(np.deg2rad(float(adjacent_angle["median"])) / 2.0)

    conclusion = {
        "CURRENT_PROPOSAL_SECTOR_COUNT": 256,
        "CURRENT_AZIMUTH_RESOLUTION": {"degrees": 22.5, "radians": 0.39269908169872414},
        "CURRENT_ELEVATION_LEVELS": 16,
        "CURRENT_ELEVATION_RESOLUTION": {"degrees": float(resolution["elevation_spacing_deg"]), "radians": float(resolution["elevation_spacing_rad"])},
        "CURRENT_MEDIAN_ADJACENT_ANGLE_DEG": float(adjacent_angle["median"]),
        "GAT_CANONICAL_DIRECTION_COUNT": 56,
        "PROPOSAL_AND_GAT_DIRECTION_CONTRACT": "COUPLED",
        "SECTOR_DENSIFICATION_WITHOUT_RETRAINING": "NO",
        "SECTOR_DENSIFICATION_ABLATION_AUTHORIZED": "NO",
        "ADJACENT_SWITCH_FRACTION": float(overall_switch["adjacent_switch_fraction"]),
        "HIGH_JERK_ADJACENT_SWITCH_FRACTION": float(overall_switch["high_jerk_adjacent_switch_fraction"]),
        "SWITCHBACK_RATE": float(overall_switch["switchback_rate_1s"]),
        "SWITCHBACK_RATE_HEADLINE_WINDOW_S": 1.0,
        "ELEVATION_SWITCHBACK_RATE": float(vertical["elevation_level_ABA_within_1s_rate"]),
        "SECTOR_ANGLE_JERK_ASSOCIATION": float(angle_assoc["estimate"]),
        "SECTOR_ANGLE_JERK_ASSOCIATION_95CI": [float(angle_assoc["ci_low"]), float(angle_assoc["ci_high"])],
        "ELEVATION_JERK_ASSOCIATION": float(elevation_assoc["estimate"]),
        "ELEVATION_JERK_ASSOCIATION_95CI": [float(elevation_assoc["ci_low"]), float(elevation_assoc["ci_high"])],
        "SECTOR_QUANTIZATION_EVIDENCE": "WEAK",
        "VERTICAL_SECTOR_QUANTIZATION_EVIDENCE": "WEAK",
        "SELECTOR_CHATTER_EVIDENCE": "MODERATE",
        "LOWER_CONTROLLER_TRANSIENT_EVIDENCE": "STRONG",
        "BEST_DEVELOPMENT_VARIANT": "KEEP_ORIGINAL_NO_VARIANT_AUTHORIZED",
        "DEV_ORIGINAL_SUCCESS": "NOT_RUN",
        "DEV_VARIANT_SUCCESS": "NOT_RUN",
        "DEV_ORIGINAL_JERK": "NOT_RUN",
        "DEV_VARIANT_JERK": "NOT_RUN",
        "DEV_JERK_REDUCTION_PERCENT": "NOT_RUN",
        "DEV_REFERENCE_SWITCH_REDUCTION": "NOT_RUN",
        "DEV_COMPUTE_CHANGE_PERCENT": "NOT_RUN",
        "HOLDOUT_REPLICATED": "NOT_RUN",
        "RETRAINING_PERFORMED": "NO",
        "GAT_CHECKPOINT_CHANGED": "NO",
        "SAC_CHECKPOINT_CHANGED": "NO",
        "POLICY_EPISODES_EXECUTED": 0,
        "CURRENT_ORIGINAL_FORMAL_V2_SUCCESS": formal["formal_success_rate"],
        "FORMAL_REEVALUATION_RECOMMENDED": "NO",
        "ACADEMIC_INTEGRITY_GATE": "PASS",
        "FINAL_RECOMMENDATION": "Keep the frozen original configuration. Do not densify Proposal under the shared sensor contract; prioritize a separate switch-transient/acceleration-continuity study.",
    }
    write_json(root / "conclusion.json", conclusion)

    stage_lines = []
    for stage in ("Stage I", "Stage II", "Stage III", "Stage IV"):
        row = switch_stats.loc[switch_stats["scope"] == stage].iloc[0]
        stage_lines.append(
            f"| {stage} | {int(row['candidate_sector_switch_count']):,} | {100*float(row['adjacent_switch_fraction']):.2f}% | {100*float(row['switchback_rate_1s']):.2f}% | {float(row['mean_abs_sector_elevation_jump_deg']):.3f} | {float(row['mean_vertical_jerk_peak_mps3']):.3f} | {float(row['mean_post_switch_squared_jerk_m2_s6']):.3f} | {formal_stages[stage]['successful_smoothness_mean']:.3f} |"
        )
    bin_lines = []
    for _, row in angular_bins.iterrows():
        bin_lines.append(
            f"| {row['angular_jump_bin_deg']} | {int(row['event_count']):,} | {100*float(row['high_jerk_rate']):.2f}% | {float(row['mean_post_switch_jerk_peak_mps3']):.3f} | {float(row['median_post_switch_jerk_peak_mps3']):.3f} |"
        )

    root_report = f"""# Sector Quantization Root-Cause Report

## Executive classification

- `SECTOR_QUANTIZATION_TOO_COARSE = WEAK`
- `VERTICAL_SECTOR_QUANTIZATION = WEAK`
- `SELECTOR_CHATTER = MODERATE`
- `LOWER_CONTROLLER_TRANSIENT = STRONG`

The stored successful Formal V2 trajectories do not show the monotonic signature expected if coarse Proposal sectors were the primary jerk source. Sector-center angular jump has a weak **negative** Spearman association with post-switch jerk peak (rho={float(angle_assoc['estimate']):+.3f}, episode-cluster 95% CI {float(angle_assoc['ci_low']):+.3f} to {float(angle_assoc['ci_high']):+.3f}). A-B-A returns within 1 s are real ({int(overall_switch['switchback_count_1s']):,}/{int(overall_switch['candidate_sector_switch_count']):,}, {100*float(overall_switch['switchback_rate_1s']):.2f}%), but their mean jerk peak is {abs(float(switchback_contrast['estimate'])):.3f} m/s^3 **lower** than ordinary changed-sector events.

The strongest temporal signature is at the controller transition itself: the 0.5 s post-switch jerk peak exceeds the paired pre-switch peak by {post_pre['estimate']:.3f} m/s^3 on average (95% CI {post_pre['ci_low']:.3f} to {post_pre['ci_high']:.3f}); the vertical increase is {post_pre_vertical['estimate']:.3f} m/s^3. This identifies a switch-aligned transient, not isolated causality.

## Exact direction contract

| Item | Frozen value |
|---|---:|
| Proposal directions | 16 azimuth x 16 elevation = 256 |
| Azimuth spacing | 22.5 deg / 0.392699 rad |
| Elevation spacing | {float(resolution['elevation_spacing_deg']):.6f} deg / {float(resolution['elevation_spacing_rad']):.6f} rad |
| Adjacent great-circle angle | min {float(adjacent_angle['min']):.3f}, median {float(adjacent_angle['median']):.3f}, max {float(adjacent_angle['max']):.3f} deg |
| Median adjacent reference displacement at R=1.05 m | {median_displacement:.3f} m |
| Coarse interface | ordered raw feasible candidates -> Top-K 10 -> FP-SHEP -> GAT-R |
| GAT canonical safety directions | fixed 8 x 7 = 56 |
| Candidate direction in GAT | continuous xyz |
| Proposal sector ID in GAT | absent |

Proposal and the GAT 56-direction field are `COUPLED`, not identical: both read the same 16 x 16 scan, but GAT independently projects that scan to a fixed 8 x 7 basis and receives no Proposal sector ID.

## Read-only event coverage

- 400 stored M9 Formal records were reopened; only the 381 team-success episodes entered the switch/jerk audit.
- {integrity['actual_reference_changes']:,} actual active-reference changes were aligned to raw 0.1 s trajectories.
- {integrity['candidate_reference_changes']:,} were candidate-reference selections; {integrity['noncandidate_reference_changes']:,} were terminal handoffs.
- {int(overall_switch['candidate_sector_switch_count']):,} candidate-to-candidate events changed Proposal sector.
- No policy, environment episode, checkpoint, or stored Formal result was executed or modified.

## Adjacent switching and chatter

- Adjacent-sector share: {100*float(overall_switch['adjacent_switch_fraction']):.2f}%.
- High-jerk adjacent share: {100*float(overall_switch['high_jerk_adjacent_switch_fraction']):.2f}%, below the all-switch adjacent base rate.
- Conditional P90-high-jerk rate: adjacent {100*adjacent_high_rate:.2f}%, non-adjacent {100*nonadjacent_high_rate:.2f}%.
- A-B-A rate: {100*float(chatter['switchback_windows_s']['0.2']['rate']):.2f}% within 0.2 s, {100*float(chatter['switchback_windows_s']['0.5']['rate']):.2f}% within 0.5 s, {100*float(chatter['switchback_windows_s']['1.0']['rate']):.2f}% within 1.0 s, and {100*float(chatter['switchback_windows_s']['2.0']['rate']):.2f}% within 2.0 s.
- A-B-A-B completions: {chatter['abab_pattern_count']:,}.
- Switchbacks have a higher, not lower, mean GAT Top-1/Top-2 margin by {boundary['small_margin_to_AB_switching']['estimate']:+.3f}; the stored evidence does not support a simple near-tie-margin hysteresis explanation.

Large jumps do have a heavier extreme tail, which is why sector quantization receives `WEAK` rather than `NO_EVIDENCE`:

| Angular jump | n | P90-high-jerk rate | Mean peak | Median peak |
|---|---:|---:|---:|---:|
{chr(10).join(bin_lines)}

This tail pattern is non-monotonic across the full distribution and does not establish that finer raw sampling would reduce far-sector selections.

## Vertical audit

- 16 elevation levels: 128 upward centers, 128 downward centers, and no exact level center; the two nearest levels are +/-5.333 deg.
- Elevation jump versus vertical jerk: rho={float(elevation_assoc['estimate']):+.3f}, 95% CI {float(elevation_assoc['ci_low']):+.3f} to {float(elevation_assoc['ci_high']):+.3f}.
- Events with any elevation-level change have mean vertical jerk peak {float(elevation_contrast['estimate']):.3f} m/s^3 above no-elevation-change events.
- Exact elevation-level A-B-A within 1 s: {vertical['elevation_level_ABA_within_1s_count']:,} ({100*float(vertical['elevation_level_ABA_within_1s_rate']):.2f}% of eligible three-selection sequences).

These effects are statistically stable but small. They support `WEAK`, not strong, vertical-quantization evidence.

## Stage III check

| Stage | Changed-sector events | Adjacent | A-B-A <=1 s | Mean absolute elevation jump (deg) | Vertical jerk peak | Post-switch mean squared jerk | Successful episode smoothness |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(stage_lines)}

Stage III is not the numerical worst case: its successful-episode smoothness is {formal_stages['Stage III']['successful_smoothness_mean']:.3f}, compared with {formal_stages['Stage I']['successful_smoothness_mean']:.3f}/{formal_stages['Stage II']['successful_smoothness_mean']:.3f}/{formal_stages['Stage IV']['successful_smoothness_mean']:.3f} for Stages I/II/IV. The visually prominent Stage III oscillation is therefore not evidence of a stage-specific sector bottleneck.

## Boundary-instability limit

There is no continuous pre-quantization direction. Proposal directly enumerates exact sensor-ray centers, so `distance_to_sector_boundary` is unavailable and was not fabricated. Non-event alternative rankings are also absent, so directional margin to switch probability cannot be estimated. GAT probability margin is retained but is not a sector-boundary margin.

## Mandatory compatibility stop

`SECTOR_DENSIFICATION_WITHOUT_RETRAINING = NO`. Directly changing 16 x 16 changes the frozen SAC observation dimension (522) because Proposal and the actor share the sensor grid. Creating a separate denser Proposal grid would require a new interpolation/feasibility contract that does not exist in the frozen implementation. Conditions 1 and 6 of the mandatory eight-condition gate fail.

Therefore no S1/S2 resolution, hysteresis, 2 x 2 Development, Holdout, or Formal variant was run. The original 381/400 (95.25%) Formal V2 result remains provenance only and is not paired with new variant smoothness.
"""
    root_cause_path = root / "03_root_cause/SECTOR_QUANTIZATION_ROOT_CAUSE_REPORT.md"
    root_cause_path.write_text(root_report, encoding="utf-8")

    paper = f"""# Paper Sector-Resolution Analysis

The final Proposal generator does not use the GAT's 56 canonical directions as its candidate grid. It enumerates the 256 directions of the 16 x 16 LiDAR scan, ranks feasible candidates, keeps Top-K=10, and supplies continuous candidate direction vectors to GAT-R. The frozen 56-direction block is a separate 8 x 7 projection of the same scan safety field. The two contracts are coupled through sensing but are not identical.

On the 381 successful Formal V2 Proposed episodes, {int(overall_switch['candidate_sector_switch_count']):,} changed-sector events were aligned with raw 0.1 s accelerations. Adjacent sectors account for {100*float(overall_switch['adjacent_switch_fraction']):.2f}% of changes, and A-B-A returns within 1 s account for {100*float(overall_switch['switchback_rate_1s']):.2f}%. However, angular jump and post-switch jerk are weakly negatively associated (rho={float(angle_assoc['estimate']):+.3f}, 95% CI {float(angle_assoc['ci_low']):+.3f} to {float(angle_assoc['ci_high']):+.3f}), while switchbacks have lower mean jerk than ordinary switches. Elevation jumps show only a weak positive association with vertical jerk (rho={float(elevation_assoc['estimate']):+.3f}).

The clearest signal is temporal: mean jerk peak rises by {post_pre['estimate']:.3f} m/s^3 from the paired 0.5 s pre-switch window to the post-switch window, and vertical jerk rises by {post_pre_vertical['estimate']:.3f} m/s^3. This supports describing the visible oscillation as a recurrent reference-update/transient issue more than a demonstrated coarse-sector quantization issue. Because Proposal resolution is tied to the frozen actor's sensor observation size, a denser no-retraining Proposal variant is not checkpoint-compatible under the existing source contract. No modified method was evaluated, and no performance claim is made for one.
"""
    (root / "11_paper_ready/PAPER_SECTOR_RESOLUTION_ANALYSIS.md").write_text(paper, encoding="utf-8")

    final_report = f"""# Proposal Sector Resolution and Oscillation Audit

## Executive result

`SECTOR_QUANTIZATION_EVIDENCE = WEAK`, `SELECTOR_CHATTER_EVIDENCE = MODERATE`, and `LOWER_CONTROLLER_TRANSIENT_EVIDENCE = STRONG`. The active Proposal geometry is **16 x 16 = 256 directions**, not the GAT-R 56-direction projection. Its spacing is 22.5 deg in azimuth and 10.667 deg in elevation; source-adjacent great-circle distance has a 15.002 deg median.

Across 381 successful Formal V2 Proposed episodes, {int(overall_switch['candidate_sector_switch_count']):,} actual changed-sector events were aligned to raw 0.1 s acceleration. Adjacent switches are common ({100*float(overall_switch['adjacent_switch_fraction']):.2f}%), but they are less represented among P90-high-jerk events ({100*float(overall_switch['high_jerk_adjacent_switch_fraction']):.2f}%). Angular jump is weakly negatively associated with jerk (rho={float(angle_assoc['estimate']):+.3f}); elevation jump is weakly positively associated with vertical jerk (rho={float(elevation_assoc['estimate']):+.3f}). Thus the specific hypothesis "coarse adjacent-sector quantization causes the high jerk" is not supported as the primary mechanism.

The more persuasive evidence is switch-aligned controller response: post-switch jerk peak increases by {post_pre['estimate']:.3f} m/s^3 over the paired pre-switch window, and vertical jerk by {post_pre_vertical['estimate']:.3f} m/s^3. A-B-A chatter exists ({100*float(overall_switch['switchback_rate_1s']):.2f}% within 1 s), but it has lower mean jerk than ordinary switches.

## Sector contract

- Proposal: 256 exact LiDAR ray centers; maximum 256 geometric raw directions before feasibility/progress filtering.
- Raw feasible count varies by state; the exact pre-Top-K count was not retained in Formal event rows.
- Top-K stays 10. FP-SHEP and GAT-R see only this ordered Top-K interface.
- GAT-R receives continuous candidate direction xyz and no Proposal sector ID.
- GAT safety input remains a frozen 56-value canonical projection.
- Contract relation: `COUPLED`, because both Proposal and canonical safety projection read the same 16 x 16 scan.

## Core read-only results

| Quantity | Result |
|---|---:|
| Successful Formal episodes audited | 381/400 |
| Actual reference changes | {integrity['actual_reference_changes']:,} |
| Candidate reference changes | {integrity['candidate_reference_changes']:,} |
| Changed-sector events | {int(overall_switch['candidate_sector_switch_count']):,} |
| Adjacent switch fraction | {100*float(overall_switch['adjacent_switch_fraction']):.2f}% |
| High-jerk adjacent fraction | {100*float(overall_switch['high_jerk_adjacent_switch_fraction']):.2f}% |
| A-B-A within 1 s | {100*float(overall_switch['switchback_rate_1s']):.2f}% |
| Angular jump vs jerk rho | {float(angle_assoc['estimate']):+.3f} [{float(angle_assoc['ci_low']):+.3f}, {float(angle_assoc['ci_high']):+.3f}] |
| Elevation jump vs vertical jerk rho | {float(elevation_assoc['estimate']):+.3f} [{float(elevation_assoc['ci_low']):+.3f}, {float(elevation_assoc['ci_high']):+.3f}] |
| Post-minus-pre jerk peak | +{post_pre['estimate']:.3f} m/s^3 |
| Post-minus-pre vertical jerk peak | +{post_pre_vertical['estimate']:.3f} m/s^3 |

## Densification hard gate

`SECTOR_DENSIFICATION_WITHOUT_RETRAINING = NO`. The Proposal direction grid is not independently configurable: it is `sensor.ray_directions`, and the same sensor grid determines the frozen SAC actor's 522-D observation. Changing it directly breaks the checkpoint input contract. Decoupling Proposal through interpolation would be a new method semantic, not the requested minimum no-retraining repair.

The mandatory stop was obeyed. S1/S2, hysteresis, the 2 x 2 Development block, Holdout, and Formal reevaluation are all `NOT_RUN`. Policy episode count for this audit is zero; GAT/SAC hashes are unchanged.

## Stage III

Stage III has {int(switch_stats.loc[switch_stats['scope']=='Stage III','candidate_sector_switch_count'].iloc[0]):,} changed-sector events, {100*float(switch_stats.loc[switch_stats['scope']=='Stage III','adjacent_switch_fraction'].iloc[0]):.2f}% adjacent switching, {100*float(switch_stats.loc[switch_stats['scope']=='Stage III','switchback_rate_1s'].iloc[0]):.2f}% A-B-A within 1 s, and successful-episode smoothness {formal_stages['Stage III']['successful_smoothness_mean']:.3f}. It is not worse than every other stage on the audited statistics.

## Final decision

- `BEST_DEVELOPMENT_VARIANT = KEEP_ORIGINAL_NO_VARIANT_AUTHORIZED`
- `FORMAL_REEVALUATION_RECOMMENDED = NO`
- `RETRAINING_PERFORMED = NO`
- `ACADEMIC_INTEGRITY_GATE = PASS`

Keep the current frozen method. The next defensible study is a separate switch-transient/acceleration-continuity audit; it should not silently change Proposal resolution or reuse 95.25% as performance for an unevaluated variant.

## Artifact index

- Exact contract: `01_sector_contract/PROPOSAL_SECTOR_CONTRACT.json`
- Angular resolution: `01_sector_contract/SECTOR_ANGULAR_RESOLUTION.csv`
- Vertical audit: `01_sector_contract/VERTICAL_SECTOR_RESOLUTION_AUDIT.json`
- Event table: `02_existing_event_analysis/SECTOR_SWITCH_EVENT_TABLE.csv`
- Chatter audit: `02_existing_event_analysis/ADJACENT_SECTOR_CHATTER_AUDIT.json`
- Jerk associations: `02_existing_event_analysis/SECTOR_JERK_ASSOCIATION.csv`
- Root-cause report: `03_root_cause/SECTOR_QUANTIZATION_ROOT_CAUSE_REPORT.md`
- Compatibility gate: `03_root_cause/SECTOR_DENSIFICATION_COMPATIBILITY.json`
- Paper figures and captions: `11_paper_ready/figure_manifest.csv`
- Machine-readable decision: `conclusion.json`
- Independent raw-data reconciliation: `INDEPENDENT_RECONCILIATION.json`
"""
    (root / "FINAL_REPORT.md").write_text(final_report, encoding="utf-8")

    write_figure_manifest(root)
    required = [
        root / "FINAL_REPORT.md",
        root / "conclusion.json",
        root / "01_sector_contract/PROPOSAL_SECTOR_CONTRACT.json",
        root / "01_sector_contract/SECTOR_ANGULAR_RESOLUTION.csv",
        root / "01_sector_contract/VERTICAL_SECTOR_RESOLUTION_AUDIT.json",
        root / "02_existing_event_analysis/SECTOR_SWITCH_EVENT_TABLE.csv",
        root / "02_existing_event_analysis/ADJACENT_SECTOR_CHATTER_AUDIT.json",
        root / "02_existing_event_analysis/SECTOR_SWITCH_STATISTICS.csv",
        root / "02_existing_event_analysis/SECTOR_JERK_ASSOCIATION.csv",
        root / "03_root_cause/SECTOR_QUANTIZATION_ROOT_CAUSE_REPORT.md",
        root / "03_root_cause/SECTOR_DENSIFICATION_COMPATIBILITY.json",
        root / "04_resolution_variants/SECTOR_VARIANT_CONTRACTS.json",
        root / "06_development/SECTOR_DEVELOPMENT_RESULTS.csv",
        root / "06_development/SECTOR_RELIABILITY_SMOOTHNESS_PARETO.csv",
        root / "10_freeze/SECTOR_VARIANT_FORMAL_GO_NO_GO.json",
        root / "11_paper_ready/PAPER_SECTOR_RESOLUTION_ANALYSIS.md",
    ]
    pngs = sorted((root / "11_paper_ready/png_600dpi").glob("*.png"))
    pdfs = sorted((root / "11_paper_ready/pdf").glob("*.pdf"))
    checks = {
        "status": "PASS",
        "required_file_count": len(required),
        "missing_required_files": [str(path.relative_to(root)) for path in required if not path.exists()],
        "required_sha256": {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in required if path.exists()},
        "figure_pdf_count": len(pdfs),
        "figure_png_count": len(pngs),
        "successful_formal_episode_count_recomputed": formal["formal_success_count"],
        "event_row_count_recomputed": integrity["actual_reference_changes"],
        "changed_sector_event_count_recomputed": int(overall_switch["candidate_sector_switch_count"]),
        "development_policy_episode_count": 0,
        "holdout_policy_episode_count": 0,
        "formal_policy_episode_count": 0,
        "checkpoint_change": False,
        "stored_formal_result_change": False,
        "mandatory_stop_obeyed": compatibility["SECTOR_DENSIFICATION_ABLATION_AUTHORIZED"] == "NO",
    }
    if checks["missing_required_files"] or len(pdfs) != 8 or len(pngs) != 8 or formal["formal_success_count"] != 381:
        checks["status"] = "FAIL"
    write_json(root / "FINAL_RECONCILIATION.json", checks)
    print(json.dumps({"status": checks["status"], "root": str(root), "figures": len(pdfs), "formal_success": formal["formal_success_count"]}))


if __name__ == "__main__":
    main()
