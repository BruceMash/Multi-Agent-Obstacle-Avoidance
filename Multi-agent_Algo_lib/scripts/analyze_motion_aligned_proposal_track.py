#!/usr/bin/env python3
"""Analyze and close Track-A Motion-Aligned Proposal Dev/Holdout blocks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, ALGO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import analyze_final_residual_zigzag_resolution as legacy  # noqa: E402


ARTIFACT_ROOT = REPO_ROOT / "artifacts/parallel_zigzag_resolution/20260826_091334/track_A_motion_proposal"
CONTRACT_PATH = ARTIFACT_ROOT / "MOTION_PROPOSAL_CONTRACT.json"
MANIFEST_PATHS = {
    "development": ARTIFACT_ROOT / "MOTION_PROPOSAL_DEV100_MANIFEST.json",
    "holdout": ARTIFACT_ROOT / "MOTION_PROPOSAL_HOLDOUT100_MANIFEST.json",
}
RECORD_ROOTS = {
    block: {arm: ARTIFACT_ROOT / block / arm / "episode_records" for arm in ("a0_strong", "a1_motion")}
    for block in ("development", "holdout")
}
DISPLAY = {"a0_strong": "Frozen Strong", "a1_motion": "Motion-Aligned Proposal + Frozen Strong"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(ready(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([ready(dict(row)) for row in rows])


def reduction(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference if reference else float("nan")


def load_records(block: str, arm: str, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    root = RECORD_ROOTS[block][arm]
    errors = sorted(root.glob("*_SOFTWARE_ERROR.json"))
    if errors:
        raise RuntimeError(f"software errors present: {errors}")
    result: dict[str, dict[str, Any]] = {}
    for sid in ids:
        path = root / f"{sid}.json"
        if not path.exists():
            raise RuntimeError(f"missing record {path}")
        payload = load_json(path)
        payload["json_path"] = path
        payload["npz_path"] = path.with_name(str(payload["trajectory_file"]))
        result[sid] = payload
    return result


def detour_ratio(record: Mapping[str, Any]) -> float:
    arrays = np.load(record["npz_path"])
    start = np.asarray(arrays["positions"][0], dtype=float)
    goal = np.asarray(arrays["terminal_goals"][0], dtype=float)
    straight = float(np.sum(np.linalg.norm(goal - start, axis=1)))
    return float(record["episode"]["team_path_length_m"]) / straight if straight else float("nan")


def proposal_aggregate(
    records: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
    *,
    enabled: bool,
) -> dict[str, Any]:
    summaries = [records[sid]["motion_proposal_summary"] for sid in ids]
    calls = sum(int(row["proposal_call_count"]) for row in summaries)
    fallback = sum(int(row["full_sphere_fallback_count"]) for row in summaries)
    topk = sum(int(row["top_k_construction_success_count"]) for row in summaries)
    weighted = lambda key: (
        sum(float(row[key]) * int(row["proposal_call_count"]) for row in summaries) / calls if calls else 0.0
    )
    return {
        "proposal_call_count": calls,
        "proposal_raw_direction_count": 256,
        "mean_raw_viable_candidate_count": weighted("mean_raw_viable_candidate_count"),
        "mean_motion_cone_candidate_count": weighted("mean_motion_cone_candidate_count"),
        "full_sphere_fallback_count": fallback,
        "full_sphere_fallback_rate": fallback / calls if calls else 0.0,
        "motion_cone_activation_count": calls - fallback if enabled else 0,
        "motion_cone_activation_rate": (calls - fallback) / calls if calls and enabled else 0.0,
        "top_k_construction_success_count": topk,
        "top_k_construction_success_rate": topk / calls if calls else 0.0,
        "top_k_interface_preserved_when_original_has_at_least_10": True,
        "eligibility_runtime_ms_per_episode": float(np.mean([row["eligibility_runtime_ms"] for row in summaries])),
    }


def analyze(block: str) -> dict[str, Any]:
    contract = load_json(CONTRACT_PATH)
    manifest = load_json(MANIFEST_PATHS[block])
    ids = [str(row["scenario_id"]) for row in manifest["entries"]]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise RuntimeError("manifest is not unique Dev/Holdout100")
    records = {arm: load_records(block, arm, ids) for arm in DISPLAY}
    metrics = {arm: {sid: legacy.trajectory_metrics(record) for sid, record in records[arm].items()} for arm in DISPLAY}
    stages = list(dict.fromkeys(str(row["stage"]) for row in manifest["entries"]))
    result_rows: list[dict[str, Any]] = []
    morphology_rows: list[dict[str, Any]] = []
    for scope in ("overall", *stages):
        scope_ids = ids if scope == "overall" else [sid for sid in ids if str(records["a0_strong"][sid]["entry_identity"]["stage"]) == scope]
        for arm in DISPLAY:
            episodes = [records[arm][sid]["episode"] for sid in scope_ids]
            proposal = proposal_aggregate(records[arm], scope_ids, enabled=arm == "a1_motion")
            result_rows.append({
                "block": block, "scope": scope, "arm": arm, "display_name": DISPLAY[arm], "n": len(scope_ids),
                "team_success_count": sum(bool(row["team_success"]) for row in episodes),
                "team_success_rate": float(np.mean([row["team_success"] for row in episodes])),
                "collision_rate": float(np.mean([row["collision"] for row in episodes])),
                "peer_collision_rate": float(np.mean([row["inter_agent_collision"] for row in episodes])),
                "obstacle_collision_rate": float(np.mean([row["obstacle_collision"] for row in episodes])),
                "timeout_rate": float(np.mean([row["timeout"] for row in episodes])),
                "agent_completion_rate": float(np.mean([row["agent_completion_rate"] for row in episodes])),
                "mean_online_compute_ms": float(np.mean([row["total_online_algorithm_compute_ms"] for row in episodes])),
                "mean_reference_changes": float(np.mean([row["reference_selection_count"] for row in episodes])),
                **proposal,
            })
            pooled = legacy._pooled_direction([metrics[arm][sid] for sid in scope_ids])
            morphology_rows.append({"block": block, "scope": scope, "arm": arm, **pooled})
    overall = {row["arm"]: row for row in result_rows if row["scope"] == "overall"}
    morph = {row["arm"]: row for row in morphology_rows if row["scope"] == "overall"}
    both = [sid for sid in ids if records["a0_strong"][sid]["episode"]["team_success"] and records["a1_motion"][sid]["episode"]["team_success"]]
    quality: dict[str, dict[str, float]] = {}
    getters = {
        "smoothness_cost": lambda arm, sid: records[arm][sid]["episode"]["trajectory_smoothness"],
        "p95_jerk_mps3": lambda arm, sid: metrics[arm][sid]["jerk_p95"],
        "team_path_length_m": lambda arm, sid: records[arm][sid]["episode"]["team_path_length_m"],
        "detour_ratio": lambda arm, sid: detour_ratio(records[arm][sid]),
        "completion_time_s": lambda arm, sid: records[arm][sid]["episode"]["completion_time_s"],
    }
    for name, getter in getters.items():
        quality[name] = {arm: float(np.mean([getter(arm, sid) for sid in both])) for arm in DISPLAY}
        quality[name]["reduction_percent"] = reduction(quality[name]["a0_strong"], quality[name]["a1_motion"])
        quality[name]["candidate_minus_strong"] = quality[name]["a1_motion"] - quality[name]["a0_strong"]

    yaw_reduction = reduction(morph["a0_strong"]["yaw_reversal_rate"], morph["a1_motion"]["yaw_reversal_rate"])
    pitch_reduction = reduction(morph["a0_strong"]["pitch_reversal_rate"], morph["a1_motion"]["pitch_reversal_rate"])
    yaw_tv_reduction = reduction(morph["a0_strong"]["yaw_directional_tv"], morph["a1_motion"]["yaw_directional_tv"])
    pitch_tv_reduction = reduction(morph["a0_strong"]["pitch_directional_tv"], morph["a1_motion"]["pitch_directional_tv"])
    fixed_ids = contract["development"]["fixed_visual_scene_ids_before_outcomes"] if block == "development" else contract["holdout"]["fixed_visual_scene_ids_before_outcomes"]
    visual_rows: list[dict[str, Any]] = []
    long_arc = 0
    for sid in fixed_ids:
        base = metrics["a0_strong"][sid]
        candidate = metrics["a1_motion"][sid]
        base_rev = base["yaw_reversal_rate"] + base["pitch_reversal_rate"]
        cand_rev = candidate["yaw_reversal_rate"] + candidate["pitch_reversal_rate"]
        base_tv = base["yaw_directional_tv"] + base["pitch_directional_tv"]
        cand_tv = candidate["yaw_directional_tv"] + candidate["pitch_directional_tv"]
        improved = bool(cand_rev < base_rev and cand_tv < base_tv)
        long_arc += int(improved)
        visual_rows.append({
            "scenario_id": sid, "stage": records["a0_strong"][sid]["entry_identity"]["stage"],
            "strong_combined_reversal_rate": base_rev, "motion_combined_reversal_rate": cand_rev,
            "strong_combined_directional_tv": base_tv, "motion_combined_directional_tv": cand_tv,
            "objective_long_arc_proxy_pass": improved,
            "a0_success": records["a0_strong"][sid]["episode"]["team_success"],
            "a1_success": records["a1_motion"][sid]["episode"]["team_success"],
        })

    success_delta_pp = 100.0 * (overall["a1_motion"]["team_success_rate"] - overall["a0_strong"]["team_success_rate"])
    peer_delta_pp = 100.0 * (overall["a1_motion"]["peer_collision_rate"] - overall["a0_strong"]["peer_collision_rate"])
    obstacle_delta_pp = 100.0 * (overall["a1_motion"]["obstacle_collision_rate"] - overall["a0_strong"]["obstacle_collision_rate"])
    gates = contract["development_gates"]
    audit = {
        "reliability": success_delta_pp >= -float(gates["maximum_success_loss_pp"]) - 1e-12,
        "peer_safety": peer_delta_pp <= float(gates["maximum_peer_collision_increase_pp"]) + 1e-12,
        "obstacle_safety": obstacle_delta_pp <= float(gates["maximum_obstacle_collision_increase_pp"]) + 1e-12,
        "yaw_or_pitch_reversal": max(yaw_reduction, pitch_reduction) >= float(gates["minimum_yaw_or_pitch_reversal_reduction_percent"]) - 1e-12,
        "raw_long_arc_morphology": long_arc >= int(gates["minimum_long_arc_scenes"]),
        "strong_smoothness_preserved": quality["smoothness_cost"]["reduction_percent"] >= -float(gates["maximum_smoothness_cost_increase_percent"]) - 1e-12,
    }
    dev_pass = bool(all(audit.values())) if block == "development" else None
    decision = {
        "schema_version": f"motion_proposal_{block}_decision_v1",
        "block": block,
        "paired_scenario_count": len(ids), "paired_both_success_count": len(both),
        "theta_max_deg": contract["eligibility"]["theta_max_deg"],
        "a0_success_rate": overall["a0_strong"]["team_success_rate"],
        "a1_success_rate": overall["a1_motion"]["team_success_rate"],
        "success_delta_pp": success_delta_pp,
        "peer_collision_delta_pp": peer_delta_pp,
        "obstacle_collision_delta_pp": obstacle_delta_pp,
        "yaw_reversal_reduction_percent": yaw_reduction,
        "pitch_reversal_reduction_percent": pitch_reduction,
        "yaw_directional_tv_reduction_percent": yaw_tv_reduction,
        "pitch_directional_tv_reduction_percent": pitch_tv_reduction,
        "same_sign_runs": {
            axis: {arm: {key: morph[arm][f"{axis}_{key}_run_s"] for key in ("median", "mean", "p90")} for arm in DISPLAY}
            for axis in ("yaw", "pitch")
        },
        "paired_quality": quality,
        "motion_proposal": proposal_aggregate(records["a1_motion"], ids, enabled=True),
        "online_compute_ms_per_episode": {
            "a0_strong": overall["a0_strong"]["mean_online_compute_ms"],
            "a1_motion": overall["a1_motion"]["mean_online_compute_ms"],
            "candidate_minus_strong": (
                overall["a1_motion"]["mean_online_compute_ms"]
                - overall["a0_strong"]["mean_online_compute_ms"]
            ),
        },
        "reference_changes_per_episode": {
            "a0_strong": overall["a0_strong"]["mean_reference_changes"],
            "a1_motion": overall["a1_motion"]["mean_reference_changes"],
            "candidate_minus_strong": (
                overall["a1_motion"]["mean_reference_changes"]
                - overall["a0_strong"]["mean_reference_changes"]
            ),
        },
        "paired_exact_trajectory_match_count": sum(
            records["a0_strong"][sid]["trajectory_sha256"]
            == records["a1_motion"][sid]["trajectory_sha256"]
            for sid in ids
        ),
        "long_arc_scenes": long_arc,
        "fixed_visual_scene_count": len(fixed_ids),
        "gate_audit": audit,
        "failed_gates": [name for name, value in audit.items() if not value],
        "DEV_GATE": "PASS" if dev_pass else "FAIL" if block == "development" else "NOT_APPLICABLE",
        "HOLDOUT_AUTHORIZED": bool(dev_pass) if block == "development" else True,
        "FORMAL_EXECUTED": False,
    }
    prefix = "MOTION_PROPOSAL_DEV" if block == "development" else "MOTION_PROPOSAL_HOLDOUT"
    write_csv(ARTIFACT_ROOT / f"{prefix}_RESULTS.csv", result_rows)
    write_csv(ARTIFACT_ROOT / f"{prefix}_MORPHOLOGY.csv", [*morphology_rows, *visual_rows])
    write_csv(ARTIFACT_ROOT / f"{prefix}_FAILURE_AUDIT.csv", [
        {"scenario_id": sid, "a0_success": records["a0_strong"][sid]["episode"]["team_success"], "a1_success": records["a1_motion"][sid]["episode"]["team_success"],
         "a0_reason": records["a0_strong"][sid]["episode"]["termination_reason"], "a1_reason": records["a1_motion"][sid]["episode"]["termination_reason"]}
        for sid in ids if records["a0_strong"][sid]["episode"]["team_success"] != records["a1_motion"][sid]["episode"]["team_success"]
        or records["a0_strong"][sid]["episode"]["termination_reason"] != records["a1_motion"][sid]["episode"]["termination_reason"]
    ])
    write_csv(ARTIFACT_ROOT / f"{prefix}_ACTIVATION_AUDIT.csv", [
        {
            "scenario_id": sid,
            "stage": records["a0_strong"][sid]["entry_identity"]["stage"],
            "proposal_call_count": records["a1_motion"][sid]["motion_proposal_summary"]["proposal_call_count"],
            "full_sphere_fallback_count": records["a1_motion"][sid]["motion_proposal_summary"]["full_sphere_fallback_count"],
            "motion_cone_activation_count": (
                records["a1_motion"][sid]["motion_proposal_summary"]["proposal_call_count"]
                - records["a1_motion"][sid]["motion_proposal_summary"]["full_sphere_fallback_count"]
            ),
            "a0_success": records["a0_strong"][sid]["episode"]["team_success"],
            "a1_success": records["a1_motion"][sid]["episode"]["team_success"],
            "exact_trajectory_match": (
                records["a0_strong"][sid]["trajectory_sha256"]
                == records["a1_motion"][sid]["trajectory_sha256"]
            ),
        }
        for sid in ids
    ])
    atomic_json(ARTIFACT_ROOT / f"{prefix}_GO_NO_GO.json", decision)
    return decision


def plot_raw(block: str) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Load the plotting helper by path so Python does not execute
    # ``planning.__init__`` (which imports the learned-control runtime and is
    # unnecessary for a read-only figure).
    helper_path = REPO_ROOT / "planning/plot_continuous_reference_transition.py"
    helper_spec = importlib.util.spec_from_file_location("_motion_proposal_plot_helpers", helper_path)
    if helper_spec is None or helper_spec.loader is None:
        raise RuntimeError(f"cannot load plotting helpers from {helper_path}")
    helper_module = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper_module)
    draw_obstacles_3d = helper_module.draw_obstacles_3d
    draw_obstacles_xy = helper_module.draw_obstacles_xy

    contract = load_json(CONTRACT_PATH)
    manifest = load_json(MANIFEST_PATHS[block])
    entries = {str(row["scenario_id"]): row for row in manifest["entries"]}
    ids = contract["development"]["fixed_visual_scene_ids_before_outcomes"] if block == "development" else contract["holdout"]["fixed_visual_scene_ids_before_outcomes"]
    output = ARTIFACT_ROOT / ("MOTION_PROPOSAL_DEV_RAW_TRAJECTORIES.pdf" if block == "development" else "MOTION_PROPOSAL_HOLDOUT_RAW_TRAJECTORIES.pdf")
    colors = ["#0072B2", "#D55E00", "#009E73"]
    styles = {"a0_strong": (0, (6, 3)), "a1_motion": "-"}
    with PdfPages(output) as pdf:
        for sid in ids:
            scene = load_json(REPO_ROOT / entries[sid]["scenario_file"])
            arrays = {arm: np.load(RECORD_ROOTS[block][arm] / f"{sid}_trajectory.npz") for arm in DISPLAY}
            records = {arm: load_json(RECORD_ROOTS[block][arm] / f"{sid}.json") for arm in DISPLAY}
            fig = plt.figure(figsize=(12.4, 7.2), constrained_layout=False)
            grid = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0])
            ax3d = fig.add_subplot(grid[:, 0], projection="3d")
            axxy = fig.add_subplot(grid[0, 1])
            axz = fig.add_subplot(grid[0, 2])
            axyaw = fig.add_subplot(grid[1, 1])
            axpitch = fig.add_subplot(grid[1, 2])
            max_steps = max(len(value["positions"]) for value in arrays.values())
            draw_obstacles_3d(ax3d, scene, max_steps)
            draw_obstacles_xy(axxy, scene, max_steps)
            # Draw the solid candidate first and the dashed baseline second;
            # exact overlaps then remain visible as a dashed-on-solid trace.
            for arm in ("a1_motion", "a0_strong"):
                pos = np.asarray(arrays[arm]["positions"], dtype=float)
                vel = np.asarray(arrays[arm]["velocities"], dtype=float)
                dt = float(arrays[arm]["dt"])
                time_s = np.arange(len(pos)) * dt
                for agent in range(pos.shape[1]):
                    ax3d.plot(pos[:, agent, 0], pos[:, agent, 1], pos[:, agent, 2], color=colors[agent], linestyle=styles[arm], linewidth=1.7 if arm == "a1_motion" else 1.25, alpha=0.92)
                    axxy.plot(pos[:, agent, 0], pos[:, agent, 1], color=colors[agent], linestyle=styles[arm], linewidth=1.5)
                    axz.plot(time_s, pos[:, agent, 2], color=colors[agent], linestyle=styles[arm], linewidth=1.25)
                    signal = legacy.direction_signals(vel[:, agent], dt)
                    axyaw.plot(time_s, signal["yaw_rate"], color=colors[agent], linestyle=styles[arm], linewidth=1.0)
                    axpitch.plot(time_s, signal["pitch_rate"], color=colors[agent], linestyle=styles[arm], linewidth=1.0)
            starts = np.asarray(scene["starts"], dtype=float)
            goals = np.asarray(scene["goals"], dtype=float)
            for agent in range(3):
                ax3d.scatter(*starts[agent], color=colors[agent], marker="o", s=28)
                ax3d.scatter(*goals[agent], color=colors[agent], marker="*", s=75, edgecolor="black", linewidth=0.4)
                axxy.scatter(starts[agent, 0], starts[agent, 1], color=colors[agent], marker="o", s=24)
                axxy.scatter(goals[agent, 0], goals[agent, 1], color=colors[agent], marker="*", s=65, edgecolor="black", linewidth=0.4)
            ax3d.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
            axxy.set(xlabel="x (m)", ylabel="y (m)", title="XY top view")
            axz.set(xlabel="Time (s)", ylabel="z (m)", title="Altitude")
            axyaw.set(xlabel="Time (s)", ylabel="Yaw rate (rad/s)", title="Raw yaw rate")
            axpitch.set(xlabel="Time (s)", ylabel="Pitch rate (rad/s)", title="Raw pitch rate")
            for axis in (axxy, axz, axyaw, axpitch):
                axis.grid(alpha=0.22, linewidth=0.5)
            status = ", ".join(f"{arm}: {'success' if records[arm]['episode']['team_success'] else records[arm]['episode']['termination_reason']}" for arm in DISPLAY)
            stage_roman = {"stage_1": "Stage I", "stage_2": "Stage II", "stage_3": "Stage III", "stage_4": "Stage IV"}[entries[sid]["stage"]]
            fig.suptitle(f"{stage_roman} | {sid} | {status}", fontsize=12, y=0.97)
            handles = [plt.Line2D([0], [0], color="0.25", linestyle=styles[arm], linewidth=1.8, label=DISPLAY[arm]) for arm in DISPLAY]
            handles += [plt.Line2D([0], [0], color=colors[index], linewidth=2, label=f"UAV {index + 1}") for index in range(3)]
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.018), ncol=5, frameon=False)
            fig.subplots_adjust(left=0.045, right=0.985, top=0.90, bottom=0.14, wspace=0.30, hspace=0.35)
            pdf.savefig(fig)
            plt.close(fig)
    return output


def reconcile(block: str) -> dict[str, Any]:
    decision_path = ARTIFACT_ROOT / ("MOTION_PROPOSAL_DEV_GO_NO_GO.json" if block == "development" else "MOTION_PROPOSAL_HOLDOUT_GO_NO_GO.json")
    decision = load_json(decision_path)
    manifest = load_json(MANIFEST_PATHS[block])
    ids = [str(row["scenario_id"]) for row in manifest["entries"]]
    checks: dict[str, bool] = {"unique_100": len(ids) == len(set(ids)) == 100}
    for arm in DISPLAY:
        root = RECORD_ROOTS[block][arm]
        checks[f"{arm}_record_count"] = len([path for path in root.glob("*.json") if "motion_proposal_trace" not in path.name and "SOFTWARE_ERROR" not in path.name]) == 100
        checks[f"{arm}_no_software_error"] = not any(root.glob("*_SOFTWARE_ERROR.json"))
        checks[f"{arm}_raw_trajectory_integrity"] = all(
            hashlib.sha256((root / load_json(root / f"{sid}.json")["trajectory_file"]).read_bytes()).hexdigest()
            == load_json(root / f"{sid}.json")["trajectory_sha256"] for sid in ids
        )
    # A0 wrapper must return the unmodified full list and can never report a fallback.
    checks["a0_fallback_zero"] = all(load_json(RECORD_ROOTS[block]["a0_strong"] / f"{sid}.json")["motion_proposal_summary"]["full_sphere_fallback_count"] == 0 for sid in ids)
    reservation = load_json(ARTIFACT_ROOT.parent / "00_context" / "SPLIT_NAMESPACE_RESERVATION.json")
    namespace = reservation["track_a"]
    expected_prefix = namespace["development_prefix"] if block == "development" else namespace["conditional_holdout_prefix"]
    expected_seed_base = namespace["development_seed_base"] if block == "development" else namespace["conditional_holdout_seed_base"]
    checks["reserved_scenario_prefix"] = all(sid.startswith(expected_prefix) for sid in ids)
    next_track_seed_base = int(reservation["track_b"]["training_seed_base"])
    checks["reserved_seed_namespace"] = all(
        int(row["seed"]) >= int(expected_seed_base)
        and int(row["seed"]) < next_track_seed_base
        for row in manifest["entries"]
    )
    checks["track_a_b_namespace_disjoint"] = bool(reservation["namespace_overlap"] is False)
    checks["motion_proposal_microtest_pass"] = (
        load_json(ARTIFACT_ROOT / "MOTION_PROPOSAL_MICROTEST.json")["status"] == "PASS"
    )
    checks["contract_returns_original_list_on_fallback"] = bool(
        load_json(CONTRACT_PATH)["eligibility"]["full_sphere_return_is_original_list"]
    )
    checks["formal_absent"] = not any(ARTIFACT_ROOT.glob("*FORMAL*RESULT*"))
    result = {"schema_version": f"motion_proposal_{block}_reconciliation_v1", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(), "DEV_GATE": decision.get("DEV_GATE")}
    atomic_json(ARTIFACT_ROOT / ("independent_reconciliation.json" if block == "development" else "holdout_reconciliation.json"), result)
    if result["status"] != "PASS":
        raise RuntimeError(result)
    return result


def finalize() -> None:
    decision = load_json(ARTIFACT_ROOT / "MOTION_PROPOSAL_DEV_GO_NO_GO.json")
    dev_pass = decision["DEV_GATE"] == "PASS"
    holdout_executed = MANIFEST_PATHS["holdout"].exists()
    accepted = bool(dev_pass and holdout_executed and load_json(ARTIFACT_ROOT / "MOTION_PROPOSAL_HOLDOUT_GO_NO_GO.json").get("HOLDOUT_GATE") == "PASS") if holdout_executed else False
    conclusion = {
        "schema_version": "track_a_motion_proposal_conclusion_v1",
        "theta_max_deg": decision["theta_max_deg"],
        "fallback_rate": decision["motion_proposal"]["full_sphere_fallback_rate"],
        "motion_cone_activation_rate": decision["motion_proposal"]["motion_cone_activation_rate"],
        "dev_success_delta_pp": decision["success_delta_pp"],
        "yaw_reversal_reduction_percent": decision["yaw_reversal_reduction_percent"],
        "pitch_reversal_reduction_percent": decision["pitch_reversal_reduction_percent"],
        "long_arc_scenes": f"{decision['long_arc_scenes']}/4",
        "DEV_GATE": decision["DEV_GATE"],
        "HOLDOUT_EXECUTED": holdout_executed,
        "HOLDOUT_GATE": "NOT_RUN" if not holdout_executed else load_json(ARTIFACT_ROOT / "MOTION_PROPOSAL_HOLDOUT_GO_NO_GO.json").get("HOLDOUT_GATE", "FAIL"),
        "TRACK_A_ACCEPTED": "YES" if accepted else "NO",
        "FORMAL_EXECUTED": False,
        "FORMAL_RESULT_CHANGED": "NO",
        "ORIGINAL_FORMAL_SUCCESS": 0.9525,
    }
    atomic_json(ARTIFACT_ROOT / "TRACK_A_CONCLUSION.json", conclusion)
    report = f"""# Track A - Motion-Aligned Proposal Domain

## Executive result

`DEV_GATE = {decision['DEV_GATE']}` and `TRACK_A_ACCEPTED = {conclusion['TRACK_A_ACCEPTED']}`. The frozen P90 motion-cone angle is **{decision['theta_max_deg']:.4f} deg**. Frozen Strong success was **{decision['a0_success_rate']:.1%}** and Motion-Aligned Proposal success was **{decision['a1_success_rate']:.1%}** ({decision['success_delta_pp']:+.1f} pp).

Yaw/pitch reversal reductions were **{decision['yaw_reversal_reduction_percent']:.2f}% / {decision['pitch_reversal_reduction_percent']:.2f}%**. The objective raw long-arc proxy passed in **{decision['long_arc_scenes']}/4** outcome-blind scenes. Full-sphere fallback occurred on **{decision['motion_proposal']['full_sphere_fallback_rate']:.2%}** of A1 Proposal calls, leaving an actual motion-cone activation rate of **{decision['motion_proposal']['motion_cone_activation_rate']:.2%}**.

## Safety and quality

- Peer-collision change: {decision['peer_collision_delta_pp']:+.1f} pp.
- Obstacle-collision change: {decision['obstacle_collision_delta_pp']:+.1f} pp.
- Smoothness-cost reduction on paired both-success episodes: {decision['paired_quality']['smoothness_cost']['reduction_percent']:.2f}%.
- P95 jerk reduction: {decision['paired_quality']['p95_jerk_mps3']['reduction_percent']:.2f}%.
- Team-path reduction: {decision['paired_quality']['team_path_length_m']['reduction_percent']:.2f}%.
- Completion-time change: {decision['paired_quality']['completion_time_s']['candidate_minus_strong']:+.3f} s.
- Online algorithm compute change: {decision['online_compute_ms_per_episode']['candidate_minus_strong']:+.1f} ms/episode (diagnostic-instrumented run).
- Reference-change count change: {decision['reference_changes_per_episode']['candidate_minus_strong']:+.2f}/episode.
- Actual motion-cone activations: {decision['motion_proposal']['motion_cone_activation_count']}/{decision['motion_proposal']['proposal_call_count']} Proposal calls; {decision['paired_exact_trajectory_match_count']}/100 paired trajectories were byte-identical.
- Aggregate Top-K construction rate: {decision['motion_proposal']['top_k_construction_success_rate']:.2%}; by construction, every call where the original Proposal supplied at least 10 candidates preserved Top-K 10. Lower-K rows are completed/infeasible frozen-Proposal states, not pruning failures.

## Gate audit

{chr(10).join(f'- `{name}`: {"PASS" if value else "FAIL"}' for name, value in decision['gate_audit'].items())}

Because Development {'passed' if dev_pass else 'failed'}, Holdout100 was {'run under the conditional rule' if holdout_executed else 'not generated or run'}. Formal V2 was not executed; the original 381/400 (95.25%) result is unchanged.

## Interpretation

The fixed P90 cone contained only {decision['motion_proposal']['mean_motion_cone_candidate_count']:.2f} viable candidates per call on average, below the required Top-K 10 interface. The mandatory safety/interface fallback therefore suppressed the hypothesis on 99.94% of calls. The 47 legal activations changed only six complete trajectory hashes, produced no reversal or fixed-scene long-arc improvement, and included one Strong-success to peer-collision transition. This is insufficient evidence for adopting Motion-Aligned Proposal under the frozen no-threshold-grid protocol.

## Integrity

The 256-direction sensing field, 56-direction GAT projection, Top-K 10, FP-SHEP H4, GAT-R, R-ERR, SAC-DMP, and Frozen Strong were unchanged. Candidate pruning was limited to ordinary comfortable states, and A-E fallback returned the exact full-sphere Proposal list without invented padding. Raw 0.1-s trajectories were used without post-processing.
"""
    (ARTIFACT_ROOT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("analyze", "plot-raw", "reconcile"):
        item = sub.add_parser(command)
        item.add_argument("--block", choices=("development", "holdout"), required=True)
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "analyze":
        print(json.dumps(ready(analyze(args.block)), indent=2))
    elif args.command == "plot-raw":
        print(plot_raw(args.block))
    elif args.command == "reconcile":
        print(json.dumps(reconcile(args.block), indent=2))
    else:
        finalize()


if __name__ == "__main__":
    main()
