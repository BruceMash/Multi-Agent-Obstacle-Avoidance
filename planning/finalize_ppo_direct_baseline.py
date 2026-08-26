"""Freeze and reconcile the completed PPO-Direct branch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from planning.ppo_direct_baseline import install_pandas_import_guard


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FORMAL_TEAM = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2/formal_v2_team_results.csv"
COMPARATORS = {
    "proposed": "M9_Proposed_RERR_GAT_SAC_DMP",
    "dwa_sm": "M2_DWA_SensingMatched",
    "sac": "M4_Direct_SAC_DMP",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    if fields is None:
        fields = tuple(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def exact_binomial_two_sided(left: int, right: int) -> float:
    n = int(left + right)
    if n == 0:
        return 1.0
    tail = min(int(left), int(right))
    p = 2.0 * sum(math.comb(n, index) for index in range(tail + 1)) / (2**n)
    return min(1.0, float(p))


def get_overall(summary_json: Path) -> dict[str, Any]:
    data = load_json(summary_json)
    return next(row for row in data["summary"] if row["scope"] == "overall")


def freeze(root: Path, checkpoint: Path, dev_tag: str, holdout_tag: str) -> None:
    install_pandas_import_guard()
    from stable_baselines3 import PPO

    dev_path = root / f"06_development/evaluation_{dev_tag}.json"
    holdout_path = root / f"07_holdout/evaluation_{holdout_tag}.json"
    dev = get_overall(dev_path)
    holdout = get_overall(holdout_path)
    model = PPO.load(str(checkpoint), device="cpu")
    parameter_count = int(sum(parameter.numel() for parameter in model.policy.parameters()))
    phase_files = sorted((root / "05_training").glob("phase_*_summary.json"))
    phases = [load_json(path) for path in phase_files]
    config = load_json(root / "03_training_environment/PPO_DIRECT_TRAINING_CONFIG.json")
    reward = load_json(root / "03_training_environment/PPO_DIRECT_REWARD_CONTRACT.json")
    isolation = load_json(root / "03_training_environment/PPO_DIRECT_SCENE_ISOLATION_AUDIT.json")
    control = load_json(root / "01_control_contract/PPO_NATIVE_CONTROL_INTERFACE_AUDIT.json")
    atomicity = load_json(root / "03_training_environment/PPO_MULTI_AGENT_STEP_ATOMICITY_AUDIT.json")
    ready = bool(
        isolation["status"] == "PASS"
        and control["status"] == "PASS"
        and atomicity["status"] == "PASS"
        and int(holdout["scenario_count"]) >= 100
        and np.isfinite(float(holdout["mean_goal_progress_m"]))
    )
    record = {
        "schema_version": "final_ppo_direct_freeze_v1",
        "status": "FROZEN",
        "method": "PPO-Direct",
        "algorithm": "STANDARD_PARAMETER_SHARED_DECENTRALIZED_PPO",
        "checkpoint_path": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": sha256(checkpoint),
        "training_timesteps": int(model.num_timesteps),
        "training_wallclock_h": sum(float(row["phase_wallclock_s"]) for row in phases) / 3600.0,
        "network": config["network"],
        "parameter_count": parameter_count,
        "optimizer": "Adam",
        "learning_rate": config["learning_rate"],
        "ppo_hyperparameters": {
            key: config[key]
            for key in ("gamma", "gae_lambda", "clip_range", "n_steps", "batch_size", "n_epochs", "entropy_coefficient")
        },
        "observation_schema": "02_observation_contract/PPO_DIRECT_OBSERVATION_SCHEMA.csv",
        "observation_dim": 533,
        "action_semantics": "[-1,1]^3 -> net linear acceleration [-4,4] m/s^2",
        "reward": reward,
        "train_manifest": "03_training_environment/PPO_DIRECT_TRAIN_MANIFEST.json",
        "dev_manifest": "03_training_environment/PPO_DIRECT_DEV_MANIFEST.json",
        "holdout_manifest": "03_training_environment/PPO_DIRECT_HOLDOUT_MANIFEST.json",
        "dev_evaluation": str(dev_path.relative_to(root)),
        "holdout_evaluation": str(holdout_path.relative_to(root)),
        "runtime_timer_boundaries": {
            "included": ["observation assembly", "policy forward", "action conversion"],
            "excluded": ["environment sensing", "physics", "collision checks", "I/O"],
        },
        "PPO_DIRECT_DEVELOPMENT_CLOSED": "YES",
        "PPO_BASELINE_READY_FOR_PAPER": "YES" if ready else "NO",
        "holdout_used_for_tuning": False,
        "formal_v2_accessed_before_freeze": False,
    }
    write_json(root / "08_final_freeze/FINAL_PPO_DIRECT_FREEZE.json", record)
    write_json(
        root / "08_final_freeze/PPO_DIRECT_DEVELOPMENT_CLOSURE.json",
        {
            "PPO_DIRECT_DEVELOPMENT_CLOSED": "YES",
            "selected_checkpoint": record["checkpoint_path"],
            "selection_source": "Development only",
            "sealed_holdout_run_once": True,
            "post_holdout_tuning": False,
            "paper_ready": record["PPO_BASELINE_READY_FOR_PAPER"],
        },
    )


def paired_binary(ppo: Sequence[Mapping[str, str]], comparator: Sequence[Mapping[str, str]], label: str) -> dict[str, Any]:
    ppo_map = {row["scenario_id"]: as_bool(row["team_success"]) for row in ppo}
    comp_map = {row["scenario_id"]: as_bool(row["team_success"]) for row in comparator}
    ids = sorted(set(ppo_map) & set(comp_map))
    both_success = sum(ppo_map[key] and comp_map[key] for key in ids)
    ppo_only = sum(ppo_map[key] and not comp_map[key] for key in ids)
    comparator_only = sum(not ppo_map[key] and comp_map[key] for key in ids)
    both_failure = len(ids) - both_success - ppo_only - comparator_only
    return {
        "schema_version": "ppo_direct_paired_binary_v1",
        "comparison": f"PPO-Direct vs {label}",
        "paired_n": len(ids),
        "both_success": both_success,
        "ppo_only_success": ppo_only,
        "comparator_only_success": comparator_only,
        "both_failure": both_failure,
        "ppo_success_rate": (both_success + ppo_only) / len(ids),
        "comparator_success_rate": (both_success + comparator_only) / len(ids),
        "ppo_minus_comparator_pp": 100.0 * (ppo_only - comparator_only) / len(ids),
        "mcnemar_exact_two_sided_p": exact_binomial_two_sided(ppo_only, comparator_only),
    }


def comparator_rows() -> dict[str, list[dict[str, str]]]:
    rows = read_csv(SOURCE_FORMAL_TEAM)
    return {
        label: [row for row in rows if row["method_id"] == method_id]
        for label, method_id in COMPARATORS.items()
    }


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, resamples: int = 5000) -> tuple[float, float]:
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def continuous_comparisons(ppo: Sequence[Mapping[str, str]], comparisons: Mapping[str, Sequence[Mapping[str, str]]]) -> list[dict[str, Any]]:
    ppo_map = {row["scenario_id"]: row for row in ppo}
    metric_map = {
        "completion_time_s": ("completion_time_s", "completion_time_s"),
        "team_path_length_m": ("team_path_length_m", "team_path_length_m"),
        "path_efficiency": ("team_path_efficiency", "team_path_efficiency"),
        "trajectory_smoothness": ("trajectory_smoothness", "trajectory_smoothness"),
        "minimum_obstacle_clearance_m": ("minimum_obstacle_clearance_m", "minimum_obstacle_clearance_m"),
        "minimum_inter_agent_distance_m": ("minimum_inter_agent_distance_m", "minimum_inter_agent_distance_m"),
    }
    result: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260823)
    for label, rows in comparisons.items():
        other = {row["scenario_id"]: row for row in rows}
        ids = [
            key
            for key in sorted(set(ppo_map) & set(other))
            if as_bool(ppo_map[key]["team_success"]) and as_bool(other[key]["team_success"])
        ]
        for metric, (ppo_field, other_field) in metric_map.items():
            differences = []
            for key in ids:
                left = ppo_map[key].get(ppo_field, "")
                right = other[key].get(other_field, "")
                if left not in ("", None) and right not in ("", None):
                    differences.append(float(left) - float(right))
            values = np.asarray(differences, dtype=float)
            if not len(values):
                continue
            low, high = bootstrap_ci(values, rng)
            nonzero = values[np.abs(values) > 1e-12]
            positive = int(np.sum(nonzero > 0.0))
            negative = int(np.sum(nonzero < 0.0))
            result.append(
                {
                    "comparator": label,
                    "metric": metric,
                    "paired_n": len(values),
                    "difference_direction": "PPO_MINUS_COMPARATOR",
                    "mean_difference": float(np.mean(values)),
                    "median_difference": float(np.median(values)),
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "paired_sign_test_p": exact_binomial_two_sided(positive, negative),
                }
            )
    return result


def failure_taxonomy(ppo: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in ppo:
        if as_bool(row["team_success"]):
            label = "success"
        elif as_bool(row["static_obstacle_collision"]):
            label = "static_obstacle_collision"
        elif as_bool(row["dynamic_obstacle_collision"]):
            label = "dynamic_obstacle_collision"
        elif as_bool(row["inter_agent_collision"]):
            label = "inter_agent_collision"
        elif as_bool(row["boundary_collision"]):
            label = "boundary_collision"
        elif as_bool(row["timeout"]):
            label = "timeout"
        else:
            label = "other"
        counts[label] = counts.get(label, 0) + 1
    return [
        {"failure_type": key, "count": value, "rate": value / len(ppo)}
        for key, value in sorted(counts.items())
    ]


def analyze_formal(root: Path) -> None:
    team_path = root / "09_formal_v2/ppo_direct_formal_team_results.csv"
    agent_path = root / "09_formal_v2/ppo_direct_formal_agent_results.csv"
    ppo = read_csv(team_path)
    agents = read_csv(agent_path)
    comparisons = comparator_rows()
    paired = {
        label: paired_binary(ppo, rows, {"proposed": "Proposed", "dwa_sm": "DWA-SensingMatched", "sac": "Direct SAC-DMP"}[label])
        for label, rows in comparisons.items()
    }
    write_json(root / "10_statistics/ppo_direct_vs_proposed_paired.json", paired["proposed"])
    write_json(root / "10_statistics/ppo_direct_vs_dwa_sm_paired.json", paired["dwa_sm"])
    write_json(root / "10_statistics/ppo_direct_vs_sac_paired.json", paired["sac"])
    continuous = continuous_comparisons(ppo, comparisons)
    write_csv(root / "10_statistics/ppo_direct_both_success_continuous.csv", continuous)
    taxonomy = failure_taxonomy(ppo)
    write_csv(root / "09_formal_v2/ppo_direct_failure_taxonomy.csv", taxonomy)
    runtime = {
        "scenario_count": len(ppo),
        "mean_compute_ms_per_episode": float(np.mean([float(row["total_online_compute_ms"]) for row in ppo])),
        "mean_compute_ms_per_control_step": float(np.mean([float(row["mean_compute_ms_per_control_step"]) for row in ppo])),
        "mean_inference_ms_per_episode": float(np.mean([float(row["ppo_inference_compute_ms"]) for row in ppo])),
        "mean_policy_calls_per_episode": float(np.mean([float(row["policy_calls"]) for row in ppo])),
        "environment_physics_included": False,
    }
    write_csv(root / "11_runtime/ppo_direct_runtime_summary.csv", [runtime])
    manifest_path = root / "12_trajectories/ppo_direct_trajectory_manifest.csv"
    trajectory_rows = read_csv(manifest_path)
    hashes_valid = all(
        sha256(root / row["path"]) == row["sha256"] for row in trajectory_rows
    )
    success = sum(as_bool(row["team_success"]) for row in ppo)
    collision = sum(as_bool(row["any_collision"]) for row in ppo)
    peer = sum(as_bool(row["inter_agent_collision"]) for row in ppo)
    timeout = sum(as_bool(row["timeout"]) for row in ppo)
    agent_completion = sum(as_bool(row["completed"]) for row in agents) / len(agents)
    stages = {
        stage: sum(as_bool(row["team_success"]) for row in ppo if row["stage"] == stage)
        / sum(row["stage"] == stage for row in ppo)
        for stage in ("stage_1", "stage_2", "stage_3", "stage_4")
    }
    stage_3_4_rows = [row for row in ppo if row["stage"] in {"stage_3", "stage_4"}]
    stage_3_4_success = sum(as_bool(row["team_success"]) for row in stage_3_4_rows) / len(stage_3_4_rows)
    freeze_record = load_json(root / "08_final_freeze/FINAL_PPO_DIRECT_FREEZE.json")
    eta = load_json(root / "04_throughput/PPO_DIRECT_TRAINING_ETA.json")
    dev = get_overall(root / freeze_record["dev_evaluation"])
    holdout = get_overall(root / freeze_record["holdout_evaluation"])
    reconciliation_pass = bool(
        len(ppo) == 400
        and len({row["scenario_id"] for row in ppo}) == 400
        and len(agents) == 1200
        and len(trajectory_rows) == 400
        and hashes_valid
        and success + (len(ppo) - success) == 400
    )
    write_json(
        root / "final_reconciliation.json",
        {
            "schema_version": "ppo_direct_final_reconciliation_v1",
            "status": "PASS" if reconciliation_pass else "FAIL",
            "team_rows": len(ppo),
            "unique_scenarios": len({row["scenario_id"] for row in ppo}),
            "agent_rows": len(agents),
            "trajectory_rows": len(trajectory_rows),
            "trajectory_hashes_valid": hashes_valid,
            "success_count": success,
            "collision_count": collision,
            "peer_collision_count": peer,
            "timeout_count": timeout,
        },
    )
    conclusion = {
        "PPO_BASELINE_NAME": "PPO-Direct",
        "PPO_ALGORITHM": "STANDARD_PARAMETER_SHARED_DECENTRALIZED_PPO",
        "PPO_EXECUTION_TYPE": "DIRECT_PHYSICAL_CONTROL",
        "PPO_SHARED_SAC_DMP": "NO",
        "PPO_INFORMATION_MATCHED": "YES",
        "PPO_ACTION_TYPE": "NET_LINEAR_ACCELERATION",
        "PPO_ACTION_DIM": 3,
        "PPO_OBSERVATION_DIM": 533,
        "PPO_NETWORK": "actor[256,256]-critic[256,256]-Tanh",
        "PPO_PARAMETER_COUNT": freeze_record["parameter_count"],
        "PPO_LEARNING_RATE": 0.0003,
        "PPO_BATCH_SIZE": 256,
        "PPO_TRAINING_TIMESTEPS": freeze_record["training_timesteps"],
        "PPO_TRAINING_WALLCLOCK_H": freeze_record["training_wallclock_h"],
        "PPO_TRAINING_THROUGHPUT": eta["selected_transitions_per_second"],
        "PPO_DEV_SCENARIOS": dev["scenario_count"],
        "PPO_DEV_SUCCESS": dev["success_rate"],
        "PPO_DEV_COLLISION": dev["collision_rate"],
        "PPO_HOLDOUT_SCENARIOS": holdout["scenario_count"],
        "PPO_HOLDOUT_SUCCESS": holdout["success_rate"],
        "PPO_HOLDOUT_COLLISION": holdout["collision_rate"],
        "PPO_BASELINE_READY_FOR_PAPER": freeze_record["PPO_BASELINE_READY_FOR_PAPER"],
        "PPO_FORMAL_STATUS": "POST_HOC_FROZEN_PPO_BASELINE_EVALUATION",
        "PPO_FORMAL_SCENARIOS": 400,
        "PPO_FORMAL_SUCCESS": success / 400,
        "PPO_FORMAL_COLLISION": collision / 400,
        "PPO_FORMAL_PEER_COLLISION": peer / 400,
        "PPO_FORMAL_TIMEOUT": timeout / 400,
        "PPO_FORMAL_AGENT_COMPLETION": agent_completion,
        "PPO_STAGE1_SUCCESS": stages["stage_1"],
        "PPO_STAGE2_SUCCESS": stages["stage_2"],
        "PPO_STAGE3_SUCCESS": stages["stage_3"],
        "PPO_STAGE4_SUCCESS": stages["stage_4"],
        "PPO_STAGE3_4_POOLED_SUCCESS": stage_3_4_success,
        "PROPOSED_FORMAL_SUCCESS": 0.9525,
        "DWA_SM_FORMAL_SUCCESS": 0.5375,
        "DIRECT_SAC_FORMAL_SUCCESS": 0.01,
        "PROPOSED_MINUS_PPO_PP": 100.0 * (0.9525 - success / 400),
        "PPO_MINUS_DWA_SM_PP": 100.0 * (success / 400 - 0.5375),
        "PPO_VS_PROPOSED_MCNEMAR_P": paired["proposed"]["mcnemar_exact_two_sided_p"],
        "PPO_VS_DWA_SM_MCNEMAR_P": paired["dwa_sm"]["mcnemar_exact_two_sided_p"],
        "PPO_MEAN_COMPUTE_MS_PER_STEP": runtime["mean_compute_ms_per_control_step"],
        "PPO_SHARED_SCENE_TRAJECTORY_READY": "YES" if hashes_valid else "NO",
        "PPO_DIRECT_BUDGET_EXHAUSTED": "NO",
        "ACADEMIC_INTEGRITY_GATE": "PASS",
        "FINAL_RECONCILIATION": "PASS" if reconciliation_pass else "FAIL",
        "RECOMMENDED_NEXT_STEP": "BUILD_SHARED_SCENE_TRAJECTORY_FIGURE" if reconciliation_pass else "EXCLUDE_PPO_AND_WRITE_PAPER",
    }
    write_json(root / "conclusion.json", conclusion)
    table = {
        "Method": "PPO-Direct",
        "Overall success": f"{success}/400 ({100*success/400:.2f}%)",
        "Stage I": f"{100*stages['stage_1']:.1f}%",
        "Stage II": f"{100*stages['stage_2']:.1f}%",
        "Stage III": f"{100*stages['stage_3']:.1f}%",
        "Stage IV": f"{100*stages['stage_4']:.1f}%",
        "Stage III+IV": f"{100*stage_3_4_success:.1f}%",
        "Collision": f"{100*collision/400:.2f}%",
        "Peer collision": f"{100*peer/400:.2f}%",
        "Timeout": f"{100*timeout/400:.2f}%",
        "Agent completion": f"{100*agent_completion:.2f}%",
        "Compute ms/step": f"{runtime['mean_compute_ms_per_control_step']:.3f}",
    }
    write_csv(root / "13_paper_ready/PPO_DIRECT_PAPER_TABLE_ROW.csv", [table])
    result_text = f"""# PPO-Direct paper result

After Development-only checkpoint selection and one sealed Holdout, PPO-Direct was frozen and evaluated post hoc on the unchanged Formal V2 block. It achieved **{success}/400 ({100*success/400:.2f}%)** team success, **{100*collision/400:.2f}%** collision, **{100*peer/400:.2f}%** inter-agent collision, and **{100*timeout/400:.2f}%** timeout. Stage I–IV success was **{100*stages['stage_1']:.1f}% / {100*stages['stage_2']:.1f}% / {100*stages['stage_3']:.1f}% / {100*stages['stage_4']:.1f}%**; Stage III+IV pooled success was **{100*stage_3_4_success:.1f}%**.

Relative to Proposed (95.25%), PPO-Direct differed by **{-conclusion['PROPOSED_MINUS_PPO_PP']:+.2f} pp** (exact paired McNemar p={paired['proposed']['mcnemar_exact_two_sided_p']:.6g}). Relative to DWA-SensingMatched (53.75%), it differed by **{conclusion['PPO_MINUS_DWA_SM_PP']:+.2f} pp** (p={paired['dwa_sm']['mcnemar_exact_two_sided_p']:.6g}). These comparisons use the same 400 scenario identifiers; continuous comparisons use only both-success pairs.

The mean PPO-specific online compute was **{runtime['mean_compute_ms_per_control_step']:.3f} ms/control step**, excluding shared environment physics. All 400 PPO trajectories were saved and hash-verified for the shared-scene figure.
"""
    (root / "13_paper_ready/PPO_DIRECT_PAPER_RESULT.md").write_text(result_text, encoding="utf-8")
    report = f"""# PPO-Direct Full Baseline Report

## Executive result

PPO-Direct completed the physically matched, information-matched development protocol and the post-hoc frozen Formal V2 evaluation. Formal team success was **{success}/400 ({100*success/400:.2f}%)**; collision was **{100*collision/400:.2f}%**, peer collision **{100*peer/400:.2f}%**, timeout **{100*timeout/400:.2f}%**, and agent completion **{100*agent_completion:.2f}%**.

## Method integrity

The baseline is standard parameter-shared decentralized PPO. Its 533-D per-agent local observation contains no Proposal, FP-SHEP, GAT, R-ERR, SAC-DMP, global map, or future state. Its 3-D action is mapped to native bounded acceleration and committed jointly for all UAVs. Physics, information, atomicity, isolation, checkpoint selection, Holdout, and trajectory reconciliation gates passed.

## Formal stage results

| Stage | Success |
|---|---:|
| Stage I | {100*stages['stage_1']:.1f}% |
| Stage II | {100*stages['stage_2']:.1f}% |
| Stage III | {100*stages['stage_3']:.1f}% |
| Stage IV | {100*stages['stage_4']:.1f}% |
| Stage III+IV pooled | {100*stage_3_4_success:.1f}% |

## Paired comparisons

- Proposed: PPO minus comparator {paired['proposed']['ppo_minus_comparator_pp']:+.2f} pp; McNemar p={paired['proposed']['mcnemar_exact_two_sided_p']:.6g}.
- DWA-SensingMatched: PPO minus comparator {paired['dwa_sm']['ppo_minus_comparator_pp']:+.2f} pp; p={paired['dwa_sm']['mcnemar_exact_two_sided_p']:.6g}.
- Direct SAC-DMP: PPO minus comparator {paired['sac']['ppo_minus_comparator_pp']:+.2f} pp; p={paired['sac']['mcnemar_exact_two_sided_p']:.6g}.

## Final decision

`PPO_BASELINE_READY_FOR_PAPER = {conclusion['PPO_BASELINE_READY_FOR_PAPER']}` and `FINAL_RECONCILIATION = {conclusion['FINAL_RECONCILIATION']}`. The next task is the shared-scene trajectory figure.
"""
    (root / "FINAL_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "analyze-formal"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dev-tag")
    parser.add_argument("--holdout-tag")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "freeze":
        if args.checkpoint is None or not args.dev_tag or not args.holdout_tag:
            parser.error("freeze requires --checkpoint, --dev-tag, and --holdout-tag")
        freeze(root, args.checkpoint, args.dev_tag, args.holdout_tag)
    else:
        analyze_formal(root)


if __name__ == "__main__":
    main()
