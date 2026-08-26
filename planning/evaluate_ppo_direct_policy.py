"""Frozen deterministic evaluator for PPO-Direct checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from planning.final_four_stage_benchmark import step_direct_accelerations
from planning.ppo_direct_baseline import (
    NUM_AGENTS,
    build_local_observations,
    install_pandas_import_guard,
    make_environment,
    normalized_actions_to_accelerations,
)


FORMAL_MANIFEST = Path(
    "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2/FORMAL_V2_MANIFEST.json"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(model: Any) -> None:
    if str(model.device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def obstacle_clearance(env: Any) -> float:
    obstacles = list(env.static_obstacles) + list(env.dynamic_obstacles)
    return min(
        (
            float(obstacle.signed_distance(dynamic.p))
            for dynamic in env.dynamics
            for obstacle in obstacles
        ),
        default=float("inf"),
    )


def episode(model: Any, entry: Mapping[str, Any], save_path: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    env = make_environment(entry)
    obs_started = time.perf_counter()
    observation = build_local_observations(env)
    observation_s = time.perf_counter() - obs_started
    positions = [env._positions().copy()]
    velocities = [env._velocities().copy()]
    applied_accelerations: list[np.ndarray] = []
    normalized_actions: list[np.ndarray] = []
    min_obstacle = obstacle_clearance(env)
    min_peer = float(env._check_collision()["min_inter_agent_distance"])
    inference_s = 0.0
    conversion_s = 0.0
    last_info: dict[str, Any] | None = None
    while True:
        synchronize(model)
        t0 = time.perf_counter()
        actions, _ = model.predict(observation, deterministic=True)
        synchronize(model)
        inference_s += time.perf_counter() - t0
        actions = np.asarray(actions, dtype=np.float32)
        t0 = time.perf_counter()
        accelerations = normalized_actions_to_accelerations(actions)
        conversion_s += time.perf_counter() - t0
        terminated, truncated, info = step_direct_accelerations(
            env, accelerations, refresh_sensors=True
        )
        positions.append(env._positions().copy())
        velocities.append(env._velocities().copy())
        normalized_actions.append(actions.copy())
        applied_accelerations.append(np.asarray(info["applied_accelerations"], dtype=float))
        min_obstacle = min(min_obstacle, obstacle_clearance(env))
        min_peer = min(min_peer, float(info["min_inter_agent_distance"]))
        t0 = time.perf_counter()
        observation = build_local_observations(env)
        observation_s += time.perf_counter() - t0
        last_info = dict(info)
        if terminated or truncated:
            break
    assert last_info is not None
    pos = np.asarray(positions, dtype=float)
    vel = np.asarray(velocities, dtype=float)
    acc = np.asarray(applied_accelerations, dtype=float)
    act = np.asarray(normalized_actions, dtype=float)
    start_dist = np.linalg.norm(np.asarray(entry["goals"]) - np.asarray(entry["starts"]), axis=1)
    final_dist = np.asarray(last_info["distance_to_goals"], dtype=float)
    success_mask = np.asarray(last_info["success_mask"], dtype=bool)
    collision_mask = np.asarray(last_info["collision_mask"], dtype=bool)
    obstacle_mask = np.asarray(last_info["obstacle_collision_mask"], dtype=bool)
    static_mask = np.asarray(last_info["static_obstacle_collision_mask"], dtype=bool)
    dynamic_mask = np.asarray(last_info["dynamic_obstacle_collision_mask"], dtype=bool)
    peer_mask = np.asarray(last_info["inter_agent_collision_mask"], dtype=bool)
    boundary_mask = np.asarray(last_info["boundary_collision_mask"], dtype=bool)
    path = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=2), axis=0)
    efficiency = start_dist / np.maximum(path, 1e-8)
    if len(acc) >= 2:
        jerk = np.diff(acc, axis=0) / 0.1
        smooth_per_agent = np.mean(np.sum(jerk**2, axis=2), axis=0)
    else:
        smooth_per_agent = np.zeros(NUM_AGENTS, dtype=float)
    compute_ms = 1000.0 * (inference_s + observation_s + conversion_s)
    team = {
        "scenario_id": entry["scenario_id"],
        "seed": int(entry["seed"]),
        "stage": entry["stage"],
        "family": entry["family"],
        "task_pattern": entry["task_pattern"],
        "team_success": bool(last_info["success"]),
        "any_collision": bool(last_info["collision"]),
        "obstacle_collision": bool(np.any(obstacle_mask)),
        "static_obstacle_collision": bool(np.any(static_mask)),
        "dynamic_obstacle_collision": bool(np.any(dynamic_mask)),
        "inter_agent_collision": bool(np.any(peer_mask)),
        "boundary_collision": bool(np.any(boundary_mask)),
        "timeout": bool(last_info["truncated"]),
        "agent_completion_rate": float(np.mean(success_mask)),
        "steps": int(last_info["steps"]),
        "completion_time_s": 0.1 * int(last_info["steps"]) if last_info["success"] else "",
        "team_path_length_m": float(np.sum(path)),
        "team_path_efficiency": float(np.mean(efficiency[success_mask])) if np.any(success_mask) else "",
        "trajectory_smoothness": float(np.mean(smooth_per_agent)),
        "minimum_obstacle_clearance_m": float(min_obstacle),
        "minimum_inter_agent_distance_m": float(min_peer),
        "initial_goal_distance_team_mean_m": float(np.mean(start_dist)),
        "final_goal_distance_team_mean_m": float(np.mean(final_dist)),
        "terminal_goal_progress_team_mean_m": float(np.mean(start_dist - final_dist)),
        "mean_speed_mps": float(np.mean(np.linalg.norm(vel, axis=2))),
        "peak_speed_mps": float(np.max(np.linalg.norm(vel, axis=2))),
        "mean_applied_acceleration_mps2": float(np.mean(np.linalg.norm(acc, axis=2))),
        "peak_applied_acceleration_mps2": float(np.max(np.linalg.norm(acc, axis=2))),
        "mean_abs_normalized_action": float(np.mean(np.abs(act))),
        "action_saturation_rate": float(np.mean(np.abs(act) >= 0.999)),
        "policy_calls": int(len(act) * NUM_AGENTS),
        "ppo_inference_compute_ms": 1000.0 * inference_s,
        "observation_assembly_compute_ms": 1000.0 * observation_s,
        "action_conversion_compute_ms": 1000.0 * conversion_s,
        "total_online_compute_ms": compute_ms,
        "mean_compute_ms_per_control_step": compute_ms / max(int(last_info["steps"]), 1),
    }
    agents = [
        {
            "scenario_id": entry["scenario_id"],
            "agent_id": agent_id,
            "completed": bool(success_mask[agent_id]),
            "any_collision": bool(collision_mask[agent_id]),
            "obstacle_collision": bool(obstacle_mask[agent_id]),
            "static_obstacle_collision": bool(static_mask[agent_id]),
            "dynamic_obstacle_collision": bool(dynamic_mask[agent_id]),
            "inter_agent_collision": bool(peer_mask[agent_id]),
            "boundary_collision": bool(boundary_mask[agent_id]),
            "initial_goal_distance_m": float(start_dist[agent_id]),
            "final_goal_distance_m": float(final_dist[agent_id]),
            "goal_progress_m": float(start_dist[agent_id] - final_dist[agent_id]),
            "path_length_m": float(path[agent_id]),
            "path_efficiency": float(efficiency[agent_id]),
            "smoothness": float(smooth_per_agent[agent_id]),
        }
        for agent_id in range(NUM_AGENTS)
    ]
    trajectory = None
    if save_path is not None:
        trajectory = {
            "schema_version": "ppo_direct_formal_trajectory_v1",
            "scenario_id": entry["scenario_id"],
            "stage": entry["stage"],
            "seed": int(entry["seed"]),
            "dt": 0.1,
            "starts": entry["starts"],
            "goals": entry["goals"],
            "static_obstacles": entry["static_obstacles"],
            "dynamic_obstacles": entry["dynamic_obstacles"],
            "positions": pos.tolist(),
            "velocities": vel.tolist(),
            "applied_accelerations": acc.tolist(),
            "normalized_actions": act.tolist(),
            "outcome": team,
        }
        write_json(save_path, trajectory)
    env.close()
    return team, agents, trajectory


def summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, list[Mapping[str, Any]]]] = [("overall", list(rows))]
    groups += [
        (stage, [row for row in rows if row["stage"] == stage])
        for stage in ("stage_1", "stage_2", "stage_3", "stage_4")
    ]
    groups.append(("stage_3_4", [row for row in rows if row["stage"] in {"stage_3", "stage_4"}]))
    result: list[dict[str, Any]] = []
    for scope, block in groups:
        successful = [row for row in block if row["team_success"]]
        result.append(
            {
                "scope": scope,
                "scenario_count": len(block),
                "success_count": sum(bool(row["team_success"]) for row in block),
                "success_rate": float(np.mean([row["team_success"] for row in block])),
                "collision_rate": float(np.mean([row["any_collision"] for row in block])),
                "static_collision_rate": float(np.mean([row["static_obstacle_collision"] for row in block])),
                "dynamic_collision_rate": float(np.mean([row["dynamic_obstacle_collision"] for row in block])),
                "peer_collision_rate": float(np.mean([row["inter_agent_collision"] for row in block])),
                "timeout_rate": float(np.mean([row["timeout"] for row in block])),
                "agent_completion_rate": float(np.mean([row["agent_completion_rate"] for row in block])),
                "mean_goal_progress_m": float(np.mean([row["terminal_goal_progress_team_mean_m"] for row in block])),
                "mean_steps": float(np.mean([row["steps"] for row in block])),
                "mean_success_completion_time_s": float(np.mean([row["completion_time_s"] for row in successful])) if successful else "",
                "mean_compute_ms_per_step": float(np.mean([row["mean_compute_ms_per_control_step"] for row in block])),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "holdout", "formal"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    install_pandas_import_guard()
    from stable_baselines3 import PPO

    root = args.root.resolve()
    manifest_path = (
        FORMAL_MANIFEST
        if args.split == "formal"
        else root
        / f"03_training_environment/PPO_DIRECT_{args.split.upper()}_MANIFEST.json"
    )
    entries = list(load_json(manifest_path)["entries"])
    if args.limit:
        count = int(args.limit) // 4
        entries = [
            row
            for stage in ("stage_1", "stage_2", "stage_3", "stage_4")
            for row in [entry for entry in entries if entry["stage"] == stage][:count]
        ]
    model = PPO.load(str(args.checkpoint), device=args.device)
    output_dir = {
        "dev": root / "06_development",
        "holdout": root / "07_holdout",
        "formal": root / "09_formal_v2",
    }[args.split]
    team_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, 1):
        trajectory_path = None
        if args.save_trajectories:
            trajectory_path = root / f"12_trajectories/formal_records/{entry['scenario_id']}.json"
        team, agents, _ = episode(model, entry, trajectory_path)
        team_rows.append(team)
        agent_rows.extend(agents)
        if trajectory_path is not None:
            trajectory_rows.append(
                {
                    "scenario_id": entry["scenario_id"],
                    "stage": entry["stage"],
                    "success": team["team_success"],
                    "path": str(trajectory_path.relative_to(root)),
                    "sha256": sha256(trajectory_path),
                }
            )
        if index % 10 == 0 or index == len(entries):
            print(
                json.dumps(
                    {
                        "split": args.split,
                        "tag": args.tag,
                        "done": index,
                        "total": len(entries),
                        "success": float(np.mean([row["team_success"] for row in team_rows])),
                        "collision": float(np.mean([row["any_collision"] for row in team_rows])),
                        "goal_progress_m": float(np.mean([row["terminal_goal_progress_team_mean_m"] for row in team_rows])),
                    }
                ),
                flush=True,
            )
    summary_rows = summary(team_rows)
    prefix = "ppo_direct_formal" if args.split == "formal" else f"PPO_DIRECT_{args.split.upper()}"
    write_csv(output_dir / f"{prefix}_team_results_{args.tag}.csv", team_rows)
    write_csv(output_dir / f"{prefix}_agent_results_{args.tag}.csv", agent_rows)
    write_csv(output_dir / f"{prefix}_summary_{args.tag}.csv", summary_rows)
    # Required stable aliases point to the most recently executed valid evaluation.
    required = {
        "dev": "PPO_DIRECT_DEV_RESULTS.csv",
        "holdout": "PPO_DIRECT_HOLDOUT_RESULTS.csv",
        "formal": "ppo_direct_formal_team_results.csv",
    }[args.split]
    write_csv(output_dir / required, team_rows)
    if args.split == "formal":
        write_csv(output_dir / "ppo_direct_formal_agent_results.csv", agent_rows)
        write_csv(output_dir / "ppo_direct_stage_summary.csv", summary_rows)
        write_csv(root / "12_trajectories/ppo_direct_trajectory_manifest.csv", trajectory_rows)
    write_json(
        output_dir / f"evaluation_{args.tag}.json",
        {
            "schema_version": "ppo_direct_evaluation_v1",
            "split": args.split,
            "tag": args.tag,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "scenario_count": len(entries),
            "summary": summary_rows,
        },
    )


if __name__ == "__main__":
    main()
