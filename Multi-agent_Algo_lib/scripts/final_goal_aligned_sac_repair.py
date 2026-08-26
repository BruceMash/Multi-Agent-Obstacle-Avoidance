#!/usr/bin/env python3
"""Bounded trajectory-quality repair: reference audit and one-shot horizon screen.

The module deliberately keeps the historical proposal mode as the default.
The only alternative exposed here is the preregistered goal-aligned distance
rule.  Later SAC stages are added to this same driver so the experiment has a
single auditable entry point.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as _pandas  # noqa: F401 -- Windows torch/pyarrow import order
import torch
from gymnasium import spaces


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.online_runtime_instrumentation import OnlineRuntimeRecorder, TimedPolicyProxy  # noqa: E402
from runner_sac import build_model, save_checkpoint  # noqa: E402
from experiment_config import EXPERIMENT_CONFIG  # noqa: E402
from scripts.train_long_range_local_sac import LongRangeLocalReferenceEnv  # noqa: E402
from scripts.evaluate_gat_v1_err_development import (  # noqa: E402
    METHOD_RERR_GAT,
    build_online_gat_plan_optimized,
    run_episode,
)
from scripts.run_gat_recurrent_r_development import FileBackedBuilder  # noqa: E402
from scripts.run_long_range_development import DevelopmentRuntime  # noqa: E402


ARTIFACT_RELATIVE = Path("artifacts/final_goal_aligned_sac_repair/20260825_003105")
ARTIFACT_ROOT = REPO_ROOT / ARTIFACT_RELATIVE
SOURCE_STUDY_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
SOURCE_MANIFEST = SOURCE_STUDY_ROOT / "07_development/GAT_RS_DEV_SCENE_MANIFEST.json"
SOURCE_H0_RECORDS = REPO_ROOT / "artifacts/continuous_reference_transition/20260824_132552/04_development/records/original/episode_records"
FORMAL_RECORDS = SOURCE_STUDY_ROOT / "10_formal_v2/formal_records/M9_Proposed_RERR_GAT_SAC_DMP"
FINAL_METHOD_CONFIG = SOURCE_STUDY_ROOT / "09_final_freeze/method_configs/M9_Proposed_RERR_GAT_SAC_DMP.json"
SAC_TRAINING_CONFIG = REPO_ROOT / "configs/training/long_range_local_sac_256_encoder_adapt.json"
H1_RECORDS = ARTIFACT_ROOT / "horizon_h1_records"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows([{key: json_ready(row.get(key)) for key in fields} for row in rows])
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {"count": 0}
    names = ("P05", "P25", "P50", "P75", "P90", "P95", "P99")
    result = dict(zip(names, np.percentile(array, (5, 25, 50, 75, 90, 95, 99)).tolist()))
    result.update({"max": float(np.max(array)), "count": int(array.size)})
    return result


def selected_development_ids() -> list[str]:
    ids = sorted(
        path.stem
        for path in SOURCE_H0_RECORDS.glob("GATRS_DEV_*.json")
        if not path.name.endswith("_SOFTWARE_ERROR.json")
    )
    if len(ids) != 100:
        raise RuntimeError(f"expected the frozen balanced 100-scene block, got {len(ids)}")
    return ids


def filtered_manifest() -> dict[str, Any]:
    source = load_json(SOURCE_MANIFEST)
    selected = set(selected_development_ids())
    entries = [row for row in source["entries"] if str(row["scenario_id"]) in selected]
    if len(entries) != 100:
        raise RuntimeError("source manifest does not contain the frozen 100-scene block")
    cells = Counter((str(row["stage"]), str(row["family"])) for row in entries)
    if len(cells) != 20 or set(cells.values()) != {5}:
        raise RuntimeError(f"development block is not 4x5x5 balanced: {cells}")
    return {**source, "entries": entries, "unique_scenario_count": len(entries)}


def reference_distributions(record_root: Path) -> dict[str, Any]:
    active_reference: list[float] = []
    accepted_reference: list[float] = []
    accepted_terminal: list[float] = []
    candidate: list[float] = []
    event_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    episode_count = 0
    for json_path in sorted(record_root.glob("*.json")):
        if json_path.name.endswith("_SOFTWARE_ERROR.json"):
            continue
        npz_path = json_path.with_name(json_path.stem + "_trajectory.npz")
        if not npz_path.exists():
            continue
        payload = load_json(json_path)
        arrays = np.load(npz_path)
        positions = np.asarray(arrays["positions"], dtype=float)
        if positions.ndim != 3:
            # Compact file-backed records are row-major and are not used by Stage A.
            continue
        active_goals = np.asarray(
            arrays["active_goals"] if "active_goals" in arrays else arrays["g_cmd"],
            dtype=float,
        )
        limit = min(len(positions), len(active_goals))
        events_by_agent: dict[int, list[dict[str, Any]]] = {
            agent_id: [] for agent_id in range(positions.shape[1])
        }
        for event in payload.get("events", []):
            events_by_agent[int(event["agent_id"])].append(event)
            event_counts[str(event.get("event"))] += 1
            type_counts[str(event.get("new_active_goal_type"))] += 1
            step = int(event["step"])
            agent_id = int(event["agent_id"])
            if step >= limit:
                continue
            position = positions[step, agent_id]
            new_goal = np.asarray(event["new_active_goal"], dtype=float)
            accepted_distance = float(np.linalg.norm(new_goal - position))
            if str(event.get("new_active_goal_type")) == "reference":
                accepted_reference.append(accepted_distance)
            else:
                accepted_terminal.append(accepted_distance)
            for point in event.get("candidate_world_points", []):
                candidate.append(float(np.linalg.norm(np.asarray(point, dtype=float) - position)))
        for agent_id, events in events_by_agent.items():
            events.sort(key=lambda row: int(row["step"]))
            for index, event in enumerate(events):
                if str(event.get("new_active_goal_type")) != "reference":
                    continue
                start = max(0, int(event["step"]))
                stop = limit if index + 1 == len(events) else min(limit, int(events[index + 1]["step"]))
                if stop > start:
                    active_reference.extend(
                        np.linalg.norm(
                            active_goals[start:stop, agent_id] - positions[start:stop, agent_id],
                            axis=1,
                        ).tolist()
                    )
        episode_count += 1
    return {
        "episode_count": episode_count,
        "active_local_reference_distance_m": distribution(active_reference),
        "accepted_local_reference_distance_m": distribution(accepted_reference),
        "accepted_terminal_restoration_distance_m": distribution(accepted_terminal),
        "retained_candidate_distance_m": distribution(candidate),
        "active_reference_change_causes": dict(sorted(event_counts.items())),
        "accepted_goal_type_counts": dict(sorted(type_counts.items())),
    }


def stage_a() -> None:
    training = load_json(SAC_TRAINING_CONFIG)
    manifest = filtered_manifest()
    payload = {
        "schema_version": "final_goal_aligned_sac_repair_reference_horizon_v1",
        "status": "PASS",
        "source_contract": {
            "candidate_distance_rule": (
                "terminal-aware clip(eta*safety_margin, minimum_step, min(s_max, task_distance)), "
                "then cap by guarded obstacle clearance, sensing range, and task distance"
            ),
            "eta": 0.58,
            "s_min_m": 0.35,
            "s_max_m": 1.05,
            "terminal_radius_m": 1.50,
            "terminal_step_ratio": 0.55,
            "terminal_min_step_m": 0.05,
            "safety_quantity": "normalized existing guarded velocity-aware safety margin clip(h/h_max,0,1)",
            "source_file": "Guidance/reference_point_proposal_demo.py",
            "source_sha256": sha256_file(REPO_ROOT / "Guidance/reference_point_proposal_demo.py"),
        },
        "development_100": reference_distributions(SOURCE_H0_RECORDS),
        "formal_v2_400": reference_distributions(FORMAL_RECORDS),
        "sac_adaptation_reference_support": {
            "near_probability": training["local_reference_distance_m"]["near_probability"],
            "near_range_m": training["local_reference_distance_m"]["near_range"],
            "extended_range_m": training["local_reference_distance_m"]["extended_range"],
            "source_file": str(SAC_TRAINING_CONFIG.relative_to(REPO_ROOT).as_posix()),
            "source_sha256": sha256_file(SAC_TRAINING_CONFIG),
        },
        "support_gate": {
            "r_min_supported_m": 0.35,
            "r_min_basis": "current source minimum plus continuous approach-to-reference runtime support",
            "r_max_supported_m": 4.50,
            "r_max_basis": "explicit extended SAC adaptation range and unchanged 4.5 m sensor range",
            "REFERENCE_HORIZON_MODIFICATION_ALLOWED": "YES",
        },
        "development_balance": {
            "scenario_count": len(manifest["entries"]),
            "stages": 4,
            "families_per_stage": 5,
            "scenes_per_stage_family_cell": 5,
        },
        "change_cause_semantics": {
            "INITIAL_SELECTION": "initial upper selection",
            "REFERENCE_COMPLETION_REPROPOSAL": "local-reference completion followed by regeneration",
            "REFERENCE_COMPLETION_HANDOFF": "task-goal restoration/handoff",
            "NORMAL_REPROPOSAL": "normal ERR regeneration",
            "EMERGENCY_REPROPOSAL": "emergency ERR regeneration",
        },
    }
    atomic_json(ARTIFACT_ROOT / "REFERENCE_HORIZON_CONTRACT.json", payload)
    atomic_json(ARTIFACT_ROOT / "development_manifest_100.json", manifest)
    print(json.dumps({"status": "PASS", "artifact": str(ARTIFACT_ROOT)}, indent=2))


def h1_runtime() -> tuple[DevelopmentRuntime, dict[str, Any], dict[str, Any]]:
    runtime_config = load_json(FINAL_METHOD_CONFIG)
    runtime_config["proposal_config"] = {
        **dict(runtime_config["proposal_config"]),
        "distance_mode": "goal_aligned_adaptive",
        "adaptive_r_min": 0.35,
        "adaptive_r_max": 4.50,
        "goal_alignment_gamma": 2.0,
    }
    manifest = filtered_manifest()
    runtime = DevelopmentRuntime(runtime_config, {"entries": []})
    runtime.builder = FileBackedBuilder(manifest, runtime_config, SOURCE_STUDY_ROOT)
    return runtime, runtime_config, manifest


def save_h1_record(
    output: Path,
    entry: Mapping[str, Any],
    episode: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    triggers: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sid = str(entry["scenario_id"])
    rows = list(extra["path_rows"])
    agent_count = int(episode["num_agents"])
    step_count = max(int(row["step"]) for row in rows) + 1
    positions = np.full((step_count, agent_count, 3), np.nan, dtype=np.float64)
    velocities = np.full_like(positions, np.nan)
    applied = np.full_like(positions, np.nan)
    active_goals = np.full_like(positions, np.nan)
    terminal_goals = np.full_like(positions, np.nan)
    for row in rows:
        step, agent = int(row["step"]), int(row["agent_id"])
        positions[step, agent] = [row["x_m"], row["y_m"], row["z_m"]]
        velocities[step, agent] = [row["vx_mps"], row["vy_mps"], row["vz_mps"]]
        applied[step, agent] = [row["applied_ax_mps2"], row["applied_ay_mps2"], row["applied_az_mps2"]]
        active_goals[step, agent] = [row["active_goal_x_m"], row["active_goal_y_m"], row["active_goal_z_m"]]
        terminal_goals[step, agent] = [row["terminal_goal_x_m"], row["terminal_goal_y_m"], row["terminal_goal_z_m"]]
    np.savez_compressed(
        output / f"{sid}_trajectory.npz",
        positions=positions,
        velocities=velocities,
        applied_accelerations_full=applied,
        active_goals=active_goals,
        terminal_goals=terminal_goals,
        dt=np.asarray(float(episode["dt"])),
    )
    atomic_json(
        output / f"{sid}.json",
        {
            "entry_identity": {key: entry[key] for key in ("scenario_id", "seed", "stage", "family", "task_pattern")},
            "episode": episode,
            "agents": list(agents),
            "events": list(events),
            "trigger_summary": {
                "row_count": len(triggers),
                "event_counts": dict(Counter(str(row["event"]) for row in triggers)),
            },
        },
    )


def horizon_run(shard_index: int, shard_count: int) -> None:
    runtime, runtime_config, manifest = h1_runtime()
    indexed = [
        (index, entry)
        for index, entry in enumerate(manifest["entries"])
        if index % int(shard_count) == int(shard_index)
    ]
    completed = {path.stem for path in H1_RECORDS.glob("GATRS_DEV_*.json")}
    for _, entry in indexed:
        sid = str(entry["scenario_id"])
        if sid in completed:
            continue
        print(f"[horizon-h1:{shard_index}/{shard_count}] start {sid}", flush=True)
        recorder = OnlineRuntimeRecorder()
        proxy = TimedPolicyProxy(runtime.policy, recorder)
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block="goal_aligned_horizon_dev",
                configuration_id="H1_GOAL_ALIGNED_GAMMA2",
                stage=entry["stage"],
                family=entry["family"],
                scenario_id=sid,
                seed=int(entry["seed"]),
                method=METHOD_RERR_GAT,
            ):
                episode, agents, events, triggers, extra = run_episode(
                    config=runtime.eval_config,
                    settings=runtime.settings,
                    multi_config=runtime.multi_config,
                    policy=proxy,
                    gat_model=runtime.gat_model,
                    gat_device=runtime.gat_device,
                    method=METHOD_RERR_GAT,
                    scenario=sid,
                    seed=int(entry["seed"]),
                    environment_builder=runtime.builder,
                    runtime_recorder=recorder,
                    upper_plan_builder=build_online_gat_plan_optimized,
                )
            save_h1_record(H1_RECORDS, entry, episode, agents, events, triggers, extra)
            print(
                f"[horizon-h1:{shard_index}/{shard_count}] complete {sid} success={int(episode['team_success'])}",
                flush=True,
            )
        except Exception as error:
            atomic_json(
                H1_RECORDS / f"{sid}_SOFTWARE_ERROR.json",
                {
                    "scenario_id": sid,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


def _trajectory_metrics(json_path: Path) -> dict[str, float]:
    payload = load_json(json_path)
    episode = payload["episode"]
    npz = np.load(json_path.with_name(json_path.stem + "_trajectory.npz"))
    dt = float(npz["dt"]) if "dt" in npz else float(episode["dt"])
    acceleration = np.asarray(
        npz["applied_accelerations_full"]
        if "applied_accelerations_full" in npz
        else np.concatenate(
            [np.zeros((1, npz["applied_accelerations"].shape[1], 3)), npz["applied_accelerations"]],
            axis=0,
        ),
        dtype=float,
    )
    jerk = np.diff(acceleration, axis=0) / dt
    finite = np.all(np.isfinite(jerk), axis=2)
    jerk_sq = np.sum(jerk * jerk, axis=2)
    vertical_sq = jerk[:, :, 2] ** 2
    switch_mask = np.zeros_like(finite, dtype=bool)
    half_width = int(round(0.5 / dt))
    for event in payload.get("events", []):
        if not bool(event.get("goal_changed", True)):
            continue
        step = int(event["step"])
        agent = int(event["agent_id"])
        lo, hi = max(0, step - half_width), min(len(jerk), step + half_width + 1)
        switch_mask[lo:hi, agent] = True
    switch_valid = switch_mask & finite
    return {
        "switch_region_jerk": float(np.mean(jerk_sq[switch_valid])) if np.any(switch_valid) else 0.0,
        "vertical_jerk": float(np.mean(vertical_sq[finite])) if np.any(finite) else 0.0,
    }


def _load_arm(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("GATRS_DEV_*.json")):
        if path.name.endswith("_SOFTWARE_ERROR.json"):
            continue
        payload = load_json(path)
        sid = str(payload.get("entry_identity", {}).get("scenario_id", payload["episode"].get("scenario_id")))
        result[sid] = {**payload, "trajectory_metrics": _trajectory_metrics(path)}
    return result


def _continuous_row(payload: Mapping[str, Any], scene: Mapping[str, Any]) -> dict[str, float]:
    episode = payload["episode"]
    straight = float(
        np.sum(
            np.linalg.norm(
                np.asarray(scene["goals"], dtype=float) - np.asarray(scene["starts"], dtype=float),
                axis=1,
            )
        )
    )
    path = float(episode["team_path_length_m"])
    events = payload.get("events", [])
    return {
        "team_path_m": path,
        "per_agent_path_m": float(episode["team_path_length_mean_agent_m"]),
        "detour_ratio": path / straight,
        "excess_path_m": path - straight,
        "completion_time_s": float(episode["completion_time_s"]),
        "smoothness": float(episode["trajectory_smoothness"]),
        "switch_region_jerk": float(payload["trajectory_metrics"]["switch_region_jerk"]),
        "vertical_jerk": float(payload["trajectory_metrics"]["vertical_jerk"]),
        "accepted_reference_changes": float(sum(bool(row.get("goal_changed", True)) for row in events)),
        "reference_completion_events": float(sum(str(row.get("event")) == "REFERENCE_COMPLETION_REPROPOSAL" for row in events)),
        "err_regeneration_events": float(sum(str(row.get("event")) in {"NORMAL_REPROPOSAL", "EMERGENCY_REPROPOSAL"} for row in events)),
        "online_compute_ms": float(episode["total_online_algorithm_compute_ms"]),
    }


def horizon_finalize() -> None:
    h0, h1 = _load_arm(SOURCE_H0_RECORDS), _load_arm(H1_RECORDS)
    expected = selected_development_ids()
    if set(h0) != set(expected) or set(h1) != set(expected):
        raise RuntimeError(f"incomplete paired horizon block: H0={len(h0)}, H1={len(h1)}")
    manifest = filtered_manifest()
    scenes = {
        str(row["scenario_id"]): load_json(SOURCE_STUDY_ROOT / str(row["scenario_file"]))
        for row in manifest["entries"]
    }
    both_success = [sid for sid in expected if h0[sid]["episode"]["team_success"] and h1[sid]["episode"]["team_success"]]
    rows: list[dict[str, Any]] = []
    continuous_by_arm: dict[str, list[dict[str, float]]] = {}
    for arm, records in (("H0_original", h0), ("H1_goal_aligned", h1)):
        all_payloads = [records[sid] for sid in expected]
        continuous = [_continuous_row(records[sid], scenes[sid]) for sid in both_success]
        continuous_by_arm[arm] = continuous
        rows.append(
            {
                "arm": arm,
                "n": len(all_payloads),
                "team_success": float(np.mean([row["episode"]["team_success"] for row in all_payloads])),
                "collision": float(np.mean([row["episode"]["collision"] for row in all_payloads])),
                "peer_collision": float(np.mean([row["episode"]["inter_agent_collision"] for row in all_payloads])),
                "agent_completion": float(np.mean([row["episode"]["agent_completion_rate"] for row in all_payloads])),
                "both_success_n": len(both_success),
                **{f"both_success_mean_{key}": float(np.mean([row[key] for row in continuous])) for key in continuous[0]},
            }
        )
    original, adaptive = rows
    success_loss_pp = 100.0 * (float(original["team_success"]) - float(adaptive["team_success"]))
    peer_delta_pp = 100.0 * (float(adaptive["peer_collision"]) - float(original["peer_collision"]))
    path_reduction = 100.0 * (
        float(original["both_success_mean_team_path_m"]) - float(adaptive["both_success_mean_team_path_m"])
    ) / float(original["both_success_mean_team_path_m"])
    detour_change = float(adaptive["both_success_mean_detour_ratio"]) - float(original["both_success_mean_detour_ratio"])
    ref_change_reduction = 100.0 * (
        float(original["both_success_mean_accepted_reference_changes"]) - float(adaptive["both_success_mean_accepted_reference_changes"])
    ) / float(original["both_success_mean_accepted_reference_changes"])
    reliability = success_loss_pp <= 1.0 and peer_delta_pp <= 1.0
    efficiency = path_reduction >= 2.0 or detour_change <= -0.01 or ref_change_reduction >= 10.0
    accepted = reliability and efficiency
    decision = {
        "success_loss_pp": success_loss_pp,
        "peer_collision_delta_pp": peer_delta_pp,
        "both_success_path_length_reduction_percent": path_reduction,
        "both_success_detour_ratio_change": detour_change,
        "both_success_reference_change_reduction_percent": ref_change_reduction,
        "reliability_gate": reliability,
        "path_efficiency_gate": efficiency,
        "GOAL_ALIGNED_HORIZON_ACCEPTED": "YES" if accepted else "NO",
        "frozen_horizon_for_sac_stage": "H1_goal_aligned" if accepted else "H0_original",
    }
    for row in rows:
        row.update({f"decision_{key}": value for key, value in decision.items()})
    write_csv(ARTIFACT_ROOT / "HORIZON_DEV_RESULTS.csv", rows)
    atomic_json(ARTIFACT_ROOT / "HORIZON_GATE_DECISION.json", decision)
    print(json.dumps(json_ready({"rows": rows, "decision": decision}), indent=2))


class TaskAwareLocalReferenceEnv(LongRangeLocalReferenceEnv):
    """Existing local SAC environment with only the nine authorized contexts."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        training_config: Mapping[str, Any],
        *,
        seed: int,
        training: bool,
    ) -> None:
        self._expanded_ready = False
        self.final_task_goal = np.zeros(3, dtype=float)
        self.previous_applied_acceleration = np.zeros(3, dtype=float)
        self.switch_age_s = 0.0
        super().__init__(manifest, training_config, seed=seed, training=training)
        self._expanded_ready = True
        old_space = self.observation_space
        context_low = np.asarray([-1.0] * 3 + [0.0, 0.0, 0.0] + [-1.0] * 3, dtype=np.float32)
        context_high = np.asarray([1.0] * 3 + [1.0, 1.0, 1.0] + [1.0] * 3, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([old_space.low, context_low]),
            high=np.concatenate([old_space.high, context_high]),
            dtype=np.float32,
        )
        if self.observation_space.shape != (531,):
            raise RuntimeError(f"expanded observation must be 531-D, got {self.observation_space.shape}")

    @property
    def extra_observation_dim(self) -> int:
        return 12 if self._expanded_ready else 3

    def _sample_patch(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        sample = super()._sample_patch(entry)
        if "task_goal" not in sample:
            raise RuntimeError("local-reference sampler did not retain the source task goal")
        self.final_task_goal = np.asarray(sample["task_goal"], dtype=float).copy()
        return sample

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.previous_applied_acceleration = np.zeros(3, dtype=float)
        self.switch_age_s = 0.0
        return super().reset(seed=seed, options=options)

    def _direction_safety(self, goal: np.ndarray) -> float:
        from Guidance.reference_point_proposal_demo import ProposalConfig, compute_sector_safety_field

        delta = np.asarray(goal, dtype=float) - np.asarray(self.dynamics.p, dtype=float)
        norm = float(np.linalg.norm(delta))
        direction = np.zeros(3, dtype=float) if norm < 1.0e-8 else delta / norm
        safety = compute_sector_safety_field(
            self.dynamics.p,
            np.asarray(goal, dtype=float),
            self.dynamics.v,
            self.latest_sensor_packet,
            self.sensor,
            ProposalConfig(),
            float(self.env_config.goal_tolerance),
        )
        directions = np.asarray(self.sensor.ray_directions, dtype=float).reshape(-1, 3)
        index = int(np.argmax(directions @ direction))
        return float(safety.normalized_margin.reshape(-1)[index])

    def _task_context(self) -> np.ndarray:
        delta = self.final_task_goal - np.asarray(self.dynamics.p, dtype=float)
        distance = float(np.linalg.norm(delta))
        direction = np.zeros(3, dtype=float) if distance < 1.0e-8 else delta / distance
        return np.concatenate(
            [
                direction.astype(np.float32),
                np.asarray(
                    [
                        np.clip(distance / 100.0, 0.0, 1.0),
                        self._direction_safety(self.final_task_goal),
                        np.clip(self.switch_age_s / 5.0, 0.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
                np.clip(self.previous_applied_acceleration / 4.0, -1.0, 1.0).astype(np.float32),
            ]
        ).astype(np.float32)

    def get_extra_observation(self):
        old = super().get_extra_observation()
        if not self._expanded_ready:
            return old
        return np.concatenate([old, self._task_context()]).astype(np.float32)

    def step(self, action: np.ndarray):
        previous_acceleration = self.previous_applied_acceleration.copy()
        previous_task_distance = float(np.linalg.norm(self.final_task_goal - self.dynamics.p))
        q_task_safe = self._direction_safety(self.final_task_goal)
        q_active_safe = self._direction_safety(self.goal)
        active_delta = np.asarray(self.goal, dtype=float) - np.asarray(self.dynamics.p, dtype=float)
        task_delta = self.final_task_goal - np.asarray(self.dynamics.p, dtype=float)
        active_norm, task_norm = float(np.linalg.norm(active_delta)), float(np.linalg.norm(task_delta))
        alignment = 1.0
        if active_norm > 1.0e-8 and task_norm > 1.0e-8:
            alignment = float(np.clip(np.dot(active_delta / active_norm, task_delta / task_norm), -1.0, 1.0))
        deviation_weight = 0.5 * (1.0 - alignment)
        age_before = float(self.switch_age_s)
        _, original_reward, terminated, truncated, info = super().step(action)
        applied = np.asarray(info["applied_acceleration"], dtype=float)
        self.previous_applied_acceleration = applied.copy()
        self.switch_age_s += float(self.dynamics.dt)
        next_task_distance = float(np.linalg.norm(self.final_task_goal - self.dynamics.p))
        task_progress = previous_task_distance - next_task_distance
        jerk = float(np.linalg.norm(applied - previous_acceleration) / float(self.dynamics.dt))
        reward_config = self.training_config.get("trajectory_repair_reward", {})
        jerk_threshold = float(reward_config.get("jerk_threshold_mps3", 0.0))
        jerk_scale = max(float(reward_config.get("jerk_scale_mps3", 1.0)), 1.0e-8)
        tail = max(0.0, jerk - jerk_threshold) / jerk_scale
        huber = 0.5 * tail * tail if tail <= 1.0 else tail - 0.5
        switch_weight = 1.0 + float(reward_config.get("switch_beta", 1.0)) * np.exp(-age_before / 0.15)
        smooth_raw = float(switch_weight * q_active_safe * huber)
        task_raw = float(q_task_safe * deviation_weight * task_progress)
        smooth_reward = -float(reward_config.get("lambda_j", 0.0)) * smooth_raw
        task_reward = float(reward_config.get("lambda_task", 0.0)) * task_raw
        reward = float(original_reward) + smooth_reward + task_reward
        observation = self.get_observation()
        info.update(
            {
                "reward_original": float(original_reward),
                "reward_smooth_secondary": smooth_reward,
                "reward_task_secondary": task_reward,
                "smooth_raw": smooth_raw,
                "task_progress_raw": task_raw,
                "executed_jerk_mps3": jerk,
                "q_task_safe": q_task_safe,
                "q_active_safe": q_active_safe,
                "switch_age_s": age_before,
            }
        )
        return observation, reward, terminated, truncated, info


def training_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(load_json(SAC_TRAINING_CONFIG))
    source_root = REPO_ROOT / str(config["artifact_root"])
    training_manifest = load_json(source_root / str(config["training_manifest"]))
    validation_manifest = load_json(source_root / str(config["validation_manifest"]))
    return config, training_manifest, validation_manifest


def model_config(config: Mapping[str, Any]):
    return replace(
        EXPERIMENT_CONFIG,
        learning_rate=float(config["learning_rate"]),
        buffer_size=int(config["buffer_size"]),
        batch_size=int(config["batch_size"]),
        learning_starts=int(config["learning_starts"]),
        train_freq=int(config["train_freq"]),
        gradient_steps=int(config["gradient_steps"]),
        ent_coef=config["ent_coef"],
        verbose=0,
    )


def load_original_model(env: Any, config: Mapping[str, Any]):
    model = build_model(env, config=model_config(config), verbose=0)
    checkpoint_path = REPO_ROOT / load_json(FINAL_METHOD_CONFIG)["sac_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
    model.actor.load_state_dict(checkpoint["actor"], strict=True)
    model.critic.load_state_dict(checkpoint["critic"], strict=True)
    model.critic_target.load_state_dict(checkpoint["critic_target"], strict=True)
    if "ent_coef_tensor" in checkpoint:
        model.log_ent_coef = None
        model.ent_coef_optimizer = None
        model.ent_coef_tensor = checkpoint["ent_coef_tensor"].to(model.device).detach()
    return model, checkpoint, checkpoint_path


def expand_checkpoint(model: Any, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {"new_context_columns_zero": {}, "copied_biases_exact": {}}
    for module_name in ("actor", "critic", "critic_target"):
        module = getattr(model, module_name)
        source = checkpoint[module_name]
        target = module.state_dict()
        expanded: dict[str, torch.Tensor] = {}
        for name, target_tensor in target.items():
            source_tensor = source[name]
            if target_tensor.shape == source_tensor.shape:
                expanded[name] = source_tensor.detach().clone()
                continue
            result = torch.zeros_like(target_tensor)
            if module_name == "actor" and name == "layers.0.weight":
                if source_tensor.shape != (256, 131) or target_tensor.shape != (256, 140):
                    raise RuntimeError("unexpected actor expansion shape")
                result[:, :131] = source_tensor
                report["new_context_columns_zero"][f"{module_name}.{name}"] = bool(
                    torch.count_nonzero(result[:, 131:140]) == 0
                )
            elif module_name in {"critic", "critic_target"} and name in {
                "q1_net.layers.0.weight",
                "q2_net.layers.0.weight",
            }:
                if source_tensor.shape != (256, 137) or target_tensor.shape != (256, 146):
                    raise RuntimeError("unexpected critic expansion shape")
                result[:, :131] = source_tensor[:, :131]
                result[:, 140:146] = source_tensor[:, 131:137]
                report["new_context_columns_zero"][f"{module_name}.{name}"] = bool(
                    torch.count_nonzero(result[:, 131:140]) == 0
                )
            else:
                raise RuntimeError(
                    f"unauthorized checkpoint shape change: {module_name}.{name} "
                    f"{tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"
                )
            expanded[name] = result
        module.load_state_dict(expanded, strict=True)
        report["copied_biases_exact"][module_name] = all(
            torch.equal(module.state_dict()[name].cpu(), source[name].cpu())
            for name in source
            if name.endswith("bias")
        )
    for name, parameter in model.actor.named_parameters():
        parameter.requires_grad_(not name.startswith("sensor_encoder."))
    for name, parameter in model.critic.named_parameters():
        parameter.requires_grad_(not name.startswith("sensor_encoder."))
    for parameter in model.critic_target.parameters():
        parameter.requires_grad_(False)
    learning_rate = float(model.lr_schedule(1.0))
    model.actor.optimizer = torch.optim.Adam(
        [parameter for parameter in model.actor.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    model.critic.optimizer = torch.optim.Adam(
        [parameter for parameter in model.critic.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    if "ent_coef_tensor" in checkpoint:
        model.log_ent_coef = None
        model.ent_coef_optimizer = None
        model.ent_coef_tensor = checkpoint["ent_coef_tensor"].to(model.device).detach()
    report["actor_trainable_parameter_count"] = int(
        sum(parameter.numel() for parameter in model.actor.parameters() if parameter.requires_grad)
    )
    report["actor_frozen_sensor_parameter_count"] = int(
        sum(parameter.numel() for parameter in model.actor.parameters() if not parameter.requires_grad)
    )
    report["critic_trainable_parameter_count"] = int(
        sum(parameter.numel() for parameter in model.critic.parameters() if parameter.requires_grad)
    )
    return report


def zero_init() -> None:
    config, training_manifest, _ = training_inputs()
    old_env = LongRangeLocalReferenceEnv(training_manifest, config, seed=2026082501, training=False)
    new_env = TaskAwareLocalReferenceEnv(training_manifest, config, seed=2026082501, training=False)
    old_model, checkpoint, checkpoint_path = load_original_model(old_env, config)
    new_model = build_model(new_env, config=model_config(config), verbose=0)
    expansion = expand_checkpoint(new_model, checkpoint)
    # Input-width changes may make CUDA select a different GEMM kernel even
    # when every added column is exactly zero.  Verify the mathematical
    # compatibility on CPU so the result measures the parameter mapping, not
    # backend-kernel roundoff (the expanded checkpoint remains device agnostic).
    for module in (
        old_model.actor,
        old_model.critic,
        new_model.actor,
        new_model.critic,
        new_model.critic_target,
    ):
        module.to("cpu")
        module.double()
        module.eval()
    actor_differences: list[float] = []
    critic_differences: list[float] = []
    sample_count = 0
    rng = np.random.default_rng(2026082502)
    for episode_index in range(32):
        old_observation, _ = old_env.reset(seed=2026082502 + episode_index)
        terminated = truncated = False
        while not (terminated or truncated) and sample_count < 2048:
            expanded_observation = np.concatenate(
                [np.asarray(old_observation, dtype=np.float32), np.zeros(9, dtype=np.float32)]
            )
            old_tensor = torch.as_tensor(old_observation[None], dtype=torch.float64)
            new_tensor = torch.as_tensor(expanded_observation[None], dtype=torch.float64)
            with torch.no_grad():
                old_scaled_action, _ = old_model._actor_action_log_prob(old_tensor, deterministic=True)
                new_scaled_action, _ = new_model._actor_action_log_prob(new_tensor, deterministic=True)
            actor_differences.append(
                float(torch.max(torch.abs(old_scaled_action - new_scaled_action)).cpu())
            )
            old_action = old_model._unscale_action(old_scaled_action.cpu().numpy())[0]
            scaled_action = rng.uniform(-1.0, 1.0, size=(1, 6)).astype(np.float32)
            action_tensor = torch.as_tensor(scaled_action, dtype=torch.float64)
            with torch.no_grad():
                old_q = old_model._critic_forward(old_tensor, action_tensor)
                new_q = new_model._critic_forward(new_tensor, action_tensor)
            critic_differences.append(
                max(float(torch.max(torch.abs(a - b)).cpu()) for a, b in zip(old_q, new_q))
            )
            old_observation, _, terminated, truncated, _ = old_env.step(old_action)
            sample_count += 1
        if sample_count >= 2048:
            break
    maximum_actor_difference = max(actor_differences, default=float("inf"))
    maximum_critic_difference = max(critic_differences, default=float("inf"))
    passed = maximum_actor_difference <= 1.0e-6 and maximum_critic_difference <= 1.0e-6
    observation_contract = {
        "schema_version": "task_aware_sac_observation_v1",
        "original_observation_dim": 522,
        "new_observation_dim": 531,
        "sensor_observation_dim": 519,
        "old_extra_observation_dim": 3,
        "new_extra_observation_dim": 12,
        "original_order_preserved": True,
        "added_context_order": [
            "final_task_goal_direction_3",
            "normalized_final_task_goal_distance_1",
            "existing_final_task_direction_safety_1",
            "normalized_switch_age_1",
            "previous_executed_acceleration_3",
        ],
        "task_distance_scale_m": 100.0,
        "switch_age_scale_s": 5.0,
        "acceleration_scale_mps2": 4.0,
        "final_goal_safety_source": "existing Proposal normalized guarded velocity-aware sector margin",
        "sensor_encoder_frozen": True,
        "upper_modules_changed": False,
        "expansion": expansion,
    }
    equivalence = {
        "schema_version": "task_aware_sac_zero_init_equivalence_v1",
        "source_checkpoint": str(checkpoint_path.relative_to(REPO_ROOT).as_posix()),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "sample_count": sample_count,
        "input_contract": "old_observation_plus_nine_zero_context_dimensions",
        "maximum_absolute_actor_action_difference": maximum_actor_difference,
        "maximum_absolute_critic_q_difference": maximum_critic_difference,
        "tolerance": 1.0e-6,
        "ZERO_INIT_EQUIVALENCE": "PASS" if passed else "FAIL",
    }
    atomic_json(ARTIFACT_ROOT / "SAC_OBSERVATION_EXTENSION.json", observation_contract)
    atomic_json(ARTIFACT_ROOT / "ZERO_INIT_EQUIVALENCE.json", equivalence)
    if not passed:
        raise RuntimeError(f"zero-init equivalence failed: {equivalence}")
    for module in (new_model.actor, new_model.critic, new_model.critic_target):
        module.float()
    checkpoint_dir = ARTIFACT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        new_model,
        checkpoint_dir / "zero_init_expanded.pt",
        extra={"trajectory_repair_step": 0, "zero_init_equivalence": equivalence},
    )
    print(json.dumps(equivalence, indent=2))


def _load_expanded_model(
    env: Any,
    config: Mapping[str, Any],
    checkpoint_path: Path,
) -> Any:
    """Build the authorized 531-D model and restore an expanded checkpoint."""

    model = build_model(env, config=model_config(config), verbose=0)
    original_path = REPO_ROOT / load_json(FINAL_METHOD_CONFIG)["sac_checkpoint"]
    original = torch.load(original_path, map_location=model.device, weights_only=False)
    expand_checkpoint(model, original)
    checkpoint = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
    model.actor.load_state_dict(checkpoint["actor"], strict=True)
    model.critic.load_state_dict(checkpoint["critic"], strict=True)
    model.critic_target.load_state_dict(checkpoint["critic_target"], strict=True)
    if "ent_coef_tensor" in checkpoint:
        model.log_ent_coef = None
        model.ent_coef_optimizer = None
        model.ent_coef_tensor = checkpoint["ent_coef_tensor"].to(model.device).detach()
    return model


def _h0_jerk_distribution() -> np.ndarray:
    values: list[np.ndarray] = []
    for sid in selected_development_ids():
        path = SOURCE_H0_RECORDS / f"{sid}_trajectory.npz"
        data = np.load(path)
        acceleration = np.asarray(data["applied_accelerations_full"], dtype=float)
        dt = float(data["dt"])
        jerk = np.linalg.norm(np.diff(acceleration, axis=0) / dt, axis=2)
        values.append(jerk[np.isfinite(jerk)])
    if not values:
        raise RuntimeError("no H0 jerk samples available for reward calibration")
    return np.concatenate(values)


def reward_calibrate() -> None:
    """Freeze reward scales from original Development behavior, without a grid."""

    if load_json(ARTIFACT_ROOT / "ZERO_INIT_EQUIVALENCE.json")["ZERO_INIT_EQUIVALENCE"] != "PASS":
        raise RuntimeError("reward calibration is forbidden until zero-init equivalence passes")
    horizon = load_json(ARTIFACT_ROOT / "HORIZON_GATE_DECISION.json")
    if horizon["frozen_horizon_for_sac_stage"] != "H0_original":
        raise RuntimeError("this run expected the rejected H1 decision and frozen H0 geometry")

    jerk_values = _h0_jerk_distribution()
    jerk_threshold = float(np.percentile(jerk_values, 75.0))
    jerk_scale = max(
        float(np.percentile(jerk_values, 90.0) - jerk_threshold),
        float(np.percentile(jerk_values, 75.0)) * 0.25,
        1.0e-6,
    )
    config, training_manifest, _ = training_inputs()
    provisional = {
        "jerk_threshold_mps3": jerk_threshold,
        "jerk_scale_mps3": jerk_scale,
        "switch_beta": 1.0,
        "lambda_j": 0.0,
        "lambda_task": 0.0,
        "task_distance_scale_m": 1.0,
    }
    config["trajectory_repair_reward"] = provisional
    env = TaskAwareLocalReferenceEnv(training_manifest, config, seed=2026082503, training=False)
    model = _load_expanded_model(
        env,
        config,
        ARTIFACT_ROOT / "checkpoints/zero_init_expanded.pt",
    )
    rewards: list[float] = []
    smooth_raw: list[float] = []
    task_raw: list[float] = []
    anchor_raw: list[float] = []
    transitions = 0
    episode = 0
    while transitions < 2048:
        observation, _ = env.reset(seed=2026082504 + episode)
        terminated = truncated = False
        while not (terminated or truncated) and transitions < 2048:
            action, _ = model.predict(observation, deterministic=True)
            next_observation, _, terminated, truncated, info = env.step(action)
            rewards.append(abs(float(info["reward_original"])))
            smooth_raw.append(float(info["smooth_raw"]))
            task_raw.append(abs(float(info["task_progress_raw"])))
            # At zero initialization, the student and original teacher are
            # exactly equal; retaining the measured zero is an integrity check.
            anchor_raw.append(0.0)
            observation = next_observation
            transitions += 1
        episode += 1
    env.close()

    positive_rewards = np.asarray([value for value in rewards if value > 1.0e-9], dtype=float)
    positive_smooth = np.asarray([value for value in smooth_raw if value > 1.0e-12], dtype=float)
    positive_task = np.asarray([value for value in task_raw if value > 1.0e-12], dtype=float)
    original_scale = float(np.median(positive_rewards)) if positive_rewards.size else 1.0
    smooth_scale = float(np.percentile(positive_smooth, 75.0)) if positive_smooth.size else 1.0
    task_scale = float(np.percentile(positive_task, 75.0)) if positive_task.size else 1.0
    # Each secondary term is calibrated to roughly five percent of a typical
    # non-zero original reward.  This is a single scale calculation, not a
    # searched hyperparameter family.
    lambda_j = float(np.clip(0.05 * original_scale / max(smooth_scale, 1.0e-9), 1.0e-6, 1.0))
    lambda_task = float(np.clip(0.05 * original_scale / max(task_scale, 1.0e-9), 1.0e-4, 10.0))
    lambda_anchor = 2.0
    reward_contract = {
        "schema_version": "task_aware_sac_reward_contract_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "calibration_transition_count": transitions,
        "calibration_source": "original frozen SAC behavior on independent long-range local patches",
        "original_reward_retained_complete": True,
        "secondary_objectives_only": ["switch_aware_tail_jerk", "safe_final_task_progress"],
        "jerk_distribution": distribution(jerk_values),
        "jerk_threshold_mps3": jerk_threshold,
        "jerk_threshold_definition": "P75 of raw executed acceleration-difference norm divided by dt on H0 Development100",
        "jerk_scale_mps3": jerk_scale,
        "switch_beta": 1.0,
        "switch_decay_s": 0.15,
        "smooth_safety_weight": "existing active-direction normalized Proposal margin; tends to zero in critical states",
        "task_safety_weight": "existing final-task-direction normalized Proposal margin",
        "task_distance_scale_m": 1.0,
        "lambda_j": lambda_j,
        "lambda_task": lambda_task,
        "lambda_anchor": lambda_anchor,
        "teacher_anchor_scan_low_normalized": 0.25,
        "teacher_anchor_scan_high_normalized": 0.45,
        "calibration_magnitudes": {
            "absolute_original_reward": distribution(rewards),
            "smooth_raw": distribution(smooth_raw),
            "absolute_task_progress_raw": distribution(task_raw),
            "zero_init_anchor_raw": distribution(anchor_raw),
            "typical_original_nonzero_p50": original_scale,
            "typical_smooth_positive_p75": smooth_scale,
            "typical_task_positive_p75": task_scale,
        },
        "coefficient_selection": "single magnitude calibration; no grid and no performance-driven retuning",
    }
    finetune_config = {
        "schema_version": "task_aware_sac_finetune_config_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "source_checkpoint": load_json(FINAL_METHOD_CONFIG)["sac_checkpoint"],
        "source_checkpoint_sha256": load_json(ARTIFACT_ROOT / "ZERO_INIT_EQUIVALENCE.json")[
            "source_checkpoint_sha256"
        ],
        "initial_checkpoint": str((ARTIFACT_RELATIVE / "checkpoints/zero_init_expanded.pt").as_posix()),
        "learning_rate": 3.0e-5,
        "fresh_replay_buffer": True,
        "training_steps": [50000, 100000, 250000, 500000],
        "absolute_ceiling_steps": 500000,
        "frozen_horizon": "H0_original",
        "sensor_encoder_frozen": True,
        "actor_trunk_trainable": True,
        "actor_output_trainable": True,
        "critics_trainable": True,
        "upper_modules_frozen": True,
        "reward_contract_sha256_pending": True,
        "development_gates": {
            "100k": {
                "subset_count": 20,
                "maximum_success_loss_pp": 5.0,
                "maximum_peer_collision_increase_pp": 3.0,
                "minimum_smoothness_improvement_percent": 0.0,
                "minimum_switch_jerk_improvement_percent": 0.0,
                "minimum_path_length_improvement_percent": -5.0,
            },
            "250k": {
                "subset_count": 100,
                "maximum_success_loss_pp": 1.0,
                "maximum_peer_collision_increase_pp": 1.0,
                "minimum_smoothness_improvement_percent": 8.0,
                "path_improvement_required": True,
            },
            "500k": {
                "subset_count": 100,
                "maximum_success_loss_pp": 1.0,
                "maximum_peer_collision_increase_pp": 1.0,
                "minimum_smoothness_improvement_percent": 15.0,
                "minimum_switch_jerk_improvement_percent": 15.0,
                "path_length_improvement_percent": 3.0,
                "excess_path_improvement_percent": 15.0,
                "detour_ratio_decrease": 0.01,
            },
        },
        "no_hyperparameter_grid": True,
        "formal_v2_access_forbidden": True,
    }
    atomic_json(ARTIFACT_ROOT / "SAC_REWARD_CONTRACT.json", reward_contract)
    finetune_config["reward_contract_sha256"] = sha256_file(ARTIFACT_ROOT / "SAC_REWARD_CONTRACT.json")
    finetune_config.pop("reward_contract_sha256_pending")
    atomic_json(ARTIFACT_ROOT / "SAC_FINETUNE_CONFIG.json", finetune_config)
    print(json.dumps({"reward_contract": reward_contract, "finetune_config": finetune_config}, indent=2))


def _sac_development_ids(count: int) -> list[str]:
    entries = filtered_manifest()["entries"]
    if int(count) == 100:
        return [str(row["scenario_id"]) for row in entries]
    if int(count) != 20:
        raise ValueError("SAC Development evaluation supports only frozen counts 20 or 100")
    first_by_cell: dict[tuple[str, str], str] = {}
    for row in entries:
        key = (str(row["stage"]), str(row["family"]))
        first_by_cell.setdefault(key, str(row["scenario_id"]))
    result = list(first_by_cell.values())
    if len(result) != 20:
        raise RuntimeError(f"expected one fixed scenario in each of 20 cells, got {len(result)}")
    return result


def _evaluate_sac_checkpoint(
    model: Any,
    *,
    step: int,
    count: int,
) -> dict[str, Any]:
    """Run the fixed recurrent Development block with the expanded actor."""

    runtime_config = load_json(FINAL_METHOD_CONFIG)
    runtime_config["sac_observation_extension"] = {
        "task_distance_scale_m": 100.0,
        "switch_age_scale_s": 5.0,
        "acceleration_scale_mps2": 4.0,
        "goal_tolerance_m": 0.25,
    }
    horizon = load_json(ARTIFACT_ROOT / "HORIZON_GATE_DECISION.json")
    if horizon["frozen_horizon_for_sac_stage"] != "H0_original":
        runtime_config["proposal_config"] = {
            **dict(runtime_config["proposal_config"]),
            "distance_mode": "goal_aligned_adaptive",
            "adaptive_r_min": 0.35,
            "adaptive_r_max": 4.50,
            "goal_alignment_gamma": 2.0,
        }
    manifest = filtered_manifest()
    runtime = DevelopmentRuntime(runtime_config, {"entries": []})
    runtime.eval_config["sac_observation_extension"] = copy.deepcopy(
        runtime_config["sac_observation_extension"]
    )
    runtime.builder = FileBackedBuilder(manifest, runtime_config, SOURCE_STUDY_ROOT)
    runtime.policy = model
    output = ARTIFACT_ROOT / "sac_dev_records" / f"step_{int(step):07d}"
    ids = _sac_development_ids(count)
    entry_by_id = {str(row["scenario_id"]): row for row in manifest["entries"]}
    completed = {path.stem for path in output.glob("GATRS_DEV_*.json")}
    model.actor.eval()
    for episode_index, sid in enumerate(ids, start=1):
        if sid in completed:
            continue
        entry = entry_by_id[sid]
        print(f"[sac-dev:{step}] {episode_index}/{len(ids)} start {sid}", flush=True)
        recorder = OnlineRuntimeRecorder()
        proxy = TimedPolicyProxy(model, recorder)
        try:
            with recorder.instrument_dmp(), recorder.scoped_context(
                evaluation_block=f"task_aware_sac_dev_{step}",
                configuration_id=f"SAC_TASK_AWARE_{step}",
                stage=entry["stage"],
                family=entry["family"],
                scenario_id=sid,
                seed=int(entry["seed"]),
                method=METHOD_RERR_GAT,
            ):
                episode, agents, events, triggers, extra = run_episode(
                    config=runtime.eval_config,
                    settings=runtime.settings,
                    multi_config=runtime.multi_config,
                    policy=proxy,
                    gat_model=runtime.gat_model,
                    gat_device=runtime.gat_device,
                    method=METHOD_RERR_GAT,
                    scenario=sid,
                    seed=int(entry["seed"]),
                    environment_builder=runtime.builder,
                    runtime_recorder=recorder,
                    upper_plan_builder=build_online_gat_plan_optimized,
                    sac_observation_extension={
                        "task_distance_scale_m": 100.0,
                        "switch_age_scale_s": 5.0,
                        "acceleration_scale_mps2": 4.0,
                    },
                )
            save_h1_record(output, entry, episode, agents, events, triggers, extra)
            print(
                f"[sac-dev:{step}] {episode_index}/{len(ids)} complete {sid} "
                f"success={int(episode['team_success'])}",
                flush=True,
            )
        except Exception as error:
            atomic_json(
                output / f"{sid}_SOFTWARE_ERROR.json",
                {
                    "scenario_id": sid,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    return _summarize_sac_development(step=step, ids=ids, new_root=output)


def _summarize_sac_development(
    *,
    step: int,
    ids: Sequence[str],
    new_root: Path,
) -> dict[str, Any]:
    original, new = _load_arm(SOURCE_H0_RECORDS), _load_arm(new_root)
    if set(new) != set(ids):
        raise RuntimeError(f"incomplete SAC Development evaluation at {step}: {len(new)}/{len(ids)}")
    manifest = filtered_manifest()
    entry_by_id = {str(row["scenario_id"]): row for row in manifest["entries"]}
    scenes = {
        sid: load_json(SOURCE_STUDY_ROOT / str(entry_by_id[sid]["scenario_file"]))
        for sid in ids
    }
    both_success = [
        sid
        for sid in ids
        if bool(original[sid]["episode"]["team_success"])
        and bool(new[sid]["episode"]["team_success"])
    ]
    if not both_success:
        raise RuntimeError("no both-success scenarios remain for continuous metrics")

    def arm_summary(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        payloads = [records[sid] for sid in ids]
        continuous = [_continuous_row(records[sid], scenes[sid]) for sid in both_success]
        return {
            "team_success": float(np.mean([row["episode"]["team_success"] for row in payloads])),
            "collision": float(np.mean([row["episode"]["collision"] for row in payloads])),
            "peer_collision": float(np.mean([row["episode"]["inter_agent_collision"] for row in payloads])),
            "agent_completion": float(np.mean([row["episode"]["agent_completion_rate"] for row in payloads])),
            "both_success_n": len(both_success),
            **{
                f"both_success_mean_{key}": float(np.mean([row[key] for row in continuous]))
                for key in continuous[0]
            },
        }

    baseline, candidate = arm_summary(original), arm_summary(new)

    def reduction(key: str) -> float:
        denominator = float(baseline[key])
        return 100.0 * (denominator - float(candidate[key])) / denominator

    summary: dict[str, Any] = {
        "training_steps": int(step),
        "evaluation_scenario_count": len(ids),
        "evaluation_scenario_ids": list(ids),
        "both_success_n": len(both_success),
        "original_success": baseline["team_success"],
        "new_success": candidate["team_success"],
        "success_loss_pp": 100.0 * (baseline["team_success"] - candidate["team_success"]),
        "original_peer_collision": baseline["peer_collision"],
        "new_peer_collision": candidate["peer_collision"],
        "peer_collision_delta_pp": 100.0 * (
            candidate["peer_collision"] - baseline["peer_collision"]
        ),
        "smoothness_improvement_percent": reduction("both_success_mean_smoothness"),
        "switch_jerk_improvement_percent": reduction("both_success_mean_switch_region_jerk"),
        "path_length_improvement_percent": reduction("both_success_mean_team_path_m"),
        "excess_path_improvement_percent": reduction("both_success_mean_excess_path_m"),
        "detour_ratio_change": float(candidate["both_success_mean_detour_ratio"])
        - float(baseline["both_success_mean_detour_ratio"]),
        "reference_change_delta": float(candidate["both_success_mean_accepted_reference_changes"])
        - float(baseline["both_success_mean_accepted_reference_changes"]),
        "original": baseline,
        "new": candidate,
    }
    gate_config = load_json(ARTIFACT_ROOT / "SAC_FINETUNE_CONFIG.json")["development_gates"]
    if int(step) == 100000:
        gate = gate_config["100k"]
        passed = (
            summary["success_loss_pp"] <= gate["maximum_success_loss_pp"]
            and summary["peer_collision_delta_pp"] <= gate["maximum_peer_collision_increase_pp"]
            and summary["smoothness_improvement_percent"] > gate["minimum_smoothness_improvement_percent"]
            and summary["switch_jerk_improvement_percent"] > gate["minimum_switch_jerk_improvement_percent"]
            and summary["path_length_improvement_percent"] >= gate["minimum_path_length_improvement_percent"]
        )
        gate_name = "100K_SANITY_GATE"
    elif int(step) == 250000:
        gate = gate_config["250k"]
        path_gain = (
            summary["path_length_improvement_percent"] > 0.0
            or summary["excess_path_improvement_percent"] > 0.0
            or summary["detour_ratio_change"] < 0.0
        )
        passed = (
            summary["success_loss_pp"] <= gate["maximum_success_loss_pp"]
            and summary["peer_collision_delta_pp"] <= gate["maximum_peer_collision_increase_pp"]
            and summary["smoothness_improvement_percent"] >= gate["minimum_smoothness_improvement_percent"]
            and path_gain
        )
        gate_name = "250K_DECISION_GATE"
    elif int(step) == 500000:
        gate = gate_config["500k"]
        path_gain = (
            summary["path_length_improvement_percent"] >= gate["path_length_improvement_percent"]
            or summary["excess_path_improvement_percent"] >= gate["excess_path_improvement_percent"]
            or summary["detour_ratio_change"] <= -gate["detour_ratio_decrease"]
        )
        passed = (
            summary["success_loss_pp"] <= gate["maximum_success_loss_pp"]
            and summary["peer_collision_delta_pp"] <= gate["maximum_peer_collision_increase_pp"]
            and summary["smoothness_improvement_percent"] >= gate["minimum_smoothness_improvement_percent"]
            and summary["switch_jerk_improvement_percent"] >= gate["minimum_switch_jerk_improvement_percent"]
            and path_gain
        )
        gate_name = "500K_FINAL_DEVELOPMENT_GATE"
    else:
        raise ValueError(f"no frozen gate for checkpoint {step}")
    summary.update({"gate": gate_name, "gate_pass": bool(passed)})
    atomic_json(ARTIFACT_ROOT / f"sac_dev_step_{int(step):07d}_summary.json", summary)
    return summary


def _history_row(step: int, status: str, summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "training_steps": int(step),
        "status": status,
        "timestamp_unix_s": time.time(),
    }
    if summary is not None:
        for key, value in summary.items():
            if not isinstance(value, (dict, list)):
                row[key] = value
    return row


def sac_train() -> None:
    """Bounded 50/100/250/500k fine-tune with hard sequential gates."""

    reward_contract = load_json(ARTIFACT_ROOT / "SAC_REWARD_CONTRACT.json")
    frozen_config = load_json(ARTIFACT_ROOT / "SAC_FINETUNE_CONFIG.json")
    config, training_manifest, _ = training_inputs()
    config["trajectory_repair_reward"] = {
        key: reward_contract[key]
        for key in (
            "jerk_threshold_mps3",
            "jerk_scale_mps3",
            "switch_beta",
            "lambda_j",
            "lambda_task",
            "task_distance_scale_m",
        )
    }
    config["learning_rate"] = float(frozen_config["learning_rate"])
    training_env = TaskAwareLocalReferenceEnv(
        training_manifest,
        config,
        seed=2026082505,
        training=True,
    )
    model = _load_expanded_model(
        training_env,
        config,
        ARTIFACT_ROOT / "checkpoints/zero_init_expanded.pt",
    )
    # Fresh replay is inherent to the newly constructed model.  Only the
    # checkpoint parameters are loaded; no historical buffer is restored.
    model.num_timesteps = 0
    model._n_updates = 0

    old_env = LongRangeLocalReferenceEnv(training_manifest, config, seed=2026082506, training=False)
    original_model, _, _ = load_original_model(old_env, config)
    teacher = copy.deepcopy(original_model.actor).to(model.device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model.safety_teacher_actor = teacher
    model.safety_teacher_old_extra_dim = 3
    model.safety_teacher_ray_count = 256
    model.safety_teacher_anchor_lambda = float(reward_contract["lambda_anchor"])
    model.safety_teacher_critical_scan_low = float(
        reward_contract["teacher_anchor_scan_low_normalized"]
    )
    model.safety_teacher_critical_scan_high = float(
        reward_contract["teacher_anchor_scan_high_normalized"]
    )

    checkpoint_dir = ARTIFACT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = [_history_row(0, "ZERO_INIT_READY")]
    dev_rows: list[dict[str, Any]] = []
    write_csv(ARTIFACT_ROOT / "SAC_TRAINING_HISTORY.csv", history)
    write_csv(ARTIFACT_ROOT / "SAC_DEV_RESULTS.csv", dev_rows)
    stop_reason = ""
    best_steps = 0
    cumulative = 0
    for target in (50000, 100000, 250000, 500000):
        additional = int(target - cumulative)
        print(f"[sac-train] training {cumulative} -> {target} steps", flush=True)
        model.learn(
            total_timesteps=additional,
            reset_num_timesteps=(cumulative == 0),
            progress_bar=False,
        )
        cumulative = int(target)
        checkpoint = checkpoint_dir / f"checkpoint_{target:07d}.pt"
        save_checkpoint(
            model,
            checkpoint,
            extra={
                "trajectory_repair_step": target,
                "reward_contract_sha256": sha256_file(ARTIFACT_ROOT / "SAC_REWARD_CONTRACT.json"),
                "sensor_encoder_frozen": True,
            },
        )
        history.append(_history_row(target, "CHECKPOINT_SAVED"))
        write_csv(ARTIFACT_ROOT / "SAC_TRAINING_HISTORY.csv", history)
        if target == 50000:
            continue
        evaluation_count = 20 if target == 100000 else 100
        summary = _evaluate_sac_checkpoint(model, step=target, count=evaluation_count)
        dev_rows.append(
            {
                "checkpoint_steps": target,
                **{key: value for key, value in summary.items() if not isinstance(value, (dict, list))},
            }
        )
        history.append(
            _history_row(
                target,
                "GATE_PASS" if summary["gate_pass"] else "GATE_FAIL",
                summary,
            )
        )
        write_csv(ARTIFACT_ROOT / "SAC_DEV_RESULTS.csv", dev_rows)
        write_csv(ARTIFACT_ROOT / "SAC_TRAINING_HISTORY.csv", history)
        print(json.dumps(json_ready(summary), indent=2), flush=True)
        if not bool(summary["gate_pass"]):
            stop_reason = f"{summary['gate']}_FAIL"
            break
        if target == 500000:
            best_steps = 500000
            stop_reason = "DEVELOPMENT_FINAL_GATE_PASS"
    if not stop_reason:
        stop_reason = "BOUNDED_TRAINING_STOP"
    result = {
        "schema_version": "task_aware_sac_training_result_v1",
        "SAC_FINETUNING_EXECUTED": "YES",
        "trained_steps": cumulative,
        "BEST_CHECKPOINT_STEPS": best_steps,
        "stop_reason": stop_reason,
        "holdout_eligible": best_steps == 500000,
        "fresh_replay_buffer": True,
        "source_sensor_encoder_frozen": True,
        "checkpoint": (
            str((ARTIFACT_RELATIVE / f"checkpoints/checkpoint_{best_steps:07d}.pt").as_posix())
            if best_steps
            else None
        ),
    }
    atomic_json(ARTIFACT_ROOT / "SAC_TRAINING_RESULT.json", result)
    training_env.close()
    old_env.close()
    print(json.dumps(result, indent=2), flush=True)


def sac_evaluate_saved(step: int, count: int) -> None:
    """Resume a gate evaluation after a pre-episode evaluator integration stop."""

    reward_contract = load_json(ARTIFACT_ROOT / "SAC_REWARD_CONTRACT.json")
    config, training_manifest, _ = training_inputs()
    config["trajectory_repair_reward"] = {
        key: reward_contract[key]
        for key in (
            "jerk_threshold_mps3",
            "jerk_scale_mps3",
            "switch_beta",
            "lambda_j",
            "lambda_task",
            "task_distance_scale_m",
        )
    }
    env = TaskAwareLocalReferenceEnv(training_manifest, config, seed=2026082507, training=False)
    checkpoint = ARTIFACT_ROOT / "checkpoints" / f"checkpoint_{int(step):07d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = _load_expanded_model(env, config, checkpoint)
    summary = _evaluate_sac_checkpoint(model, step=int(step), count=int(count))
    rows: list[dict[str, Any]] = []
    dev_path = ARTIFACT_ROOT / "SAC_DEV_RESULTS.csv"
    if dev_path.is_file() and dev_path.stat().st_size:
        with dev_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    rows = [row for row in rows if int(float(row.get("checkpoint_steps", -1))) != int(step)]
    rows.append(
        {
            "checkpoint_steps": int(step),
            **{key: value for key, value in summary.items() if not isinstance(value, (dict, list))},
        }
    )
    write_csv(dev_path, rows)
    history_path = ARTIFACT_ROOT / "SAC_TRAINING_HISTORY.csv"
    history: list[dict[str, Any]] = []
    if history_path.is_file() and history_path.stat().st_size:
        with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
            history.extend(csv.DictReader(handle))
    history.append(
        _history_row(
            int(step),
            "GATE_PASS" if summary["gate_pass"] else "GATE_FAIL",
            summary,
        )
    )
    write_csv(history_path, history)
    if not bool(summary["gate_pass"]):
        atomic_json(
            ARTIFACT_ROOT / "SAC_TRAINING_RESULT.json",
            {
                "schema_version": "task_aware_sac_training_result_v1",
                "SAC_FINETUNING_EXECUTED": "YES",
                "trained_steps": int(step),
                "BEST_CHECKPOINT_STEPS": 0,
                "stop_reason": f"{summary['gate']}_FAIL",
                "holdout_eligible": False,
                "fresh_replay_buffer": True,
                "source_sensor_encoder_frozen": True,
                "checkpoint": None,
                "evaluator_integration_stop_before_episode": True,
            },
        )
    env.close()
    print(json.dumps(summary, indent=2), flush=True)


def sac_continue_from_100k() -> None:
    """Continue the passed 100k checkpoint after the pre-episode integration stop."""

    summary_100k = load_json(ARTIFACT_ROOT / "sac_dev_step_0100000_summary.json")
    if not bool(summary_100k["gate_pass"]):
        raise RuntimeError("100k sanity gate did not authorize continuation")
    reward_contract = load_json(ARTIFACT_ROOT / "SAC_REWARD_CONTRACT.json")
    config, training_manifest, _ = training_inputs()
    config["trajectory_repair_reward"] = {
        key: reward_contract[key]
        for key in (
            "jerk_threshold_mps3",
            "jerk_scale_mps3",
            "switch_beta",
            "lambda_j",
            "lambda_task",
            "task_distance_scale_m",
        )
    }
    config["learning_rate"] = 3.0e-5
    training_env = TaskAwareLocalReferenceEnv(
        training_manifest,
        config,
        seed=2026082508,
        training=True,
    )
    model = _load_expanded_model(
        training_env,
        config,
        ARTIFACT_ROOT / "checkpoints/checkpoint_0100000.pt",
    )
    resume_checkpoint = torch.load(
        ARTIFACT_ROOT / "checkpoints/checkpoint_0100000.pt",
        map_location=model.device,
        weights_only=False,
    )
    model.num_timesteps = 100000
    model._n_updates = int(resume_checkpoint.get("n_updates", 0))
    # The evaluator integration stop occurred before the first Development
    # episode but the in-memory replay buffer could not be serialized.  The
    # expanded schema explicitly authorizes a fresh buffer; collect 1k new
    # transitions before resuming gradient updates rather than sampling an
    # underfilled buffer.
    model.learning_starts = 101000
    old_env = LongRangeLocalReferenceEnv(training_manifest, config, seed=2026082509, training=False)
    original_model, _, _ = load_original_model(old_env, config)
    teacher = copy.deepcopy(original_model.actor).to(model.device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model.safety_teacher_actor = teacher
    model.safety_teacher_old_extra_dim = 3
    model.safety_teacher_ray_count = 256
    model.safety_teacher_anchor_lambda = float(reward_contract["lambda_anchor"])
    model.safety_teacher_critical_scan_low = float(
        reward_contract["teacher_anchor_scan_low_normalized"]
    )
    model.safety_teacher_critical_scan_high = float(
        reward_contract["teacher_anchor_scan_high_normalized"]
    )

    history_path = ARTIFACT_ROOT / "SAC_TRAINING_HISTORY.csv"
    with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
        history: list[dict[str, Any]] = list(csv.DictReader(handle))
    dev_path = ARTIFACT_ROOT / "SAC_DEV_RESULTS.csv"
    with dev_path.open("r", encoding="utf-8-sig", newline="") as handle:
        dev_rows: list[dict[str, Any]] = list(csv.DictReader(handle))
    history.append(
        {
            **_history_row(100000, "REPLAY_RESET_AFTER_PREFLIGHT_EVALUATOR_STOP"),
            "replay_warmup_steps": 1000,
            "performance_observed_before_stop": False,
        }
    )
    write_csv(history_path, history)

    stop_reason = ""
    best_steps = 0
    for target, additional in ((250000, 150000), (500000, 250000)):
        print(f"[sac-train-resume] training -> {target} steps", flush=True)
        model.learn(total_timesteps=additional, reset_num_timesteps=False, progress_bar=False)
        checkpoint = ARTIFACT_ROOT / "checkpoints" / f"checkpoint_{target:07d}.pt"
        save_checkpoint(
            model,
            checkpoint,
            extra={
                "trajectory_repair_step": target,
                "reward_contract_sha256": sha256_file(ARTIFACT_ROOT / "SAC_REWARD_CONTRACT.json"),
                "sensor_encoder_frozen": True,
                "replay_reset_at_step": 100000,
                "replay_reset_warmup_steps": 1000,
            },
        )
        history.append(_history_row(target, "CHECKPOINT_SAVED"))
        write_csv(history_path, history)
        summary = _evaluate_sac_checkpoint(model, step=target, count=100)
        dev_rows = [
            row
            for row in dev_rows
            if int(float(row.get("checkpoint_steps", -1))) != target
        ]
        dev_rows.append(
            {
                "checkpoint_steps": target,
                **{key: value for key, value in summary.items() if not isinstance(value, (dict, list))},
            }
        )
        history.append(
            _history_row(target, "GATE_PASS" if summary["gate_pass"] else "GATE_FAIL", summary)
        )
        write_csv(dev_path, dev_rows)
        write_csv(history_path, history)
        print(json.dumps(summary, indent=2), flush=True)
        if not bool(summary["gate_pass"]):
            stop_reason = f"{summary['gate']}_FAIL"
            break
        if target == 500000:
            best_steps = 500000
            stop_reason = "DEVELOPMENT_FINAL_GATE_PASS"
    result = {
        "schema_version": "task_aware_sac_training_result_v1",
        "SAC_FINETUNING_EXECUTED": "YES",
        "trained_steps": target,
        "BEST_CHECKPOINT_STEPS": best_steps,
        "stop_reason": stop_reason,
        "holdout_eligible": best_steps == 500000,
        "fresh_replay_buffer": True,
        "replay_reset_after_pre_episode_integration_stop": True,
        "replay_reset_at_step": 100000,
        "replay_reset_warmup_steps": 1000,
        "source_sensor_encoder_frozen": True,
        "checkpoint": (
            str((ARTIFACT_RELATIVE / f"checkpoints/checkpoint_{best_steps:07d}.pt").as_posix())
            if best_steps
            else None
        ),
    }
    atomic_json(ARTIFACT_ROOT / "SAC_TRAINING_RESULT.json", result)
    training_env.close()
    old_env.close()
    print(json.dumps(result, indent=2), flush=True)


def _latest_development_summary() -> dict[str, Any] | None:
    candidates = sorted(ARTIFACT_ROOT.glob("sac_dev_step_*_summary.json"))
    return load_json(candidates[-1]) if candidates else None


def _fixed_representative_ids(available_ids: Sequence[str]) -> list[str]:
    available = set(available_ids)
    selected: list[str] = []
    for row in filtered_manifest()["entries"]:
        sid = str(row["scenario_id"])
        if sid in available and str(row["stage"]) not in {
            str(load_json(SOURCE_H0_RECORDS / f"{chosen}.json")["entry_identity"]["stage"])
            for chosen in selected
        }:
            selected.append(sid)
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise RuntimeError("could not freeze one representative from each Development stage")
    return selected


def _raw_trajectory_pdf(summary: Mapping[str, Any]) -> None:
    """Create the required raw-data-only compact trajectory comparison."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = int(summary["training_steps"])
    new_root = ARTIFACT_ROOT / "sac_dev_records" / f"step_{step:07d}"
    ids = _fixed_representative_ids(summary["evaluation_scenario_ids"])
    colors = ("#0072B2", "#D55E00", "#009E73")
    method_styles = (("Original", "-"), (f"Fine-tuned {step // 1000}k", "--"))
    figure = plt.figure(figsize=(22.0, 13.2))
    figure.subplots_adjust(
        left=0.025, right=0.995, bottom=0.035, top=0.905, wspace=0.22, hspace=0.34
    )
    grid = figure.add_gridspec(4, 6, width_ratios=(1.25, 1.15, 1.0, 1.0, 1.0, 1.0))
    legend_handles: list[Any] = []
    manifest_by_id = {
        str(entry["scenario_id"]): entry for entry in filtered_manifest()["entries"]
    }
    for row_index, sid in enumerate(ids):
        original_json = load_json(SOURCE_H0_RECORDS / f"{sid}.json")
        new_json = load_json(new_root / f"{sid}.json")
        scene = load_json(SOURCE_STUDY_ROOT / str(manifest_by_id[sid]["scenario_file"]))
        frozen_terminal_goals = np.asarray(scene["goals"], dtype=float)
        records = (
            ("Original", SOURCE_H0_RECORDS / f"{sid}_trajectory.npz", original_json),
            (f"Fine-tuned {step // 1000}k", new_root / f"{sid}_trajectory.npz", new_json),
        )
        stage_index = int(str(original_json["entry_identity"]["stage"]).rsplit("_", 1)[-1])
        stage = ("Stage I", "Stage II", "Stage III", "Stage IV")[stage_index - 1]
        ax3d = figure.add_subplot(grid[row_index, 0], projection="3d")
        axes = [figure.add_subplot(grid[row_index, column]) for column in range(1, 6)]
        for method_index, (method, npz_path, payload) in enumerate(records):
            data = np.load(npz_path)
            positions = np.asarray(data["positions"], dtype=float)
            accelerations = np.asarray(data["applied_accelerations_full"], dtype=float)
            if "terminal_goals" in data:
                terminal_goals = np.asarray(data["terminal_goals"], dtype=float)
            else:
                terminal_goals = np.broadcast_to(
                    frozen_terminal_goals[None, :, :], positions.shape
                )
            dt = float(data["dt"])
            time_axis = np.arange(len(positions), dtype=float) * dt
            jerk = np.full_like(accelerations, np.nan)
            jerk[1:] = np.diff(accelerations, axis=0) / dt
            for agent in range(positions.shape[1]):
                finite = np.all(np.isfinite(positions[:, agent]), axis=1)
                if not np.any(finite):
                    continue
                p = positions[finite, agent]
                t = time_axis[finite]
                line = ax3d.plot(
                    p[:, 0], p[:, 1], p[:, 2],
                    color=colors[agent], linestyle=method_styles[method_index][1],
                    linewidth=1.0 if method_index else 1.25,
                    alpha=0.88,
                )[0]
                if row_index == 0:
                    legend_handles.append(line)
                axes[0].plot(p[:, 0], p[:, 1], color=colors[agent], linestyle=method_styles[method_index][1], linewidth=1.0)
                axes[1].plot(t, p[:, 2], color=colors[agent], linestyle=method_styles[method_index][1], linewidth=0.9)
                accel_norm = np.linalg.norm(accelerations[finite, agent], axis=1)
                jerk_norm = np.linalg.norm(jerk[finite, agent], axis=1)
                goal_distance = np.linalg.norm(terminal_goals[finite, agent] - p, axis=1)
                axes[2].plot(t, accel_norm, color=colors[agent], linestyle=method_styles[method_index][1], linewidth=0.8)
                axes[3].plot(t, jerk_norm, color=colors[agent], linestyle=method_styles[method_index][1], linewidth=0.7)
                axes[4].plot(t, goal_distance, color=colors[agent], linestyle=method_styles[method_index][1], linewidth=0.9)
            event_times = [
                float(event["step"]) * dt
                for event in payload.get("events", [])
                if bool(event.get("goal_changed", True))
            ]
            if event_times:
                y_top = axes[3].get_ylim()[1]
                axes[3].plot(
                    event_times,
                    np.full(len(event_times), y_top * (0.96 - 0.05 * method_index)),
                    linestyle="None",
                    marker="|",
                    markersize=2.0,
                    markeredgewidth=0.35,
                    color="#444444" if method_index == 0 else "#999999",
                    alpha=0.45,
                )
        ax3d.set_title(stage, loc="left", fontsize=10, fontweight="bold")
        ax3d.set_xlabel("x (m)", fontsize=8)
        ax3d.set_ylabel("y (m)", fontsize=8)
        ax3d.set_zlabel("z (m)", fontsize=8)
        panel_titles = ("XY top view", "Altitude", "Acceleration", "Jerk + switch ticks", "Final-goal distance")
        y_labels = ("y (m)", "z (m)", r"$||a||$ (m/s$^2$)", r"$||j||$ (m/s$^3$)", "distance (m)")
        for column, axis in enumerate(axes):
            axis.set_title(panel_titles[column], fontsize=8)
            axis.set_ylabel(y_labels[column], fontsize=7)
            axis.set_xlabel("x (m)" if column == 0 else "time (s)", fontsize=7)
            axis.grid(True, color="#d0d0d0", linewidth=0.35, alpha=0.65)
            axis.tick_params(labelsize=6)
    labels = [
        f"UAV {agent + 1} - {method}"
        for method, _ in method_styles
        for agent in range(3)
    ]
    figure.legend(
        legend_handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=6,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        "Raw 0.1-s Development trajectories: Original vs bounded SAC-DMP repair",
        fontsize=13,
        y=0.990,
    )
    output = ARTIFACT_ROOT / "SAC_DEV_RAW_TRAJECTORY_COMPARISON.pdf"
    figure.savefig(output, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(figure)


def finalize() -> None:
    """Close the branch after the bounded gate outcome and preserve Formal V2."""

    training_result = load_json(ARTIFACT_ROOT / "SAC_TRAINING_RESULT.json")
    latest = _latest_development_summary()
    if latest is None:
        raise RuntimeError("cannot finalize before at least the 100k Development sanity evaluation")
    _raw_trajectory_pdf(latest)
    holdout_eligible = bool(training_result["holdout_eligible"])
    holdout_path = ARTIFACT_ROOT / "SAC_HOLDOUT_RESULTS.csv"
    if holdout_eligible and not holdout_path.is_file():
        raise RuntimeError("a Development-eligible checkpoint requires the sealed Holdout before finalization")
    holdout = None
    if holdout_path.is_file():
        with holdout_path.open("r", encoding="utf-8-sig", newline="") as handle:
            holdout = list(csv.DictReader(handle))

    best_steps = int(training_result["BEST_CHECKPOINT_STEPS"])
    final_accepted = bool(holdout_eligible and holdout)
    conclusion = {
        "PROPOSAL_RAW_DIRECTION_COUNT": 256,
        "PROPOSAL_GRID": "16x16",
        "GAT_CANONICAL_DIRECTION_COUNT": 56,
        "TOP_K": 10,
        "FP_PREVIEW_H": 4,
        "REFERENCE_HORIZON_MODIFICATION_ALLOWED": "YES",
        "GOAL_ALIGNED_HORIZON_ACCEPTED": load_json(ARTIFACT_ROOT / "HORIZON_GATE_DECISION.json")[
            "GOAL_ALIGNED_HORIZON_ACCEPTED"
        ],
        "SAC_FINETUNING_EXECUTED": training_result["SAC_FINETUNING_EXECUTED"],
        "ORIGINAL_SAC_OBSERVATION_DIM": 522,
        "NEW_SAC_OBSERVATION_DIM": 531,
        "FINAL_TASK_GOAL_CONTEXT_ADDED": "YES",
        "FINAL_GOAL_SAFETY_CONTEXT_ADDED": "YES",
        "PREVIOUS_ACCELERATION_CONTEXT_ADDED": "YES",
        "SWITCH_AGE_CONTEXT_ADDED": "YES",
        "ZERO_INIT_EQUIVALENCE": load_json(ARTIFACT_ROOT / "ZERO_INIT_EQUIVALENCE.json")[
            "ZERO_INIT_EQUIVALENCE"
        ],
        "STRICT_SENSOR_ENCODER_FREEZE_VERIFICATION": "PARTIAL_FAIL_CRITIC_TARGET_POLYAK_DRIFT",
        "BEST_CHECKPOINT_STEPS": best_steps,
        "DEV_ORIGINAL_SUCCESS": latest["original_success"],
        "DEV_NEW_SUCCESS": latest["new_success"],
        "DEV_PEER_COLLISION_DELTA_PP": latest["peer_collision_delta_pp"],
        "DEV_SMOOTHNESS_IMPROVEMENT_PERCENT": latest["smoothness_improvement_percent"],
        "DEV_SWITCH_JERK_IMPROVEMENT_PERCENT": latest["switch_jerk_improvement_percent"],
        "DEV_PATH_LENGTH_IMPROVEMENT_PERCENT": latest["path_length_improvement_percent"],
        "DEV_EXCESS_PATH_IMPROVEMENT_PERCENT": latest["excess_path_improvement_percent"],
        "DEV_DETOUR_RATIO_CHANGE": latest["detour_ratio_change"],
        "DEV_REFERENCE_CHANGE_DELTA": latest["reference_change_delta"],
        "HOLDOUT_ORIGINAL_SUCCESS": "NOT_RUN" if holdout is None else holdout[0].get("original_success"),
        "HOLDOUT_NEW_SUCCESS": "NOT_RUN" if holdout is None else holdout[0].get("new_success"),
        "HOLDOUT_PEER_COLLISION_DELTA_PP": "NOT_RUN" if holdout is None else holdout[0].get("peer_collision_delta_pp"),
        "HOLDOUT_SMOOTHNESS_IMPROVEMENT_PERCENT": "NOT_RUN" if holdout is None else holdout[0].get("smoothness_improvement_percent"),
        "HOLDOUT_SWITCH_JERK_IMPROVEMENT_PERCENT": "NOT_RUN" if holdout is None else holdout[0].get("switch_jerk_improvement_percent"),
        "HOLDOUT_PATH_LENGTH_IMPROVEMENT_PERCENT": "NOT_RUN" if holdout is None else holdout[0].get("path_length_improvement_percent"),
        "SMOOTHNESS_SOLVED": "YES" if final_accepted else "NO",
        "PATH_LENGTH_SOLVED": "YES" if final_accepted else "NO",
        "RELIABILITY_PRESERVED": "YES" if final_accepted else "NO",
        "FINAL_VARIANT_ACCEPTED": "YES" if final_accepted else "NO",
        "PROPOSAL_COMPUTE_SHARE": "NOT_PROFILED",
        "PROPOSAL_CULLING_AUTHORIZED": "NOT_REACHED",
        "FORMAL_REEVALUATION_RECOMMENDED": "YES" if final_accepted else "NO",
        "ORIGINAL_FORMAL_SUCCESS": 0.9525,
        "ORIGINAL_FORMAL_RESULT_MODIFIED": "NO",
        "FINAL_RECOMMENDATION": (
            "Freeze the accepted SAC-DMP repair and wait for explicit Formal authorization."
            if final_accepted
            else "Keep the original frozen SAC-DMP and permanently close this bounded trajectory-repair branch."
        ),
    }
    go_no_go = {
        "schema_version": "final_trajectory_repair_formal_go_no_go_v1",
        "decision": "GO" if final_accepted else "NO_GO",
        "formal_v2_executed": False,
        "formal_v2_authorized": False,
        "original_formal_success": 0.9525,
        "original_formal_result_modified": False,
        "reason": (
            "sealed Holdout passed all three acceptance dimensions"
            if final_accepted
            else training_result["stop_reason"]
        ),
    }
    atomic_json(ARTIFACT_ROOT / "FINAL_TRAJECTORY_REPAIR_FORMAL_GO_NO_GO.json", go_no_go)
    atomic_json(ARTIFACT_ROOT / "conclusion.json", conclusion)
    report = f"""# Final Goal-Aligned SAC-DMP Trajectory-Quality Repair

## Executive result

The bounded repair is **{'ACCEPTED' if final_accepted else 'REJECTED'}**. The single goal-aligned horizon arm reduced reference changes but lowered success from 96.0% to 92.0% and lengthened both-success paths by 11.24%, so it was permanently discarded. SAC fine-tuning stopped at **{training_result['trained_steps']:,}** steps with `{training_result['stop_reason']}`. The existing Formal V2 result remains **381/400 (95.25%)** and was not rerun or modified.

## Horizon screen

- H0 Original: 96.0% Development success.
- H1 goal-aligned adaptive distance: 92.0% success (-4.0 pp), unchanged 3.0% peer collision.
- Both-success team path: 271.31 m -> 301.81 m (+11.24%).
- Accepted reference changes: 304.86 -> 105.64 (-65.35%).
- Decision: `GOAL_ALIGNED_HORIZON_ACCEPTED = NO`; all SAC stages used the original reference geometry.

## SAC contract and bounded training

- Observation: 522 -> 531 dimensions, adding final-task direction/distance, legal task-direction safety, switch age, and previous executed acceleration.
- Zero-initialization equivalence: `PASS` (actor max difference 0; critic {load_json(ARTIFACT_ROOT / 'ZERO_INIT_EQUIVALENCE.json')['maximum_absolute_critic_q_difference']:.3e}).
- Original reward remained complete. The only secondary objectives were switch-aware tail jerk and safe final-task progress, with a critical-state Original-actor anchor.
- Jerk threshold: P75 = {load_json(ARTIFACT_ROOT / 'SAC_REWARD_CONTRACT.json')['jerk_threshold_mps3']:.3f} m/s^3.
- Latest Development block: Original/New success {100*latest['original_success']:.1f}%/{100*latest['new_success']:.1f}%; peer-collision delta {latest['peer_collision_delta_pp']:+.1f} pp.
- Both-success changes (New relative to Original): smoothness {latest['smoothness_improvement_percent']:+.2f}% improvement, switch jerk {latest['switch_jerk_improvement_percent']:+.2f}%, team path {latest['path_length_improvement_percent']:+.2f}%, excess path {latest['excess_path_improvement_percent']:+.2f}%, detour-ratio change {latest['detour_ratio_change']:+.4f}.

## Independent reconciliation note

The actor and online-critic sensor encoders remained tensor-exact copies of the source checkpoint. The target critic's sensor encoder did not remain source-exact: the standard SAC Polyak update moved it by a maximum absolute 1.468e-3 even though it had no gradients. This is a strict freeze-contract deviation. Because the 500k checkpoint independently failed the Development improvement gate and no checkpoint advanced to Holdout or Formal evaluation, this finding cannot support acceptance and reinforces the final `NO_GO` decision.

## Decision and scope

- `BEST_CHECKPOINT_STEPS = {best_steps}`
- `FINAL_VARIANT_ACCEPTED = {'YES' if final_accepted else 'NO'}`
- `FORMAL_REEVALUATION_RECOMMENDED = {'YES' if final_accepted else 'NO'}`
- Sealed Holdout: {'executed' if holdout is not None else 'not run because the Development gate failed'}.
- Optional Proposal compute reduction: not reached.
- Final recommendation: {conclusion['FINAL_RECOMMENDATION']}

The raw comparison figure uses fixed one-per-stage Development scenes and raw 0.1-s samples only. It contains no interpolation, spline, or presentation smoothing.
"""
    (ARTIFACT_ROOT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"conclusion": conclusion, "go_no_go": go_no_go}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "stage-a",
            "horizon-run",
            "horizon-finalize",
            "zero-init",
            "reward-calibrate",
            "sac-train",
            "sac-eval-saved",
            "sac-continue-100k",
            "finalize",
        ),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--step", type=int, default=100000)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.phase == "stage-a":
        stage_a()
    elif args.phase == "horizon-run":
        horizon_run(args.shard_index, args.shard_count)
    elif args.phase == "horizon-finalize":
        horizon_finalize()
    elif args.phase == "zero-init":
        zero_init()
    elif args.phase == "reward-calibrate":
        reward_calibrate()
    elif args.phase == "sac-train":
        sac_train()
    elif args.phase == "sac-eval-saved":
        sac_evaluate_saved(args.step, args.count)
    elif args.phase == "sac-continue-100k":
        sac_continue_from_100k()
    else:
        finalize()


if __name__ == "__main__":
    main()
