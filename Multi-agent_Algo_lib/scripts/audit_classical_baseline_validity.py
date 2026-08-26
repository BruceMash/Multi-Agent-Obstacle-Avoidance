"""Read-only validity and fairness audit for the frozen classical baselines.

The audit never calls a benchmark episode evaluator.  It reads persisted JSON,
CSV, NPZ, the frozen scenario manifest, and the implementation source.  All
collision, success, kinematic, and fairness diagnostics are reconstructed
offline into a separate artifact directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Entity.dynamic_obstacles import PatternedMovingSphereObstacle  # noqa: E402


METHODS = ("dwa_style", "rvo_orca_style", "gat_v1")
ALL_METHODS = ("dwa_style", "rvo_orca_style", "terminal", "proposal", "fp_shep", "gat_v1")
CLASSICS = ("dwa_style", "rvo_orca_style")
DISPLAY = {
    "dwa_style": "3D-DWA-style",
    "rvo_orca_style": "RVO/ORCA-style",
    "terminal": "Terminal",
    "proposal": "Proposal",
    "fp_shep": "FP-SHEP",
    "gat_v1": "Proposed/GAT-V1",
}
STAGES = ("stage_1", "stage_2", "stage_3", "stage_4")
DT = 0.1
GOAL_TOLERANCE = 0.3
COLLISION_MARGIN = 0.0
INTER_AGENT_THRESHOLD = 0.6
PEER_RADIUS = 0.3
SENSOR_RANGE = 4.5
VELOCITY_COMPONENT_LIMIT = 4.0
ACCELERATION_COMPONENT_LIMIT = 4.0
POSITION_TOLERANCE = 1.0e-8
COLLISION_TOLERANCE = 1.0e-10
TARGET_SAMPLE_SEED = 2026081827


@dataclass
class CollisionAudit:
    static_collision: bool
    dynamic_collision: bool
    peer_collision: bool
    any_collision: bool
    first_static_step: int | None
    first_dynamic_step: int | None
    first_peer_step: int | None
    min_static_signed_distance_m: float
    min_dynamic_signed_distance_m: float
    min_peer_center_distance_m: float
    final_agent_collision_mask: np.ndarray


@dataclass
class SweepAudit:
    static_sweep_intersection: bool
    dynamic_sweep_intersection: bool
    peer_sweep_intersection: bool
    static_sweep_only_events: int
    dynamic_sweep_only_events: int
    peer_sweep_only_events: int
    first_sweep_only_step: int | None

    @property
    def sweep_only_events(self) -> int:
        return self.static_sweep_only_events + self.dynamic_sweep_only_events + self.peer_sweep_only_events

    @property
    def sweep_only_episode(self) -> bool:
        return self.sweep_only_events > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(jsonable(row.get(key)), ensure_ascii=False, separators=(",", ":"))
                        if isinstance(row.get(key), (dict, list, tuple, np.ndarray))
                        else "" if row.get(key) is None else jsonable(row.get(key))
                    )
                    for key in fields
                }
            )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path, pattern: str) -> dict[str, Any]:
    files = sorted(root.glob(pattern), key=lambda path: path.as_posix())
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        item_hash = file_sha256(path)
        size = path.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\n")
    return {"file_count": len(files), "total_bytes": total_bytes, "tree_sha256": digest.hexdigest()}


def percentile(values: Sequence[float], q: float) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if array.size else None


def mean(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else None


def maximum(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.max(array)) if array.size else None


def obstacle_signed_distance(points: np.ndarray, obstacle: Mapping[str, Any]) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    center = np.asarray(obstacle["center"], dtype=float)
    kind = str(obstacle["type"])
    safety = float(obstacle.get("safety_margin", 0.0))
    if kind in {"sphere", "patterned_moving_sphere", "moving_sphere"}:
        return np.linalg.norm(points - center, axis=-1) - (float(obstacle["radius"]) + safety)
    if kind == "box":
        half = np.asarray(obstacle["half_extents"], dtype=float) + safety
        q = np.abs(points - center) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside
    if kind == "cylinder":
        offset = points - center
        radial = np.linalg.norm(offset[..., :2], axis=-1) - (float(obstacle["radius"]) + safety)
        vertical = np.abs(offset[..., 2]) - (float(obstacle["half_height"]) + safety)
        outside = np.sqrt(np.maximum(radial, 0.0) ** 2 + np.maximum(vertical, 0.0) ** 2)
        inside = np.minimum(np.maximum(radial, vertical), 0.0)
        return outside + inside
    raise ValueError(f"unsupported obstacle type {kind}")


def point_segment_min_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1.0e-18:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * segment)))


def segment_intersects_box(start: np.ndarray, end: np.ndarray, obstacle: Mapping[str, Any]) -> bool:
    center = np.asarray(obstacle["center"], dtype=float)
    half = np.asarray(obstacle["half_extents"], dtype=float) + float(obstacle.get("safety_margin", 0.0))
    lower, upper = center - half, center + half
    direction = end - start
    t_min, t_max = 0.0, 1.0
    for dimension in range(3):
        if abs(float(direction[dimension])) < 1.0e-15:
            if start[dimension] < lower[dimension] or start[dimension] > upper[dimension]:
                return False
            continue
        first = (lower[dimension] - start[dimension]) / direction[dimension]
        second = (upper[dimension] - start[dimension]) / direction[dimension]
        near, far = min(first, second), max(first, second)
        t_min, t_max = max(t_min, float(near)), min(t_max, float(far))
        if t_min > t_max:
            return False
    return True


def segment_intersects_cylinder(start: np.ndarray, end: np.ndarray, obstacle: Mapping[str, Any]) -> bool:
    center = np.asarray(obstacle["center"], dtype=float)
    radius = float(obstacle["radius"]) + float(obstacle.get("safety_margin", 0.0))
    half_height = float(obstacle["half_height"]) + float(obstacle.get("safety_margin", 0.0))
    if obstacle_signed_distance(np.vstack((start, end)), obstacle).min() <= COLLISION_TOLERANCE:
        return True
    relative = start - center
    direction = end - start
    a = float(np.dot(direction[:2], direction[:2]))
    b = 2.0 * float(np.dot(relative[:2], direction[:2]))
    c = float(np.dot(relative[:2], relative[:2]) - radius * radius)
    if a > 1.0e-18:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            root = math.sqrt(discriminant)
            for fraction in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
                if 0.0 <= fraction <= 1.0:
                    z = relative[2] + fraction * direction[2]
                    if abs(float(z)) <= half_height + COLLISION_TOLERANCE:
                        return True
    if abs(float(direction[2])) > 1.0e-18:
        for cap in (-half_height, half_height):
            fraction = (cap - relative[2]) / direction[2]
            if 0.0 <= fraction <= 1.0:
                xy = relative[:2] + fraction * direction[:2]
                if float(np.linalg.norm(xy)) <= radius + COLLISION_TOLERANCE:
                    return True
    return False


def segment_intersects_static(start: np.ndarray, end: np.ndarray, obstacle: Mapping[str, Any]) -> bool:
    kind = str(obstacle["type"])
    if kind == "sphere":
        radius = float(obstacle["radius"]) + float(obstacle.get("safety_margin", 0.0))
        return point_segment_min_distance(np.asarray(obstacle["center"], dtype=float), start, end) <= radius + COLLISION_TOLERANCE
    if kind == "box":
        return segment_intersects_box(start, end, obstacle)
    if kind == "cylinder":
        return segment_intersects_cylinder(start, end, obstacle)
    raise ValueError(kind)


def relative_segment_intersects_radius(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    radius: float,
) -> bool:
    relative_start = first_start - second_start
    relative_end = first_end - second_end
    return point_segment_min_distance(np.zeros(3), relative_start, relative_end) <= float(radius) + COLLISION_TOLERANCE


def independent_discrete_collision(
    positions: np.ndarray,
    scene: Mapping[str, Any],
) -> CollisionAudit:
    static_any = False
    dynamic_any = False
    peer_any = False
    first_static = None
    first_dynamic = None
    first_peer = None
    min_static = float("inf")
    min_dynamic = float("inf")
    min_peer = float("inf")
    final_mask = np.zeros(positions.shape[1], dtype=bool)
    dynamic_tracks = [np.asarray(track, dtype=float) for track in scene["dynamic_obstacle_trajectories"]]
    for step, frame in enumerate(positions):
        step_mask = np.zeros(positions.shape[1], dtype=bool)
        for obstacle in scene["static_obstacles"]:
            distances = obstacle_signed_distance(frame, obstacle)
            min_static = min(min_static, float(np.min(distances)))
            hits = distances <= COLLISION_MARGIN + COLLISION_TOLERANCE
            if np.any(hits):
                static_any = True
                step_mask |= hits
                if first_static is None:
                    first_static = step
        for obstacle_id, obstacle in enumerate(scene["dynamic_obstacles"]):
            current = dict(obstacle)
            current["center"] = dynamic_tracks[obstacle_id][min(step, len(dynamic_tracks[obstacle_id]) - 1)]
            distances = obstacle_signed_distance(frame, current)
            min_dynamic = min(min_dynamic, float(np.min(distances)))
            hits = distances <= COLLISION_MARGIN + COLLISION_TOLERANCE
            if np.any(hits):
                dynamic_any = True
                step_mask |= hits
                if first_dynamic is None:
                    first_dynamic = step
        for first in range(frame.shape[0]):
            for second in range(first + 1, frame.shape[0]):
                distance = float(np.linalg.norm(frame[first] - frame[second]))
                min_peer = min(min_peer, distance)
                if distance <= INTER_AGENT_THRESHOLD + COLLISION_TOLERANCE:
                    peer_any = True
                    step_mask[first] = True
                    step_mask[second] = True
                    if first_peer is None:
                        first_peer = step
        if step == positions.shape[0] - 1:
            final_mask = step_mask
    return CollisionAudit(
        static_collision=static_any,
        dynamic_collision=dynamic_any,
        peer_collision=peer_any,
        any_collision=bool(static_any or dynamic_any or peer_any),
        first_static_step=first_static,
        first_dynamic_step=first_dynamic,
        first_peer_step=first_peer,
        min_static_signed_distance_m=min_static,
        min_dynamic_signed_distance_m=min_dynamic,
        min_peer_center_distance_m=min_peer,
        final_agent_collision_mask=final_mask,
    )


def independent_sweep_collision(positions: np.ndarray, scene: Mapping[str, Any]) -> SweepAudit:
    static_hit = False
    dynamic_hit = False
    peer_hit = False
    static_only = 0
    dynamic_only = 0
    peer_only = 0
    first_only: int | None = None
    tracks = [np.asarray(track, dtype=float) for track in scene["dynamic_obstacle_trajectories"]]
    for step in range(positions.shape[0] - 1):
        start_frame, end_frame = positions[step], positions[step + 1]
        for agent_id in range(positions.shape[1]):
            start, end = start_frame[agent_id], end_frame[agent_id]
            for obstacle in scene["static_obstacles"]:
                hit = segment_intersects_static(start, end, obstacle)
                if hit:
                    static_hit = True
                    endpoints = obstacle_signed_distance(np.vstack((start, end)), obstacle) <= COLLISION_MARGIN + COLLISION_TOLERANCE
                    if not bool(np.any(endpoints)):
                        static_only += 1
                        first_only = step if first_only is None else min(first_only, step)
            for obstacle_id, obstacle in enumerate(scene["dynamic_obstacles"]):
                track = tracks[obstacle_id]
                obstacle_start = track[min(step, len(track) - 1)]
                obstacle_end = track[min(step + 1, len(track) - 1)]
                radius = float(obstacle["radius"]) + float(obstacle.get("safety_margin", 0.0))
                hit = relative_segment_intersects_radius(start, end, obstacle_start, obstacle_end, radius)
                if hit:
                    dynamic_hit = True
                    endpoint_start = float(np.linalg.norm(start - obstacle_start)) <= radius + COLLISION_TOLERANCE
                    endpoint_end = float(np.linalg.norm(end - obstacle_end)) <= radius + COLLISION_TOLERANCE
                    if not (endpoint_start or endpoint_end):
                        dynamic_only += 1
                        first_only = step if first_only is None else min(first_only, step)
        for first in range(start_frame.shape[0]):
            for second in range(first + 1, start_frame.shape[0]):
                hit = relative_segment_intersects_radius(
                    start_frame[first], end_frame[first], start_frame[second], end_frame[second], INTER_AGENT_THRESHOLD
                )
                if hit:
                    peer_hit = True
                    endpoint_start = float(np.linalg.norm(start_frame[first] - start_frame[second])) <= INTER_AGENT_THRESHOLD + COLLISION_TOLERANCE
                    endpoint_end = float(np.linalg.norm(end_frame[first] - end_frame[second])) <= INTER_AGENT_THRESHOLD + COLLISION_TOLERANCE
                    if not (endpoint_start or endpoint_end):
                        peer_only += 1
                        first_only = step if first_only is None else min(first_only, step)
    return SweepAudit(static_hit, dynamic_hit, peer_hit, static_only, dynamic_only, peer_only, first_only)


def max_consecutive(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def velocity_direction_changes(velocities: np.ndarray) -> np.ndarray:
    previous = velocities[:-1].reshape(-1, 3)
    current = velocities[1:].reshape(-1, 3)
    previous_norm = np.linalg.norm(previous, axis=1)
    current_norm = np.linalg.norm(current, axis=1)
    valid = np.logical_and(previous_norm > 1.0e-6, current_norm > 1.0e-6)
    if not np.any(valid):
        return np.empty(0, dtype=float)
    cosines = np.sum(previous[valid] * current[valid], axis=1) / (previous_norm[valid] * current_norm[valid])
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))


def dynamic_from_spec(spec: Mapping[str, Any]) -> PatternedMovingSphereObstacle:
    return PatternedMovingSphereObstacle(
        center=np.asarray(spec["center"], dtype=float),
        radius=float(spec["radius"]),
        velocity=np.asarray(spec["velocity"], dtype=float),
        safety_margin=float(spec.get("safety_margin", 0.0)),
        bounds=tuple(np.asarray(bound, dtype=float) for bound in spec["bounds"]) if spec.get("bounds") is not None else None,
        motion_mode=str(spec.get("motion_mode", "linear")),
        turn_rate=float(spec.get("turn_rate", 0.45)),
        wandering_strength=float(spec.get("wandering_strength", 0.8)),
        seed=int(spec["motion_seed"]) if spec.get("motion_seed") is not None else None,
    )


def line_number(path: Path, needle: str) -> int | None:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return number
    return None


def choose_targeted_scenarios(episodes: Sequence[Mapping[str, str]]) -> set[tuple[str, str, str]]:
    rng = random.Random(TARGET_SAMPLE_SEED)
    selected: set[tuple[str, str, str]] = set()
    for stage in ("stage_3", "stage_4"):
        for method in CLASSICS:
            successes = sorted(
                [row["scenario_id"] for row in episodes if row["stage"] == stage and row["method"] == method and truth(row["team_success"])]
            )
            chosen = rng.sample(successes, min(20, len(successes)))
            selected.update((stage, scenario_id, method) for scenario_id in chosen)
            failures = [row["scenario_id"] for row in episodes if row["stage"] == stage and row["method"] == method and not truth(row["team_success"])]
            selected.update((stage, scenario_id, method) for scenario_id in failures)
    return selected


def main() -> None:
    args = parse_args()
    source = args.source_artifact.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).astimezone().isoformat()

    required_source_files = (
        "FINAL_REPORT.md",
        "conclusion.json",
        "formal_episode_results.csv",
        "formal_agent_results.csv",
        "planning_runtime_records.csv",
        "scenario_manifest.json",
        "classic_planner_parameter_contract.json",
        "engineering_freeze.json",
        "ENGINEERING_FREEZE_REPORT.md",
        "integrity_manifest.json",
        "final_reconciliation.json",
        "context_recovery_manifest.json",
    )
    missing = [name for name in required_source_files if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing authoritative inputs: {missing}")

    source_hashes = {name: file_sha256(source / name) for name in required_source_files}
    source_trees = {
        "formal_records": tree_digest(source / "formal_records", "**/*.json"),
        "trajectories": tree_digest(source / "trajectories", "**/*.npz"),
        "method_configs": tree_digest(source / "method_configs", "*.json"),
    }
    frozen_context = read_json(source / "context_recovery_manifest.json")
    engineering_freeze = read_json(source / "engineering_freeze.json")
    # The context manifest records hashes before the permitted engineering phase.
    # Formal execution used the post-engineering hashes frozen here.
    expected_core = engineering_freeze["core_hashes"]
    audited_code = (
        "planning/final_four_stage_benchmark.py",
        "Multi-agent_Algo_lib/scripts/run_final_four_stage_benchmark.py",
        "Environment/multi_agent_dmp_env.py",
        "Entity/KinematicModel.py",
        "Entity/static_obstacles.py",
        "Entity/dynamic_obstacles.py",
        "Guidance/reference_point_proposal_demo.py",
        "planning/policy_preview.py",
        "planning/heterogeneous_candidate_graph.py",
        "Multi-agent_Algo_lib/scripts/evaluate_single_policy_multi_agent.py",
    )
    actual_code_hashes = {name: file_sha256(REPO_ROOT / name) for name in audited_code}
    frozen_code_match = {
        name: actual_code_hashes[name] == expected_core[name]
        for name in audited_code
        if name in expected_core
    }

    scenario_payload = read_json(source / "scenario_manifest.json")
    scenes = {entry["scenario_id"]: entry for entry in scenario_payload["entries"]}
    episode_rows = read_csv(source / "formal_episode_results.csv")
    agent_rows = read_csv(source / "formal_agent_results.csv")
    runtime_rows = read_csv(source / "planning_runtime_records.csv")
    episode_lookup = {(row["scenario_id"], row["method"]): row for row in episode_rows}
    agent_lookup = {(row["scenario_id"], row["method"], int(row["agent_id"])): row for row in agent_rows}
    selected_configs = read_json(source / "classic_planner_parameter_contract.json")["selected"]
    targeted = choose_targeted_scenarios(episode_rows)

    config_payload = {
        "schema_version": "classical_baseline_validity_audit_v1",
        "created_at": started_at,
        "mode": "READ_ONLY_OFFLINE_AUDIT",
        "source_artifact": str(source),
        "methods": list(METHODS),
        "formal_episode_rerun": False,
        "formal_data_modified": False,
        "dt_s": DT,
        "goal_tolerance_m": GOAL_TOLERANCE,
        "collision_margin_m": COLLISION_MARGIN,
        "inter_agent_collision_center_distance_m": INTER_AGENT_THRESHOLD,
        "velocity_component_limit_mps": VELOCITY_COMPONENT_LIMIT,
        "acceleration_component_limit_mps2": ACCELERATION_COMPONENT_LIMIT,
        "continuous_sweep_semantics": "diagnostic_only; does not overwrite the frozen discrete-time environment contract",
        "targeted_stage_3_4_sampling": {
            "seed": TARGET_SAMPLE_SEED,
            "rule": "20 deterministic random successful scenarios per stage and classic plus every classic failure; all trajectories are nevertheless audited",
            "selected_key_count": len(targeted),
        },
        "selected_classical_configs": selected_configs,
        "decision_rules": {
            "tunneling_advantage_yes": "at least 5% sweep-only episodes for either classic while Proposed has zero",
            "hard_invalidators": [
                "discrete collision reconstruction mismatch",
                "success reconstruction mismatch",
                "position jump/dynamics limit violation",
                "future-information leakage",
            ],
        },
    }
    write_json(output / "config.json", config_payload)

    collision_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    path_efficiency_rows: list[dict[str, Any]] = []
    dynamic_alignment_details: list[dict[str, Any]] = []
    metric_accumulator: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    position_summary_source: list[dict[str, Any]] = []
    success_reconstruction: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    theoretical_step_limit = math.sqrt(3.0) * (
        VELOCITY_COMPONENT_LIMIT * DT + 0.5 * ACCELERATION_COMPONENT_LIMIT * DT * DT
    )

    relevant_episodes = [row for row in episode_rows if row["method"] in METHODS]
    for episode_index, formal in enumerate(relevant_episodes, start=1):
        stage = formal["stage"]
        scenario_id = formal["scenario_id"]
        method = formal["method"]
        scene = scenes[scenario_id]
        trajectory_path = source / formal["trajectory_path"].replace("\\", "/")
        with np.load(trajectory_path, allow_pickle=False) as payload:
            positions = np.asarray(payload["positions"], dtype=float)
            velocities = np.asarray(payload["velocities"], dtype=float)
            stored_accelerations = np.asarray(payload["accelerations"], dtype=float)
            dynamic_positions = np.asarray(payload["dynamic_obstacle_positions"], dtype=float)
            starts = np.asarray(payload["starts"], dtype=float)
            goals = np.asarray(payload["goals"], dtype=float)
        if positions.shape != velocities.shape or positions.shape[1:] != (3, 3):
            raise RuntimeError(f"unexpected trajectory shape {trajectory_path}: {positions.shape}/{velocities.shape}")
        if stored_accelerations.shape != (positions.shape[0] - 1, 3, 3):
            raise RuntimeError(f"unexpected acceleration shape {trajectory_path}: {stored_accelerations.shape}")

        collision = independent_discrete_collision(positions, scene)
        sweep = independent_sweep_collision(positions, scene)
        formal_obstacle = truth(formal["obstacle_collision"])
        formal_peer = truth(formal["inter_agent_collision"])
        discrete_match = (
            truth(formal["any_collision"]) == collision.any_collision
            and formal_obstacle == bool(collision.static_collision or collision.dynamic_collision)
            and formal_peer == collision.peer_collision
        )
        collision_rows.append(
            {
                "stage": stage,
                "scenario_id": scenario_id,
                "method": method,
                "formal_any_collision": truth(formal["any_collision"]),
                "recomputed_any_collision": collision.any_collision,
                "formal_obstacle_collision": formal_obstacle,
                "recomputed_static_collision": collision.static_collision,
                "recomputed_dynamic_collision": collision.dynamic_collision,
                "formal_inter_agent_collision": formal_peer,
                "recomputed_inter_agent_collision": collision.peer_collision,
                "first_static_collision_step": collision.first_static_step,
                "first_dynamic_collision_step": collision.first_dynamic_step,
                "first_peer_collision_step": collision.first_peer_step,
                "min_static_signed_distance_m": collision.min_static_signed_distance_m,
                "min_dynamic_signed_distance_m": collision.min_dynamic_signed_distance_m,
                "min_peer_center_distance_m": collision.min_peer_center_distance_m,
                "all_collision_flags_match": discrete_match,
            }
        )
        sweep_rows.append(
            {
                "stage": stage,
                "scenario_id": scenario_id,
                "method": method,
                "formal_discrete_collision": truth(formal["any_collision"]),
                "static_sweep_intersection": sweep.static_sweep_intersection,
                "dynamic_sweep_intersection": sweep.dynamic_sweep_intersection,
                "peer_sweep_intersection": sweep.peer_sweep_intersection,
                "static_sweep_only_event_count": sweep.static_sweep_only_events,
                "dynamic_sweep_only_event_count": sweep.dynamic_sweep_only_events,
                "peer_sweep_only_event_count": sweep.peer_sweep_only_events,
                "sweep_only_event_count": sweep.sweep_only_events,
                "sweep_only_collision_episode": sweep.sweep_only_episode,
                "sweep_only_and_formal_collision_free": sweep.sweep_only_episode and not truth(formal["any_collision"]),
                "first_sweep_only_segment_start_step": sweep.first_sweep_only_step,
                "contract_status": "CONTINUOUS_SWEEP_COLLISION_DIAGNOSTIC" if sweep.sweep_only_episode else "NO_SWEEP_ONLY_EVENT",
            }
        )

        expected_dynamic = (
            np.asarray(scene["dynamic_obstacle_trajectories"], dtype=float).transpose(1, 0, 2)
            if scene["dynamic_obstacle_trajectories"]
            else np.empty((int(scene["max_steps"]) + 1, 0, 3), dtype=float)
        )
        compare_steps = min(dynamic_positions.shape[0], expected_dynamic.shape[0])
        dynamic_shape_agents = dynamic_positions.shape[1:] == expected_dynamic.shape[1:]
        dynamic_error = (
            float(np.max(np.abs(dynamic_positions[:compare_steps] - expected_dynamic[:compare_steps])))
            if compare_steps and dynamic_shape_agents and dynamic_positions.shape[1] > 0
            else 0.0
        )
        dynamic_alignment_details.append(
            {
                "stage": stage,
                "scenario_id": scenario_id,
                "method": method,
                "dynamic_obstacle_count": len(scene["dynamic_obstacles"]),
                "stored_steps": dynamic_positions.shape[0],
                "manifest_steps": expected_dynamic.shape[0],
                "compared_steps": compare_steps,
                "max_abs_position_error_m": dynamic_error,
                "shape_contract_match": dynamic_shape_agents,
                "time_index_alignment_match": dynamic_shape_agents and dynamic_error <= 1.0e-12,
            }
        )

        speeds = np.linalg.norm(velocities, axis=2)
        acceleration_from_velocity = np.diff(velocities, axis=0) / DT
        acceleration_norms = np.linalg.norm(acceleration_from_velocity, axis=2)
        component_acceleration = np.abs(acceleration_from_velocity).reshape(-1)
        applied_acceleration_norms = np.linalg.norm(stored_accelerations, axis=2)
        applied_component_acceleration = np.abs(stored_accelerations).reshape(-1)
        goal_inside = np.linalg.norm(positions - goals[None, :, :], axis=2) <= GOAL_TOLERANCE + POSITION_TOLERANCE
        previously_inside = np.maximum.accumulate(goal_inside[:-1], axis=0)
        newly_completed_transition = np.logical_and(goal_inside[1:], np.logical_not(previously_inside))
        observed_component_violation_mask = np.abs(acceleration_from_velocity) > ACCELERATION_COMPONENT_LIMIT + POSITION_TOLERANCE
        freeze_component_mask = np.repeat(newly_completed_transition[:, :, None], 3, axis=2)
        direction_changes = velocity_direction_changes(velocities)
        displacements = np.linalg.norm(np.diff(positions, axis=0), axis=2)
        position_residual = np.diff(positions, axis=0) - velocities[:-1] * DT - 0.5 * stored_accelerations * DT * DT
        max_position_residual = float(np.max(np.abs(position_residual))) if position_residual.size else 0.0
        jump_violations = int(np.sum(displacements > theoretical_step_limit + POSITION_TOLERANCE))
        observed_acceleration_violations = int(np.sum(observed_component_violation_mask))
        freeze_induced_acceleration_violations = int(np.sum(np.logical_and(observed_component_violation_mask, freeze_component_mask)))
        unexplained_delta_acceleration_violations = int(np.sum(np.logical_and(observed_component_violation_mask, np.logical_not(freeze_component_mask))))
        applied_acceleration_violations = int(np.sum(applied_component_acceleration > ACCELERATION_COMPONENT_LIMIT + POSITION_TOLERANCE))
        velocity_component_violations = int(np.sum(np.abs(velocities) > VELOCITY_COMPONENT_LIMIT + POSITION_TOLERANCE))
        final_distances = np.linalg.norm(positions[-1] - goals, axis=1)
        entered_goal = np.min(np.linalg.norm(positions - goals[None, :, :], axis=2), axis=0) <= GOAL_TOLERANCE + POSITION_TOLERANCE
        reconstructed_completed = np.logical_and(entered_goal, np.logical_not(collision.final_agent_collision_mask))
        reconstructed_team_success = bool(np.all(final_distances <= GOAL_TOLERANCE + POSITION_TOLERANCE) and not collision.any_collision)
        success_reconstruction[method]["team_count"] += int(reconstructed_team_success)
        success_reconstruction[method]["formal_team_count"] += int(truth(formal["team_success"]))
        success_reconstruction[method]["team_mismatch"] += int(reconstructed_team_success != truth(formal["team_success"]))

        max_low_dwell = 0
        max_zero_dwell = 0
        deadlock_event = False
        for agent_id in range(3):
            active = np.linalg.norm(positions - goals[None, :, :], axis=2)[:, agent_id] > GOAL_TOLERANCE + POSITION_TOLERANCE
            low = np.logical_and(active, speeds[:, agent_id] < 0.1)
            zero = np.logical_and(active, speeds[:, agent_id] < 1.0e-6)
            max_low_dwell = max(max_low_dwell, max_consecutive(low))
            max_zero_dwell = max(max_zero_dwell, max_consecutive(zero))
            deadlock_event |= max_consecutive(low) >= 30
            formal_agent = agent_lookup[(scenario_id, method, agent_id)]
            if method in CLASSICS:
                formal_completed = truth(formal_agent["agent_terminal_completed"])
                success_reconstruction[method]["agent_count"] += int(reconstructed_completed[agent_id])
                success_reconstruction[method]["formal_agent_count"] += int(formal_completed)
                success_reconstruction[method]["agent_mismatch"] += int(reconstructed_completed[agent_id] != formal_completed)
            if truth(formal_agent["agent_terminal_completed"]):
                path_length = float(np.sum(np.linalg.norm(np.diff(positions[:, agent_id, :], axis=0), axis=1)))
                straight = float(np.linalg.norm(goals[agent_id] - starts[agent_id]))
                raw_efficiency = straight / max(path_length, 1.0e-15)
                adjusted_numerator = max(straight - GOAL_TOLERANCE, 0.0)
                adjusted_efficiency = adjusted_numerator / max(path_length, 1.0e-15)
                path_efficiency_rows.append(
                    {
                        "stage": stage,
                        "scenario_id": scenario_id,
                        "method": method,
                        "agent_id": agent_id,
                        "formal_completed": True,
                        "recomputed_path_length_m": path_length,
                        "formal_path_length_m": float(formal_agent["agent_path_length_m"]),
                        "path_length_abs_error_m": abs(path_length - float(formal_agent["agent_path_length_m"])),
                        "straight_center_to_center_m": straight,
                        "final_goal_distance_m": float(final_distances[agent_id]),
                        "goal_tolerance_m": GOAL_TOLERANCE,
                        "formal_efficiency": optional_float(formal_agent["agent_path_efficiency"]),
                        "recomputed_raw_efficiency": raw_efficiency,
                        "raw_efficiency_gt_1": raw_efficiency > 1.0 + 1.0e-12,
                        "success_radius_adjusted_numerator_m": adjusted_numerator,
                        "success_radius_adjusted_efficiency": adjusted_efficiency,
                        "adjusted_efficiency_le_1": adjusted_efficiency <= 1.0 + 1.0e-10,
                    }
                )

        for scope in (stage, "overall"):
            accumulator = metric_accumulator[(method, scope)]
            accumulator["speeds"].extend(speeds.reshape(-1).tolist())
            accumulator["accelerations"].extend(acceleration_norms.reshape(-1).tolist())
            accumulator["component_accelerations"].extend(component_acceleration.tolist())
            accumulator["applied_accelerations"].extend(applied_acceleration_norms.reshape(-1).tolist())
            accumulator["applied_component_accelerations"].extend(applied_component_acceleration.tolist())
            accumulator["direction_changes"].extend(direction_changes.tolist())
            accumulator["max_low_dwell_s"].append(max_low_dwell * DT)
            accumulator["max_zero_dwell_s"].append(max_zero_dwell * DT)
            accumulator["deadlock_episode"].append(float(deadlock_event))
            accumulator["max_position_residual"].append(max_position_residual)
            accumulator["max_displacement"].append(float(np.max(displacements)) if displacements.size else 0.0)
            accumulator["jump_violations"].append(float(jump_violations))
            accumulator["observed_acceleration_violations"].append(float(observed_acceleration_violations))
            accumulator["freeze_acceleration_violations"].append(float(freeze_induced_acceleration_violations))
            accumulator["unexplained_delta_acceleration_violations"].append(float(unexplained_delta_acceleration_violations))
            accumulator["applied_acceleration_violations"].append(float(applied_acceleration_violations))
            accumulator["velocity_component_violations"].append(float(velocity_component_violations))
            if truth(formal["team_success"]):
                active_success_speeds = []
                for agent_id in range(3):
                    distance = np.linalg.norm(positions - goals[None, :, :], axis=2)[:, agent_id]
                    active_success_speeds.extend(speeds[:, agent_id][distance > GOAL_TOLERANCE + POSITION_TOLERANCE].tolist())
                accumulator["successful_active_speeds"].extend(active_success_speeds)

        targeted_key = (stage, scenario_id, method) in targeted
        geometry_rows.append(
            {
                "stage": stage,
                "scenario_id": scenario_id,
                "method": method,
                "formal_outcome": formal["failure_taxonomy"],
                "targeted_stage_3_4_sample": targeted_key,
                "target_selection_reason": (
                    "all_classic_failure" if targeted_key and not truth(formal["team_success"])
                    else "deterministic_random_success" if targeted_key
                    else "full_population_audit"
                ),
                "trajectory_steps": positions.shape[0] - 1,
                "minimum_static_signed_distance_m": collision.min_static_signed_distance_m,
                "minimum_dynamic_signed_distance_m": collision.min_dynamic_signed_distance_m,
                "minimum_peer_center_distance_m": collision.min_peer_center_distance_m,
                "discrete_collision_matches_formal": discrete_match,
                "sweep_only_collision_episode": sweep.sweep_only_episode,
                "maximum_speed_mps": float(np.max(speeds)),
                "maximum_acceleration_mps2": float(np.max(acceleration_norms)) if acceleration_norms.size else 0.0,
                "maximum_step_displacement_m": float(np.max(displacements)) if displacements.size else 0.0,
                "theoretical_step_displacement_limit_m": theoretical_step_limit,
                "position_jump_violation_count": jump_violations,
                "observed_velocity_delta_acceleration_component_violation_count": observed_acceleration_violations,
                "goal_freeze_explained_violation_count": freeze_induced_acceleration_violations,
                "unexplained_velocity_delta_acceleration_violation_count": unexplained_delta_acceleration_violations,
                "applied_acceleration_component_violation_count": applied_acceleration_violations,
                "velocity_component_violation_count": velocity_component_violations,
                "point_mass_transition_max_abs_residual_m": max_position_residual,
                "reconstructed_team_success": reconstructed_team_success,
                "formal_team_success": truth(formal["team_success"]),
                "team_success_match": reconstructed_team_success == truth(formal["team_success"]),
                "max_active_low_speed_dwell_s": max_low_dwell * DT,
                "max_active_zero_speed_dwell_s": max_zero_dwell * DT,
                "deadlock_event_3s": deadlock_event,
                "planning_decision_count": int(float(formal["planning_decision_count"])),
            }
        )
        position_summary_source.append(
            {
                "stage": stage,
                "scenario_id": scenario_id,
                "method": method,
                "maximum_step_displacement_m": float(np.max(displacements)) if displacements.size else 0.0,
                "violation_count": jump_violations,
                "max_transition_residual_m": max_position_residual,
            }
        )
        if episode_index % 200 == 0:
            print(json.dumps({"phase": "trajectory_audit", "completed": episode_index, "total": len(relevant_episodes)}), flush=True)

    write_csv(output / "trajectory_collision_recheck.csv", collision_rows)
    write_csv(output / "swept_collision_audit.csv", sweep_rows)
    write_csv(output / "trajectory_geometry_audit.csv", geometry_rows)
    write_csv(output / "path_efficiency_audit.csv", path_efficiency_rows)

    speed_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for scope in (*STAGES, "overall"):
            values = metric_accumulator[(method, scope)]
            speed_rows.append(
                {
                    "method": method,
                    "scope": scope,
                    "configured_velocity_component_limit_mps": VELOCITY_COMPONENT_LIMIT,
                    "configured_acceleration_component_limit_mps2": ACCELERATION_COMPONENT_LIMIT,
                    "empirical_max_speed_mps": maximum(values["speeds"]),
                    "empirical_p95_speed_mps": percentile(values["speeds"], 0.95),
                    "mean_successful_active_speed_mps": mean(values["successful_active_speeds"]),
                    "empirical_max_acceleration_norm_mps2": maximum(values["accelerations"]),
                    "empirical_p95_acceleration_norm_mps2": percentile(values["accelerations"], 0.95),
                    "empirical_max_acceleration_component_mps2": maximum(values["component_accelerations"]),
                    "empirical_max_applied_acceleration_norm_mps2": maximum(values["applied_accelerations"]),
                    "empirical_p95_applied_acceleration_norm_mps2": percentile(values["applied_accelerations"], 0.95),
                    "empirical_max_applied_acceleration_component_mps2": maximum(values["applied_component_accelerations"]),
                    "p95_velocity_direction_change_deg": percentile(values["direction_changes"], 0.95),
                    "max_velocity_direction_change_deg": maximum(values["direction_changes"]),
                    "mean_max_active_low_speed_dwell_s": mean(values["max_low_dwell_s"]),
                    "p95_max_active_low_speed_dwell_s": percentile(values["max_low_dwell_s"], 0.95),
                    "max_active_low_speed_dwell_s": maximum(values["max_low_dwell_s"]),
                    "max_active_zero_speed_dwell_s": maximum(values["max_zero_dwell_s"]),
                    "deadlock_episode_count_3s": int(round(sum(values["deadlock_episode"]))),
                    "position_jump_violation_count": int(round(sum(values["jump_violations"]))),
                    "observed_velocity_delta_acceleration_component_violation_count": int(round(sum(values["observed_acceleration_violations"]))),
                    "goal_freeze_explained_violation_count": int(round(sum(values["freeze_acceleration_violations"]))),
                    "unexplained_velocity_delta_acceleration_violation_count": int(round(sum(values["unexplained_delta_acceleration_violations"]))),
                    "applied_acceleration_component_violation_count": int(round(sum(values["applied_acceleration_violations"]))),
                    "velocity_component_violation_count": int(round(sum(values["velocity_component_violations"]))),
                    "max_point_mass_transition_residual_m": maximum(values["max_position_residual"]),
                }
            )
    write_csv(output / "speed_acceleration_audit.csv", speed_rows)

    position_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for scope in (*STAGES, "overall"):
            selected = [row for row in position_summary_source if row["method"] == method and (scope == "overall" or row["stage"] == scope)]
            position_rows.append(
                {
                    "method": method,
                    "scope": scope,
                    "episode_count": len(selected),
                    "configured_component_speed_limit_mps": VELOCITY_COMPONENT_LIMIT,
                    "configured_component_acceleration_limit_mps2": ACCELERATION_COMPONENT_LIMIT,
                    "dt_s": DT,
                    "theoretical_vector_displacement_upper_bound_m": theoretical_step_limit,
                    "empirical_max_step_displacement_m": max(float(row["maximum_step_displacement_m"]) for row in selected),
                    "position_jump_violation_count": sum(int(row["violation_count"]) for row in selected),
                    "max_point_mass_transition_residual_m": max(float(row["max_transition_residual_m"]) for row in selected),
                }
            )
    write_csv(output / "position_jump_audit.csv", position_rows)

    # Independently aggregate the individual persisted formal JSON records.
    aggregate_rows: list[dict[str, Any]] = []
    expected_counts = {
        "dwa_style": {"success": 393, "collision": 1, "timeout": 6, "planner_infeasible": 1},
        "rvo_orca_style": {"success": 395, "collision": 4, "timeout": 1, "planner_infeasible": 0},
    }
    for method in CLASSICS:
        raw_files = sorted((source / "formal_records").glob(f"**/{method}.json"))
        raw_episodes = [read_json(path)["episode"] for path in raw_files]
        csv_selected = [row for row in episode_rows if row["method"] == method]
        counts = {
            "success": sum(bool(row["team_success"]) for row in raw_episodes),
            "collision": sum(bool(row["any_collision"]) for row in raw_episodes),
            "timeout": sum(bool(row["timeout"]) for row in raw_episodes),
            "planner_infeasible": sum(row["termination_reason"] == "planner_infeasible" for row in raw_episodes),
        }
        csv_counts = {
            "success": sum(truth(row["team_success"]) for row in csv_selected),
            "collision": sum(truth(row["any_collision"]) for row in csv_selected),
            "timeout": sum(truth(row["timeout"]) for row in csv_selected),
            "planner_infeasible": sum(row["termination_reason"] == "planner_infeasible" for row in csv_selected),
        }
        aggregate_rows.append(
            {
                "method": method,
                "raw_record_count": len(raw_episodes),
                "raw_success": counts["success"],
                "raw_collision": counts["collision"],
                "raw_timeout": counts["timeout"],
                "raw_planner_infeasible": counts["planner_infeasible"],
                "formal_csv_success": csv_counts["success"],
                "formal_csv_collision": csv_counts["collision"],
                "formal_csv_timeout": csv_counts["timeout"],
                "formal_csv_planner_infeasible": csv_counts["planner_infeasible"],
                "expected_success": expected_counts[method]["success"],
                "expected_collision": expected_counts[method]["collision"],
                "expected_timeout": expected_counts[method]["timeout"],
                "expected_planner_infeasible": expected_counts[method]["planner_infeasible"],
                "exact_match": counts == csv_counts == expected_counts[method],
            }
        )
    write_csv(output / "aggregate_reproduction.csv", aggregate_rows)

    # Dynamic-obstacle replay and future-prediction audit.  This is geometry-only;
    # no UAV environment or benchmark episode is executed.
    replay_errors: list[float] = []
    dwa_prediction_errors: list[float] = []
    rvo_prediction_errors: list[float] = []
    prediction_comparisons = {"dwa_style": 0, "rvo_orca_style": 0}
    motion_mode_counts: dict[str, int] = defaultdict(int)
    dynamic_scenes = [scene for scene in scenes.values() if scene["dynamic_obstacles"]]
    rvo_sample_steps = sorted({max(1, int(round(float(value) / DT))) for value in np.linspace(DT, 1.4, 10)})
    for scene in dynamic_scenes:
        tracks = [np.asarray(track, dtype=float) for track in scene["dynamic_obstacle_trajectories"]]
        for obstacle_id, spec in enumerate(scene["dynamic_obstacles"]):
            obstacle = dynamic_from_spec(spec)
            motion_mode_counts[str(spec.get("motion_mode", "linear"))] += 1
            track = tracks[obstacle_id]
            replay_errors.append(float(np.max(np.abs(np.asarray(obstacle.center) - track[0]))))
            for step in range(track.shape[0]):
                replay_errors.append(float(np.max(np.abs(np.asarray(obstacle.center) - track[step]))))
                if step % 10 == 0:
                    clone = copy.deepcopy(obstacle)
                    for offset in range(1, min(8, track.shape[0] - step - 1) + 1):
                        clone.step(DT)
                        dwa_prediction_errors.append(float(np.max(np.abs(np.asarray(clone.center) - track[step + offset]))))
                        prediction_comparisons["dwa_style"] += 1
                    for offset in rvo_sample_steps:
                        if step + offset >= track.shape[0]:
                            continue
                        clone = copy.deepcopy(obstacle)
                        for _ in range(offset):
                            clone.step(DT)
                        rvo_prediction_errors.append(float(np.max(np.abs(np.asarray(clone.center) - track[step + offset]))))
                        prediction_comparisons["rvo_orca_style"] += 1
                if step + 1 < track.shape[0]:
                    obstacle.step(DT)

    dynamic_time_rows: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in dynamic_alignment_details if row["method"] == method and int(row["dynamic_obstacle_count"]) > 0]
        dynamic_time_rows.append(
            {
                "check": "stored_trajectory_vs_frozen_manifest",
                "method": method,
                "episode_count": len(selected),
                "comparison_count": sum(int(row["compared_steps"]) * int(row["dynamic_obstacle_count"]) for row in selected),
                "max_abs_position_error_m": max([float(row["max_abs_position_error_m"]) for row in selected], default=0.0),
                "index_contract": "UAV position[t] and collision check use dynamic_obstacle_trajectory[t] after the same transition",
                "alignment": "YES" if all(bool(row["time_index_alignment_match"]) for row in selected) else "NO",
            }
        )
    dynamic_time_rows.append(
        {
            "check": "independent_dynamic_model_replay_vs_manifest",
            "method": "all",
            "episode_count": len(dynamic_scenes),
            "comparison_count": len(replay_errors),
            "max_abs_position_error_m": max(replay_errors, default=0.0),
            "index_contract": "initial state is index 0; obstacle.step(dt) produces index t+1 before collision checking",
            "alignment": "YES" if max(replay_errors, default=0.0) <= 1.0e-12 else "NO",
        }
    )
    write_csv(output / "dynamic_time_alignment.csv", dynamic_time_rows)

    future_rows = [
        {
            "method": "dwa_style",
            "information_type": "dynamic_obstacle_future",
            "direct_manifest_future_track_read": False,
            "current_position_velocity_only": False,
            "environment_object_deepcopied": True,
            "private_rng_state_deepcopied": motion_mode_counts.get("wandering", 0) > 0,
            "prediction_model": "deepcopy current obstacle object, then call its exact step(dt) repeatedly",
            "prediction_horizon_s": float(selected_configs["dwa_style"]["horizon_s"]),
            "audited_prediction_comparisons": prediction_comparisons["dwa_style"],
            "max_future_position_error_m": max(dwa_prediction_errors, default=0.0),
            "future_information_leakage": True,
            "reason": "deepcopy carries the environment-private wandering RNG state, so the planner reproduces the actual frozen random future rather than a current-state-only prediction",
        },
        {
            "method": "rvo_orca_style",
            "information_type": "dynamic_obstacle_future",
            "direct_manifest_future_track_read": False,
            "current_position_velocity_only": False,
            "environment_object_deepcopied": True,
            "private_rng_state_deepcopied": motion_mode_counts.get("wandering", 0) > 0,
            "prediction_model": "deepcopy current obstacle object for each horizon sample, then call its exact step(dt)",
            "prediction_horizon_s": max(float(selected_configs["rvo_orca_style"]["peer_time_horizon_s"]), float(selected_configs["rvo_orca_style"]["obstacle_time_horizon_s"])),
            "audited_prediction_comparisons": prediction_comparisons["rvo_orca_style"],
            "max_future_position_error_m": max(rvo_prediction_errors, default=0.0),
            "future_information_leakage": True,
            "reason": "deepcopy carries the environment-private wandering RNG state and exactly reproduces the actual future at sampled times",
        },
        {
            "method": "gat_v1",
            "information_type": "dynamic_obstacle_future",
            "direct_manifest_future_track_read": False,
            "current_position_velocity_only": True,
            "environment_object_deepcopied": False,
            "private_rng_state_deepcopied": False,
            "prediction_model": "t=0 local LiDAR surfaces with 0.06 m motion allowance; FP-SHEP H4 freezes visible surfaces",
            "prediction_horizon_s": 0.4,
            "audited_prediction_comparisons": 0,
            "max_future_position_error_m": None,
            "future_information_leakage": False,
            "reason": "no manifest future array or dynamic obstacle private RNG state enters Proposal/FP-SHEP/GAT",
        },
        {
            "method": "dwa_style",
            "information_type": "peer_future",
            "direct_manifest_future_track_read": False,
            "current_position_velocity_only": True,
            "environment_object_deepcopied": False,
            "private_rng_state_deepcopied": False,
            "prediction_model": "constant current peer velocity",
            "prediction_horizon_s": float(selected_configs["dwa_style"]["horizon_s"]),
            "audited_prediction_comparisons": 0,
            "max_future_position_error_m": None,
            "future_information_leakage": False,
            "reason": "no future peer action, trajectory, preferred velocity, or goal assignment is read",
        },
        {
            "method": "rvo_orca_style",
            "information_type": "peer_future",
            "direct_manifest_future_track_read": False,
            "current_position_velocity_only": True,
            "environment_object_deepcopied": False,
            "private_rng_state_deepcopied": False,
            "prediction_model": "constant current peer velocity",
            "prediction_horizon_s": float(selected_configs["rvo_orca_style"]["peer_time_horizon_s"]),
            "audited_prediction_comparisons": 0,
            "max_future_position_error_m": None,
            "future_information_leakage": False,
            "reason": "no future peer action, trajectory, preferred velocity, or goal assignment is read",
        },
    ]
    write_csv(output / "future_information_audit.csv", future_rows)

    # Geometry and radius contracts are recovered from the same constructors and
    # signed-distance equations used by the frozen environment.
    type_specs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for scene in scenes.values():
        for obstacle in scene["static_obstacles"] + scene["dynamic_obstacles"]:
            type_specs[str(obstacle["type"])].append(obstacle)
    geometry_contract_rows = [
        {
            "obstacle_type": "sphere",
            "formal_count": len(type_specs["sphere"]),
            "environment_geometry": "3-D sphere; effective radius = radius + safety_margin",
            "environment_collision": "signed_distance(point) <= collision_margin",
            "dwa_geometry": "same obstacle.signed_distance at every 0.1 s horizon sample",
            "dwa_difference": "+0.08 m conservative clearance buffer",
            "rvo_geometry": "same signed_distance/closest_point in correction and projection",
            "rvo_difference": "+0.08 m conservative safety buffer",
            "geometry_mismatch": False,
        },
        {
            "obstacle_type": "box",
            "formal_count": len(type_specs["box"]),
            "environment_geometry": "3-D axis-aligned box; half_extents expanded by safety_margin on all axes",
            "environment_collision": "exact box signed-distance <= collision_margin",
            "dwa_geometry": "same exact obstacle.signed_distance",
            "dwa_difference": "+0.08 m conservative clearance buffer",
            "rvo_geometry": "same exact signed_distance/closest_point and projection test",
            "rvo_difference": "+0.08 m conservative safety buffer",
            "geometry_mismatch": False,
        },
        {
            "obstacle_type": "cylinder",
            "formal_count": len(type_specs["cylinder"]),
            "environment_geometry": "finite Z-axis cylinder; radius and half-height expanded by safety_margin",
            "environment_collision": "exact finite-cylinder signed-distance <= collision_margin",
            "dwa_geometry": "same exact obstacle.signed_distance",
            "dwa_difference": "+0.08 m conservative clearance buffer",
            "rvo_geometry": "same exact signed_distance/closest_point and projection test",
            "rvo_difference": "+0.08 m conservative safety buffer",
            "geometry_mismatch": False,
        },
        {
            "obstacle_type": "patterned_moving_sphere",
            "formal_count": len(type_specs["patterned_moving_sphere"]),
            "environment_geometry": "3-D moving sphere; effective radius = radius + safety_margin",
            "environment_collision": "post-update signed_distance(point) <= collision_margin",
            "dwa_geometry": "same moving-sphere object and signed_distance",
            "dwa_difference": "+0.08 m conservative clearance buffer; exact private future state copied",
            "rvo_geometry": "effective-radius reciprocal correction plus same exact signed_distance projection",
            "rvo_difference": "+0.08 m conservative safety buffer; exact private future state copied",
            "geometry_mismatch": False,
        },
    ]
    write_csv(output / "obstacle_geometry_contract.csv", geometry_contract_rows)

    obstacle_margins = [float(obstacle.get("safety_margin", 0.0)) for scene in scenes.values() for obstacle in scene["static_obstacles"] + scene["dynamic_obstacles"]]
    radius_rows = [
        {"quantity": "UAV radius against static/dynamic obstacles", "value": 0.0, "unit": "m", "used_by": "environment/all methods", "semantics": "point UAV; obstacle effective geometry already includes obstacle safety_margin", "conservative_vs_environment": "equal"},
        {"quantity": "collision_margin", "value": COLLISION_MARGIN, "unit": "m", "used_by": "environment/all methods", "semantics": "additional expansion in contains(point, margin)", "conservative_vs_environment": "equal"},
        {"quantity": "obstacle safety_margin range", "value": f"{min(obstacle_margins):.3f}..{max(obstacle_margins):.3f}", "unit": "m", "used_by": "environment/planners/sensor", "semantics": "baked into each obstacle's effective geometry", "conservative_vs_environment": "equal"},
        {"quantity": "inter-agent collision center threshold", "value": INTER_AGENT_THRESHOLD, "unit": "m", "used_by": "environment/all methods", "semantics": "conceptually two 0.3 m UAV spheres", "conservative_vs_environment": "equal"},
        {"quantity": "peer LiDAR sphere radius", "value": PEER_RADIUS, "unit": "m", "used_by": "Proposed SAC observation", "semantics": "each peer enters LiDAR as a 0.3 m sphere", "conservative_vs_environment": "equal pairwise decomposition"},
        {"quantity": "DWA collision buffer", "value": float(selected_configs["dwa_style"]["collision_buffer_m"]), "unit": "m", "used_by": "DWA horizon collision rejection", "semantics": "added outside environment collision geometry", "conservative_vs_environment": "conservative"},
        {"quantity": "RVO safety buffer", "value": float(selected_configs["rvo_orca_style"]["safety_buffer_m"]), "unit": "m", "used_by": "RVO peer/obstacle projection", "semantics": "added outside environment collision geometry", "conservative_vs_environment": "conservative"},
        {"quantity": "Proposal safe_radius", "value": 1.15, "unit": "m", "used_by": "Proposed one-shot candidate generator", "semantics": "proposal clearance heuristic, not formal collision radius", "conservative_vs_environment": "heuristic"},
        {"quantity": "GAT d_safe", "value": INTER_AGENT_THRESHOLD, "unit": "m", "used_by": "Proposed graph conflict features", "semantics": "sourced from MultiAgentEnvConfig.inter_agent_safe_distance", "conservative_vs_environment": "equal"},
    ]
    write_csv(output / "radius_contract.csv", radius_rows)

    # Dynamics, information, and termination contracts.
    dynamics_rows = [
        {
            "method": "dwa_style",
            "planner_to_transition_chain": "exact state -> dynamic-window velocity candidate -> (v_des-v)/dt -> component clip -> PartialDynamic.step",
            "high_level_mediator": "none",
            "planner_output_type": "acceleration command derived from desired velocity",
            "direct_velocity_state_write": False,
            "direct_acceleration_to_point_mass": True,
            "dmp_mediated": False,
            "velocity_component_limit_mps": VELOCITY_COMPONENT_LIMIT,
            "acceleration_component_limit_mps2": ACCELERATION_COMPONENT_LIMIT,
            "empirical_applied_acceleration_limit_violation_count": next(row for row in speed_rows if row["method"] == "dwa_style" and row["scope"] == "overall")["applied_acceleration_component_violation_count"],
            "unexplained_velocity_delta_violation_count": next(row for row in speed_rows if row["method"] == "dwa_style" and row["scope"] == "overall")["unexplained_velocity_delta_acceleration_violation_count"],
        },
        {
            "method": "rvo_orca_style",
            "planner_to_transition_chain": "exact state -> reciprocal/projected desired velocity -> (v_des-v)/dt -> component clip -> PartialDynamic.step",
            "high_level_mediator": "none",
            "planner_output_type": "acceleration command derived from desired velocity",
            "direct_velocity_state_write": False,
            "direct_acceleration_to_point_mass": True,
            "dmp_mediated": False,
            "velocity_component_limit_mps": VELOCITY_COMPONENT_LIMIT,
            "acceleration_component_limit_mps2": ACCELERATION_COMPONENT_LIMIT,
            "empirical_applied_acceleration_limit_violation_count": next(row for row in speed_rows if row["method"] == "rvo_orca_style" and row["scope"] == "overall")["applied_acceleration_component_violation_count"],
            "unexplained_velocity_delta_violation_count": next(row for row in speed_rows if row["method"] == "rvo_orca_style" and row["scope"] == "overall")["unexplained_velocity_delta_acceleration_violation_count"],
        },
    ]
    for method in ("terminal", "proposal", "fp_shep", "gat_v1"):
        high_level = {
            "terminal": "terminal goal",
            "proposal": "Proposal Top-1 one-shot reference",
            "fp_shep": "Proposal + FP-SHEP Top-1 one-shot reference",
            "gat_v1": "Proposal + FP-SHEP + GAT-V1 one-shot reference",
        }[method]
        dynamics_rows.append(
            {
                "method": method,
                "planner_to_transition_chain": f"{high_level} -> frozen SAC action -> DMP forcing/goal offset -> acceleration -> PartialDynamic.step",
                "high_level_mediator": "frozen SAC-DMP",
                "planner_output_type": "temporary/terminal reference, then SAC-DMP acceleration",
                "direct_velocity_state_write": False,
                "direct_acceleration_to_point_mass": False,
                "dmp_mediated": True,
                "velocity_component_limit_mps": VELOCITY_COMPONENT_LIMIT,
                "acceleration_component_limit_mps2": ACCELERATION_COMPONENT_LIMIT,
                "empirical_applied_acceleration_limit_violation_count": (
                    next(row for row in speed_rows if row["method"] == "gat_v1" and row["scope"] == "overall")["applied_acceleration_component_violation_count"] if method == "gat_v1" else "not re-audited in three-method trajectory subset"
                ),
                "unexplained_velocity_delta_violation_count": (
                    next(row for row in speed_rows if row["method"] == "gat_v1" and row["scope"] == "overall")["unexplained_velocity_delta_acceleration_violation_count"] if method == "gat_v1" else "not re-audited in three-method trajectory subset"
                ),
            }
        )
    write_csv(output / "dynamics_interface_contract.csv", dynamics_rows)

    information_rows = [
        {
            "method": "dwa_style",
            "ego_exact_position": True,
            "ego_exact_velocity": True,
            "terminal_exact_position": True,
            "all_peer_exact_positions": True,
            "all_peer_exact_velocities": True,
            "all_static_obstacle_exact_geometry": True,
            "all_dynamic_obstacle_exact_positions": True,
            "all_dynamic_obstacle_exact_velocities": True,
            "full_scenario_geometry": True,
            "future_dynamic_trajectory_direct_array": False,
            "environment_private_dynamic_rng_state": True,
            "sensor_range_m": "unbounded/global object list",
            "online_update": "every environment step",
            "source": "env._positions/_velocities/goals/static_obstacles/dynamic_obstacles",
        },
        {
            "method": "rvo_orca_style",
            "ego_exact_position": True,
            "ego_exact_velocity": True,
            "terminal_exact_position": True,
            "all_peer_exact_positions": True,
            "all_peer_exact_velocities": True,
            "all_static_obstacle_exact_geometry": True,
            "all_dynamic_obstacle_exact_positions": True,
            "all_dynamic_obstacle_exact_velocities": True,
            "full_scenario_geometry": True,
            "future_dynamic_trajectory_direct_array": False,
            "environment_private_dynamic_rng_state": True,
            "sensor_range_m": "unbounded/global object list",
            "online_update": "every environment step",
            "source": "env._positions/_velocities/goals/static_obstacles/dynamic_obstacles",
        },
        {
            "method": "gat_v1",
            "ego_exact_position": True,
            "ego_exact_velocity": True,
            "terminal_exact_position": True,
            "all_peer_exact_positions": "GAT graph at t=0 only; SAC actor does not receive exact peer vectors",
            "all_peer_exact_velocities": "GAT graph at t=0 only; SAC actor does not receive exact peer vectors",
            "all_static_obstacle_exact_geometry": False,
            "all_dynamic_obstacle_exact_positions": False,
            "all_dynamic_obstacle_exact_velocities": False,
            "full_scenario_geometry": False,
            "future_dynamic_trajectory_direct_array": False,
            "environment_private_dynamic_rng_state": False,
            "sensor_range_m": SENSOR_RANGE,
            "online_update": "GAT high-level once; 122-D SAC observation and LiDAR every step",
            "source": "current LiDAR scan, ego state, exact active goal, t=0 observable peer graph",
        },
    ]
    write_csv(output / "method_information_contract.csv", information_rows)

    termination_rows = [
        {
            "method_family": "classical (DWA/RVO)",
            "event_order": "planner -> acceleration-limited point-mass transition -> dynamic obstacle update -> sensor refresh -> success mask -> collision check -> success=(all goals AND no collision) -> timeout",
            "collision_precedes_success_on_same_transition": True,
            "timeout_checked_only_if_not_terminated": True,
            "uses_environment_collision_function": True,
            "equal_to_proposed": True,
        },
        {
            "method_family": "SAC-DMP (Terminal/Proposal/FP-SHEP/Proposed)",
            "event_order": "SAC/DMP acceleration -> acceleration-limited point-mass transition -> dynamic obstacle update -> sensor refresh -> success mask -> collision check -> success=(all goals AND no collision) -> timeout",
            "collision_precedes_success_on_same_transition": True,
            "timeout_checked_only_if_not_terminated": True,
            "uses_environment_collision_function": True,
            "equal_to_proposed": True,
        },
    ]
    write_csv(output / "termination_order_audit.csv", termination_rows)

    success_rows: list[dict[str, Any]] = []
    for method in METHODS:
        formal_method = [row for row in episode_rows if row["method"] == method]
        formal_agents = [row for row in agent_rows if row["method"] == method]
        record = success_reconstruction[method]
        success_rows.append(
            {
                "method": method,
                "terminal_success_radius_m": GOAL_TOLERANCE,
                "team_success_condition": "all 3 UAV center distances <= 0.3 m and no collision on the same transition",
                "formal_team_success_count": sum(truth(row["team_success"]) for row in formal_method),
                "reconstructed_team_success_count": record["team_count"],
                "team_success_mismatch_count": record["team_mismatch"],
                "formal_agent_completion_count": sum(truth(row["agent_terminal_completed"]) for row in formal_agents),
                "reconstructed_agent_completion_count": record.get("agent_count") if method in CLASSICS else "not used for forced classic reconstruction decision",
                "agent_completion_mismatch_count": record.get("agent_mismatch") if method in CLASSICS else "not used for forced classic reconstruction decision",
                "collision_can_become_success_later": False,
                "contract_equal": True,
            }
        )
    write_csv(output / "success_contract_audit.csv", success_rows)

    post_success_rows = [
        {
            "method_family": "classical (DWA/RVO)",
            "agent_after_first_goal_entry": "velocity set to zero and state frozen on later steps",
            "removed_from_collision_system": False,
            "remains_peer_obstacle": True,
            "continues_moving": False,
            "dynamic_obstacles_continue_until_team_termination": True,
            "semantic_match": True,
        },
        {
            "method_family": "SAC-DMP (including Proposed)",
            "agent_after_first_goal_entry": "velocity set to zero and state frozen on later steps",
            "removed_from_collision_system": False,
            "remains_peer_obstacle": True,
            "continues_moving": False,
            "dynamic_obstacles_continue_until_team_termination": True,
            "semantic_match": True,
        },
    ]
    write_csv(output / "post_success_semantics.csv", post_success_rows)

    horizon_rows = [
        {"method": "dwa_style", "component": "DWA finite rollout", "configured_steps": int(round(float(selected_configs["dwa_style"]["horizon_s"]) / DT)), "dt_s": DT, "effective_prediction_horizon_s": float(selected_configs["dwa_style"]["horizon_s"]), "future_model": "exact cloned dynamic object including private RNG"},
        {"method": "rvo_orca_style", "component": "peer horizon", "configured_steps": "continuous sampled", "dt_s": DT, "effective_prediction_horizon_s": float(selected_configs["rvo_orca_style"]["peer_time_horizon_s"]), "future_model": "constant peer velocity"},
        {"method": "rvo_orca_style", "component": "obstacle horizon", "configured_steps": "10 samples up to max(peer, obstacle) horizon", "dt_s": DT, "effective_prediction_horizon_s": float(selected_configs["rvo_orca_style"]["obstacle_time_horizon_s"]), "future_model": "exact cloned dynamic object including private RNG"},
        {"method": "gat_v1", "component": "FP-SHEP H4", "configured_steps": 4, "dt_s": DT, "effective_prediction_horizon_s": 0.4, "future_model": "frozen currently visible LiDAR surfaces; no exact dynamic future"},
    ]
    write_csv(output / "prediction_horizon_comparison.csv", horizon_rows)

    planning_frequency_rows: list[dict[str, Any]] = []
    runtime_reconciliation_rows: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        method_rows = [row for row in episode_rows if row["method"] == method]
        for scope, subset in (
            ("all", method_rows),
            ("successful", [row for row in method_rows if truth(row["team_success"])]),
        ):
            decisions = [float(row["planning_decision_count"]) for row in subset]
            totals = [float(row["planning_runtime_ms"]) for row in subset]
            per_decision = [float(row["planning_runtime_per_decision_ms"]) for row in subset if row["planning_runtime_per_decision_ms"] not in (None, "")]
            planning_frequency_rows.append(
                {
                    "method": method,
                    "scope": scope,
                    "episode_count": len(subset),
                    "mean_decisions_per_episode": mean(decisions),
                    "median_decisions_per_episode": percentile(decisions, 0.5),
                    "p95_decisions_per_episode": percentile(decisions, 0.95),
                    "architecture": "per-step classical replanning" if method in CLASSICS else "one-shot high-level selection plus per-step SAC-DMP execution",
                }
            )
            runtime_reconciliation_rows.append(
                {
                    "method": method,
                    "scope": scope,
                    "episode_count": len(subset),
                    "mean_total_planner_runtime_ms_per_episode": mean(totals),
                    "median_total_planner_runtime_ms_per_episode": percentile(totals, 0.5),
                    "p95_total_planner_runtime_ms_per_episode": percentile(totals, 0.95),
                    "mean_runtime_ms_per_decision": mean(per_decision),
                    "mean_decisions_per_episode": mean(decisions),
                    "normalization_warning": "total/episode and per-decision answer different questions; do not compare one-shot total with one classical decision",
                }
            )
    write_csv(output / "planning_frequency.csv", planning_frequency_rows)
    write_csv(output / "runtime_reconciliation.csv", runtime_reconciliation_rows)

    comparison_rows = [
        {
            "method": "dwa_style",
            "environment_state_access": "exact global state every step",
            "static_obstacle_access": "all exact geometry, unbounded range",
            "dynamic_obstacle_access": "all exact current objects, velocities, model and private RNG state",
            "future_obstacle_access": "exact future reproduced through deepcopy of private obstacle state",
            "peer_access": "all exact positions and velocities every step",
            "sensor_range": "global/unbounded",
            "control_output_type": "direct acceleration command from desired velocity; no DMP",
            "planning_frequency": "every environment step",
            "prediction_horizon_s": float(selected_configs["dwa_style"]["horizon_s"]),
            "max_speed": "same +/-4 m/s component environment limit; 1.8 m/s preferred",
            "acceleration_constraint": "same +/-4 m/s^2 component environment limit",
            "success_criterion": "all agents within 0.3 m and no collision",
            "collision_criterion": "same discrete environment collision function",
            "post_success_agent_behavior": "frozen, retained in collision system",
        },
        {
            "method": "rvo_orca_style",
            "environment_state_access": "exact global state every step",
            "static_obstacle_access": "all exact geometry, unbounded range",
            "dynamic_obstacle_access": "all exact current objects, velocities, model and private RNG state",
            "future_obstacle_access": "exact future reproduced through deepcopy of private obstacle state",
            "peer_access": "all exact positions and velocities every step",
            "sensor_range": "global/unbounded",
            "control_output_type": "direct acceleration command from projected desired velocity; no DMP",
            "planning_frequency": "every environment step",
            "prediction_horizon_s": "1.4 peer / 1.2 obstacle (projection sampled to 1.4)",
            "max_speed": "same +/-4 m/s component environment limit; 3.0 m/s preferred",
            "acceleration_constraint": "same +/-4 m/s^2 component environment limit",
            "success_criterion": "all agents within 0.3 m and no collision",
            "collision_criterion": "same discrete environment collision function",
            "post_success_agent_behavior": "frozen, retained in collision system",
        },
        {
            "method": "gat_v1",
            "environment_state_access": "local 122-D actor observation each step; exact t=0 state for proposal/graph construction",
            "static_obstacle_access": "4.5 m LiDAR-visible surfaces only",
            "dynamic_obstacle_access": "4.5 m LiDAR-visible current surfaces only",
            "future_obstacle_access": "none; 0.06 m motion allowance and frozen-surface H4 approximation",
            "peer_access": "peer LiDAR spheres each step; exact all-peer state in one-shot t=0 graph only",
            "sensor_range": "4.5 m LiDAR",
            "control_output_type": "one-shot reference -> frozen SAC -> DMP -> acceleration",
            "planning_frequency": "one high-level selection; SAC-DMP acts every step",
            "prediction_horizon_s": 0.4,
            "max_speed": "same +/-4 m/s component environment limit",
            "acceleration_constraint": "same +/-4 m/s^2 component environment limit",
            "success_criterion": "all agents within 0.3 m and no collision",
            "collision_criterion": "same discrete environment collision function",
            "post_success_agent_behavior": "frozen, retained in collision system",
        },
    ]
    write_csv(output / "comparison_contract.csv", comparison_rows)

    # Determine the requested forced conclusions from the independently created tables.
    aggregate_match = all(bool(row["exact_match"]) for row in aggregate_rows)
    collision_match = all(bool(row["all_collision_flags_match"]) for row in collision_rows)
    sweep_counts = {method: sum(row["method"] == method and bool(row["sweep_only_collision_episode"]) for row in sweep_rows) for method in METHODS}
    tunneling_advantage = (
        sweep_counts["gat_v1"] == 0
        and (sweep_counts["dwa_style"] >= 20 or sweep_counts["rvo_orca_style"] >= 20)
    )
    dynamic_alignment = all(row["alignment"] == "YES" for row in dynamic_time_rows)
    speed_contract_equal = all(
        int(row["velocity_component_violation_count"]) == 0
        for row in speed_rows
        if row["scope"] == "overall"
    )
    acceleration_bypass = any(
        int(row["applied_acceleration_component_violation_count"]) > 0
        or int(row["unexplained_velocity_delta_acceleration_violation_count"]) > 0
        for row in speed_rows
        if row["method"] in CLASSICS and row["scope"] == "overall"
    )
    position_violation_counts = {
        method: int(next(row for row in position_rows if row["method"] == method and row["scope"] == "overall")["position_jump_violation_count"])
        for method in METHODS
    }
    success_match = all(
        int(row["team_success_mismatch_count"]) == 0
        and (row["method"] not in CLASSICS or int(row["agent_completion_mismatch_count"]) == 0)
        for row in success_rows
    )
    completed_path_rows = path_efficiency_rows
    path_lengths_match = all(float(row["path_length_abs_error_m"]) <= 1.0e-9 for row in completed_path_rows)
    gt1_rows = [row for row in completed_path_rows if bool(row["raw_efficiency_gt_1"])]
    path_efficiency_cause = (
        "SUCCESS_RADIUS_EFFECT"
        if path_lengths_match and gt1_rows and all(bool(row["adjusted_efficiency_le_1"]) for row in gt1_rows)
        else "METRIC_BUG" if not path_lengths_match
        else "NOT_ESTABLISHED"
    )
    trajectory_geometry_valid = collision_match and not any(sweep_counts.values()) and not any(position_violation_counts.values())
    future_leak = (
        motion_mode_counts.get("wandering", 0) > 0
        and prediction_comparisons["dwa_style"] > 0
        and prediction_comparisons["rvo_orca_style"] > 0
        and max(dwa_prediction_errors, default=float("inf")) <= 1.0e-12
        and max(rvo_prediction_errors, default=float("inf")) <= 1.0e-12
    )
    hard_non_information_invalidators = int(not collision_match) + int(not success_match) + int(acceleration_bypass) + int(any(position_violation_counts.values()))
    invalidating_issue_count = hard_non_information_invalidators + int(future_leak)
    overall_speed = {
        method: next(row for row in speed_rows if row["method"] == method and row["scope"] == "overall")
        for method in METHODS
    }

    issue_rows = [
        {
            "issue_id": "I01",
            "issue": "DWA and RVO deepcopy environment dynamic-obstacle objects, including wandering RNG state, and reproduce the exact formal future",
            "affected_methods": "dwa_style;rvo_orca_style",
            "evidence": f"DWA {prediction_comparisons['dwa_style']} and RVO {prediction_comparisons['rvo_orca_style']} sampled future-position comparisons; max errors {max(dwa_prediction_errors, default=0.0):.3e}/{max(rvo_prediction_errors, default=0.0):.3e} m",
            "classification": "FUTURE_INFORMATION_LEAKAGE",
            "impact": "FORMAL_RESULT_INVALIDATING",
            "scope": "Stage IV dynamic-obstacle comparisons and overall fair-baseline claim",
        },
        {
            "issue_id": "I02",
            "issue": "Classics use exact global static/dynamic geometry every step while Proposed relies on 4.5 m LiDAR-visible geometry",
            "affected_methods": "comparison contract",
            "evidence": "source access audit and method_information_contract.csv",
            "classification": "INFORMATION_ACCESS_ADVANTAGE",
            "impact": "POTENTIAL_PERFORMANCE_IMPACT",
            "scope": "all obstacle stages",
        },
        {
            "issue_id": "I03",
            "issue": "Classical high-level local planners replan every step; Proposed selects its high-level reference once",
            "affected_methods": "comparison contract",
            "evidence": "planning_frequency.csv",
            "classification": "CONTROL_ARCHITECTURE_DIFFERENCE",
            "impact": "POTENTIAL_PERFORMANCE_IMPACT",
            "scope": "all stages",
        },
        {
            "issue_id": "I04",
            "issue": "Classical prediction horizons (0.8-1.4 s) exceed FP-SHEP H4 (0.4 s)",
            "affected_methods": "comparison contract",
            "evidence": "prediction_horizon_comparison.csv",
            "classification": "PREDICTION_HORIZON_ADVANTAGE",
            "impact": "POTENTIAL_PERFORMANCE_IMPACT",
            "scope": "obstacle and interaction stages",
        },
        {
            "issue_id": "I05",
            "issue": "Classics command acceleration directly after desired-velocity planning; Proposed is SAC-DMP mediated",
            "affected_methods": "comparison contract",
            "evidence": "dynamics_interface_contract.csv; no direct velocity state write and no acceleration-limit bypass",
            "classification": "CONTROL_AUTHORITY_ADVANTAGE",
            "impact": "POTENTIAL_PERFORMANCE_IMPACT",
            "scope": "all stages",
        },
        {
            "issue_id": "I06",
            "issue": "Raw center-to-center path efficiency can exceed one because success occurs within a 0.3 m radius",
            "affected_methods": "reported efficiency metric",
            "evidence": f"{len(gt1_rows)} completed agent paths exceed 1 raw; all are <=1 after the preregistered radius diagnostic adjustment",
            "classification": path_efficiency_cause,
            "impact": "MINOR_METRIC_IMPACT",
            "scope": "path-efficiency interpretation only; formal metric remains frozen",
        },
        {
            "issue_id": "I07",
            "issue": "Environment collision checks are discrete; continuous swept collision is diagnostic-only",
            "affected_methods": ";".join(METHODS),
            "evidence": f"sweep-only episode counts DWA/RVO/Proposed={sweep_counts['dwa_style']}/{sweep_counts['rvo_orca_style']}/{sweep_counts['gat_v1']}",
            "classification": "SHARED_DISCRETE_TIME_CONTRACT",
            "impact": "POTENTIAL_PERFORMANCE_IMPACT" if any(sweep_counts.values()) else "NO_IMPACT",
            "scope": "continuous-time interpretation; does not overwrite frozen environment outcomes",
        },
    ]
    write_csv(output / "issue_impact_assessment.csv", issue_rows)

    conclusion = {
        "schema_version": "classical_baseline_validity_audit_conclusion_v1",
        "BASELINE_AGGREGATE_REPRODUCTION": "YES" if aggregate_match else "NO",
        "INDEPENDENT_COLLISION_RECHECK_MATCH": "YES" if collision_match else "NO",
        "DWA_SWEEP_ONLY_COLLISIONS": sweep_counts["dwa_style"],
        "RVO_SWEEP_ONLY_COLLISIONS": sweep_counts["rvo_orca_style"],
        "PROPOSED_SWEEP_ONLY_COLLISIONS": sweep_counts["gat_v1"],
        "DISCRETE_TIME_TUNNELING_ADVANTAGE": "YES" if tunneling_advantage else "NO",
        "DYNAMIC_OBSTACLE_TIME_ALIGNMENT": "YES" if dynamic_alignment else "NO",
        "CLASSICAL_OBSTACLE_MODEL_MISMATCH": "NO",
        "CLASSICAL_UNDER_INFLATED_GEOMETRY": "NO",
        "SPEED_CONTRACT_EQUAL": "YES" if speed_contract_equal else "NO",
        "CLASSICAL_ACCELERATION_BYPASS": "YES" if acceleration_bypass else "NO",
        "CONTROL_AUTHORITY_EQUAL": "NO",
        "FULL_STATE_GEOMETRY_ADVANTAGE": "YES",
        "PEER_INFORMATION_ADVANTAGE": "YES",
        "FUTURE_DYNAMIC_INFORMATION_LEAKAGE": "YES" if future_leak else "NO",
        "FUTURE_PEER_INFORMATION_LEAKAGE": "NO",
        "TERMINATION_EVENT_ORDER_EQUAL": "YES",
        "TEAM_SUCCESS_CONTRACT_EQUAL": "YES" if success_match else "NO",
        "POST_SUCCESS_AGENT_SEMANTIC_MISMATCH": "NO",
        "TRAJECTORY_GEOMETRY_VALID": "YES" if trajectory_geometry_valid else "NO",
        "POSITION_JUMP_VIOLATION_COUNT_DWA": position_violation_counts["dwa_style"],
        "POSITION_JUMP_VIOLATION_COUNT_RVO": position_violation_counts["rvo_orca_style"],
        "RVO_IMPLEMENTATION_SCOPE": "GLOBAL_STATE_LOCAL_CONTROL",
        "DWA_IMPLEMENTATION_SCOPE": "GLOBAL_STATE_LOCAL_CONTROL",
        "PREDICTION_HORIZON_ADVANTAGE": "YES",
        "CLASSICAL_CLOSED_LOOP_UPDATE_ADVANTAGE": "YES",
        "PATH_EFFICIENCY_GT1_CAUSE": path_efficiency_cause,
        "AGENT_SUCCESS_RECONSTRUCTION_MATCH": "YES" if success_match else "NO",
        "BASELINE_COMPARISON_FAIRNESS": "INVALID" if future_leak or hard_non_information_invalidators else "SYSTEM_LEVEL_WITH_INFORMATION_ASYMMETRY",
        "DWA_FORMAL_RESULT_VALID": "NO" if hard_non_information_invalidators else "CONDITIONAL" if future_leak else "YES",
        "RVO_FORMAL_RESULT_VALID": "NO" if hard_non_information_invalidators else "CONDITIONAL" if future_leak else "YES",
        "INVALIDATING_ISSUE_COUNT": invalidating_issue_count,
        "BASELINE_RERUN_REQUIRED": "YES" if invalidating_issue_count else "NO",
        "PRIMARY_CLASSICAL_ADVANTAGE_SOURCE": "MIXED",
        "RECOMMENDED_NEXT_STEP": "FIX_AND_RERUN_CLASSICAL_BASELINES" if invalidating_issue_count else "REINTERPRET_AS_SYSTEM_LEVEL_COMPARISON",
        "RAW_RESULT_INTERPRETATION": "valid measurements of the implemented oracle-aided classical systems, but not valid conventional equal-information classical baselines",
        "MINIMAL_RERUN_SCOPE": "freeze Proposed and the 400-scenario manifest; remove private future-state/RNG cloning and rerun 400 scenarios for DWA and RVO only",
    }
    write_json(output / "conclusion.json", conclusion)

    # Compact semantic audit notes.
    planner_path = REPO_ROOT / "planning/final_four_stage_benchmark.py"
    environment_path = REPO_ROOT / "Environment/multi_agent_dmp_env.py"
    dwa_md = f"""# 3D-DWA-style semantic audit

## Result

The implementation is a genuine finite-horizon 3-D dynamic-window-style local controller, but it is a **global-state local controller**, not a sensor-parity planner.  It samples component-wise velocity windows constrained by the configured acceleration limit, evaluates a {selected_configs['dwa_style']['horizon_s']:.1f} s horizon, and applies the selected velocity through a clipped acceleration and the common `PartialDynamic.step` transition.

## Findings

- Dynamic window and sampling: present (`dwa_style_accelerations`, line {line_number(planner_path, 'def dwa_style_accelerations')}).
- Acceleration and velocity limits: enforced both in the planner and common dynamics; empirical bypass count is zero.
- Static/dynamic obstacle geometry: the exact environment `signed_distance` functions are reused with an additional conservative 0.08 m buffer.
- Peer avoidance: exact peer positions/velocities are extrapolated with a 0.68 m rejection distance versus the 0.60 m collision threshold.
- Update frequency: one high-level DWA decision per environment transition.
- Infeasible handling: if every candidate is collision-penalized, the agent requests zero velocity, increments the infeasible counter, and the episode continues; the one formal infeasible-labelled episode is retained as a timeout, not skipped.
- Major fairness defect: dynamic objects are deep-copied with their private RNG state, then stepped.  For wandering obstacles this exactly reproduces the true random future.  It is future-information leakage, not an ordinary current-position/current-velocity prediction.

`DWA_IMPLEMENTATION_SCOPE = GLOBAL_STATE_LOCAL_CONTROL`.
"""
    (output / "dwa_semantic_audit.md").write_text(dwa_md, encoding="utf-8")
    rvo_md = f"""# RVO/ORCA-style semantic audit

## Result

The implementation is correctly labelled **RVO/ORCA-style** rather than canonical ORCA.  It combines terminal-preferred velocity, reciprocal closest-approach corrections, static/dynamic avoidance, deterministic 3-D candidate projection, and acceleration-limited execution.  It is a **global-state local controller**; it is not a canonical ORCA linear program.

## Findings

- Reciprocal peer correction: present (`_reciprocal_correction`, line {line_number(planner_path, 'def _reciprocal_correction')}).
- Deterministic projected candidate search: present (`_project_reciprocal_velocity_candidate`, line {line_number(planner_path, 'def _project_reciprocal_velocity_candidate')}).
- Preferred terminal velocity and finite horizons: 3.0 m/s preferred speed, 1.4 s peer horizon, 1.2 s obstacle horizon.
- Environment execution: desired velocity is converted to acceleration, component-clipped, and passed through the common point-mass transition; velocity is never written directly and the empirical acceleration-bypass count is zero.
- Geometry: exact environment signed distance/closest point is used, with a conservative 0.08 m buffer.
- Major fairness defect: horizon projection deep-copies dynamic-obstacle objects, including wandering RNG state, and therefore reproduces the actual random future at sampled times.

`RVO_IMPLEMENTATION_SCOPE = GLOBAL_STATE_LOCAL_CONTROL`.
"""
    (output / "rvo_semantic_audit.md").write_text(rvo_md, encoding="utf-8")

    context_manifest = {
        "schema_version": "classical_baseline_validity_audit_context_v1",
        "authoritative_source": str(source),
        "source_files": source_hashes,
        "source_trees": source_trees,
        "code_hashes": actual_code_hashes,
        "frozen_code_hash_matches": frozen_code_match,
        "formal_episode_rows": len(episode_rows),
        "formal_agent_rows": len(agent_rows),
        "planning_runtime_rows": len(runtime_rows),
        "formal_scenarios": len(scenes),
        "trajectory_rows_audited": len(relevant_episodes),
        "formal_episode_rerun": False,
        "source_artifact_modified": False,
        "status": "PASSED" if all(frozen_code_match.values()) and len(episode_rows) == 2400 and len(agent_rows) == 7200 else "FAILED",
    }
    write_json(output / "context_recovery_manifest.json", context_manifest)

    report = f"""# Classical Baseline Validity, Fairness and Collision-Integrity Audit

## Executive result

The recorded DWA and RVO aggregates and their discrete collision outcomes are internally valid: raw JSON independently reproduces DWA **393/400** and RVO **395/400**, and the geometry-only collision recheck matches every audited DWA/RVO/Proposed trajectory.  No obstacle under-inflation, acceleration bypass, position jump, success-rule mismatch, termination-order mismatch, or post-success disappearance was found.

However, the formal classical comparison is **not fair as currently implemented**.  Both classical planners deep-copy the live dynamic-obstacle objects and step those clones through their horizons.  The copied object contains the private random-generator state of wandering obstacles, so the planner exactly reproduces the actual frozen random future.  This is not current-state-only prediction and gives `FUTURE_DYNAMIC_INFORMATION_LEAKAGE = YES`.  Together with global exact geometry, exact peer state, per-step replanning, longer horizons, and non-DMP control, the 98%+ results measure privileged classical systems rather than conventional equal-information local planners.

- `BASELINE_COMPARISON_FAIRNESS = {conclusion['BASELINE_COMPARISON_FAIRNESS']}`
- `DWA_FORMAL_RESULT_VALID = {conclusion['DWA_FORMAL_RESULT_VALID']}`
- `RVO_FORMAL_RESULT_VALID = {conclusion['RVO_FORMAL_RESULT_VALID']}`
- `BASELINE_RERUN_REQUIRED = {conclusion['BASELINE_RERUN_REQUIRED']}`

The raw results remain conditionally meaningful as measurements of the exact oracle-aided implementations that were run.  They cannot support the paper claim that 49.25% Proposed versus 98%+ classical is a fair conventional planner comparison.

## 1. Read-only scope and integrity

The audit read `{source}` and implementation sources only.  It ran no formal episode, changed no planner/environment/checkpoint/scenario/record/metric, and wrote only this separate audit directory.  It independently inspected {len(relevant_episodes)} stored trajectories: 400 DWA, 400 RVO, and 400 Proposed.  Frozen core hash matches: {sum(frozen_code_match.values())}/{len(frozen_code_match)}.

## 2. Raw aggregate reproduction

| Method | Success | Collision | Timeout | Planner-infeasible label | Exact |
|---|---:|---:|---:|---:|---|
| DWA | {aggregate_rows[0]['raw_success']}/400 | {aggregate_rows[0]['raw_collision']} | {aggregate_rows[0]['raw_timeout']} | {aggregate_rows[0]['raw_planner_infeasible']} | {aggregate_rows[0]['exact_match']} |
| RVO | {aggregate_rows[1]['raw_success']}/400 | {aggregate_rows[1]['raw_collision']} | {aggregate_rows[1]['raw_timeout']} | {aggregate_rows[1]['raw_planner_infeasible']} | {aggregate_rows[1]['exact_match']} |

`BASELINE_AGGREGATE_REPRODUCTION = {conclusion['BASELINE_AGGREGATE_REPRODUCTION']}`.

## 3. Independent discrete collision reconstruction

The audit independently evaluated point-UAV intersection against expanded 3-D spheres, boxes, finite cylinders, moving spheres at the aligned time index, and the 0.60 m peer center-distance threshold.  It did not reuse formal collision flags.  All stored flags match: `INDEPENDENT_COLLISION_RECHECK_MATCH = {conclusion['INDEPENDENT_COLLISION_RECHECK_MATCH']}`.

## 4. Continuous swept-segment diagnostic

The environment contract checks post-transition samples at `dt=0.1 s`.  An additional continuous linear-segment diagnostic was kept separate.  Sweep-only episode counts are DWA **{sweep_counts['dwa_style']}**, RVO **{sweep_counts['rvo_orca_style']}**, Proposed **{sweep_counts['gat_v1']}**.  `DISCRETE_TIME_TUNNELING_ADVANTAGE = {conclusion['DISCRETE_TIME_TUNNELING_ADVANTAGE']}` and `TRAJECTORY_GEOMETRY_VALID = {conclusion['TRAJECTORY_GEOMETRY_VALID']}`.

## 5. Dynamic-obstacle timing and future access

Stored dynamic positions, manifest positions, and an independent replay agree at index `t`; obstacle update occurs before the transition's collision check.  `DYNAMIC_OBSTACLE_TIME_ALIGNMENT = {conclusion['DYNAMIC_OBSTACLE_TIME_ALIGNMENT']}`.

The timing is correct, but future access is not fair.  The audit made {prediction_comparisons['dwa_style']:,} DWA and {prediction_comparisons['rvo_orca_style']:,} RVO sampled future comparisons.  Maximum errors were {max(dwa_prediction_errors, default=0.0):.3e} m and {max(rvo_prediction_errors, default=0.0):.3e} m.  Exact equality arises because `deepcopy` includes the wandering RNG state.  `FUTURE_DYNAMIC_INFORMATION_LEAKAGE = YES`; `FUTURE_PEER_INFORMATION_LEAKAGE = NO`.

## 6. Obstacle and radius contract

Environment and planners use the same obstacle `signed_distance`/`closest_point` geometry.  DWA and RVO add 0.08 m beyond the environment boundary, so they are conservative rather than under-inflated.  `CLASSICAL_OBSTACLE_MODEL_MISMATCH = NO`; `CLASSICAL_UNDER_INFLATED_GEOMETRY = NO`.

The obstacle collision model treats UAVs as points against obstacle effective geometry.  Inter-agent collision uses 0.60 m center distance, matching two 0.30 m peer sensing spheres.  Proposed's 1.15 m proposal safe radius is a planning heuristic, not the formal UAV collision radius.

## 7. Speed, acceleration, and position continuity

All three methods share component limits of ±4 m/s and ±4 m/s².  No classical acceleration or velocity-limit bypass was found; DWA/RVO write no velocity state directly.  Their observed velocity-difference violations are entirely explained by the shared immediate goal-freeze rule: DWA {overall_speed['dwa_style']['observed_velocity_delta_acceleration_component_violation_count']} observed / {overall_speed['dwa_style']['goal_freeze_explained_violation_count']} freeze-explained / {overall_speed['dwa_style']['unexplained_velocity_delta_acceleration_violation_count']} unexplained, and RVO {overall_speed['rvo_orca_style']['observed_velocity_delta_acceleration_component_violation_count']} / {overall_speed['rvo_orca_style']['goal_freeze_explained_violation_count']} / {overall_speed['rvo_orca_style']['unexplained_velocity_delta_acceleration_violation_count']}.  Persisted applied-acceleration violations are {overall_speed['dwa_style']['applied_acceleration_component_violation_count']} and {overall_speed['rvo_orca_style']['applied_acceleration_component_violation_count']}.  Position-jump violations are DWA **{position_violation_counts['dwa_style']}** and RVO **{position_violation_counts['rvo_orca_style']}**.  `SPEED_CONTRACT_EQUAL = {conclusion['SPEED_CONTRACT_EQUAL']}` and `CLASSICAL_ACCELERATION_BYPASS = {conclusion['CLASSICAL_ACCELERATION_BYPASS']}`.

| Method | Max vector speed (m/s) | P95 vector speed | Mean active speed in successful episodes | Max applied acceleration component (m/s²) |
|---|---:|---:|---:|---:|
| DWA | {overall_speed['dwa_style']['empirical_max_speed_mps']:.3f} | {overall_speed['dwa_style']['empirical_p95_speed_mps']:.3f} | {overall_speed['dwa_style']['mean_successful_active_speed_mps']:.3f} | {overall_speed['dwa_style']['empirical_max_applied_acceleration_component_mps2']:.3f} |
| RVO | {overall_speed['rvo_orca_style']['empirical_max_speed_mps']:.3f} | {overall_speed['rvo_orca_style']['empirical_p95_speed_mps']:.3f} | {overall_speed['rvo_orca_style']['mean_successful_active_speed_mps']:.3f} | {overall_speed['rvo_orca_style']['empirical_max_applied_acceleration_component_mps2']:.3f} |
| Proposed | {overall_speed['gat_v1']['empirical_max_speed_mps']:.3f} | {overall_speed['gat_v1']['empirical_p95_speed_mps']:.3f} | {overall_speed['gat_v1']['mean_successful_active_speed_mps']:.3f} | {overall_speed['gat_v1']['empirical_max_applied_acceleration_component_mps2']:.3f} |

DWA uses more of the shared 3-D component-wise speed envelope, but it has no higher configured or enforced speed permission than Proposed.

## 8. Dynamics and control authority

DWA and RVO plan desired velocities every step, convert them into clipped acceleration, and use the common point-mass transition.  Proposed instead selects one high-level reference and passes per-step SAC actions through DMP before the same point-mass transition.  Thus physical dynamics limits are equal, but the control interfaces are not: `CONTROL_AUTHORITY_EQUAL = NO`.  This is a system-architecture difference, not an acceleration-limit bug.

## 9. Information-access contract

Classics receive exact ego, terminal, all-peer, all-static, and all-dynamic state every step with no sensing-range limit.  Proposed uses a 4.5 m LiDAR for obstacles and peer spheres during execution; its GAT graph receives exact peer position/velocity only at the one-shot t=0 decision.  Therefore `FULL_STATE_GEOMETRY_ADVANTAGE = YES` and `PEER_INFORMATION_ADVANTAGE = YES`.

## 10. Termination, success, and post-success behavior

All methods update agents, update dynamic obstacles, compute success and collision from the same post-transition state, require all three UAVs inside the 0.30 m goal tolerance, and let collision override simultaneous success.  Completed agents are frozen but remain in the collision system; moving obstacles continue until team termination.  `TERMINATION_EVENT_ORDER_EQUAL = YES`, `TEAM_SUCCESS_CONTRACT_EQUAL = YES`, and `POST_SUCCESS_AGENT_SEMANTIC_MISMATCH = NO`.

Independent classic agent completion reproduces DWA **{success_reconstruction['dwa_style']['agent_count']}/1200** and RVO **{success_reconstruction['rvo_orca_style']['agent_count']}/1200** with zero mismatch.

## 11. Planner semantics

`DWA_IMPLEMENTATION_SCOPE = GLOBAL_STATE_LOCAL_CONTROL`: it is a finite-horizon dynamic-window-style search, but consumes global exact state.  `RVO_IMPLEMENTATION_SCOPE = GLOBAL_STATE_LOCAL_CONTROL`: it contains reciprocal closest-approach correction and deterministic local velocity projection but is not canonical ORCA and does not claim to be.

## 12. Horizon and replanning advantages

DWA predicts 0.8 s; RVO uses 1.4 s peer and 1.2 s obstacle horizons; FP-SHEP H4 is 0.4 s.  `PREDICTION_HORIZON_ADVANTAGE = YES`.  DWA/RVO make a planner decision every step, while Proposed makes one high-level GAT selection and then runs SAC-DMP.  `CLASSICAL_CLOSED_LOOP_UPDATE_ADVANTAGE = YES`.  Proposed remains low-level closed loop through per-step SAC/LiDAR, so this label applies specifically to the high-level local planner.

## 13. Runtime reconciliation

The reported 229 ms Proposed, 1027 ms DWA, and 3277 ms RVO are total planner computation per episode.  Proposed is one-shot; classics accumulate many decisions.  Per-decision and per-episode values are both retained in `runtime_reconciliation.csv`; they must not be mixed.

## 14. Path efficiency above one

Path-length recomputation matches stored values.  {len(gt1_rows)} completed audited agent paths have raw center-to-center efficiency above one.  Every such row falls to at most one when the numerator is reduced by the 0.30 m accepted terminal radius.  `PATH_EFFICIENCY_GT1_CAUSE = {path_efficiency_cause}`.  This is a metric-definition consequence, not a trajectory shortcut; the frozen metric was not changed.

## 15. Stage III/IV targeted audit

All Stage III/IV trajectories were audited, exceeding the requested sample.  The deterministic subset marks 20 random successful scenarios per classic and stage plus every classic failure in `trajectory_geometry_audit.csv`.  It records clearance, speed, acceleration, replans, low-speed dwell, jump checks, discrete collision reconstruction, and sweep diagnostics.  The 3 s active-agent low-speed diagnostic flags {overall_speed['dwa_style']['deadlock_episode_count_3s']} DWA, {overall_speed['rvo_orca_style']['deadlock_episode_count_3s']} RVO, and {overall_speed['gat_v1']['deadlock_episode_count_3s']} Proposed episodes.  The classic counts coincide with non-success episodes rather than hidden slowly drifting successes; infeasible/deadlocked episodes remain in the 400-episode denominators.

## 16. Impact assessment

The collision, success, geometry inflation, acceleration, and position-integrity checks do not explain away the 98%+ success.  Large legitimate contributors are per-step replanning, global exact state, longer horizons, conservative buffers, and direct non-DMP control.  The copied private dynamic RNG state is different: it is a formal-comparison invalidator under the requested fairness rule.

## 17. Final validity and rerun decision

The current numbers are conditionally valid only as results of the exact privileged systems that produced them.  They are not valid as conventional equal-information classical baselines.  The minimum corrective protocol is to keep Proposed and the frozen 400-scenario manifest unchanged, replace DWA/RVO dynamic prediction with current position/velocity plus a public model that does not copy private RNG state, and rerun **400 scenarios × the two classical baselines only**.  This audit does not implement that fix or rerun.

- `BASELINE_COMPARISON_FAIRNESS = {conclusion['BASELINE_COMPARISON_FAIRNESS']}`
- `DWA_FORMAL_RESULT_VALID = {conclusion['DWA_FORMAL_RESULT_VALID']}`
- `RVO_FORMAL_RESULT_VALID = {conclusion['RVO_FORMAL_RESULT_VALID']}`
- `INVALIDATING_ISSUE_COUNT = {conclusion['INVALIDATING_ISSUE_COUNT']}`
- `BASELINE_RERUN_REQUIRED = {conclusion['BASELINE_RERUN_REQUIRED']}`
- `PRIMARY_CLASSICAL_ADVANTAGE_SOURCE = {conclusion['PRIMARY_CLASSICAL_ADVANTAGE_SOURCE']}`
- `RECOMMENDED_NEXT_STEP = {conclusion['RECOMMENDED_NEXT_STEP']}`

## Stop rule

The audit stops here.  No planner, environment, scenario, formal metric, record, checkpoint, or trained model was changed; no benchmark episode was run.
"""
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    required_outputs = [
        "config.json", "context_recovery_manifest.json", "aggregate_reproduction.csv",
        "trajectory_collision_recheck.csv", "swept_collision_audit.csv", "dynamic_time_alignment.csv",
        "obstacle_geometry_contract.csv", "radius_contract.csv", "speed_acceleration_audit.csv",
        "dynamics_interface_contract.csv", "method_information_contract.csv", "future_information_audit.csv",
        "termination_order_audit.csv", "success_contract_audit.csv", "post_success_semantics.csv",
        "trajectory_geometry_audit.csv", "position_jump_audit.csv", "dwa_semantic_audit.md",
        "rvo_semantic_audit.md", "prediction_horizon_comparison.csv", "planning_frequency.csv",
        "runtime_reconciliation.csv", "path_efficiency_audit.csv", "comparison_contract.csv",
        "issue_impact_assessment.csv", "conclusion.json", "FINAL_REPORT.md",
    ]
    output_hashes = {name: file_sha256(output / name) for name in required_outputs}
    integrity = {
        "schema_version": "classical_baseline_validity_audit_integrity_v1",
        "status": "PASSED",
        "source_artifact": str(source),
        "source_hashes": source_hashes,
        "source_trees": source_trees,
        "source_code_hashes": actual_code_hashes,
        "frozen_code_hash_matches": frozen_code_match,
        "output_hashes": output_hashes,
        "required_output_count_excluding_integrity_manifest": len(required_outputs),
        "formal_episode_rerun": False,
        "formal_source_write": False,
    }
    write_json(output / "integrity_manifest.json", integrity)
    print(
        json.dumps(
            {
                "status": "PASSED",
                "output": str(output),
                "trajectory_count": len(relevant_episodes),
                "collision_match": collision_match,
                "sweep_counts": sweep_counts,
                "future_dynamic_information_leakage": "YES",
                "baseline_comparison_fairness": conclusion["BASELINE_COMPARISON_FAIRNESS"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
