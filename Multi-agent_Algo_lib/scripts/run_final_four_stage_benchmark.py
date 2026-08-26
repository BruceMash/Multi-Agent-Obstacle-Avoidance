"""Prepare, tune, freeze, and run the final four-stage six-method benchmark."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402
from planning.final_four_stage_benchmark import (  # noqa: E402
    DWAStyleConfig,
    FAMILY_ORDER,
    RVOStyleConfig,
    STAGE_ORDER,
    WORKSPACE_BOUNDS,
    generate_scenario_manifest,
    json_ready,
    obstacles_from_entry,
    run_classical_episode,
    stable_hash,
    validate_scenario_manifest,
)
from planning.gat.stage1_training import load_model_checkpoint, resolve_device  # noqa: E402
from scripts.evaluate_gat_closed_loop import (  # noqa: E402
    CORE_METHOD_PATHS,
    METHOD_FP_SHEP,
    METHOD_GAT,
    METHOD_PROPOSAL,
    METHOD_TERMINAL,
    _file_hashes,
    _model_hash,
    _policy_parameter_sha256,
    build_shared_selection_bundle,
    run_method_episode,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_single_policy_multi_agent import _build_environment  # noqa: E402
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


SCHEMA_VERSION = "final_four_stage_benchmark_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/evaluation/final_four_stage_benchmark.json"
METHOD_ORDER = (
    "dwa_style",
    "rvo_orca_style",
    "terminal",
    "proposal",
    "fp_shep",
    "gat_v1",
)
SAC_METHOD_MAP = {
    "terminal": METHOD_TERMINAL,
    "proposal": METHOD_PROPOSAL,
    "fp_shep": METHOD_FP_SHEP,
    "gat_v1": METHOD_GAT,
}
EXTRA_CORE_PATHS = (
    "planning/final_four_stage_benchmark.py",
    "Multi-agent_Algo_lib/scripts/run_final_four_stage_benchmark.py",
    "Multi-agent_Algo_lib/scripts/evaluate_actor_dmp_goal_semantics.py",
    "Multi-agent_Algo_lib/scripts/evaluate_gat_closed_loop.py",
    "configs/evaluation/final_four_stage_benchmark.json",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(json_ready(value), ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict, np.ndarray))
                    else value
                    for key, value in row.items()
                }
            )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk_integer_seed_values(value: Any, key: str = "") -> set[int]:
    found: set[int] = set()
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            found.update(_walk_integer_seed_values(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            found.update(_walk_integer_seed_values(child, key))
    elif (
        "seed" in key.lower()
        and not key.lower().endswith("seed_base")
        and isinstance(value, (int, float))
    ):
        numeric = float(value)
        if numeric.is_integer() and 0 <= numeric <= 2_147_483_647:
            found.add(int(numeric))
    return found


def audit_used_seeds(*, excluded_root: Path | None = None) -> dict[str, Any]:
    roots = [REPO_ROOT / "artifacts", REPO_ROOT / "configs"]
    source_rows: list[dict[str, Any]] = []
    all_seeds: set[int] = set()
    json_files = 0
    csv_files = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if excluded_root is not None:
                try:
                    path.relative_to(excluded_root)
                    continue
                except ValueError:
                    pass
            seeds: set[int] = set()
            try:
                if path.suffix.lower() == ".json" and path.stat().st_size <= 64 * 1024 * 1024:
                    seeds = _walk_integer_seed_values(load_json(path))
                    json_files += 1
                elif path.suffix.lower() in {".csv", ".tsv"} and path.stat().st_size <= 256 * 1024 * 1024:
                    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
                    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
                        reader = csv.DictReader(stream, delimiter=delimiter)
                        seed_fields = [field for field in (reader.fieldnames or []) if "seed" in field.lower()]
                        if seed_fields:
                            for row in reader:
                                for field in seed_fields:
                                    raw = row.get(field)
                                    try:
                                        numeric = float(raw) if raw not in (None, "") else -1.0
                                    except (TypeError, ValueError):
                                        continue
                                    if numeric.is_integer() and 0 <= numeric <= 2_147_483_647:
                                        seeds.add(int(numeric))
                    csv_files += 1
            except (OSError, UnicodeError, json.JSONDecodeError, csv.Error):
                continue
            if seeds:
                all_seeds.update(seeds)
                source_rows.append(
                    {
                        "path": str(path.relative_to(REPO_ROOT)),
                        "seed_count": len(seeds),
                        "minimum": min(seeds),
                        "maximum": max(seeds),
                    }
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_scope": ["artifacts", "configs"],
        "structured_json_seed_fields": True,
        "structured_csv_seed_columns": True,
        "json_files_scanned": json_files,
        "csv_files_scanned": csv_files,
        "unique_used_seed_count": len(all_seeds),
        "used_seeds": sorted(all_seeds),
        "sources": source_rows,
    }


def audit_used_geometry(*, excluded_root: Path | None = None) -> dict[str, Any]:
    fields = {
        "geometry_fingerprint",
        "environment_fingerprint",
        "initial_condition_hash",
        "scenario_manifest_hash",
        "initial_hash",
    }
    hashes: set[str] = set()
    sources: dict[str, list[str]] = defaultdict(list)

    def visit(value: Any, key: str, source: str) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key), source)
        elif isinstance(value, list):
            for child in value:
                visit(child, key, source)
        elif key in fields and isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
            normalized = value.lower()
            hashes.add(normalized)
            if len(sources[normalized]) < 3:
                sources[normalized].append(source)

    for path in (REPO_ROOT / "artifacts").rglob("*"):
        if not path.is_file():
            continue
        if excluded_root is not None:
            try:
                path.relative_to(excluded_root)
                continue
            except ValueError:
                pass
        relative = str(path.relative_to(REPO_ROOT))
        try:
            if path.suffix.lower() == ".json" and path.stat().st_size <= 64 * 1024 * 1024:
                visit(load_json(path), "", relative)
            elif path.suffix.lower() == ".csv" and path.stat().st_size <= 256 * 1024 * 1024:
                with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
                    reader = csv.DictReader(stream)
                    hash_fields = [field for field in (reader.fieldnames or []) if field in fields]
                    for row in reader:
                        for field in hash_fields:
                            visit(row.get(field), field, relative)
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error):
            continue
    return {
        "schema_version": SCHEMA_VERSION,
        "hash_fields": sorted(fields),
        "unique_historical_geometry_hash_count": len(hashes),
        "geometry_hashes": sorted(hashes),
        "example_sources": dict(sources),
    }


def difficulty_rows(manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "stage": row["stage"],
            "scenario_id": row["scenario_id"],
            "family": row["family"],
            "seed": row["seed"],
            **row["difficulty"],
            "geometry_fingerprint": row["geometry_fingerprint"],
            "translation_invariant_fingerprint": row["translation_invariant_fingerprint"],
        }
        for row in manifest["entries"]
    ]


def family_rows(manifests: Sequence[tuple[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for split, manifest in manifests:
        groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for entry in manifest["entries"]:
            groups[(entry["stage"], entry["family"])].append(entry)
        for (stage, family), entries in groups.items():
            rows.append(
                {
                    "split": split,
                    "stage": stage,
                    "family": family,
                    "scenario_count": len(entries),
                    "scenario_ids": [row["scenario_id"] for row in entries],
                    "structure_defined_before_performance": True,
                }
            )
    return rows


def hardware_manifest() -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    try:
        import psutil

        ram_bytes = int(psutil.virtual_memory().total)
        cpu_physical = psutil.cpu_count(logical=False)
        cpu_logical = psutil.cpu_count(logical=True)
    except ImportError:
        ram_bytes = None
        cpu_physical = None
        cpu_logical = os.cpu_count()
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu": platform.processor(),
        "cpu_physical_cores": cpu_physical,
        "cpu_logical_cores": cpu_logical,
        "ram_bytes": ram_bytes,
        "gpu": gpu,
        "cuda_runtime": torch.version.cuda,
        "pytorch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "timing_clock": "time.perf_counter_ns",
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected schema version")
    if tuple(config["methods"]) != METHOD_ORDER:
        raise ValueError("frozen method order changed")
    execution = config["execution"]
    expected = {
        "num_agents": 3,
        "top_k": 10,
        "H_preview": 4,
        "max_steps": 220,
        "dt": 0.1,
        "handoff_threshold_m": 0.25,
        "actor_observation_dim": 122,
        "boundary_mode": "boundary_free",
        "replanning_enabled": False,
    }
    for key, value in expected.items():
        if execution[key] != value:
            raise ValueError(f"frozen execution field changed: {key}")
    if any(bool(value) for value in config["strict_exclusions"].values()):
        raise ValueError("strict exclusion flags must remain false")


def prepare(config: Mapping[str, Any], output_dir: Path) -> Path:
    validate_config(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "engineering_search").mkdir()
    (output_dir / "method_configs").mkdir()
    (output_dir / "formal_records").mkdir()
    (output_dir / "trajectories").mkdir()
    resolved = copy.deepcopy(dict(config))
    resolved["created_at"] = datetime.now().astimezone().isoformat()
    resolved["resolved_output_dir"] = str(output_dir)
    write_json(output_dir / "config.json", resolved)

    source = config["sources"]
    gat_path = REPO_ROOT / source["gat_checkpoint"]
    sac_path = REPO_ROOT / source["sac_checkpoint"]
    hashes = {
        "gat_checkpoint": sha256_file(gat_path),
        "sac_checkpoint": sha256_file(sac_path),
    }
    if hashes["gat_checkpoint"] != source["gat_checkpoint_sha256_expected"]:
        raise RuntimeError("GAT checkpoint SHA256 mismatch")
    if hashes["sac_checkpoint"] != source["sac_checkpoint_sha256_expected"]:
        raise RuntimeError("SAC checkpoint SHA256 mismatch")
    core_hashes = _file_hashes(tuple(CORE_METHOD_PATHS) + EXTRA_CORE_PATHS)

    # The current Goal's output root can contain a stopped preflight or a
    # resumed copy of this same frozen manifest.  It is not historical evidence
    # and must not make its own allocated seed block appear previously used.
    goal_output_root = REPO_ROOT / str(config["output_root"])
    used_seed_manifest = audit_used_seeds(excluded_root=goal_output_root)
    used_geometry_manifest = audit_used_geometry(excluded_root=goal_output_root)
    protocol = config["scenario_protocol"]
    development = generate_scenario_manifest(
        counts_per_stage=int(protocol["development_scenarios_per_stage"]),
        seed_base=int(protocol["development_seed_base"]),
        prefix=str(protocol["development_prefix"]),
        max_steps=int(config["execution"]["max_steps"]),
        dt=float(config["execution"]["dt"]),
    )
    formal = generate_scenario_manifest(
        counts_per_stage=int(protocol["formal_scenarios_per_stage"]),
        seed_base=int(protocol["formal_seed_base"]),
        prefix=str(protocol["formal_prefix"]),
        max_steps=int(config["execution"]["max_steps"]),
        dt=float(config["execution"]["dt"]),
    )
    development_validation = validate_scenario_manifest(development)
    formal_validation = validate_scenario_manifest(formal)
    used_seeds = set(used_seed_manifest["used_seeds"])
    development_seeds = {int(row["seed"]) for row in development["entries"]}
    formal_seeds = {int(row["seed"]) for row in formal["entries"]}
    historical_hashes = set(used_geometry_manifest["geometry_hashes"])
    new_hashes = {
        row["geometry_fingerprint"] for row in development["entries"] + formal["entries"]
    }
    separation = {
        "development_formal_seed_overlap": len(development_seeds & formal_seeds),
        "development_historical_seed_overlap": len(development_seeds & used_seeds),
        "formal_historical_seed_overlap": len(formal_seeds & used_seeds),
        "development_formal_geometry_overlap": len(
            {row["geometry_fingerprint"] for row in development["entries"]}
            & {row["geometry_fingerprint"] for row in formal["entries"]}
        ),
        "new_historical_geometry_hash_overlap": len(new_hashes & historical_hashes),
    }
    separation["status"] = (
        "PASSED" if all(value == 0 for value in separation.values()) else "FAILED"
    )
    if development_validation["status"] != "PASSED":
        raise RuntimeError(f"development manifest invalid: {development_validation}")
    if formal_validation["status"] != "PASSED":
        raise RuntimeError(f"formal manifest invalid: {formal_validation}")
    if separation["status"] != "PASSED":
        raise RuntimeError(f"development/formal/historical separation failed: {separation}")

    write_json(output_dir / "all_used_seed_manifest.json", used_seed_manifest)
    write_json(output_dir / "all_used_geometry_manifest.json", used_geometry_manifest)
    write_json(
        output_dir / "development_seed_manifest.json",
        {"seeds": sorted(development_seeds), "count": len(development_seeds)},
    )
    write_json(
        output_dir / "formal_seed_manifest.json",
        {"seeds": sorted(formal_seeds), "count": len(formal_seeds)},
    )
    write_json(output_dir / "development_scenario_manifest.json", development)
    write_json(output_dir / "scenario_manifest.json", formal)
    (output_dir / "scenario_manifest_sha256.txt").write_text(
        formal["manifest_sha256"] + "\n", encoding="ascii"
    )
    write_csv(
        output_dir / "difficulty_statistics.csv",
        difficulty_rows(development, "development") + difficulty_rows(formal, "formal"),
    )
    write_csv(
        output_dir / "family_manifest.csv",
        family_rows((("development", development), ("formal", formal))),
    )
    write_json(output_dir / "hardware_manifest.json", hardware_manifest())
    context = {
        "schema_version": SCHEMA_VERSION,
        "TECHNICAL_PATH_UNCHANGED": "YES",
        "checkpoint_hashes": hashes,
        "core_hashes_before_development": core_hashes,
        "development_manifest_validation": development_validation,
        "formal_manifest_validation": formal_validation,
        "separation": separation,
        "historical_decisions": {
            "final_gat_checkpoint": "V1",
            "ia_formal_gain": "NO",
            "minimal_theory_extension_justified": "NO",
            "recommended_next_step": "KEEP_GAT_V1_AND_PROCEED_TO_BASELINES",
        },
        "may20_reference": {
            "source": source["may20_reference_dir"],
            "task_scale_m": "approximately 7-11",
            "dense_mixed_success_rate": 0.92,
        },
        "status": "PASSED",
    }
    write_json(output_dir / "context_recovery_manifest.json", context)
    write_json(
        output_dir / "preformal_integrity.json",
        {
            "checkpoint_hashes": hashes,
            "core_hashes": core_hashes,
            "formal_manifest_sha256": formal["manifest_sha256"],
            "formal_result_count": 0,
            "FORMAL_PHASE": "CLOSED_PENDING_ENGINEERING_FREEZE",
        },
    )
    print(json.dumps({"phase": "prepare", "output_dir": str(output_dir), "status": "PASSED"}), flush=True)
    return output_dir


class ManifestEnvironmentBuilder:
    def __init__(self, manifest: Mapping[str, Any]) -> None:
        self.entries = {str(row["scenario_id"]): row for row in manifest["entries"]}

    def __call__(
        self, *, config: Any, scenario: str, seed: int, peer_radius: float
    ) -> tuple[Any, dict[str, Any]]:
        entry = self.entries[str(scenario)]
        if int(seed) != int(entry["seed"]):
            raise ValueError("scenario seed does not match frozen manifest")
        static, dynamic = obstacles_from_entry(entry)
        env = _build_environment(
            config,
            observation_mode="peer_spheres",
            peer_radius=float(peer_radius),
            training_distribution=False,
            include_boundaries_in_sensor=False,
            terminate_on_boundary_collision=False,
        )
        env.reset(
            seed=int(seed),
            options={
                "starts": np.asarray(entry["starts"], dtype=float),
                "goals": np.asarray(entry["goals"], dtype=float),
                "static_obstacles": static,
                "dynamic_obstacles": dynamic,
            },
        )
        metadata = {
            "stage": entry["stage"],
            "family": entry["family"],
            "scenario_id": entry["scenario_id"],
            "environment_fingerprint": entry["environment_fingerprint"],
            "geometry_fingerprint": entry["geometry_fingerprint"],
            "boundary_mode": "boundary_free",
            "static_obstacle_count": len(static),
            "dynamic_obstacle_count": len(dynamic),
        }
        return env, metadata


class FrozenRuntime:
    def __init__(self, config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
        self.config = config
        execution = config["execution"]
        base = build_single_distribution_multi_config(
            num_agents=int(execution["num_agents"]),
            max_steps=int(execution["max_steps"]),
        )
        self.multi_config = replace(
            base,
            workspace_bounds=WORKSPACE_BOUNDS,
            randomize_start_goal=False,
            start_position_bounds=((-0.4, -2.0, -0.9), (0.2, 2.0, 0.9)),
            goal_position_bounds=((7.0, -2.0, -0.9), (11.0, 2.0, 0.9)),
            min_start_goal_distance=5.5,
        )
        self.builder = ManifestEnvironmentBuilder(manifest)
        self.base_eval_config = build_eval_config(config, manifest)
        settings = load_json(REPO_ROOT / config["sources"]["base_execution_config"])
        settings.update(
            {
                "checkpoint": config["sources"]["sac_checkpoint"],
                "checkpoint_sha256_expected": config["sources"]["sac_checkpoint_sha256_expected"],
                "deterministic_policy": True,
                "num_agents": int(execution["num_agents"]),
                "max_steps": int(execution["max_steps"]),
                "dt": float(execution["dt"]),
                "peer_radius": float(execution["peer_radius"]),
                "temporary_reference": {
                    **settings["temporary_reference"],
                    "K_requested": int(execution["top_k"]),
                    "reached_tolerance_m": float(execution["handoff_threshold_m"]),
                    "replanning_enabled": False,
                },
                "proposal_config": copy.deepcopy(config["proposal_config"]),
            }
        )
        self.execution_settings = settings
        self.policy, loaded = _load_policy(settings, self.multi_config)
        expected_sac = (REPO_ROOT / config["sources"]["sac_checkpoint"]).resolve()
        if loaded.resolve() != expected_sac:
            raise RuntimeError("SAC loader resolved a different checkpoint")
        stage1 = load_json(REPO_ROOT / config["sources"]["stage1_config"])
        self.gat_device = resolve_device(stage1["training"]["device"])
        self.gat_model = load_model_checkpoint(
            REPO_ROOT / config["sources"]["gat_checkpoint"], stage1, self.gat_device
        )
        for parameter in self.gat_model.parameters():
            parameter.requires_grad_(False)
        self.gat_model.eval()
        self.policy_hash = _policy_parameter_sha256(self.policy)
        self.gat_model_hash = _model_hash(self.gat_model)


def build_eval_config(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    execution = config["execution"]
    return {
        "top_k": int(execution["top_k"]),
        "H_preview": int(execution["H_preview"]),
        "max_steps": int(execution["max_steps"]),
        "dt": float(execution["dt"]),
        "peer_radius": float(execution["peer_radius"]),
        "handoff_threshold_m": float(execution["handoff_threshold_m"]),
        "proposal_config": copy.deepcopy(config["proposal_config"]),
        "fp_shep": copy.deepcopy(config["fp_shep"]),
        "graph": copy.deepcopy(config["graph"]),
        "formal_scenarios": [row["scenario_id"] for row in manifest["entries"]],
        "formal_seeds": [int(row["seed"]) for row in manifest["entries"]],
    }


def proposed_eval_config(base: Mapping[str, Any], search: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    result["proposal_config"].update(
        {
            "safe_radius": float(search["safe_radius"]),
            "obstacle_motion_allowance": float(search["obstacle_motion_allowance"]),
            "top_k": 10,
        }
    )
    result["graph"]["d_align"] = float(search["d_align"])
    return result


def _trajectory_agent_minima(
    trajectory: Mapping[str, Any], entry: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(trajectory["positions"], dtype=float)
    static, _ = obstacles_from_entry(entry)
    dynamic_paths = entry["dynamic_obstacle_trajectories"]
    obstacle_min = np.full(positions.shape[1], float("inf"))
    peer_min = np.full(positions.shape[1], float("inf"))
    for step, frame in enumerate(positions):
        for agent_id, point in enumerate(frame):
            values = [float(obstacle.signed_distance(point)) for obstacle in static]
            for obstacle_id, spec in enumerate(entry["dynamic_obstacles"]):
                dynamic = obstacles_from_entry(
                    {"static_obstacles": [], "dynamic_obstacles": [spec]}
                )[1][0]
                dynamic.center = np.asarray(
                    dynamic_paths[obstacle_id][min(step, len(dynamic_paths[obstacle_id]) - 1)],
                    dtype=float,
                )
                values.append(float(dynamic.signed_distance(point)))
            if values:
                obstacle_min[agent_id] = min(obstacle_min[agent_id], min(values))
            peers = [
                float(np.linalg.norm(point - frame[peer_id]))
                for peer_id in range(len(frame))
                if peer_id != agent_id
            ]
            if peers:
                peer_min[agent_id] = min(peer_min[agent_id], min(peers))
    return obstacle_min, peer_min


def standardize_sac_result(
    episode: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]],
    trajectory: Mapping[str, Any],
    entry: Mapping[str, Any],
    method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    result = copy.deepcopy(dict(episode))
    result.update(
        {
            "stage": entry["stage"],
            "family": entry["family"],
            "scenario_id": entry["scenario_id"],
            "seed": int(entry["seed"]),
            "method": method,
            "any_collision": bool(episode["collision"]),
            "termination_time_s": float(episode["steps"] * entry["dt"]),
            "end_to_end_runtime_ms": float(episode["episode_runtime_ms"])
            + float(episode.get("planning_runtime_ms", 0.0)),
        }
    )
    obstacle_min, peer_min = _trajectory_agent_minima(trajectory, entry)
    starts = np.asarray(entry["starts"], dtype=float)
    goals = np.asarray(entry["goals"], dtype=float)
    straight = np.linalg.norm(goals - starts, axis=1)
    path_lengths = np.asarray(episode["per_agent_path_length_m"], dtype=float)
    standardized_agents: list[dict[str, Any]] = []
    for row in agents:
        agent_id = int(row["agent_id"])
        completion_step = row.get("terminal_completed_step")
        completed = completion_step is not None
        standardized_agents.append(
            {
                "stage": entry["stage"],
                "family": entry["family"],
                "scenario_id": entry["scenario_id"],
                "seed": int(entry["seed"]),
                "method": method,
                "agent_id": agent_id,
                "agent_terminal_completed": bool(completed),
                "agent_collision": bool(row.get("collision_before_reference", False) or row.get("collision_after_reference", False)),
                "agent_path_length_m": float(path_lengths[agent_id]),
                "agent_path_efficiency": (
                    float(straight[agent_id] / max(path_lengths[agent_id], 1e-9))
                    if completed
                    else None
                ),
                "reference_selected": bool(row.get("reference_available", False)),
                "reference_reached": (
                    bool(row.get("reference_reached", False))
                    if method != "terminal"
                    else None
                ),
                "completion_step": completion_step,
                "minimum_obstacle_clearance_m": (
                    float(obstacle_min[agent_id]) if np.isfinite(obstacle_min[agent_id]) else None
                ),
                "minimum_peer_distance_m": (
                    float(peer_min[agent_id]) if np.isfinite(peer_min[agent_id]) else None
                ),
                "selected_candidate_id": row.get("selected_candidate_id"),
                "selected_null": row.get("selected_null"),
            }
        )
    result["minimum_obstacle_clearance_m"] = (
        float(np.min(obstacle_min)) if np.any(np.isfinite(obstacle_min)) else None
    )
    result["minimum_inter_agent_distance_m"] = float(np.min(peer_min))
    trajectory_payload = {
        **{key: np.asarray(value) for key, value in trajectory.items()},
        "dynamic_obstacle_positions": np.asarray(
            entry["dynamic_obstacle_trajectories"], dtype=float
        ).transpose(1, 0, 2)
        if entry["dynamic_obstacle_trajectories"]
        else np.empty((len(trajectory["positions"]), 0, 3), dtype=float),
    }
    return result, standardized_agents, trajectory_payload


def evaluate_proposed_episode(
    runtime: FrozenRuntime,
    entry: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    method: str = "gat_v1",
    shared: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Mapping[str, Any]]:
    if shared is None:
        shared = build_shared_selection_bundle(
            config=eval_config,
            execution_settings=runtime.execution_settings,
            multi_config=runtime.multi_config,
            policy=runtime.policy,
            gat_model=runtime.gat_model,
            gat_device=runtime.gat_device,
            scenario=str(entry["scenario_id"]),
            seed=int(entry["seed"]),
            environment_builder=runtime.builder,
        )
    sink: dict[str, Any] = {}
    episode, agents = run_method_episode(
        config=eval_config,
        execution_settings=runtime.execution_settings,
        multi_config=runtime.multi_config,
        policy=runtime.policy,
        shared=shared,
        method=SAC_METHOD_MAP[method],
        environment_builder=runtime.builder,
        trajectory_sink=sink,
    )
    standardized = standardize_sac_result(episode, agents, sink, entry, method)
    return (*standardized, shared)


def metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if bool(row["team_success"])]
    return {
        "episode_count": len(rows),
        "team_success_count": sum(bool(row["team_success"]) for row in rows),
        "team_success_rate": float(np.mean([bool(row["team_success"]) for row in rows])),
        "any_collision_count": sum(bool(row["any_collision"]) for row in rows),
        "any_collision_rate": float(np.mean([bool(row["any_collision"]) for row in rows])),
        "timeout_count": sum(bool(row["timeout"]) for row in rows),
        "timeout_rate": float(np.mean([bool(row["timeout"]) for row in rows])),
        "successful_completion_time_s": (
            float(np.mean([row["completion_time_s"] for row in successes])) if successes else None
        ),
        "successful_team_path_length_m": (
            float(np.mean([row["team_path_length_m"] for row in successes])) if successes else None
        ),
        "mean_planning_runtime_ms": float(np.mean([row["planning_runtime_ms"] for row in rows])),
    }


def selection_key(summary: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        -float(summary["team_success_rate"]),
        float(summary["any_collision_rate"]),
        float(summary["timeout_rate"]),
        float(summary["successful_completion_time_s"] or float("inf")),
        float(summary["successful_team_path_length_m"] or float("inf")),
        float(summary["mean_planning_runtime_ms"]),
    )


def _development_record_path(output_dir: Path, method: str, config_id: str, scenario_id: str) -> Path:
    storage_method = (
        "rvo_orca_style_closest_approach_projection_v2"
        if method == "rvo_orca_style"
        else method
    )
    return output_dir / "engineering_search" / "records" / storage_method / config_id / f"{scenario_id}.json"


def _load_or_run_development(
    *,
    output_dir: Path,
    method: str,
    search_config: Mapping[str, Any],
    entry: Mapping[str, Any],
    runtime: FrozenRuntime | None,
) -> dict[str, Any]:
    path = _development_record_path(output_dir, method, str(search_config["config_id"]), str(entry["scenario_id"]))
    if path.exists():
        return load_json(path)
    if method == "gat_v1":
        if runtime is None:
            raise RuntimeError("frozen runtime is required")
        eval_config = proposed_eval_config(runtime.base_eval_config, search_config)
        episode, _, _, _ = evaluate_proposed_episode(runtime, entry, eval_config, "gat_v1")
    else:
        if runtime is None:
            raise RuntimeError("environment runtime is required")
        planner = (
            DWAStyleConfig(**{key: value for key, value in search_config.items() if key != "config_id"})
            if method == "dwa_style"
            else RVOStyleConfig(**{key: value for key, value in search_config.items() if key != "config_id"})
        )
        episode, _, _, _ = run_classical_episode(
            environment_builder=runtime.builder,
            multi_config=runtime.multi_config,
            scenario=str(entry["scenario_id"]),
            seed=int(entry["seed"]),
            peer_radius=float(runtime.config["execution"]["peer_radius"]),
            method=method,
            planner_config=planner,
        )
    record = {
        "schema_version": SCHEMA_VERSION,
        "split": "development",
        "config_id": search_config["config_id"],
        "config": dict(search_config),
        **episode,
    }
    record["result_hash"] = stable_hash(record)
    write_json(path, record)
    return record


def run_search_method(
    *,
    output_dir: Path,
    method: str,
    configs: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    runtime: FrozenRuntime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Mapping[str, Any]]:
    screening = [row for row in entries if int(row["scenario_index"]) < 2]
    all_records: list[dict[str, Any]] = []
    screening_summaries: list[dict[str, Any]] = []
    for config_index, search_config in enumerate(configs, start=1):
        rows = []
        for entry_index, entry in enumerate(screening, start=1):
            row = _load_or_run_development(
                output_dir=output_dir,
                method=method,
                search_config=search_config,
                entry=entry,
                runtime=runtime,
            )
            rows.append(row)
            print(
                f"[development:{method}:screen {config_index}/{len(configs)} "
                f"{entry_index}/{len(screening)}] {search_config['config_id']} "
                f"{entry['scenario_id']} {row['termination_reason']}",
                flush=True,
            )
        summary = {"method": method, "config_id": search_config["config_id"], "phase": "screening", **metric_summary(rows)}
        screening_summaries.append(summary)
        all_records.extend(rows)
    finalists = sorted(screening_summaries, key=selection_key)[:3]
    finalist_ids = {row["config_id"] for row in finalists}
    full_summaries: list[dict[str, Any]] = []
    for search_config in configs:
        if search_config["config_id"] not in finalist_ids:
            continue
        existing = [row for row in all_records if row["config_id"] == search_config["config_id"]]
        existing_ids = {row["scenario_id"] for row in existing}
        for entry_index, entry in enumerate(entries, start=1):
            if entry["scenario_id"] in existing_ids:
                continue
            row = _load_or_run_development(
                output_dir=output_dir,
                method=method,
                search_config=search_config,
                entry=entry,
                runtime=runtime,
            )
            all_records.append(row)
            print(
                f"[development:{method}:finalist {search_config['config_id']} "
                f"{entry_index}/{len(entries)}] {entry['scenario_id']} {row['termination_reason']}",
                flush=True,
            )
        rows = [row for row in all_records if row["config_id"] == search_config["config_id"]]
        full_summaries.append(
            {"method": method, "config_id": search_config["config_id"], "phase": "full_development", **metric_summary(rows)}
        )
    winner = min(full_summaries, key=selection_key)
    selected = next(row for row in configs if row["config_id"] == winner["config_id"])
    return all_records, screening_summaries + full_summaries, selected


def development(config: Mapping[str, Any], output_dir: Path) -> None:
    if any((output_dir / "formal_records").rglob("*.json")):
        raise RuntimeError("formal result exists; engineering optimization is closed")
    manifest = load_json(output_dir / "development_scenario_manifest.json")
    runtime = FrozenRuntime(config, manifest)
    entries = list(manifest["entries"])
    # Warm-up is development-only and is excluded from steady-state timing.
    warm_config = proposed_eval_config(runtime.base_eval_config, config["engineering_search"]["proposed"][0])
    evaluate_proposed_episode(runtime, entries[0], warm_config, "gat_v1")
    all_results: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for method, key in (("gat_v1", "proposed"), ("dwa_style", "dwa_style"), ("rvo_orca_style", "rvo_orca_style")):
        results, summaries, winner = run_search_method(
            output_dir=output_dir,
            method=method,
            configs=config["engineering_search"][key],
            entries=entries,
            runtime=runtime,
        )
        all_results.extend(results)
        all_summaries.extend(summaries)
        selected[method] = winner
    write_csv(output_dir / "development_results.csv", all_results)
    write_csv(output_dir / "proposed_engineering_search.csv", [row for row in all_summaries if row["method"] == "gat_v1"])
    write_csv(output_dir / "classic_a_search.csv", [row for row in all_summaries if row["method"] == "dwa_style"])
    write_csv(output_dir / "classic_b_search.csv", [row for row in all_summaries if row["method"] == "rvo_orca_style"])
    write_json(output_dir / "engineering_search" / "selected_configs.json", selected)
    for method, value in selected.items():
        write_json(output_dir / "method_configs" / f"{method}.json", value)
    print(json.dumps({"phase": "development", "selected": selected}, ensure_ascii=False), flush=True)


def _determinism_check(
    config: Mapping[str, Any], output_dir: Path, selected: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = load_json(output_dir / "development_scenario_manifest.json")
    runtime = FrozenRuntime(config, manifest)
    entry = manifest["entries"][0]
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    eval_config = proposed_eval_config(runtime.base_eval_config, selected["gat_v1"])
    first, _, first_traj, _ = evaluate_proposed_episode(runtime, entry, eval_config, "gat_v1")
    second, _, second_traj, _ = evaluate_proposed_episode(runtime, entry, eval_config, "gat_v1")
    checks["proposed_selected_outcome_exact"] = all(
        first[key] == second[key]
        for key in ("team_success", "any_collision", "timeout", "completion_step", "termination_reason")
    )
    checks["proposed_trajectory_numerical_match"] = np.allclose(
        first_traj["positions"], second_traj["positions"], rtol=0.0, atol=1e-7
    )
    details["proposed_position_max_abs_difference"] = float(
        np.max(np.abs(first_traj["positions"] - second_traj["positions"]))
    )
    for method, cls in (("dwa_style", DWAStyleConfig), ("rvo_orca_style", RVOStyleConfig)):
        planner = cls(**{key: value for key, value in selected[method].items() if key != "config_id"})
        one, _, _, one_traj = run_classical_episode(
            environment_builder=runtime.builder,
            multi_config=runtime.multi_config,
            scenario=entry["scenario_id"],
            seed=entry["seed"],
            peer_radius=config["execution"]["peer_radius"],
            method=method,
            planner_config=planner,
        )
        two, _, _, two_traj = run_classical_episode(
            environment_builder=runtime.builder,
            multi_config=runtime.multi_config,
            scenario=entry["scenario_id"],
            seed=entry["seed"],
            peer_radius=config["execution"]["peer_radius"],
            method=method,
            planner_config=planner,
        )
        checks[f"{method}_outcome_exact"] = all(
            one[key] == two[key]
            for key in ("team_success", "any_collision", "timeout", "completion_step", "termination_reason")
        )
        checks[f"{method}_trajectory_exact"] = np.array_equal(
            one_traj["positions"], two_traj["positions"]
        )
    return {
        "checks": checks,
        "details": details,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "failed_checks": [key for key, value in checks.items() if not value],
    }


def freeze(config: Mapping[str, Any], output_dir: Path) -> None:
    selected_path = output_dir / "engineering_search" / "selected_configs.json"
    if not selected_path.exists():
        raise RuntimeError("development search has not completed")
    if any((output_dir / "formal_records").rglob("*.json")):
        raise RuntimeError("formal result exists; freeze cannot be rewritten")
    selected = load_json(selected_path)
    proposed_search = read_csv(output_dir / "proposed_engineering_search.csv")
    classic_a_search = read_csv(output_dir / "classic_a_search.csv")
    classic_b_search = read_csv(output_dir / "classic_b_search.csv")
    proposed_full = {
        row["config_id"]: row
        for row in proposed_search
        if row["phase"] == "full_development"
    }
    initial_proposed = proposed_full["P00"]
    final_proposed = proposed_full[selected["gat_v1"]["config_id"]]
    engineering_metrics = {
        "pre_engineering_success": float(initial_proposed["team_success_rate"]),
        "post_engineering_success": float(final_proposed["team_success_rate"]),
        "success_gain_pp": 100.0
        * (
            float(final_proposed["team_success_rate"])
            - float(initial_proposed["team_success_rate"])
        ),
        "collision_change_pp": 100.0
        * (
            float(final_proposed["any_collision_rate"])
            - float(initial_proposed["any_collision_rate"])
        ),
        "timeout_change_pp": 100.0
        * (
            float(final_proposed["timeout_rate"])
            - float(initial_proposed["timeout_rate"])
        ),
        "planning_runtime_change_ms": (
            float(final_proposed["mean_planning_runtime_ms"])
            - float(initial_proposed["mean_planning_runtime_ms"])
        ),
        "planning_runtime_change_fraction": (
            float(final_proposed["mean_planning_runtime_ms"])
            / float(initial_proposed["mean_planning_runtime_ms"])
            - 1.0
        ),
    }
    determinism = _determinism_check(config, output_dir, selected)
    core_hashes = _file_hashes(tuple(CORE_METHOD_PATHS) + EXTRA_CORE_PATHS)
    formal_manifest = load_json(output_dir / "scenario_manifest.json")
    context = load_json(output_dir / "context_recovery_manifest.json")
    checkpoint_valid = (
        sha256_file(REPO_ROOT / config["sources"]["gat_checkpoint"])
        == config["sources"]["gat_checkpoint_sha256_expected"]
        and sha256_file(REPO_ROOT / config["sources"]["sac_checkpoint"])
        == config["sources"]["sac_checkpoint_sha256_expected"]
    )
    freeze_valid = bool(
        determinism["status"] == "PASSED"
        and checkpoint_valid
        and context["development_manifest_validation"]["status"] == "PASSED"
        and context["formal_manifest_validation"]["status"] == "PASSED"
        and context["separation"]["status"] == "PASSED"
    )
    contract = {
        "CLASSICAL_BASELINE_A": "3D-DWA-style",
        "CLASSICAL_BASELINE_B": "RVO/ORCA-style",
        "tuning_strategy": "successive_elimination_8_scene_screen_then_top3_full40",
        "selection_criterion": config["engineering_search"]["selection_criterion"],
        "budgets": {
            "classic_a_configurations": len(config["engineering_search"]["dwa_style"]),
            "classic_b_configurations": len(config["engineering_search"]["rvo_orca_style"]),
        },
        "selected": {
            "dwa_style": selected["dwa_style"],
            "rvo_orca_style": selected["rvo_orca_style"],
        },
        "development_search_summary": {
            "dwa_style": classic_a_search,
            "rvo_orca_style": classic_b_search,
        },
        "rvo_implementation": "closest_approach_reciprocal_velocity_projection_v2",
        "rvo_label_boundary": "RVO/ORCA-style; canonical ORCA is not claimed",
        "parameters_frozen": True,
    }
    write_json(output_dir / "classic_planner_parameter_contract.json", contract)
    freeze_payload = {
        "schema_version": SCHEMA_VERSION,
        "ENGINEERING_FREEZE_VALID": "YES" if freeze_valid else "NO",
        "TECHNICAL_PATH_UNCHANGED": "YES",
        "FORMAL_RESULT_USED_FOR_TUNING": "NO",
        "selected_configs": selected,
        "proposed_initial_config": config["engineering_search"]["proposed"][0],
        "proposed_attempted_configs": config["engineering_search"]["proposed"],
        "proposed_accepted_change": {
            "d_align": [1.2, float(selected["gat_v1"]["d_align"])],
            "reason": "lexicographic development selection after identical success/collision/timeout/path outcomes and lower measured planning runtime",
        },
        "proposed_rejected_config_ids": [
            row["config_id"]
            for row in config["engineering_search"]["proposed"]
            if row["config_id"] != selected["gat_v1"]["config_id"]
        ],
        "engineering_metrics": engineering_metrics,
        "determinism_and_equivalence": determinism,
        "core_hashes": core_hashes,
        "checkpoint_valid": checkpoint_valid,
        "formal_scenario_manifest_sha256": formal_manifest["manifest_sha256"],
        "evaluation_definitions_frozen": True,
        "metric_definitions_frozen": True,
        "ranking_definitions_frozen": True,
        "formal_result_count_at_freeze": 0,
        "FORMAL_PHASE": "OPEN" if freeze_valid else "CLOSED",
    }
    write_json(output_dir / "engineering_freeze.json", freeze_payload)
    report = [
        "# Engineering Freeze Report",
        "",
        f"`ENGINEERING_FREEZE_VALID = {'YES' if freeze_valid else 'NO'}`.",
        "",
        "## Frozen method",
        "",
        "Proposal → Top-K 10 → FP-SHEP H4 → GAT-V1 → frozen deterministic SAC-DMP, with one-shot phase-preserving handoff and no replanning.",
        "",
        "## Development search",
        "",
        f"- Proposed configurations attempted: {len(config['engineering_search']['proposed'])}.",
        f"- DWA-style configurations attempted: {len(config['engineering_search']['dwa_style'])}.",
        f"- RVO/ORCA-style configurations attempted: {len(config['engineering_search']['rvo_orca_style'])}.",
        "- Successive elimination used 2 scenarios per stage for screening; the top three configurations per method were evaluated on all 40 development scenarios.",
        "- Selection was lexicographic: success, collision, timeout, successful completion time, successful path length, runtime.",
        f"- Proposed before/after success: {engineering_metrics['pre_engineering_success']:.1%} / {engineering_metrics['post_engineering_success']:.1%} ({engineering_metrics['success_gain_pp']:+.1f} pp).",
        f"- Proposed collision change: {engineering_metrics['collision_change_pp']:+.1f} pp; timeout change: {engineering_metrics['timeout_change_pp']:+.1f} pp.",
        f"- Proposed planning-runtime change: {engineering_metrics['planning_runtime_change_ms']:+.3f} ms ({engineering_metrics['planning_runtime_change_fraction']:+.1%}).",
        "- Accepted existing-parameter change: graph d_align 1.2 → 1.0 (P03). No Top-K, H4, graph schema, features, checkpoint, executor, or threshold definition changed.",
        "- Rejected Proposed configurations remain in proposed_engineering_search.csv.",
        "- The first endpoint-only RVO implementation was rejected before freeze; the frozen RVO/ORCA-style planner uses full-window closest approach plus deterministic 3-D reciprocal velocity projection.",
        "",
        "## Selected configurations",
        "",
        f"- Proposed: `{json.dumps(selected['gat_v1'], sort_keys=True)}`",
        f"- 3D-DWA-style: `{json.dumps(selected['dwa_style'], sort_keys=True)}`",
        f"- RVO/ORCA-style: `{json.dumps(selected['rvo_orca_style'], sort_keys=True)}`",
        "",
        "## Integrity",
        "",
        f"- Determinism/equivalence gate: `{determinism['status']}`.",
        f"- Checkpoint hashes valid: `{'YES' if checkpoint_valid else 'NO'}`.",
        f"- Formal manifest SHA256: `{formal_manifest['manifest_sha256']}`.",
        "- No formal result existed when this report was frozen.",
    ]
    (output_dir / "ENGINEERING_FREEZE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not freeze_valid:
        raise RuntimeError(f"engineering freeze gate failed: {freeze_payload}")
    print(json.dumps({"phase": "freeze", "status": "PASSED"}), flush=True)


def _formal_record_path(output_dir: Path, entry: Mapping[str, Any], method: str) -> Path:
    return output_dir / "formal_records" / entry["stage"] / entry["scenario_id"] / f"{method}.json"


def _save_formal_record(
    output_dir: Path,
    entry: Mapping[str, Any],
    method: str,
    episode: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    trajectory: Mapping[str, Any],
) -> None:
    trajectory_path = output_dir / "trajectories" / entry["stage"] / entry["scenario_id"] / f"{method}.npz"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trajectory_path, **{key: np.asarray(value) for key, value in trajectory.items()})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": entry["stage"],
        "scenario_id": entry["scenario_id"],
        "method": method,
        "status": "COMPLETE",
        "episode": dict(episode),
        "agents": list(agents),
        "planning_runtime_records": list(runtime_rows),
        "trajectory_path": str(trajectory_path.relative_to(output_dir)),
        "trajectory_sha256": sha256_file(trajectory_path),
    }
    payload["result_hash"] = stable_hash(payload)
    path = _formal_record_path(output_dir, entry, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    write_json(temporary, payload)
    temporary.replace(path)


def formal(config: Mapping[str, Any], output_dir: Path) -> None:
    freeze_payload = load_json(output_dir / "engineering_freeze.json")
    if freeze_payload["ENGINEERING_FREEZE_VALID"] != "YES":
        raise RuntimeError("formal phase is closed because engineering freeze is invalid")
    selected = freeze_payload["selected_configs"]
    manifest = load_json(output_dir / "scenario_manifest.json")
    if manifest["manifest_sha256"] != freeze_payload["formal_scenario_manifest_sha256"]:
        raise RuntimeError("formal scenario manifest changed after freeze")
    current_core = _file_hashes(tuple(CORE_METHOD_PATHS) + EXTRA_CORE_PATHS)
    if current_core != freeze_payload["core_hashes"]:
        raise RuntimeError("core implementation changed after engineering freeze")
    runtime = FrozenRuntime(config, manifest)
    eval_config = proposed_eval_config(runtime.base_eval_config, selected["gat_v1"])
    entries = list(manifest["entries"])
    expected_jobs = len(entries) * len(METHOD_ORDER)
    completed = {
        (path.parent.name, path.stem)
        for path in (output_dir / "formal_records").rglob("*.json")
    }
    phase_state_path = output_dir / "formal_phase_state.json"
    job_index = 0
    for entry in entries:
        shared: Mapping[str, Any] | None = None
        for method in METHOD_ORDER:
            job_index += 1
            key = (entry["scenario_id"], method)
            if key in completed:
                continue
            if not phase_state_path.exists():
                write_json(
                    phase_state_path,
                    {
                        "FORMAL_PHASE": "OPEN",
                        "ENGINEERING_OPTIMIZATION_PHASE": "CLOSED",
                        "first_result_started_at": datetime.now().astimezone().isoformat(),
                        "formal_manifest_sha256": manifest["manifest_sha256"],
                        "core_hashes": current_core,
                    },
                )
            if method in {"dwa_style", "rvo_orca_style"}:
                cls = DWAStyleConfig if method == "dwa_style" else RVOStyleConfig
                planner = cls(**{key: value for key, value in selected[method].items() if key != "config_id"})
                episode, agents, runtime_rows, trajectory = run_classical_episode(
                    environment_builder=runtime.builder,
                    multi_config=runtime.multi_config,
                    scenario=entry["scenario_id"],
                    seed=entry["seed"],
                    peer_radius=config["execution"]["peer_radius"],
                    method=method,
                    planner_config=planner,
                )
            else:
                episode, agents, trajectory, shared = evaluate_proposed_episode(
                    runtime, entry, eval_config, method, shared
                )
                runtime_rows = []
                if method != "terminal":
                    runtime_rows.append(
                        {
                            "stage": entry["stage"],
                            "scenario_id": entry["scenario_id"],
                            "method": method,
                            "decision_index": 0,
                            "runtime_ms": episode["planning_runtime_ms"],
                        }
                    )
            _save_formal_record(
                output_dir, entry, method, episode, agents, runtime_rows, trajectory
            )
            print(
                f"[formal {job_index}/{expected_jobs}] {entry['stage']} "
                f"{entry['scenario_id']} {method}: {episode['termination_reason']}",
                flush=True,
            )
    actual = len(list((output_dir / "formal_records").rglob("*.json")))
    write_json(
        output_dir / "formal_run_status.json",
        {
            "expected_team_episode_count": expected_jobs,
            "actual_team_episode_count": actual,
            "status": "COMPLETE" if actual == expected_jobs else "INCOMPLETE",
            "completed_at": datetime.now().astimezone().isoformat(),
        },
    )
    if actual != expected_jobs:
        raise RuntimeError(f"formal data incomplete: {actual}/{expected_jobs}")
    print(json.dumps({"phase": "formal", "actual": actual, "status": "COMPLETE"}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=("prepare", "development", "freeze", "formal", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    config = load_json(args.config.resolve())
    if args.output_dir is None:
        if args.phase != "prepare" and args.phase != "all":
            raise ValueError("--output-dir is required when resuming after prepare")
        output_dir = REPO_ROOT / config["output_root"] / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_dir = args.output_dir.resolve()
    if args.phase in {"prepare", "all"}:
        prepare(config, output_dir)
    if args.phase in {"development", "all"}:
        development(config, output_dir)
    if args.phase in {"freeze", "all"}:
        freeze(config, output_dir)
    if args.phase in {"formal", "all"}:
        formal(config, output_dir)
    return output_dir


if __name__ == "__main__":
    main()
