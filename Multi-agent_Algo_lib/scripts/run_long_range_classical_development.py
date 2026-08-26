"""Resumable development evaluation for long-range classical baselines.

This runner consumes only the frozen 200-scenario development manifest.  It is
deliberately incapable of opening or creating the formal manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as _pandas  # noqa: F401  # initialize Windows pyarrow before torch users


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.final_four_stage_benchmark import DWAStyleConfig, run_classical_episode  # noqa: E402
from planning.semi_structured_long_range_benchmark import WORKSPACE_BOUNDS, json_ready  # noqa: E402
from planning.sensing_matched_classical import run_sensing_matched_episode  # noqa: E402
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.run_long_range_contract_pilot import (  # noqa: E402
    LongRangeManifestEnvironmentBuilder,
)


SCHEMA_VERSION = "semi_structured_long_range_classical_development_v1"
SUPPORTED_METHODS = {"dwa_sensing_matched", "dwa_fullstate"}
SOURCE_PATHS = (
    "Environment/multi_agent_dmp_env.py",
    "planning/final_four_stage_benchmark.py",
    "planning/sensing_matched_classical.py",
    "planning/semi_structured_long_range_benchmark.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_contract_pilot.py",
    "Multi-agent_Algo_lib/scripts/run_long_range_classical_development.py",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    raw = load_json(path)
    base_name = raw.get("base_config")
    if base_name is None:
        return raw
    base_path = (REPO_ROOT / str(base_name)).resolve()
    if base_path == path.resolve():
        raise ValueError("classical development config cannot inherit itself")
    return _deep_merge(load_config(base_path), raw)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    root = (REPO_ROOT / str(config["artifact_root"])).resolve()
    manifest = root / str(config["development_manifest"])
    output = root / str(config["output_subdir"])
    return root, manifest, output


def selected_ids(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    available = [str(row["scenario_id"]) for row in manifest["entries"]]
    requested = config.get("development_scenario_ids")
    result = available if requested is None else [str(value) for value in requested]
    if not result or len(result) != len(set(result)):
        raise ValueError("development scenario ids must be non-empty and unique")
    missing = sorted(set(result) - set(available))
    if missing:
        raise ValueError(f"unknown development scenario ids: {missing}")
    return result


def build_runtime(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[Any, Any]:
    sensor = config["sensor"]
    base = build_single_distribution_multi_config(
        num_agents=int(config["num_agents"]), max_steps=int(config["max_steps"])
    )
    multi_config = replace(
        base,
        workspace_bounds=tuple(tuple(row) for row in WORKSPACE_BOUNDS),
        randomize_start_goal=False,
        start_position_bounds=None,
        goal_position_bounds=None,
        min_start_goal_distance=0.0,
        sensing_radius=float(sensor["range_m"]),
        sensor_azimuth_bins=int(sensor["azimuth_bins"]),
        sensor_elevation_bins=int(sensor["elevation_bins"]),
        sensor_goal_distance_clip=2.0 * float(sensor["range_m"]),
    )
    return LongRangeManifestEnvironmentBuilder(manifest, config), multi_config


def planner_config(config: Mapping[str, Any]) -> DWAStyleConfig:
    values = {key: value for key, value in config["planner"].items() if key != "method"}
    return DWAStyleConfig(**values)


def prepare(config_path: Path) -> None:
    raw_config = load_json(config_path)
    config = load_config(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected classical development schema")
    if str(config["planner"]["method"]) not in SUPPORTED_METHODS:
        raise ValueError("unsupported classical method")
    root, manifest_path, output = resolve_paths(config)
    manifest = load_json(manifest_path)
    if len(manifest["entries"]) != 200:
        raise RuntimeError("development manifest must contain exactly 200 scenarios")
    scenario_ids = selected_ids(config, manifest)
    output.mkdir(parents=True, exist_ok=True)
    source_files = [
        *SOURCE_PATHS,
        str(config_path.relative_to(REPO_ROOT)).replace("\\", "/"),
    ]
    if raw_config.get("base_config") is not None:
        source_files.append(str(raw_config["base_config"]))
    source_hashes = {
        path: sha256_file(REPO_ROOT / path)
        for path in source_files
    }
    atomic_json(
        output / "development_freeze.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "configuration_id": config["configuration_id"],
            "method": config["planner"]["method"],
            "planner": asdict(planner_config(config)),
            "scenario_count": len(scenario_ids),
            "scenario_ids": scenario_ids,
            "development_manifest_sha256": sha256_file(manifest_path),
            "source_hashes": source_hashes,
            "formal_data_used": False,
        },
    )
    print(json.dumps({"phase": "prepare", "status": "PASS", "scenario_count": len(scenario_ids)}))


def run(config_path: Path, shard_count: int, shard_index: int) -> None:
    config = load_config(config_path)
    _, manifest_path, output = resolve_paths(config)
    freeze = load_json(output / "development_freeze.json")
    manifest = load_json(manifest_path)
    entries = {str(row["scenario_id"]): row for row in manifest["entries"]}
    builder, multi_config = build_runtime(config, manifest)
    method = str(config["planner"]["method"])
    dwa = planner_config(config)
    records = output / "episode_records"
    records.mkdir(parents=True, exist_ok=True)
    for ordinal, scenario_id in enumerate(freeze["scenario_ids"]):
        if ordinal % int(shard_count) != int(shard_index):
            continue
        destination = records / f"{scenario_id}.json"
        if destination.exists():
            continue
        row = entries[scenario_id]
        print(f"[long-range-classical] start {config['configuration_id']} {scenario_id}", flush=True)
        try:
            kwargs = dict(
                environment_builder=builder,
                multi_config=multi_config,
                scenario=scenario_id,
                seed=int(row["seed"]),
                peer_radius=float(config["peer_radius"]),
                planner_config=dwa,
            )
            if method == "dwa_sensing_matched":
                episode, agents, runtime, trajectory = run_sensing_matched_episode(
                    method=method, **kwargs
                )
            else:
                episode, agents, runtime, trajectory = run_classical_episode(
                    method="dwa_style", **kwargs
                )
                episode["method"] = method
                for agent in agents:
                    agent["method"] = method
                for decision in runtime:
                    decision["method"] = method
            payload = {
                "schema_version": SCHEMA_VERSION,
                "configuration_id": config["configuration_id"],
                "scenario": {
                    "scenario_id": scenario_id,
                    "seed": int(row["seed"]),
                    "stage": row["stage"],
                    "family": row["family"],
                    "task_pattern": row["task_pattern"],
                    "environment_fingerprint": row["environment_fingerprint"],
                },
                "episode": episode,
                "agents": agents,
                "runtime_summary": {
                    "decision_count": len(runtime),
                    "mean_decision_ms": float(np.mean([item["runtime_ms"] for item in runtime]))
                    if runtime
                    else 0.0,
                },
                "software_error": False,
            }
            if bool(config.get("save_trajectories", False)):
                trajectory_path = records / f"{scenario_id}_trajectory.npz"
                np.savez_compressed(trajectory_path, **trajectory)
                payload["trajectory_file"] = trajectory_path.name
            atomic_json(destination, payload)
            print(
                f"[long-range-classical] complete {scenario_id} "
                f"success={int(episode['team_success'])} collision={int(episode['any_collision'])}",
                flush=True,
            )
        except Exception as exc:  # preserve every attempted development scene
            atomic_json(
                destination,
                {
                    "schema_version": SCHEMA_VERSION,
                    "configuration_id": config["configuration_id"],
                    "scenario": row,
                    "software_error": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


def aggregate(rows: Sequence[Mapping[str, Any]], scope: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "n": len(rows),
        "team_success_rate": float(np.mean([bool(row["team_success"]) for row in rows])),
        "collision_rate": float(np.mean([bool(row["any_collision"]) for row in rows])),
        "obstacle_collision_rate": float(
            np.mean([bool(row["obstacle_collision"]) for row in rows])
        ),
        "static_obstacle_collision_rate": float(
            np.mean([bool(row.get("static_obstacle_collision", False)) for row in rows])
        ),
        "dynamic_obstacle_collision_rate": float(
            np.mean([bool(row.get("dynamic_obstacle_collision", False)) for row in rows])
        ),
        "inter_agent_collision_rate": float(
            np.mean([bool(row["inter_agent_collision"]) for row in rows])
        ),
        "boundary_collision_rate": float(
            np.mean([bool(row.get("boundary_collision", False)) for row in rows])
        ),
        "timeout_rate": float(np.mean([bool(row["timeout"]) for row in rows])),
        "mean_compute_ms": float(np.mean([float(row["planning_runtime_ms"]) for row in rows])),
    }


def finalize(config_path: Path) -> None:
    config = load_config(config_path)
    _, _, output = resolve_paths(config)
    freeze = load_json(output / "development_freeze.json")
    expected = list(freeze["scenario_ids"])
    payloads = {
        path.stem: load_json(path)
        for path in (output / "episode_records").glob("*.json")
        if not path.stem.endswith("_trajectory")
    }
    missing = sorted(set(expected) - set(payloads))
    unexpected = sorted(set(payloads) - set(expected))
    errors = [key for key, value in payloads.items() if bool(value.get("software_error"))]
    if missing or unexpected or errors:
        raise RuntimeError(f"incomplete classical development: missing={missing}, unexpected={unexpected}, errors={errors}")
    rows: list[dict[str, Any]] = []
    for scenario_id in expected:
        payload = payloads[scenario_id]
        row = dict(payload["episode"])
        row.update(
            {
                "configuration_id": config["configuration_id"],
                "task_pattern": payload["scenario"]["task_pattern"],
                "environment_fingerprint": payload["scenario"]["environment_fingerprint"],
            }
        )
        rows.append(row)
    summaries = [aggregate(rows, "overall")]
    for stage in ("stage_1", "stage_2", "stage_3", "stage_4"):
        summaries.append(aggregate([row for row in rows if row["stage"] == stage], stage))
    write_csv(output / "development_team_results.csv", rows)
    write_csv(output / "development_summary.csv", summaries)
    atomic_json(
        output / "development_reconciliation.json",
        {
            "status": "PASS",
            "completed_scenarios": len(rows),
            "expected_scenarios": len(expected),
            "missing_scenario_ids": missing,
            "unexpected_scenario_ids": unexpected,
            "software_error_count": len(errors),
            "overall": summaries[0],
            "formal_data_used": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "finalize"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    if arguments.phase == "prepare":
        prepare(config_path)
    elif arguments.phase == "run":
        run(config_path, arguments.shard_count, arguments.shard_index)
    else:
        finalize(config_path)


if __name__ == "__main__":
    main()
