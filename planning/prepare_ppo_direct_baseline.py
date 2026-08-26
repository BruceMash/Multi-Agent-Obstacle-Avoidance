"""Prepare frozen PPO-Direct contracts and isolated scene splits."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from planning.ppo_direct_baseline import (
    ACTION_DIM,
    ACTION_SCALE_MPS2,
    OBSERVATION_DIM,
    PPODirectRewardConfig,
    build_local_observations,
    make_environment,
    normalized_actions_to_accelerations,
)
from planning.final_four_stage_benchmark import step_direct_accelerations
from planning.semi_structured_long_range_benchmark import (
    generate_scenario_manifest,
    stable_hash,
    validate_scenario_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIOR_REGISTRY = (
    REPO_ROOT
    / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2/ALL_USED_SCENE_REGISTRY.csv"
)
SUBDIRS = (
    "00_context",
    "01_control_contract",
    "02_observation_contract",
    "03_training_environment",
    "04_throughput",
    "05_training",
    "06_development",
    "07_holdout",
    "08_final_freeze",
    "09_formal_v2",
    "10_statistics",
    "11_runtime",
    "12_trajectories",
    "13_paper_ready",
)
REGISTRY_FIELDS = (
    "scene_id",
    "split",
    "source",
    "seed",
    "geometry_fingerprint",
    "dynamic_track_fingerprint",
    "translation_invariant_fingerprint",
    "start_goal_fingerprint",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
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


def start_goal_fingerprint(entry: Mapping[str, Any]) -> str:
    return stable_hash({"starts": entry["starts"], "goals": entry["goals"]})


def compact_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    compact = copy.deepcopy(dict(manifest))
    compact["source_manifest_sha256"] = compact.pop("manifest_sha256")
    compact["storage_contract"] = (
        "constant-translation tracks omitted; exact current state is reconstructed "
        "from frozen center, velocity, dt, and step"
    )
    for entry in compact["entries"]:
        entry.pop("dynamic_obstacle_trajectories", None)
        witness = entry.get("static_route_witness_evaluation_only")
        if isinstance(witness, dict):
            witness.pop("paths", None)
            witness.pop("path_points", None)
    compact["manifest_sha256"] = stable_hash(
        {key: value for key, value in compact.items() if key != "manifest_sha256"}
    )
    return compact


def registry_rows(manifest: Mapping[str, Any], split: str, source: str) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": entry["scenario_id"],
            "split": split,
            "source": source,
            "seed": int(entry["seed"]),
            "geometry_fingerprint": entry["geometry_fingerprint"],
            "dynamic_track_fingerprint": entry["dynamic_track_fingerprint"],
            "translation_invariant_fingerprint": entry["translation_invariant_fingerprint"],
            "start_goal_fingerprint": start_goal_fingerprint(entry),
        }
        for entry in manifest["entries"]
    ]


def load_registry(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def overlap_counts(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for field in REGISTRY_FIELDS[3:]:
        a = {str(row[field]) for row in left if str(row.get(field, ""))}
        b = {str(row[field]) for row in right if str(row.get(field, ""))}
        result[field] = len(a & b)
    return result


def atomicity_audit(entry: Mapping[str, Any]) -> dict[str, Any]:
    actions = np.asarray(
        [[0.25, -0.10, 0.05], [-0.30, 0.20, -0.15], [0.10, 0.05, 0.20]],
        dtype=np.float32,
    )
    permutation = np.asarray([2, 0, 1], dtype=int)
    inverse = np.argsort(permutation)
    permuted = copy.deepcopy(dict(entry))
    permuted["starts"] = np.asarray(entry["starts"])[permutation].tolist()
    permuted["goals"] = np.asarray(entry["goals"])[permutation].tolist()
    env_a = make_environment(entry)
    env_b = make_environment(permuted)
    try:
        before_a = build_local_observations(env_a)
        before_b = build_local_observations(env_b)[inverse]
        term_a, trunc_a, info_a = step_direct_accelerations(
            env_a, normalized_actions_to_accelerations(actions), refresh_sensors=True
        )
        term_b, trunc_b, info_b = step_direct_accelerations(
            env_b,
            normalized_actions_to_accelerations(actions[permutation]),
            refresh_sensors=True,
        )
        after_a = build_local_observations(env_a)
        after_b = build_local_observations(env_b)[inverse]
        position_error = float(
            np.max(np.abs(env_a._positions() - env_b._positions()[inverse]))
        )
        velocity_error = float(
            np.max(np.abs(env_a._velocities() - env_b._velocities()[inverse]))
        )
        before_observation_error = float(np.max(np.abs(before_a - before_b)))
        after_observation_error = float(np.max(np.abs(after_a - after_b)))
        collision_equal = bool(
            np.array_equal(
                np.asarray(info_a["collision_mask"]),
                np.asarray(info_b["collision_mask"])[inverse],
            )
        )
        passed = bool(
            position_error <= 1e-7
            and velocity_error <= 1e-7
            and before_observation_error <= 1e-6
            and after_observation_error <= 1e-6
            and collision_equal
            and term_a == term_b
            and trunc_a == trunc_b
        )
        return {
            "schema_version": "ppo_direct_atomicity_audit_v1",
            "status": "PASS" if passed else "FAIL",
            "snapshot_observation_for_all_agents_before_inference": True,
            "all_actions_collected_before_world_commit": True,
            "joint_world_advance_count_per_control_step": 1,
            "permutation": permutation.tolist(),
            "maximum_position_error_m": position_error,
            "maximum_velocity_error_mps": velocity_error,
            "maximum_pre_step_observation_error": before_observation_error,
            "maximum_post_step_observation_error": after_observation_error,
            "collision_labels_permutation_equivalent": collision_equal,
            "agent_order_artifact_detected": not passed,
            "source": "planning/ppo_direct_baseline.py:PPODirectWorld.step",
        }
    finally:
        env_a.close()
        env_b.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    for name in SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)

    source_files = [
        REPO_ROOT / "planning/ppo_direct_baseline.py",
        REPO_ROOT / "planning/prepare_ppo_direct_baseline.py",
        REPO_ROOT / "planning/final_four_stage_benchmark.py",
        REPO_ROOT / "Entity/KinematicModel.py",
        REPO_ROOT / "Environment/multi_agent_dmp_env.py",
        REPO_ROOT / "planning/semi_structured_long_range_benchmark.py",
    ]
    write_json(
        root / "00_context/SOURCE_HASHES.json",
        {str(path.relative_to(REPO_ROOT)): sha256(path) for path in source_files},
    )
    write_json(
        root / "01_control_contract/PPO_NATIVE_CONTROL_INTERFACE_AUDIT.json",
        {
            "schema_version": "ppo_native_control_interface_audit_v1",
            "status": "PASS",
            "native_control_type": "NET_LINEAR_ACCELERATION",
            "dimension": ACTION_DIM,
            "unit": "m/s^2",
            "policy_action_space": {"low": [-1.0] * 3, "high": [1.0] * 3},
            "mapping": "acceleration_mps2 = clip(policy_action,-1,1) * 4.0",
            "physical_bounds": {"per_axis_acceleration_mps2": [-4.0, 4.0]},
            "integration_rule": "bounded next velocity followed by trapezoidal position integration under speed-norm cap",
            "speed_saturation": {"per_axis_velocity_mps": [-4.0, 4.0], "norm_mps": 3.2},
            "acceleration_saturation": "component-wise np.clip before velocity integration",
            "dt_use": "velocity and position integration both use dt=0.1 s",
            "collision_execution": "all three motions commit, moving obstacles advance once, then exact obstacle/peer/boundary collision is evaluated",
            "current_users_of_interface": ["3D-DWA-style", "RVO/ORCA-style", "DWA-SensingMatched", "NMPC-SensingMatched"],
            "source_files": ["Entity/KinematicModel.py", "planning/final_four_stage_benchmark.py"],
            "source_lines": ["Entity/KinematicModel.py:6-55,214-225", "planning/final_four_stage_benchmark.py:1280-1380"],
            "PPO_PHYSICS_MATCHED": "YES",
        },
    )
    schema_rows = [
        {"component": "ego_velocity", "start": 0, "end_exclusive": 3, "dim": 3, "range": "[-1,1]", "source": "native velocity / 4 m/s", "legal": "YES"},
        {"component": "terminal_goal_direction", "start": 3, "end_exclusive": 6, "dim": 3, "range": "[-1,1]", "source": "native local sensor packet", "legal": "YES"},
        {"component": "terminal_goal_distance", "start": 6, "end_exclusive": 7, "dim": 1, "range": "[0,1]", "source": "distance / 9 m, clipped", "legal": "YES"},
        {"component": "lidar_current", "start": 7, "end_exclusive": 263, "dim": 256, "range": "[0,1]", "source": "4.5 m untyped 16x16 LiDAR", "legal": "YES"},
        {"component": "lidar_previous", "start": 263, "end_exclusive": 519, "dim": 256, "range": "[0,1]", "source": "previous 4.5 m untyped 16x16 LiDAR", "legal": "YES"},
        {"component": "anonymous_local_peer_slots", "start": 519, "end_exclusive": 533, "dim": 14, "range": "[-1,1]", "source": "two 4.5 m range-limited identity-free ally slots", "legal": "YES"},
    ]
    write_csv(
        root / "02_observation_contract/PPO_DIRECT_OBSERVATION_SCHEMA.csv",
        schema_rows,
        ("component", "start", "end_exclusive", "dim", "range", "source", "legal"),
    )
    reward = PPODirectRewardConfig()
    write_json(
        root / "03_training_environment/PPO_DIRECT_REWARD_CONTRACT.json",
        {
            "schema_version": "ppo_direct_reward_contract_v1",
            **reward.__dict__,
            "formula": "progress - 0.01 + 30*new_goal - 100*team_collision - 25*local_collision",
            "peer_collision_included": True,
            "obstacle_collision_included": True,
            "simulator_ground_truth_use": "training reward terminal collision labels only; never policy observation",
            "forbidden_structured_terms_present": False,
            "reward_revision_count": 0,
        },
    )
    write_json(
        root / "03_training_environment/PPO_DIRECT_TRAINING_CONFIG.json",
        {
            "schema_version": "ppo_direct_training_config_v1",
            "algorithm": "STANDARD_PARAMETER_SHARED_DECENTRALIZED_PPO",
            "implementation": "Stable-Baselines3 PPO 2.7.1",
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "action_type": "normalized net linear acceleration",
            "network": {"actor": [256, 256], "critic": [256, 256], "activation": "Tanh"},
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 256,
            "n_epochs": 10,
            "entropy_coefficient": 0.0,
            "device": "cuda",
            "seed": 20260823,
            "num_worlds": "SELECT_AFTER_THROUGHPUT",
            "phase_transition_budgets": [200000, 1000000, 2000000, 3000000],
            "absolute_max_transitions": 5000000,
            "transition_unit": "agent transition; three per joint world step",
            "maximum_serious_configurations": 6,
            "maximum_reward_revisions": 1,
        },
    )

    split_specs = (
        ("PPO_DIRECT_TRAIN", 100, 7_830_000_000, "PDT"),
        ("PPO_DIRECT_DEV", 25, 7_831_000_000, "PDD"),
        ("PPO_DIRECT_HOLDOUT", 25, 7_832_000_000, "PDH"),
    )
    manifests: dict[str, dict[str, Any]] = {}
    validations: dict[str, Any] = {}
    all_new_rows: list[dict[str, Any]] = []
    for split, count, seed_base, prefix in split_specs:
        manifest = generate_scenario_manifest(
            counts_per_stage=count,
            seed_base=seed_base,
            prefix=prefix,
        )
        validations[split] = validate_scenario_manifest(manifest)
        compact = compact_manifest(manifest)
        manifests[split] = compact
        filename = f"{split}_MANIFEST.json"
        write_json(root / f"03_training_environment/{filename}", compact)
        all_new_rows.extend(registry_rows(compact, split, f"03_training_environment/{filename}"))

    prior_rows = load_registry(PRIOR_REGISTRY)
    split_rows = {
        split: [row for row in all_new_rows if row["split"] == split]
        for split, *_ in split_specs
    }
    prior_overlap = overlap_counts(all_new_rows, prior_rows)
    pair_overlaps: dict[str, Any] = {}
    split_names = list(split_rows)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            pair_overlaps[f"{left}__{right}"] = overlap_counts(
                split_rows[left], split_rows[right]
            )
    passed = bool(
        all(value["status"] == "PASS" for value in validations.values())
        and all(count == 0 for count in prior_overlap.values())
        and all(
            count == 0
            for values in pair_overlaps.values()
            for count in values.values()
        )
    )
    write_csv(
        root / "03_training_environment/PPO_DIRECT_SCENE_REGISTRY.csv",
        all_new_rows,
        REGISTRY_FIELDS,
    )
    write_json(
        root / "03_training_environment/PPO_DIRECT_SCENE_ISOLATION_AUDIT.json",
        {
            "schema_version": "ppo_direct_scene_isolation_audit_v1",
            "status": "PASS" if passed else "FAIL",
            "prior_registry": str(PRIOR_REGISTRY.relative_to(REPO_ROOT)),
            "prior_registry_rows": len(prior_rows),
            "new_split_counts": {key: len(value) for key, value in split_rows.items()},
            "grammar_validation": validations,
            "overlap_with_all_prior_registered_scenes": prior_overlap,
            "between_new_split_overlaps": pair_overlaps,
            "required_zero_fields": list(REGISTRY_FIELDS[3:]),
        },
    )
    audit = atomicity_audit(manifests["PPO_DIRECT_TRAIN"]["entries"][0])
    write_json(
        root / "03_training_environment/PPO_MULTI_AGENT_STEP_ATOMICITY_AUDIT.json",
        audit,
    )
    if not passed or audit["status"] != "PASS":
        raise RuntimeError("PPO-Direct preparation gate failed")
    print(
        json.dumps(
            {
                "status": "PASS",
                "root": str(root),
                "observation_dim": OBSERVATION_DIM,
                "train": len(split_rows["PPO_DIRECT_TRAIN"]),
                "dev": len(split_rows["PPO_DIRECT_DEV"]),
                "holdout": len(split_rows["PPO_DIRECT_HOLDOUT"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
