"""Freeze independent long-range training, validation, and development splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.semi_structured_long_range_benchmark import (  # noqa: E402
    FAMILY_ORDER,
    STAGE_ORDER,
    generate_scenario_manifest,
    json_ready,
    validate_scenario_manifest,
)


DEFAULT_ARTIFACT = REPO_ROOT / "artifacts/semi_structured_long_range_main_benchmark/20260820_193228"
SPLITS = {
    "training": {
        "counts_per_stage": 50,
        "seed_base": 1_300_000_000,
        "prefix": "TRAIN_LR_",
        "directory": "07_training",
        "filename": "training_manifest.json",
    },
    "validation": {
        "counts_per_stage": 20,
        "seed_base": 1_400_000_000,
        "prefix": "VAL_LR_",
        "directory": "07_training",
        "filename": "validation_manifest.json",
    },
    "development": {
        "counts_per_stage": 50,
        "seed_base": 1_500_000_000,
        "prefix": "DEV_LR_",
        "directory": "08_development",
        "filename": "development_manifest.json",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_registry(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "scene_id", "split", "stage", "family", "task_pattern", "seed",
        "geometry_fingerprint", "translation_invariant_fingerprint",
        "dynamic_track_fingerprint", "performance_episode_count",
        "eligible_for_training", "eligible_for_development", "eligible_for_formal",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def registry_row(entry: Mapping[str, Any], split: str) -> dict[str, Any]:
    return {
        "scene_id": entry["scenario_id"],
        "split": split,
        "stage": entry["stage"],
        "family": entry["family"],
        "task_pattern": entry["task_pattern"],
        "seed": int(entry["seed"]),
        "geometry_fingerprint": entry["geometry_fingerprint"],
        "translation_invariant_fingerprint": entry["translation_invariant_fingerprint"],
        "dynamic_track_fingerprint": entry["dynamic_track_fingerprint"],
        "performance_episode_count": 0,
        "eligible_for_training": split == "training",
        "eligible_for_development": split == "development",
        "eligible_for_formal": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    registry_path = root / "00_context/ALL_USED_SCENE_REGISTRY.csv"
    existing = read_registry(registry_path)
    prior_seeds = {int(row["seed"]) for row in existing}
    prior_geometry = {str(row["geometry_fingerprint"]) for row in existing}
    prior_translation = {str(row["translation_invariant_fingerprint"]) for row in existing}
    manifests: dict[str, dict[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}
    new_rows: list[dict[str, Any]] = []

    for split, spec in SPLITS.items():
        print(f"[split-freeze] generating {split}", flush=True)
        manifest = generate_scenario_manifest(
            counts_per_stage=int(spec["counts_per_stage"]),
            seed_base=int(spec["seed_base"]),
            prefix=str(spec["prefix"]),
        )
        validation = validate_scenario_manifest(manifest)
        if validation["status"] != "PASS":
            raise RuntimeError(f"{split} manifest failed: {validation['errors']}")
        entries = manifest["entries"]
        seeds = {int(row["seed"]) for row in entries}
        geometry = {str(row["geometry_fingerprint"]) for row in entries}
        translation = {str(row["translation_invariant_fingerprint"]) for row in entries}
        overlaps = {
            "seed": len(seeds & prior_seeds),
            "geometry": len(geometry & prior_geometry),
            "translation": len(translation & prior_translation),
        }
        if any(overlaps.values()):
            raise RuntimeError(f"{split} overlaps earlier registry: {overlaps}")
        validation["overlap_with_all_earlier_registered_splits"] = overlaps
        validation["family_count_per_stage"] = {
            stage: {
                family: sum(
                    row["stage"] == stage and row["family"] == family for row in entries
                )
                for family in FAMILY_ORDER
            }
            for stage in STAGE_ORDER
        }
        output = root / str(spec["directory"]) / str(spec["filename"])
        write_json(output, manifest)
        write_json(output.with_name(output.stem + "_validation.json"), validation)
        manifest["file_sha256"] = sha256_file(output)
        manifests[split] = manifest
        validations[split] = validation
        new_rows.extend(registry_row(row, split) for row in entries)
        prior_seeds.update(seeds)
        prior_geometry.update(geometry)
        prior_translation.update(translation)
        print(f"[split-freeze] {split} count={len(entries)} PASS", flush=True)

    by_id = {str(row["scene_id"]): row for row in existing}
    for row in new_rows:
        by_id[str(row["scene_id"])] = row
    write_csv(registry_path, list(by_id.values()))
    cross = {
        "schema_version": "long_range_training_development_split_freeze_v1",
        "status": "PASS",
        "formal_manifest_generated": False,
        "splits": {
            split: {
                "scenario_count": len(manifest["entries"]),
                "seed_base": manifest["seed_base"],
                "manifest_semantic_sha256": manifest["manifest_sha256"],
                "manifest_file_sha256": manifest["file_sha256"],
            }
            for split, manifest in manifests.items()
        },
        "pairwise_overlap": {},
        "development_structure": "4 stages x 5 families x 10 scenes = 200",
        "training_or_development_result_count_at_freeze": 0,
    }
    for left_index, left in enumerate(manifests):
        for right in list(manifests)[left_index + 1 :]:
            left_rows = manifests[left]["entries"]
            right_rows = manifests[right]["entries"]
            cross["pairwise_overlap"][f"{left}__{right}"] = {
                "seed": len({row["seed"] for row in left_rows} & {row["seed"] for row in right_rows}),
                "geometry": len({row["geometry_fingerprint"] for row in left_rows} & {row["geometry_fingerprint"] for row in right_rows}),
                "translation": len({row["translation_invariant_fingerprint"] for row in left_rows} & {row["translation_invariant_fingerprint"] for row in right_rows}),
            }
    if any(any(values.values()) for values in cross["pairwise_overlap"].values()):
        raise RuntimeError("new split overlap detected")
    write_json(root / "00_context/training_development_split_freeze.json", cross)

    tuning_budget = [
        {"method": "Proposed", "screening_configs": 12, "full_dev_finalists": 3, "training_runs_max": 4, "selection_metric": "success/collision/stage/family/runtime lexicographic"},
        {"method": "R-ERR+FP-SHEP", "screening_configs": 8, "full_dev_finalists": 2, "training_runs_max": 0, "selection_metric": "success/collision/stage/family/runtime lexicographic"},
        {"method": "DWA-SensingMatched", "screening_configs": 12, "full_dev_finalists": 3, "training_runs_max": 0, "selection_metric": "success/collision/runtime lexicographic"},
        {"method": "DWA-FullState", "screening_configs": 12, "full_dev_finalists": 3, "training_runs_max": 0, "selection_metric": "success/collision/runtime lexicographic"},
        {"method": "Waypoint-PPO", "screening_configs": 0, "full_dev_finalists": 0, "training_runs_max": 1, "selection_metric": "readiness gate before performance"},
        {"method": "Direct/Proposal/FP-SHEP/OneShot ablations", "screening_configs": 0, "full_dev_finalists": 1, "training_runs_max": 0, "selection_metric": "inherit final common SAC and applicable upper parameters"},
    ]
    budget_path = root / "09_baseline_tuning/METHOD_TUNING_BUDGET.csv"
    fields = tuple(tuning_budget[0].keys())
    temporary = budget_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tuning_budget)
    temporary.replace(budget_path)
    write_json(
        root / "09_baseline_tuning/HYPERPARAMETER_SEARCH_SPACE.json",
        {
            "schema_version": "long_range_predevelopment_search_space_v1",
            "status": "FROZEN_BEFORE_DEVELOPMENT_RESULTS",
            "development_screen_subset": "first two scenes per stage/family (40 total), fixed before results",
            "proposal": {"safe_radius_m": [0.9, 1.15, 1.35], "s_max_m": [0.8, 1.05, 1.3], "top_k": [8, 10, 14]},
            "fp_shep": {"H_preview_steps": [4, 6, 8], "d_safe_m": [0.6, 0.75, 0.9], "normalization_scale_multiplier": [0.8, 1.0, 1.25]},
            "gat": {"checkpoint": ["historical_v1", "long_range_finetuned"], "learning_rate": [0.00003, 0.0001, 0.0003], "hidden_architecture_change": False},
            "rerr": {"T_rep_s": [2.5, 5.0, 7.5], "T_dwell_s": [0.5, 1.0, 1.5], "p_min_mps": [0.02, 0.02666666666666667, 0.05], "h_rep_m": [0.25, 0.35, 0.5], "d_hand_m": [0.2, 0.25, 0.35]},
            "sac": {"architecture": "unchanged", "learning_rate": [0.00003, 0.0001, 0.0003], "fine_tune_steps": [250000, 500000, 1000000], "action_semantics_change": False},
            "dwa": {"horizon_s": [0.8, 1.2, 1.8], "velocity_samples_per_axis": [3, 5], "preferred_speed_mps": [1.6, 2.2, 2.8], "clearance_weight": [1.0, 1.5, 2.0], "goal_weight": [1.5, 2.0, 2.5], "collision_buffer_m": [0.08, 0.15, 0.25]},
            "formal_data_available_at_freeze": False,
        },
    )
    write_json(root / "07_training/training_split_freeze.json", cross)
    print(json.dumps({"status": "PASS", "counts": {key: len(value["entries"]) for key, value in manifests.items()}}), flush=True)


if __name__ == "__main__":
    main()
