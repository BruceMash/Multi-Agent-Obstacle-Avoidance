"""Independently reconcile frozen GAT-rescue scene-file manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.semi_structured_long_range_benchmark import stable_hash  # noqa: E402


DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
)
SPLITS = {
    "training": "02_recurrent_dataset/GAT_RS_TRAIN_SCENE_MANIFEST.json",
    "development": "07_development/GAT_RS_DEV_SCENE_MANIFEST.json",
    "holdout": "08_holdout/GAT_RS_HOLDOUT_SCENE_MANIFEST.json",
}
FIELDS = (
    "seed",
    "geometry_fingerprint",
    "dynamic_track_fingerprint",
    "translation_invariant_fingerprint",
    "start_goal_fingerprint",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def scene_values(scene: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(scene["seed"]),
        "geometry_fingerprint": str(scene["geometry_fingerprint"]),
        "dynamic_track_fingerprint": str(scene["dynamic_track_fingerprint"]),
        "translation_invariant_fingerprint": str(
            scene["translation_invariant_fingerprint"]
        ),
        "start_goal_fingerprint": stable_hash(
            {"starts": scene["starts"], "goals": scene["goals"]}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    errors: list[str] = []
    split_sets: dict[str, dict[str, set[Any]]] = {}
    summaries: dict[str, Any] = {}

    for split, relative in SPLITS.items():
        manifest_path = root / relative
        manifest = load_json(manifest_path)
        entries = manifest["entries"]
        current = {name: set() for name in FIELDS}
        stage_counts: dict[str, int] = {}
        family_stage_counts: dict[str, int] = {}
        total_bytes = 0
        for ordinal, index in enumerate(entries, start=1):
            scene_path = root / index["scenario_file"]
            if not scene_path.is_file():
                errors.append(f"missing_scene:{split}:{index['scenario_id']}")
                continue
            observed_sha = sha256_file(scene_path)
            if observed_sha != index["scenario_file_sha256"]:
                errors.append(f"scene_hash:{split}:{index['scenario_id']}")
            total_bytes += scene_path.stat().st_size
            scene = load_json(scene_path)
            if scene["scenario_id"] != index["scenario_id"]:
                errors.append(f"scene_id:{split}:{index['scenario_id']}")
            values = scene_values(scene)
            for name in FIELDS:
                if values[name] != index[name]:
                    errors.append(
                        f"index_value:{split}:{index['scenario_id']}:{name}"
                    )
                current[name].add(values[name])
            stage = str(scene["stage"])
            family = str(scene["family"])
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            key = f"{stage}/{family}"
            family_stage_counts[key] = family_stage_counts.get(key, 0) + 1
            tracks = scene["dynamic_obstacle_trajectories"]
            if len(tracks) != len(scene["dynamic_obstacles"]):
                errors.append(f"track_count:{split}:{index['scenario_id']}")
            if any(len(track) != int(scene["max_steps"]) + 1 for track in tracks):
                errors.append(f"track_length:{split}:{index['scenario_id']}")
            if ordinal % 100 == 0:
                print(f"[independent-reconcile] {split}: {ordinal}/{len(entries)}", flush=True)

        expected = int(manifest["unique_scenario_count"])
        if len(entries) != expected:
            errors.append(f"manifest_count:{split}")
        internal = {name: len(entries) - len(current[name]) for name in FIELDS}
        if any(internal.values()):
            errors.append(f"internal_duplicate:{split}:{internal}")
        split_sets[split] = current
        summaries[split] = {
            "manifest_file_sha256": sha256_file(manifest_path),
            "scene_count": len(entries),
            "scene_bytes": total_bytes,
            "stage_counts": stage_counts,
            "family_stage_counts": family_stage_counts,
            "internal_duplicates": internal,
            "all_scene_hashes_match": not any(
                item.startswith(f"scene_hash:{split}:") for item in errors
            ),
            "all_index_values_match_scene_files": not any(
                item.startswith(f"index_value:{split}:") for item in errors
            ),
            "all_dynamic_track_lengths_match": not any(
                item.startswith(f"track_length:{split}:") for item in errors
            ),
        }

    pairwise: dict[str, dict[str, int]] = {}
    names = list(split_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pairwise[f"{left}__{right}"] = {
                name: len(split_sets[left][name] & split_sets[right][name])
                for name in FIELDS
            }
    if any(any(values.values()) for values in pairwise.values()):
        errors.append(f"pairwise_overlap:{pairwise}")

    registry_path = root / "02_recurrent_dataset/ALL_USED_SCENE_REGISTRY.csv"
    with registry_path.open("r", newline="", encoding="utf-8-sig") as handle:
        registry = list(csv.DictReader(handle))
    new_registry = [row for row in registry if row["split"] in SPLITS]
    expected_new = sum(summary["scene_count"] for summary in summaries.values())
    if len(new_registry) != expected_new:
        errors.append(f"registry_count:{len(new_registry)}:{expected_new}")
    formal_files = list((root / "10_formal_v2").rglob("*"))
    formal_files = [
        path
        for path in formal_files
        if path.is_file() and path.name.lower() != "readme.md"
    ]
    if formal_files:
        errors.append("formal_v2_directory_not_empty")

    result = {
        "schema_version": "gat_recurrent_split_independent_reconciliation_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "splits": summaries,
        "pairwise_overlap": pairwise,
        "combined_registry_sha256": sha256_file(registry_path),
        "combined_registry_total_rows": len(registry),
        "combined_registry_new_rows": len(new_registry),
        "formal_v2_file_count": len(formal_files),
        "formal_v2_generated": False if not formal_files else "VIOLATION",
        "performance_rows_read": 0,
        "old_formal_v1_episode_rows_read": 0,
    }
    atomic_json(
        root / "02_recurrent_dataset/independent_split_reconciliation.json",
        result,
    )
    print(json.dumps({"status": result["status"], "errors": errors}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
