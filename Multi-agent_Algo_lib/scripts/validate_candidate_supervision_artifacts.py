"""Validate serialized candidate-supervision artifacts without training."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))


def _coerce(value: str) -> Any:
    if value == "":
        return None
    if value == "null":
        return "null"
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return [
            {key: _coerce(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def validate_artifacts(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = _read_csv(root / "manifest.csv")
    graph_rows = _read_csv(root / "graph_records.csv")
    mapping_rows = _read_csv(root / "class_mapping.csv")
    state_rows = _read_csv(root / "state_groups.csv")
    class_rows = _read_csv(root / "candidate_class_records.csv")
    if int(config["H_preview"]) != 4:
        raise AssertionError("serialized config changed formal H_preview")
    if config.get("diagnostic_preview_excluded_from_graph") is not True:
        raise AssertionError("diagnostic preview exclusion is not explicit")
    if config.get("selection_scope") != "train_validation_only":
        raise AssertionError("selection scope is not train+validation only")
    if len(manifest) != len(graph_rows) * len(config["H_label"]):
        raise AssertionError("manifest does not contain one row per graph/H_label")
    if len({(row["sample_id"], int(row["H_label"])) for row in manifest}) != len(manifest):
        raise AssertionError("duplicate sample/H_label manifest rows")

    split_seeds = {
        split: {int(seed) for seed in seeds}
        for split, seeds in config["split_seeds"].items()
    }
    state_group_splits: dict[str, set[str]] = defaultdict(set)
    for row in state_rows:
        split = str(row["split"])
        seed = int(row["seed"])
        if seed not in split_seeds[split]:
            raise AssertionError("state group is assigned to the wrong seed split")
        state_group_splits[str(row["state_group_id"])].add(split)
        if int(row["timestep"]) not in {int(value) for value in config["state_sample_timesteps"]}:
            raise AssertionError("unexpected sampled timestep")
    if any(len(values) != 1 for values in state_group_splits.values()):
        raise AssertionError("state group leaks across dataset splits")
    for row in manifest:
        expected_eligible = str(row["split"]) in {"train", "validation"}
        if bool(row["selection_eligible"]) != expected_eligible:
            raise AssertionError("manifest selection eligibility is inconsistent")
        if bool(row["test_used_for_selection"]):
            raise AssertionError("test split was marked as selection input")
        for key in ("graph_path", "label_path", "rollout_path", "state_snapshot_path"):
            if not (root / str(row[key])).is_file():
                raise AssertionError(f"missing artifact referenced by {key}")

    mapping_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mapping_rows:
        mapping_by_sample[str(row["sample_id"])].append(row)
    graph_horizons: set[int] = set()
    for index, row in enumerate(graph_rows):
        sample_id = str(row["sample_id"])
        graph = torch.load(
            root / str(row["graph_path"]),
            map_location="cpu",
            weights_only=False,
        )
        graph_horizons.add(int(graph.graph_metadata["H"]))
        if tuple(graph["null"].x.shape) != (1, 5):
            raise AssertionError("null node feature schema changed")
        if int(graph["proposal"].num_nodes) != int(row["K_actual"]):
            raise AssertionError("graph proposal count does not match K_actual")
        if tuple(graph.proposal_node_to_candidate_id) != tuple(range(int(row["K_actual"]))):
            raise AssertionError("proposal-node candidate mapping is not contiguous")
        if any("diagnostic" in str(key).lower() for key in graph.keys()):
            raise AssertionError("diagnostic H_label data leaked into graph")
        mappings = sorted(mapping_by_sample[sample_id], key=lambda item: int(item["class_index"]))
        if [int(item["class_index"]) for item in mappings] != list(range(int(row["class_count"]))):
            raise AssertionError("class mapping is not contiguous")
        if mappings[0]["class_kind"] != "null" or mappings[0]["proposal_node_index"] is not None:
            raise AssertionError("class zero is not an external null branch")
        for class_index, mapping in enumerate(mappings[1:], start=1):
            if int(mapping["proposal_node_index"]) != class_index - 1:
                raise AssertionError("class-to-proposal mapping is inconsistent")
    if graph_horizons != {4}:
        raise AssertionError("not all serialized graphs use H_preview=4")

    for row in manifest:
        label = json.loads((root / str(row["label_path"])).read_text(encoding="utf-8"))
        if int(label["H_preview_formal"]) != 4:
            raise AssertionError("label points to a non-H4 formal graph")
        if not label["diagnostic_preview_excluded_from_graph"]:
            raise AssertionError("diagnostic preview graph exclusion is false")
        if not label["diagnostic_preview_excluded_from_formal_J_preview"]:
            raise AssertionError("diagnostic preview formal-quality exclusion is false")
        if label["final_supervision_target_frozen"]:
            raise AssertionError("dataset incorrectly freezes a final target")
        for key in (
            "formal_J_preview_3", "formal_J_preview_4",
            "provisional_primary_target", "companion_target",
        ):
            values = np.asarray(label[key], dtype=float)
            if values.shape != (int(row["class_count"]),) or not np.all(np.isfinite(values)):
                raise AssertionError(f"invalid finite target vector: {key}")
        with np.load(root / str(row["rollout_path"]), allow_pickle=False) as bundle:
            if int(bundle["formal_preview_positions"].shape[1]) != 5:
                raise AssertionError("formal preview trajectory is not H=4 plus initial state")
            if int(bundle["J_target_3"].shape[0]) != int(row["class_count"]):
                raise AssertionError("rollout target class dimension mismatch")
            if bundle["real_positions"].dtype == object:
                raise AssertionError("rollout bundle contains object arrays")

    class_splits: dict[str, set[str]] = defaultdict(set)
    for row in class_rows:
        class_splits[str(row["state_group_id"])].add(str(row["split"]))
    if any(len(values) != 1 for values in class_splits.values()):
        raise AssertionError("candidate labels leak state groups across splits")
    return {
        "artifact_root": str(root),
        "state_group_count": len(state_rows),
        "ego_graph_count": len(graph_rows),
        "label_manifest_count": len(manifest),
        "candidate_class_record_count": len(class_rows),
        "formal_graph_horizons": sorted(graph_horizons),
        "null_node_shape": [1, 5],
        "state_group_split_leakage": False,
        "diagnostic_preview_leaked_into_graph": False,
        "test_used_for_selection": False,
        "object_array_serialization_used": False,
        "validation_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_artifacts(args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
