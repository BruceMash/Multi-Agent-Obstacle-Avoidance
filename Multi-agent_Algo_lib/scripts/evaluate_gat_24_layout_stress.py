"""Evaluate the frozen Stage-I GAT on the frozen 24-layout stress set.

The module is an evaluation-only adapter.  It restores geometry from the
already frozen manifest, builds one current candidate/preview/graph bundle per
layout, and never uses historical selections or outcomes as online inputs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import threading
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# Pandas is imported transitively by the historical SAC runner.  PyArrow is an
# unused optional dependency in this evaluator and the local binary is not ABI
# compatible with the evaluation environment.
sys.modules.setdefault("pyarrow", None)

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Entity.static_obstacles import StaticSphereObstacle  # noqa: E402
from Environment.multi_agent_dmp_env import MultiAgentDMPEnv  # noqa: E402
from Guidance.reference_point_proposal_demo import ProposalConfig  # noqa: E402
from planning.gat.stage1_training import (  # noqa: E402
    load_model_checkpoint,
    resolve_device,
)
from planning.geometry_generalization_scenarios import (  # noqa: E402
    FROZEN_LAYOUT_TABLE_SHA256,
    build_scenario_manifest,
    stable_hash as geometry_stable_hash,
)
from planning.goal_semantics_diagnosis import VARIANT_A, VARIANT_D  # noqa: E402
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    scoped_historical_preview_and_multi_agent_transition,
)
from planning.pre_gat_closed_loop import FPSHEPOnlineScoreSpec  # noqa: E402
from scripts.evaluate_actor_dmp_goal_semantics import (  # noqa: E402
    run_variant_episode,
    write_csv,
    write_json,
)
import scripts.evaluate_gat_closed_loop as closed_loop  # noqa: E402
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_single_policy_multi_agent import _build_environment  # noqa: E402
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


SCHEMA_VERSION = "gat_24_layout_stress_v1"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/evaluation/gat_24_layout_stress.json"
METHOD_TERMINAL = closed_loop.METHOD_TERMINAL
METHOD_PROPOSAL = closed_loop.METHOD_PROPOSAL
METHOD_FP_SHEP = closed_loop.METHOD_FP_SHEP
METHOD_GAT = closed_loop.METHOD_GAT
METHOD_ORDER = closed_loop.METHOD_ORDER
METHOD_NAMES = closed_loop.METHOD_NAMES

_PATCH_LOCK = threading.RLock()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _parse_optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _parse_optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _assert_frozen_config(config: Mapping[str, Any]) -> None:
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected stress-test schema")
    exact = {
        "geometry_manifest_sha256_expected": (
            "488098107fde63ac707aa7e173aa4471fc140c84e224293c03fd77b7dc0a501c"
        ),
        "layout_count": 24,
        "layouts_per_family": 4,
        "num_agents": 3,
        "static_obstacle_count": 2,
        "dynamic_obstacle_count": 0,
        "top_k": 10,
        "H_preview": 4,
        "max_steps": 220,
        "handoff_threshold_m": 0.25,
        "forcing_gate": HISTORICAL_GATE_NAME,
        "execution_protocol": "one_shot_temporary_reference_handoff",
        "boundary_mode": "boundary_free",
    }
    for key, expected in exact.items():
        if config[key] != expected:
            raise ValueError(f"frozen setting changed: {key}")
    if list(config["families"]) != list("ABCDEF"):
        raise ValueError("families must remain A..F")
    if tuple(config["methods"]) != METHOD_ORDER:
        raise ValueError("method order changed")
    if not bool(config["peer_spheres"]):
        raise ValueError("peer spheres must remain enabled")
    tolerances = config["compatibility_tolerances"]
    if not bool(tolerances["frozen_before_outcome_evaluation"]):
        raise ValueError("compatibility tolerances must be frozen before outcomes")
    if bool(tolerances["threshold_adjustment_after_outcomes_allowed"]):
        raise ValueError("post-outcome tolerance adjustment is forbidden")
    if any(bool(value) for value in config["strict_exclusions"].values()):
        raise ValueError("strict exclusion flags must remain false")
    representatives = config["execution_equivalence_representatives"]
    if set(representatives) != set("ABCDEF"):
        raise ValueError("execution equivalence must cover all six families")


def _runtime_closed_loop_config(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = copy.deepcopy(dict(config))
    runtime["peer_radius"] = float(config["peer_radius_m"])
    runtime["dt"] = 0.1
    runtime["formal_scenarios"] = [str(config["scenario"])]
    runtime["formal_seeds"] = list(range(1000, 1000 + int(config["layout_count"])))
    runtime["smoke_seeds"] = []
    return runtime


def _build_execution_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = _load_json(REPO_ROOT / str(config["base_execution_config"]))
    settings.update(
        {
            "checkpoint": config["sac_checkpoint"],
            "checkpoint_sha256_expected": config["sac_checkpoint_sha256_expected"],
            "deterministic_policy": True,
            "num_agents": int(config["num_agents"]),
            "max_steps": int(config["max_steps"]),
            "dt": 0.1,
            "peer_radius": float(config["peer_radius_m"]),
            "scenarios": [str(config["scenario"])],
            "seeds": list(range(1000, 1000 + int(config["layout_count"]))),
            "temporary_reference": {
                **settings["temporary_reference"],
                "K_requested": int(config["top_k"]),
                "reached_tolerance_m": float(config["handoff_threshold_m"]),
                "replanning_enabled": False,
            },
            "proposal_config": copy.deepcopy(config["proposal_config"]),
        }
    )
    return settings


def build_manifest_environment_builder(
    manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> Callable[..., tuple[Any, dict[str, Any]]]:
    """Restore exact starts/goals/obstacles from manifest records."""

    by_seed = {int(row["evaluation_seed"]): copy.deepcopy(row) for row in manifest["layouts"]}
    stress_config = dict(config)
    expected_scenario = str(stress_config["scenario"])

    def builder(
        *, config: Any, scenario: str, seed: int, peer_radius: float
    ) -> tuple[Any, dict[str, Any]]:
        if str(scenario) != expected_scenario:
            raise ValueError("unexpected frozen stress scenario")
        if int(seed) not in by_seed:
            raise KeyError(f"unknown frozen layout seed: {seed}")
        if int(config.num_agents) != 3:
            raise ValueError("frozen stress environment requires three agents")
        if not np.isclose(float(peer_radius), 0.3):
            raise ValueError("peer radius differs from manifest")
        record = by_seed[int(seed)]
        obstacles = [
            StaticSphereObstacle(
                center=np.asarray(item["center"], dtype=float),
                radius=float(item["radius_m"]),
                safety_margin=float(item["safety_margin_m"]),
            )
            for item in record["obstacles"]
        ]
        options = {
            "starts": np.asarray(record["starts"], dtype=float).copy(),
            "goals": np.asarray(record["terminal_goals"], dtype=float).copy(),
            "static_obstacles": obstacles,
            "dynamic_obstacles": [],
        }
        env = _build_environment(
            config,
            observation_mode="peer_spheres",
            peer_radius=float(peer_radius),
            training_distribution=False,
            include_boundaries_in_sensor=False,
            terminate_on_boundary_collision=False,
        )
        env.reset(seed=int(seed), options=copy.deepcopy(options))
        restored = {
            "starts": env.starts,
            "terminal_goals": env.goals,
            "obstacles": [
                {
                    "center": item.center,
                    "radius_m": item.radius,
                    "safety_margin_m": item.safety_margin,
                }
                for item in env.static_obstacles
            ],
        }
        expected = {
            "starts": record["starts"],
            "terminal_goals": record["terminal_goals"],
            "obstacles": [
                {
                    "center": item["center"],
                    "radius_m": item["radius_m"],
                    "safety_margin_m": item["safety_margin_m"],
                }
                for item in record["obstacles"]
            ],
        }
        if geometry_stable_hash(restored) != geometry_stable_hash(expected):
            env.close()
            raise RuntimeError("exact manifest geometry restoration failed")
        return env, {
            "scene_type": expected_scenario,
            "scenario_role": stress_config["scenario_role"],
            "scenario_geometry_version": "frozen_manifest_exact_restore_v1",
            "scenario_geometry_hash": str(record["layout_hash"]),
            "seed_set_role": str(record["layout_set_role"]),
            "layout_id": str(record["layout_id"]),
            "family": str(record["family"]),
            "layout_hash": str(record["layout_hash"]),
            "relative_geometry_hash": str(record["relative_geometry_hash"]),
            "observation_mode": "peer_spheres",
            "peer_spheres_enabled": True,
            "include_boundaries_in_sensor": False,
            "terminate_on_boundary_collision": False,
            "static_obstacle_count": 2,
            "dynamic_obstacle_count": 0,
            "geometry_restored_from_frozen_manifest": True,
        }

    return builder


def config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


@contextmanager
def _scoped_closed_loop_environment_builder(builder: Callable[..., Any]):
    """Evaluation-only adapter for the already implemented shared bundle helper."""

    with _PATCH_LOCK:
        original = closed_loop.build_closed_loop_environment
        closed_loop.build_closed_loop_environment = builder
        try:
            yield
        finally:
            closed_loop.build_closed_loop_environment = original
        if closed_loop.build_closed_loop_environment is not original:
            raise RuntimeError("closed-loop environment builder leaked after scope")


@contextmanager
def capture_execution_trace():
    """Capture read-only step traces without changing environment dynamics."""

    trace: list[dict[str, Any]] = []
    with _PATCH_LOCK:
        original = MultiAgentDMPEnv.step

        def wrapped(self: Any, actions: np.ndarray):
            action_copy = np.asarray(actions, dtype=float).copy()
            result = original(self, actions)
            trace.append(
                {
                    "action": action_copy,
                    "position": self._positions().copy(),
                    "velocity": self._velocities().copy(),
                    "phase": np.asarray([item.phase for item in self.dmps], dtype=float),
                    "dmp_goal": np.asarray([item.goal for item in self.dmps], dtype=float),
                    "terminated": bool(result[2]),
                    "truncated": bool(result[3]),
                }
            )
            return result

        MultiAgentDMPEnv.step = wrapped
        try:
            yield trace
        finally:
            MultiAgentDMPEnv.step = original
        if MultiAgentDMPEnv.step is not original:
            raise RuntimeError("step instrumentation leaked after scope")


def _trace_payload(trace: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_jsonable(row) for row in trace]


def _trace_hash(trace: Sequence[Mapping[str, Any]]) -> str:
    return closed_loop.stable_hash(_trace_payload(trace))


def _max_abs_error(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], key: str
) -> float | None:
    if len(left) != len(right):
        return None
    if not left:
        return 0.0
    return float(
        max(
            np.max(
                np.abs(
                    np.asarray(a[key], dtype=float) - np.asarray(b[key], dtype=float)
                )
            )
            for a, b in zip(left, right, strict=True)
        )
    )


def _allclose_optional(
    left: Any, right: Any, *, atol: float, rtol: float
) -> bool:
    if left is None or right is None:
        return left is None and right is None
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    return a.shape == b.shape and bool(
        np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _manifest_audit(
    manifest_path: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    actual_hash = _sha256_file(manifest_path)
    manifest = _load_json(manifest_path)
    families = Counter(str(row["family"]) for row in manifest["layouts"])
    current_manifest = build_scenario_manifest()
    checks = {
        "file_hash_match": actual_hash
        == str(config["geometry_manifest_sha256_expected"]),
        "layout_count": len(manifest["layouts"]) == 24,
        "families": sorted(families) == list("ABCDEF"),
        "four_layouts_per_family": all(families[item] == 4 for item in "ABCDEF"),
        "frozen_layout_table_hash": manifest["frozen_layout_table_sha256"]
        == FROZEN_LAYOUT_TABLE_SHA256,
        "current_table_manifest_exact": geometry_stable_hash(current_manifest)
        == geometry_stable_hash(manifest),
        "all_three_agents": all(len(row["starts"]) == 3 for row in manifest["layouts"]),
        "all_two_static_obstacles": all(
            len(row["obstacles"]) == 2 for row in manifest["layouts"]
        ),
        "all_boundary_free": all(bool(row["boundary_free"]) for row in manifest["layouts"]),
        "all_peer_spheres": all(
            row["peer_sensing_mode"] == "peer_spheres" for row in manifest["layouts"]
        ),
    }
    return manifest, {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "actual_sha256": actual_hash,
        "expected_sha256": config["geometry_manifest_sha256_expected"],
        "family_counts": dict(sorted(families.items())),
    }


def _build_all_shared_bundles(
    *,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    execution_settings: Mapping[str, Any],
    multi_config: Any,
    policy: Any,
    gat_model: torch.nn.Module,
    gat_device: torch.device,
    environment_builder: Callable[..., Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cache: dict[str, dict[str, Any]] = {}
    reconstruction_rows: list[dict[str, Any]] = []
    gat_rows: list[dict[str, Any]] = []
    with _scoped_closed_loop_environment_builder(environment_builder):
        for index, layout in enumerate(manifest["layouts"], start=1):
            layout_id = str(layout["layout_id"])
            shared = closed_loop.build_shared_selection_bundle(
                config=runtime_config,
                execution_settings=execution_settings,
                multi_config=multi_config,
                policy=policy,
                gat_model=gat_model,
                gat_device=gat_device,
                scenario=str(config["scenario"]),
                seed=int(layout["evaluation_seed"]),
            )
            shared["layout_id"] = layout_id
            shared["family"] = str(layout["family"])
            shared["layout_hash"] = str(layout["layout_hash"])
            shared["relative_geometry_hash"] = str(layout["relative_geometry_hash"])
            cache[layout_id] = shared
            reconstruction_rows.extend(
                {"layout_id": layout_id, "family": layout["family"], **row}
                for row in shared["reconstruction_rows"]
            )
            gat_rows.extend(
                {
                    "layout_id": layout_id,
                    "family": layout["family"],
                    "seed": int(layout["evaluation_seed"]),
                    "agent_id": agent_id,
                    "candidate_bundle_hash": shared["candidate_bundle_hash"],
                    "preview_bundle_hash": shared["preview_bundle_hash"],
                    **diagnostic,
                }
                for agent_id, diagnostic in enumerate(shared["gat_diagnostics"])
            )
            print(f"[shared {index}/24] {layout_id}", flush=True)
    return cache, reconstruction_rows, gat_rows


def _historical_schema_complete(
    historical_dir: Path, config: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    episode_rows = _read_csv(historical_dir / "per_episode.csv")
    agent_rows = _read_csv(historical_dir / "per_agent.csv")
    episode_fields = set(episode_rows[0]) if episode_rows else set()
    agent_fields = set(agent_rows[0]) if agent_rows else set()
    missing_episode = sorted(
        set(config["historical_reuse"]["required_episode_fields"]) - episode_fields
    )
    missing_agent = sorted(
        set(config["historical_reuse"]["required_per_agent_fields"]) - agent_fields
    )
    complete = not missing_episode and not missing_agent
    return complete, {
        "required_episode_fields": config["historical_reuse"]["required_episode_fields"],
        "required_per_agent_fields": config["historical_reuse"]["required_per_agent_fields"],
        "missing_episode_fields": missing_episode,
        "missing_per_agent_fields": missing_agent,
        "complete": complete,
    }


def _candidate_and_preview_reproduction_audit(
    *,
    shared_cache: Mapping[str, Mapping[str, Any]],
    historical_dir: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tolerances = config["compatibility_tolerances"]
    catol = float(tolerances["candidate_atol"])
    crtol = float(tolerances["candidate_rtol"])
    patol = float(tolerances["preview_feature_atol"])
    prtol = float(tolerances["preview_feature_rtol"])
    satol = float(tolerances["fp_shep_score_atol"])
    srtol = float(tolerances["fp_shep_score_rtol"])
    bundle_rows = _read_csv(historical_dir / "candidate_bundle.csv")
    selection_rows = [
        row
        for row in _read_csv(historical_dir / "candidate_selection.csv")
        if row["method"] == "fp_shep_top1_one_shot"
    ]
    bundle_index: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in bundle_rows:
        bundle_index[(row["layout_id"], int(row["agent_id"]))].append(row)
    for rows_for_agent in bundle_index.values():
        rows_for_agent.sort(key=lambda row: int(row["candidate_index"]))
    selection_index = {
        (row["layout_id"], int(row["agent_id"])): row for row in selection_rows
    }
    rows: list[dict[str, Any]] = []
    candidate_match = True
    preview_match = True
    score_match = True
    selection_match = True
    maximum_preview_error = 0.0
    maximum_score_error = 0.0
    historical_null_current_positive_infinity_count = 0
    for layout_id, shared in shared_cache.items():
        bundle = shared["candidate_bundle"]
        fp_records = shared["plans"][METHOD_FP_SHEP]["candidate_records"]
        for agent_id, candidates in enumerate(bundle.per_agent):
            historical_bundle = bundle_index[(layout_id, agent_id)]
            historical_bundle_head = historical_bundle[0]
            historical_selection = selection_index[(layout_id, agent_id)]
            historical_points = np.asarray(
                [json.loads(row["candidate_world_point"]) for row in historical_bundle],
                dtype=float,
            )
            historical_scores = np.asarray(
                [float(row["proposal_score"]) for row in historical_bundle],
                dtype=float,
            )
            historical_metadata = [
                json.loads(row["candidate_metadata"]) for row in historical_bundle
            ]
            current_points = np.asarray([item.point for item in candidates], dtype=float)
            current_scores = np.asarray([item.score for item in candidates], dtype=float)
            current_metadata = [item.metadata for item in candidates]
            count_exact = (
                len(candidates) == int(historical_bundle_head["K_t"])
                and len(historical_bundle) == int(historical_bundle_head["K_t"])
            )
            order_exact = [int(item.original_index) for item in candidates] == list(
                range(len(candidates))
            ) and [int(row["original_index"]) for row in historical_bundle] == list(
                range(len(historical_bundle))
            )
            position_match = _allclose_optional(
                current_points, historical_points, atol=catol, rtol=crtol
            )
            coarse_score_match = _allclose_optional(
                current_scores, historical_scores, atol=catol, rtol=crtol
            )
            sector_exact = [
                (int(item["azimuth_index"]), int(item["elevation_index"]))
                for item in current_metadata
            ] == [
                (int(item["azimuth_index"]), int(item["elevation_index"]))
                for item in historical_metadata
            ]
            bundle_hash_exact = (
                shared["candidate_bundle_hash"]
                == historical_bundle_head["candidate_set_hash"]
            )
            candidate_ok = all(
                (
                    count_exact,
                    order_exact,
                    position_match,
                    coarse_score_match,
                    sector_exact,
                    bundle_hash_exact,
                )
            )
            historical_previews = json.loads(
                historical_selection["fp_shep_candidate_records"]
            )
            current_previews = fp_records[agent_id]["fp_shep_candidate_records"]
            preview_count_exact = len(current_previews) == len(historical_previews)
            feature_error = 0.0
            score_error = 0.0
            feature_ok = preview_count_exact
            scores_ok = preview_count_exact
            if preview_count_exact:
                for current, historical in zip(
                    current_previews, historical_previews, strict=True
                ):
                    if int(current["candidate_id"]) != int(historical["candidate_id"]):
                        feature_ok = False
                        scores_ok = False
                        continue
                    current_features = np.asarray(
                        [
                            current["preview_task_progress"],
                            current["preview_min_clearance"],
                            current["preview_max_execution_deviation"],
                            current["preview_terminal_speed"],
                        ],
                        dtype=float,
                    )
                    historical_features = np.asarray(
                        [
                            historical["preview_task_progress"],
                            historical["preview_min_clearance"],
                            historical["preview_max_execution_deviation"],
                            historical["preview_terminal_speed"],
                        ],
                        dtype=float,
                    )
                    for current_value, historical_value in zip(
                        current_features,
                        [
                            historical["preview_task_progress"],
                            historical["preview_min_clearance"],
                            historical["preview_max_execution_deviation"],
                            historical["preview_terminal_speed"],
                        ],
                        strict=True,
                    ):
                        if historical_value is None and np.isposinf(current_value):
                            historical_null_current_positive_infinity_count += 1
                    finite_pair = np.isfinite(current_features) & np.isfinite(
                        historical_features
                    )
                    local_feature_error = (
                        float(
                            np.max(
                                np.abs(
                                    current_features[finite_pair]
                                    - historical_features[finite_pair]
                                )
                            )
                        )
                        if np.any(finite_pair)
                        else 0.0
                    )
                    local_score_error = abs(
                        float(current["fp_shep_online_score"])
                        - float(historical["fp_shep_online_score"])
                    )
                    feature_error = max(feature_error, local_feature_error)
                    score_error = max(score_error, local_score_error)
                    feature_ok &= bool(
                        np.allclose(
                            current_features,
                            historical_features,
                            atol=patol,
                            rtol=prtol,
                            equal_nan=True,
                        )
                    )
                    scores_ok &= bool(
                        np.isclose(
                            float(current["fp_shep_online_score"]),
                            float(historical["fp_shep_online_score"]),
                            atol=satol,
                            rtol=srtol,
                            equal_nan=True,
                        )
                    )
            current_selected = fp_records[agent_id]["selected_candidate_id"]
            historical_selected = int(historical_selection["selected_candidate_index"])
            selected_exact = int(current_selected) == historical_selected
            maximum_preview_error = max(maximum_preview_error, feature_error)
            maximum_score_error = max(maximum_score_error, score_error)
            candidate_match &= candidate_ok
            preview_match &= feature_ok
            score_match &= scores_ok
            selection_match &= selected_exact
            rows.append(
                {
                    "layout_id": layout_id,
                    "family": shared["family"],
                    "agent_id": agent_id,
                    "candidate_count_exact": count_exact,
                    "candidate_order_exact": order_exact,
                    "candidate_position_match": position_match,
                    "candidate_sector_exact": sector_exact,
                    "candidate_score_match": coarse_score_match,
                    "candidate_bundle_hash_exact": bundle_hash_exact,
                    "preview_feature_match": feature_ok,
                    "preview_feature_max_absolute_error": feature_error,
                    "fp_shep_score_match": scores_ok,
                    "fp_shep_score_max_absolute_error": score_error,
                    "selected_candidate_exact": selected_exact,
                    "current_selected_candidate": current_selected,
                    "historical_selected_candidate": historical_selected,
                }
            )
    audit = {
        "CANDIDATE_REPRODUCTION_MATCH": "YES" if candidate_match else "NO",
        "PREVIEW_FEATURE_REPRODUCTION_MATCH": "YES" if preview_match else "NO",
        "FP_SHEP_SCORE_REPRODUCTION_MATCH": "YES" if score_match else "NO",
        "SELECTION_REPRODUCTION_MATCH": "YES" if selection_match else "NO",
        "maximum_preview_feature_absolute_error": maximum_preview_error,
        "maximum_fp_shep_score_absolute_error": maximum_score_error,
        "historical_null_current_positive_infinity_count": (
            historical_null_current_positive_infinity_count
        ),
        "historical_preview_trajectory_available": False,
        "preview_trajectory_reproduction_match": "NOT_AVAILABLE",
        "candidate_atol": catol,
        "candidate_rtol": crtol,
        "preview_feature_atol": patol,
        "preview_feature_rtol": prtol,
        "fp_shep_score_atol": satol,
        "fp_shep_score_rtol": srtol,
    }
    return audit, rows


def _run_raw_episode(
    *,
    method: str,
    shared: Mapping[str, Any],
    policy: Any,
    multi_config: Any,
    execution_settings: Mapping[str, Any],
    environment_builder: Callable[..., Any],
    selection_plan: Mapping[str, Any] | None,
    candidate_sets: Sequence[Sequence[Any]] | None = None,
    selection_method: str = "proposal_sac_dmp",
    score_spec: FPSHEPOnlineScoreSpec | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    execution_semantics: list[str | None] = []

    def observer(_kwargs: dict[str, Any], transition: Any) -> None:
        execution_semantics.append(
            transition.controller_info.get("forcing_gate_semantics")
        )

    variant = VARIANT_A if method == METHOD_TERMINAL else VARIANT_D
    with capture_execution_trace() as trace:
        with scoped_historical_preview_and_multi_agent_transition(
            execution_observer=observer
        ):
            raw, _, _ = run_variant_episode(
                policy=policy,
                multi_config=multi_config,
                settings=execution_settings,
                scenario=str(execution_settings["scenarios"][0]),
                seed=int(shared["seed"]),
                variant=variant,
                candidate_sets=candidate_sets,
                selection_method=selection_method,
                score_spec=score_spec,
                environment_builder=environment_builder,
                selection_plan=selection_plan,
            )
    raw["_execution_gate_trace"] = execution_semantics
    return raw, trace


def _execution_equivalence_audit(
    *,
    manifest: Mapping[str, Any],
    shared_cache: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    policy: Any,
    multi_config: Any,
    execution_settings: Mapping[str, Any],
    environment_builder: Callable[..., Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tolerances = config["compatibility_tolerances"]
    atol = float(tolerances["execution_atol"])
    rtol = float(tolerances["execution_rtol"])
    trajectory_atol = float(tolerances["trajectory_atol"])
    trajectory_rtol = float(tolerances["trajectory_rtol"])
    score_spec = FPSHEPOnlineScoreSpec.from_mapping(
        {
            "name": config["fp_shep"]["score_name"],
            "definition_status": "frozen_stress_compatibility",
            "H_preview": int(config["H_preview"]),
            "formula": config["fp_shep"]["formula"],
            "weights": {
                "progress": 1.0,
                "clearance": 1.0,
                "deviation": 1.0,
                "terminal_speed": 0.0,
            },
            "normalization": config["fp_shep"]["normalization"],
            "normalization_recalibrated": False,
            "real_outcome_used_to_tune_score": False,
        }
    )
    rows: list[dict[str, Any]] = []
    representatives = config["execution_equivalence_representatives"]
    manifest_index = {row["layout_id"]: row for row in manifest["layouts"]}
    for family in "ABCDEF":
        layout_id = str(representatives[family])
        if str(manifest_index[layout_id]["family"]) != family:
            raise RuntimeError("representative family mapping is invalid")
        shared = shared_cache[layout_id]
        for method in (METHOD_PROPOSAL, METHOD_FP_SHEP):
            selector = (
                "proposal_sac_dmp"
                if method == METHOD_PROPOSAL
                else "fp_shep_sac_dmp"
            )
            legacy, legacy_trace = _run_raw_episode(
                method=method,
                shared=shared,
                policy=policy,
                multi_config=multi_config,
                execution_settings=execution_settings,
                environment_builder=environment_builder,
                selection_plan=None,
                candidate_sets=shared["candidate_bundle"].per_agent,
                selection_method=selector,
                score_spec=score_spec,
            )
            planned, planned_trace = _run_raw_episode(
                method=method,
                shared=shared,
                policy=policy,
                multi_config=multi_config,
                execution_settings=execution_settings,
                environment_builder=environment_builder,
                selection_plan=shared["plans"][method],
            )
            selected_reference_exact = (
                legacy["temporary_reference_hash"]
                == planned["temporary_reference_hash"]
            )
            trace_length_exact = len(legacy_trace) == len(planned_trace)
            errors = {
                key: _max_abs_error(legacy_trace, planned_trace, key)
                for key in ("action", "position", "velocity", "phase", "dmp_goal")
            }
            action_match = trace_length_exact and errors["action"] is not None and errors[
                "action"
            ] <= atol
            position_match = (
                trace_length_exact
                and errors["position"] is not None
                and errors["position"] <= trajectory_atol
            )
            velocity_match = (
                trace_length_exact
                and errors["velocity"] is not None
                and errors["velocity"] <= trajectory_atol
            )
            phase_match = trace_length_exact and errors["phase"] is not None and errors[
                "phase"
            ] <= atol
            dmp_goal_match = (
                trace_length_exact
                and errors["dmp_goal"] is not None
                and errors["dmp_goal"] <= atol
            )
            handoff_exact = (
                legacy["temporary_reference_reached_count"]
                == planned["temporary_reference_reached_count"]
                and legacy["reference_reached_steps"]
                == planned["reference_reached_steps"]
            )
            termination_exact = all(
                legacy[key] == planned[key]
                for key in (
                    "success",
                    "collision",
                    "obstacle_collision",
                    "inter_agent_collision",
                    "truncated",
                    "termination_reason",
                    "steps",
                )
            )
            trajectory_hash_exact = _trace_hash(legacy_trace) == _trace_hash(
                planned_trace
            )
            all_match = all(
                (
                    selected_reference_exact,
                    trace_length_exact,
                    action_match,
                    position_match,
                    velocity_match,
                    phase_match,
                    dmp_goal_match,
                    handoff_exact,
                    termination_exact,
                )
            )
            rows.append(
                {
                    "family": family,
                    "layout_id": layout_id,
                    "method": method,
                    "selected_reference_exact": selected_reference_exact,
                    "trace_length_exact": trace_length_exact,
                    "legacy_step_count": len(legacy_trace),
                    "selection_plan_step_count": len(planned_trace),
                    "action_match": action_match,
                    "action_max_absolute_error": errors["action"],
                    "position_match": position_match,
                    "position_max_absolute_error": errors["position"],
                    "velocity_match": velocity_match,
                    "velocity_max_absolute_error": errors["velocity"],
                    "phase_match": phase_match,
                    "phase_max_absolute_error": errors["phase"],
                    "dmp_goal_match": dmp_goal_match,
                    "dmp_goal_max_absolute_error": errors["dmp_goal"],
                    "handoff_exact": handoff_exact,
                    "handoff_count": planned["temporary_reference_reached_count"],
                    "termination_exact": termination_exact,
                    "termination_reason": planned["termination_reason"],
                    "trajectory_hash_exact": trajectory_hash_exact,
                    "legacy_trajectory_hash": _trace_hash(legacy_trace),
                    "selection_plan_trajectory_hash": _trace_hash(planned_trace),
                    "all_match": all_match,
                }
            )
    required_families = set("ABCDEF")
    covered_families = {row["family"] for row in rows}
    audit = {
        "EXECUTION_REPRODUCTION_MATCH": (
            "YES"
            if covered_families == required_families
            and len(rows) == 12
            and all(row["all_match"] for row in rows)
            else "NO"
        ),
        "family_coverage": sorted(covered_families),
        "representatives": dict(representatives),
        "comparison_count": len(rows),
        "execution_atol": atol,
        "execution_rtol": rtol,
        "trajectory_atol": trajectory_atol,
        "trajectory_rtol": trajectory_rtol,
        "termination_types_observed": sorted(
            {str(row["termination_reason"]) for row in rows}
        ),
        "handoff_observed": any(int(row["handoff_count"]) > 0 for row in rows),
    }
    return audit, rows


def _standardize_fresh_episode(
    *,
    raw: Mapping[str, Any],
    method: str,
    shared: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    execution_trace = [
        {"forcing_gate_semantics": value}
        for value in raw.get("_execution_gate_trace", [])
    ]
    episode, agents = closed_loop._standardize_episode(
        raw, method=method, shared=shared, execution_trace=execution_trace
    )
    common = {
        "schema_version": SCHEMA_VERSION,
        "layout_id": shared["layout_id"],
        "family": shared["family"],
        "layout_hash": shared["layout_hash"],
        "relative_geometry_hash": shared["relative_geometry_hash"],
        "geometry_manifest_sha256": None,
        "result_source": "fresh_current_evaluator",
    }
    episode.update(common)
    for row in agents:
        row.update(common)
        agent_id = int(row["agent_id"])
        row["path_length_m"] = float(raw["path_length_per_agent_m"][agent_id])
        row["mean_speed_mps"] = float(raw["mean_speed_per_agent_mps"][agent_id])
        row["peak_speed_mps"] = float(raw["peak_speed_per_agent_mps"][agent_id])
    return episode, agents


def _run_fresh_methods(
    *,
    manifest: Mapping[str, Any],
    shared_cache: Mapping[str, Mapping[str, Any]],
    methods: Sequence[str],
    policy: Any,
    multi_config: Any,
    execution_settings: Mapping[str, Any],
    environment_builder: Callable[..., Any],
    manifest_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    total = len(manifest["layouts"]) * len(methods)
    counter = 0
    for layout in manifest["layouts"]:
        shared = shared_cache[str(layout["layout_id"])]
        for method in methods:
            plan = shared["plans"].get(method)
            raw, _ = _run_raw_episode(
                method=method,
                shared=shared,
                policy=policy,
                multi_config=multi_config,
                execution_settings=execution_settings,
                environment_builder=environment_builder,
                selection_plan=plan,
            )
            episode, agent_rows = _standardize_fresh_episode(
                raw=raw, method=method, shared=shared
            )
            episode["geometry_manifest_sha256"] = manifest_hash
            for row in agent_rows:
                row["geometry_manifest_sha256"] = manifest_hash
            episodes.append(episode)
            agents.extend(agent_rows)
            counter += 1
            print(
                f"[episode {counter}/{total}] {layout['layout_id']} {method}: "
                f"{episode['termination_reason']}",
                flush=True,
            )
    return episodes, agents


def _load_reusable_historical_results(
    *,
    historical_dir: Path,
    shared_cache: Mapping[str, Mapping[str, Any]],
    manifest_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Import verified A/B/C records only after all compatibility gates pass."""

    method_map = {
        "terminal_goal_baseline": METHOD_TERMINAL,
        "proposal_top1_one_shot": METHOD_PROPOSAL,
        "fp_shep_top1_one_shot": METHOD_FP_SHEP,
    }
    old_agents = _read_csv(historical_dir / "per_agent.csv")
    agents_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in old_agents:
        agents_by_key[(row["layout_id"], row["method"])].append(row)
    episodes: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    for old in _read_csv(historical_dir / "per_episode.csv"):
        method = method_map[str(old["method"])]
        layout_id = str(old["layout_id"])
        shared = shared_cache[layout_id]
        plan = shared["plans"].get(method)
        old_agent_rows = sorted(
            agents_by_key[(layout_id, old["method"])],
            key=lambda row: int(row["agent_id"]),
        )
        normalized_agents: list[dict[str, Any]] = []
        per_agent_paths: list[float] = []
        for old_agent in old_agent_rows:
            agent_id = int(old_agent["agent_id"])
            plan_record = (
                plan["candidate_records"][agent_id]
                if plan is not None
                else {
                    "selected_candidate_id": None,
                    "selected_null": False,
                    "no_candidate_fallback": False,
                }
            )
            path_length = float(old_agent["path_length_m"])
            per_agent_paths.append(path_length)
            normalized_agents.append(
                {
                    **dict(old_agent),
                    "schema_version": SCHEMA_VERSION,
                    "method": method,
                    "method_display_name": METHOD_NAMES[method],
                    "scenario": old["scenario"],
                    "seed": int(old["seed"]),
                    "layout_id": layout_id,
                    "family": old["family"],
                    "layout_hash": old["layout_hash"],
                    "relative_geometry_hash": old["relative_geometry_hash"],
                    "geometry_manifest_sha256": manifest_hash,
                    "reference_selected_type": closed_loop._reference_selected_type(
                        method, plan_record
                    ),
                    "selected_candidate_id": plan_record.get("selected_candidate_id"),
                    "selected_null": bool(plan_record.get("selected_null", False)),
                    "candidate_bundle_hash": shared["candidate_bundle_hash"],
                    "preview_bundle_hash": (
                        shared["preview_bundle_hash"]
                        if method == METHOD_FP_SHEP
                        else None
                    ),
                    "path_length_m": path_length,
                    "reference_reached": _parse_bool(old_agent["reference_reached"]),
                    "terminal_completed_after_reference": _parse_bool(
                        old_agent["terminal_completed_after_reference"]
                    ),
                    "result_source": "verified_historical_artifact",
                }
            )
        selected_count = sum(
            row["reference_selected_type"] == "proposal" for row in normalized_agents
        )
        reached_count = sum(bool(row["reference_reached"]) for row in normalized_agents)
        completed_count = sum(
            bool(row["terminal_completed_after_reference"])
            for row in normalized_agents
        )
        minimum_obstacle = _parse_optional_float(old["minimum_obstacle_clearance_m"])
        episode = {
            "schema_version": SCHEMA_VERSION,
            "method": method,
            "method_display_name": METHOD_NAMES[method],
            "scenario": old["scenario"],
            "seed": int(old["seed"]),
            "layout_id": layout_id,
            "family": old["family"],
            "layout_hash": old["layout_hash"],
            "relative_geometry_hash": old["relative_geometry_hash"],
            "geometry_manifest_sha256": manifest_hash,
            "team_success": _parse_bool(old["team_success"]),
            "obstacle_collision": _parse_bool(old["obstacle_collision"]),
            "inter_agent_collision": _parse_bool(old["inter_agent_collision"]),
            "collision": _parse_bool(old["collision"]),
            "timeout": _parse_bool(old["timeout"]),
            "termination_reason": old["termination_reason"],
            "completion_step": (
                int(old["steps"]) if _parse_bool(old["team_success"]) else None
            ),
            "completion_time_s": _parse_optional_float(old["completion_time_s"]),
            "steps": int(old["steps"]),
            "team_path_length_m": float(old["path_length_team_sum_m"]),
            "team_path_length_mean_agent_m": float(old["path_length_m"]),
            "per_agent_path_length_m": per_agent_paths,
            "trajectory_smoothness": float(old["trajectory_smoothness"]),
            "mean_speed_mps": float(old["mean_speed_team_mps"]),
            "peak_speed_mps": float(old["peak_speed_team_mps"]),
            "mean_acceleration_mps2": float(old["mean_applied_acceleration_mps2"]),
            "peak_acceleration_mps2": float(old["max_applied_acceleration_mps2"]),
            "heading_yaw_change_available": False,
            "minimum_obstacle_clearance_m": minimum_obstacle,
            "minimum_obstacle_clearance_unbounded": minimum_obstacle is None,
            "minimum_inter_agent_distance_m": float(
                old["minimum_inter_agent_distance_m"]
            ),
            "reference_selected_types": [
                row["reference_selected_type"] for row in normalized_agents
            ],
            "reference_selected_type": (
                "mixed"
                if len(
                    {row["reference_selected_type"] for row in normalized_agents}
                )
                > 1
                else normalized_agents[0]["reference_selected_type"]
            ),
            "reference_selected_count": selected_count,
            "reference_reached_count": reached_count,
            "reference_reached": reached_count > 0,
            "reference_reach_step": [
                _parse_optional_int(row["reference_reached_step"])
                for row in old_agent_rows
            ],
            "reference_to_terminal_success_count": completed_count,
            "reference_to_terminal_success": reached_count > 0
            and completed_count == reached_count,
            "candidate_bundle_hash": shared["candidate_bundle_hash"],
            "preview_bundle_hash": (
                shared["preview_bundle_hash"] if method == METHOD_FP_SHEP else None
            ),
            "selection_plan_hash": (
                plan["selection_plan_hash"] if plan is not None else None
            ),
            "initial_condition_hash": old["initial_condition_hash"],
            "scenario_manifest_hash": shared["initial_condition_hash"],
            "proposal_reconstruction_equivalent": True,
            "graph_schema_match": True,
            "GAT_checkpoint_used": False,
            "FP_SHEP_selector_used": method == METHOD_FP_SHEP,
            "historical_gate": HISTORICAL_GATE_NAME,
            "execution_transition_call_count": int(
                old["execution_transition_call_count"]
            ),
            "execution_historical_gate_verified": _parse_bool(
                old["execution_historical_gate_verified"]
            ),
            "preview_historical_gate_verified": True,
            "handoff_count": reached_count,
            "maximum_handoff_count_per_agent": 1 if reached_count else 0,
            "replanning_count": 0,
            "phase_reset_on_switch": _parse_bool(old["phase_reset_on_switch"]),
            "terminal_task_goals_unchanged": _parse_bool(
                old["terminal_task_goals_unchanged"]
            ),
            "max_steps": 220,
            "episode_runtime_ms": None,
            "result_source": "verified_historical_artifact",
        }
        episodes.append(episode)
        agents.extend(normalized_agents)
    return episodes, agents


def _historical_outcome_reproduction(
    episodes: Sequence[Mapping[str, Any]], historical_dir: Path
) -> dict[str, Any]:
    method_map = {
        "terminal_goal_baseline": METHOD_TERMINAL,
        "proposal_top1_one_shot": METHOD_PROPOSAL,
        "fp_shep_top1_one_shot": METHOD_FP_SHEP,
    }
    current = {
        (row["layout_id"], row["method"]): row
        for row in episodes
        if row["method"] in method_map.values()
    }
    rows = _read_csv(historical_dir / "per_episode.csv")
    checks: list[dict[str, Any]] = []
    for historical in rows:
        method = method_map[str(historical["method"])]
        row = current.get((historical["layout_id"], method))
        if row is None:
            continue
        discrete = {
            "team_success": bool(row["team_success"])
            == _parse_bool(historical["team_success"]),
            "obstacle_collision": bool(row["obstacle_collision"])
            == _parse_bool(historical["obstacle_collision"]),
            "inter_agent_collision": bool(row["inter_agent_collision"])
            == _parse_bool(historical["inter_agent_collision"]),
            "timeout": bool(row["timeout"]) == _parse_bool(historical["timeout"]),
            "termination_reason": str(row["termination_reason"])
            == str(historical["termination_reason"]),
            "steps": int(row["steps"]) == int(historical["steps"]),
        }
        checks.append(
            {
                "layout_id": historical["layout_id"],
                "method": method,
                **discrete,
                "all_discrete_match": all(discrete.values()),
            }
        )
    success_counts = Counter(
        row["method"] for row in episodes if bool(row["team_success"])
    )
    return {
        "comparison_count": len(checks),
        "all_discrete_outcomes_match": bool(checks)
        and all(row["all_discrete_match"] for row in checks),
        "fresh_success_counts": {
            method: int(success_counts[method])
            for method in (METHOD_TERMINAL, METHOD_PROPOSAL, METHOD_FP_SHEP)
        },
        "historical_expected_success_counts": {
            METHOD_TERMINAL: 2,
            METHOD_PROPOSAL: 2,
            METHOD_FP_SHEP: 6,
        },
        "rows": checks,
    }


def _failure_category(
    episode: Mapping[str, Any], agents: Sequence[Mapping[str, Any]]
) -> str:
    if bool(episode["team_success"]):
        return "SUCCESS"
    if bool(episode["obstacle_collision"]):
        return "OBSTACLE_COLLISION"
    if bool(episode["inter_agent_collision"]):
        return "INTER_AGENT_COLLISION"
    selected = [row for row in agents if row["reference_selected_type"] == "proposal"]
    if bool(episode["timeout"]):
        if not selected or not any(bool(row["reference_reached"]) for row in selected):
            return "TIMEOUT_BEFORE_REFERENCE"
        if selected and any(not bool(row["reference_reached"]) for row in selected):
            return "REFERENCE_UNREACHABLE_OR_STAGNATION"
        if selected and all(bool(row["reference_reached"]) for row in selected):
            return "AFTER_REFERENCE_FAILURE"
    return "OTHER_SOFTWARE_OR_ENVIRONMENT_FAILURE"


def _failure_decomposition(
    episodes: Sequence[Mapping[str, Any]], agents: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    agent_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in agents:
        agent_groups[(row["method"], int(row["seed"]))].append(row)
    labeled = [
        {
            **dict(row),
            "primary_failure_category": _failure_category(
                row, agent_groups[(row["method"], int(row["seed"]))]
            ),
        }
        for row in episodes
    ]
    output: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        for family in ["overall", *list("ABCDEF")]:
            members = [
                row
                for row in labeled
                if row["method"] == method
                and (family == "overall" or row["family"] == family)
            ]
            counts = Counter(row["primary_failure_category"] for row in members)
            for category, count in sorted(counts.items()):
                output.append(
                    {
                        "method": method,
                        "family": family,
                        "primary_failure_category": category,
                        "count": count,
                        "rate": count / len(members) if members else None,
                        "episode_count": len(members),
                    }
                )
    return output


def _summary_stats(values: Iterable[Any]) -> dict[str, Any]:
    array = np.asarray(
        [float(value) for value in values if value is not None], dtype=float
    )
    array = array[np.isfinite(array)]
    if not array.size:
        return {"mean": None, "std": None, "median": None, "count": 0}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "count": int(array.size),
    }


def _family_summary(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in "ABCDEF":
        for method in METHOD_ORDER:
            members = [
                row
                for row in episodes
                if row["family"] == family and row["method"] == method
            ]
            record: dict[str, Any] = {
                "family": family,
                "method": method,
                "episode_count": len(members),
            }
            for key in (
                "team_success",
                "obstacle_collision",
                "inter_agent_collision",
                "collision",
                "timeout",
            ):
                count = sum(bool(row[key]) for row in members)
                record[f"{key}_count"] = count
                record[f"{key}_rate"] = count / len(members) if members else None
            rows.append(record)
    return rows


def _reference_summary(
    episodes: Sequence[Mapping[str, Any]], agents: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        for family in ["overall", *list("ABCDEF")]:
            method_agents = [
                row
                for row in agents
                if row["method"] == method
                and (family == "overall" or row["family"] == family)
            ]
            selected = [
                row for row in method_agents if row["reference_selected_type"] == "proposal"
            ]
            reached = [row for row in selected if bool(row["reference_reached"])]
            completed = [
                row for row in reached if bool(row["terminal_completed_after_reference"])
            ]
            method_episodes = [
                row
                for row in episodes
                if row["method"] == method
                and (family == "overall" or row["family"] == family)
            ]
            all_refs_reached = sum(
                row["reference_selected_count"] > 0
                and row["reference_reached_count"] == row["reference_selected_count"]
                for row in method_episodes
            )
            team_completed_after = sum(
                row["team_success"]
                and row["reference_selected_count"] > 0
                and row["reference_reached_count"] == row["reference_selected_count"]
                for row in method_episodes
            )
            rows.append(
                {
                    "method": method,
                    "family": family,
                    "reference_selected_count": len(selected),
                    "reference_reached_count": len(reached),
                    "reference_reach_rate": len(reached) / len(selected) if selected else None,
                    "terminal_completed_after_reference_count": len(completed),
                    "reached_to_terminal_conversion_rate": (
                        len(completed) / len(reached) if reached else None
                    ),
                    "team_episode_count": len(method_episodes),
                    "all_references_reached_count": all_refs_reached,
                    "team_completed_after_refs_count": team_completed_after,
                }
            )
    return rows


def _enrich_family_summary(
    family_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    disagreements: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reference_index = {
        (row["family"], row["method"]): row
        for row in reference_rows
        if row["family"] != "overall"
    }
    disagreement_by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in disagreements:
        disagreement_by_family[str(row["family"])].append(row)
    output: list[dict[str, Any]] = []
    for row in family_rows:
        family = str(row["family"])
        method = str(row["method"])
        reference = reference_index[(family, method)]
        disagreement_members = disagreement_by_family[family]
        output.append(
            {
                **dict(row),
                "reference_selected_count": reference["reference_selected_count"],
                "reference_reached_count": reference["reference_reached_count"],
                "reference_reach_rate": reference["reference_reach_rate"],
                "reached_to_terminal_conversion_rate": reference[
                    "reached_to_terminal_conversion_rate"
                ],
                "gat_fp_selector_disagreement_count": sum(
                    bool(item["disagreement"]) for item in disagreement_members
                ),
                "gat_fp_selector_disagreement_rate": (
                    sum(bool(item["disagreement"]) for item in disagreement_members)
                    / len(disagreement_members)
                    if disagreement_members
                    else None
                ),
            }
        )
    return output


def _rate_class(improvement: float, config: Mapping[str, Any]) -> str:
    thresholds = config["effect_classification"]
    if improvement >= float(thresholds["yes_minimum_improvement"]):
        return "YES"
    if improvement > float(thresholds["weak_strict_minimum_improvement"]):
        return "WEAK"
    return "NO"


def _build_conclusion(
    *,
    episodes: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    integrity: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    overall: dict[str, dict[str, float]] = {}
    for method in METHOD_ORDER:
        members = [row for row in episodes if row["method"] == method]
        overall[method] = {
            "success": sum(bool(row["team_success"]) for row in members) / len(members),
            "collision": sum(bool(row["collision"]) for row in members) / len(members),
            "timeout": sum(bool(row["timeout"]) for row in members) / len(members),
            "inter_agent_collision": sum(
                bool(row["inter_agent_collision"]) for row in members
            )
            / len(members),
        }
    fp = overall[METHOD_FP_SHEP]
    gat = overall[METHOD_GAT]
    proposal = overall[METHOD_PROPOSAL]
    success_gain = gat["success"] - fp["success"]
    collision_gain = fp["collision"] - gat["collision"]
    timeout_gain = fp["timeout"] - gat["timeout"]
    inter_gain = fp["inter_agent_collision"] - gat["inter_agent_collision"]
    proposal_gain = gat["success"] - proposal["success"]

    family_lookup = {(row["family"], row["method"]): row for row in family_rows}
    family_differences = {
        family: float(family_lookup[(family, METHOD_GAT)]["team_success_rate"])
        - float(family_lookup[(family, METHOD_FP_SHEP)]["team_success_rate"])
        for family in "ABCDEF"
    }
    positive_families = sum(value > 0.0 for value in family_differences.values())
    regressed_families = sum(value < 0.0 for value in family_differences.values())
    success_class = _rate_class(success_gain, config)
    family_rule = config["family_generalization_rule"]
    if (
        success_class == family_rule["yes_requires_overall_success_class"]
        and positive_families >= int(family_rule["yes_minimum_positive_families"])
        and regressed_families <= int(family_rule["yes_maximum_regressed_families"])
    ):
        family_class = "YES"
    elif success_gain > 0.0 and positive_families >= int(
        family_rule["weak_minimum_positive_families"]
    ):
        family_class = "WEAK"
    else:
        family_class = "NO"
    if success_class == "YES" and family_class == "YES":
        geometry_gain = "YES"
    elif success_gain > 0.0 and family_class in {"YES", "WEAK"}:
        geometry_gain = "WEAK"
    else:
        geometry_gain = "NO"

    ref_lookup = {(row["method"], row["family"]): row for row in reference_rows}
    fp_ref = ref_lookup[(METHOD_FP_SHEP, "overall")]
    gat_ref = ref_lookup[(METHOD_GAT, "overall")]
    ref_gain = (
        float(gat_ref["reference_reach_rate"] or 0.0)
        - float(fp_ref["reference_reach_rate"] or 0.0)
    )
    post_gain = (
        float(gat_ref["reached_to_terminal_conversion_rate"] or 0.0)
        - float(fp_ref["reached_to_terminal_conversion_rate"] or 0.0)
    )

    sp_rule = config["spatiotemporal_claim_rule"]
    if (
        success_gain >= float(sp_rule["supported_requires_success_improvement"])
        and inter_gain
        >= float(sp_rule["supported_requires_inter_agent_collision_improvement"])
        and positive_families >= int(sp_rule["supported_minimum_positive_families"])
    ):
        spatiotemporal = "SUPPORTED"
    elif success_gain > 0.0 or inter_gain > 0.0:
        spatiotemporal = "WEAK"
    else:
        spatiotemporal = "NOT_ESTABLISHED"

    return {
        "GEOMETRY_MANIFEST_HASH_MATCH": "YES"
        if integrity["geometry_manifest_hash_match"]
        else "NO",
        "HISTORICAL_STRESS_RESULTS_REUSABLE": integrity[
            "historical_stress_results_reusable"
        ],
        "CANDIDATE_BUNDLE_FAIRNESS": "YES"
        if integrity["candidate_bundle_fairness"]
        else "NO",
        "PREVIEW_FAIRNESS": "YES" if integrity["preview_fairness"] else "NO",
        "GAT_CHECKPOINT_VALID": "YES"
        if integrity["gat_checkpoint_valid"]
        else "NO",
        "STRESS_PIPELINE_VALID": "YES"
        if integrity["stress_pipeline_valid"]
        else "NO",
        "GAT_VS_PROPOSAL_STRESS_GAIN": _rate_class(proposal_gain, config),
        "GAT_VS_FP_SHEP_STRESS_SUCCESS_GAIN": success_class,
        "GAT_VS_FP_SHEP_STRESS_COLLISION_GAIN": _rate_class(
            collision_gain, config
        ),
        "REFERENCE_EXECUTABILITY_GAIN": _rate_class(ref_gain, config),
        "POST_REFERENCE_COMPLETION_GAIN": _rate_class(post_gain, config),
        "FAMILY_GENERALIZATION": family_class,
        "GEOMETRY_GENERALIZATION_GAIN": geometry_gain,
        "SPATIOTEMPORAL_COORDINATION_CLAIM": spatiotemporal,
        "FINAL_METHOD": "Stage-I GAT",
        "FINAL_CHECKPOINT": config["gat_checkpoint"],
        "NEXT_STEP": (
            "Organize final experiment tables, trajectory figures, and paper Results; "
            "do not tune the frozen method from this stress outcome."
        ),
        "rate_improvements": {
            "gat_minus_fp_success": success_gain,
            "fp_minus_gat_collision": collision_gain,
            "fp_minus_gat_timeout": timeout_gain,
            "fp_minus_gat_inter_agent_collision": inter_gain,
            "gat_minus_proposal_success": proposal_gain,
            "gat_minus_fp_reference_reach": ref_gain,
            "gat_minus_fp_post_reference_conversion": post_gain,
        },
        "family_success_rate_differences": family_differences,
        "positive_family_count": positive_families,
        "regressed_family_count": regressed_families,
        "automatic_training_started": False,
        "automatic_method_modification_started": False,
    }


def _final_report(
    *,
    conclusion: Mapping[str, Any],
    method_summary: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    compatibility: Mapping[str, Any],
) -> str:
    overall = {
        row["method"]: row
        for row in method_summary
        if row["scenario"] == "overall"
    }
    lines = [
        "# Stage-I GAT 24-Layout Frozen Geometry Stress Test",
        "",
        "## Frozen protocol",
        "",
        "- 24 non-translation-equivalent layouts; six families; four layouts each.",
        "- Top-K=10, H_preview=4, max_steps=220, handoff=0.25 m.",
        "- Historical vector goal-eff gate, one-shot handoff, boundary-free.",
        "- Historical artifacts were never used by GAT graph construction or online selection.",
        "",
        "## Compatibility",
        "",
    ]
    for key in (
        "CANDIDATE_REPRODUCTION_MATCH",
        "PREVIEW_FEATURE_REPRODUCTION_MATCH",
        "FP_SHEP_SCORE_REPRODUCTION_MATCH",
        "SELECTION_REPRODUCTION_MATCH",
        "EXECUTION_REPRODUCTION_MATCH",
        "HISTORICAL_STRESS_RESULTS_REUSABLE",
    ):
        lines.append(f"- `{key} = {compatibility[key]}`")
    lines.extend(["", "## Overall team results", ""])
    for method in METHOD_ORDER:
        row = overall[method]
        lines.append(
            f"- {METHOD_NAMES[method]}: success {row['team_success_count']}/"
            f"{row['episode_count']} ({100.0*float(row['team_success_rate']):.1f}%); "
            f"collision {100.0*float(row['collision_rate']):.1f}%; "
            f"timeout {100.0*float(row['timeout_rate']):.1f}%."
        )
    pair_counts = Counter(row["pair_outcome"] for row in paired)
    lines.extend(["", "## GAT vs FP-SHEP paired outcomes", ""])
    lines.extend(f"- {key}: {value}" for key, value in sorted(pair_counts.items()))
    lines.extend(["", "## Decisions", ""])
    for key in (
        "GAT_VS_PROPOSAL_STRESS_GAIN",
        "GAT_VS_FP_SHEP_STRESS_SUCCESS_GAIN",
        "GAT_VS_FP_SHEP_STRESS_COLLISION_GAIN",
        "REFERENCE_EXECUTABILITY_GAIN",
        "POST_REFERENCE_COMPLETION_GAIN",
        "FAMILY_GENERALIZATION",
        "GEOMETRY_GENERALIZATION_GAIN",
        "SPATIOTEMPORAL_COORDINATION_CLAIM",
        "FINAL_METHOD",
        "FINAL_CHECKPOINT",
        "NEXT_STEP",
    ):
        lines.append(f"- `{key} = {conclusion[key]}`")
    lines.extend(
        [
            "",
            "## Stop rule",
            "",
            "The run stopped after the frozen 24-layout evaluation and analysis. "
            "No training, geometry change, method tuning, or new scenario family was started.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment(config: Mapping[str, Any], output_dir: Path) -> Path:
    _assert_frozen_config(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    evaluator_path = Path(__file__).resolve()
    evaluator_hash_before = _sha256_file(evaluator_path)
    resolved_config = copy.deepcopy(dict(config))
    resolved_config.update(
        {
            "resolved_output_dir": str(output_dir),
            "created_at": datetime.now().isoformat(),
            "config_sha256_before_outcomes": _sha256_file(DEFAULT_CONFIG_PATH),
            "evaluator_sha256_before_outcomes": evaluator_hash_before,
            "compatibility_tolerances_frozen_before_outcomes": True,
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
        }
    )
    write_json(output_dir / "config.json", resolved_config)

    manifest_path = REPO_ROOT / str(config["frozen_geometry_manifest"])
    manifest, manifest_audit = _manifest_audit(manifest_path, config)
    (output_dir / "geometry_manifest.json").write_bytes(manifest_path.read_bytes())
    if manifest_audit["status"] != "PASSED":
        write_json(output_dir / "integrity_manifest.json", manifest_audit)
        raise RuntimeError(f"geometry manifest gate failed: {manifest_audit['failed_checks']}")
    environment_builder = build_manifest_environment_builder(manifest, config)

    runtime_config = _runtime_closed_loop_config(config)
    execution_settings = _build_execution_settings(config)
    multi_config = build_single_distribution_multi_config(num_agents=3, max_steps=220)
    sac_path = REPO_ROOT / str(config["sac_checkpoint"])
    gat_path = REPO_ROOT / str(config["gat_checkpoint"])
    if _sha256_file(sac_path) != config["sac_checkpoint_sha256_expected"]:
        raise RuntimeError("SAC checkpoint hash mismatch")
    if _sha256_file(gat_path) != config["gat_checkpoint_sha256_expected"]:
        raise RuntimeError("GAT checkpoint hash mismatch")
    policy, loaded_sac = _load_policy(execution_settings, multi_config)
    if loaded_sac.resolve() != sac_path.resolve():
        raise RuntimeError("SAC loader resolved a different checkpoint")
    stage1_config = _load_json(REPO_ROOT / str(config["stage1_config"]))
    gat_device = resolve_device(stage1_config["training"]["device"])
    gat_model = load_model_checkpoint(gat_path, stage1_config, gat_device)
    gat_model.eval()
    for parameter in gat_model.parameters():
        parameter.requires_grad_(False)
    policy_hash_before = closed_loop._policy_parameter_sha256(policy)
    gat_hash_before = closed_loop._model_hash(gat_model)
    core_before = closed_loop._file_hashes(closed_loop.CORE_METHOD_PATHS)

    # Online plans are fully constructed before any historical selection or
    # outcome artifact is loaded.
    shared_cache, reconstruction_rows, gat_rows = _build_all_shared_bundles(
        manifest=manifest,
        config=config,
        runtime_config=runtime_config,
        execution_settings=execution_settings,
        multi_config=multi_config,
        policy=policy,
        gat_model=gat_model,
        gat_device=gat_device,
        environment_builder=environment_builder,
    )
    online_plan_hash = closed_loop.stable_hash(
        {
            layout_id: {
                "candidate": shared["candidate_bundle_hash"],
                "preview": shared["preview_bundle_hash"],
                "plans": {
                    method: plan["selection_plan_hash"]
                    for method, plan in shared["plans"].items()
                },
            }
            for layout_id, shared in shared_cache.items()
        }
    )

    historical_dir = REPO_ROOT / str(config["historical_artifact_dir"])
    reproduction, reproduction_rows = _candidate_and_preview_reproduction_audit(
        shared_cache=shared_cache,
        historical_dir=historical_dir,
        config=config,
    )
    execution_audit, execution_rows = _execution_equivalence_audit(
        manifest=manifest,
        shared_cache=shared_cache,
        config=config,
        policy=policy,
        multi_config=multi_config,
        execution_settings=execution_settings,
        environment_builder=environment_builder,
    )
    schema_complete, schema_audit = _historical_schema_complete(
        historical_dir, config
    )
    key_reproduction = all(
        reproduction[key] == "YES"
        for key in (
            "CANDIDATE_REPRODUCTION_MATCH",
            "PREVIEW_FEATURE_REPRODUCTION_MATCH",
            "FP_SHEP_SCORE_REPRODUCTION_MATCH",
            "SELECTION_REPRODUCTION_MATCH",
        )
    ) and execution_audit["EXECUTION_REPRODUCTION_MATCH"] == "YES"
    historical_reusable = bool(key_reproduction and schema_complete)
    compatibility = {
        **reproduction,
        **execution_audit,
        "HISTORICAL_REQUIRED_SCHEMA_COMPLETE": "YES" if schema_complete else "NO",
        "HISTORICAL_STRESS_RESULTS_REUSABLE": "YES" if historical_reusable else "NO",
        "historical_schema_audit": schema_audit,
        "online_plan_hash_frozen_before_historical_load": online_plan_hash,
        "historical_artifact_used_for_gat_input": False,
        "historical_artifact_used_for_online_selection": False,
        "rerun_decision": (
            "RUN_GAT_ONLY_REUSE_HISTORICAL_ABC"
            if historical_reusable
            else "RERUN_ALL_FOUR_METHODS_96_TEAM_EPISODES"
        ),
    }
    write_json(output_dir / "historical_compatibility_audit.json", compatibility)
    write_csv(output_dir / "reproduction_audit.csv", reproduction_rows)
    write_csv(output_dir / "execution_equivalence.csv", execution_rows)

    if historical_reusable:
        episodes, agents = _load_reusable_historical_results(
            historical_dir=historical_dir,
            shared_cache=shared_cache,
            manifest_hash=manifest_audit["actual_sha256"],
        )
        gat_episodes, gat_agents = _run_fresh_methods(
            manifest=manifest,
            shared_cache=shared_cache,
            methods=(METHOD_GAT,),
            policy=policy,
            multi_config=multi_config,
            execution_settings=execution_settings,
            environment_builder=environment_builder,
            manifest_hash=manifest_audit["actual_sha256"],
        )
        episodes.extend(gat_episodes)
        agents.extend(gat_agents)
    else:
        episodes, agents = _run_fresh_methods(
            manifest=manifest,
            shared_cache=shared_cache,
            methods=METHOD_ORDER,
            policy=policy,
            multi_config=multi_config,
            execution_settings=execution_settings,
            environment_builder=environment_builder,
            manifest_hash=manifest_audit["actual_sha256"],
        )
    if len(episodes) != 96 or len(agents) != 288:
        raise RuntimeError("formal stress-test episode/agent count mismatch")

    historical_outcomes = _historical_outcome_reproduction(episodes, historical_dir)
    compatibility["fresh_historical_outcome_reproduction"] = historical_outcomes
    write_json(output_dir / "historical_compatibility_audit.json", compatibility)

    method_summary = closed_loop.aggregate_method_summary(episodes)
    paired, disagreements = closed_loop.build_gat_fp_pairing(episodes, agents)
    seed_to_layout = {
        int(row["evaluation_seed"]): (row["layout_id"], row["family"])
        for row in manifest["layouts"]
    }
    for row in paired:
        row["layout_id"], row["family"] = seed_to_layout[int(row["seed"])]
    for row in disagreements:
        row["layout_id"], row["family"] = seed_to_layout[int(row["seed"])]
    reference_summary = _reference_summary(episodes, agents)
    family_summary = _enrich_family_summary(
        _family_summary(episodes), reference_summary, disagreements
    )
    failure_rows = _failure_decomposition(episodes, agents)
    statistical_tests = closed_loop.build_statistical_tests(episodes)
    statistical_tests["statistical_unit"] = "layout_team_episode"
    statistical_tests["layout_count"] = 24
    for comparison in statistical_tests["comparisons"].values():
        for scope in comparison.values():
            for metric in scope.values():
                metric["statistical_unit"] = "layout_team_episode"
    both_success_pairs = [row for row in paired if bool(row["both_success"])]
    statistical_tests["continuous_paired_gat_vs_fp_shep"] = {
        "subset": "both_methods_successful"
        " for completion/path/smoothness; all pairs for minimum distance",
        "completion_time_difference_s": _summary_stats(
            row["completion_time_paired_difference_s"]
            for row in both_success_pairs
        ),
        "team_path_length_difference_m": _summary_stats(
            row["path_length_paired_difference_m"] for row in both_success_pairs
        ),
        "trajectory_smoothness_difference": _summary_stats(
            row["smoothness_paired_difference"] for row in both_success_pairs
        ),
        "minimum_inter_agent_distance_difference_m": _summary_stats(
            row["minimum_inter_agent_distance_difference_m"] for row in paired
        ),
    }

    candidate_fairness = all(
        len(set(shared["candidate_hashes_by_method"].values())) == 1
        for shared in shared_cache.values()
    )
    preview_fairness = all(
        len(set(shared["preview_hashes_by_method"].values())) == 1
        for shared in shared_cache.values()
    )
    integrity = {
        "status": "PASSED",
        "geometry_manifest_hash_match": manifest_audit["status"] == "PASSED",
        "historical_stress_results_reusable": compatibility[
            "HISTORICAL_STRESS_RESULTS_REUSABLE"
        ],
        "candidate_bundle_fairness": candidate_fairness,
        "preview_fairness": preview_fairness,
        "gat_checkpoint_valid": _sha256_file(gat_path)
        == config["gat_checkpoint_sha256_expected"],
        "stress_pipeline_valid": True,
        "proposal_reconstruction_equivalent": all(
            shared["proposal_reconstruction_equivalent"]
            for shared in shared_cache.values()
        ),
        "graph_schema_match": all(
            shared["graph_schema_match"] for shared in shared_cache.values()
        ),
        "preview_historical_gate": all(
            shared["preview_historical_gate_verified"]
            for shared in shared_cache.values()
        ),
        "execution_historical_gate": all(
            bool(row["execution_historical_gate_verified"]) for row in episodes
        ),
        "one_shot": all(int(row["maximum_handoff_count_per_agent"]) <= 1 for row in episodes),
        "no_replanning": all(int(row["replanning_count"]) == 0 for row in episodes),
        "terminal_goals_unchanged": all(
            bool(row["terminal_task_goals_unchanged"]) for row in episodes
        ),
        "phase_not_reset": all(not bool(row["phase_reset_on_switch"]) for row in episodes),
        "episode_count": len(episodes),
        "agent_record_count": len(agents),
        "layout_count": len(shared_cache),
        "family_counts": manifest_audit["family_counts"],
        "online_plan_built_before_historical_load": True,
        "historical_artifact_used_for_gat_input": False,
        "historical_artifact_used_for_online_selection": False,
        "core_source_hashes_before": core_before,
        "core_source_hashes_after": closed_loop._file_hashes(
            closed_loop.CORE_METHOD_PATHS
        ),
        "core_sources_unchanged": core_before
        == closed_loop._file_hashes(closed_loop.CORE_METHOD_PATHS),
        "sac_policy_unchanged": policy_hash_before
        == closed_loop._policy_parameter_sha256(policy),
        "gat_model_unchanged": gat_hash_before == closed_loop._model_hash(gat_model),
        "checkpoint_hashes": {
            "sac": _sha256_file(sac_path),
            "gat": _sha256_file(gat_path),
        },
        "config_sha256_after_outcomes": _sha256_file(DEFAULT_CONFIG_PATH),
        "config_unchanged_after_outcomes": _sha256_file(DEFAULT_CONFIG_PATH)
        == resolved_config["config_sha256_before_outcomes"],
        "evaluator_sha256_after_outcomes": _sha256_file(evaluator_path),
        "evaluator_unchanged_after_outcomes": _sha256_file(evaluator_path)
        == evaluator_hash_before,
        "runtime_seconds": time.perf_counter() - started,
        "training_performed": False,
        "method_or_geometry_modified_from_outcomes": False,
    }
    integrity["status"] = (
        "PASSED"
        if all(
            (
                integrity["geometry_manifest_hash_match"],
                integrity["candidate_bundle_fairness"],
                integrity["preview_fairness"],
                integrity["gat_checkpoint_valid"],
                integrity["proposal_reconstruction_equivalent"],
                integrity["graph_schema_match"],
                integrity["preview_historical_gate"],
                integrity["execution_historical_gate"],
                integrity["one_shot"],
                integrity["no_replanning"],
                integrity["terminal_goals_unchanged"],
                integrity["phase_not_reset"],
                integrity["core_sources_unchanged"],
                integrity["sac_policy_unchanged"],
                integrity["gat_model_unchanged"],
                integrity["config_unchanged_after_outcomes"],
                integrity["evaluator_unchanged_after_outcomes"],
                execution_audit["EXECUTION_REPRODUCTION_MATCH"] == "YES",
            )
        )
        else "FAILED"
    )
    integrity["stress_pipeline_valid"] = integrity["status"] == "PASSED"
    conclusion = _build_conclusion(
        episodes=episodes,
        family_rows=family_summary,
        reference_rows=reference_summary,
        integrity=integrity,
        config=config,
    )

    write_csv(output_dir / "episode_results.csv", episodes)
    write_csv(output_dir / "per_agent_results.csv", agents)
    write_csv(output_dir / "method_summary.csv", method_summary)
    write_csv(output_dir / "family_summary.csv", family_summary)
    write_csv(output_dir / "failure_decomposition.csv", failure_rows)
    write_csv(output_dir / "paired_gat_vs_fp_shep.csv", paired)
    write_csv(output_dir / "selector_disagreement.csv", disagreements)
    write_csv(output_dir / "reference_transition_summary.csv", reference_summary)
    write_csv(output_dir / "proposal_reconstruction_audit.csv", reconstruction_rows)
    write_csv(output_dir / "gat_selection_diagnostics.csv", gat_rows)
    write_json(output_dir / "statistical_tests.json", statistical_tests)
    write_json(output_dir / "integrity_manifest.json", integrity)
    write_json(output_dir / "conclusion.json", conclusion)
    (output_dir / "FINAL_REPORT.md").write_text(
        _final_report(
            conclusion=conclusion,
            method_summary=method_summary,
            paired=paired,
            compatibility=compatibility,
        ),
        encoding="utf-8",
    )
    if integrity["status"] != "PASSED":
        raise RuntimeError("final stress pipeline integrity failed")
    return output_dir


def main() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = _load_json(args.config.resolve())
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / str(config["output_root"]) / stamp
    else:
        output_dir = args.output_dir.resolve()
    result = run_experiment(config, output_dir)
    print(result)
    return result


if __name__ == "__main__":
    main()
