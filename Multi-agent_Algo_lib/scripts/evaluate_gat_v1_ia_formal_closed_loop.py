#!/usr/bin/env python3
"""Frozen independent formal closed-loop evaluation of GAT-V1 versus V1-IA."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import binomtest


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.gat.stage1_training import load_model_checkpoint, resolve_device  # noqa: E402
from scripts import evaluate_gat_closed_loop as legacy  # noqa: E402
from scripts import evaluate_gat_stage1_v2_closed_loop as paired  # noqa: E402
from scripts.evaluate_pre_gat_closed_loop import _policy_parameter_sha256  # noqa: E402
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


SCHEMA_VERSION = "gat_v1_ia_formal_closed_loop_v1"
METHOD_V1 = "gat_v1"
METHOD_IA = "gat_v1_ia"
METHODS = (METHOD_V1, METHOD_IA)
SCENARIOS = ("open", "sparse_static", "multi_agent")
DEFAULT_CONFIG = REPO_ROOT / "configs/evaluation/gat_v1_ia_formal_closed_loop.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    fieldnames = list(fields) if fields is not None else list(rows[0]) if rows else ["schema_version", "status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(json_ready(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_hash(model: torch.nn.Module) -> str:
    return paired._model_hash(model)


def core_hashes(paths: Iterable[str]) -> dict[str, str]:
    return {path: sha256_file(REPO_ROOT / path) for path in paths}


def _integer_values(value: Any) -> set[int]:
    result: set[int] = set()
    if isinstance(value, bool):
        return result
    if isinstance(value, int):
        result.add(int(value))
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        result.add(int(value))
    elif isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[-+]?\d+(?:\.0+)?", text):
            result.add(int(float(text)))
        for left, right in re.findall(r"(?<!\d)(\d{1,6})\s*[-–—]\s*(\d{1,6})(?!\d)", text):
            start, stop = int(left), int(right)
            if start <= stop and stop - start <= 10000:
                result.update(range(start, stop + 1))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            result.update(_integer_values(item))
    return {item for item in result if 0 <= item <= 10_000_000}


def _seed_field(name: str) -> bool:
    normalized = name.lower()
    if "seed" not in normalized:
        return False
    if any(token in normalized for token in ("count", "overlap", "hash", "fingerprint")):
        return False
    return normalized not in {"seed_audit", "seed_manifest"}


def audit_all_used_seeds(
    roots: Sequence[Path], *, exclude_dir: Path | None = None
) -> dict[str, Any]:
    """Conservatively recover seed evidence from CSV, JSON, and report text."""

    used: set[int] = set()
    evidence: list[dict[str, Any]] = []
    scanned = Counter()
    parse_errors: list[dict[str, str]] = []

    def excluded(path: Path) -> bool:
        if exclude_dir is None:
            return False
        try:
            path.resolve().relative_to(exclude_dir.resolve())
            return True
        except ValueError:
            return False

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or excluded(path):
                continue
            suffix = path.suffix.lower()
            if suffix not in {".csv", ".json", ".md", ".txt"}:
                continue
            scanned[suffix] += 1
            relative = str(path.relative_to(REPO_ROOT))
            try:
                if suffix == ".csv":
                    with path.open("r", encoding="utf-8-sig", newline="") as handle:
                        reader = csv.DictReader(handle)
                        fields = [field for field in (reader.fieldnames or []) if _seed_field(field)]
                        found: set[int] = set()
                        for row in reader:
                            for field in fields:
                                found.update(_integer_values(row.get(field)))
                    if found:
                        used.update(found)
                        evidence.append(
                            {
                                "path": relative,
                                "source_type": "csv_seed_columns",
                                "fields": fields,
                                "seed_values": sorted(found),
                            }
                        )
                elif suffix == ".json":
                    payload = load_json(path)
                    found_by_field: dict[str, set[int]] = {}

                    def visit(value: Any, location: str = "$") -> None:
                        if isinstance(value, Mapping):
                            for key, item in value.items():
                                child = f"{location}.{key}"
                                if _seed_field(str(key)):
                                    values = _integer_values(item)
                                    if values:
                                        found_by_field.setdefault(child, set()).update(values)
                                visit(item, child)
                        elif isinstance(value, list):
                            for index, item in enumerate(value):
                                visit(item, f"{location}[{index}]")

                    visit(payload)
                    found = set().union(*found_by_field.values()) if found_by_field else set()
                    if found:
                        used.update(found)
                        evidence.append(
                            {
                                "path": relative,
                                "source_type": "json_seed_fields",
                                "fields": sorted(found_by_field),
                                "seed_values": sorted(found),
                            }
                        )
                else:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                    found: set[int] = set()
                    for index, line in enumerate(lines):
                        if "seed" not in line.lower():
                            continue
                        found.update(_integer_values(line))
                        next_index = index + 1
                        while next_index < len(lines) and not lines[next_index].strip():
                            next_index += 1
                        if next_index < len(lines) and re.fullmatch(
                            r"[\s`\[\](),:;0-9.\-–—]+", lines[next_index]
                        ):
                            found.update(_integer_values(lines[next_index]))
                    if found:
                        used.update(found)
                        evidence.append(
                            {
                                "path": relative,
                                "source_type": "report_seed_mentions",
                                "seed_values": sorted(found),
                            }
                        )
            except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as error:
                parse_errors.append({"path": relative, "error": str(error)})
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_scope": [str(root.relative_to(REPO_ROOT)) for root in roots if root.exists()],
        "scanned_file_counts": dict(sorted(scanned.items())),
        "seed_evidence_source_count": len(evidence),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors,
        "all_used_seed_values": sorted(used),
        "used_seed_values_0_to_9999": sorted(value for value in used if value <= 9999),
        "evidence": evidence,
    }


def choose_seed_block(
    used: Iterable[int], *, block_size: int, minimum: int, maximum: int
) -> list[int]:
    used_set = set(int(value) for value in used)
    for start in range(int(minimum), int(maximum) - int(block_size) + 2):
        block = list(range(start, start + int(block_size)))
        if not (set(block) & used_set):
            return block
    raise RuntimeError("no unused contiguous formal seed block found")


def validate_config(config: Mapping[str, Any]) -> None:
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected formal config schema")
    if tuple(config["methods"]) != METHODS or tuple(config["formal_scenarios"]) != SCENARIOS:
        raise ValueError("formal method or scenario set changed")
    frozen = {
        "top_k": 10,
        "H_preview": 4,
        "max_steps": 220,
        "handoff_threshold_m": 0.25,
        "actor_observation_dim": 122,
        "forcing_gate": "historical_vector_goal_eff_gate",
        "boundary_mode": "boundary_free",
    }
    for key, expected in frozen.items():
        if config["execution"][key] != expected:
            raise ValueError(f"frozen execution field changed: {key}")
    if any(bool(value) for value in config["strict_exclusions"].values()):
        raise ValueError("strict exclusion flags must remain false")
    if float(config["interaction_risk_contract"]["d_safe"]) != 0.6:
        raise ValueError("d_safe changed")
    if not config["formal_gain_decision"]["thresholds_frozen_before_outcomes"]:
        raise ValueError("formal decision thresholds were not frozen")


def build_context(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    v1_checkpoint = REPO_ROOT / sources["v1_checkpoint"]
    ia_checkpoint = REPO_ROOT / sources["ia_checkpoint"]
    ia_conclusion = load_json(REPO_ROOT / sources["ia_experiment_dir"] / "conclusion.json")
    historical = load_json(REPO_ROOT / sources["historical_v1_formal_dir"] / "conclusion.json")
    checks = {
        "v1_training_artifact_exists": (REPO_ROOT / sources["v1_training_dir"]).is_dir(),
        "historical_v1_formal_exists": (REPO_ROOT / sources["historical_v1_formal_dir"]).is_dir(),
        "interaction_capacity_audit_exists": (REPO_ROOT / sources["interaction_capacity_audit_dir"]).is_dir(),
        "ia_experiment_exists": (REPO_ROOT / sources["ia_experiment_dir"]).is_dir(),
        "v1_checkpoint_hash": sha256_file(v1_checkpoint) == sources["v1_checkpoint_sha256_expected"],
        "ia_checkpoint_hash": sha256_file(ia_checkpoint) == sources["ia_checkpoint_sha256_expected"],
        "ia_target_valid": ia_conclusion.get("IA_TARGET_VALID") == "YES",
        "ia_model_gain": ia_conclusion.get("MODEL_INTERACTION_GAIN") == "YES",
        "ia_development_gain": ia_conclusion.get("IA_CLOSED_LOOP_GAIN") == "YES",
        "ia_selected_checkpoint_hash": ia_conclusion.get("selected_checkpoint_sha256")
        == sources["ia_checkpoint_sha256_expected"],
        "historical_formal_seed_overlap_excluded": historical.get("FINAL_TEST_SEED_OVERLAP") == 0,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "V1_CHECKPOINT_SHA256": sha256_file(v1_checkpoint),
        "IA_CHECKPOINT_SHA256": sha256_file(ia_checkpoint),
        "ia_authoritative_conclusion": {
            key: ia_conclusion[key]
            for key in ("IA_TARGET_VALID", "MODEL_INTERACTION_GAIN", "IA_CLOSED_LOOP_GAIN", "selected_optimization_seed", "selected_best_epoch")
        },
        "historical_v1_formal_provenance": {
            "seed_block": "30-49",
            "v1_team_success": "41/60",
            "reused_for_current_formal": False,
        },
        "AGENTS_md_present": (REPO_ROOT / "AGENTS.md").is_file(),
        "CODEX_HANDOFF_md_present": (REPO_ROOT / "CODEX_HANDOFF.md").is_file(),
    }


def build_closed_config(config: Mapping[str, Any], formal_seeds: Sequence[int]) -> dict[str, Any]:
    base = load_json(REPO_ROOT / config["sources"]["frozen_closed_loop_config"])
    execution = config["execution"]
    base.update(
        {
            "formal_scenarios": list(SCENARIOS),
            "formal_seeds": list(formal_seeds),
            "smoke_seeds": list(formal_seeds[: int(config["smoke_gate"]["seed_count"])]),
            "v1_checkpoint": config["sources"]["v1_checkpoint"],
            "v1_checkpoint_sha256_expected": config["sources"]["v1_checkpoint_sha256_expected"],
            "v2_checkpoint": config["sources"]["ia_checkpoint"],
            "v2_checkpoint_sha256_expected": config["sources"]["ia_checkpoint_sha256_expected"],
            "sac_checkpoint": config["sources"]["sac_checkpoint"],
            "sac_checkpoint_sha256_expected": config["sources"]["sac_checkpoint_sha256_expected"],
            "base_execution_config": config["sources"]["base_execution_config"],
            "num_agents": execution["num_agents"],
            "top_k": execution["top_k"],
            "H_preview": execution["H_preview"],
            "max_steps": execution["max_steps"],
            "dt": execution["dt"],
            "peer_radius": execution["peer_radius"],
            "handoff_threshold_m": execution["handoff_threshold_m"],
            "forcing_gate": execution["forcing_gate"],
            "actor_observation_dim": execution["actor_observation_dim"],
            "phase": execution["phase"],
            "boundary_mode": execution["boundary_mode"],
            "proposal_config": copy.deepcopy(config["proposal_config"]),
        }
    )
    return base


def classify_selection(row: Mapping[str, Any], d_safe: float) -> str:
    if bool(row.get("selected_null")):
        return "NULL"
    if row.get("selected_candidate_id") is None:
        return "NO_CANDIDATE"
    edge_count = row.get("selected_align_edge_count")
    d_min = row.get("selected_minimum_d_min")
    t_risk = row.get("selected_maximum_T_risk")
    try:
        edge_count_value = int(edge_count)
        d_min_value = float(d_min)
        t_risk_value = float(t_risk)
    except (TypeError, ValueError):
        return "NEUTRAL"
    if edge_count_value <= 0 or not math.isfinite(d_min_value) or not math.isfinite(t_risk_value):
        return "NEUTRAL"
    return "RISKY" if d_min_value < d_safe or t_risk_value > 0.0 else "SAFE"


def relabel_episode(
    episode: dict[str, Any], agents: list[dict[str, Any]], method: str, d_safe: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    display = "GAT-V1" if method == METHOD_V1 else "GAT-V1-IA"
    episode.update(
        {
            "schema_version": SCHEMA_VERSION,
            "method": method,
            "method_display_name": display,
            "checkpoint_role": "V1" if method == METHOD_V1 else "IA",
        }
    )
    for row in agents:
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "method": method,
                "method_display_name": display,
                "checkpoint_role": "V1" if method == METHOD_V1 else "IA",
                "selection_source": method,
                "selected_type": row.get("reference_selected_type"),
                "selected_candidate_rank": row.get("selected_proposal_rank_1based"),
                "preview_hash": row.get("preview_bundle_hash"),
                "selected_interaction_risk_status": classify_selection(row, d_safe),
                "d_safe": d_safe,
            }
        )
    return episode, agents


def run_seed_subset(
    *,
    closed_config: Mapping[str, Any],
    execution_settings: Mapping[str, Any],
    multi_config: Any,
    policy: Any,
    v1_model: torch.nn.Module,
    ia_model: torch.nn.Module,
    device: torch.device,
    seeds: Sequence[int],
    phase: str,
    d_safe: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    jobs = [(scenario, int(seed), method) for scenario in SCENARIOS for seed in seeds for method in METHODS]
    shared_cache: dict[tuple[str, int], dict[str, Any]] = {}
    for index, (scenario, seed, method) in enumerate(jobs, start=1):
        key = (scenario, seed)
        if key not in shared_cache:
            shared = legacy.build_shared_selection_bundle(
                config=closed_config,
                execution_settings=execution_settings,
                multi_config=multi_config,
                policy=policy,
                gat_model=v1_model,
                gat_device=device,
                scenario=scenario,
                seed=seed,
            )
            paired.add_v2_checkpoint_plan(shared, ia_model, device)
            if len(set(shared["graph_input_hashes_by_method"].values())) != 1:
                raise RuntimeError("V1_GRAPH_INPUT_HASH != IA_GRAPH_INPUT_HASH")
            shared_cache[key] = shared
        internal = paired.METHOD_V1 if method == METHOD_V1 else paired.METHOD_V2
        episode, rows = paired.run_method_episode(
            config=closed_config,
            execution_settings=execution_settings,
            multi_config=multi_config,
            policy=policy,
            shared=shared_cache[key],
            method=internal,
        )
        episode, rows = relabel_episode(episode, rows, method, d_safe)
        episodes.append(episode)
        agents.extend(rows)
        print(
            f"[{phase} {index}/{len(jobs)}] {scenario} seed={seed} {method}: {episode['termination_reason']}",
            flush=True,
        )
    paired._annotate_agent_team_outcomes(agents, episodes)
    return episodes, agents


def paired_index(episodes: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    return {(str(row["scenario"]), int(row["seed"]), str(row["method"])): row for row in episodes}


def pair_fairness(episodes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    index = paired_index(episodes)
    mismatches = Counter()
    for scenario, seed in sorted({(row["scenario"], int(row["seed"])) for row in episodes}):
        v1 = index[(scenario, seed, METHOD_V1)]
        ia = index[(scenario, seed, METHOD_IA)]
        mismatches["candidate"] += int(v1["candidate_bundle_hash"] != ia["candidate_bundle_hash"])
        mismatches["preview"] += int(v1["preview_bundle_hash"] != ia["preview_bundle_hash"])
        mismatches["graph"] += int(v1["graph_input_hash"] != ia["graph_input_hash"])
        mismatches["initial_condition"] += int(v1["initial_condition_hash"] != ia["initial_condition_hash"])
    return {
        "candidate_hash_mismatch_count": mismatches["candidate"],
        "preview_hash_mismatch_count": mismatches["preview"],
        "graph_hash_mismatch_count": mismatches["graph"],
        "initial_condition_hash_mismatch_count": mismatches["initial_condition"],
    }


def smoke_gate(
    *,
    config: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
    core_before: Mapping[str, str],
    core_after: Mapping[str, str],
    model_before: Mapping[str, str],
    model_after: Mapping[str, str],
    policy_before: str,
    policy_after: str,
) -> dict[str, Any]:
    fairness = pair_fairness(episodes)
    checks = {
        "team_episode_count": len(episodes) == int(config["smoke_gate"]["expected_team_episode_count"]),
        "agent_record_count": len(agents) == int(config["smoke_gate"]["expected_agent_record_count"]),
        "candidate_fairness": fairness["candidate_hash_mismatch_count"] == 0,
        "preview_fairness": fairness["preview_hash_mismatch_count"] == 0,
        "graph_equality": fairness["graph_hash_mismatch_count"] == 0,
        "initial_condition_fairness": fairness["initial_condition_hash_mismatch_count"] == 0,
        "finite_logits": all(bool(row.get("logits_finite")) for row in agents),
        "class_mapping": all(bool(row.get("class_mapping_valid")) for row in agents),
        "null_mapping": all(bool(row.get("selected_null")) == (int(row.get("selected_class")) == 0) for row in agents),
        "one_shot_no_replanning": all(int(row.get("replanning_count", 0)) == 0 for row in episodes),
        "maximum_one_handoff": all(int(row.get("maximum_handoff_count_per_agent", 0)) <= 1 for row in episodes),
        "phase_preserving_handoff": all(not bool(row.get("phase_reset_on_switch")) for row in episodes),
        "terminal_goals_unchanged": all(bool(row.get("terminal_task_goals_unchanged")) for row in episodes),
        "historical_gate": all(bool(row.get("execution_historical_gate_verified")) for row in episodes),
        "selection_plan_unchanged": all(bool(row.get("selection_plan_unchanged")) for row in episodes),
        "core_hash_unchanged": dict(core_before) == dict(core_after),
        "model_hash_unchanged": dict(model_before) == dict(model_after),
        "policy_hash_unchanged": policy_before == policy_after,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "fairness": fairness,
        "success_outcomes_used_for_tuning": False,
        "smoke_semantics_changed_after_run": False,
        "smoke_episodes_included_in_formal": all(checks.values()),
    }


def mean_finite(values: Iterable[Any]) -> float | None:
    array = np.asarray([float(value) for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else None


def summary_rows(
    episodes: Sequence[Mapping[str, Any]], agents: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for scope in ("overall", *SCENARIOS):
            episode_members = [row for row in episodes if row["method"] == method and (scope == "overall" or row["scenario"] == scope)]
            agent_members = [row for row in agents if row["method"] == method and (scope == "overall" or row["scenario"] == scope)]
            successful = [row for row in episode_members if bool(row["team_success"])]
            count = len(episode_members)
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "method": method,
                "scope": scope,
                "episode_count": count,
                "agent_count": len(agent_members),
            }
            for field in ("team_success", "any_collision", "obstacle_collision", "inter_agent_collision", "timeout"):
                total = sum(bool(row[field]) for row in episode_members)
                record[f"{field}_count"] = total
                record[f"{field}_rate"] = total / count if count else None
            completed = sum(bool(row.get("agent_terminal_completed")) for row in agent_members)
            record.update(
                {
                    "agent_completion_count": completed,
                    "agent_completion_rate": completed / len(agent_members) if agent_members else None,
                    "mean_success_completion_step": mean_finite(row["completion_step"] for row in successful),
                    "mean_success_completion_time_s": mean_finite(row["completion_time_s"] for row in successful),
                    "mean_success_team_path_length_m": mean_finite(row["team_path_length_m"] for row in successful),
                    "mean_team_path_length_m": mean_finite(row["team_path_length_m"] for row in episode_members),
                    "mean_minimum_obstacle_clearance_m": mean_finite(row["minimum_obstacle_clearance_m"] for row in episode_members),
                    "mean_minimum_inter_agent_distance_m": mean_finite(row["minimum_inter_agent_distance_m"] for row in episode_members),
                    "mean_trajectory_smoothness": mean_finite(row["trajectory_smoothness"] for row in episode_members),
                }
            )
            rows.append(record)
    return [row for row in rows if row["scope"] == "overall"], [row for row in rows if row["scope"] != "overall"]


def selection_rows(agents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "schema_version", "method", "scenario", "seed", "agent_id", "team_success", "team_collision",
        "selected_type", "selected_null", "selected_candidate_id", "selected_candidate_rank",
        "selected_proposal_rank_by_coarse_score", "selected_proposal_rank_by_FP_SHEP",
        "selected_interaction_risk_status", "d_safe", "selected_minimum_d_min", "selected_maximum_T_risk",
        "selected_align_edge_count", "null_probability", "GAT_confidence", "class_count",
        "class_mapping_valid", "logits_finite", "candidate_bundle_hash", "preview_hash", "graph_input_hash",
    )
    return [{field: row.get(field) for field in fields} for row in agents]


def risky_analysis(agents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes = [
        ("scenario", "overall", lambda row: True),
        *[("scenario", scenario, lambda row, scenario=scenario: row["scenario"] == scenario) for scenario in SCENARIOS],
        ("outcome", "successful", lambda row: bool(row["team_success"])),
        ("outcome", "failed", lambda row: not bool(row["team_success"])),
        ("outcome", "collision", lambda row: bool(row["team_collision"])),
    ]
    for method in METHODS:
        for scope_type, scope, predicate in scopes:
            members = [row for row in agents if row["method"] == method and predicate(row)]
            counts = Counter(str(row["selected_interaction_risk_status"]) for row in members)
            proposal_count = counts["SAFE"] + counts["RISKY"] + counts["NEUTRAL"]
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "method": method,
                    "scope_type": scope_type,
                    "scope": scope,
                    "agent_decision_count": len(members),
                    "proposal_selection_count": proposal_count,
                    "safe_selection_count": counts["SAFE"],
                    "risky_selection_count": counts["RISKY"],
                    "neutral_selection_count": counts["NEUTRAL"],
                    "null_selection_count": counts["NULL"],
                    "no_candidate_count": counts["NO_CANDIDATE"],
                    "risky_selection_rate_all_agents": counts["RISKY"] / len(members) if members else None,
                    "risky_selection_rate_given_proposal": counts["RISKY"] / proposal_count if proposal_count else None,
                    "primary_rate_denominator": "all_agent_decisions",
                }
            )
    return rows


def null_analysis(agents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for scope in ("overall", *SCENARIOS):
            members = [row for row in agents if row["method"] == method and (scope == "overall" or row["scenario"] == scope)]
            probabilities = np.asarray([float(row["null_probability"]) for row in members], dtype=float)
            null_count = sum(bool(row["selected_null"]) for row in members)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "method": method,
                    "scope": scope,
                    "agent_count": len(members),
                    "null_selection_count": null_count,
                    "null_selection_rate": null_count / len(members) if members else None,
                    "mean_null_probability": float(np.mean(probabilities)) if probabilities.size else None,
                    "median_null_probability": float(np.median(probabilities)) if probabilities.size else None,
                    "p90_null_probability": float(np.percentile(probabilities, 90)) if probabilities.size else None,
                    "p95_null_probability": float(np.percentile(probabilities, 95)) if probabilities.size else None,
                    "null_calibration_performed": False,
                }
            )
    return rows


def reference_analysis(agents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for scope in ("overall", *SCENARIOS):
            members = [row for row in agents if row["method"] == method and (scope == "overall" or row["scenario"] == scope)]
            selected = [row for row in members if bool(row["reference_selected"])]
            reached = [row for row in selected if bool(row["reference_reached"])]
            completed = [row for row in reached if bool(row["terminal_completed_after_reference"])]
            obstacle = sum(bool(row["post_reference_obstacle_collision"]) for row in reached)
            inter = sum(bool(row["post_reference_inter_agent_collision"]) for row in reached)
            timeout = sum(bool(row["post_reference_timeout"]) for row in reached)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "method": method,
                    "scope": scope,
                    "agent_count": len(members),
                    "reference_selection_count": len(selected),
                    "reference_selection_rate": len(selected) / len(members) if members else None,
                    "reference_reach_count": len(reached),
                    "reference_reach_rate": len(reached) / len(selected) if selected else None,
                    "reached_to_terminal_count": len(completed),
                    "reached_to_terminal_rate": len(completed) / len(reached) if reached else None,
                    "post_reference_obstacle_collision_count": obstacle,
                    "post_reference_obstacle_collision_rate": obstacle / len(reached) if reached else None,
                    "post_reference_inter_agent_collision_count": inter,
                    "post_reference_inter_agent_collision_rate": inter / len(reached) if reached else None,
                    "post_reference_timeout_count": timeout,
                    "post_reference_timeout_rate": timeout / len(reached) if reached else None,
                }
            )
    return rows


def build_paired_outcomes(
    episodes: Sequence[Mapping[str, Any]], agents: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    episode_lookup = paired_index(episodes)
    agent_lookup = {(row["scenario"], int(row["seed"]), row["method"], int(row["agent_id"])): row for row in agents}
    rows: list[dict[str, Any]] = []
    for scenario, seed in sorted({(row["scenario"], int(row["seed"])) for row in episodes}):
        v1 = episode_lookup[(scenario, seed, METHOD_V1)]
        ia = episode_lookup[(scenario, seed, METHOD_IA)]
        if v1["team_success"] and ia["team_success"]:
            outcome = "BOTH_SUCCESS"
        elif ia["team_success"]:
            outcome = "IA_ONLY_SUCCESS"
        elif v1["team_success"]:
            outcome = "V1_ONLY_SUCCESS"
        else:
            outcome = "BOTH_FAIL"
        risk_safe_agents = 0
        for agent_id in range(3):
            left = agent_lookup[(scenario, seed, METHOD_V1, agent_id)]
            right = agent_lookup[(scenario, seed, METHOD_IA, agent_id)]
            risk_safe_agents += int(
                left["selected_interaction_risk_status"] == "RISKY"
                and right["selected_interaction_risk_status"] == "SAFE"
            )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario": scenario,
                "seed": seed,
                "pair_outcome": outcome,
                "both_success": outcome == "BOTH_SUCCESS",
                "ia_only_success": outcome == "IA_ONLY_SUCCESS",
                "v1_only_success": outcome == "V1_ONLY_SUCCESS",
                "both_fail": outcome == "BOTH_FAIL",
                "v1_team_success": bool(v1["team_success"]),
                "ia_team_success": bool(ia["team_success"]),
                "v1_any_collision": bool(v1["any_collision"]),
                "ia_any_collision": bool(ia["any_collision"]),
                "v1_inter_agent_collision": bool(v1["inter_agent_collision"]),
                "ia_inter_agent_collision": bool(ia["inter_agent_collision"]),
                "v1_timeout": bool(v1["timeout"]),
                "ia_timeout": bool(ia["timeout"]),
                "v1_collision_to_ia_success": bool(v1["any_collision"] and ia["team_success"]),
                "v1_timeout_to_ia_success": bool(v1["timeout"] and ia["team_success"]),
                "v1_success_to_ia_collision": bool(v1["team_success"] and ia["any_collision"]),
                "v1_success_to_ia_timeout": bool(v1["team_success"] and ia["timeout"]),
                "matched_agent_v1_risky_to_ia_safe_count": risk_safe_agents,
                "v1_risky_failure_to_ia_safe_success": bool(
                    not v1["team_success"] and ia["team_success"] and risk_safe_agents > 0
                ),
                "candidate_bundle_hash_match": v1["candidate_bundle_hash"] == ia["candidate_bundle_hash"],
                "preview_hash_match": v1["preview_bundle_hash"] == ia["preview_bundle_hash"],
                "graph_input_hash_match": v1["graph_input_hash"] == ia["graph_input_hash"],
                "completion_time_difference_ia_minus_v1_s": (
                    float(ia["completion_time_s"]) - float(v1["completion_time_s"])
                    if outcome == "BOTH_SUCCESS" else None
                ),
                "team_path_length_difference_ia_minus_v1_m": (
                    float(ia["team_path_length_m"]) - float(v1["team_path_length_m"])
                    if outcome == "BOTH_SUCCESS" else None
                ),
                "trajectory_smoothness_difference_ia_minus_v1": (
                    float(ia["trajectory_smoothness"]) - float(v1["trajectory_smoothness"])
                    if outcome == "BOTH_SUCCESS" else None
                ),
                "minimum_inter_agent_distance_difference_ia_minus_v1_m": (
                    float(ia["minimum_inter_agent_distance_m"]) - float(v1["minimum_inter_agent_distance_m"])
                    if outcome == "BOTH_SUCCESS" else None
                ),
            }
        )
    return rows


def mcnemar_rows(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = paired_index(episodes)
    rows: list[dict[str, Any]] = []
    for scope in ("overall", "multi_agent", "open", "sparse_static"):
        keys = sorted(
            {
                (row["scenario"], int(row["seed"]))
                for row in episodes
                if scope == "overall" or row["scenario"] == scope
            }
        )
        if not keys:
            continue
        for metric in ("team_success", "any_collision", "inter_agent_collision", "timeout"):
            v1 = [bool(index[(*key, METHOD_V1)][metric]) for key in keys]
            ia = [bool(index[(*key, METHOD_IA)][metric]) for key in keys]
            v1_only = sum(left and not right for left, right in zip(v1, ia))
            ia_only = sum(right and not left for left, right in zip(v1, ia))
            discordant = v1_only + ia_only
            p_value = float(binomtest(min(v1_only, ia_only), discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
            v1_rate = float(np.mean(v1))
            ia_rate = float(np.mean(ia))
            positive_improvement = ia_rate - v1_rate if metric == "team_success" else v1_rate - ia_rate
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "pair_count": len(keys),
                    "both_false_count": sum(not left and not right for left, right in zip(v1, ia)),
                    "both_true_count": sum(left and right for left, right in zip(v1, ia)),
                    "v1_only_true_count": v1_only,
                    "ia_only_true_count": ia_only,
                    "discordant_count": discordant,
                    "v1_rate": v1_rate,
                    "ia_rate": ia_rate,
                    "ia_minus_v1_rate": ia_rate - v1_rate,
                    "absolute_rate_difference": abs(ia_rate - v1_rate),
                    "improvement_positive_rate_difference": positive_improvement,
                    "exact_two_sided_p_value": p_value,
                }
            )
    return rows


def descriptive(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0 if array.size else None,
        "median": float(np.median(array)) if array.size else None,
    }


def continuous_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    index = paired_index(episodes)
    field_names = (
        "completion_time_s",
        "team_path_length_m",
        "trajectory_smoothness",
        "minimum_inter_agent_distance_m",
    )
    rows: list[dict[str, Any]] = []
    for scope in ("overall", *SCENARIOS):
        keys = sorted(
            {
                (row["scenario"], int(row["seed"]))
                for row in episodes
                if (scope == "overall" or row["scenario"] == scope)
                and bool(row["team_success"])
                and bool(index[(row["scenario"], int(row["seed"]), METHOD_V1)]["team_success"])
                and bool(index[(row["scenario"], int(row["seed"]), METHOD_IA)]["team_success"])
            }
        )
        for field in field_names:
            v1 = [float(index[(*key, METHOD_V1)][field]) for key in keys]
            ia = [float(index[(*key, METHOD_IA)][field]) for key in keys]
            difference = [right - left for left, right in zip(v1, ia)]
            rows.append(
                {
                    "scope": scope,
                    "metric": field,
                    "subset": "both_methods_team_success",
                    "pair_count": len(keys),
                    "v1": descriptive(v1),
                    "ia": descriptive(ia),
                    "paired_difference_ia_minus_v1": descriptive(difference),
                    "failed_completion_time_filled_with_zero": False,
                }
            )
    return {"schema_version": SCHEMA_VERSION, "rows": rows}


def lookup(rows: Sequence[Mapping[str, Any]], **keys: Any) -> Mapping[str, Any]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in keys.items())]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row for {keys}, found {len(matches)}")
    return matches[0]


def formal_decision(
    *,
    config: Mapping[str, Any],
    methods: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
    risky: Sequence[Mapping[str, Any]],
    nulls: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    formal_manifest: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    v1 = lookup(methods, method=METHOD_V1, scope="overall")
    ia = lookup(methods, method=METHOD_IA, scope="overall")
    v1_multi = lookup(scenarios, method=METHOD_V1, scope="multi_agent")
    ia_multi = lookup(scenarios, method=METHOD_IA, scope="multi_agent")
    v1_risk = lookup(risky, method=METHOD_V1, scope_type="scenario", scope="overall")
    ia_risk = lookup(risky, method=METHOD_IA, scope_type="scenario", scope="overall")
    v1_null = lookup(nulls, method=METHOD_V1, scope="overall")
    ia_null = lookup(nulls, method=METHOD_IA, scope="overall")
    v1_ref = lookup(references, method=METHOD_V1, scope="overall")
    ia_ref = lookup(references, method=METHOD_IA, scope="overall")
    success_test = lookup(tests, scope="overall", metric="team_success")
    collision_test = lookup(tests, scope="overall", metric="any_collision")
    success_gain = float(ia["team_success_rate"] - v1["team_success_rate"])
    collision_change = float(ia["any_collision_rate"] - v1["any_collision_rate"])
    multi_gain = float(ia_multi["team_success_rate"] - v1_multi["team_success_rate"])
    thresholds = config["formal_gain_decision"]
    yes = (
        success_gain >= float(thresholds["yes_minimum_overall_team_success_gain"])
        and collision_change <= float(thresholds["yes_maximum_overall_collision_worsening"])
        and multi_gain > float(thresholds["material_multi_agent_regression"])
    )
    weak = (
        success_gain < float(thresholds["yes_minimum_overall_team_success_gain"])
        and multi_gain >= float(thresholds["weak_minimum_multi_agent_success_gain"])
        and success_gain >= 0.0
        and collision_change <= 0.0
    )
    gain = "YES" if yes else "WEAK" if weak else "NO"
    if gain == "YES":
        final_checkpoint = "IA"
    elif gain == "NO":
        final_checkpoint = "V1"
    else:
        weak_checks = {
            "overall_success_non_decrease": success_gain >= 0.0,
            "multi_agent_gain": multi_gain >= float(thresholds["weak_minimum_multi_agent_success_gain"]),
            "any_collision_non_worsening": collision_change <= 0.0,
            "inter_agent_collision_non_worsening": ia["inter_agent_collision_rate"] <= v1["inter_agent_collision_rate"],
            "timeout_non_worsening": ia["timeout_rate"] <= v1["timeout_rate"],
            "reference_reach_retained": ia_ref["reference_reach_rate"] - v1_ref["reference_reach_rate"]
            >= -float(thresholds["weak_final_ia_maximum_reference_reach_decline"]),
            "reached_to_terminal_retained": ia_ref["reached_to_terminal_rate"] - v1_ref["reached_to_terminal_rate"]
            >= -float(thresholds["weak_final_ia_maximum_reached_to_terminal_decline"]),
        }
        final_checkpoint = "IA" if all(weak_checks.values()) else "V1"
    proceed = final_checkpoint == "IA"
    headroom = "NO" if gain == "YES" else "YES" if gain == "NO" else "NOT_ESTABLISHED"
    recommendation = "FINAL_BASELINE_COMPARISON" if proceed else "KEEP_V1_AND_STOP_IA"
    return {
        "schema_version": SCHEMA_VERSION,
        "CONTEXT_RECOVERY_VALID": "YES" if context["status"] == "PASSED" else "NO",
        "V1_CHECKPOINT_VALID": "YES" if context["checks"]["v1_checkpoint_hash"] else "NO",
        "IA_CHECKPOINT_VALID": "YES" if context["checks"]["ia_checkpoint_hash"] else "NO",
        "IA_CHECKPOINT_SHA256": context["IA_CHECKPOINT_SHA256"],
        "FINAL_TEST_SEED_BLOCK": formal_manifest["formal_seed_block"],
        "FINAL_TEST_SEED_OVERLAP": formal_manifest["FINAL_TEST_SEED_OVERLAP"],
        "CANDIDATE_BUNDLE_FAIRNESS": "YES" if integrity["fairness"]["candidate_hash_mismatch_count"] == 0 else "NO",
        "PREVIEW_FAIRNESS": "YES" if integrity["fairness"]["preview_hash_mismatch_count"] == 0 else "NO",
        "V1_IA_GRAPH_INPUT_EQUAL": "YES" if integrity["fairness"]["graph_hash_mismatch_count"] == 0 else "NO",
        "CORE_INTEGRITY_VALID": "YES" if integrity["status"] == "PASSED" else "NO",
        "V1_TEAM_SUCCESS": v1["team_success_rate"],
        "IA_TEAM_SUCCESS": ia["team_success_rate"],
        "IA_SUCCESS_GAIN_PP": 100.0 * success_gain,
        "V1_ANY_COLLISION": v1["any_collision_rate"],
        "IA_ANY_COLLISION": ia["any_collision_rate"],
        "COLLISION_CHANGE_PP": 100.0 * collision_change,
        "V1_TIMEOUT": v1["timeout_rate"],
        "IA_TIMEOUT": ia["timeout_rate"],
        "V1_MULTI_AGENT_SUCCESS": v1_multi["team_success_rate"],
        "IA_MULTI_AGENT_SUCCESS": ia_multi["team_success_rate"],
        "MULTI_AGENT_SUCCESS_GAIN_PP": 100.0 * multi_gain,
        "V1_INTER_AGENT_COLLISION": v1["inter_agent_collision_rate"],
        "IA_INTER_AGENT_COLLISION": ia["inter_agent_collision_rate"],
        "V1_RISKY_SELECTION_RATE": v1_risk["risky_selection_rate_all_agents"],
        "IA_RISKY_SELECTION_RATE": ia_risk["risky_selection_rate_all_agents"],
        "RISKY_SELECTION_CHANGE_PP": 100.0 * (
            ia_risk["risky_selection_rate_all_agents"] - v1_risk["risky_selection_rate_all_agents"]
        ),
        "V1_NULL_RATE": v1_null["null_selection_rate"],
        "IA_NULL_RATE": ia_null["null_selection_rate"],
        "V1_REFERENCE_REACH_RATE": v1_ref["reference_reach_rate"],
        "IA_REFERENCE_REACH_RATE": ia_ref["reference_reach_rate"],
        "V1_REACHED_TO_TERMINAL": v1_ref["reached_to_terminal_rate"],
        "IA_REACHED_TO_TERMINAL": ia_ref["reached_to_terminal_rate"],
        "MCMENAR_SUCCESS_P": success_test["exact_two_sided_p_value"],
        "MCMENAR_COLLISION_P": collision_test["exact_two_sided_p_value"],
        "IA_FORMAL_GAIN": gain,
        "IA_TEAM_SUCCESS_90_TARGET_REACHED": "YES" if ia["team_success_rate"] >= float(thresholds["team_success_90_target"]) else "NO",
        "FINAL_GAT_CHECKPOINT": final_checkpoint,
        "PROCEED_TO_FINAL_BASELINE_COMPARISON": "YES" if proceed else "NO",
        "EXISTING_GAT_SUPERVISION_ONLY_HEADROOM_EXHAUSTED": headroom,
        "RECOMMENDED_NEXT_STEP": recommendation,
        "formal_gain_thresholds": thresholds,
        "training_or_tuning_performed": False,
        "formal_result_used_offline_top1_for_selection": False,
        "status": "COMPLETE",
    }


def report_text(
    conclusion: Mapping[str, Any],
    methods: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
    risky: Sequence[Mapping[str, Any]],
    nulls: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
) -> str:
    method = {(row["method"], row["scope"]): row for row in methods}
    scenario = {(row["method"], row["scope"]): row for row in scenarios}
    risk = {(row["method"], row["scope_type"], row["scope"]): row for row in risky}
    null = {(row["method"], row["scope"]): row for row in nulls}
    ref = {(row["method"], row["scope"]): row for row in references}
    pair_counts = Counter(row["pair_outcome"] for row in pairs)
    success_test = lookup(tests, scope="overall", metric="team_success")
    collision_test = lookup(tests, scope="overall", metric="any_collision")
    lines = [
        "# Formal GAT-V1 vs GAT-V1-IA Closed-Loop Evaluation",
        "",
        "## Executive result",
        "",
        f"`IA_FORMAL_GAIN = {conclusion['IA_FORMAL_GAIN']}` and `FINAL_GAT_CHECKPOINT = {conclusion['FINAL_GAT_CHECKPOINT']}`. "
        f"IA team success is {conclusion['IA_TEAM_SUCCESS']:.1%} versus V1 {conclusion['V1_TEAM_SUCCESS']:.1%} "
        f"({conclusion['IA_SUCCESS_GAIN_PP']:+.1f} pp); any collision is {conclusion['IA_ANY_COLLISION']:.1%} versus "
        f"{conclusion['V1_ANY_COLLISION']:.1%} ({conclusion['COLLISION_CHANGE_PP']:+.1f} pp IA minus V1).",
        "",
        "This conclusion uses only the newly frozen independent formal seed block and does not use the development gain as formal evidence.",
        "",
        "## Frozen protocol and integrity",
        "",
        f"- Formal seeds: `{conclusion['FINAL_TEST_SEED_BLOCK']}`; overlap: {conclusion['FINAL_TEST_SEED_OVERLAP']}.",
        f"- V1/IA checkpoint valid: `{conclusion['V1_CHECKPOINT_VALID']}` / `{conclusion['IA_CHECKPOINT_VALID']}`.",
        f"- Candidate/preview/graph fairness: `{conclusion['CANDIDATE_BUNDLE_FAIRNESS']}` / `{conclusion['PREVIEW_FAIRNESS']}` / `{conclusion['V1_IA_GRAPH_INPUT_EQUAL']}`.",
        f"- Core integrity: `{conclusion['CORE_INTEGRITY_VALID']}`.",
        "- Top-K 10, H4 preview, 220 steps, deterministic frozen SAC-DMP, historical vector gate, boundary-free one-shot phase-preserving handoff.",
        "",
        "## Formal team outcomes",
        "",
        "| scope | method | success | any collision | obstacle collision | inter-agent collision | timeout | agent completion |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in ("overall", *SCENARIOS):
        for name in METHODS:
            row = method[(name, scope)] if scope == "overall" else scenario[(name, scope)]
            lines.append(
                f"| {scope} | {name} | {row['team_success_count']}/{row['episode_count']} ({row['team_success_rate']:.1%}) | "
                f"{row['any_collision_rate']:.1%} | {row['obstacle_collision_rate']:.1%} | "
                f"{row['inter_agent_collision_rate']:.1%} | {row['timeout_rate']:.1%} | {row['agent_completion_rate']:.1%} |"
            )
    lines.extend(
        [
            "",
            "## Interaction, null, and reference behavior",
            "",
            f"Overall risky selection per agent decision is V1 {conclusion['V1_RISKY_SELECTION_RATE']:.1%} versus IA "
            f"{conclusion['IA_RISKY_SELECTION_RATE']:.1%} ({conclusion['RISKY_SELECTION_CHANGE_PP']:+.1f} pp). "
            f"Conditional-on-proposal rates are {risk[(METHOD_V1, 'scenario', 'overall')]['risky_selection_rate_given_proposal']:.1%} and "
            f"{risk[(METHOD_IA, 'scenario', 'overall')]['risky_selection_rate_given_proposal']:.1%}.",
            f"Null selection is V1 {conclusion['V1_NULL_RATE']:.1%} versus IA {conclusion['IA_NULL_RATE']:.1%}; mean null probability is "
            f"{null[(METHOD_V1, 'overall')]['mean_null_probability']:.3f}/{null[(METHOD_IA, 'overall')]['mean_null_probability']:.3f}.",
            f"Reference reach conditional on selection is {conclusion['V1_REFERENCE_REACH_RATE']:.1%}/{conclusion['IA_REFERENCE_REACH_RATE']:.1%}; "
            f"reached-to-terminal completion is {conclusion['V1_REACHED_TO_TERMINAL']:.1%}/{conclusion['IA_REACHED_TO_TERMINAL']:.1%}.",
            "",
            "For multi_agent (mixed obstacle and agent-interaction scenario):",
            "",
            f"- Team success: V1 {conclusion['V1_MULTI_AGENT_SUCCESS']:.1%}, IA {conclusion['IA_MULTI_AGENT_SUCCESS']:.1%} ({conclusion['MULTI_AGENT_SUCCESS_GAIN_PP']:+.1f} pp).",
            f"- Risky selection: V1 {risk[(METHOD_V1, 'scenario', 'multi_agent')]['risky_selection_rate_all_agents']:.1%}, IA {risk[(METHOD_IA, 'scenario', 'multi_agent')]['risky_selection_rate_all_agents']:.1%}.",
            f"- Null selection: V1 {null[(METHOD_V1, 'multi_agent')]['null_selection_rate']:.1%}, IA {null[(METHOD_IA, 'multi_agent')]['null_selection_rate']:.1%}.",
            f"- Reference reach: V1 {ref[(METHOD_V1, 'multi_agent')]['reference_reach_rate']:.1%}, IA {ref[(METHOD_IA, 'multi_agent')]['reference_reach_rate']:.1%}.",
            "",
            "## Paired and statistical analysis",
            "",
            f"Paired outcomes: both success {pair_counts['BOTH_SUCCESS']}, IA-only {pair_counts['IA_ONLY_SUCCESS']}, "
            f"V1-only {pair_counts['V1_ONLY_SUCCESS']}, both fail {pair_counts['BOTH_FAIL']}.",
            f"McNemar success discordant counts V1-only/IA-only are {success_test['v1_only_true_count']}/{success_test['ia_only_true_count']} "
            f"(two-sided exact p={success_test['exact_two_sided_p_value']:.6f}).",
            f"McNemar any-collision discordant counts V1-only/IA-only are {collision_test['v1_only_true_count']}/{collision_test['ia_only_true_count']} "
            f"(two-sided exact p={collision_test['exact_two_sided_p_value']:.6f}).",
            "Continuous paired metrics are reported only for both-success episodes; failed completion times are never filled with zero.",
            "",
            "## Final decision and stop rule",
            "",
            f"- `IA_FORMAL_GAIN = {conclusion['IA_FORMAL_GAIN']}`",
            f"- `IA_TEAM_SUCCESS_90_TARGET_REACHED = {conclusion['IA_TEAM_SUCCESS_90_TARGET_REACHED']}`",
            f"- `FINAL_GAT_CHECKPOINT = {conclusion['FINAL_GAT_CHECKPOINT']}`",
            f"- `PROCEED_TO_FINAL_BASELINE_COMPARISON = {conclusion['PROCEED_TO_FINAL_BASELINE_COMPARISON']}`",
            f"- `EXISTING_GAT_SUPERVISION_ONLY_HEADROOM_EXHAUSTED = {conclusion['EXISTING_GAT_SUPERVISION_ONLY_HEADROOM_EXHAUSTED']}`",
            f"- `RECOMMENDED_NEXT_STEP = {conclusion['RECOMMENDED_NEXT_STEP']}`",
            "",
            "The experiment stops after this two-checkpoint formal comparison. No baseline methods, V2, ERR, 24-layout, retraining, calibration, post-hoc threshold change, or extra seeds were run.",
            "",
        ]
    )
    return "\n".join(lines)


def empty_outputs(output_dir: Path) -> None:
    for name in (
        "smoke_results.csv", "episode_results.csv", "agent_results.csv", "selection_results.csv",
        "method_summary.csv", "scenario_summary.csv", "risky_selection_analysis.csv", "null_analysis.csv",
        "reference_transition_analysis.csv", "paired_outcomes.csv",
    ):
        write_csv(output_dir / name, [])
    write_json(output_dir / "statistical_tests.json", {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"})
    write_json(output_dir / "continuous_metrics.json", {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"})


def run_experiment(config: Mapping[str, Any], output_dir: Path) -> Path:
    validate_config(config)
    started = time.perf_counter()
    resume_preflight = output_dir.is_dir()
    if resume_preflight:
        required = (
            "config.json", "context_recovery_manifest.json", "all_used_seeds_manifest.json",
            "formal_seed_manifest.json", "episode_results.csv", "smoke_results.csv",
        )
        if any(not (output_dir / name).is_file() for name in required):
            raise RuntimeError("existing output directory is not a resumable preflight artifact")
        if list(csv.DictReader((output_dir / "episode_results.csv").open("r", encoding="utf-8-sig"))):
            raise RuntimeError("preflight resume is forbidden after any formal episode")
        if list(csv.DictReader((output_dir / "smoke_results.csv").open("r", encoding="utf-8-sig"))):
            raise RuntimeError("preflight resume is forbidden after any smoke episode")
        context = load_json(output_dir / "context_recovery_manifest.json")
        seed_audit = load_json(output_dir / "all_used_seeds_manifest.json")
        formal_manifest = load_json(output_dir / "formal_seed_manifest.json")
        formal_seeds = [int(value) for value in formal_manifest["formal_seeds"]]
        if formal_manifest["status"] != "PASSED" or formal_seeds != list(range(60, 80)):
            raise RuntimeError("frozen preflight seed manifest is invalid")
        resolved_config = load_json(output_dir / "config.json")
        previous_sac_hash = resolved_config["sources"]["sac_checkpoint_sha256_expected"]
        resolved_config["sources"] = copy.deepcopy(config["sources"])
        resolved_config.update(
            {
                "preflight_resume_at": datetime.now().astimezone().isoformat(),
                "preflight_episode_count_before_resume": 0,
                "preflight_correction": {
                    "field": "sources.sac_checkpoint_sha256_expected",
                    "previous_value": previous_sac_hash,
                    "corrected_value": config["sources"]["sac_checkpoint_sha256_expected"],
                    "reason": "transcription_error_in_expected_hash_only",
                    "evaluation_semantics_changed": False,
                    "formal_seed_block_changed": False,
                },
            }
        )
        write_json(output_dir / "config.json", resolved_config)
        print(f"resuming zero-episode preflight with frozen seed block: {formal_manifest['formal_seed_block']}", flush=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        empty_outputs(output_dir)
        context = build_context(config)
        write_json(output_dir / "context_recovery_manifest.json", context)
        if context["status"] != "PASSED":
            raise RuntimeError(f"context recovery failed: {context['checks']}")

        roots = [(REPO_ROOT / value).resolve() for value in config["seed_audit"]["roots"]]
        seed_audit = audit_all_used_seeds(roots, exclude_dir=output_dir)
        used = set(seed_audit["all_used_seed_values"])
        for start, stop in config["seed_audit"]["explicitly_forbidden_intervals"]:
            used.update(range(int(start), int(stop) + 1))
        used.update(int(value) for value in config["seed_audit"]["known_ia_development_seeds"])
        formal_seeds = choose_seed_block(
            used,
            block_size=int(config["seed_audit"]["contiguous_block_size"]),
            minimum=int(config["seed_audit"]["minimum_seed"]),
            maximum=int(config["seed_audit"]["maximum_seed_for_block_search"]),
        )
        overlap = sorted(set(formal_seeds) & used)
        seed_audit.update(
            {
                "explicit_forbidden_values": sorted(
                    set().union(
                        *[
                            set(range(int(start), int(stop) + 1))
                            for start, stop in config["seed_audit"]["explicitly_forbidden_intervals"]
                        ],
                        set(config["seed_audit"]["known_ia_development_seeds"]),
                    )
                ),
                "selected_formal_seed_block": formal_seeds,
                "selection_rule": "minimum_start_contiguous_20_seed_block_absent_from_all_recovered_usage",
            }
        )
        formal_manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASSED" if not overlap and len(formal_seeds) == 20 else "FAILED",
            "formal_seeds": formal_seeds,
            "formal_seed_block": f"{formal_seeds[0]}-{formal_seeds[-1]}",
            "smoke_seeds": formal_seeds[: int(config["smoke_gate"]["seed_count"])],
            "FINAL_TEST_SEED_OVERLAP": len(overlap),
            "overlap_values": overlap,
            "seed_block_frozen_before_any_outcome": True,
            "result_dependent_replacement_allowed": False,
            "all_used_seed_manifest_sha256": None,
        }
        write_json(output_dir / "all_used_seeds_manifest.json", seed_audit)
        formal_manifest["all_used_seed_manifest_sha256"] = sha256_file(output_dir / "all_used_seeds_manifest.json")
        write_json(output_dir / "formal_seed_manifest.json", formal_manifest)
        if formal_manifest["status"] != "PASSED":
            raise RuntimeError("formal seed audit failed")

        resolved_config = copy.deepcopy(dict(config))
        resolved_config.update(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "resolved_output_dir": str(output_dir),
                "formal_seeds": formal_seeds,
                "smoke_seeds": formal_manifest["smoke_seeds"],
                "formal_seed_manifest_sha256_before_outcomes": sha256_file(output_dir / "formal_seed_manifest.json"),
                "thresholds_frozen_before_outcomes": True,
            }
        )
        write_json(output_dir / "config.json", resolved_config)
        print(f"formal seed block frozen: {formal_manifest['formal_seed_block']}", flush=True)

    if context["status"] != "PASSED":
        raise RuntimeError(f"context recovery failed: {context['checks']}")

    closed_config = build_closed_config(config, formal_seeds)
    core_paths = tuple(config["expected_core_hashes"])
    core_before = core_hashes(core_paths)
    if core_before != dict(config["expected_core_hashes"]):
        raise RuntimeError("authoritative core source hash mismatch")
    auxiliary_paths = (
        "Multi-agent_Algo_lib/scripts/evaluate_gat_closed_loop.py",
        "Multi-agent_Algo_lib/scripts/evaluate_gat_stage1_v2_closed_loop.py",
        "Multi-agent_Algo_lib/scripts/evaluate_gat_v1_ia_formal_closed_loop.py",
        "configs/evaluation/gat_v1_ia_formal_closed_loop.json",
    )
    auxiliary_before = core_hashes(auxiliary_paths)
    checkpoint_paths = {
        "v1": (REPO_ROOT / config["sources"]["v1_checkpoint"]).resolve(),
        "ia": (REPO_ROOT / config["sources"]["ia_checkpoint"]).resolve(),
        "sac": (REPO_ROOT / config["sources"]["sac_checkpoint"]).resolve(),
    }
    checkpoint_before = {name: sha256_file(path) for name, path in checkpoint_paths.items()}
    if checkpoint_before != {
        "v1": config["sources"]["v1_checkpoint_sha256_expected"],
        "ia": config["sources"]["ia_checkpoint_sha256_expected"],
        "sac": config["sources"]["sac_checkpoint_sha256_expected"],
    }:
        raise RuntimeError("checkpoint hash mismatch before formal outcomes")

    execution_settings = legacy._build_execution_settings(closed_config)
    multi_config = build_single_distribution_multi_config(
        num_agents=int(config["execution"]["num_agents"]),
        max_steps=int(config["execution"]["max_steps"]),
    )
    policy, loaded_sac = _load_policy(execution_settings, multi_config)
    if loaded_sac.resolve() != checkpoint_paths["sac"]:
        raise RuntimeError("SAC loader resolved an unexpected checkpoint")
    policy_before = _policy_parameter_sha256(policy)
    stage1_config = load_json(REPO_ROOT / config["sources"]["stage1_config"])
    device = resolve_device(stage1_config["training"]["device"])
    v1_model = load_model_checkpoint(checkpoint_paths["v1"], stage1_config, device)
    ia_model = load_model_checkpoint(checkpoint_paths["ia"], stage1_config, device)
    payload_v1 = torch.load(checkpoint_paths["v1"], map_location="cpu", weights_only=False)
    payload_ia = torch.load(checkpoint_paths["ia"], map_location="cpu", weights_only=False)
    strict_checkpoint_checks = {
        "v1_schema": payload_v1.get("schema_version") == "gat_stage1_training_v1",
        "ia_schema": payload_ia.get("schema_version") == "gat_stage1_training_v1",
        "same_model_config": payload_v1.get("model_config") == payload_ia.get("model_config"),
        "ia_optimization_seed": int(payload_ia.get("optimization_seed")) == 1,
        "ia_epoch": int(payload_ia.get("epoch")) == 17,
    }
    if not all(strict_checkpoint_checks.values()):
        raise RuntimeError(f"strict checkpoint load failed: {strict_checkpoint_checks}")
    for model in (v1_model, ia_model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model_before = {"v1": model_hash(v1_model), "ia": model_hash(ia_model)}

    smoke_seeds = formal_manifest["smoke_seeds"]
    smoke_episodes, smoke_agents = run_seed_subset(
        closed_config=closed_config,
        execution_settings=execution_settings,
        multi_config=multi_config,
        policy=policy,
        v1_model=v1_model,
        ia_model=ia_model,
        device=device,
        seeds=smoke_seeds,
        phase="smoke",
        d_safe=float(config["interaction_risk_contract"]["d_safe"]),
    )
    write_csv(output_dir / "smoke_results.csv", smoke_episodes)
    smoke = smoke_gate(
        config=config,
        episodes=smoke_episodes,
        agents=smoke_agents,
        core_before=core_before,
        core_after=core_hashes(core_paths),
        model_before=model_before,
        model_after={"v1": model_hash(v1_model), "ia": model_hash(ia_model)},
        policy_before=policy_before,
        policy_after=_policy_parameter_sha256(policy),
    )
    write_json(output_dir / "smoke_gate.json", smoke)
    if smoke["status"] != "PASSED":
        write_json(
            output_dir / "conclusion.json",
            {"schema_version": SCHEMA_VERSION, "status": "SMOKE_FAILED", "IA_FORMAL_GAIN": "NOT_RUN", "failed_checks": smoke["checks"]},
        )
        raise RuntimeError(f"smoke gate failed: {smoke['checks']}")

    remaining = [seed for seed in formal_seeds if seed not in smoke_seeds]
    formal_episodes, formal_agents = run_seed_subset(
        closed_config=closed_config,
        execution_settings=execution_settings,
        multi_config=multi_config,
        policy=policy,
        v1_model=v1_model,
        ia_model=ia_model,
        device=device,
        seeds=remaining,
        phase="formal",
        d_safe=float(config["interaction_risk_contract"]["d_safe"]),
    )
    episodes = smoke_episodes + formal_episodes
    agents = smoke_agents + formal_agents
    method_rows, scenario_rows = summary_rows(episodes, agents)
    selections = selection_rows(agents)
    risky_rows = risky_analysis(agents)
    null_rows = null_analysis(agents)
    reference_rows = reference_analysis(agents)
    pair_rows = build_paired_outcomes(episodes, agents)
    test_rows = mcnemar_rows(episodes)
    continuous = continuous_metrics(episodes)

    core_after = core_hashes(core_paths)
    auxiliary_after = core_hashes(auxiliary_paths)
    checkpoint_after = {name: sha256_file(path) for name, path in checkpoint_paths.items()}
    model_after = {"v1": model_hash(v1_model), "ia": model_hash(ia_model)}
    policy_after = _policy_parameter_sha256(policy)
    fairness = pair_fairness(episodes)
    integrity_checks = {
        "expected_team_episode_count": len(episodes) == int(config["formal_scale"]["expected_team_episode_count"]),
        "expected_agent_record_count": len(agents) == int(config["formal_scale"]["expected_agent_record_count"]),
        "every_scenario_seed_method_once": len({(row["scenario"], int(row["seed"]), row["method"]) for row in episodes}) == 120,
        "candidate_fairness": fairness["candidate_hash_mismatch_count"] == 0,
        "preview_fairness": fairness["preview_hash_mismatch_count"] == 0,
        "graph_equality": fairness["graph_hash_mismatch_count"] == 0,
        "initial_condition_fairness": fairness["initial_condition_hash_mismatch_count"] == 0,
        "finite_logits": all(bool(row["logits_finite"]) for row in agents),
        "class_mapping": all(bool(row["class_mapping_valid"]) for row in agents),
        "one_shot_no_replanning": all(int(row.get("replanning_count", 0)) == 0 for row in episodes),
        "core_hash_unchanged": core_before == core_after,
        "auxiliary_hash_unchanged": auxiliary_before == auxiliary_after,
        "checkpoint_hash_unchanged": checkpoint_before == checkpoint_after,
        "model_hash_unchanged": model_before == model_after,
        "policy_hash_unchanged": policy_before == policy_after,
        "seed_manifest_unchanged": resolved_config["formal_seed_manifest_sha256_before_outcomes"]
        == sha256_file(output_dir / "formal_seed_manifest.json"),
    }
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED" if all(integrity_checks.values()) else "FAILED",
        "checks": integrity_checks,
        "fairness": fairness,
        "strict_checkpoint_checks": strict_checkpoint_checks,
        "core_hashes_before": core_before,
        "core_hashes_after": core_after,
        "auxiliary_hashes_before": auxiliary_before,
        "auxiliary_hashes_after": auxiliary_after,
        "checkpoint_hashes_before": checkpoint_before,
        "checkpoint_hashes_after": checkpoint_after,
        "model_hashes_before": model_before,
        "model_hashes_after": model_after,
        "policy_hash_before": policy_before,
        "policy_hash_after": policy_after,
        "runtime_seconds": time.perf_counter() - started,
        "baseline_methods_executed": False,
        "formal_seed_replacement_count": 0,
        "failed_episode_deletion_count": 0,
        "training_or_tuning_performed": False,
    }
    write_json(output_dir / "integrity_manifest.json", integrity)
    if integrity["status"] != "PASSED":
        raise RuntimeError(f"final integrity failed: {integrity_checks}")

    conclusion = formal_decision(
        config=config,
        methods=method_rows,
        scenarios=scenario_rows,
        risky=risky_rows,
        nulls=null_rows,
        references=reference_rows,
        tests=test_rows,
        context=context,
        formal_manifest=formal_manifest,
        integrity=integrity,
    )
    write_csv(output_dir / "episode_results.csv", episodes)
    write_csv(output_dir / "agent_results.csv", agents)
    write_csv(output_dir / "selection_results.csv", selections)
    write_csv(output_dir / "method_summary.csv", method_rows)
    write_csv(output_dir / "scenario_summary.csv", scenario_rows)
    write_csv(output_dir / "risky_selection_analysis.csv", risky_rows)
    write_csv(output_dir / "null_analysis.csv", null_rows)
    write_csv(output_dir / "reference_transition_analysis.csv", reference_rows)
    write_csv(output_dir / "paired_outcomes.csv", pair_rows)
    write_json(output_dir / "statistical_tests.json", {"schema_version": SCHEMA_VERSION, "rows": test_rows})
    write_json(output_dir / "continuous_metrics.json", continuous)
    write_json(output_dir / "conclusion.json", conclusion)
    (output_dir / "FINAL_REPORT.md").write_text(
        report_text(conclusion, method_rows, scenario_rows, risky_rows, null_rows, reference_rows, pair_rows, test_rows),
        encoding="utf-8",
    )
    print(json.dumps(conclusion, indent=2), flush=True)
    print(f"output_dir={output_dir}", flush=True)
    return output_dir


def main() -> Path:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (REPO_ROOT / config["output_root"] / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    )
    return run_experiment(config, output_dir)


if __name__ == "__main__":
    main()
