#!/usr/bin/env python3
"""Run the gated 30-episode GAT-V1 versus GAT-V1-IA development comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
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

from planning.gat.stage1_training import load_model_checkpoint, resolve_device  # noqa: E402
from scripts import evaluate_gat_closed_loop as legacy  # noqa: E402
from scripts import evaluate_gat_stage1_v2_closed_loop as paired  # noqa: E402
from scripts.evaluate_pre_gat_closed_loop import _policy_parameter_sha256  # noqa: E402
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


SCHEMA_VERSION = "gat_v1_interaction_aware_supervision_v1"
METHOD_V1 = "gat_v1"
METHOD_IA = "gat_v1_ia"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
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
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields = list(rows[0]) if rows else ["schema_version", "status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
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


def scan_executed_seed_usage(exclude_dir: Path) -> tuple[set[int], list[dict[str, Any]]]:
    """Audit actual CSV records, excluding manifests/configs and this experiment."""

    used: set[int] = set()
    evidence: list[dict[str, Any]] = []
    artifact_root = REPO_ROOT / "artifacts"
    for path in artifact_root.rglob("*.csv"):
        try:
            path.resolve().relative_to(exclude_dir.resolve())
            continue
        except ValueError:
            pass
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                seed_fields = [
                    field for field in (reader.fieldnames or [])
                    if field.lower() in {"seed", "evaluation_seed", "episode_seed", "scenario_seed"}
                ]
                if not seed_fields:
                    continue
                found: set[int] = set()
                row_count = 0
                for row in reader:
                    row_count += 1
                    for field in seed_fields:
                        raw = row.get(field)
                        if raw is None or raw == "":
                            continue
                        try:
                            value = int(float(raw))
                        except ValueError:
                            continue
                        found.add(value)
                if found:
                    used.update(found)
                    evidence.append(
                        {
                            "path": str(path.relative_to(REPO_ROOT)),
                            "seed_fields": seed_fields,
                            "row_count": row_count,
                            "minimum_seed": min(found),
                            "maximum_seed": max(found),
                            "candidate_development_seed_hits": sorted(found & set(range(55, 60))),
                        }
                    )
        except (OSError, UnicodeError, csv.Error):
            continue
    return used, evidence


def mean_finite(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def relabel(
    episode: dict[str, Any], agents: list[dict[str, Any]], method: str
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
            }
        )
    return episode, agents


def summaries(
    episodes: Sequence[Mapping[str, Any]], agents: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in (METHOD_V1, METHOD_IA):
        for scope in ("overall", "open", "sparse_static", "multi_agent"):
            episode_members = [
                row for row in episodes
                if row["method"] == method and (scope == "overall" or row["scenario"] == scope)
            ]
            agent_members = [
                row for row in agents
                if row["method"] == method and (scope == "overall" or row["scenario"] == scope)
            ]
            selected = [row for row in agent_members if bool(row.get("reference_selected"))]
            reached = [row for row in selected if bool(row.get("reference_reached"))]
            reached_terminal = [row for row in reached if bool(row.get("terminal_completed_after_reference"))]
            completed_agents = sum(bool(row.get("agent_terminal_completed")) for row in agent_members)
            count = len(episode_members)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "method": method,
                    "scope": scope,
                    "episode_count": count,
                    "agent_count": len(agent_members),
                    "team_success_count": sum(bool(row["team_success"]) for row in episode_members),
                    "team_success_rate": sum(bool(row["team_success"]) for row in episode_members) / count if count else None,
                    "agent_completion_count": completed_agents,
                    "agent_completion_rate": completed_agents / len(agent_members) if agent_members else None,
                    "any_collision_count": sum(bool(row["any_collision"]) for row in episode_members),
                    "any_collision_rate": sum(bool(row["any_collision"]) for row in episode_members) / count if count else None,
                    "obstacle_collision_count": sum(bool(row["obstacle_collision"]) for row in episode_members),
                    "obstacle_collision_rate": sum(bool(row["obstacle_collision"]) for row in episode_members) / count if count else None,
                    "inter_agent_collision_count": sum(bool(row["inter_agent_collision"]) for row in episode_members),
                    "inter_agent_collision_rate": sum(bool(row["inter_agent_collision"]) for row in episode_members) / count if count else None,
                    "timeout_count": sum(bool(row["timeout"]) for row in episode_members),
                    "timeout_rate": sum(bool(row["timeout"]) for row in episode_members) / count if count else None,
                    "null_selection_count": sum(bool(row.get("selected_null")) for row in agent_members),
                    "null_selection_rate": sum(bool(row.get("selected_null")) for row in agent_members) / len(agent_members) if agent_members else None,
                    "reference_selection_count": len(selected),
                    "reference_selection_rate": len(selected) / len(agent_members) if agent_members else None,
                    "reference_reach_count": len(reached),
                    "reference_reach_rate": len(reached) / len(selected) if selected else None,
                    "reached_to_terminal_count": len(reached_terminal),
                    "reached_to_terminal_rate": len(reached_terminal) / len(reached) if reached else None,
                    "mean_minimum_inter_agent_distance_m": mean_finite(row["minimum_inter_agent_distance_m"] for row in episode_members),
                    "mean_success_completion_time_s": mean_finite(row["completion_time_s"] for row in episode_members if row["team_success"]),
                    "mean_success_team_path_length_m": mean_finite(row["team_path_length_m"] for row in episode_members if row["team_success"]),
                    "mean_team_path_length_m": mean_finite(row["team_path_length_m"] for row in episode_members),
                }
            )
    return rows


def paired_rows(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {(row["scenario"], int(row["seed"]), row["method"]): row for row in episodes}
    rows: list[dict[str, Any]] = []
    for scenario, seed in sorted({(row["scenario"], int(row["seed"])) for row in episodes}):
        v1 = index[(scenario, seed, METHOD_V1)]
        ia = index[(scenario, seed, METHOD_IA)]
        if v1["team_success"] and ia["team_success"]:
            outcome = "BOTH_SUCCESS"
        elif ia["team_success"]:
            outcome = "IA_ONLY_SUCCESS"
        elif v1["team_success"]:
            outcome = "V1_ONLY_SUCCESS"
        else:
            outcome = "BOTH_FAIL"
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
                "candidate_bundle_hash_match": v1["candidate_bundle_hash"] == ia["candidate_bundle_hash"],
                "preview_bundle_hash_match": v1["preview_bundle_hash"] == ia["preview_bundle_hash"],
                "graph_input_hash_match": v1["graph_input_hash"] == ia["graph_input_hash"],
                "completion_time_difference_ia_minus_v1_s": (
                    float(ia["completion_time_s"]) - float(v1["completion_time_s"])
                    if outcome == "BOTH_SUCCESS" else None
                ),
                "team_path_length_difference_ia_minus_v1_m": float(ia["team_path_length_m"]) - float(v1["team_path_length_m"]),
                "minimum_inter_agent_distance_difference_ia_minus_v1_m": float(ia["minimum_inter_agent_distance_m"]) - float(v1["minimum_inter_agent_distance_m"]),
            }
        )
    return rows


def extend_report(path: Path, conclusion: Mapping[str, Any], scenario_rows: Sequence[Mapping[str, Any]]) -> None:
    lookup = {(row["method"], row["scope"]): row for row in scenario_rows}
    existing = path.read_text(encoding="utf-8")
    marker = "\n## Development closed-loop result\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    existing = existing.replace(
        "`IA_CLOSED_LOOP_GAIN = NOT_RUN`.",
        f"`IA_CLOSED_LOOP_GAIN = {conclusion['IA_CLOSED_LOOP_GAIN']}`.",
    ).replace(
        "`EXISTING_GAT_SUPERVISION_ONLY_HEADROOM_EXHAUSTED = NOT_ESTABLISHED`",
        f"`EXISTING_GAT_SUPERVISION_ONLY_HEADROOM_EXHAUSTED = {conclusion['EXISTING_GAT_SUPERVISION_ONLY_HEADROOM_EXHAUSTED']}`",
    ).replace(
        "`RECOMMENDED_NEXT_STEP = KEEP_GAT_V1`",
        f"`RECOMMENDED_NEXT_STEP = {conclusion['RECOMMENDED_NEXT_STEP']}`",
    )
    lines = [
        marker.strip("\n"),
        "",
        "The offline gate triggered the pre-frozen five-seed development comparison (seeds 55-59); no seed was replaced.",
        "",
        "| scope | V1 success | IA success | V1 collision | IA collision | V1 timeout | IA timeout |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in ("overall", "open", "sparse_static", "multi_agent"):
        v1 = lookup[(METHOD_V1, scope)]
        ia = lookup[(METHOD_IA, scope)]
        lines.append(
            f"| {scope} | {v1['team_success_rate']:.1%} | {ia['team_success_rate']:.1%} | "
            f"{v1['any_collision_rate']:.1%} | {ia['any_collision_rate']:.1%} | "
            f"{v1['timeout_rate']:.1%} | {ia['timeout_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"`IA_CLOSED_LOOP_GAIN = {conclusion['IA_CLOSED_LOOP_GAIN']}`; overall success gain "
            f"{conclusion['IA_DEVELOPMENT_SUCCESS_GAIN_PP']:.1f} pp; multi-agent success "
            f"V1 {conclusion['V1_MULTI_AGENT_SUCCESS']:.1%} versus IA {conclusion['IA_MULTI_AGENT_SUCCESS']:.1%}.",
            "",
            f"- `EXISTING_GAT_SUPERVISION_ONLY_HEADROOM_EXHAUSTED = {conclusion['EXISTING_GAT_SUPERVISION_ONLY_HEADROOM_EXHAUSTED']}`",
            f"- `RECOMMENDED_NEXT_STEP = {conclusion['RECOMMENDED_NEXT_STEP']}`",
            "",
            "This was development evidence only. No formal seeds, 24-layout data, second target, or theory extension were used.",
            "",
        ]
    )
    path.write_text(existing.rstrip() + "\n\n" + "\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    config = read_json(output_dir / "config.json")
    conclusion = read_json(output_dir / "conclusion.json")
    if conclusion["IA_TARGET_VALID"] != "YES" or conclusion["MODEL_INTERACTION_GAIN"] not in {"YES", "WEAK"}:
        raise RuntimeError("development gate was not triggered")
    if conclusion["status"] != "OFFLINE_COMPLETE_DEVELOPMENT_PENDING":
        raise RuntimeError(f"unexpected resumable state: {conclusion['status']}")
    started = time.perf_counter()
    seeds = [int(seed) for seed in config["development"]["seeds"]]
    scenarios = list(config["development"]["scenarios"])
    if seeds != list(range(55, 60)) or scenarios != ["open", "sparse_static", "multi_agent"]:
        raise RuntimeError("development seeds or scenarios differ from the pre-frozen config")

    historical_used, evidence = scan_executed_seed_usage(output_dir)
    overlap = sorted(set(seeds) & historical_used)
    forbidden_overlap = sorted(
        set(seeds)
        & (set(config["development"]["forbidden_training_seeds"])
           | set(config["development"]["forbidden_formal_seeds"])
           | set(config["development"]["known_err_development_seeds"]))
    )
    seed_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED" if not overlap and not forbidden_overlap else "FAILED",
        "audit_scope": "all pre-existing artifact CSV records with exact seed-like columns; manifests and configs excluded",
        "historical_executed_seed_union": sorted(historical_used),
        "historical_evidence_file_count": len(evidence),
        "historical_evidence": evidence,
        "frozen_seeds": seeds,
        "development_seed_overlap": len(overlap),
        "overlap_values": overlap,
        "forbidden_seed_overlap": len(forbidden_overlap),
        "forbidden_overlap_values": forbidden_overlap,
        "seed_interval_frozen_before_outcomes": True,
        "result_dependent_seed_replacement_allowed": False,
    }
    write_json(output_dir / "development_seed_manifest.json", seed_manifest)
    if seed_manifest["status"] != "PASSED":
        raise RuntimeError(f"development seed audit failed: {overlap}, {forbidden_overlap}")

    closed_config = read_json(REPO_ROOT / config["sources"]["closed_loop_config"])
    closed_config["formal_scenarios"] = scenarios
    closed_config["formal_seeds"] = seeds
    closed_config["smoke_seeds"] = []
    closed_config["v2_checkpoint"] = str((output_dir / "checkpoints/best_validation.pt").relative_to(REPO_ROOT))
    closed_config["v2_checkpoint_sha256_expected"] = sha256_file(output_dir / "checkpoints/best_validation.pt")
    closed_config["interpretation"] = {
        "name": "same_V1_graph_and_execution_checkpoint_only_V1_vs_IA",
        "closed_loop_control_variable": "checkpoint_parameters_only",
        "same_online_graph_input": True,
        "same_candidate_bundle": True,
        "same_environment_and_executor": True,
    }
    core_before = {path: sha256_file(REPO_ROOT / path) for path in paired.CORE_METHOD_PATHS}
    expected_core = dict(closed_config["expected_core_hashes"])
    if core_before != expected_core:
        raise RuntimeError("frozen core source hash mismatch before development evaluation")

    execution_settings = legacy._build_execution_settings(closed_config)
    multi_config = build_single_distribution_multi_config(
        num_agents=int(closed_config["num_agents"]), max_steps=int(closed_config["max_steps"])
    )
    policy, loaded_sac = _load_policy(execution_settings, multi_config)
    expected_sac = (REPO_ROOT / closed_config["sac_checkpoint"]).resolve()
    if loaded_sac.resolve() != expected_sac or sha256_file(expected_sac) != closed_config["sac_checkpoint_sha256_expected"]:
        raise RuntimeError("frozen SAC checkpoint mismatch")
    policy_hash_before = _policy_parameter_sha256(policy)
    stage1_config = read_json(REPO_ROOT / closed_config["stage1_config"])
    device = resolve_device(stage1_config["training"]["device"])
    v1_checkpoint = (REPO_ROOT / closed_config["v1_checkpoint"]).resolve()
    ia_checkpoint = output_dir / "checkpoints/best_validation.pt"
    v1_model = load_model_checkpoint(v1_checkpoint, stage1_config, device)
    ia_model = load_model_checkpoint(ia_checkpoint, stage1_config, device)
    for model in (v1_model, ia_model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model_hash_before = {"v1": paired._model_hash(v1_model), "ia": paired._model_hash(ia_model)}

    episodes: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    shared_cache: dict[tuple[str, int], dict[str, Any]] = {}
    jobs = [(scenario, seed, method) for scenario in scenarios for seed in seeds for method in (METHOD_V1, METHOD_IA)]
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
                raise RuntimeError("V1 and IA graph hashes differ")
            shared_cache[key] = shared
        internal_method = paired.METHOD_V1 if method == METHOD_V1 else paired.METHOD_V2
        episode, rows = paired.run_method_episode(
            config=closed_config,
            execution_settings=execution_settings,
            multi_config=multi_config,
            policy=policy,
            shared=shared_cache[key],
            method=internal_method,
        )
        episode, rows = relabel(episode, rows, method)
        episodes.append(episode)
        agents.extend(rows)
        print(
            f"[development {index}/{len(jobs)}] {scenario} seed={seed} {method}: "
            f"{episode['termination_reason']}",
            flush=True,
        )
    paired._annotate_agent_team_outcomes(agents, episodes)
    scenario_rows = summaries(episodes, agents)
    pairs = paired_rows(episodes)

    lookup = {(row["method"], row["scope"]): row for row in scenario_rows}
    v1_overall = lookup[(METHOD_V1, "overall")]
    ia_overall = lookup[(METHOD_IA, "overall")]
    v1_multi = lookup[(METHOD_V1, "multi_agent")]
    ia_multi = lookup[(METHOD_IA, "multi_agent")]
    overall_gain = float(ia_overall["team_success_rate"] - v1_overall["team_success_rate"])
    multi_gain = float(ia_multi["team_success_rate"] - v1_multi["team_success_rate"])
    collision_worsening = float(ia_overall["any_collision_rate"] - v1_overall["any_collision_rate"])
    yes = (
        (overall_gain >= float(config["development"]["overall_team_success_gain_yes"])
         or multi_gain >= float(config["development"]["multi_agent_team_success_gain_yes"]))
        and collision_worsening <= float(config["development"]["maximum_overall_collision_worsening"])
    )
    ia_only = sum(row["ia_only_success"] for row in pairs)
    v1_only = sum(row["v1_only_success"] for row in pairs)
    weak = ia_only > v1_only and collision_worsening <= float(
        config["development"]["maximum_overall_collision_worsening"]
    )
    gain = "YES" if yes else "WEAK" if weak else "NO"

    core_after = {path: sha256_file(REPO_ROOT / path) for path in paired.CORE_METHOD_PATHS}
    policy_hash_after = _policy_parameter_sha256(policy)
    model_hash_after = {"v1": paired._model_hash(v1_model), "ia": paired._model_hash(ia_model)}
    fairness = {
        "episode_count": len(episodes),
        "agent_row_count": len(agents),
        "expected_episode_count": int(config["development"]["expected_team_episode_count"]),
        "candidate_hash_mismatch_count": sum(not row["candidate_bundle_hash_match"] for row in pairs),
        "preview_hash_mismatch_count": sum(not row["preview_bundle_hash_match"] for row in pairs),
        "graph_hash_mismatch_count": sum(not row["graph_input_hash_match"] for row in pairs),
        "selection_plan_mutation_count": sum(not bool(row["selection_plan_unchanged"]) for row in episodes),
        "replanning_count": sum(int(row["replanning_count"]) for row in episodes),
        "core_hash_unchanged": core_before == core_after,
        "policy_hash_unchanged": policy_hash_before == policy_hash_after,
        "model_hash_unchanged": model_hash_before == model_hash_after,
    }
    if not (
        fairness["episode_count"] == fairness["expected_episode_count"]
        and fairness["candidate_hash_mismatch_count"] == 0
        and fairness["preview_hash_mismatch_count"] == 0
        and fairness["graph_hash_mismatch_count"] == 0
        and fairness["selection_plan_mutation_count"] == 0
        and fairness["replanning_count"] == 0
        and fairness["core_hash_unchanged"]
        and fairness["policy_hash_unchanged"]
        and fairness["model_hash_unchanged"]
    ):
        raise RuntimeError(f"development integrity failure: {fairness}")

    write_csv(output_dir / "development_episode_results.csv", episodes)
    write_csv(output_dir / "development_agent_results.csv", agents)
    write_csv(output_dir / "development_paired.csv", pairs)
    write_csv(output_dir / "development_scenario_summary.csv", scenario_rows)
    conclusion.update(
        {
            "DEVELOPMENT_SEED_OVERLAP": len(overlap),
            "V1_DEVELOPMENT_TEAM_SUCCESS": v1_overall["team_success_rate"],
            "IA_DEVELOPMENT_TEAM_SUCCESS": ia_overall["team_success_rate"],
            "IA_DEVELOPMENT_SUCCESS_GAIN_PP": 100.0 * overall_gain,
            "V1_MULTI_AGENT_SUCCESS": v1_multi["team_success_rate"],
            "IA_MULTI_AGENT_SUCCESS": ia_multi["team_success_rate"],
            "IA_CLOSED_LOOP_GAIN": gain,
            "EXISTING_GAT_SUPERVISION_ONLY_HEADROOM_EXHAUSTED": "NO" if gain == "YES" else "YES",
            "RECOMMENDED_NEXT_STEP": "FORMAL_GAT_V1_IA_EVALUATION" if gain == "YES" else "ONLY_THEN_CONSIDER_MINIMAL_JOINT_DECISION_EXTENSION",
            "status": "COMPLETE",
            "development_pair_outcome_counts": {
                name: sum(row["pair_outcome"] == name for row in pairs)
                for name in ("BOTH_SUCCESS", "IA_ONLY_SUCCESS", "V1_ONLY_SUCCESS", "BOTH_FAIL")
            },
            "development_overall_collision_change_pp": 100.0 * collision_worsening,
            "development_multi_agent_success_gain_pp": 100.0 * multi_gain,
        }
    )
    write_json(output_dir / "conclusion.json", conclusion)
    integrity = read_json(output_dir / "integrity_manifest.json")
    integrity.update(
        {
            "development_closed_loop_performed": True,
            "development_runtime_seconds": time.perf_counter() - started,
            "development_fairness": fairness,
            "development_core_hashes_before": core_before,
            "development_core_hashes_after": core_after,
            "development_v1_checkpoint_sha256": sha256_file(v1_checkpoint),
            "development_ia_checkpoint_sha256": sha256_file(ia_checkpoint),
            "formal_seeds_used_in_development": False,
            "layout_24_artifact_read_in_development": False,
        }
    )
    write_json(output_dir / "integrity_manifest.json", integrity)
    extend_report(output_dir / "FINAL_REPORT.md", conclusion, scenario_rows)
    print(json.dumps({key: conclusion[key] for key in (
        "V1_DEVELOPMENT_TEAM_SUCCESS", "IA_DEVELOPMENT_TEAM_SUCCESS",
        "IA_DEVELOPMENT_SUCCESS_GAIN_PP", "V1_MULTI_AGENT_SUCCESS",
        "IA_MULTI_AGENT_SUCCESS", "IA_CLOSED_LOOP_GAIN", "RECOMMENDED_NEXT_STEP"
    )}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
