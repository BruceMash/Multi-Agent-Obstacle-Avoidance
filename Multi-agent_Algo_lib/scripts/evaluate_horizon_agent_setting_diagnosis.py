"""Run the strict S50/S220/M50/M220 reference-transition diagnosis.

No training path is imported or called.  Horizon pairs reuse identical
scenario seeds, candidate selection, checkpoint semantics, and execution code;
``max_steps`` is their sole active configuration difference.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiment_config import EXPERIMENT_CONFIG  # noqa: E402
from planning.goal_semantics_diagnosis import VARIANT_D  # noqa: E402
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    scoped_historical_multi_agent_transition,
)
from planning.horizon_agent_setting_diagnosis import (  # noqa: E402
    HORIZON_LONG,
    HORIZON_SHORT,
    SCHEMA_VERSION,
    analyze_diagnosis,
    render_final_report,
    stable_hash,
    validate_horizon_only_pair,
)
from planning.reference_transition_finetuning import (  # noqa: E402
    TASK_REFERENCE,
    TASK_TERMINAL,
    build_reference_transition_env,
    load_checkpoint_weights_only,
    sha256_file,
    smoke_training_config,
    state_dict_sha256,
)
from runner_sac import build_model  # noqa: E402
from scripts.evaluate_actor_dmp_goal_semantics import (  # noqa: E402
    run_variant_episode,
)
from scripts.evaluate_historical_gate_abcd_rerun import (  # noqa: E402
    _augment_episode_with_transition_diagnostics,
    _transition_trace_observer,
)
from scripts.evaluate_reference_transition_finetuning import (  # noqa: E402
    jsonable,
    run_evaluation_episode,
    write_csv,
    write_json,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402
from scripts.evaluate_pre_gat_closed_loop import _policy_parameter_sha256  # noqa: E402


DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "evaluation" / "horizon_agent_setting_diagnosis.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _single_contract(
    config: Any,
    settings: Mapping[str, Any],
    *,
    max_steps: int,
) -> dict[str, Any]:
    values = asdict(config)
    values.pop("max_steps", None)
    return {
        "agent_setting": "single_agent",
        "max_steps": int(max_steps),
        "experiment_config_without_max_steps": values,
        "checkpoint_sha256": settings["checkpoint_sha256_expected"],
        "deterministic_policy": True,
        "actor_observation_dimension": 122,
        "action_dimension": 6,
        "historical_vector_gate": settings["historical_transition"],
        "temporary_reference": settings["single_agent"],
        "training_enabled": False,
    }


def _multi_contract(
    config: Any,
    settings: Mapping[str, Any],
    *,
    max_steps: int,
) -> dict[str, Any]:
    values = asdict(config)
    values.pop("max_steps", None)
    return {
        "agent_setting": "multi_agent",
        "max_steps": int(max_steps),
        "environment_config_without_max_steps": values,
        "checkpoint_sha256": settings["checkpoint_sha256_expected"],
        "deterministic_policy": True,
        "actor_observation_dimension": 122,
        "action_dimension": 6,
        "historical_vector_gate": settings["historical_transition"],
        "temporary_reference": settings["multi_agent"],
        "training_enabled": False,
    }


def _standardize_single(
    raw: Mapping[str, Any],
    *,
    protocol: str,
    max_steps: int,
    contract_hash: str,
) -> dict[str, Any]:
    available = int(bool(raw["reference_available"]))
    reached = int(bool(raw["temporary_reference_reached"]))
    reached_terminal = int(bool(raw["reached_then_terminal_success"]))
    reference_steps = [
        int(value) for value in raw.get("reference_reached_steps", [])
    ]
    terminal_steps = [
        int(value) for value in raw.get("terminal_completion_steps", [])
    ]
    stage2_steps = (
        [int(terminal_steps[0]) - int(reference_steps[0])]
        if reached_terminal and reference_steps and terminal_steps
        else []
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "agent_setting": "single_agent",
        "max_steps": int(max_steps),
        "scenario": str(raw["scene"]),
        "seed": int(raw["seed"]),
        "episode_steps": int(raw["episode_length"]),
        "team_terminal_success": bool(raw["terminal_success"]),
        "team_terminal_completion_step": raw["terminal_completion_step"],
        "collision": bool(raw["collision"]),
        "obstacle_collision": bool(raw["collision"]),
        "inter_agent_collision": False,
        "timeout": bool(raw["timeout"]),
        "termination_reason": str(raw["termination_reason"]),
        "temporary_reference_available_count": available,
        "temporary_reference_reached_count": reached,
        "agent_temporary_reference_reached_rate": (
            reached / available if available else None
        ),
        "reached_then_terminal_completion_count": reached_terminal,
        "agent_reached_then_terminal_completion_rate": (
            reached_terminal / reached if reached else None
        ),
        "team_stage1_success": bool(available and reached == available),
        "reference_reached_steps": reference_steps,
        "terminal_completion_steps": terminal_steps,
        "stage2_completion_steps": stage2_steps,
        "remaining_steps_after_reference": list(
            raw.get("remaining_steps_after_reference", [])
        ),
        "collision_before_reference_reached_count": int(
            raw["collision_before_reference_reached_count"]
        ),
        "collision_after_reference_reached_count": int(
            raw["collision_after_reference_reached_count"]
        ),
        "timeout_after_reference_reached": bool(
            raw["timeout_after_reference_reached"]
        ),
        "terminal_progress_m": float(raw["terminal_progress_m"]),
        "path_length_m": float(raw["path_length_m"]),
        "trajectory_smoothness": float(raw["trajectory_smoothness"]),
        "initial_condition_hash": str(raw["initial_condition_hash"]),
        "temporary_reference_hash": str(raw["temporary_reference_hash"]),
        "state_at_step_50_hash": raw["state_at_step_50_hash"],
        "final_state_hash": str(raw["final_state_hash"]),
        "horizon_independent_contract_hash": contract_hash,
        "reference_generation_count": int(raw["reference_generation_count"]),
        "reference_handoff_count": int(raw["reference_handoff_count"]),
        "phase_reset_on_switch": False,
        "forcing_gate_semantics": str(raw["forcing_gate_semantics"]),
        "checkpoint_unchanged": True,
        "policy_parameters_unchanged": True,
        "training_performed": False,
        "GAT_used": False,
        "FP_SHEP_used": False,
    }


def _standardize_multi(
    raw: Mapping[str, Any],
    *,
    protocol: str,
    max_steps: int,
    contract_hash: str,
) -> dict[str, Any]:
    available = int(raw["temporary_reference_count"])
    reached = int(raw["temporary_reference_reached_count"])
    reached_terminal = int(raw["reached_then_terminal_completion_count"])
    reference_steps = [
        int(value) for value in raw["reference_reached_steps"] if value is not None
    ]
    terminal_steps = [
        int(value) for value in raw["terminal_success_steps"] if value is not None
    ]
    stage2_steps = [
        int(completion_step) - int(reference_step)
        for reference_step, completion_step in zip(
            raw["reference_reached_steps"],
            raw["terminal_success_steps"],
            strict=True,
        )
        if reference_step is not None
        and completion_step is not None
        and int(completion_step) >= int(reference_step)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "agent_setting": "multi_agent",
        "max_steps": int(max_steps),
        "scenario": str(raw["scenario"]),
        "seed": int(raw["seed"]),
        "episode_steps": int(raw["steps"]),
        "team_terminal_success": bool(raw["success"]),
        "team_terminal_completion_step": raw["team_terminal_completion_step"],
        "collision": bool(raw["collision"]),
        "obstacle_collision": bool(raw["obstacle_collision"]),
        "inter_agent_collision": bool(raw["inter_agent_collision"]),
        "timeout": bool(raw["truncated"]),
        "termination_reason": str(raw["termination_reason"]),
        "temporary_reference_available_count": available,
        "temporary_reference_reached_count": reached,
        "agent_temporary_reference_reached_rate": (
            reached / available if available else None
        ),
        "reached_then_terminal_completion_count": reached_terminal,
        "agent_reached_then_terminal_completion_rate": (
            reached_terminal / reached if reached else None
        ),
        "team_stage1_success": bool(raw["team_stage1_success"]),
        "reference_reached_steps": reference_steps,
        "terminal_completion_steps": terminal_steps,
        "stage2_completion_steps": stage2_steps,
        "remaining_steps_after_reference": list(
            raw["remaining_steps_after_reference"]
        ),
        "collision_before_reference_reached_count": int(
            raw["collision_before_reference_reached_count"]
        ),
        "collision_after_reference_reached_count": int(
            raw["collision_after_reference_reached_count"]
        ),
        "timeout_after_reference_reached": bool(
            raw["timeout_after_reference_reached"]
        ),
        "terminal_progress_m": float(raw["terminal_progress_team_mean_m"]),
        "path_length_m": float(raw["path_length_team_mean_m"]),
        "path_length_team_sum_m": float(raw["path_length_team_sum_m"]),
        "trajectory_smoothness": float(raw["trajectory_smoothness"]),
        "minimum_obstacle_clearance_m": float(raw["minimum_obstacle_clearance_m"]),
        "minimum_inter_agent_distance_m": float(
            raw["minimum_inter_agent_distance_m"]
        ),
        "initial_condition_hash": str(raw["initial_condition_hash"]),
        "temporary_reference_hash": str(raw["temporary_reference_hash"]),
        "state_at_step_50_hash": raw["state_at_step_50_hash"],
        "final_state_hash": str(raw["final_state_hash"]),
        "horizon_independent_contract_hash": contract_hash,
        "reference_generation_count": available,
        "reference_handoff_count": reached,
        "phase_reset_on_switch": bool(raw["phase_reset_on_switch"]),
        "forcing_gate_semantics": HISTORICAL_GATE_NAME,
        "checkpoint_unchanged": True,
        "policy_parameters_unchanged": True,
        "training_performed": False,
        "GAT_used": False,
        "FP_SHEP_used": False,
    }


def _assert_protocol_exclusions(settings: Mapping[str, Any]) -> None:
    exclusions = settings["strict_exclusions"]
    if any(bool(value) for value in exclusions.values()):
        raise ValueError("all strict exclusion flags must remain false")
    if tuple(int(value) for value in settings["horizons"]) != (
        HORIZON_SHORT,
        HORIZON_LONG,
    ):
        raise ValueError("diagnosis horizons must be exactly [50, 220]")
    if settings["multi_agent"]["variant"] != VARIANT_D:
        raise ValueError("multi-agent diagnosis must use existing Variant D")


def run_experiment(settings: Mapping[str, Any], output_dir: Path) -> Path:
    _assert_protocol_exclusions(settings)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = (REPO_ROOT / str(settings["checkpoint"])).resolve()
    checkpoint_hash_before = sha256_file(checkpoint)
    if checkpoint_hash_before != str(settings["checkpoint_sha256_expected"]):
        raise RuntimeError("checkpoint hash does not match audited best_eval_model.pt")

    single_base = smoke_training_config(EXPERIMENT_CONFIG)
    single_configs = {
        horizon: replace(single_base, max_steps=int(horizon))
        for horizon in (HORIZON_SHORT, HORIZON_LONG)
    }
    multi_configs = {
        horizon: build_single_distribution_multi_config(
            num_agents=int(settings["multi_agent"]["num_agents"]),
            max_steps=int(horizon),
        )
        for horizon in (HORIZON_SHORT, HORIZON_LONG)
    }
    single_contracts = {
        horizon: _single_contract(
            single_configs[horizon], settings, max_steps=horizon
        )
        for horizon in (HORIZON_SHORT, HORIZON_LONG)
    }
    multi_contracts = {
        horizon: _multi_contract(multi_configs[horizon], settings, max_steps=horizon)
        for horizon in (HORIZON_SHORT, HORIZON_LONG)
    }
    contract_validation = {
        "single": validate_horizon_only_pair(
            single_contracts[HORIZON_SHORT], single_contracts[HORIZON_LONG]
        ),
        "multi": validate_horizon_only_pair(
            multi_contracts[HORIZON_SHORT], multi_contracts[HORIZON_LONG]
        ),
    }
    augmented_config = copy.deepcopy(dict(settings))
    augmented_config.update(
        {
            "resolved_output_dir": str(output_dir),
            "single_horizon_contracts": single_contracts,
            "multi_horizon_contracts": multi_contracts,
            "horizon_only_contract_validation": contract_validation,
        }
    )
    write_json(output_dir / "config.json", augmented_config)

    model_env = build_reference_transition_env(
        single_configs[HORIZON_LONG], forced_task=TASK_TERMINAL
    )
    try:
        single_model = build_model(
            model_env,
            config=single_configs[HORIZON_LONG],
            tensorboard_log=None,
            verbose=0,
        )
        single_base_settings = _load_json(
            (REPO_ROOT / settings["single_agent"]["base_config"]).resolve()
        )
        warm_start = load_checkpoint_weights_only(
            single_model,
            checkpoint,
            learning_rate=float(single_base_settings["learning"]["actor_lr"]),
        )
    finally:
        model_env.close()
    single_actor_before = state_dict_sha256(single_model.actor.state_dict())
    single_critic_before = state_dict_sha256(single_model.critic.state_dict())
    single_replay_before = int(single_model.replay_buffer.size())

    multi_base_settings = _load_json(
        (REPO_ROOT / settings["multi_agent"]["base_config"]).resolve()
    )
    multi_base_settings.update(
        {
            "checkpoint": settings["checkpoint"],
            "checkpoint_sha256_expected": settings["checkpoint_sha256_expected"],
            "deterministic_policy": True,
            "num_agents": int(settings["multi_agent"]["num_agents"]),
            "scenarios": list(settings["multi_agent"]["scenarios"]),
            "seeds": list(settings["multi_agent"]["seeds"]),
            "variant_order": [VARIANT_D],
        }
    )
    multi_policy, loaded_checkpoint = _load_policy(
        multi_base_settings, multi_configs[HORIZON_LONG]
    )
    if loaded_checkpoint.resolve() != checkpoint:
        raise RuntimeError("single and multi evaluators resolved different checkpoints")
    multi_policy_before = _policy_parameter_sha256(multi_policy)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    job_count = 2 * (
        len(settings["single_agent"]["scenarios"])
        * len(settings["single_agent"]["seeds"])
        + len(settings["multi_agent"]["scenarios"])
        * len(settings["multi_agent"]["seeds"])
    )
    job_index = 0
    # Run the short horizon before the long horizon for each setting.  Every
    # episode is nevertheless created from a fresh deterministic reset.
    for horizon, protocol in ((50, "S50"), (220, "S220")):
        contract_hash = contract_validation["single"][
            "horizon_independent_contract_hash"
        ]
        for scenario in settings["single_agent"]["scenarios"]:
            for seed in settings["single_agent"]["seeds"]:
                raw, _ = run_evaluation_episode(
                    model=single_model,
                    config=single_configs[horizon],
                    evaluation_step=0,
                    evaluation_kind="horizon_agent_setting_diagnosis",
                    scene=str(scenario),
                    seed=int(seed),
                    task=TASK_REFERENCE,
                    proposal_top_k=int(
                        settings["single_agent"]["proposal_top_k"]
                    ),
                    reference_tolerance=float(
                        settings["single_agent"][
                            "temporary_reference_tolerance_m"
                        ]
                    ),
                )
                rows.append(
                    _standardize_single(
                        raw,
                        protocol=protocol,
                        max_steps=horizon,
                        contract_hash=contract_hash,
                    )
                )
                job_index += 1
                print(
                    f"[{job_index}/{job_count}] {protocol} {scenario} seed={seed}: "
                    f"{raw['termination_reason']}",
                    flush=True,
                )

    for horizon, protocol in ((50, "M50"), (220, "M220")):
        contract_hash = contract_validation["multi"][
            "horizon_independent_contract_hash"
        ]
        multi_settings = copy.deepcopy(multi_base_settings)
        multi_settings["max_steps"] = int(horizon)
        for scenario in settings["multi_agent"]["scenarios"]:
            for seed in settings["multi_agent"]["seeds"]:
                trace: list[dict[str, Any]] = []
                observer = _transition_trace_observer(
                    trace,
                    near_reference_radius_m=float(
                        multi_settings["waypoint_attenuation_diagnostic"][
                            "near_reference_radius_m"
                        ]
                    ),
                )
                with scoped_historical_multi_agent_transition(observer):
                    raw, _, _ = run_variant_episode(
                        policy=multi_policy,
                        multi_config=multi_configs[horizon],
                        settings=multi_settings,
                        scenario=str(scenario),
                        seed=int(seed),
                        variant=VARIANT_D,
                    )
                _augment_episode_with_transition_diagnostics(
                    raw, trace, multi_settings
                )
                rows.append(
                    _standardize_multi(
                        raw,
                        protocol=protocol,
                        max_steps=horizon,
                        contract_hash=contract_hash,
                    )
                )
                job_index += 1
                print(
                    f"[{job_index}/{job_count}] {protocol} {scenario} seed={seed}: "
                    f"{raw['termination_reason']}",
                    flush=True,
                )

    checkpoint_hash_after = sha256_file(checkpoint)
    integrity = {
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "checkpoint_unchanged": checkpoint_hash_after == checkpoint_hash_before,
        "single_actor_unchanged": (
            single_actor_before == state_dict_sha256(single_model.actor.state_dict())
        ),
        "single_critic_unchanged": (
            single_critic_before == state_dict_sha256(single_model.critic.state_dict())
        ),
        "single_replay_size_before": single_replay_before,
        "single_replay_size_after": int(single_model.replay_buffer.size()),
        "single_replay_unchanged": (
            single_replay_before == int(single_model.replay_buffer.size())
        ),
        "multi_policy_unchanged": (
            multi_policy_before == _policy_parameter_sha256(multi_policy)
        ),
        "training_performed": False,
        "gradient_update_count": 0,
        "warm_start_audit": jsonable(warm_start),
    }
    if not all(
        integrity[key]
        for key in (
            "checkpoint_unchanged",
            "single_actor_unchanged",
            "single_critic_unchanged",
            "single_replay_unchanged",
            "multi_policy_unchanged",
        )
    ):
        raise RuntimeError("evaluation mutated frozen policy/checkpoint state")

    analysis = analyze_diagnosis(rows)
    if analysis["pairing_integrity"]["status"] != "PASSED":
        raise RuntimeError(
            f"paired horizon validation failed: "
            f"{analysis['pairing_integrity']['errors']}"
        )
    integrity["paired_horizon_validation"] = analysis["pairing_integrity"]
    integrity["runtime_seconds"] = float(time.perf_counter() - started)
    integrity["episode_count"] = len(rows)

    write_csv(output_dir / "per_episode.csv", rows)
    write_csv(output_dir / "stage_summary.csv", analysis["stage_rows"])
    write_csv(output_dir / "horizon_summary.csv", analysis["horizon_rows"])
    write_csv(output_dir / "paired_comparison.csv", analysis["paired_rows"])
    write_csv(output_dir / "censoring_analysis.csv", analysis["censoring_rows"])
    write_json(output_dir / "integrity.json", integrity)
    write_json(output_dir / "conclusion.json", analysis["conclusion"])
    (output_dir / "FINAL_REPORT.md").write_text(
        render_final_report(config=augmented_config, analysis=analysis),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "episode_count": len(rows),
                "runtime_seconds": integrity["runtime_seconds"],
                **analysis["conclusion"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    settings = _load_json(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT / str(settings["output_dir"]) / timestamp
    )
    return run_experiment(settings, output_dir)


if __name__ == "__main__":
    main()
