"""Re-run the established A/B/C/D diagnosis with the historical vector gate.

The previous evaluator is reused as the authoritative definition of variants,
candidate generation, observation conditioning, one-shot return behavior,
environment construction, and metrics.  This script changes exactly one
execution variable inside a bounded context: the DMP forcing gate/transition.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
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

from planning.goal_semantics_diagnosis import (  # noqa: E402
    VARIANT_DISPLAY_NAMES,
    VARIANT_ORDER,
    terminal_checkpoint_observations,
)
from planning.historical_forcing_gate import (  # noqa: E402
    HISTORICAL_GATE_NAME,
    compare_current_and_historical_gate,
    scoped_historical_multi_agent_transition,
)
from scripts.evaluate_actor_dmp_goal_semantics import (  # noqa: E402
    run_variant_episode,
    validate_pairing,
    write_csv,
    write_json,
)
from scripts.evaluate_frozen_policy_waypoint_guidance import (  # noqa: E402
    predict_actions_without_postprocessing,
)
from scripts.evaluate_pre_gat_closed_loop import (  # noqa: E402
    _jsonable,
    _policy_parameter_sha256,
    _scenario_hash,
    _scene_snapshot,
    build_closed_loop_environment,
)
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_temporary_reference_interface import _load_policy  # noqa: E402


DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "evaluation" / "historical_gate_abcd_rerun.json"
)
SCHEMA_VERSION = "historical_gate_abcd_rerun_v1"


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _transition_trace_observer(
    sink: list[dict[str, Any]],
    *,
    near_reference_radius_m: float,
) -> Any:
    def observe(arguments: dict[str, Any], transition: Any) -> None:
        position = np.asarray(arguments["position"], dtype=float)
        active_goal = np.asarray(arguments["active_goal"], dtype=float)
        terminal_goal = np.asarray(arguments["terminal_goal"], dtype=float)
        info = transition.controller_info
        forcing = np.asarray(info["forcing"], dtype=float)
        contribution = np.asarray(info["forcing_contribution"], dtype=float)
        gate = np.asarray(info["forcing_gate"], dtype=float)
        distance_before = float(np.linalg.norm(active_goal - position))
        distance_after = float(
            np.linalg.norm(active_goal - np.asarray(transition.position, dtype=float))
        )
        temporary_active = bool(
            np.linalg.norm(active_goal - terminal_goal) > 1.0e-8
        )
        forcing_norm = float(np.linalg.norm(forcing))
        sink.append(
            {
                "temporary_active": temporary_active,
                "distance_to_active_before_m": distance_before,
                "distance_to_active_after_m": distance_after,
                "near_active_reference": bool(
                    temporary_active
                    and min(distance_before, distance_after)
                    <= float(near_reference_radius_m)
                ),
                "historical_gate_norm": float(np.linalg.norm(gate)),
                "historical_gate_min_axis": float(np.min(gate)),
                "historical_gate_max_axis": float(np.max(gate)),
                "forcing_norm": forcing_norm,
                "forcing_contribution_norm": float(np.linalg.norm(contribution)),
                "forcing_attenuation_ratio": float(
                    np.linalg.norm(contribution) / max(forcing_norm, 1.0e-12)
                ),
                "speed_mps": float(np.linalg.norm(transition.velocity)),
                "commanded_acceleration_mps2": float(
                    np.linalg.norm(transition.commanded_acceleration)
                ),
            }
        )

    return observe


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(np.mean([float(row[key]) for row in rows]))


def _augment_episode_with_transition_diagnostics(
    episode: dict[str, Any],
    trace: list[dict[str, Any]],
    settings: Mapping[str, Any],
) -> None:
    diagnostic = settings["waypoint_attenuation_diagnostic"]
    temporary_rows = [row for row in trace if bool(row["temporary_active"])]
    near_rows = [row for row in trace if bool(row["near_active_reference"])]
    low_speed_threshold = float(diagnostic["low_speed_threshold_mps"])
    attenuation_threshold = float(
        diagnostic["forcing_attenuation_ratio_threshold"]
    )
    reached_count = int(episode.get("temporary_reference_reached_count") or 0)
    available_count = int(episode.get("temporary_reference_count") or 0)
    reached_terminal_count = reached_count if bool(episode["success"]) else 0
    episode.update(
        {
            "forcing_gate_semantics": HISTORICAL_GATE_NAME,
            "forcing_gate_distance_source": "goal_eff_per_axis",
            "historical_transition_scope_applied": True,
            "transition_count": len(trace),
            "temporary_active_transition_count": len(temporary_rows),
            "near_active_reference_transition_count": len(near_rows),
            "near_active_reference_mean_gate_norm": _mean(
                near_rows, "historical_gate_norm"
            ),
            "near_active_reference_mean_forcing_attenuation_ratio": _mean(
                near_rows, "forcing_attenuation_ratio"
            ),
            "near_active_reference_mean_forcing_contribution_norm": _mean(
                near_rows, "forcing_contribution_norm"
            ),
            "near_active_reference_mean_speed_mps": _mean(near_rows, "speed_mps"),
            "near_active_reference_low_speed_count": sum(
                float(row["speed_mps"]) <= low_speed_threshold for row in near_rows
            ),
            "near_active_reference_low_speed_total": len(near_rows),
            "near_active_reference_low_speed_rate": (
                float(
                    np.mean(
                        [
                            float(row["speed_mps"]) <= low_speed_threshold
                            for row in near_rows
                        ]
                    )
                )
                if near_rows
                else None
            ),
            "near_active_reference_attenuated_count": sum(
                float(row["forcing_attenuation_ratio"]) <= attenuation_threshold
                for row in near_rows
            ),
            "near_active_reference_attenuated_total": len(near_rows),
            "near_active_reference_attenuated_rate": (
                float(
                    np.mean(
                        [
                            float(row["forcing_attenuation_ratio"])
                            <= attenuation_threshold
                            for row in near_rows
                        ]
                    )
                )
                if near_rows
                else None
            ),
            "temporary_reference_reached_count_metric": reached_count,
            "temporary_reference_reached_total_metric": available_count,
            "temporary_reference_reached_rate_metric": (
                reached_count / available_count if available_count else None
            ),
            "reached_then_terminal_success_count": reached_terminal_count,
            "reached_then_terminal_success_total": reached_count,
            "reached_then_terminal_success_rate": (
                reached_terminal_count / reached_count if reached_count else None
            ),
        }
    )


def _gate_diagnostic_rows(
    *,
    policy: Any,
    multi_config: Any,
    settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in settings["scenarios"]:
        for seed in settings["seeds"]:
            env, metadata = build_closed_loop_environment(
                config=multi_config,
                scenario=str(scenario),
                seed=int(seed),
                peer_radius=float(settings["peer_radius"]),
            )
            try:
                fingerprint_before = _scenario_hash(_scene_snapshot(env))
                observations = terminal_checkpoint_observations(env)
                actions = predict_actions_without_postprocessing(
                    policy, observations, tuple(env.action_shape)
                )
                for agent_index in range(int(env.num_agents)):
                    result = compare_current_and_historical_gate(
                        config=env.dmps[agent_index].config,
                        position=env.dynamics[agent_index].p,
                        velocity=env.dynamics[agent_index].v,
                        action=actions[agent_index],
                        active_goal=env.goals[agent_index],
                        terminal_goal=env.goals[agent_index],
                        phase=float(env.dmps[agent_index].phase),
                    )
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "scenario": str(scenario),
                            "seed": int(seed),
                            "agent_id": int(agent_index),
                            "state_hash": fingerprint_before,
                            "position": env.dynamics[agent_index].p.tolist(),
                            "velocity": env.dynamics[agent_index].v.tolist(),
                            "terminal_goal": env.goals[agent_index].tolist(),
                            "same_action": actions[agent_index].tolist(),
                            **result,
                            **metadata,
                        }
                    )
                fingerprint_after = _scenario_hash(_scene_snapshot(env))
                if fingerprint_before != fingerprint_after:
                    raise RuntimeError("gate diagnostic advanced or mutated environment state")
            finally:
                env.close()
    return rows


def run_experiment(settings: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", copy.deepcopy(dict(settings)))
    multi_config = build_single_distribution_multi_config(
        num_agents=int(settings["num_agents"]),
        max_steps=int(settings["max_steps"]),
    )
    policy, checkpoint = _load_policy(settings, multi_config)
    expected_checkpoint_hash = str(settings["checkpoint_sha256_expected"])
    checkpoint_hash_before = _sha256(checkpoint)
    if checkpoint_hash_before != expected_checkpoint_hash:
        raise RuntimeError("checkpoint hash does not match the audited checkpoint")
    policy_hash_before = _policy_parameter_sha256(policy)

    episodes: list[dict[str, Any]] = []
    jobs = [
        (str(scenario), int(seed), str(variant))
        for scenario in settings["scenarios"]
        for seed in settings["seeds"]
        for variant in VARIANT_ORDER
    ]
    started = time.perf_counter()
    for index, (scenario, seed, variant) in enumerate(jobs, start=1):
        trace: list[dict[str, Any]] = []
        observer = _transition_trace_observer(
            trace,
            near_reference_radius_m=float(
                settings["waypoint_attenuation_diagnostic"][
                    "near_reference_radius_m"
                ]
            ),
        )
        with scoped_historical_multi_agent_transition(observer):
            episode, _, _ = run_variant_episode(
                policy=policy,
                multi_config=multi_config,
                settings=settings,
                scenario=scenario,
                seed=seed,
                variant=variant,
            )
        _augment_episode_with_transition_diagnostics(episode, trace, settings)
        episodes.append(episode)
        print(
            f"[{index}/{len(jobs)}] {scenario} seed={seed} "
            f"{VARIANT_DISPLAY_NAMES[variant]}: {episode['termination_reason']}",
            flush=True,
        )

    pairing = validate_pairing(episodes)
    if pairing["status"] != "PASSED":
        raise RuntimeError(f"paired validation failed: {pairing['errors']}")
    gate_rows = _gate_diagnostic_rows(
        policy=policy,
        multi_config=multi_config,
        settings=settings,
    )
    checkpoint_hash_after = _sha256(checkpoint)
    policy_hash_after = _policy_parameter_sha256(policy)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("checkpoint changed during evaluation")
    if policy_hash_after != policy_hash_before:
        raise RuntimeError("frozen Actor parameters changed during evaluation")

    # Prove that leaving the context restored the exact current callable.
    import Environment.multi_agent_dmp_env as environment_module
    from Environment.frozen_sac_dmp_execution import propagate_sac_dmp_action

    scalar_transition_restored = (
        environment_module.propagate_sac_dmp_action is propagate_sac_dmp_action
    )
    if not scalar_transition_restored:
        raise RuntimeError("current scalar transition was not restored")

    for row in episodes:
        row.update(
            {
                "checkpoint_sha256": checkpoint_hash_after,
                "checkpoint_unchanged": True,
                "policy_parameters_unchanged": True,
                "current_scalar_transition_restored": True,
                "training_performed": False,
                "GAT_used": False,
                "FP_SHEP_selector_used": False,
            }
        )
    write_csv(output_dir / "per_episode.csv", episodes)
    write_csv(output_dir / "gate_diagnostic.csv", gate_rows)

    from scripts.analyze_historical_gate_abcd_rerun import analyze_run

    analyze_run(output_dir)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            _jsonable(
                {
                    "output_dir": output_dir,
                    "episodes": len(episodes),
                    "gate_diagnostic_rows": len(gate_rows),
                    "runtime_seconds": elapsed,
                    "pairing": pairing,
                    "checkpoint_unchanged": True,
                    "policy_parameters_unchanged": True,
                    "current_scalar_transition_restored": True,
                }
            ),
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
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT / str(settings["output_dir"]) / timestamp
    )
    return run_experiment(settings, output_dir)


if __name__ == "__main__":
    main()
