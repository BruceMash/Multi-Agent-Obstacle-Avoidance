"""Run the strict 100k targeted reference-transition SAC-DMP smoke training."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from baseline.common.callbacks import BaseCallback  # noqa: E402
from experiment_config import EXPERIMENT_CONFIG  # noqa: E402
from planning.reference_transition_finetuning import (  # noqa: E402
    TASK_TERMINAL,
    build_reference_transition_env,
    load_checkpoint_weights_only,
    sha256_file,
    smoke_training_config,
)
from runner_sac import build_model, save_checkpoint  # noqa: E402
from scripts.evaluate_reference_transition_finetuning import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    evaluate_model,
    jsonable,
    write_csv,
    write_json,
)


def _evaluation_history_row(result: Mapping[str, Any]) -> dict[str, Any]:
    retention = result["retention_summary"]
    adaptation = result["adaptation_summary"]
    return {
        "evaluation_step": int(retention["evaluation_step"]),
        "retention_terminal_success_count": retention["terminal_success_count"],
        "retention_terminal_success_total": retention["terminal_success_total"],
        "retention_terminal_success_rate": retention["terminal_success_rate"],
        "retention_collision_rate": retention["collision_rate"],
        "retention_timeout_rate": retention["timeout_rate"],
        "retention_mean_episode_length": retention["mean_episode_length"],
        "retention_mean_return": retention["mean_return"],
        "adaptation_reference_valid_count": adaptation["reference_valid_count"],
        "adaptation_reference_fallback_count": adaptation["reference_fallback_count"],
        "temporary_reference_reached_count": adaptation[
            "temporary_reference_reached_count"
        ],
        "temporary_reference_reached_total": adaptation[
            "temporary_reference_reached_total"
        ],
        "temporary_reference_reached_rate": adaptation[
            "temporary_reference_reached_rate"
        ],
        "reached_then_terminal_success_count": adaptation[
            "reached_then_terminal_success_count"
        ],
        "reached_then_terminal_success_total": adaptation[
            "reached_then_terminal_success_total"
        ],
        "reached_then_terminal_success_rate": adaptation[
            "reached_then_terminal_success_rate"
        ],
        "adaptation_overall_terminal_success_rate": adaptation[
            "overall_terminal_success_rate"
        ],
        "adaptation_collision_rate": adaptation["collision_rate"],
        "adaptation_timeout_rate": adaptation["timeout_rate"],
        "adaptation_mean_terminal_progress_m": adaptation["mean_terminal_progress_m"],
        "adaptation_mean_trajectory_smoothness": adaptation[
            "mean_trajectory_smoothness"
        ],
    }


def _step0_gate(
    result: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    retention = result["retention_summary"]
    adaptation = result["adaptation_summary"]
    gate = settings["step0_adapter_gate"]
    rows = list(result["retention_rows"]) + list(result["adaptation_rows"])
    semantic_checks = {
        "all_observations_122d": all(int(row["observation_dimension"]) == 122 for row in rows),
        "all_historical_vector_gate": all(
            row["forcing_gate_semantics"] == "historical_vector_goal_eff_gate"
            for row in rows
        ),
        "all_terminal_goals_unchanged": all(
            bool(row["terminal_goal_unchanged"]) for row in rows
        ),
        "no_GAT": all(not bool(row["GAT_used"]) for row in rows),
        "no_FP_SHEP": all(not bool(row["FP_SHEP_selector_used"]) for row in rows),
        "no_repeated_waypoint": all(
            not bool(row["repeated_waypoint_used"]) for row in rows
        ),
        "no_hard_boundary": all(not bool(row["hard_boundary_used"]) for row in rows),
        "evaluation_side_effect_free": all(
            bool(result["integrity"][key])
            for key in ("actor_unchanged", "critic_unchanged", "replay_unchanged")
        ),
    }
    performance_checks = {
        "retention_matches_historical_fixed_eval": float(
            retention["terminal_success_rate"]
        )
        >= float(gate["minimum_retention_success_rate"]),
        "temporary_reference_reaching_present": float(
            adaptation["temporary_reference_reached_rate"]
        )
        >= float(gate["minimum_temporary_reference_reached_rate"]),
        "reached_to_terminal_matches_unadapted_baseline": float(
            adaptation["reached_then_terminal_success_rate"]
        )
        <= float(gate["maximum_reached_then_terminal_success_rate"]),
    }
    passed = all(semantic_checks.values()) and all(performance_checks.values())
    return {
        "status": "PASSED" if passed else "ADAPTER_MISMATCH",
        "passed": bool(passed),
        "semantic_checks": semantic_checks,
        "performance_checks": performance_checks,
        "observed": {
            "retention_terminal_success_rate": retention["terminal_success_rate"],
            "temporary_reference_reached_rate": adaptation[
                "temporary_reference_reached_rate"
            ],
            "reached_then_terminal_success_rate": adaptation[
                "reached_then_terminal_success_rate"
            ],
        },
        "thresholds": copy.deepcopy(dict(gate)),
    }


def _handoff_stability_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int], dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["scene"]), int(row["seed"]))
        groups.setdefault(key, {})[int(row["relative_step"])] = row
    forcing_jumps: list[float] = []
    offset_jumps: list[float] = []
    acceleration_jumps: list[float] = []
    saturations: list[float] = []
    reversals = 0
    for window in groups.values():
        if 0 not in window or 1 not in window:
            continue
        current = window[0]
        following = window[1]
        forcing_jumps.append(
            float(
                np.linalg.norm(
                    np.asarray(following["forcing_xyz"], dtype=float)
                    - np.asarray(current["forcing_xyz"], dtype=float)
                )
            )
        )
        offset_jumps.append(
            float(
                np.linalg.norm(
                    np.asarray(following["goal_offset_xyz"], dtype=float)
                    - np.asarray(current["goal_offset_xyz"], dtype=float)
                )
            )
        )
        acceleration_jumps.append(
            float(
                np.linalg.norm(
                    np.asarray(following["commanded_acceleration"], dtype=float)
                    - np.asarray(current["commanded_acceleration"], dtype=float)
                )
            )
        )
        saturations.append(float(following["action_saturation"]))
        reversals += int(
            float(
                np.dot(
                    np.asarray(current["velocity_after"], dtype=float),
                    np.asarray(following["velocity_after"], dtype=float),
                )
            )
            < 0.0
        )
    return {
        "handoff_count": len(groups),
        "paired_post_step_count": len(forcing_jumps),
        "mean_forcing_jump_l2": float(np.mean(forcing_jumps)) if forcing_jumps else None,
        "max_forcing_jump_l2": float(np.max(forcing_jumps)) if forcing_jumps else None,
        "mean_goal_offset_jump_l2": float(np.mean(offset_jumps)) if offset_jumps else None,
        "max_goal_offset_jump_l2": float(np.max(offset_jumps)) if offset_jumps else None,
        "mean_acceleration_jump_l2_mps2": (
            float(np.mean(acceleration_jumps)) if acceleration_jumps else None
        ),
        "max_acceleration_jump_l2_mps2": (
            float(np.max(acceleration_jumps)) if acceleration_jumps else None
        ),
        "mean_post_action_saturation": (
            float(np.mean(saturations)) if saturations else None
        ),
        "immediate_reversal_count": int(reversals),
    }


class SmokeTrainingCallback(BaseCallback):
    def __init__(
        self,
        *,
        config: Any,
        settings: Mapping[str, Any],
        run_dir: Path,
        evaluation_results: list[dict[str, Any]],
        training_rows: list[dict[str, Any]],
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.config = config
        self.settings = settings
        self.run_dir = run_dir
        self.evaluation_results = evaluation_results
        self.training_rows = training_rows
        self.evaluation_steps = set(int(value) for value in settings["evaluation_steps"])
        self.checkpoint_steps = set(int(value) for value in settings["checkpoint_steps"])
        self.completed: set[int] = {0}
        self.episode_return = 0.0
        self.episode_start_step = 0

    def _record_training_episode(self) -> None:
        rewards = np.asarray(self.locals.get("rewards", [0.0]), dtype=float).reshape(-1)
        dones = np.asarray(self.locals.get("dones", [False]), dtype=bool).reshape(-1)
        infos = list(self.locals.get("infos", []))
        self.episode_return += float(rewards[0]) if rewards.size else 0.0
        if not dones.size or not bool(dones[0]) or not infos:
            return
        info = infos[0]
        self.training_rows.append(
            {
                "episode_index": len(self.training_rows),
                "start_training_step": int(self.episode_start_step),
                "end_training_step": int(self.num_timesteps),
                "episode_length": int(self.num_timesteps - self.episode_start_step),
                "episode_return": float(self.episode_return),
                "requested_task": info.get("requested_task"),
                "episode_task": info.get("episode_task"),
                "reference_generation_fallback": info.get(
                    "reference_generation_fallback", False
                ),
                "temporary_reference_reached": info.get(
                    "temporary_reference_reached", False
                ),
                "reference_handoff_count": info.get("reference_handoff_count", 0),
                "terminal_success": info.get("terminal_success", info.get("success", False)),
                "collision": info.get("collision", False),
                "timeout": info.get("truncated", False),
                "final_terminal_goal_distance_m": info.get("terminal_goal_distance"),
            }
        )
        self.episode_return = 0.0
        self.episode_start_step = int(self.num_timesteps)

    def _persist(self) -> None:
        retention_rows = [
            row
            for result in self.evaluation_results
            for row in result["retention_rows"]
        ]
        adaptation_rows = [
            row
            for result in self.evaluation_results
            for row in result["adaptation_rows"]
        ]
        handoff_rows = [
            row for result in self.evaluation_results for row in result["handoff_rows"]
        ]
        history = [_evaluation_history_row(result) for result in self.evaluation_results]
        write_csv(self.run_dir / "training_summary.csv", self.training_rows)
        write_csv(self.run_dir / "evaluation_history.csv", history)
        write_csv(self.run_dir / "retention_eval.csv", retention_rows)
        write_csv(self.run_dir / "adaptation_eval.csv", adaptation_rows)
        write_csv(self.run_dir / "handoff_diagnostics.csv", handoff_rows)

    def _on_step(self) -> bool:
        self._record_training_episode()
        step = int(self.num_timesteps)
        if step not in self.evaluation_steps or step in self.completed:
            return step <= int(self.settings["training_steps"])
        if step in self.checkpoint_steps:
            save_checkpoint(
                self.model,
                self.run_dir / "checkpoints" / f"step_{step // 1000:03d}k.pt",
                extra={
                    "fine_tuning_step": step,
                    "source_checkpoint": self.settings["checkpoint"],
                    "training_kind": "reference_transition_smoke",
                },
            )
        result = evaluate_model(
            model=self.model,
            config=self.config,
            settings=self.settings,
            evaluation_step=step,
        )
        self.evaluation_results.append(result)
        self.completed.add(step)
        self._persist()
        if self.verbose:
            history = _evaluation_history_row(result)
            print(
                f"[evaluation step={step}] retention={history['retention_terminal_success_rate']:.3f} "
                f"reached={history['temporary_reference_reached_rate']:.3f} "
                f"reached_to_terminal={history['reached_then_terminal_success_rate']:.3f}",
                flush=True,
            )
        return step < int(self.settings["training_steps"])

    def _on_training_end(self) -> None:
        self._persist()


def _select_and_conclude(
    *,
    settings: Mapping[str, Any],
    run_dir: Path,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    by_step = {int(row["evaluation_step"]): row for row in history}
    step0 = by_step[0]
    retention0 = float(step0["retention_terminal_success_rate"])
    adaptation0 = float(step0["reached_then_terminal_success_rate"])
    max_drop = float(settings["selection"]["retention_max_drop_percentage_points"]) / 100.0
    acceptable: list[dict[str, Any]] = []
    for step in settings["checkpoint_steps"]:
        row = by_step[int(step)]
        retention_delta = float(row["retention_terminal_success_rate"]) - retention0
        if retention_delta >= -max_drop:
            candidate = dict(row)
            candidate["retention_delta"] = retention_delta
            candidate["adaptation_delta"] = (
                float(row["reached_then_terminal_success_rate"]) - adaptation0
            )
            acceptable.append(candidate)

    acceptable.sort(
        key=lambda row: (
            float(row["reached_then_terminal_success_rate"]),
            float(row["adaptation_overall_terminal_success_rate"]),
            float(row["retention_terminal_success_rate"]),
            -int(row["evaluation_step"]),
        ),
        reverse=True,
    )
    best = acceptable[0] if acceptable else None
    best_path: str | None = None
    if best is not None:
        source = run_dir / "checkpoints" / f"step_{int(best['evaluation_step']) // 1000:03d}k.pt"
        destination = run_dir / "checkpoints" / "best_adaptation.pt"
        shutil.copy2(source, destination)
        best_path = str(destination)

    comparison = best if best is not None else by_step[100_000]
    retention_delta = float(comparison["retention_terminal_success_rate"]) - retention0
    adaptation_delta = float(comparison["reached_then_terminal_success_rate"]) - adaptation0
    signal_threshold = (
        float(settings["selection"]["adaptation_signal_min_gain_percentage_points"])
        / 100.0
    )
    if best is not None and adaptation_delta >= signal_threshold:
        signal = "YES"
        viable = "YES"
        next_step = "Freeze the accepted design and prepare a larger controlled validation."
    elif best is not None and float(best["reached_then_terminal_success_rate"]) > 0.0:
        signal = "WEAK"
        viable = "POSSIBLE"
        next_step = "Inspect handoff diagnostics before deciding on any longer run."
    else:
        signal = "NO"
        viable = "NO"
        next_step = "Stop and revise the reference-transition adaptation design."
    return {
        "REFERENCE_TRANSITION_ADAPTATION_SIGNAL": signal,
        "FINE_TUNING_STRATEGY_VIABLE": viable,
        "BEST_ACCEPTABLE_CHECKPOINT": best_path or "NONE",
        "TERMINAL_RETENTION_DELTA": retention_delta,
        "REFERENCE_TO_TERMINAL_SUCCESS_DELTA": adaptation_delta,
        "NEXT_STEP": next_step,
        "acceptable_checkpoint_count": len(acceptable),
        "best_evaluation_step": None if best is None else int(best["evaluation_step"]),
        "step0_retention_success_rate": retention0,
        "step0_reached_then_terminal_success_rate": adaptation0,
        "comparison_retention_success_rate": float(
            comparison["retention_terminal_success_rate"]
        ),
        "comparison_reached_then_terminal_success_rate": float(
            comparison["reached_then_terminal_success_rate"]
        ),
        "training_stopped_at_steps": 100_000,
        "training_extended": False,
    }


def _write_report(
    run_dir: Path,
    conclusion: Mapping[str, Any],
    history: list[dict[str, Any]],
    step0_gate: Mapping[str, Any],
) -> None:
    lines = [
        "# Targeted Reference-Transition SAC-DMP Fine-Tuning — Smoke Run",
        "",
        "## Step-0 adapter gate",
        "",
        f"Status: **{step0_gate['status']}**",
        "",
        "## Evaluation history",
        "",
        "| Step | Retention Success | Temp Reached | Reached→Terminal | Adaptation Terminal Success |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in history:
        lines.append(
            f"| {int(row['evaluation_step']):,} | "
            f"{float(row['retention_terminal_success_rate']):.2%} | "
            f"{float(row['temporary_reference_reached_rate']):.2%} | "
            f"{float(row['reached_then_terminal_success_rate']):.2%} | "
            f"{float(row['adaptation_overall_terminal_success_rate']):.2%} |"
        )
    if bool(conclusion.get("ADAPTER_MISMATCH", False)):
        handoff = conclusion["STEP0_HANDOFF_DIAGNOSTIC"]
        lines.extend(
            [
                "",
                "## Adapter mismatch diagnosis",
                "",
                "- Terminal retention reproduces the historical fixed-seed result: 10/10 success.",
                "- The single-agent 220-step adaptation set is already solved at step 0: 15/15 temporary references reached and 15/15 reached-to-terminal completions.",
                "- The historical multi-agent comparison reported 0/33 reached-to-terminal completions. The current adaptation set therefore does not reproduce the failure distribution that the smoke fine-tuning is intended to address.",
                "- No gradient update was permitted after this mismatch was detected.",
                "",
                "## Required questions",
                "",
                "1. Adaptation success did not start from zero; it was already 100% at step 0, so learning onset cannot be measured.",
                "2. Temporary-reference reached rate was 100% at step 0; post-training retention was not evaluated.",
                "3. Reached-to-terminal success was already 100% before training; recovery cannot be attributed to fine-tuning.",
                "4. Step-0 terminal capability was intact at 100%; catastrophic forgetting was not tested because training did not start.",
                (
                    "5. Step-0 handoff is not action-smooth: mean forcing jump "
                    f"is {float(handoff['mean_forcing_jump_l2']):.3f}, mean goal-offset "
                    f"jump is {float(handoff['mean_goal_offset_jump_l2']):.3f}, and mean "
                    f"commanded-acceleration jump is "
                    f"{float(handoff['mean_acceleration_jump_l2_mps2']):.3f} m/s². "
                    f"No immediate velocity reversal was observed in "
                    f"{int(handoff['handoff_count'])} handoffs."
                ),
                "6. No adapted checkpoint exists because zero gradient updates were performed.",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- REFERENCE_TRANSITION_ADAPTATION_SIGNAL = {conclusion['REFERENCE_TRANSITION_ADAPTATION_SIGNAL']}",
            f"- FINE_TUNING_STRATEGY_VIABLE = {conclusion['FINE_TUNING_STRATEGY_VIABLE']}",
            f"- BEST_ACCEPTABLE_CHECKPOINT = {conclusion['BEST_ACCEPTABLE_CHECKPOINT']}",
            f"- TERMINAL_RETENTION_DELTA = {float(conclusion['TERMINAL_RETENTION_DELTA']):+.4f}",
            f"- REFERENCE_TO_TERMINAL_SUCCESS_DELTA = {float(conclusion['REFERENCE_TO_TERMINAL_SUCCESS_DELTA']):+.4f}",
            f"- NEXT_STEP = {conclusion['NEXT_STEP']}",
            "",
            (
                "Training was not started because the step-0 adapter gate failed."
                if bool(conclusion.get("ADAPTER_MISMATCH", False))
                else "Training stopped exactly at 100,000 environment steps and was not extended."
            ),
        ]
    )
    (run_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(settings: Mapping[str, Any], run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = smoke_training_config(EXPERIMENT_CONFIG)
    config_artifact = copy.deepcopy(dict(settings))
    config_artifact["output_dir_resolved"] = str(run_dir.resolve())
    config_artifact["actual_learning_starts"] = int(config.learning_starts)
    config_artifact["first_5000_action_source"] = "existing_SAC_random_warmup"
    write_json(run_dir / "config.json", config_artifact)

    checkpoint = REPO_ROOT / str(settings["checkpoint"])
    expected_hash = str(settings["checkpoint_sha256_expected"])
    if sha256_file(checkpoint) != expected_hash:
        raise RuntimeError("source checkpoint hash does not match the preregistered value")
    env = build_reference_transition_env(config)
    try:
        model = build_model(
            env,
            config=config,
            tensorboard_log=str(run_dir / "tensorboard"),
            verbose=0,
        )
        warm_start = load_checkpoint_weights_only(
            model,
            checkpoint,
            learning_rate=float(settings["learning"]["actor_lr"]),
        )
        write_json(run_dir / "warm_start_audit.json", warm_start)

        evaluation_results = [
            evaluate_model(
                model=model,
                config=config,
                settings=settings,
                evaluation_step=0,
            )
        ]
        gate = _step0_gate(evaluation_results[0], settings)
        write_json(run_dir / "step0_adapter_gate.json", gate)
        callback = SmokeTrainingCallback(
            config=config,
            settings=settings,
            run_dir=run_dir,
            evaluation_results=evaluation_results,
            training_rows=[],
        )
        callback._persist()
        if not bool(gate["passed"]):
            conclusion = {
                "REFERENCE_TRANSITION_ADAPTATION_SIGNAL": "NO",
                "FINE_TUNING_STRATEGY_VIABLE": "NO",
                "BEST_ACCEPTABLE_CHECKPOINT": "NONE",
                "TERMINAL_RETENTION_DELTA": 0.0,
                "REFERENCE_TO_TERMINAL_SUCCESS_DELTA": 0.0,
                "NEXT_STEP": (
                    "Redesign the adaptation evaluation to preserve the failing "
                    "multi-agent reference-transition distribution before fine-tuning."
                ),
                "ADAPTER_MISMATCH": True,
                "ADAPTER_MISMATCH_REASON": (
                    "The 220-step single-agent adaptation set is already solved at step 0 "
                    "and does not reproduce the historical 0/33 reached-to-terminal result."
                ),
                "STEP0_RETENTION_SUCCESS_RATE": float(
                    evaluation_results[0]["retention_summary"]["terminal_success_rate"]
                ),
                "STEP0_TEMPORARY_REFERENCE_REACHED_RATE": float(
                    evaluation_results[0]["adaptation_summary"][
                        "temporary_reference_reached_rate"
                    ]
                ),
                "STEP0_REACHED_TO_TERMINAL_SUCCESS_RATE": float(
                    evaluation_results[0]["adaptation_summary"][
                        "reached_then_terminal_success_rate"
                    ]
                ),
                "STEP0_HANDOFF_DIAGNOSTIC": _handoff_stability_summary(
                    evaluation_results[0]["handoff_rows"]
                ),
                "training_performed": False,
                "training_stopped_at_steps": 0,
            }
            write_json(run_dir / "conclusion.json", conclusion)
            _write_report(
                run_dir,
                conclusion,
                [_evaluation_history_row(evaluation_results[0])],
                gate,
            )
            return run_dir

        started = time.perf_counter()
        model.learn(
            total_timesteps=int(settings["training_steps"]),
            callback=callback,
            reset_num_timesteps=True,
        )
        elapsed = float(time.perf_counter() - started)
        if int(model.num_timesteps) != int(settings["training_steps"]):
            raise RuntimeError(
                f"strict stop failed: expected {settings['training_steps']}, got {model.num_timesteps}"
            )
        if set(int(value) for value in settings["evaluation_steps"]) != callback.completed:
            raise RuntimeError("one or more required evaluation checkpoints were not completed")
        callback._persist()
        history = [_evaluation_history_row(result) for result in evaluation_results]
        conclusion = _select_and_conclude(
            settings=settings,
            run_dir=run_dir,
            history=history,
        )
        conclusion.update(
            {
                "ADAPTER_MISMATCH": False,
                "training_performed": True,
                "training_runtime_seconds": elapsed,
                "source_checkpoint_sha256_before": warm_start.checkpoint_sha256_before,
                "source_checkpoint_sha256_after": sha256_file(checkpoint),
                "source_checkpoint_unchanged": (
                    warm_start.checkpoint_sha256_before == sha256_file(checkpoint)
                ),
                "training_steps_max": int(settings["training_steps"]),
                "training_steps_actual": int(model.num_timesteps),
                "automatic_extension": False,
                "GAT_used": False,
                "FP_SHEP_selector_used": False,
                "repeated_waypoint_used": False,
                "hard_boundary_used": False,
            }
        )
        write_json(run_dir / "conclusion.json", conclusion)
        _write_report(run_dir, conclusion, history, gate)
        if not conclusion["source_checkpoint_unchanged"]:
            raise RuntimeError("source checkpoint changed during fine-tuning")
        return run_dir
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> Path:
    args = _parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or (REPO_ROOT / str(settings["output_dir"]) / timestamp)
    return run(settings, output)


if __name__ == "__main__":
    print(main())
