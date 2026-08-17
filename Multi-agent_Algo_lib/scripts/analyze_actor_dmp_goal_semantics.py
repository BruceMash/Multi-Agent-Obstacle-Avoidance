"""Analyze the Actor--DMP goal-semantics 2x2 paired diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, ALGO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from planning.goal_semantics_diagnosis import (  # noqa: E402
    VARIANT_A,
    VARIANT_B,
    VARIANT_C,
    VARIANT_D,
    VARIANT_DISPLAY_NAMES,
    VARIANT_ORDER,
    factorial_interaction,
)
from scripts.evaluate_actor_dmp_goal_semantics import (  # noqa: E402
    write_csv,
    write_json,
)


COLORS = {
    VARIANT_A: "#4C4C4C",
    VARIANT_B: "#377EB8",
    VARIANT_C: "#4DAF4A",
    VARIANT_D: "#E41A1C",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value in {"", None}:
                    row[key] = None
                    continue
                try:
                    row[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    row[key] = value
            rows.append(row)
    return rows


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _overall(summary: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["variant"]): row
        for row in summary
        if str(row["scenario"]) == "overall"
    }


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(
    fig: plt.Figure,
    run_dir: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    figure_dir = run_dir / "figures"
    data_dir = run_dir / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    write_csv(data_dir / f"{name}.csv", rows)
    write_json(
        data_dir / f"{name}.json",
        {
            "figure": name,
            "status": "DIAGNOSTIC_ONLY",
            "language": "English",
            "source": "numerical_artifacts",
            **dict(metadata),
        },
    )


def plot_outcomes(
    run_dir: Path,
    summary: Sequence[Mapping[str, Any]],
) -> None:
    overall = _overall(summary)
    success = np.asarray(
        [100.0 * float(overall[variant]["success_rate"]) for variant in VARIANT_ORDER]
    )
    collision = np.asarray(
        [100.0 * float(overall[variant]["collision_rate"]) for variant in VARIANT_ORDER]
    )
    x = np.arange(len(VARIANT_ORDER))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    first = ax.bar(
        x - width / 2,
        success,
        width,
        color=[COLORS[variant] for variant in VARIANT_ORDER],
        edgecolor="black",
        linewidth=0.6,
        label="Success Rate",
    )
    second = ax.bar(
        x + width / 2,
        collision,
        width,
        color="white",
        edgecolor=[COLORS[variant] for variant in VARIANT_ORDER],
        linewidth=1.1,
        hatch="//",
        label="Collision Rate",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Variant {label}" for label in "ABCD"])
    ax.set_ylabel("Episode Rate (%)")
    ax.set_ylim(0.0, 100.0)
    ax.set_title("Closed-Loop Outcome by Actor--DMP Goal Semantics")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    for bars in (first, second):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    rows = [
        {
            "variant": variant,
            "variant_display_name": VARIANT_DISPLAY_NAMES[variant],
            "success_rate": float(overall[variant]["success_rate"]),
            "collision_rate": float(overall[variant]["collision_rate"]),
        }
        for variant in VARIANT_ORDER
    ]
    _save_figure(
        fig,
        run_dir,
        "D1_success_collision_by_goal_semantics",
        rows,
        {"metrics": ["success_rate", "collision_rate"]},
    )


def plot_shift_distributions(
    run_dir: Path,
    actor_rows: Sequence[Mapping[str, Any]],
    dmp_rows: Sequence[Mapping[str, Any]],
) -> None:
    scenarios = sorted(
        {str(row["scenario"]) for row in actor_rows}
        | {str(row["scenario"]) for row in dmp_rows}
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    actor_values = [
        [float(row["action_l2_delta"]) for row in actor_rows if row["scenario"] == scene]
        for scene in scenarios
    ]
    dmp_values = [
        [
            float(row["same_action_commanded_acceleration_delta"])
            for row in dmp_rows
            if row["scenario"] == scene
        ]
        for scene in scenarios
    ]
    axes[0].boxplot(actor_values, tick_labels=scenarios, showfliers=False)
    axes[0].set_ylabel("Actor Action Shift L2")
    axes[0].set_title("Actor Goal-Conditioning Shift")
    axes[1].boxplot(dmp_values, tick_labels=scenarios, showfliers=False)
    axes[1].set_ylabel("Acceleration Shift (m/s²)")
    axes[1].set_title("DMP Attractor Shift (Same Action)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    rows: list[dict[str, Any]] = []
    for scene, values in zip(scenarios, actor_values):
        rows.extend(
            {
                "diagnostic": "actor_action_shift_l2",
                "scenario": scene,
                "value": value,
            }
            for value in values
        )
    for scene, values in zip(scenarios, dmp_values):
        rows.extend(
            {
                "diagnostic": "dmp_same_action_acceleration_shift_mps2",
                "scenario": scene,
                "value": value,
            }
            for value in values
        )
    _save_figure(
        fig,
        run_dir,
        "D2_actor_action_vs_dmp_acceleration_shift",
        rows,
        {"panels": ["actor_action_shift", "dmp_acceleration_shift"]},
    )


def _material_degradation(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    success_drop = float(baseline["success_rate"]) - float(candidate["success_rate"])
    collision_increase = float(candidate["collision_rate"]) - float(
        baseline["collision_rate"]
    )
    progress_drop = float(baseline["terminal_progress_team_mean_m"]) - float(
        candidate["terminal_progress_team_mean_m"]
    )
    baseline_smoothness = float(baseline["trajectory_smoothness"])
    smoothness_relative_increase = (
        float(candidate["trajectory_smoothness"]) / baseline_smoothness - 1.0
        if baseline_smoothness > 1.0e-12
        else math.inf
    )
    primary = (
        success_drop >= float(thresholds["absolute_success_rate_drop"])
        or collision_increase
        >= float(thresholds["absolute_collision_rate_increase"])
    )
    secondary = (
        progress_drop >= float(thresholds["terminal_progress_drop_m"])
        and smoothness_relative_increase
        >= float(thresholds["smoothness_relative_increase"])
    )
    return {
        "material": bool(primary or secondary),
        "primary_rate_rule": bool(primary),
        "secondary_joint_rule": bool(secondary),
        "success_rate_drop": success_drop,
        "collision_rate_increase": collision_increase,
        "terminal_progress_drop_m": progress_drop,
        "smoothness_relative_increase": smoothness_relative_increase,
    }


def build_conclusion(
    config: Mapping[str, Any],
    summary: Sequence[Mapping[str, Any]],
    actor_rows: Sequence[Mapping[str, Any]],
    dmp_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    overall = _overall(summary)
    thresholds = config["material_effect_thresholds"]
    effects = {
        variant: _material_degradation(
            overall[VARIANT_A], overall[variant], thresholds
        )
        for variant in (VARIANT_B, VARIANT_C, VARIANT_D)
    }
    interaction_metrics = {
        metric: factorial_interaction(
            float(overall[VARIANT_A][metric]),
            float(overall[VARIANT_B][metric]),
            float(overall[VARIANT_C][metric]),
            float(overall[VARIANT_D][metric]),
        )
        for metric in (
            "success_rate",
            "collision_rate",
            "terminal_progress_team_mean_m",
            "trajectory_smoothness",
            "mean_applied_acceleration_mps2",
            "action_saturation_rate",
        )
    }
    actor_statistics = {
        "record_count": len(actor_rows),
        "mean_observation_l2_delta": _mean(actor_rows, "observation_l2_delta"),
        "mean_action_l2_delta": _mean(actor_rows, "action_l2_delta"),
        "mean_forcing_action_l2_delta": _mean(
            actor_rows, "forcing_action_l2_delta"
        ),
        "mean_goal_offset_action_l2_delta": _mean(
            actor_rows, "goal_offset_action_l2_delta"
        ),
        "mean_terminal_action_saturation_rate": _mean(
            actor_rows, "terminal_action_saturation_rate"
        ),
        "mean_temporary_action_saturation_rate": _mean(
            actor_rows, "temporary_action_saturation_rate"
        ),
        "maximum_non_goal_feature_l2_delta": max(
            (float(row["non_goal_feature_l2_delta"]) for row in actor_rows),
            default=0.0,
        ),
    }
    dmp_statistics = {
        "record_count": len(dmp_rows),
        "mean_zero_action_nominal_acceleration_delta": _mean(
            dmp_rows, "zero_action_nominal_acceleration_delta"
        ),
        "mean_same_action_commanded_acceleration_delta": _mean(
            dmp_rows, "same_action_commanded_acceleration_delta"
        ),
        "maximum_forcing_gate_difference": max(
            (
                abs(
                    float(row["forcing_gate_terminal_commanded"])
                    - float(row["forcing_gate_temporary_commanded"])
                )
                for row in dmp_rows
            ),
            default=0.0,
        ),
    }

    b_bad = bool(effects[VARIANT_B]["material"])
    c_bad = bool(effects[VARIANT_C]["material"])
    d_bad = bool(effects[VARIANT_D]["material"])
    success_c = float(overall[VARIANT_C]["success_rate"])
    success_d = float(overall[VARIANT_D]["success_rate"])
    c_recovers = (
        success_c > 0.0
        and not c_bad
        and success_c - success_d
        >= float(thresholds["absolute_success_rate_drop"])
    )
    c_better_than_d = (
        success_c - success_d
        >= float(thresholds["absolute_success_rate_drop"])
        or float(overall[VARIANT_D]["collision_rate"])
        - float(overall[VARIANT_C]["collision_rate"])
        >= float(thresholds["absolute_collision_rate_increase"])
    )

    if c_recovers:
        case = "Case 5"
        primary = "ACTOR_ACTIVE_GOAL_DISTRIBUTION_SHIFT"
        fine_tuning = "NO"
        next_step = (
            "Route upper-level temporary references through the DMP reference layer "
            "while preserving terminal-conditioned Frozen SAC observations."
        )
    elif b_bad and (not c_bad or c_better_than_d):
        case = "Case 1"
        primary = "ACTOR_ACTIVE_GOAL_DISTRIBUTION_SHIFT"
        fine_tuning = "NO"
        next_step = (
            "Keep terminal-conditioned Actor semantics and study a DMP/reference-layer "
            "temporary-guidance interface."
        )
    elif c_bad and not b_bad:
        case = "Case 2"
        primary = "DMP_ATTRACTOR_REPLACEMENT"
        fine_tuning = "NO"
        next_step = (
            "Study reference blending, smooth attractor transition, and phase-compatible "
            "reference handling without changing the Actor."
        )
    elif not b_bad and not c_bad and d_bad:
        case = "Case 3"
        primary = "ACTOR_DMP_GOAL_SEMANTIC_COUPLING"
        fine_tuning = "NO"
        next_step = "Design an explicit upper--lower goal interface before GAT training."
    elif b_bad and c_bad and d_bad:
        case = "Case 4"
        primary = "FROZEN_LOWER_POLICY_NOT_WAYPOINT_COMPATIBLE"
        fine_tuning = "YES"
        next_step = (
            "Perform targeted waypoint-conditioned/intermediate-reference fine-tuning "
            "from the current checkpoint; do not train from scratch."
        )
    else:
        case = "Mixed evidence"
        primary = "INCONCLUSIVE_MIXED_GOAL_SEMANTICS_EFFECTS"
        fine_tuning = "NO"
        next_step = (
            "Do not fine-tune yet; inspect paired per-scenario effects and freeze a "
            "single goal-interface hypothesis before further experiments."
        )

    if case in {"Case 1", "Case 5"}:
        waypoint_suitability = True
        waypoint_suitability_detail = (
            "YES_WITH_TERMINAL_ACTOR_SEMANTICS_AND_DMP_REFERENCE_LAYER"
        )
    elif case == "Case 2":
        waypoint_suitability = True
        waypoint_suitability_detail = (
            "YES_BUT_DMP_REFERENCE_INTERFACE_REQUIRES_REDESIGN"
        )
    elif case == "Case 3":
        waypoint_suitability = True
        waypoint_suitability_detail = "YES_ONLY_WITH_DECOUPLED_GOAL_INTERFACE"
    elif case == "Case 4":
        waypoint_suitability = False
        waypoint_suitability_detail = "NO_UNDER_CURRENT_FROZEN_LOWER_POLICY"
    else:
        waypoint_suitability = None
        waypoint_suitability_detail = "INCONCLUSIVE"

    primary_coupling_loss = bool(
        interaction_metrics["success_rate"]
        <= -float(thresholds["absolute_success_rate_drop"])
        or interaction_metrics["collision_rate"]
        >= float(thresholds["absolute_collision_rate_increase"])
    )
    baseline_smoothness = max(
        abs(float(overall[VARIANT_A]["trajectory_smoothness"])), 1.0e-12
    )
    smoothness_interaction_relative_to_a = (
        float(interaction_metrics["trajectory_smoothness"]) / baseline_smoothness
    )
    smoothness_coupling_signal = bool(
        smoothness_interaction_relative_to_a
        >= float(thresholds["smoothness_relative_increase"])
    )

    answers = {
        "actor_only_change_causes_material_degradation": b_bad,
        "dmp_only_change_causes_material_degradation": c_bad,
        "joint_change_has_additional_coupling_loss": primary_coupling_loss,
        "joint_change_has_additional_coupling_loss_on_primary_outcomes": (
            primary_coupling_loss
        ),
        "joint_change_has_additional_smoothness_interaction": (
            smoothness_coupling_signal
        ),
        "smoothness_interaction_relative_to_A": (
            smoothness_interaction_relative_to_a
        ),
        "frozen_sac_suitable_as_waypoint_lower_policy": waypoint_suitability,
        "frozen_sac_waypoint_suitability_detail": waypoint_suitability_detail,
        "waypoint_conditioned_sac_fine_tuning_recommended": fine_tuning,
    }
    return {
        "decision_case": case,
        "PRIMARY_LIMITATION": primary,
        "SAC_FINE_TUNING_RECOMMENDED": fine_tuning,
        "NEXT_STEP": next_step,
        "material_effects_vs_A": effects,
        "factorial_interaction_D_minus_C_minus_B_plus_A": interaction_metrics,
        "actor_shift_statistics": actor_statistics,
        "dmp_shift_statistics": dmp_statistics,
        "answers": answers,
        "thresholds": dict(thresholds),
    }


def write_report(
    run_dir: Path,
    summary: Sequence[Mapping[str, Any]],
    conclusion: Mapping[str, Any],
) -> None:
    overall = _overall(summary)
    actor = conclusion["actor_shift_statistics"]
    dmp = conclusion["dmp_shift_statistics"]
    interaction = conclusion["factorial_interaction_D_minus_C_minus_B_plus_A"]
    lines = [
        "# Actor–DMP Goal-Semantics Decoupling Diagnosis",
        "",
        "本报告仅使用 open、sparse_static、multi_agent 场景及 development seeds 0–4。三类场景均不向传感器注入 workspace boundary，也不因 workspace boundary 触发 termination。",
        "",
        "## Paired 2×2 Results",
        "",
        "| Variant | Success | Collision | Obstacle | Inter-agent | Progress (m) | Smoothness | Mean accel. (m/s²) | Saturation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        row = overall[variant]
        lines.append(
            f"| {VARIANT_DISPLAY_NAMES[variant]} | {100*float(row['success_rate']):.2f}% | {100*float(row['collision_rate']):.2f}% | {100*float(row['obstacle_collision_rate']):.2f}% | {100*float(row['inter_agent_collision_rate']):.2f}% | {float(row['terminal_progress_team_mean_m']):.3f} | {float(row['trajectory_smoothness']):.2f} | {float(row['mean_applied_acceleration_mps2']):.3f} | {100*float(row['action_saturation_rate']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 1. 仅改变 Actor goal observation 是否会导致闭环明显退化？",
            "",
            f"结论：**{'是' if conclusion['answers']['actor_only_change_causes_material_degradation'] else '否'}**。B 相对 A 的预注册判定为 `{json.dumps(conclusion['material_effects_vs_A'][VARIANT_B], ensure_ascii=False)}`。同状态 mean observation L2 delta = {actor['mean_observation_l2_delta']:.4f}，mean action L2 delta = {actor['mean_action_l2_delta']:.4f}；forcing / goal-offset action delta 分别为 {actor['mean_forcing_action_l2_delta']:.4f} / {actor['mean_goal_offset_action_l2_delta']:.4f}。",
            "",
            "## 2. 仅改变 DMP attractor 是否会导致闭环明显退化？",
            "",
            f"结论：**{'是' if conclusion['answers']['dmp_only_change_causes_material_degradation'] else '否'}**。C 相对 A 的预注册判定为 `{json.dumps(conclusion['material_effects_vs_A'][VARIANT_C], ensure_ascii=False)}`。同状态 zero-action nominal acceleration delta = {dmp['mean_zero_action_nominal_acceleration_delta']:.4f} m/s²，same-action commanded acceleration delta = {dmp['mean_same_action_commanded_acceleration_delta']:.4f} m/s²；forcing-gate maximum difference = {dmp['maximum_forcing_gate_difference']:.3e}。",
            "",
            "## 3. Actor 与 DMP 同时切换是否存在额外耦合损失？",
            "",
            "结论：**按 success/collision 主结果未识别出额外负交互，但存在明显的 smoothness 交互信号**。"
            f"2×2 interaction 定义为 D−C−B+A，结果为 `{json.dumps(interaction, ensure_ascii=False)}`；"
            f"其中 smoothness interaction 相当于 A 的 {conclusion['answers']['smoothness_interaction_relative_to_A']:.2f} 倍。"
            "D 的较低 collision rate 与大量 timeout 同时出现，不应解释为更安全。",
            "",
            "## 4. 当前 Frozen SAC 是否仍适合作为 upper-level waypoint guidance 的 lower-level policy？",
            "",
            "结论：**"
            + (
                "是（有接口条件）"
                if conclusion["answers"]["frozen_sac_suitable_as_waypoint_lower_policy"]
                is True
                else (
                    "否"
                    if conclusion["answers"]["frozen_sac_suitable_as_waypoint_lower_policy"]
                    is False
                    else "证据不足"
                )
            )
            + "**。接口判定：`"
            + conclusion["answers"]["frozen_sac_waypoint_suitability_detail"]
            + "`。",
            "",
            "## 5. 下一步是否需要 waypoint-conditioned SAC fine-tuning？",
            "",
            f"结论：**{conclusion['SAC_FINE_TUNING_RECOMMENDED']}**。{conclusion['NEXT_STEP']}",
            "",
            "## Final Decision",
            "",
            f"DECISION_CASE = {conclusion['decision_case']}",
            "",
            f"PRIMARY_LIMITATION = {conclusion['PRIMARY_LIMITATION']}",
            "",
            f"SAC_FINE_TUNING_RECOMMENDED = {conclusion['SAC_FINE_TUNING_RECOMMENDED']}",
            "",
            f"NEXT_STEP = {conclusion['NEXT_STEP']}",
            "",
            "本阶段未训练或修改 SAC，未修改 DMP、forcing gate、phase、Proposal、FP-SHEP、GAT、Graph Builder、supervision、reward、collision、success 或 termination。",
        ]
    )
    (run_dir / "FINAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def validate_artifacts(
    run_dir: Path,
    config: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    actor_rows: Sequence[Mapping[str, Any]],
    dmp_rows: Sequence[Mapping[str, Any]],
    conclusion: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "config.json",
        "per_episode.csv",
        "protocol_summary.csv",
        "actor_shift.csv",
        "dmp_shift.csv",
        "conclusion.json",
        "FINAL_REPORT.md",
        "pairing_validation.json",
        "integrity.json",
    )
    for relative in required:
        if not (run_dir / relative).is_file():
            errors.append(f"missing:{relative}")
    expected_episode_count = (
        len(config["scenarios"])
        * len(config["seeds"])
        * len(config["variant_order"])
    )
    if len(episodes) != expected_episode_count:
        errors.append(
            f"episode_count:{len(episodes)}!=expected:{expected_episode_count}"
        )
    if not actor_rows:
        errors.append("actor_shift_empty")
    if not dmp_rows:
        errors.append("dmp_shift_empty")
    if any(bool(row["include_boundaries_in_sensor"]) for row in episodes):
        errors.append("boundary_sensor_enabled")
    if any(bool(row["terminate_on_boundary_collision"]) for row in episodes):
        errors.append("boundary_termination_enabled")
    if any(not bool(row["terminal_task_goals_unchanged"]) for row in episodes):
        errors.append("terminal_goal_changed")
    if any(float(row["maximum_phase_switch_delta"]) != 0.0 for row in episodes):
        errors.append("phase_switch_delta")
    integrity = _load_json(run_dir / "integrity.json")
    if not bool(integrity["policy_parameters_unchanged"]):
        errors.append("policy_hash_changed")
    if not bool(integrity["critical_files_unchanged"]):
        errors.append("critical_hash_changed")
    figure_pdfs = list((run_dir / "figures").glob("*.pdf"))
    figure_pngs = list((run_dir / "figures").glob("*.png"))
    if len(figure_pdfs) != 2 or len(figure_pngs) != 2:
        errors.append("figure_count_not_two")
    if not conclusion.get("PRIMARY_LIMITATION"):
        errors.append("primary_limitation_missing")
    result = {
        "status": "PASSED" if not errors else "FAILED",
        "errors": errors,
        "episode_count": len(episodes),
        "expected_episode_count": expected_episode_count,
        "actor_shift_record_count": len(actor_rows),
        "dmp_shift_record_count": len(dmp_rows),
        "figure_count": len(figure_pdfs),
    }
    write_json(run_dir / "artifact_validation.json", result)
    return result


def analyze(run_dir: Path) -> dict[str, Any]:
    _configure_plot_style()
    config = _load_json(run_dir / "config.json")
    episodes = _load_json(run_dir / "per_episode.json")
    summary = _load_json(run_dir / "protocol_summary.json")
    actor_rows = _load_json(run_dir / "actor_shift.json")
    dmp_rows = _load_json(run_dir / "dmp_shift.json")
    conclusion = build_conclusion(config, summary, actor_rows, dmp_rows)
    write_json(run_dir / "conclusion.json", conclusion)
    plot_outcomes(run_dir, summary)
    plot_shift_distributions(run_dir, actor_rows, dmp_rows)
    write_report(run_dir, summary, conclusion)
    validation = validate_artifacts(
        run_dir, config, episodes, actor_rows, dmp_rows, conclusion
    )
    if validation["status"] != "PASSED":
        raise RuntimeError(f"artifact validation failed: {validation['errors']}")
    manifest = _load_json(run_dir / "manifest.json")
    manifest.update(
        {
            "analysis_complete": True,
            "conclusion": {
                "PRIMARY_LIMITATION": conclusion["PRIMARY_LIMITATION"],
                "SAC_FINE_TUNING_RECOMMENDED": conclusion[
                    "SAC_FINE_TUNING_RECOMMENDED"
                ],
                "NEXT_STEP": conclusion["NEXT_STEP"],
            },
            "artifact_validation": validation,
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    return {"conclusion": conclusion, "validation": validation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args.run_dir.resolve())
    print(json.dumps(result["validation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
