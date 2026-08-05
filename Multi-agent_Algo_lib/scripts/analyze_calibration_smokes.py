from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TRAIN_KEYS = (
    "critic_loss",
    "critic_gradient_norm",
    "critic_gradient_clip_rate",
    "actor_loss",
    "actor_gradient_norm",
    "actor_gradient_clip_rate",
    "q_replay",
    "q_critic_gap",
    "q_replay_variance",
    "entropy",
    "alpha",
    "action_forcing_abs_mean",
    "action_offset_abs_mean",
    "forcing_saturation_rate",
    "goal_offset_saturation_rate",
    "pre_tanh_proxy_std",
    "effective_forcing_norm",
    "nominal_drive_norm",
    "acceleration_clip_rate",
    "velocity_clip_rate",
)

REWARD_KEYS = (
    "reward_progress",
    "reward_obstacle_potential_penalty",
    "reward_boundary_potential_penalty",
    "reward_inter_agent_potential_penalty",
    "reward_stagnation_penalty",
    "reward_acceleration_penalty",
    "reward_acceleration_clip_penalty",
    "reward_individual_success_bonus",
    "reward_team_success_bonus",
    "reward_team_collision_penalty",
    "reward_local_collision_penalty",
    "reward_team_timeout_penalty",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def numeric(value: Any) -> float:
    text = str(value).strip().lower()
    if text == "true":
        return 1.0
    if text == "false":
        return 0.0
    return float(text)


def mean_field(rows: list[dict[str, str]], key: str, tail: int | None = None) -> float:
    selected = rows[-tail:] if tail else rows
    values = finite([numeric(row.get(key, "nan")) for row in selected])
    return float(np.mean(values)) if values.size else float("nan")


def last_field(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return float("nan")
    return numeric(rows[-1].get(key, "nan"))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    config = load_json(run_dir / "config.json")
    args = config.get("script_args", config.get("args", config))
    train = read_csv(run_dir / "train_metrics.csv")
    episodes = read_csv(run_dir / "metrics.csv")
    evaluation = read_csv(run_dir / "eval_metrics.csv")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "seed": int(args.get("seed", config.get("seed", -1))),
        "total_steps": int(args.get("total_steps", -1)),
        "forcing_limit": float(args.get("forcing_limit", float("nan"))),
        "forcing_gate_kappa": float(args.get("forcing_gate_kappa", float("nan"))),
        "actor_lr": float(args.get("actor_lr", float("nan"))),
        "critic_lr": float(args.get("critic_lr", float("nan"))),
        "episode_count": len(episodes),
        "episode_success_rate": mean_field(episodes, "success"),
        "episode_collision_rate": mean_field(episodes, "collision"),
        "episode_agent_success_rate": mean_field(episodes, "agent_success_rate"),
        "eval_success_rate": last_field(evaluation, "eval_success_rate"),
        "eval_agent_success_rate": last_field(evaluation, "eval_agent_success_rate"),
        "eval_mean_reward": last_field(evaluation, "eval_mean_reward"),
        "eval_mean_len": last_field(evaluation, "eval_mean_len"),
        "eval_inter_agent_collision_rate": last_field(
            evaluation, "eval_inter_agent_collision_rate"
        ),
        "eval_obstacle_collision_rate": last_field(
            evaluation, "eval_obstacle_collision_rate"
        ),
    }
    for key in TRAIN_KEYS:
        summary[f"train_tail_{key}"] = mean_field(train, key, tail=5)
    effective = summary["train_tail_effective_forcing_norm"]
    nominal = summary["train_tail_nominal_drive_norm"]
    summary["train_tail_effective_nominal_ratio"] = (
        effective / nominal if np.isfinite(effective) and nominal > 0.0 else float("nan")
    )
    for key in REWARD_KEYS:
        summary[f"episode_mean_{key}"] = mean_field(episodes, key)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["forcing_limit"], row["actor_lr"], row["critic_lr"])
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (forcing_limit, actor_lr, critic_lr), members in sorted(groups.items()):
        grouped: dict[str, Any] = {
            "forcing_limit": forcing_limit,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "seed_count": len(members),
            "seeds": ",".join(str(member["seed"]) for member in members),
        }
        for key in members[0]:
            if key in grouped or key in {"run_dir", "seed"}:
                continue
            values = finite(
                [float(member[key]) for member in members if isinstance(member[key], (int, float))]
            )
            if values.size:
                grouped[f"mean_{key}"] = float(np.mean(values))
                grouped[f"std_{key}"] = float(np.std(values))
        output.append(grouped)
    return output


def plot_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [f"f={row['forcing_limit']:g}\nseed={row['seed']}" for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = (
        ("eval_success_rate", "Evaluation success rate"),
        ("train_tail_forcing_saturation_rate", "Forcing saturation rate"),
        ("train_tail_effective_nominal_ratio", "Effective forcing / nominal drive"),
        ("train_tail_critic_gradient_clip_rate", "Critic gradient clip rate"),
    )
    for axis, (key, title) in zip(axes.flat, panels):
        axis.bar(x, [row[key] for row in rows], color="#2878B5")
        axis.set_xticks(x, labels, rotation=20)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "smoke_comparison.png", dpi=180)
    plt.close(figure)


def write_report(
    path: Path, rows: list[dict[str, Any]], groups: list[dict[str, Any]], source_root: Path
) -> None:
    lines = [
        "# Calibration smoke summary",
        "",
        f"Source: `{source_root}`",
        "",
        "The table reports the last five training log windows and the final fixed-seed evaluation.",
        "These short runs diagnose scale and numerical behavior; they do not establish convergence.",
        "",
        "| forcing | seed | train success | eval success | forcing saturation | eff./nominal | accel clip | actor grad clip | critic grad clip |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {forcing_limit:.3g} | {seed} | {episode_success_rate:.3f} | "
            "{eval_success_rate:.3f} | {train_tail_forcing_saturation_rate:.3f} | "
            "{train_tail_effective_nominal_ratio:.3f} | {train_tail_acceleration_clip_rate:.3f} | "
            "{train_tail_actor_gradient_clip_rate:.3f} | {train_tail_critic_gradient_clip_rate:.3f} |".format(
                **row
            )
        )
    lines.extend(["", "## Cross-seed groups", ""])
    for group in groups:
        lines.append(
            "- forcing={forcing_limit:g}, actor_lr={actor_lr:g}, critic_lr={critic_lr:g}: "
            "eval success={mean_eval_success_rate:.3f}±{std_eval_success_rate:.3f}, "
            "forcing saturation={mean_train_tail_forcing_saturation_rate:.3f}, "
            "effective/nominal={mean_train_tail_effective_nominal_ratio:.3f}.".format(**group)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate short MASAC-DMP calibration runs.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    run_dirs = sorted({path.parent for path in args.input_root.rglob("train_metrics.csv")})
    rows = [summarize_run(run_dir) for run_dir in run_dirs]
    if not rows:
        raise FileNotFoundError(f"No train_metrics.csv found below {args.input_root}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root or args.input_root / f"summary_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    groups = group_rows(rows)
    write_csv(output_dir / "run_summary.csv", rows)
    write_csv(output_dir / "group_summary.csv", groups)
    plot_summary(rows, output_dir)
    write_report(output_dir / "report.md", rows, groups, args.input_root)
    print(output_dir)


if __name__ == "__main__":
    main()
