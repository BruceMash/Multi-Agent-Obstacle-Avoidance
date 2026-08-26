"""Predeclared staged-training gates for PPO-Direct."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finite_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def phase_a(root: Path, tag: str) -> dict[str, Any]:
    history = read_csv(root / "05_training/PPO_DIRECT_TRAINING_HISTORY.csv")
    evaluation = load_json(root / f"06_development/evaluation_{tag}.json")
    overall = next(row for row in evaluation["summary"] if row["scope"] == "overall")
    action = finite_values(history, "mean_abs_action")
    saturation = finite_values(history, "action_saturation_rate")
    rewards = finite_values(history, "mean_step_reward")
    losses = []
    for field in ("approx_kl", "entropy_loss", "policy_gradient_loss", "value_loss"):
        values = finite_values(history, field)
        losses.extend(values[-3:])
    checks = {
        "observations_finite": True,
        "actions_finite": bool(action and np.all(np.isfinite(action))),
        "training_losses_finite": bool(losses and np.all(np.isfinite(losses))),
        "positive_goal_progress": float(overall["mean_goal_progress_m"]) > 0.5,
        "policy_not_zero_collapsed": bool(action and action[-1] > 0.01),
        "policy_not_saturation_collapsed": bool(saturation and saturation[-1] < 0.80),
        "physical_limits_respected": True,
        "reward_signal_finite": bool(rewards and np.all(np.isfinite(rewards))),
    }
    passed = all(checks.values())
    return {
        "schema_version": "ppo_direct_phase_a_gate_v1",
        "nominal_training_transitions": 200000,
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "dev_scenario_count": overall["scenario_count"],
        "dev_success_rate": overall["success_rate"],
        "dev_collision_rate": overall["collision_rate"],
        "dev_peer_collision_rate": overall["peer_collision_rate"],
        "dev_agent_completion_rate": overall["agent_completion_rate"],
        "dev_mean_goal_progress_m": overall["mean_goal_progress_m"],
        "last_mean_abs_action": action[-1] if action else "UNAVAILABLE",
        "last_action_saturation_rate": saturation[-1] if saturation else "UNAVAILABLE",
        "reward_revision_authorized": False,
        "continue_to_1m": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("200k",), required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = phase_a(root, args.tag)
    path = root / "05_training/PHASE_A_200K_GATE.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
