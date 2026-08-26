"""Select the PPO-Direct checkpoint using complete Development evaluations only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_steps(path: Path) -> int:
    match = re.search(r"_(\d+)\.zip$", path.name)
    if match is None:
        raise ValueError(f"checkpoint name has no timestep suffix: {path}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    candidates: list[dict[str, Any]] = []
    for evaluation_path in sorted((root / "06_development").glob("evaluation_*.json")):
        evaluation = load_json(evaluation_path)
        overall = next(row for row in evaluation["summary"] if row["scope"] == "overall")
        if int(overall["scenario_count"]) != 100:
            continue
        checkpoint = Path(evaluation["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = (Path.cwd() / checkpoint).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        candidates.append(
            {
                "tag": evaluation["tag"],
                "evaluation": str(evaluation_path.relative_to(root)),
                "checkpoint": str(checkpoint.relative_to(root)),
                "checkpoint_sha256": sha256(checkpoint),
                "training_timesteps": checkpoint_steps(checkpoint),
                "dev_scenarios": 100,
                "team_success_rate": float(overall["success_rate"]),
                "collision_rate": float(overall["collision_rate"]),
                "peer_collision_rate": float(overall["peer_collision_rate"]),
                "agent_completion_rate": float(overall["agent_completion_rate"]),
                "timeout_rate": float(overall["timeout_rate"]),
                "mean_goal_progress_m": float(overall["mean_goal_progress_m"]),
            }
        )
    if not candidates:
        raise RuntimeError("no complete 100-scenario Development evaluation found")

    ranked = sorted(
        candidates,
        key=lambda row: (
            -row["team_success_rate"],
            row["collision_rate"],
            row["peer_collision_rate"],
            row["training_timesteps"],
        ),
    )
    selected = ranked[0]
    record = {
        "schema_version": "ppo_direct_dev_checkpoint_selection_v1",
        "selection_data": "Development only",
        "holdout_accessed": False,
        "formal_v2_accessed": False,
        "primary_metric": "team_success_rate_higher_is_better",
        "tie_break_1": "collision_rate_lower_is_better",
        "tie_break_2": "peer_collision_rate_lower_is_better",
        "tie_break_3": "training_timesteps_lower_is_better",
        "candidate_count": len(ranked),
        "candidates_ranked": ranked,
        "selected": selected,
    }
    output = root / "06_development/PPO_DIRECT_CHECKPOINT_SELECTION.json"
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(selected, ensure_ascii=False))


if __name__ == "__main__":
    main()
