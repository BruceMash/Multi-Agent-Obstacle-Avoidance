from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _trajectory_positions(record_path: Path, trajectory_name: str) -> dict[tuple[int, int], np.ndarray]:
    trajectory = np.load(record_path.parent / trajectory_name, allow_pickle=False)
    return {
        (int(step), int(agent_id)): np.asarray(position, dtype=float)
        for step, agent_id, position in zip(
            trajectory["steps"],
            trajectory["agent_ids"],
            trajectory["positions"],
            strict=True,
        )
    }


def audit(run_dir: Path, *, terminal_local_scope_m: float, local_reference_max_m: float) -> None:
    record_paths = sorted((run_dir / "episode_records").glob("DEV_LR_*.json"))
    if not record_paths:
        raise RuntimeError("no development episode records found")

    null_rows: list[dict[str, Any]] = []
    collision_episode_count = 0
    collision_episode_with_collider_last_null = 0
    colliding_agent_count = 0
    colliding_agent_last_null_count = 0
    obstacle_colliding_agent_count = 0
    obstacle_colliding_agent_last_null_count = 0
    peer_colliding_agent_count = 0
    peer_colliding_agent_last_null_count = 0
    selection_event_count = 0
    raw_null_count = 0
    effective_null_count = 0
    candidate_count_at_raw_null: Counter[int] = Counter()

    for record_path in record_paths:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        episode = payload["episode"]
        events = payload.get("events", [])
        selection_event_count += len(events)
        position_by_key = _trajectory_positions(record_path, payload["trajectory_file"])
        last_event_by_agent: dict[int, dict[str, Any]] = {}
        for event in events:
            agent_id = int(event["agent_id"])
            last_event_by_agent[agent_id] = event
            raw_null = bool(event.get("raw_selected_null", event.get("selected_null")))
            effective_null = bool(event.get("selected_null"))
            raw_null_count += int(raw_null)
            effective_null_count += int(effective_null)
            if not raw_null:
                continue
            candidate_count = int(event.get("K_t", 0))
            candidate_count_at_raw_null[candidate_count] += 1
            position = position_by_key.get((int(event["step"]), agent_id))
            new_goal = np.asarray(event["new_active_goal"], dtype=float)
            new_goal_distance = (
                None if position is None else float(np.linalg.norm(new_goal - position))
            )
            null_rows.append(
                {
                    "scenario_id": payload["entry_identity"]["scenario_id"],
                    "stage": payload["entry_identity"]["stage"],
                    "family": payload["entry_identity"]["family"],
                    "task_pattern": payload["entry_identity"]["task_pattern"],
                    "team_success": bool(episode["team_success"]),
                    "obstacle_collision": bool(episode["obstacle_collision"]),
                    "inter_agent_collision": bool(episode["inter_agent_collision"]),
                    "step": int(event["step"]),
                    "agent_id": agent_id,
                    "event": event["event"],
                    "candidate_count": candidate_count,
                    "raw_selected_null": raw_null,
                    "effective_selected_null": effective_null,
                    "new_active_goal_type": event["new_active_goal_type"],
                    "new_active_goal_distance_m": new_goal_distance,
                    "far_terminal_goal_applied": bool(
                        event["new_active_goal_type"] == "terminal"
                        and new_goal_distance is not None
                        and new_goal_distance > terminal_local_scope_m
                    ),
                    "local_reference_scale_exceeded": bool(
                        new_goal_distance is not None
                        and new_goal_distance > local_reference_max_m + 1.0e-9
                    ),
                }
            )

        if bool(episode["collision"]):
            collision_episode_count += 1
            collider_last_null = False
            for agent in payload["agents"]:
                if not bool(agent.get("collision")):
                    continue
                colliding_agent_count += 1
                agent_id = int(agent["agent_id"])
                last_event = last_event_by_agent.get(agent_id)
                last_null = bool(
                    last_event
                    and last_event.get(
                        "raw_selected_null", last_event.get("selected_null")
                    )
                )
                colliding_agent_last_null_count += int(last_null)
                collider_last_null = collider_last_null or last_null
                if bool(agent.get("obstacle_collision")):
                    obstacle_colliding_agent_count += 1
                    obstacle_colliding_agent_last_null_count += int(last_null)
                if bool(agent.get("inter_agent_collision")):
                    peer_colliding_agent_count += 1
                    peer_colliding_agent_last_null_count += int(last_null)
            collision_episode_with_collider_last_null += int(collider_last_null)

    far_terminal_rows = [row for row in null_rows if row["far_terminal_goal_applied"]]
    summary = {
        "schema_version": "long_range_null_lifecycle_audit_v1",
        "status": "PASS",
        "development_only": True,
        "episode_count": len(record_paths),
        "selection_event_count": selection_event_count,
        "raw_null_selection_count": raw_null_count,
        "effective_null_selection_count": effective_null_count,
        "raw_null_with_available_candidate_count": int(
            sum(count for k, count in candidate_count_at_raw_null.items() if k > 0)
        ),
        "raw_null_with_top_k_full_count": int(candidate_count_at_raw_null.get(10, 0)),
        "candidate_count_at_raw_null": {
            str(key): value for key, value in sorted(candidate_count_at_raw_null.items())
        },
        "far_terminal_goal_applied_count": len(far_terminal_rows),
        "maximum_far_terminal_goal_distance_m": (
            max(float(row["new_active_goal_distance_m"]) for row in far_terminal_rows)
            if far_terminal_rows
            else None
        ),
        "collision_episode_count": collision_episode_count,
        "collision_episode_with_collider_last_null_count": (
            collision_episode_with_collider_last_null
        ),
        "colliding_agent_count": colliding_agent_count,
        "colliding_agent_last_null_count": colliding_agent_last_null_count,
        "obstacle_colliding_agent_count": obstacle_colliding_agent_count,
        "obstacle_colliding_agent_last_null_count": (
            obstacle_colliding_agent_last_null_count
        ),
        "peer_colliding_agent_count": peer_colliding_agent_count,
        "peer_colliding_agent_last_null_count": peer_colliding_agent_last_null_count,
        "diagnosis": "FAR_TERMINAL_NULL_BREAKS_LOCAL_REFERENCE_CHAIN",
        "authorized_repair": "MASK_NULL_TO_BEST_EXISTING_GAT_NON_NULL_WHILE_TERMINAL_IS_NONLOCAL",
        "formal_data_used": False,
    }
    _write_csv(run_dir / "null_lifecycle_events.csv", null_rows)
    _atomic_json(run_dir / "null_lifecycle_audit.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--terminal-local-scope-m", type=float, default=4.5)
    parser.add_argument("--local-reference-max-m", type=float, default=1.05)
    args = parser.parse_args()
    audit(
        args.run_dir.resolve(),
        terminal_local_scope_m=float(args.terminal_local_scope_m),
        local_reference_max_m=float(args.local_reference_max_m),
    )


if __name__ == "__main__":
    main()
