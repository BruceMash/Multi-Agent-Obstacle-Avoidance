from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from planning.long_range_collision_recheck import audit_trajectory_collisions


def _entry() -> dict:
    return {
        "workspace_bounds": [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],
        "static_obstacles": [
            {
                "type": "box",
                "center": [4.0, 4.0, 4.0],
                "half_extents": [0.5, 0.5, 0.5],
                "safety_margin": 0.0,
            }
        ],
        "dynamic_obstacles": [
            {
                "type": "moving_sphere_constant_translation",
                "center": [8.0, 8.0, 8.0],
                "velocity": [0.0, 0.0, 0.0],
                "radius": 0.5,
                "safety_margin": 0.0,
                "motion_model": "constant_direction_translation",
                "future_available_to_planner": False,
            }
        ],
        "dynamic_obstacle_trajectories": [
            [[8.0, 8.0, 8.0], [7.0, 7.0, 7.0], [6.0, 6.0, 6.0]]
        ],
    }


def test_collision_recheck_decomposes_all_four_types() -> None:
    positions = np.asarray(
        [
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            [[4.0, 4.0, 4.0], [7.0, 7.0, 7.0], [7.4, 7.0, 7.0]],
            [[6.0, 6.0, 6.0], [-0.1, 5.0, 5.0], [9.0, 9.0, 9.0]],
        ],
        dtype=float,
    )
    result = audit_trajectory_collisions(positions, _entry())
    assert result["static_obstacle_collision"]
    assert result["dynamic_obstacle_collision"]
    assert result["inter_agent_collision"]
    assert result["boundary_collision"]
    assert result["first_static_obstacle_collision_step"] == 1
    assert result["first_dynamic_obstacle_collision_step"] == 1
    assert result["first_inter_agent_collision_step"] == 1
    assert result["first_boundary_collision_step"] == 2


def test_collision_recheck_excludes_initial_frame() -> None:
    positions = np.asarray(
        [
            [[4.0, 4.0, 4.0], [4.2, 4.0, 4.0], [-1.0, 3.0, 3.0]],
            [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]],
        ],
        dtype=float,
    )
    result = audit_trajectory_collisions(positions, _entry())
    assert not result["any_collision"]


def test_collision_recheck_matches_existing_long_range_records_when_present() -> None:
    root = Path(
        "artifacts/semi_structured_long_range_main_benchmark/20260820_193228"
    )
    manifest_path = root / "08_development/development_manifest.json"
    records = root / "08_development/D05_interaction_feasibility_mask_full200/episode_records"
    if not manifest_path.is_file() or not records.is_dir():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {row["scenario_id"]: row for row in manifest["entries"]}
    checked = 0
    for record_path in sorted(records.glob("DEV_LR_*.json"))[:20]:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if "summary" not in record:
            continue
        with np.load(records / record["trajectory_file"], allow_pickle=False) as archive:
            positions = archive["positions"]
            if positions.ndim == 2:
                steps = archive["steps"]
                agent_ids = archive["agent_ids"]
                unique_steps = sorted(set(map(int, steps)))
                lookup = {
                    (int(step), int(agent_id)): point
                    for step, agent_id, point in zip(steps, agent_ids, positions)
                }
                positions = np.asarray(
                    [
                        [lookup[(step, agent_id)] for agent_id in range(3)]
                        for step in unique_steps
                    ],
                    dtype=float,
                )
            result = audit_trajectory_collisions(
                positions, entries[record["entry_identity"]["scenario_id"]]
            )
        episode = record["episode"]
        assert result["obstacle_collision"] == bool(episode["obstacle_collision"])
        assert result["inter_agent_collision"] == bool(episode["inter_agent_collision"])
        assert result["any_collision"] == bool(episode["collision"])
        checked += 1
    assert checked > 0
