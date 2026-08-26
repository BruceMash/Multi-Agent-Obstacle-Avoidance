#!/usr/bin/env python3
"""Generate the outcome-blind fixed-scene raw Holdout trajectory PDF."""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for search_path in (REPO_ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import plot_safety_adaptive_jerk_limiter as plot


plot.SOURCE_MANIFEST = plot.SOURCE_STUDY_ROOT / "08_holdout/GAT_RS_HOLDOUT_SCENE_MANIFEST.json"
plot.SOURCE_ORIGINAL = plot.SOURCE_STUDY_ROOT / "13_objective_revision/08_holdout/GAT_R_FP_ANCHOR_HOLDOUT400/episode_records"
plot.DEV_ROOT = plot.ARTIFACT_ROOT / "holdout_records"
plot.OUTPUT = plot.ARTIFACT_ROOT / "JERK_LIMITER_HOLDOUT_RAW_TRAJECTORIES.pdf"
plot.ARMS = ("original", "strong")
plot.ARM_LABELS = {"original": "Original", "strong": "Strong"}
plot.LINE_STYLES = {"original": "-", "strong": ":"}
plot.FIXED_SCENES = tuple(f"GATRS_HOLDOUT_{stage}_000" for stage in range(1, 5))
plot.ACCELERATION_TITLE = "Velocity-derived acceleration proxy"
plot.ACCELERATION_YLABEL = r"$\|\Delta v/\Delta t\|$ (m/s$^2$)"
plot.JERK_TITLE = "Velocity-derived 0.1 s jerk proxy"
plot.JERK_YLABEL = r"$\|\Delta^2 v/\Delta t^2\|$ (m/s$^3$)"


def load_record_with_common_velocity_differences(arm: str, sid: str):
    root = plot.record_root(arm)
    record = json.loads((root / f"{sid}.json").read_text(encoding="utf-8"))
    archive = np.load(root / f"{sid}_trajectory.npz")
    position = np.asarray(archive["positions"], dtype=float)
    velocity = np.asarray(archive["velocities"], dtype=float)
    if position.ndim == 2:
        steps = np.asarray(archive["steps"], dtype=int)
        agent_ids = np.asarray(archive["agent_ids"], dtype=int)
        dense_position = np.full((int(np.max(steps)) + 1, 3, 3), np.nan, dtype=float)
        dense_velocity = np.full_like(dense_position, np.nan)
        dense_position[steps, agent_ids] = position
        dense_velocity[steps, agent_ids] = velocity
        position, velocity = dense_position, dense_velocity
    acceleration = np.full_like(velocity, np.nan)
    acceleration[0] = 0.0
    acceleration[1:] = np.diff(velocity, axis=0) / plot.DT
    return record, {"positions": position, "applied_accelerations_full": acceleration}


plot.load_record = load_record_with_common_velocity_differences


if __name__ == "__main__":
    plot.generate()
