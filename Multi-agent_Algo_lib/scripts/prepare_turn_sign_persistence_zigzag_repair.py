#!/usr/bin/env python3
"""Recover and freeze execution-side turn-persistence thresholds before performance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "artifacts/turn_sign_persistence_zigzag_repair/20260825_222130"
SOURCE_ROOT = REPO_ROOT / "artifacts/safety_adaptive_jerk_limiter/20260825_115704/dev_records/strong/episode_records"
EARLY_CONTRACT = REPO_ROOT / "artifacts/final_residual_zigzag_resolution/20260825_183146/STRONG_EARLY_BYPASS_CONTRACT.json"
METHOD_CONFIG = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/09_final_freeze/method_configs/M9_Proposed_RERR_GAT_SAC_DMP.json"
CONTROL_CONTRACT = ARTIFACT_ROOT / "TURN_PERSISTENCE_CONTROL_CONTRACT.json"
THRESHOLDS = ARTIFACT_ROOT / "TURN_PERSISTENCE_THRESHOLDS.json"
VELOCITY_EPS = 1.0e-9
COMMAND_EPS = 1.0e-9


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sign(value: float) -> int:
    return 1 if value > COMMAND_EPS else -1 if value < -COMMAND_EPS else 0


def main() -> None:
    if CONTROL_CONTRACT.exists() or THRESHOLDS.exists():
        raise RuntimeError("threshold/control freeze already exists; refusing overwrite")
    early = load_json(EARLY_CONTRACT)
    config = load_json(METHOD_CONFIG)
    h_rep = float(config["err"]["h_rep_m"])
    h_emg = float(config["err"]["h_emg_m"])
    if h_rep != 0.35 or h_emg != 0.0:
        raise RuntimeError("source safety thresholds do not match the recovered contract")
    files = sorted(
        path for path in SOURCE_ROOT.glob("*.json")
        if not path.name.endswith("_SOFTWARE_ERROR.json")
    )
    if len(files) != 100:
        raise RuntimeError(f"expected 100 Frozen Strong Development records, found {len(files)}")
    lateral: list[float] = []
    vertical: list[float] = []
    record_hashes: dict[str, dict[str, str]] = {}
    safe_rows = 0
    for path in files:
        payload = load_json(path)
        trajectory_path = path.with_name(str(payload["trajectory_file"]))
        trace_path = path.with_name(str(payload["limiter_trace_file"]))
        trajectory = np.load(trajectory_path)
        trace = np.load(trace_path)
        velocities = np.asarray(trajectory["velocities"], dtype=float)
        accepted = np.zeros((3, 2), dtype=np.int8)
        for row_index in range(len(trace["step"])):
            step = int(trace["step"][row_index])
            agent = int(trace["agent_id"][row_index])
            velocity = velocities[min(step, len(velocities) - 1), agent]
            strong = np.asarray(trace["limited_acceleration"][row_index], dtype=float)
            margin = float(trace["safety_margin_m"][row_index])
            speed_xy = float(np.linalg.norm(velocity[:2]))
            commands: list[float | None] = [None, float(strong[2])]
            if speed_xy > VELOCITY_EPS:
                tangent = velocity[:2] / speed_xy
                normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
                commands[0] = float(np.dot(strong[:2], normal))
            else:
                accepted[agent, 0] = 0
            comfortable = margin > h_rep
            safe_rows += int(comfortable)
            for axis, command in enumerate(commands):
                if command is None:
                    continue
                requested = sign(command)
                if requested == 0:
                    continue
                previous = int(accepted[agent, axis])
                if previous != 0 and requested != previous and comfortable:
                    (lateral if axis == 0 else vertical).append(abs(float(command)))
                accepted[agent, axis] = requested
        record_hashes[str(payload["entry_identity"]["scenario_id"])] = {
            "json_sha256": sha256(path),
            "trajectory_sha256": sha256(trajectory_path),
            "limiter_trace_sha256": sha256(trace_path),
        }
    if not lateral or not vertical:
        raise RuntimeError("safe-state reversal distribution is empty")
    a_lat = float(np.percentile(np.asarray(lateral, dtype=float), 75))
    a_vert = float(np.percentile(np.asarray(vertical, dtype=float), 75))
    created = datetime.now(timezone.utc).isoformat()
    control = {
        "schema_version": "turn_persistence_control_contract_v1",
        "status": "FROZEN_BEFORE_NEW_PERFORMANCE",
        "created_at": created,
        "execution_order": [
            "raw SAC-DMP acceleration",
            "Frozen Strong jerk limiter",
            "turn-sign persistence",
            "original physical acceleration saturation",
            "vehicle dynamics"
        ],
        "dt_s": 0.1,
        "horizontal_geometry": {
            "velocity": "current executed horizontal velocity v_xy before the control step",
            "valid_heading": "norm(v_xy) > 1e-9 m/s",
            "tangent": "v_xy/norm(v_xy)",
            "normal": "[-e_t_y,e_t_x]",
            "tangential_component": "dot(a_strong_xy,e_t), preserved exactly before physical clipping",
            "turn_component": "dot(a_strong_xy,e_n)"
        },
        "vertical_geometry": {"turn_component": "a_strong_z"},
        "state": {
            "per_axis": ["accepted_sign", "pending_opposite_sign", "pending_count"],
            "n_persist": 2,
            "reset": ["episode start", "horizontal heading invalid for horizontal axis", "episode termination"],
            "ERR_reference_change_reset": False
        },
        "safety_priority": {
            "online_margin": "existing active-direction safety margin m_t",
            "warning_bypass_m": h_rep,
            "hard_bypass_m": h_emg,
            "action": "bypass Strong and persistence; raw acceleration proceeds to original physical saturation"
        },
        "frozen_strong": {
            "j_smooth_mps3": early["frozen_strong"]["j_smooth_mps3"],
            "changed": False
        },
        "sources": {
            "method_config": METHOD_CONFIG.relative_to(REPO_ROOT).as_posix(),
            "method_config_sha256": sha256(METHOD_CONFIG),
            "early_bypass_contract": EARLY_CONTRACT.relative_to(REPO_ROOT).as_posix(),
            "early_bypass_contract_sha256": sha256(EARLY_CONTRACT),
            "frozen_strong_development_record_count": len(files)
        },
        "performance_episodes_observed": 0
    }
    thresholds = {
        "schema_version": "turn_persistence_thresholds_v1",
        "status": "FROZEN_BEFORE_NEW_PERFORMANCE",
        "created_at": created,
        "N_PERSIST": 2,
        "PERCENTILE": 75,
        "A_REV_LAT": a_lat,
        "A_REV_VERT": a_vert,
        "units": "m/s^2",
        "comfortable_safe_condition": "m_t > h_rep = 0.35 m",
        "distribution": {
            "horizontal_safe_opposite_request_count": len(lateral),
            "vertical_safe_opposite_request_count": len(vertical),
            "horizontal_min_mps2": float(np.min(lateral)),
            "horizontal_median_mps2": float(np.median(lateral)),
            "horizontal_p75_mps2": a_lat,
            "horizontal_max_mps2": float(np.max(lateral)),
            "vertical_min_mps2": float(np.min(vertical)),
            "vertical_median_mps2": float(np.median(vertical)),
            "vertical_p75_mps2": a_vert,
            "vertical_max_mps2": float(np.max(vertical)),
            "comfortable_safe_trace_rows": safe_rows
        },
        "numerical": {
            "velocity_epsilon_mps": VELOCITY_EPS,
            "command_epsilon_mps2": COMMAND_EPS
        },
        "source": {
            "record_root": SOURCE_ROOT.relative_to(REPO_ROOT).as_posix(),
            "record_count": len(files),
            "record_hashes_sha256": hashlib.sha256(
                json.dumps(record_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        },
        "grid_search": False,
        "performance_episodes_observed": 0
    }
    atomic_json(CONTROL_CONTRACT, control)
    atomic_json(THRESHOLDS, thresholds)
    print(json.dumps({"A_REV_LAT": a_lat, "A_REV_VERT": a_vert, "horizontal_n": len(lateral), "vertical_n": len(vertical)}, indent=2))


if __name__ == "__main__":
    main()
