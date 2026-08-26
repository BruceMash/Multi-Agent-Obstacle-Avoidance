"""Independent reconciliation for the sector-resolution oscillation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "artifacts/sector_resolution_oscillation_audit/20260824_110611"
FORMAL_RECORDS = REPO_ROOT / (
    "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2/"
    "formal_records/M9_Proposed_RERR_GAT_SAC_DMP"
)
METHOD_CONFIG = REPO_ROOT / (
    "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/09_final_freeze/"
    "method_configs/M9_Proposed_RERR_GAT_SAC_DMP.json"
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def close(left: float, right: float, atol: float = 1.0e-12) -> bool:
    return abs(float(left) - float(right)) <= atol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    conclusion = load(root / "conclusion.json")
    contract = load(root / "01_sector_contract/PROPOSAL_SECTOR_CONTRACT.json")
    compatibility = load(root / "03_root_cause/SECTOR_DENSIFICATION_COMPATIBILITY.json")
    method_config = load(METHOD_CONFIG)
    events = pd.read_csv(root / "02_existing_event_analysis/SECTOR_SWITCH_EVENT_TABLE.csv")
    switches = events[(events["candidate_to_candidate_switch"] == 1) & (events["candidate_changed"] == 1)].copy()
    success_count = 0
    formal_count = 0
    for path in sorted(FORMAL_RECORDS.glob("*.json")):
        episode = load(path)["episode"]
        formal_count += 1
        success_count += int(bool(episode["team_success"]))
    angle_rho = float(spearmanr(switches["sector_center_angular_jump_deg"].round(9), switches["post_switch_jerk_peak"]).statistic)
    elevation_rho = float(spearmanr(switches["sector_elevation_jump_deg"].abs().round(9), switches["post_switch_vertical_jerk_peak"]).statistic)
    adjacent_fraction = float(switches["sectors_adjacent"].mean())
    high_jerk_adjacent = float(switches.loc[switches["high_jerk_switch"] == 1, "sectors_adjacent"].mean())
    switchback = float(switches["switchback_within_1_0s"].mean())
    gat_checkpoint = REPO_ROOT / method_config["gat_checkpoint"]
    sac_checkpoint = REPO_ROOT / method_config["sac_checkpoint"]
    dev = pd.read_csv(root / "06_development/SECTOR_DEVELOPMENT_RESULTS.csv")
    assertions = {
        "formal_record_count_400": formal_count == 400,
        "successful_formal_count_381": success_count == 381,
        "formal_success_0_9525": close(success_count / formal_count, conclusion["CURRENT_ORIGINAL_FORMAL_V2_SUCCESS"]),
        "event_row_count_116164": len(events) == 116164,
        "candidate_reference_count_114054": int(events["is_candidate_reference"].sum()) == 114054,
        "terminal_handoff_count_2110": int((events["new_active_goal_type"] == "terminal").sum()) == 2110,
        "changed_sector_count_45116": len(switches) == 45116,
        "adjacent_fraction_reproduced": close(adjacent_fraction, conclusion["ADJACENT_SWITCH_FRACTION"]),
        "high_jerk_adjacent_fraction_reproduced": close(high_jerk_adjacent, conclusion["HIGH_JERK_ADJACENT_SWITCH_FRACTION"]),
        "switchback_rate_reproduced": close(switchback, conclusion["SWITCHBACK_RATE"]),
        "angle_spearman_reproduced": close(angle_rho, conclusion["SECTOR_ANGLE_JERK_ASSOCIATION"]),
        "elevation_spearman_reproduced": close(elevation_rho, conclusion["ELEVATION_JERK_ASSOCIATION"]),
        "proposal_sector_count_256": contract["proposal_candidate_sector_count"] == 256,
        "proposal_geometry_16x16": contract["proposal_azimuth_bins"] == 16 and contract["proposal_elevation_bins"] == 16,
        "top_k_10": contract["top_k"] == 10 and method_config["top_k"] == 10,
        "gat_canonical_56": contract["gat_canonical_projection_direction_count"] == 56,
        "contract_coupled": conclusion["PROPOSAL_AND_GAT_DIRECTION_CONTRACT"] == "COUPLED",
        "densification_gate_failed": compatibility["SECTOR_DENSIFICATION_ABLATION_AUTHORIZED"] == "NO",
        "all_development_rows_not_run": bool(dev["status"].eq("NOT_RUN_DENSIFICATION_HARD_GATE_FAILED").all()),
        "no_audit_checkpoint_files": not any(root.rglob("*.pt")) and not any(root.rglob("*.pth")),
        "gat_hash_unchanged": digest(gat_checkpoint) == method_config["gat_checkpoint_sha256_expected"],
        "sac_hash_unchanged": digest(sac_checkpoint) == method_config["sac_checkpoint_sha256_expected"],
        "figure_pdf_count_8": len(list((root / "11_paper_ready/pdf").glob("*.pdf"))) == 8,
        "figure_png_count_8": len(list((root / "11_paper_ready/png_600dpi").glob("*.png"))) == 8,
        "formal_reevaluation_no": conclusion["FORMAL_REEVALUATION_RECOMMENDED"] == "NO",
        "academic_integrity_pass": conclusion["ACADEMIC_INTEGRITY_GATE"] == "PASS",
    }
    result = {
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "assertions": assertions,
        "recomputed": {
            "formal_count": formal_count,
            "success_count": success_count,
            "event_rows": len(events),
            "changed_sector_events": len(switches),
            "adjacent_switch_fraction": adjacent_fraction,
            "high_jerk_adjacent_switch_fraction": high_jerk_adjacent,
            "switchback_rate_1s": switchback,
            "angle_jerk_spearman": angle_rho,
            "elevation_vertical_jerk_spearman": elevation_rho,
        },
        "audit_scripts": {
            "planning/audit_sector_resolution_oscillation.py": digest(REPO_ROOT / "planning/audit_sector_resolution_oscillation.py"),
            "planning/plot_sector_resolution_oscillation_audit.py": digest(REPO_ROOT / "planning/plot_sector_resolution_oscillation_audit.py"),
            "planning/finalize_sector_resolution_oscillation_audit.py": digest(REPO_ROOT / "planning/finalize_sector_resolution_oscillation_audit.py"),
            "planning/verify_sector_resolution_oscillation_audit.py": digest(Path(__file__).resolve()),
        },
    }
    (root / "INDEPENDENT_RECONCILIATION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "assertion_count": len(assertions), "failed": [key for key, value in assertions.items() if not value]}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
