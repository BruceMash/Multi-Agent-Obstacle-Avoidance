"""Independent reconciliation for the peer communication-contract closure."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "peer_communication_contract_closure" / "20260819_174406"
RECORD_ROOT = ROOT / "artifacts" / "rerr_safety_closure" / "20260819_134216" / "diagnostic_records"
RUNTIME = ROOT / "artifacts" / "rerr_runtime_compression" / "20260819_162022"
DT = 0.1
D_SAFE = 0.6
RANGE_M = 1.2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    conclusion = read_json(OUTPUT / "conclusion.json")
    paper_rows = read_csv(OUTPUT / "paper_peer_communication_contract.csv")
    dataflow_rows = read_csv(OUTPUT / "code_peer_state_dataflow.csv")
    diagnostic_rows = read_csv(OUTPUT / "ally_range_and_clipping_diagnostic.csv")
    summary = read_json(OUTPUT / "ally_range_and_clipping_summary.json")
    manifest = read_json(OUTPUT / "context_recovery_manifest.json")
    runtime = read_json(RUNTIME / "conclusion.json")
    records = [
        read_json(path)
        for path in sorted(RECORD_ROOT.glob("stage_*/SR*/gat_v1_rerr.json"))
    ]

    raw_peer_collision_count = sum(
        bool(record["episode"]["inter_agent_collision"]) for record in records
    )
    range_leads = [float(row["ally_1_2m_boundary_lead_s"]) for row in diagnostic_rows]
    clipped_leads = [float(row["clipped_h4_lead_s"]) for row in diagnostic_rows]
    exact_leads = [float(row["unclipped_h4_lead_s"]) for row in diagnostic_rows]
    attachment = manifest["goal_attachment"]
    attachment_path = Path(attachment["path"])
    checks = {
        "reissued_goal_attachment_verified": attachment_path.is_file()
        and sha256(attachment_path) == attachment["sha256"]
        and attachment["byte_identical_to_prior_audited_goal"] is True
        and conclusion["GOAL_ATTACHMENT_IDENTICAL_TO_PRIOR_AUDITED_GOAL"] == "YES",
        "paper_sha_matches": all(
            row["paper_sha256"] == sha256(ROOT / "hire-rl-body.tex") for row in paper_rows
        ),
        "paper_has_active_neighbor_state_equations": any(
            row["evidence_id"] == "P06" and row["active"] == "YES" for row in paper_rows
        ),
        "paper_has_no_declared_continuous_contract": conclusion[
            "PAPER_CONTINUOUS_PEER_STATE"
        ]
        == "UNDEFINED",
        "commented_drafts_excluded": sum(
            row["active"] == "NO_COMMENTED_DRAFT" for row in paper_rows
        )
        == 2,
        "dataflow_has_A_B_C_and_actor": {row["category"] for row in dataflow_rows}
        == {
            "A_PRIVATE_SIMULATOR_STATE",
            "B_UPPER_EVENT_EXACT_PEER_INFORMATION",
            "C_EVERY_STEP_FLAT_ALLY_BLOCK",
            "LOWER_FROZEN_ACTOR",
        },
        "raw_record_count_80": len(records) == 80,
        "raw_peer_collision_count_10": raw_peer_collision_count == 10,
        "diagnostic_row_count_10": len(diagnostic_rows) == 10,
        "range_median_1_2": abs(float(np.median(range_leads)) - 1.2) <= 1.0e-12,
        "range_ge05_9": sum(value >= 0.5 - 1.0e-12 for value in range_leads) == 9,
        "clipped_h4_ge05_4": sum(value >= 0.5 - 1.0e-12 for value in clipped_leads) == 4,
        "unclipped_h4_ge05_2": sum(value >= 0.5 - 1.0e-12 for value in exact_leads) == 2,
        "both_h4_medians_0_35": abs(float(np.median(clipped_leads)) - 0.35) <= 1.0e-12
        and abs(float(np.median(exact_leads)) - 0.35) <= 1.0e-12,
        "clipping_never_delayed": all(
            float(row["clipping_lead_change_s"]) >= -1.0e-12
            for row in diagnostic_rows
        ),
        "root_cause_communication_undefined": conclusion[
            "PEER_OBSERVABILITY_ROOT_CAUSE"
        ]
        == "COMMUNICATION_MODEL_UNDEFINED",
        "mandatory_stop_obeyed": conclusion["PHASE_F_EXECUTED"] == "NO"
        and conclusion["NEW_DEVELOPMENT_SCENARIO_COUNT"] == 0
        and conclusion["NEW_TEAM_EPISODE_COUNT"] == 0,
        "fairness_not_changed": conclusion[
            "FINAL_EQUAL_INFORMATION_CONTRACT_UPDATE_REQUIRED"
        ]
        == "NO",
        "runtime_freeze_preserved": conclusion["OPTIMIZED_RERR_FREEZE_PRESERVED"]
        == "YES"
        and abs(
            float(conclusion["OPTIMIZED_RERR_TOTAL_COMPUTE_MS"])
            - float(runtime["OPTIMIZED_TOTAL_COMPUTE_MS"])
        )
        <= 1.0e-12,
        "report_exists": (OUTPUT / "FINAL_REPORT.md").is_file(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(OUTPUT.iterdir())
        if path.is_file() and path.name != "final_reconciliation.json"
    }
    result = {
        "status": status,
        "independent_recomputation": True,
        "checks": checks,
        "recomputed": {
            "paper_rows": len(paper_rows),
            "dataflow_rows": len(dataflow_rows),
            "raw_records": len(records),
            "peer_collisions": raw_peer_collision_count,
            "range_entry_median_lead_s": float(np.median(range_leads)),
            "range_entry_ge_0_5s_count": sum(
                value >= 0.5 - 1.0e-12 for value in range_leads
            ),
            "clipped_h4_ge_0_5s_count": sum(
                value >= 0.5 - 1.0e-12 for value in clipped_leads
            ),
            "unclipped_h4_ge_0_5s_count": sum(
                value >= 0.5 - 1.0e-12 for value in exact_leads
            ),
        },
        "artifact_sha256": artifact_hashes,
    }
    (OUTPUT / "final_reconciliation.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError("peer communication-contract reconciliation failed")


if __name__ == "__main__":
    main()
