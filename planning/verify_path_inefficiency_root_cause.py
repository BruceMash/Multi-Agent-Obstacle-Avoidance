"""Independent raw-record and artifact reconciliation for the path audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2/formal_records/M9_Proposed_RERR_GAT_SAC_DMP"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); args = parser.parse_args(); output = args.root.resolve()
    required = [
        "01_metric_contract/PATH_METRIC_CONTRACT.json",
        "02_formal_path_decomposition/FORMAL_PATH_DECOMPOSITION.csv",
        "02_formal_path_decomposition/FORMAL_EPISODE_PATH_SUMMARY.csv",
        "03_err_segment_analysis/ERR_SEGMENT_EFFICIENCY.csv",
        "03_err_segment_analysis/ERR_SEGMENT_EFFICIENCY_SUMMARY.json",
        "04_cancellation/REFERENCE_SEQUENCE_CANCELLATION.csv",
        "05_path_jerk_relation/ERR_PATH_ASSOCIATION.json",
        "05_path_jerk_relation/PATH_SMOOTHNESS_ASSOCIATION.json",
        "05_path_jerk_relation/JERK_COST_TEMPORAL_ATTRIBUTION.json",
        "06_stable_reference/STABLE_REFERENCE_JERK_AUDIT.csv",
        "07_tacv_medium_diagnostic/TACV_MEDIUM_PATH_DIAGNOSTIC.csv",
        "08_stage_analysis/STAGE_PATH_INEFFICIENCY_SUMMARY.csv",
        "10_root_cause/PATH_INEFFICIENCY_ROOT_CAUSE_REPORT.md",
        "10_root_cause/PATH_REPAIR_GO_NO_GO.json",
        "10_root_cause/PATH_ROOT_CAUSE_SUMMARY.csv",
        "11_paper_ready/figure_manifest.csv",
        "11_paper_ready/PAPER_PATH_EFFICIENCY_DIAGNOSTIC.md",
        "conclusion.json", "FINAL_REPORT.md",
    ]
    checks: list[dict[str, Any]] = []
    for relative in required:
        path = output / relative
        checks.append({"check": f"required:{relative}", "pass": path.exists() and path.stat().st_size > 0, "detail": path.stat().st_size if path.exists() else "missing"})

    agents = read_csv(output / required[1]); episodes = read_csv(output / required[2]); segments = read_csv(output / required[3]); cancellation = read_csv(output / required[5])
    checks += [
        {"check": "successful_episode_rows", "pass": len(episodes) == 381, "detail": len(episodes)},
        {"check": "successful_agent_rows", "pass": len(agents) == 1143, "detail": len(agents)},
        {"check": "unique_episode_keys", "pass": len({r["scenario_id"] for r in episodes}) == 381, "detail": len({r["scenario_id"] for r in episodes})},
        {"check": "unique_agent_keys", "pass": len({(r["scenario_id"],r["agent_id"]) for r in agents}) == len(agents), "detail": len({(r["scenario_id"],r["agent_id"]) for r in agents})},
        {"check": "unique_segment_keys", "pass": len({(r["scenario_id"],r["agent_id"],r["segment_index"]) for r in segments}) == len(segments), "detail": len(segments)},
        {"check": "unique_cancellation_keys", "pass": len({(r["scenario_id"],r["agent_id"],r["first_segment_index"]) for r in cancellation}) == len(cancellation), "detail": len(cancellation)},
    ]

    raw_success: dict[str, dict[str, Any]] = {}
    for path in sorted(FORMAL.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["episode"]["team_success"]:
            raw_success[record["episode"]["scenario_id"]] = record
    checks.append({"check": "raw_success_count", "pass": len(raw_success) == 381, "detail": len(raw_success)})

    agent_by_episode = defaultdict(list)
    for row in agents: agent_by_episode[row["scenario_id"]].append(row)
    max_path = 0.0; max_smooth = 0.0
    for scenario, record in raw_success.items():
        rows = agent_by_episode[scenario]
        max_path = max(max_path, abs(sum(float(row["path_length_m"]) for row in rows) - float(record["episode"]["team_path_length_m"])))
        max_smooth = max(max_smooth, abs(sum(float(row["smoothness_m2_s6"]) for row in rows)/len(rows) - float(record["episode"]["trajectory_smoothness"])))
    checks += [
        {"check": "raw_team_path_reproduction", "pass": max_path < 1e-8, "detail": max_path},
        {"check": "raw_smoothness_reproduction", "pass": max_smooth < 1e-8, "detail": max_smooth},
    ]

    segment_by_agent = defaultdict(float)
    for row in segments: segment_by_agent[(row["scenario_id"],row["agent_id"])] += float(row["segment_path_length_m"])
    max_segment = max(abs(float(row["path_length_m"]) - segment_by_agent[(row["scenario_id"],row["agent_id"])]) for row in agents)
    checks.append({"check": "segment_path_partition", "pass": max_segment < 1e-8, "detail": max_segment})

    temporal = json.loads((output / required[8]).read_text(encoding="utf-8"))
    cost_partition = temporal["within_0p5s"]["cost_fraction"] + temporal["outside_0p5s"]["cost_fraction"]
    checks.append({"check": "temporal_cost_partition", "pass": abs(cost_partition-1.0) < 1e-12, "detail": cost_partition})

    conclusion = json.loads((output / "conclusion.json").read_text(encoding="utf-8"))
    mandatory = ["ORIGINAL_FORMAL_SUCCESS","ORIGINAL_FORMAL_SUCCESSFUL_EPISODES","MEAN_PATH_LENGTH_SUCCESS","MEDIAN_PATH_LENGTH_SUCCESS","MEAN_MISSION_DISTANCE","MEAN_DETOUR_RATIO","MEDIAN_DETOUR_RATIO","MEAN_EXCESS_PATH_M","MEAN_BACKWARD_PATH_M","MEAN_BACKWARD_RATIO","MEAN_LATERAL_MOTION_M","MEAN_VERTICAL_VARIATION_M","MEAN_VERTICAL_RATIO","MEAN_ERR_ACCEPTED_CHANGES","MEDIAN_ERR_SEGMENT_EFFICIENCY","NEGATIVE_PROGRESS_SEGMENT_FRACTION","LOW_EFFICIENCY_SEGMENT_FRACTION_ETA_LT_0P2","PATH_SHARE_IN_LOW_EFFICIENCY_SEGMENTS","EXCESS_PATH_SHARE_ASSOCIATED_WITH_LOW_EFFICIENCY_SEGMENTS","MEAN_SEGMENT_CANCELLATION","HIGH_CANCELLATION_PAIR_FRACTION","ERR_COUNT_VS_EXCESS_PATH_RHO","LOW_EFFICIENCY_SHARE_VS_SMOOTHNESS_RHO","DETOUR_RATIO_VS_SMOOTHNESS_RHO","VERTICAL_VARIATION_VS_SMOOTHNESS_RHO","JERK_COST_WITHIN_0P2S_OF_REFERENCE_CHANGE","JERK_COST_WITHIN_0P5S_OF_REFERENCE_CHANGE","JERK_COST_OUTSIDE_0P5S_SWITCH_WINDOWS","STABLE_REFERENCE_JERK_LEVEL","TACV_MEDIUM_PATH_CHANGE","TACV_MEDIUM_DETOUR_RATIO_CHANGE","TACV_MEDIUM_LOW_EFFICIENCY_PATH_SHARE_CHANGE","TACV_MEDIUM_ERR_COUNT_CHANGE","PATH_EXCESS_PRIMARY_SOURCE","PATH_AND_JERK_SHARED_CAUSE_EVIDENCE","FRAMEWORK_PRESERVING_PATH_REPAIR_HEADROOM","NEW_SIMULATION_RUN","METHOD_CHANGED","FORMAL_RESULT_CHANGED","ACADEMIC_INTEGRITY_GATE","NEXT_STEP_RECOMMENDATION","FINAL_RECOMMENDATION"]
    checks.append({"check": "conclusion_mandatory_fields", "pass": all(key in conclusion for key in mandatory), "detail": [key for key in mandatory if key not in conclusion]})
    checks.append({"check": "method_formal_freeze", "pass": conclusion["NEW_SIMULATION_RUN"]=="NO" and conclusion["METHOD_CHANGED"]=="NO" and conclusion["FORMAL_RESULT_CHANGED"]=="NO", "detail": {key:conclusion[key] for key in ("NEW_SIMULATION_RUN","METHOD_CHANGED","FORMAL_RESULT_CHANGED")}})
    checks.append({"check": "hard_stop_applied", "pass": conclusion["FRAMEWORK_PRESERVING_PATH_REPAIR_HEADROOM"]=="LOW" and conclusion["NEXT_STEP_RECOMMENDATION"]=="NO CHANGE - KEEP ORIGINAL", "detail": {"headroom":conclusion["FRAMEWORK_PRESERVING_PATH_REPAIR_HEADROOM"],"recommendation":conclusion["NEXT_STEP_RECOMMENDATION"]}})

    figures = read_csv(output / "11_paper_ready/figure_manifest.csv")
    figure_ok = len(figures)==14
    for row in figures:
        for field in ("pdf","png_600dpi","source_data","caption"):
            path=Path(row[field]); figure_ok &= path.exists() and path.stat().st_size>0
    checks.append({"check":"figure_A_to_N_outputs","pass":bool(figure_ok),"detail":len(figures)})

    pass_all = all(bool(row["pass"]) for row in checks)
    result = {"status":"PASS" if pass_all else "FAIL","independent_logic":True,"checks":checks,"counts":{"episodes":len(episodes),"agents":len(agents),"segments":len(segments),"cancellation_pairs":len(cancellation),"figures":len(figures)},"artifact_hashes":{relative:digest(output/relative) for relative in required if (output/relative).exists()}}
    path = output / "10_root_cause/independent_verification.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"failed":[row for row in checks if not row["pass"]]},ensure_ascii=False,indent=2))
    if not pass_all: raise SystemExit(1)


if __name__ == "__main__": main()
