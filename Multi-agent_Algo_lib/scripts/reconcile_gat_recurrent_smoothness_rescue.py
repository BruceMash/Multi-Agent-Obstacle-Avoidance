#!/usr/bin/env python3
"""Independent end-to-end reconciliation for the GAT recurrent rescue artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244"
DEV = ROOT / "07_development"
SCHEMA = "gat_recurrent_smoothness_rescue_independent_reconciliation_v1"
ARMS = {
    "fp_shep": DEV / "FP_SHEP_RERR_DEV400/development_team_results.csv",
    "gat_v1": DEV / "GAT_V1_RERR_DEV400/development_team_results.csv",
    "gat_r": DEV / "GAT_R_RERR_DEV400/development_team_results.csv",
    "gat_rs": DEV / "GAT_RS_RERR_DEV400/development_team_results.csv",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exact_two_sided_binomial(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def close(left: float, right: float, tol: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tol)


def main() -> None:
    errors: list[str] = []
    checks: dict[str, Any] = {}

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    raw = {name: rows(path) for name, path in ARMS.items()}
    for name, arm_rows in raw.items():
        require(len(arm_rows) == 400, f"{name}: expected 400 rows")
        require(len({r['scenario_id'] for r in arm_rows}) == 400, f"{name}: duplicate scenario id")
        require(all(not truth(r["software_error"]) for r in arm_rows), f"{name}: software error")
        require(all(truth(r["scene_reconstruction_match"]) for r in arm_rows), f"{name}: scene mismatch")
        require(
            {r["stage"] for r in arm_rows} == {"stage_1", "stage_2", "stage_3", "stage_4"},
            f"{name}: stage set",
        )
        for stage in ("stage_1", "stage_2", "stage_3", "stage_4"):
            require(sum(r["stage"] == stage for r in arm_rows) == 100, f"{name}/{stage}: count")

    key_sets = {
        name: {(r["scenario_id"], r["seed"], r["stage"], r["family"]) for r in arm_rows}
        for name, arm_rows in raw.items()
    }
    first_keys = key_sets["fp_shep"]
    require(all(value == first_keys for value in key_sets.values()), "paired identity mismatch")

    computed: dict[str, dict[str, Any]] = {}
    indexed: dict[str, dict[str, dict[str, str]]] = {}
    for name, arm_rows in raw.items():
        indexed[name] = {r["scenario_id"]: r for r in arm_rows}
        computed[name] = {
            "n": len(arm_rows),
            "success_count": sum(truth(r["team_success"]) for r in arm_rows),
            "collision_count": sum(truth(r["collision"]) for r in arm_rows),
            "obstacle_collision_count": sum(truth(r["obstacle_collision"]) for r in arm_rows),
            "peer_collision_count": sum(truth(r["inter_agent_collision"]) for r in arm_rows),
            "timeout_count": sum(truth(r["timeout"]) for r in arm_rows),
            "stage_success": {
                stage: sum(truth(r["team_success"]) for r in arm_rows if r["stage"] == stage)
                for stage in ("stage_1", "stage_2", "stage_3", "stage_4")
            },
        }
    expected = {
        "fp_shep": (375, 25, 4, 21, (93, 95, 96, 91)),
        "gat_v1": (358, 42, 10, 32, (89, 89, 91, 89)),
        "gat_r": (326, 74, 3, 71, (78, 82, 83, 83)),
        "gat_rs": (355, 45, 3, 42, (90, 90, 90, 85)),
    }
    for name, (success, collision, obstacle, peer, stage_values) in expected.items():
        value = computed[name]
        require(value["success_count"] == success, f"{name}: success mismatch")
        require(value["collision_count"] == collision, f"{name}: collision mismatch")
        require(value["obstacle_collision_count"] == obstacle, f"{name}: obstacle mismatch")
        require(value["peer_collision_count"] == peer, f"{name}: peer mismatch")
        require(tuple(value["stage_success"].values()) == stage_values, f"{name}: stage mismatch")

    fp = indexed["fp_shep"]
    rs = indexed["gat_rs"]
    rr = indexed["gat_r"]
    fp_only_success = sum(truth(fp[k]["team_success"]) and not truth(rs[k]["team_success"]) for k in fp)
    rs_only_success = sum(truth(rs[k]["team_success"]) and not truth(fp[k]["team_success"]) for k in fp)
    fp_only_peer = sum(truth(fp[k]["inter_agent_collision"]) and not truth(rs[k]["inter_agent_collision"]) for k in fp)
    rs_only_peer = sum(truth(rs[k]["inter_agent_collision"]) and not truth(fp[k]["inter_agent_collision"]) for k in fp)
    success_p = exact_two_sided_binomial(fp_only_success, fp_only_success + rs_only_success)
    peer_p = exact_two_sided_binomial(fp_only_peer, fp_only_peer + rs_only_peer)
    require((fp_only_success, rs_only_success) == (42, 22), "RS/FP success discordance mismatch")
    require((fp_only_peer, rs_only_peer) == (19, 40), "RS/FP peer discordance mismatch")

    rs_fp_pairs = [
        float(rs[k]["trajectory_smoothness"]) - float(fp[k]["trajectory_smoothness"])
        for k in fp
        if truth(fp[k]["team_success"]) and truth(rs[k]["team_success"])
    ]
    rs_r_pairs = [
        float(rs[k]["trajectory_smoothness"]) - float(rr[k]["trajectory_smoothness"])
        for k in rr
        if truth(rr[k]["team_success"]) and truth(rs[k]["team_success"])
    ]
    require(len(rs_fp_pairs) == 333, "RS/FP both-success count")
    require(len(rs_r_pairs) == 290, "RS/R both-success count")
    require(close(fmean(rs_fp_pairs), -1.7389763754170904), "RS/FP jerk delta")
    require(close(fmean(rs_r_pairs), 42.202496940219426), "RS/R jerk delta")

    dev_decision = load_json(DEV / "DEV_GATE_DECISION.json")
    conclusion = load_json(ROOT / "conclusion.json")
    require(dev_decision["DEV_GATE"] == "FAIL", "Dev gate must fail")
    require(conclusion["DEV_GATE"] == "FAIL", "conclusion Dev gate")
    require(conclusion["HOLDOUT_GATE"] == "NOT_RUN", "conclusion Holdout gate")
    require(conclusion["FINAL_GAT_RS_FREEZE"] == "NO", "conclusion freeze")
    require(conclusion["GAT_INCREMENT_UNDER_RERR"] == "NEGATIVE", "conclusion increment")
    require(conclusion["ACADEMIC_INTEGRITY_GATE"] == "PASS", "integrity field")
    require(close(conclusion["FP_DEV_SUCCESS"], 375 / 400), "conclusion FP success")
    require(close(conclusion["GAT_RS_DEV_SUCCESS"], 355 / 400), "conclusion RS success")
    require(close(conclusion["GAT_RS_SMOOTHNESS_PAIRED_DELTA"], fmean(rs_fp_pairs)), "conclusion jerk")
    require(close(conclusion["GAT_RS_VS_FP_DEV_MCNEMAR_P"], success_p), "success p mismatch")
    require(close(conclusion["GAT_RS_VS_FP_PEER_COLLISION_MCNEMAR_P"], peer_p), "peer p mismatch")

    split = load_json(ROOT / "02_recurrent_dataset/independent_split_reconciliation.json")
    require(split["status"] == "PASS", "split reconciliation")
    for pair, values in split["pairwise_overlap"].items():
        require(all(value == 0 for value in values.values()), f"overlap in {pair}")
    require(split["old_formal_v1_episode_rows_read"] == 0, "Formal V1 row read")
    require(split["performance_rows_read"] == 0, "reserved performance read before Dev")

    dataset = load_json(ROOT / "02_recurrent_dataset/recurrent_dataset_reconciliation.json")
    require(dataset["episode_count"] == 200, "training panel episode count")
    require(dataset["state_count"] == 1185, "recurrent state count")
    require(dataset["candidate_branch_count_primary"] == 11852, "branch count")
    require(dataset["eligible_smoothness_pair_count"] == 6770, "pair count")
    require(dataset["formal_v1_used_for_training"] is False, "Formal V1 training use")
    require(dataset["holdout_opened"] is False, "Holdout opened in dataset phase")

    pair_rows = rows(ROOT / "03_counterfactual_rollouts/eligible_smoothness_pairs.csv")
    require(len(pair_rows) == 6770, "eligible pair CSV count")
    require(all(truth(r["safety_equivalent"]) for r in pair_rows), "non-safety-equivalent pair")
    require(all(truth(r["progress_equivalent"]) for r in pair_rows), "non-progress-equivalent pair")
    require(all(not truth(r["null_involved"]) for r in pair_rows), "null in smoothness pair")
    require(
        all(float(r["preferred_jerk_m2_s6"]) < float(r["disfavored_jerk_m2_s6"]) for r in pair_rows),
        "pair direction not lower jerk",
    )

    checkpoint_checks: dict[str, Any] = {}
    for name, manifest_path in {
        "gat_r": ROOT / "05_gat_r_training/GAT_R_TRAINING_MANIFEST.json",
        "gat_rs": ROOT / "06_gat_rs_training/GAT_RS_TRAINING_MANIFEST.json",
    }.items():
        manifest = load_json(manifest_path)
        checkpoint = REPO_ROOT / manifest["selected_checkpoint"]
        observed = digest(checkpoint)
        match = observed == manifest["selected_checkpoint_sha256"]
        checkpoint_checks[name] = {"observed_sha256": observed, "match": match}
        require(match, f"{name}: checkpoint hash")
        require(manifest["gat_runtime_input_changed"] is False, f"{name}: runtime input")
        require(manifest["gat_runtime_role_changed"] is False, f"{name}: runtime role")

    holdout_marker = rows(ROOT / "08_holdout/holdout_selector_comparison.csv")
    holdout_tests = load_json(ROOT / "08_holdout/holdout_paired_tests.json")
    require(len(holdout_marker) == 1 and holdout_marker[0]["status"] == "NOT_RUN", "Holdout marker")
    require(holdout_marker[0]["performance_row_count"] == "0", "Holdout performance count")
    require(holdout_tests["performance_row_count"] == 0, "Holdout paired rows")

    formal_team = rows(ROOT / "10_formal_v2/formal_v2_team_results.csv")
    formal_agent = rows(ROOT / "10_formal_v2/formal_v2_agent_results.csv")
    formal_manifest = load_json(ROOT / "10_formal_v2/FORMAL_V2_MANIFEST.json")
    require(len(formal_team) == 0, "Formal team rows must be zero")
    require(len(formal_agent) == 0, "Formal agent rows must be zero")
    require(formal_manifest["status"] == "NOT_GENERATED", "Formal manifest status")
    require(formal_manifest["scenario_count"] == 0, "Formal scenario count")

    required = [
        "01_smoothness_metric_audit/SMOOTHNESS_METRIC_AUDIT.md",
        "01_smoothness_metric_audit/smoothness_metric_definition.json",
        "02_recurrent_dataset/recurrent_state_sampling_distribution.csv",
        "02_recurrent_dataset/AMBIGUOUS_DECISION_DATASET.csv",
        "03_counterfactual_rollouts/candidate_counterfactual_rollouts.csv",
        "03_counterfactual_rollouts/candidate_safety_labels.csv",
        "03_counterfactual_rollouts/candidate_progress_labels.csv",
        "03_counterfactual_rollouts/candidate_smoothness_labels.csv",
        "03_counterfactual_rollouts/eligible_smoothness_pairs.csv",
        "04_feature_alignment/gat_train_vs_longrange_features.csv",
        "04_feature_alignment/goal_distance_distribution_audit.json",
        "02_recurrent_dataset/null_calibration.csv",
        "02_recurrent_dataset/peer_conflict_training_subset.csv",
        "05_gat_r_training/GAT_R_TRAINING_MANIFEST.json",
        "06_gat_rs_training/GAT_RS_TRAINING_MANIFEST.json",
        "05_gat_r_training/training_curves_gat_r.csv",
        "06_gat_rs_training/training_curves_gat_rs.csv",
        "05_gat_r_training/checkpoint_selection_history.csv",
        "06_gat_rs_training/checkpoint_selection_history.csv",
        "07_development/dev_selector_comparison.csv",
        "07_development/dev_paired_success_tests.json",
        "07_development/dev_both_success_continuous_tests.json",
        "08_holdout/holdout_selector_comparison.csv",
        "08_holdout/holdout_paired_tests.json",
        "09_final_freeze/FINAL_GAT_RS_FREEZE.json",
        "10_formal_v2/FORMAL_V2_MANIFEST.json",
        "10_formal_v2/formal_v2_team_results.csv",
        "10_formal_v2/formal_v2_agent_results.csv",
        "10_formal_v2/formal_v2_fp_vs_gat_rs.json",
        "10_formal_v2/formal_v2_continuous_paired_tests.json",
        "10_formal_v2/formal_v2_stage_summary.csv",
        "10_formal_v2/formal_v2_failure_taxonomy.csv",
        "10_formal_v2/formal_v2_runtime_summary.csv",
        "12_paper_ready/paper_gat_claim_matrix.csv",
        "conclusion.json",
        "FINAL_REPORT.md",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    require(not missing, f"missing required artifacts: {missing}")

    report_text = (ROOT / "FINAL_REPORT.md").read_text(encoding="utf-8")
    for literal in (
        "375/400 (93.75%)",
        "358/400 (89.50%)",
        "326/400 (81.50%)",
        "355/400 (88.75%)",
        "`DEV_GATE = FAIL`",
        "`HOLDOUT_GATE = NOT_RUN`",
        "`FORMAL_V2 = NOT_GENERATED`",
    ):
        require(literal in report_text, f"report missing {literal}")

    checks.update(
        {
            "raw_results": computed,
            "paired_identity_match": all(value == first_keys for value in key_sets.values()),
            "gat_rs_vs_fp_success_discordance": {"fp_only": fp_only_success, "gat_rs_only": rs_only_success, "p": success_p},
            "gat_rs_vs_fp_peer_collision_discordance": {"fp_only": fp_only_peer, "gat_rs_only": rs_only_peer, "p": peer_p},
            "paired_smoothness": {
                "gat_rs_vs_fp": {"n": len(rs_fp_pairs), "mean_delta": fmean(rs_fp_pairs)},
                "gat_rs_vs_gat_r": {"n": len(rs_r_pairs), "mean_delta": fmean(rs_r_pairs)},
            },
            "checkpoint_checks": checkpoint_checks,
            "eligible_smoothness_pair_rows": len(pair_rows),
            "holdout_performance_rows": 0,
            "formal_team_rows": len(formal_team),
            "formal_agent_rows": len(formal_agent),
            "required_artifact_count": len(required),
            "missing_required_artifacts": missing,
        }
    )
    output = {
        "schema_version": SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": checks,
        "conclusion_sha256": digest(ROOT / "conclusion.json"),
        "final_report_sha256": digest(ROOT / "FINAL_REPORT.md"),
    }
    out_path = ROOT / "11_statistics/independent_final_reconciliation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "errors": errors, "checks": checks}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
