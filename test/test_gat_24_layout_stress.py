from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ALGO = ROOT / "Multi-agent_Algo_lib"
if str(ALGO) not in sys.path:
    sys.path.insert(0, str(ALGO))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))
sys.modules.setdefault("pyarrow", None)

from scripts import evaluate_gat_24_layout_stress as stress
from scripts.evaluate_single_policy_aligned_multi_agent import (
    build_single_distribution_multi_config,
)


CONFIG_PATH = ROOT / "configs/evaluation/gat_24_layout_stress.json"


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_config_contract_and_tolerances():
    config = load_config()
    stress._assert_frozen_config(config)
    tolerance = config["compatibility_tolerances"]
    assert tolerance["candidate_atol"] == 1e-6
    assert tolerance["preview_feature_atol"] == 1e-6
    assert tolerance["fp_shep_score_atol"] == 1e-6
    assert tolerance["execution_atol"] == 1e-7
    assert tolerance["trajectory_atol"] == 1e-7
    assert all(tolerance[key] == 0.0 for key in tolerance if key.endswith("_rtol"))
    assert tolerance["threshold_adjustment_after_outcomes_allowed"] is False


def test_manifest_hash_and_layout_family_counts():
    config = load_config()
    path = ROOT / config["frozen_geometry_manifest"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == config[
        "geometry_manifest_sha256_expected"
    ]
    manifest, audit = stress._manifest_audit(path, config)
    assert audit["status"] == "PASSED"
    assert len(manifest["layouts"]) == 24
    assert audit["family_counts"] == {key: 4 for key in "ABCDEF"}


def test_execution_equivalence_representatives_cover_all_families():
    config = load_config()
    representatives = config["execution_equivalence_representatives"]
    assert set(representatives) == set("ABCDEF")
    assert len(set(representatives.values())) == 6
    manifest = json.loads(
        (ROOT / config["frozen_geometry_manifest"]).read_text(encoding="utf-8")
    )
    family_by_id = {row["layout_id"]: row["family"] for row in manifest["layouts"]}
    assert all(family_by_id[layout_id] == family for family, layout_id in representatives.items())


def test_manifest_environment_restores_exact_geometry():
    config = load_config()
    manifest = json.loads(
        (ROOT / config["frozen_geometry_manifest"]).read_text(encoding="utf-8")
    )
    builder = stress.build_manifest_environment_builder(manifest, config)
    multi = build_single_distribution_multi_config(num_agents=3, max_steps=220)
    record = manifest["layouts"][0]
    env, metadata = builder(
        config=multi,
        scenario=config["scenario"],
        seed=int(record["evaluation_seed"]),
        peer_radius=0.3,
    )
    try:
        assert np.array_equal(env.starts, np.asarray(record["starts"], dtype=float))
        assert np.array_equal(
            env.goals, np.asarray(record["terminal_goals"], dtype=float)
        )
        assert len(env.static_obstacles) == 2
        assert metadata["geometry_restored_from_frozen_manifest"] is True
        assert metadata["layout_id"] == record["layout_id"]
    finally:
        env.close()


def test_historical_schema_forces_full_rerun_when_required_metric_missing():
    config = load_config()
    complete, audit = stress._historical_schema_complete(
        ROOT / config["historical_artifact_dir"], config
    )
    assert complete is False
    assert "path_length_m" in audit["missing_per_agent_fields"]


def test_rate_classification_is_for_rate_improvements_only():
    config = load_config()
    assert stress._rate_class(0.10, config) == "YES"
    assert stress._rate_class(0.099, config) == "WEAK"
    assert stress._rate_class(1e-9, config) == "WEAK"
    assert stress._rate_class(0.0, config) == "NO"
    assert stress._rate_class(-0.01, config) == "NO"


def test_failure_taxonomy_is_mutually_exclusive_and_priority_ordered():
    base = {
        "team_success": False,
        "obstacle_collision": False,
        "inter_agent_collision": False,
        "timeout": True,
    }
    agents = [
        {
            "reference_selected_type": "proposal",
            "reference_reached": False,
        }
    ]
    assert stress._failure_category(base, agents) == "TIMEOUT_BEFORE_REFERENCE"
    assert (
        stress._failure_category({**base, "inter_agent_collision": True}, agents)
        == "INTER_AGENT_COLLISION"
    )
    assert (
        stress._failure_category({**base, "obstacle_collision": True}, agents)
        == "OBSTACLE_COLLISION"
    )
    reached = [{"reference_selected_type": "proposal", "reference_reached": True}]
    assert stress._failure_category(base, reached) == "AFTER_REFERENCE_FAILURE"


def test_trace_error_reports_stepwise_max_absolute_error():
    left = [
        {
            "position": np.zeros((3, 3)),
            "action": np.zeros((3, 6)),
        },
        {
            "position": np.ones((3, 3)),
            "action": np.ones((3, 6)),
        },
    ]
    right = [
        {
            "position": np.zeros((3, 3)),
            "action": np.zeros((3, 6)),
        },
        {
            "position": np.ones((3, 3)) + 2e-8,
            "action": np.ones((3, 6)) - 3e-8,
        },
    ]
    assert np.isclose(stress._max_abs_error(left, right, "position"), 2e-8)
    assert np.isclose(stress._max_abs_error(left, right, "action"), 3e-8)


def test_historical_artifact_is_not_configured_as_online_input():
    config = load_config()
    assert config["artifact_policy"]["historical_artifact_used_for_online_decision"] is False
    assert config["artifact_policy"]["current_preview_is_only_fp_shep_and_gat_input"] is True
    source = (
        ROOT / "Multi-agent_Algo_lib/scripts/evaluate_gat_24_layout_stress.py"
    ).read_text(encoding="utf-8")
    assert source.index("_build_all_shared_bundles(") < source.index(
        "_candidate_and_preview_reproduction_audit("
    )


def test_core_modules_are_not_modified_by_stress_adapter_import():
    critical = [
        ROOT / "Guidance/reference_point_proposal_demo.py",
        ROOT / "planning/policy_preview.py",
        ROOT / "planning/heterogeneous_candidate_graph.py",
        ROOT / "planning/gat/candidate_selector.py",
        ROOT / "Controller/dmp_rl.py",
        ROOT / "Environment/multi_agent_dmp_env.py",
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in critical}
    spec = importlib.util.spec_from_file_location("stress_reimport", stress.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in critical}
    assert before == after

