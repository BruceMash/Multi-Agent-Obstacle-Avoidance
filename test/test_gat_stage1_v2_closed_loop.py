from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "Multi-agent_Algo_lib"
for path in (REPO_ROOT, ALGO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from planning.gat.stage1_training import load_model_checkpoint  # noqa: E402
from scripts import evaluate_gat_stage1_v2_closed_loop as module  # noqa: E402


def _config() -> dict:
    return json.loads(module.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_config_and_seed_manifest() -> None:
    config = _config()
    module._assert_frozen_config(config)
    manifest = module.build_seed_manifest(config)
    assert manifest["status"] == "PASSED"
    assert manifest["FINAL_TEST_SEED_OVERLAP"] == 0
    assert manifest["formal_seeds"] == list(range(30, 50))

    changed = copy.deepcopy(config)
    changed["formal_seeds"][-1] = 50
    with pytest.raises(ValueError, match="30..49"):
        module._assert_frozen_config(changed)


def test_authoritative_context_and_checkpoint_metadata_recover() -> None:
    config = _config()
    context = module.build_context_recovery_manifest(config)
    assert context["status"] == "PASSED"
    assert context["CONTEXT_RECOVERY_VALID"] == "YES"
    assert context["stress_set_status"] == "DIAGNOSTIC_ONLY_AFTER_OBSERVATION"
    assert context["checkpoints"]["v1"]["metadata"]["tensor_count"] == 68
    assert context["checkpoints"]["v2"]["metadata"]["parameter_count"] == 96449


def test_both_checkpoints_strict_load_same_architecture() -> None:
    config = _config()
    stage1 = json.loads(
        (REPO_ROOT / config["stage1_config"]).read_text(encoding="utf-8")
    )
    device = torch.device("cpu")
    v1 = load_model_checkpoint(REPO_ROOT / config["v1_checkpoint"], stage1, device)
    v2 = load_model_checkpoint(REPO_ROOT / config["v2_checkpoint"], stage1, device)
    assert set(v1.state_dict()) == set(v2.state_dict())
    assert sum(value.numel() for value in v1.state_dict().values()) == 96449


class _Graph:
    graph_metadata = {"H": 4, "feature_schema_version": "same"}
    node_types = ["proposal"]
    edge_types = [("proposal", "to", "proposal")]

    def __init__(self, value: float) -> None:
        self.stores = {
            "proposal": {"x": torch.tensor([[value, 2.0]])},
            ("proposal", "to", "proposal"): {
                "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
                "edge_attr": torch.tensor([[3.0]])
            },
        }

    def __getitem__(self, key):
        return self.stores[key]


def test_graph_hash_covers_online_tensor_values() -> None:
    first = module.graph_input_hash([_Graph(1.0)])
    assert first == module.graph_input_hash([_Graph(1.0)])
    assert first != module.graph_input_hash([_Graph(1.1)])


def test_v2_execution_uses_plan_shim_without_candidate_reordering(monkeypatch) -> None:
    v1_plan = {"selection_plan_hash": "v1"}
    v2_plan = {"selection_plan_hash": "v2"}
    shared = {
        "plans": {
            module.METHOD_V1: v1_plan,
            module.METHOD_V2: v2_plan,
            module.legacy.METHOD_GAT: v1_plan,
        },
        "graph_input_hash": "graph",
    }
    captured = {}

    def fake_run_method_episode(**kwargs):
        captured["method"] = kwargs["method"]
        captured["plan"] = kwargs["shared"]["plans"][module.legacy.METHOD_GAT]
        return (
            {
                "collision": False,
                "obstacle_collision": False,
                "inter_agent_collision": False,
            },
            [],
        )

    monkeypatch.setattr(module.legacy, "run_method_episode", fake_run_method_episode)
    episode, _ = module.run_method_episode(
        config={},
        execution_settings={},
        multi_config=None,
        policy=None,
        shared=shared,
        method=module.METHOD_V2,
    )
    assert captured["method"] == module.legacy.METHOD_GAT
    assert captured["plan"] is v2_plan
    assert episode["method"] == module.METHOD_V2
    assert episode["graph_input_hash"] == "graph"


def _episode(method: str, seed: int, success: bool, collision=False, timeout=False):
    return {
        "scenario": "open",
        "seed": seed,
        "method": method,
        "team_success": success,
        "any_collision": collision,
        "obstacle_collision": collision,
        "inter_agent_collision": False,
        "timeout": timeout,
        "candidate_bundle_hash": "candidate",
        "preview_bundle_hash": "preview",
        "graph_input_hash": "graph" if method in {module.METHOD_V1, module.METHOD_V2} else None,
        "completion_step": 10 if success else None,
        "completion_time_s": 1.0 if success else None,
        "team_path_length_m": 3.0,
        "minimum_obstacle_clearance_m": 1.0,
        "minimum_inter_agent_distance_m": 1.0,
    }


def test_pairing_and_mcnemar_count_team_episode_once() -> None:
    episodes = []
    for seed, v1, v2 in ((30, True, True), (31, False, True), (32, True, False), (33, False, False)):
        episodes.extend(
            [
                _episode(module.METHOD_V1, seed, v1, collision=not v1),
                _episode(module.METHOD_V2, seed, v2, timeout=not v2),
                _episode(module.METHOD_FP_SHEP, seed, v1),
            ]
        )
    paired = module.build_pairing(episodes, baseline=module.METHOD_V1)
    assert Counter(row["pair_outcome"] for row in paired) == {
        "BOTH_SUCCESS": 1,
        "V2_ONLY_SUCCESS": 1,
        "V1_ONLY_SUCCESS": 1,
        "BOTH_FAIL": 1,
    }
    test = module._mcnemar_exact([True, False, True, False], [True, True, False, False])
    assert test["discordant_count"] == 2
    assert test["exact_two_sided_p_value"] == 1.0


def test_combined_gain_matches_frozen_rule() -> None:
    config = _config()
    assert module._combined_gain(0.10, -0.20, -0.20, config) == "YES"
    assert module._combined_gain(0.00, 0.10, -0.05, config) == "YES"
    assert module._combined_gain(0.00, 0.10, -0.051, config) == "WEAK"
    assert module._combined_gain(0.0, 0.0, 0.0, config) == "NO"
