"""
MAPPO training entry for the multi-agent DMP-RL environment.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

if "object" not in np.__dict__:
    setattr(np, "object", object)

import MARL as marl

from experiment_config import MAPPO_EXPERIMENT_CONFIG, MAPPOExperimentConfig


def _patch_tensorboardx_writer() -> None:
    try:
        from tensorboardX import record_writer
    except ImportError:
        return

    original_open_file = record_writer.open_file
    if getattr(original_open_file, "_dmp_parent_patch", False):
        return

    def open_file_with_parent(path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return original_open_file(path)

    open_file_with_parent._dmp_parent_patch = True
    record_writer.open_file = open_file_with_parent


def build_env(config: MAPPOExperimentConfig = MAPPO_EXPERIMENT_CONFIG) -> tuple[Any, dict[str, Any]]:
    """Build the MARLlib-compatible multi-agent environment."""
    return marl.make_env(
        config.environment_name,
        config.map_name,
        core_env_kwargs=config.build_core_env_kwargs(),
    )


def build_algo(config: MAPPOExperimentConfig = MAPPO_EXPERIMENT_CONFIG) -> Any:
    """Build MAPPO with the centralized experiment config."""
    return marl.algos.mappo(
        config.hyperparam_source,
        **config.build_algo_args(),
    )


def build_model(
    env: tuple[Any, dict[str, Any]],
    algo: Any,
    config: MAPPOExperimentConfig = MAPPO_EXPERIMENT_CONFIG,
) -> tuple[Any, dict[str, Any]]:
    """Build the centralized-critic model used by MAPPO."""
    return marl.build_model(
        env,
        algo,
        config.build_model_preference(),
    )


def _write_run_config(
    run_dir: Path,
    config: MAPPOExperimentConfig,
    stop_config: dict[str, Any],
    running_params: dict[str, Any],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    payload = {
        "experiment_config": asdict(config),
        "core_env_kwargs": config.build_core_env_kwargs(),
        "algo_args": config.build_algo_args(),
        "model_preference": config.build_model_preference(),
        "stop": stop_config,
        "running_params": running_params,
    }
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(payload, config_file, indent=2, ensure_ascii=False, default=str)
    return config_path


def train(
    training_iteration: int | None = None,
    output_root: str | None = None,
    config: MAPPOExperimentConfig = MAPPO_EXPERIMENT_CONFIG,
) -> dict[str, str]:
    """Run MAPPO training and return artifact paths."""
    output_root = config.output_root if output_root is None else output_root

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / timestamp

    env = build_env(config)
    algo = build_algo(config)
    model = build_model(env, algo, config)
    stop_config = config.build_stop_config(training_iteration=training_iteration)
    running_params = config.build_running_params(local_dir=str(run_dir))
    config_path = _write_run_config(
        run_dir=run_dir,
        config=config,
        stop_config=stop_config,
        running_params=running_params,
    )

    _patch_tensorboardx_writer()
    algo.fit(
        env,
        model,
        stop=stop_config,
        **running_params,
    )

    ray_results_dir = run_dir / f"mappo_{config.model_core_arch}_{config.map_name}"
    return {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "ray_results_dir": str(ray_results_dir),
    }


def main() -> None:
    outputs = train()
    print("MAPPO training finished.")
    print(f"run_dir: {outputs['run_dir']}")
    print(f"config: {outputs['config']}")
    print(f"ray_results_dir: {outputs['ray_results_dir']}")


if __name__ == "__main__":
    main()
