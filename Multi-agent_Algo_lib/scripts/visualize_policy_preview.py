"""Standalone 3-D debug visualization for FP-SHEP candidate previews."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, ALGO_ROOT, SCRIPTS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from Environment.frozen_sac_dmp_execution import freeze_policy  # noqa: E402
from Guidance.reference_point_proposal_demo import (  # noqa: E402
    ProposalConfig,
    propose_reference_points,
)
from experiment_config import EXPERIMENT_CONFIG  # noqa: E402
from planning.policy_preview import (  # noqa: E402
    CandidatePreview,
    PreviewInitialState,
    PreviewLocalContext,
    adapt_candidate_proposals,
    build_preview_inputs_from_env,
    preview_candidates,
)
from runner_sac import build_env as build_single_env  # noqa: E402
from runner_sac import build_model as build_single_model  # noqa: E402
from runner_sac import load_checkpoint  # noqa: E402
from scripts.evaluate_single_policy_aligned_multi_agent import (  # noqa: E402
    build_single_distribution_multi_config,
)
from scripts.evaluate_single_policy_multi_agent import SinglePolicyMultiAgentEnv  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "evaluation" / "frozen_policy_preview.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def plot_policy_preview(
    *,
    initial_state: PreviewInitialState,
    local_context: PreviewLocalContext,
    candidates: list[Any],
    previews: list[CandidatePreview],
    output_path: Path,
) -> Path:
    """Plot current state, task goal, local hits, candidates, and trajectories."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(10.5, 7.5), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(*initial_state.position, s=110, marker="o", color="#111827", label="Current UAV")
    axis.scatter(*initial_state.task_goal, s=150, marker="*", color="#dc2626", label="Task goal")
    known = local_context.visible_surface_points
    if known.shape[0]:
        axis.scatter(known[:, 0], known[:, 1], known[:, 2], s=14, alpha=0.42, color="#6b7280", label="Known LiDAR surfaces")
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, max(len(previews), 1)))
    for index, (candidate, preview) in enumerate(zip(candidates, previews, strict=True)):
        point = np.asarray(getattr(candidate, "point", candidate), dtype=float)
        trajectory = preview.trajectory.positions
        axis.scatter(*point, s=48, marker="^", color=colors[index])
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            linewidth=2.0,
            color=colors[index],
            label=f"C{index + 1}: Δd={preview.task_progress:.2f}, c={preview.min_clearance:.2f}",
        )
    all_points = [initial_state.position[None, :], initial_state.task_goal[None, :]]
    if known.shape[0]:
        all_points.append(known)
    all_points.extend(preview.trajectory.positions for preview in previews)
    combined = np.concatenate(all_points, axis=0)
    center = 0.5 * (combined.min(axis=0) + combined.max(axis=0))
    span = max(float(np.ptp(combined, axis=0).max()), 1.0)
    for setter, value in zip(
        (axis.set_xlim, axis.set_ylim, axis.set_zlim),
        center,
        strict=True,
    ):
        setter(value - 0.55 * span, value + 0.55 * span)
    axis.set_xlabel("X / m")
    axis.set_ylabel("Y / m")
    axis.set_zlabel("Z / m")
    axis.set_title("FP-SHEP closed-loop candidate previews")
    axis.legend(loc="upper left", fontsize=8)
    axis.grid(True, alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def run_visualization(config_path: Path = DEFAULT_CONFIG) -> dict[str, Path]:
    settings = json.loads(Path(config_path).read_text(encoding="utf-8"))
    seed = int(settings["seed"])
    agent_index = int(settings["agent_index"])
    checkpoint = (REPO_ROOT / settings["checkpoint"]).resolve()
    output_dir = (REPO_ROOT / settings["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_single_distribution_multi_config(num_agents=3, max_steps=20)
    env = SinglePolicyMultiAgentEnv(
        **config.build_core_env_kwargs(),
        observation_mode="peer_spheres",
        peer_radius=0.3,
        include_boundaries_in_sensor=False,
        terminate_on_boundary_collision=False,
    )
    reference_env = build_single_env(config=EXPERIMENT_CONFIG, action_guidance_enabled=False)
    try:
        env.reset(seed=seed)
        model = build_single_model(reference_env, config=EXPERIMENT_CONFIG, verbose=0)
        load_checkpoint(model, checkpoint)
        freeze_policy(model)
        initial, local = build_preview_inputs_from_env(env, agent_index)
        proposal_config = ProposalConfig(top_k=int(settings["consumer_top_k"]))
        proposals = propose_reference_points(
            initial.position,
            initial.task_goal,
            initial.velocity,
            env.latest_sensor_packets[agent_index],
            env.sensors[agent_index],
            proposal_config,
            float(env.env_config.goal_tolerance),
        )
        consumer_top_k = (
            int(settings["consumer_top_k"])
            if settings.get("candidate_consumer") == "reference_point_demo_top_k"
            else None
        )
        candidates = adapt_candidate_proposals(proposals, consumer_top_k=consumer_top_k)
        all_preview_start = time.perf_counter_ns()
        previews = preview_candidates(
            initial_state=initial,
            local_context=local,
            candidates=candidates,
            policy=model,
            horizon=int(settings["horizon"]),
            dmp_config=env.dmps[agent_index].config,
            dynamics=env.dynamics[agent_index],
            debug_candidate_index=settings.get("debug_candidate_index"),
        )
        all_preview_time_ms = (time.perf_counter_ns() - all_preview_start) / 1.0e6
        image_path = plot_policy_preview(
            initial_state=initial,
            local_context=local,
            candidates=candidates,
            previews=previews,
            output_path=output_dir / "fp_shep_preview.png",
        )
        report = {
            "seed": seed,
            "checkpoint": checkpoint,
            "candidate_consumer": settings.get("candidate_consumer"),
            "viable_candidate_count": len(proposals),
            "preview_candidate_count": len(candidates),
            "horizon": int(settings["horizon"]),
            "preview_calls": len(previews),
            "all_candidate_preview_time_ms": float(all_preview_time_ms),
            "single_candidate_preview_time_ms": [row.performance.total_ms for row in previews],
            "performance_breakdown_ms": [asdict(row.performance) for row in previews],
            "features": [
                {
                    "task_progress": row.task_progress,
                    "min_clearance": row.min_clearance,
                    "max_execution_deviation": row.max_execution_deviation,
                    "terminal_speed": row.terminal_speed,
                    "metadata": row.metadata,
                }
                for row in previews
            ],
        }
        report_path = output_dir / "fp_shep_preview.json"
        report_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
        return {"image": image_path, "report": report_path}
    finally:
        env.close()
        reference_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    outputs = run_visualization(parse_args().config)
    print(json.dumps(_jsonable(outputs), ensure_ascii=False, indent=2))
