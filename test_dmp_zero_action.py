"""Zero-action DMP reachability test.

This script isolates the current DMP controller from SAC, obstacles, sensors,
and reward shaping. It checks whether the nominal DMP structure can drive the
point-mass dynamics from the configured start to the configured goal when the
RL action is always zero.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from Controller.dmp_rl import DMPConfig, SecondOrderDMPController
from Entity.KinematicModel import PartialDynamic


def _load_experiment_config():
    """Load experiment config without requiring the full Gym environment stack."""
    try:
        from experiment_config import EXPERIMENT_CONFIG

        return EXPERIMENT_CONFIG, "import"
    except ModuleNotFoundError:
        config_path = Path(__file__).with_name("experiment_config.py")
        tree = ast.parse(config_path.read_text(encoding="utf-8"))

        values = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "SACExperimentConfig":
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value is not None:
                        try:
                            values[item.target.id] = ast.literal_eval(item.value)
                        except ValueError:
                            continue
                break
        if not values:
            raise RuntimeError("failed to read SACExperimentConfig defaults from experiment_config.py")
        return SimpleNamespace(**values), "static"


def _build_dynamics_config(config) -> dict[str, object]:
    return {
        "velocity_clip": config.velocity_clip,
        "accelerate_clip": config.accelerate_clip,
        "time_step": config.time_step,
    }


def _build_dmp_config(config) -> DMPConfig:
    return DMPConfig(
        dt=config.time_step,
        dims=config.dmp_dims,
        K_alpha=config.k_alpha,
        K_beta=config.k_beta,
        alpha_s=config.alpha_s,
        tau=config.tau,
        forcing_term_max=config.forcing_term_max,
        forcing_term_min=config.forcing_term_min,
        goal_offset_max=config.goal_offset_max,
    )


def _save_diagnostic_plot(
    start: np.ndarray,
    goal: np.ndarray,
    positions: np.ndarray,
    distances: np.ndarray,
    commanded_accelerations: np.ndarray,
    applied_accelerations: np.ndarray,
    output_path: Path,
) -> None:
    steps = np.arange(1, len(distances) + 1)
    commanded_norms = np.linalg.norm(commanded_accelerations, axis=1)
    applied_norms = np.linalg.norm(applied_accelerations, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    ax_traj = fig.add_subplot(2, 2, 1, projection="3d")
    ax_distance = fig.add_subplot(2, 2, 2)
    ax_acc_norm = fig.add_subplot(2, 2, 3)
    ax_acc_axis = fig.add_subplot(2, 2, 4)

    ax_traj.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="#2563eb", linewidth=2.0, label="trajectory")
    ax_traj.scatter(start[0], start[1], start[2], color="#16a34a", s=60, label="start")
    ax_traj.scatter(goal[0], goal[1], goal[2], color="#dc2626", s=60, label="goal")
    ax_traj.set_title("3D trajectory")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_zlabel("z")
    ax_traj.legend(loc="best")

    stacked = np.vstack([positions, start[None, :], goal[None, :]])
    lower = stacked.min(axis=0)
    upper = stacked.max(axis=0)
    span = np.maximum(upper - lower, 0.5)
    center = 0.5 * (lower + upper)
    lower = center - 0.55 * span
    upper = center + 0.55 * span
    ax_traj.set_xlim(lower[0], upper[0])
    ax_traj.set_ylim(lower[1], upper[1])
    ax_traj.set_zlim(lower[2], upper[2])
    if hasattr(ax_traj, "set_box_aspect"):
        ax_traj.set_box_aspect(upper - lower)

    ax_distance.plot(steps, distances, color="#0891b2", linewidth=2.0)
    ax_distance.set_title("Distance to goal")
    ax_distance.set_xlabel("step")
    ax_distance.set_ylabel("distance")
    ax_distance.grid(True, alpha=0.3)

    ax_acc_norm.plot(steps, commanded_norms, color="#ea580c", linewidth=1.8, label="commanded")
    ax_acc_norm.plot(steps, applied_norms, color="#2563eb", linewidth=1.8, label="applied")
    ax_acc_norm.set_title("Acceleration norm")
    ax_acc_norm.set_xlabel("step")
    ax_acc_norm.set_ylabel("norm")
    ax_acc_norm.grid(True, alpha=0.3)
    ax_acc_norm.legend(loc="best")

    axis_labels = ("ax", "ay", "az")
    axis_colors = ("#2563eb", "#16a34a", "#dc2626")
    for axis_index, (label, color) in enumerate(zip(axis_labels, axis_colors)):
        ax_acc_axis.plot(steps, applied_accelerations[:, axis_index], color=color, linewidth=1.8, label=label)
    ax_acc_axis.set_title("Applied acceleration by axis")
    ax_acc_axis.set_xlabel("step")
    ax_acc_axis.set_ylabel("acceleration")
    ax_acc_axis.grid(True, alpha=0.3)
    ax_acc_axis.legend(loc="best")

    fig.suptitle("Zero-action DMP diagnostics", fontsize=16)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_action_sensitivity_cases(config) -> list[tuple[str, np.ndarray]]:
    dims = int(config.dmp_dims)
    forcing_max = float(config.forcing_term_max)
    goal_offset_max = float(config.goal_offset_max)
    cases: list[tuple[str, np.ndarray]] = [("zero", np.zeros(2 * dims, dtype=np.float32))]

    for axis in range(dims):
        positive_forcing = np.zeros(2 * dims, dtype=np.float32)
        negative_forcing = np.zeros(2 * dims, dtype=np.float32)
        positive_forcing[axis] = forcing_max
        negative_forcing[axis] = -forcing_max
        cases.append((f"forcing_+axis{axis}", positive_forcing))
        cases.append((f"forcing_-axis{axis}", negative_forcing))

    for axis in range(dims):
        positive_offset = np.zeros(2 * dims, dtype=np.float32)
        negative_offset = np.zeros(2 * dims, dtype=np.float32)
        positive_offset[dims + axis] = goal_offset_max
        negative_offset[dims + axis] = -goal_offset_max
        cases.append((f"goal_offset_+axis{axis}", positive_offset))
        cases.append((f"goal_offset_-axis{axis}", negative_offset))

    all_positive_forcing = np.zeros(2 * dims, dtype=np.float32)
    all_negative_forcing = np.zeros(2 * dims, dtype=np.float32)
    all_positive_forcing[:dims] = forcing_max
    all_negative_forcing[:dims] = -forcing_max
    cases.append(("forcing_all_positive", all_positive_forcing))
    cases.append(("forcing_all_negative", all_negative_forcing))
    return cases


def _evaluate_action_sensitivity(
    config,
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    cases = _build_action_sensitivity_cases(config)
    accelerate_low = np.asarray(config.accelerate_clip[0], dtype=float)
    accelerate_high = np.asarray(config.accelerate_clip[1], dtype=float)

    for name, action in cases:
        dmp = SecondOrderDMPController(_build_dmp_config(config))
        dmp.reset(position, goal)
        commanded_acceleration, _ = dmp.compute_acceleration(position, velocity, action, sensor_packet=None)
        applied_acceleration = np.clip(commanded_acceleration, accelerate_low, accelerate_high)
        results.append(
            {
                "name": name,
                "commanded": commanded_acceleration,
                "applied": applied_acceleration,
                "is_clipped": not np.allclose(commanded_acceleration, applied_acceleration),
            }
        )
    return results


def _print_action_sensitivity(
    label: str,
    position: np.ndarray,
    velocity: np.ndarray,
    results: list[dict[str, object]],
) -> None:
    applied = np.asarray([item["applied"] for item in results], dtype=float)
    commanded = np.asarray([item["commanded"] for item in results], dtype=float)
    applied_range = applied.max(axis=0) - applied.min(axis=0)
    commanded_range = commanded.max(axis=0) - commanded.min(axis=0)
    unique_applied = np.unique(np.round(applied, decimals=4), axis=0)
    clipped_count = sum(1 for item in results if bool(item["is_clipped"]))

    print(f"Action sensitivity: {label}")
    print(f"state_position: {position}")
    print(f"state_velocity: {velocity}")
    print(f"commanded_acceleration_axis_range: {commanded_range}")
    print(f"applied_acceleration_axis_range: {applied_range}")
    print(f"unique_applied_accelerations: {len(unique_applied)}/{len(results)}")
    print(f"clipped_action_cases: {clipped_count}/{len(results)}")
    print("sample_cases:")
    for item in results[:7]:
        print(
            f"  {item['name']}: "
            f"commanded={np.asarray(item['commanded'])}, "
            f"applied={np.asarray(item['applied'])}"
        )
    print()


def main() -> None:
    config, config_source = _load_experiment_config()
    dynamics = PartialDynamic(_build_dynamics_config(config))
    dmp = SecondOrderDMPController(_build_dmp_config(config))

    start = np.asarray(config.default_start, dtype=float)
    goal = np.asarray(config.default_goal, dtype=float)
    zero_action = np.zeros(2 * config.dmp_dims, dtype=np.float32)

    dynamics.reset({"position": start, "velocity": np.zeros(config.dmp_dims, dtype=float)})
    dmp.reset(start, goal)

    distances: list[float] = []
    positions: list[np.ndarray] = [dynamics.p.copy()]
    velocities: list[np.ndarray] = [dynamics.v.copy()]
    commanded_accelerations: list[np.ndarray] = []
    applied_accelerations: list[np.ndarray] = []
    commanded_norms: list[float] = []
    applied_norms: list[float] = []
    clipped_steps = 0
    reached_step: int | None = None

    print("Zero-action DMP reachability test")
    print(f"config_source: {config_source}")
    print(f"start: {start}")
    print(f"goal: {goal}")
    print(f"goal_tolerance: {config.goal_tolerance}")
    print(f"max_steps: {config.max_steps}")
    print(
        "DMP params: "
        f"K_alpha={config.k_alpha}, K_beta={config.k_beta}, tau={config.tau}, "
        f"forcing_range=[{config.forcing_term_min}, {config.forcing_term_max}], "
        f"goal_offset_max={config.goal_offset_max}"
    )
    print(
        "Dynamics limits: "
        f"accelerate_clip={config.accelerate_clip}, velocity_clip={config.velocity_clip}"
    )
    print()

    for step in range(1, config.max_steps + 1):
        commanded_acceleration, _ = dmp.compute_acceleration(
            dynamics.p,
            dynamics.v,
            zero_action,
            sensor_packet=None,
        )
        applied_acceleration = np.clip(
            commanded_acceleration,
            dynamics.accelerate_min,
            dynamics.accelerate_max,
        )

        if not np.allclose(commanded_acceleration, applied_acceleration):
            clipped_steps += 1

        commanded_norms.append(float(np.linalg.norm(commanded_acceleration)))
        applied_norms.append(float(np.linalg.norm(applied_acceleration)))
        commanded_accelerations.append(commanded_acceleration.copy())
        applied_accelerations.append(applied_acceleration.copy())

        dynamics.step(applied_acceleration)
        positions.append(dynamics.p.copy())
        velocities.append(dynamics.v.copy())
        distance = float(np.linalg.norm(goal - dynamics.p))
        distances.append(distance)

        if reached_step is None and distance <= config.goal_tolerance:
            reached_step = step

    final_distance = distances[-1]
    min_distance = min(distances)
    saturation_ratio = clipped_steps / max(1, config.max_steps)

    print("Result")
    print(f"reached: {reached_step is not None}")
    print(f"reached_step: {reached_step}")
    print(f"final_distance: {final_distance:.6f}")
    print(f"min_distance: {min_distance:.6f}")
    print(f"final_position: {dynamics.p}")
    print(f"final_velocity: {dynamics.v}")
    print()
    print("Acceleration diagnostics")
    print(f"max_commanded_acceleration_norm: {max(commanded_norms):.6f}")
    print(f"mean_commanded_acceleration_norm: {np.mean(commanded_norms):.6f}")
    print(f"max_applied_acceleration_norm: {max(applied_norms):.6f}")
    print(f"mean_applied_acceleration_norm: {np.mean(applied_norms):.6f}")
    print(f"clipped_steps: {clipped_steps}/{config.max_steps}")
    print(f"acceleration_clip_ratio: {saturation_ratio:.3f}")

    sensitivity_indices = {
        "initial": 0,
        "mid_trajectory": min(len(positions) - 1, max(1, config.max_steps // 4)),
        "near_goal": min(len(positions) - 1, max(1, (reached_step or config.max_steps) - 1)),
    }
    print()
    print("Action sensitivity diagnostics")
    for label, index in sensitivity_indices.items():
        sensitivity_results = _evaluate_action_sensitivity(
            config=config,
            position=positions[index],
            velocity=velocities[index],
            goal=goal,
        )
        _print_action_sensitivity(
            label=label,
            position=positions[index],
            velocity=velocities[index],
            results=sensitivity_results,
        )

    output_path = Path("artifacts") / "dmp_zero_action_test" / "zero_action_dmp_diagnostics.png"
    _save_diagnostic_plot(
        start=start,
        goal=goal,
        positions=np.asarray(positions, dtype=float),
        distances=np.asarray(distances, dtype=float),
        commanded_accelerations=np.asarray(commanded_accelerations, dtype=float),
        applied_accelerations=np.asarray(applied_accelerations, dtype=float),
        output_path=output_path,
    )
    print()
    print(f"diagnostic_plot: {output_path}")


if __name__ == "__main__":
    main()
