"""Visualize the current obstacle scene generated from experiment_config.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from Environment.single_agent_dmp_env import SingleAgentDMPEnv
from experiment_config import EXPERIMENT_CONFIG, SACExperimentConfig


def build_scene_env(config: SACExperimentConfig = EXPERIMENT_CONFIG) -> SingleAgentDMPEnv:
    """Build the same obstacle environment used by the SAC training entry."""
    dynamics_config = config.build_dynamics_config()
    sensor_config = config.build_sensor_config()
    dmp_config = config.build_dmp_config()
    env_config = config.build_env_config()
    fixed_box = config.build_fixed_box()
    start_goal_generator = config.build_start_goal_generator()
    static_obstacle_generator = config.build_static_obstacle_generator(fixed_box)
    dynamic_obstacle_generator = config.build_dynamic_obstacle_generator()
    static_obstacles = config.build_static_obstacles(fixed_box)

    env = SingleAgentDMPEnv(
        dynamics_config=dynamics_config,
        sensor_config=sensor_config,
        dmp_config=dmp_config,
        env_config=env_config,
        start_goal_generator=start_goal_generator,
        static_obstacles=static_obstacles,
        static_obstacle_generator=static_obstacle_generator,
        dynamic_obstacles=[],
        dynamic_obstacle_generator=dynamic_obstacle_generator,
    )
    env._default_start = np.asarray(config.default_start, dtype=float)
    env._default_goal = np.asarray(config.default_goal, dtype=float)
    env.goal = env._default_goal.copy()
    return env


def _plot_sphere_3d(
    ax,
    center: np.ndarray,
    radius: float,
    color: str,
    alpha: float,
    label: str | None = None,
) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 28)
    v = np.linspace(0.0, np.pi, 18)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0.0, shade=False)
    ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=28, label=label)


def _plot_box_3d(
    ax,
    center: np.ndarray,
    half_extents: np.ndarray,
    color: str,
    alpha: float = 0.9,
    label: str | None = None,
    linestyle: str = "-",
    linewidth: float = 1.3,
) -> None:
    x0, y0, z0 = center - half_extents
    x1, y1, z1 = center + half_extents
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for start_index, end_index in edges:
        xs = [vertices[start_index, 0], vertices[end_index, 0]]
        ys = [vertices[start_index, 1], vertices[end_index, 1]]
        zs = [vertices[start_index, 2], vertices[end_index, 2]]
        ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
    ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=28, label=label)


def _set_tight_3d_axes(ax, points: list[np.ndarray], padding: float = 0.2) -> None:
    stacked = np.vstack(points)
    lower = stacked.min(axis=0) - padding
    upper = stacked.max(axis=0) + padding
    span = upper - lower
    min_span = 0.5
    for axis_index in range(3):
        if span[axis_index] < min_span:
            center = 0.5 * (lower[axis_index] + upper[axis_index])
            lower[axis_index] = center - 0.5 * min_span
            upper[axis_index] = center + 0.5 * min_span
            span[axis_index] = min_span

    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(span)


def _format_vec(vector: np.ndarray) -> str:
    return "(" + ", ".join(f"{value:.3f}" for value in vector) + ")"


def _collect_axis_points(
    start: np.ndarray,
    goal: np.ndarray,
    env: SingleAgentDMPEnv,
    include_dynamic_bounds: bool,
) -> list[np.ndarray]:
    points = [start, goal]
    for obstacle in env.static_obstacles + env.dynamic_obstacles:
        center = np.asarray(obstacle.center, dtype=float)
        if hasattr(obstacle, "expanded_half_extents"):
            half_extents = np.asarray(obstacle.expanded_half_extents, dtype=float)
            points.extend([center - half_extents, center + half_extents])
        elif hasattr(obstacle, "effective_radius"):
            radius = float(obstacle.effective_radius)
            points.extend([center - radius, center + radius])
        bounds = getattr(obstacle, "bounds", None)
        if include_dynamic_bounds and bounds is not None:
            lower, upper = bounds
            points.extend([np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)])
    return points


def save_scene_visualization(
    env: SingleAgentDMPEnv,
    start: np.ndarray,
    goal: np.ndarray,
    save_path: str | Path,
    seed: int,
    episode_index: int,
    fixed_start_goal: bool,
    hide_dynamic_bounds: bool,
    view_elev: float = 26.0,
    view_azim: float = -58.0,
    title: str | None = None,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax.scatter([start[0]], [start[1]], [start[2]], color="tab:green", marker="o", s=70, label="start")
    ax.scatter([goal[0]], [goal[1]], [goal[2]], color="tab:orange", marker="*", s=130, label="goal")
    ax.plot(
        [start[0], goal[0]],
        [start[1], goal[1]],
        [start[2], goal[2]],
        color="0.45",
        linewidth=1.0,
        linestyle="--",
        label="start-goal line",
    )

    label_used: set[str] = set()
    for index, obstacle in enumerate(env.static_obstacles):
        center = np.asarray(obstacle.center, dtype=float)
        if hasattr(obstacle, "expanded_half_extents"):
            label = "fixed box" if "fixed box" not in label_used else None
            label_used.add("fixed box")
            _plot_box_3d(
                ax,
                center,
                np.asarray(obstacle.expanded_half_extents, dtype=float),
                color="tab:red",
                alpha=0.9,
                label=label,
            )
        elif hasattr(obstacle, "effective_radius"):
            label = "static sphere" if "static sphere" not in label_used else None
            label_used.add("static sphere")
            _plot_sphere_3d(
                ax,
                center,
                float(obstacle.effective_radius),
                color="tab:blue",
                alpha=0.18,
                label=label,
            )
        else:
            print(f"skip unsupported static obstacle {index}: {type(obstacle).__name__}")

    for index, obstacle in enumerate(env.dynamic_obstacles):
        center = np.asarray(obstacle.center, dtype=float)
        if hasattr(obstacle, "effective_radius"):
            label = "dynamic sphere" if "dynamic sphere" not in label_used else None
            label_used.add("dynamic sphere")
            _plot_sphere_3d(
                ax,
                center,
                float(obstacle.effective_radius),
                color="tab:purple",
                alpha=0.18,
                label=label,
            )

            velocity = np.asarray(obstacle.velocity, dtype=float)
            if float(np.linalg.norm(velocity)) > 1e-8:
                ax.quiver(
                    center[0],
                    center[1],
                    center[2],
                    velocity[0],
                    velocity[1],
                    velocity[2],
                    color="tab:purple",
                    linewidth=1.8,
                    arrow_length_ratio=0.18,
                    label="dynamic velocity" if "dynamic velocity" not in label_used else None,
                )
                label_used.add("dynamic velocity")
        else:
            print(f"skip unsupported dynamic obstacle {index}: {type(obstacle).__name__}")

        bounds = getattr(obstacle, "bounds", None)
        if not hide_dynamic_bounds and bounds is not None:
            lower, upper = bounds
            lower = np.asarray(lower, dtype=float)
            upper = np.asarray(upper, dtype=float)
            label = "dynamic bounds" if "dynamic bounds" not in label_used else None
            label_used.add("dynamic bounds")
            _plot_box_3d(
                ax,
                0.5 * (lower + upper),
                0.5 * (upper - lower),
                color="0.35",
                alpha=0.28,
                label=label,
                linestyle="--",
                linewidth=0.8,
            )

    mode = "fixed start/goal" if fixed_start_goal else "training reset"
    if title is None:
        title = (
            f"Current Scene Generated from experiment_config.py | seed={seed} | "
            f"episode={episode_index} | {mode}"
        )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.grid(True, alpha=0.3)
    _set_tight_3d_axes(
        ax,
        _collect_axis_points(
            start,
            goal,
            env,
            include_dynamic_bounds=not hide_dynamic_bounds,
        ),
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def save_random_3d_side_views(
    seed: int,
    episodes: int,
    fixed_start_goal: bool,
    hide_dynamic_bounds: bool,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Generate multiple randomized 3D perspective images for scene inspection."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    output_dir = Path(output_dir) if output_dir is not None else Path("artifacts") / f"random_side_views_seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed + 2026)
    saved_paths: list[str] = []
    env = build_scene_env()
    try:
        for episode_index in range(episodes):
            reset_seed = seed if episode_index == 0 else None
            reset_options = None
            if fixed_start_goal:
                reset_options = {
                    "start": np.asarray(EXPERIMENT_CONFIG.default_start, dtype=float),
                    "goal": np.asarray(EXPERIMENT_CONFIG.default_goal, dtype=float),
                }

            if reset_options is None:
                env.reset(seed=reset_seed)
            else:
                env.reset(seed=reset_seed, options=reset_options)

            view_elev = float(rng.uniform(18.0, 38.0))
            view_azim = float(rng.uniform(-75.0, -25.0))
            save_path = output_dir / f"side_view_episode_{episode_index:03d}.png"
            title = (
                f"Episode {episode_index} randomized 3D view "
                f"(elev={view_elev:.1f}, azim={view_azim:.1f})"
            )
            saved_paths.append(
                save_scene_visualization(
                    env,
                    env.dynamics.p.copy(),
                    env.goal.copy(),
                    save_path,
                    seed,
                    episode_index,
                    fixed_start_goal,
                    hide_dynamic_bounds,
                    view_elev=view_elev,
                    view_azim=view_azim,
                    title=title,
                )
            )
            print_scene_summary(
                env,
                env.dynamics.p.copy(),
                env.goal.copy(),
                seed,
                episode_index,
                fixed_start_goal,
            )
            print(f"visualization: {saved_paths[-1]}")
    finally:
        env.close()

    combined_path = output_dir / "side_views_combined.png"
    _save_combined_view_grid(saved_paths, combined_path)
    saved_paths.append(str(combined_path))
    print(f"combined visualization: {combined_path}")
    return saved_paths


def _save_combined_view_grid(image_paths: list[str], save_path: str | Path) -> None:
    if not image_paths:
        return

    cols = min(3, len(image_paths))
    rows = int(np.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.4 * rows))
    axes_array = np.asarray(axes, dtype=object).reshape(rows, cols)

    for index, ax in enumerate(axes_array.flat):
        ax.axis("off")
        if index >= len(image_paths):
            continue
        image = plt.imread(image_paths[index])
        ax.imshow(image)
        episode_label = Path(image_paths[index]).stem.replace("side_view_", "3d_view_")
        ax.set_title(episode_label.replace("_", " "), fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_scene_summary(
    env: SingleAgentDMPEnv,
    start: np.ndarray,
    goal: np.ndarray,
    seed: int,
    episode_index: int,
    fixed_start_goal: bool,
) -> None:
    print("scene summary")
    print(f"seed: {seed}")
    print(f"episode: {episode_index}")
    print(f"mode: {'fixed_start_goal' if fixed_start_goal else 'training_reset'}")
    print(f"start: {_format_vec(start)}")
    print(f"goal: {_format_vec(goal)}")
    print(f"static_obstacles: {len(env.static_obstacles)}")
    print(f"dynamic_obstacles: {len(env.dynamic_obstacles)}")

    for index, obstacle in enumerate(env.static_obstacles):
        center = np.asarray(obstacle.center, dtype=float)
        if hasattr(obstacle, "expanded_half_extents"):
            half_extents = np.asarray(obstacle.half_extents, dtype=float)
            expanded_half = np.asarray(obstacle.expanded_half_extents, dtype=float)
            print(
                f"static[{index}] box center={_format_vec(center)} "
                f"half_extents={_format_vec(half_extents)} "
                f"expanded_half_extents={_format_vec(expanded_half)}"
            )
        elif hasattr(obstacle, "effective_radius"):
            print(
                f"static[{index}] sphere center={_format_vec(center)} "
                f"radius={float(obstacle.radius):.3f} "
                f"effective_radius={float(obstacle.effective_radius):.3f}"
            )
        else:
            print(f"static[{index}] unsupported type={type(obstacle).__name__}")

    for index, obstacle in enumerate(env.dynamic_obstacles):
        center = np.asarray(obstacle.center, dtype=float)
        velocity = np.asarray(obstacle.velocity, dtype=float)
        speed = float(np.linalg.norm(velocity))
        print(
            f"dynamic[{index}] sphere center={_format_vec(center)} "
            f"radius={float(obstacle.radius):.3f} "
            f"effective_radius={float(obstacle.effective_radius):.3f} "
            f"velocity={_format_vec(velocity)} speed={speed:.3f}"
        )
        bounds = getattr(obstacle, "bounds", None)
        if bounds is not None:
            lower, upper = bounds
            print(f"dynamic[{index}] bounds lower={_format_vec(lower)} upper={_format_vec(upper)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the current generated obstacle scene.")
    parser.add_argument("--seed", type=int, default=20240517, help="Environment reset seed.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of generated reset scenes to visualize.",
    )
    parser.add_argument(
        "--fixed-start-goal",
        action="store_true",
        help="Use default_start/default_goal and only randomize obstacles.",
    )
    parser.add_argument(
        "--hide-dynamic-bounds",
        action="store_true",
        help="Hide the dashed box showing dynamic obstacle movement bounds.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Multiple episodes append episode index before the suffix.",
    )
    parser.add_argument(
        "--random-side-views",
        action="store_true",
        help="Generate randomized 3D perspective views into artifacts/random_side_views_seed_<seed>.",
    )
    parser.add_argument(
        "--random-side-output-dir",
        type=str,
        default=None,
        help="Output directory for --random-side-views.",
    )
    return parser.parse_args()


def _resolve_output_path(
    output: str | None,
    seed: int,
    episode_index: int,
    episodes: int,
    fixed_start_goal: bool,
) -> Path:
    if output is not None:
        output_path = Path(output)
        if episodes == 1:
            return output_path
        if output_path.suffix:
            return output_path.with_name(
                f"{output_path.stem}_episode_{episode_index:03d}{output_path.suffix}"
            )
        return output_path / f"current_scene_seed_{seed}_episode_{episode_index:03d}.png"

    mode = "fixed_start_goal" if fixed_start_goal else "training"
    if episodes == 1:
        return Path("artifacts") / f"current_scene_{mode}_seed_{seed}.png"
    return Path("artifacts") / f"current_scene_{mode}_seed_{seed}_episode_{episode_index:03d}.png"


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    if args.random_side_views:
        save_random_3d_side_views(
            seed=args.seed,
            episodes=args.episodes,
            fixed_start_goal=args.fixed_start_goal,
            hide_dynamic_bounds=args.hide_dynamic_bounds,
            output_dir=args.random_side_output_dir,
        )
        return

    env = build_scene_env()
    try:
        for episode_index in range(args.episodes):
            reset_seed = args.seed if episode_index == 0 else None
            reset_options = None
            if args.fixed_start_goal:
                reset_options = {
                    "start": np.asarray(EXPERIMENT_CONFIG.default_start, dtype=float),
                    "goal": np.asarray(EXPERIMENT_CONFIG.default_goal, dtype=float),
                }

            if reset_options is None:
                env.reset(seed=reset_seed)
            else:
                env.reset(seed=reset_seed, options=reset_options)

            start = env.dynamics.p.copy()
            goal = env.goal.copy()
            output_path = _resolve_output_path(
                args.output,
                args.seed,
                episode_index,
                args.episodes,
                args.fixed_start_goal,
            )
            saved_path = save_scene_visualization(
                env,
                start,
                goal,
                output_path,
                args.seed,
                episode_index,
                args.fixed_start_goal,
                args.hide_dynamic_bounds,
            )
            print_scene_summary(env, start, goal, args.seed, episode_index, args.fixed_start_goal)
            print(f"visualization: {saved_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
