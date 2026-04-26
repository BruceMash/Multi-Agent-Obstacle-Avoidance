"""
单智能体 SAC 测试脚本。

功能：
1. 加载训练得到的 .pt 模型（checkpoint / best_model / final_model 均可）
2. 在单智能体环境执行一个完整回合
3. 输出测试指标，并保存可视化图片（轨迹 + 奖励分量）
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import runner_sac


def _find_latest_model(output_root: str = "artifacts") -> Path:
    """
    在 artifacts 目录下自动查找最近一次训练目录中的 best_model.pt。
    找不到 best_model.pt 时回退到 final_model.pt。
    """
    root = Path(output_root)
    if not root.exists():
        raise FileNotFoundError(f"输出目录不存在: {root}")

    run_dirs = [path for path in root.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"未找到训练目录: {root}")

    latest_dir = sorted(run_dirs)[-1]
    best_path = latest_dir / "best_model.pt"
    final_path = latest_dir / "final_model.pt"

    if best_path.exists():
        return best_path
    if final_path.exists():
        return final_path
    raise FileNotFoundError(f"在 {latest_dir} 中未找到 best_model.pt 或 final_model.pt")


def _extract_obstacles(env) -> list[dict[str, Any]]:
    """提取用于绘图的障碍物几何信息。"""
    obstacles: list[dict[str, Any]] = []

    obstacle_groups = [
        ("static", env.static_obstacles),
        ("dynamic", env.dynamic_obstacles),
    ]
    for obstacle_kind, obstacle_list in obstacle_groups:
        for obstacle in obstacle_list:
            item: dict[str, Any] = {
                "kind": obstacle_kind,
                "center": np.asarray(obstacle.center, dtype=float).copy(),
            }
            if hasattr(obstacle, "expanded_half_extents"):
                item["type"] = "box"
                item["half_extents"] = np.asarray(obstacle.expanded_half_extents, dtype=float).copy()
            elif hasattr(obstacle, "effective_radius"):
                item["type"] = "sphere"
                item["radius"] = float(obstacle.effective_radius)
            else:
                continue
            obstacles.append(item)

    return obstacles


def _summarize_env(env) -> dict[str, Any]:
    """
    汇总测试场景信息。

    测试脚本和训练脚本共用 runner_sac.build_env()，这里单独把场景信息保存下来，
    方便从命令行输出中直接确认本次测试是否带了障碍物。
    """
    return {
        "static_obstacles": int(len(env.static_obstacles)),
        "dynamic_obstacles": int(len(env.dynamic_obstacles)),
        "static_obstacle_types": [type(obstacle).__name__ for obstacle in env.static_obstacles],
        "dynamic_obstacle_types": [type(obstacle).__name__ for obstacle in env.dynamic_obstacles],
    }   # 汇总场景信息


def _plot_circle(ax, cx: float, cy: float, r: float, color: str = "tab:red", alpha: float = 0.2) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 80)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    ax.fill(x, y, color=color, alpha=alpha)
    ax.plot(x, y, color=color, linewidth=1.0)


def _plot_sphere_3d(
    ax,
    center: np.ndarray,
    radius: float,
    color: str = "tab:red",
    alpha: float = 0.15,
    label: str | None = None,
) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 16)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0.0, shade=False)
    ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=24, label=label)


def _plot_box_3d(
    ax,
    center: np.ndarray,
    half_extents: np.ndarray,
    color: str = "tab:red",
    alpha: float = 0.8,
    label: str | None = None,
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
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for s, e in edges:
        xs = [vertices[s, 0], vertices[e, 0]]
        ys = [vertices[s, 1], vertices[e, 1]]
        zs = [vertices[s, 2], vertices[e, 2]]
        ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=1.2)
    ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=24, label=label)


def _set_equal_3d_axes(ax, points: list[np.ndarray], padding: float = 0.6) -> None:
    """让三维图的 X/Y/Z 比例一致，避免球形障碍物显示成椭球。"""
    stacked = np.vstack(points)
    lower = stacked.min(axis=0) - padding
    upper = stacked.max(axis=0) + padding
    center = 0.5 * (lower + upper)
    radius = 0.5 * float(np.max(upper - lower))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _save_visualization(result: dict[str, Any], save_path: str | Path) -> str:
    """
    保存测试可视化：
    1. 三维轨迹图（包含起点、终点、障碍物）
    2. 奖励分量曲线图
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory = np.asarray(result["trajectory"], dtype=float)
    start = np.asarray(result["start"], dtype=float)
    goal = np.asarray(result["goal"], dtype=float)
    obstacles = result["obstacles"]

    reward_total_series = np.asarray(result["reward_total_series"], dtype=float)
    reward_step_series = np.asarray(result["reward_step_series"], dtype=float)
    reward_obstacle_penalty_series = np.asarray(result["reward_obstacle_penalty_series"], dtype=float)
    reward_step_penalty_series = np.asarray(result["reward_step_penalty_series"], dtype=float)

    fig = plt.figure(figsize=(14, 6))
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_rw = fig.add_subplot(1, 2, 2)

    # 子图1：3D 轨迹
    ax_3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="tab:blue", linewidth=2.0, label="trajectory")
    ax_3d.scatter([start[0]], [start[1]], [start[2]], color="tab:green", marker="o", s=60, label="start")
    ax_3d.scatter([goal[0]], [goal[1]], [goal[2]], color="tab:orange", marker="*", s=120, label="goal")

    axis_points = [trajectory.min(axis=0), trajectory.max(axis=0), start, goal]
    obstacle_label_used: set[str] = set()
    for obstacle in obstacles:
        center = obstacle["center"]
        color = "tab:red" if obstacle["kind"] == "static" else "tab:purple"
        if obstacle["type"] == "sphere":
            radius = obstacle["radius"]
            label_key = f"{obstacle['kind']} sphere"
            label = None if label_key in obstacle_label_used else label_key
            obstacle_label_used.add(label_key)
            _plot_sphere_3d(ax_3d, center, radius, color=color, label=label)
            axis_points.extend([center - radius, center + radius])
        elif obstacle["type"] == "box":
            half_extents = obstacle["half_extents"]
            label_key = f"{obstacle['kind']} box"
            label = None if label_key in obstacle_label_used else label_key
            obstacle_label_used.add(label_key)
            _plot_box_3d(ax_3d, center, half_extents, color=color, label=label)
            axis_points.extend([center - half_extents, center + half_extents])

    ax_3d.set_title("Trajectory (3D)")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.grid(True, alpha=0.3)
    _set_equal_3d_axes(ax_3d, axis_points)
    ax_3d.legend(loc="best")

    # 子图2：奖励分量
    steps = np.arange(len(reward_total_series))
    ax_rw.plot(steps, reward_total_series, label="reward_total", linewidth=1.8)
    ax_rw.plot(steps, reward_step_series, label="reward_step", linewidth=1.2)
    ax_rw.plot(steps, reward_obstacle_penalty_series, label="obstacle_penalty", linewidth=1.2)
    ax_rw.plot(steps, reward_step_penalty_series, label="step_penalty", linewidth=1.2)
    ax_rw.set_title("Reward Components")
    ax_rw.set_xlabel("Step")
    ax_rw.set_ylabel("Value")
    ax_rw.grid(True, alpha=0.3)
    ax_rw.legend(loc="best")

    fig.suptitle(
        f"Model Test | success={result['success']} | collision={result['collision']} | total_reward={result['total_reward']:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def test_single_episode(
    model_path: str | Path | None = None,
    deterministic: bool = True,
    output_root: str = "artifacts",
) -> dict[str, Any]:
    """加载模型并执行单回合测试。"""
    checkpoint_path = Path(model_path) if model_path is not None else _find_latest_model(output_root=output_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    env = runner_sac.build_env()
    env_summary = _summarize_env(env)
    model = runner_sac.build_model(env, buffer_size=1_000, verbose=0)
    model = runner_sac.load_checkpoint(model, checkpoint_path)

    start = np.array([0.0, 0.0, 0.0], dtype=float)
    goal = np.array([8.0, 0.0, 0.0], dtype=float)
    observation, info = env.reset(options={"start": start, "goal": goal})

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False
    trajectory = [env.dynamics.p.copy()]

    reward_total_series: list[float] = []
    reward_step_series: list[float] = []
    reward_obstacle_penalty_series: list[float] = []
    reward_step_penalty_series: list[float] = []

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        step_count += 1
        trajectory.append(env.dynamics.p.copy())

        reward_total_series.append(float(reward))
        reward_step_series.append(float(info.get("reward_step_reward", 0.0)))
        reward_obstacle_penalty_series.append(float(info.get("reward_obstacle_potential_penalty", 0.0)))
        reward_step_penalty_series.append(float(info.get("reward_step_penalty", 0.0)))

    result: dict[str, Any] = {
        "model_path": str(checkpoint_path),
        "steps": int(step_count),
        "total_reward": float(total_reward),
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "truncated": bool(info.get("truncated", False)),
        "distance_to_goal": float(info.get("distance_to_goal", np.nan)),
        "trajectory_points": int(len(trajectory)),
        "trajectory": np.asarray(trajectory, dtype=float),
        "start": start,
        "goal": goal,
        "obstacles": _extract_obstacles(env),
        "env_summary": env_summary,
        "reward_total_series": reward_total_series,
        "reward_step_series": reward_step_series,
        "reward_obstacle_penalty_series": reward_obstacle_penalty_series,
        "reward_step_penalty_series": reward_step_penalty_series,
    }

    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="单智能体 SAC 模型测试")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型路径（.pt）。不传则自动使用 artifacts 下最近一次训练的 best_model.pt / final_model.pt",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="使用随机动作采样测试（默认关闭，即确定性测试）",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="artifacts",
        help="自动查找最新模型时使用的根目录",
    )
    parser.add_argument(
        "--vis-path",
        type=str,
        default=None,
        help="可视化输出图片路径（png）。不传则自动保存到模型目录下 test_vis_时间戳.png",
    )
    args = parser.parse_args()

    result = test_single_episode(
        model_path=args.model,
        deterministic=not args.stochastic,
        output_root=args.output_root,
    )

    if args.vis_path is not None:
        vis_path = Path(args.vis_path)
    else:
        model_parent = Path(result["model_path"]).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_path = model_parent / f"test_vis_{timestamp}.png"

    saved_vis = _save_visualization(result, vis_path)

    print("test finished")
    print(f"model_path: {result['model_path']}")
    print(f"steps: {result['steps']}")
    print(f"total_reward: {result['total_reward']:.3f}")
    print(f"success: {result['success']}")
    print(f"collision: {result['collision']}")
    print(f"truncated: {result['truncated']}")
    print(f"distance_to_goal: {result['distance_to_goal']:.3f}")
    print(f"trajectory_points: {result['trajectory_points']}")
    print(f"static_obstacles: {result['env_summary']['static_obstacles']}")
    print(f"dynamic_obstacles: {result['env_summary']['dynamic_obstacles']}")
    print(f"visualization: {saved_vis}")


if __name__ == "__main__":
    main()
