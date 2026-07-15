from __future__ import annotations

import argparse
import copy
import html
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ALGO_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
for path in (PROJECT_ROOT, ALGO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Entity.static_obstacles import AxisAlignedBoxObstacle
from MASAC.curriculum import CurriculumScenarioGenerator, build_curriculum_stages


def fallback_config():
    """在训练环境依赖未安装时提供与默认训练配置一致的绘图参数。"""
    return SimpleNamespace(
        workspace_bounds=((0.0, 0.0, 0.0), (9.0, 4.5, 2.4)),
        curriculum_phase2_box_counts=(1, 2, 3),
        curriculum_phase2_sphere_counts=(1, 2, 3),
        curriculum_phase3_dynamic_counts=(1, 2, 3),
        curriculum_box_half_extent_range=(0.20, 0.38),
        curriculum_box_height_range=(0.45, 1.10),
        curriculum_aerial_sphere_radius_range=(0.18, 0.30),
        curriculum_dynamic_sphere_radius_range=(0.20, 0.30),
        curriculum_dynamic_speed_range=(0.25, 0.60),
        curriculum_aerial_min_center_height=0.65,
        curriculum_obstacle_safety_margin=0.05,
        curriculum_start_goal_clearance=0.55,
        curriculum_obstacle_separation=0.12,
        curriculum_placement_attempts=1000,
        curriculum_curved_turn_rate=0.45,
        curriculum_wandering_strength=0.8,
        time_step=0.1,
    )


def load_config():
    try:
        from MASAC.config import MASAC_EXPERIMENT_CONFIG

        return MASAC_EXPERIMENT_CONFIG
    except ModuleNotFoundError as error:
        if error.name not in {"gym", "gymnasium"}:
            raise
        return fallback_config()


def fixed_agent_points():
    starts = np.array(
        [[0.8, 0.7, 0.7], [0.8, 2.25, 1.2], [0.8, 3.8, 1.7]], dtype=float
    )
    goals = np.array(
        [[8.2, 0.7, 0.7], [8.2, 2.25, 1.2], [8.2, 3.8, 1.7]], dtype=float
    )
    return starts, goals


def svg_circle(cx, cy, radius, fill, stroke="#0f172a", width=1.2, opacity=1.0):
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>'
    )


def generate_svg(config, output_path: Path, seed: int):
    stages = build_curriculum_stages(
        config.curriculum_phase2_box_counts,
        config.curriculum_phase2_sphere_counts,
        config.curriculum_phase3_dynamic_counts,
    )
    starts, goals = fixed_agent_points()
    width, height = 1500, 820
    panel_width, panel_height = 350, 300
    x_origins = (55, 420, 785, 1150)
    y_origins = (115, 455)
    workspace_lower, workspace_upper = np.asarray(config.workspace_bounds, dtype=float)
    span = workspace_upper - workspace_lower
    plot_width, plot_height = 310, 155
    colors = ("#22d3ee", "#fbbf24", "#a78bfa")
    mode_colors = {"linear": "#60a5fa", "curved": "#c084fc", "wandering": "#34d399"}

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs><filter id=\"shadow\"><feDropShadow dx=\"0\" dy=\"4\" stdDeviation=\"5\" flood-opacity=\"0.22\"/></filter></defs>",
        '<rect width="100%" height="100%" fill="#07111f"/>',
        '<text x="55" y="48" fill="#e2e8f0" font-size="28" font-family="Microsoft YaHei, sans-serif" font-weight="700">MASAC 基于成功率的课程训练场景</text>',
        '<text x="55" y="78" fill="#94a3b8" font-size="15" font-family="Microsoft YaHei, sans-serif">整队成功率达到 0.80 且累计满 100 个 episode 后，场景按箭头方向单向晋级</text>',
    ]

    def project(point, origin_x, origin_y):
        normalized = (np.asarray(point) - workspace_lower) / span
        return origin_x + 18 + normalized[0] * plot_width, origin_y + 62 + (1.0 - normalized[1]) * plot_height

    for index, stage in enumerate(stages):
        row = 0 if index < 4 else 1
        column = index if index < 4 else index - 4
        ox, oy = x_origins[column], y_origins[row]
        generator = CurriculumScenarioGenerator(config, stage)
        static = generator.generate_static(
            starts=starts, goals=goals, seed=seed + index * 17 + 1
        )
        dynamic = generator.generate_dynamic(
            starts=starts,
            goals=goals,
            seed=seed + index * 17 + 2,
            static_obstacles=static,
        )
        title = "Phase 1 · 无障碍物" if stage.phase == 1 else f"Phase {stage.phase} · 密度 L{stage.level}"
        subtitle = (
            f"地面长方体 {stage.ground_box_count}　空中静态球 {stage.aerial_sphere_count}　动态球 {stage.dynamic_sphere_count}"
        )
        svg.extend(
            [
                f'<rect x="{ox}" y="{oy}" width="{panel_width}" height="{panel_height}" rx="14" fill="#0f1d31" stroke="#243b5a" filter="url(#shadow)"/>',
                f'<text x="{ox + 18}" y="{oy + 29}" fill="#f8fafc" font-size="18" font-family="Microsoft YaHei, sans-serif" font-weight="700">{html.escape(title)}</text>',
                f'<text x="{ox + 18}" y="{oy + 51}" fill="#8da3be" font-size="11" font-family="Microsoft YaHei, sans-serif">{html.escape(subtitle)}</text>',
                f'<rect x="{ox + 18}" y="{oy + 62}" width="{plot_width}" height="{plot_height}" rx="5" fill="#081523" stroke="#39516d"/>',
            ]
        )
        for start_index, point in enumerate(starts):
            x, y = project(point, ox, oy)
            svg.append(svg_circle(x, y, 5.5, colors[start_index], width=1.0))
        for goal_index, point in enumerate(goals):
            x, y = project(point, ox, oy)
            svg.append(svg_circle(x, y, 7.0, "none", colors[goal_index], 2.2))

        for obstacle in static:
            x, y = project(obstacle.center, ox, oy)
            if isinstance(obstacle, AxisAlignedBoxObstacle):
                rect_width = obstacle.half_extents[0] * 2.0 / span[0] * plot_width
                rect_height = obstacle.half_extents[1] * 2.0 / span[1] * plot_height
                svg.append(
                    f'<rect x="{x - rect_width / 2:.2f}" y="{y - rect_height / 2:.2f}" width="{rect_width:.2f}" height="{rect_height:.2f}" fill="#f97316" stroke="#fed7aa" stroke-width="1.1"/>'
                )
            else:
                radius = max(5.0, obstacle.radius / span[0] * plot_width)
                altitude_opacity = 0.55 + 0.4 * obstacle.center[2] / span[2]
                svg.append(svg_circle(x, y, radius, "#ef4444", "#fecaca", 1.1, altitude_opacity))

        for dynamic_index, obstacle in enumerate(dynamic):
            simulated = copy.deepcopy(obstacle)
            trajectory = [simulated.center.copy()]
            for _ in range(50):
                trajectory.append(simulated.step(float(config.time_step)))
            points = " ".join(
                f"{px:.2f},{py:.2f}" for px, py in (project(point, ox, oy) for point in trajectory)
            )
            color = mode_colors[obstacle.motion_mode]
            svg.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="5 3" opacity="0.85"/>'
            )
            x, y = project(obstacle.center, ox, oy)
            svg.append(svg_circle(x, y, 6.0, color, "#e0f2fe", 1.3))
            svg.append(
                f'<text x="{x + 8:.2f}" y="{y - 7:.2f}" fill="{color}" font-size="10" font-family="sans-serif">{obstacle.motion_mode}</text>'
            )

        footer_y = oy + 248
        if index < len(stages) - 1:
            svg.append(
                f'<text x="{ox + 18}" y="{footer_y}" fill="#67e8f9" font-size="12" font-family="Microsoft YaHei, sans-serif">SR₁₀₀ ≥ 0.80 → 进入下一难度</text>'
            )
        else:
            svg.append(
                f'<text x="{ox + 18}" y="{footer_y}" fill="#34d399" font-size="12" font-family="Microsoft YaHei, sans-serif">最终训练阶段</text>'
            )

    legend_x, legend_y = x_origins[3], y_origins[1]
    svg.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="350" height="300" rx="14" fill="#0b1728" stroke="#243b5a"/>',
            f'<text x="{legend_x + 22}" y="{legend_y + 36}" fill="#f8fafc" font-size="18" font-family="Microsoft YaHei, sans-serif" font-weight="700">图例与课程机制</text>',
            f'<rect x="{legend_x + 24}" y="{legend_y + 62}" width="20" height="14" fill="#f97316"/><text x="{legend_x + 55}" y="{legend_y + 74}" fill="#cbd5e1" font-size="13" font-family="Microsoft YaHei, sans-serif">地面长方体障碍物</text>',
            f'<circle cx="{legend_x + 34}" cy="{legend_y + 104}" r="8" fill="#ef4444"/><text x="{legend_x + 55}" y="{legend_y + 109}" fill="#cbd5e1" font-size="13" font-family="Microsoft YaHei, sans-serif">空中静态球形障碍物</text>',
            f'<circle cx="{legend_x + 34}" cy="{legend_y + 140}" r="8" fill="#60a5fa"/><text x="{legend_x + 55}" y="{legend_y + 145}" fill="#cbd5e1" font-size="13" font-family="Microsoft YaHei, sans-serif">动态球及预测运动轨迹</text>',
            f'<text x="{legend_x + 24}" y="{legend_y + 184}" fill="#94a3b8" font-size="12" font-family="Microsoft YaHei, sans-serif">Phase 2：静态障碍物逐级增加</text>',
            f'<text x="{legend_x + 24}" y="{legend_y + 209}" fill="#94a3b8" font-size="12" font-family="Microsoft YaHei, sans-serif">Phase 3：保留最高静态密度</text>',
            f'<text x="{legend_x + 24}" y="{legend_y + 234}" fill="#94a3b8" font-size="12" font-family="Microsoft YaHei, sans-serif">并依次加入直线、曲线、随机游走</text>',
            f'<text x="{legend_x + 24}" y="{legend_y + 269}" fill="#64748b" font-size="11" font-family="Microsoft YaHei, sans-serif">○ 目标位置　● 起始位置</text>',
        ]
    )
    svg.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def _cuboid_faces(center, half_extents):
    center = np.asarray(center, dtype=float)
    half = np.asarray(half_extents, dtype=float)
    vertices = np.array(
        [
            center + np.array([sx * half[0], sy * half[1], sz * half[2]])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )
    indices = (
        (0, 1, 3, 2),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 3, 7, 5),
    )
    return [[vertices[index] for index in face] for face in indices]


def _draw_workspace(ax, lower, upper, offset_x):
    lower = np.asarray(lower, dtype=float) + np.array([offset_x, 0.0, 0.0])
    upper = np.asarray(upper, dtype=float) + np.array([offset_x, 0.0, 0.0])
    corners = np.array(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ]
    )
    edges = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )
    for start, end in edges:
        ax.plot(*corners[[start, end]].T, color="#64748b", linewidth=0.75, alpha=0.75)
    ground_x = np.array([[lower[0], upper[0]], [lower[0], upper[0]]])
    ground_y = np.array([[lower[1], lower[1]], [upper[1], upper[1]]])
    ground_z = np.full((2, 2), lower[2])
    ax.plot_surface(ground_x, ground_y, ground_z, color="#dbeafe", alpha=0.14, shade=False)


def _draw_sphere(ax, center, radius, color, alpha=0.82):
    u = np.linspace(0.0, 2.0 * np.pi, 18)
    v = np.linspace(0.0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0.15, shade=True)


def generate_matplotlib_3d(config, png_path: Path, pdf_path: Path, seed: int, dpi: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    stages = build_curriculum_stages(
        config.curriculum_phase2_box_counts,
        config.curriculum_phase2_sphere_counts,
        config.curriculum_phase3_dynamic_counts,
    )
    starts, goals = fixed_agent_points()
    lower, upper = np.asarray(config.workspace_bounds, dtype=float)
    span = upper - lower
    stage_gap = 2.2
    stage_stride = span[0] + stage_gap
    total_x = stage_stride * (len(stages) - 1) + span[0]
    agent_colors = ("#06b6d4", "#f59e0b", "#8b5cf6")
    mode_colors = {"linear": "#2563eb", "curved": "#9333ea", "wandering": "#059669"}

    fig = plt.figure(figsize=(28, 6.2), facecolor="#f8fafc")
    ax = fig.add_axes((0.01, 0.14, 0.98, 0.76), projection="3d", facecolor="#f8fafc")
    for stage_index, stage in enumerate(stages):
        offset_x = stage_index * stage_stride
        offset = np.array([offset_x, 0.0, 0.0])
        generator = CurriculumScenarioGenerator(config, stage)
        static = generator.generate_static(
            starts=starts,
            goals=goals,
            seed=seed + stage_index * 17 + 1,
        )
        dynamic = generator.generate_dynamic(
            starts=starts,
            goals=goals,
            seed=seed + stage_index * 17 + 2,
            static_obstacles=static,
        )
        _draw_workspace(ax, lower, upper, offset_x)

        for agent_index, (start, goal) in enumerate(zip(starts, goals)):
            shifted_start = start + offset
            shifted_goal = goal + offset
            color = agent_colors[agent_index]
            ax.scatter(*shifted_start, s=28, color=color, edgecolors="white", linewidths=0.7, depthshade=True)
            ax.scatter(
                *shifted_goal,
                s=38,
                facecolors="none",
                edgecolors=color,
                linewidths=1.6,
                depthshade=False,
            )

        for obstacle in static:
            shifted_center = np.asarray(obstacle.center, dtype=float) + offset
            if isinstance(obstacle, AxisAlignedBoxObstacle):
                faces = _cuboid_faces(shifted_center, obstacle.half_extents)
                collection = Poly3DCollection(
                    faces,
                    facecolor="#f97316",
                    edgecolor="#9a3412",
                    linewidth=0.45,
                    alpha=0.78,
                )
                ax.add_collection3d(collection)
            else:
                _draw_sphere(ax, shifted_center, obstacle.radius, "#ef4444", alpha=0.76)

        for obstacle in dynamic:
            simulated = copy.deepcopy(obstacle)
            trajectory = [simulated.center.copy()]
            for _ in range(65):
                trajectory.append(simulated.step(float(config.time_step)))
            trajectory = np.asarray(trajectory) + offset[None, :]
            color = mode_colors[obstacle.motion_mode]
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=color,
                linewidth=1.35,
                linestyle="--",
                alpha=0.9,
            )
            shifted_center = np.asarray(obstacle.center, dtype=float) + offset
            _draw_sphere(ax, shifted_center, obstacle.radius, color, alpha=0.9)
        stage_title = "Phase 1" if stage.phase == 1 else f"Phase {stage.phase}-L{stage.level}"
        counts = (
            f"B{stage.ground_box_count} · S{stage.aerial_sphere_count}"
            f" · D{stage.dynamic_sphere_count}"
        )
        ax.text(
            offset_x + 0.5 * span[0],
            upper[1] + 0.38,
            upper[2] + 0.28,
            f"{stage_title}\n{counts}",
            fontsize=8.5,
            fontweight="bold",
            color="#0f172a",
            ha="center",
            linespacing=1.35,
        )

        if stage_index < len(stages) - 1:
            arrow_start = offset_x + span[0] + 0.25
            ax.quiver(
                arrow_start,
                0.5 * span[1],
                upper[2] + 0.42,
                stage_gap - 0.5,
                0.0,
                0.0,
                color="#0891b2",
                linewidth=1.25,
                arrow_length_ratio=0.24,
            )

    ax.set_xlim(lower[0] - 0.6, total_x + 0.6)
    ax.set_ylim(lower[1] - 0.25, upper[1] + 0.85)
    ax.set_zlim(lower[2] - 0.10, upper[2] + 0.65)
    ax.set_box_aspect((total_x, span[1] * 3.8, span[2] * 4.2), zoom=1.0)
    ax.set_proj_type("ortho")
    ax.view_init(elev=18, azim=-72)
    ax.set_axis_off()
    fig.suptitle(
        "MASAC 基于成功率的课程训练场景：SR$_{100}$ ≥ 0.80 后单向晋级",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
        y=0.965,
    )
    legend_handles = [
        Patch(facecolor="#f97316", edgecolor="#9a3412", label="地面长方体障碍物"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ef4444", markeredgecolor="#991b1b", label="空中静态球"),
        Line2D([0], [0], color="#2563eb", linestyle="--", label="直线模式"),
        Line2D([0], [0], color="#9333ea", linestyle="--", label="曲线模式"),
        Line2D([0], [0], color="#059669", linestyle="--", label="随机游走"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#06b6d4", markeredgecolor="white", label="智能体起点"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#06b6d4", label="智能体目标点"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=7,
        frameon=False,
        fontsize=8,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_stage_images(config, output_dir: Path, seed: int, dpi: int) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False
    stages = build_curriculum_stages(
        config.curriculum_phase2_box_counts,
        config.curriculum_phase2_sphere_counts,
        config.curriculum_phase3_dynamic_counts,
    )
    starts, goals = fixed_agent_points()
    lower, upper = np.asarray(config.workspace_bounds, dtype=float)
    span = upper - lower
    agent_colors = ("#06b6d4", "#f59e0b", "#8b5cf6")
    mode_colors = {"linear": "#2563eb", "curved": "#9333ea", "wandering": "#059669"}
    mode_labels = {"linear": "直线模式", "curved": "曲线模式", "wandering": "随机游走"}
    outputs = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for stage_index, stage in enumerate(stages):
        generator = CurriculumScenarioGenerator(config, stage)
        static = generator.generate_static(
            starts=starts,
            goals=goals,
            seed=seed + stage_index * 17 + 1,
        )
        dynamic = generator.generate_dynamic(
            starts=starts,
            goals=goals,
            seed=seed + stage_index * 17 + 2,
            static_obstacles=static,
        )
        fig = plt.figure(figsize=(9.6, 6.8), facecolor="#f8fafc")
        ax = fig.add_subplot(111, projection="3d", facecolor="#f8fafc")
        _draw_workspace(ax, lower, upper, 0.0)

        for agent_index, (start, goal) in enumerate(zip(starts, goals)):
            color = agent_colors[agent_index]
            ax.scatter(
                *start,
                s=54,
                color=color,
                edgecolors="white",
                linewidths=0.9,
                depthshade=True,
            )
            ax.scatter(
                *goal,
                s=70,
                facecolors="none",
                edgecolors=color,
                linewidths=2.0,
                depthshade=False,
            )

        for obstacle in static:
            if isinstance(obstacle, AxisAlignedBoxObstacle):
                collection = Poly3DCollection(
                    _cuboid_faces(obstacle.center, obstacle.half_extents),
                    facecolor="#f97316",
                    edgecolor="#9a3412",
                    linewidth=0.7,
                    alpha=0.82,
                )
                ax.add_collection3d(collection)
            else:
                _draw_sphere(ax, obstacle.center, obstacle.radius, "#ef4444", alpha=0.80)

        present_modes = []
        for obstacle in dynamic:
            simulated = copy.deepcopy(obstacle)
            trajectory = [simulated.center.copy()]
            for _ in range(65):
                trajectory.append(simulated.step(float(config.time_step)))
            trajectory = np.asarray(trajectory)
            color = mode_colors[obstacle.motion_mode]
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=color,
                linewidth=2.0,
                linestyle="--",
                alpha=0.95,
            )
            _draw_sphere(ax, obstacle.center, obstacle.radius, color, alpha=0.92)
            present_modes.append(obstacle.motion_mode)

        stage_title = "Phase 1：无障碍物" if stage.phase == 1 else f"Phase {stage.phase} · Level {stage.level}"
        count_text = (
            f"地面长方体 {stage.ground_box_count}　|　空中静态球 {stage.aerial_sphere_count}"
            f"　|　动态球 {stage.dynamic_sphere_count}"
        )
        ax.set_title(
            f"{stage_title}\n{count_text}",
            fontsize=15,
            fontweight="bold",
            color="#0f172a",
            pad=16,
            linespacing=1.45,
        )
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        ax.set_box_aspect(span)
        ax.view_init(elev=25, azim=-62)
        ax.set_xlabel("X / m", labelpad=8)
        ax.set_ylabel("Y / m", labelpad=8)
        ax.set_zlabel("Z / m", labelpad=6)
        ax.set_xticks((0.0, 3.0, 6.0, 9.0))
        ax.set_yticks((0.0, 1.5, 3.0, 4.5))
        ax.set_zticks((0.0, 0.8, 1.6, 2.4))
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor((0.75, 0.80, 0.87, 0.6))

        legend_handles = [
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="#06b6d4",
                markeredgecolor="white", markersize=8, label="智能体起点"
            ),
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="none",
                markeredgecolor="#06b6d4", markeredgewidth=1.8, markersize=8,
                label="智能体目标点"
            ),
        ]
        if stage.ground_box_count:
            legend_handles.append(
                Patch(facecolor="#f97316", edgecolor="#9a3412", label="地面长方体")
            )
        if stage.aerial_sphere_count:
            legend_handles.append(
                Line2D(
                    [0], [0], marker="o", color="none", markerfacecolor="#ef4444",
                    markeredgecolor="#991b1b", markersize=8, label="空中静态球"
                )
            )
        for mode in dict.fromkeys(present_modes):
            legend_handles.append(
                Line2D(
                    [0], [0], color=mode_colors[mode], linestyle="--",
                    linewidth=2.0, label=mode_labels[mode]
                )
            )
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(4, len(legend_handles)),
            frameon=False,
            fontsize=9,
        )
        fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.17)
        output_path = output_dir / f"{stage.name}.png"
        fig.savefig(output_path, dpi=int(dpi), facecolor=fig.get_facecolor())
        plt.close(fig)
        outputs.append(output_path)
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save each MASAC curriculum stage as an independent Matplotlib 3D image."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/masac_curriculum/3d_stages"),
    )
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = generate_stage_images(load_config(), args.output_dir, args.seed, args.dpi)
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
