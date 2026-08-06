"""Generate scientific figures for the frozen-policy corridor-reservation study.

The visualizer consumes only persisted artifacts.  It never replays the policy and
therefore cannot change the frozen actor, DMP state, environment state, or random
number generators.  Every available figure is written as both PNG and PDF.

Expected artifact files are ``config.json``, ``summary.csv``, ``episodes.csv``,
``agents.csv``, ``corridor_events.csv``, ``waypoint_switches.csv``,
``rollout_predictions.csv``, and ``trajectories.csv``.  Each input is optional:
figures whose source data are unavailable are skipped with an explicit message.
This is important when a staged Go/No-Go run stops before E3 or E4.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = (
    REPO_ROOT / "artifacts" / "frozen_policy_corridor_reservation_minimal_20260805"
)

TABLE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "summary": ("summary.csv",),
    "episodes": ("episodes.csv",),
    "agents": ("agents.csv",),
    "events": ("corridor_events.csv", "events.csv"),
    "switches": ("waypoint_switches.csv", "switches.csv"),
    "predictions": ("rollout_predictions.csv", "predictions.csv"),
    "trajectories": (
        "trajectories.csv",
        "trajectory.csv",
        "trajectory_timeseries.csv",
        "time_series.csv",
        "timeseries.csv",
        "step_log.csv",
        "steps.csv",
    ),
}

GROUP_ORDER = ("E0", "E1", "E2", "E3", "E4")
GROUP_COLORS = {
    "E0": "#456990",
    "E1": "#2A9D8F",
    "E2": "#E9A23B",
    "E3": "#E76F51",
    "E4": "#7B2CBF",
}
AGENT_COLORS = (
    "#1676B8",
    "#E45756",
    "#2A9D8F",
    "#8F5DA2",
    "#F28E2B",
    "#76B7B2",
)
WAYPOINT_ORDER = (
    "PROGRESS",
    "DECELERATION",
    "HOLD",
    "ENTRY",
    "EXIT",
    "TERMINAL",
)
RISK_METRICS = (
    (
        "team_success_rate",
        ("team_success_rate", "team_success", "success_rate"),
        "团队成功率 / Team success",
        True,
    ),
    (
        "inter_agent_collision_rate",
        (
            "inter_agent_collision_rate",
            "any_inter_agent_collision",
            "inter_agent_collision",
        ),
        "机间碰撞率 / Inter-agent collision",
        True,
    ),
    (
        "obstacle_collision_rate",
        ("obstacle_collision_rate", "any_obstacle_collision", "obstacle_collision"),
        "障碍物碰撞率 / Obstacle collision",
        True,
    ),
    (
        "timeout_rate",
        ("timeout_rate", "timeout"),
        "超时率 / Timeout",
        True,
    ),
    (
        "out_of_bounds_rate",
        (
            "out_of_bounds_rate",
            "any_out_of_bounds",
            "out_of_bounds",
            "boundary_excursion_rate",
        ),
        "越界率 / Out of bounds",
        True,
    ),
    (
        "occupancy_overlap_time",
        (
            "occupancy_overlap_time_mean",
            "mean_occupancy_overlap_time",
            "occupancy_overlap_time",
        ),
        "占用重叠时间 / Occupancy overlap (s)",
        False,
    ),
)


@dataclass(frozen=True, order=True)
class EpisodeKey:
    """Stable identifier used across artifact tables."""

    group: str
    seed: int
    episode_index: int = 0
    route_agent_id: int = -1

    @property
    def slug(self) -> str:
        route = "" if self.route_agent_id < 0 else f"_route{self.route_agent_id}"
        return _slugify(
            f"{self.group}_seed{self.seed}_episode{self.episode_index}{route}"
        )


@dataclass
class ArtifactData:
    artifact_dir: Path
    config: dict[str, Any] = field(default_factory=dict)
    summary: list[dict[str, str]] = field(default_factory=list)
    episodes: list[dict[str, str]] = field(default_factory=list)
    agents: list[dict[str, str]] = field(default_factory=list)
    events: list[dict[str, str]] = field(default_factory=list)
    switches: list[dict[str, str]] = field(default_factory=list)
    predictions: list[dict[str, str]] = field(default_factory=list)
    trajectories: list[dict[str, str]] = field(default_factory=list)
    loaded_paths: dict[str, Path] = field(default_factory=dict)


def configure_matplotlib() -> None:
    """Apply a restrained scientific-report style."""

    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.facecolor": "#F5F7FA",
            "savefig.facecolor": "#F5F7FA",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#667085",
            "axes.labelcolor": "#344054",
            "axes.titlecolor": "#1D2939",
            "xtick.color": "#475467",
            "ytick.color": "#475467",
            "grid.color": "#D0D5DD",
            "grid.linestyle": "--",
            "grid.alpha": 0.45,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return slug.strip("._-") or "unknown"


def _norm_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _row_value(row: Mapping[str, Any], *names: str, default: Any = "") -> Any:
    lookup = {_norm_key(key): value for key, value in row.items()}
    for name in names:
        key = _norm_key(name)
        if key in lookup and lookup[key] not in (None, ""):
            return lookup[key]
    return default


def _to_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else default
    text = str(value).strip().replace("%", "")
    if not text:
        return default
    try:
        number = float(text)
    except ValueError:
        return default
    if "%" in str(value):
        number /= 100.0
    return number if math.isfinite(number) else default


def _to_int(value: Any, default: int = 0) -> int:
    number = _to_float(value)
    return int(number) if math.isfinite(number) else default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on", "success", "succeeded"}:
        return True
    if text in {"false", "0", "no", "n", "off", "none", "nan", ""}:
        return False
    return default


def _group_name(row: Mapping[str, Any]) -> str:
    return str(
        _row_value(row, "group", "experiment_group", "controller", "method", default="")
    ).strip()


def _canonical_group(group: str) -> str:
    match = re.match(r"\s*(E[0-4])(?:\b|[_:-])", str(group), flags=re.IGNORECASE)
    return match.group(1).upper() if match else str(group).strip()


def _group_sort_key(group: str) -> tuple[int, str]:
    canonical = _canonical_group(group)
    try:
        return GROUP_ORDER.index(canonical), str(group)
    except ValueError:
        return len(GROUP_ORDER), str(group)


def _agent_id(row: Mapping[str, Any]) -> int:
    return _to_int(
        _row_value(
            row,
            "environment_agent_id",
            "agent_id",
            "uav_id",
            "agent",
            "route_agent_id",
            default=0,
        )
    )


def _row_time(row: Mapping[str, Any], dt: float) -> float:
    value = _to_float(_row_value(row, "time", "timestamp", "elapsed_time"))
    if math.isfinite(value):
        return value
    return dt * _to_float(_row_value(row, "step", "timestep", "frame"), default=0.0)


def _episode_key(row: Mapping[str, Any]) -> EpisodeKey:
    return EpisodeKey(
        group=_group_name(row),
        seed=_to_int(_row_value(row, "seed", "random_seed", default=0)),
        episode_index=_to_int(
            _row_value(row, "episode_index", "episode_id", "episode", default=0)
        ),
        route_agent_id=_to_int(_row_value(row, "route_agent_id", default=-1), default=-1),
    )


def _same_episode(row: Mapping[str, Any], key: EpisodeKey) -> bool:
    if _group_name(row) != key.group:
        return False
    if _to_int(_row_value(row, "seed", "random_seed", default=0)) != key.seed:
        return False
    episode_value = _row_value(row, "episode_index", "episode_id", "episode", default="")
    if episode_value not in (None, "") and _to_int(episode_value) != key.episode_index:
        return False
    route_value = _row_value(row, "route_agent_id", default="")
    if (
        key.route_agent_id >= 0
        and route_value not in (None, "")
        and _to_int(route_value, default=-1) != key.route_agent_id
    ):
        return False
    return True


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_artifacts(artifact_dir: Path) -> ArtifactData:
    """Load all present inputs while treating absent staged outputs as normal."""

    data = ArtifactData(artifact_dir=artifact_dir)
    config_candidates = ("config.json", "config_snapshot.json")
    for filename in config_candidates:
        path = artifact_dir / filename
        if path.is_file():
            data.config = json.loads(path.read_text(encoding="utf-8-sig"))
            data.loaded_paths["config"] = path
            break

    for table_name, filenames in TABLE_CANDIDATES.items():
        for filename in filenames:
            path = artifact_dir / filename
            if path.is_file():
                setattr(data, table_name, _read_csv(path))
                data.loaded_paths[table_name] = path
                break
    return data


def _recursive_find(value: Any, target_names: Iterable[str]) -> Any:
    targets = {_norm_key(name) for name in target_names}
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _norm_key(key) in targets:
                return child
        for child in value.values():
            found = _recursive_find(child, targets)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found = _recursive_find(child, targets)
            if found is not None:
                return found
    return None


def _metadata(config: Mapping[str, Any]) -> Mapping[str, Any]:
    found = _recursive_find(config, ("resolved_corridor_metadata", "corridor_metadata"))
    return found if isinstance(found, Mapping) else {}


def _config_dt(config: Mapping[str, Any]) -> float:
    value = _recursive_find(config, ("dt", "time_step", "timestep_seconds"))
    dt = _to_float(value, default=0.1)
    return dt if dt > 0.0 else 0.1


def _as_xyz(value: Any) -> np.ndarray | None:
    if isinstance(value, Mapping):
        nested = _row_value(
            value,
            "position",
            "point",
            "center",
            "coordinate",
            "coordinates",
            default=None,
        )
        if nested is not None:
            return _as_xyz(nested)
        x = _to_float(_row_value(value, "x"))
        y = _to_float(_row_value(value, "y"))
        z = _to_float(_row_value(value, "z", default=0.0), default=0.0)
        if math.isfinite(x) and math.isfinite(y):
            return np.asarray([x, y, z], dtype=float)
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size >= 2 and np.isfinite(array[:2]).all():
            return np.asarray(
                [array[0], array[1], array[2] if array.size >= 3 else 0.0], dtype=float
            )
    return None


def _bounds_xy(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if isinstance(value, Mapping):
        pairs = (
            ("lower", "upper"),
            ("minimum", "maximum"),
            ("min_corner", "max_corner"),
            ("lower_bound", "upper_bound"),
        )
        for lower_name, upper_name in pairs:
            lower = _as_xyz(_row_value(value, lower_name, default=None))
            upper = _as_xyz(_row_value(value, upper_name, default=None))
            if lower is not None and upper is not None:
                return np.minimum(lower, upper), np.maximum(lower, upper)
        center = _as_xyz(_row_value(value, "center", default=None))
        half = _as_xyz(
            _row_value(value, "half_extents", "half_extent", "half_size", default=None)
        )
        if center is not None and half is not None:
            return center - np.abs(half), center + np.abs(half)
        xmin = _to_float(_row_value(value, "x_min", "xmin"))
        xmax = _to_float(_row_value(value, "x_max", "xmax"))
        ymin = _to_float(_row_value(value, "y_min", "ymin"))
        ymax = _to_float(_row_value(value, "y_max", "ymax"))
        if all(math.isfinite(v) for v in (xmin, xmax, ymin, ymax)):
            return np.asarray([xmin, ymin, 0.0]), np.asarray([xmax, ymax, 0.0])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
        if array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 2:
            lower = np.asarray([array[0, 0], array[0, 1], 0.0])
            upper = np.asarray([array[1, 0], array[1, 1], 0.0])
            return np.minimum(lower, upper), np.maximum(lower, upper)
        if array.ndim == 1 and array.size >= 4:
            lower = np.asarray([array[0], array[1], 0.0])
            upper = np.asarray([array[2], array[3], 0.0])
            return np.minimum(lower, upper), np.maximum(lower, upper)
    return None


def _save_figure(fig: Any, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for suffix, kwargs in (
        (".png", {"dpi": 240}),
        (".pdf", {}),
    ):
        output_path = output_base.with_suffix(suffix)
        fig.savefig(output_path, bbox_inches="tight", **kwargs)
        saved.append(output_path)
    plt.close(fig)
    return saved


def _deduplicate_legend(ax: Any, **kwargs: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_") and label not in unique:
            unique[label] = handle
    if unique:
        ax.legend(unique.values(), unique.keys(), **kwargs)


def _row_xyz(row: Mapping[str, Any], prefix: str = "") -> np.ndarray | None:
    names = (
        (f"{prefix}x", f"{prefix}y", f"{prefix}z"),
        (f"{prefix}position_x", f"{prefix}position_y", f"{prefix}position_z"),
    )
    for x_name, y_name, z_name in names:
        x = _to_float(_row_value(row, x_name))
        y = _to_float(_row_value(row, y_name))
        z = _to_float(_row_value(row, z_name, default=0.0), default=0.0)
        if math.isfinite(x) and math.isfinite(y):
            return np.asarray([x, y, z], dtype=float)
    return None


def _trajectory_episode_keys(rows: Sequence[Mapping[str, Any]]) -> list[EpisodeKey]:
    return sorted({_episode_key(row) for row in rows}, key=lambda key: (_group_sort_key(key.group), key))


def _episode_rows(
    rows: Sequence[Mapping[str, Any]], key: EpisodeKey
) -> list[Mapping[str, Any]]:
    return [row for row in rows if _same_episode(row, key)]


def _gate_xy(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Resolve a gate to a 2-D line segment."""

    if isinstance(value, Mapping):
        start = _as_xyz(_row_value(value, "start", "point_a", "lower", default=None))
        end = _as_xyz(_row_value(value, "end", "point_b", "upper", default=None))
        if start is not None and end is not None:
            return start[:2], end[:2]
        x = _to_float(_row_value(value, "x", "plane_x", "value"))
        ymin = _to_float(_row_value(value, "y_min", "ymin", default=-0.6), default=-0.6)
        ymax = _to_float(_row_value(value, "y_max", "ymax", default=0.6), default=0.6)
        if math.isfinite(x):
            return np.asarray([x, ymin]), np.asarray([x, ymax])
    point = _as_xyz(value)
    if point is not None:
        return np.asarray([point[0], point[1] - 0.6]), np.asarray([point[0], point[1] + 0.6])
    scalar = _to_float(value)
    if math.isfinite(scalar):
        return np.asarray([scalar, -0.6]), np.asarray([scalar, 0.6])
    return None


def _draw_corridor_geometry(ax: Any, metadata: Mapping[str, Any]) -> None:
    conflict = _row_value(metadata, "conflict_region", default=None)
    bounds = _bounds_xy(conflict)
    if bounds is not None:
        lower, upper = bounds
        ax.add_patch(
            Rectangle(
                (lower[0], lower[1]),
                upper[0] - lower[0],
                upper[1] - lower[1],
                facecolor="#FEC84B",
                edgecolor="#B54708",
                linewidth=1.2,
                alpha=0.14,
                hatch="//",
                label="冲突区 / conflict region",
                zorder=0,
            )
        )

    gate_specs = (
        ("entry_gate_a", "#12B76A", "入口 / entry"),
        ("entry_gate_b", "#12B76A", "_entry"),
        ("exit_gate_a", "#7F56D9", "出口 / exit"),
        ("exit_gate_b", "#7F56D9", "_exit"),
    )
    for field_name, color, label in gate_specs:
        segment = _gate_xy(_row_value(metadata, field_name, default=None))
        if segment is not None:
            start, end = segment
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linestyle="--",
                linewidth=1.6,
                label=label,
                zorder=2,
            )

    for index, field_name in enumerate(("hold_zone_a", "hold_zone_b")):
        value = _row_value(metadata, field_name, default=None)
        bounds = _bounds_xy(value)
        if bounds is not None:
            lower, upper = bounds
            ax.add_patch(
                Rectangle(
                    (lower[0], lower[1]),
                    upper[0] - lower[0],
                    upper[1] - lower[1],
                    facecolor="#84CAFF",
                    edgecolor="#1570EF",
                    alpha=0.18,
                    linewidth=1.1,
                    label="等待区 / hold zone" if index == 0 else "_hold",
                    zorder=1,
                )
            )
            continue
        center = _as_xyz(value)
        if center is not None:
            ax.add_patch(
                Circle(
                    (center[0], center[1]),
                    radius=0.18,
                    facecolor="#84CAFF",
                    edgecolor="#1570EF",
                    alpha=0.65,
                    label="等待点 / hold" if index == 0 else "_hold",
                    zorder=3,
                )
            )


def _switch_position(
    row: Mapping[str, Any], trajectory_rows: Sequence[Mapping[str, Any]], dt: float
) -> np.ndarray | None:
    for prefix in ("new_waypoint_", "new_", "waypoint_", ""):
        point = _row_xyz(row, prefix)
        if point is not None:
            return point
    agent = _agent_id(row)
    switch_time = _row_time(row, dt)
    candidates = [item for item in trajectory_rows if _agent_id(item) == agent]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda item: abs(_row_time(item, dt) - switch_time))
    return _row_xyz(nearest)


def plot_top_down_trajectory(
    *,
    key: EpisodeKey,
    trajectory_rows: Sequence[Mapping[str, Any]],
    switch_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    dt: float,
    output_base: Path,
    title_prefix: str = "",
) -> list[Path]:
    """Plot XY trajectories, geometry, waypoint switches, and failure locations."""

    valid_rows = [row for row in trajectory_rows if _row_xyz(row) is not None]
    if not valid_rows:
        return []

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    _draw_corridor_geometry(ax, metadata)
    agents = sorted({_agent_id(row) for row in valid_rows})
    for color_index, agent in enumerate(agents):
        rows = sorted(
            (row for row in valid_rows if _agent_id(row) == agent),
            key=lambda row: _row_time(row, dt),
        )
        points = np.stack([_row_xyz(row) for row in rows], axis=0)
        color = AGENT_COLORS[color_index % len(AGENT_COLORS)]
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=2.0,
            label=f"UAV {agent}",
            zorder=4,
        )
        ax.scatter(
            points[0, 0],
            points[0, 1],
            marker="o",
            s=54,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=6,
        )
        ax.scatter(
            points[-1, 0],
            points[-1, 1],
            marker="*",
            s=105,
            facecolor=color,
            edgecolor="#344054",
            linewidth=0.6,
            zorder=6,
        )

    for index, row in enumerate(switch_rows):
        point = _switch_position(row, valid_rows, dt)
        if point is None:
            continue
        ax.scatter(
            point[0],
            point[1],
            marker="s",
            s=33,
            facecolor="none",
            edgecolor="#101828",
            linewidth=0.9,
            label="路径点切换 / switch" if index == 0 else "_switch",
            zorder=7,
        )

    collision_rows = [
        row
        for row in valid_rows
        if any(
            _to_bool(_row_value(row, name))
            for name in (
                "inter_agent_collision",
                "obstacle_collision",
                "out_of_bounds",
                "collision",
            )
        )
    ]
    for index, row in enumerate(collision_rows):
        point = _row_xyz(row)
        assert point is not None
        ax.scatter(
            point[0],
            point[1],
            marker="X",
            s=92,
            facecolor="#D92D20",
            edgecolor="white",
            linewidth=0.7,
            label="碰撞/越界 / failure" if index == 0 else "_failure",
            zorder=9,
        )

    conflict_events = []
    for row in event_rows:
        label = str(
            _row_value(row, "event_type", "event", "transition", "status", default="")
        ).lower()
        point = _row_xyz(row)
        if "conflict" in label and point is not None:
            conflict_events.append(point)
    if conflict_events:
        points = np.stack(conflict_events)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            marker="P",
            s=60,
            facecolor="#F79009",
            edgecolor="#7A2E0E",
            label="冲突事件 / conflict",
            zorder=8,
        )

    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    heading = f"{title_prefix} " if title_prefix else ""
    ax.set_title(
        f"{heading}{key.group}: seed={key.seed} 顶视轨迹 / Top-down trajectory"
    )
    ax.grid(True)
    ax.set_aspect("equal", adjustable="datalim")
    _deduplicate_legend(ax, loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    return _save_figure(fig, output_base)


def _centerline_projection(
    points: np.ndarray, metadata: Mapping[str, Any]
) -> np.ndarray:
    centerline = _row_value(metadata, "centerline", default=None)
    origin = np.zeros(3, dtype=float)
    direction: np.ndarray | None = None
    if isinstance(centerline, Mapping):
        origin_value = _as_xyz(
            _row_value(centerline, "origin", "start", "point", default=None)
        )
        end_value = _as_xyz(_row_value(centerline, "end", default=None))
        direction_value = _as_xyz(_row_value(centerline, "direction", default=None))
        if origin_value is not None:
            origin = origin_value
        if end_value is not None:
            direction = end_value - origin
        elif direction_value is not None:
            direction = direction_value
    elif isinstance(centerline, Sequence) and not isinstance(centerline, (str, bytes)):
        try:
            array = np.asarray(centerline, dtype=float)
        except (TypeError, ValueError):
            array = np.zeros((0, 0))
        if array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 2:
            origin[: array.shape[1]] = array[0, : min(array.shape[1], 3)]
            end = np.zeros(3, dtype=float)
            end[: array.shape[1]] = array[1, : min(array.shape[1], 3)]
            direction = end - origin
    if direction is None or float(np.linalg.norm(direction[:2])) < 1.0e-9:
        ranges = np.ptp(points[:, :2], axis=0)
        direction = np.asarray([1.0, 0.0, 0.0]) if ranges[0] >= ranges[1] else np.asarray([0.0, 1.0, 0.0])
        origin = np.zeros(3, dtype=float)
    direction = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    return (points - origin) @ direction


def _speed(row: Mapping[str, Any]) -> float:
    speed = _to_float(_row_value(row, "speed", "velocity_norm"))
    if math.isfinite(speed):
        return speed
    velocity = np.asarray(
        [
            _to_float(_row_value(row, "vx", "velocity_x"), default=0.0),
            _to_float(_row_value(row, "vy", "velocity_y"), default=0.0),
            _to_float(_row_value(row, "vz", "velocity_z"), default=0.0),
        ]
    )
    return float(np.linalg.norm(velocity))


def _waypoint_type(row: Mapping[str, Any]) -> str:
    value = str(
        _row_value(
            row,
            "active_waypoint_type",
            "waypoint_type",
            "candidate_type",
            default="UNKNOWN",
        )
    ).strip()
    return value.upper() if value else "UNKNOWN"


def _owner_label(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan", "-1"}:
        return "NONE"
    number = _to_float(text)
    return str(int(number)) if math.isfinite(number) else text


def plot_temporal_diagnostics(
    *,
    key: EpisodeKey,
    trajectory_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    dt: float,
    output_base: Path,
) -> list[Path]:
    """Plot longitudinal motion, speed, waypoint type, and reservation owner."""

    valid_rows = [row for row in trajectory_rows if _row_xyz(row) is not None]
    if not valid_rows:
        return []
    fig, axes = plt.subplots(4, 1, figsize=(12.5, 11.0), sharex=True)
    agents = sorted({_agent_id(row) for row in valid_rows})
    all_waypoint_types = list(WAYPOINT_ORDER)
    observed_types = sorted({_waypoint_type(row) for row in valid_rows})
    for value in observed_types:
        if value not in all_waypoint_types:
            all_waypoint_types.append(value)
    type_index = {value: index for index, value in enumerate(all_waypoint_types)}

    for color_index, agent in enumerate(agents):
        rows = sorted(
            (row for row in valid_rows if _agent_id(row) == agent),
            key=lambda row: _row_time(row, dt),
        )
        times = np.asarray([_row_time(row, dt) for row in rows], dtype=float)
        points = np.stack([_row_xyz(row) for row in rows], axis=0)
        color = AGENT_COLORS[color_index % len(AGENT_COLORS)]
        axes[0].plot(
            times,
            _centerline_projection(points, metadata),
            color=color,
            linewidth=1.8,
            label=f"UAV {agent}",
        )
        axes[1].plot(
            times,
            [_speed(row) for row in rows],
            color=color,
            linewidth=1.6,
            label=f"UAV {agent}",
        )
        axes[2].step(
            times,
            [type_index[_waypoint_type(row)] for row in rows],
            where="post",
            color=color,
            linewidth=1.5,
            label=f"UAV {agent}",
        )

    owner_by_time: dict[float, str] = {}
    for row in valid_rows:
        owner = _row_value(row, "reservation_owner", "owner_agent_id", default="")
        if owner not in (None, ""):
            owner_by_time[_row_time(row, dt)] = _owner_label(owner)
    for row in event_rows:
        owner = _row_value(row, "reservation_owner", "owner_agent_id", default="")
        if owner not in (None, ""):
            owner_by_time[_row_time(row, dt)] = _owner_label(owner)
    if owner_by_time:
        owner_labels = ["NONE"] + [
            value
            for value in sorted(
                {item for item in owner_by_time.values() if item != "NONE"},
                key=lambda item: (_to_int(item, default=10**9), item),
            )
        ]
        owner_index = {value: index for index, value in enumerate(owner_labels)}
        owner_times = np.asarray(sorted(owner_by_time), dtype=float)
        axes[3].step(
            owner_times,
            [owner_index[owner_by_time[time]] for time in owner_times],
            where="post",
            color="#7F56D9",
            linewidth=1.9,
        )
        axes[3].set_yticks(range(len(owner_labels)), owner_labels)
    else:
        axes[3].text(
            0.5,
            0.5,
            "reservation owner 未记录",
            ha="center",
            va="center",
            transform=axes[3].transAxes,
            color="#667085",
        )
        axes[3].set_yticks([])

    axes[0].set_ylabel("纵向位置 / m")
    axes[0].set_title("沿走廊中心线的位置—时间 / Longitudinal position")
    axes[1].set_ylabel("速度 / m s$^{-1}$")
    axes[1].set_title("速度—时间 / Speed")
    axes[2].set_ylabel("Waypoint type")
    axes[2].set_yticks(range(len(all_waypoint_types)), all_waypoint_types, fontsize=8)
    axes[2].set_title("Active waypoint 类型状态")
    axes[3].set_ylabel("Owner")
    axes[3].set_xlabel("时间 / s")
    axes[3].set_title("走廊 reservation owner 状态")
    for ax in axes:
        ax.grid(True)
    _deduplicate_legend(axes[0], loc="best", ncol=min(3, len(agents)), fontsize=8)
    fig.suptitle(
        f"{key.group}: seed={key.seed} 时序执行诊断 / Temporal diagnostics",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return _save_figure(fig, output_base)


def _mean_bool(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> float:
    values = [_to_bool(_row_value(row, *names)) for row in rows]
    return float(np.mean(values)) if values else math.nan


def _summary_metric_values(
    summary_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, dict[str, float]]]:
    groups = sorted(
        {
            _group_name(row)
            for row in list(summary_rows) + list(episode_rows) + list(event_rows)
            if _group_name(row)
        },
        key=_group_sort_key,
    )
    values: dict[str, dict[str, float]] = {group: {} for group in groups}
    for group in groups:
        group_summary = [row for row in summary_rows if _group_name(row) == group]
        group_episodes = [row for row in episode_rows if _group_name(row) == group]
        group_events = [row for row in event_rows if _group_name(row) == group]
        for metric, aliases, _title, percent in RISK_METRICS:
            candidates = [
                _to_float(_row_value(row, *aliases))
                for row in group_summary
                if math.isfinite(_to_float(_row_value(row, *aliases)))
            ]
            if candidates:
                value = float(np.mean(candidates))
                if percent and 1.0 < value <= 100.0:
                    value /= 100.0
                values[group][metric] = value
                continue
            if metric == "team_success_rate":
                values[group][metric] = _mean_bool(group_episodes, aliases)
            elif metric == "inter_agent_collision_rate":
                values[group][metric] = _mean_bool(group_episodes, aliases)
            elif metric == "obstacle_collision_rate":
                values[group][metric] = _mean_bool(group_episodes, aliases)
            elif metric == "timeout_rate":
                values[group][metric] = _mean_bool(group_episodes, aliases)
            elif metric == "out_of_bounds_rate":
                values[group][metric] = _mean_bool(group_episodes, aliases)
            else:
                overlap = [
                    _to_float(_row_value(row, *aliases))
                    for row in list(group_episodes) + list(group_events)
                    if math.isfinite(_to_float(_row_value(row, *aliases)))
                ]
                values[group][metric] = float(np.mean(overlap)) if overlap else math.nan
    return groups, values


def plot_group_risk_metrics(
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    output_base: Path,
) -> list[Path]:
    """Plot success and explicit risk-transfer metrics across executed groups."""

    groups, values = _summary_metric_values(summary_rows, episode_rows, event_rows)
    if not groups:
        return []
    if not any(
        math.isfinite(value)
        for group_values in values.values()
        for value in group_values.values()
    ):
        return []
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.4))
    for ax, (metric, _aliases, title, percent) in zip(axes.flat, RISK_METRICS):
        plotted_groups = [group for group in groups if math.isfinite(values[group].get(metric, math.nan))]
        metric_values = [values[group][metric] for group in plotted_groups]
        if not plotted_groups:
            ax.text(
                0.5,
                0.5,
                "未记录 / not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#667085",
            )
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        colors = [
            GROUP_COLORS.get(_canonical_group(group), "#667085") for group in plotted_groups
        ]
        bars = ax.bar(
            np.arange(len(plotted_groups)),
            metric_values,
            color=colors,
            edgecolor="white",
            linewidth=0.7,
            width=0.72,
        )
        ax.set_xticks(np.arange(len(plotted_groups)), plotted_groups, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y")
        if percent:
            ax.set_ylim(0.0, max(1.05, max(metric_values, default=0.0) * 1.18))
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        else:
            ax.set_ylim(0.0, max(0.1, max(metric_values, default=0.0) * 1.2))
        for bar, value in zip(bars, metric_values):
            label = f"{value:.1%}" if percent else f"{value:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.02 * max(ax.get_ylim()[1], 1.0e-6),
                label,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#344054",
            )
    fig.suptitle(
        "冻结底层策略走廊协调：性能与风险转移 / Performance and risk transfer",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    return _save_figure(fig, output_base)


def _actual_event_times(
    event_rows: Sequence[Mapping[str, Any]], dt: float
) -> dict[tuple[str, int, int], dict[str, float]]:
    actual: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    for row in event_rows:
        key = (
            _group_name(row),
            _to_int(_row_value(row, "seed", default=0)),
            _agent_id(row),
        )
        explicit_entry = _to_float(_row_value(row, "actual_entry_time"))
        explicit_exit = _to_float(_row_value(row, "actual_exit_time"))
        if math.isfinite(explicit_entry):
            actual[key]["entry"] = explicit_entry
        if math.isfinite(explicit_exit):
            actual[key]["exit"] = explicit_exit
        label = str(
            _row_value(row, "event_type", "event", "transition", "new_state", default="")
        ).upper()
        event_time = _row_time(row, dt)
        if ("ENTRY" in label or "ENTER" in label) and (
            "ACTUAL" in label or "CROSS" in label or "OCCUP" in label
        ):
            actual[key].setdefault("entry", event_time)
        if "EXIT" in label and ("ACTUAL" in label or "CROSS" in label or "RELEASE" in label):
            actual[key].setdefault("exit", event_time)
    return actual


def _prediction_pairs(
    prediction_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    dt: float,
) -> dict[str, list[tuple[float, float, str]]]:
    actual_lookup = _actual_event_times(event_rows, dt)
    pairs: dict[str, list[tuple[float, float, str]]] = {"entry": [], "exit": []}
    for row in prediction_rows:
        group = _group_name(row)
        # E4 may be absent after an earlier Go/No-Go stop; do not synthesize it.
        lookup_key = (
            group,
            _to_int(_row_value(row, "seed", default=0)),
            _agent_id(row),
        )
        for kind in ("entry", "exit"):
            predicted = _to_float(
                _row_value(
                    row,
                    f"predicted_{kind}_time",
                    f"prediction_{kind}_time",
                    f"predicted_corridor_{kind}_time",
                )
            )
            observed = _to_float(
                _row_value(
                    row,
                    f"actual_{kind}_time",
                    f"observed_{kind}_time",
                    f"true_{kind}_time",
                )
            )
            if not math.isfinite(observed):
                observed = actual_lookup.get(lookup_key, {}).get(kind, math.nan)
            if math.isfinite(predicted) and math.isfinite(observed):
                pairs[kind].append((predicted, observed, group))
    return pairs


def plot_prediction_accuracy(
    *,
    prediction_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    dt: float,
    output_base: Path,
) -> list[Path]:
    """Plot policy-rollout entry/exit predictions against observed times."""

    if not prediction_rows:
        return []
    pairs = _prediction_pairs(prediction_rows, event_rows, dt)
    if not pairs["entry"] and not pairs["exit"]:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    specifications = (
        ("entry", "入口时间 / Corridor entry"),
        ("exit", "出口时间 / Corridor exit"),
    )
    for ax, (kind, title) in zip(axes, specifications):
        entries = pairs[kind]
        if not entries:
            ax.text(
                0.5,
                0.5,
                "未记录 / not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#667085",
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        groups = sorted({entry[2] for entry in entries}, key=_group_sort_key)
        for group in groups:
            points = np.asarray(
                [(predicted, actual) for predicted, actual, label in entries if label == group],
                dtype=float,
            )
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=38,
                alpha=0.78,
                color=GROUP_COLORS.get(_canonical_group(group), "#667085"),
                edgecolor="white",
                linewidth=0.5,
                label=group,
            )
        all_points = np.asarray([(entry[0], entry[1]) for entry in entries], dtype=float)
        lower = float(np.min(all_points))
        upper = float(np.max(all_points))
        padding = max(0.5, 0.06 * (upper - lower if upper > lower else 1.0))
        ax.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            color="#344054",
            linestyle="--",
            linewidth=1.1,
            label="理想预测 / y=x",
        )
        mae = float(np.mean(np.abs(all_points[:, 0] - all_points[:, 1])))
        ax.text(
            0.04,
            0.94,
            f"MAE = {mae:.2f} s\nn = {len(entries)}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#D0D5DD", "alpha": 0.85},
        )
        ax.set_xlabel("预测时间 / s")
        ax.set_ylabel("真实时间 / s")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        _deduplicate_legend(ax, loc="lower right", fontsize=8)
    fig.suptitle(
        "反事实 rollout 时间预测校准 / Counterfactual-rollout calibration",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return _save_figure(fig, output_base)


def _event_label(row: Mapping[str, Any]) -> str:
    label = str(
        _row_value(
            row,
            "event_type",
            "event",
            "transition",
            "new_state",
            "status",
            default="EVENT",
        )
    ).strip()
    old_state = str(_row_value(row, "previous_state", "old_state", "state_from", default="")).strip()
    new_state = str(_row_value(row, "new_state", "state_to", default="")).strip()
    if old_state and new_state and old_state != new_state:
        return f"{old_state}→{new_state}"
    return label or "EVENT"


def _event_style(label: str) -> tuple[str, str]:
    value = label.upper()
    if any(token in value for token in ("COLLISION", "OUT_OF_BOUNDS", "FAIL")):
        return "X", "#D92D20"
    if "RELEASE" in value or "EXIT" in value:
        return "v", "#7F56D9"
    if "ENTRY" in value or "ENTER" in value or "OCCUP" in value:
        return "^", "#12B76A"
    if "AUTH" in value or "REQUEST" in value or "RESERV" in value:
        return "D", "#F79009"
    if "HOLD" in value or "DECEL" in value:
        return "s", "#1570EF"
    if "WP:" in value or "WAYPOINT" in value or "SWITCH" in value:
        return "o", "#475467"
    if "TIMEOUT" in value or "DEADLOCK" in value:
        return "P", "#B42318"
    return "o", "#667085"


def plot_failure_timeline(
    *,
    key: EpisodeKey,
    episode_row: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
    switch_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    dt: float,
    output_base: Path,
) -> list[Path]:
    """Save an event-level diagnostic timeline for one failed seed."""

    records: list[tuple[float, int, str]] = []
    for row in event_rows:
        records.append((_row_time(row, dt), _agent_id(row), _event_label(row)))
    for row in switch_rows:
        new_type = str(
            _row_value(row, "new_type", "new_waypoint_type", "waypoint_type", default="?")
        )
        records.append((_row_time(row, dt), _agent_id(row), f"WP:{new_type}"))
    collision_names = (
        ("inter_agent_collision", "INTER_AGENT_COLLISION"),
        ("obstacle_collision", "OBSTACLE_COLLISION"),
        ("out_of_bounds", "OUT_OF_BOUNDS"),
    )
    seen_collision: set[tuple[int, str]] = set()
    for row in sorted(trajectory_rows, key=lambda item: _row_time(item, dt)):
        for field_name, label in collision_names:
            marker_key = (_agent_id(row), label)
            if _to_bool(_row_value(row, field_name)) and marker_key not in seen_collision:
                records.append((_row_time(row, dt), _agent_id(row), label))
                seen_collision.add(marker_key)

    failure_type = str(
        _row_value(episode_row, "first_failure_type", default="FAILURE")
    ).strip()
    final_time = _to_float(
        _row_value(episode_row, "makespan", "episode_time", "time"),
        default=math.nan,
    )
    if not math.isfinite(final_time):
        final_time = dt * _to_float(
            _row_value(episode_row, "episode_steps", "steps", default=0.0), default=0.0
        )
    if failure_type and failure_type.upper() not in {"NONE", "SUCCESS", "N/A"}:
        records.append((final_time, -1, f"FIRST:{failure_type}"))
    if not records:
        return []

    agents = sorted({agent for _time, agent, _label in records if agent >= 0})
    lanes = agents + [-1]
    lane_index = {agent: index for index, agent in enumerate(lanes)}
    lane_labels = [f"UAV {agent}" for agent in agents] + ["Episode"]
    fig_height = max(4.2, 1.05 * len(lanes) + 2.3)
    fig, ax = plt.subplots(figsize=(13.2, fig_height))
    for index, label in enumerate(lane_labels):
        ax.axhline(index, color="#EAECF0", linewidth=1.0, zorder=0)
        ax.text(
            -0.01,
            index,
            label,
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform(),
            color="#344054",
            fontsize=9,
        )

    annotation_levels: defaultdict[int, int] = defaultdict(int)
    legend_handles: dict[str, Line2D] = {}
    for event_time, agent, label in sorted(records):
        lane = lane_index[agent if agent in lane_index else -1]
        marker, color = _event_style(label)
        ax.scatter(
            event_time,
            lane,
            marker=marker,
            s=58,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        category = label.split(":", 1)[0]
        legend_handles.setdefault(
            category,
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=7,
                label=category,
            ),
        )
        level = annotation_levels[lane] % 4
        annotation_levels[lane] += 1
        offset = (12 + 9 * level) * (1 if level % 2 == 0 else -1)
        ax.annotate(
            label,
            (event_time, lane),
            xytext=(3, offset),
            textcoords="offset points",
            rotation=35,
            ha="left",
            va="bottom" if offset > 0 else "top",
            fontsize=6.8,
            color="#344054",
            arrowprops={"arrowstyle": "-", "color": "#98A2B3", "lw": 0.45},
        )

    ax.set_yticks([])
    ax.set_ylim(-0.65, len(lanes) - 0.35)
    ax.set_xlabel("时间 / s")
    ax.set_title(
        f"{key.group}: seed={key.seed} 失败事件时间线 / Failure-event timeline\n"
        f"first_failure_type = {failure_type or 'UNKNOWN'}",
        fontsize=12,
    )
    ax.grid(True, axis="x")
    ax.legend(
        handles=list(legend_handles.values()),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(5, max(1, len(legend_handles))),
        fontsize=7.5,
    )
    fig.tight_layout()
    return _save_figure(fig, output_base)


def _is_failure_episode(row: Mapping[str, Any]) -> bool:
    failure = str(_row_value(row, "first_failure_type", default="")).strip().upper()
    if failure and failure not in {"NONE", "SUCCESS", "N/A", "NA"}:
        return True
    success_value = _row_value(row, "team_success", "success", default="")
    return success_value not in (None, "") and not _to_bool(success_value)


def _representative_keys(data: ArtifactData) -> list[EpisodeKey]:
    available = _trajectory_episode_keys(data.trajectories)
    representatives: list[EpisodeKey] = []
    groups = sorted({key.group for key in available}, key=_group_sort_key)
    episode_lookup = {_episode_key(row): row for row in data.episodes}
    for group in groups:
        candidates = [key for key in available if key.group == group]
        failed = [
            key
            for key in candidates
            if key in episode_lookup and _is_failure_episode(episode_lookup[key])
        ]
        representatives.append((failed or candidates)[0])
    return representatives


def _failure_rows(data: ArtifactData) -> list[tuple[EpisodeKey, Mapping[str, Any]]]:
    output: list[tuple[EpisodeKey, Mapping[str, Any]]] = []
    seen: set[EpisodeKey] = set()
    for row in data.episodes:
        if not _is_failure_episode(row):
            continue
        key = _episode_key(row)
        if key not in seen:
            output.append((key, row))
            seen.add(key)
    return sorted(output, key=lambda item: (_group_sort_key(item[0].group), item[0]))


def generate_visualizations(
    data: ArtifactData,
    *,
    output_dir: Path,
    max_failure_seeds: int = 0,
) -> tuple[list[Path], list[str]]:
    """Generate every figure supported by the available staged artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dt = _config_dt(data.config)
    metadata = _metadata(data.config)
    generated: list[Path] = []
    skipped: list[str] = []

    risk_paths = plot_group_risk_metrics(
        summary_rows=data.summary,
        episode_rows=data.episodes,
        event_rows=data.events,
        output_base=output_dir / "01_group_performance_and_risk",
    )
    if risk_paths:
        generated.extend(risk_paths)
    else:
        skipped.append("group risk figure: summary/episode metrics unavailable")

    prediction_paths = plot_prediction_accuracy(
        prediction_rows=data.predictions,
        event_rows=data.events,
        dt=dt,
        output_base=output_dir / "02_rollout_prediction_calibration",
    )
    if prediction_paths:
        generated.extend(prediction_paths)
    else:
        skipped.append("prediction figure: E4 not executed or matched actual times unavailable")

    representatives = _representative_keys(data)
    if not representatives:
        skipped.append("trajectory/time-state figures: trajectories.csv unavailable")
    for key in representatives:
        trajectory_rows = _episode_rows(data.trajectories, key)
        switch_rows = _episode_rows(data.switches, key)
        event_rows = _episode_rows(data.events, key)
        generated.extend(
            plot_top_down_trajectory(
                key=key,
                trajectory_rows=trajectory_rows,
                switch_rows=switch_rows,
                event_rows=event_rows,
                metadata=metadata,
                dt=dt,
                output_base=output_dir / f"03_{key.slug}_trajectory",
            )
        )
        generated.extend(
            plot_temporal_diagnostics(
                key=key,
                trajectory_rows=trajectory_rows,
                event_rows=event_rows,
                metadata=metadata,
                dt=dt,
                output_base=output_dir / f"04_{key.slug}_temporal_diagnostics",
            )
        )

    failures = _failure_rows(data)
    if max_failure_seeds > 0:
        failures = failures[:max_failure_seeds]
    if not failures and data.episodes:
        skipped.append("failure figures: no failed seed in executed groups")
    elif not data.episodes:
        skipped.append("failure figures: episodes.csv unavailable")
    failure_dir = output_dir / "failures"
    for key, episode_row in failures:
        trajectory_rows = _episode_rows(data.trajectories, key)
        switch_rows = _episode_rows(data.switches, key)
        event_rows = _episode_rows(data.events, key)
        if trajectory_rows:
            generated.extend(
                plot_top_down_trajectory(
                    key=key,
                    trajectory_rows=trajectory_rows,
                    switch_rows=switch_rows,
                    event_rows=event_rows,
                    metadata=metadata,
                    dt=dt,
                    output_base=failure_dir / f"{key.slug}_trajectory",
                    title_prefix="失败回合 / Failed episode",
                )
            )
        else:
            skipped.append(f"failure trajectory {key.slug}: trajectory rows unavailable")
        timeline_paths = plot_failure_timeline(
            key=key,
            episode_row=episode_row,
            event_rows=event_rows,
            switch_rows=switch_rows,
            trajectory_rows=trajectory_rows,
            dt=dt,
            output_base=failure_dir / f"{key.slug}_event_timeline",
        )
        if timeline_paths:
            generated.extend(timeline_paths)
        else:
            skipped.append(f"failure timeline {key.slug}: event data unavailable")

    return generated, skipped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing the staged experiment CSV/config artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure directory (default: <artifact-dir>/figures).",
    )
    parser.add_argument(
        "--max-failure-seeds",
        type=int,
        default=0,
        help="Maximum failed episodes to visualize; 0 keeps every failed seed.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    artifact_dir = args.artifact_dir.expanduser().resolve()
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_dir}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else artifact_dir / "figures"
    )
    configure_matplotlib()
    data = load_artifacts(artifact_dir)
    generated, skipped = generate_visualizations(
        data,
        output_dir=output_dir,
        max_failure_seeds=max(0, int(args.max_failure_seeds)),
    )

    print(f"Artifact directory: {artifact_dir}")
    print(
        "Loaded inputs: "
        + (", ".join(sorted(data.loaded_paths)) if data.loaded_paths else "none")
    )
    print(f"Generated {len(generated)} files in: {output_dir}")
    for reason in skipped:
        print(f"Skipped: {reason}")
    return output_dir


if __name__ == "__main__":
    main()
