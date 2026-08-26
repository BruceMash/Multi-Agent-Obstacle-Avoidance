"""Generate paper-ready PPO-Direct learning, outcome, and shared-scene figures.

This module is post-processing only.  It reads frozen PPO-Direct and Formal V2
artifacts and never runs or changes a policy, environment, or evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = REPO_ROOT / "artifacts/gat_recurrent_smoothness_rescue/20260821_180244/10_formal_v2"
FORMAL_RECORDS = FORMAL_ROOT / "formal_records"
STAGES = ("Stage I", "Stage II", "Stage III", "Stage IV")
STAGE_KEYS = ("stage_1", "stage_2", "stage_3", "stage_4")
METHODS = (
    ("M2_DWA_SensingMatched", "DWA-SensingMatched", "--"),
    ("PPO_DIRECT", "PPO-Direct", "-"),
    ("M4_Direct_SAC_DMP", "Direct SAC-DMP", ":"),
    ("M9_Proposed_RERR_GAT_SAC_DMP", "Proposed", "-."),
)
AGENT_COLORS = ("#0072B2", "#D55E00", "#009E73")
METHOD_COLORS = ("#0072B2", "#CC79A7", "#E69F00", "#009E73")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if fields is None:
        fields = tuple(materialized[0].keys()) if materialized else ()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: Any, out: Path, stem: str) -> tuple[Path, Path]:
    pdf = out / "figures_pdf" / f"{stem}.pdf"
    png = out / "figures_png_600dpi" / f"{stem}.png"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def rolling_mean(values: np.ndarray, width: int = 12) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        lo = max(0, index - width + 1)
        result[index] = float(np.nanmean(values[lo : index + 1]))
    return result


def learning_curve(root: Path, out: Path) -> dict[str, Any]:
    history = read_csv(root / "05_training/PPO_DIRECT_TRAINING_HISTORY.csv")
    x = np.asarray([float(row["timesteps"]) / 1e6 for row in history])
    success = np.asarray([float(row["recent_team_success"]) for row in history])
    collision = np.asarray([float(row["recent_collision"]) for row in history])
    completion = np.asarray([float(row["recent_agent_completion"]) for row in history])
    source_rows = []
    for index, row in enumerate(history):
        source_rows.append(
            {
                "timesteps": row["timesteps"],
                "train_success_raw": success[index],
                "train_success_rolling12": rolling_mean(success)[index],
                "train_collision_raw": collision[index],
                "train_collision_rolling12": rolling_mean(collision)[index],
                "train_agent_completion_rolling12": rolling_mean(completion)[index],
            }
        )
    dev_points = []
    for path in sorted((root / "06_development").glob("evaluation_*_full.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        timesteps = int(Path(record["checkpoint"]).stem.rsplit("_", 1)[-1])
        overall = next(row for row in record["summary"] if row["scope"] == "overall")
        dev_points.append(
            {
                "timesteps": timesteps,
                "success": float(overall["success_rate"]),
                "collision": float(overall["collision_rate"]),
                "selected": timesteps == 2002944,
            }
        )
    write_csv(out / "source_data/figure_01_learning_curve.csv", source_rows)
    write_csv(out / "source_data/figure_01_dev_points.csv", dev_points)

    fig, axis = plt.subplots(figsize=(7.2, 3.5))
    axis.plot(x, 100 * rolling_mean(success), color="#0072B2", linewidth=1.35, label="Train success (rolling 12)")
    axis.plot(x, 100 * rolling_mean(collision), color="#D55E00", linewidth=1.2, linestyle="--", label="Train collision (rolling 12)")
    axis.plot(x, 100 * rolling_mean(completion), color="#009E73", linewidth=1.1, linestyle="-.", label="Agent completion (rolling 12)")
    dx = np.asarray([row["timesteps"] / 1e6 for row in dev_points])
    dy = np.asarray([100 * row["success"] for row in dev_points])
    axis.scatter(dx, dy, marker="D", s=34, facecolor="white", edgecolor="#7A3E9D", linewidth=1.1, zorder=5, label="Fixed Dev success")
    selected = next(row for row in dev_points if row["selected"])
    axis.scatter([selected["timesteps"] / 1e6], [100 * selected["success"]], marker="*", s=95, color="#7A3E9D", edgecolor="black", linewidth=0.35, zorder=6)
    axis.annotate("selected 2.00M: 55%", (selected["timesteps"] / 1e6, 100 * selected["success"]), xytext=(8, 8), textcoords="offset points")
    axis.set_xlim(0, max(x) * 1.01)
    axis.set_ylim(0, 103)
    axis.set_xlabel("Training transitions (millions)")
    axis.set_ylabel("Episode rate (%)")
    axis.set_title("PPO-Direct training and fixed-Dev checkpoint selection")
    axis.grid(axis="y", color="0.9", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    pdf, png = save_figure(fig, out, "figure_01_ppo_direct_learning_curve")
    caption = (
        "Figure 1. PPO-Direct training outcomes (12-update rolling means) and deterministic evaluations "
        "on the unchanged 100-scenario Development set. The 2.00M checkpoint was selected by team "
        "success before Holdout or Formal V2 access; the 3.00M continuation regressed to 28% Dev success."
    )
    caption_path = out / "captions/figure_01_ppo_direct_learning_curve_caption.txt"
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return {"figure": 1, "title": "PPO-Direct learning curve", "pdf": pdf, "png": png, "caption": caption_path}


def success_comparison(root: Path, out: Path) -> dict[str, Any]:
    baseline = [
        *read_csv(FORMAL_ROOT / "formal_v2_stage_summary.csv"),
        *read_csv(FORMAL_ROOT / "overall_method_summary.csv"),
    ]
    ppo = read_csv(root / "09_formal_v2/ppo_direct_stage_summary.csv")
    scope_map = {"overall": "Overall", "stage_1": "Stage I", "stage_2": "Stage II", "stage_3": "Stage III", "stage_4": "Stage IV"}
    source_rows = []
    for method_id, display_name, _ in METHODS:
        if method_id == "PPO_DIRECT":
            values = {scope_map[row["scope"]]: float(row["success_rate"]) for row in ppo if row["scope"] in scope_map}
        else:
            values = {
                ("Overall" if row["scope"] == "overall" else row["scope"]): float(row["success_rate"])
                for row in baseline
                if row["method_id"] == method_id and row["scope"] in (*STAGES, "overall")
            }
        for scope in ("Overall", *STAGES):
            source_rows.append({"method_id": method_id, "display_name": display_name, "scope": scope, "success_rate": values[scope]})
    write_csv(out / "source_data/figure_02_success_comparison.csv", source_rows)

    fig, axis = plt.subplots(figsize=(7.2, 3.55))
    labels = ("Overall", *STAGES)
    x = np.arange(len(labels))
    for index, (method_id, display_name, linestyle) in enumerate(METHODS):
        rows = [row for row in source_rows if row["method_id"] == method_id]
        values = [100 * next(row["success_rate"] for row in rows if row["scope"] == label) for label in labels]
        axis.plot(x, values, color=METHOD_COLORS[index], linestyle=linestyle, marker=("o", "D", "s", "^")[index], linewidth=1.55, markersize=4.5, label=display_name)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 103)
    axis.set_ylabel("Team success (%)")
    axis.set_title("Frozen Formal V2 success comparison")
    axis.grid(axis="y", color="0.9", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="lower left")
    pdf, png = save_figure(fig, out, "figure_02_ppo_direct_success_comparison")
    caption = (
        "Figure 2. Frozen Formal V2 team success for the information-matched classical baseline, the "
        "post-hoc frozen PPO-Direct baseline, Direct SAC-DMP, and Proposed. Every point uses the same "
        "100 scenarios per stage; no comparator was rerun."
    )
    caption_path = out / "captions/figure_02_ppo_direct_success_comparison_caption.txt"
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return {"figure": 2, "title": "Formal success comparison", "pdf": pdf, "png": png, "caption": caption_path}


def select_shared_scenes(root: Path) -> list[dict[str, str]]:
    ppo_rows = read_csv(root / "09_formal_v2/ppo_direct_formal_team_results.csv")
    formal_rows = read_csv(FORMAL_ROOT / "formal_v2_team_results.csv")
    ppo_by_scene = {row["scenario_id"]: row for row in ppo_rows}
    by_method_scene = {(row["method_id"], row["scenario_id"]): row for row in formal_rows}
    selected = []
    for stage, stage_key in zip(STAGES, STAGE_KEYS):
        eligible = []
        for scenario_id, ppo_row in ppo_by_scene.items():
            if ppo_row["stage"] != stage_key or not truth(ppo_row["team_success"]):
                continue
            if not truth(by_method_scene[("M2_DWA_SensingMatched", scenario_id)]["team_success"]):
                continue
            if not truth(by_method_scene[("M9_Proposed_RERR_GAT_SAC_DMP", scenario_id)]["team_success"]):
                continue
            eligible.append(ppo_row)
        if not eligible:
            raise RuntimeError(f"no triple-success shared-scene anchor for {stage}")
        times = np.asarray([float(row["completion_time_s"]) for row in eligible])
        median = float(np.median(times))
        anchor = min(eligible, key=lambda row: (abs(float(row["completion_time_s"]) - median), row["scenario_id"]))
        selected.append({"stage": stage, "stage_key": stage_key, "scenario_id": anchor["scenario_id"], "selection_rule": "PPO completion closest to triple-success stage median"})
    return selected


def load_positions(root: Path, method_id: str, scenario_id: str) -> np.ndarray:
    if method_id == "PPO_DIRECT":
        record = json.loads((root / f"12_trajectories/formal_records/{scenario_id}.json").read_text(encoding="utf-8"))
        return np.asarray(record["positions"], dtype=float)
    with np.load(FORMAL_RECORDS / method_id / f"{scenario_id}_trajectory.npz", allow_pickle=False) as archive:
        return np.asarray(archive["positions"], dtype=float)


def plot_static(axis: Any, spec: Mapping[str, Any]) -> None:
    center = np.asarray(spec["center"], dtype=float)
    if spec["type"] == "box":
        half = np.asarray(spec["half_extents"], dtype=float)
        z0, z1 = max(0.0, center[2] - half[2]), center[2] + half[2]
        corners = [(center[0] + sx * half[0], center[1] + sy * half[1]) for sx in (-1, 1) for sy in (-1, 1)]
        for x, y in corners:
            axis.plot([x, x], [y, y], [z0, z1], color="0.62", linewidth=0.35, alpha=0.45)
        for z in (z0, z1):
            xs = [center[0]-half[0], center[0]+half[0], center[0]+half[0], center[0]-half[0], center[0]-half[0]]
            ys = [center[1]-half[1], center[1]-half[1], center[1]+half[1], center[1]+half[1], center[1]-half[1]]
            axis.plot(xs, ys, [z] * 5, color="0.62", linewidth=0.35, alpha=0.45)
    else:
        theta = np.linspace(0, 2 * np.pi, 24)
        radius = float(spec["radius"])
        z0 = max(0.0, center[2] - float(spec["half_height"]))
        z1 = center[2] + float(spec["half_height"])
        for z in (z0, z1):
            axis.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta), z, color="0.62", linewidth=0.35, alpha=0.45)


def shared_scene_trajectories(root: Path, out: Path) -> dict[str, Any]:
    anchors = select_shared_scenes(root)
    write_csv(out / "source_data/figure_03_shared_scene_selection.csv", anchors)
    trajectory_rows = []
    fig = plt.figure(figsize=(9.0, 7.4))
    for panel, anchor in enumerate(anchors, start=1):
        axis = fig.add_subplot(2, 2, panel, projection="3d")
        scenario_id = anchor["scenario_id"]
        ppo_record = json.loads((root / f"12_trajectories/formal_records/{scenario_id}.json").read_text(encoding="utf-8"))
        for spec in ppo_record["static_obstacles"]:
            plot_static(axis, spec)
        for obstacle in ppo_record["dynamic_obstacles"]:
            center = np.asarray(obstacle["center"], dtype=float)
            velocity = np.asarray(obstacle["velocity"], dtype=float)
            axis.scatter(*center, color="0.28", marker="x", s=11, linewidth=0.6)
            end = np.clip(center + 5.0 * velocity, [0, 0, 0], [100, 100, 4])
            axis.plot([center[0], end[0]], [center[1], end[1]], [center[2], end[2]], color="0.35", linestyle=(0, (1, 2)), linewidth=0.5, alpha=0.65)
        for method_id, display_name, linestyle in METHODS:
            positions = load_positions(root, method_id, scenario_id)
            for agent in range(3):
                path = positions[:, agent]
                axis.plot(path[:, 0], path[:, 1], path[:, 2], color=AGENT_COLORS[agent], linestyle=linestyle, linewidth=1.0, alpha=0.82)
                stride = max(1, len(path) // 100)
                for step in range(0, len(path), stride):
                    trajectory_rows.append({"stage": anchor["stage"], "scenario_id": scenario_id, "method_id": method_id, "display_name": display_name, "agent_id": agent + 1, "step": step, "x_m": path[step, 0], "y_m": path[step, 1], "z_m": path[step, 2]})
        starts = np.asarray(ppo_record["starts"], dtype=float)
        goals = np.asarray(ppo_record["goals"], dtype=float)
        for agent in range(3):
            axis.scatter(*starts[agent], color=AGENT_COLORS[agent], marker="^", s=20, edgecolor="black", linewidth=0.25)
            axis.scatter(*goals[agent], color=AGENT_COLORS[agent], marker="*", s=34, edgecolor="black", linewidth=0.25)
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 100)
        axis.set_zlim(0, 4)
        axis.set_xlabel("x (m)", labelpad=1)
        axis.set_ylabel("y (m)", labelpad=1)
        axis.set_zlabel("z (m)", labelpad=0)
        axis.set_title(anchor["stage"], pad=2)
        axis.view_init(elev=28, azim=-57)
        axis.grid(True, linewidth=0.25, color="0.9")
    write_csv(out / "source_data/figure_03_shared_scene_trajectories.csv", trajectory_rows)
    agent_handles = [Line2D([0], [0], color=AGENT_COLORS[index], linewidth=1.5, label=f"UAV {index + 1}") for index in range(3)]
    method_handles = [Line2D([0], [0], color="0.2", linewidth=1.25, linestyle=linestyle, label=display_name) for _, display_name, linestyle in METHODS]
    marker_handles = [Line2D([0], [0], color="0.2", marker="^", linestyle="None", label="start"), Line2D([0], [0], color="0.2", marker="*", linestyle="None", label="goal")]
    fig.legend(handles=[*agent_handles, *method_handles, *marker_handles], ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.subplots_adjust(top=0.90, wspace=0.02, hspace=0.09)
    pdf, png = save_figure(fig, out, "figure_03_shared_scene_3d_trajectories")
    caption = (
        "Figure 3. Shared-scene 3-D trajectories across the four frozen stages. UAV identity is encoded "
        "by color and method by line style; triangles and stars denote starts and goals. Each stage anchor "
        "is selected before plotting as the PPO success closest to the median PPO completion time among "
        "scenarios where PPO-Direct, DWA-SensingMatched, and Proposed all succeed. Direct SAC-DMP is shown "
        "on the same scenario regardless of outcome. Axes start at zero; no negative coordinate is introduced."
    )
    caption_path = out / "captions/figure_03_shared_scene_3d_trajectories_caption.txt"
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return {"figure": 3, "title": "Shared-scene 3-D trajectories", "pdf": pdf, "png": png, "caption": caption_path}


def generate(root: Path) -> None:
    setup_style()
    out = root / "13_paper_ready"
    figures = [learning_curve(root, out), success_comparison(root, out), shared_scene_trajectories(root, out)]
    manifest_rows = []
    for item in figures:
        manifest_rows.append(
            {
                "figure": item["figure"],
                "title": item["title"],
                "pdf": item["pdf"].relative_to(root).as_posix(),
                "png_600dpi": item["png"].relative_to(root).as_posix(),
                "caption": item["caption"].relative_to(root).as_posix(),
                "status": "YES",
            }
        )
    write_csv(out / "PPO_DIRECT_FIGURE_MANIFEST.csv", manifest_rows)
    print(json.dumps({"figure_count": len(figures), "manifest": str((out / 'PPO_DIRECT_FIGURE_MANIFEST.csv').relative_to(root))}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    generate(root)


if __name__ == "__main__":
    main()
