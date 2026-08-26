#!/usr/bin/env python3
"""Create the compact paper figure for recurrent-selector ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "artifacts/recurrent_selector_ablation_confirmation/20260822_143118"
OUT = ROOT / "10_paper_ready"
METHODS = (
    "M1_RERR_Proposal_SAC_DMP",
    "M8_RERR_FP_SHEP_SAC_DMP",
    "M9_Proposed_RERR_GAT_SAC_DMP",
)
LABELS = ("Proposal", "FP-SHEP", "GAT-R")
COLORS = ("#4C78A8", "#F58518", "#54A24B")
HATCHES = ("///", "\\\\", "...")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = read_csv(ROOT / "stage_summary.csv")
    overall = {row["method_id"]: row for row in summary if row["scope"] == "overall"}
    pfp = load_json(ROOT / "proposal_vs_fp_paired.json")["scopes"]["overall"]["team_success"]
    fpg = load_json(ROOT / "fp_vs_gat_paired.json")["scopes"]["overall"]["team_success"]
    metrics = (
        ("success_rate", "Team success", "Success rate (%)"),
        ("collision_rate", "Any collision", "Episode rate (%)"),
        ("inter_agent_collision_rate", "Inter-agent collision", "Episode rate (%)"),
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8), constrained_layout=True)
    source_rows: list[dict[str, object]] = []
    for axis, (field, title, ylabel) in zip(axes, metrics):
        values = np.asarray([100.0 * float(overall[method][field]) for method in METHODS])
        x = np.arange(3)
        bars = axis.bar(x, values, width=0.64, color=COLORS, edgecolor="#303030", linewidth=0.55)
        for bar, hatch, value, label, method in zip(bars, HATCHES, values, LABELS, METHODS):
            bar.set_hatch(hatch)
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(0.7, 0.018 * max(values.max(), 1.0)),
                f"{value:.2f}", ha="center", va="bottom", fontsize=8,
            )
            source_rows.append(
                {"panel": title, "metric": field, "method_id": method, "method": label, "rate_percent": value}
            )
        axis.set_title(title, pad=7)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, LABELS)
        ceiling = max(values.max() * 1.20, 8.0)
        if field == "success_rate":
            ceiling = min(100.0, max(100.0, ceiling))
        axis.set_ylim(0.0, ceiling)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", length=0)
    axes[0].text(
        0.5, 0.06,
        f"FP − Proposal: {pfp['right_minus_left_rate_pp']:+.2f} pp\n"
        f"GAT-R − FP: {fpg['right_minus_left_rate_pp']:+.2f} pp",
        transform=axes[0].transAxes, ha="center", va="bottom", fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "white", "edgecolor": "#A0A0A0", "linewidth": 0.55},
    )
    fig.suptitle("Independent recurrent-selector ablation (400 matched scenarios)", fontsize=11, y=1.03)
    pdf = OUT / "selector_ablation_figure.pdf"
    png = OUT / "selector_ablation_figure.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(ROOT / "selector_ablation_figure.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "selector_ablation_figure.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    with (OUT / "selector_ablation_figure_source_data.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    caption = (
        "Figure. Independent recurrent-selector ablation on 400 newly generated matched scenarios. "
        "All arms use the identical frozen Proposal pool, R-ERR trigger/lifecycle contract, SAC-DMP "
        "executor, physical limits, and outcome definitions; only the final selector differs. Bars show "
        "overall team success, any-collision, and inter-agent-collision rates. Exact paired McNemar tests "
        f"give FP-SHEP minus Proposal {pfp['right_minus_left_rate_pp']:+.2f} percentage points "
        f"(p={pfp['exact_two_sided_mcnemar_p']:.3g}) and GAT-R minus FP-SHEP "
        f"{fpg['right_minus_left_rate_pp']:+.2f} percentage points "
        f"(p={fpg['exact_two_sided_mcnemar_p']:.3g}). Non-significance is not interpreted as equivalence."
    )
    (ROOT / "paper_caption.md").write_text(caption + "\n", encoding="utf-8")
    (OUT / "paper_caption.md").write_text(caption + "\n", encoding="utf-8")
    conclusion_path = ROOT / "conclusion.json"
    conclusion = load_json(conclusion_path)
    conclusion["PAPER_FIGURES_READY"] = "YES"
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plot": "PASS", "pdf": str(pdf), "png": str(png)}))


if __name__ == "__main__":
    main()
