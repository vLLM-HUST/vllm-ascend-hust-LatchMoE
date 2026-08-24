#!/usr/bin/env python3
"""Render the two-panel Motivation figure from the independent audit JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


MODELS = ["Qwen3\n30B-A3B", "GLM-4.7\nFlash", "Qwen3-Next\n80B-A3B"]
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#666666"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("paper/data/audits/motivation_reaudit.json"),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("paper/figures/motivation_characterization"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("status") != "verified" or len(audit.get("models", [])) != 3:
        raise ValueError("Motivation audit must contain three verified models")
    rows = audit["models"]

    miss_bearing = np.array([
        row["paper_facing"]["decode_miss_bearing_rate_pct"] for row in rows
    ])
    update_bearing = np.array([
        row["paper_facing"]["decode_update_bearing_rate_pct"] for row in rows
    ])
    active = [row["paper_facing"]["capacity_32"]["active_hbm_bytes"] for row in rows]
    active_experts = [row["paper_facing"]["prefill_active_experts"] for row in rows]
    p50 = np.array([item["p50_nearest_rank"] for item in active_experts], dtype=float) / 32.0
    p95 = np.array([item["p95_nearest_rank"] for item in active_experts], dtype=float) / 32.0
    maximum = np.array([item["max"] for item in active_experts], dtype=float) / 32.0

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8.3,
        "ytick.labelsize": 8.3,
        "legend.fontsize": 8.3,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "latchmoe-motivation-v1",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), constrained_layout=True)

    x = np.arange(3)
    width = 0.32
    ax = axes[0]
    bars_a = ax.bar(
        x - width / 2,
        miss_bearing,
        width,
        color=BLUE,
        edgecolor="#111111",
        linewidth=0.5,
        hatch="///",
        label="At least one cache miss",
        zorder=3,
    )
    bars_b = ax.bar(
        x + width / 2,
        update_bearing,
        width,
        color=ORANGE,
        edgecolor="#111111",
        linewidth=0.5,
        hatch="...",
        label="Mapping updated",
        zorder=3,
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Decode layer invocations (%)")
    ax.set_xticks(x, MODELS)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    for bars in (bars_a, bars_b):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.8,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                color="#222222",
            )
    ax.text(-0.18, 1.03, "(a) Residency changes despite sparse decode", transform=ax.transAxes,
            fontweight="bold", ha="left", va="bottom")

    ax = axes[1]
    ax.axhline(1.0, color="#A0A0A0", linewidth=1.2, linestyle="--", zorder=1)
    ax.fill_between([-0.45, 2.45], 0.75, 1.0, color="#F2F2F2", zorder=0)
    # Models are categorical configurations, so markers are deliberately not
    # connected. Small offsets keep equal values (GLM p95=max) visible.
    offsets = (-0.10, 0.0, 0.10)
    ax.scatter(x + offsets[0], p50, marker="o", s=38, color=BLUE,
               edgecolor="white", linewidth=0.5, label="p50", zorder=3)
    ax.scatter(x + offsets[1], p95, marker="s", s=34, color=ORANGE,
               edgecolor="white", linewidth=0.5, label="p95", zorder=3)
    ax.scatter(x + offsets[2], maximum, marker="^", s=40, color=GREEN,
               edgecolor="white", linewidth=0.5, label="maximum", zorder=3)
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.75, 20)
    ax.set_yticks([1, 2, 4, 8, 16], labels=["1×", "2×", "4×", "8×", "16×"])
    ax.set_ylabel("Prefill active experts / 32 slots")
    ax.set_xticks(x, MODELS)
    ax.set_xlim(-0.45, 2.45)
    ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left", columnspacing=0.8,
              handletextpad=0.4)
    ax.text(2.42, 1.05, "slot capacity", color=GRAY, fontsize=8.2, ha="right", va="bottom")
    ax.text(-0.18, 1.03, "(b) One-request prefill exceeds 32 slots", transform=ax.transAxes,
            fontweight="bold", ha="left", va="bottom")

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    common = {"bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(
        args.output_stem.with_suffix(".pdf"),
        metadata={"Creator": "LatchMoE deterministic plot script", "CreationDate": None, "ModDate": None},
        **common,
    )
    fig.savefig(
        args.output_stem.with_suffix(".svg"),
        metadata={"Creator": "LatchMoE deterministic plot script", "Date": None},
        **common,
    )
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=240, **common)
    plt.close(fig)


if __name__ == "__main__":
    main()
