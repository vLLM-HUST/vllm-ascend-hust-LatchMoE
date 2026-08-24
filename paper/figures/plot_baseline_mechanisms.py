#!/usr/bin/env python3
"""Plot full-resident cost and matched mechanism effects from audited data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from paper_plot_style import BLUE, GRAY, ORANGE, configure, finish_axes, save


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if data.get("status") != "passed":
        raise SystemExit("formal campaign audit has not passed")

    configure()
    fig, (cost_ax, effect_ax) = plt.subplots(
        1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [0.8, 1.35]}
    )

    arms = data["baseline"]["arms"]
    ratios = []
    for metric in ("ttft_p50_ms", "tpot_p50_ms"):
        ratios.append([
            float(latch[metric]) / float(full[metric])
            for latch, full in zip(arms["latchmoe"], arms["full_resident"], strict=True)
        ])
    x = np.arange(2)
    medians = [float(np.median(values)) for values in ratios]
    cost_ax.bar(
        x, medians, width=0.55, color=ORANGE, edgecolor="#333333",
        linewidth=0.8, hatch="//", zorder=2, label="LatchMoE",
    )
    offsets = (-0.11, 0.0, 0.11)
    for metric_index, values in enumerate(ratios):
        for repeat, (offset, value) in enumerate(zip(offsets, values, strict=True), 1):
            cost_ax.plot(metric_index + offset, value, marker="o", color=BLUE,
                         markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        cost_ax.text(metric_index, medians[metric_index] + 0.10,
                     f"{medians[metric_index]:.2f}$\\times$", ha="center", va="bottom")
    cost_ax.axhline(1.0, color=GRAY, linewidth=1.0, linestyle="--", zorder=1)
    cost_ax.text(1.48, 1.04, "full resident", color=GRAY, ha="right", va="bottom")
    cost_ax.set_xticks(x, ["TTFT p50", "TPOT p50"])
    cost_ax.set_ylabel("Latency / full-resident latency")
    cost_ax.set_ylim(0, max(medians) * 1.22)
    cost_ax.set_title("(a) Cost of bounded offloading", loc="left", fontweight="bold")
    finish_axes(cost_ax)

    issue = data["issue17"]["paired_statistics"]
    overlap = data["overlap"]["statistics"]
    rows = [
        ("Bounded waves, TTFT", issue["ttft"], BLUE, "o"),
        ("Overlap, TTFT", overlap["ttft"], BLUE, "s"),
        ("Bounded waves, TPOT", issue["tpot"], ORANGE, "o"),
        ("Overlap, TPOT", overlap["tpot"], ORANGE, "s"),
    ]
    y = np.arange(len(rows))[::-1]
    effect_ax.add_patch(Rectangle((-5, -0.5), 10, 2.0, facecolor="#EEEEEE",
                                  edgecolor="none", zorder=0))
    effect_ax.axvline(0, color=GRAY, linewidth=0.9, zorder=1)
    effect_ax.vlines([-5, 5], -0.5, 1.5, color=GRAY, linewidth=0.7,
                     linestyle=":", zorder=1)
    for pos, (label, result, color, marker) in zip(y, rows, strict=True):
        estimate = 100.0 * float(result["estimate"])
        interval = result["interval"]
        low = 100.0 * float(interval["lower"])
        high = 100.0 * float(interval["upper"])
        effect_ax.errorbar(
            estimate, pos, xerr=[[estimate - low], [high - estimate]],
            fmt=marker, color=color, ecolor=color, capsize=2.5,
            markeredgecolor="white", markeredgewidth=0.5, zorder=3,
        )
        effect_ax.text(high + 0.8, pos, f"{estimate:+.1f}%", va="center", color=color)
    effect_ax.set_yticks(y, [row[0] for row in rows])
    effect_ax.set_xlabel("Paired latency effect (lower is better)")
    effect_ax.set_xlim(-40, 9)
    effect_ax.set_title("(b) One-factor mechanism effects", loc="left", fontweight="bold")
    effect_ax.text(0, 1.48, "TPOT equivalence band", ha="center", va="bottom",
                   color=GRAY, fontsize=8.2)
    finish_axes(effect_ax)
    effect_ax.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.75, zorder=0)
    effect_ax.grid(axis="y", visible=False)

    fig.tight_layout(w_pad=2.0)
    save(fig, Path(args.output_stem))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
