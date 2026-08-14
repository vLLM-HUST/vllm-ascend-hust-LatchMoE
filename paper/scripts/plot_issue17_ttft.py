#!/usr/bin/env python3
"""Regenerate the matched Issue 17 TTFT figure from the checked-in bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile

import matplotlib.pyplot as plt
import numpy as np


SUMMARY_PATH = "issue-17-matched-ttft/matched_summary.original.json"


def load_summary(bundle_path: Path) -> dict[str, object]:
    with tarfile.open(bundle_path, "r:gz") as archive:
        member = archive.extractfile(SUMMARY_PATH)
        if member is None:
            raise RuntimeError(f"{SUMMARY_PATH} is missing from {bundle_path}")
        return json.load(member)


def min_max(values: list[float], center: float) -> tuple[float, float]:
    return center - min(values), max(values) - center


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = load_summary(args.bundle)
    arms = summary["arms"]
    full = arms["full_layer"]
    wave = arms["multi_wave"]

    full_values = np.array(
        [full["ttft_p50_ms_median"], full["ttft_p95_ms_median"]],
        dtype=float,
    )
    wave_values = np.array(
        [wave["ttft_p50_ms_median"], wave["ttft_p95_ms_median"]],
        dtype=float,
    )
    full_ranges = [
        min_max(full["ttft_p50_ms_by_repeat"], full_values[0]),
        min_max(full["ttft_p95_ms_by_repeat"], full_values[1]),
    ]
    wave_ranges = [
        min_max(wave["ttft_p50_ms_by_repeat"], wave_values[0]),
        min_max(wave["ttft_p95_ms_by_repeat"], wave_values[1]),
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.15), constrained_layout=True)
    x = np.arange(2)
    width = 0.34
    error_style = {"elinewidth": 0.8, "capsize": 2.5, "capthick": 0.8}

    full_bars = ax.bar(
        x - width / 2,
        full_values,
        width,
        label="Full-layer",
        color="#999999",
        edgecolor="#333333",
        linewidth=0.7,
        yerr=np.array(full_ranges).T,
        error_kw=error_style,
    )
    wave_bars = ax.bar(
        x + width / 2,
        wave_values,
        width,
        label="Multi-wave",
        color="#0072B2",
        edgecolor="#003B5C",
        linewidth=0.7,
        yerr=np.array(wave_ranges).T,
        error_kw=error_style,
    )

    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x, ["TTFT p50", "TTFT p95"])
    ax.set_ylim(0, 1120)
    ax.set_yticks(np.arange(0, 1101, 200))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.bar_label(full_bars, fmt="%.0f", padding=4, fontsize=7)
    ax.bar_label(wave_bars, fmt="%.0f", padding=4, fontsize=7)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()

