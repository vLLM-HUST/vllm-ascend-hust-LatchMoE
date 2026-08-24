#!/usr/bin/env python3
"""Plot the audited online slot-capacity sensitivity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import BLUE, GREEN, ORANGE, configure, finish_axes, save


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if data.get("status") != "passed":
        raise SystemExit("formal campaign audit has not passed")
    rows = data["capacity"]["rows"]
    slots = np.asarray([int(row["slots"]) for row in rows])

    configure()
    fig, (latency_ax, pressure_ax) = plt.subplots(1, 2, figsize=(7.0, 2.25))
    latency_ax.plot(slots, [row["ttft_p50_ms"] for row in rows], color=BLUE,
                    marker="o", label="TTFT p50")
    latency_ax.plot(slots, [row["ttft_p95_ms"] for row in rows], color=ORANGE,
                    marker="s", linestyle="--", label="TTFT p95")
    latency_ax.set_xticks(slots)
    latency_ax.set_xlabel("Physical slots")
    latency_ax.set_ylabel("TTFT (ms)")
    latency_ax.set_title("(a) Descriptive prefill latency", loc="left", fontweight="bold")
    latency_ax.legend(frameon=False, ncol=2, loc="best")
    finish_axes(latency_ax)

    waves = np.asarray([
        float(row["waves_total"]) / max(1, int(row["wave_events"])) for row in rows
    ])
    storage = np.asarray([float(row["slot_bank_bytes"]) / 1024**3 for row in rows])
    pressure_ax.plot(slots, waves, color=BLUE, marker="o", label="Waves/invocation")
    pressure_ax.set_xticks(slots)
    pressure_ax.set_xlabel("Physical slots")
    pressure_ax.set_ylabel("Mean waves / invocation", color=BLUE)
    pressure_ax.tick_params(axis="y", labelcolor=BLUE)
    pressure_ax.set_title("(b) Reuse–storage tradeoff", loc="left", fontweight="bold")
    finish_axes(pressure_ax)
    storage_ax = pressure_ax.twinx()
    storage_ax.plot(slots, storage, color=GREEN, marker="^", linestyle="--",
                    label="Slot-bank storage")
    storage_ax.set_ylabel("Persistent slot bank (GiB)", color=GREEN)
    storage_ax.tick_params(axis="y", labelcolor=GREEN, direction="out", length=3, width=0.8)
    storage_ax.spines["top"].set_visible(False)
    handles = pressure_ax.get_lines() + storage_ax.get_lines()
    pressure_ax.legend(handles, [line.get_label() for line in handles],
                       frameon=False, loc="best")

    fig.tight_layout(w_pad=2.2)
    save(fig, Path(args.output_stem))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
