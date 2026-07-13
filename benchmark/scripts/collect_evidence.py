#!/usr/bin/env python3
"""Collect SEW-Offload benchmark evidence from unit artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


BYTES_PER_GIB = 1024**3


WEIGHTS_RE = re.compile(r"Loading model weights took ([\-0-9.]+) GB")
KV_RE = re.compile(r"Available KV cache memory: ([\-0-9.]+) GiB")
KV_TOKENS_RE = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")
CONCURRENCY_RE = re.compile(
    r"Maximum concurrency for ([0-9,]+) tokens per request: ([\-0-9.]+)x"
)


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def _read_text(path: str | Path) -> str:
    target = Path(path)
    if not target.exists():
        return ""
    return target.read_text(encoding="utf-8", errors="replace")


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    return float(str(value).replace(",", ""))


def _last_int(pattern: re.Pattern[str], text: str) -> int | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[0]
    return int(str(value).replace(",", ""))


def _classify_failure(text: str, result: dict[str, Any]) -> str:
    if result.get("status") == "ok":
        return ""
    if "No available memory for the cache blocks" in text:
        return "kv_cache_capacity_failure"
    if "capture model contains a stream that was not joined" in text:
        return "aclgraph_unjoined_stream"
    if "Not allow to synchronize captured-stream" in text:
        return "captured_stream_synchronization"
    if result.get("error"):
        return str(result["error"])
    return "failed"


def _profile_summary(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    summary: dict[str, Any] = {
        "profile_records": 0,
        "host_store_gib": None,
        "slot_bank_gib": None,
        "total_managed_gib": None,
        "registered_layers": None,
        "num_slots": None,
        "h2d_gib_total": 0.0,
        "stage_ms_total": 0.0,
        "max_active_experts": 0,
        "max_wave_count": 0,
        "b2_events": 0,
        "decode_stage_events": 0,
    }
    if not target.exists():
        return summary

    h2d_bytes_total = 0
    stage_ms_total = 0.0
    with target.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            summary["profile_records"] += 1
            ledger = record.get("memory_ledger") or {}
            if ledger:
                summary["host_store_gib"] = ledger.get("host_store_bytes", 0) / BYTES_PER_GIB
                summary["slot_bank_gib"] = ledger.get("slot_bank_bytes", 0) / BYTES_PER_GIB
                summary["total_managed_gib"] = ledger.get("total_managed_bytes", 0) / BYTES_PER_GIB
                summary["registered_layers"] = ledger.get("registered_layers")

            payload = record.get("payload") or {}
            name = str(record.get("name", ""))
            if "num_slots" in payload:
                summary["num_slots"] = payload.get("num_slots")
            if "n_active" in payload:
                summary["max_active_experts"] = max(
                    int(summary["max_active_experts"]), int(payload.get("n_active") or 0)
                )
            if "n_waves" in payload:
                summary["max_wave_count"] = max(
                    int(summary["max_wave_count"]), int(payload.get("n_waves") or 0)
                )
            if name == "b2_work_conserving_prefill":
                summary["b2_events"] += 1
            if name == "decode_fixed_slot_stage":
                summary["decode_stage_events"] += 1

            sample_rate = int(payload.get("profile_sample_rate") or 1)
            h2d_bytes_total += int(payload.get("h2d_bytes") or 0) * sample_rate
            if "stage_ms" in payload:
                stage_ms_total += float(payload.get("stage_ms") or 0.0) * sample_rate
            wave_summary = payload.get("wave_summary") or {}
            h2d_bytes_total += int(wave_summary.get("h2d_bytes") or 0)
            if "stage_ms" in wave_summary:
                stage_ms_total += float(wave_summary.get("stage_ms") or 0.0)
            if "wave_count" in wave_summary:
                summary["max_wave_count"] = max(
                    int(summary["max_wave_count"]), int(wave_summary.get("wave_count") or 0)
                )

    summary["h2d_gib_total"] = h2d_bytes_total / BYTES_PER_GIB
    summary["stage_ms_total"] = stage_ms_total
    return summary


def collect_unit(path: str | Path) -> dict[str, Any]:
    result_path = Path(path)
    result = _read_json(result_path)
    benchmark = _read_json(result.get("benchmark_json", ""))
    server_log = _read_text(result.get("server_log", ""))
    profile = _profile_summary(result.get("profile_jsonl", ""))
    max_model_len = None
    max_concurrency = None
    matches = CONCURRENCY_RE.findall(server_log)
    if matches:
        max_model_len = int(str(matches[-1][0]).replace(",", ""))
        max_concurrency = float(matches[-1][1])

    row = {
        "case": result.get("case", {}).get("name"),
        "workload": result.get("workload", {}).get("name"),
        "status": result.get("status"),
        "stage": result.get("stage"),
        "failure_reason": _classify_failure(server_log, result),
        "successful_requests": benchmark.get("successful_requests", 0),
        "failed_requests": benchmark.get("failed_requests", 0),
        "median_ttft_ms": benchmark.get("median_ttft_ms", 0.0),
        "median_tpot_ms": benchmark.get("median_tpot_ms", 0.0),
        "output_throughput_tok_s": benchmark.get("output_throughput", 0.0),
        "request_throughput": benchmark.get("request_throughput", 0.0),
        "weights_gb": _last_float(WEIGHTS_RE, server_log),
        "available_kv_gib": _last_float(KV_RE, server_log),
        "kv_cache_tokens": _last_int(KV_TOKENS_RE, server_log),
        "max_model_len": max_model_len,
        "max_concurrency": max_concurrency,
        "graph_capture_completed": "Graph capturing finished" in server_log,
        "moe_offload_stage_seen": "vllm::moe_offload_stage" in server_log,
        "prefetch_offloader_seen": "PrefetchOffloader" in server_log,
        "unit_result": str(result_path),
        "server_log": str(result.get("server_log", "")),
        "benchmark_json": str(result.get("benchmark_json", "")),
        "profile_jsonl": str(result.get("profile_jsonl", "")),
    }
    row.update(profile)
    return row


def discover_unit_results(paths: list[str]) -> list[Path]:
    results: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_file():
            results.append(path)
        elif path.is_dir():
            results.extend(sorted(path.glob("*/*/unit_result.json")))
        else:
            raise FileNotFoundError(path)
    return sorted(dict.fromkeys(results))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _label(row: dict[str, Any]) -> str:
    return str(row.get("case", "")).replace("sew_", "").replace("_", "\n")


def _svg_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_svg_bar_chart(
    path: Path,
    *,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
    zero_line: bool = False,
) -> None:
    width = 760
    height = 330
    margin_left = 70
    margin_right = 20
    margin_top = 42
    margin_bottom = 92
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_v = max([0.0, *values])
    min_v = min([0.0, *values]) if zero_line else 0.0
    if max_v == min_v:
        max_v = min_v + 1.0
    scale = plot_h / (max_v - min_v)
    zero_y = margin_top + max_v * scale
    bar_w = plot_w / max(1, len(values)) * 0.66
    palette = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-family="Arial, sans-serif" font-size="15">{_svg_escape(title)}</text>',
        f'<text x="16" y="{margin_top + plot_h / 2:.1f}" transform="rotate(-90 16 {margin_top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{_svg_escape(ylabel)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<text x="{margin_left - 8}" y="{margin_top + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="10">{max_v:.2f}</text>',
        f'<text x="{margin_left - 8}" y="{margin_top + plot_h + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="10">{min_v:.2f}</text>',
    ]
    if zero_line:
        parts.append(
            f'<line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width - margin_right}" y2="{zero_y:.1f}" stroke="#555" stroke-width="1" stroke-dasharray="4 3"/>'
        )
    for idx, (label, value) in enumerate(zip(labels, values)):
        center = margin_left + plot_w / len(values) * (idx + 0.5)
        x = center - bar_w / 2
        if value >= 0:
            y = margin_top + (max_v - value) * scale
            h = max(1.0, zero_y - y)
        else:
            y = zero_y
            h = max(1.0, (0 - value) * scale)
        color = palette[idx % len(palette)]
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{center:.1f}" y="{max(12, y - 5):.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">{value:.2f}</text>'
        )
        label_lines = label.split("\n")
        for line_idx, line in enumerate(label_lines[:4]):
            parts.append(
                f'<text x="{center:.1f}" y="{margin_top + plot_h + 18 + line_idx * 12}" text-anchor="middle" font-family="Arial, sans-serif" font-size="9">{_svg_escape(line)}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_svg_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if ok_rows:
        labels = [_label(row) for row in ok_rows]
        _write_svg_bar_chart(
            output_dir / "week2_smoke_ttft.svg",
            title="Week 2 smoke TTFT",
            labels=labels,
            values=[float(row.get("median_ttft_ms") or 0.0) for row in ok_rows],
            ylabel="TTFT (ms)",
        )
        _write_svg_bar_chart(
            output_dir / "week2_smoke_tpot.svg",
            title="Week 2 smoke TPOT",
            labels=labels,
            values=[float(row.get("median_tpot_ms") or 0.0) for row in ok_rows],
            ylabel="TPOT (ms/token)",
        )
        _write_svg_bar_chart(
            output_dir / "week2_smoke_throughput.svg",
            title="Week 2 smoke output throughput",
            labels=labels,
            values=[float(row.get("output_throughput_tok_s") or 0.0) for row in ok_rows],
            ylabel="Output throughput (tok/s)",
        )

    labels = [_label(row) for row in rows]
    _write_svg_bar_chart(
        output_dir / "week2_smoke_weights.svg",
        title="Week 2 smoke model-weight memory",
        labels=labels,
        values=[float(row.get("weights_gb") or 0.0) for row in rows],
        ylabel="Model weights on NPU (GB)",
    )
    _write_svg_bar_chart(
        output_dir / "week2_smoke_kv_cache.svg",
        title="Week 2 smoke available KV cache",
        labels=labels,
        values=[float(row.get("available_kv_gib") or 0.0) for row in rows],
        ylabel="Available KV cache (GiB)",
        zero_line=True,
    )
    h2d_rows = [row for row in rows if float(row.get("h2d_gib_total") or 0.0) > 0]
    if h2d_rows:
        _write_svg_bar_chart(
            output_dir / "week2_smoke_h2d.svg",
            title="Week 2 smoke profiled H2D traffic",
            labels=[_label(row) for row in h2d_rows],
            values=[float(row.get("h2d_gib_total") or 0.0) for row in h2d_rows],
            ylabel="Profiled H2D traffic (GiB)",
        )


def write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        write_svg_plots(output_dir, rows)
        return

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    palette = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )

    if ok_rows:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), constrained_layout=True)
        metrics = [
            ("median_ttft_ms", "TTFT (ms)"),
            ("median_tpot_ms", "TPOT (ms/token)"),
            ("output_throughput_tok_s", "Output throughput (tok/s)"),
        ]
        labels = [_label(row) for row in ok_rows]
        for ax, (key, ylabel) in zip(axes, metrics):
            values = [float(row.get(key) or 0.0) for row in ok_rows]
            ax.bar(labels, values, color=palette[: len(values)])
            ax.set_ylabel(ylabel)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", rotation=0)
        fig.suptitle("Week 2 smoke serving metrics")
        fig.savefig(output_dir / "week2_smoke_serving_metrics.png", dpi=300)
        fig.savefig(output_dir / "week2_smoke_serving_metrics.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), constrained_layout=True)
    labels = [_label(row) for row in rows]
    weights = [float(row.get("weights_gb") or 0.0) for row in rows]
    kv = [float(row.get("available_kv_gib") or 0.0) for row in rows]
    axes[0].bar(labels, weights, color=palette[0])
    axes[0].set_ylabel("Model weights on NPU (GB)")
    axes[1].bar(labels, kv, color=palette[1])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Available KV cache (GiB)")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Week 2 smoke memory evidence")
    fig.savefig(output_dir / "week2_smoke_memory_evidence.png", dpi=300)
    fig.savefig(output_dir / "week2_smoke_memory_evidence.pdf")
    plt.close(fig)

    h2d_rows = [row for row in rows if float(row.get("h2d_gib_total") or 0.0) > 0]
    if h2d_rows:
        fig, ax = plt.subplots(figsize=(4.2, 2.4), constrained_layout=True)
        labels = [_label(row) for row in h2d_rows]
        values = [float(row.get("h2d_gib_total") or 0.0) for row in h2d_rows]
        ax.bar(labels, values, color=palette[2])
        ax.set_ylabel("Profiled H2D traffic (GiB)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.suptitle("Week 2 smoke H2D evidence")
        fig.savefig(output_dir / "week2_smoke_h2d_evidence.png", dpi=300)
        fig.savefig(output_dir / "week2_smoke_h2d_evidence.pdf")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Run dirs or unit_result.json files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    unit_results = discover_unit_results(args.paths)
    rows = [collect_unit(path) for path in unit_results]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "week2_smoke_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(output_dir / "week2_smoke_summary.csv", rows)
    if not args.no_plots:
        write_plots(output_dir, rows)
    print(f"Wrote {len(rows)} evidence rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
