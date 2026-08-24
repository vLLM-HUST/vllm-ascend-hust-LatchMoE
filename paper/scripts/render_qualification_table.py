#!/usr/bin/env python3
"""Render Evaluation Table 1 from the audited qualification summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DISPLAY = {
    "qwen3-30b-a3b": "Qwen3-30B-A3B",
    "glm-4.7-flash": "GLM-4.7-Flash",
    "qwen3-next-80b-a3b-instruct": "Qwen3-Next-80B-A3B",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if report.get("status") != "passed" or len(report.get("rows") or []) != 3:
        raise SystemExit("qualification audit must pass for exactly three models")
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{Three-model semantic and graph qualification. $L/E/S$ denotes managed MoE layers, routed experts per layer, and resident shared experts. Router/boundary counts are exact native comparisons; locks/replays are address locks and observed graph replays; overflow/H2D counts exercise wave composition and transfer leases. Every row passes all 11 gates, including exact end-to-end tokens and zero eager fallback.}",
        r"  \label{tab:qualification}",
        r"  \begin{tabular}{lrrrrrrc}",
        r"    \toprule",
        r"    Model & $L/E/S$ & top-$k$ & Router & Boundary & Locks/replays & Overflow/H2D & Result \\",
        r"    \midrule",
    ]
    for row in report["rows"]:
        result = "Pass" if row["status"] == "passed" else "Fail"
        lines.append(
            "    "
            + DISPLAY[row["model_id"]]
            + f" & {row['moe_layers']}/{row['routed_experts']}/{row['shared_experts']}"
            + f" & {row['top_k']} & {row['router_records']} & {row['layer_boundary_records']}"
            + f" & {row['graph_address_locks']}/{row['graph_replays']}"
            + f" & {row['overflow_events']}/{row['h2d_stages']} & {result} \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
