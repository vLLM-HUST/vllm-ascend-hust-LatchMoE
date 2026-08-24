#!/usr/bin/env python3
"""Render the evaluation resource table from the audited ledger JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GIB = 1024**3
MODEL_LABELS = {
    "qwen3-30b-a3b": "Qwen3-30B-A3B",
    "glm-4.7-flash": "GLM-4.7-Flash",
    "qwen3-next-80b-a3b-instruct": "Qwen3-Next-80B-A3B",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise SystemExit("resource ledger audit has not passed")

    rows = []
    for item in payload["rows"]:
        mode = item["modes"]["latch-graph"]
        if mode["num_slots"] != 32 or mode["original_expert_weights_retained"]:
            raise SystemExit(f"unexpected qualified ledger for {item['model_id']}")
        device_bytes = (
            mode["slot_bank_bytes"]
            + mode["prefill_stage_bank_bytes"]
            + mode["resident_shared_weight_bytes"]
        )
        rows.append(
            "    {} & {:.1f} & {:.1f} & {:.1f} & {} \\\\".format(
                MODEL_LABELS[item["model_id"]],
                mode["host_store_bytes"] / GIB,
                device_bytes / GIB,
                mode["h2d_bytes"] / GIB,
                mode["graph_replay_issues"],
            )
        )

    table = "\n".join([
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{Audited resources in each model's graph-qualified LatchMoE run. Device storage sums the 32-slot bank, prefill stage banks, and resident shared-expert weights; H2D is the profiled transfer volume for that qualification workload. Original managed expert banks are released in every row.}",
        r"  \label{tab:resources}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Model & Host & Device & H2D & Replays \\",
        r"          & (GiB) & (GiB) & (GiB) & \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
