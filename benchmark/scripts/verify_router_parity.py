#!/usr/bin/env python3
"""Fail closed when native and LatchMoE router artifacts diverge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_moe_offload_ascend.moe_offload.router_parity import (
    compare_router_artifacts,
    load_router_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--seam", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    report = compare_router_artifacts(
        load_router_artifact(args.native, role="native"),
        load_router_artifact(args.seam, role="seam"),
        atol=args.atol,
        rtol=args.rtol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
