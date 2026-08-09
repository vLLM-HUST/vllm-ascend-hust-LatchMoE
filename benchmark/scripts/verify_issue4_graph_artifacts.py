#!/usr/bin/env python3
"""Fail closed unless a LatchMoE run contains graph-path evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    console = (run_dir / "console.log").read_text(encoding="utf-8", errors="replace")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    profile_path = run_dir / "moe_offload_profile.jsonl"
    profile_events = [
        json.loads(line)
        for line in profile_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = {
        "seam_guard_pass": _count(r"SEW_SEAM_SELECT .*config_guard=PASS", console),
        "seam_guard_fail": _count(r"SEW_SEAM_SELECT .*config_guard=FAIL", console),
        "graph_slot_probe": _count(r"GRAPH_COMPAT_SLOT", console),
        "capture_slot_probe": _count(r"GRAPH_COMPAT_SLOT .*capturing=True", console),
        "capture_weights_none": _count(r"GRAPH_COMPAT_SLOT .*capture_weights_none=True", console),
        "aclgraph_replay": _count(r"Replaying aclgraph", console),
        "forbidden_path_marker": _count(r"\b(?:FALLBACK|BYPASS|FULL_WEIGHT|NATIVE)\b", console),
        "h2d_events": sum(
            int(event.get("payload", {}).get("h2d_bytes", 0)) > 0
            for event in profile_events
        ),
    }
    failures: list[str] = []
    if summary.get("status") != "ok":
        failures.append(f"summary status is {summary.get('status')!r}")
    for key in ("seam_guard_pass", "graph_slot_probe", "capture_slot_probe", "aclgraph_replay", "h2d_events"):
        if counts[key] <= 0:
            failures.append(f"missing required evidence: {key}")
    for key in ("seam_guard_fail", "capture_weights_none", "forbidden_path_marker"):
        if counts[key] != 0:
            failures.append(f"forbidden evidence observed: {key}={counts[key]}")

    result = {
        "status": "passed" if not failures else "failed",
        "summary_status": summary.get("status"),
        "counts": counts,
        "failures": failures,
    }
    (run_dir / "graph_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
