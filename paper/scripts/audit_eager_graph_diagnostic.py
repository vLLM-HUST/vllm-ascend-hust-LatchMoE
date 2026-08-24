#!/usr/bin/env python3
"""Audit the matched one-request eager-versus-PIECEWISE diagnostic.

This diagnostic is deliberately narrower than the formal serving campaigns:
it checks the same fixed-slot offline workload in eager and graph modes, then
records the exact output-token match and the latency difference.  It must not
be interpreted as a multi-request performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events


def _checks(mode: str, summary: dict[str, Any], events: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    if summary.get("status") != "ok":
        failures.append(f"{mode}: summary status is not ok")
    if int(summary.get("completed", 0)) != 1:
        failures.append(f"{mode}: expected exactly one completed request")
    if int(summary.get("total_output_tokens", 0)) != 4:
        failures.append(f"{mode}: expected exactly four output tokens")
    if not summary.get("graph_compatible_offload"):
        failures.append(f"{mode}: fixed-slot graph-compatible flag is absent")
    diagnostics = summary.get("qualification_diagnostics") or {}
    for key in ("stage_seam", "cpu_first_load", "release_original_expert_weights"):
        if not diagnostics.get(key):
            failures.append(f"{mode}: diagnostic gate {key} is absent")
    if mode == "eager" and not diagnostics.get("diagnostic_eager"):
        failures.append("eager: explicit diagnostic-eager gate is absent")
    if mode == "graph" and diagnostics.get("diagnostic_eager"):
        failures.append("graph: eager diagnostic gate is unexpectedly enabled")
    if not any(event.get("name") == "graph_slot_address_validate" for event in events):
        failures.append(f"{mode}: no replay-visible slot-address validation event")
    if mode == "graph":
        replay = [event for event in events if event.get("name") == "graph_replay_issue"]
        if not replay:
            failures.append("graph: no PIECEWISE replay event")
        elif not any(
            (event.get("payload") or {}).get("runtime_mode") == "PIECEWISE"
            and (event.get("payload") or {}).get("synchronizes_npu") is False
            for event in replay
        ):
            failures.append("graph: replay event did not record PIECEWISE/non-synchronizing mode")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager-dir", required=True, type=Path)
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    roots = {"eager": args.eager_dir, "graph": args.graph_dir}
    summaries: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, Any]] = {}
    events: dict[str, list[dict[str, Any]]] = {}
    failures: list[str] = []
    source_files: dict[str, dict[str, str]] = {}
    for mode, root in roots.items():
        summary_path = root / "summary.json"
        output_path = root / "outputs.jsonl"
        profile_path = root / "moe_offload_profile.jsonl"
        if not all(path.is_file() for path in (summary_path, output_path, profile_path)):
            failures.append(f"{mode}: required raw file is missing")
            continue
        summary = _load(summary_path)
        output_lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(output_lines) != 1:
            failures.append(f"{mode}: expected exactly one output record")
            output = {}
        else:
            output = json.loads(output_lines[0])
        profile = _profile_events(profile_path)
        summaries[mode] = summary
        outputs[mode] = output
        events[mode] = profile
        failures.extend(_checks(mode, summary, profile))
        source_files[mode] = {
            "summary": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "outputs": str(output_path),
            "outputs_sha256": _sha256(output_path),
            "profile": str(profile_path),
            "profile_sha256": _sha256(profile_path),
        }

    eager_output = outputs.get("eager", {})
    graph_output = outputs.get("graph", {})
    eager_ids = eager_output.get("output_token_ids") or []
    graph_ids = graph_output.get("output_token_ids") or []
    token_match = eager_ids == graph_ids and bool(eager_ids)
    if not token_match:
        failures.append("eager/graph: output token IDs do not match exactly")

    eager_summary = summaries.get("eager", {})
    graph_summary = summaries.get("graph", {})
    eager_ttft = float((eager_summary.get("ttft_ms") or {}).get("median", 0.0))
    graph_ttft = float((graph_summary.get("ttft_ms") or {}).get("median", 0.0))
    eager_tpot = float((eager_summary.get("tpot_ms") or {}).get("median", 0.0))
    graph_tpot = float((graph_summary.get("tpot_ms") or {}).get("median", 0.0))
    if not all(value > 0 for value in (eager_ttft, graph_ttft, eager_tpot, graph_tpot)):
        failures.append("eager/graph: latency medians are incomplete")

    payload = {
        "schema_version": "latchmoe-eager-graph-diagnostic-v1",
        "status": "passed" if not failures else "failed",
        "scope": {
            "requests_per_mode": 1,
            "output_tokens": 4,
            "concurrency": 1,
            "model": eager_summary.get("model"),
            "device_visibility": "ASCEND_RT_VISIBLE_DEVICES=6",
            "slots": eager_summary.get("num_slots"),
            "offload_gb": 14,
            "interpretation": "qualification diagnostic, not a variance or throughput campaign",
        },
        "exact_output_token_match": token_match,
        "output_token_ids_sha256": hashlib.sha256(
            json.dumps(eager_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest() if eager_ids else None,
        "latency_ms": {
            "eager": {"ttft_median": eager_ttft, "tpot_median": eager_tpot},
            "graph": {"ttft_median": graph_ttft, "tpot_median": graph_tpot},
            "graph_relative_effect": {
                "ttft": graph_ttft / eager_ttft - 1.0 if eager_ttft else None,
                "tpot": graph_tpot / eager_tpot - 1.0 if eager_tpot else None,
            },
        },
        "profile_event_counts": {
            mode: {
                "total": len(events.get(mode, [])),
                "graph_slot_address_validate": sum(
                    event.get("name") == "graph_slot_address_validate"
                    for event in events.get(mode, [])
                ),
                "graph_replay_issue": sum(
                    event.get("name") == "graph_replay_issue"
                    for event in events.get(mode, [])
                ),
            }
            for mode in ("eager", "graph")
        },
        "source_files": source_files,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output), "failures": failures}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
