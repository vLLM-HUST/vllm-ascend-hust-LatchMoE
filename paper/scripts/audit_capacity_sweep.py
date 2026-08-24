#!/usr/bin/env python3
"""Audit exactness, capacity, graph, and costs for the online slot sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


EXPECTED = {16: "sew_14gb_slots16", 32: "sew_14gb_slots32", 64: "sew_14gb_slots64"}
FORBIDDEN = ("address fingerprint changed", "stale H2D completion", "NPU out of memory", "ACL_ERROR", "EZ9999")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _percentile(values: list[float], percentile: float) -> float | None:
    """Return a linearly interpolated percentile without adding a dependency."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--expected-requests", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    campaign_path = Path(args.campaign).resolve()
    campaign = _read(campaign_path)
    failures = []
    rows = []
    oracle = None
    source_identity = None
    for item in campaign.get("units") or []:
        slots = int(item["slots"])
        unit = Path(item["unit_dir"])
        required = {
            name: unit / name for name in (
                "unit_manifest.json", "unit_result.json", "benchmark.json", "server.log",
                "moe_profile.jsonl", "npu_samples.jsonl", "release_ack.json",
            )
        }
        missing = [name for name, path in required.items() if not path.is_file() or path.stat().st_size == 0]
        if missing:
            failures.append(f"slots {slots}: missing {missing}")
            continue
        manifest = _read(required["unit_manifest.json"])
        result = _read(required["unit_result.json"])
        benchmark = _read(required["benchmark.json"])
        release = _read(required["release_ack.json"])
        records = _jsonl(required["moe_profile.jsonl"])
        npu_samples = _jsonl(required["npu_samples.jsonl"])
        log = required["server.log"].read_text(encoding="utf-8", errors="replace")
        if (manifest.get("case") or {}).get("name") != EXPECTED[slots]:
            failures.append(f"slots {slots}: case mismatch")
        selected = manifest.get("selected_env") or {}
        if selected.get("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS") != str(slots):
            failures.append(f"slots {slots}: selected slot count differs")
        server_command = manifest.get("server_command") or []
        try:
            kv_index = server_command.index("--kv-cache-memory-bytes")
            if server_command[kv_index + 1] != "536870912":
                failures.append(f"slots {slots}: KV-cache contract differs")
        except (ValueError, IndexError):
            failures.append(f"slots {slots}: missing explicit 512-MiB KV-cache contract")
        if result.get("status") != "ok" or result.get("release_status") != "released" or release.get("status") != "released":
            failures.append(f"slots {slots}: service or release failed")
        if benchmark.get("successful_requests") != args.expected_requests or benchmark.get("failed_requests") != 0:
            failures.append(f"slots {slots}: request gate failed")
        outputs = {row["request_id"]: row.get("output_token_ids") for row in benchmark.get("per_request") or []}
        ttft_values_ms = [
            float(row["ttft_s"]) * 1000.0
            for row in benchmark.get("per_request") or []
            if row.get("ttft_s") is not None
        ]
        if oracle is None:
            oracle = outputs
        elif outputs != oracle:
            failures.append(f"slots {slots}: output tokens differ")
        provenance = manifest.get("provenance") or {}
        identity = provenance.get("runtime_source_sha256")
        if not identity:
            failures.append(f"slots {slots}: exact runtime source identity missing")
        if source_identity is None:
            source_identity = identity
        elif identity != source_identity:
            failures.append(f"slots {slots}: runtime source identity differs")
        graph_capture = "Graph capturing finished" in log
        graph_replay = bool(re.search(r"Replaying aclgraph", log))
        errors = {marker: log.count(marker) for marker in FORBIDDEN if marker in log}
        if not graph_capture or not graph_replay or errors:
            failures.append(f"slots {slots}: graph/runtime gate failed {errors}")
        wave_events = [record for record in records if record.get("name") == "b2_work_conserving_prefill"]
        wave_counts = []
        h2d_bytes = 0
        for record in wave_events:
            payload = record.get("payload") or {}
            observed_slots = int(payload.get("num_slots") or 0)
            if observed_slots != slots:
                failures.append(f"slots {slots}: wave event reports {observed_slots} slots")
            summary = payload.get("wave_summary") or {}
            wave_counts.append(int(payload.get("n_waves") or summary.get("wave_count") or 0))
            h2d_bytes += int(summary.get("h2d_bytes") or 0)
        if not wave_events:
            failures.append(f"slots {slots}: no capacity-bounded wave events")
        ledgers = [record["memory_ledger"] for record in records if isinstance(record.get("memory_ledger"), dict)]
        final = ledgers[-1] if ledgers else {}
        hbm_samples = [
            int(sample["hbm_usage_percent"])
            for sample in npu_samples
            if sample.get("hbm_usage_percent") is not None
        ]
        rows.append({
            "slots": slots,
            "normalized_capacity": slots / 128.0,
            "successful_requests": benchmark.get("successful_requests"),
            "ttft_p50_ms": (benchmark.get("ttft_ms") or {}).get("p50"),
            "ttft_p95_ms": _percentile(ttft_values_ms, 95.0),
            "tpot_p50_ms": (benchmark.get("tpot_ms") or {}).get("p50"),
            "throughput_tok_s": benchmark.get("output_throughput_tok_s"),
            "wave_events": len(wave_events),
            "waves_total": sum(wave_counts),
            "waves_p50": sorted(wave_counts)[len(wave_counts) // 2] if wave_counts else None,
            "h2d_bytes": h2d_bytes,
            "host_store_bytes": final.get("host_store_bytes"),
            "slot_bank_bytes": final.get("slot_bank_bytes"),
            "prefill_stage_bank_bytes": final.get("prefill_stage_bank_bytes"),
            "hbm_peak_percent": max(hbm_samples) if hbm_samples else None,
            "hbm_final_percent": hbm_samples[-1] if hbm_samples else None,
            "graph_capture": graph_capture,
            "graph_replay": graph_replay,
            "unit_dir": str(unit),
            "artifact_sha256": {name: _sha(path) for name, path in required.items()},
        })
    if sorted(row["slots"] for row in rows) != [16, 32, 64]:
        failures.append("capacity sweep does not contain exactly 16/32/64 slots")
    output = {
        "schema_version": "latchmoe-capacity-sweep-audit-v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "campaign": str(campaign_path),
        "campaign_sha256": _sha(campaign_path),
        "scope": "one service start per capacity; latency is descriptive",
        "rows": sorted(rows, key=lambda row: row["slots"]),
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(output_path), "failures": failures}))
    return 0 if output["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
