#!/usr/bin/env python3
"""Fail closed unless one managed Issue #7 unit satisfies every graph gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


FORBIDDEN_GRAPH_ERRORS = (
    "requires PIECEWISE-only ACLGraph",
    "capture model contains a stream that was not joined",
    "Not allow to synchronize captured-stream",
    "address fingerprint changed",
    "stale H2D completion",
    "ownership changed while compute was in flight",
    "CUDA out of memory",
    "NPU out of memory",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid profile JSONL line {line_number}: {exc}") from exc
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_unit(unit_dir: Path, *, minimum_requests: int = 1) -> dict[str, Any]:
    failures: list[str] = []
    required = {
        name: unit_dir / name
        for name in (
            "unit_manifest.json",
            "unit_result.json",
            "benchmark.json",
            "server.log",
            "client.log",
            "launcher_lifecycle.log",
            "moe_profile.jsonl",
        )
    }
    for name, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty artifact: {name}")
    if failures:
        return {"status": "failed", "failures": failures, "unit_dir": str(unit_dir)}

    manifest = _read_json(required["unit_manifest.json"])
    result = _read_json(required["unit_result.json"])
    benchmark = _read_json(required["benchmark.json"])
    server_log = required["server.log"].read_text(encoding="utf-8", errors="replace")
    records = _profile_records(required["moe_profile.jsonl"])
    provenance = manifest.get("provenance") or {}

    if result.get("status") != "ok":
        failures.append(f"unit status is {result.get('status')!r}")
    if result.get("release_status") != "released":
        failures.append(f"release status is {result.get('release_status')!r}")
    release_ack = Path(str(result.get("release_ack") or ""))
    if not release_ack.is_file():
        failures.append("release ACK is missing")
    else:
        ack = _read_json(release_ack)
        if ack.get("status") != "released":
            failures.append(f"release ACK status is {ack.get('status')!r}")

    successful = int(benchmark.get("successful_requests") or 0)
    failed = int(benchmark.get("failed_requests") or 0)
    if successful < int(minimum_requests) or failed != 0:
        failures.append(
            f"request gate failed: successful={successful}, failed={failed}, "
            f"minimum={minimum_requests}"
        )
    if int(benchmark.get("total_output_tokens") or 0) <= 0:
        failures.append("no output tokens were produced")

    graph_capture = "Graph capturing finished" in server_log
    graph_replay = bool(re.search(r"Replaying aclgraph", server_log))
    graph_config = (
        "LATCHMOE_GRAPH_CONFIG cudagraph_mode=PIECEWISE" in server_log
        and "splitting_op=vllm::moe_offload_stage" in server_log
    )
    for label, observed in (
        ("Graph capture", graph_capture),
        ("Graph replay", graph_replay),
        ("PIECEWISE splitting-op configuration", graph_config),
    ):
        if not observed:
            failures.append(f"missing {label} evidence")
    for marker in FORBIDDEN_GRAPH_ERRORS:
        if marker in server_log:
            failures.append(f"forbidden runtime marker observed: {marker}")

    by_name: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_name.setdefault(str(record.get("name", "")), []).append(record)
    locks = by_name.get("graph_slot_address_lock", [])
    validations = by_name.get("graph_slot_address_validate", [])
    lock_by_layer = {int(record["layer_id"]): record for record in locks}
    validation_by_layer = {int(record["layer_id"]): record for record in validations}
    if not lock_by_layer:
        failures.append("missing graph slot-address lock events")
    for layer_id, lock in lock_by_layer.items():
        validation = validation_by_layer.get(layer_id)
        if validation is None:
            failures.append(f"layer {layer_id} lacks replay-time address validation")
            continue
        lock_payload = lock.get("payload") or {}
        validation_payload = validation.get("payload") or {}
        for key in ("w13_data_ptr", "w2_data_ptr", "log2phy_data_ptr"):
            if lock_payload.get(key) != validation_payload.get(key):
                failures.append(f"layer {layer_id} changed {key}")
        if validation_payload.get("matches_capture_fingerprint") is not True:
            failures.append(f"layer {layer_id} address validation did not pass")

    generation_checks = by_name.get(
        "slot_generation_protected_until_compute_complete", []
    )
    if not generation_checks:
        failures.append("missing slot generation/compute lease evidence")
    if any(
        (record.get("payload") or {}).get("leases_still_match") is not True
        for record in generation_checks
    ):
        failures.append("slot generation changed before compute completion")

    decode_stages = by_name.get("decode_fixed_slot_stage", [])
    h2d_stages = [
        record
        for record in decode_stages
        if int((record.get("payload") or {}).get("h2d_bytes") or 0) > 0
    ]
    if not h2d_stages:
        failures.append("missing H2D decode staging evidence")
    for record in h2d_stages:
        payload = record.get("payload") or {}
        if payload.get("consumer_dependency_installed") is not True:
            failures.append("H2D consumer dependency was not installed")
        if payload.get("mapping_published_after_ready") is not True:
            failures.append("mapping was not published after H2D readiness")

    b2_events = by_name.get("b2_work_conserving_prefill", [])
    if not b2_events:
        failures.append("missing multi-wave prefill evidence")
    for record in b2_events:
        payload = record.get("payload") or {}
        num_slots = int(payload.get("num_slots") or 0)
        wave_plan = payload.get("wave_plan") or {}
        waves = wave_plan.get("waves") or payload.get("waves") or []
        summary = payload.get("wave_summary") or {}
        if num_slots <= 0 or int(payload.get("n_active") or 0) <= 0:
            failures.append("wave capacity metadata is missing")
        if waves and any(len(wave.get("experts") or []) > num_slots for wave in waves):
            failures.append("wave active working set exceeds slot capacity")
        if not waves and int(summary.get("wave_count") or 0) <= 1:
            failures.append("multi-wave capacity evidence has fewer than two waves")
        if int(summary.get("wave_count") or 0) <= 0:
            failures.append("wave profile has no executed waves")
        if int(summary.get("prefetch_after_compute_issues") or 0) != 0:
            failures.append("late after-compute H2D prefetch observed")

    for key in (
        "repository_head_sha",
        "repository_parent_sha",
        "compatibility_lock_sha256",
        "model_config_sha256",
        "dataset_manifest_sha256",
        "device",
        "runtime_bundle_sha256",
    ):
        if not provenance.get(key):
            failures.append(f"missing provenance field: {key}")
    if provenance.get("runtime_paths_dirty") is not False:
        failures.append("evidence was not produced from clean runtime paths")

    artifact_sha256 = {
        name: _sha256(path)
        for name, path in required.items()
        if path.is_file()
    }
    return {
        "status": "passed" if not failures else "failed",
        "unit_dir": str(unit_dir),
        "failures": failures,
        "counts": {
            "successful_requests": successful,
            "output_tokens": int(benchmark.get("total_output_tokens") or 0),
            "profile_records": len(records),
            "address_lock_layers": len(lock_by_layer),
            "address_validation_layers": len(validation_by_layer),
            "h2d_decode_stages": len(h2d_stages),
            "b2_prefill_events": len(b2_events),
        },
        "timing": timing_breakdown(records),
        "artifact_sha256": artifact_sha256,
    }


def timing_breakdown(records: list[dict[str, Any]]) -> dict[str, float | int]:
    totals = {
        "h2d_bytes": 0,
        "h2d_copy_enqueue_ms": 0.0,
        "waiting_event_ms": 0.0,
        "slot_update_ms": 0.0,
        "wave_prefill_compute_ms": 0.0,
        "wave_prefill_stage_issue_ms": 0.0,
        "wave_prefill_stage_wait_ms": 0.0,
    }
    for record in records:
        payload = record.get("payload") or {}
        sample_rate = int(payload.get("profile_sample_rate") or 1)
        if record.get("name") == "decode_fixed_slot_stage":
            totals["h2d_bytes"] += int(payload.get("h2d_bytes") or 0) * sample_rate
            totals["h2d_copy_enqueue_ms"] += float(payload.get("load_enqueue_ms") or 0.0) * sample_rate
            totals["waiting_event_ms"] += float(payload.get("ready_wait_ms") or 0.0) * sample_rate
            totals["slot_update_ms"] += float(payload.get("mapping_ms") or 0.0) * sample_rate
        if record.get("name") == "b2_work_conserving_prefill":
            summary = payload.get("wave_summary") or {}
            totals["h2d_bytes"] += int(summary.get("h2d_bytes") or 0)
            totals["wave_prefill_compute_ms"] += float(summary.get("mlp_ms") or 0.0)
            totals["wave_prefill_stage_issue_ms"] += float(summary.get("stage_issue_ms") or 0.0)
            totals["wave_prefill_stage_wait_ms"] += float(summary.get("stage_wait_ms") or 0.0)
    return {
        key: int(value) if key == "h2d_bytes" else round(float(value), 3)
        for key, value in totals.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-dir", required=True)
    parser.add_argument("--minimum-requests", type=int, default=1)
    args = parser.parse_args()
    unit_dir = Path(args.unit_dir)
    report = verify_unit(unit_dir, minimum_requests=args.minimum_requests)
    report_path = unit_dir / "graph_correctness.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    marker = unit_dir / ("PASSED.txt" if report["status"] == "passed" else "FAILED.txt")
    marker.write_text(
        "all Issue #7 graph gates passed\n"
        if report["status"] == "passed"
        else "\n".join(report["failures"]) + "\n",
        encoding="utf-8",
    )
    if report["status"] != "passed":
        print("; ".join(report["failures"]))
        return 1
    print(f"Issue #7 graph unit passed: {unit_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
