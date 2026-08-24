#!/usr/bin/env python3
"""Digest and summarize qualification memory/control ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODELS = (
    "qwen3-30b-a3b",
    "glm-4.7-flash",
    "qwen3-next-80b-a3b-instruct",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _payload_total(records: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for record in records:
        payload = record.get("payload") or {}
        if isinstance(payload.get(key), (int, float)):
            total += float(payload[key])
        wave = payload.get("wave_summary") or {}
        if isinstance(wave.get(key), (int, float)):
            total += float(wave[key])
    return total


def _control_ms(records: list[dict[str, Any]]) -> float:
    total = 0.0
    for record in records:
        payload = record.get("payload") or {}
        mapping_ms = payload.get("mapping_ms")
        if isinstance(mapping_ms, (int, float)):
            total += float(mapping_ms)
        value = payload.get("control_ms")
        if isinstance(value, (int, float)):
            total += float(value)
        elif isinstance(value, dict) and isinstance(value.get("end_to_end"), (int, float)):
            total += float(value["end_to_end"])
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for model in MODELS:
        parser.add_argument(f"--{model}-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    failures = []
    rows = []
    for model in MODELS:
        root = Path(getattr(args, f"{model.replace('-', '_')}_root")).resolve()
        report_path = root / "qualification_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "passed" or report.get("failed_gates"):
            failures.append(f"{model}: qualification report failed")
        modes = {}
        for mode in ("latch-graph", "overflow-graph"):
            summary_path = root / mode / "summary.json"
            profile_path = root / mode / "moe_offload_profile.jsonl"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            records = _jsonl(profile_path)
            ledgers = [record["memory_ledger"] for record in records if isinstance(record.get("memory_ledger"), dict)]
            if summary.get("status") != "ok" or not ledgers:
                failures.append(f"{model}/{mode}: missing successful memory ledger")
                continue
            final = ledgers[-1]
            modes[mode] = {
                "num_slots": summary.get("num_slots"),
                "host_store_bytes": final.get("host_store_bytes"),
                "slot_bank_bytes": final.get("slot_bank_bytes"),
                "prefill_stage_bank_bytes": final.get("prefill_stage_bank_bytes"),
                "prefill_stage_mapping_bytes": final.get("prefill_stage_mapping_bytes"),
                "resident_shared_weight_bytes": final.get("resident_shared_weight_bytes"),
                "original_expert_weight_bytes": final.get("original_expert_weight_bytes"),
                "original_expert_weights_retained": final.get("original_expert_weights_retained"),
                "registered_layers": final.get("registered_layers"),
                "total_npu_slot_bytes": final.get("total_npu_slot_bytes"),
                "h2d_bytes": int(_payload_total(records, "h2d_bytes")),
                "stage_wait_ms": _payload_total(records, "stage_wait_ms") + _payload_total(records, "ready_wait_ms"),
                "host_control_ms": _control_ms(records),
                "graph_replay_issues": sum(record.get("name") == "graph_replay_issue" for record in records),
                "address_locks": sum(record.get("name") == "graph_slot_address_lock" for record in records),
                "address_validations": sum(record.get("name") == "graph_slot_address_validate" for record in records),
                "release_original_records": sum(record.get("name") == "release_original_expert_weights" for record in records),
                "summary_sha256": _sha(summary_path),
                "profile_sha256": _sha(profile_path),
            }
        rows.append({
            "model_id": model,
            "qualification_root": str(root),
            "qualification_report_sha256": _sha(report_path),
            "modes": modes,
        })
    output = {
        "schema_version": "latchmoe-resource-ledger-audit-v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "rows": rows,
        "scope_note": "latch-graph uses 32 stable slots; overflow-graph is a separate 4-slot stress arm",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(output_path), "failures": failures}))
    return 0 if output["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
