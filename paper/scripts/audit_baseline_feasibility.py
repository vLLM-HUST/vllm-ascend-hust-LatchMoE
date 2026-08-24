#!/usr/bin/env python3
"""Audit matched one-request feasibility units for the four baseline paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_CASES = {
    "full_resident": "no_offload_kv512m_aclgraph",
    "native_prefetch": "native_prefetch_14gb",
    "legacy_layered": "legacy_layered_14gb",
    "latchmoe": "sew_14gb_autoslots",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in EXPECTED_CASES:
        parser.add_argument(f"--{arm.replace('_', '-')}", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        arm: Path(getattr(args, arm)).resolve()
        for arm in EXPECTED_CASES
    }
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    oracle: dict[str, list[int]] | None = None
    reference_contract: dict[str, Any] | None = None
    for arm, unit_dir in paths.items():
        required = {
            name: unit_dir / name
            for name in (
                "unit_manifest.json", "unit_result.json", "benchmark.json",
                "server.log", "client.log", "release_ack.json",
            )
        }
        missing = [name for name, path in required.items() if not path.is_file() or path.stat().st_size == 0]
        if missing:
            failures.append(f"{arm}: missing artifacts {missing}")
            continue
        manifest = _read(required["unit_manifest.json"])
        result = _read(required["unit_result.json"])
        benchmark = _read(required["benchmark.json"])
        release = _read(required["release_ack.json"])
        log = required["server.log"].read_text(encoding="utf-8", errors="replace")
        case_name = (manifest.get("case") or {}).get("name")
        if case_name != EXPECTED_CASES[arm]:
            failures.append(f"{arm}: unexpected case {case_name!r}")
        if result.get("status") != "ok" or result.get("release_status") != "released":
            failures.append(f"{arm}: unit did not complete and release")
        if release.get("status") != "released":
            failures.append(f"{arm}: release ACK failed")
        if benchmark.get("successful_requests") != 1 or benchmark.get("failed_requests") != 0:
            failures.append(f"{arm}: one-request success gate failed")
        requests = benchmark.get("per_request") or []
        outputs = {
            str(item["request_id"]): [int(token) for token in item.get("output_token_ids") or []]
            for item in requests
        }
        if oracle is None:
            oracle = outputs
        elif outputs != oracle:
            failures.append(f"{arm}: output request IDs or token arrays differ")
        provenance = manifest.get("provenance") or {}
        contract = {
            key: provenance.get(key)
            for key in (
                "model_config_sha256", "dataset_manifest_sha256", "device",
                "vllm_root_sha", "seam_root_sha", "compatibility_lock_sha256",
            )
        }
        if reference_contract is None:
            reference_contract = contract
        elif contract != reference_contract:
            failures.append(f"{arm}: model/workload/runtime contract differs")
        graph_capture = "Graph capturing finished" in log
        graph_replay = bool(re.search(r"Replaying aclgraph", log))
        if not graph_capture or not graph_replay:
            failures.append(f"{arm}: graph capture/replay missing")
        rows.append({
            "arm": arm,
            "case": case_name,
            "status": "passed",
            "graph_capture": graph_capture,
            "graph_replay": graph_replay,
            "successful_requests": benchmark.get("successful_requests"),
            "failed_requests": benchmark.get("failed_requests"),
            "prompt_tokens": requests[0].get("prompt_tokens") if requests else None,
            "output_tokens": requests[0].get("output_tokens") if requests else None,
            "output_token_ids_sha256": requests[0].get("output_token_ids_sha256") if requests else None,
            "unit_dir": str(unit_dir),
            "artifact_sha256": {name: _sha256(path) for name, path in required.items()},
            "selected_env": manifest.get("selected_env") or {},
            "provenance_contract": contract,
        })
    report = {
        "schema_version": "latchmoe-baseline-feasibility-v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "scope": "one cold request; feasibility and exactness only, not performance",
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": len(rows), "output": str(output)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
