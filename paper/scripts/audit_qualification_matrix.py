#!/usr/bin/env python3
"""Digest-check and consolidate the three-model qualification matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_GATES = {
    "router_parity",
    "layer_boundary_parity",
    "native_oracle",
    "latchmoe_eager_diagnostic",
    "token_exactness",
    "piecewise_capture",
    "piecewise_replay",
    "zero_eager_fallback",
    "prefill_overflow",
    "decode_cache_churn",
    "h2d_lease_release",
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
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    matrix_path = Path(args.matrix).resolve()
    registry_path = Path(args.registry).resolve()
    matrix = _read(matrix_path)
    registry = _read(registry_path)
    registry_rows = {row["id"]: row for row in registry["models"]}
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    for row in matrix.get("rows") or []:
        model_id = row["id"]
        report_path = Path(row["evidence"]["qualification_report"])
        if not report_path.is_file():
            failures.append(f"{model_id}: missing qualification report {report_path}")
            continue
        observed_sha = _sha256(report_path)
        expected_sha = row["evidence"]["qualification_report_sha256"]
        if observed_sha != expected_sha:
            failures.append(f"{model_id}: qualification report digest mismatch")
            continue
        report = _read(report_path)
        gates = report.get("gates") or {}
        if set(gates) != REQUIRED_GATES or any(value != "passed" for value in gates.values()):
            failures.append(f"{model_id}: required qualification gates are incomplete")
        if report.get("status") != "passed" or report.get("failed_gates"):
            failures.append(f"{model_id}: report status is not passed")
        if report.get("model_id") != model_id:
            failures.append(f"{model_id}: report model identity differs")
        model = registry_rows.get(model_id)
        if model is None:
            failures.append(f"{model_id}: missing registry row")
            continue
        details = report.get("details") or {}
        selection = row["capability_config"]["selection"]
        rows.append({
            "model_id": model_id,
            "architecture": model["architecture"][0],
            "moe_layers": model["memory_estimate"]["moe_layers"],
            "routed_experts": row["capability_config"]["routed_expert_count"],
            "top_k": selection["top_k"],
            "shared_experts": row["capability_config"]["shared_expert_count"],
            "output_contract": row["capability_config"]["output_contract"],
            "status": report["status"],
            "router_records": details["router_parity"]["records_compared"],
            "layer_boundary_records": details["layer_boundary"]["records"],
            "graph_address_locks": details["piecewise_capture"]["address_locks"],
            "graph_address_validations": details["piecewise_capture"]["address_validations"],
            "graph_replays": details["piecewise_replay"]["replays"],
            "overflow_events": details["prefill_overflow"]["events"],
            "decode_steps": details["decode_cache_churn"]["decode_steps"],
            "h2d_stages": details["h2d_lease_release"]["h2d_stages"],
            "lease_records": details["h2d_lease_release"]["lease_records"],
            "release_records": details["h2d_lease_release"]["release_records"],
            "qualification_report": str(report_path),
            "qualification_report_sha256": observed_sha,
        })
    if set(registry_rows) != {row["id"] for row in matrix.get("rows") or []}:
        failures.append("registry and qualification matrix model sets differ")
    output = {
        "schema_version": "latchmoe-paper-qualification-summary-v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "matrix": str(matrix_path),
        "matrix_sha256": _sha256(matrix_path),
        "registry": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "required_gates": sorted(REQUIRED_GATES),
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": output["status"], "rows": len(rows), "output": str(output_path)}))
    return 0 if output["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
