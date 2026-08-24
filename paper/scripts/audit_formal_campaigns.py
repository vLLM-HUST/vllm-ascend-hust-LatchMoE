#!/usr/bin/env python3
"""Package and independently audit the formal performance campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


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


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _requests(path: Path) -> dict[str, dict[str, Any]]:
    rows = (_read(path).get("per_request") or [])
    return {str(row["request_id"]): row for row in rows}


def _audit_artifacts(unit: dict[str, Any], failures: list[str], label: str) -> None:
    root = Path(str(unit.get("unit_dir") or ""))
    recorded = unit.get("artifact_sha256") or {}
    if not root.is_dir() or not recorded:
        failures.append(f"{label}: missing unit directory or artifact ledger")
        return
    for name, expected in recorded.items():
        path = root / name
        if not path.is_file():
            failures.append(f"{label}: missing {name}")
        elif _sha(path) != expected:
            failures.append(f"{label}: digest mismatch for {name}")


def _baseline_diagnostics(summary: dict[str, Any], failures: list[str]) -> list[dict[str, Any]]:
    reports = summary.get("units") or []
    if len(reports) != 12:
        failures.append(f"baseline: expected 12 units, found {len(reports)}")
    oracles = {
        int(report["repeat"]): Path(report["unit_dir"]) / "benchmark.json"
        for report in reports
        if report.get("arm") == "full_resident" and report.get("status") == "passed"
    }
    rows = []
    for report in reports:
        label = str(report.get("stage") or report.get("unit_dir"))
        _audit_artifacts(report, failures, f"baseline {label}")
        unit = Path(report["unit_dir"])
        benchmark_path = unit / "benchmark.json"
        row: dict[str, Any] = {
            "repeat": report.get("repeat"),
            "position": report.get("position"),
            "arm": report.get("arm"),
            "status": report.get("status"),
            "unit_dir": str(unit),
        }
        if benchmark_path.is_file():
            benchmark = _read(benchmark_path)
            ttft = [float(item["ttft_s"]) * 1000.0 for item in benchmark.get("per_request") or []]
            tpot = [
                (float(item["total_s"]) - float(item["ttft_s"]))
                / max(1, int(item["output_tokens"])) * 1000.0
                for item in benchmark.get("per_request") or []
            ]
            row.update({
                "successful_requests": benchmark.get("successful_requests"),
                "failed_requests": benchmark.get("failed_requests"),
                "ttft_p50_ms": _percentile(ttft, 0.50),
                "ttft_p95_ms": _percentile(ttft, 0.95),
                "tpot_p50_ms": _percentile(tpot, 0.50),
                "tpot_p95_ms": _percentile(tpot, 0.95),
                "throughput_tok_s": benchmark.get("output_throughput_tok_s"),
            })
            oracle_path = oracles.get(int(report["repeat"]))
            if oracle_path and oracle_path.is_file():
                oracle = _requests(oracle_path)
                observed = _requests(benchmark_path)
                missing = sorted(set(oracle) - set(observed))
                extra = sorted(set(observed) - set(oracle))
                mismatched = sorted(
                    request_id for request_id in set(oracle) & set(observed)
                    if oracle[request_id].get("output_token_ids")
                    != observed[request_id].get("output_token_ids")
                )
                row["exactness"] = {
                    "oracle": str(oracle_path),
                    "missing_requests": missing,
                    "extra_requests": extra,
                    "mismatched_requests": mismatched,
                    "mismatched_count": len(mismatched),
                    "passed": not (missing or extra or mismatched),
                }
                expected_status = "passed" if row["exactness"]["passed"] else "unsupported"
                if report.get("status") != expected_status:
                    failures.append(
                        f"baseline {label}: verifier status {report.get('status')} "
                        f"does not match exactness-derived {expected_status}"
                    )
        npu_path = unit / "npu_samples.jsonl"
        if npu_path.is_file():
            samples = _jsonl(npu_path)
            hbm = [int(item["hbm_usage_percent"]) for item in samples if item.get("hbm_usage_percent") is not None]
            row["hbm_peak_percent"] = max(hbm) if hbm else None
            row["hbm_final_percent"] = hbm[-1] if hbm else None
        rows.append(row)
    return rows


def _audit_overlap(summary: dict[str, Any], failures: list[str]) -> None:
    units = summary.get("units") or []
    if len(units) != 6:
        failures.append(f"overlap: expected 6 units, found {len(units)}")
    for report in units:
        _audit_artifacts(report, failures, f"overlap {report.get('stage')}")
    statistics = summary.get("statistics") or {}
    ttft = statistics.get("ttft") or {}
    interval = ttft.get("interval") or {}
    recomputed_gate = (
        float(ttft.get("estimate", 1.0)) <= -0.05
        and float(interval.get("upper", 1.0)) < 0.0
    )
    if bool(ttft.get("causal_benefit_gate")) != recomputed_gate:
        failures.append("overlap: causal-benefit gate does not match preregistered rule")
    tpot = statistics.get("tpot") or {}
    tpot_interval = tpot.get("interval") or {}
    recomputed_equivalence = (
        float(tpot_interval.get("lower", -1.0)) >= -0.05
        and float(tpot_interval.get("upper", 1.0)) <= 0.05
    )
    if bool(tpot.get("equivalence_gate")) != recomputed_equivalence:
        failures.append("overlap: TPOT equivalence gate does not match preregistered rule")


def _audit_capacity(summary: dict[str, Any], failures: list[str]) -> None:
    rows = summary.get("rows") or []
    if [int(row["slots"]) for row in rows] != [16, 32, 64]:
        failures.append("capacity: rows are not the fixed 16/32/64-slot sweep")
    for row in rows:
        _audit_artifacts(row, failures, f"capacity slots-{row.get('slots')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--overlap", required=True)
    parser.add_argument("--capacity", required=True)
    parser.add_argument("--issue17", required=True)
    parser.add_argument("--workload-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "baseline": Path(args.baseline).resolve(),
        "overlap": Path(args.overlap).resolve(),
        "capacity": Path(args.capacity).resolve(),
        "issue17": Path(args.issue17).resolve(),
        "workload_manifest": Path(args.workload_manifest).resolve(),
    }
    payloads = {name: _read(path) for name, path in paths.items() if name != "workload_manifest"}
    failures: list[str] = []
    for name, payload in payloads.items():
        if payload.get("status") != "passed":
            failures.append(f"{name}: source audit status is {payload.get('status')!r}")
    baseline_rows = _baseline_diagnostics(payloads["baseline"], failures)
    _audit_overlap(payloads["overlap"], failures)
    _audit_capacity(payloads["capacity"], failures)
    manifest_rows = _jsonl(paths["workload_manifest"])
    prefill = [row for row in manifest_rows if row.get("bucket") == "prefill_heavy"]
    prompt_lengths = [int(row["prompt_tokens"]) for row in prefill]
    output_limits = sorted({int(row["max_output_tokens"]) for row in prefill})
    seeds = sorted({int(row["seed"]) for row in prefill})
    output = {
        "schema_version": "latchmoe-formal-campaign-audit-v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "source_sha256": {name: _sha(path) for name, path in paths.items()},
        "source_paths": {name: str(path) for name, path in paths.items()},
        "workload": {
            "bucket": "prefill_heavy", "candidate_requests": len(prefill),
            "prompt_tokens_min": min(prompt_lengths),
            "prompt_tokens_p50": _percentile([float(value) for value in prompt_lengths], 0.50),
            "prompt_tokens_max": max(prompt_lengths),
            "output_token_limits": output_limits, "seeds": seeds,
        },
        "baseline_diagnostics": baseline_rows,
        "baseline": payloads["baseline"],
        "overlap": payloads["overlap"],
        "capacity": payloads["capacity"],
        "issue17": payloads["issue17"],
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(output_path), "failures": failures}))
    return 0 if output["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
