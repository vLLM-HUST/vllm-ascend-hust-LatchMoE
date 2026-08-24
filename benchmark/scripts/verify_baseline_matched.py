#!/usr/bin/env python3
"""Verify and summarize the four-arm matched graph-baseline campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any


CASES = {
    "full_resident": "no_offload_kv512m_aclgraph",
    "native_prefetch": "native_prefetch_14gb_kv512m",
    "legacy_layered": "legacy_layered_14gb_kv512m",
    "latchmoe": "sew_14gb_autoslots_kv512m",
}
EXPECTED_ORDERS = [
    ["full_resident", "native_prefetch", "legacy_layered", "latchmoe"],
    ["latchmoe", "legacy_layered", "native_prefetch", "full_resident"],
    ["legacy_layered", "latchmoe", "full_resident", "native_prefetch"],
]
REPLICATES = 10_000
SEED = 20_260_823
FORBIDDEN = (
    "capture model contains a stream that was not joined", "address fingerprint changed",
    "stale H2D completion", "ownership changed while compute was in flight",
    "NPU out of memory", "CUDA out of memory", "ACL_ERROR", "EZ9999",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requests(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in benchmark.get("per_request") or []:
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in rows:
            raise ValueError(f"missing or duplicate request_id {request_id!r}")
        rows[request_id] = row
    return rows


def _percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    offset = (len(values) - 1) * probability
    low = int(offset)
    high = min(low + 1, len(values) - 1)
    weight = offset - low
    return values[low] * (1 - weight) + values[high] * weight


def _latency(row: dict[str, Any], metric: str) -> float:
    ttft = float(row["ttft_s"]) * 1000.0
    if metric == "ttft":
        return ttft
    return (float(row["total_s"]) - float(row["ttft_s"])) / max(1, int(row["output_tokens"])) * 1000.0


def _verification_status(failures: list[str], unsupported_reasons: list[str]) -> str:
    if failures:
        return "failed"
    if unsupported_reasons:
        return "unsupported"
    return "passed"


def verify_unit(unit: Path, arm: str, expected_requests: int, oracle: Path | None) -> dict[str, Any]:
    failures: list[str] = []
    unsupported_reasons: list[str] = []
    names = (
        "unit_manifest.json", "unit_result.json", "benchmark.json", "server.log",
        "client.log", "launcher_lifecycle.log", "npu_samples.jsonl", "release_ack.json",
    )
    files = {name: unit / name for name in names}
    for name, path in files.items():
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty artifact: {name}")
    if failures:
        return {"status": "failed", "failures": failures, "unit_dir": str(unit)}
    manifest = _read(files["unit_manifest.json"])
    result = _read(files["unit_result.json"])
    benchmark = _read(files["benchmark.json"])
    release = _read(files["release_ack.json"])
    log = files["server.log"].read_text(encoding="utf-8", errors="replace")
    if (manifest.get("case") or {}).get("name") != CASES[arm]:
        failures.append("case identity differs")
    command = manifest.get("server_command") or []
    try:
        kv_index = command.index("--kv-cache-memory-bytes")
        if command[kv_index + 1] != "536870912":
            failures.append("KV-cache contract differs")
    except (ValueError, IndexError):
        failures.append("missing explicit 512-MiB KV-cache contract")
    if result.get("status") != "ok" or result.get("release_status") != "released" or release.get("status") != "released":
        failures.append("managed execution or release failed")
    successful = int(benchmark.get("successful_requests") or 0)
    failed = int(benchmark.get("failed_requests") or 0)
    observed = _requests(benchmark)
    if successful != expected_requests or failed != 0 or len(observed) != expected_requests:
        failures.append(f"request gate failed: successful={successful}, failed={failed}, records={len(observed)}")
    if oracle is not None:
        expected = _requests(_read(oracle))
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        mismatched = sorted(
            key for key in set(expected) & set(observed)
            if expected[key].get("output_token_ids") != observed[key].get("output_token_ids")
        )
        if missing or extra or mismatched:
            unsupported_reasons.append("request IDs or output-token arrays differ from oracle")
    graph_capture = "Graph capturing finished" in log
    graph_replay = bool(re.search(r"Replaying aclgraph", log))
    if not graph_capture or not graph_replay:
        failures.append("graph capture/replay missing")
    errors = {marker: log.count(marker) for marker in FORBIDDEN if marker in log}
    if errors:
        failures.append(f"forbidden runtime markers: {errors}")
    provenance = manifest.get("provenance") or {}
    for key in (
        "model_config_sha256", "dataset_manifest_sha256", "device", "vllm_root_sha",
        "seam_root_sha", "compatibility_lock_sha256", "runtime_bundle_sha256",
        "runtime_source_sha256",
    ):
        if not provenance.get(key):
            failures.append(f"missing provenance: {key}")
    status = _verification_status(failures, unsupported_reasons)
    return {
        "status": status, "failures": failures,
        "unsupported_reasons": unsupported_reasons,
        "comparable": status == "passed",
        "unit_dir": str(unit), "arm": arm, "successful_requests": successful,
        "failed_requests": failed, "graph_capture": graph_capture, "graph_replay": graph_replay,
        "errors": errors, "provenance": provenance,
        "artifact_sha256": {name: _sha(path) for name, path in files.items()},
    }


def _arm_repeat(unit_report: dict[str, Any]) -> dict[str, Any]:
    benchmark_path = Path(unit_report["unit_dir"]) / "benchmark.json"
    if not benchmark_path.is_file():
        return {
            "repeat": unit_report["repeat"], "position": unit_report["position"],
            "status": unit_report.get("status"), "comparable": False,
        }
    benchmark = _read(benchmark_path)
    requests = _requests(benchmark)
    ttft = [_latency(row, "ttft") for row in requests.values()]
    tpot = [_latency(row, "tpot") for row in requests.values()]
    return {
        "repeat": unit_report["repeat"], "position": unit_report["position"],
        "status": unit_report.get("status"),
        "comparable": bool(unit_report.get("comparable")),
        "ttft_p50_ms": _percentile(ttft, 0.50), "ttft_p95_ms": _percentile(ttft, 0.95),
        "tpot_p50_ms": _percentile(tpot, 0.50), "tpot_p95_ms": _percentile(tpot, 0.95),
        "throughput_tok_s": float(benchmark.get("output_throughput_tok_s") or 0.0),
    }


def _pair_logs(latch: dict[str, Any], baseline: dict[str, Any], metric: str) -> list[float]:
    left = _requests(_read(Path(baseline["unit_dir"]) / "benchmark.json"))
    right = _requests(_read(Path(latch["unit_dir"]) / "benchmark.json"))
    if set(left) != set(right):
        raise ValueError("request IDs differ")
    return [math.log(_latency(right[key], metric) / _latency(left[key], metric)) for key in sorted(left)]


def _estimate(logs: list[list[float]]) -> float:
    return math.exp(statistics.fmean(statistics.median(values) for values in logs)) - 1.0


def _bootstrap(logs: list[list[float]], confidence: float = 0.95) -> dict[str, Any]:
    rng = random.Random(SEED)
    samples: list[float] = []
    for _ in range(REPLICATES):
        medians = []
        for _ in range(len(logs)):
            values = logs[rng.randrange(len(logs))]
            selected = [values[rng.randrange(len(values))] for _ in range(len(values))]
            medians.append(statistics.median(selected))
        samples.append(math.exp(statistics.fmean(medians)) - 1.0)
    alpha = (1.0 - confidence) / 2.0
    return {
        "replicates": REPLICATES, "seed": SEED, "confidence": confidence,
        "lower": _percentile(samples, alpha), "upper": _percentile(samples, 1.0 - alpha),
    }


def summarize(campaign_path: Path) -> dict[str, Any]:
    campaign = _read(campaign_path)
    failures: list[str] = []
    if [list(order) for order in campaign.get("orders") or []] != EXPECTED_ORDERS:
        failures.append("counterorder differs from preregistration")
    reports = []
    unsupported_units = []
    for item in campaign.get("units") or []:
        report = _read(Path(item["unit_dir"]) / "baseline_verification.json")
        report.update({
            "repeat": item["repeat"], "position": item["position"],
            "stage": item["stage"], "arm": item["arm"],
        })
        reports.append(report)
        if report.get("status") == "failed":
            failures.append(f"{item['stage']} failed verification")
        elif report.get("status") == "unsupported":
            unsupported_units.append({
                "stage": item["stage"], "arm": report.get("arm"),
                "reasons": report.get("unsupported_reasons") or [],
            })
    if len(reports) != 12:
        failures.append(f"expected 12 units, found {len(reports)}")
    if reports:
        reference = reports[0].get("provenance") or {}
        keys = ("model_config_sha256", "dataset_manifest_sha256", "device", "vllm_root_sha", "seam_root_sha", "compatibility_lock_sha256", "runtime_source_sha256")
        for report in reports[1:]:
            current = report.get("provenance") or {}
            if any(current.get(key) != reference.get(key) for key in keys):
                failures.append(f"{report['stage']} provenance differs")
    arms: dict[str, list[dict[str, Any]]] = {arm: [] for arm in CASES}
    for report in reports:
        arms[report["arm"]].append(_arm_repeat(report))
    for arm, values in arms.items():
        values.sort(key=lambda item: item["repeat"])
        if len(values) != 3:
            failures.append(f"{arm} has {len(values)} repeats")
    comparisons: dict[str, Any] = {}
    if not failures:
        for baseline in ("full_resident", "native_prefetch", "legacy_layered"):
            baseline_reports = [row for row in reports if row["arm"] == baseline]
            latch_reports = [row for row in reports if row["arm"] == "latchmoe"]
            if any(row.get("status") != "passed" for row in baseline_reports + latch_reports):
                comparisons[baseline] = {
                    "status": "unsupported",
                    "reason": "one or more repeats failed exact-output comparability",
                }
                continue
            ttft_logs, tpot_logs = [], []
            for repeat in range(1, 4):
                latch = next(row for row in reports if row["arm"] == "latchmoe" and row["repeat"] == repeat)
                other = next(row for row in reports if row["arm"] == baseline and row["repeat"] == repeat)
                ttft_logs.append(_pair_logs(latch, other, "ttft"))
                tpot_logs.append(_pair_logs(latch, other, "tpot"))
            comparisons[baseline] = {
                "status": "passed",
                "estimator": "exp(mean_repeat(median_request(log(latchmoe/baseline))))-1",
                "ttft_effect": _estimate(ttft_logs), "ttft_interval_95": _bootstrap(ttft_logs, 0.95),
                "tpot_effect": _estimate(tpot_logs), "tpot_interval_95": _bootstrap(tpot_logs, 0.95),
            }
    return {
        "status": "passed" if not failures else "failed", "failures": failures,
        "campaign_path": str(campaign_path), "units": reports, "arms": arms,
        "comparisons": comparisons, "unsupported_units": unsupported_units,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-dir")
    parser.add_argument("--arm", choices=tuple(CASES))
    parser.add_argument("--oracle-benchmark")
    parser.add_argument("--expected-requests", type=int, default=64)
    parser.add_argument("--campaign")
    parser.add_argument("--output")
    args = parser.parse_args()
    if bool(args.unit_dir) == bool(args.campaign):
        parser.error("select exactly one of --unit-dir or --campaign")
    if args.unit_dir:
        if not args.arm:
            parser.error("--arm is required with --unit-dir")
        report = verify_unit(Path(args.unit_dir), args.arm, args.expected_requests, Path(args.oracle_benchmark) if args.oracle_benchmark else None)
        output = Path(args.output) if args.output else Path(args.unit_dir) / "baseline_verification.json"
    else:
        report = summarize(Path(args.campaign))
        output = Path(args.output) if args.output else Path(args.campaign).with_name("baseline_summary.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "failures": report["failures"]}))
    return 0 if report["status"] in {"passed", "unsupported"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
