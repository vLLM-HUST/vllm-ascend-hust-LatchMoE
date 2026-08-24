#!/usr/bin/env python3
"""Verify serial/overlap units and apply the preregistered paired estimator."""

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


ARM_CASES = {
    "serial": "sew_14gb_serial_stage_matched",
    "overlap": "sew_14gb_overlap_stage_matched",
}
ARM_ASYNC = {"serial": "0", "overlap": "1"}
EXPECTED_ORDER = ["serial", "overlap", "overlap", "serial", "serial", "overlap"]
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_823
FORBIDDEN_MARKERS = (
    "requires PIECEWISE-only ACLGraph",
    "capture model contains a stream that was not joined",
    "Not allow to synchronize captured-stream",
    "address fingerprint changed",
    "stale H2D completion",
    "ownership changed while compute was in flight",
    "NPU out of memory",
    "CUDA out of memory",
    "ACL_ERROR",
    "EZ9999",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take percentile of an empty sequence")
    offset = (len(ordered) - 1) * probability
    low = int(offset)
    high = min(low + 1, len(ordered) - 1)
    weight = offset - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _requests(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in benchmark.get("per_request") or []:
        request_id = str(item.get("request_id") or "")
        if not request_id or request_id in result:
            raise ValueError(f"missing or duplicate request_id: {request_id!r}")
        result[request_id] = item
    return result


def _latency(item: dict[str, Any], metric: str) -> float:
    ttft_s = float(item["ttft_s"])
    if metric == "ttft":
        return ttft_s * 1000.0
    output_tokens = int(item["output_tokens"])
    return (float(item["total_s"]) - ttft_s) / max(1, output_tokens) * 1000.0


def _clean_selected_env(env: dict[str, Any]) -> dict[str, Any]:
    ignored = {
        "LATCHMOE_CUSTODY_STATE",
        "VLLM_ASCEND_MOE_GMM_PROFILE_PATH",
        "VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH",
        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH",
        "VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD",
    }
    return {key: value for key, value in env.items() if key not in ignored}


def verify_unit(
    unit_dir: Path,
    *,
    arm: str,
    expected_requests: int,
    oracle_benchmark: Path | None,
) -> dict[str, Any]:
    failures: list[str] = []
    required_names = (
        "unit_manifest.json", "unit_result.json", "benchmark.json", "server.log",
        "client.log", "launcher_lifecycle.log", "moe_profile.jsonl",
        "npu_samples.jsonl", "release_ack.json",
    )
    required = {name: unit_dir / name for name in required_names}
    for name, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty artifact: {name}")
    if failures:
        return {"status": "failed", "unit_dir": str(unit_dir), "failures": failures}

    manifest = _read_json(required["unit_manifest.json"])
    result = _read_json(required["unit_result.json"])
    benchmark = _read_json(required["benchmark.json"])
    release = _read_json(required["release_ack.json"])
    server_log = required["server.log"].read_text(encoding="utf-8", errors="replace")
    selected_env = manifest.get("selected_env") or {}
    provenance = manifest.get("provenance") or {}
    if (manifest.get("case") or {}).get("name") != ARM_CASES[arm]:
        failures.append("case name does not match arm")
    if selected_env.get("VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD") != ARM_ASYNC[arm]:
        failures.append("async-load setting does not match arm")
    if result.get("status") != "ok" or result.get("release_status") != "released":
        failures.append("managed unit did not complete and release cleanly")
    if release.get("status") != "released":
        failures.append("release ACK is not released")
    successful = int(benchmark.get("successful_requests") or 0)
    failed = int(benchmark.get("failed_requests") or 0)
    if successful != expected_requests or failed != 0:
        failures.append(f"request gate failed: successful={successful}, failed={failed}")
    observed = _requests(benchmark)
    if len(observed) != expected_requests:
        failures.append(f"request records present for {len(observed)}/{expected_requests}")
    for request_id, item in observed.items():
        if not isinstance(item.get("output_token_ids"), list):
            failures.append(f"missing output token IDs: {request_id}")
        if float(item.get("ttft_s") or 0.0) <= 0 or float(item.get("total_s") or 0.0) <= 0:
            failures.append(f"nonpositive latency: {request_id}")

    oracle_report = None
    if oracle_benchmark is not None:
        expected = _requests(_read_json(oracle_benchmark))
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        mismatched = sorted(
            request_id for request_id in set(expected) & set(observed)
            if expected[request_id].get("output_token_ids") != observed[request_id].get("output_token_ids")
        )
        oracle_report = {
            "path": str(oracle_benchmark),
            "sha256": _sha256(oracle_benchmark),
            "missing_request_ids": missing,
            "extra_request_ids": extra,
            "mismatched_request_ids": mismatched,
        }
        if missing or extra or mismatched:
            failures.append("request IDs or output token IDs differ from oracle")

    graph_capture = "Graph capturing finished" in server_log
    graph_replay = bool(re.search(r"Replaying aclgraph", server_log))
    graph_config = (
        "LATCHMOE_GRAPH_CONFIG cudagraph_mode=PIECEWISE" in server_log
        and "splitting_op=vllm::moe_offload_stage" in server_log
    )
    if not graph_capture or not graph_replay or not graph_config:
        failures.append("PIECEWISE Graph capture/replay evidence is incomplete")
    error_counts = {marker: server_log.count(marker) for marker in FORBIDDEN_MARKERS if marker in server_log}
    if error_counts:
        failures.append(f"forbidden runtime markers observed: {error_counts}")
    for key in (
        "repository_head_sha", "repository_parent_sha", "compatibility_lock_sha256",
        "model_config_sha256", "dataset_manifest_sha256", "device",
        "runtime_bundle_sha256", "vllm_root_sha", "seam_root_sha",
    ):
        if not provenance.get(key):
            failures.append(f"missing provenance field: {key}")
    if not provenance.get("runtime_source_sha256") or not provenance.get("runtime_source_files"):
        failures.append("exact runtime source identity is missing")

    return {
        "status": "passed" if not failures else "failed",
        "unit_dir": str(unit_dir),
        "arm": arm,
        "failures": failures,
        "requests": {"successful": successful, "failed": failed},
        "graph": {"capture": graph_capture, "replay": graph_replay, "config": graph_config},
        "error_counts": error_counts,
        "selected_env": selected_env,
        "matched_env": _clean_selected_env(selected_env),
        "provenance": provenance,
        "oracle": oracle_report,
        "artifact_sha256": {name: _sha256(path) for name, path in required.items()},
    }


def _paired_logs(serial: dict[str, Any], overlap: dict[str, Any], metric: str) -> list[float]:
    left = _requests(_read_json(Path(serial["unit_dir"]) / "benchmark.json"))
    right = _requests(_read_json(Path(overlap["unit_dir"]) / "benchmark.json"))
    if set(left) != set(right):
        raise ValueError("paired request ID sets differ")
    return [math.log(_latency(right[key], metric) / _latency(left[key], metric)) for key in sorted(left)]


def _estimate(pair_logs: list[list[float]]) -> float:
    return math.exp(statistics.fmean(statistics.median(values) for values in pair_logs)) - 1.0


def _bootstrap(pair_logs: list[list[float]], confidence: float) -> dict[str, float | int]:
    rng = random.Random(BOOTSTRAP_SEED)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        medians: list[float] = []
        for _ in range(len(pair_logs)):
            values = pair_logs[rng.randrange(len(pair_logs))]
            resampled = [values[rng.randrange(len(values))] for _ in range(len(values))]
            medians.append(statistics.median(resampled))
        samples.append(math.exp(statistics.fmean(medians)) - 1.0)
    alpha = (1.0 - confidence) / 2.0
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence": confidence,
        "lower": _percentile(samples, alpha),
        "upper": _percentile(samples, 1.0 - alpha),
    }


def summarize_campaign(campaign_path: Path) -> dict[str, Any]:
    campaign = _read_json(campaign_path)
    units = campaign.get("units") or []
    failures: list[str] = []
    if [item.get("arm") for item in units] != EXPECTED_ORDER:
        failures.append("campaign order is not the fixed AB/BA/AB sequence")
    reports: list[dict[str, Any]] = []
    for item in units:
        report = _read_json(Path(item["unit_dir"]) / "overlap_verification.json")
        reports.append({"stage": item["stage"], **report})
        if report.get("status") != "passed":
            failures.append(f"{item['stage']} verification failed")

    if reports:
        reference_env = reports[0].get("matched_env")
        reference_provenance = reports[0].get("provenance") or {}
        comparable = (
            "repository_head_sha", "repository_parent_sha", "compatibility_lock_sha256",
            "model_config_sha256", "dataset_manifest_sha256", "device",
            "runtime_bundle_sha256", "vllm_root_sha", "seam_root_sha",
            "runtime_source_sha256",
        )
        for report in reports[1:]:
            if report.get("matched_env") != reference_env:
                failures.append(f"{report['stage']} differs beyond the async-load factor")
            provenance = report.get("provenance") or {}
            if any(provenance.get(key) != reference_provenance.get(key) for key in comparable):
                failures.append(f"{report['stage']} provenance differs from the first unit")

    pairs: list[dict[str, Any]] = []
    ttft_logs: list[list[float]] = []
    tpot_logs: list[list[float]] = []
    for pair_id in range(1, 4):
        serial = next((item for item in reports if item["stage"] == f"pair-{pair_id}-serial"), None)
        overlap = next((item for item in reports if item["stage"] == f"pair-{pair_id}-overlap"), None)
        if serial is None or overlap is None:
            failures.append(f"pair {pair_id} is incomplete")
            continue
        try:
            pair_ttft = _paired_logs(serial, overlap, "ttft")
            pair_tpot = _paired_logs(serial, overlap, "tpot")
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            failures.append(f"pair {pair_id} cannot be paired: {exc}")
            continue
        ttft_logs.append(pair_ttft)
        tpot_logs.append(pair_tpot)
        pairs.append({
            "pair": pair_id,
            "request_count": len(pair_ttft),
            "ttft_effect": math.exp(statistics.median(pair_ttft)) - 1.0,
            "tpot_effect": math.exp(statistics.median(pair_tpot)) - 1.0,
        })

    statistics_report: dict[str, Any] = {}
    if len(ttft_logs) == 3 and not failures:
        ttft_estimate = _estimate(ttft_logs)
        ttft_interval = _bootstrap(ttft_logs, 0.95)
        tpot_estimate = _estimate(tpot_logs)
        tpot_interval = _bootstrap(tpot_logs, 0.90)
        statistics_report = {
            "estimator": "exp(mean_start_pair(median_request(log(overlap/serial))))-1",
            "pairs": pairs,
            "ttft": {
                "estimate": ttft_estimate,
                "interval": ttft_interval,
                "causal_benefit_gate": ttft_estimate <= -0.05 and float(ttft_interval["upper"]) < 0.0,
            },
            "tpot": {
                "estimate": tpot_estimate,
                "interval": tpot_interval,
                "equivalence_gate": float(tpot_interval["lower"]) >= -0.05 and float(tpot_interval["upper"]) <= 0.05,
            },
        }
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "campaign_path": str(campaign_path),
        "units": reports,
        "statistics": statistics_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-dir")
    parser.add_argument("--arm", choices=tuple(ARM_CASES))
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
        report = verify_unit(
            Path(args.unit_dir),
            arm=args.arm,
            expected_requests=args.expected_requests,
            oracle_benchmark=Path(args.oracle_benchmark) if args.oracle_benchmark else None,
        )
        output = Path(args.output) if args.output else Path(args.unit_dir) / "overlap_verification.json"
    else:
        report = summarize_campaign(Path(args.campaign))
        output = Path(args.output) if args.output else Path(args.campaign).with_name("overlap_summary.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "failures": report["failures"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
