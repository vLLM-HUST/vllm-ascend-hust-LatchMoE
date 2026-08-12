#!/usr/bin/env python3
"""Verify and summarize the bounded matched A/B campaign from Issue #17."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ARM_CASES = {
    "full_layer": "sew_14gb_full_layer_matched",
    "multi_wave": "sew_14gb_multi_wave_matched",
}
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


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{number}: {exc}") from exc
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    offset = (len(ordered) - 1) * pct / 100.0
    low = int(offset)
    high = min(low + 1, len(ordered) - 1)
    weight = offset - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _request_outputs(benchmark: dict[str, Any]) -> dict[str, list[int]]:
    outputs: dict[str, list[int]] = {}
    for item in benchmark.get("per_request") or []:
        request_id = str(item.get("request_id") or "")
        token_ids = item.get("output_token_ids")
        if not request_id or not isinstance(token_ids, list):
            continue
        if request_id in outputs:
            raise ValueError(f"duplicate request_id: {request_id}")
        outputs[request_id] = [int(token) for token in token_ids]
    return outputs


def _distribution(benchmark: dict[str, Any]) -> dict[str, Any]:
    ttft = []
    tpot = []
    for item in benchmark.get("per_request") or []:
        first = item.get("ttft_s")
        output_tokens = int(item.get("output_tokens") or 0)
        total = float(item.get("total_s") or 0.0)
        if first is None or output_tokens <= 0:
            continue
        first_s = float(first)
        ttft.append(first_s * 1000.0)
        tpot.append((total - first_s) / output_tokens * 1000.0)
    return {
        "count": len(ttft),
        "ttft_ms": {
            "p50": percentile(ttft, 50),
            "p95": percentile(ttft, 95),
            "min": min(ttft) if ttft else 0.0,
            "max": max(ttft) if ttft else 0.0,
            "values": ttft,
        },
        "tpot_ms": {
            "p50": percentile(tpot, 50),
            "p95": percentile(tpot, 95),
            "min": min(tpot) if tpot else 0.0,
            "max": max(tpot) if tpot else 0.0,
            "values": tpot,
        },
    }


def verify_unit(
    unit_dir: Path,
    *,
    arm: str,
    expected_requests: int = 200,
    oracle_benchmark: Path | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    required_names = (
        "unit_manifest.json",
        "unit_result.json",
        "benchmark.json",
        "server.log",
        "client.log",
        "launcher_lifecycle.log",
        "moe_profile.jsonl",
        "npu_samples.jsonl",
        "release_ack.json",
    )
    required = {name: unit_dir / name for name in required_names}
    for name, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty artifact: {name}")
    if failures:
        return {"status": "failed", "unit_dir": str(unit_dir), "failures": failures}

    if arm not in ARM_CASES:
        raise ValueError(f"unknown arm: {arm}")
    manifest = _read_json(required["unit_manifest.json"])
    result = _read_json(required["unit_result.json"])
    benchmark = _read_json(required["benchmark.json"])
    release = _read_json(required["release_ack.json"])
    records = _jsonl(required["moe_profile.jsonl"])
    npu_samples = _jsonl(required["npu_samples.jsonl"])
    server_log = required["server.log"].read_text(encoding="utf-8", errors="replace")
    provenance = manifest.get("provenance") or {}
    selected_env = manifest.get("selected_env") or {}

    if (manifest.get("case") or {}).get("name") != ARM_CASES[arm]:
        failures.append("case name does not match the declared arm")
    observed_mode = selected_env.get("VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE")
    if observed_mode != arm:
        failures.append(f"overflow mode is {observed_mode!r}, expected {arm!r}")
    if result.get("status") != "ok" or result.get("release_status") != "released":
        failures.append("managed unit did not complete and release cleanly")
    if release.get("status") != "released":
        failures.append("release ACK is not released")
    successful = int(benchmark.get("successful_requests") or 0)
    failed = int(benchmark.get("failed_requests") or 0)
    if successful != expected_requests or failed != 0:
        failures.append(f"request gate failed: successful={successful}, failed={failed}")

    outputs = _request_outputs(benchmark)
    if len(outputs) != expected_requests:
        failures.append(f"token arrays present for {len(outputs)}/{expected_requests} requests")
    oracle_report = None
    if oracle_benchmark is not None:
        oracle = _read_json(oracle_benchmark)
        expected = _request_outputs(oracle)
        missing = sorted(set(expected) - set(outputs))
        extra = sorted(set(outputs) - set(expected))
        mismatched = sorted(
            request_id
            for request_id in set(expected) & set(outputs)
            if expected[request_id] != outputs[request_id]
        )
        exact_tokens = sum(
            len(expected[request_id])
            for request_id in set(expected) & set(outputs)
            if request_id not in mismatched
        )
        oracle_report = {
            "path": str(oracle_benchmark),
            "sha256": _sha256(oracle_benchmark),
            "expected_requests": len(expected),
            "exact_requests": len(expected) - len(missing) - len(mismatched),
            "exact_tokens": exact_tokens,
            "missing_request_ids": missing,
            "extra_request_ids": extra,
            "mismatched_request_ids": mismatched,
        }
        if missing or extra or mismatched:
            failures.append("request IDs or output token IDs differ from matched oracle")

    graph_capture = "Graph capturing finished" in server_log
    graph_replay = bool(re.search(r"Replaying aclgraph", server_log))
    graph_config = (
        "LATCHMOE_GRAPH_CONFIG cudagraph_mode=PIECEWISE" in server_log
        and "splitting_op=vllm::moe_offload_stage" in server_log
    )
    if not graph_capture or not graph_replay or not graph_config:
        failures.append("PIECEWISE Graph capture/replay evidence is incomplete")
    error_counts = {
        marker: server_log.count(marker)
        for marker in FORBIDDEN_MARKERS
        if marker in server_log
    }
    if error_counts:
        failures.append(f"forbidden NPU/ACL/OOM/runtime markers observed: {error_counts}")

    names: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        names.setdefault(str(record.get("name") or ""), []).append(record)
    full_events = names.get("b2_reference_full_layer_prefill", [])
    wave_events = names.get("b2_work_conserving_prefill", [])
    fallback_events = [
        record
        for record in full_events
        if (record.get("payload") or {}).get("fallback_reason") not in (None, "")
    ]
    if arm == "full_layer":
        if not full_events or wave_events:
            failures.append("full_layer arm did not exclusively use reference_full_layer")
    else:
        if not wave_events or full_events:
            failures.append("multi_wave arm fell back to or mixed with full_layer")
        invalid_modes = sorted(
            {
                str((record.get("payload") or {}).get("execution_mode"))
                for record in wave_events
                if (record.get("payload") or {}).get("execution_mode")
                != "pair_microbatch"
            }
        )
        if invalid_modes:
            failures.append(f"unexpected multi_wave execution modes: {invalid_modes}")

    for key in (
        "repository_head_sha",
        "repository_parent_sha",
        "compatibility_lock_sha256",
        "model_config_sha256",
        "dataset_manifest_sha256",
        "device",
        "runtime_bundle_sha256",
        "vllm_root_sha",
        "seam_root_sha",
    ):
        if not provenance.get(key):
            failures.append(f"missing provenance field: {key}")
    if provenance.get("runtime_paths_dirty") is not False:
        failures.append("runtime paths were dirty")

    distribution = _distribution(benchmark)
    if distribution["count"] != expected_requests:
        failures.append("raw latency distribution is incomplete")
    hbm = [
        int(item["hbm_usage_percent"])
        for item in npu_samples
        if item.get("hbm_usage_percent") is not None
    ]
    return {
        "status": "passed" if not failures else "failed",
        "unit_dir": str(unit_dir),
        "arm": arm,
        "failures": failures,
        "requests": {
            "successful": successful,
            "failed": failed,
            "output_tokens": int(benchmark.get("total_output_tokens") or 0),
        },
        "distribution": distribution,
        "throughput_tok_s": float(benchmark.get("output_throughput") or 0.0),
        "runtime": {
            "graph_capture": graph_capture,
            "graph_replay": graph_replay,
            "full_layer_events": len(full_events),
            "multi_wave_events": len(wave_events),
            "fallback_events": len(fallback_events),
            "error_counts": error_counts,
        },
        "memory": {
            "hbm_peak_percent": max(hbm) if hbm else None,
            "hbm_final_percent": hbm[-1] if hbm else None,
        },
        "provenance": provenance,
        "oracle": oracle_report,
        "artifact_sha256": {
            name: _sha256(path) for name, path in required.items()
        },
    }


def summarize_campaign(campaign_path: Path) -> dict[str, Any]:
    campaign = _read_json(campaign_path)
    units = campaign.get("units") or []
    failures: list[str] = []
    if [item.get("arm") for item in units] != [
        "full_layer",
        "multi_wave",
        "multi_wave",
        "full_layer",
        "full_layer",
        "multi_wave",
    ]:
        failures.append("campaign order is not the fixed AB/BA/AB sequence")
    reports = []
    baseline: dict[str, Any] | None = None
    comparable_keys = (
        "repository_head_sha",
        "repository_parent_sha",
        "compatibility_lock_sha256",
        "model_config_sha256",
        "dataset_manifest_sha256",
        "device",
        "runtime_bundle_sha256",
        "vllm_root_sha",
        "seam_root_sha",
    )
    for item in units:
        unit_dir = Path(item["unit_dir"])
        report = _read_json(unit_dir / "issue17_verification.json")
        reports.append({"stage": item["stage"], **report})
        if report.get("status") != "passed":
            failures.append(f"{item['stage']} verification failed")
        observed = report.get("provenance") or {}
        if baseline is None:
            baseline = observed
        elif any(observed.get(key) != baseline.get(key) for key in comparable_keys):
            failures.append(f"{item['stage']} provenance differs from the first unit")
    arms: dict[str, list[dict[str, Any]]] = {"full_layer": [], "multi_wave": []}
    for report in reports:
        arms[str(report["arm"])].append(report)
    if any(len(values) != 3 for values in arms.values()):
        failures.append("each arm must contain exactly three independent repeats")

    def arm_summary(values: list[dict[str, Any]]) -> dict[str, Any]:
        p50 = [item["distribution"]["ttft_ms"]["p50"] for item in values]
        p95 = [item["distribution"]["ttft_ms"]["p95"] for item in values]
        tpot50 = [item["distribution"]["tpot_ms"]["p50"] for item in values]
        tpot95 = [item["distribution"]["tpot_ms"]["p95"] for item in values]
        return {
            "repeats": len(values),
            "ttft_p50_ms_by_repeat": p50,
            "ttft_p95_ms_by_repeat": p95,
            "ttft_p50_ms_median": percentile(p50, 50),
            "ttft_p95_ms_median": percentile(p95, 50),
            "tpot_p50_ms_by_repeat": tpot50,
            "tpot_p95_ms_by_repeat": tpot95,
            "throughput_tok_s_by_repeat": [item["throughput_tok_s"] for item in values],
            "successful_requests": sum(item["requests"]["successful"] for item in values),
            "output_tokens": sum(item["requests"]["output_tokens"] for item in values),
            "fallback_events": sum(item["runtime"]["fallback_events"] for item in values),
            "error_counts": sum(len(item["runtime"]["error_counts"]) for item in values),
        }

    summaries = {arm: arm_summary(values) for arm, values in arms.items()}
    paired = []
    for pair in range(1, 4):
        left = next((item for item in reports if item["arm"] == "full_layer" and item["stage"].startswith(f"pair-{pair}-")), None)
        right = next((item for item in reports if item["arm"] == "multi_wave" and item["stage"].startswith(f"pair-{pair}-")), None)
        if left is None or right is None:
            failures.append(f"pair {pair} is incomplete")
            continue
        full_p50 = left["distribution"]["ttft_ms"]["p50"]
        wave_p50 = right["distribution"]["ttft_ms"]["p50"]
        full_p95 = left["distribution"]["ttft_ms"]["p95"]
        wave_p95 = right["distribution"]["ttft_ms"]["p95"]
        paired.append(
            {
                "pair": pair,
                "ttft_p50_reduction_percent": (full_p50 - wave_p50) / full_p50 * 100.0,
                "ttft_p95_reduction_percent": (full_p95 - wave_p95) / full_p95 * 100.0,
            }
        )
    full = summaries["full_layer"]
    wave = summaries["multi_wave"]
    comparison: dict[str, Any] = {"paired_reductions": paired}
    if full["repeats"] and wave["repeats"]:
        comparison.update(
            {
                "ttft_p50_reduction_percent": (
                    full["ttft_p50_ms_median"] - wave["ttft_p50_ms_median"]
                ) / full["ttft_p50_ms_median"] * 100.0,
                "ttft_p95_reduction_percent": (
                    full["ttft_p95_ms_median"] - wave["ttft_p95_ms_median"]
                ) / full["ttft_p95_ms_median"] * 100.0,
            }
        )
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "campaign_path": str(campaign_path),
        "units": reports,
        "arms": summaries,
        "comparison": comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-dir")
    parser.add_argument("--arm", choices=tuple(ARM_CASES))
    parser.add_argument("--oracle-benchmark")
    parser.add_argument("--expected-requests", type=int, default=200)
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
            oracle_benchmark=(Path(args.oracle_benchmark) if args.oracle_benchmark else None),
        )
        output = Path(args.output) if args.output else Path(args.unit_dir) / "issue17_verification.json"
    else:
        report = summarize_campaign(Path(args.campaign))
        output = Path(args.output) if args.output else Path(args.campaign).with_name("matched_summary.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "failures": report["failures"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
