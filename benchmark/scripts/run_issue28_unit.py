#!/usr/bin/env python3
"""Adapt one standard benchmark-suite unit to the Issue #28 artifact contract."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from issue28_campaign import contract_digest, load_contract, read_json  # noqa: E402


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def _copy_if_present(source: Path, target: Path, names: list[str]) -> list[str]:
    copied: list[str] = []
    for name in names:
        path = source / name
        if path.is_file():
            shutil.copy2(path, target / name)
            copied.append(name)
    return copied


def _materialize_artifacts(source: Path, target: Path, contract: dict[str, Any], digest: str) -> None:
    benchmark = read_json(source / "benchmark.json") if (source / "benchmark.json").is_file() else {}
    per_request = benchmark.get("per_request") or []
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    outputs: dict[str, list[int]] = {}
    for item in per_request:
        request_id = str(item.get("request_id") or "")
        if request_id:
            outputs[request_id] = [int(token) for token in item.get("output_token_ids") or []]
        first = item.get("ttft_s")
        output_tokens = int(item.get("output_tokens") or 0)
        total = float(item.get("total_s") or 0.0)
        if first is not None and output_tokens > 0:
            ttft_ms.append(float(first) * 1000.0)
            tpot_ms.append((total - float(first)) / output_tokens * 1000.0)
    identity = {
        "campaign_id": contract["campaign_id"],
        "contract_sha256": digest,
        "identity": {
            "model": contract.get("model", {}),
            "request_manifest": contract.get("request_manifest", {}),
            "serving": contract.get("serving", {}),
        },
    }
    _write(target / "metrics.json", {
        **identity,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "throughput_tok_s": [float(benchmark.get("output_throughput") or 0.0)],
        "successful_requests": int(benchmark.get("successful_requests") or 0),
        "failed_requests": int(benchmark.get("failed_requests") or 0),
    })
    _write(target / "outputs.json", {**identity, "request_outputs": outputs})
    server_log = (source / "server.log").read_text(encoding="utf-8", errors="replace") if (source / "server.log").is_file() else ""
    profile = _jsonl(source / "moe_profile.jsonl")
    wave_counts = [
        int(((record.get("payload") or {}).get("wave_summary") or {}).get("wave_count") or 0)
        for record in profile
    ]
    fallback_count = sum(
        1
        for record in profile
        if (record.get("payload") or {}).get("fallback_reason") not in (None, "")
    )
    _write(target / "runtime.json", {
        **identity,
        "graph_mode": "PIECEWISE" if "cudagraph_mode=PIECEWISE" in server_log else "unknown",
        "graph_capture": "Graph capturing finished" in server_log,
        "graph_replay": "Replaying aclgraph" in server_log,
        "fallback_count": fallback_count,
        "wave_count": max(wave_counts or [1]),
    })
    npu_samples = _jsonl(source / "npu_samples.jsonl")
    hbm_values = [float(item["hbm_usage_percent"]) for item in npu_samples if item.get("hbm_usage_percent") is not None]
    hbm_capacity_mb = float((contract.get("serving") or {}).get("hbm_capacity_mb") or 0.0)
    _write(target / "memory.json", {
        **identity,
        "hbm_peak_percent": max(hbm_values) if hbm_values else None,
        "hbm_peak_mb": max(hbm_values) * hbm_capacity_mb / 100.0 if hbm_values and hbm_capacity_mb else None,
    })
    h2d_events = [
        record for record in profile
        if int((record.get("payload") or {}).get("h2d_bytes") or 0) > 0
    ]
    dependency_ok = bool(h2d_events) and all(
        (record.get("payload") or {}).get("consumer_dependency_installed") is True
        for record in h2d_events
    )
    _write(target / "transfers.json", {
        **identity,
        "h2d_count": len(h2d_events),
        "h2d_dependency_ok": dependency_ok,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--host-python", required=True, type=Path)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--seam-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--startup-timeout-s", type=float, default=1200.0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = load_contract(args.contract.resolve())
    digest = contract_digest(contract)
    suite_root = output / "suite"
    suite_root.mkdir()
    command = [
        str(args.python.resolve()), str(REPO_ROOT / "benchmark/scripts/run_suite.py"),
        "--config", str(args.config.resolve()), "--output-root", str(suite_root),
        "--case", args.case, "--workload", args.workload,
        "--managed-backend", "locked-host", "--server-manager", str(REPO_ROOT / "benchmark/scripts/manage_locked_host_runtime.py"),
        "--device", str(args.device), "--host-python", str(args.host_python.resolve()),
        "--vllm-root", str(args.vllm_root.resolve()), "--seam-root", str(args.seam_root.resolve()),
        "--release-ack-dir", str(output / "release-acks"), "--python", str(args.python.resolve()),
        "--manifest", str(args.manifest.resolve()), "--model-path", str(args.model_path.resolve()),
        "--dataset-path", str(args.dataset_path.resolve()), "--startup-timeout-s", str(args.startup_timeout_s),
        "--max-num-seqs", "1", "--client-concurrency", "1",
    ]
    if args.max_requests > 0:
        command.extend(["--max-requests", str(args.max_requests)])
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(completed.stdout, end="")
    suites = sorted(path for path in suite_root.glob("sew-offload-ascend-v1-*") if path.is_dir())
    if len(suites) != 1:
        _write(output / "unit_result.json", {"status": "failed", "release_status": "not_released", "error": f"expected one suite directory, found {len(suites)}", "raw_artifacts": ["stdout.log"]})
        return completed.returncode or 1
    source = suites[0] / args.case / args.workload
    if not source.is_dir():
        _write(output / "unit_result.json", {"status": "failed", "release_status": "not_released", "error": f"missing suite unit: {source}", "raw_artifacts": ["stdout.log"]})
        return completed.returncode or 1
    copied = _copy_if_present(source, output, [
        "server.log", "client.log", "launcher_lifecycle.log", "moe_profile.jsonl", "moe_trace.jsonl", "npu_samples.jsonl", "release_ack.json", "benchmark.json", "summary.md", "PASSED.txt", "FAILED.txt",
    ])
    source_result = read_json(source / "unit_result.json") if (source / "unit_result.json").is_file() else {}
    status = "success" if source_result.get("status") == "ok" and completed.returncode == 0 else "failed"
    _materialize_artifacts(source, output, contract, digest)
    raw = [name for name in copied if name not in {"unit_manifest.json", "unit_result.json"}]
    raw.extend(name for name in ("metrics.json", "outputs.json", "runtime.json", "memory.json", "transfers.json") if name not in raw)
    _write(output / "unit_result.json", {
        "status": status,
        "release_status": source_result.get("release_status", "not_released"),
        "raw_artifacts": raw,
        "source_unit": str(source),
        "source_result": source_result,
    })
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
