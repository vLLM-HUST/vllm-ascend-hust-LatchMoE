# SPDX-License-Identifier: Apache-2.0
"""Minimal SEW-Offload benchmark runner.

The runner follows benchmark/configs/sew_offload_v1.yaml and reports only the
current-stage metrics: throughput, TTFT, and TPOT.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch_npu
import yaml
from vllm import LLM, SamplingParams

from tools.sharegpt_manifest import (
    assert_no_random_dataset,
    build_sharegpt_manifest,
)


DEFAULT_CONFIG = "benchmark/configs/sew_offload_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--case-name", default="native_prefetch_experts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--prepare-smoke-manifest", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-requests-per-bucket", type=int, default=1)
    parser.add_argument("--buckets", default="")
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-memory-mb", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-backend", default="prefetch")
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--cpu-offload-params", default="")
    parser.add_argument("--offload-group-size", type=int, default=4)
    parser.add_argument("--offload-num-in-group", type=int, default=1)
    parser.add_argument("--offload-prefetch-step", type=int, default=1)
    parser.add_argument("--offload-params", default="experts")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    required = {
        "benchmark",
        "model",
        "dataset",
        "workload_buckets",
        "concurrency",
        "offload_budget",
        "metrics",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"benchmark config missing keys: {sorted(missing)}")
    return config


def _bucket_filter(buckets: str) -> set[str] | None:
    selected = _csv_set(buckets)
    return selected or None


def prepare_sharegpt_manifest(
    config: dict[str, Any],
    manifest_path: Path,
    requests_per_bucket: int,
    buckets: set[str] | None,
) -> None:
    # requests_per_bucket <= 0 means "use each bucket's full configured count".
    cap = requests_per_bucket if requests_per_bucket and requests_per_bucket > 0 else None
    written = build_sharegpt_manifest(
        config=config,
        manifest_path=manifest_path,
        model_path=config["model"]["path"],
        requests_per_bucket=cap,
        buckets=buckets,
    )
    print(f"MANIFEST_OK written={written} path={manifest_path}", flush=True)


def load_manifest(
    manifest_path: Path,
    buckets: set[str] | None,
    max_requests: int,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if buckets is not None and record.get("bucket") not in buckets:
                continue
            requests.append(record)
            if max_requests and len(requests) >= max_requests:
                break
    if not requests:
        raise ValueError(f"no requests selected from manifest: {manifest_path}")
    return requests


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * percentile / 100
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    weight = k - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 90),
    }


def _metrics_from_outputs(outputs: list[Any], duration_s: float) -> dict[str, Any]:
    ttfts_ms: list[float] = []
    tpots_ms: list[float] = []
    output_tokens = 0
    per_request: list[dict[str, Any]] = []

    for output in outputs:
        completion = output.outputs[0]
        num_tokens = len(completion.token_ids)
        output_tokens += num_tokens

        metrics = output.metrics
        ttft_ms = 0.0
        tpot_ms = 0.0
        if metrics is not None:
            ttft_ms = float(metrics.first_token_latency) * 1000
            if num_tokens > 1 and metrics.last_token_ts and metrics.first_token_ts:
                tpot_ms = (
                    float(metrics.last_token_ts - metrics.first_token_ts)
                    / (num_tokens - 1)
                    * 1000
                )
        ttfts_ms.append(ttft_ms)
        if num_tokens > 1:
            tpots_ms.append(tpot_ms)
        per_request.append(
            {
                "request_id": output.request_id,
                "output_tokens": num_tokens,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
            }
        )

    ttft_summary = _summarize(ttfts_ms)
    tpot_summary = _summarize(tpots_ms)
    return {
        "duration_s": duration_s,
        "completed": len(outputs),
        "total_output_tokens": output_tokens,
        "request_throughput_req_s": len(outputs) / duration_s if duration_s else 0.0,
        "request_throughput": len(outputs) / duration_s if duration_s else 0.0,
        "output_throughput_tok_s": output_tokens / duration_s if duration_s else 0.0,
        "output_throughput": output_tokens / duration_s if duration_s else 0.0,
        "median_ttft_ms": ttft_summary["median"],
        "median_tpot_ms": tpot_summary["median"],
        "ttft_ms": ttft_summary,
        "tpot_ms": tpot_summary,
        "per_request": per_request,
    }


def run_benchmark(args: argparse.Namespace, config: dict[str, Any], requests: list[dict[str, Any]]) -> dict[str, Any]:
    cpu_offload_params = _csv_set(args.cpu_offload_params)
    offload_params = _csv_set(args.offload_params)

    print(f"CASE {args.case_name}", flush=True)
    print(f"benchmark {config['benchmark']['id']} {config['benchmark']['config_version']}", flush=True)
    print(f"model {config['model']['path']}", flush=True)
    print(f"ASCEND_RT_VISIBLE_DEVICES {os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}", flush=True)
    print(f"torch {torch.__version__} torch_npu {torch_npu.__version__}", flush=True)
    print(
        "offload "
        f"backend={args.offload_backend} cpu_gb={args.cpu_offload_gb} "
        f"cpu_params={sorted(cpu_offload_params)} group_size={args.offload_group_size} "
        f"num_in_group={args.offload_num_in_group} prefetch_step={args.offload_prefetch_step} "
        f"params={sorted(offload_params)}",
        flush=True,
    )
    print(
        f"requests={len(requests)} concurrency={args.concurrency} "
        f"max_model_len={args.max_model_len} max_num_batched_tokens={args.max_num_batched_tokens}",
        flush=True,
    )

    load_t0 = time.perf_counter()
    llm = LLM(
        model=config["model"]["path"],
        tensor_parallel_size=int(config["model"]["tensor_parallel_size"]),
        trust_remote_code=False,
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_mb * 1024 * 1024,
        max_model_len=args.max_model_len,
        max_num_seqs=args.concurrency,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_expert_parallel=False,
        seed=int(config["dataset"]["seed"]),
        offload_backend=args.offload_backend,
        cpu_offload_gb=args.cpu_offload_gb,
        cpu_offload_params=cpu_offload_params,
        offload_group_size=args.offload_group_size,
        offload_num_in_group=args.offload_num_in_group,
        offload_prefetch_step=args.offload_prefetch_step,
        offload_params=offload_params,
        disable_log_stats=False,
    )
    load_s = time.perf_counter() - load_t0
    print(f"LOAD_OK seconds={load_s:.3f}", flush=True)

    if args.warmup:
        warmup = requests[:1]
        warmup_params = [
            SamplingParams(
                max_tokens=min(8, int(req["max_output_tokens"])),
                temperature=0.0,
                top_p=1.0,
                ignore_eos=args.ignore_eos,
            )
            for req in warmup
        ]
        llm.generate([req["prompt"] for req in warmup], warmup_params, use_tqdm=False)

    sampling_params = [
        SamplingParams(
            max_tokens=int(req["max_output_tokens"]),
            temperature=float(req.get("temperature", 0.0)),
            top_p=float(req.get("top_p", 1.0)),
            ignore_eos=args.ignore_eos,
        )
        for req in requests
    ]
    gen_t0 = time.perf_counter()
    outputs = llm.generate(
        [req["prompt"] for req in requests],
        sampling_params,
        use_tqdm=False,
    )
    gen_s = time.perf_counter() - gen_t0
    summary = _metrics_from_outputs(outputs, gen_s)
    summary["load_seconds"] = load_s
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(args.config)
    assert_no_random_dataset(config)
    selected_buckets = _bucket_filter(args.buckets)
    manifest_path = Path(args.manifest or config["dataset"]["manifest_path"])

    if args.prepare_smoke_manifest:
        prepare_sharegpt_manifest(
            config,
            manifest_path,
            args.smoke_requests_per_bucket,
            selected_buckets,
        )
        if args.prepare_only:
            print(f"PREPARE_OK manifest={manifest_path}", flush=True)
            return

    summary_path = output_dir / "summary.json"
    try:
        requests = load_manifest(manifest_path, selected_buckets, args.max_requests)
        summary = run_benchmark(args, config, requests)
        summary.update(
            {
                "status": "ok",
                "case_name": args.case_name,
                "manifest": str(manifest_path),
                "buckets": sorted(selected_buckets) if selected_buckets else "all",
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    except BaseException as exc:
        failure = {
            "status": "failed",
            "case_name": args.case_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "manifest": str(manifest_path),
        }
        summary_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(f"CASE_FAILED {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
