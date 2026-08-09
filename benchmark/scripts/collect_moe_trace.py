# SPDX-License-Identifier: Apache-2.0
"""Collect routed MoE expert traces for SEW-Offload.

This script enables MVP-B trace-only mode, runs a small vLLM workload, and
exports the in-memory MoE trace collector as JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

import yaml
from vllm import LLM, SamplingParams

from benchmark.scripts.sharegpt_manifest import (
    assert_no_random_dataset,
    build_sharegpt_manifest,
)
from vllm_moe_offload_ascend.moe_offload.runtime import get_moe_offload_runtime, reset_moe_offload_runtime


DEFAULT_CONFIG = "benchmark/configs/sew_offload_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--prepare-smoke-manifest", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-requests-per-bucket", type=int, default=1)
    parser.add_argument("--buckets", default="short_chat")
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-memory-mb", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.enforce_eager:
        parser.error("--enforce-eager is forbidden for graph-only LatchMoE runs")
    return args


def csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _bucket_target_prompt_tokens(bucket: dict[str, Any], index: int) -> int:
    prompt_tokens = bucket["prompt_tokens"]
    if prompt_tokens == "mixed":
        mixed_targets = [192, 768, 3072, 384]
        return mixed_targets[index % len(mixed_targets)]
    low, high = prompt_tokens
    if high <= low:
        return int(low)
    ratio = (index % 5) / 4
    return int(round(low + (high - low) * ratio))


def prepare_sharegpt_manifest(
    *,
    config: dict[str, Any],
    manifest_path: Path,
    requests_per_bucket: int,
    buckets: set[str] | None,
    model_path: str | None = None,
) -> None:
    # requests_per_bucket <= 0 means "use each bucket's full configured count".
    cap = requests_per_bucket if requests_per_bucket and requests_per_bucket > 0 else None
    written = build_sharegpt_manifest(
        config=config,
        manifest_path=manifest_path,
        model_path=model_path or config["model"]["path"],
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


def collect_trace(args: argparse.Namespace, config: dict[str, Any], requests: list[dict[str, Any]]) -> int:
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_ENABLED"] = "1"
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY"] = "1"
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH"] = str(args.output)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    reset_moe_offload_runtime()

    model_path = args.model or config["model"]["path"]
    llm = LLM(
        model=model_path,
        tensor_parallel_size=int(config["model"]["tensor_parallel_size"]),
        trust_remote_code=False,
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_mb * 1024 * 1024,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_expert_parallel=False,
        seed=int(config["dataset"]["seed"]),
        disable_log_stats=False,
    )
    sampling_params = [
        SamplingParams(
            max_tokens=int(req["max_output_tokens"]),
            temperature=float(req.get("temperature", 0.0)),
            top_p=float(req.get("top_p", 1.0)),
            ignore_eos=args.ignore_eos,
        )
        for req in requests
    ]
    llm.generate([req["prompt"] for req in requests], sampling_params, use_tqdm=False)
    jsonl_records = _count_jsonl_records(output_path)
    if jsonl_records > 0:
        return jsonl_records
    return get_moe_offload_runtime().export_trace(output_path)


def _count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    assert_no_random_dataset(config)
    selected_buckets = csv_set(args.buckets) or None
    manifest_path = Path(args.manifest or config["dataset"]["manifest_path"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.prepare_smoke_manifest:
        prepare_sharegpt_manifest(
            config=config,
            manifest_path=manifest_path,
            requests_per_bucket=args.smoke_requests_per_bucket,
            buckets=selected_buckets,
            model_path=args.model,
        )
        if args.prepare_only:
            print(f"PREPARE_OK manifest={manifest_path}", flush=True)
            return

    try:
        requests = load_manifest(manifest_path, selected_buckets, args.max_requests)
        num_records = collect_trace(args, config, requests)
        summary = {
            "status": "ok",
            "output": str(output_path),
            "manifest": str(manifest_path),
            "num_requests": len(requests),
            "num_trace_records": num_records,
        }
        print("TRACE_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    except BaseException as exc:
        print(f"TRACE_FAILED {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
