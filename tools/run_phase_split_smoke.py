#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""D.11 Phase Split NPU semantic smoke runner.

Minimal runner that exercises the post-dispatch phase split code path on a real
NPU and compares output token ids against a no-offload baseline.

Usage::

    # 1-token baseline
    python tools/sew_offload/run_phase_split_smoke.py --mode no_offload \\
        --output-dir artifacts/sew_offload/runs/d11_no_offload_1tok \\
        --max-output-tokens 1

    # 1-token candidate
    python tools/sew_offload/run_phase_split_smoke.py --mode phase_split \\
        --output-dir artifacts/sew_offload/runs/d11_phase_split_1tok \\
        --max-output-tokens 1

    # Compare
    python tools/sew_offload/compare_smoke_outputs.py \\
        --baseline artifacts/sew_offload/runs/d11_no_offload_1tok/outputs.jsonl \\
        --candidate artifacts/sew_offload/runs/d11_phase_split_1tok/outputs.jsonl \\
        --output artifacts/sew_offload/runs/d11_correctness_compare_1tok.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml
from vllm import LLM, SamplingParams


DEFAULT_CONFIG = "benchmark/configs/sew_offload_v1.yaml"
DEFAULT_MODEL = "/data/shared-models/Qwen3-30B-A3B"
DEFAULT_PROMPT = "请用中文回答：什么是大语言模型？"
DEFAULT_TEMPERATURE = 0.0

PHASE_SPLIT_ENV_VARS = (
    "VLLM_ASCEND_MOE_OFFLOAD_PHASE_SPLIT",
    "VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D.11 Phase Split NPU Smoke")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=("no_offload", "phase_split"),
        default="phase_split",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-output-tokens", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-memory-mb", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser.parse_args()


def configure_env(mode: str, output_dir: Path) -> None:
    for env_name in PHASE_SPLIT_ENV_VARS:
        os.environ.pop(env_name, None)

    if mode == "no_offload":
        return

    if mode == "phase_split":
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_PHASE_SPLIT"] = "1"
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH"] = str(
            output_dir / "moe_offload_profile.jsonl"
        )


def run_smoke(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    configure_env(args.mode, output_dir)

    print(f"[D.11 smoke] mode={args.mode} model={args.model} max_tokens={args.max_output_tokens}")

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "enforce_eager": args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_memory_bytes": args.kv_cache_memory_mb * 1024 * 1024,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_expert_parallel": False,
        "disable_log_stats": False,
    }

    t_start = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_s = time.perf_counter() - t_start
    print(f"[D.11 smoke] LLM loaded in {load_s:.1f}s")

    sampling_params = SamplingParams(
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=args.max_output_tokens,
        ignore_eos=True,
    )

    prompts = [args.prompt]

    t_gen_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    gen_s = time.perf_counter() - t_gen_start
    print(f"[D.11 smoke] generation completed in {gen_s:.1f}s")

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / gen_s if gen_s > 0 else 0.0

    output_records = []
    for output in outputs:
        completion = output.outputs[0]
        token_ids = [int(tid) for tid in completion.token_ids]
        output_records.append(
            {
                "request_id": output.request_id,
                "output_text": completion.text,
                "output_token_ids": token_ids,
                "output_tokens": len(token_ids),
            }
        )
        print(f"[D.11 smoke] output token_ids: {token_ids}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = output_dir / "outputs.jsonl"
    with outputs_path.open("w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    profile_path = output_dir / "moe_offload_profile.jsonl"
    profile_events = []
    if profile_path.exists():
        with profile_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    profile_events.append(json.loads(line))
    phase_split_events = [e for e in profile_events if e.get("event") == "phase_split"]

    summary = {
        "status": "ok" if outputs else "error",
        "mode": args.mode,
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "load_seconds": round(load_s, 3),
        "generate_seconds": round(gen_s, 3),
        "output_throughput_tok_s": round(throughput, 4),
        "output_token_ids": output_records[0]["output_token_ids"] if output_records else [],
        "phase_split_profile_events": len(phase_split_events),
        "phase_split_plans": [
            e.get("phase_plan", {}) for e in phase_split_events if "phase_plan" in e
        ],
        "env": {
            k: os.environ.get(k, "")
            for k in PHASE_SPLIT_ENV_VARS
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[D.11 smoke] summary written to {summary_path}")
    print(f"[D.11 smoke] status={summary['status']} throughput={throughput:.4f} tok/s")

    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    try:
        summary = run_smoke(args, output_dir)
        if summary["status"] != "ok":
            print(f"[D.11 smoke] FAILED: {summary}", file=sys.stderr)
            sys.exit(1)
    except Exception:
        traceback.print_exc()
        error_summary = {
            "status": "error",
            "mode": args.mode,
            "error": traceback.format_exc(),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(error_summary, indent=2), encoding="utf-8"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
