# SPDX-License-Identifier: Apache-2.0
"""Run a minimal fixed-slot SEW-Offload smoke workload.

This runner is intentionally small and artifact-oriented. It is for checking
whether the MVP-D fixed-slot path reaches generation on a real NPU; it is not a
publishable benchmark runner.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from vllm import LLM, SamplingParams

from vllm_moe_offload_ascend import register as register_moe_offload
from benchmark.scripts.collect_moe_trace import (
    csv_set,
    load_manifest,
    prepare_sharegpt_manifest,
)
from benchmark.scripts.sharegpt_manifest import assert_no_random_dataset
from vllm_moe_offload_ascend.moe_offload.runtime import get_moe_offload_runtime, reset_moe_offload_runtime


DEFAULT_CONFIG = "benchmark/configs/sew_offload_v1.yaml"
SEW_OFFLOAD_ENV_VARS = (
    "VLLM_ASCEND_MOE_OFFLOAD_ENABLED",
    "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY",
    "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS",
    "VLLM_ASCEND_MOE_OFFLOAD_POLICY",
    "VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD",
    "VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES",
    "VLLM_ASCEND_MOE_OFFLOAD_TRACE_MAX_RECORDS",
    "VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH",
    "VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS",
    "VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS",
    "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME",
    "VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE",
    "VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD",
    "VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH",
    "VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM",
    "VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD",
    "VLLM_ASCEND_MOE_OFFLOAD_COMPARE_ROUTER",
    "VLLM_ASCEND_MOE_OFFLOAD_COMPARE_LAYER_BOUNDARY",
    "VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL",
    "VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE",
    "VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT",
    "VLLM_ASCEND_MOE_ROUTER_PARITY_PATH",
    "VLLM_ASCEND_MOE_ROUTER_PARITY_MAX_TOKENS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=("no_offload", "trace_only", "fixed_slot_sync"),
        default="fixed_slot_sync",
        help="Single-process smoke mode. Run modes separately for correctness comparison.",
    )
    parser.add_argument("--model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--prepare-smoke-manifest", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-requests-per-bucket", type=int, default=1)
    parser.add_argument("--inline-prompt")
    parser.add_argument("--inline-prompts-jsonl")
    parser.add_argument("--inline-max-output-tokens", type=int, default=1)
    parser.add_argument("--override-max-output-tokens", type=int)
    parser.add_argument("--buckets", default="short_chat")
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-memory-mb", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--ascend-additional-config-json",
        default="{}",
        help=(
            "JSON object forwarded as vLLM-Ascend additional_config and recorded "
            "in the smoke summary, for example '{\"mix_placement\": true}'."
        ),
    )
    parser.add_argument(
        "--disable-ascend-norm-quant-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable vLLM and Ascend norm-quant compiler passes for CANN compatibility probes.",
    )
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--diagnostic-eager",
        action="store_true",
        help="Allow eager execution only for an explicitly labelled qualification diagnostic.",
    )
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-slots", type=int, default=16)
    parser.add_argument("--resident-layer-ids", default="")
    parser.add_argument("--release-original-expert-weights", action="store_true")
    parser.add_argument("--layered-runtime", action="store_true")
    parser.add_argument("--fanout-threshold", type=int, default=0)
    parser.add_argument(
        "--stage-seam",
        action="store_true",
        help="Enable the graph splitting seam for a fixed-slot qualification run.",
    )
    parser.add_argument(
        "--cpu-first-load",
        action="store_true",
        help="Load routed expert weights CPU-first before fixed-slot registration.",
    )
    parser.add_argument(
        "--router-parity",
        action="store_true",
        help="Capture bounded native and seam router snapshots in eager diagnostics.",
    )
    parser.add_argument("--router-parity-max-tokens", type=int, default=64)
    parser.add_argument(
        "--layer-boundary-parity",
        action="store_true",
        help="Compare native and seam MoE tuple outputs in eager diagnostics.",
    )
    parser.add_argument(
        "--wave-prefill",
        action="store_true",
        help="Enable the multi-wave prefill qualification path.",
    )
    parser.add_argument("--offload-backend", default="prefetch")
    parser.add_argument("--offload-group-size", type=int, default=4)
    parser.add_argument("--offload-num-in-group", type=int, default=1)
    parser.add_argument("--offload-prefetch-step", type=int, default=1)
    parser.add_argument("--offload-params", default="experts")
    parser.add_argument(
        "--with-native-offload-backend",
        action="store_true",
        help=(
            "Also engage vLLM's native offloader in fixed_slot_sync mode. "
            "Disabled by default because SEW manages its own offloading and "
            "the native backend requires pinned CPU storage (conflicts with "
            "SEW host store, fails profile_run with 'not pinned' assertion)."
        ),
    )
    args = parser.parse_args()
    if args.enforce_eager and not args.diagnostic_eager:
        parser.error("--enforce-eager requires --diagnostic-eager")
    if args.diagnostic_eager and not args.enforce_eager:
        parser.error("--diagnostic-eager requires --enforce-eager")
    if args.stage_seam and args.mode != "fixed_slot_sync":
        parser.error("--stage-seam requires --mode fixed_slot_sync")
    if args.cpu_first_load and not args.stage_seam:
        parser.error("--cpu-first-load requires --stage-seam")
    if args.router_parity and (not args.stage_seam or not args.diagnostic_eager):
        parser.error("--router-parity requires an eager fixed-slot seam diagnostic")
    if args.layer_boundary_parity and (not args.stage_seam or not args.diagnostic_eager):
        parser.error("--layer-boundary-parity requires an eager fixed-slot seam diagnostic")
    if args.wave_prefill and not args.stage_seam:
        parser.error("--wave-prefill requires --stage-seam")
    try:
        args.ascend_additional_config = json.loads(
            args.ascend_additional_config_json
        )
    except json.JSONDecodeError as exc:
        parser.error(f"--ascend-additional-config-json is not valid JSON: {exc.msg}")
    if not isinstance(args.ascend_additional_config, dict):
        parser.error("--ascend-additional-config-json must decode to a JSON object")
    return args


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_inline_request(*, prompt: str, max_output_tokens: int) -> dict[str, Any]:
    return {
        "request_id": "inline_0000",
        "bucket": "inline",
        "prompt": prompt,
        "max_output_tokens": int(max_output_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "dataset": "inline_smoke",
    }


def load_inline_prompts_jsonl(
    path: str | Path,
    *,
    default_max_output_tokens: int,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"inline prompt record {index} is missing a non-empty prompt")
            request_id = str(record.get("request_id") or f"inline_{index:04d}")
            request = make_inline_request(
                prompt=prompt,
                max_output_tokens=int(record.get("max_output_tokens", default_max_output_tokens)),
            )
            request["request_id"] = request_id
            requests.append(request)
    if not requests:
        raise ValueError(f"no inline prompts selected from {path}")
    return requests


def override_request_max_output_tokens(
    requests: list[dict[str, Any]],
    *,
    max_output_tokens: int | None,
) -> list[dict[str, Any]]:
    if max_output_tokens is None:
        return requests
    return [
        {
            **request,
            "max_output_tokens": int(max_output_tokens),
        }
        for request in requests
    ]


def configure_sew_offload_env(
    mode: str,
    *,
    num_slots: int,
    max_num_seqs_hint: int | None = None,
    resident_layer_ids: str = "",
    release_original_expert_weights: bool = False,
    layered_runtime: bool = False,
    fanout_threshold: int = 0,
    trace_path: str = "moe_offload_trace.jsonl",
) -> None:
    if mode == "no_offload":
        for env_name in SEW_OFFLOAD_ENV_VARS:
            os.environ.pop(env_name, None)
        return
    elif mode == "trace_only":
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_ENABLED"] = "1"
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY"] = "1"
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"] = "0"
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH"] = trace_path
    elif mode == "fixed_slot_sync":
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_ENABLED"] = "1"
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY"] = "0"
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"] = str(int(num_slots))
    else:
        raise ValueError(f"unsupported smoke mode: {mode}")
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD"] = "0"
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES"] = "1"
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS"] = resident_layer_ids
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS"] = (
        "1" if release_original_expert_weights else "0"
    )
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME"] = "1" if layered_runtime else "0"
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE"] = (
        "1" if mode == "fixed_slot_sync" else "0"
    )
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD"] = str(int(fanout_threshold))
    if max_num_seqs_hint is not None:
        if int(max_num_seqs_hint) <= 0:
            raise ValueError("max_num_seqs_hint must be greater than zero")
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT"] = str(
            int(max_num_seqs_hint)
        )
    os.environ.setdefault("VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH", "moe_offload_profile.jsonl")


def _csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _output_records(outputs: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for output in outputs:
        completion = output.outputs[0]
        token_ids = [int(token_id) for token_id in getattr(completion, "token_ids", [])]
        output_text = getattr(completion, "text", "")
        if not isinstance(output_text, str):
            output_text = ""
        records.append(
            {
                "request_id": output.request_id,
                "output_text": output_text,
                "output_token_ids": token_ids,
                "output_tokens": len(token_ids),
            }
        )
    return records


def _write_outputs_jsonl(output_dir: Path, outputs: list[Any]) -> None:
    records = _output_records(outputs)
    with (output_dir / "outputs.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_profile_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


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


def _smoke_compilation_config(
    *,
    mode: str,
    enforce_eager: bool,
    disable_ascend_norm_quant_fusion: bool,
) -> dict[str, Any] | None:
    config: dict[str, Any] = {}
    if mode == "fixed_slot_sync" and not enforce_eager:
        config["cudagraph_mode"] = "PIECEWISE"
    if disable_ascend_norm_quant_fusion:
        config["pass_config"] = {"fuse_norm_quant": False}
    return config or None


def _metrics_from_outputs(outputs: list[Any], duration_s: float) -> dict[str, Any]:
    total_output_tokens = 0
    ttfts_ms: list[float] = []
    tpots_ms: list[float] = []
    per_request: list[dict[str, Any]] = []
    for output in outputs:
        completion = output.outputs[0]
        num_tokens = len(completion.token_ids)
        total_output_tokens += num_tokens
        request_metrics = getattr(output, "metrics", None)
        ttft_ms = 0.0
        tpot_ms = 0.0
        if request_metrics is not None:
            ttft_ms = float(getattr(request_metrics, "first_token_latency", 0.0)) * 1000
            first_token_ts = getattr(request_metrics, "first_token_ts", None)
            last_token_ts = getattr(request_metrics, "last_token_ts", None)
            if num_tokens > 1 and first_token_ts and last_token_ts:
                tpot_ms = float(last_token_ts - first_token_ts) / (num_tokens - 1) * 1000
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
    return {
        "duration_s": duration_s,
        "completed": len(outputs),
        "total_output_tokens": total_output_tokens,
        "output_throughput_tok_s": total_output_tokens / duration_s if duration_s else 0.0,
        "ttft_ms": _summarize(ttfts_ms),
        "tpot_ms": _summarize(tpots_ms),
        "per_request": per_request,
    }


def _build_llm_kwargs(args: argparse.Namespace, config: dict[str, Any], mode: str) -> dict[str, Any]:
    model_path = args.model or config["model"]["path"]
    kwargs: dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": int(config["model"]["tensor_parallel_size"]),
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "enforce_eager": args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_memory_bytes": args.kv_cache_memory_mb * 1024 * 1024,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_expert_parallel": False,
        "seed": int(config["dataset"]["seed"]),
        "disable_log_stats": False,
        "enable_prefix_caching": False,
    }
    if mode == "fixed_slot_sync" and getattr(args, "with_native_offload_backend", False):
        # SEW manages its own offloading via host store + slot bank + original
        # weight release. The native vLLM offloader is a separate system that
        # requires pinned CPU storage and conflicts with SEW when both manage
        # experts (it fails in profile_run with "CPU storage ... is not pinned").
        # Only engage it when explicitly opted in.
        kwargs.update(
            {
                "offload_backend": args.offload_backend,
                "offload_group_size": args.offload_group_size,
                "offload_num_in_group": args.offload_num_in_group,
                "offload_prefetch_step": args.offload_prefetch_step,
                "offload_params": _csv_set(args.offload_params),
            }
        )
    additional_config = dict(getattr(args, "ascend_additional_config", {}) or {})
    disable_norm_quant = bool(getattr(args, "disable_ascend_norm_quant_fusion", False))
    if disable_norm_quant:
        compile_config = dict(
            additional_config.get("ascend_compilation_config") or {}
        )
        compile_config["fuse_norm_quant"] = False
        additional_config["ascend_compilation_config"] = compile_config
    if additional_config:
        kwargs["additional_config"] = additional_config
    compilation_config = _smoke_compilation_config(
        mode=mode,
        enforce_eager=bool(args.enforce_eager),
        disable_ascend_norm_quant_fusion=disable_norm_quant,
    )
    if compilation_config is not None:
        kwargs["compilation_config"] = compilation_config
    return kwargs


def run_smoke(
    args: argparse.Namespace,
    config: dict[str, Any],
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_jsonl_path = output_dir / "moe_offload_profile.jsonl"
    trace_jsonl_path = output_dir / "moe_offload_trace.jsonl"
    if profile_jsonl_path.exists():
        profile_jsonl_path.unlink()
    if trace_jsonl_path.exists():
        trace_jsonl_path.unlink()
    mode = getattr(args, "mode", "fixed_slot_sync")
    configure_sew_offload_env(
        mode,
        num_slots=args.num_slots,
        max_num_seqs_hint=(
            int(args.max_num_seqs)
            if bool(getattr(args, "wave_prefill", False))
            else None
        ),
        resident_layer_ids=getattr(args, "resident_layer_ids", ""),
        release_original_expert_weights=getattr(args, "release_original_expert_weights", False),
        layered_runtime=getattr(args, "layered_runtime", False),
        fanout_threshold=getattr(args, "fanout_threshold", 0),
        trace_path=str(trace_jsonl_path),
    )
    router_parity_path = output_dir / "moe_router_parity.jsonl"
    if bool(getattr(args, "stage_seam", False)):
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM"] = "1"
    if bool(getattr(args, "cpu_first_load", False)):
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD"] = "1"
    if bool(getattr(args, "router_parity", False)):
        if router_parity_path.exists():
            router_parity_path.unlink()
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_COMPARE_ROUTER"] = "1"
        os.environ["VLLM_ASCEND_MOE_ROUTER_PARITY_PATH"] = str(router_parity_path)
        os.environ["VLLM_ASCEND_MOE_ROUTER_PARITY_MAX_TOKENS"] = str(
            max(1, int(getattr(args, "router_parity_max_tokens", 64)))
        )
    if bool(getattr(args, "layer_boundary_parity", False)):
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_COMPARE_LAYER_BOUNDARY"] = "1"
    if bool(getattr(args, "wave_prefill", False)):
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL"] = "1"
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE"] = "multi_wave"
    if mode != "no_offload":
        os.environ["VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH"] = str(profile_jsonl_path)
        # vLLM selects a single platform plugin; explicitly install LatchMoE hooks.
        register_moe_offload()
    reset_moe_offload_runtime()

    llm_kwargs = _build_llm_kwargs(args, config, mode)
    load_t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_s = time.perf_counter() - load_t0

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
    outputs = llm.generate([req["prompt"] for req in requests], sampling_params, use_tqdm=False)
    gen_s = time.perf_counter() - gen_t0

    _write_outputs_jsonl(output_dir, outputs)
    summary = _metrics_from_outputs(outputs, gen_s)
    summary.update(
        {
            "status": "ok",
            "mode": mode,
            "model": llm_kwargs["model"],
            "num_slots": int(args.num_slots) if mode == "fixed_slot_sync" else 0,
            "layered_runtime": bool(getattr(args, "layered_runtime", False)) if mode == "fixed_slot_sync" else False,
            "graph_compatible_offload": bool(
                mode == "fixed_slot_sync"
            ),
            "fanout_threshold": int(getattr(args, "fanout_threshold", 0)) if mode == "fixed_slot_sync" else 0,
            "disable_ascend_norm_quant_fusion": bool(
                getattr(args, "disable_ascend_norm_quant_fusion", False)
            ),
            "ascend_additional_config": llm_kwargs.get("additional_config", {}),
            "load_seconds": load_s,
            "manifest": str(args.manifest),
            "buckets": args.buckets,
            "moe_offload_profile": get_moe_offload_runtime().profiling_summary(),
            "moe_offload_profile_jsonl": str(profile_jsonl_path),
            "moe_offload_profile_jsonl_events": _read_profile_jsonl(profile_jsonl_path),
            "qualification_diagnostics": {
                "diagnostic_eager": bool(getattr(args, "diagnostic_eager", False)),
                "stage_seam": bool(getattr(args, "stage_seam", False)),
                "cpu_first_load": bool(getattr(args, "cpu_first_load", False)),
                "release_original_expert_weights": bool(
                    getattr(args, "release_original_expert_weights", False)
                ),
                "graph_compatible_offload": bool(mode == "fixed_slot_sync"),
                "router_parity": bool(getattr(args, "router_parity", False)),
                "router_parity_artifact": (
                    str(router_parity_path)
                    if bool(getattr(args, "router_parity", False))
                    else None
                ),
                "layer_boundary_parity": bool(
                    getattr(args, "layer_boundary_parity", False)
                ),
                "wave_prefill": bool(getattr(args, "wave_prefill", False)),
                "ascend_additional_config": llm_kwargs.get(
                    "additional_config", {}
                ),
            },
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_fixed_slot_smoke(
    args: argparse.Namespace,
    config: dict[str, Any],
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    args.mode = "fixed_slot_sync"
    return run_smoke(args, config, requests)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    assert_no_random_dataset(config)
    selected_buckets = csv_set(args.buckets) or None
    manifest_path = Path(args.manifest or config["dataset"]["manifest_path"])
    args.manifest = str(manifest_path)

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
        if args.inline_prompts_jsonl is not None:
            requests = load_inline_prompts_jsonl(
                args.inline_prompts_jsonl,
                default_max_output_tokens=args.inline_max_output_tokens,
            )
        elif args.inline_prompt is not None:
            requests = [
                make_inline_request(
                    prompt=args.inline_prompt,
                    max_output_tokens=args.inline_max_output_tokens,
                )
            ]
        else:
            requests = load_manifest(manifest_path, selected_buckets, args.max_requests)
        requests = override_request_max_output_tokens(
            requests,
            max_output_tokens=args.override_max_output_tokens,
        )
        summary = run_smoke(args, config, requests)
        print("SMOKE_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    except BaseException as exc:
        failure = {
            "status": "failed",
            "mode": args.mode,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "manifest": str(manifest_path),
            "num_slots": int(args.num_slots),
        }
        (output_dir / "summary.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(f"SMOKE_FAILED {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
