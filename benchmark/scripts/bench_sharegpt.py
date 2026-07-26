#!/usr/bin/env python3
"""Benchmark TTFT, TPOT, and throughput on a running vLLM OpenAI server.

The defaults match the local ShareGPT smoke script, while CLI flags expose the
knobs needed for repeatable single-op vs SEW comparisons.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://localhost:8016"
DEFAULT_MODEL = "qwen3-30b-a3b"
DEFAULT_DATASET = "/data/shared_datasets/ShareGPT_V3_unfiltered_cleaned_split.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--n-prompts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--max-prompt-chars", type=int, default=800)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--tokenizer",
        default="",
        help=(
            "Optional local/HF tokenizer path. When set, output tokens are "
            "counted from generated text instead of streaming chunks."
        ),
    )
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    return parser.parse_args()


def load_prompts(path: str | Path, n: int, max_chars: int, seed: int) -> list[str]:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    prompts: list[str] = []
    for item in data:
        convs = item.get("conversations") or item.get("conversation") or []
        for turn in convs:
            role = turn.get("from", turn.get("role", ""))
            val = turn.get("value", turn.get("content", ""))
            if role in ("human", "user") and 10 < len(val) <= max_chars:
                prompts.append(val)
                break
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:n]


def _load_tokenizer(path: str):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "--tokenizer requires transformers to be installed"
        ) from exc
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _count_output_tokens(text: str, chunk_count: int, tokenizer: Any | None) -> int:
    if tokenizer is None:
        return int(chunk_count)
    if not text:
        return 0
    return int(len(tokenizer.encode(text, add_special_tokens=False)))


async def stream_request(
    session: Any,
    *,
    base_url: str,
    model: str,
    prompt: str,
    idx: int,
    max_tokens: int,
    tokenizer: Any | None,
) -> dict[str, Any]:
    """Send one streaming chat request and return per-request metrics."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
    }
    ttft = None
    chunk_count = 0
    text_parts: list[str] = []
    t0 = time.perf_counter()
    status = 0
    error = ""
    try:
        async with session.post(f"{base_url}/v1/chat/completions", json=payload) as resp:
            status = int(resp.status)
            if resp.status >= 400:
                error = await resp.text()
                return {
                    "idx": idx,
                    "status": status,
                    "error": error[:1000],
                    "ttft_s": None,
                    "output_tokens": 0,
                    "total_s": time.perf_counter() - t0,
                    "chunks": 0,
                }
            async for raw in resp.content:
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                delta = obj["choices"][0]["delta"].get("content", "")
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunk_count += 1
                    text_parts.append(delta)
    except Exception as exc:  # network failures should be visible in the JSON.
        error = f"{type(exc).__name__}: {exc}"

    total = time.perf_counter() - t0
    text = "".join(text_parts)
    out_tokens = _count_output_tokens(text, chunk_count, tokenizer)
    return {
        "idx": idx,
        "status": status,
        "error": error,
        "ttft_s": ttft,
        "output_tokens": out_tokens,
        "total_s": total,
        "chunks": chunk_count,
        "output_chars": len(text),
    }


def percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    ordered = sorted(data)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return float(ordered[lo])
    weight = k - lo
    return float(ordered[lo] * (1 - weight) + ordered[hi] * weight)


def summarize(
    results: list[dict[str, Any]],
    *,
    wall_s: float,
    args: argparse.Namespace,
    tokenizer_used: bool,
) -> dict[str, Any]:
    ttfts: list[float] = []
    tpots: list[float] = []
    e2els: list[float] = []
    total_out = 0
    success = 0
    for result in results:
        ttft = result.get("ttft_s")
        out_tok = int(result.get("output_tokens") or 0)
        e2el = float(result.get("total_s") or 0.0)
        if ttft is None or out_tok <= 0:
            continue
        success += 1
        ttfts.append(float(ttft) * 1000.0)
        tpots.append((e2el - float(ttft)) / max(1, out_tok) * 1000.0)
        e2els.append(e2el)
        total_out += out_tok

    def stat(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
        return {
            "mean": float(statistics.fmean(values)),
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p99": percentile(values, 99),
        }

    ttft = stat(ttfts)
    tpot = stat(tpots)
    e2el_mean = float(statistics.fmean(e2els)) if e2els else 0.0
    throughput = total_out / wall_s if wall_s else 0.0
    payload = {
        "status": "ok" if success == len(results) else "partial",
        "dashboard_label": args.label or args.model,
        "model": args.model,
        "base_url": args.base_url,
        "dataset": args.dataset,
        "n_prompts": int(args.n_prompts),
        "successful_requests": int(success),
        "failed_requests": int(len(results) - success),
        "concurrency": int(args.concurrency),
        "max_tokens": int(args.max_tokens),
        "max_prompt_chars": int(args.max_prompt_chars),
        "seed": int(args.seed),
        "tokenizer": args.tokenizer or None,
        "tokenizer_used": bool(tokenizer_used),
        "wall_s": float(wall_s),
        "total_output_tokens": int(total_out),
        "output_throughput": float(throughput),
        "output_throughput_tok_s": float(throughput),
        "request_throughput": success / wall_s if wall_s else 0.0,
        "median_ttft_ms": float(ttft["p50"]),
        "median_tpot_ms": float(tpot["p50"]),
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "e2el_s": {"mean": e2el_mean},
        "per_request": results,
    }
    return payload


async def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import aiohttp
    except Exception as exc:
        raise RuntimeError(
            "bench_sharegpt.py requires aiohttp in the benchmark Python "
            "environment."
        ) from exc

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    prompts = load_prompts(args.dataset, args.n_prompts, args.max_prompt_chars, args.seed)
    tokenizer = _load_tokenizer(args.tokenizer)
    print(
        f"Loaded {len(prompts)} prompts. Concurrency={args.concurrency}. "
        f"Tokenizer={'yes' if tokenizer is not None else 'no'}. Starting benchmark ...\n"
    )

    results: list[dict[str, Any] | None] = [None] * len(prompts)
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=float(args.request_timeout_s))

    async def bounded(session: Any, prompt: str, idx: int):
        async with sem:
            return await stream_request(
                session,
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                idx=idx,
                max_tokens=args.max_tokens,
                tokenizer=tokenizer,
            )

    wall_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        tasks = [bounded(session, p, i) for i, p in enumerate(prompts)]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            idx = int(result["idx"])
            results[idx] = result
            ttft = result.get("ttft_s")
            out_tok = int(result.get("output_tokens") or 0)
            e2el = float(result.get("total_s") or 0.0)
            if ttft is not None and out_tok > 0:
                tpot = (e2el - float(ttft)) / max(1, out_tok) * 1000.0
                print(
                    f"  [{idx + 1:3d}] TTFT={float(ttft) * 1000:6.1f}ms  "
                    f"TPOT={tpot:6.2f}ms/tok  out={out_tok}tok"
                )
            else:
                print(
                    f"  [{idx + 1:3d}] failed status={result.get('status')} "
                    f"error={str(result.get('error', ''))[:120]}"
                )
            sys.stdout.flush()

    wall = time.perf_counter() - wall_start
    finalized = [item for item in results if item is not None]
    return summarize(finalized, wall_s=wall, args=args, tokenizer_used=tokenizer is not None)


def print_summary(summary: dict[str, Any]) -> None:
    ttft = summary["ttft_ms"]
    tpot = summary["tpot_ms"]
    e2el = summary["e2el_s"]
    print(f"""
{'=' * 55}
Results ({summary['successful_requests']} successful requests, concurrency={summary['concurrency']})
{'=' * 55}
TTFT  (ms)   mean={ttft['mean']:.1f}  p50={ttft['p50']:.1f}  p90={ttft['p90']:.1f}  p99={ttft['p99']:.1f}
TPOT  (ms)   mean={tpot['mean']:.2f}  p50={tpot['p50']:.2f}  p90={tpot['p90']:.2f}  p99={tpot['p99']:.2f}
E2EL  (s)    mean={e2el['mean']:.2f}
Throughput   {summary['output_throughput']:.1f} output-tokens/s  (wall {summary['wall_s']:.1f}s, {summary['total_output_tokens']} tok total)
{'=' * 55}
""")


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run(args))
    print_summary(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
