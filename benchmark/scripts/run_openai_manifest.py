#!/usr/bin/env python3
"""Run a JSONL workload manifest against an OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_requests(path: str | Path, *, bucket: str, max_requests: int) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("bucket") != bucket:
                continue
            requests.append(record)
            if max_requests and len(requests) >= max_requests:
                break
    if not requests:
        raise ValueError(f"no requests for bucket {bucket!r} in manifest {path}")
    return requests


def _load_tokenizer(path: str):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("--tokenizer requires transformers") from exc
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
    request_record: dict[str, Any],
    tokenizer: Any | None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": str(request_record["prompt"])}],
        "max_tokens": int(request_record["max_output_tokens"]),
        "stream": True,
        "temperature": float(request_record.get("temperature", 0.0)),
        "top_p": float(request_record.get("top_p", 1.0)),
    }
    request_id = str(request_record.get("request_id", ""))
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
                    "request_id": request_id,
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
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    total = time.perf_counter() - t0
    text = "".join(text_parts)
    out_tokens = _count_output_tokens(text, chunk_count, tokenizer)
    return {
        "request_id": request_id,
        "status": status,
        "error": error,
        "ttft_s": ttft,
        "output_tokens": out_tokens,
        "total_s": total,
        "chunks": chunk_count,
        "prompt_tokens": request_record.get("prompt_tokens"),
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
    request_count: int,
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
    return {
        "status": "ok" if success == request_count else "partial",
        "model": args.model,
        "base_url": args.base_url,
        "manifest": args.manifest,
        "bucket": args.bucket,
        "n_prompts": int(request_count),
        "successful_requests": int(success),
        "failed_requests": int(request_count - success),
        "concurrency": int(args.concurrency),
        "tokenizer": args.tokenizer or None,
        "tokenizer_used": bool(tokenizer_used),
        "wall_s": float(wall_s),
        "total_output_tokens": int(total_out),
        "output_throughput": total_out / wall_s if wall_s else 0.0,
        "output_throughput_tok_s": total_out / wall_s if wall_s else 0.0,
        "request_throughput": success / wall_s if wall_s else 0.0,
        "median_ttft_ms": float(ttft["p50"]),
        "median_tpot_ms": float(tpot["p50"]),
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "e2el_s": {"mean": float(statistics.fmean(e2els)) if e2els else 0.0},
        "per_request": results,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import aiohttp
    except Exception as exc:
        raise RuntimeError("run_openai_manifest.py requires aiohttp") from exc
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")

    requests = load_requests(args.manifest, bucket=args.bucket, max_requests=args.max_requests)
    tokenizer = _load_tokenizer(args.tokenizer)
    print(
        f"Loaded {len(requests)} requests for bucket={args.bucket}. "
        f"Concurrency={args.concurrency}. Tokenizer={'yes' if tokenizer else 'no'}."
    )
    results: list[dict[str, Any] | None] = [None] * len(requests)
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=float(args.request_timeout_s))

    async def bounded(session: Any, request_record: dict[str, Any], idx: int):
        async with sem:
            result = await stream_request(
                session,
                base_url=args.base_url,
                model=args.model,
                request_record=request_record,
                tokenizer=tokenizer,
            )
            result["idx"] = idx
            return result

    wall_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        tasks = [bounded(session, request, idx) for idx, request in enumerate(requests)]
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
                    f"  [{idx + 1:3d}] TTFT={float(ttft) * 1000:6.1f}ms "
                    f"TPOT={tpot:6.2f}ms/tok out={out_tok}tok"
                )
            else:
                print(
                    f"  [{idx + 1:3d}] failed status={result.get('status')} "
                    f"error={str(result.get('error', ''))[:120]}"
                )
            sys.stdout.flush()

    finalized = [item for item in results if item is not None]
    wall_s = time.perf_counter() - wall_start
    return summarize(
        finalized,
        wall_s=wall_s,
        args=args,
        request_count=len(requests),
        tokenizer_used=tokenizer is not None,
    )


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run(args))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
