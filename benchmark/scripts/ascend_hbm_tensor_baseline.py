#!/usr/bin/env python3
"""HBM-resident tensor baseline for Ascend-UVA-like add bandwidth probes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=4)
    parser.add_argument("--elements", type=int, required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--verify-output", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device_id)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    src = torch.zeros(args.elements, device=f"npu:{args.device_id}", dtype=dtype)
    ones = torch.ones_like(src)
    for _ in range(max(0, args.warmup)):
        _ = src + ones
    torch.npu.synchronize()
    started = time.perf_counter()
    out = None
    for _ in range(max(1, args.repeat)):
        out = src + ones
    torch.npu.synchronize()
    elapsed_s = time.perf_counter() - started
    assert out is not None
    approx_read_bytes = args.elements * src.element_size() * max(1, args.repeat)

    result = {
        "probe": "ascend_hbm_tensor_add_baseline",
        "timestamp_utc": _now(),
        "python": sys.executable,
        "torch": getattr(torch, "__version__", None),
        "device_id": args.device_id,
        "elements": args.elements,
        "dtype": str(dtype),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "elapsed_s": elapsed_s,
        "avg_ms": elapsed_s * 1000.0 / max(1, args.repeat),
        "approx_source_read_gib_s": approx_read_bytes / (1024**3) / elapsed_s if elapsed_s > 0 else None,
        "src_data_ptr": hex(src.data_ptr()),
        "out_data_ptr": hex(out.data_ptr()),
        "status": "ok",
    }
    if args.verify_output:
        sample = out[:16].cpu()
        result["output_sample"] = sample.tolist()
        result["allclose_to_one"] = bool(torch.allclose(sample, torch.ones_like(sample)))

    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
