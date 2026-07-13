#!/usr/bin/env python3
"""Probe torch_npu NPUGraph replay over an Ascend host-registered pointer.

The key question is whether a captured graph that reads a host-registered
device-visible pointer observes host-side content updates at replay time. This
is a minimal U2 probe for an Ascend-UVA-like expert access path; it is not a
vLLM grouped-MLP integration test.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import struct
import sys
import time
from pathlib import Path
from typing import Any

from ascend_uva_probe import (
    ACL_HOST_REGISTER_MAPPED,
    _call_record,
    _load_acl,
    configure_signatures,
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_half_values(page: mmap.mmap, elements: int, value: float) -> None:
    pattern = struct.pack("<e", value)
    chunk_elems = min(elements, 1 << 20)
    chunk = pattern * chunk_elems
    offset = 0
    remaining = elements
    while remaining > 0:
        n = min(chunk_elems, remaining)
        nbytes = n * 2
        page[offset : offset + nbytes] = chunk[:nbytes]
        offset += nbytes
        remaining -= n


def _write(result: dict[str, Any], out: Path | None, exit_code: int = 0) -> int:
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=4)
    parser.add_argument("--size-mib", type=int, default=1)
    parser.add_argument("--elements", type=int, default=1024)
    parser.add_argument("--replay-values", default="2,4", help="Comma-separated host values written before graph replay.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    size = args.size_mib * 1024 * 1024
    result: dict[str, Any] = {
        "probe": "ascend_uva_like_npugraph_replay",
        "timestamp_utc": _now(),
        "device_id": args.device_id,
        "size_mib": args.size_mib,
        "elements": args.elements,
        "replay_values": args.replay_values,
        "python": sys.executable,
        "status": "unknown",
        "steps": {},
    }

    page: mmap.mmap | None = None
    host_ptr: ctypes.c_void_p | None = None
    lib: ctypes.CDLL | None = None
    try:
        import torch
        import torch_npu

        result["torch"] = getattr(torch, "__version__", None)
        result["torch_npu"] = getattr(torch_npu, "__version__", None)
        torch.npu.set_device(args.device_id)
        result["steps"]["torch_set_device"] = {"ok": True, "device": args.device_id}

        element_size = 2
        needed = args.elements * element_size
        if needed > size:
            result["status"] = "invalid_args"
            result["error"] = f"elements require {needed} bytes but size is {size}"
            return _write(result, args.out, exit_code=2)

        lib = _load_acl()
        configure_signatures(lib)
        result["steps"]["aclrtSetDevice"] = _call_record(lib.aclrtSetDevice, args.device_id)
        if not result["steps"]["aclrtSetDevice"].get("ok"):
            result["status"] = "runtime_init_failed"
            return _write(result, args.out, exit_code=1)

        page = mmap.mmap(-1, size)
        _write_half_values(page, args.elements, 0.0)
        host_ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(page)))
        device_ptr = ctypes.c_void_p()
        register = _call_record(
            lib.aclrtHostRegister,
            host_ptr,
            ctypes.c_uint64(size),
            ACL_HOST_REGISTER_MAPPED,
            ctypes.byref(device_ptr),
        )
        register["host_ptr"] = hex(host_ptr.value) if host_ptr.value else None
        register["device_ptr"] = hex(device_ptr.value) if device_ptr.value else None
        result["steps"]["aclrtHostRegister"] = register
        if not register.get("ok"):
            result["status"] = "host_register_failed"
            return _write(result, args.out, exit_code=1)

        device = torch.device(f"npu:{args.device_id}")
        metadata = {
            "nbytes": size,
            "data_ptr": int(device_ptr.value),
            "size": (size // element_size,),
            "stride": (1,),
            "dtype": torch.float16,
            "device": device,
            "storage_offset": 0,
        }
        storage = torch_npu._C._construct_storage_from_data_pointer(
            metadata["data_ptr"], metadata["device"], metadata["nbytes"]
        )
        tensor = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)
        result["steps"]["construct_tensor"] = {
            "ok": True,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "data_ptr": hex(tensor.data_ptr()),
            "check_npu_data_ptr": bool(torch_npu._C._check_npu_data_ptr(storage)),
        }

        src = tensor[: args.elements]
        ones = torch.ones_like(src)
        out = torch.empty_like(src)

        # Eager warmup establishes that the compute path can read the pointer.
        out.copy_(src + ones)
        torch.npu.synchronize()
        eager_sample = out[: min(16, args.elements)].cpu()
        result["steps"]["eager_before_capture"] = {
            "ok": bool(torch.allclose(eager_sample, torch.ones_like(eager_sample))),
            "output_sample": eager_sample.tolist(),
        }

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out.copy_(src + ones)
        torch.npu.synchronize()
        capture_sample = out[: min(16, args.elements)].cpu()
        result["steps"]["capture"] = {
            "ok": bool(torch.allclose(capture_sample, torch.ones_like(capture_sample))),
            "output_sample": capture_sample.tolist(),
        }

        replay_results: list[dict[str, Any]] = []
        for raw_value in [v.strip() for v in args.replay_values.split(",") if v.strip()]:
            value = float(raw_value)
            torch.npu.synchronize()
            _write_half_values(page, args.elements, value)
            started = time.perf_counter()
            graph.replay()
            torch.npu.synchronize()
            elapsed_s = time.perf_counter() - started
            sample = out[: min(16, args.elements)].cpu()
            expected = torch.full_like(sample, value + 1.0)
            replay_results.append(
                {
                    "host_value": value,
                    "expected": value + 1.0,
                    "ok": bool(torch.allclose(sample, expected)),
                    "output_sample": sample.tolist(),
                    "elapsed_ms": elapsed_s * 1000.0,
                }
            )
        result["steps"]["replay"] = replay_results
        result["status"] = "graph_replay_observes_host_updates" if all(r["ok"] for r in replay_results) else "graph_replay_failed_or_stale"
        return _write(result, args.out, exit_code=0 if all(r["ok"] for r in replay_results) else 1)
    except Exception as exc:
        result["status"] = "probe_exception"
        result["error"] = repr(exc)
        return _write(result, args.out, exit_code=2)
    finally:
        if lib is not None and host_ptr is not None and hasattr(lib, "aclrtHostUnregister"):
            result.setdefault("steps", {})["aclrtHostUnregister"] = _call_record(lib.aclrtHostUnregister, host_ptr)
        if page is not None:
            try:
                page.close()
            except BufferError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
