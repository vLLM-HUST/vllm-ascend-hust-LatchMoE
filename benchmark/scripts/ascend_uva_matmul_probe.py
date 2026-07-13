#!/usr/bin/env python3
"""Probe MoE-shaped matmul over an Ascend host-registered weight pointer.

This U3 microbenchmark compares a host-registered weight matrix against an HBM
resident weight matrix with the same shape. It approximates the grouped-MLP
expert-weight read question without integrating into vLLM.
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


def _fill_half(page: mmap.mmap, elements: int, value: float) -> None:
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
    parser.add_argument("--m", type=int, default=16, help="Activation rows / token count.")
    parser.add_argument("--k", type=int, default=4096, help="Hidden dimension.")
    parser.add_argument("--n", type=int, default=4096, help="Expert projection output dimension.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--verify-output", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    weight_elements = args.k * args.n
    weight_nbytes = weight_elements * 2
    size_mib = (weight_nbytes + (1024 * 1024 - 1)) // (1024 * 1024)
    size = size_mib * 1024 * 1024
    result: dict[str, Any] = {
        "probe": "ascend_uva_like_matmul_weight_read",
        "timestamp_utc": _now(),
        "device_id": args.device_id,
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "weight_mib": weight_nbytes / (1024 * 1024),
        "alloc_size_mib": size_mib,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "verify_output": args.verify_output,
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

        lib = _load_acl()
        configure_signatures(lib)
        result["steps"]["aclrtSetDevice"] = _call_record(lib.aclrtSetDevice, args.device_id)
        if not result["steps"]["aclrtSetDevice"].get("ok"):
            result["status"] = "runtime_init_failed"
            return _write(result, args.out, exit_code=1)

        page = mmap.mmap(-1, size)
        _fill_half(page, weight_elements, 1.0)
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
            "size": (size // 2,),
            "stride": (1,),
            "dtype": torch.float16,
            "device": device,
            "storage_offset": 0,
        }
        storage = torch_npu._C._construct_storage_from_data_pointer(
            metadata["data_ptr"], metadata["device"], metadata["nbytes"]
        )
        tensor = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)
        weight_host = tensor[:weight_elements].reshape(args.k, args.n)
        result["steps"]["construct_weight_tensor"] = {
            "ok": True,
            "shape": list(weight_host.shape),
            "dtype": str(weight_host.dtype),
            "device": str(weight_host.device),
            "data_ptr": hex(weight_host.data_ptr()),
            "check_npu_data_ptr": bool(torch_npu._C._check_npu_data_ptr(storage)),
        }

        activation = torch.ones((args.m, args.k), device=f"npu:{args.device_id}", dtype=torch.float16)

        def run_matmul(weight: Any) -> tuple[Any, float]:
            out = None
            for _ in range(max(0, args.warmup)):
                out = activation @ weight
            torch.npu.synchronize()
            started = time.perf_counter()
            for _ in range(max(1, args.repeat)):
                out = activation @ weight
            torch.npu.synchronize()
            return out, time.perf_counter() - started

        weight_hbm = torch.ones((args.k, args.n), device=f"npu:{args.device_id}", dtype=torch.float16)
        try:
            out_hbm, elapsed_hbm = run_matmul(weight_hbm)
            hbm_record: dict[str, Any] = {
                "ok": True,
                "elapsed_s": elapsed_hbm,
                "avg_ms": elapsed_hbm * 1000.0 / max(1, args.repeat),
                "approx_weight_read_gib_s": (
                    weight_nbytes * max(1, args.repeat) / (1024**3) / elapsed_hbm
                    if elapsed_hbm > 0
                    else None
                ),
                "out_data_ptr": hex(out_hbm.data_ptr()),
            }
            if args.verify_output:
                sample = out_hbm[: min(2, args.m), : min(8, args.n)].cpu()
                expected = torch.full_like(sample, float(args.k))
                hbm_record["output_sample"] = sample.tolist()
                hbm_record["allclose_to_k"] = bool(torch.allclose(sample, expected, rtol=0, atol=1))
        except Exception as exc:
            hbm_record = {"ok": False, "error": repr(exc)}
        result["steps"]["hbm_matmul"] = hbm_record

        try:
            out_host, elapsed_host = run_matmul(weight_host)
            host_record: dict[str, Any] = {
                "ok": True,
                "elapsed_s": elapsed_host,
                "avg_ms": elapsed_host * 1000.0 / max(1, args.repeat),
                "approx_weight_read_gib_s": (
                    weight_nbytes * max(1, args.repeat) / (1024**3) / elapsed_host
                    if elapsed_host > 0
                    else None
                ),
                "out_data_ptr": hex(out_host.data_ptr()),
            }
            if args.verify_output:
                sample = out_host[: min(2, args.m), : min(8, args.n)].cpu()
                expected = torch.full_like(sample, float(args.k))
                host_record["output_sample"] = sample.tolist()
                host_record["allclose_to_k"] = bool(torch.allclose(sample, expected, rtol=0, atol=1))
        except Exception as exc:
            host_record = {"ok": False, "error": repr(exc)}
        result["steps"]["host_registered_matmul"] = host_record

        if not hbm_record.get("ok"):
            result["status"] = "hbm_reference_failed"
            return _write(result, args.out, exit_code=1)
        if not host_record.get("ok"):
            result["status"] = "host_registered_matmul_failed"
            return _write(result, args.out, exit_code=1)

        host_bw = host_record.get("approx_weight_read_gib_s")
        hbm_bw = hbm_record.get("approx_weight_read_gib_s")
        result["relative_to_hbm"] = host_bw / hbm_bw if host_bw and hbm_bw else None
        result["status"] = "matmul_probe_ok"
        return _write(result, args.out)
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
