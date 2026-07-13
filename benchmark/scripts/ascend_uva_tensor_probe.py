#!/usr/bin/env python3
"""Try wrapping an Ascend host-registered device pointer as a torch_npu Tensor.

This probes the framework-integration layer for an Ascend-UVA-like baseline.
It does not prove AI Core grouped-MLP readability. By default it only attempts
to construct storage/tensor metadata around the pointer. Use --try-op to force
small data-access operations, which may fail if the pointer is not accepted by
the NPU runtime as a normal tensor allocation.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import sys
import time
from pathlib import Path
from typing import Any

from ascend_uva_probe import (
    ACL_HOST_REGISTER_MAPPED,
    ACL_SUCCESS,
    _call_record,
    _load_acl,
    configure_signatures,
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    parser.add_argument("--dtype", choices=["uint8", "float16", "bfloat16"], default="uint8")
    parser.add_argument(
        "--try-copy",
        action="store_true",
        help="Deprecated alias for --try-op copy.",
    )
    parser.add_argument(
        "--try-op",
        choices=["none", "copy", "device_copy", "add"],
        default="none",
        help="Optional operation after wrapping the pointer.",
    )
    parser.add_argument("--op-elements", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--init", choices=["pattern", "zero"], default="pattern")
    parser.add_argument(
        "--verify-output",
        action="store_true",
        help="For add/device_copy, copy regular HBM output back to CPU for validation.",
    )
    parser.add_argument(
        "--standalone-acl-init",
        action="store_true",
        help="Call aclInit/aclFinalize explicitly. Off by default because torch_npu owns ACL lifecycle.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    size = args.size_mib * 1024 * 1024
    result: dict[str, Any] = {
        "probe": "ascend_uva_like_torch_npu_tensor_wrap",
        "timestamp_utc": _now(),
        "device_id": args.device_id,
        "size_mib": args.size_mib,
        "dtype": args.dtype,
        "try_op": "copy" if args.try_copy else args.try_op,
        "op_elements": args.op_elements,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "init": args.init,
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
        result["torch_npu"] = getattr(torch_npu, "__file__", None)

        if hasattr(torch, "npu") and hasattr(torch.npu, "set_device"):
            torch.npu.set_device(args.device_id)
            result["steps"]["torch_set_device"] = {"ok": True, "device": args.device_id}
        else:
            result["steps"]["torch_set_device"] = {"ok": False, "error": "torch.npu.set_device missing"}

        lib = _load_acl()
        configure_signatures(lib)
        if args.standalone_acl_init:
            result["steps"]["aclInit"] = _call_record(lib.aclInit, None)
        else:
            result["steps"]["aclInit"] = {"skipped": True, "reason": "torch_npu owns ACL lifecycle"}
        result["steps"]["aclrtSetDevice"] = _call_record(lib.aclrtSetDevice, args.device_id)
        if args.standalone_acl_init and not result["steps"]["aclInit"].get("ok"):
            result["status"] = "runtime_init_failed"
            return _write(result, args.out, exit_code=1)
        if not result["steps"]["aclrtSetDevice"].get("ok"):
            result["status"] = "runtime_init_failed"
            return _write(result, args.out, exit_code=1)

        page = mmap.mmap(-1, size)
        init_nbytes = min(size, max(4096, args.op_elements * 4))
        if args.init == "zero":
            page[:init_nbytes] = b"\x00" * init_nbytes
        else:
            page[:init_nbytes] = bytes([i % 251 for i in range(init_nbytes)])
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

        dtype = {
            "uint8": torch.uint8,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[args.dtype]
        element_size = torch.empty((), dtype=dtype).element_size()
        numel = size // element_size
        device = torch.device(f"npu:{args.device_id}")
        metadata = {
            "nbytes": size,
            "data_ptr": int(device_ptr.value),
            "size": (numel,),
            "stride": (1,),
            "dtype": dtype,
            "device": device,
            "storage_offset": 0,
        }

        storage = torch_npu._C._construct_storage_from_data_pointer(
            metadata["data_ptr"], metadata["device"], metadata["nbytes"]
        )
        result["steps"]["construct_storage"] = {
            "ok": True,
            "storage_data_ptr": hex(storage.data_ptr()),
            "storage_nbytes": storage.nbytes(),
            "check_npu_data_ptr": bool(torch_npu._C._check_npu_data_ptr(storage)),
        }

        tensor = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)
        result["steps"]["construct_tensor"] = {
            "ok": True,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "data_ptr": hex(tensor.data_ptr()),
            "storage_data_ptr": hex(tensor.untyped_storage().data_ptr()),
        }

        try_op = "copy" if args.try_copy else args.try_op
        op_elements = max(1, min(int(args.op_elements), int(numel)))
        if try_op == "copy":
            try:
                sample = tensor[: min(16, op_elements)].cpu()
                result["steps"]["try_op"] = {
                    "ok": True,
                    "op": try_op,
                    "values": sample.tolist(),
                }
            except Exception as exc:
                result["steps"]["try_op"] = {
                    "ok": False,
                    "op": try_op,
                    "error": repr(exc),
                }
        elif try_op == "device_copy":
            try:
                src = tensor[:op_elements]
                dst = torch.empty_like(src)
                dst.copy_(src)
                torch.npu.synchronize()
                op_result = {
                    "ok": True,
                    "op": try_op,
                    "elements": op_elements,
                    "dst_data_ptr": hex(dst.data_ptr()),
                }
                if args.verify_output:
                    op_result["output_sample"] = dst[: min(16, op_elements)].cpu().tolist()
                result["steps"]["try_op"] = op_result
            except Exception as exc:
                result["steps"]["try_op"] = {
                    "ok": False,
                    "op": try_op,
                    "error": repr(exc),
                }
        elif try_op == "add":
            try:
                src = tensor[:op_elements]
                if src.dtype == torch.uint8:
                    src = src.to(torch.float16)
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
                approx_read_bytes = op_elements * src.element_size() * max(1, args.repeat)
                op_result = {
                    "ok": True,
                    "op": try_op,
                    "elements": op_elements,
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "elapsed_s": elapsed_s,
                    "avg_ms": elapsed_s * 1000.0 / max(1, args.repeat),
                    "approx_source_read_gib_s": (
                        approx_read_bytes / (1024**3) / elapsed_s if elapsed_s > 0 else None
                    ),
                    "out_data_ptr": hex(out.data_ptr()),
                    "out_dtype": str(out.dtype),
                }
                if args.verify_output:
                    sample = out[: min(16, op_elements)].cpu()
                    op_result["output_sample"] = sample.tolist()
                    op_result["allclose_to_one"] = bool(
                        torch.allclose(sample, torch.ones_like(sample))
                    )
                result["steps"]["try_op"] = op_result
            except Exception as exc:
                result["steps"]["try_op"] = {
                    "ok": False,
                    "op": try_op,
                    "error": repr(exc),
                }

        result["status"] = "tensor_wrap_possible"
        return _write(result, args.out)
    except Exception as exc:
        result["status"] = "probe_exception"
        result["error"] = repr(exc)
        return _write(result, args.out, exit_code=2)
    finally:
        if lib is not None and host_ptr is not None and hasattr(lib, "aclrtHostUnregister"):
            result.setdefault("steps", {})["aclrtHostUnregister"] = _call_record(lib.aclrtHostUnregister, host_ptr)
        if args.standalone_acl_init and lib is not None and hasattr(lib, "aclrtResetDevice"):
            result.setdefault("steps", {})["aclrtResetDevice"] = _call_record(lib.aclrtResetDevice, args.device_id)
        if args.standalone_acl_init and lib is not None and hasattr(lib, "aclFinalize"):
            result.setdefault("steps", {})["aclFinalize"] = _call_record(lib.aclFinalize)
        if page is not None:
            try:
                page.close()
            except BufferError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
