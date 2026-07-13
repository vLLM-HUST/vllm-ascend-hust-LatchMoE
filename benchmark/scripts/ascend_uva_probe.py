#!/usr/bin/env python3
"""Probe Ascend UVA-like host-memory mechanisms.

This script intentionally stays below vLLM. It answers the first feasibility
question for an Ascend-UVA-like expert offload baseline:

1. Does the current CANN runtime report host-memory mapping support for the
   target device?
2. Can a host allocation be registered and mapped to a device pointer?
3. Can managed memory be allocated by the runtime?

The script does not prove that AI Core grouped-MLP kernels can consume the
mapped pointer. That requires the follow-up microbenchmark described in the
experiment protocol.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ACL_SUCCESS = 0
ACL_HOST_REGISTER_MAPPED = 0

HAC_TYPES = {
    0: "STARS",
    1: "AICPU",
    2: "AIC",
    3: "AIV",
    4: "PCIEDMA",
    5: "RDMA",
    6: "SDMA",
    7: "DVPP",
    8: "UDMA",
    9: "CCU",
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_acl() -> ctypes.CDLL:
    candidates = [
        "libascendcl.so",
        "/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so",
        "/usr/local/Ascend/cann-9.0.0/aarch64-linux/devlib/libascendcl.so",
        "/usr/local/Ascend/cann-9.0.0/aarch64-linux/devlib/linux/aarch64/libascendcl.so",
    ]
    errors: list[str] = []
    for lib in candidates:
        try:
            return ctypes.CDLL(lib)
        except OSError as exc:
            errors.append(f"{lib}: {exc}")
    raise RuntimeError("failed to load libascendcl.so: " + " | ".join(errors))


def _set_sig(lib: ctypes.CDLL, name: str, argtypes: list[Any], restype: Any = ctypes.c_int) -> bool:
    try:
        fn = getattr(lib, name)
    except AttributeError:
        return False
    fn.argtypes = argtypes
    fn.restype = restype
    return True


def _call_record(fn: Any, *args: Any) -> dict[str, Any]:
    try:
        rc = int(fn(*args))
        return {"available": True, "rc": rc, "ok": rc == ACL_SUCCESS}
    except AttributeError:
        return {"available": False, "error": "missing_symbol"}
    except Exception as exc:  # pragma: no cover - diagnostic path.
        return {"available": True, "error": repr(exc), "ok": False}


def _npu_smi() -> dict[str, Any]:
    exe = shutil.which("npu-smi")
    if not exe:
        return {"available": False}
    try:
        out = subprocess.run([exe, "info"], check=False, text=True, capture_output=True, timeout=10)
        return {
            "available": True,
            "returncode": out.returncode,
            "stdout_head": out.stdout.splitlines()[:40],
            "stderr": out.stderr,
        }
    except Exception as exc:  # pragma: no cover - diagnostic path.
        return {"available": True, "error": repr(exc)}


def _probe_capabilities(lib: ctypes.CDLL, device_id: int) -> dict[str, Any]:
    if not hasattr(lib, "aclrtHostMemMapCapabilities"):
        return {"available": False, "error": "missing_symbol"}
    out: dict[str, Any] = {}
    for hac_type, name in HAC_TYPES.items():
        cap = ctypes.c_int(-1)
        rc = int(lib.aclrtHostMemMapCapabilities(device_id, hac_type, ctypes.byref(cap)))
        out[name] = {
            "hac_type": hac_type,
            "rc": rc,
            "ok": rc == ACL_SUCCESS,
            "capability": cap.value,
            "supported": rc == ACL_SUCCESS and cap.value == 1,
        }
    return {"available": True, "by_hac_type": out}


def _probe_malloc_host(lib: ctypes.CDLL, size: int) -> dict[str, Any]:
    if not hasattr(lib, "aclrtMallocHost"):
        return {"available": False, "error": "missing_symbol"}
    host_ptr = ctypes.c_void_p()
    record = _call_record(lib.aclrtMallocHost, ctypes.byref(host_ptr), size)
    record["host_ptr"] = hex(host_ptr.value) if host_ptr.value else None
    if record.get("ok") and hasattr(lib, "aclrtFreeHost"):
        record["free"] = _call_record(lib.aclrtFreeHost, host_ptr)
    return record


def _probe_host_register_legacy(lib: ctypes.CDLL, size: int) -> dict[str, Any]:
    if not hasattr(lib, "aclrtHostRegister"):
        return {"available": False, "error": "missing_symbol"}
    page = mmap.mmap(-1, size)
    host_ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(page)))
    device_ptr = ctypes.c_void_p()
    record = _call_record(
        lib.aclrtHostRegister,
        host_ptr,
        ctypes.c_uint64(size),
        ACL_HOST_REGISTER_MAPPED,
        ctypes.byref(device_ptr),
    )
    record["host_ptr"] = hex(host_ptr.value) if host_ptr.value else None
    record["device_ptr"] = hex(device_ptr.value) if device_ptr.value else None
    if record.get("ok") and hasattr(lib, "aclrtHostUnregister"):
        record["unregister"] = _call_record(lib.aclrtHostUnregister, host_ptr)
    page.close()
    return record


def _probe_host_register_v2(lib: ctypes.CDLL, size: int) -> dict[str, Any]:
    if not hasattr(lib, "aclrtHostRegisterV2"):
        return {"available": False, "error": "missing_symbol"}
    page = mmap.mmap(-1, size)
    host_ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(page)))
    record = _call_record(lib.aclrtHostRegisterV2, host_ptr, ctypes.c_uint64(size), 0)
    record["host_ptr"] = hex(host_ptr.value) if host_ptr.value else None
    if record.get("ok") and hasattr(lib, "aclrtHostGetDevicePointer"):
        device_ptr = ctypes.c_void_p()
        get_record = _call_record(lib.aclrtHostGetDevicePointer, host_ptr, ctypes.byref(device_ptr), 0)
        get_record["device_ptr"] = hex(device_ptr.value) if device_ptr.value else None
        record["get_device_pointer"] = get_record
    if record.get("ok") and hasattr(lib, "aclrtHostUnregister"):
        record["unregister"] = _call_record(lib.aclrtHostUnregister, host_ptr)
    page.close()
    return record


def _probe_managed_alloc(lib: ctypes.CDLL, size: int) -> dict[str, Any]:
    if not hasattr(lib, "aclrtMemAllocManaged"):
        return {"available": False, "error": "missing_symbol"}
    ptr = ctypes.c_void_p()
    record = _call_record(lib.aclrtMemAllocManaged, ctypes.byref(ptr), ctypes.c_uint64(size), 0)
    record["ptr"] = hex(ptr.value) if ptr.value else None
    if record.get("ok") and hasattr(lib, "aclrtFree"):
        record["free"] = _call_record(lib.aclrtFree, ptr)
    return record


def configure_signatures(lib: ctypes.CDLL) -> None:
    _set_sig(lib, "aclInit", [ctypes.c_char_p])
    _set_sig(lib, "aclFinalize", [])
    _set_sig(lib, "aclrtSetDevice", [ctypes.c_int])
    _set_sig(lib, "aclrtResetDevice", [ctypes.c_int])
    _set_sig(lib, "aclrtHostMemMapCapabilities", [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)])
    _set_sig(lib, "aclrtMallocHost", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t])
    _set_sig(lib, "aclrtFreeHost", [ctypes.c_void_p])
    _set_sig(lib, "aclrtHostRegister", [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)])
    _set_sig(lib, "aclrtHostRegisterV2", [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32])
    _set_sig(lib, "aclrtHostGetDevicePointer", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32])
    _set_sig(lib, "aclrtHostUnregister", [ctypes.c_void_p])
    _set_sig(lib, "aclrtMemAllocManaged", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64, ctypes.c_uint32])
    _set_sig(lib, "aclrtFree", [ctypes.c_void_p])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=4, help="Physical Ascend device id to probe.")
    parser.add_argument("--size-mib", type=int, default=64, help="Allocation/register size for API probes.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--skip-malloc-host", action="store_true", help="Skip aclrtMallocHost/freeHost probe.")
    parser.add_argument("--skip-register", action="store_true", help="Skip host-register probes.")
    parser.add_argument("--skip-managed", action="store_true", help="Skip aclrtMemAllocManaged probe.")
    args = parser.parse_args()

    size = args.size_mib * 1024 * 1024
    result: dict[str, Any] = {
        "probe": "ascend_uva_like_expert_access_feasibility",
        "timestamp_utc": _now(),
        "device_id": args.device_id,
        "size_mib": args.size_mib,
        "python": sys.executable,
        "platform": platform.platform(),
        "env": {
            "ASCEND_HOME_PATH": os.environ.get("ASCEND_HOME_PATH"),
            "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
        "npu_smi": _npu_smi(),
        "status": "unknown",
        "steps": {},
    }

    try:
        lib = _load_acl()
        configure_signatures(lib)
        result["steps"]["aclInit"] = _call_record(lib.aclInit, None)
        if not result["steps"]["aclInit"].get("ok"):
            result["status"] = "acl_init_failed"
            return write_result(result, args.out)

        result["steps"]["aclrtSetDevice"] = _call_record(lib.aclrtSetDevice, args.device_id)
        if not result["steps"]["aclrtSetDevice"].get("ok"):
            result["status"] = "set_device_failed"
            return write_result(result, args.out)

        result["steps"]["host_mem_map_capabilities"] = _probe_capabilities(lib, args.device_id)
        result["steps"]["malloc_host"] = (
            {"skipped": True} if args.skip_malloc_host else _probe_malloc_host(lib, size)
        )
        if args.skip_register:
            result["steps"]["host_register_legacy"] = {"skipped": True}
            result["steps"]["host_register_v2"] = {"skipped": True}
        else:
            result["steps"]["host_register_legacy"] = _probe_host_register_legacy(lib, size)
            result["steps"]["host_register_v2"] = _probe_host_register_v2(lib, size)
        result["steps"]["managed_alloc"] = (
            {"skipped": True} if args.skip_managed else _probe_managed_alloc(lib, size)
        )

        supported = any(
            item.get("supported")
            for item in result["steps"]["host_mem_map_capabilities"].get("by_hac_type", {}).values()
        )
        register_ok = bool(result["steps"]["host_register_legacy"].get("ok") or result["steps"]["host_register_v2"].get("ok"))
        managed_ok = bool(result["steps"]["managed_alloc"].get("ok"))
        result["status"] = "runtime_mapping_possible" if (supported or register_ok or managed_ok) else "runtime_mapping_not_available"
        return finish_and_write(result, args.out, lib, args.device_id)
    except Exception as exc:
        result["status"] = "probe_exception"
        result["error"] = repr(exc)
        return finish_and_write(result, args.out, locals().get("lib"), args.device_id, exit_code=2)


def write_result(result: dict[str, Any], out: Path | None, exit_code: int = 0) -> int:
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return exit_code


def finish_and_write(
    result: dict[str, Any],
    out: Path | None,
    lib: ctypes.CDLL | None,
    device_id: int,
    exit_code: int = 0,
) -> int:
    if lib is not None:
        cleanup: dict[str, Any] = {}
        if hasattr(lib, "aclrtResetDevice"):
            cleanup["aclrtResetDevice"] = _call_record(lib.aclrtResetDevice, device_id)
        if hasattr(lib, "aclFinalize"):
            cleanup["aclFinalize"] = _call_record(lib.aclFinalize)
        result["steps"]["cleanup"] = cleanup
    return write_result(result, out, exit_code=exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
