#!/usr/bin/env python3
"""Run Ascend-UVA-like tensor access probes in isolated subprocesses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python executable inside the torch_npu environment.")
    parser.add_argument("--device-id", type=int, default=4)
    parser.add_argument("--size-mib", type=int, default=1)
    parser.add_argument("--op-elements", type=int, default=1024)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark/artifacts/reports/ascend_uva_feasibility"))
    parser.add_argument("--timeout-s", type=int, default=180)
    args = parser.parse_args()

    script = Path(__file__).with_name("ascend_uva_tensor_probe.py")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        {"name": "construct_uint8", "dtype": "uint8", "op": "none", "init": "pattern"},
        {"name": "copy_uint8", "dtype": "uint8", "op": "copy", "init": "pattern"},
        {"name": "device_copy_uint8", "dtype": "uint8", "op": "device_copy", "init": "pattern"},
        {"name": "add_float16_zero", "dtype": "float16", "op": "add", "init": "zero"},
    ]
    summary: dict[str, Any] = {
        "probe": "ascend_uva_like_tensor_access_matrix",
        "timestamp_utc": _now(),
        "device_id": args.device_id,
        "size_mib": args.size_mib,
        "op_elements": args.op_elements,
        "python": args.python,
        "jobs": [],
    }

    for job in jobs:
        child_out = args.out_dir / f"probe_device{args.device_id}_tensor_{job['name']}.json"
        cmd = [
            args.python,
            str(script),
            "--device-id",
            str(args.device_id),
            "--size-mib",
            str(args.size_mib),
            "--dtype",
            job["dtype"],
            "--try-op",
            job["op"],
            "--op-elements",
            str(args.op_elements),
            "--init",
            job["init"],
            "--out",
            str(child_out),
        ]
        started = time.monotonic()
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=args.timeout_s)
            elapsed_s = time.monotonic() - started
            record = {
                "name": job["name"],
                "cmd": cmd,
                "returncode": proc.returncode,
                "elapsed_s": elapsed_s,
                "stdout_tail": proc.stdout.splitlines()[-80:],
                "stderr_tail": proc.stderr.splitlines()[-80:],
                "child_json": str(child_out) if child_out.exists() else None,
            }
            if proc.returncode == 0:
                record["status"] = "completed"
            elif proc.returncode < 0:
                record["status"] = f"signal_{-proc.returncode}"
            elif proc.returncode == 139:
                record["status"] = "segmentation_fault"
            else:
                record["status"] = "failed"
        except subprocess.TimeoutExpired as exc:
            elapsed_s = time.monotonic() - started
            record = {
                "name": job["name"],
                "cmd": cmd,
                "status": "timeout",
                "elapsed_s": elapsed_s,
                "timeout_s": args.timeout_s,
                "stdout_tail": (exc.stdout or "").splitlines()[-80:] if isinstance(exc.stdout, str) else [],
                "stderr_tail": (exc.stderr or "").splitlines()[-80:] if isinstance(exc.stderr, str) else [],
                "child_json": str(child_out) if child_out.exists() else None,
            }
        summary["jobs"].append(record)

    summary["status"] = (
        "all_completed"
        if all(job.get("returncode") == 0 for job in summary["jobs"])
        else "some_failed_or_crashed"
    )
    out = args.out_dir / f"probe_device{args.device_id}_tensor_access_matrix.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "all_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
