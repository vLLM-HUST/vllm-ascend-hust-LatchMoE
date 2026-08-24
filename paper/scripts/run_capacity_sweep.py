#!/usr/bin/env python3
"""Run the frozen 16/32/64-slot Qwen3 online capacity sweep."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SUITE = REPO_ROOT / "benchmark" / "scripts" / "run_suite.py"
AUDIT = REPO_ROOT / "paper" / "scripts" / "audit_capacity_sweep.py"
CASES = {16: "sew_14gb_slots16", 32: "sew_14gb_slots32", 64: "sew_14gb_slots64"}


def _wait_port() -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        sock = socket.socket()
        try:
            sock.bind(("127.0.0.1", 8026))
            return
        except OSError:
            time.sleep(1)
        finally:
            sock.close()
    raise TimeoutError("port 8026 was not released")


def _run(command: list[str]) -> int:
    process = subprocess.Popen(command, cwd=REPO_ROOT, start_new_session=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGINT)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", type=int, choices=(5, 6), required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--vllm-root", required=True)
    parser.add_argument("--seam-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--requests", type=int, default=32)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ack_dir = output_root / "release-acks"
    units = []
    for slots, case in CASES.items():
        _wait_port()
        stage_root = output_root / f"slots-{slots}"
        before = set(stage_root.glob("sew-offload-ascend-v1-*"))
        command = [
            str(args.python), str(RUN_SUITE), "--output-root", str(stage_root),
            "--case", case, "--workload", "prefill_heavy",
            "--max-requests", str(args.requests), "--client-concurrency", "1",
            "--max-num-seqs", "1", "--device", str(args.device),
            "--managed-backend", "locked-host",
            "--server-manager", str(REPO_ROOT / "benchmark" / "scripts" / "manage_locked_host_runtime.py"),
            "--host-python", str(args.python), "--vllm-root", str(Path(args.vllm_root).resolve()),
            "--seam-root", str(Path(args.seam_root).resolve()), "--release-ack-dir", str(ack_dir),
            "--startup-timeout-s", "1200", "--python", str(args.python),
            "--manifest", str(Path(args.manifest).resolve()), "--model-path", str(Path(args.model_path).resolve()),
            "--dataset-path", str(Path(args.dataset_path).resolve()),
            "--custody-unit-prefix", f"capacity-slots-{slots}",
        ]
        returncode = _run(command)
        created = set(stage_root.glob("sew-offload-ascend-v1-*")) - before
        if len(created) != 1:
            raise RuntimeError(f"slots {slots}: expected one suite, found {len(created)}")
        unit = created.pop() / case / "prefill_heavy"
        units.append({"slots": slots, "case": case, "unit_dir": str(unit), "runner_returncode": returncode})
        (output_root / "campaign.json").write_text(json.dumps({"units": units}, indent=2), encoding="utf-8")
        if returncode != 0:
            break
    checked = subprocess.run(
        [str(args.python), str(AUDIT), "--campaign", str(output_root / "campaign.json"),
         "--expected-requests", str(args.requests), "--output", str(output_root / "capacity_summary.json")],
        cwd=REPO_ROOT,
        check=False,
    )
    if checked.returncode == 0:
        (output_root / "PASSED.txt").write_text("16/32/64-slot capacity sweep passed\n", encoding="utf-8")
    return checked.returncode


if __name__ == "__main__":
    raise SystemExit(main())
