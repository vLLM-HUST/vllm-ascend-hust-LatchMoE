#!/usr/bin/env python3
"""Run three counterordered starts of four graph-mode serving baselines."""

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
VERIFY = REPO_ROOT / "benchmark" / "scripts" / "verify_baseline_matched.py"
ORDERS = (
    ("full_resident", "native_prefetch", "legacy_layered", "latchmoe"),
    ("latchmoe", "legacy_layered", "native_prefetch", "full_resident"),
    ("legacy_layered", "latchmoe", "full_resident", "native_prefetch"),
)
CASES = {
    "full_resident": "no_offload_kv512m_aclgraph",
    "native_prefetch": "native_prefetch_14gb_kv512m",
    "legacy_layered": "legacy_layered_14gb_kv512m",
    "latchmoe": "sew_14gb_autoslots_kv512m",
}


def _latest_suite(root: Path, before: set[Path]) -> Path:
    created = {path for path in root.glob("sew-offload-ascend-v1-*") if path.is_dir()} - before
    if len(created) != 1:
        raise RuntimeError(f"expected one new suite directory, found {len(created)}")
    return created.pop()


def _run(command: list[str]) -> int:
    process = subprocess.Popen(command, cwd=REPO_ROOT, start_new_session=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGINT)
        try:
            return process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            return process.wait(timeout=30)


def _wait_port(timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sock = socket.socket()
        try:
            sock.bind(("127.0.0.1", 8026))
            return
        except OSError:
            time.sleep(1.0)
        finally:
            sock.close()
    raise TimeoutError("port 8026 was not released")


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
    parser.add_argument("--workload", default="prefill_heavy")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--startup-timeout-s", type=float, default=1200)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ack_dir = output_root / "release-acks"
    units: list[dict[str, object]] = []
    oracle: Path | None = None
    for repeat, order in enumerate(ORDERS, 1):
        for position, arm in enumerate(order, 1):
            _wait_port()
            stage = f"repeat-{repeat}-position-{position}-{arm}"
            stage_root = output_root / stage
            before = set(stage_root.glob("sew-offload-ascend-v1-*"))
            command = [
                str(args.python), str(RUN_SUITE),
                "--output-root", str(stage_root),
                "--case", CASES[arm],
                "--workload", args.workload,
                "--max-requests", str(args.requests),
                "--client-concurrency", "1",
                "--max-num-seqs", "1",
                "--device", str(args.device),
                "--managed-backend", "locked-host",
                "--server-manager", str(REPO_ROOT / "benchmark" / "scripts" / "manage_locked_host_runtime.py"),
                "--host-python", str(args.python),
                "--vllm-root", str(Path(args.vllm_root).resolve()),
                "--seam-root", str(Path(args.seam_root).resolve()),
                "--release-ack-dir", str(ack_dir),
                "--startup-timeout-s", str(args.startup_timeout_s),
                "--python", str(args.python),
                "--manifest", str(Path(args.manifest).resolve()),
                "--model-path", str(Path(args.model_path).resolve()),
                "--dataset-path", str(Path(args.dataset_path).resolve()),
                "--custody-unit-prefix", f"baseline-{stage}",
            ]
            returncode = _run(command)
            suite = _latest_suite(stage_root, before)
            unit = suite / CASES[arm] / args.workload
            units.append({
                "repeat": repeat, "position": position, "stage": stage,
                "arm": arm, "unit_dir": str(unit), "runner_returncode": returncode,
            })
            campaign = output_root / "campaign.json"
            campaign.write_text(
                json.dumps({"orders": ORDERS, "units": units}, indent=2), encoding="utf-8"
            )
            verify = [
                str(args.python), str(VERIFY), "--unit-dir", str(unit),
                "--arm", arm, "--expected-requests", str(args.requests),
            ]
            if oracle is not None:
                verify.extend(["--oracle-benchmark", str(oracle)])
            checked = subprocess.run(verify, cwd=REPO_ROOT, check=False)
            if oracle is None and returncode == 0 and checked.returncode == 0 and (unit / "benchmark.json").is_file():
                oracle = unit / "benchmark.json"
    checked = subprocess.run(
        [str(args.python), str(VERIFY), "--campaign", str(output_root / "campaign.json"),
         "--output", str(output_root / "baseline_summary.json")],
        cwd=REPO_ROOT,
        check=False,
    )
    if checked.returncode == 0:
        (output_root / "PASSED.txt").write_text("Matched four-baseline campaign passed\n", encoding="utf-8")
    return checked.returncode


if __name__ == "__main__":
    raise SystemExit(main())
