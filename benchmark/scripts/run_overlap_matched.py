#!/usr/bin/env python3
"""Run the preregistered AB/BA/AB serial-versus-overlap campaign."""

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
VERIFY = REPO_ROOT / "benchmark" / "scripts" / "verify_overlap_matched.py"
ORDER = (
    ("pair-1-serial", "serial"),
    ("pair-1-overlap", "overlap"),
    ("pair-2-overlap", "overlap"),
    ("pair-2-serial", "serial"),
    ("pair-3-serial", "serial"),
    ("pair-3-overlap", "overlap"),
)
CASES = {
    "serial": "sew_14gb_serial_stage_matched",
    "overlap": "sew_14gb_overlap_stage_matched",
}


def _latest_suite(root: Path, before: set[Path]) -> Path:
    created = {path for path in root.glob("sew-offload-ascend-v1-*") if path.is_dir()} - before
    if len(created) != 1:
        raise RuntimeError(f"expected one new suite directory, found {len(created)}")
    return created.pop()


def _run_managed(command: list[str]) -> int:
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


def _wait_for_port_release(host: str, port: int, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sock = socket.socket()
        try:
            sock.bind((host, port))
            return
        except OSError:
            time.sleep(1.0)
        finally:
            sock.close()
    raise TimeoutError(f"{host}:{port} was not released within {timeout_s:.0f}s")


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
    for stage, arm in ORDER:
        _wait_for_port_release("127.0.0.1", 8026)
        stage_root = output_root / stage
        before = {path for path in stage_root.glob("sew-offload-ascend-v1-*")}
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
            "--custody-unit-prefix", f"overlap-{stage}",
        ]
        returncode = _run_managed(command)
        suite = _latest_suite(stage_root, before)
        unit = suite / CASES[arm] / args.workload
        units.append({
            "stage": stage,
            "arm": arm,
            "unit_dir": str(unit),
            "runner_returncode": returncode,
        })
        campaign = output_root / "campaign.json"
        campaign.write_text(
            json.dumps({"order": [list(value) for value in ORDER], "units": units}, indent=2),
            encoding="utf-8",
        )
        if returncode != 0:
            return returncode
        verify = [
            str(args.python), str(VERIFY),
            "--unit-dir", str(unit),
            "--arm", arm,
            "--expected-requests", str(args.requests),
        ]
        if oracle is not None:
            verify.extend(["--oracle-benchmark", str(oracle)])
        checked = subprocess.run(verify, cwd=REPO_ROOT, check=False)
        if checked.returncode != 0:
            return checked.returncode
        if oracle is None:
            oracle = unit / "benchmark.json"

    checked = subprocess.run(
        [
            str(args.python), str(VERIFY),
            "--campaign", str(output_root / "campaign.json"),
            "--output", str(output_root / "overlap_summary.json"),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    if checked.returncode == 0:
        (output_root / "PASSED.txt").write_text(
            "Matched serial/overlap AB/BA/AB campaign passed integrity gates\n",
            encoding="utf-8",
        )
    return checked.returncode


if __name__ == "__main__":
    raise SystemExit(main())
