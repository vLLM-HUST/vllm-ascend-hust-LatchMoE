#!/usr/bin/env python3
"""Run the gated Issue #7 smoke, short gate, and three independent startups."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SUITE = REPO_ROOT / "benchmark" / "scripts" / "run_suite.py"
VERIFY = REPO_ROOT / "benchmark" / "scripts" / "verify_issue7_graph_unit.py"


def _latest_suite(output_root: Path, before: set[Path]) -> Path:
    candidates = {
        path for path in output_root.glob("sew-offload-ascend-v1-*") if path.is_dir()
    } - before
    if len(candidates) != 1:
        raise RuntimeError(f"expected one new suite directory, found {len(candidates)}")
    return candidates.pop()


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
    parser.add_argument("--short-requests", type=int, default=11)
    parser.add_argument("--repeat-requests", type=int, default=32)
    parser.add_argument("--startup-timeout-s", type=float, default=1200)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ack_dir = output_root / "release-acks"
    manifest = Path(args.manifest).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    stages = [
        ("smoke", "smoke", 1),
        ("short-gate", "mixed_chat", int(args.short_requests)),
        *[
            (f"repeat-{index}", "mixed_chat", int(args.repeat_requests))
            for index in range(1, 4)
        ],
    ]
    bundle_results = []
    repeat_oracle: Path | None = None
    for stage_name, workload, requests in stages:
        stage_root = output_root / stage_name
        before = {path for path in stage_root.glob("sew-offload-ascend-v1-*")}
        command = [
            str(args.python),
            str(RUN_SUITE),
            "--output-root",
            str(stage_root),
            "--case",
            "sew_14gb_autoslots",
            "--workload",
            workload,
            "--max-requests",
            str(requests),
            "--client-concurrency",
            "1",
            "--max-num-seqs",
            "1",
            "--device",
            str(args.device),
            "--managed-backend",
            "locked-host",
            "--server-manager",
            str(REPO_ROOT / "benchmark" / "scripts" / "manage_locked_host_runtime.py"),
            "--host-python",
            str(args.python),
            "--vllm-root",
            str(args.vllm_root),
            "--seam-root",
            str(args.seam_root),
            "--release-ack-dir",
            str(ack_dir),
            "--startup-timeout-s",
            str(args.startup_timeout_s),
            "--python",
            str(args.python),
            "--manifest",
            str(manifest),
            "--model-path",
            str(Path(args.model_path).resolve()),
            "--dataset-path",
            str(Path(args.dataset_path).resolve()),
            "--custody-unit-prefix",
            f"issue7-{stage_name}",
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        suite_dir = _latest_suite(stage_root, before)
        unit_dir = suite_dir / "sew_14gb_autoslots" / workload
        if completed.returncode != 0:
            (unit_dir / "FAILED.txt").write_text(
                f"suite runner exited with code {completed.returncode}\n",
                encoding="utf-8",
            )
            return completed.returncode
        verify_command = [
            str(args.python),
            str(VERIFY),
            "--unit-dir",
            str(unit_dir),
            "--minimum-requests",
            str(requests),
        ]
        if stage_name == "repeat-1":
            short = next(item for item in bundle_results if item["stage"] == "short-gate")
            verify_command.extend(
                ["--oracle-benchmark", str(Path(short["unit_dir"]) / "benchmark.json")]
            )
        elif stage_name.startswith("repeat-") and repeat_oracle is not None:
            verify_command.extend(["--oracle-benchmark", str(repeat_oracle)])
        verified = subprocess.run(
            verify_command,
            cwd=REPO_ROOT,
            check=False,
        )
        report = json.loads(
            (unit_dir / "graph_correctness.json").read_text(encoding="utf-8")
        )
        bundle_results.append(
            {
                "stage": stage_name,
                "requests": requests,
                "suite_dir": str(suite_dir),
                "unit_dir": str(unit_dir),
                "verification": report,
            }
        )
        (output_root / "bundle_results.json").write_text(
            json.dumps(bundle_results, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if verified.returncode != 0:
            return verified.returncode
        if stage_name == "repeat-1":
            repeat_oracle = unit_dir / "benchmark.json"

    repeat_units = [item for item in bundle_results if item["stage"].startswith("repeat-")]
    if len(repeat_units) != 3 or any(
        item["verification"].get("status") != "passed" for item in repeat_units
    ):
        raise RuntimeError("three independent startup gate did not pass")
    (output_root / "PASSED.txt").write_text(
        "smoke, short gate, and three independent Graph startups passed\n",
        encoding="utf-8",
    )
    print(f"Issue #7 graph bundle passed: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
