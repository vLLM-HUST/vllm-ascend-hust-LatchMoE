#!/usr/bin/env python3
"""Run the fixed AB/BA/AB 200-request matched campaign for Issue #17."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SUITE = REPO_ROOT / "benchmark" / "scripts" / "run_suite.py"
VERIFY = REPO_ROOT / "benchmark" / "scripts" / "verify_issue17_matched_ttft.py"
ORDER = (
    ("pair-1-full_layer", "full_layer"),
    ("pair-1-multi_wave", "multi_wave"),
    ("pair-2-multi_wave", "multi_wave"),
    ("pair-2-full_layer", "full_layer"),
    ("pair-3-full_layer", "full_layer"),
    ("pair-3-multi_wave", "multi_wave"),
)
CASES = {
    "full_layer": "sew_14gb_full_layer_matched",
    "multi_wave": "sew_14gb_multi_wave_matched",
}


def _latest_suite(root: Path, before: set[Path]) -> Path:
    created = {path for path in root.glob("sew-offload-ascend-v1-*") if path.is_dir()} - before
    if len(created) != 1:
        raise RuntimeError(f"expected one new suite directory, found {len(created)}")
    return created.pop()


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
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--startup-timeout-s", type=float, default=1200)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ack_dir = output_root / "release-acks"
    units = []
    oracle: Path | None = None
    for stage, arm in ORDER:
        stage_root = output_root / stage
        before = {path for path in stage_root.glob("sew-offload-ascend-v1-*")}
        command = [
            str(args.python), str(RUN_SUITE),
            "--output-root", str(stage_root),
            "--case", CASES[arm],
            "--workload", "mixed_chat",
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
            "--custody-unit-prefix", f"issue17-{stage}",
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        suite = _latest_suite(stage_root, before)
        unit = suite / CASES[arm] / "mixed_chat"
        item = {"stage": stage, "arm": arm, "unit_dir": str(unit), "runner_returncode": completed.returncode}
        units.append(item)
        campaign = output_root / "campaign.json"
        campaign.write_text(json.dumps({"order": [list(value) for value in ORDER], "units": units}, indent=2), encoding="utf-8")
        if completed.returncode != 0:
            return completed.returncode
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

    campaign = output_root / "campaign.json"
    checked = subprocess.run(
        [str(args.python), str(VERIFY), "--campaign", str(campaign), "--output", str(output_root / "matched_summary.json")],
        cwd=REPO_ROOT,
        check=False,
    )
    if checked.returncode == 0:
        (output_root / "PASSED.txt").write_text("Issue #17 matched AB/BA/AB campaign passed\n", encoding="utf-8")
    return checked.returncode


if __name__ == "__main__":
    raise SystemExit(main())
