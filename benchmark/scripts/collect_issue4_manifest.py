#!/usr/bin/env python3
"""Write a machine-readable Issue #4 reproduction manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return getattr(exc, "output", None) or str(exc)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    distributions = {
        "torch": "torch",
        "torch_npu": "torch_npu",
        "vllm": "vllm",
        "vllm_ascend": "vllm-ascend",
    }
    for name, distribution in distributions.items():
        try:
            module = __import__(name)
            module_version = getattr(module, "__version__", None)
            versions[name] = module_version or version(distribution)
        except PackageNotFoundError:
            versions[name] = "distribution metadata unavailable"
        except Exception as exc:
            versions[name] = f"unavailable: {type(exc).__name__}"
    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--seam-root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--status", choices=("started", "passed", "failed"), required=True)
    parser.add_argument("--command-file", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "reproducibility.json"
    previous: dict[str, object] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))

    command_file = Path(args.command_file)
    command = command_file.read_text(encoding="utf-8") if command_file.exists() else ""
    smoke_command_path = run_dir / "smoke_command.txt"
    smoke_command = (
        smoke_command_path.read_text(encoding="utf-8")
        if smoke_command_path.exists()
        else ""
    )
    tracked_env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key == "ASCEND_RT_VISIBLE_DEVICES"
        or key == "PYTHONPATH"
        or key.startswith("VLLM_ASCEND_MOE_")
        or key.startswith("SEW_")
    }
    status_history = list(previous.get("status_history", []))
    status_history.append(
        {
            "status": args.status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    payload = {
        "schema_version": 1,
        "status": args.status,
        "status_history": status_history,
        "runtime_root": str(Path(args.runtime_root).resolve()),
        "runtime_commit": _git_revision(Path(args.runtime_root)),
        "seam_root": str(Path(args.seam_root).resolve()),
        "seam_commit": _git_revision(Path(args.seam_root)),
        "device": str(args.device),
        "command": command,
        "smoke_command": smoke_command,
        "environment": tracked_env,
        "package_versions": _package_versions(),
        "npu_smi": _command_output(["npu-smi", "info"]),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
