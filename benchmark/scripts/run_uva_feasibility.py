#!/usr/bin/env python3
"""Run the Ascend-UVA-like feasibility probes from the benchmark config.

Some probes are expected to return non-zero because the failure itself is the
evidence, e.g. host-registered matmul weights failing while HBM references pass.
This runner records those outcomes and treats the aggregate verdict JSON as the
paper-facing pass/fail boundary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_DIR.parent
DEFAULT_CONFIG = BENCHMARK_DIR / "configs" / "sew_offload_v1.yaml"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to read the benchmark YAML config.") from exc
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def build_plan(config: dict[str, Any], only: set[str] | None = None) -> list[dict[str, Any]]:
    uva = config.get("uva_like_feasibility") or {}
    commands = uva.get("commands") or {}
    if not isinstance(commands, dict):
        raise ValueError("uva_like_feasibility.commands must be a mapping")
    expected_nonzero = set(uva.get("expected_nonzero_commands") or [])
    plan = []
    for name, command in commands.items():
        if only and name not in only:
            continue
        if not isinstance(command, list) or not command:
            raise ValueError(f"command {name} must be a non-empty list")
        plan.append(
            {
                "name": str(name),
                "command": [str(item) for item in command],
                "expected_nonzero": str(name) in expected_nonzero,
            }
        )
    if only:
        missing = sorted(only - {item["name"] for item in plan})
        if missing:
            raise KeyError(f"unknown UVA commands: {', '.join(missing)}")
    return plan


def _tail(text: str, limit: int = 80) -> list[str]:
    return text.splitlines()[-limit:]


def run_plan(plan: list[dict[str, Any]], cwd: Path, timeout_s: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in plan:
        started = time.monotonic()
        try:
            proc = subprocess.run(
                item["command"],
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
            elapsed_s = time.monotonic() - started
            expected_nonzero = bool(item.get("expected_nonzero"))
            if proc.returncode == 0:
                status = "ok"
            elif expected_nonzero:
                status = "expected_nonzero"
            else:
                status = "unexpected_nonzero"
            record = {
                "name": item["name"],
                "command": item["command"],
                "expected_nonzero": expected_nonzero,
                "returncode": proc.returncode,
                "status": status,
                "elapsed_s": elapsed_s,
                "stdout_tail": _tail(proc.stdout),
                "stderr_tail": _tail(proc.stderr),
            }
        except subprocess.TimeoutExpired as exc:
            elapsed_s = time.monotonic() - started
            record = {
                "name": item["name"],
                "command": item["command"],
                "expected_nonzero": bool(item.get("expected_nonzero")),
                "status": "timeout",
                "elapsed_s": elapsed_s,
                "timeout_s": timeout_s,
                "stdout_tail": _tail(exc.stdout or "") if isinstance(exc.stdout, str) else [],
                "stderr_tail": _tail(exc.stderr or "") if isinstance(exc.stderr, str) else [],
            }
        records.append(record)
        if record["status"] in {"unexpected_nonzero", "timeout"}:
            break
    return records


def derive_runner_status(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    verdict_path: Path,
) -> dict[str, Any]:
    uva = config.get("uva_like_feasibility") or {}
    expected_verdict = uva.get("expected_verdict")
    unexpected = [record for record in records if record.get("status") in {"unexpected_nonzero", "timeout"}]
    verdict = None
    if verdict_path.exists():
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    verdict_value = (verdict or {}).get("verdict")
    verdict_matches = expected_verdict is None or verdict_value == expected_verdict
    if unexpected:
        status = "failed_unexpected_command"
    elif verdict is None:
        status = "failed_missing_verdict"
    elif not verdict_matches:
        status = "failed_verdict_mismatch"
    else:
        status = "ok"
    return {
        "status": status,
        "expected_verdict": expected_verdict,
        "verdict": verdict_value,
        "verdict_matches": verdict_matches,
        "unexpected_commands": unexpected,
        "verdict_path": str(verdict_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only", nargs="*", default=None, help="Optional subset of UVA command names to run.")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark/artifacts/reports/ascend_uva_feasibility/e0_runner_manifest.json"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    plan = build_plan(config, set(args.only) if args.only else None)
    uva = config.get("uva_like_feasibility") or {}
    verdict_path = REPO_ROOT / "benchmark/artifacts/reports/ascend_uva_feasibility/e0_ascend_uva_like_verdict.json"
    collect_command = (uva.get("commands") or {}).get("collect_summary") or []
    if "--verdict-out" in collect_command:
        idx = collect_command.index("--verdict-out")
        if idx + 1 < len(collect_command):
            verdict_path = REPO_ROOT / str(collect_command[idx + 1])

    manifest: dict[str, Any] = {
        "runner": "ascend_uva_like_feasibility",
        "timestamp_utc": _now(),
        "config": str(args.config),
        "dry_run": args.dry_run,
        "plan": plan,
        "records": [],
        "summary": {},
    }
    if not args.dry_run:
        manifest["records"] = run_plan(plan, REPO_ROOT, args.timeout_s)
        manifest["summary"] = derive_runner_status(config, manifest["records"], verdict_path)
    else:
        manifest["summary"] = {"status": "dry_run", "verdict_path": str(verdict_path)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    print(json.dumps(manifest["summary"], ensure_ascii=False, sort_keys=True))
    return 0 if manifest["summary"].get("status") in {"ok", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
