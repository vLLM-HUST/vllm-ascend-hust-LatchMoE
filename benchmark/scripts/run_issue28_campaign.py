#!/usr/bin/env python3
"""Run the fixed, three-repeat matched campaign for Issue #28.

The contract supplies the exact command for each baseline.  Commands are
argv arrays (never shell strings), and each unit gets a fresh output directory
and a fresh process.  A non-zero command is retained as an artifact; an OOM
is accepted only when that arm explicitly allows a capacity failure.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from issue28_campaign import (  # noqa: E402
    CAPACITY_STATUSES,
    arm_map,
    contains_capacity_marker,
    contract_digest,
    expected_units,
    expand_tokens,
    inherited_environment,
    load_contract,
    read_json,
    selected_environment,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(command: list[str], *, cwd: Path, env: dict[str, str], stdout: Path, stderr: Path) -> int:
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        try:
            return int(process.wait())
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            try:
                return int(process.wait(timeout=120))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                return int(process.wait(timeout=30))


def _external_status(unit_dir: Path) -> str | None:
    result = unit_dir / "unit_result.json"
    if not result.is_file() or result.stat().st_size == 0:
        return None
    try:
        value = read_json(result).get("status")
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return str(value) if value is not None else None


def _unit_manifest(
    contract: dict[str, Any],
    digest: str,
    item: dict[str, Any],
    command: list[str],
    environment: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema": "latchmoe.issue28.unit/v1",
        "campaign_id": contract["campaign_id"],
        "contract_sha256": digest,
        "order_index": item["order_index"],
        "unit_id": item["unit_id"],
        "arm": item["arm"],
        "repeat": item["repeat"],
        "command": command,
        "selected_env": selected_environment(environment),
        "identity": {
            "model": contract.get("model", {}),
            "request_manifest": contract.get("request_manifest", {}),
            "serving": contract.get("serving", {}),
        },
    }


def _status_allowed(status: str | None, expected: str) -> bool:
    if status == "success" and expected in {"success", "success_or_capacity_failure"}:
        return True
    return status == "capacity_failure" and expected in {"success_or_capacity_failure", "capacity_failure"}


def _campaign_payload(
    contract: dict[str, Any],
    contract_path: Path,
    digest: str,
    units: list[dict[str, Any]],
    failures: list[str],
    status: str,
) -> dict[str, Any]:
    return {
        "schema": "latchmoe.issue28.campaign-result/v1",
        "campaign_id": contract["campaign_id"],
        "contract": str(contract_path.resolve()),
        "contract_sha256": digest,
        "order": expected_units(contract),
        "units": units,
        "status": status,
        "failures": failures,
    }


def run_campaign(
    contract_path: Path,
    output_root: Path,
    *,
    python: Path,
    dry_run: bool,
    resume: bool = False,
) -> int:
    if dry_run and resume:
        raise ValueError("--dry-run and --resume cannot be used together")
    contract = load_contract(contract_path)
    digest = contract_digest(contract)
    repo_root = Path(__file__).resolve().parents[2]
    output_root.mkdir(parents=True, exist_ok=True)
    arms = arm_map(contract)
    base_env = inherited_environment()
    units: list[dict[str, Any]] = []
    overall_failures: list[str] = []
    existing_by_id: dict[str, dict[str, Any]] = {}
    campaign_path = output_root / "campaign.json"
    if resume and campaign_path.is_file():
        previous = read_json(campaign_path)
        if previous.get("campaign_id") != contract["campaign_id"]:
            raise ValueError("--resume campaign_id does not match the contract")
        if previous.get("contract_sha256") != digest:
            raise ValueError("--resume contract digest does not match the existing campaign")
        existing_by_id = {
            str(record["unit_id"]): record
            for record in previous.get("units", [])
            if isinstance(record, dict) and record.get("unit_id")
        }
    for item in expected_units(contract):
        arm = arms[item["arm"]]
        unit_dir = output_root / item["unit_id"]
        if unit_dir.exists() and any(unit_dir.iterdir()):
            expected = str(arm["expected_status"])
            external = _external_status(unit_dir)
            runner_result = unit_dir / "runner_result.json"
            if resume and _status_allowed(external, expected) and runner_result.is_file():
                previous_record = existing_by_id.get(item["unit_id"])
                if previous_record is not None:
                    recorded_dir = Path(str(previous_record.get("unit_dir", ""))).resolve()
                    if recorded_dir != unit_dir.resolve():
                        raise ValueError(
                            f"existing unit_dir for {item['unit_id']} does not match output root: "
                            f"{recorded_dir} != {unit_dir.resolve()}"
                        )
                runner = read_json(runner_result)
                record = {
                    **item,
                    "unit_dir": str(unit_dir),
                    "status": external,
                    "returncode": int(runner.get("returncode", 0)),
                }
                units.append(record)
                if not _status_allowed(external, expected):
                    overall_failures.append(
                        f"{item['unit_id']}: status={external}, expected={expected}"
                    )
                _write_json(
                    campaign_path,
                    _campaign_payload(
                        contract,
                        contract_path,
                        digest,
                        units,
                        overall_failures,
                        "failed" if overall_failures else "running",
                    ),
                )
                continue
            if resume:
                backup_dir = unit_dir.with_name(f"{unit_dir.name}.aborted-{time.time_ns()}")
                unit_dir.rename(backup_dir)
                print(
                    f"issue28 campaign: preserved incomplete {unit_dir.name} as {backup_dir.name}",
                    file=sys.stderr,
                )
            else:
                raise RuntimeError(f"refusing to reuse non-empty unit directory: {unit_dir}")
        unit_dir.mkdir(parents=True, exist_ok=True)
        values = {
            "python": str(python),
            "repo_root": str(repo_root),
            "contract": str(contract_path.resolve()),
            "output_root": str(output_root.resolve()),
            "unit_dir": str(unit_dir.resolve()),
            "campaign_id": str(contract["campaign_id"]),
            "arm": item["arm"],
            "repeat": str(item["repeat"]),
            "device": str(contract.get("serving", {}).get("device", "")),
            # Sequential units use isolated ports to avoid a kernel/socket
            # release race after the previous managed server exits.
            "port": str(8100 + int(item["order_index"])),
            "model_path": str(contract.get("model", {}).get("checkpoint", "")),
            "request_manifest": str(contract.get("request_manifest", {}).get("path", "")),
            "vllm_root": str(contract.get("runtime", {}).get("vllm_root", "")),
            "seam_root": str(contract.get("runtime", {}).get("seam_root", "")),
        }
        command = expand_tokens([str(token) for token in arm["command"]], values)
        environment = dict(base_env)
        for key, value in (arm.get("env") or {}).items():
            if value is None:
                environment.pop(str(key), None)
            else:
                environment[str(key)] = str(value)
        _write_json(unit_dir / "unit_manifest.json", _unit_manifest(contract, digest, item, command, environment))
        started_ns = time.time_ns()
        if dry_run:
            returncode = 0
            status = "planned"
            (unit_dir / "stdout.log").write_text("dry-run: command was not executed\n", encoding="utf-8")
            (unit_dir / "stderr.log").write_text("", encoding="utf-8")
        else:
            returncode = _run(
                command,
                cwd=repo_root,
                env=environment,
                stdout=unit_dir / "stdout.log",
                stderr=unit_dir / "stderr.log",
            )
            existing = _external_status(unit_dir)
            combined = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in (unit_dir / "stdout.log", unit_dir / "stderr.log")
            )
            if existing in CAPACITY_STATUSES:
                status = existing
            elif returncode != 0 and contains_capacity_marker(combined):
                status = "capacity_failure"
            elif existing:
                status = existing
            else:
                status = "success" if returncode == 0 else "failed"
        runner_result = {
            "schema": "latchmoe.issue28.runner-result/v1",
            "campaign_id": contract["campaign_id"],
            "contract_sha256": digest,
            "unit_id": item["unit_id"],
            "returncode": returncode,
            "status": status,
            "expected_status": arm["expected_status"],
            "started_at_ns": started_ns,
            "finished_at_ns": time.time_ns(),
            "capacity_marker_observed": status == "capacity_failure",
            "identity": {
                "model": contract.get("model", {}),
                "request_manifest": contract.get("request_manifest", {}),
                "serving": contract.get("serving", {}),
            },
        }
        _write_json(unit_dir / "runner_result.json", runner_result)
        if not (unit_dir / "unit_result.json").is_file():
            _write_json(
                unit_dir / "unit_result.json",
                {
                    "status": status,
                    "release_status": "released" if status in {"success", "planned"} else "not_released",
                    "raw_artifacts": ["stdout.log", "stderr.log", "runner_result.json"],
                },
            )
        record = {**item, "unit_dir": str(unit_dir), "status": status, "returncode": returncode}
        units.append(record)
        expected = str(arm["expected_status"])
        allowed = _status_allowed(status, expected)
        if not dry_run and not allowed:
            overall_failures.append(f"{item['unit_id']}: status={status}, expected={expected}")
        _write_json(
            campaign_path,
            _campaign_payload(
                contract,
                contract_path,
                digest,
                units,
                overall_failures,
                "planned" if dry_run else ("failed" if overall_failures else "running"),
            ),
        )

    final_status = "planned" if dry_run else ("failed" if overall_failures else "completed")
    _write_json(campaign_path, _campaign_payload(contract, contract_path, digest, units, overall_failures, final_status))
    return 0 if dry_run or not overall_failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed units and preserve an incomplete unit before rerunning it",
    )
    args = parser.parse_args()
    try:
        return run_campaign(
            args.contract.resolve(),
            args.output_root.resolve(),
            python=args.python.resolve(),
            dry_run=args.dry_run,
            resume=args.resume,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"issue28 campaign runner: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
