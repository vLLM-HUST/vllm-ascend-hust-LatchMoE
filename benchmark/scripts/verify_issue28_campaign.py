#!/usr/bin/env python3
"""Fail-closed verifier for the Issue #28 matched campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from issue28_campaign import (  # noqa: E402
    CAPACITY_STATUSES,
    REQUIRED_SUCCESS_ARTIFACTS,
    SUCCESS_STATUSES,
    arm_map,
    contract_digest,
    expected_units,
    load_contract,
    metric_values,
    percentile,
    read_json,
    sha256_file,
    summarize,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _identity_failures(payload: dict[str, Any], contract: dict[str, Any], digest: str) -> list[str]:
    failures: list[str] = []
    if payload.get("campaign_id") != contract["campaign_id"]:
        failures.append("campaign_id does not match contract")
    if payload.get("contract_sha256") != digest:
        failures.append("contract_sha256 does not match contract")
    identity = payload.get("identity") or {}
    for key in ("model", "request_manifest", "serving"):
        if identity.get(key) != contract.get(key, {}):
            failures.append(f"unit identity field differs: {key}")
    return failures


def _contract_freeze_failures(contract: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    model_digest = str((contract.get("model") or {}).get("config_sha256") or "")
    request_digest = str((contract.get("request_manifest") or {}).get("sha256") or "")
    if not model_digest or model_digest.startswith("FILL_"):
        failures.append("model config digest is not frozen")
    if not request_digest or request_digest.startswith("FILL_"):
        failures.append("request manifest digest is not frozen")
    if not bool((contract.get("request_manifest") or {}).get("order_frozen")):
        failures.append("request order is not explicitly frozen")
    return failures


def _metric_artifact(unit_dir: Path, name: str) -> dict[str, Any] | None:
    path = unit_dir / name
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def verify_unit(
    unit_dir: Path,
    *,
    contract: dict[str, Any],
    digest: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    manifest_path = unit_dir / "unit_manifest.json"
    runner_path = unit_dir / "runner_result.json"
    result_path = unit_dir / "unit_result.json"
    for path in (manifest_path, runner_path, result_path, unit_dir / "stdout.log", unit_dir / "stderr.log"):
        if not path.is_file() or path.stat().st_size == 0 and path.name not in {"stderr.log"}:
            failures.append(f"missing or empty artifact: {path.name}")
    if failures:
        return {"status": "failed", "unit_id": expected["unit_id"], "failures": failures}

    manifest = read_json(manifest_path)
    runner = read_json(runner_path)
    result = read_json(result_path)
    if manifest.get("schema") != "latchmoe.issue28.unit/v1":
        failures.append("unit manifest schema is invalid")
    if manifest.get("unit_id") != expected["unit_id"]:
        failures.append("unit_id does not match fixed order")
    if manifest.get("arm") != expected["arm"] or int(manifest.get("repeat", 0)) != expected["repeat"]:
        failures.append("arm/repeat does not match fixed order")
    if int(manifest.get("order_index", -1)) != expected["order_index"]:
        failures.append("order_index does not match fixed order")
    failures.extend(_identity_failures(manifest, contract, digest))
    failures.extend(_identity_failures(runner, contract, digest))

    arm = arm_map(contract)[expected["arm"]]
    status = str(result.get("status") or runner.get("status") or "failed")
    if status in CAPACITY_STATUSES:
        if arm["expected_status"] not in {"success_or_capacity_failure", "capacity_failure"}:
            failures.append("capacity failure is not allowed for this arm")
        if int(runner.get("returncode", 0)) == 0:
            failures.append("capacity failure was reported with a zero return code")
        return {
            "status": "failed" if failures else "accepted_capacity_failure",
            "unit_id": expected["unit_id"],
            "arm": expected["arm"],
            "repeat": expected["repeat"],
            "failures": failures,
            "runner": runner,
        }
    if status not in SUCCESS_STATUSES:
        failures.append(f"unit did not succeed: {status}")
    if arm["expected_status"] == "capacity_failure":
        failures.append("capacity-only arm unexpectedly reported success")
    if int(runner.get("returncode", 1)) != 0:
        failures.append("successful unit has non-zero runner return code")

    artifacts: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_SUCCESS_ARTIFACTS:
        payload = _metric_artifact(unit_dir, name)
        if payload is None:
            failures.append(f"missing or invalid success artifact: {name}")
        else:
            artifacts[name] = payload
    raw_artifacts = result.get("raw_artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        failures.append("unit_result.raw_artifacts is empty")
    else:
        for raw in raw_artifacts:
            raw_path = unit_dir / str(raw)
            if not raw_path.is_file() or raw_path.stat().st_size == 0:
                failures.append(f"missing or empty raw artifact: {raw}")

    for payload in artifacts.values():
        failures.extend(_identity_failures(payload, contract, digest))
    runtime = artifacts.get("runtime.json", {})
    memory = artifacts.get("memory.json", {})
    transfers = artifacts.get("transfers.json", {})
    metrics = artifacts.get("metrics.json", {})
    outputs = artifacts.get("outputs.json", {})
    if runtime.get("graph_mode") != "PIECEWISE":
        failures.append("runtime graph_mode is not PIECEWISE")
    if runtime.get("graph_capture") is not True or runtime.get("graph_replay") is not True:
        failures.append("PIECEWISE graph capture/replay evidence is incomplete")
    if int(runtime.get("fallback_count", -1)) != 0:
        failures.append("runtime fallback_count is not zero")
    # Native vLLM prefetch is a comparison baseline, not a LatchMoE wave
    # scheduler.  Its H2D/wave counters are not emitted by the LatchMoE
    # profile, so absence is reported in the artifact but is not confused with
    # a failed LatchMoE dependency check.  The other three offload paths have
    # explicit profile events and must expose their transfer/wave evidence.
    # The full-layer arm is the explicit reference path: it stages the whole
    # layer and therefore does not expose a multi-wave count.  Wave evidence
    # is required for the layered and multi-wave schedulers, but absence from
    # the reference path must not turn an otherwise complete run into a false
    # negative.
    requires_wave_metrics = expected["arm"] in {"legacy_layered", "latchmoe_multi_wave"}
    requires_h2d_metrics = expected["arm"] != "native_prefetch"
    requires_consumer_dependency = expected["arm"] == "latchmoe_multi_wave"
    if requires_wave_metrics and int(runtime.get("wave_count", 0) or 0) <= 0:
        failures.append("runtime wave_count is missing")
    if requires_h2d_metrics and int(transfers.get("h2d_count", 0) or 0) <= 0:
        failures.append("H2D transfer count is missing")
    if requires_consumer_dependency:
        if transfers.get("h2d_dependency_evidence") is not True:
            failures.append("H2D consumer-dependency evidence is missing")
        elif transfers.get("h2d_dependency_ok") is not True:
            failures.append("H2D consumer dependency was not installed for every checked transfer")
    if not metric_values(metrics, "ttft_ms"):
        failures.append("TTFT samples are missing")
    if not metric_values(metrics, "tpot_ms"):
        failures.append("TPOT samples are missing")
    if not metric_values(metrics, "throughput_tok_s"):
        failures.append("throughput sample is missing")
    if not isinstance(memory.get("hbm_peak_mb"), (int, float)):
        failures.append("HBM peak is missing")
    if not isinstance(outputs.get("request_outputs"), dict) or not outputs["request_outputs"]:
        failures.append("request output token arrays are missing")
    expected_requests = int((contract.get("request_manifest") or {}).get("request_count") or 0)
    if expected_requests > 0 and len(outputs.get("request_outputs") or {}) != expected_requests:
        failures.append(
            "request output count does not match the frozen manifest: "
            f"{len(outputs.get('request_outputs') or {})}/{expected_requests}"
        )

    return {
        "status": "passed" if not failures else "failed",
        "unit_id": expected["unit_id"],
        "arm": expected["arm"],
        "repeat": expected["repeat"],
        "failures": failures,
        "runner": runner,
        "metrics": {
            key: summarize(metric_values(metrics, key))
            for key in ("ttft_ms", "tpot_ms", "throughput_tok_s")
        },
        "outputs": outputs.get("request_outputs", {}),
        "memory": memory,
        "artifact_sha256": {
            name: sha256_file(unit_dir / name)
            for name in REQUIRED_SUCCESS_ARTIFACTS
            if (unit_dir / name).is_file()
        },
    }


def verify_campaign(campaign_path: Path, contract_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    contract = load_contract(contract_path)
    digest = contract_digest(contract)
    campaign = read_json(campaign_path)
    failures: list[str] = _contract_freeze_failures(contract)
    if campaign.get("campaign_id") != contract["campaign_id"]:
        failures.append("campaign_id does not match contract")
    if campaign.get("contract_sha256") != digest:
        failures.append("campaign contract_sha256 does not match contract")
    expected = expected_units(contract)
    observed = campaign.get("units") or []
    if [item.get("unit_id") for item in observed] != [item["unit_id"] for item in expected]:
        failures.append("campaign units are not in the fixed AB/BA order")
    reports: list[dict[str, Any]] = []
    by_id = {str(item.get("unit_id")): item for item in observed}
    for item in expected:
        record = by_id.get(item["unit_id"])
        if record is None:
            failures.append(f"missing campaign unit: {item['unit_id']}")
            continue
        report = verify_unit(Path(str(record.get("unit_dir"))), contract=contract, digest=digest, expected=item)
        reports.append(report)
        if report["status"] == "failed":
            failures.extend(f"{item['unit_id']}: {failure}" for failure in report.get("failures", []))

    arms = arm_map(contract)
    arm_reports: dict[str, Any] = {}
    successful_outputs: dict[str, dict[str, list[int]]] = {}
    for arm_name, arm in arms.items():
        arm_units = [report for report in reports if report.get("arm") == arm_name]
        accepted = [report for report in arm_units if report.get("status") in {"passed", "accepted_capacity_failure"}]
        if len(accepted) != 3:
            failures.append(f"{arm_name}: expected 3 accepted repeats, found {len(accepted)}")
        if arm["expected_status"] == "success" and any(report.get("status") != "passed" for report in arm_units):
            failures.append(f"{arm_name}: all three repeats must succeed")
        passed = [report for report in arm_units if report.get("status") == "passed"]
        if passed:
            successful_outputs[arm_name] = passed[0].get("outputs", {})
            for report in passed[1:]:
                if report.get("outputs", {}) != successful_outputs[arm_name]:
                    failures.append(f"{arm_name}: output token arrays differ between repeats")
        arm_reports[arm_name] = {
            "expected_status": arm["expected_status"],
            "accepted_repeats": len(accepted),
            "successful_repeats": len(passed),
            "capacity_failures": sum(report.get("status") == "accepted_capacity_failure" for report in arm_units),
            "ttft_ms": summarize([report["metrics"]["ttft_ms"]["median"] for report in passed if report.get("metrics")]),
            "tpot_ms": summarize([report["metrics"]["tpot_ms"]["median"] for report in passed if report.get("metrics")]),
            "throughput_tok_s": summarize([report["metrics"]["throughput_tok_s"]["median"] for report in passed if report.get("metrics")]),
        }
    if len(successful_outputs) < 2:
        failures.append("fewer than two arms produced successful matched data")
    all_outputs = list(successful_outputs.values())
    if all_outputs and any(outputs != all_outputs[0] for outputs in all_outputs[1:]):
        failures.append("successful arms produced different request output token arrays")
    report = {
        "schema": "latchmoe.issue28.verification/v1",
        "status": "passed" if not failures else "failed",
        "campaign_id": contract["campaign_id"],
        "contract_sha256": digest,
        "failures": failures,
        "units": reports,
        "arms": arm_reports,
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = verify_campaign(args.campaign.resolve(), args.contract.resolve(), args.output.resolve() if args.output else None)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"issue28 campaign verifier: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "failures": report["failures"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
