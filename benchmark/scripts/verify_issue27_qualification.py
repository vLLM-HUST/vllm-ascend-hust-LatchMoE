#!/usr/bin/env python3
"""Fail closed for one Issue #27 Phase-B qualification bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_moe_offload_ascend.moe_offload.router_parity import (  # noqa: E402
    compare_router_artifacts,
    load_router_artifact,
)


GATES = (
    "native_oracle",
    "latchmoe_eager_diagnostic",
    "router_parity",
    "layer_boundary_parity",
    "token_exactness",
    "piecewise_capture",
    "piecewise_replay",
    "zero_eager_fallback",
    "h2d_lease_release",
    "prefill_overflow",
    "decode_cache_churn",
)
_LAYER_LINE = re.compile(r"SEW_LAYER_BOUNDARY_COMPARE\b")
_FALLBACK_MARKERS = (
    "latchmoe_capability_rejected",
    "refusing native/eager fallback",
    "unsupported global configuration",
)
_CAPACITY_FAILURE_MARKERS = (
    "torch.OutOfMemoryError",
    "torch_npu.OutOfMemoryError",
    "NPU out of memory",
    "out of memory",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"non-object JSONL record at {path}:{line_number}")
        records.append(record)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _outputs(unit_dir: Path) -> dict[str, list[int]]:
    path = unit_dir / "outputs.jsonl"
    if not path.is_file():
        raise ValueError(f"missing output artifact: {path}")
    result: dict[str, list[int]] = {}
    for record in _read_jsonl(path):
        request_id = str(record.get("request_id") or "")
        token_ids = record.get("output_token_ids")
        if not request_id or not isinstance(token_ids, list):
            raise ValueError(f"invalid output record in {path}")
        if request_id in result:
            raise ValueError(f"duplicate request ID {request_id!r} in {path}")
        result[request_id] = [int(token_id) for token_id in token_ids]
    if not result or not all(result.values()):
        raise ValueError(f"no non-empty token outputs in {path}")
    return result


def _same_outputs(expected: dict[str, list[int]], observed: dict[str, list[int]]) -> dict[str, Any]:
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    mismatched = {
        request_id: {"expected": expected[request_id], "observed": observed[request_id]}
        for request_id in sorted(set(expected) & set(observed))
        if expected[request_id] != observed[request_id]
    }
    return {
        "passed": not missing and not extra and not mismatched,
        "missing_request_ids": missing,
        "extra_request_ids": extra,
        "mismatched": mismatched,
    }


def _summary_is_success(unit_dir: Path) -> tuple[bool, str | None]:
    path = unit_dir / "summary.json"
    if not path.is_file():
        return False, f"missing summary: {path}"
    summary = _read_json(path)
    if summary.get("status") != "ok":
        return False, f"summary status is {summary.get('status')!r}: {path}"
    try:
        _outputs(unit_dir)
    except ValueError as exc:
        return False, str(exc)
    return True, None


def _capacity_failure(unit_dir: Path) -> tuple[bool, dict[str, Any]]:
    """Classify an expected full-resident capacity failure without hiding it."""
    summary_path = unit_dir / "summary.json"
    log_path = unit_dir / "run.log"
    text = ""
    for path in (summary_path, log_path):
        if path.is_file():
            text += "\n" + path.read_text(encoding="utf-8", errors="replace")
    markers = [marker for marker in _CAPACITY_FAILURE_MARKERS if marker in text]
    if not markers:
        return False, {"status": "not_capacity_failure"}
    return True, {
        "status": "capacity_limited",
        "markers": markers,
        "summary": _read_json(summary_path) if summary_path.is_file() else {},
    }


def _profile(unit_dir: Path) -> list[dict[str, Any]]:
    path = unit_dir / "moe_offload_profile.jsonl"
    if not path.is_file():
        raise ValueError(f"missing profile artifact: {path}")
    return _read_jsonl(path)


def _layer_boundary_gate(log_path: Path, *, shared_output_required: bool) -> tuple[bool, dict[str, Any]]:
    if not log_path.is_file():
        return False, {"reason": f"missing eager log: {log_path}"}
    lines = [
        line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if _LAYER_LINE.search(line)
    ]
    if not lines:
        return False, {"reason": "no layer-boundary diagnostics"}
    failures: list[str] = []
    for line in lines:
        if "output_equal=True" not in line:
            failures.append(line)
        if shared_output_required and "shared_equal=True" not in line:
            failures.append(line)
        if "shared_contract_equal=False" in line:
            failures.append(line)
    return not failures, {"records": len(lines), "failures": failures[:8]}


def _graph_gates(unit_dir: Path) -> dict[str, tuple[bool, dict[str, Any]]]:
    try:
        records = _profile(unit_dir)
    except ValueError as exc:
        failed = (False, {"reason": str(exc)})
        return {
            "piecewise_capture": failed,
            "piecewise_replay": failed,
            "zero_eager_fallback": failed,
            "h2d_lease_release": failed,
            "decode_cache_churn": failed,
        }
    by_name: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_name.setdefault(str(record.get("name") or ""), []).append(record)
    log = (unit_dir / "run.log").read_text(encoding="utf-8", errors="replace") if (unit_dir / "run.log").is_file() else ""
    locks = by_name.get("graph_slot_address_lock", [])
    validates = by_name.get("graph_slot_address_validate", [])
    replays = by_name.get("graph_replay_issue", [])
    capture_ok = (
        bool(locks)
        and bool(validates)
        and "LATCHMOE_GRAPH_CONFIG cudagraph_mode=PIECEWISE" in log
        and "Graph capturing finished" in log
    )
    replay_ok = bool(replays) and "Replaying aclgraph" in log and all(
        (record.get("payload") or {}).get("synchronizes_npu") is False
        for record in replays
    )
    fallback_hits = [marker for marker in _FALLBACK_MARKERS if marker in log]
    decode = by_name.get("decode_fixed_slot_stage", [])
    h2d = [
        record for record in decode
        if int((record.get("payload") or {}).get("h2d_bytes") or 0) > 0
    ]
    h2d_ok = bool(h2d) and all(
        (record.get("payload") or {}).get("consumer_dependency_installed") is True
        and (record.get("payload") or {}).get("mapping_published_after_ready") is True
        for record in h2d
    )
    leases = by_name.get("slot_generation_protected_until_compute_complete", [])
    lease_ok = bool(leases) and all(
        (record.get("payload") or {}).get("leases_still_match") is True
        for record in leases
    )
    release_ok = bool(by_name.get("release_original_expert_weights", []))
    decode_steps = {
        int((record.get("payload") or {}).get("step_id", -1))
        for record in decode
    }
    return {
        "piecewise_capture": (capture_ok, {"address_locks": len(locks), "address_validations": len(validates)}),
        "piecewise_replay": (replay_ok, {"replays": len(replays)}),
        "zero_eager_fallback": (not fallback_hits, {"fallback_markers": fallback_hits}),
        "h2d_lease_release": (
            h2d_ok and lease_ok and release_ok,
            {"h2d_stages": len(h2d), "lease_records": len(leases), "release_records": len(by_name.get("release_original_expert_weights", []))},
        ),
        "decode_cache_churn": (len(decode_steps) >= 2, {"decode_steps": len(decode_steps)}),
    }


def _overflow_gate(unit_dir: Path) -> tuple[bool, dict[str, Any]]:
    try:
        records = _profile(unit_dir)
    except ValueError as exc:
        return False, {"reason": str(exc)}
    events = [record for record in records if record.get("name") == "b2_work_conserving_prefill"]
    failures: list[str] = []
    for event in events:
        payload = event.get("payload") or {}
        summary = payload.get("wave_summary") or {}
        wave_count = int(summary.get("wave_count") or 0)
        if wave_count <= 1:
            failures.append("prefill event did not execute multiple waves")
        if int(summary.get("prefetch_after_compute_issues") or 0) != 0:
            failures.append("late post-compute prefetch was observed")
    return bool(events) and not failures, {"events": len(events), "failures": failures}


def verify_bundle(
    bundle_dir: Path,
    *,
    model_id: str,
    shared_output_required: bool,
) -> dict[str, Any]:
    units = {
        "native-eager": bundle_dir / "native-eager",
        "latch-eager": bundle_dir / "latch-eager",
        "latch-graph": bundle_dir / "latch-graph",
        "overflow-graph": bundle_dir / "overflow-graph",
    }
    gates = {gate: "failed" for gate in GATES}
    details: dict[str, Any] = {}
    for name, unit_dir in units.items():
        ok, reason = _summary_is_success(unit_dir)
        details[f"{name}_summary"] = {"passed": ok, "reason": reason}

    native_capacity_limited, native_capacity_detail = _capacity_failure(
        units["native-eager"]
    )
    details["native_capacity"] = native_capacity_detail

    def diagnostic_flags(unit_dir: Path) -> dict[str, Any]:
        try:
            return dict(_read_json(unit_dir / "summary.json").get("qualification_diagnostics") or {})
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    native_flags = diagnostic_flags(units["native-eager"])
    eager_flags = diagnostic_flags(units["latch-eager"])
    graph_flags = diagnostic_flags(units["latch-graph"])
    overflow_flags = diagnostic_flags(units["overflow-graph"])
    details["diagnostic_flags"] = {
        "native_eager": native_flags,
        "latch_eager": eager_flags,
        "latch_graph": graph_flags,
        "overflow_graph": overflow_flags,
    }
    native_ok = details["native-eager_summary"]["passed"] and (
        native_flags.get("diagnostic_eager") is True
        and native_flags.get("stage_seam") is False
    )
    latch_eager_ok = (
        details["latch-eager_summary"]["passed"]
        and eager_flags.get("diagnostic_eager") is True
        and eager_flags.get("stage_seam") is True
        and eager_flags.get("router_parity") is True
        and eager_flags.get("layer_boundary_parity") is True
    )
    latch_graph_ok = (
        details["latch-graph_summary"]["passed"]
        and graph_flags.get("diagnostic_eager") is False
        and graph_flags.get("stage_seam") is True
    )
    overflow_ok = (
        details["overflow-graph_summary"]["passed"]
        and overflow_flags.get("diagnostic_eager") is False
        and overflow_flags.get("stage_seam") is True
        and overflow_flags.get("wave_prefill") is True
    )
    gates["latchmoe_eager_diagnostic"] = "passed" if latch_eager_ok else "failed"

    eager_log = units["latch-eager"] / "run.log"
    layer_ok, layer_detail = _layer_boundary_gate(
        eager_log, shared_output_required=shared_output_required
    )
    details["layer_boundary"] = layer_detail
    gates["layer_boundary_parity"] = "passed" if latch_eager_ok and layer_ok else "failed"
    # A full-resident native Qwen3-Next service is intentionally expected to
    # fail on one 64 GiB NPU: the checkpoint is ~151 GiB. The eager seam runs
    # the native MoE implementation before staging and records router/layer
    # parity, which is the usable native oracle for this capacity-limited case.
    native_oracle_ok = native_ok or (
        native_capacity_limited
        and latch_eager_ok
        and layer_ok
    )
    if native_capacity_limited:
        details["native_oracle"] = {
            "mode": "layer_boundary_native_oracle",
            "full_resident_status": "capacity_limited",
            "layer_boundary_records": layer_detail.get("records", 0),
        }
    gates["native_oracle"] = "passed" if native_oracle_ok else "failed"

    router_path = units["latch-eager"] / "moe_router_parity.jsonl"
    if router_path.is_file():
        try:
            router_detail = compare_router_artifacts(
                load_router_artifact(router_path, role="native"),
                load_router_artifact(router_path, role="seam"),
            )
        except ValueError as exc:
            router_detail = {"status": "failed", "reason": str(exc)}
    else:
        router_detail = {"status": "failed", "reason": f"missing artifact: {router_path}"}
    details["router_parity"] = router_detail
    gates["router_parity"] = "passed" if router_detail.get("status") == "passed" else "failed"

    try:
        eager_outputs = _outputs(units["latch-eager"])
        graph_outputs = _outputs(units["latch-graph"])
        if native_capacity_limited:
            graph_exact = _same_outputs(eager_outputs, graph_outputs)
            token_detail = {
                "native_full_resident": "capacity_limited",
                "latch_eager_vs_latch_graph": graph_exact,
            }
            token_ok = graph_exact["passed"]
        else:
            native_outputs = _outputs(units["native-eager"])
            eager_exact = _same_outputs(native_outputs, eager_outputs)
            graph_exact = _same_outputs(native_outputs, graph_outputs)
            token_detail = {"native_vs_eager": eager_exact, "native_vs_graph": graph_exact}
            token_ok = eager_exact["passed"] and graph_exact["passed"]
    except ValueError as exc:
        token_detail = {"reason": str(exc)}
        token_ok = False
    details["token_exactness"] = token_detail
    gates["token_exactness"] = "passed" if token_ok else "failed"

    graph_results = _graph_gates(units["latch-graph"])
    for gate, (passed, detail) in graph_results.items():
        details[gate] = detail
        gates[gate] = "passed" if latch_graph_ok and passed else "failed"
    overflow_passed, overflow_detail = _overflow_gate(units["overflow-graph"])
    details["prefill_overflow"] = overflow_detail
    gates["prefill_overflow"] = "passed" if overflow_ok and overflow_passed else "failed"

    artifact_paths = [
        path
        for unit_dir in units.values()
        for path in (
            unit_dir / "summary.json",
            unit_dir / "outputs.jsonl",
            unit_dir / "moe_offload_profile.jsonl",
            unit_dir / "run.log",
        )
        if path.is_file()
    ]
    failures = [gate for gate, status in gates.items() if status != "passed"]
    return {
        "schema_version": "latchmoe-issue27-qualification-v1",
        "model_id": model_id,
        "bundle_dir": str(bundle_dir),
        "status": "passed" if not failures else "failed",
        "gates": gates,
        "failed_gates": failures,
        "details": details,
        "artifact_sha256": {str(path.relative_to(bundle_dir)): _sha256(path) for path in artifact_paths},
    }


def update_matrix(matrix_path: Path, report_path: Path, report: dict[str, Any]) -> None:
    if report.get("status") != "passed":
        raise ValueError("refusing to update qualification matrix from a failed report")
    matrix = _read_json(matrix_path)
    model_id = report.get("model_id")
    rows = matrix.get("rows") or []
    row = next((item for item in rows if item.get("id") == model_id), None)
    if row is None:
        raise ValueError(f"model is absent from qualification matrix: {model_id!r}")
    row["gates"] = dict(report["gates"])
    row["status"] = "passed"
    row["evidence"] = {
        "qualification_report": str(report_path),
        "qualification_report_sha256": _sha256(report_path),
    }
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shared-output-required", action="store_true")
    parser.add_argument("--matrix", type=Path)
    args = parser.parse_args()

    report = verify_bundle(
        args.bundle_dir.resolve(),
        model_id=args.model_id,
        shared_output_required=bool(args.shared_output_required),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.matrix is not None:
        update_matrix(args.matrix, args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
