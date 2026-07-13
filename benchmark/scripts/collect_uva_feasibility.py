#!/usr/bin/env python3
"""Aggregate Ascend-UVA-like feasibility probe artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def add_row(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    rows.append(
        {
            "gate": kwargs.get("gate", ""),
            "artifact": kwargs.get("artifact", ""),
            "status": kwargs.get("status", ""),
            "size_mib": kwargs.get("size_mib", ""),
            "operation": kwargs.get("operation", ""),
            "returncode": kwargs.get("returncode", ""),
            "ok": kwargs.get("ok", ""),
            "avg_ms": kwargs.get("avg_ms", ""),
            "approx_source_read_gib_s": kwargs.get("approx_source_read_gib_s", ""),
            "relative_to_hbm": kwargs.get("relative_to_hbm", ""),
            "note": kwargs.get("note", ""),
        }
    )


def collect_rows(d: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    runtime_14g = load(d / "probe_device4_14gb_register_only.json")
    if runtime_14g:
        reg = runtime_14g["steps"].get("host_register_legacy", {})
        add_row(
            rows,
            gate="U0_runtime_mapping",
            artifact="probe_device4_14gb_register_only.json",
            status=runtime_14g.get("status"),
            size_mib=runtime_14g.get("size_mib"),
            operation="aclrtHostRegister",
            ok=reg.get("ok"),
            note=f"device_ptr={reg.get('device_ptr')}; HostRegisterV2 ok={runtime_14g['steps'].get('host_register_v2', {}).get('ok')}",
        )

    wrap = load(d / "probe_device4_tensor_construct_uint8.json") or load(d / "probe_device4_tensor_wrap_1mib.json")
    if wrap:
        add_row(
            rows,
            gate="U0_framework_wrapping",
            artifact=Path(wrap.get("_artifact", "probe_device4_tensor_construct_uint8.json")).name,
            status=wrap.get("status"),
            size_mib=wrap.get("size_mib"),
            operation="torch_npu_private_tensor_wrap",
            ok=wrap.get("steps", {}).get("construct_tensor", {}).get("ok"),
            note=f"check_npu_data_ptr={wrap.get('steps', {}).get('construct_storage', {}).get('check_npu_data_ptr')}",
        )

    matrix = load(d / "probe_device4_tensor_access_matrix.json")
    if matrix:
        for job in matrix.get("jobs", []):
            child = load(Path(job["child_json"])) if job.get("child_json") else None
            child_try_op = (child or {}).get("steps", {}).get("try_op", {})
            op_ok = child_try_op.get("ok")
            if op_ok is None:
                op_ok = job.get("returncode") == 0
            add_row(
                rows,
                gate="U1_tensor_access_matrix",
                artifact=Path(job.get("child_json") or "probe_device4_tensor_access_matrix.json").name,
                status=job.get("status"),
                size_mib=matrix.get("size_mib"),
                operation=job.get("name"),
                returncode=job.get("returncode"),
                ok=op_ok,
                note=(
                    child_try_op.get("error", "")
                    or ("; ".join(job.get("stderr_tail", [])[-3:]) if job.get("stderr_tail") else "")
                ),
            )

    host64 = load(d / "probe_device4_tensor_add_float16_zero_64mib_timing.json")
    hbm64 = load(d / "probe_device4_hbm_add_float16_zero_64mib_timing.json")
    host256 = load(d / "probe_device4_tensor_add_float16_zero_256mib_timing.json")
    hbm256 = load(d / "probe_device4_hbm_add_float16_zero_256mib_timing.json")
    for label, host, hbm in [("64MiB", host64, hbm64), ("256MiB", host256, hbm256)]:
        if host:
            op = host.get("steps", {}).get("try_op", {})
            hbm_bw = hbm.get("approx_source_read_gib_s") if hbm else None
            host_bw = op.get("approx_source_read_gib_s")
            rel = (host_bw / hbm_bw) if host_bw and hbm_bw else ""
            add_row(
                rows,
                gate="U1_elementwise_read_bandwidth",
                artifact=f"probe_device4_tensor_add_float16_zero_{label.lower()}_timing.json",
                status=host.get("status"),
                size_mib=host.get("size_mib"),
                operation=f"host_registered_add_{label}",
                ok=op.get("ok") and op.get("allclose_to_one"),
                avg_ms=op.get("avg_ms"),
                approx_source_read_gib_s=host_bw,
                relative_to_hbm=rel,
                note="output allclose to one" if op.get("allclose_to_one") else "",
            )
        if hbm:
            add_row(
                rows,
                gate="U1_hbm_reference",
                artifact=f"probe_device4_hbm_add_float16_zero_{label.lower()}_timing.json",
                status=hbm.get("status"),
                size_mib=int(hbm.get("elements", 0)) * 2 // (1024 * 1024),
                operation=f"hbm_add_{label}",
                ok=hbm.get("allclose_to_one"),
                avg_ms=hbm.get("avg_ms"),
                approx_source_read_gib_s=hbm.get("approx_source_read_gib_s"),
                relative_to_hbm=1.0,
                note="HBM resident reference",
            )

    graph = load(d / "probe_device4_npugraph_replay_1mib_1024.json")
    if graph:
        replay = graph.get("steps", {}).get("replay", [])
        add_row(
            rows,
            gate="U2_npugraph_replay",
            artifact="probe_device4_npugraph_replay_1mib_1024.json",
            status=graph.get("status"),
            size_mib=graph.get("size_mib"),
            operation="npugraph_replay_host_update",
            ok=bool(replay) and all(item.get("ok") for item in replay),
            note="; ".join(
                f"host={item.get('host_value')} output={item.get('output_sample', [''])[0]}"
                for item in replay
            ),
        )

    for name in [
        "probe_device4_matmul_m16_k1024_n1024.json",
        "probe_device4_matmul_m16_k4096_n4096.json",
    ]:
        matmul = load(d / name)
        if not matmul:
            continue
        host = matmul.get("steps", {}).get("host_registered_matmul", {})
        hbm = matmul.get("steps", {}).get("hbm_matmul", {})
        shape = f"m{matmul.get('m')}_k{matmul.get('k')}_n{matmul.get('n')}"
        add_row(
            rows,
            gate="U3_moe_shaped_matmul",
            artifact=name,
            status=matmul.get("status"),
            size_mib=matmul.get("alloc_size_mib"),
            operation=f"host_registered_weight_matmul_{shape}",
            ok=host.get("ok"),
            avg_ms=host.get("avg_ms", ""),
            approx_source_read_gib_s=host.get("approx_weight_read_gib_s", ""),
            relative_to_hbm=matmul.get("relative_to_hbm", ""),
            note=host.get("error", ""),
        )
        add_row(
            rows,
            gate="U3_hbm_matmul_reference",
            artifact=name,
            status="ok" if hbm.get("ok") else "failed",
            size_mib=matmul.get("alloc_size_mib"),
            operation=f"hbm_weight_matmul_{shape}",
            ok=hbm.get("ok"),
            avg_ms=hbm.get("avg_ms", ""),
            approx_source_read_gib_s=hbm.get("approx_weight_read_gib_s", ""),
            relative_to_hbm=1.0 if hbm.get("ok") else "",
            note="HBM resident matmul reference" if hbm.get("ok") else hbm.get("error", ""),
        )

    return rows


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "ok"}
    return bool(value)


def derive_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the paper-facing E0 verdict from gate rows."""

    by_gate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_gate.setdefault(str(row.get("gate", "")), []).append(row)

    def any_ok(gate: str) -> bool:
        return any(_truthy(row.get("ok")) for row in by_gate.get(gate, []))

    def all_ok(gate: str) -> bool:
        rows_for_gate = by_gate.get(gate, [])
        return bool(rows_for_gate) and all(_truthy(row.get("ok")) for row in rows_for_gate)

    host_matmul_rows = [
        row
        for row in by_gate.get("U3_moe_shaped_matmul", [])
        if str(row.get("operation", "")).startswith("host_registered_weight_matmul")
    ]
    hbm_matmul_rows = by_gate.get("U3_hbm_matmul_reference", [])
    host_matmul_any_ok = any(_truthy(row.get("ok")) for row in host_matmul_rows)
    host_matmul_any_failed = any(not _truthy(row.get("ok")) for row in host_matmul_rows)
    hbm_matmul_all_ok = bool(hbm_matmul_rows) and all(_truthy(row.get("ok")) for row in hbm_matmul_rows)

    required_present = {
        "U0_runtime_mapping": bool(by_gate.get("U0_runtime_mapping")),
        "U0_framework_wrapping": bool(by_gate.get("U0_framework_wrapping")),
        "U1_tensor_access_matrix": bool(by_gate.get("U1_tensor_access_matrix")),
        "U2_npugraph_replay": bool(by_gate.get("U2_npugraph_replay")),
        "U3_moe_shaped_matmul": bool(host_matmul_rows),
    }

    if not all(required_present.values()):
        overall = "incomplete"
        comparison_to_sew = "not_applicable_missing_required_gates"
        primary_blocker = "missing_required_gate"
    elif host_matmul_any_failed and not host_matmul_any_ok and hbm_matmul_all_ok:
        overall = "not_viable_as_sew_baseline"
        comparison_to_sew = "compatibility_failure_not_latency_throughput_comparison"
        primary_blocker = "host_registered_matmul_weight_path_fails_507057"
    elif host_matmul_any_ok:
        overall = "requires_sew_performance_comparison"
        comparison_to_sew = "run_common_moe_shaped_or_vllm_harness"
        primary_blocker = "none_at_current_gate"
    else:
        overall = "inconclusive"
        comparison_to_sew = "not_applicable_until_u3_is_decisive"
        primary_blocker = "u3_inconclusive"

    failed_rows = [row for row in rows if not _truthy(row.get("ok"))]
    return {
        "verdict": overall,
        "comparison_to_sew": comparison_to_sew,
        "primary_blocker": primary_blocker,
        "offload_budget_gb": 14,
        "passed_gates": {
            "runtime_mapping_14gb": any_ok("U0_runtime_mapping"),
            "framework_tensor_wrapping": any_ok("U0_framework_wrapping"),
            "simple_ai_core_elementwise_read": any(
                _truthy(row.get("ok")) and row.get("operation") == "add_float16_zero"
                for row in by_gate.get("U1_tensor_access_matrix", [])
            ),
            "elementwise_bandwidth_reference": all_ok("U1_hbm_reference"),
            "simple_npugraph_replay": any_ok("U2_npugraph_replay"),
            "hbm_matmul_reference": hbm_matmul_all_ok,
        },
        "failed_gates": {
            "d2h_copy_or_sdma_copy": any(
                (not _truthy(row.get("ok")))
                and row.get("operation") in {"copy_uint8", "device_copy_uint8"}
                for row in by_gate.get("U1_tensor_access_matrix", [])
            ),
            "host_registered_matmul_weight_path": host_matmul_any_failed,
        },
        "required_present": required_present,
        "failure_evidence": [
            {
                "gate": row.get("gate"),
                "artifact": row.get("artifact"),
                "operation": row.get("operation"),
                "status": row.get("status"),
                "note": row.get("note"),
            }
            for row in failed_rows
        ],
        "allowed_claim": (
            "Ascend exposes a partial UVA-like remote-read path on this CANN 9.0/910B2 stack, "
            "but it is not a drop-in CUDA UVAOffloader port for MoE expert execution: simple "
            "elementwise reads and simple NPUGraph replay work, while copy/SDMA paths fail and "
            "host-registered matmul weights fail with 507057. SEW should be compared against this "
            "as a compatibility-failure baseline unless a lower-level grouped-MLP path is made runnable."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("benchmark/artifacts/reports/ascend_uva_feasibility"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark/artifacts/reports/ascend_uva_feasibility/e0_ascend_uva_like_summary.csv"),
    )
    parser.add_argument(
        "--verdict-out",
        type=Path,
        default=Path("benchmark/artifacts/reports/ascend_uva_feasibility/e0_ascend_uva_like_verdict.json"),
    )
    args = parser.parse_args()

    rows = collect_rows(args.artifact_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    verdict = derive_verdict(rows)
    args.verdict_out.parent.mkdir(parents=True, exist_ok=True)
    args.verdict_out.write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.out)
    print(args.verdict_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
