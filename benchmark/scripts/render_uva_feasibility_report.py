#!/usr/bin/env python3
"""Render a paper-ready Ascend-UVA-like feasibility report from E0 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_DIR = Path("benchmark/artifacts/reports/ascend_uva_feasibility")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value)
    if text.lower() == "true":
        return "yes"
    if text.lower() == "false":
        return "no"
    return text


def _fmt_float(value: Any, digits: int = 2) -> str:
    if value in {"", None}:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _short_note(note: str, limit: int = 90) -> str:
    text = " ".join((note or "").split())
    if len(text) <= limit:
        return text or "-"
    return text[: limit - 3] + "..."


def _row_by_op(rows: list[dict[str, str]], operation: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("operation") == operation:
            return row
    return None


def render(summary_rows: list[dict[str, str]], verdict: dict[str, Any], runner: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Ascend UVA-like Feasibility Report")
    lines.append("")
    lines.append("This report is generated from real E0 probe artifacts. It should be regenerated after rerunning the UVA feasibility suite.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Verdict | `{verdict.get('verdict')}` |")
    lines.append(f"| Comparison to SEW | `{verdict.get('comparison_to_sew')}` |")
    lines.append(f"| Primary blocker | `{verdict.get('primary_blocker')}` |")
    lines.append(f"| Offload budget | {verdict.get('offload_budget_gb')} GiB |")
    lines.append("")
    lines.append("Allowed claim:")
    lines.append("")
    lines.append(f"> {verdict.get('allowed_claim')}")
    lines.append("")

    lines.append("## Runner Summary")
    lines.append("")
    runner_summary = runner.get("summary", {})
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Runner status | `{runner_summary.get('status')}` |")
    lines.append(f"| Expected verdict | `{runner_summary.get('expected_verdict')}` |")
    lines.append(f"| Observed verdict | `{runner_summary.get('verdict')}` |")
    lines.append(f"| Verdict matches | {_fmt_bool(runner_summary.get('verdict_matches'))} |")
    lines.append("")
    lines.append("| Command | Return code | Runner status | Expected nonzero |")
    lines.append("|---|---:|---|---|")
    for record in runner.get("records", []):
        lines.append(
            f"| `{record.get('name')}` | {record.get('returncode', '-')} | "
            f"`{record.get('status')}` | {_fmt_bool(record.get('expected_nonzero'))} |"
        )
    lines.append("")

    lines.append("## Gate Evidence")
    lines.append("")
    lines.append("| Gate | Operation | OK | Status | Size MiB | Avg ms | Bandwidth GiB/s | Relative to HBM | Note |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---|")
    for row in summary_rows:
        lines.append(
            f"| `{row.get('gate')}` | `{row.get('operation')}` | {_fmt_bool(row.get('ok'))} | "
            f"`{row.get('status')}` | {row.get('size_mib') or '-'} | {_fmt_float(row.get('avg_ms'))} | "
            f"{_fmt_float(row.get('approx_source_read_gib_s'))} | {_fmt_float(row.get('relative_to_hbm'), 4)} | "
            f"{_short_note(row.get('note', ''))} |"
        )
    lines.append("")

    host64 = _row_by_op(summary_rows, "host_registered_add_64MiB")
    hbm64 = _row_by_op(summary_rows, "hbm_add_64MiB")
    host256 = _row_by_op(summary_rows, "host_registered_add_256MiB")
    hbm256 = _row_by_op(summary_rows, "hbm_add_256MiB")
    matmul2 = _row_by_op(summary_rows, "host_registered_weight_matmul_m16_k1024_n1024")
    matmul32 = _row_by_op(summary_rows, "host_registered_weight_matmul_m16_k4096_n4096")
    lines.append("## Paper-Ready Interpretation")
    lines.append("")
    lines.append("- The 14 GiB host expert store can be registered through legacy `aclrtHostRegister`, so the runtime mapping layer is not the blocker.")
    if host64 and hbm64 and host256 and hbm256:
        lines.append(
            "- Simple host-registered elementwise reads work but are slow: "
            f"{_fmt_float(host64.get('approx_source_read_gib_s'))} GiB/s vs "
            f"{_fmt_float(hbm64.get('approx_source_read_gib_s'))} GiB/s at 64 MiB, and "
            f"{_fmt_float(host256.get('approx_source_read_gib_s'))} GiB/s vs "
            f"{_fmt_float(hbm256.get('approx_source_read_gib_s'))} GiB/s at 256 MiB."
        )
    lines.append("- Simple `torch.npu.NPUGraph` replay observes host-side content updates, so graph replay alone is not the first hard blocker.")
    if matmul2 and matmul32:
        lines.append(
            "- Host-registered matmul weights fail for both 2 MiB and 32 MiB weight tiles with `507057`, while HBM matmul references pass."
        )
    lines.append(
        "- Therefore, the fair SEW comparison is a compatibility-failure baseline: direct UVA-like expert matmul is not runnable, whereas SEW stages experts into HBM fixed slots before grouped MLP."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/ascend_uva_like_report.md"))
    args = parser.parse_args()

    summary_rows = _load_rows(args.artifact_dir / "e0_ascend_uva_like_summary.csv")
    verdict = _load_json(args.artifact_dir / "e0_ascend_uva_like_verdict.json")
    runner = _load_json(args.artifact_dir / "e0_runner_manifest.json")
    report = render(summary_rows, verdict, runner)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
