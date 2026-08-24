#!/usr/bin/env python3
"""Extract and verify the portable LatchMoE raw-artifact bundle.

This wrapper deliberately writes all derived audit output below the caller's
work directory. It never overwrites the checked-in compact summaries or audit
artifacts, so a reviewer can inspect the independent result files directly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _find_unit(root: Path, marker: str, leaf: str) -> Path:
    candidates = sorted(
        path for path in root.rglob(leaf) if marker in str(path) and path.is_dir()
    )
    if len(candidates) != 1:
        raise RuntimeError(f"expected one {marker}/{leaf} unit under {root}, found {candidates}")
    return candidates[0]


def _run(repo: Path, script: str, args: list[str], output: Path) -> dict[str, Any]:
    command = [sys.executable, str(repo / "paper/scripts" / script), *args]
    result = subprocess.run(command, cwd=repo, check=False)
    payload: dict[str, Any] = {"returncode": result.returncode, "output": str(output)}
    if output.is_file():
        try:
            payload["status"] = json.loads(output.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            payload["status"] = "invalid_output"
    else:
        payload["status"] = "missing_output"
    return payload


def _motivation_config(repo: Path, raw: Path, work: Path) -> Path:
    specs = [
        (
            "Qwen3-30B-A3B",
            "motivation_profile_summary.json",
            raw / "latchmoe-motivation-qwen3-fullrate-v2"
            / "sew-offload-ascend-v1-20260822T030434Z"
            / "sew_14gb_autoslots/mixed_chat",
            12,
            200,
            9437184,
            ["--client-concurrency", "1", "--max-num-seqs", "1", "--routing-profile"],
        ),
        (
            "GLM-4.7-Flash",
            "motivation_glm_profile_summary.json",
            raw / "latchmoe-motivation-glm-fullrate-v2"
            / "sew-offload-ascend-v1-20260822T034632Z"
            / "sew_14gb_autoslots/mixed_chat",
            11,
            100,
            18874368,
            ["--client-concurrency", "1", "--max-num-seqs", "1", "--routing-profile"],
        ),
        (
            "Qwen3-Next-80B-A3B-Instruct",
            "motivation_qwen3_next_profile_summary.json",
            raw / "latchmoe-motivation-qwen3-next-profile64"
            / "sew-offload-ascend-v1-20260822T050040Z"
            / "sew_28gb_autoslots/mixed_chat",
            48,
            64,
            6291456,
            [
                "--client-concurrency",
                "1",
                "--max-num-seqs",
                "1",
                "--max-requests",
                "64",
                "--request-output-tokens",
                "32",
                "--routing-profile",
            ],
        ),
    ]
    datasets = []
    summary_dir = work / "motivation_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for name, summary_name, artifact_dir, layers, requests, expert_bytes, argv in specs:
        summary = json.loads((repo / "paper/data" / summary_name).read_text(encoding="utf-8"))
        summary["source"]["profile"] = str(artifact_dir / "moe_profile.jsonl")
        summary_path = summary_dir / summary_name
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        datasets.append(
            {
                "name": name,
                "summary": str(summary_path),
                "artifact_dir": str(artifact_dir),
                "managed_layers": layers,
                "requests": requests,
                "skipped_requests": 1,
                "expert_bytes": expert_bytes,
                "required_argv": argv,
            }
        )
    config = work / "motivation_sources.json"
    config.write_text(json.dumps({"schema_version": 1, "datasets": datasets}, indent=2) + "\n")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    work = (args.workdir or Path(tempfile.mkdtemp(prefix="latchmoe-artifact-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive_path = args.archive.resolve()
    # GNU tar keeps this wrapper usable on Python versions whose stdlib
    # tarfile module predates zstd support. The archive was produced with the
    # same command, so extraction is deterministic and does not require a
    # Python zstandard package.
    subprocess.run(
        ["tar", "--zstd", "-xf", str(archive_path), "-C", str(raw)],
        check=True,
    )

    formal = raw / "latchmoe-formal-20260824"
    issue17 = repo / "paper/data/issue17_audit.json"
    workload = repo / "benchmark/artifacts/workloads/issue13_sharegpt.jsonl"
    outputs = work / "verification_results"
    outputs.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    results["formal_campaigns"] = _run(
        repo,
        "audit_formal_campaigns.py",
        [
            "--baseline", str(formal / "baseline-matched-r3/baseline_summary.json"),
            "--overlap", str(formal / "overlap-abba-r2/overlap_summary.json"),
            "--capacity", str(formal / "capacity-r1/capacity_summary.json"),
            "--issue17", str(issue17),
            "--workload-manifest", str(workload),
            "--output", str(outputs / "formal_campaigns.json"),
        ],
        outputs / "formal_campaigns.json",
    )
    results["qualification_matrix"] = _run(
        repo,
        "audit_qualification_matrix.py",
        [
            "--matrix", str(repo / "benchmark/registry/qualification_matrix_v2.json"),
            "--registry", str(repo / "benchmark/registry/model_registry_v2.json"),
            "--output", str(outputs / "qualification_matrix.json"),
        ],
        outputs / "qualification_matrix.json",
    )
    results["capacity_sweep"] = _run(
        repo,
        "audit_capacity_sweep.py",
        ["--campaign", str(formal / "capacity-r1/campaign.json"), "--output", str(outputs / "capacity_sweep.json")],
        outputs / "capacity_sweep.json",
    )
    results["resource_ledgers"] = _run(
        repo,
        "audit_resource_ledgers.py",
        [
            "--qwen3-30b-a3b-root", str(formal / "qwen3-qualification-r1"),
            "--glm-4.7-flash-root", str(raw / "latchmoe-issue27-npu5/glm-phase-b-r13"),
            "--qwen3-next-80b-a3b-instruct-root", str(raw / "latchmoe-issue27-npu5/qwen3-next-gdn-vendor-r14"),
            "--output", str(outputs / "resource_ledgers.json"),
        ],
        outputs / "resource_ledgers.json",
    )
    results["motivation"] = _run(
        repo,
        "audit_motivation_characterization.py",
        ["--config", str(_motivation_config(repo, raw, work)), "--output", str(outputs / "motivation.json")],
        outputs / "motivation.json",
    )
    diagnostic = raw / "eager_graph_probe_20260824"
    if diagnostic.is_dir():
        results["eager_graph_diagnostic"] = _run(
            repo,
            "audit_eager_graph_diagnostic.py",
            [
                "--eager-dir", str(diagnostic / "eager_spawn3"),
                "--graph-dir", str(diagnostic / "graph_spawn"),
                "--output", str(outputs / "eager_graph_diagnostic.json"),
            ],
            outputs / "eager_graph_diagnostic.json",
        )

    overall = all(item["returncode"] == 0 for item in results.values())
    report = {
        "schema_version": "latchmoe-portable-artifact-verification-v1",
        "status": "passed" if overall else "failed",
        "archive": str(args.archive.resolve()),
        "workdir": str(work),
        "results": results,
    }
    report_path = work / "artifact_verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
