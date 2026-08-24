#!/usr/bin/env python3
"""Build a digest manifest for the portable LatchMoE raw-artifact archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCES = {
    "formal_baseline": (
        Path("/workspace/latchmoe-formal-20260824/baseline-matched-r3"),
        "latchmoe-formal-20260824/baseline-matched-r3",
    ),
    "formal_overlap": (
        Path("/workspace/latchmoe-formal-20260824/overlap-abba-r2"),
        "latchmoe-formal-20260824/overlap-abba-r2",
    ),
    "formal_capacity": (
        Path("/workspace/latchmoe-formal-20260824/capacity-r1"),
        "latchmoe-formal-20260824/capacity-r1",
    ),
    "baseline_feasibility": (
        Path("/workspace/latchmoe-formal-20260824/baseline-smoke"),
        "latchmoe-formal-20260824/baseline-smoke",
    ),
    "qualification_qwen3": (
        Path("/workspace/latchmoe-formal-20260824/qwen3-qualification-r1"),
        "latchmoe-formal-20260824/qwen3-qualification-r1",
    ),
    "motivation_qwen3": (
        Path("/workspace/latchmoe-motivation-qwen3-fullrate-v2"),
        "latchmoe-motivation-qwen3-fullrate-v2",
    ),
    "motivation_glm": (
        Path("/workspace/latchmoe-motivation-glm-fullrate-v2"),
        "latchmoe-motivation-glm-fullrate-v2",
    ),
    "motivation_qwen3_next": (
        Path("/workspace/latchmoe-motivation-qwen3-next-profile64"),
        "latchmoe-motivation-qwen3-next-profile64",
    ),
    "qualification_glm": (
        Path("/workspace/latchmoe-issue27-npu5/glm-phase-b-r13"),
        "latchmoe-issue27-npu5/glm-phase-b-r13",
    ),
    "qualification_qwen3_next": (
        Path("/root/workspace/latchmoe-issue27-npu5/qwen3-next-gdn-vendor-r14"),
        "latchmoe-issue27-npu5/qwen3-next-gdn-vendor-r14",
    ),
    "eager_graph_diagnostic": (
        Path("/workspace/vllm-ascend-hust-LatchMoE/benchmark/artifacts/eager_graph_probe_20260824"),
        "eager_graph_probe_20260824",
    ),
}

AUXILIARY_FILES = {
    "issue17_bundle": (
        Path("/workspace/vllm-ascend-hust-LatchMoE/docs/evidence/bundles/issue-17-matched-ttft-31621de.tar.gz"),
        "docs/evidence/bundles/issue-17-matched-ttft-31621de.tar.gz",
    ),
    "issue7_bundle": (
        Path("/workspace/vllm-ascend-hust-LatchMoE/docs/evidence/bundles/issue-7-graph-lifecycle-743045d.tar.gz"),
        "docs/evidence/bundles/issue-7-graph-lifecycle-743045d.tar.gz",
    ),
    "workload_manifest": (
        Path("/workspace/vllm-ascend-hust-LatchMoE/benchmark/artifacts/workloads/issue13_sharegpt.jsonl"),
        "benchmark/artifacts/workloads/issue13_sharegpt.jsonl",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    files: list[dict[str, str | int]] = []
    missing: list[str] = []
    for source_id, (root, archive_prefix) in SOURCES.items():
        if not root.is_dir():
            missing.append(str(root))
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            files.append(
                {
                    "source_id": source_id,
                    "source_path": str(path),
                    "archive_path": f"{archive_prefix}/{path.relative_to(root)}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    for source_id, (path, archive_path) in AUXILIARY_FILES.items():
        if not path.is_file():
            missing.append(str(path))
            continue
        files.append(
            {
                "source_id": source_id,
                "source_path": str(path),
                "archive_path": archive_path,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    archive = args.archive.resolve()
    payload = {
        "schema_version": "latchmoe-portable-artifact-manifest-v1",
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size if archive.is_file() else None,
            "sha256": sha256(archive) if archive.is_file() else None,
        },
        "sources": {
            source_id: {"source_path": str(root), "archive_prefix": prefix}
            for source_id, (root, prefix) in SOURCES.items()
        },
        "auxiliary_files": {
            source_id: {"source_path": str(path), "archive_path": archive_path}
            for source_id, (path, archive_path) in AUXILIARY_FILES.items()
        },
        "missing_sources": missing,
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed" if not missing else "missing", "files": len(files), "output": str(args.output)}))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
