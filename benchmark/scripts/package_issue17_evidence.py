#!/usr/bin/env python3
"""Package a portable, verifier-ready Issue #17 evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

REQUIRED_UNIT_FILES = (
    "unit_manifest.json",
    "unit_result.json",
    "benchmark.json",
    "server.log",
    "client.log",
    "launcher_lifecycle.log",
    "moe_profile.jsonl",
    "npu_samples.jsonl",
    "release_ack.json",
    "issue17_verification.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(output_root: Path, destination: Path) -> None:
    output_root = output_root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    campaign = _read_json(output_root / "campaign.json")
    summary = _read_json(output_root / "matched_summary.json")
    if summary.get("status") != "passed" or not (output_root / "PASSED.txt").is_file():
        raise ValueError("campaign has not passed its final verifier")

    units = campaign.get("units") or []
    if len(units) != 6:
        raise ValueError(f"expected six campaign units, found {len(units)}")

    with tempfile.TemporaryDirectory(prefix="issue17-evidence-") as temp:
        root = Path(temp) / "issue-17-matched-ttft"
        root.mkdir()
        portable_units = []
        for item in units:
            stage = str(item["stage"])
            source = Path(item["unit_dir"])
            target = root / "units" / stage
            target.mkdir(parents=True)
            for name in REQUIRED_UNIT_FILES:
                artifact = source / name
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise ValueError(f"missing or empty artifact: {artifact}")
                shutil.copy2(artifact, target / name)
            portable_units.append(
                {
                    "stage": stage,
                    "arm": item["arm"],
                    "unit_dir": f"units/{stage}",
                    "runner_returncode": item["runner_returncode"],
                }
            )

        portable_campaign = {"order": campaign["order"], "units": portable_units}
        (root / "campaign.json").write_text(
            json.dumps(portable_campaign, indent=2) + "\n", encoding="utf-8"
        )
        shutil.copy2(output_root / "matched_summary.json", root / "matched_summary.original.json")
        shutil.copy2(output_root / "PASSED.txt", root / "PASSED.txt")

        provenance = summary["units"][0]["provenance"]
        workload = Path(provenance["dataset_manifest_path"])
        if _sha256(workload) != provenance["dataset_manifest_sha256"]:
            raise ValueError("workload manifest digest differs from campaign provenance")
        shutil.copy2(workload, root / "issue17_sharegpt_mixed_200.jsonl")

        metadata = {
            "campaign_status": summary["status"],
            "source_campaign_sha256": _sha256(output_root / "campaign.json"),
            "source_summary_sha256": _sha256(output_root / "matched_summary.json"),
            "workload_sha256": _sha256(workload),
            "units": [item["stage"] for item in units],
        }
        (root / "BUNDLE.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (root / "README.md").write_text(
            "# Issue #17 matched TTFT evidence\n\n"
            "Run the repository verifier from this directory. Verify each unit against "
            "`units/pair-1-full_layer/benchmark.json`, then run the campaign verifier "
            "against `campaign.json`. `SHA256SUMS` covers every packaged evidence file.\n",
            encoding="utf-8",
        )

        digest_lines = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                digest_lines.append(f"{_sha256(path)}  {path.relative_to(root)}")
        (root / "SHA256SUMS").write_text("\n".join(digest_lines) + "\n", encoding="utf-8")

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(root, arcname=root.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    package(args.output_root, args.destination)
    print(json.dumps({"bundle": str(args.destination.resolve()), "sha256": _sha256(args.destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
