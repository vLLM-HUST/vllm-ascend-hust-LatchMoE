#!/usr/bin/env python3
"""Fail-closed verifier for the immutable LatchMoE A1--A18 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import fitz


FROZEN = {
    "sections/00_intro.tex": "80962fc6c50741b9c2d3933f1af94a7ef13b585118d002da32e9ea31e4417679",
    "sections/01_background.tex": "f41cf8233d48c8345f3ba3fd0922afeffdafac3185e2675a32680aad91e8bf3e",
}
REQUIRED_AUDITS = (
    "PROOF_AUDIT.json",
    "PAPER_CLAIM_AUDIT.json",
    "CITATION_AUDIT.json",
    "KILL_ARGUMENT.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    paper = args.paper_dir.resolve()
    repo = paper.parent
    manifest_path = (args.manifest or paper / ".aris/acceptance_manifest.json").resolve()
    failures: list[str] = []

    for rel, expected in FROZEN.items():
        path = paper / rel
        if not path.is_file() or sha(path) != expected:
            failures.append(f"A1 frozen hash mismatch: {rel}")

    main_tex = (paper / "main.tex").read_text(encoding="utf-8")
    if "\\documentclass[sigplan,anonymous,review,nonacm]{acmart}" not in main_tex:
        failures.append("A2 official anonymous acmart class missing")
    forbidden = re.compile(r"\\(?:vspace|fontsize|geometry|linespread|enlargethispage)\b")
    for path in [paper / "main.tex", *sorted((paper / "sections").glob("*.tex"))]:
        if forbidden.search(path.read_text(encoding="utf-8")):
            failures.append(f"A2 forbidden layout command: {path.relative_to(paper)}")
    pdf = fitz.open(paper / "main.pdf")
    conclusion_pages = [index + 1 for index, page in enumerate(pdf) if "Conclusion" in page.get_text()]
    if not conclusion_pages or max(conclusion_pages) > 11:
        failures.append(f"A2 conclusion page is not within 11: {conclusion_pages}")
    # The table is included by the Evaluation section rather than main.tex.
    # Count the complete manuscript sources so the contract check reflects the
    # actual document and does not report a false duplicate/missing placement.
    manuscript_sources = [main_tex] + [
        path.read_text(encoding="utf-8") for path in sorted((paper / "sections").glob("*.tex"))
    ]
    qualification_inputs = sum(source.count("\\input{generated/qualification_table}") for source in manuscript_sources)
    if qualification_inputs != 1:
        failures.append("A17 qualification table is not placed exactly once")

    status_files = {
        "A4": (paper / "data/audits/motivation_reaudit.json", "verified"),
        "A10": (paper / "data/qualification_summary.json", "passed"),
        "A12": (paper / "data/formal_campaigns.json", "passed"),
        "A16": (paper / "data/resource_ledgers.json", "passed"),
    }
    for aid, (path, expected) in status_files.items():
        try:
            actual = json.loads(path.read_text(encoding="utf-8")).get("status")
        except Exception as error:
            failures.append(f"{aid} cannot parse {path.name}: {error}")
            continue
        if actual != expected:
            failures.append(f"{aid} status {actual!r}, expected {expected!r}")

    for name in REQUIRED_AUDITS:
        path = paper / name
        if not path.is_file():
            failures.append(f"A18 missing mandatory audit: {name}")
        else:
            verdict = json.loads(path.read_text(encoding="utf-8")).get("verdict")
            if verdict in {"FAIL", "BLOCKED", "ERROR", None}:
                failures.append(f"A18 blocking audit {name}: {verdict}")

    if not manifest_path.is_file():
        failures.append("A18 acceptance manifest missing")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for assertion in manifest.get("assertions", []):
            if assertion.get("status") != "pass":
                failures.append(f"manifest {assertion.get('id')} is {assertion.get('status')}")
            for artifact in assertion.get("artifacts", []):
                path = Path(artifact["path"])
                if not path.is_absolute():
                    path = repo / path
                if not path.is_file() or sha(path) != artifact.get("sha256"):
                    failures.append(f"stale artifact: {artifact['path']}")

    result = {
        "schema_version": "latchmoe-acceptance-check-v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "conclusion_pages": conclusion_pages,
        "pdf_pages_including_references": len(pdf),
    }
    output = args.json_out or paper / ".aris/acceptance-check-report.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
