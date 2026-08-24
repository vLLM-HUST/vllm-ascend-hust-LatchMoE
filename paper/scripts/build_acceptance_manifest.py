#!/usr/bin/env python3
"""Build a hash-pinned A1--A18 acceptance manifest for LatchMoE."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ARTIFACTS = {
    "A1": ["paper/sections/00_intro.tex", "paper/sections/01_background.tex", "paper/FROZEN_SECTION_SUGGESTIONS.md"],
    "A2": ["paper/main.tex", "paper/main.pdf"],
    "A3": ["paper/figures/render_latchmoe_architecture_compact.py", "paper/figures/latchmoe_architecture.pdf", "paper/figures/motivation_characterization.pdf", "paper/figures/baseline_mechanisms.pdf", "paper/figures/capacity_sensitivity.pdf", "paper/.aris/traces/figure_recheck.md"],
    "A4": ["paper/scripts/audit_motivation_characterization.py", "paper/data/audits/motivation_reaudit.json"],
    "A5": ["paper/data/audits/motivation_reaudit.json", "paper/sections/02_motivation.tex"],
    "A6": ["paper/data/audits/motivation_reaudit.json", "paper/figures/motivation_characterization.pdf"],
    "A7": ["paper/sections/03_design.tex", "paper/figures/latchmoe_architecture.pdf"],
    "A8": ["paper/data/qualification_summary.json", "paper/sections/03_design.tex"],
    "A9": ["paper/data/qualification_summary.json", "paper/figures/latchmoe_architecture.pdf"],
    "A10": ["paper/data/qualification_summary.json", "paper/generated/qualification_table.tex"],
    "A11": ["paper/data/qualification_summary.json", "paper/data/issue17_audit.json"],
    "A12": ["paper/data/formal_campaigns.json", "paper/sections/05_evaluation.tex"],
    "A13": ["paper/data/formal_campaigns.json", "paper/generated/formal_results_macros.tex"],
    "A14": ["paper/sections/05_evaluation.tex", "paper/sections/06_discussion.tex", "paper/sections/08_conclusion.tex"],
    "A15": ["paper/data/formal_campaigns.json", "paper/sections/05_evaluation.tex"],
    "A16": ["paper/data/resource_ledgers.json", "paper/generated/resource_table.tex", "paper/sections/05_evaluation.tex"],
    "A17": ["paper/main.tex", "paper/sections/05_evaluation.tex", "paper/generated/qualification_table.tex"],
    "A18": ["paper/CLAIM_LEDGER.md", "paper/EXPERIMENT_AUDIT.json", "paper/PROOF_AUDIT.json", "paper/PAPER_CLAIM_AUDIT.json", "paper/CITATION_AUDIT.json", "paper/KILL_ARGUMENT.json", "paper/.aris/audit-verifier-report.json"],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    paper = repo / "paper"
    missing: list[str] = []
    assertions = []
    for aid in [f"A{i}" for i in range(1, 19)]:
        rows = []
        for rel in ARTIFACTS[aid]:
            path = repo / rel
            if not path.is_file():
                missing.append(rel)
                continue
            rows.append({"path": rel, "sha256": sha(path)})
        assertions.append({
            "id": aid,
            "status": "pass" if len(rows) == len(ARTIFACTS[aid]) else "pending",
            "artifacts": rows,
        })
    checker = paper / "scripts/check_acceptance_contract.py"
    external = Path("/root/.codex/aris_repo/tools/verify_paper_audits.sh")
    manifest = {
        "schema_version": "latchmoe-acceptance-manifest-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not missing else "pending",
        "missing": missing,
        "contract": {"path": "PAPER_ACCEPTANCE_CONTRACT.md", "sha256": sha(repo / "PAPER_ACCEPTANCE_CONTRACT.md")},
        "plan": {"path": "PAPER_PLAN.md", "sha256": sha(repo / "PAPER_PLAN.md")},
        "assertions": assertions,
        "contract_checker": {
            "path": "paper/scripts/check_acceptance_contract.py",
            "sha256": sha(checker),
            "command": "python3 paper/scripts/check_acceptance_contract.py paper",
            "inputs": ["paper/.aris/acceptance_manifest.json", "paper/main.pdf", "paper/main.tex", "paper/sections", "paper/*_AUDIT.json"],
            "fail_criteria": "any A1--A18 artifact missing/stale, frozen hash mismatch, main text after page 11, forbidden layout command, or blocking mandatory audit",
        },
        "external_audit_verifier": {
            "path": str(external),
            "sha256": sha(external),
            "command": "bash /root/.codex/aris_repo/tools/verify_paper_audits.sh paper --assurance submission",
            "inputs": ["paper/PROOF_AUDIT.json", "paper/PAPER_CLAIM_AUDIT.json", "paper/CITATION_AUDIT.json", "paper/KILL_ARGUMENT.json"],
            "fail_criteria": "missing/invalid/stale mandatory audit or FAIL/BLOCKED/ERROR verdict",
        },
    }
    output = args.output or paper / ".aris/acceptance_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output": str(output), "missing": missing}))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
