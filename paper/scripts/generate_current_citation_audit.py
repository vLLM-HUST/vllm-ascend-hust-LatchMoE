#!/usr/bin/env python3
"""Emit the current citation-audit ledger from the web-backed review record.

This is intentionally a small, deterministic aggregator: the substantive
existence/metadata/context judgements are recorded in ``evidence`` below and
the script derives citation sites, hashes, counts, and the machine-readable
ledger from the current manuscript.  It does not touch any TeX source.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
SECTIONS = PAPER / "sections"
RUN = "2026-08-24_run02"

# Current primary-source review record.  URLs are primary landing pages or
# official proceedings/documentation pages used during the web-backed review.
# Occurrence verdicts are deliberately explicit: a citation can support one
# use (e.g., Related Work) while being wrong for another use in a frozen
# paragraph.
EVIDENCE = {
    "shazeer2017outrageously": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://arxiv.org/abs/1701.06538", "https://openreview.net/forum?id=B1ckMDqlg"],
        "note": "Real paper; current metadata and MoE routing uses are supported.",
    },
    "fedus2022switch": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://www.jmlr.org/papers/v23/21-0998.html"],
        "note": "Official JMLR record confirms title, authors, 2022 volume 23(120), pages, and sparse routing context.",
    },
    "jiang2024mixtral": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://arxiv.org/abs/2401.04088"],
        "note": "Official arXiv record confirms title, authors, year, and token-wise top-k expert routing context.",
    },
    "dai2024deepseekmoe": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://aclanthology.org/2024.acl-long.70/"],
        "note": "Current ACL proceedings metadata matches the local entry and supports the MoE architecture/routing uses.",
    },
    "du2022glam": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://proceedings.mlr.press/v162/du22c.html"],
        "note": "Official PMLR record confirms metadata and sparse MoE capacity/compute context.",
    },
    "rajbhandari2022deepspeedmoe": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://proceedings.mlr.press/v162/rajbhandari22a.html"],
        "note": "Official PMLR record confirms metadata and MoE inference/parallelism context.",
    },
    "hwang2024pregated": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://www.iscaconf.org/isca2024/program/", "https://doi.org/10.1109/ISCA59077.2024.00078"],
        "note": "Official ISCA program/DOI identify the paper; its pre-gating and CPU--GPU offload discussion supports the cited uses.",
    },
    "xue2024moeinfinity": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://arxiv.org/html/2401.14361v3", "https://github.com/EfficientMoE/MoE-Infinity/#citation"],
        "note": "The current v3 HTML has six authors, including Chuanhao Sun; local metadata now matches and the cache/prefetch/offload uses are supported.",
    },
    "li2026commitmoe": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://ojs.aaai.org/index.php/AAAI/article/view/39454"],
        "note": "Official AAAI record confirms volume 40 issue 27, pages, authors, year, DOI, and offloading/prediction context.",
    },
    "song2024promoe": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://arxiv.org/abs/2410.22134", "https://dblp.org/rec/journals/corr/abs-2410-22134.html"],
        "note": "Current arXiv record confirms the four-author 2024 preprint and proactive caching context.",
    },
    "tang2024hobbit": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://arxiv.org/abs/2411.01433", "https://dblp.org/rec/journals/corr/abs-2411-01433.html"],
        "note": "Official arXiv/DBLP records confirm metadata and mixed-precision offloading/prefetch/cache context.",
    },
    "kong2024swapmoe": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://aclanthology.org/2024.acl-long.363/"],
        "note": "Official ACL record confirms metadata and tunable-memory expert mapping/swapping context.",
    },
    "cao2025moelightning": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://arxiv.org/abs/2411.11217", "https://doi.org/10.1145/3669940.3707267"],
        "note": "Current ASPLOS metadata is complete; the paper supports broad expert offloading, paged movement, and pipeline overlap. The Background cache sentence is a group citation, not a claim that this paper alone defines every cache semantic.",
    },
    "fang2025klotski": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://yuyue.github.io/res/paper/Klotski-ASPLOS2025.pdf", "https://arxiv.org/abs/2502.06888"],
        "note": "Current ASPLOS metadata is complete. The paper supports multi-batch expert-aware prefetch/overlap and qualitatively supports greater prefill working sets; the word 'substantially' remains a weak frozen wording choice, not a fabricated citation.",
    },
    "li2023lina": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://www.usenix.org/conference/atc23/presentation/li-jiamin"],
        "note": "Official USENIX record confirms metadata and distributed MoE scheduling/parallelism context.",
    },
    "vllmUVA": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://docs.vllm.ai/en/stable/api/vllm/model_executor/offloader/uva/"],
        "note": "Official vLLM documentation supports pinned host storage and UVA access semantics.",
    },
    "nvidiaCudaBestPractices": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/"],
        "note": "Official NVIDIA guide supports UVA and mapped pinned host memory behavior.",
    },
    "nvidiaCudaGraph": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/torch-integration.html"],
        "note": "Official NVIDIA integration guide supports graph-address lifetime and update/recapture semantics.",
    },
    "echo2026": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://www.usenix.org/conference/osdi26/presentation/liu-guangda"],
        "note": "Official USENIX OSDI record supports graph-friendly dynamic KV-cache eviction/recall and prefetch overlap.",
    },
    "grace2026": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://www.usenix.org/conference/osdi26/presentation/ghosh"],
        "note": "Official USENIX OSDI record supports compiler transformations that increase CUDA Graph coverage.",
    },
    "cloudmoe2026": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://www.usenix.org/conference/osdi26/presentation/wang-wenxin"],
        "note": "Official USENIX OSDI record supports CPU--GPU hybrid stream-loading and overlap for local MoE serving.",
    },
    "vtensor2024": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://arxiv.org/abs/2407.15309"],
        "note": "Official arXiv record supports virtual-memory-backed tensor management that decouples computation from physical-page management.",
    },
    "flexgen": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://proceedings.mlr.press/v202/sheng23a.html"],
        "note": "Official PMLR record confirms current ten-author metadata and heterogeneous GPU/CPU/disk placement context.",
    },
    "kwon2020nimble": {
        "verdict": "REPLACE", "axis_failures": ["CONTEXT"],
        "sources": ["https://proceedings.mlsys.org/paper_files/paper/2021/hash/5b47430e24a5a1f9fe21f0e8eb814131-Abstract.html"],
        "note": "Nimble is a dynamic-neural-network compiler/runtime, not a source for CUDA Graph capture/replay or host-launch-gap reduction. Its Related Work use is valid, but the frozen Introduction and Background graph-replay uses are wrong-context and require author approval before any edit.",
    },
    "ye2025flashinfer": {
        "verdict": "KEEP", "axis_failures": [],
        "sources": ["https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html"],
        "note": "Official MLSys record confirms current proceedings metadata and supports graph-compatible dynamic scheduling/workspace-address claims.",
    },
}


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def extract_sites():
    files = [PAPER / "main.tex", *sorted(SECTIONS.glob("*.tex"))]
    sites = []
    for path in files:
        rel = path.relative_to(PAPER).as_posix()
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            for match in re.finditer(r"\\cite(?:[A-Za-z]*)?\{([^}]*)\}", line):
                for key in (x.strip() for x in match.group(1).split(",")):
                    sites.append({"key": key, "file": rel, "line": line_no, "text": line.strip()})
    return sites


def main() -> None:
    sites = extract_sites()
    keys = sorted({s["key"] for s in sites})
    if set(keys) != set(EVIDENCE):
        raise SystemExit(f"evidence/citation key mismatch: {set(keys) ^ set(EVIDENCE)}")
    contexts = PAPER / ".aris/citation-audit/contexts.txt"
    contexts.write_text("\n".join(f"{s['key']}\t{s['file']}:{s['line']}\t{s['text']}" for s in sites) + "\n")

    per_entry = []
    for key in keys:
        ev = EVIDENCE[key]
        uses = []
        for s in [x for x in sites if x["key"] == key]:
            use_verdict = "WRONG" if key == "kwon2020nimble" and s["file"] in {"sections/00_intro.tex", "sections/01_background.tex"} else "SUPPORTS"
            use = {"file": s["file"], "line": s["line"], "verdict": use_verdict}
            if use_verdict == "WRONG":
                use["note"] = "Nimble does not establish CUDA Graph capture/replay or host-launch-gap reduction."
            if key == "fang2025klotski" and s["file"] == "sections/01_background.tex":
                use["verdict"] = "WEAK"
                use["note"] = "Supports the qualitative prefill working-set observation but not the exact quantifier 'substantially'."
            uses.append(use)
        per_entry.append({"key": key, "verdict": ev["verdict"], "axis_failures": ev["axis_failures"], "note": ev["note"], "uses": uses, "sources": ev["sources"]})

    counts = {v: sum(e["verdict"] == v for e in per_entry) for v in ("KEEP", "FIX", "REPLACE", "REMOVE")}
    input_hashes = {"references.bib": sha256(PAPER / "references.bib"), "main.tex": sha256(PAPER / "main.tex")}
    cited_files = sorted({s["file"] for s in sites})
    for rel in cited_files:
        input_hashes[rel] = sha256(PAPER / rel)
    input_hashes[".aris/citation-audit/contexts.txt"] = sha256(contexts)

    run_dir = PAPER / ".aris/traces/citation-audit" / RUN
    run_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "skill": "citation-audit", "run_id": RUN,
        "started_at": datetime.now(timezone.utc).isoformat(), "executor": "codex",
        "executor_model": "gpt-5.6-sol", "executor_family": "openai",
        "review_independence": "same-family", "acceptance_status": "provisional",
        "project_dir": str(PAPER), "protocol": "fresh web-backed primary-source review",
    }
    (run_dir / "run.meta.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    for entry in per_entry:
        lines = [f"# Current web-backed citation review: `{entry['key']}`", "", "Reviewer route: direct primary-source/web verification; same-family provisional.", "", f"**Existence:** YES. Sources: " + ", ".join(f"[{u}]({u})" for u in entry["sources"]) + ".", "", f"**Metadata:** {EVIDENCE[entry['key']]['note']}", "", "**Context:**"]
        for use in entry["uses"]:
            note = f" — {use['note']}" if "note" in use else ""
            lines.append(f"- `{use['file']}:{use['line']}`: **{use['verdict']}**{note}")
        lines += ["", f"**Verdict:** **{entry['verdict']}**.", ""]
        (run_dir / f"{entry['key']}.md").write_text("\n".join(lines))

    overall = "FAIL" if any(e["verdict"] in {"REPLACE", "REMOVE"} for e in per_entry) else ("WARN" if any(e["verdict"] == "FIX" for e in per_entry) else "PASS")
    reason = "wrong_context" if overall == "FAIL" else ("metadata_drift" if overall == "WARN" else "all_entries_keep")
    payload = {
        "audit_skill": "citation-audit", "verdict": overall, "reason_code": reason,
        "summary": "All 25 cited works are real and current metadata is verified; one frozen Nimble occurrence has wrong CUDA-Graph context.",
        "audited_input_hashes": input_hashes,
        "trace_path": f".aris/traces/citation-audit/{RUN}/",
        "thread_id": "/root/citation_audit_final",
        "executor_model": "codex-gpt-5.6-sol", "executor_family": "openai",
        "reviewer_model": "gpt-5.6-sol", "reviewer_family": "openai",
        "review_independence": "same-family", "acceptance_status": "provisional",
        "reviewer_reasoning": "xhigh", "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "details": {"total_entries": len(per_entry), "counts": counts, "existence_counts": {"real": len(per_entry), "unverifiable": 0, "nonexistent": 0}, "per_entry": per_entry},
    }
    (PAPER / "CITATION_AUDIT.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    md = ["# Citation Audit Report", "", "**Date**: 2026-08-24", "**Bib file**: references.bib", "**Total cited entries**: 25", "", "## Summary", "", "| Verdict | Count |", "|---|---:|", *[f"| {k} | {counts[k]} |" for k in ("KEEP", "FIX", "REPLACE", "REMOVE")], "", "## Result", "", "All 25 cited works were checked against current primary-source pages. Current metadata is clean after the previously recorded metadata repairs. The only remaining blocking finding is `kwon2020nimble` in the frozen CUDA-Graph sentences of Introduction and Background; its Related Work use is appropriate. No frozen section was modified.", "", "## Priority finding", "", "### REPLACE occurrence context: `kwon2020nimble`", "", "Nimble is a dynamic-neural-network compiler/runtime. It does not establish CUDA Graph capture/replay or host-launch-gap reduction. The uses at `sections/00_intro.tex:17` and `sections/01_background.tex:29` are wrong-context; the use at `sections/07_related_work.tex:28` supports Nimble's actual contribution. Record for author decision because both affected sections are frozen.", "", "## All-clean entries", "", "`" + "`, `".join(k for k in keys if k != "kwon2020nimble") + "`", ""]
    (PAPER / "CITATION_AUDIT.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
