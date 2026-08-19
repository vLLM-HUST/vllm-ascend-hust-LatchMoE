# LatchMoE Evidence Map

This working file records the evidence boundary for the post-Introduction
draft. It is not part of the manuscript.

| ID | Source | Level | Supports | Cannot support | Planned use | Risk |
|---|---|---|---|---|---|---|
| E1 | `paper/sections/00_intro.tex` | L1 user manuscript | problem framing, key insight, three design challenges, claimed contributions | implementation details or results not stated elsewhere | all sections, terminology alignment | author must verify the retained framing |
| E2 | `vllm_moe_offload_ascend/moe_offload/slot_bank.py` and `slot_mapping.py` | L1 implementation | fixed allocations, owner/version leases, slot states, logical-to-physical mapping, fail-closed checks | online performance | Design, Implementation | code describes the current branch only |
| E3 | `vllm_moe_offload_ascend/moe_offload/transfer_engine.py` and `runtime.py` | L1 implementation | asynchronous H2D events, consumer dependency, readiness publication, compute protection | measured overlap benefit | Design, Implementation | mechanism evidence, not performance evidence |
| E4 | `vllm_moe_offload_ascend/moe_offload/phase_split.py` and `pipeline.py` | L1 implementation | bounded wave planning, policy-neutral routed-expert ordering | exact online equivalence or latency improvement | Design, Implementation | requires E8/E9 for empirical claims |
| E5 | `vllm_moe_offload_ascend/moe_offload/cpu_first_loader.py`, `autoconfig.py`, and `config.py` | L1 implementation | CPU-first loading, capacity configuration, supported runtime controls | universal memory feasibility | Implementation | configuration names retain legacy `SEW` identifiers |
| E6 | `tests/` | L1 executable tests | host-side lifecycle, mapping, wave, launcher, and artifact contracts | NPU graph performance | Implementation, Evaluation methodology | full collection requires the pinned external runtime stack |
| E7 | `docs/evidence/issue-7-graph-lifecycle.md` and its checked-in bundle | L1 data and logs | three independent graph-only starts; exact tokens; 12/12 stable layer pointers; H2D ordering; bounded waves; release; scoped TTFT/TPOT | broad performance superiority | Evaluation Q1 | one model, one NPU, TP1, concurrency 1 |
| E8 | `docs/evidence/issue-13-multi-wave-prefill.md` | L1 data and logs | native recombination equivalence; issued-before-compute schedule; stage wait; managed memory accounting | causal latency gain from overlap alone | Evaluation Q3 | stability campaign is not the matched A/B |
| E9 | `docs/evidence/issue-17-matched-ttft.md` and its checked-in bundle | L1 data and logs | three matched full-layer/multi-wave pairs; 35.21% TTFT-p50 and 22.96% TTFT-p95 reductions; exact tokens; zero fallback | other models, devices, budgets, concurrency, prefix cache, or decode speedup | Abstract, Introduction, Evaluation Q2, Conclusion | narrow qualified boundary |
| E10 | `benchmark/configs/sew_offload_v1.yaml` and `benchmark/scenarios/sew_offload_scenarios.json` | L1 experiment plan | planned graph-only baseline/workload/capacity/ablation matrix | results for cells that have not run | Chapter blueprint, Discussion | planning data must not be written as results |
| E11 | `paper/references.bib` and citations already used by the Introduction | L3 metadata plus user manuscript context | citation-level placement of related systems and prior directions | unverified method details or numerical comparisons | Background, Related Work | independently verified; authors must still confirm their own reading |
| E12 | `paper/figures/graph_breakdown.png` | L1 user figure | the four displayed eager/graph latency totals and qualitative breakdown | exact unlabeled stack values or vector quality | Figure 1 in Introduction | raster source; original plotting data absent |
| E13 | `benchmark/artifacts/motivation-fullrate-20260818/sew-offload-ascend-v1-20260818T145649Z/sew_14gb_autoslots/mixed_chat/{moe_profile.jsonl,server.log,benchmark.json,unit_manifest.json,release_ack.json}`, `paper/scripts/analyze_motivation_profile.py`, and `paper/data/motivation_profile_summary.json` | L1 raw profile, run metadata, and reproducible derived analysis | 200-request routing characterization: prefill active-set overflow and full-rate decode cache misses and logical-to-slot mapping-entry rewrites for Qwen3-30B-A3B with 32 slots per managed layer | other models, capacities, policies, workloads, or concurrency | Motivation | raw 315-MB profile is locally retained and ignored by Git; include it in the anonymous artifact before submission |

## Contribution Trace

| Contribution | Design/implementation location | Experiment or figure | Status |
|---|---|---|---|
| Address-stable expert-slot abstraction | Sections 3.2 and 4.2 | Issue 7 pointer and replay gates | verified within E7 scope |
| Replay-boundary staging with safe publication | Sections 3.3 and 4.3 | Issue 7 H2D ordering gates | verified within E7 scope |
| Versioned compute-protected slot reuse | Sections 3.4 and 4.3 | Issue 7 lease checks; host fault injection | verified within E7 scope |
| Exact capacity-bounded multi-wave execution | Sections 3.5 and 4.4 | Issue 13 exactness; Issue 17 matched A/B | verified within E8/E9 scope |
| Broad service-performance improvement | none | planned matrix only | missing; not claimed in the draft |
