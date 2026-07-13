# SEW-Offload CCF-A Experiment Plan

Date: 2026-07-08

## Positioning

SEW-Offload should be evaluated as a **single-card, memory-constrained, graph-compatible MoE expert offload system on Ascend**, not as a generic cloud LLM serving scheduler and not as an offline large-batch throughput engine.

The closest experiment philosophy comes from MoE offload papers such as MoE-Infinity, Fiddler, FineMoE/fMoE, HOBBIT, FluxMoE, and MoE-Lightning. General LLM serving papers such as vLLM/PagedAttention, Sarathi-Serve, and Splitwise are useful for metric design and trace realism, but their request-rate sweep should be secondary for this project.

## Related-Work Lessons

| Paper family | What they usually prove | Useful lesson for SEW-Offload |
|---|---|---|
| vLLM / PagedAttention | Real prompt/output length distributions, request-rate sweep, latency-throughput frontier | Use ShareGPT and p50/p90/p99 TTFT/TPOT; do not make cloud request-rate sweep the main claim. |
| FlexGen | Explicit memory constraint, fixed input/output lengths such as 512/32, throughput under offload | Always report resource constraint and feasibility together with throughput. |
| MoE-Infinity | Personal-machine MoE offload, often batch size 1, expert movement/caching behavior | Batch/concurrency 1 is defensible for local memory-constrained MoE if explained clearly. |
| Fiddler | Local single-batch MoE inference, length sweep, CPU-GPU orchestration | Separate decode-heavy, balanced, prefill-heavy, and long-prefill regimes. |
| FineMoE / fMoE | TTFT, TPOT, expert hit rate, cache-limit sensitivity, batch-size sensitivity | We must report slot hit/miss, active experts, H2D bytes, and slot-budget sensitivity, not only TPOT. |
| MoE-Lightning | High-throughput batch inference on memory-constrained GPUs | Useful for generation-length sweep, but not the right main baseline because SEW targets low-concurrency graph replay. |
| HOBBIT / CommitMoE | Mixed precision or approximate expert selection, speed plus quality preservation | Since SEW should not change model semantics, run output/logit equivalence instead of full accuracy benchmarks unless needed. |
| FluxMoE | Expert paging, stable virtual addresses, memory for KV | Closest conceptual neighbor; SEW must emphasize Ascend ACLGraph replay, fixed physical slots, and B2 overflow. |
| Sarathi-Serve / Splitwise | Phase-aware metrics, TTFT/TBT, production/trace realism | Use as secondary motivation for prefill/decode separation and tail latency metrics. |

## Paper Claims and Required Evidence

| Claim | Reviewer question | Evidence needed | Dataset/workload | Baselines | Metrics | Status |
|---|---|---|---|---|---|---|
| C1. SEW enables MoE serving under tight NPU HBM by offloading experts. | Does the model actually require offload under the paper's memory budget? | No-offload capacity probe, SEW success under 14 GiB/28 GiB expert offload. | ShareGPT mixed, fixed-length buckets | no-offload default, SEW | success/OOM, peak HBM, KV size, CPU expert store | partial |
| C2. SEW is graph-replay-compatible while existing dynamic offload paths are not. | Why does this paper exist? | Legacy/native capture failure logs; SEW graph capture and replay logs; stable slot/log2phy evidence. | mixed_chat + small diagnostic | native prefetch, legacy/layered, SEW | graph capture status, failure reason, replay evidence | partial |
| C3. Graph replay matters quantitatively. | Is ACLGraph actually useful, or just engineering decoration? | SEW capture-on vs SEW eager under same offload; full-resident ACLGraph vs Eager upper-bound. | mixed_chat, decode-heavy | SEW eager, full-resident eager | TPOT, TTFT, output throughput | partial/done |
| C4. Slot-stable virtualization reduces offload overhead. | Why do fixed slots help? | Slot-count and offload-budget sweep; hit/miss and H2D analysis. | mixed_chat, decode-heavy | slot 8/16/32/64, 14/28 GiB | hit rate, H2D bytes, staging time, TPOT | planned |
| C5. B2 wave prefill is necessary for long prompts. | What happens when active experts exceed slots? | no-B2 vs B2 vs B2+transfer-aware on prefill-heavy/long-context. | prefill_heavy, long_context_prefill | SEW variants | TTFT, wave count, active expert count, H2D bytes | planned |
| C6. Compute-protected slot lifecycle is necessary for correctness. | Could slot reuse overwrite weights still being read? | stress/diagnostic with protection disabled or assertions; output/logit equivalence with protection enabled. | small deterministic prompt set + stress | protected vs unsafe diagnostic | mismatch/crash/assertion, exact output/logit diff | planned |
| C7. The result is robust enough for a paper, not a one-off demo. | Are results stable and reproducible? | 3 repeated runs, tail latency, hardware/process logs. | all main workloads | full system and key ablations | median, p90/p99, mean/std or min/max | planned |

## Required Experiment Stack

### E1. Main End-to-End Performance

Purpose: the main paper result.

Run:

| Workload | Prompt setting | Output | Requests | Why |
|---|---|---:|---:|---|
| `mixed_chat` | ShareGPT real distribution | 128 | 200 | Main realistic workload. |
| `decode_heavy` | short prompts, 64-256 | 256 | 128 | Decode hot path and TPOT. |
| `prefill_heavy` | 1024-2048 | 32 | 64 | TTFT and B2 prefill pressure. |

Methods:

| Method/case | Role |
|---|---|
| `no_offload_capacity_probe` | feasibility baseline; expected KV/HBM failure under default tight budget |
| `legacy_layered_14gb` / `legacy_layered_14gb_eager` | compatibility/failure evidence |
| `native_prefetch_14gb` / `native_prefetch_14gb_eager` | compatibility/failure evidence; be honest about CUDA-only failure on Ascend |
| `sew_14gb_capture_disabled` | same offload runtime, eager mode |
| `sew_14gb_autoslots` | main SEW result |
| `sew_28gb_slots32_capture_disabled` | same runtime, larger budget, eager |
| `sew_28gb_slots32` | main SEW larger-budget result |

Report:

- successful/failed requests
- TTFT p50/p90/p99
- TPOT p50/p90/p99
- output throughput
- peak HBM / available KV
- graph capture/replay status

### E2. Graph Efficiency Isolation

Purpose: separate "graph is faster" from "offload is different".

Already completed:

| Case | Result |
|---|---|
| full-resident ACLGraph, 512 MiB KV | 200/200, TTFT p50 200.22 ms, TPOT p50 33.23 ms/token, 28.45 tok/s |
| full-resident Eager, 512 MiB KV | 200/200, TTFT p50 322.19 ms, TPOT p50 147.35 ms/token, 6.64 tok/s |
| Gain | TPOT 4.43x lower, throughput 4.28x higher |

Also required:

| Pair | Why |
|---|---|
| `sew_14gb_autoslots` vs `sew_14gb_capture_disabled` | graph benefit under actual offload, already has first mixed_chat result |
| `sew_28gb_slots32` vs `sew_28gb_slots32_capture_disabled` | graph benefit under larger expert-residency budget |
| decode-heavy graph on/off | decode hot path isolates TPOT most cleanly |

### E3. Memory Feasibility and Cost Shifting

Purpose: show what memory is traded.

Table to fill:

| Case | Peak HBM | NPU weights | Slot bank | Host expert store | KV cache | Status |
|---|---:|---:|---:|---:|---:|---|
| no-offload default | TBD | 56.90 GB | 0 | 0 | failed | KV/HBM capacity failure |
| full-resident KV512M | TBD | 56.90 GB | 0 | 0 | 0.50 GiB | ok |
| SEW 14 GiB | TBD | 46.78 GB | 3.375 GiB | 13.50 GiB | 1.81 GiB observed | ok |
| SEW 28 GiB slots32 | TBD | TBD | TBD | TBD | TBD | planned/formal rerun |

This experiment prevents a reviewer from saying "maybe offload only wins because you changed memory pressure silently."

### E4. Slot Count and Offload Budget Sensitivity

Purpose: explain the latency-memory tradeoff.

Run matrix:

| Offload budget | Slot counts | Workloads |
|---|---|---|
| 14 GiB | 8, 16, 32 | mixed_chat, decode_heavy, prefill_heavy |
| 28 GiB | 32, 64 | mixed_chat, decode_heavy, prefill_heavy |

Report:

- TTFT/TPOT/throughput
- slot hit/miss rate
- H2D bytes per request
- staging time
- active expert count
- wave count
- available KV cache

Expected interpretation:

- too few slots: high miss rate and H2D overhead
- too many slots: KV pressure / possible capacity failure
- AutoConfig should land near the best safe point, not necessarily the largest slot count

### E5. B2 Wave Prefill and Transfer-Aware Scheduling

Purpose: prove the prefill mechanism, not just the decode path.

Run:

| Variant | Workloads | Reviewer question |
|---|---|---|
| `sew_14gb_no_b2` | prefill_heavy, long_context_prefill | Does fixed-slot staging fail or slow when active experts exceed slots? |
| `sew_14gb_no_transfer_aware` | prefill_heavy, long_context_prefill | Does transfer-aware scheduling matter beyond correctness? |
| `sew_14gb_autoslots` | prefill_heavy, long_context_prefill | Full B2 mechanism |

Report:

- TTFT p50/p90/p99
- max/mean active experts
- max/mean wave count
- H2D bytes
- prefill staging time
- failure reason if no-B2 fails

### E6. Correctness and Slot Lifecycle Safety

Purpose: systems reviewers will ask whether dynamic slot reuse is safe.

Minimum:

| Test | Method |
|---|---|
| Output equivalence | deterministic decoding on 32-64 prompts; compare SEW against full-resident no-offload when KV512M fits |
| Logit equivalence | if available, compare selected token logits or hidden outputs with tolerance |
| Lifecycle stress | intentionally high miss/eviction workload; verify no overwrite assertions and no output mismatch |
| Unsafe diagnostic | only if implemented safely: disable compute-protection events and show concrete assertion/crash/mismatch |

Do not run an unsafe performance benchmark as a main figure. Treat this as correctness evidence.

### E7. Controlled Fixed-Length Buckets

Purpose: make the experiment comparable to offloading literature.

Add or generate controlled buckets:

| Bucket | Input | Output | Related-work rationale |
|---|---:|---:|---|
| short-decode | 128 | 128 or 256 | HOBBIT/Fiddler-style small prompt/decode stress |
| standard-offload | 512 | 32 | FlexGen/MoE-Infinity anchor |
| balanced-paper | 512 | 128 | more stable TPOT for SEW |
| long-prefill | 2048 | 32 | Fiddler/Sarathi-style prefill stress |
| max-safe | 3800-4096 | 16 | B2 and KV edge |

Use real text from ShareGPT where possible. If synthetic padding is used, label it as controlled synthetic and keep it out of the headline table.

### E8. Concurrency Robustness, Not Cloud-Serving Contest

Purpose: show the runtime does not collapse when more than one route exists.

Recommended:

| Setting | Goal |
|---|---|
| concurrency 1 | main paper setting |
| concurrency 2 | robustness |
| concurrency 4 | stress/appendix if KV permits |

Use closed-loop concurrency first. Poisson request-rate sweeps can be future work unless the system becomes stable enough to support them without derailing the paper.

## Figure/Table Plan

| Figure/Table | Content | Main/Appendix |
|---|---|---|
| Figure 1 | Existing dynamic offload fails under ACLGraph vs SEW fixed-slot replay | main |
| Figure 2 | System overview: router, stage op, host store, slot bank, log2phy, captured MLP | main |
| Table 1 | Main ShareGPT mixed/decode/prefill performance | main |
| Figure 3 | SEW capture-on vs capture-off; full-resident ACLGraph vs Eager inset | main |
| Figure 4 | Memory feasibility and cost shifting | main |
| Figure 5 | Slot/budget sensitivity: TPOT + hit rate + KV capacity | main |
| Figure 6 | B2 prefill: active experts, wave count, TTFT | main |
| Table 2 | Baseline failure taxonomy with honest root causes | main |
| Table 3 | Correctness/equivalence checks | main or appendix |
| Appendix | fixed-length buckets, concurrency 2/4, repeated-run raw numbers | appendix |

## Execution Priority

| Priority | Experiment | Claim defended | Cost | Dependency | Stop condition |
|---|---|---|---|---|---|
| P0 | Repeat `mixed_chat`: SEW 14 capture-on/off, full-resident graph/eager if needed | C3, stability | high | existing configs | 3 valid runs each |
| P0 | Run `decode_heavy`: SEW 14 capture-on/off | C3 | medium | existing configs | 128/128 success |
| P0 | Run `prefill_heavy`: SEW 14 full/no-B2/no-transfer-aware | C5 | medium | existing configs | B2 evidence or failure logs |
| P1 | Run SEW 28 slots32 capture-on/off on mixed/decode/prefill | C1/C3 | high | NPU time | 3 workload results |
| P1 | Slot sweep 14 GiB: 8/16/32 | C4 | high | profile metrics | clear trend or failure boundary |
| P1 | Slot sweep 28 GiB: 32/64 | C4 | medium | profile metrics | safe point and unsafe point identified |
| P1 | Memory table extraction | C1 | low | logs/artifacts | all main rows filled |
| P1 | Correctness equivalence against full-resident KV512M | C6 | medium | deterministic decoding harness | no mismatch or bounded logit diff |
| P2 | Controlled fixed-length 512/32, 512/128, 2048/32, max-safe | comparability | medium | workload generation | one table complete |
| P2 | Concurrency 2/4 robustness | robustness | high | KV capacity | no crash or documented boundary |

## What Not To Do

1. Do not use smoke results as paper results.
2. Do not claim native prefetch is graph-incompatible if the real failure is CUDA-only API on Ascend. State the real root cause.
3. Do not compare SEW batch=1 against MoE-Lightning/FlexGen large-batch throughput as a headline.
4. Do not hide full-resident KV512M as "no offload always fits"; distinguish default tight-budget failure from manual-KV full-resident upper bound.
5. Do not invent quality numbers. If SEW does not approximate experts, correctness/equivalence is enough.

## Source URLs

- vLLM / PagedAttention: https://arxiv.org/abs/2309.06180
- FlexGen: https://arxiv.org/abs/2303.06865
- Sarathi-Serve: https://www.usenix.org/system/files/osdi24-agrawal.pdf
- Splitwise: https://arxiv.org/abs/2311.18677
- MoE-Infinity: https://arxiv.org/abs/2401.14361
- Fiddler: https://arxiv.org/abs/2402.07033
- FineMoE / fMoE: https://arxiv.org/abs/2502.05370
- HOBBIT: https://arxiv.org/abs/2411.01433
- MoE-Lightning: https://arxiv.org/abs/2411.11217
- CommitMoE: https://ojs.aaai.org/index.php/AAAI/article/view/39454
- FluxMoE: https://arxiv.org/abs/2604.02715

