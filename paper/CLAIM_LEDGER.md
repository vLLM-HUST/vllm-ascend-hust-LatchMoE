# LatchMoE final quantitative claim ledger

This ledger maps every paper-facing measured value to a machine-readable audit
and its SHA-256. Percentages are rounded only at the LaTeX rendering boundary;
the audit JSON retains full precision.

## Pinned audit sources

| Source | SHA-256 | Role |
|---|---|---|
| `data/audits/motivation_reaudit.json` | `7ba3cbe690b40921134a364a987ed7ae3bd58a595221bd49ac7a4026d01633fe` | Raw-event Motivation recomputation and custody |
| `data/qualification_summary.json` | `2f76538e76549ae06afae1c70984363a1d9e4ce277aa508b21d1285ec8fbb848` | Three-model 11-gate capability qualification |
| `data/formal_campaigns.json` | `219f37cff69ab34b9d7b95891c84c733bc1cc8a922e6f13bd8bf581ed193e67a` | Baseline, Issue-17, overlap, capacity, workload and artifact audit |
| `data/issue17_audit.json` | `ad7376e09a985099c07dcfd9d428589c077f2425658f56bdcb0525db4a9926fa` | Portable full-layer/multi-wave bundle audit |
| `data/resource_ledgers.json` | `84429b7bdb921f167b446da3298761b5e155ca0255b453567a75267c3a5a4697` | Three-model memory, transfer, control and release ledger |
| `generated/formal_results_macros.tex` | `ec5499b5263487908e9254b3ff2cb6702fd08ed1014d4444d3898c9ad88a1a70` | Deterministic paper-facing rounding |

## Frozen Introduction result

The author-frozen Introduction reports four-workload eager latency of
133.7--137.4 ms/token, graph latency of 32.6--33.6 ms/token, and an approximate
75% reduction. Its only retained evidence is the author-supplied raster
`figures/graph_breakdown.png` (SHA-256
`445bbc8140252f9c8e67a0e10a67f328ca2a87b40bf615ac66e300a7495dcc37`).
No raw measurements or plotting source were found in the repository or
available experiment directories, so this one frozen result is not
independently regenerable. `FROZEN_SECTION_SUGGESTIONS.md` asks the authors to
recover and audit its source before submission; the writing workflow did not
alter or silently relabel it.

## Motivation and characterization

All rows use one Ascend 910B2, BF16, TP1, prefix caching disabled, and both
concurrency controls set to one. Cross-model rows establish recurrence, not a
ranking, because request counts and output limits differ.

| Model | Scope (requests/layers) | Decode access miss | Miss-bearing invocations | Mapping-update invocations | Same-layer Jaccard p50 | Prefill active experts p50/p95/max | Waves at 32 slots p50/p95/max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 200/12 | 20.5521% | 68.7083% | 68.0938% | 0.333333 | 99/117/126 | 4/4/4 |
| GLM-4.7-Flash | 100/11 | 14.0200% | 41.0251% | 40.3772% | 0.142857 | 61/64/64 | 2/2/2 |
| Qwen3-Next-80B-A3B-Instruct | 64/48 | 38.1975% | 94.2241% | 93.8980% | 0.250000 | 252/401/496 | 8/13/16 |

Additional paper-facing facts from the same audit:

- All 2,400/1,100/3,072 profiled prefill-layer invocations overflow 32 slots.
- Active-weight p50 is 0.87/1.07/1.48 GiB, versus 0.28/0.56/0.19 GiB of
  32-slot storage for Qwen3/GLM/Qwen3-Next.
- At 64 slots, Qwen3 overflow is 99.29% and Qwen3-Next overflow is 100%; at
  128 slots, Qwen3-Next overflow is 99.48%.
- The three traces contain 8,571/2,200/26,695 prefill waves and
  1.58/0.60/4.62 TB H2D traffic. Stage wait is
  1,459.871/391.919/4,683.342 ms, or 20.24%/24.03%/17.17% of recorded MLP time.
- Every re-audited decode publication follows readiness, every prefetch is
  issued before its target compute, failed requests are zero, and each service
  has a release acknowledgment.

## Three-model qualification

Each row passes the same 11 gates. End-to-end qualification is one short
request per mode; Qwen3-Next uses eager-seam versus graph exactness because a
native full-resident end-to-end run exceeds the device capacity, while native
semantics are still checked at router and layer boundaries.

| Model | L/E/S | top-k | Router/boundary records | Address locks/replays | Overflow/H2D records |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 48/128/0 | 8 | 192/192 | 48/97 | 144/239 |
| GLM-4.7-Flash | 46/64/1 | 4 | 184/184 | 46/94 | 92/226 |
| Qwen3-Next-80B-A3B-Instruct | 48/512/1 | 10 | 192/192 | 48/97 | 144/240 |

The matched eager-versus-PIECEWISE diagnostic in
`paper/data/eager_graph_diagnostic.json` uses one Qwen3 request and four
output tokens. It produces identical output-token IDs in both modes, with
TTFT 2235.661983/2165.771722 ms and TPOT 262.042458/140.633295 ms/token
(eager/graph). The result is qualification evidence only; it is not a
repeated-start estimator or a throughput/concurrency claim.

## Formal Qwen3 performance campaigns

Unless stated otherwise, these use the same frozen ShareGPT prefill-heavy
workload, 1,034--1,979 prompt tokens (median 1,394), 32 generated tokens,
temperature zero, seed 42, one Ascend 910B2, BF16, TP1, concurrency one,
`max_num_seqs=1`, disabled prefix caching, PIECEWISE ACLGraph, and a 512-MiB KV
cache. The Qwen3 performance arms manage 12 of 48 MoE layers (IDs 3, 7, 11,
..., 47) under the 14-GiB AutoConfig; every compared arm uses the same
selection. Every reportable unit passes exact-token, graph, address, lifecycle,
failure, raw-artifact and release-ACK gates.

| Claim | Design | Result | Uncertainty/scope |
|---|---|---|---|
| Full-resident cost anchor | Three counterordered starts, 32 requests/unit | repeat-median TTFT p50 179.13 ms and TPOT p50 13.19 ms/token; sampled peak HBM median 97% | Cost anchor, not memory matched |
| LatchMoE bounded offload | Same starts/workload, 14-GiB target | repeat-median TTFT p50 662.22 ms and TPOT p50 54.50 ms/token; sampled peak HBM median 80% | vs full resident: TTFT +261.283% (95% CI +240.088% to +277.323%); TPOT +309.515% (95% CI +299.132% to +319.376%) |
| Unsupported native/legacy cells | Three starts, 96 completed requests/arm | native prefetch 30/96 mismatched arrays; legacy layered offload 30/96 | Retained as unsupported; no latency claim |
| Bounded waves vs full-layer staging | Three AB/BA/AB pairs, 200 requests/unit, 1,200 total | TTFT -35.1257%; TPOT +0.2483% | TTFT 95% CI [-35.6741%, -34.5963%]; TPOT 90% CI [-1.1360%, +1.6765%], wholly inside ±5% |
| Conventional bounded-wave summary | Same six units | median-across-start TTFT p50 reduction 35.21%; p95 reduction 22.96% | Descriptive companion to paired estimator |
| Overlap vs serial staging | Three AB/BA/AB pairs, 64 requests/unit, 384 total; prefetch depth is the only factor | TTFT -27.8713%; TPOT -4.9484% | TTFT 95% CI [-30.1835%, -25.2428%], causal gate passes; TPOT 90% CI [-5.7989%, -4.1502%], equivalence gate fails |

The paired estimator is
`exp(mean_start median_request(log(T_treatment/T_control))) - 1`. Intervals use
10,000 hierarchical paired-bootstrap replicates with seed 20260823, resampling
start pairs and then paired request IDs.

## Slot-capacity sensitivity

Each point is one fresh, exact-token, graph-qualified start over the first 32
requests; results are descriptive rather than uncertainty claims.

| Slots | TTFT p50/p95 (ms) | TPOT p50 (ms/token) | Total waves (p50/invocation) | H2D | Slot/stage storage | Sampled peak/final HBM |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 679.36/754.92 | 60.28 | 2,808 (7) | 318.5 GiB | 1.69/0.281 GiB | 77%/5% |
| 32 | 611.46/665.12 | 55.85 | 1,512 (4) | 268.4 GiB | 3.38/0.562 GiB | 80%/5% |
| 64 | 452.67/529.06 | 45.65 | 792 (2) | 173.8 GiB | 6.75/1.125 GiB | 87%/5% |

Thus 16→64 slots reduces TTFT p50/p95 by 33.37%/29.92%, while increasing
persistent slot storage and sampled peak HBM.

## Qualification-run resource ledger

| Model | Host | Device | H2D | Host control | Graph replays |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 54.0 GiB | 14.1 GiB | 65.5 GiB | 10,062.854 ms | 97 |
| GLM-4.7-Flash | 51.8 GiB | 26.7 GiB | 22.3 GiB | 73.095 ms | 94 |
| Qwen3-Next-80B-A3B-Instruct | 144.0 GiB | 9.7 GiB | 152.6 GiB | 44,852.769 ms | 97 |

Device storage is the audited sum of persistent slots, prefill stage banks, and
resident shared-expert weights. Qualification profiles contain one original
expert-bank release event per managed layer; formal campaign and Motivation
services additionally retain service-level release acknowledgments.
Persistent slot banks are 13.5/25.9/9.0 GiB and transient stage banks are
0.56/0/0.38 GiB for Qwen3/GLM/Qwen3-Next; mapping buffers are at most 0.19 MiB.
