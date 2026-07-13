# LatchMoE Four-Stage Optimization Results — 2026-07-10

## Locked protocol

- Model: Qwen3-30B-A3B, BF16, TP=1.
- Hardware: one Ascend 910B2-class NPU, physical device 4.
- Dataset: frozen `mixed_chat` records from the ShareGPT manifest.
- Requests: first 50 records; `max_output_tokens=128`.
- Offload: 14 GiB target, AutoConfig 32 slots, SEW/ACLGraph enabled.
- Tasks 1–2 use client concurrency 1. Tasks 3–4 use concurrency 8 and
  `max_num_seqs=8` to force mixed prefill/decode expert-union overflow.

The concurrency-1 and concurrency-8 rows are different serving regimes and must
not be compared as a mechanism ablation. Task 1 vs 2 and Task 3 vs 4 are the two
controlled comparisons.

## End-to-end results

| Stage | Concurrency | Success | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms/token) | Output throughput (tok/s) |
|---|---:|---:|---:|---:|---:|
| 1. Corrected B2 memory ledger | 1 | 50/50 | 1105.05 / 1656.90 | 76.30 / 87.62 | 11.589 |
| 2. Shared layout-scoped stage pool | 1 | 50/50 | 1084.29 / 1710.17 | 73.98 / 89.32 | 12.004 |
| 3. Exact mixed-phase pair-wave executor | 8 | 50/50 | 3944.23 / 5194.70 | 84.68 / 106.73 | 61.911 |
| 4. NPU pair planner + scatter descriptor | 8 | 50/50 | 3311.30 / 4334.12 | 80.86 / 95.31 | 68.925 |

Task 3 produced 240 mixed-phase B2 events. The maximum active-expert union was
127 with a 32-slot capacity, and calls completed in 2–5 exact waves. This proves
that the benchmark exercised the new mixed overflow path rather than only the
legacy prefill path.

## Memory validation

Before sharing, the runtime allocated 24 physical stage banks: 12 offloaded
layers times two buffers. Each bank contained 301,989,888 managed bytes. The NPU
allocator reported a 301,990,912-byte increment per allocation, a 1 KiB
difference. The corrected ledger reports:

| Configuration | Stage banks | Stage-bank HBM | Main + stage slot HBM |
|---|---:|---:|---:|
| Per-layer double buffers | 24 | 6.7500 GiB | 10.1250 GiB |
| Shared layout-scoped double buffer | 2 | 0.5625 GiB | 3.9375 GiB |

The shared pool therefore removes 6.1875 GiB (91.7%) of stage-bank HBM. Cross-
layer reuse is protected by compute-stream release events consumed by the H2D
transfer stream before overwriting a shared buffer.

## Device-control attribution

Task 4 recorded `pair_planner_mode=npu_device` for all 286 B2 events. Against
Task 3 under the same frozen workload and concurrency:

| B2 control component | Task 3 mean (ms) | Task 4 mean (ms) | Change |
|---|---:|---:|---:|
| Pair-index launch/control | 0.068 | 0.016 | -76.5% |
| Wave microbatch planning | 4.920 | 2.382 | -51.6% |
| Final scatter | 1.199 | 0.792 | -33.9% |
| B2 end-to-end per event | 17.854 | 15.393 | -13.8% |

Task 4 improves output throughput by 11.3%, reduces TTFT p50/p99 by 16.0%/16.6%,
and reduces TPOT p50/p99 by 4.5%/10.7% relative to Task 3. These are one-run
measurements; repeated runs are required before reporting confidence intervals.

## Raw artifacts

- Task 1: `benchmark/artifacts/runs/task1_ledger_mixed50/sew-offload-ascend-v1-20260710T074355Z`
- Task 2: `benchmark/artifacts/runs/task2_shared_pool_mixed50/sew-offload-ascend-v1-20260710T080034Z`
- Task 3: `benchmark/artifacts/runs/task3_exact_mixed_pair_wave_mixed50_c8/sew-offload-ascend-v1-20260710T081920Z`
- Task 4: `benchmark/artifacts/runs/task4_npu_pair_scatter_mixed50_c8/sew-offload-ascend-v1-20260710T105256Z`

Each directory retains the unit manifest, server/client logs, benchmark JSON,
and MoE profile JSONL. No synthetic or mock measurements are used above.

## Follow-up: Task 4 at concurrency 1

Task 4 was additionally rerun with the same concurrency-1 protocol as Task 2:
the same first 50 ShareGPT `mixed_chat` records and 128 maximum output tokens.
This isolates the NPU pair/scatter planner from the shared-pool change.

| Configuration | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms/token) | Throughput (tok/s) |
|---|---:|---:|---:|
| Task 2: Host pair planner | 1084.29 / 1710.17 | 73.98 / 89.32 | 12.004 |
| Task 4: NPU pair/scatter planner | 978.82 / 1879.11 | 74.56 / 84.88 | 12.039 |

Both runs completed 50/50 requests and produced 6,400 output tokens. Task 4
recorded `pair_planner_mode=npu_device` for all 610 prefill B2 events. The
host-observed control breakdown changed as follows:

| B2 control component | Host planner mean (ms) | NPU planner mean (ms) | Change |
|---|---:|---:|---:|
| Pair-index launch/control | 0.065 | 0.016 | -75.4% |
| Wave microbatch planning | 2.615 | 1.968 | -24.7% |
| Final scatter | 1.195 | 0.829 | -30.6% |
| B2 end-to-end per event | 13.753 | 13.493 | -1.9% |

At concurrency 1, the optimization primarily affects prompt processing: a
single-request decode activates at most top-8 experts per layer and normally
fits in 32 slots, so it does not use the pair-wave executor. Consequently, TTFT
p50 improves by 9.7%, while throughput changes by only +0.29% and TPOT p50 is
effectively unchanged (+0.79%). TTFT p99 regresses by 9.9% in this single run,
so the current evidence does not support a claim of improved concurrency-1 tail
latency. Repeated runs are required to distinguish an outlier from a systematic
tail effect.

Raw artifact:
`benchmark/artifacts/runs/task4_npu_pair_scatter_mixed50_c1/sew-offload-ascend-v1-20260710T123250Z`.
