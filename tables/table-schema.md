# Table Schema

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| T0: Ascend UVA-like feasibility | Decide whether CUDA-UVA-style expert access is a viable baseline or threat model on Ascend. | Gate x device x CANN version x probe size, plus one verdict row. | API support, return code, mapped device pointer, tensor wrapping, copy/SDMA status, simple AI Core bandwidth, NPUGraph replay status, MoE-shaped matmul status, verdict, comparison boundary to SEW. | `benchmark/artifacts/reports/ascend_uva_feasibility/*.json`, `benchmark/artifacts/reports/ascend_uva_feasibility/e0_ascend_uva_like_summary.csv`, `benchmark/artifacts/reports/ascend_uva_feasibility/e0_ascend_uva_like_verdict.json`. | Experiment runner. |
| T1: Benchmark feasibility | Show success/failure under each case and workload. | Cases x workloads x repetitions. | status, failure reason, server readiness, graph capture completed, request success/failure. | `unit_result.json`, `server.log`, `benchmark.json` when present. | Experiment runner. |
| T2: End-to-end serving | Main performance comparison under fixed HBM budgets. | Successful cases x workloads x repetitions. | TTFT p50/p90/p99, TPOT p50/p90/p99, request latency p50/p90/p99 when available, output throughput, request throughput. | `benchmark.json`, collected per-request metrics. | Experiment runner. |
| T3: Memory cost | Explain HBM, KV, slot-bank, and host-store tradeoff. | Cases x workloads. | weights GiB, available KV GiB, KV tokens, slot-bank GiB, host-store GiB, original weights retained, peak HBM if available. | `server.log`, `moe_profile.jsonl`, profiler output. | Experiment runner. |
| T4: Offload overhead | Explain expert movement and B2 execution cost. | SEW cases x workloads x repetitions. | H2D bytes, stage ms, prefill stage ms, active experts, wave count, slot hit/miss rate. | `moe_profile.jsonl`, `moe_trace.jsonl`. | Experiment runner. |
| T5: Ablation | Attribute mechanism value. | Full SEW and ablation variants. | TTFT, TPOT, throughput, H2D bytes, wave count, failures. | Full raw run dirs. | Experiment runner. |
| T6: Graph evidence | Support the graph-replay-safe claim. | Cases x graph mode. | graph capture status, `moe_offload_stage` seen, stable address/log2phy evidence, failure class and exact log signature. | `server.log`, debug trace, unit result. | Experiment runner. |
| T7: Correctness and lifecycle safety | Bound model semantics and slot reuse safety. | Prompt set x case x check. | output exact match, logit max/mean diff, stale mapping count, COMPUTING overwrite violations, unit tests. | correctness scripts, pytest logs, trace assertions. | Experiment runner. |
| T8: Concurrency robustness | Secondary evidence that fixed-slot mechanism remains stable at low concurrency. | concurrency x case x workload. | success rate, TTFT/TPOT tails, slot hit/miss, failures. | run dirs for max concurrency 2/4. | Experiment runner. |

Aggregation rules:

- Paper tables use repeated full workloads, not smoke runs.
- Successful performance rows report median across repetitions plus per-request
  p90/p99 when available.
- Failure rows keep the failure as data and report the classified root cause.
- E0/T0 rows must keep the gate boundary explicit: runtime registration is not
  evidence for AI Core readability, ACLGraph replay, or vLLM-MoE performance.
- Any mock or planning table must include `PLANNING DATA - replace before
  submission`.
