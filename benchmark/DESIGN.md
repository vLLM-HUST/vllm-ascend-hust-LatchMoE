# Benchmark Design

## What We Borrow From vllm-hust-benchmark

`vllm-hust-benchmark` is valuable because it stays thin. It does not fork the
real benchmark runtime. Instead, it provides:

1. A stable entrypoint.
2. Scenario definitions stored as data.
3. Command rendering from scenarios.
4. Standard artifact export.
5. Baseline specs that can be rerun and compared.

This benchmark keeps the same engineering boundary, but specializes the
scenario vocabulary for MoE offloading on Ascend.

## Evaluation Gap

General vLLM serving benchmarks measure TTFT, TPOT, throughput, and latency.
They do not diagnose the stateful costs that decide whether dynamic MoE
offloading works under Ascend ACLGraph replay:

- fixed physical expert slot residency;
- logical-to-physical expert mapping stability;
- H2D expert movement and staging time;
- active expert overflow during prefill;
- graph-capture and replay compatibility;
- memory feasibility under a fixed HBM budget.

The benchmark therefore treats offload state as a first-class metric surface,
not a hidden implementation detail.

## Standard Workloads

All workloads are sampled from real ShareGPT prompts by token bucket.

| Workload | Role | Main pressure |
|---|---|---|
| `smoke` | Minimal validation | End-to-end harness health. |
| `mixed_chat` | Main serving result | Representative chat mixture. |
| `decode_heavy` | Hot decode path | TPOT, slot hits, graph replay. |
| `prefill_heavy` | Long prompt, short output | TTFT, B2 waves, active expert fanout. |
| `long_context_prefill` | Optional stress | Wave count, H2D bytes, memory edge. |

Synthetic and random datasets are not default benchmark inputs. They may be
used only for local debugging outside this standard artifact path.

## Standard Cases

The case taxonomy mirrors the paper claims:

- Capacity and baseline probes: `no_offload_capacity_probe`,
  `native_prefetch_14gb`, `legacy_layered_14gb`.
- Main system: `sew_14gb_autoslots`, `sew_28gb_autoslots`.
- Ablations: no B2, no transfer-aware schedule, no CPU-first loading.
- Sensitivity: explicit slot counts around AutoConfig.

Native prefetch and SEW are always run in separate server processes. Native
vLLM offload flags must never be mixed with `VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE=1`.

## Metrics

The benchmark preserves the normal serving metrics:

- TTFT: mean, p50, p90, p99.
- TPOT: mean, p50, p90, p99.
- request throughput.
- output token throughput.
- success, failure, and error rate.

It adds MoE-offload metrics when profile events are available:

- peak or configured HBM budget;
- CPU expert store and slot-bank memory;
- H2D bytes;
- staging time;
- slot hit/miss rate;
- active expert count;
- wave count;
- graph-capture evidence from logs.

The runner records raw logs and JSONL first. Derived tables and plots should be
regenerated from artifacts instead of hand-editing numbers.

## Experiment Groups

| Group | Question |
|---|---|
| `e0_smoke` | Does the harness and one SEW path work? |
| `e1_end_to_end` | Does SEW improve feasibility and serving performance? |
| `e2_memory_graph` | Does SEW produce the graph-compatible evidence claimed? |
| `e3_b2_prefill` | Are B2 waves necessary for prefill overflow? |
| `e4_ablation` | Which mechanisms carry the benefit? |
| `e5_slot_sensitivity` | How sensitive is performance to slot budget? |

The benchmark is intentionally configured before the run. If a case fails, the
failure belongs in the result table with the recorded reason.
