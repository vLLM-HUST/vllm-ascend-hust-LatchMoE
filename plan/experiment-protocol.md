# SEW-Offload Experiment Protocol

Date: 2026-07-09

This protocol evaluates SEW-Offload as a single-card, memory-constrained,
graph-replay-safe MoE expert offload runtime on Ascend. The paper is not a
generic cloud-serving throughput contest. Smoke runs validate the harness only;
paper claims require repeated full-workload runs or classified failure logs.

## Paper Claims Under Test

| Claim | Evidence family | Allowed wording before full evidence |
|---|---|---|
| C0. CUDA-UVA-like offload is a necessary threat model, but its Ascend feasibility must be tested layer by layer. | E0 | "Runtime host registration observed; simple AI Core elementwise read and simple NPUGraph replay work, but copy/SDMA paths fail and host-registered matmul weights fail with `507057`" for the current CANN/910B2 stack; no full vLLM grouped-MLP or SEW-comparison claim yet. |
| C1. SEW enables MoE serving under tight HBM budgets by moving nonresident experts to host memory. | E1, E2 | "Smoke-supported feasibility" until repeated ShareGPT buckets finish. |
| C2. SEW is graph-replay-safe while existing dynamic offload paths are not. | E2, graph failure logs, stable slot/log2phy evidence | "Graph compatibility observed in logs" until trace-backed table is complete. |
| C3. ACLGraph replay improves the same SEW offload path over eager execution. | E1 capture-on/off pairs | "Capture-on/off ablation" only for cases with same budget, slots, and workload. |
| C4. B2 prefill waves handle active expert sets larger than the slot budget. | E3 | "Planned" until no-B2/B2/transfer-aware variants are run. |
| C5. Slot lifecycle protection prevents unsafe overwrite under overlapped transfer and compute. | E6 | "Correctness mechanism" until stress or assertion evidence exists. |
| C6. SEW preserves model semantics. | E6 output/logit equivalence | No quality claim before equivalence checks. |

## Dataset and Splits

Dataset: `/data/shared_datasets/ShareGPT_V3_unfiltered_cleaned_split.json`

Manifest generator:

```bash
python benchmark/scripts/sew_bench.py prepare-workloads
```

Manifest path:

```text
benchmark/artifacts/workloads/sharegpt_qwen3_30b_a3b_v1.jsonl
```

Required buckets:

| Bucket | Prompt tokens | Output tokens | Role |
|---|---:|---:|---|
| `mixed_chat` | mixed 128-4096 | 128 | Main serving mix. |
| `decode_heavy` | 64-256 | 256 | Decode hot-path stress. |
| `prefill_heavy` | 1024-2048 | 32 | B2 and TTFT stress. |
| `long_context_prefill` | 2048-4096 | 16 | Optional B2/memory-edge stress. |
| `smoke` | 64-512 | 8 | Harness validation only. |

Seed: `42`.

Split rule: workload manifests must be generated once per tokenizer/model pair
and reused across baselines. Do not let a baseline receive a different prompt
set. If a prompt fails tokenization or exceeds `max_model_len`, record the
filter rule in the manifest metadata.

## Baselines

| Case | Purpose | Fairness boundary |
|---|---|---|
| `ascend_uva_like_probe_14gb` | Test whether CANN can provide a CUDA-UVA-like host-backed expert address at the same 14 GiB budget. | Pre-serving runtime probe only; not a vLLM performance baseline until AI Core, ACLGraph, and MoE integration gates pass. |
| `no_offload_capacity_probe` | Capacity and KV feasibility. | Same model, same server shape, no expert offload. |
| `native_prefetch_14gb` | Existing dynamic prefetch capture attempt. | Native prefetch flags only, no SEW env; failure is graph evidence, not a performance number. |
| `native_prefetch_14gb_eager` | Native prefetch performance baseline when graph capture is disabled. | Same native flags plus `--enforce-eager`; compare only to SEW under same workload family. |
| `legacy_layered_14gb` | Plugin path before graph-compatible seam. | Plugin offload without SEW dataplane; capture failure supports Figure 1. |
| `legacy_layered_14gb_eager` | Legacy eager performance baseline. | Same plugin path plus `--enforce-eager`. |
| `sew_14gb_capture_disabled` | Internal eager ablation. | Same SEW runtime, same budget, capture disabled. |
| `sew_14gb_autoslots` | Main 14 GiB SEW point. | AutoConfig slots, SEW enabled, capture on. |
| `sew_28gb_slots32` | Viable 28 GiB SEW point found in Week 2 smoke. | Explicit 32 slots until AutoConfig becomes KV-aware. |

`sew_28gb_autoslots` remains a diagnostic case, not a main comparison, until
AutoConfig is fixed.

## Metrics

Serving metrics:

- TTFT p50/p90/p99.
- TPOT p50/p90/p99.
- End-to-end request latency p50/p90/p99 when available.
- Output throughput.
- Request success and failure counts.

Memory and offload metrics:

- Model weights on NPU.
- Available KV cache memory.
- Slot-bank memory.
- Host expert-store memory.
- H2D bytes.
- Stage time.
- Active expert count.
- Wave count.
- Slot hit/miss rate.
- Slot-bank memory and host-store memory.

Graph evidence:

- `vllm::moe_offload_stage` appears in splitting ops.
- `Graph capturing finished` appears in server log.
- Native/legacy failures are classified from ACLGraph error text.
- Stable slot tensors and persistent `log2phy` buffers are recorded by debug
  assertions or trace events when available.

Correctness metrics:

- Deterministic output exact match where feasible.
- Logit max absolute difference / mean absolute difference when logits are
  captured.
- Slot mapping validation: no stale `log2phy`, no unmapped active expert, no
  overwrite of COMPUTING slots.

## Experiment Families

| ID | Question | Workloads | Cases | Primary artifacts |
|---|---|---|---|---|
| E0 | Can a CUDA-UVA-like expert access path be ported to Ascend at the 14 GiB offload budget? | Runtime/API probe, AI Core microbenchmark, ACLGraph microbenchmark, MoE-shaped microbenchmark | `ascend_uva_like_probe_14gb`, future U1-U4 probes, SEW 14 GiB for performance reference after U3 | T0, `docs/experiments/ascend_uva_like_feasibility.md` |
| E1 | Does SEW improve feasibility and serving performance under fixed HBM budgets? | `mixed_chat`, `decode_heavy`, `prefill_heavy` | no-offload, eager baselines, SEW eager, SEW capture-on | T1, T2, F3 |
| E2 | Does SEW provide graph-replay-safe evidence? | `mixed_chat`, `prefill_heavy`, diagnostic smoke | native/legacy capture attempts, SEW capture-on | T1, T6, F1/F2 |
| E3 | Are B2 waves necessary for prefill overflow? | `prefill_heavy`, `long_context_prefill` | no-B2, B2 without transfer-aware, full SEW | T4, F5 |
| E4 | Which modules carry the benefit? | `mixed_chat`, `prefill_heavy` | full SEW and ablations | T5, F6 |
| E5 | How sensitive is SEW to slot count and offload budget? | all main buckets | 14 GiB slots 8/16/32, 28 GiB slots 32/64 | T3, F4 |
| E6 | Does SEW preserve semantics and slot lifecycle safety? | 32-64 deterministic prompts plus stress traces | full-resident when feasible, SEW, unsafe/debug variants if allowed | T7 |
| E7 | Does the mechanism remain stable under low concurrency? | `mixed_chat`, maybe `decode_heavy` | SEW 14/28 at concurrency 2/4 | T8, F7 |

## Repetition Rule

For paper numbers, run at least three repetitions per successful case and
workload. Report median plus p90/p99 from per-request metrics. Smoke runs must
not be used as final results.

For failure evidence, repetition is not required if the failure is deterministic
and the log contains a stable root-cause signature. Still preserve at least one
full raw artifact directory per failure class.

For E0, do not collapse the gates. Runtime host registration, AI Core
readability, ACLGraph replay, framework integration, and MoE-shaped performance
are separate evidence layers. A pass at one layer must not be reported as a pass
at the next layer.

## Hardware and Software

Use `ASCEND_RT_VISIBLE_DEVICES=4` unless a later lock file records a different
free NPU. Record CANN, driver, plugin commit, model path, dataset path, and
environment variables for every batch. Do not record vLLM-Ascend commit unless
the user explicitly asks to lift that constraint.

The main paper hardware line should state "single Ascend 910B-class NPU" unless
the exact card and memory are locked for submission.

## Artifact Rule

Every unit must preserve:

- `unit_manifest.json`
- `server.log`
- `client.log` when client runs
- `benchmark.json` when serving succeeds
- `unit_result.json`
- `moe_profile.jsonl` when profile hooks are reached

Derived reports must be regenerated from raw artifacts with
`benchmark/scripts/collect_evidence.py`.

## Mock and Smoke Boundary

Smoke artifacts and planning figures may be used in internal reports, but final
paper prose must not say "results show" using smoke data. Any planning table or
mock figure must be marked `PLANNING DATA - replace before submission`.
