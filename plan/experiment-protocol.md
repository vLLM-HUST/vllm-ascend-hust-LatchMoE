# SEW-Offload Experiment Protocol

Date: 2026-06-30

This protocol covers the Qwen3-30B-A3B single-Ascend-NPU evaluation for the
CCF-A systems-paper plan. Smoke runs are allowed only as harness validation;
paper claims require repeated full-workload runs.

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
| `smoke` | 64-512 | 8 | Harness validation only. |

Seed: `42`.

## Baselines

| Case | Purpose | Fairness boundary |
|---|---|---|
| `no_offload_capacity_probe` | Capacity and KV feasibility. | Same model, same server shape, no expert offload. |
| `native_prefetch_14gb` | Existing dynamic prefetch baseline. | Native prefetch flags only, no SEW env. |
| `legacy_layered_14gb` | Plugin path before graph-compatible seam. | Plugin offload without SEW dataplane. |
| `sew_14gb_autoslots` | Main 14 GiB SEW point. | AutoConfig slots, SEW enabled. |
| `sew_28gb_slots32` | Viable 28 GiB SEW point found in Week 2 smoke. | Explicit 32 slots until AutoConfig becomes KV-aware. |

`sew_28gb_autoslots` remains a diagnostic case, not a main comparison, until
AutoConfig is fixed.

## Metrics

Serving metrics:

- TTFT p50/p90/p99.
- TPOT p50/p90/p99.
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

Graph evidence:

- `vllm::moe_offload_stage` appears in splitting ops.
- `Graph capturing finished` appears in server log.
- Native/legacy failures are classified from ACLGraph error text.

## Repetition Rule

For paper numbers, run at least three repetitions per successful case and
workload. Report median plus p90/p99 from per-request metrics. Smoke runs must
not be used as final results.

## Hardware and Software

Use `ASCEND_RT_VISIBLE_DEVICES=4` unless a later lock file records a different
free NPU. Record CANN, driver, plugin commit, model path, dataset path, and
environment variables for every batch. Do not record vLLM-Ascend commit unless
the user explicitly asks to lift that constraint.

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
