# Week 2 Core Evidence Summary

Date: 2026-06-30

Scope: execute the first Week-2 core-evidence loop on the refactored
`benchmark/` harness.

User constraint carried forward: do not execute or record the vLLM-Ascend
commit. This summary does not record a vLLM-Ascend commit.

## What Changed

1. Fixed the canonical benchmark model/tokenizer path to the locally available
   Qwen3-30B-A3B path.
2. Hardened benchmark config validation so missing model, tokenizer, or
   dataset paths fail early.
3. Fixed `run_suite.py` readiness detection by bypassing environment proxies
   for localhost `/v1/models`.
4. Added `collect_evidence.py` to extract normalized evidence from raw unit
   artifacts and produce initial CSV/JSON/SVG plots.
5. Added tests for benchmark config, localhost readiness, and evidence
   extraction.
6. Updated the default benchmark matrix to use `sew_28gb_slots32` as the
   current viable 28 GiB point; `sew_28gb_autoslots` is kept as a diagnostic
   case until AutoConfig becomes KV-aware.

## Validation

Config validation:

```bash
/root/miniconda3/bin/conda run -n vllm-hust-dev \
  python benchmark/scripts/sew_bench.py validate
```

Result:

```text
OK /root/vllm-moe-offload-ascend/benchmark/configs/sew_offload_v1.yaml
```

Full unit tests:

```bash
/root/miniconda3/bin/conda run -n vllm-hust-dev python -m pytest tests -q
```

Result:

```text
109 passed, 3 warnings in 17.92s
```

## Smoke Evidence

All runs used:

- NPU: `ASCEND_RT_VISIBLE_DEVICES=4`
- Model: `/data/shared_models/modelscope_cache/Qwen/Qwen3-30B-A3B`
- Dataset manifest:
  `benchmark/artifacts/workloads/sharegpt_qwen3_30b_a3b_v1.jsonl`
- Workload: `smoke`, one real ShareGPT prompt, 8 max output tokens
- `max_model_len=4096`, `max_num_seqs=1`

Summary table:

| Case | Status | Key result |
|---|---|---|
| `no_offload_capacity_probe` | failed | `56.9001 GB` weights; KV cache `-2.48 GiB`; capacity failure. |
| `native_prefetch_14gb` | failed | `43.4001 GB` weights; KV cache `11.02 GiB`; ACLGraph failed with unjoined stream. |
| `legacy_layered_14gb` | failed | `46.7751 GB` weights; KV cache `7.65 GiB`; captured stream synchronization/D2H failure. |
| `sew_14gb_autoslots` | ok | `46.7751 GB` weights; KV cache `1.81 GiB`; graph capture completed; one request succeeded. |
| `sew_28gb_autoslots` | failed | AutoConfig selected `64` slots; slot bank `13.5 GiB`; KV cache `-4.96 GiB`. |
| `sew_28gb_slots32` | ok | `36.6501 GB` weights; KV cache `5.20 GiB`; graph capture completed; one request succeeded. |

Successful smoke serving metrics:

| Case | TTFT ms | TPOT ms/token | Output tok/s |
|---|---:|---:|---:|
| `sew_14gb_autoslots` | 4340.145 | 91.951 | 1.572 |
| `sew_28gb_slots32` | 1414.637 | 77.531 | 3.901 |

These are smoke-only numbers and must not be reported as final paper results.

## Important Finding

The Week-2 smoke loop found that offload budget and slot budget must be treated
as separate experimental dimensions. More expert offload does not automatically
mean more usable KV cache: `sew_28gb_autoslots` failed because 64 slots consumed
too much NPU memory, while `sew_28gb_slots32` succeeded and provided a larger
KV cache than `sew_14gb_autoslots`.

This suggests the next AutoConfig step should be KV-aware slot capping rather
than scaling slots only with offload budget.

## Generated Reports

Evidence report directory:

```text
benchmark/artifacts/reports/week2_smoke_20260630/
```

Key files:

- `week2_smoke_summary.csv`
- `week2_smoke_summary.json`
- `week2_smoke_ttft.svg`
- `week2_smoke_tpot.svg`
- `week2_smoke_throughput.svg`
- `week2_smoke_weights.svg`
- `week2_smoke_kv_cache.svg`
- `week2_smoke_h2d.svg`

## Next Week-2 Actions

1. Make AutoConfig KV-aware for `28 GiB` so the default path does not select an
   infeasible slot bank at `max_model_len=4096`.
2. Generate full ShareGPT manifests for `mixed_chat`, `decode_heavy`, and
   `prefill_heavy`.
3. Run formal repeated benchmarks for `sew_14gb_autoslots`,
   `sew_28gb_slots32`, no-offload probe, native prefetch, and legacy/layered.
4. Replace smoke plots with Matplotlib/Seaborn plots from repeated raw runs in
   a plotting environment that has matplotlib installed.
5. Draft Figure 1 and Figure 2 after the first full result table is populated.
