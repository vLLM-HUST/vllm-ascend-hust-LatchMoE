# Week 1 Execution Summary

Date: 2026-06-30

Scope: execute Week 1 of `docs/ccf_a_paper_roadmap.md` using the additional
evaluation requirements from `docs/evaluation_design_notes.md`.

User constraint: do not execute or record the vLLM-Ascend commit. This was
honored; no vLLM-Ascend commit is recorded here.

## Completed Items

1. Frozen the system name and thesis.
2. Created a clean Week-1 smoke config set.
3. Recorded environment, model, dataset, hardware, and plugin repository state
   without recording vLLM-Ascend commit.
4. Ran unit tests in the `vllm-hust-dev` Conda environment.
5. Ran smoke probes for no-offload, native prefetch, legacy/layered, and
   SEW-Offload.

## Files Created

| File | Purpose |
|---|---|
| `demo/week1_smoke_config.json` | Clean smoke config set for Week 1. |
| `docs/week1_reproducibility_lock.md` | Environment and reproducibility lock, excluding vLLM-Ascend commit. |
| `docs/week1_execution_summary.md` | This execution summary. |
| `demo_runs/week1_smoke/...` | Raw smoke artifacts. |

## Verification

Unit test command:

```bash
/root/miniconda3/bin/conda run -n vllm-hust-dev python -m pytest tests -q
```

Observed result:

```text
103 passed, 3 warnings in 16.76s
```

## Smoke Outcomes

| Case | Outcome | Key evidence |
|---|---|---|
| `no_offload_capacity_probe` | Expected capacity failure | Loaded weights took `56.9001 GB`; available KV cache memory was `-2.19 GiB`; server failed with `No available memory for the cache blocks`. |
| `native_prefetch_14gb` | Failed during ACLGraph capture | `PrefetchOffloader` was enabled and weights dropped to `43.4001 GB`, but capture failed with `capture model contains a stream that was not joined to the original stream`. |
| `legacy_layered_14gb` | Failed during captured path | Profile was produced, but captured execution attempted D2H/copy sync and failed with `Not allow to synchronize captured-stream`. |
| `sew_14gb_slots32` | Smoke success | `vllm::moe_offload_stage` appeared in splitting ops, graph capture finished, server reached `/v1/models`, and one benchmark request succeeded. |

## SEW Smoke Metrics

These are smoke-only numbers, not paper results.

| Metric | Value |
|---|---:|
| Successful requests | 1 |
| Failed requests | 0 |
| Median TTFT | 744.159 ms |
| Median TPOT | 58.095 ms/token |
| Output throughput | 6.524 tokens/s |

SEW artifact directory:

```text
demo_runs/week1_smoke/week1-smoke-20260630-20260630T111454Z/sew_14gb_slots32/
```

Important artifacts:

| Artifact | Status |
|---|---|
| `case_manifest.json` | present |
| `server.log` | present |
| `moe_profile.jsonl` | present, 132 lines |
| `benchmark.json` | present |
| `case_result.json` | present |
| `summary.md` | present |

## Research Interpretation

The Week-1 smoke run supports the paper's problem framing:

1. No-offload cannot reserve KV cache on the selected single 910B2 NPU.
2. Native prefetch reduces weight memory but fails ACLGraph capture due to an
   unjoined stream.
3. Legacy/layered offload reaches plugin profiling but performs a captured
   D2H/synchronizing copy, which ACLGraph rejects.
4. SEW-Offload moves the MoE offload stage into an explicit graph-compatible
   seam and successfully reaches graph capture plus one request.

This is not yet a performance claim. It is Week-1 feasibility and artifact
validation for the evaluation plan.

## Follow-up Fixes Before Week 2

1. Investigate why `run_annual_demo_suite.py` did not proceed to spawn
   `bench_sharegpt.py` after SEW `/v1/models` readiness, even though manual
   readiness and manual benchmark both succeeded.
2. Add an explicit failure-result writer to the runner so expected OOM/capture
   failures produce `case_result.json` instead of only `server.log`.
3. Add a full paired 14 GiB / 28 GiB run after the runner issue is fixed.
4. Add graph evidence extraction scripts for `vllm::moe_offload_stage`,
   graph-capture completion, and profile `b2_work_conserving_prefill` events.
