# Benchmark Harness

This directory is the standard benchmark harness for `vllm-moe-offload-ascend`.
Its design principles: keep benchmark scenarios as data, render commands from a
stable registry, preserve raw artifacts, and separate run orchestration from
result summarization.

The benchmark is tailored to one systems question:

> Can slot-stable expert offloading make MoE expert offloading feasible and
> graph-compatible on a memory-constrained single Ascend 910B-class NPU?

## Layout

| Path | Purpose |
|---|---|
| `configs/sew_offload_v1.yaml` | Canonical model, dataset, workload, case, metric, and artifact contract. |
| `scenarios/sew_offload_scenarios.json` | Workload registry, stored as data. |
| `scripts/sew_bench.py` | Validate config, list cases/workloads, prepare ShareGPT manifests, render plans, summarize runs. |
| `scripts/run_suite.py` | Standard server benchmark runner for `case x workload` experiments. |
| `scripts/run_openai_manifest.py` | OpenAI-compatible streaming client for JSONL workload manifests. |
| `scripts/collect_evidence.py` | Collect cross-run evidence tables from unit artifacts. |
| `scripts/sharegpt_manifest.py` | Build tokenized ShareGPT workload manifests. |
| `scripts/bench_sharegpt.py` | Standalone OpenAI-compatible ShareGPT benchmark client. |
| `schemas/sew_offload_config.schema.json` | Human-readable schema for the canonical config. |
| `artifacts/` | Generated manifests and run outputs. Ignored by git except small placeholders. |

## Quick Start

Before running on a new server, check the real deployment paths used by the
canonical config:

```bash
test -d /data/shared_models/modelscope_cache/Qwen/Qwen3-30B-A3B
test -f /data/shared_datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```

If the model or dataset lives elsewhere, edit
`benchmark/configs/sew_offload_v1.yaml` before preparing workloads. For formal
experiments in this project, use only an explicitly reserved and idle physical
NPU5 or NPU6. This repository is graph-only and rejects forced eager cases.

```bash
export ASCEND_RT_VISIBLE_DEVICES=5
```

Validate the benchmark definition:

```bash
python benchmark/scripts/sew_bench.py validate
```

Inspect the default matrix:

```bash
python benchmark/scripts/sew_bench.py list-cases
python benchmark/scripts/sew_bench.py list-workloads
python benchmark/scripts/sew_bench.py render-plan --case sew_14gb_autoslots --workload smoke
```

Build a real ShareGPT workload manifest:

```bash
python benchmark/scripts/sew_bench.py prepare-workloads \
  --bucket smoke \
  --requests-per-bucket 1
```

Dry-run one benchmark unit without starting vLLM:

```bash
python benchmark/scripts/run_suite.py \
  --case sew_14gb_autoslots \
  --workload smoke \
  --dry-run
```

Run one real unit in the configured vLLM/Ascend environment:

```bash
python benchmark/scripts/run_suite.py \
  --case sew_14gb_autoslots \
  --workload mixed_chat
```

Summarize a completed run directory:

```bash
python benchmark/scripts/sew_bench.py summarize benchmark/artifacts/runs/<run-id>
```

Collect cross-run evidence and initial plots:

```bash
python benchmark/scripts/collect_evidence.py \
  benchmark/artifacts/runs/<run-id> \
  --output-dir benchmark/artifacts/reports/<report-id>
```

## Design Boundary

The main benchmark path is server-based because the evaluation needs TTFT,
TPOT, throughput, graph-capture evidence, server logs, and profile JSONL
artifacts from the same serving boundary users will operate.

The default supported boundary is intentionally narrow:

- Qwen3-30B-A3B, unquantized MoE.
- Single Ascend 910B-class NPU.
- OpenAI-compatible vLLM server.
- Low-concurrency serving first, `max_num_seqs=1`.
- ShareGPT-derived real prompts only.
- Offload budgets of 14 GiB and 28 GiB, plus slot sensitivity.

## Artifact Contract

Each benchmark unit writes:

- `unit_manifest.json`: commands, environment, case, workload, and paths.
- `server.log`: vLLM server stdout and stderr.
- `client.log`: benchmark client stdout and stderr.
- `benchmark.json`: serving metrics from the streaming client.
- `unit_result.json`: normalized status and pointers to artifacts.
- `moe_profile.jsonl`: SEW/MoE runtime profile events when enabled.
- `moe_trace.jsonl`: routed expert trace events when enabled.

Failed units still write `unit_result.json`; expected OOM or graph-capture
failures are evidence, not missing data.
