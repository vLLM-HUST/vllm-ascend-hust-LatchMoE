# Benchmark Harness

This directory is the standard benchmark harness for `vllm-moe-offload-ascend`.
Its design principles: keep benchmark scenarios as data, render commands from a
stable registry, preserve raw artifacts, and separate run orchestration from
result summarization.

The benchmark is tailored to one systems question:

> Can slot-stable expert offloading make MoE expert offloading feasible and
> graph-compatible on a memory-constrained single Ascend 910B-class NPU?

The canonical YAML remains the routed-only Qwen3 baseline. Shared-expert and
complex-router runs must supply their checkpoint path explicitly and are only
reportable after the qualification matrix records their native, parity,
PIECEWISE, overflow, decode, and token gates as passed.

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
The benchmark launcher pins `VLLM_ENGINE_ENABLE_PREFIX_CACHING=0` and passes
`--no-enable-prefix-caching`; prefix-cache reuse is outside the LatchMoE
qualification scope.

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

Record a shared/router qualification unit with the immutable Phase-A registry
and bounded eager router artifacts. The parity verifier rejects missing native
or seam records as well as the first ID/weight/logit mismatch:

```bash
python benchmark/scripts/run_suite.py \
  --case sew_14gb_autoslots \
  --workload smoke \
  --model-path /root/data/shared_models/strict-models/GLM-4.7-Flash \
  --dataset-path /root/data/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
  --capability-registry benchmark/registry/model_registry_v2.json \
  --router-parity \
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

- `unit_manifest.json`: commands, environment, case, workload, paths, parent
  SHA, dependency SHAs, model/dataset hashes, device, and runtime-bundle identity.
- `server.log`: vLLM server stdout and stderr.
- `launcher_lifecycle.log`: manager start, status, stop, and release output.
- `client.log`: benchmark client stdout and stderr.
- `benchmark.json`: serving metrics from the streaming client.
- `unit_result.json`: normalized status and pointers to artifacts.
- `moe_profile.jsonl`: SEW/MoE runtime profile events when enabled.
- `npu_samples.jsonl`: bounded-rate physical HBM and NPU-utilization samples,
  including a post-release sample.
- `release_ack.json`: portable unit-local release proof (the external custody
  directory retains the same acknowledgement).
- `moe_trace.jsonl`: routed expert trace events when enabled.
- `moe_router_parity.jsonl`: bounded eager native/seam router snapshots when
  `--router-parity` is enabled.
- `router_parity_report.json`: fail-closed parity result for the corresponding
  unit; absent records are a failure, not a pass.
- `PASSED.txt` or `FAILED.txt`: fail-closed unit disposition.

Every `unit_manifest.json` also records a checkpoint-derived capability
descriptor digest, Phase-A registry digest/row match, and router-parity
coordinates. A config-derived descriptor can contain unresolved router
ownership; it is provenance rather than a substitute for the materialized
runner's capability guard.

Failed units still write `unit_result.json` and `FAILED.txt`; expected OOM or
graph-capture failures are evidence, not missing data. A request-success result
is downgraded to failure when the manager cannot prove service release.
# Managed graph-mode launch

Non-dry suite runs start and stop the service only through a pinned repository
manager. The default container backend uses
`third_party/vllm-hust-dev-hub/manage.sh`, an immutable image digest, and a
unique container. On a systemd-less host, `--managed-backend locked-host` uses
`benchmark/scripts/manage_locked_host_runtime.py` with an explicit Python and
the compatibility-locked vLLM/vLLM-Ascend checkouts. Both backends require a
physical NPU5 or NPU6 and a release-ack directory. The runner rejects occupied
ports/devices, active custody state, forced eager, and external manager paths.
`--no-start-server` remains a probe mode and must not be labeled
repository-owned online evidence.

## Issue #7 graph correctness bundle

`run_issue7_graph_bundle.py` enforces the required order: one-request ShareGPT
smoke, a short mixed-chat gate, then three independent managed service starts.
Every stage runs PIECEWISE ACLGraph only and is checked by
`verify_issue7_graph_unit.py`. The verifier requires non-empty output, graph
capture and replay, fixed slot-address equality, generation-protected compute,
H2D-before-mapping publication, capacity-bounded multi-wave prefill, complete
provenance, exact token IDs against the preceding gate, physical HBM samples,
and a successful release ACK. It also emits H2D copy enqueue, waiting/event,
slot update, wave-prefill compute, stage issue/wait, and sampled Graph replay
issue timing. Replay issue time is CPU-side dispatch timing and does not imply
device-kernel execution time.

Example for a locked host stack:

```bash
python benchmark/scripts/run_issue7_graph_bundle.py \
  --output-root benchmark/artifacts/issue7-npu5 \
  --device 5 \
  --python /workspace/latchmoe-issue13-venv/bin/python \
  --vllm-root /workspace/latchmoe-issue13-stack/vllm-hust \
  --seam-root /workspace/latchmoe-issue13-stack/vllm-ascend-hust \
  --manifest benchmark/artifacts/workloads/issue13_sharegpt.jsonl \
  --model-path /root/data/shared_models/strict-models/Qwen3-30B-A3B \
  --dataset-path /root/data/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Issue #17 matched TTFT campaign

`run_issue17_matched_ttft.py` runs three independent `full_layer` and three
independent `multi_wave` PIECEWISE Graph starts in a fixed AB/BA/AB order. Both
arms use the same 200-request `mixed_chat` manifest; the only case difference
is `VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE`. The dedicated verifier derives
TTFT/TPOT p50 and p95 from every raw request, requires exact token IDs across
all units, rejects multi-wave fallback, checks NPU/ACL/OOM markers and release
ACKs, and fails if provenance differs between arms.

After a passing campaign, `package_issue17_evidence.py` copies only the six
declared formal units into a portable archive, rewrites the campaign unit paths
relative to the bundle root, includes the fixed workload manifest, and emits a
SHA-256 list for every packaged file. It refuses to package a campaign without
both a passing summary and `PASSED.txt`.
