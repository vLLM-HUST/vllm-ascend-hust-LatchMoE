# Evaluation Design Notes from Related Work

This note translates experiment patterns from recent LLM serving and MoE
offloading papers into an executable evaluation plan for SEW-Offload.

## Related-Work Experiment Patterns

| Work | What they evaluate | Useful lesson for SEW-Offload |
|---|---|---|
| vLLM / PagedAttention | OPT and LLaMA models; ShareGPT and Alpaca length traces; request-rate sweep; normalized latency, throughput, memory batching behavior. Source: https://arxiv.org/abs/2309.06180 | Serving papers should show latency-throughput curves under realistic prompt/output length distributions, not only one fixed batch. |
| FlexGen | OPT models under limited GPU memory; offload baselines; throughput-latency frontier; HELM benchmark execution under commodity GPU constraints. Source: https://arxiv.org/abs/2303.06865 | Offload papers should make the resource constraint explicit and report feasibility, memory placement, and throughput together. |
| MoE-Infinity | Switch, NLLB-MoE, Mixtral, Arctic; BIGBench/FLAN/MMLU tasks; Azure-style arrivals; DeepSpeed-Inference, Llama.cpp, Mixtral-Offloading, BrainStorm baselines; GPU blocking time and bandwidth. Source: https://arxiv.org/abs/2401.14361 | MoE offloading papers should include expert movement metrics, not just request latency. |
| fMoE | Mixtral, Qwen1.5-MoE, Phi-3.5-MoE; LMSYS-Chat-1M and ShareGPT; Azure traces; TTFT, TPOT, expert hit rate, overhead, cache-limit sensitivity, ablation. Prototyped on HuggingFace Transformers (eager, no graph capture). Source: https://arxiv.org/abs/2502.05370 | The strongest recent MoE-offload papers separate prefill and decode metrics and evaluate hit/miss behavior. Eager execution: does not engage graph replay, so it is a metric-design reference, not a graph-compatibility competitor. |
| MoE-Lightning | Mixtral 8x7B, Mixtral 8x22B, DBRX; T4/L4 and multi-T4 hardware; MTBench and HELM-style long-prompt tasks; generation throughput; batch/output-length sensitivity. Source: https://arxiv.org/abs/2411.11217 | Throughput-oriented offload papers justify workloads by prompt length and generation length regimes. |
| HOBBIT / CommitMoE-style work | Representative MoE models under constrained-device or GPU-memory-limited setups; speed and model-quality preservation. Sources: https://arxiv.org/abs/2411.01433 and https://ojs.aaai.org/index.php/AAAI/article/view/39454 | If the inference procedure changes numerical behavior, quality benchmarks are mandatory. SEW-Offload should instead show output equivalence because it should not alter model semantics. |
| FluxMoE (closest prior mechanism) | Expert paging via PagedTensor: stable virtual addresses for expert tensors, physical blocks bound pre-launch, MoE treated as static compute graph + streamed params. GPU / PyTorch / Triton; motivated by reclaiming GPU memory for KV cache. Source: https://arxiv.org/abs/2604.02715 | Head-to-head related work: mechanically nearest to slot-stable virtualization but graph-agnostic (no CUDA-graph / capture / replay claim) and GPU-only. SEW-Offload must report the delta: graph-replay-safe goal, Ascend ACLGraph, per-request compute-protected staging, and B2 wave overflow. See Novelty Verification in ccf_a_paper_roadmap.md. |
| Surviving Partial Rank Failures in Wide EP MoE | States "CUDA-graph execution and online reconfiguration are structurally opposed"; graphs freeze pointer identities at capture; keeps graph structure intact while changing routing/peers on failure. Source: https://arxiv.org/abs/2605.10670 | Independent confirmation the graph-vs-dynamic-state tension is real. Solves it for EP membership changes on failure, not for offload staging. Cite as motivation support and adjacent related work. |

## Recommended Model Set

### Primary model: Qwen3-30B-A3B

Use Qwen3-30B-A3B as the main model.

Rationale:

1. It is the model already supported by the repository's compatibility boundary.
2. It is a modern MoE LLM with enough experts to make expert-slot management
   meaningful: 30.5B total parameters, 3.3B activated parameters, 48 layers,
   128 experts, and 8 activated experts according to the public model card.
3. It is practical for single-card Ascend 910B-class evaluation while still
   creating real memory pressure.

### Optional secondary model

Only add a second model after the Qwen3-30B-A3B story is stable.

Preferred secondary role:

| Candidate | Use only if | Why |
|---|---|---|
| Qwen1.5-MoE | vLLM-Ascend and this plugin support its MoE implementation with low engineering risk. | Shows the method is not hard-coded only for Qwen3. |
| DeepSeek-V2-Lite-MoE | It runs correctly on the local Ascend stack without broad model-specific patches. | Gives a second routing/expert shape. |
| Mixtral 8x7B | Ascend backend and plugin support are confirmed. | Makes comparison to GPU MoE-offload literature easier. |

Do not spend early time forcing broad model coverage. For this paper, a strong
single-model, single-platform system story is better than weak multi-model
coverage with fragile support.

## Recommended Dataset and Workload Stack

### Tier 1: Realistic serving prompts

Use ShareGPT as the default workload source because it is already in the local
benchmark harness and is widely used by LLM serving papers.

Required splits:

| Workload name | Prompt selection | Max output tokens | Purpose |
|---|---|---|---|
| Mixed-Chat | Random ShareGPT prompts after tokenization filtering | 100 or 128 | Main end-to-end serving result. |
| Decode-Heavy | Short prompts, fixed longer generation | 128 or 256 | Stresses decode hot path and ACLGraph replay benefit. |
| Prefill-Heavy | Long prompts, short generation | 16 or 32 | Stresses B2 prefill waves and active expert overflow. |
| Long-Context-Prefill | Longest safe prompt bucket under current `max_model_len` | 16 | Stress test for wave count, H2D, and TTFT. |

### Tier 2: Controlled synthetic prompt lengths

Use controlled prompt-length buckets after the ShareGPT result is available.

Recommended buckets:

| Bucket | Input tokens | Output tokens | Purpose |
|---|---:|---:|---|
| Short | 64 | 128 | Decode-dominant path. |
| Medium | 512 | 128 | Balanced serving. |
| Long | 2048 | 32 | Prefill-dominant path. |
| Max-safe | 4096 or current stable max | 16 | B2 stress and memory edge. |

The controlled buckets should be generated with real text when possible. If
synthetic repeated text is used, mark it as synthetic and do not use it as the
main end-to-end claim.

### Tier 3: Arrival-rate or online serving trace (CUT FROM v1)

**Decision (2026-07-01): cut from v1, listed as future work.** Tier 3 belongs to
the vLLM/fMoE "serving maturity" paradigm (online arrival, normalized-latency-
vs-request-rate). It is orthogonal to SEW-Offload's core claims (graph
compatibility, memory feasibility, correctness), and the closest neighbor
(HOBBIT) does not do it. The main experiment is single-request (batch=1); see
the concurrency note in E7. If revisited later:

1. Poisson arrivals with request-rate sweep, following the vLLM-style setup.
2. Optional Azure-style trace replay if the trace is available locally and the
   license allows use.

### Tier 4: Quality and correctness guard

SEW-Offload should not change model semantics. Therefore, the main quality
experiment is not MMLU/GSM8K performance. The main correctness guard is:

1. Same prompts, same decoding parameters, same seed where applicable.
2. Compare outputs or logits against a non-offload or full-resident path on a
   small sample.
3. Report exact-match for deterministic decoding or bounded numerical
   difference for logits.

Add LM Harness tasks only if reviewers may suspect quality degradation. Use a
small sanity suite rather than turning the paper into a model-quality paper.

## Metrics

### End-to-end serving metrics

| Metric | Report as | Claim supported |
|---|---|---|
| TTFT | median, p90, p99 | Prefill overhead and B2 behavior. |
| TPOT | median, p90, p99 | Decode hot-path performance. |
| Output throughput | tokens/s | Overall serving efficiency. |
| Request latency | median, p90, p99 | Per-request latency at batch=1 (main); also the concurrency-robustness section. |
| Success/OOM | per case | Memory-constrained feasibility. |

### Memory and offload metrics

| Metric | Report as | Claim supported |
|---|---|---|
| Peak HBM | GiB and percentage of device memory | Feasibility under fixed budget. |
| CPU expert-store memory | GiB | Cost shifted to host memory. |
| H2D bytes | per request, per token, per layer if available | Expert movement cost. |
| Staging time | average and tail | Graph-external overhead. |
| Slot hit/miss rate | per layer and aggregate | Fixed-slot reuse quality. |
| Active expert count | distribution per layer/phase | Why B2 waves are necessary. |
| Wave count | distribution | B2 prefill capacity behavior. |

### Graph-compatibility metrics

| Metric | Report as | Claim supported |
|---|---|---|
| Captured/replayed MoE path status | yes/no plus evidence | ACLGraph compatibility. |
| Graph breaks around MoE | count or trace evidence | Existing dynamic offload mismatch. |
| Decode host overhead | per token or timeline | Benefit of keeping hot path replayable. |
| Stable slot/log2phy addresses | trace or debug assertion | Fixed-address abstraction correctness. |

## Required Experiments

### E1: End-to-end performance under memory budgets

Compare:

1. No offload capacity probe.
2. Native prefetch baseline.
3. Legacy/layered offload path if still supported.
4. SEW-Offload 14 GiB.
5. SEW-Offload 28 GiB.

Run on Mixed-Chat, Decode-Heavy, and Prefill-Heavy workloads.

Primary figure: grouped bars for TTFT, TPOT, and output throughput.

### E2: Memory feasibility and cost shifting

Show:

1. Which systems OOM or fail to serve at each budget.
2. Peak HBM.
3. CPU expert-store memory.
4. Slot-bank memory footprint.

Primary figure: memory stacked bar plus success/OOM table.

### E3: ACLGraph compatibility evidence

Show that SEW-Offload moves dynamic staging outside the captured MLP replay
boundary:

1. Router-stage-MLP execution trace.
2. Stable slot tensors and persistent `log2phy` buffer.
3. Stage op no-op during capture and active before replay.
4. Decode hot-path timing with and without SEW.

Primary figure: timeline or trace-backed diagram.

### E4: B2 prefill overflow stress

Compare:

1. SEW without B2.
2. SEW with B2.
3. SEW with B2 plus transfer-aware schedule.

Use Prefill-Heavy and Long-Context-Prefill workloads.

Report active experts, slot budget, wave count, TTFT, H2D bytes, and staging
time.

### E5: Ablation study

Minimum ablations:

| Variant | Expected evidence |
|---|---|
| Full SEW-Offload | Best complete system. |
| No B2 wave prefill | Fails or slows on active-set overflow. |
| No transfer-aware scheduling | Similar correctness, worse prefill or stage overlap. |
| No CPU-first loading | Higher load-time HBM pressure or worse startup feasibility. |
| Smaller slot count | Higher miss/staging overhead. |

The compute-protected lifecycle is mostly a correctness mechanism. If disabling
it is unsafe, use stress tests and assertions instead of a knowingly unsafe
performance run.

### E6: Slot-count and budget sensitivity

Run slot counts around the AutoConfig choice:

| Offload budget | Slot counts |
|---|---|
| 14 GiB | 8, 16, 32 |
| 28 GiB | 32, 64 |

Report TPOT, TTFT, H2D bytes, hit/miss rate, and wave count.

### E7: Concurrency robustness (secondary section, not a throughput contest)

**Decision (2026-07-01): the main experiment is single-request (`batch=1`),
following HOBBIT (all-batch=1) and MoE-Infinity (single-card, low RPS).**
SEW-Offload sells graph compatibility, memory feasibility, and correctness, not
throughput, so it does not enter the FlexGen / MoE-Lightning large-batch arena.

1. `max_num_seqs=1` is the headline setting for all Claim 3 / Claim 5 results.
2. Add a single concurrency section at 2/4 routes whose goal is to show the
   fixed-slot mechanism stays correct and does not crash under concurrency, and
   how slot hit rate shifts. One figure. Not a throughput maximization.
3. KV floor reserves for `kv_reserve_seqs=4` at init so this section runs
   without changing slot count (which would re-trigger graph capture).

### E8: Correctness and regression

Report:

1. Unit-test result.
2. Output/logit equivalence check on a small prompt set.
3. Slot mapping validation.
4. No stale `log2phy` and no slot overwrite under stress.

## Result Figures

| Figure | Chart type | Data source |
|---|---|---|
| Figure 6: End-to-end performance | Grouped bars | Benchmark JSONs. |
| Figure 7: Memory feasibility | Stacked bars + OOM table | Server logs and profile JSONL. |
| Figure 8: B2 prefill behavior | Line or grouped bars | Profile JSONL wave events. |
| Figure 9: Slot sensitivity | Line plot | Slot-count experiments. |
| Figure 10: Runtime breakdown | Stacked bars or timeline | Profiling events. |

All experimental plots should be generated from raw artifacts with
Matplotlib/Seaborn. Do not hand-edit numerical figures.

## What Not to Do

1. Do not use MMLU/GSM8K as the main evidence. The system does not claim better
   model quality.
2. Do not compare against GPU-only MoE systems that cannot run under the same
   memory budget unless explicitly labeled as an upper bound.
3. Do not hide unsupported model/backend boundaries.
4. Do not over-invest in unsupported models before the Qwen3-30B-A3B result is
   complete.
5. Do not report only average latency. Tail latency matters for serving papers.
6. Do not call exploratory smoke results final paper numbers.

## Minimum Strong Evaluation for a First Submission Draft

If time is limited, the minimum defensible evaluation is:

1. Qwen3-30B-A3B on single-card Ascend 910B-class NPU.
   Runtime constraint: do not use NPU 0-3; select only from NPU 4-7.
2. ShareGPT Mixed-Chat, Decode-Heavy, and Prefill-Heavy workloads.
3. No-offload, native prefetch capture-on failure, native prefetch eager,
   legacy/layered capture-on failure if available, legacy/layered eager,
   SEW capture-disabled, SEW 14 GiB, and SEW 28 GiB.
4. Metrics: TTFT, TPOT, throughput, peak HBM, H2D bytes, slot hit/miss, active
   expert count, wave count, success/OOM.
5. Ablations: no B2, no transfer-aware schedule, slot-count sensitivity,
   CPU-first loading.
6. Correctness: unit tests plus output/logit equivalence on a small prompt set.
7. Graph evidence: trace or profiler artifact showing dynamic staging outside
   graph replay and stable MLP-side slot addresses.

This minimum set directly supports the paper's central claim: SEW-Offload makes
dynamic MoE expert offloading practical under Ascend ACLGraph constraints.
