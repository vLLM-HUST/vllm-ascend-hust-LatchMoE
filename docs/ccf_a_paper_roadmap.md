# CCF-A Systems Paper Roadmap

## Target

Target format: English full paper for a CCF-A-level systems venue.

Working thesis:

> SEW-Offload makes dynamic MoE expert offloading compatible with Ascend
> ACLGraph replay by virtualizing logical experts into compute-protected,
> fixed-address physical slots and moving dynamic staging decisions outside the
> graph replay boundary.

The paper should be positioned as a technique/system paper, not as a vLLM patch
report. The main artifact is a graph-compatible MoE offloading runtime for
memory-constrained Ascend NPU inference.

## Paper Shape

| Section | Role | Key claim |
|---|---|---|
| 1 Introduction | Problem and motivation | Dynamic MoE offloading conflicts with ACLGraph replay stability. |
| 2 Background and Motivation | Ascend graph replay, MoE serving, offloading | Existing offloading paths do not expose a graph-stable expert state abstraction. |
| 3 Overview | Runtime boundary and abstractions | SEW-Offload separates eager expert staging from captured MLP execution. |
| 4 Design | Core mechanisms | Fixed slots, persistent `log2phy`, B2 prefill waves, and protected slot lifecycle. |
| 5 Implementation | vLLM-Ascend integration | Router-stage-MLP seam, custom ops, AutoConfig, CPU-first loading, profiling. |
| 6 Evaluation | Performance and correctness evidence | SEW-Offload expands feasible serving under tight HBM budgets while preserving replay-friendly execution. |
| 7 Related Work | Systems context | MoE serving, offloading, CUDA/ACL graph execution, heterogeneous memory runtimes. Must position against FluxMoE (address-stable, graph-agnostic) and graph-safe reconfiguration work; see Novelty Verification. |
| 8 Conclusion | Takeaway | Dynamic MoE offloading can be made graph-compatible through slot-stable virtualization. |

## Code-to-Paper Mapping

| Paper module | Code path | Paper treatment |
|---|---|---|
| Plugin integration | `vllm_moe_offload_ascend/__init__.py`, `patches/patch_fused_moe.py` | Implementation detail. Explain how the runtime is inserted into vLLM-Ascend. |
| Router-stage-MLP seam | `ops/fused_moe/moe_router_op.py`, `moe_offload_stage_op.py`, `moe_mlp_op.py` | Core design. This is the graph-compatible execution boundary. |
| Fixed-slot runtime | `moe_offload/runtime.py`, `slot_bank.py`, `slot_mapping.py` | Core design. This supports the fixed-address logical-to-physical expert abstraction. |
| Host expert store | `moe_offload/host_store.py`, `cpu_first_loader.py` | Implementation and memory optimization. Keep secondary unless evaluation shows major impact. |
| Transfer engine | `moe_offload/transfer_engine.py` | Core implementation. Use it for H2D overhead and overlap analysis. |
| B2 prefill waves | `moe_offload/phase_split.py`, B2 hooks in `patch_fused_moe.py` | Core design. This addresses active expert sets larger than slot capacity. |
| AutoConfig | `moe_offload/autoconfig.py`, `autoconfig_advisor.py` | Usability layer and evaluation setup. Do not make it the main novelty. |
| Profiling and demo harness | `tools/run_annual_demo_suite.py`, `tools/analyze_ascend_moe_profile.py` | Reproducibility and experiment pipeline. |

## Main Claims to Prove

1. Feasibility: SEW-Offload enables MoE models to run under HBM budgets where
   no-offload execution fails.
2. Graph compatibility: the decode hot path can replay with stable slot tensor
   addresses and persistent mapping buffers, while dynamic staging happens
   before replay.
3. Performance (graph-capture value under a fixed peak-HBM budget): under a
   fixed peak-HBM budget explicitly partitioned into a reserved KV region and an
   expert-slot region, SEW-Offload sustains ACLGraph replay during MoE offload
   while native prefetch and legacy offload must fall back to eager execution.
   This splits into two sub-claims:
   - **(3a) Capability**: only SEW-Offload keeps graph capture alive during
     dynamic offload; native prefetch and legacy paths crash under capture
     (`stream not joined` / `Not allow to synchronize captured-stream`). Proven
     directly by the Figure 1 motivation, no extra experiment.
   - **(3b) Performance lift**: with peak HBM held equal, SEW-Offload (capture
     on) improves median/p99 TPOT and decode throughput by [X] over the best
     eager offload baseline, and by [Y] over a SEW-Offload-with-capture-disabled
     ablation.
   "Same memory budget" is defined as a fixed peak HBM; the two orthogonal
   sweep axes are offload budget and the KV/slot partition. Depends on
   verifying (Phase 1A) that the baselines run in eager mode.
4. Prefill robustness: B2 waves handle prefill active sets that exceed the slot
   budget without requiring all active experts to be resident at once.
5. Correctness: protected slot states prevent overwriting weights that are still
   consumed by compute streams.

## Novelty Verification

Verification pass completed 2026-07-01 via literature search. Purpose: confirm
whether any prior work already makes dynamic MoE expert offloading compatible
with graph-capture replay. This gate resolves the F1/F2 (novelty vs. closest
prior work, venue fit) risk that was previously marked `unverified`.

### Novelty axis under test

The only defensible novelty axis is: **routing-driven dynamic expert offload
made compatible with graph-capture replay (CUDA Graph / Ascend ACLGraph), by
holding physical slot addresses and `log2phy` buffers stable across replay and
moving staging outside the replay boundary.** Memory savings alone, MoE
offloading alone, and Ascend porting alone are not novel.

### Verdict: NOVELTY HOLDS, but PARTIAL

No published work claims dynamic offload + graph-replay compatibility as a
contribution, and the Ascend ACLGraph + offload combination is absent from both
literature and production systems. However, one 2026 paper (FluxMoE)
independently built a mechanically similar address-stable design, so the paper
must cite it and carve an explicit delta or a reviewer will reframe the
contribution as a port.

### Evidence map

| Bucket | Systems | Implication for SEW-Offload |
|---|---|---|
| Offload, eager, no graph | fMoE (arXiv 2502.05370, HF Transformers prototype), MoE-Infinity (2401.14361), MoE-Lightning (2411.11217), HOBBIT (2411.01433), ProMoE (2410.22134), Fiddler (ICLR'25) | Full-text checks found no CUDA-graph / graph-capture engagement. Dynamic movement keeps them in eager mode. This is the gap SEW-Offload fills. |
| Graph, no offload (all experts resident) | Production vLLM / SGLang / TensorRT-LLM; vLLM-Ascend ACLGraph | vLLM forum confirms dynamic expert offload is unsupported and all experts must be resident to use graph mode (issue #38256 is an open request). Validates the either/or tension. |
| Address-stable, graph-agnostic (THREAT) | FluxMoE (arXiv 2604.02715, Apr 2026): PagedTensor reserves stable virtual addresses, binds physical blocks pre-launch, treats MoE as static compute graph + streamed params | Mechanically closest prior art. Does NOT claim graph-capture-replay compatibility, is GPU/PyTorch, and is motivated by KV-cache memory reclaim, not replay stability. Must be cited and differentiated. |
| Graph-safe dynamic reconfiguration (support + related work) | "Surviving Partial Rank Failures in Wide EP MoE" (arXiv 2605.10670): states "CUDA-graph execution and online reconfiguration are structurally opposed"; graphs freeze pointer identities at capture | Independent confirmation the tension is real and recognized. Solves it for EP membership changes on failure, not for offload staging. |
| Ascend MoE, orthogonal | Relay Buffer comms (2605.06055), Ascend MoE training (2505.04519), Multi-core interleaved scheduling (2605.23764) | None combines ACLGraph with expert offload. Ascend ACLGraph + dynamic offload is unoccupied. |

### Defensible delta against FluxMoE (required in Related Work and Introduction)

1. **Goal**: SEW-Offload targets graph-**capture/replay** compatibility as the
   stated objective (staging moved outside the replay boundary); FluxMoE never
   mentions graph capture, eager mode, or replay, and is motivated by reclaiming
   GPU memory for KV cache.
2. **Platform**: Ascend ACLGraph realization; FluxMoE is NVIDIA GPU / PyTorch /
   Triton.
3. **Staging trigger**: per-request routing-driven staging plus
   compute-protected slot lifecycle (LOADING/READY/COMPUTING with cross-stream
   events); FluxMoE uses memory-budget-driven paging without a compute-protected
   overlap guard.
4. **Prefill overflow**: capacity-bounded B2 waves for active sets larger than
   the slot budget; not addressed by FluxMoE.

### Actions this gate forces

1. Add FluxMoE (2604.02715) and arXiv 2605.10670 as primary Related Work
   entries; position SEW-Offload as "graph-replay-safe dynamic offload" between
   FluxMoE (address-stable but graph-agnostic) and 2605.10670 (graph-safe
   reconfiguration but for EP failure, not offload).
2. Correct the baseline name: the paper's "FineMoE" is **fMoE** (arXiv
   2502.05370). Fix in `docs/evaluation_design_notes.md`.
3. Novelty claim in the Introduction must be phrased as graph-**replay**
   compatibility, not generic "offloading," to stay outside FluxMoE's territory.

### Residual unverified items

- SwapMoE, EdgeMoE, Klotski, Mixtral-Offloading, ktransformers, MoBiLE
  (2510.12357), DAOP (2501.10375): believed eager, not full-text confirmed for
  graph usage. Confirm before final submission if any is used as a baseline.
- All arXiv IDs above are search-derived; confirm exact titles/venues at
  citation time and do not cite any that cannot be re-fetched.

## Evaluation Matrix

Detailed related-work-derived evaluation notes live in
`docs/evaluation_design_notes.md`.

### Baselines

Under graph capture, native prefetch and legacy paths crash by design (this is
the Figure 1 motivation, not a bug to fix). The fair performance comparison is
therefore against these baselines in **eager** mode, plus an internal
capture-disabled ablation, all at the same peak HBM.

| Baseline | Mode | Purpose |
|---|---|---|
| No offload | — | Capacity and HBM feasibility probe (expected OOM under tight budget). |
| Native prefetch | eager | Fair performance comparison; also shown crashing under capture (Fig. 1). |
| Legacy/layered offload path | eager | Fair performance comparison; also shown crashing under capture (Fig. 1). |
| SEW-Offload, capture disabled | eager | Internal ablation isolating the graph-capture benefit (the [Y] in Claim 3b). |
| SEW-Offload | capture on | Full system. |

### Configurations

| Dimension | Required values |
|---|---|
| Model | Qwen3-30B-A3B first. Add another MoE model only if engineering cost is manageable. |
| Hardware | Single-card Ascend 910B-class NPU. |
| Device selection | Do not use NPU 0-3 for experiments. Select only from NPU 4-7, preferring an idle card with enough free HBM before each run. |
| Memory budget | 14 GiB and 28 GiB offload budgets, each at a fixed peak HBM. Sweep the KV/slot partition as an orthogonal axis. |
| Slot count | AutoConfig (KV-aware) values plus sensitivity around 8, 16, 32, 64 slots if feasible. |
| Workload | Decode-heavy, prefill-heavy, and mixed ShareGPT-style prompts. |
| Concurrency | **Main experiment is single-request (`max_num_seqs=1`)**, following HOBBIT (all-batch=1) and MoE-Infinity (single-card, low RPS). Concurrency (2/4 routes) is a secondary "does the fixed-slot mechanism stay correct and not crash under concurrency" section, not a throughput contest. KV floor reserves for `kv_reserve_seqs=4` so this section runs without re-capture. |

### Metrics

| Metric | Why it matters |
|---|---|
| Success/OOM | Shows memory-constrained feasibility. |
| Peak HBM | Connects performance to memory pressure. |
| TTFT | Captures prefill and first-token overhead. |
| TPOT | Main decode hot-path metric. |
| Output throughput | End-to-end serving throughput. |
| H2D bytes and staging time | Explains offload overhead. |
| Wave count and active expert count | Explains B2 behavior. |
| Slot hit/miss rate | Explains fixed-slot reuse quality. |
| Graph replay eligibility | Supports the ACLGraph-specific claim. |

### Ablations

| Ablation | Expected question answered |
|---|---|
| Graph capture on vs off (SEW-Offload) | How much does graph replay buy over eager execution of the same offload path? This isolates the [Y] in Claim 3b. |
| Without B2 prefill waves | Are waves necessary for long prompts or wide routing? |
| Without transfer-aware wave scheduling | Does scheduling matter beyond correctness? |
| Without compute-protected slot lifecycle | Is READY-only slot reuse unsafe under overlap? Must reproduce a concrete wrong-output or crash signal, otherwise Claim 5 is unfalsifiable. |
| KV/slot partition sweep (KV-aware AutoConfig) | How does trading HBM between reserved KV and expert slots move hit rate and TPOT? Defines the "same peak HBM" controlled variable. |
| Different slot budgets | How sensitive is the method to HBM allocation? |
| Without CPU-first loading | Is CPU-first a supporting memory optimization or a core result? |

## Figure Plan

### Figure 1: Motivating Example

Use an "Existing vs Ours" layout.

Left panel: dynamic MoE offload path where expert IDs, H2D movement, and mapping
updates sit inside or across the graph replay boundary, causing unstable replay
conditions.

Right panel: SEW-Offload path where the router produces logical expert IDs,
stage updates fixed physical slots and persistent `log2phy` before replay, and
the captured MLP reads stable addresses.

### Figure 2: System Overview

Use a system architecture diagram:

1. Router and active expert discovery.
2. Stage op and runtime decision logic.
3. CPU host expert store.
4. Fixed NPU slot bank.
5. Persistent `log2phy` buffer.
6. Captured grouped MLP replay.
7. Optional B2 prefill wave path.

### Experimental Figures

Generate with Matplotlib/Seaborn from raw result files.

1. Performance comparison: grouped bars for TPOT, TTFT, throughput.
2. Memory feasibility: peak HBM and OOM/success table.
3. B2 prefill behavior: active experts, slot budget, wave count.
4. Ablation chart: full SEW-Offload vs removed components.
5. Sensitivity plot: slot count vs performance and memory.

### Image Generation Policy

Generated bitmap images are allowed for concept exploration, visual inspiration,
or raster-style illustrations, but they should not replace deterministic paper
figures when the figure is better expressed as vector diagrams or scripted
plots.

Before any image-generation call:

1. Decide whether generation is necessary. Figure 1 and Figure 2 should first be
   sketched as structured vector diagrams; experimental figures must be
   generated from data with Matplotlib/Seaborn.
2. Write a polished prompt first, including the figure's role, target venue
   style, exact text if any, layout constraints, and avoid list.
3. Prefer one image per call (`n=1`) and inspect the result before iterating.
4. Record the final prompt, tool, model, output path, and any manual edits.
5. Save project-bound assets inside the repository; do not leave final assets
   only in a tool-specific generated-image directory.

Tool choice:

| Tool | Use when |
|---|---|
| Built-in `imagegen` | A normal raster asset is useful and no direct NowCoding API path is required. |
| `nowcoding-image-generation` | A raster asset should be generated through the NowCoding OpenAI-compatible `gpt-image-2` image API. |
| draw.io / PowerPoint / Figma | Structured mechanism diagrams such as Figure 1 and Figure 2. |
| Matplotlib / Seaborn | Quantitative experimental figures. |

Cost rule: no image API call should be made until the prompt is specific enough
that a reviewer could infer the intended composition from the text alone.

## Immediate Work Plan

Phase 0A (novelty gate) is **done** (see Novelty Verification: novelty HOLDS but
PARTIAL; FluxMoE must be differentiated). Phase 0B's static KV-aware slot cap is
implemented in AutoConfig; the remaining critical path is:
`1A eager/capture matrix -> 1B define "same peak HBM" -> 2A/2B core experiments`.

### Phase 0B: KV-aware slot cap (static cap and runtime backstop implemented)

1. Done: `derive_num_slots_defaults` subtracts KV floor, activation reserve, and
   B2 staging-buffer capacity before capping slot count. It reuses
   `estimate_kv_cache_gib`.
2. Done: `kv_reserve_seqs` (default 4) and `kv_reserve_ctx`
   (default `min(max_model_len, 8192)`) are decoupled from `max_num_seqs=1`.
3. Done: runtime fail-fast backstop wraps vLLM KV-capacity checks and the
   resolved KV-cache report path; when MoE offload slots make one full request
   impossible, the error explicitly says to reduce slots/offload budget or
   request shape.
4. Done: regression tests sweep HBM budgets and lock the reserve-aware slot cap,
   including a Qwen3-like KV shape that keeps 14 GiB at 32 slots while capping
   the old unsafe 28 GiB / 64-slot choice.
5. Red line: this is a static init-time decision. Never make the slot cap
   runtime-adaptive; changing slot count re-triggers graph capture and breaks
   the paper's core thesis.

### Phase 1: Make Claim 3 measurable (~2 days)

1. (1A) Verify the hypothesis: run native prefetch and legacy with graph capture
   disabled (eager) and confirm they produce tokens. Record a
   {baseline} x {capture on/off} success/crash matrix.
2. (1B) Define the controlled variable: fixed peak HBM, KV region and slot
   region explicitly split (built on 0B).
3. (1C) Reframe the baseline crashes as the Figure 1 motivation; add the
   SEW-Offload-capture-disabled ablation as the fair eager performance baseline.

### Phase 2: Core evidence (~5-7 days)

Main experiment is **single-request (batch=1)**; concurrency is a secondary
section.

1. (2A) Claim 3 alignment experiment at fixed peak HBM: SEW (capture on) vs
   {SEW capture off, native prefetch eager, legacy eager}. Report TTFT/TPOT/
   decode throughput median + p99. Fills [X]/[Y].
2. (2B) Orthogonal budget sweep: offload budget x KV/slot partition; plot hit
   rate and TPOT vs KV headroom.
3. (2C) Claim 5 decisive experiment: without compute-protected slot lifecycle,
   reproduce a concrete wrong-output or crash; with it, correct.
4. (2D) Concurrency section: 2/4 routes, show the fixed-slot mechanism stays
   correct and does not crash. One figure. Not a throughput contest.
5. (2E) Graph replay eligibility: capture + replay N decode steps, show slot
   tensor addresses stable across replays, using `trace_collector.py`.

### Phase 3: Writing (~3-4 days)

1. Rewrite this roadmap's remaining sections around the finalized story.
2. Write Sections 2-5; write Evaluation after plots are script-generated.
3. Follow the Zhang-style conventions: C1/C2/C3 <-> D1/D2/D3 numbering, paired
   numbers with honest cost disclosure (state the low-concurrency, 1.5-3.9 tok/s
   regime openly as scope), Figure 1 from real error messages.
4. Run a pre-submission review focused on novelty (FluxMoE delta), correctness,
   and experiment sufficiency.

### Out of scope for v1

Tier 3 online arrival-rate traces (Poisson / Azure) are **cut from v1** and
listed as future work. They belong to the vLLM/fMoE "serving maturity" paradigm,
are orthogonal to the core claims, and the closest neighbor (HOBBIT) does not do
them.

Execution note: the first Week-2 smoke loop is summarized in
`docs/week2_execution_summary.md`. It found the old 28 GiB AutoConfig slot
choice was not KV-cache-safe at `max_model_len=4096`; the static AutoConfig cap
now accounts for KV reserve, activation reserve, B2 staging buffers, and a
physical slot-bank cap for 64 GiB 910B-class cards. 2026-07-02 smoke rerun:
`sew_28gb_autoslots` selected 32 slots and succeeded with 5.20 GiB available KV
cache (`benchmark/artifacts/runs/sew-offload-ascend-v1-20260702T052334Z`).
The smoke eager/capture matrix is in
`benchmark/artifacts/runs/sew-offload-ascend-v1-20260702T053049Z`: SEW 14 GiB
capture-on/off and SEW 28 GiB slots32 capture-on/off all succeeded; native
prefetch capture-on and both legacy layered cases failed.

## Verification Command

Current local test command:

```bash
/root/miniconda3/bin/conda run -n vllm-hust-dev python -m pytest tests -q
```

Observed result on 2026-06-30:

```text
109 passed, 3 warnings in 17.92s
```

## Integrity Rules

1. Do not claim graph compatibility without either code-level evidence or a
   captured/replayed trace.
2. Do not report exploratory benchmark numbers as final paper results.
3. Do not hide unsupported boundaries: single-card, unquantized MoE, and
   Qwen3-30B-A3B should be stated clearly. State the single-request (batch=1)
   main-experiment scope openly (following HOBBIT / MoE-Infinity); do not imply
   high-throughput large-batch serving is a target.
4. Do not describe CPU-first loading as a main contribution unless ablation
   shows it is central.
5. Treat every result as invalid until the exact config, commit, environment,
   and raw output file are recorded.
6. Do not claim novelty of "dynamic MoE offloading" in general. The verified
   novel axis is graph-replay-safe dynamic offload on Ascend ACLGraph; FluxMoE
   (arXiv 2604.02715) already holds address-stable expert tensors on GPU, so the
   Introduction and Related Work must cite it and state the delta (see Novelty
   Verification).
