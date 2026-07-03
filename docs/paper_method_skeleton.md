# Paper Method Skeleton

## Paper-Type Positioning

- Type: Technique Paper.
- Rationale: The paper proposes a runtime mechanism for an existing systems
  problem: efficient MoE serving on memory-constrained Ascend NPUs.

## Thinking Template

| Stage | Content |
|---|---|
| Research background | MoE LLM serving is attractive because sparse activation reduces per-token compute, but the full expert weight set still stresses accelerator memory. vLLM and vLLM-Ascend provide high-throughput serving and graph capture, yet dynamic expert routing makes CPU/NPU expert movement difficult to combine with stable captured execution on Ascend hardware. |
| Limitation 1 | Existing offload/prefetch paths treat expert movement as a weight-loading policy, but do not expose a graph-compatible state abstraction that keeps physical expert addresses stable while logical experts vary per request. |
| Limitation 2 | A single fixed-slot staging step fails when prefill activates more distinct experts than available slots, which is common for long prompts or broad routing distributions. |
| Limitation 3 | Naive slot reuse is unsafe under overlapped transfer and compute streams because H2D staging can overwrite a slot still being read by GroupedMatmul. |
| Key Idea / Our Goal | Virtualize dynamic MoE experts as a compute-protected fixed-slot runtime, where eager staging updates stable logical-to-physical mappings and capacity overflow is executed as transfer-aware prefill waves. |
| Challenge 1 | Graph capture cannot include host-side active-set discovery, H2D decisions, or dynamically allocated mapping tensors, yet the captured MLP must read correct expert slots. |
| Challenge 2 | Prefill working sets can exceed the slot budget, so the runtime needs a bounded execution plan that preserves correctness without requiring all active experts to be resident at once. |
| Challenge 3 | Slot reuse crosses host bookkeeping, transfer streams, and compute streams, so a cache-style READY flag is insufficient for correctness under overlap. |
| Methodology topic sentence | SEW-Offload is a slot-stable MoE runtime that separates eager routing decisions from graph-captured expert execution. |
| Module A (addresses Challenge 1) | Slot-Stable Expert Virtualization keeps physical slot tensors and `log2phy` buffers at stable addresses while eager staging mutates their contents before graph replay. |
| Module B (addresses Challenge 2) | Capacity-Bounded Wave Prefill splits oversized prefill active sets into waves, stages each wave into bounded slot buffers, and scatters per-wave results back to the original token order. |
| Module C (addresses Challenge 3) | Compute-Protected Slot Lifecycle promotes slots through LOADING, READY, and COMPUTING states and uses transfer/compute events before eviction. |
| Contribution 1 | We introduce SEW-Offload, a graph-compatible fixed-slot runtime for Ascend MoE serving. (Section 3) |
| Contribution 2 | We design capacity-bounded B2 prefill waves with transfer-aware scheduling for active sets larger than the slot budget. (Section 4) |
| Contribution 3 | We present a compute-protected slot lifecycle that prevents cross-stream overwrite races during overlapped H2D and GroupedMatmul execution. (Section 4) |
| Contribution 4 | We evaluate SEW-Offload on Qwen3-30B-A3B under 14 GiB and 28 GiB offload budgets against no-offload and native prefetch baselines. (Section 5) |

## Method Section Outline

### 3. Overview

Introduce the runtime boundary: SEW-Offload manages expert residency between
router output and grouped MLP execution. The central abstraction is a fixed
physical slot bank plus a stable `log2phy` mapping.

### 3.1 Slot-Stable Expert Virtualization

Explain logical expert IDs, physical slot IDs, persistent slot tensors, and the
stable `log2phy` buffer read by the captured graph.

### 3.2 Graph-Compatible Staging Seam

Describe the router-stage-MLP split. The stage op performs active-set discovery,
slot allocation, H2D staging, and mapping updates outside graph capture.

### 3.3 Capacity-Bounded Wave Prefill

Describe B2 waves, hit/miss/mixed waves, temporary stage banks, async prefetch,
and scatter back to full-token output.

### 3.4 Compute-Protected Slot Lifecycle

Describe state transitions, transfer events, compute events, eviction rules, and
failure rollback.

## Self-Consistency Checks

- Check 1 Limitations -> Key Idea: pass.
- Check 2 Key Idea -> Challenges: pass.
- Check 3 Challenges -> Methodology: pass.
- Check 4 Methodology -> Contributions: pass.

## Severity Summary

- 0 CRITICAL, 1 MAJOR, 1 MINOR.
- MAJOR: phase semantics still need a crisp correctness statement in the paper;
  the implementation gates unknown-phase overflow with `max_num_seqs_hint`.
- MINOR: CPU-first loading is currently a strong systems optimization, but should
  be framed as an implementation module or ablation unless experiments show it is
  central to the paper's main result.
