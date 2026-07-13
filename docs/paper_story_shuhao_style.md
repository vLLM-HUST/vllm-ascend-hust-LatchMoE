# SEW-Offload Paper Story, Shuhao Zhang Style

## One-Sentence Story

SEW-Offload studies dynamic MoE expert offloading on memory-constrained Ascend
NPU serving, where routed expert state changes per request but ACLGraph replay
requires stable tensor addresses; it virtualizes logical experts into
compute-protected fixed physical slots, moves staging decisions outside graph
replay, and uses capacity-bounded prefill waves to keep serving feasible under
tight HBM budgets.

## Positioning

This is a systems technique paper, not a plugin report. The paper's core
object is a graph-replay-safe state management runtime for dynamic MoE expert
weights on Ascend.

The defensible novelty is not generic expert offloading. It is the combination
of:

1. dynamic routing-driven expert movement,
2. Ascend ACLGraph capture/replay constraints,
3. fixed-address expert-state virtualization,
4. bounded prefill execution when active experts exceed slot capacity.

## Background

MoE LLMs reduce activated compute per token, but they do not remove the memory
pressure caused by the full expert weight set. On single-card Ascend serving,
that pressure competes directly with KV cache and graph-captured execution.
vLLM-Ascend can serve efficiently when tensor addresses and execution paths are
stable, but MoE offload decisions are data dependent: the router selects
different experts per request, prefill can activate a broad expert set, and
host-to-device expert movement must be coordinated with NPU compute streams.

This creates a state-management problem. The system must change which logical
experts are resident, without changing the physical addresses seen by the
captured grouped MLP replay.

## Existing Limitations

### Limitation 1: Offload Paths Treat Expert Movement as Loading, Not State Virtualization

Native prefetch and legacy layered offload paths move expert weights, but they
do not expose a stable physical expert-state abstraction to the graph-captured
MLP. As a result, dynamic movement and graph replay remain structurally
misaligned.

### Limitation 2: Fixed Slots Alone Do Not Solve Prefill Overflow

A fixed slot bank makes addresses stable, but prefill can activate more
distinct experts than available slots. A naive implementation either requires
all active experts to be resident or fails when active expert fanout exceeds the
slot budget.

### Limitation 3: Ready/Not-Ready Is Too Weak Under Overlap

Expert transfer and grouped MLP compute run across different streams. If slot
reuse is governed only by a cache-style ready flag, H2D staging can overwrite a
slot whose weights are still being consumed by compute.

## Problem Essence

How can dynamic MoE expert residency be changed on Ascend while preserving a
stable graph replay boundary and bounded NPU memory usage?

## Key Idea

Separate logical expert identity from physical expert residency. SEW-Offload
keeps fixed physical expert slots and persistent `log2phy` buffers at stable
addresses, updates their contents and mapping in an eager staging step before
replay, and lets the captured grouped MLP consume only stable slot tensors.

## Technical Challenges and Mechanisms

| Challenge | Why naive fails | SEW-Offload mechanism |
|---|---|---|
| Graph replay requires stable addresses, but expert routing is data dependent. | Putting routing-driven H2D decisions inside capture breaks ACLGraph replay; allocating new mapping tensors changes replay-visible state. | Slot-stable expert virtualization: persistent slot tensors and persistent `log2phy` buffers are allocated once; staging mutates contents before replay. |
| Prefill can activate more experts than slots. | Loading the full active set violates the HBM budget; dropping experts changes model semantics. | Capacity-bounded B2 prefill waves: split active experts into slot-sized waves, compute each wave, and scatter outputs back to token order. |
| Slot reuse must be safe under transfer/compute overlap. | A READY-only cache state does not know whether grouped MLP is still reading the slot. | Compute-protected lifecycle: slots move through LOADING, READY, and COMPUTING states, with transfer and compute events guarding eviction. |

## Contribution Draft

1. We identify graph-replay-safe dynamic expert offloading as a state
   virtualization problem on Ascend MoE serving, and introduce SEW-Offload, a
   fixed-slot runtime that decouples logical expert routing from physical expert
   residency.
2. We design a graph-compatible staging seam that updates fixed NPU expert
   slots and persistent `log2phy` buffers before ACLGraph replay, allowing the
   captured grouped MLP to consume stable addresses.
3. We introduce capacity-bounded B2 prefill waves with transfer-aware
   scheduling, enabling prefill active expert sets larger than the slot budget.
4. We implement SEW-Offload in vLLM-Ascend and evaluate feasibility,
   graph-compatibility, performance, memory cost, and correctness on
   Qwen3-30B-A3B under controlled Ascend HBM budgets.

## Introduction Flow

1. Start with memory-constrained single-card MoE serving on Ascend.
2. Explain why MoE expert weights are a dynamic state object, not a static
   parameter blob.
3. Show the ACLGraph tension: graph replay wants stable addresses; routing
   wants dynamic expert movement.
4. Use Figure 1 to show native/legacy offload failing at the graph boundary
   while SEW updates fixed slots before replay.
5. State the three challenges: stable replay state, prefill overflow, safe slot
   reuse under overlap.
6. Present SEW-Offload as the state virtualization runtime.
7. Close with contribution bullets and a controlled experiment promise.

## Figure 1 Story

Use an "Existing vs. SEW-Offload" motivated example:

- Left: routed experts trigger dynamic H2D movement across or inside the replay
  boundary. The captured stream sees synchronization/copy or changing state and
  fails to replay.
- Right: router output is handled by the eager staging seam; the runtime updates
  fixed physical slots and persistent `log2phy`; ACLGraph replay consumes only
  stable slot tensors.

The figure should name the state objects: logical experts, physical slots,
`log2phy`, H2D transfers, grouped MLP replay, and KV memory budget.

## Core Claim Boundary

Allowed headline:

> SEW-Offload makes dynamic MoE expert offloading graph-replay-safe on Ascend
> under tight HBM budgets.

Avoid claiming:

- General MoE offload superiority across all hardware.
- Better model quality.
- Cloud-scale serving throughput.
- Broad model generality before a second model is validated.

## Reviewer Risk Register

| Risk | Why it matters | Defense |
|---|---|---|
| "This is only an Ascend engineering patch." | Systems reviewers need a transferable abstraction. | Frame fixed-slot expert virtualization and graph-replay-safe state management as the core abstraction; keep plugin details in implementation. |
| "FluxMoE already has stable addresses." | Closest prior mechanism overlaps conceptually. | Position SEW as graph-replay-safe dynamic offload on Ascend with compute-protected staging and B2 overflow, not merely stable paging. |
| "Batch=1 is too weak." | Cloud serving reviewers may expect request-rate sweeps. | Use MoE-offload literature as scope precedent; add concurrency robustness as secondary, not headline. |
| "Performance gains are just capture-on/off." | That is actually one claim, but not the whole paper. | Pair performance with graph failure evidence, memory feasibility, slot/budget sensitivity, B2 overflow, and correctness. |
| "Correctness is assumed." | Offload should not change outputs. | Add deterministic output/logit equivalence and slot lifecycle stress tests. |
