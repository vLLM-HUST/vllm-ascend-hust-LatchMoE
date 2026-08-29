# LatchMoE seven-question research contract

This advisor-authored contract maps the existing manuscript to its current,
narrow evidence boundary. It does not replace the student-authored mechanism,
experiments, or results. The authoritative paper entry remains `paper/main.tex`.

## 1. What is the research problem?

How can a bounded expert-offload runtime change the logical experts resident in
HBM without changing the graph-visible storage bindings captured by accelerator
graph replay? The unit of correctness is one logical-expert ownership interval:
transfer readiness, mapping publication, replayed consumption, compute
completion, and safe slot reuse must agree on `(layer, expert, generation)`.

## 2. Why does it matter?

Sparse activation bounds arithmetic but not the total expert-weight footprint.
Host-backed staging makes oversized MoE models deployable, while eager launch
and graph recapture can expose host overhead on the repeated decode path. A
runtime that preserves graph replay under bounded residency could retain a
stable execution interface without requiring every expert to remain in HBM.
The value is conditional: the current full-resident anchor shows that the
evaluated offload path reclaims sampled HBM at a large TTFT and TPOT cost.

## 3. What common assumption and related-work boundary are tested?

The tested assumption is that changing logical expert placement must also
change the addresses consumed by captured computation, or else force eager
execution or graph repair. Existing expert caching, prediction, prefetching,
placement, and pipelining work primarily decides which weights to retain or
move. Graph-oriented runtimes preserve replay state, and virtual-memory systems
provide stable addressing. LatchMoE's paper boundary is the conjunction:
bounded expert replacement behind replay-stable physical slots, with explicit
ready-before-publication and release-before-reuse ordering. It does not claim a
new router, placement policy, universal graph requirement, or superiority over
published offload systems.

## 4. What is the mechanism hypothesis?

Address-stable slot virtualization can decouple logical ownership from the
physical buffers bound into a captured graph. Replay-boundary staging updates
slot contents and the logical-to-physical map outside the captured expert
computation; capacity-bounded waves cover routed working sets larger than the
slot bank; versioned leases prevent stale publication and premature reuse. The
falsifiable claim is that these mechanisms preserve exact output and lifecycle
invariants while enabling the implemented dynamic offload path to use
PIECEWISE graph replay.

## 5. Why is the hypothesis feasible?

The repository contains a vLLM-Ascend carrier, fixed-slot and mapping storage,
transfer events, lifecycle checks, wave execution, graph capture/replay probes,
and fail-closed verifiers. Existing evidence qualifies the mechanism on the
reported model/device tuples. Feasibility does not establish broad performance,
multi-NPU behavior, quantized-layout support, or correctness for every workload.

## 6. What is the matched evaluation contract?

- **Correctness and graph qualification:** compare native or capacity-feasible
  oracle output with the same request under the LatchMoE eager seam and
  PIECEWISE graph path. Require complete token-array identity, fixed address
  fingerprints, zero eager fallback, ready-before-use, balanced leases,
  release acknowledgement, and retained raw receipts.
- **Deployment cost:** compare full residency and LatchMoE with frozen model,
  workload order, source/runtime pins, graph mode, device, tensor parallelism,
  concurrency, KV budget, and prefix-cache setting. Full residency is a cost
  anchor, not a memory-matched offload baseline.
- **Mechanism effect:** compare multi-wave with graph-compatible full-layer
  staging, then overlap depth one with serial depth zero. These are the
  strongest currently admissible internal baselines because native prefetch
  and legacy offload fail exact-token comparability in recorded campaigns.
- **Metrics:** exact-token coverage, capture/replay and fallback counts,
  ownership/lease violations, TTFT, TPOT, throughput, HBM, host and slot bytes,
  H2D bytes, stage wait, waves, failures, and release custody. Report full
  distributions and counterordered independent starts where the current paper
  claims uncertainty.
- **Performance gate:** correctness and custody pass first. A treatment claim
  requires repeated matched advantage over its strongest admissible arm and
  preregistered uncertainty/equivalence criteria. A completed arm with an
  oracle mismatch remains a negative result and is excluded from latency
  ranking.
- **Stop conditions:** stop the current mechanism if address identity or lease
  correctness cannot be closed after one bounded debug cycle. Stop broad graph
  or workload claims when the matched eager/graph or cross-workload oracle does
  not close. Do not move thresholds after held-out results. A negative result
  closes only the tested mechanism, model, workload, capacity, or concurrency
  cell.

## 7. What is the citable takeaway if the gates hold?

Graph replay needs stable graph-visible storage, not static logical residency.
For a bounded dynamic working set with an explicit staging boundary, logical
owners can be virtualized over fixed physical slots if publication and reuse
are lifecycle-ordered. The current evidence supports this takeaway for the
implemented, narrowly qualified LatchMoE path; it does not support universal
offload speedup or multi-model performance generality.

## Ascend architecture causal chain

| Link | Current statement and evidence boundary |
|---|---|
| Architecture fact | The carrier uses PIECEWISE ACLGraph replay, explicit host-to-device expert staging, fixed NPU slot/map buffers, and device-specific fused-MoE seams. These are code, graph, and lifecycle facts. |
| Invalidated assumption | Routing-driven logical replacement need not force graph-visible pointer rebinding. The graph can retain fixed buffers while ownership metadata and contents change at a legal boundary. |
| Real counterexample | Recorded Ascend runs show fixed slot/map addresses across replay while logical owners and overflow waves change; the historical stable-slot-off diagnostic also shows that another native dynamic prefetch path can replay, so LatchMoE does not claim all dynamic-address systems must fail. |
| Native mechanism | Replay-boundary staging, address-stable slot/map buffers, PIECEWISE split placement, asynchronous H2D readiness, and versioned lease/release checks form the current Ascend carrier. |
| Predictive boundary | The mechanism is expected to qualify only when routed work can be partitioned within the slot/stage budget, the runtime exposes a legal eager staging seam, graph-visible addresses remain fixed, and transfer/lifecycle cost does not erase the deployment benefit. |
| Generalization class | The abstraction may apply to accelerator runtimes that expose a stable captured-storage ABI plus an out-of-graph update boundary. Multi-NPU ownership, quantized layouts, other graph implementations, and broader concurrency remain unverified classes. |

## Evidence ledger and known negative results

The machine-readable and claim-level sources remain `paper/CLAIMS.md`,
`paper/CLAIM_LEDGER.md`, `paper/EVIDENCE_MAP.md`, and `paper/data/`. In
particular:

- matched real-online campaigns support only their recorded Qwen3, single
  Ascend 910B2, BF16, TP1, low-concurrency, no-prefix-cache boundaries;
- single-start capacity points and one-request eager/graph measurements are
  descriptive or qualification evidence, not uncertainty or throughput claims;
- native-prefetch and legacy-offload cells that fail exact-token comparison are
  retained as negative/unsupported cells and do not enter latency ranking;
- the historical eager-to-graph raster is derived and non-regenerable until its
  raw source is recovered;
- the standard-workload matched campaign closed with claim shrink rather than a
  broad workload or performance claim; and
- the current submission audit remains blocked by a citation audit and stale
  semantic review artifacts documented in `paper/ACCEPTANCE_STATUS.md`.

Implementation, environment setup, new experiments, and result generation
remain the responsibility of the student owners.
