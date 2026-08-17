# LatchMoE ASPLOS 2027 Blueprint

This working file plans the argument and evidence allocation. It is not part of
the manuscript.

## Venue Contract

This plan targets the ASPLOS 2027 September cycle (full paper due September 9,
2026, AoE). The official CFP requires the `acmart` class with
`sigplan,anonymous,review,nonacm`, permits at most 11 pages for the main paper,
and excludes references, appendices, and an AI-disclosure-only acknowledgment
from that limit. The submission must remain self-contained without appendices.

ASPLOS 2027 first evaluates only pages 1--2. Those pages must therefore make
four judgments possible without later context: the work advances an ASPLOS
core discipline, the offloading/replay conflict is structural, address-stable
residency is a distinct cross-layer insight, and the implementation has at
least one exactness result and one bounded performance result. A contribution
to MoE serving alone is insufficient unless the paper makes its architecture,
programming-system, or operating-system advance explicit.

## Reviewer Thesis

LatchMoE shows that graph replay needs stable physical storage rather than
static expert residency. A fixed-slot data plane, an eager staging boundary,
and a versioned transfer/compute lifecycle let logical experts change while the
captured MoE path continues to reference stable addresses. Exact multi-wave
execution extends the abstraction to routed working sets larger than the slot
bank. Current evidence validates the mechanism and a narrow prefill TTFT gain;
it does not establish broad serving superiority.

## Claim-Evidence Spine

| Claim | Why it matters | Evidence | Paper location | Status |
|---|---|---|---|---|
| Dynamic expert replacement conflicts with replay-visible bindings | establishes the systems gap | Introduction argument; Figure 1 | Background 2.2 | supported by user manuscript |
| Stable slots decouple logical residency from graph-visible storage | establishes the key insight | code paths E2-E4; overview figure | Design 3.1-3.3 | implemented |
| Versioned dependencies prevent stale or premature reuse | establishes correctness under overlap | E2, E3, E6, E7 | Design 3.4; Evaluation Q1 | implemented and narrowly qualified |
| Multi-wave execution preserves semantics within bounded capacity | handles oversized prefill sets | E4, E7, E8, E9 | Design 3.5; Evaluation Q2-Q3 | narrowly qualified |
| Multi-wave lowers TTFT relative to full-layer staging | establishes measured value | E9 | Evaluation Q2 | verified only for the matched contract |
| Benefits generalize across serving conditions | would establish breadth | E10 planned matrix | Discussion | missing; explicitly not claimed |

## ASPLOS Section Plan and Page Budget

| Section | Target pages | Role | Main judgment | Evidence IDs | Open gaps |
|---|---:|---|---|---|---|
| Title and Abstract | 0.45 | miniature argument | fixed slots preserve replay; multi-wave improves matched prefill TTFT | E1, E7, E9 | keep the evidence boundary in the abstract |
| 1 Introduction | 1.55 | pass rapid review by page 2 | this is a cross-layer replay abstraction, not only an MoE optimization | E1, E7, E9, E12 | compress so design, realization, and result all appear by page 2 |
| 2 Background and Problem | 0.75 | formalize conflict and requirements | residency may change while graph-visible storage must not | E1, E11, E12 | Figure 1 needs vector source |
| 3 Design | 2.30 | explain novel mechanisms by challenge | eager boundary plus fixed slots and leases provide a stable replay interface | E1-E4 | add a concrete slot-lifecycle trace if space permits |
| 4 Implementation | 0.65 | establish realization | hook-enabled vLLM-Ascend realizes the design and fails closed | E2-E6 | exact dependencies belong in the artifact appendix |
| 5 Evaluation | 3.50 | test claims from correctness outward | graph path is exact and stable; multi-wave improves matched TTFT | E6-E9 | strongest baselines, ablations, model/capacity/concurrency breadth |
| 6 Discussion and Limitations | 0.55 | state lesson, costs, and boundary | address stability is the durable lesson; policy and broad performance remain separate | E7-E10 | prefix-cache and multi-device studies |
| 7 Related Work | 0.85 | locate the cross-layer contribution | prior offloading optimizes movement; LatchMoE makes dynamic residency replay-compatible | E11 | authors must confirm every cited abstract/full text |
| 8 Conclusion | 0.20 | answer Introduction | mechanism and narrow result support the insight, not broad superiority | E7, E9 | none for Draft status |

The 10.8-page target leaves roughly 0.2 pages of layout slack. References and
the AI-disclosure acknowledgment are outside the 11-page main-paper limit,
but the paper may not rely on appendices for acceptance-critical evidence.

## First-Two-Page Contract

1. Page 1 states the replay/offloading incompatibility and quantifies why
   eager fallback is unacceptable.
2. By the middle of page 2, the paper states the stable-storage insight and
   distinguishes it from caching, prediction, and prefetching.
3. Before page 2 ends, the reader sees the fixed-slot abstraction, eager
   staging boundary, versioned lifecycle, implemented stack, exactness gate,
   and the 35.21% matched TTFT-p50 result with its narrow scope.
4. Figure 1 remains evidence for the structural conflict. It must be rebuilt
   as vector artwork with at least 8 pt, preferably 9 pt, final-size text.

## Remaining Evaluation Sequence

1. Run matched graph-compatible no-offload, native prefetch, legacy layered,
   and LatchMoE cases on the same manifest and serving contract.
2. Complete 14/28-GiB and 8/16/32/64-slot sensitivity with retained failures.
3. Repeat mixed, decode-heavy, prefill-heavy, and long-context workloads.
4. Add concurrency and a second supported MoE model before making a broad
   serving claim.
5. Report three independent starts, dispersion, exact-token gates, graph
   fallback counts, HBM accounting, and raw artifact identities for every cell.
