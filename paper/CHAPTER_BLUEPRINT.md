# LatchMoE Post-Introduction Blueprint

This working file plans the argument and evidence allocation. It is not part of
the manuscript.

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

## Section Plan and Page Budget

| Section | Target pages | Role | Main judgment | Evidence IDs | Open gaps |
|---|---:|---|---|---|---|
| Abstract | 0.3 | miniature argument | fixed slots preserve replay; multi-wave improves matched prefill TTFT | E1, E7, E9 | update when broad matrix exists |
| 2 Background and Problem | 1.0-1.5 | formalize conflict and requirements | residency may change while graph-visible storage must not | E1, E11, E12 | Figure 1 needs vector source |
| 3 Design | 3.0-3.5 | explain novel mechanisms by challenge | eager boundary plus fixed slots and leases provide a stable replay interface | E1-E4 | add worked trace if space permits |
| 4 Implementation | 0.8-1.0 | establish realization and reproducibility | hook-enabled vLLM-Ascend realizes the design and fails closed | E2-E6 | exact dependency versions belong in artifact appendix |
| 5 Evaluation | 2.5-3.0 current, 3.5 final | test current claims from system inward | graph path is exact and stable; multi-wave improves matched TTFT | E6-E9 | native/legacy baselines, model/capacity/concurrency breadth |
| 6 Discussion and Limitations | 0.8-1.0 | bound lessons and scope | address stability is the durable lesson; policy and broad performance remain separate | E7-E10 | prefix-cache and multi-device studies |
| 7 Related Work | 1.0-1.5 | locate contribution | prior offloading optimizes placement/movement; LatchMoE targets replay-stable storage | E11 | authors must confirm every cited abstract/full text |
| 8 Conclusion | 0.3 | answer Introduction | mechanism and narrow result support the insight, not broad superiority | E7, E9 | none for Draft status |

## Remaining Evaluation Sequence

1. Run matched graph-compatible no-offload, native prefetch, legacy layered,
   and LatchMoE cases on the same manifest and serving contract.
2. Complete 14/28-GiB and 8/16/32/64-slot sensitivity with retained failures.
3. Repeat mixed, decode-heavy, prefill-heavy, and long-context workloads.
4. Add concurrency and a second supported MoE model before making a broad
   serving claim.
5. Report three independent starts, dispersion, exact-token gates, graph
   fallback counts, HBM accounting, and raw artifact identities for every cell.

