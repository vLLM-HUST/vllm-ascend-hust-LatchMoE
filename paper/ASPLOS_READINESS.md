# ASPLOS 2027 Readiness Audit

Audit date: 2026-08-14. Assumed cycle: September 2026 submission for ASPLOS
2027. Official source: <https://www.asplos-conference.org/asplos2027/cfp/>.

## Verdict

The manuscript is a coherent systems draft, but it is not ready for ASPLOS
submission. The main risk is research evidence, not prose completeness. Its
current graph-only qualification and three matched TTFT pairs support a narrow
mechanism claim, while ASPLOS reviewers will expect convincing comparisons,
causal attribution, and robustness. The Introduction has been compressed to
close the complete problem--insight--design--evidence narrative within the
two-page rapid review window.

## CRITICAL

### C1. The current evaluation does not establish the full ASPLOS claim

The paper has one matched comparison between multi-wave and graph-compatible
full-layer staging under one Qwen3-30B-A3B, Ascend 910B2, TP1, 14-GiB,
32-slot, concurrency-1 configuration. It does not yet compare against the
strongest credible no-offload, native-prefetch, or legacy layered-offloading
baselines under one contract. It also lacks mechanism ablations and breadth
across model, capacity, workload, and concurrency. Until these experiments
exist, the paper cannot claim broad serving improvement or generality.

Repair: execute the planned matrix in `benchmark/configs/sew_offload_v1.yaml`
and `benchmark/scenarios/sew_offload_scenarios.json`, retaining failed cells
and reporting independent starts, dispersion, exact-token gates, fallback
counts, memory accounting, and artifact identities.

## MAJOR

### M1. Figure 1 violates the submission figure guidance

`figures/graph_breakdown.png` is raster, and its text renders below the
preferred 9 pt size and in places below the permitted 8 pt minimum. ASPLOS
warns that undersized figure text can lead to rejection even when HotCRP's
format check passes.

Repair: recover the underlying profiling table and regenerate the figure as
PDF with at least 8 pt, preferably 9 pt, final-size text. Preserve a second
encoding beyond color.

### M2. The paper needs a stronger ASPLOS-specific lesson

The strongest durable contribution is not the particular predictor or an
Ascend-only speedup. It is the separation between replay-visible address
identity and dynamically changing logical ownership. Design, Discussion, and
Related Work should consistently explain this as a reusable accelerator
runtime abstraction and state which aspects are hardware/runtime specific.

Repair: after completing the experiments, add a concrete design-alternative
comparison covering pointer rebinding, graph recapture/update, static
residency, and address-stable slot virtualization.

### M3. Reproducibility details must move into a submission-safe artifact

The implementation is described, but exact dependency versions, commands,
raw manifests, and result identities are not yet packaged as an anonymized
artifact. The main paper must remain self-contained even if reviewers ignore
the appendix and artifact.

Repair: prepare an anonymous repository and an appendix containing only
supporting reproducibility detail. Keep claim-critical setup, baselines,
metrics, and validity threats in the 11-page paper.

### M4. References need an ASPLOS formatting audit

ASPLOS requires full, non-abbreviated author names without `et al.`, clickable
in-text citations, and preferably DOI links in reference entries. The active
bibliography entries mostly contain full names and many DOI fields, but this
must be checked from the rendered `ACM-Reference-Format` output. Duplicate
unused BibTeX entries should not be confused with the active citation set.

Repair: inspect the final `.bbl` and PDF after every bibliography change;
add verified DOI or document URLs where absent.

### M5. Mandatory-template line breaking needs a cleanup pass

The first ASPLOS-template build exposed widespread overfull lines that were not
visible in the previous USENIX layout. A legal emergency line-breaking stretch
and ordinary prose edits removed the horizontal overflow without changing font
size, margins, or vertical spacing. One small overfull page box remains for the
final rendered cleanup after figure replacement.

Repair: use ordinary prose edits and legal hyphenation/line-breaking controls;
do not change margins, font sizes, caption spacing, or vertical spacing.

## MINOR

### N1. Topic selection must match reviewer expertise

Use focused submission topics around heterogeneous accelerators, memory and
I/O, and systems for parallel computation. Avoid broad topic selection that
obscures the graph-runtime contribution.

### N2. The AI disclosure must remain isolated

ASPLOS requires full disclosure under the ACM authorship policy. The draft now
contains an acknowledgment describing Codex-assisted structuring, drafting,
editing, citation checking, and plotting-code generation. Do not place other
acknowledgments in the anonymous submission; this section is outside the page
limit only when used solely for AI disclosure.

### N3. Submission logistics need an author-side check

Confirm all authors and conflicts at submission time, remove identifying PDF
metadata and repository links, and provide a one-page change note if this is a
resubmission. A paper rejected in one ASPLOS cycle cannot be submitted to the
next cycle specified by the CFP's resubmission rule.

## Confirmed Format Contract

| Item | ASPLOS 2027 requirement | Draft status |
|---|---|---|
| Template | `acmart` with `sigplan,anonymous,review,nonacm` | migrated |
| Main-paper limit | 11 pages | passes; main content ends on page 9 |
| Excluded from limit | references, appendices, AI-only acknowledgment | structurally compliant |
| Review model | double blind; first round reads pages 1--2 only | passes structural pagination audit |
| Body font | 10 pt | template controlled |
| Figure/table text | at least 8 pt, preferably 9 pt | Figure 1 fails |
| Bibliography | `ACM-Reference-Format`, full names, hyperlinks, DOI preferred | 19 rendered entries; metadata completion pending |
| Appendix | unlimited but optional to reviewers | do not place critical evidence there |
| Generative AI | use must be fully disclosed | disclosure added; authors must verify wording |

## Resolved During This Audit

The first ASPLOS-template build placed the contribution list on page 3 and
ended page 2 mid-paragraph. The Introduction was reduced from 1,457 to 696
source words without adding evidence or broadening claims. The complete
problem, structural conflict, key insight, three mechanisms, implementation,
bounded result, and contribution list now appear before Background begins on
page 2.

## Repair Order

1. Finish the matched baseline, ablation, and robustness experiments.
2. Rewrite and paginate pages 1--2 around the ASPLOS rapid-review contract.
3. Rebuild Figure 1 from source data as vector artwork.
4. Package and anonymize the artifact, then audit references and PDF metadata.
5. Run a full pre-submission review against the official format checker and
   SIGPLAN empirical-evaluation guidelines.
