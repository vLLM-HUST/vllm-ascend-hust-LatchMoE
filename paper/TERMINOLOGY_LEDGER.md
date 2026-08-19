# LatchMoE Terminology Ledger

This ledger defines the canonical terminology for the manuscript. Use one
canonical term for each concept across all sections, figures, captions, and
tables.

| Concept | Canonical term | Usage rule |
|---|---|---|
| Overall technique | **expert offloading** | Use for the general technique of keeping some expert weights outside accelerator HBM and moving or accessing them on demand. |
| Setting studied in this paper | **cache-based expert offloading** | Use when referring specifically to offloading with a bounded HBM-resident expert cache. |
| HBM-resident storage structure | **expert cache** | Use for the bounded set of expert weights currently resident in HBM. |
| Cache access outcome | **expert cache hit** / **expert cache miss** | Use when a requested expert is resident / non-resident in the expert cache. |
| Residency changes | **expert admission** / **expert eviction** | Use for adding an expert to / removing an expert from the expert cache. Use **replacement** only when emphasizing that admission reuses storage released by an eviction. |
| Expert-to-storage relation | **device storage assignment** | Use for the runtime association between a logical expert and its current device storage. |
| Binding observed by captured computation | **graph-visible storage binding** | Use for an address or storage reference embedded in or consumed by captured computation. |

## Consistency Rules

- Use **expert offloading** as the default umbrella term.
- Use **cache-based expert offloading** when the bounded expert cache is part
  of the claim or mechanism under discussion.
- Do not use **dynamic expert caching** as a synonym for cache-based expert
  offloading. It can be misread as a cache-policy contribution.
- Use **expert cache** for the HBM-resident structure, not for the overall
  offloading technique.
- Use **resident expert** and **non-resident expert** only relative to the
  expert cache at a specified inference step.
- Keep **device storage assignment** distinct from **graph-visible storage
  binding**: the former may change at runtime, while the latter is the state
  whose replay stability the paper studies.
