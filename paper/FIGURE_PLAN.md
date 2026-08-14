# LatchMoE Figure Plan

This working file applies the figure-design audit to the three storytelling
figures. It is not part of the manuscript.

## Figure 1: Graph-Replay Motivation

### 1. Figure type
- Type: motivated-example / experimental-results hybrid
- Reason: it quantifies the host-gap penalty that makes eager fallback unacceptable.

### 2. Paradigm recommendation
- Paradigm: before/after stacked-bar comparison.
- Why this paradigm: the stack separates device execution from host-induced gaps while the eager/graph pair exposes the intervention.
- Alternatives considered and rejected: a pipeline would not provide evidence; a line chart has no ordered independent variable.

### 3. Layout sketch
- Canvas: single-column, approximately 3.35 by 1.9 inches.
- Panels: four workload groups; each group contains adjacent Eager and Graph bars.
- Arrows and connections: none; numeric totals sit above bars.
- Colour assignment: device execution in bluish green; host-induced gap in orange, with a second non-colour encoding.

### 4. Labelling and annotations
- Element names: ShareGPT, LongBench, HumanEval, GSM8K; Eager; Graph; Device Execution; Host-induced Device Gaps.
- Critical highlights: total latency above every bar and the aggregate reduction in the caption.
- Font sizes: at least 8 pt after insertion.
- Colour palette: Okabe-Ito.

### 5. Tool suggestion
- Primary: Matplotlib from the original profiling table.
- Alternative: PGFPlots.
- Reason: the current PNG cannot be edited or regenerated from values stored in this repository.

### 6. Universal rule audit
- [ ] Vector format: fail, current source is PNG (MAJOR).
- [ ] Font size: fail after compiled-size inspection; labels and legend are below the target size (MAJOR).
- [ ] Colour-blind safe: likely pass, but the stack relies mainly on colour and dotted fill.
- [ ] Self-contained caption: pass.
- [ ] Honest axis range: pass; latency bars start at zero.
- [ ] No chartjunk: pass.

### 7. Integrity gate result
- Gates 1-4 and 7: pass.
- Gate 5: fail until the original profiling data is supplied and the figure is regenerated as PDF with at least 8 pt text.
- Gate 6: pass; this is the running motivation used by the Introduction.

### 8. Severity summary
- 0 CRITICAL, 2 MAJOR, 0 MINOR.
- Top actions: recover the source data; regenerate as PDF; increase final-size text to at least 8 pt; retain the same workload order and caption claim.

## Figure 2: LatchMoE Overview

### 1. Figure type
- Type: solution-overview.
- Reason: it must provide the mental map for the captured and eager parts of one MoE layer.

### 2. Paradigm recommendation
- Paradigm: multi-layer system architecture.
- Why this paradigm: initialization and online execution occur at different frequencies, while the captured router, eager staging boundary, and captured expert MLP interact rather than form a simple data-transformation pipeline.
- Alternatives considered and rejected: a plain pipeline hides the host store and transfer dependencies; a flat architecture hides initialization versus replay-time behavior.

### 3. Layout sketch
- Canvas: two-column width, approximately 7.0 by 2.5 inches.
- Panels: top row contains CPU-First Host Store and Fixed Expert-Slot Allocation; bottom row contains Captured Router, Replay-Boundary Staging, and Captured Expert MLP. The Versioned Slot Lifecycle and Capacity-Bounded Wave Executor sit below the staging boundary.
- Arrows and connections: routed expert IDs flow router to staging; H2D weights flow host store to fixed slots; logical-to-slot IDs flow mapping buffer to the MLP; readiness and compute events form a dashed dependency loop.
- Colour assignment: captured blocks in blue, eager/control blocks in orange, stable storage in green, existing runtime context in grey.

### 4. Labelling and annotations
- Element names: CPU-First Host Store, Placement Plan, Transfer Engine, Fixed Expert-Slot Bank, Persistent Logical-to-Slot Mapping, Captured Router, Eager Staging Boundary, Captured Expert MLP, Versioned Slot Lifecycle, Capacity-Bounded Wave Executor.
- Critical highlights: “stable address” on slot and mapping allocations; “dynamic contents” on H2D and mapping updates.
- Font sizes: 8-9 pt post-scaling.
- Colour palette: Okabe-Ito with solid/dashed boundary encoding.

### 5. Tool suggestion
- Primary: TikZ in the Draft, then draw.io/Figma for camera-ready polish.
- Alternative: retain TikZ if typography and spacing pass final-size inspection.
- Reason: the architecture is structured and must remain vector and terminology-aligned with subsection names.

### 6. Universal rule audit
- [ ] Vector format: pass by construction.
- [ ] Font size: pass after compiled-size inspection.
- [ ] Colour-blind safe: pass in specification with boundary styles in addition to colour.
- [ ] Self-contained caption: pass.
- [ ] Honest axis range: not applicable.
- [ ] No chartjunk: pass.

### 7. Integrity gate result
- Gates 1-4 and 7: pass.
- Gate 5: pass after compiled inspection; no text or arrow overlaps remain.
- Gate 6: not applicable.

### 8. Severity summary
- 0 CRITICAL, 0 MAJOR, 0 MINOR.
- Top actions: retain the current terminology and vector export; polish alignment in draw.io/Figma only if the venue's camera-ready style requires it.

## Figure 3: Matched Multi-Wave TTFT

### 1. Figure type
- Type: experimental-results.
- Reason: it compares two execution modes on TTFT p50 and p95 under one matched contract.

### 2. Paradigm recommendation
- Paradigm: grouped bar with min-max repeat whiskers.
- Why this paradigm: there are two methods, two summary metrics, and three independent service starts per method.
- Alternatives considered and rejected: a line chart implies an ordered trend; a box plot over three repeat-level summaries is unstable and hides the central matched comparison.

### 3. Layout sketch
- Canvas: single-column, 3.35 by 2.15 inches.
- Panels: TTFT p50 and TTFT p95 groups; Full-layer and Multi-wave bars in each group.
- Arrows and connections: none; exact repeat medians are printed above bars, and whiskers show the repeat-level min-max range.
- Colour assignment: Full-layer in neutral grey; Multi-wave in blue.

### 4. Labelling and annotations
- Element names: Full-layer staging, Multi-wave staging, TTFT p50, TTFT p95, Latency (ms).
- Critical highlights: 35.21% and 22.96% reductions in the caption, not as decorative callouts.
- Font sizes: 8 pt ticks/legend, 9 pt axis label.
- Colour palette: Okabe-Ito plus fixed left/right bar position.

### 5. Tool suggestion
- Primary: repository Matplotlib script reading the checked-in Issue 17 bundle.
- Alternative: PGFPlots generated from the same JSON summary.
- Reason: it is reproducible, versioned, and data-bound.

### 6. Universal rule audit
- [ ] Vector format: pass (PDF output).
- [ ] Font size: pass after compiled-size inspection.
- [ ] Colour-blind safe: pass; colour and position both identify modes.
- [ ] Self-contained caption: pass.
- [ ] Honest axis range: pass; zero baseline for latency bars.
- [ ] No chartjunk: pass.

### 7. Integrity gate result
- Gates 1-7: pass after compiled-size inspection.

### 8. Severity summary
- 0 CRITICAL, 0 MAJOR, 0 MINOR in the specification.
- Top actions: regenerate after bundle changes; keep the narrow configuration in the caption; do not present TPOT as improved.
