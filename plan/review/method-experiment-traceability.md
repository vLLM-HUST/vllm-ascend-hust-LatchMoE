# Method-Experiment Traceability

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| Dynamic MoE offload can be made ACLGraph-compatible on Ascend. | Router-stage-MLP seam and `moe_offload_stage`. | E1, E3 on `sew_14gb_autoslots` and `sew_28gb_slots32`. | Graph evidence table, Figure 1, Figure 2. | SEW can complete graph capture where native/legacy paths fail. | Smoke-supported; needs repeated workloads. |
| Fixed physical slots provide a stable expert-state abstraction. | Slot bank, persistent `log2phy`, fixed-slot runtime. | E2, E6. | Memory table, slot sensitivity plot. | Slot budget trades KV cache for expert residency. | Strong smoke evidence for 28 GiB autoslots vs slots32. |
| B2 prefill waves handle active expert overflow. | `phase_split.py`, B2 wave prefill path. | E4 on `prefill_heavy` and `long_context_prefill`. | B2 wave figure. | Active expert sets can exceed slot count without full residency. | Profile events observed; needs ablation. |
| Transfer-aware scheduling reduces staging overhead. | Transfer-aware B2 wave schedule. | E4, E5. | Ablation chart. | Scheduling improves prefill/stage behavior without changing semantics. | Not yet measured in Week2 smoke. |
| CPU-first loading reduces startup HBM pressure. | CPU-first expert loader. | E5. | Startup memory table. | CPU-first is a supporting optimization if ablation shows benefit. | Not yet measured in Week2 smoke. |
| SEW preserves model semantics. | Same model weights and decoding path after staging. | E8 output/logit equivalence. | Correctness table. | SEW does not change generated outputs under deterministic settings beyond expected numerical tolerance. | Not yet run. |
