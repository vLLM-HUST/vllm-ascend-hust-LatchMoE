# Method-Experiment Traceability

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| C0. Ascend-UVA-like expert access is evaluated as a threat model before claiming SEW is necessary. | CANN host-register/managed-memory probe, torch_npu pointer-wrapping probe, AI Core read microbenchmark, NPUGraph replay probe, MoE-shaped matmul probe. | E0 Ascend UVA-like feasibility at 14 GiB. | T0 UVA-like feasibility, feasibility note. | CANN 9.0 / 910B2 can register a 14 GiB host range through legacy `aclrtHostRegister`; torch_npu private APIs can wrap a 1 MiB registered pointer as an NPU tensor; simple AI Core elementwise `add` can read host-registered `float16` correctly and simple NPUGraph replay observes host updates. However, D2H copy segfaults, device copy fails with SDMA runtime error `507001`, elementwise host reads are only 8.69-9.06 GiB/s versus 339-399 GiB/s from HBM, and host-registered `torch.matmul` weights fail with runtime error `507057` while HBM references pass. CUDA-UVA equivalence is therefore not supported for MoE expert execution on the tested stack. | Runtime, tensor-wrap, access-matrix, simple graph replay, and MoE-shaped matmul probes completed; full vLLM grouped-MLP integration not attempted because the direct matmul gate fails. |
| C1. SEW enables MoE serving under tight HBM budgets by offloading nonresident experts. | Host expert store, CPU-first loading, fixed NPU slot bank, AutoConfig budget. | E1 end-to-end, E2 memory feasibility, E5 slot/budget sensitivity. | T1 feasibility, T3 memory cost, F4 slot/budget sensitivity. | SEW expands feasible serving under explicit HBM budgets where no-offload fails or leaves insufficient KV. | Smoke-supported; needs repeated ShareGPT buckets. |
| C2. Dynamic MoE offload can be made ACLGraph-compatible on Ascend. | Router-stage-MLP seam, `moe_offload_stage`, persistent slot tensors, persistent `log2phy`. | E2 graph compatibility on native/legacy capture attempts and SEW capture-on. | T6 graph evidence, Figure 1, Figure 2. | SEW completes graph capture/replay while native/legacy dynamic offload paths produce classified graph-boundary failures. | Smoke-supported; needs trace-backed table. |
| C3. Fixed physical slots provide a stable expert-state abstraction. | Slot bank, logical-to-physical mapping, stable replay-visible tensor addresses. | E2, E5. | T3 memory cost, F4 sensitivity, debug trace appendix. | Slot budget trades expert residency against KV cache while preserving replay-visible address stability. | Strong smoke evidence for 28 GiB autoslots vs slots32; needs formal rerun. |
| C4. B2 prefill waves handle active expert overflow. | `phase_split.py`, B2 wave planner, per-wave scatter/gather. | E3 on `prefill_heavy` and `long_context_prefill`. | T4 offload overhead, F5 B2 wave behavior. | Active expert sets larger than the slot budget can be executed without making all active experts resident at once. | Profile events observed; needs no-B2/B2 ablation. |
| C5. Transfer-aware scheduling reduces staging overhead for B2 waves. | `B2PrefillAsyncSchedule`, transfer-aware issue order, transfer engine. | E3 and E4. | T5 ablation, F6 mechanism ablation. | Transfer-aware wave ordering reduces staging or TTFT overhead without changing semantics. | Not yet measured in full workload. |
| C6. Compute-protected slot lifecycle prevents unsafe overwrite under overlap. | `SlotState` LOADING/READY/COMPUTING, transfer/compute events, eviction guard. | E6 lifecycle stress and assertions. | T7 correctness and safety. | SEW avoids reusing slots still consumed by grouped MLP under overlapped H2D/compute. | Not yet run; must not be overclaimed. |
| C7. CPU-first loading reduces startup HBM pressure. | CPU-first expert loader and original expert weight release. | E4 ablation, startup memory log comparison. | T5 ablation, T3 memory cost. | CPU-first is a supporting optimization if it lowers startup peak HBM or avoids load-time OOM. | Not yet measured; keep secondary. |
| C8. SEW preserves model semantics. | Same expert weights and grouped MLP computation after staging. | E6 output/logit equivalence. | T7 correctness table. | SEW does not intentionally change outputs under deterministic decoding beyond expected numerical tolerance. | Not yet run. |

## Claims Not Allowed Yet

- Do not claim broad hardware generality beyond Ascend until another platform is
  implemented.
- Do not claim model quality improvement.
- Do not claim cloud-scale serving throughput; concurrency is secondary
  robustness evidence.
- Do not claim transfer-aware scheduling or CPU-first loading as core
  contributions until ablations support them.
- Do not claim Ascend lacks a UVA-like mechanism. Current evidence shows legacy
  host registration can produce a device pointer at 14 GiB, and simple AI Core
  elementwise compute can read it. The supported claim is narrower: the path is
  operation-dependent, copy/SDMA/D2H paths are unsafe on the tested stack, and
  matrix-weight execution fails in the tested `torch.matmul` path.
