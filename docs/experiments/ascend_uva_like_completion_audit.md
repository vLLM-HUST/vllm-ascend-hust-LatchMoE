# Ascend UVA-like Feasibility Completion Audit

Date: 2026-07-09

## Objective Under Audit

Explore and implement Ascend UVA-like Expert Access Feasibility to answer:

- whether CUDA UVA-style expert access can be ported to Ascend NPU;
- if it can, how it compares with SEW;
- if it cannot, why not and where the implementation is blocked;
- under the same 14 GiB offload budget and benchmark setting.

## Evidence Inventory

| Requirement | Evidence | Status |
|---|---|---|
| Use 14 GiB offload budget. | `probe_device4_14gb_register_only.json`, `e0_ascend_uva_like_summary.csv`, `e0_runner_manifest.json`. | Satisfied for E0 feasibility. |
| Check runtime mapping feasibility. | Legacy `aclrtHostRegister` succeeds for 14 GiB; `HostRegisterV2` and managed allocation are not usable through tested paths. | Satisfied. |
| Check framework exposure to vLLM/torch_npu-like tensor path. | Private `torch_npu` storage/tensor constructors wrap a host-registered pointer and `_check_npu_data_ptr=True`. | Satisfied for metadata wrapping. |
| Check safe tensor access. | D2H `copy_uint8` terminates by signal 11; device copy fails with SDMA/runtime `507001`; elementwise `add_float16_zero` succeeds. | Satisfied; access is operation-dependent. |
| Check simple AI Core read performance. | Host-registered elementwise read is 8.69-9.06 GiB/s versus HBM 339-399 GiB/s. | Satisfied for elementwise read; not an MoE performance claim. |
| Check graph replay semantics. | Simple `torch.npu.NPUGraph` replay observes host-side updates. | Satisfied for simple elementwise graph. |
| Check MoE-shaped expert weight access. | Host-registered matmul weights fail with `507057` at 2 MiB and 32 MiB weight tiles while HBM matmul references pass. | Satisfied for `torch.matmul`/torch_npu matrix path. |
| Compare to SEW if runnable. | Direct host-registered expert matmul is not runnable, so there is no fair TTFT/TPOT performance comparison. The comparison boundary is a compatibility-failure baseline. | Satisfied by negative gate. |
| Provide reproducible benchmark procedure. | `run_uva_feasibility.py` executes YAML-configured probes and treats expected nonzero probes as evidence; verdict is checked automatically. | Satisfied. |
| Provide paper-ready report. | `docs/experiments/ascend_uva_like_report.md`. | Satisfied. |

## Current Verdict

`not_viable_as_sew_baseline`

The tested Ascend stack exposes a partial UVA-like mechanism, but it is not a
drop-in CUDA UVAOffloader port for MoE expert execution. The blocker is not the
14 GiB host registration layer, nor simple graph replay. The blocker observed
in the current evidence is the matrix/expert-weight operator path:
host-registered `torch.matmul` weights fail with `507057`, while HBM-resident
weights pass for the same shapes.

## Claim Allowed In Paper/Presentation

Allowed:

> Ascend CANN 9.0 / 910B2 provides partial host-registered remote-read
> capability: a 14 GiB host expert store can be mapped, simple AI Core reads can
> consume it, and simple NPUGraph replay can observe host updates. However, the
> path is not a CUDA UVAOffloader equivalent for MoE serving: copy/SDMA paths
> fail or crash, simple remote reads are far slower than HBM, and
> host-registered expert-weight matmul fails with `507057`. Therefore, SEW
> should compare against this path as a compatibility-failure baseline, and its
> fixed HBM slot staging remains necessary for grouped MLP execution.

Not allowed:

- Ascend has no UVA-like mechanism.
- Ascend UVA-like offload is slower than SEW in full vLLM serving.
- No possible lower-level Ascend C kernel could ever read host-registered
  matrix weights.

## Remaining Stronger Evidence, If Needed

The current evidence is enough to reject a direct vLLM/torch_npu CUDA-UVA-style
port. A stronger hardware-level impossibility claim would require a custom
Ascend C or vendor-supported grouped-matrix probe that bypasses
`torch.matmul`/torch_npu operator selection. If that lower-level kernel can run,
then SEW should be compared against its latency and bandwidth. If it also fails,
the paper can strengthen the blocker from "torch_npu matrix path" to
"custom matrix kernel path" on the tested stack.
