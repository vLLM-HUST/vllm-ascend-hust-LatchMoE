# Ascend UVA-like Feasibility Report

This report is generated from real E0 probe artifacts. It should be regenerated after rerunning the UVA feasibility suite.

## Verdict

| Field | Value |
|---|---|
| Verdict | `not_viable_as_sew_baseline` |
| Comparison to SEW | `compatibility_failure_not_latency_throughput_comparison` |
| Primary blocker | `host_registered_matmul_weight_path_fails_507057` |
| Offload budget | 14 GiB |

Allowed claim:

> Ascend exposes a partial UVA-like remote-read path on this CANN 9.0/910B2 stack, but it is not a drop-in CUDA UVAOffloader port for MoE expert execution: simple elementwise reads and simple NPUGraph replay work, while copy/SDMA paths fail and host-registered matmul weights fail with 507057. SEW should be compared against this as a compatibility-failure baseline unless a lower-level grouped-MLP path is made runnable.

## Runner Summary

| Field | Value |
|---|---|
| Runner status | `ok` |
| Expected verdict | `not_viable_as_sew_baseline` |
| Observed verdict | `not_viable_as_sew_baseline` |
| Verdict matches | yes |

| Command | Return code | Runner status | Expected nonzero |
|---|---:|---|---|
| `small_runtime_probe` | 0 | `ok` | no |
| `budget_register_probe` | 0 | `ok` | no |
| `tensor_access_matrix` | 1 | `expected_nonzero` | yes |
| `npugraph_replay_probe` | 0 | `ok` | no |
| `matmul_probe_2mib` | 1 | `expected_nonzero` | yes |
| `matmul_probe_32mib` | 1 | `expected_nonzero` | yes |
| `collect_summary` | 0 | `ok` | no |

## Gate Evidence

| Gate | Operation | OK | Status | Size MiB | Avg ms | Bandwidth GiB/s | Relative to HBM | Note |
|---|---|---|---|---:|---:|---:|---:|---|
| `U0_runtime_mapping` | `aclrtHostRegister` | yes | `runtime_mapping_possible` | 14336 | - | - | - | device_ptr=0x1ffc7fe00000; HostRegisterV2 ok=False |
| `U0_framework_wrapping` | `torch_npu_private_tensor_wrap` | yes | `tensor_wrap_possible` | 1 | - | - | - | check_npu_data_ptr=True |
| `U1_tensor_access_matrix` | `construct_uint8` | yes | `completed` | 1 | - | - | - | - |
| `U1_tensor_access_matrix` | `copy_uint8` | no | `signal_11` | 1 | - | - | - | - |
| `U1_tensor_access_matrix` | `device_copy_uint8` | no | `completed` | 1 | - | - | - | RuntimeError('npuSynchronizeDevice:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:57... |
| `U1_tensor_access_matrix` | `add_float16_zero` | yes | `completed` | 1 | - | - | - | - |
| `U1_elementwise_read_bandwidth` | `host_registered_add_64MiB` | yes | `tensor_wrap_possible` | 64 | 7.19 | 8.69 | 0.0218 | output allclose to one |
| `U1_hbm_reference` | `hbm_add_64MiB` | yes | `ok` | 64 | 0.16 | 399.21 | 1.0000 | HBM resident reference |
| `U1_elementwise_read_bandwidth` | `host_registered_add_256MiB` | yes | `tensor_wrap_possible` | 256 | 27.61 | 9.06 | 0.0267 | output allclose to one |
| `U1_hbm_reference` | `hbm_add_256MiB` | yes | `ok` | 256 | 0.74 | 339.31 | 1.0000 | HBM resident reference |
| `U2_npugraph_replay` | `npugraph_replay_host_update` | yes | `graph_replay_observes_host_updates` | 1 | - | - | - | host=2.0 output=3.0; host=4.0 output=5.0 |
| `U3_moe_shaped_matmul` | `host_registered_weight_matmul_m16_k1024_n1024` | no | `host_registered_matmul_failed` | 2 | - | - | - | RuntimeError('npuSynchronizeDevice:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:56... |
| `U3_hbm_matmul_reference` | `hbm_weight_matmul_m16_k1024_n1024` | yes | `ok` | 2 | 0.07 | 28.43 | 1.0000 | HBM resident matmul reference |
| `U3_moe_shaped_matmul` | `host_registered_weight_matmul_m16_k4096_n4096` | no | `host_registered_matmul_failed` | 32 | - | - | - | RuntimeError('npuSynchronizeDevice:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:56... |
| `U3_hbm_matmul_reference` | `hbm_weight_matmul_m16_k4096_n4096` | yes | `ok` | 32 | 0.12 | 271.04 | 1.0000 | HBM resident matmul reference |

## Paper-Ready Interpretation

- The 14 GiB host expert store can be registered through legacy `aclrtHostRegister`, so the runtime mapping layer is not the blocker.
- Simple host-registered elementwise reads work but are slow: 8.69 GiB/s vs 399.21 GiB/s at 64 MiB, and 9.06 GiB/s vs 339.31 GiB/s at 256 MiB.
- Simple `torch.npu.NPUGraph` replay observes host-side content updates, so graph replay alone is not the first hard blocker.
- Host-registered matmul weights fail for both 2 MiB and 32 MiB weight tiles with `507057`, while HBM matmul references pass.
- Therefore, the fair SEW comparison is a compatibility-failure baseline: direct UVA-like expert matmul is not runnable, whereas SEW stages experts into HBM fixed slots before grouped MLP.

