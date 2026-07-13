# Ascend UVA-like Expert Access Feasibility

Date: 2026-07-09

## Question

Can CUDA UVA-style expert offload be ported to Ascend NPU serving, and if it
can, is it a viable baseline against SEW-Offload under the same 14 GiB offload
budget?

This question must be answered before claiming that Ascend "has no equivalent
path" to CUDA UVA. CANN 9.0 exposes host-memory registration and managed-memory
symbols, so the real question is whether those primitives form a usable path
for graph-captured MoE expert computation.

## Hypothesis Under Test

CUDA vLLM's UVAOffloader works because offloaded parameters are exposed as
stable CUDA-visible host-backed views. The captured CUDA graph sees stable
weight pointers, while the kernel pays remote host-memory access cost.

The Ascend equivalent would require all of the following:

1. **Runtime mapping:** a 14 GiB host expert store can be registered and mapped
   to a device-visible pointer.
2. **AI Core consumption:** Ascend C / grouped-MLP kernels can read that mapped
   pointer as ordinary expert-weight input.
3. **ACLGraph stability:** replay can capture a kernel that reads the mapped
   address, and replay observes host-side content updates with explicit
   synchronization.
4. **Framework integration:** the mapped pointer can be exposed as a
   `torch_npu`/vLLM-Ascend tensor or an operator input without rewriting the
   entire MoE path.
5. **Performance viability:** host-backed direct access is competitive with, or
   at least meaningfully close to, explicit H2D staging into SEW fixed NPU
   slots.

Failing any one of these gates means "Ascend UVA-like offload" is not a direct
replacement for SEW. It may still be useful as a diagnostic baseline.

## Current Runtime Probe

Script:

```bash
python benchmark/scripts/ascend_uva_probe.py \
  --device-id 4 \
  --size-mib 14336 \
  --skip-malloc-host \
  --skip-managed \
  --out benchmark/artifacts/reports/ascend_uva_feasibility/probe_device4_14gb_register_only.json
```

Environment:

- CANN: `/usr/local/Ascend/cann-9.0.0`
- Device: Ascend 910B2, NPU 4
- Probe size: 14 GiB (`14336` MiB), matching the SEW primary offload budget

Observed result:

| Probe | Result | Interpretation |
|---|---|---|
| `aclInit` + `aclrtSetDevice(4)` | success | Runtime can initialize the target NPU. |
| `aclrtHostMemMapCapabilities` | returns `207000` for all HAC types | Capability query did not provide a positive support signal on this stack. |
| `aclrtHostRegister` legacy path | success, returns device pointer | A 14 GiB host allocation can be registered into a device-visible address range. |
| `aclrtHostRegisterV2` | fails with `107000` | The newer V2 + `aclrtHostGetDevicePointer` path is not usable with current flags. |
| `aclrtMemAllocManaged` | not tested in 14 GiB run; 1 MiB run failed with `100000` | Managed/UVM-like allocation is not currently usable through the default flag path. |

The first gate is therefore **partially passed**: legacy host registration can
produce a device pointer at 14 GiB scale. This is not enough to claim CUDA UVA
equivalence.

## Framework Tensor Probe

Script:

```bash
/root/miniconda3/bin/conda run -n vllm-hust-dev python \
  benchmark/scripts/ascend_uva_tensor_probe.py \
  --device-id 4 \
  --size-mib 1 \
  --out benchmark/artifacts/reports/ascend_uva_feasibility/probe_device4_tensor_wrap_1mib.json
```

Observed result:

| Probe | Result | Interpretation |
|---|---|---|
| `aclrtHostRegister` | success | The host range is mapped to a device pointer. |
| `torch_npu._C._construct_storage_from_data_pointer` | success | The private torch_npu storage constructor can wrap the mapped pointer. |
| `torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata` | success | The pointer can be represented as a `npu:4` tensor in metadata. |
| `torch_npu._C._check_npu_data_ptr` | `true` | torch_npu accepts the pointer as NPU data pointer metadata. |

This means the framework layer is **not immediately blocked at tensor
construction**. However, this still does not prove safe reads.

Access-matrix probe:

```bash
/root/miniconda3/bin/conda run -n vllm-hust-dev python \
  benchmark/scripts/ascend_uva_tensor_matrix.py \
  --device-id 4 \
  --size-mib 1 \
  --artifact-dir benchmark/artifacts/reports/ascend_uva_feasibility \
  --out benchmark/artifacts/reports/ascend_uva_feasibility/probe_device4_tensor_access_matrix.json
```

Observed result:

| Probe | Result | Interpretation |
|---|---|---|
| Construct `uint8` tensor | success | Metadata-level wrapping is possible. |
| `tensor[:16].cpu()` | process terminated by signal 11 | Direct D2H copy from the wrapped pointer is unsafe on this stack. |
| NPU-to-NPU `device_copy` | child process returned, but operation failed with runtime error `507001` and SDMA task exception | SDMA copy does not treat this pointer as ordinary safe tensor memory. |
| AI Core elementwise `add` from `float16` host-registered source | success | At least one AI Core compute path can read the mapped host pointer correctly. |

Bandwidth probe:

| Input | Path | Avg time | Approx source-read bandwidth | Relative to HBM |
|---|---:|---:|---:|---:|
| 64 MiB `float16` | host-registered `add` | 7.189 ms | 8.69 GiB/s | 2.18% |
| 64 MiB `float16` | HBM-resident `add` | 0.157 ms | 399.21 GiB/s | 100% |
| 256 MiB `float16` | host-registered `add` | 27.608 ms | 9.06 GiB/s | 2.67% |
| 256 MiB `float16` | HBM-resident `add` | 0.737 ms | 339.31 GiB/s | 100% |

Therefore:

- Runtime host registration is possible.
- Private torch_npu tensor wrapping is possible.
- Safe tensor data access is **operation-dependent**: AI Core elementwise reads
  can work, but copy/SDMA/D2H paths failed or crashed.
- The working elementwise read path is roughly **37x-46x slower than HBM** in
  this microbenchmark.

This is a strong warning that "constructable as a tensor" is not equivalent to
"usable as an expert weight input." U1 is partially supported for a simple
AI Core elementwise read, but grouped-MLP expert access and ACLGraph replay
remain separate gates.

## Graph Replay Probe

Script:

```bash
/root/miniconda3/bin/conda run -n vllm-hust-dev python \
  benchmark/scripts/ascend_uva_graph_probe.py \
  --device-id 4 \
  --size-mib 1 \
  --elements 1024 \
  --replay-values 2,4 \
  --out benchmark/artifacts/reports/ascend_uva_feasibility/probe_device4_npugraph_replay_1mib_1024.json
```

Observed result:

| Probe | Result | Interpretation |
|---|---|---|
| Eager `out.copy_(src + 1)` | output sample is all `1.0` | Simple compute can read the initially zero host-registered source. |
| `torch.npu.graph` capture | capture succeeds; output sample is all `1.0` | The simple compute path is capturable. |
| Replay after writing host value `2.0` | output sample is all `3.0` | Replay observes host-side content update. |
| Replay after writing host value `4.0` | output sample is all `5.0` | Replay is not merely reusing the value observed during capture. |

This means U2 is **partially passed for a simple torch_npu elementwise graph**.
Graph replay itself is not the first observed hard blocker. However, this is
still much weaker than proving vLLM ACLGraph replay of grouped MoE kernels.

## MoE-Shaped Matmul Probe

Script:

```bash
/root/miniconda3/bin/conda run -n vllm-hust-dev python \
  benchmark/scripts/ascend_uva_matmul_probe.py \
  --device-id 4 \
  --m 16 \
  --k 4096 \
  --n 4096 \
  --warmup 1 \
  --repeat 3 \
  --verify-output \
  --out benchmark/artifacts/reports/ascend_uva_feasibility/probe_device4_matmul_m16_k4096_n4096.json
```

Observed result:

| Shape | Weight path | Result | Approx weight-read bandwidth | Interpretation |
|---|---|---:|---:|---|
| `16 x 1024 @ 1024 x 1024` | HBM resident | success, output verified | 26.96 GiB/s | HBM reference works. |
| `16 x 1024 @ 1024 x 1024` | host-registered | fails with runtime error `507057` | n/a | MatMul path does not accept the host-registered weight pointer. |
| `16 x 4096 @ 4096 x 4096` | HBM resident | success, output verified | 301.62 GiB/s | HBM reference works. |
| `16 x 4096 @ 4096 x 4096` | host-registered | fails with runtime error `507057` | n/a | Failure persists at a larger expert-weight tile. |

The same host-registered pointer class that works for elementwise `add` fails
for matrix multiplication. This is the strongest current evidence against a
straight CUDA-UVA-style port for MoE expert weights on this stack: MoE execution
is dominated by matrix/grouped-matrix kernels, not by elementwise reads.

## Aggregate Verdict

Aggregate artifacts:

```bash
python benchmark/scripts/run_uva_feasibility.py \
  --out benchmark/artifacts/reports/ascend_uva_feasibility/e0_runner_manifest.json

python benchmark/scripts/render_uva_feasibility_report.py \
  --out docs/experiments/ascend_uva_like_report.md
```

The full YAML-driven runner completed with `status=ok`. It treats
`tensor_access_matrix`, `matmul_probe_2mib`, and `matmul_probe_32mib` as
expected-nonzero probes because their failures are the evidence under test.

| Command | Return code | Runner status | Meaning |
|---|---:|---|---|
| `small_runtime_probe` | 0 | `ok` | 1 MiB runtime probe completed. |
| `budget_register_probe` | 0 | `ok` | 14 GiB host-register probe completed. |
| `tensor_access_matrix` | 1 | `expected_nonzero` | Copy/SDMA failure evidence recorded while elementwise read passes. |
| `npugraph_replay_probe` | 0 | `ok` | Simple graph replay observes host updates. |
| `matmul_probe_2mib` | 1 | `expected_nonzero` | Host-registered small matmul failure evidence recorded. |
| `matmul_probe_32mib` | 1 | `expected_nonzero` | Host-registered larger matmul failure evidence recorded. |
| `collect_summary` | 0 | `ok` | CSV summary and verdict JSON regenerated. |

The generated paper-ready report is `docs/experiments/ascend_uva_like_report.md`.

Current machine-readable verdict:

| Field | Value |
|---|---|
| `verdict` | `not_viable_as_sew_baseline` |
| `comparison_to_sew` | `compatibility_failure_not_latency_throughput_comparison` |
| `primary_blocker` | `host_registered_matmul_weight_path_fails_507057` |

This is the correct comparison boundary against SEW. Since direct
host-registered expert-weight matmul does not run while the same HBM-resident
matmul does run, the UVA-like path should be reported as a compatibility
failure baseline rather than assigned TTFT/TPOT/throughput numbers. SEW remains
the runnable path because it stages experts into HBM fixed slots before grouped
MLP execution.

## Remaining Gates

### Gate U1: AI Core Readability

Current status: **partially passed for a simple torch_npu elementwise compute
path**. A wrapped host-registered `float16` tensor can be read by an AI Core
`add` path and produce correct HBM output. However, this is not enough for MoE:
grouped MLP may use different kernels, layouts, DMA behavior, and graph capture
contracts. The current matmul probe shows that at least the tested
`torch.matmul` path fails on host-registered weights.

Next, build a minimal Ascend C kernel that reads a registered host pointer and
writes a checksum or copy result to NPU HBM. Run the same kernel with:

- HBM input allocated by `aclrtMalloc`
- host-registered input returned by `aclrtHostRegister`

Evidence:

- success/failure return code
- checksum correctness
- kernel time
- effective GB/s

If the host-registered pointer cannot be used by AI Core kernels, the
CUDA-UVA-like path fails at the kernel-consumption layer.

### Gate U2: ACLGraph Replay

Current status: **partially passed for a simple torch_npu NPUGraph elementwise
path**. Replay observes host-side content updates. Remaining work is to test
the actual vLLM ACLGraph path and grouped-MoE kernels.

Evidence:

- graph capture completed for simple elementwise compute
- replay observed updated host contents
- grouped-MLP/vLLM ACLGraph evidence is still missing

If replay does not observe correct data, the path fails at the graph-stability
or memory-consistency layer.

### Gate U3: MoE-like Weight Access

Current status: **failed for torch_npu matmul with host-registered weights**.
HBM references pass for the same shapes, but host-registered weights fail with
runtime error `507057`. This suggests the blocker is not address registration
or tensor metadata; it is the matrix/MatMul operator path over remote
host-backed weight memory.

Still compare direct host-registered reads with SEW explicit H2D staging for a
full MoE-shaped matrix access if a lower-level custom Ascend C kernel is later
implemented.

Minimum shape should approximate one Qwen3-30B-A3B expert projection tile and
then scale toward the 14 GiB offloaded expert store. Compare:

| Path | Description |
|---|---|
| HBM resident | Expert weight stays in NPU HBM. |
| Ascend UVA-like | Kernel reads host-registered expert weight directly. |
| SEW fixed slot | Expert is copied H2D into fixed NPU slot, then grouped MLP reads HBM. |

Metrics:

- TTFT/TPOT micro-equivalent when integrated
- raw kernel time
- remote-read bandwidth
- explicit H2D bytes and staging time
- ACLGraph replay status

### Gate U4: vLLM-Ascend Integration

Only after U1-U3 pass should we attempt a vLLM integration. The integration
question is whether the host-registered pointer can be presented to the
vLLM-Ascend grouped MLP path without breaking tensor metadata, format
conversion, ACLGraph capture, or operator launch contracts.

## Decision Matrix

| Outcome | Meaning for the paper |
|---|---|
| U1 fails broadly | Existing CANN mapping primitives are not enough for expert-weight reads; SEW remains necessary for device-local execution. |
| U1 passes only for simple compute paths | Ascend has a partial remote-read mechanism, but it is not a drop-in CUDA-UVA-equivalent tensor interface. |
| U2 passes only for simple compute paths | Graph replay can observe host updates, but this does not rescue MoE if grouped matrix kernels reject the pointer. |
| U3 fails for matrix kernels | Ascend-UVA-like expert access is not a viable drop-in MoE baseline; SEW remains necessary to stage experts into HBM before grouped MLP. |
| U1/U2/U3 pass but U3 is slow | Ascend-UVA-like access is a correctness/compatibility baseline, but SEW is needed for performance by staging hot experts into HBM. |
| U1-U4 pass and performance is competitive | SEW fixed slots alone are not enough novelty; pivot to a hybrid design or treat Ascend-UVA-like offload as the stronger baseline. |

## Current Claim Boundary

Allowed:

> On the tested CANN 9.0 / 910B2 stack, a 14 GiB host allocation can be
> registered through the legacy `aclrtHostRegister` API and mapped to a
> device-visible pointer. Private `torch_npu` APIs can wrap the pointer as an
> NPU tensor, and a simple AI Core elementwise read can consume it correctly.
> However, copy/SDMA/D2H paths failed or crashed, and the successful elementwise
> read path is roughly 37x-46x slower than HBM in 64-256 MiB microbenchmarks.
> A simple `torch.npu.NPUGraph` replay can observe host-side content updates,
> but `torch.matmul` with host-registered weights fails with runtime error
> `507057` even for a 2 MiB weight tile. Therefore the current blocker is the
> matrix/grouped-MLP operator path rather than only runtime address mapping.

Not allowed:

> Ascend has no UVA-like mechanism.

Not allowed yet:

> Ascend UVA-like offload is slower than SEW.

Current U3 evidence supports a narrower negative claim: host-registered
`torch.matmul` failed while HBM-resident matmul succeeded for the same shapes.
A full "slower than SEW" claim still requires SEW-vs-UVA comparison under a
common MoE-shaped or vLLM-serving harness; if direct host matmul cannot be made
to run, report it as a compatibility failure rather than a performance number.
