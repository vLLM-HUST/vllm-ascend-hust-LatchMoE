# Issue 13 Multi-Wave Prefill Evidence

This record covers the native-recombine multi-wave Prefill fix for
[Issue #13](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/issues/13).
Raw server logs, profiles, and per-request token arrays remain in the host
artifact directory; their SHA-256 digests are recorded below.

## Scope

- Model: Qwen3-30B-A3B, BF16, unquantized MoE
- Device: physical NPU5, one Ascend 910B2, TP1
- Serving shape: `max_num_seqs=1`, client concurrency 1
- Workload: ShareGPT `mixed_chat`, 128 output tokens per request
- Offload: 14 GiB budget, 12 offloaded layers, 32 slots
- Multi-wave: default `multi_wave`, native layer-level top-k recombine
- Overlap: asynchronous H2D, prefetch depth 1, two prefill stage buffers,
  transfer-aware scheduling
- Prefix cache: disabled; observed hit rate 0.0%
- Graph: PIECEWISE ACLGraph with `vllm::moe_offload_stage` as a splitting op

This qualification promotes `multi_wave` to the default overflow mode. The
blocking `full_layer` path remains an explicit mode and the automatic fallback
for recoverable multi-wave preflight/native-recombine qualification failures.
NPU/ACL, OOM, and arbitrary runtime failures are not swallowed. The result does
not extend to higher serving concurrency, prefix-cache reuse, or other models
and slot counts; force `full_layer` outside the validated boundary.

## Root Cause and Fix

The former wave path combined each wave independently and then accumulated
wave outputs. That changed the BF16 evaluation order relative to the native
single-pass MoE combine and could eventually change generated tokens even when
all transfers were complete.

The fixed path preserves every routed pair's original flattened top-k offset.
It concatenates wave-local permuted MLP outputs, reconstructs one global native
unpermute index, and invokes the dispatcher's ordinary `token_combine` exactly
once with the original top-k weights and output shape. Incomplete or duplicate
pair coverage fails closed under strict validation. The non-equivalent
wave-local combine fallback is rejected.

## Correctness Gates

The same 11 historical trigger requests were compared by `request_id` against
the correctness-first `full_layer` Eager oracle:

| Mode | Requests | Compared output tokens | Exact matches |
|---|---:|---:|---:|
| serial multi-wave Eager, strict validation | 11/11 | 1,408 | 1,408 |
| overlapped multi-wave Eager | 11/11 | 1,408 | 1,408 |
| overlapped multi-wave PIECEWISE Graph | 11/11 | 1,408 | 1,408 |

An additional Ascend 910B2 tensor-level check compared the original
`torch_npu.npu_moe_token_unpermute` with the rebuilt native combine inputs for
136 BF16 routed pairs split across five interleaved waves. The result was
bitwise equal (`torch.equal=True`, maximum absolute difference 0.0).

The requested 200-request run was used as the Graph stability and performance
gate. Per the final qualification decision, it was not repeated against a new
200-request `full_layer` Graph oracle; the exact token gate above therefore
remains the 11 historical trigger requests.

## 200-Request Graph Result

| Metric | Result |
|---|---:|
| Successful requests | 200/200 |
| Failed requests | 0 |
| Output tokens | 25,600 |
| TTFT p50 | 596.62 ms |
| TTFT p95 | 755.86 ms |
| TTFT p99 / max | 811.15 / 848.15 ms |
| TPOT p50 | 51.79 ms/token |
| TPOT p95 | 55.91 ms/token |
| Wall time | 1,442.31 s |
| Output throughput | 17.75 token/s |
| Prefix-cache hit rate | 0.0% |
| Runtime/NPU/ACL errors | 0 |

Compared with the Issue #11 no-prefix-cache `full_layer` Graph TTFT p50 of
871.42 ms, multi-wave improves p50 by 31.5%. Its p95 of 755.86 ms is also below
that historical p50 baseline. The 200-request run completed after successful
PIECEWISE compilation and capture; the graph pool used 0.02 GiB.

## Overlap and Memory Evidence

The Graph qualification profile (startup/warmup, the 11-request correctness
gate, and the 200-request stability run) contains 2,544 multi-wave
prefill-layer events and 9,077 waves. Every wave was issued before its compute
began. Of these, 6,533
were next-wave prefetch issues submitted before the current wave's compute;
none were late, after-compute issues. Aggregate stage wait was 1,593.87 ms,
versus 7,493.41 ms of profiled MLP time. The maximum per-layer stage wait was
1.27 ms. The only observed combine mode was `native_recombine_payload`.

Profiled transfer volume was 1,678,742,913,024 H2D bytes in total, with a
maximum of 1,207,959,552 bytes for one prefill-layer event. The final memory
ledger reported:

| Allocation | Bytes | GiB |
|---|---:|---:|
| host expert store | 14,495,514,624 | 13.50 |
| persistent slot bank | 3,623,878,656 | 3.38 |
| two-buffer prefill stage bank | 603,979,776 | 0.56 |
| total NPU slot/stage storage | 4,227,870,720 | 3.94 |
| PIECEWISE graph pool | reported as 0.02 GiB | 0.02 |

Original NPU expert weights were released for all 12 managed layers. Periodic
`npu-smi` samples before, during, and after the completed run reported 91% HBM
utilization on the 65,536 MB device, with no OOM or memory-growth failure.

## Artifact Digests

Host artifact root:
`benchmark/artifacts/issue13-npu5` (ignored by Git because the raw profile is
large).

| Artifact | SHA-256 |
|---|---|
| `full_layer_eager/benchmark_11.json` | `d46be2928213de385a94a08bc71eaa1fef2eb0e72fccd3ac9b10957fb6ba5692` |
| `wave_serial_eager/benchmark_11.json` | `57b210b9f8c30bc96f3f3b182bfc83a515d4266d60cc0183e28fe7edac68566e` |
| `wave_overlap_eager/benchmark_11.json` | `892d5fb12dcefd0a44705b834be6b584f24ec32d30b3802795d6a6ae91ca7d64` |
| `wave_overlap_graph/benchmark_11.json` | `5c32b5d0e7af3ba96a0a2ff44f36d36b58f74c959aa129cd5fca3abe6092f2d4` |
| `wave_overlap_graph/benchmark_200.json` | `36fbdcffe28609c1558c27e195a48cf48dec909c46642333fdd1173f6e4d7b9f` |
| `wave_overlap_graph/server.log` | `a0de65ce7d4d4b7434c2b4af5a3c9096c928aa5ee1e32aad8928090d8fcdd2fa` |
| `wave_overlap_graph/moe_profile.jsonl` | `4d948e46f4b363fe04399c274fa634328d3bf64cc45f67737c0bad3de5f07231` |
