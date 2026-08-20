# Issue 28: Qwen3-Next qualification boundary

This record captures the Qwen3-Next result obtained after Issue 26 was closed.
It is a correctness-reachability record, not a matched performance result.

## Successful LatchMoE run

- Environment: `/root/miniconda3/envs/latchmoe`
- Model: `/workspace/shared_models/strict-models/Qwen3-Next-80B-A3B-Instruct`
- One Ascend NPU (device 5), BF16, TP1, PIECEWISE
- `max_num_seqs=1`, four slots, 128 MiB KV-cache reservation
- CPU-first load, original expert-weight release, layered runtime, wave prefill
- Explicit qualification-only setting:
  `LATCHMOE_ENABLE_GDN_PYTORCH_FALLBACK=1`

The run completed with `status=ok`, one request, and one generated token. The
output token ID was `[576]` (`" The"`). The recorded request TTFT was
12,881.62 ms and model loading took 1,206.60 s. These timings include a
qualification path with reference fallbacks and must not be used as a
performance comparison.

The run exercised four compatibility boundaries that were absent from the
locked CANN/vLLM-Ascend combination:

1. causal-conv state update: explicit opt-in PyTorch depthwise fallback;
2. fused GDN gating: explicit opt-in PyTorch fallback;
3. recurrent GDN rule: bridge to the public `torch_npu` operator;
4. chunk GDN prefill: explicit opt-in reuse of the existing 310P reference
   implementation.

The run also completed model compilation, PIECEWISE graph capture, staged
loading, expert-weight release, and the actual request. The generated
`summary.json` reported `graph_compatible_offload=true` and
`wave_prefill=true`.

## Why this does not close the matched gate

The native Qwen3-Next attempt under the same small single-card memory budget
failed during model construction because the device could not allocate the
required resident memory. It therefore has no native output or latency sample
to pair with the successful LatchMoE request. The successful run itself also
reported `router_parity=false` and `layer_boundary_parity=false`.

Consequently this record supports only “the gated/internal-router tuple can
reach one request with the qualification fallbacks enabled.” It does not
support exact native-vs-LatchMoE equality, cross-architecture generality,
decode or mixed-chat claims, pressure-point claims, or any performance claim.
The fallback is intentionally opt-in and fails closed for speculative decoding.
