# Issue 17 Matched TTFT Evidence

This record closes the bounded matched A/B follow-up requested by
[Issue #17](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/issues/17).
It replaces the preliminary cross-campaign 31.5% comparison from Issue #13
with three matched pairs. It does not broaden the qualified configuration.

## Matched Contract

- Qwen3-30B-A3B, BF16, TP1 on one physical Ascend 910B2 (NPU6)
- ShareGPT `mixed_chat`, the same ordered 200-request manifest in every unit,
  128 output tokens per request, client concurrency 1
- 14 GiB host offload budget, 32 slots, `max_num_seqs=1`
- prefix cache disabled; PIECEWISE ACLGraph capture and replay required
- one fresh managed service start and release ACK per unit
- fixed AB/BA/AB order: full, wave, wave, full, full, wave
- the two cases differ only in
  `VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE`

All six units used repository runtime `31621de`, vLLM `ad7125a`, and
vLLM-Ascend `4806367`. The model config SHA-256 is `2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297`;
the request manifest SHA-256 is `3ec21bcb1174b48943cbae4b8f588711041ed9f173f584898149028ebb63ba9a`.

## Per-Repeat Results

| Order | Arm | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | Throughput |
|---:|---|---:|---:|---:|---:|---:|
| 1 | full_layer | 927.06 ms | 970.75 ms | 53.87 ms/tok | 56.72 ms/tok | 16.34 tok/s |
| 2 | multi_wave | 602.45 ms | 759.96 ms | 55.34 ms/tok | 58.94 ms/tok | 16.59 tok/s |
| 3 | multi_wave | 600.65 ms | 755.68 ms | 54.42 ms/tok | 57.68 ms/tok | 16.83 tok/s |
| 4 | full_layer | 931.90 ms | 989.53 ms | 54.74 ms/tok | 57.78 ms/tok | 16.09 tok/s |
| 5 | full_layer | 921.45 ms | 980.89 ms | 54.33 ms/tok | 57.80 ms/tok | 16.16 tok/s |
| 6 | multi_wave | 598.72 ms | 752.12 ms | 53.87 ms/tok | 57.12 ms/tok | 17.02 tok/s |

The median of the three repeat-level TTFT p50 values is 927.06 ms for
`full_layer` and 600.65 ms for `multi_wave`, a 35.21% reduction. The
corresponding p95 medians are 980.89 and 755.68 ms, a 22.96% reduction.
Pair-level p50 reductions are 35.02%, 35.55%, and 35.02%; pair-level p95
reductions are 21.71%, 23.63%, and 23.32%.

TPOT remains comparable rather than showing the same reduction. The three
repeat p50 values are 53.87/54.74/54.33 ms/token for `full_layer` and
55.34/54.42/53.87 ms/token for `multi_wave`. The supported conclusion is a
TTFT improvement for this Prefill-heavy boundary, not a general Decode speedup.

## Correctness and Runtime Gates

Each arm completed 600/600 requests and 76,800 output tokens. All five units
after the first oracle matched its 200 request IDs and 25,600 token IDs
exactly; therefore all six repeats produced the same output token arrays.
Every unit captured and replayed PIECEWISE Graph. Each `full_layer` unit
recorded 2,412 full-layer events and no wave events; each `multi_wave` unit
recorded 2,412 wave events and no full-layer events. There were zero fallback
events and zero NPU/ACL/OOM forbidden markers. Every managed service emitted a
release ACK, and the final physical HBM sample returned to 5% after every unit.

The complete raw 200-point TTFT/TPOT distributions, request token arrays,
commands, environments, provenance, server/client logs, profiles, physical NPU
samples, verifier reports, and release ACKs are stored in the portable bundle
`docs/evidence/bundles/issue-17-matched-ttft-31621de.tar.gz`.
The archive is 9.9 MiB with SHA-256
`de682c4073263901cc632a7f3542509a9158bd733558dbbe548bb8ae4b4a8ab4`.
After extraction, all 66 packaged evidence files pass the included
`SHA256SUMS`; rerunning all six unit verifiers and the campaign verifier from
the portable relative paths reproduces the passing summary above.

## Claim Boundary

The earlier 31.5% number compared different campaigns and remains historical
diagnostic context only. The matched result above supersedes it. This evidence
does not claim the same reduction for another model, device, offload budget,
slot count, serving concurrency, prefix-cache mode, eager mode, or workload.
