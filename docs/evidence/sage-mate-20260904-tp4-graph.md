# Sage Mate TP4 graph qualification — 2026-09-04

## Verdict

`Qwen3-30B-A3B` BF16 is functionally compatible with LatchMoE on Ascend TP4
in PIECEWISE graph mode for the exact Sage Mate baseline below. Performance is
degraded, so this result does not establish an optimization benefit.

Dense `Qwen3.8-27B` has no routed experts and is **Not Applicable** to
LatchMoE.

## Immutable inputs

- vLLM-HUST: `762f85b311fbab0bcf8921dd216f5093cd58b9b8`
  (`0.28.1rc1.dev319`)
- vLLM-Ascend-HUST seam: `2c8c722107a54127999a64c4eb0ec86139df8c26`
  (based on `4e57439e`, seam ABI 2)
- LatchMoE: `63781f3dd0235f933735bfd8ce614d388093c0b5`
- Wheel SHA-256:
  `dd74be3ea4d1be44ffeda19a2366672e594c2647959e06fb079cbcc80456f99b`
- Image: `sage-mate/mod-compat:latchmoe-r011`
- Image ID:
  `sha256:7f3d5a5ed9c30070dfcbf6d55c5e6a0672110b701a2d9fa17fbe79b4499776ac`
- Model: local ModelScope `Qwen/Qwen3-30B-A3B`, BF16, 128 routed experts,
  8 selected per token
- Topology: Ascend NPU 0–3, TP4, PP1, PIECEWISE graph; NPU 4–7 were not used

## Runtime evidence

- All four ranks called the current router and MLP seams during both capture
  and live execution. Each rank recorded 324 capture-time MLP calls; live
  concurrency recorded 96 router and 96 MLP calls per rank.
- 388 graph replay events completed. All 48 address validations matched the
  capture fingerprint; zero address mismatches were recorded.
- The host store held 3,623,878,656 bytes. Device slot allocation totalled
  1,056,976,896 bytes, and the original offloaded weights were released.
- Four simultaneous HTTP requests all returned status 200 with correct
  answers (`143`, `413`, `Au`, and `分布式系统`).
- A streamed request was cancelled after its first chunk; the running-request
  gauge returned from 1 to 0 in about 0.53 seconds. Two subsequent requests
  returned 200. Their unconstrained generated text was not byte-identical,
  which is recorded rather than hidden.
- A malformed request returned 400; the following valid request returned 200
  and `81`. The service remained healthy.
- Explicit correctness probes returned `689` for `37 * 19` and `分布式系统`
  for the translation prompt.
- Rollback restored the original Qwen3.8-27B TP4 graph service, which returned
  HTTP 200 and `LATCHMOE_ROLLBACK_OK`.

## Performance boundary

For ten requests, LatchMoE measured TTFT p50/p95 3.678/7.730 seconds,
end-to-end latency p50/p95 25.941/31.808 seconds, and output throughput
p50/p95 2.907/3.010 tok/s. The matched no-plugin baseline output throughput
was about 23.57 tok/s. This lane is therefore labelled
functionally `compatible`; effectiveness is `not-beneficial-in-tested-cell` for
this exact measured configuration and must not be advertised as a
speedup.

## Evidence custody

The raw local bundle is retained by the parent campaign at
`results/sage-mate-mod-compat-20260904T053524Z/latchmoe-r001/candidate-r011/`.
Key SHA-256 values:

- `chat-benchmark.json`: `9f085b143d10929e94938bbdc6d3cf080ef8afb7d739c93ec0239e9a63abe272`
- `concurrency-cancel-recovery.json`: `eba9af9636d41b464cfa4082ff53a134508062084efa334356d3619e9e466663`
- `seam-graph-events.log`: `c7eb59b9640406f3848321e211e1da457cb600fbaaae6c8b84bd9801670f2e91`
- `production-rollback-response.json`: `a18397218c30914ba358e7039ab875cea8aa79bc9753eb10f5838da83bb9816c`

The repository host suite passed 396 tests with 2 skips. Skips cover optional
environmental paths and are not substituted for the device qualification.
