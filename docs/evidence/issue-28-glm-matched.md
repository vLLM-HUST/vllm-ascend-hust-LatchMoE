# Issue 28: GLM matched campaign evidence

This record is the narrow, reproducible evidence unit produced for Issue 28.
It is deliberately not a general performance claim.

## Frozen boundary

- Campaign: `issue28-glm-prefill-v1`
- Model: GLM-4.7-Flash, BF16, config SHA256
  `dc9b97c7c9bed726a2e6939da4234d5c43abb3edec8812068c9a1af1dbc13acb`
- Device: one Ascend NPU, device 5, 65,536 MB HBM
- TP: 1; `max_num_seqs=1`; four LatchMoE slots; prefix cache disabled
- Graph mode: PIECEWISE
- KV cache reservation: 512 MB
- Frozen request manifest: one prefill-heavy request, SHA256
  `5c0a46fa93b4782c056923088fb5d88c5286148fca59652d414002961c368b1a`
- Campaign contract SHA256:
  `0e2749ec6291e1ad321cc59a5e8c1af4ea7211a6492c384de685ccf4d8a7fef6`

The raw campaign directory is retained at
`/workspace/latchmoe-issue28-campaign-r5` on the qualification host. The
checked-in contract and verifier are the reproducibility anchors; raw unit
files are not copied into the repository because they contain host-specific
paths and logs.

## Acceptance result

The campaign executed all 15 units in the frozen order and the verifier
returned `status=passed` with no failures.

| Arm | Accepted repeats | Result | Repeat-level TTFT median / p95 (ms) |
|---|---:|---|---:|
| `full_resident` | 3 | 3 accepted capacity failures | — |
| `native_prefetch` | 3 | success | 948.46 / 961.03 |
| `legacy_layered` | 3 | success | 941.39 / 941.44 |
| `latchmoe_full_layer` | 3 | success | 1314.35 / 1378.09 |
| `latchmoe_multi_wave` | 3 | success | 1149.05 / 1387.79 |

The verifier also reports repeat-level TPOT and throughput. Their medians are
respectively 370.84 ms and 1.653 token/s for `native_prefetch`, 368.20 ms and
1.666 token/s for `legacy_layered`, 63.86 ms and 2.548 token/s for
`latchmoe_full_layer`, and 68.82 ms and 2.731 token/s for
`latchmoe_multi_wave`.

All successful arms produced the same output token IDs:

```text
issue26_prefill_heavy_0000 -> [785, 1196, 374, 10156]
```

The HBM peak was 60,948.48 MB for the native/legacy arms and 59,637.76 MB for
the two LatchMoE arms. The three full-resident failures were accepted only
because the contract explicitly declares capacity failure as an expected
outcome for that arm; they are not treated as successful runs.

## What this supports

Within this exact GLM trigger, the external-shared/complex-router path reaches
the same output as the matched reference arms under PIECEWISE execution, and
the fixed campaign machinery can distinguish a valid run from a declared
capacity failure.

## What this does not support

This is one architecture, one device, one request shape, one slot setting, and
one prefill-heavy trigger. It does not close the Issue 28 three-architecture
generality gate, does not establish broad performance superiority, and does
not provide decode, mixed-chat, pressure-point, or multi-wave sensitivity
evidence. In particular, the reported arm medians must not be presented as a
claim that LatchMoE is faster than every reference path.
