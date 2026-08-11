# Issue 11 Transfer Readiness Evidence

This record covers the supported LatchMoE path fixed for
[Issue #11](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/issues/11).
It is intentionally compact: raw server logs and per-request token arrays stay
in the immutable host evidence directory, while their SHA-256 digests are
recorded here.

## Scope

- Model: Qwen3-30B-A3B, BF16, unquantized MoE
- Device: one Ascend 910B-class NPU, TP1
- Serving shape: `max_num_seqs=1`, client concurrency 1
- Workload: 200 ShareGPT `mixed_chat` requests, 128 output tokens each
- Offload: 14 GiB budget, 12 offloaded layers, 32 slots
- Supported overflow path: `full_layer`
- Prefix cache: disabled
- Compared modes: Eager and PIECEWISE ACLGraph

The `experimental_wave` overflow mode and prefix-cache reuse remain outside the
correctness contract.

## Provenance

- Pre-PR `main`: `ea1c3603d0784545652f3b76aa96c22a086c9914`
- Validated runtime fix: `c82dc066db8788fbc288eac13da50d85efe35473`
- vLLM-Ascend seam: `fffbd1eb75db455e4c90dfb2b8455d0e66ff5b25`
- Raw evidence root:
  `/home/changwu/latchmoe-evidence/issue11-close-c82dc06-20260811`

## Results

| Gate | Graph | Eager |
|---|---:|---:|
| Successful requests | 200/200 | 200/200 |
| Failed requests | 0 | 0 |
| Output tokens | 25,600 | 25,600 |
| Median TTFT | 871.42 ms | 1,030.84 ms |
| Median TPOT | 54.89 ms/token | 200.07 ms/token |
| Prefix-cache hit rate | 0.0% | 0.0% |
| Runtime/NPU errors | 0 | 0 |

The Graph log contains `Replaying aclgraph`. Exact comparison by `request_id`
found 200/200 matching `output_token_ids`. Both profiles contain 12 layer
registrations and 12/12 original-expert-weight releases. The final memory
ledger reports 14,495,514,624 host-store bytes and 3,623,878,656 slot-bank
bytes.

Host tests at the validated fix commit: `247 passed, 21 warnings`.
Host tests after the PR launcher/documentation cleanup:
`248 passed, 21 warnings`.

## Artifact Digests

| Artifact | SHA-256 |
|---|---|
| `graph/benchmark.json` | `297df6a7c3d8619260d80aa17de13f41fe2c568d7f8ae2ae56cde62d77145fb0` |
| `graph/server.log` | `f5ee19ee52aeaf34af27b4b9b26677d56a5e31f0d1a5455171966f54d27c9ffc` |
| `graph/moe_profile.jsonl` | `380f2e82413951fd1d8c782e623fbf55eb6cdd02e5f0d7ea80e09915204892c4` |
| `eager/benchmark.json` | `ce22f8adaf588369bdd13e454ee604eb12709e7823c630578c5c0bec4ef0ac07` |
| `eager/server.log` | `51ecd59d27df16f7af7b58db17cc793774dc91394e90f75b3ac438972f3392bd` |
| `eager/moe_profile.jsonl` | `c30f9a3cb2a12b65b2062b412afd3f4229810499621378cd9094689298804b0c` |

## Root Cause and Rollback Boundary

The primary defect was a publication-order race: a slot mapping could become
visible before the exact consumer stream had a dependency on the H2D event.
The fixed order is enqueue copy, install the consumer-stream wait, validate the
slot lease/version, publish READY, then publish the mapping. D2D content checks
did not find expert misregistration.

The pull request must use a merge commit. The pre-merge `main` SHA and the PR
merge SHA form the rollback boundary; reverting that merge commit removes the
Issue #11 runtime, launcher, benchmark, documentation, and test changes as one
unit without rewriting `main` history.
