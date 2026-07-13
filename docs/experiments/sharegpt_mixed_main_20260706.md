# ShareGPT Mixed-Chat Formal Run

This is the first formal ShareGPT result prepared for the Wednesday paper
presentation. It should be used as the current performance slide. It is still
a single run, so the paper should later add repeated runs and baseline
comparisons.

## Experiment Setting

| Item | Setting |
|---|---|
| Case | `sew_14gb_autoslots` |
| Model | Qwen3-30B-A3B |
| Hardware | 1x Ascend 910B-class NPU, `ASCEND_RT_VISIBLE_DEVICES=4` |
| Dataset | ShareGPT, `mixed_chat` bucket |
| Requests | 200 real ShareGPT prompts |
| Prompt length | min 128, p50 541, p90 1936, p99 3576, max 3803 tokens |
| Output length | max 128 tokens |
| Concurrency | 1 closed-loop client concurrency |
| Serving shape | `max_model_len=4096`, `max_num_seqs=1`, `max_num_batched_tokens=4096` |
| Offload setting | `VLLM_ASCEND_MOE_OFFLOAD_GB=14`, SEW dataplane on, AutoConfig 32 slots |

## Main Result

| Metric | Value |
|---|---:|
| Successful requests | 200 / 200 |
| Failed requests | 0 |
| TTFT p50 / p90 / p99 | 1373.6 / 2041.1 / 2672.8 ms |
| TPOT p50 / p90 / p99 | 83.57 / 90.10 / 96.04 ms/token |
| Output throughput | 10.47 tokens/s |
| Mean end-to-end latency | 12.22 s/request |
| Total output tokens | 25,600 |

## System Evidence

| Evidence | Value |
|---|---:|
| Graph capture completed | true |
| `vllm::moe_offload_stage` seen | true |
| NPU model weights | 46.7751 GB |
| Available KV cache | 1.81 GiB |
| KV cache tokens | 19,712 |
| Registered offloaded MoE layers | 12 |
| Host expert store | 13.50 GiB |
| Slot bank | 3.375 GiB |
| Number of slots | 32 |
| Max active experts | 126 |
| Max wave count | 4 |
| Profile records | 307,236 |

## Artifacts

- Run directory: `benchmark/artifacts/runs/sharegpt_mixed_main/sew-offload-ascend-v1-20260706T080714Z`
- Benchmark JSON: `benchmark/artifacts/runs/sharegpt_mixed_main/sew-offload-ascend-v1-20260706T080714Z/sew_14gb_autoslots/mixed_chat/benchmark.json`
- Unit summary: `benchmark/artifacts/runs/sharegpt_mixed_main/sew-offload-ascend-v1-20260706T080714Z/sew_14gb_autoslots/mixed_chat/summary.md`
- Evidence summary: `benchmark/artifacts/reports/sharegpt_mixed_main_20260706/sharegpt_mixed_main_summary.json`
- Updated PPT: `docs/PPT/2026-7-8-SEW-Offload论文汇报.pptx`

## Presentation Note

Use this result as the PPT's current main evidence, not as the final paper
number. The next evidence step should run at least three repetitions and add
baseline comparison against eager native/legacy offload and SEW capture-disabled
ablation under the same ShareGPT bucket.
