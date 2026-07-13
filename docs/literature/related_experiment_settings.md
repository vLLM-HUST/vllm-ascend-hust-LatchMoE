# Related Experiment Settings for SEW-Offload

SEW-Offload 的实验不能按“云端高并发 LLM serving”论文照抄。最接近我们的 MoE offloading 文献通常把主场景设为单机、低并发或 offline batch inference；vLLM、Sarathi-Serve、Splitwise 这类 serving 论文则强调 request-rate sweep、真实长度分布和 tail latency。对我们更合理的路线是：主实验采用 MoE offloading 论文常见的 single-request / low-concurrency 设置，补一个 vLLM-style realistic serving trace 小节，而不是一开始就和大型云端 continuous batching 系统拼 QPS。

## Comparison Table

| Paper / system | Workload role | Input length | Output length | Batch / concurrency / arrival | Dataset or trace | Metrics | Takeaway for SEW-Offload |
|---|---|---:|---:|---|---|---|---|
| [vLLM / PagedAttention](https://arxiv.org/pdf/2309.06180) | Online serving throughput-latency frontier | Uses tokenized real distributions. ShareGPT mean input 161.31; Alpaca mean input 19.31 | ShareGPT mean output 337.99; Alpaca mean output 58.45 | Poisson arrivals with request-rate sweeps; 1-hour traces, 15-min for OPT-175B; figures sweep roughly ShareGPT 0-2.5 req/s and Alpaca up to 30 req/s depending model | ShareGPT and Alpaca | Normalized latency = request e2e latency / output length; sustainable request rate; average batched requests | Use this only as a secondary “serving realism” pattern. It is a dense/KV-cache serving paper, not a memory-constrained MoE-offload paper. |
| [FlexGen](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf) | Throughput-oriented offline generation under tight GPU memory | 512 | 32 | Effective batch size tuned large; OPT-175B reaches effective batch 144/256 depending experiment; Petals comparison uses request batch 2 and 6 parallel client processes | Synthetic fixed-length generation; also HELM sub-scenarios, Lambada, WikiText, data wrangling | Generation throughput token/s; accuracy/perplexity for approximate methods | Good precedent for fixed length 512/32 and high-throughput/offload framing, but its latency-insensitive batch-processing goal is different from our graph-replay hot path. |
| [MoE-Infinity](https://arxiv.org/html/2401.14361v3) | Personal-machine MoE offloading | 512 in end-to-end experiments | 32 in end-to-end experiments | Explicitly assumes batch size 1 for personal-machine MoE serving | MMLU, BIGBench, FLAN-like task mix; LongBench for long-context study | TPOT / decoding latency; cache capacity; robustness under workload changes | This is the closest workload philosophy for our main experiment: batch=1 is defensible for single-card memory-constrained MoE. Add long-context / prefill stress separately. |
| [FineMoE / fMoE](https://arxiv.org/html/2502.05370v1) | MoE serving with expert offload and prompt heterogeneity | Real prompt lengths from LMSYS-Chat-1M / ShareGPT; paper does not expose a fixed default length in the visible text | Real output/generation behavior; not summarized as one fixed value in the visible text | Main setting is serving; sensitivity increases inference batch size from 1 to 4 | LMSYS-Chat-1M, ShareGPT; Mixtral-8x7B, Qwen1.5-MoE, Phi-3.5-MoE | TTFT, TPOT, request latency CDF, expert hit rate, cache-limit sweep, ablation | Strong precedent for our missing pieces: report TTFT and TPOT separately, include expert hit/miss rate, and sweep expert cache/slot memory budgets. |
| [HOBBIT](https://arxiv.org/html/2411.01433v2) | Edge-device MoE offloading with mixed precision | Alpaca samples: half length 16, half length 128 | Four groups: 32 or 128 | The paper reports generation speed groups rather than online request-rate serving; batch/concurrency not emphasized | 60 high-quality Alpaca samples for speed; GSM8K and TruthfulQA for accuracy | Prefill latency, decoding speed token/s, accuracy | Good precedent for small controlled length matrix: [16,32], [16,128], [128,32], [128,128]. For SEW, replace with larger Ascend-relevant lengths. |
| [Fiddler](https://arxiv.org/html/2402.07033v3) | Local MoE inference across small/long/beam workloads | Single-batch setting uses 32/64/128/256; long-prefill setting uses 512/1024/2048/4096; beam search uses input 32 | Single-batch setting uses 64/128/256/512; beam search uses output 64 | Evaluates single batch, long prefill, and beam search width 4/8/12/16 | Synthetic length sweeps; ShareGPT appears as reference dataset | End-to-end latency / speedup over baselines | Very useful for our experiment axes: separate decode-heavy small prompts from prefill-heavy 512-4096 prompts. |
| [MoE-Lightning](https://pschafhalter.com/papers/2025-asplos-moe-lightning.pdf) | Offline high-throughput MoE batch inference | Uses MTBench settings S1/S2/S6/S8/S9; one policy case explicitly uses prompt 512 | Sweeps generation length 32/64/128/256; one ablation uses generation length 128 | Large micro-batches and optimized policies; table reports micro-batch/policy parameters rather than interactive concurrency | MTBench; Mixtral 8x7B, Mixtral 8x22B, DBRX | Generation throughput token/s; policy/CPU memory trade-off; ablations | Do not copy as our main story unless we pivot to offline throughput. Still useful for generation-length sweep 32/64/128/256 and memory-policy ablations. |
| [MoE-Gen](https://arxiv.org/html/2503.09716v1) | Offline high-throughput MoE inference with module-based batching | Example table uses context length 768 = prompt 512 + decode 256 | 256 in the headline example | Optimizes very large module batches; also studies batch size 1 and 32 | Offline inference tasks; DeepSeek and Mixtral | Throughput token/s, module utilization, module batch size | Mostly orthogonal to us. It confirms that batch=1 and batch=32 are different regimes, so SEW should not mix them in one headline claim. |
| [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf) | Online LLM serving with chunked prefill and stall-free batching | Chunking overhead study uses prefill lengths 2K/4K/8K; chunk lengths 512/1024/2048 | Dataset-driven serving output; not a single fixed OSL in the quoted setup | Evaluates 128 requests for ablation; token budget 1024 in Table 4; online capacity under SLO elsewhere | openchat_sharegpt4 and arxiv_summarization | P50 TTFT, P99 TBT, SLO capacity | Strong precedent for our B2 prefill: use chunk/wave size sensitivity and report TTFT plus per-token tail latency. |
| [Splitwise](https://arxiv.org/html/2311.18677v2) | Cluster-level phase splitting | Uses production prompt/token size distributions; profiles multiple input/output sizes | Production distributions | Poisson arrival rate tuned for load; simulator reports per-request TTFT/TBT/E2E and utilization | Production traces from its characterization section | TTFT, TBT, E2E, utilization | Useful as rationale for phase-aware metrics, but not a direct baseline for single-card Ascend expert offload. |

## Patterns Across Papers

MoE offloading papers are much more forgiving about low concurrency than general serving papers. MoE-Infinity explicitly builds its premise around batch size one on personal machines, and Fiddler reports single-batch, long-prefill, and beam-search workloads separately. FineMoE does include batch-size sensitivity, but only from 1 to 4 in the visible paper text. This means our current `max_num_seqs=1` setting is not embarrassing if the paper is framed as single-card memory-constrained MoE serving, but it must be defended as a scope choice and complemented with a small concurrency robustness section.

The common fixed-length anchors are 512/32 and 512/256. FlexGen and MoE-Infinity both use prompt length 512 and generation length 32 for core offload experiments. MoE-Gen’s headline example uses context length 768, decomposed as prompt 512 plus decode 256. MoE-Lightning sweeps generation lengths 32, 64, 128, and 256. Fiddler extends the prefill side to 512, 1024, 2048, and 4096. For SEW-Offload, this suggests one realistic fixed matrix rather than one smoke prompt: `(128,128)`, `(512,128)`, `(2048,32)`, and `(4096,16 or 32)`.

Serving papers use real length distributions and request-rate sweeps, but those are usually not the first experiment in memory-constrained offload papers. vLLM tokenizes ShareGPT and Alpaca, then generates Poisson arrivals across request rates and reports normalized latency. Sarathi-Serve evaluates openchat_sharegpt4 and arxiv_summarization, emphasizing P50 TTFT and P99 TBT under token budgets. We can borrow this as a secondary section after the fixed-length evidence is stable.

Most strong papers report an explanatory metric, not only TTFT/TPOT. FineMoE reports expert hit rate and cache-limit sensitivity. MoE-Infinity reports cache capacity and adaptation to workload changes. Sarathi-Serve reports chunking overhead. For SEW-Offload, the matching explanatory metrics are slot hit/miss rate, active expert count, wave count, H2D bytes, staging time, graph-capture status, and KV/slot memory partition.

## Recommended SEW-Offload Matrix

### Main Fixed-Length Benchmarks

Use Qwen3-30B-A3B on one Ascend 910B-class NPU. Keep `max_num_seqs=1` as the main paper setting and state that this follows personal/local MoE-offload papers rather than cloud serving papers.

| Workload | Prompt tokens | Output tokens | Why |
|---|---:|---:|---|
| Decode-heavy | 128 | 128 or 256 | Tests replay-friendly decode hot path and TPOT. |
| Balanced | 512 | 128 | Matches the 512-token anchor used by FlexGen/MoE-Infinity while generating enough tokens for stable TPOT. |
| Prefill-heavy | 2048 | 32 | Stresses B2 wave prefill and TTFT without requiring extremely long generation. |
| Long-prefill stress | 4096 | 16 or 32 | Matches Fiddler/Sarathi-style long-prefill stress and should expose active expert overflow. |

Run at least 3 repetitions per case. Report median/p90/p99 TTFT and TPOT, output throughput, success/failure, and peak HBM. Do not mix the 1-request smoke numbers with this table.

### Systems and Baselines

| Case | Mode | Claim |
|---|---|---|
| No offload | capacity probe | Shows full-resident path fails under tight HBM budget. |
| Native prefetch | eager and capture-attempt | Eager gives a performance baseline; capture-attempt gives structural failure evidence. |
| Legacy/layered offload | eager and capture-attempt | Shows old plugin path is not graph-compatible. |
| SEW-Offload capture disabled | eager ablation | Isolates benefit of ACLGraph replay from the same slot runtime. |
| SEW-Offload capture on | full system | Main result. |

### Memory and Slot Sensitivity

Sweep offload budget and slot count separately. The Week-2 smoke already showed that `28GiB autoslots` can fail because 64 slots consume too much HBM, while `28GiB slots32` succeeds. Turn that into a paper result:

| Budget | Slot counts |
|---|---|
| 14 GiB | 8, 16, 32 |
| 28 GiB | 16, 32, 64 |

For each point, report peak HBM, available KV cache, slot-bank memory, slot hit/miss rate, H2D bytes, TTFT, and TPOT.

### B2 Wave Prefill

Use prompt lengths 2048 and 4096, output length 16 or 32. Compare:

| Variant | Expected evidence |
|---|---|
| No B2 wave | Fails or requires all active experts resident when active set exceeds slots. |
| B2 wave, transfer-aware off | Correct but slower or more staging overhead. |
| B2 wave, transfer-aware on | Correct and lower stage time / TTFT. |

Report active experts per layer, slot budget, wave count, H2D bytes, stage time, TTFT. This mirrors Sarathi’s chunk-size overhead logic but for expert waves rather than attention prefill chunks.

### Realistic Serving Add-On

After fixed-length experiments work, add one ShareGPT-derived serving workload:

| Workload | Source | Arrival / concurrency |
|---|---|---|
| Mixed-chat trace | ShareGPT tokenized prompts, filtered to fit `max_model_len=4096` | Either Poisson request rates low enough for single-card Ascend, or closed-loop concurrency 1/2/4. |

Report P50/P90/P99 TTFT, TPOT, request latency, success rate, graph-capture status, and slot hit/miss. Do not make this the headline until the system can survive repeated runs.

## Practical Change to the PPT Story

The current PPT should not apologize that the experiments are weak. It should say the current experiments are smoke evidence and then show this literature-backed plan. A better experiment slide title is:

> Current evidence is smoke-level; the paper result will follow the MoE-offload convention: fixed-length memory-constrained benchmarks first, realistic serving trace second.

The teacher-facing message is: our experiments are not merely “bad”; they are currently under-specified. Related papers succeed because they separate regimes: batch=1 local/offload, large-batch offline throughput, and online request-rate serving. SEW-Offload should do the same.

## Source Notes

- vLLM / PagedAttention: ShareGPT and Alpaca length distributions, Poisson arrivals, normalized latency, and request-rate sweeps are described in the evaluation section of the paper.
- FlexGen: fixed prompt length 512 and output length 32 are used in its core offloading comparisons; the paper also evaluates HELM, Lambada, WikiText, and data-wrangling tasks.
- MoE-Infinity: end-to-end evaluation uses prompt length 512 and decode length 32, and explicitly motivates batch size one for personal-machine MoE serving.
- FineMoE / fMoE: visible evaluation text emphasizes LMSYS-Chat-1M / ShareGPT, TTFT/TPOT, cache-limit sensitivity, expert hit rate, and batch-size sensitivity from 1 to 4.
- HOBBIT: speed evaluation uses Alpaca samples with input lengths 16 and 128 and four input/output groups: [16,32], [16,128], [128,32], [128,128].
- Fiddler: reports input length sweeps 32/64/128/256 with output 64/128/256/512, plus long-prefill input lengths 512/1024/2048/4096.
- Sarathi-Serve: prefill chunking experiments use prefill lengths 2K/4K/8K and chunk lengths 512/1024/2048; ablation table reports 128 requests with token budget 1024.
