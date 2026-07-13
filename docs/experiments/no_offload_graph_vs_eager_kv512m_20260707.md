# Qwen3-30B-A3B Full-Resident ACLGraph vs Eager on ShareGPT

Date: 2026-07-07

## Conclusion for PPT

When Qwen3-30B-A3B is fully resident on one 910B-class NPU and KV capacity is manually limited to 512 MiB, ACLGraph is substantially faster than Eager under the same ShareGPT mixed-chat workload:

| Mode | Success | TTFT p50 (ms) | TPOT p50 (ms/token) | TPOT p90 (ms/token) | Output throughput (tok/s) | Wall time (s) |
|---|---:|---:|---:|---:|---:|---:|
| ACLGraph | 200/200 | 200.22 | 33.23 | 34.87 | 28.45 | 899.85 |
| Eager | 200/200 | 322.19 | 147.35 | 150.41 | 6.64 | 3855.30 |
| Gain | - | 37.9% lower | **4.43x lower** | 4.31x lower | **4.28x higher** | 4.28x faster |

This is the cleanest measurement of graph execution efficiency because there is no expert offload in either run: all model weights stay on the NPU, and both runs use the same explicit KV capacity.

## Experimental Setup

| Item | Setting |
|---|---|
| Model | Qwen3-30B-A3B |
| Model path | `/root/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B` |
| Hardware | Single Huawei 910B-class NPU, `ASCEND_RT_VISIBLE_DEVICES=4` |
| Precision | bfloat16 |
| Tensor parallel | 1 |
| Dataset | ShareGPT V3, `mixed_chat` bucket |
| Manifest | `benchmark/artifacts/workloads/sharegpt_qwen3_30b_a3b_v1.jsonl` |
| Requests | 200 |
| Concurrency | 1 |
| Output length | 128 tokens |
| Prompt tokens | min 128, p50 547, p90 2072, p99 3582, max 3803 |
| `max_model_len` | 4096 |
| `max_num_seqs` | 1 |
| `max_num_batched_tokens` | 4096 |
| KV capacity | `--kv-cache-memory-bytes 536870912` (512 MiB) |
| Effective KV cache | 5,376 tokens; maximum concurrency for 4,096-token requests: 1.31x |
| Offload | Disabled / env cleared |

## Cases

| Case | Extra server args | Purpose |
|---|---|---|
| `no_offload_kv512m_aclgraph` | `--kv-cache-memory-bytes 536870912` | Full-resident ACLGraph baseline |
| `no_offload_kv512m_eager` | `--enforce-eager --kv-cache-memory-bytes 536870912` | Full-resident Eager baseline |

## Command

```bash
ASCEND_RT_VISIBLE_DEVICES=4 /root/miniconda3/bin/conda run -n vllm-hust-dev \
  python benchmark/scripts/run_suite.py \
  --case no_offload_kv512m_aclgraph \
  --case no_offload_kv512m_eager \
  --workload mixed_chat \
  --python /root/miniconda3/envs/vllm-hust-dev/bin/python \
  --output-root benchmark/artifacts/runs/no_offload_graph_vs_eager_kv512m
```

## Artifact Paths

Run root:

`benchmark/artifacts/runs/no_offload_graph_vs_eager_kv512m/sew-offload-ascend-v1-20260707T135837Z`

ACLGraph:

- `no_offload_kv512m_aclgraph/mixed_chat/benchmark.json`
- `no_offload_kv512m_aclgraph/mixed_chat/server.log`
- `no_offload_kv512m_aclgraph/mixed_chat/client.log`

Eager:

- `no_offload_kv512m_eager/mixed_chat/benchmark.json`
- `no_offload_kv512m_eager/mixed_chat/server.log`
- `no_offload_kv512m_eager/mixed_chat/client.log`

## Log Evidence

ACLGraph run:

- Server args include `kv_cache_memory_bytes: 536870912`.
- Weight load: `Loading model weights took 56.9001 GB`.
- KV cache: `GPU KV cache size: 5,376 tokens`.
- Graph capture: `Graph capturing finished in 2 secs, took 0.03 GiB`.
- Runtime replay: `Replaying aclgraph`.

Eager run:

- Server args include `enforce_eager: True` and `kv_cache_memory_bytes: 536870912`.
- vLLM logs: `Cudagraph is disabled under eager mode`.
- vLLM logs: `Compilation disabled, using eager mode by default`.
- Weight load: `Loading model weights took 56.9001 GB`.
- KV cache: `GPU KV cache size: 5,376 tokens`.

## Suggested PPT Text

**Full-resident graph efficiency.** To isolate graph execution from offload effects, we run Qwen3-30B-A3B fully resident on a single NPU and manually cap KV cache to 512 MiB. Under identical ShareGPT mixed-chat requests, ACLGraph reduces median TPOT from 147.35 ms/token to 33.23 ms/token, a 4.43x latency improvement, and increases output throughput from 6.64 to 28.45 tok/s.

