# Compatibility Matrix

This document defines the supported boundary for the annual demo and the paper
evaluation. Anything outside the supported set is a research backlog item, not a
demo promise.

## Supported Demo Boundary

| Dimension | Supported |
|---|---|
| Hardware | Single-card Ascend 910B-class NPU |
| Model | Qwen3-30B-A3B, unquantized MoE |
| Serving API | vLLM OpenAI-compatible server |
| Concurrency | Low-concurrency demo, `max_num_seqs=1` by default |
| Main path | SEW fixed-slot dataplane with AutoConfig |
| Offload budgets | `14` GiB and `28` GiB presets |
| Expected slots | KV-aware AutoConfig; 14 GiB is typically 32 slots, while 28 GiB is capped by KV reserve and may be below 64 slots |
| Prefill overflow | B2 wave prefill only when phase is prefill or explicit `max_num_seqs` hint proves prompt-shaped overflow |
| Loading | CPU-first loading for offloaded unquantized MoE experts |
| Baselines | no-offload capacity probe and native prefetch baseline |

## Not Supported Yet

| Dimension | Status |
|---|---|
| Multi-card serving | Not supported for demo claims |
| Quantized MoE | Not supported by CPU-first loading path yet |
| Arbitrary MoE architectures | Not claimed; Qwen3-30B-A3B is the demo target |
| High-concurrency decode | Not claimed; B2 overflow handoff is intentionally gated |
| Upstream vLLM-Ascend without hooks | Not plug-and-play; vllm-ascend-hust hook seam is required |
| Mixing native prefetch with SEW dataplane | Unsupported except as a separate baseline process |

## Recommended Preset

Use the annual demo runner presets:

```bash
python tools/run_annual_demo_suite.py \
  --config demo/annual_demo_config.json \
  --case sew_14gb_slots32
```

The preset enables the SEW dataplane, B2 prefill, transfer-aware wave scheduling,
CPU-first loading, and `VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT=1`.

## Baseline Separation

Use separate cases and separate server processes for:

- `native_prefetch_baseline`: vLLM native prefetch offload.
- `sew_14gb_slots32` / `sew_28gb_slots32`: SEW fixed-slot dataplane.
- `sew_28gb_slots64`: high-slot diagnostic sensitivity case; it may fail KV capacity.

Do not set `--offload-backend prefetch`, `--cpu-offload-gb`, or native offload
grouping flags when running a SEW case.
