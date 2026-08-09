# Issue #4 Reproduction

LatchMoE keeps its runtime implementation in this repository. The Ascend fused-MoE hook is a separately checked out dependency, pinned by `seam.lock`; this repository does not contain a Git submodule and must not replace the runtime package with the hook repository. Issue #4 uses the public `feature/latchmoe-offload-seam-v1-v021` seam because the evidence stack is pinned to vLLM 0.21.0 and CANN 9.0.0. The similarly named current-main branch is a separate compatibility line and is not interchangeable with this lock.

Run the graph smoke from a clean LatchMoE checkout with the matching clean seam checkout:

```bash
benchmark/scripts/run_issue4_graph_repro.sh \
  --seam-root /path/to/vllm-ascend-hust \
  --model /path/to/Qwen3-30B-A3B \
  --device 0
```

The run directory contains the wrapper command, exact smoke subcommand, environment, both commit SHAs, NPU inventory, raw console output, generated tokens, H2D profile events, and `graph_verification.json`. The verifier fails if capture/replay evidence is absent or if it observes a seam guard failure, missing capture weights, or an uppercase fallback/bypass/full-weight/native path marker.
