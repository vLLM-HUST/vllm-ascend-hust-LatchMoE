# vllm-moe-offload-ascend

**LatchMoE: Address-stable, Graph-compatible expert offloading framework for MoE inference**

> 📄 **Paper**: [USENIX-style draft](paper/main.pdf) and
> [claim ledger](paper/CLAIMS.md). The current performance table remains
> preliminary until its raw run manifests and repetitions are checked in.

- 负责人：李昶吾（`@Li-changwu`）、陈德斌（`@pluviophile-chen`）
- 当前工作：[运行时边界与 graph 路径证据 #4](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/issues/4)

---

## Project Boundary

LatchMoE is the runtime-mechanism project. It owns the address-stable slot
bank, Host Store and H2D path, graph-replay lifecycle, wave prefill, and a
generic placement-plan adapter. Policy research such as Static, frequency,
route-transition, EPLB matched comparisons, and experiment orchestration is
owned by the separate
[`intellistream/ascend-moe-control`](https://github.com/intellistream/ascend-moe-control)
project. New control policies must not be added to LatchMoE's main branch.

Graph-path failures must be diagnosed directly. Forced eager execution is not
an acceptable substitute for graph validation.

### Prefix-cache boundary

LatchMoE currently does **not** support vLLM prefix-cache reuse. The supported
path always recomputes the complete prompt. Prefix-cache hits change the
prefill token count and KV/block metadata, which is a separate vLLM-Ascend
compatibility path and is not part of LatchMoE's correctness or performance
claims. The `latchmoe` launcher therefore adds `--no-enable-prefix-caching`
automatically and rejects an explicit `--enable-prefix-caching` request.

## What LatchMoE Does

Serving a MoE model whose expert weights exceed single-NPU HBM requires
**expert offloading**: keep most experts in host memory and load the routed
ones on demand. On Ascend, however, efficient decoding depends on **ACLGraph
replay**, and replay assumes stable tensor addresses and a fixed task
topology; per-step changes must happen outside the captured region. Naive
offloading copies weights and synchronizes inside the replay region, which
either breaks graph capture or forces a fallback to eager execution that is
heavily host-bound on Ascend. Existing MoE offloading systems simply give up
graph replay and run eager.

**LatchMoE** makes offloading and graph replay compatible:

1. **Slot-stable expert virtualization** — a pre-allocated NPU slot bank plus
   a pinned host expert store. Captured GroupedMatmul kernels always read
   from the same slot addresses; routing changes only rewrite slot *indices*,
   never tensor addresses.
2. **Replay-boundary staging** — a staging controller executes all
   routing-driven host-to-device expert transfers outside the captured
   region, on a dedicated transfer stream.
3. **Correctness-first overflow staging** — when a prompt-shaped working set
   exceeds slot capacity, the supported path stages the full expert layer and
   executes one native MoE pass. Capacity-bounded multi-wave prefill remains
   experimental because it changes BF16 evaluation order.
4. **Compute-protected slot lifecycle** — event ordering between the compute
   and transfer streams guarantees that asynchronous loads never overwrite a
   slot still referenced by in-flight compute.
5. **KV-aware AutoConfig** — a single environment variable
   (`VLLM_ASCEND_MOE_OFFLOAD_GB`) derives resident layers and slot capacity
   at startup, bounded by physical HBM, KV-cache reserve, and a minimum
   net-saving constraint.

The integration is **non-invasive**: the plugin registers through vLLM's
`vllm.platform_plugins` entry point and replaces null-stub hook points in the
hook-enabled vllm-ascend fork. The target model, OpenAI-compatible API, and
scheduler semantics are untouched; uninstalling the plugin restores stock
behavior.

---

## Key Results

Engineering measurements with Qwen3-30B-A3B (BF16, unquantized MoE), a single
Ascend 910B-class NPU (64 GB), TP1, `max_num_seqs=1`, 200 real ShareGPT
mixed-chat requests, and a 14 GiB offload budget. Both sides run the same
LatchMoE runtime and offload configuration; only ACLGraph capture is toggled.

| Metric | Eager offload (capture off) | LatchMoE (capture on) | Improvement |
|---|---:|---:|---:|
| TTFT p50 (ms) | 2283.2 | 1373.6 | **1.66×** |
| TPOT p50 (ms/token) | 192.5 | 83.6 | **2.30×** |
| Output throughput (tok/s) | 4.71 | 10.47 | **2.22×** |

Context under the same memory budget:

- Serving without offloading fails KV-cache allocation outright: the model
  weights alone exceed the HBM budget.
- Naive per-step on-demand staging in eager mode does produce tokens, but at
  roughly 3 s/token TPOT — about 36× slower than LatchMoE — because every
  decode step re-transfers the offloaded experts.
- Slot hit rate warms from ~50% to 75–88% over the run under the deadline
  staging policy.

---

## Architecture

LatchMoE is organized into four layers:

```text
┌────────────────────────────────────────┐
│  vLLM / vllm-ascend Hook Seam          │  ← platform plugin, null-stub patching
├────────────────────────────────────────┤
│  Compute-Protected Slot Lifecycle      │  ← transfer stream + event ordering
├────────────────────────────────────────┤
│  Correctness-First Overflow Staging    │  ← full-layer native MoE pass
├────────────────────────────────────────┤
│  Slot-Stable Expert Virtualization     │  ← fixed slot bank + pinned host store
└────────────────────────────────────────┘
```

Per step, the data path is: router output → staging controller (outside the
replay boundary) → fixed NPU expert slots → captured MLP replay.

## Repository Layout

```text
vllm_moe_offload_ascend/
  moe_offload/      runtime core: slot bank, host store, transfer engine,
                    phase split, prefill residency, autoconfig, profiling
  ops/fused_moe/    router / staging / MLP ops and the graph seam injection
  patches/          monkey-patches replacing vllm-ascend null-stub hooks
benchmark/          config-driven serving benchmark harness (see benchmark/README.md)
tests/              host-side unit tests
```

---

## Requirements

- Ascend 910B-class NPU with a working CANN / torch-npu environment
- vLLM 0.21.0 and the **hook-enabled vllm-ascend fork**
  ([`vLLM-HUST/vllm-ascend-hust`, branch
  `feature/latchmoe-offload-seam-v1-v021`](https://github.com/vLLM-HUST/vllm-ascend-hust/tree/feature/latchmoe-offload-seam-v1-v021),
  commit `fffbd1eb75db455e4c90dfb2b8455d0e66ff5b25`);
  the Issue #4 dependency contract is pinned in
  [`repro/issue4/seam.lock`](repro/issue4/seam.lock), and stock vllm-ascend does
  not contain the MoE offload hook seam
- Python ≥ 3.10
- Validated configuration: Qwen3-30B-A3B (unquantized MoE), BF16, TP1,
  single NPU, low-concurrency serving (`max_num_seqs=1`)
- Do **not** combine with vLLM's native weight-offload flags
  (`--cpu-offload-gb`, `--offload-backend prefetch`, `--offload-group-size`);
  the plugin manages expert offload through its own dataplane

---

## Installation

Install vLLM 0.21.0 and the hook-enabled vllm-ascend fork into one Python
environment first. For the Issue #4 reproduction stack, check out the exact
seam commit recorded in `repro/issue4/seam.lock`, then install LatchMoE into
the same environment:

```bash
git clone --branch feature/latchmoe-offload-seam-v1-v021 \
  https://github.com/vLLM-HUST/vllm-ascend-hust.git
git -C vllm-ascend-hust checkout fffbd1eb75db455e4c90dfb2b8455d0e66ff5b25

git clone https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE.git
cd vllm-ascend-hust-LatchMoE
python -m pip install -e . --no-deps --no-build-isolation
```

The plugin auto-registers through the `vllm.platform_plugins` entry point as
`moe_offload_ascend`. If your deployment filters plugins with `VLLM_PLUGINS`,
include `moe_offload_ascend` in the list. To verify the installation:

```bash
python -m vllm_moe_offload_ascend check
```

The check is deliberately run by the same Python interpreter that will launch
vLLM. It verifies the `vllm`, `vllm_ascend`, and plugin module locations, both
required platform entry points, and any `VLLM_PLUGINS` filter. The editable
install also provides the equivalent `latchmoe check` command.

See the [Chinese installation and launch guide](docs/quickstart_zh.md) for the
complete source-install, existing-stack, overlay, and troubleshooting flows.

---

## Quick Start

With sibling `vllm-hust-dev-hub`, the repository manifest handles installation,
entry-point validation, safe defaults, and environment injection:

```bash
cd ../vllm-hust-dev-hub
./manage.sh restart --optimization latchmoe --offload-gb 14
```

The equivalent low-level invocation is documented below for standalone use.

```bash
# Offload budget in GiB; setting this enables the plugin via AutoConfig
export VLLM_ASCEND_MOE_OFFLOAD_GB=14

# Graph-compatible fixed-slot dataplane
export VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE=1

# Serving-shape hint for low-concurrency serving
export VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT=1

# Canonical: the active Python owns both plugin discovery and vLLM startup.
python -m vllm_moe_offload_ascend serve \
  /path/to/Qwen3-30B-A3B --trust-remote-code \
  --no-enable-prefix-caching

# Equivalent console script created by pip install.
latchmoe serve /path/to/Qwen3-30B-A3B --trust-remote-code \
  --no-enable-prefix-caching
```

SEW AutoConfig releases the original NPU copies of offloaded expert weights
after the host store and fixed-slot banks are ready. Set
`VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS=0` only for retained-
weight diagnostics; that mode exercises staging but does not reclaim the
target expert-weight HBM.

When the SEW dataplane is enabled, Graph serving uses pure `PIECEWISE`
ACLGraph. LatchMoE forces this mode because routing-dependent expert staging
must execute eagerly between captured pieces on every decode step. `FULL`,
`FULL_DECODE_ONLY`, and `FULL_AND_PIECEWISE` are unsafe for expert offload: a
FULL decode graph would capture the staging seam as a no-op and replay stale
slot mappings. `--enforce-eager` remains supported and keeps graph mode at
`NONE`.

A successful startup logs the following signals:

```text
moe_offload_ascend -> vllm_moe_offload_ascend:register
Enabled Ascend MoE offload autoconfig from VLLM_ASCEND_MOE_OFFLOAD_GB
LATCHMOE_GRAPH_CONFIG cudagraph_mode=PIECEWISE splitting_op=vllm::moe_offload_stage status=enabled
```

To disable, unset `VLLM_ASCEND_MOE_OFFLOAD_GB` or uninstall the plugin;
vllm-ascend then falls back to its null-stub hooks.

---

## Configuration

All configuration is environment-variable based. The main knobs:

| Environment Variable | Default | Description |
|---|---|---|
| `VLLM_ASCEND_MOE_OFFLOAD_GB` | unset (disabled) | Target offload size (GiB); setting it enables AutoConfig |
| `VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE` | `0` | Enable the graph-compatible fixed-slot dataplane |
| `VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT` | `0` | Serving-shape hint for the prefill-overflow handoff; set `1` for low-concurrency serving |
| `VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS` | auto | Expert override for the derived slot count |
| `VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS` | auto | Comma-separated layer IDs kept fully resident |
| `VLLM_ASCEND_MOE_OFFLOAD_POLICY` | `deadline` | Staging policy (`deadline` / `lru`) |
| `VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD` | `1` on SEW path | Load experts on a dedicated transfer stream |
| `VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE` | `full_layer` | Correctness-first behavior when active experts exceed slot capacity; `experimental_wave` enables the unqualified multi-wave path |
| `VLLM_ASCEND_MOE_OFFLOAD_TRANSFER_AWARE_SCHEDULE` | `1` on SEW path | Reorder wave staging/compute by per-wave H2D bytes |
| `VLLM_ASCEND_MOE_OFFLOAD_PREFILL_PREFETCH_DEPTH` | `1` | Software-pipeline prefetch depth for wave prefill |
| `VLLM_ASCEND_MOE_OFFLOAD_PREFILL_BUFFER_COUNT` | `2` | Stage buffer count for wave prefill |
| `VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_FRACTION` | `0.12` | Max fraction of physical HBM usable by the slot bank |
| `VLLM_ASCEND_MOE_OFFLOAD_KV_RESERVE_SEQS` | `4` | Sequences of KV cache reserved by AutoConfig |
| `VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD` | `0` | Materialize offloaded experts directly in host memory at load time (avoids startup HBM peak) |
| `VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS` | `1` with SEW | Release full NPU copies after fixed-slot initialization; set `0` only for retained-weight diagnostics |
| `VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY` | `0` | Collect routing traces without actual offloading |
| `VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH` | unset | Offload/stage profile JSONL output path |

Additional expert-level overrides are documented in
`vllm_moe_offload_ascend/moe_offload/config.py` and `autoconfig.py`.
The `experimental_wave` overflow mode changes BF16 evaluation order and is not
currently token-equivalent to native single-pass MoE. Do not use it for
correctness or performance claims.

---

## Benchmark

The repository ships a config-driven serving benchmark harness under
`benchmark/` (canonical config, ShareGPT workload manifests, `case ×
workload` runner, and artifact contract):

```bash
python benchmark/scripts/sew_bench.py validate
python benchmark/scripts/sew_bench.py prepare-workloads --bucket smoke --requests-per-bucket 1
python benchmark/scripts/run_suite.py --case sew_14gb_autoslots --workload smoke
```

See [benchmark/README.md](benchmark/README.md) for the full workflow.

## Testing

Host-side unit tests (no NPU required):

```bash
python3 -m pytest tests/ -q
```

## License

Apache-2.0
