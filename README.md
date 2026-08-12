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
`vllm.general_plugins` entry point and installs implementations only on the
explicit extension points exposed by the hook-enabled vllm-ascend fork. It
does not alias plugin modules into the `vllm_ascend.*` namespace. The target
model, OpenAI-compatible API, and scheduler semantics are untouched;
uninstalling the plugin restores stock behavior.

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
│  vLLM / vllm-ascend Hook Seam          │  ← general plugin, explicit seam adapter
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
  patches/          adapter installing implementations on explicit host hooks
benchmark/          config-driven serving benchmark harness (see benchmark/README.md)
tests/              host-side unit tests
```

---

## Requirements

- Ascend 910B-class NPU with a working CANN / torch-npu environment
- Python ≥ 3.10
- The exact host stack in the compatibility table below
- Validated configuration: Qwen3-30B-A3B (unquantized MoE), BF16, TP1,
  single NPU, low-concurrency serving (`max_num_seqs=1`)
- Do **not** combine with vLLM's native weight-offload flags
  (`--cpu-offload-gb`, `--offload-backend prefetch`, `--offload-group-size`);
  the plugin manages expert offload through its own dataplane

---

## 固定兼容版本

LatchMoE 当前只维护一条已验证兼容线。分支名不是版本锁，部署时必须同时
checkout 下表中的完整 commit：

| Component | Repository / branch | Locked commit | Version |
|---|---|---|---|
| vLLM | `vLLM-HUST/vllm-hust` | `ad7125a431e176d4161099480a66f0169609a690` | `0.21.0` |
| Ascend hook seam | `vLLM-HUST/vllm-ascend-hust`, `feature/latchmoe-offload-seam-v1-v021` | `4806367eeeb7d62b32078ae90cd929cc06d825fe` | seam ABI 1 |
| Torch / Torch-NPU | environment packages | — | `2.10.0` / `2.10.0.post2` |
| CANN | system runtime | — | `9.0.1` |

`feature/latchmoe-offload-seam-v1-v021` 是唯一的 LatchMoE seam 分支。
`feature/latchmoe-offload-seam-v1` 对应另一条未经本项目正确性门禁验证的宿主
主线，不应再用于安装。它承载的
[`vllm-ascend-hust#214`](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/214)
已经关闭，远端 `feature/latchmoe-offload-seam-v1` 也已删除。

完整机器可读锁位于
[`vllm_moe_offload_ascend/compatibility.lock`](vllm_moe_offload_ascend/compatibility.lock)。

## 安装

可行的部署方式是把两个宿主仓库固定到上述 commit，然后在同一个 Python
环境安装 LatchMoE。项目故意不把 vLLM 或 vLLM-Ascend 写入普通 PyPI
`dependencies`：在 Ascend 环境中让 pip 自动解析依赖，可能下载 CUDA/上游包并
覆盖已经匹配的 Torch-NPU 软件栈。

以下命令假设 CANN 9.0.1、Torch 2.10.0 和 Torch-NPU 2.10.0.post2 已由基础镜像或运维环境
提供：

推荐先 clone 本仓库，再用机器可读 lock 驱动安装。脚本会把两个宿主 checkout
到固定 commit、补充 vLLM `v0.21.0` tag、始终调用当前 `sys.executable -m pip`、
跳过与 LatchMoE seam 无关的自定义算子编译，并在最后执行环境检查：

```bash
git clone https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE.git
cd vllm-ascend-hust-LatchMoE

# 先检查将要执行的命令；不会 clone、checkout 或安装。
python tools/install_locked_stack.py \
  --workspace /path/to/latchmoe-stack --dry-run

# 安装到当前 python；目标目录必须不存在或是 clean Git checkout。
python tools/install_locked_stack.py \
  --workspace /path/to/latchmoe-stack
```

需要手工安装时，等价步骤如下：

```bash
# 1. 固定 vLLM-HUST
git clone https://github.com/vLLM-HUST/vllm-hust.git
git -C vllm-hust fetch origin ad7125a431e176d4161099480a66f0169609a690
git -C vllm-hust fetch https://github.com/vllm-project/vllm.git \
  refs/tags/v0.21.0:refs/tags/v0.21.0
git -C vllm-hust checkout ad7125a431e176d4161099480a66f0169609a690

# 2. 固定唯一的 vLLM-Ascend hook seam
git clone --branch feature/latchmoe-offload-seam-v1-v021 \
  https://github.com/vLLM-HUST/vllm-ascend-hust.git
git -C vllm-ascend-hust checkout 4806367eeeb7d62b32078ae90cd929cc06d825fe

# 3. 在同一解释器中安装两个宿主和 LatchMoE
VLLM_TARGET_DEVICE=empty python -m pip install \
  --no-deps --no-build-isolation -e ./vllm-hust
COMPILE_CUSTOM_KERNELS=0 python -m pip install \
  --no-deps --no-build-isolation -e ./vllm-ascend-hust

git clone https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE.git
python -m pip install --no-deps --no-build-isolation \
  -e ./vllm-ascend-hust-LatchMoE
```

额外 fetch `v0.21.0` tag 是必要的：HUST 仓库当前没有公开这一 tag，但固定的
commit 与上游 `v0.21.0` 完全相同；setuptools-scm 需要该 tag 才会生成正确的
`0.21.0` 包版本。

`VLLM_TARGET_DEVICE=empty` 只关闭 vLLM-HUST 自身的 CUDA 构建；运行时 NPU
platform 仍由 Ascend plugin 提供。`COMPILE_CUSTOM_KERNELS=0` 则避免安装 seam 时
编译整套与 Qwen3/LatchMoE 无关的 Ascend 自定义算子。需要这些额外算子的其他
模型应由基础 vLLM-Ascend 镜像单独提供，不能把它们的构建结果当作 LatchMoE
安装是否成功的门禁。

安装完成后无需复制源码、修改 `PYTHONPATH` 或手动调用 `register()`。vLLM 0.21.0
会在 API、EngineCore 和 Worker 进程自动发现 `vllm.general_plugins`，并调用
`vllm_moe_offload_ascend.register()`。该函数先把全部 LatchMoE 配置项注册到
`vllm.envs` 和 `vllm_ascend.envs`，再安装 seam adapter。

如果没有设置 `VLLM_PLUGINS`，插件会自动加载；如果部署环境使用白名单，必须
同时保留 Ascend platform plugin 和 LatchMoE general plugin：

```bash
export VLLM_PLUGINS=ascend,moe_offload_ascend
```

安装后核验：

```bash
python -m vllm_moe_offload_ascend check
```

`check` 会核验模块路径、vLLM platform/general 两组 entry point、固定版本与
commit、seam ABI、CANN/Torch 版本以及 `VLLM_PLUGINS` 白名单。可编辑安装还会
提供等价的 `latchmoe check` 命令。

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
export VLLM_WORKER_MULTIPROC_METHOD=spawn

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

`VLLM_ASCEND_MOE_OFFLOAD_GB=14` selects a partial-residency plan; it does not
mean that the whole model will fit in 14 GiB. For a constrained-card validation
that offloads every MoE layer, explicitly select an empty resident-layer set,
enable CPU-first loading, and choose a tested slot capacity:

```bash
export VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS=
export VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD=1
export VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=16
```

This override trades substantially more host-to-device traffic for a lower HBM
footprint. Keep AutoConfig's residency plan for normal serving unless the
capacity trade-off has been measured for the target workload.

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
