# vllm-moe-offload-ascend

**LatchMoE: Address-stable, Graph-compatible expert offloading framework for MoE inference**

> 📄 **Paper**: [USENIX-style draft](paper/main.pdf) and
> [claim ledger](paper/CLAIMS.md). Graph-path lifecycle evidence is checked in;
> broad performance claims still require the full matched evaluation matrix.

- 负责人：李昶吾（`@Li-changwu`）、陈德斌（`@pluviophile-chen`）
- 当前证据：[Issue #7 graph lifecycle bundle](docs/evidence/issue-7-graph-lifecycle.md)

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
3. **Capacity-bounded multi-wave prefill** — when a prompt-shaped working set
   exceeds slot capacity, the default path executes exact routed pairs in
   bounded waves and performs one native combine. Recoverable qualification
   failures fall back to the blocking `full_layer` path.
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

LatchMoE also publishes a static Extension Manager manifest under the
project-owned `vllm_hust.extension_bundles` namespace. Discovery reads package
metadata only and does not import or activate the runtime. The Manager records
LatchMoE as a trusted in-process vLLM-Ascend worker extension; vLLM remains the
lifecycle owner and the `latchmoe` launcher remains responsible for the strict
environment check and for rejecting prefix caching.

```bash
pip install vllm-hust-ext vllm-moe-offload-ascend
vllm-hust-ext extension check org.vllm-hust.latchmoe
vllm-hust-ext extension enable org.vllm-hust.latchmoe
vllm-hust-ext run -- latchmoe serve /path/to/model
```

The check must be supplied with evidence for MoE offload seam ABI 2. A matching
vLLM version alone does not prove that the required vLLM-Ascend seam is present.

---

## Qualified Graph Result

Issue #7 validates the graph lifecycle on Qwen3-30B-A3B BF16, one Ascend
910B2 NPU, TP1, `max_num_seqs=1`, prefix cache disabled, and a 14 GiB offload
budget. Three independent service starts used the same 11-request ShareGPT
gate; each completed 11/11 requests and 1,408 output tokens with exact token-ID
agreement across starts.

| Metric | Repeat 1 | Repeat 2 | Repeat 3 | Mean ± population SD |
|---|---:|---:|---:|---:|
| TTFT p50 (ms) | 573.64 | 537.70 | 573.11 | 561.48 ± 16.82 |
| TPOT p50 (ms/token) | 55.33 | 55.93 | 53.56 | 54.94 ± 1.01 |
| Output throughput (tok/s) | 16.52 | 16.50 | 17.08 | 16.70 ± 0.27 |
| Physical HBM peak / post-release | 91% / 5% | 91% / 5% | 91% / 5% | — |

All starts recorded explicit PIECEWISE capture/replay, stable slot addresses,
generation protection, H2D-before-consume, bounded multi-wave execution, raw
timing/profile logs, and release ACKs. See the
[Issue #7 evidence record](docs/evidence/issue-7-graph-lifecycle.md) and its
checked-in raw bundle. These are graph-correctness and narrow-configuration
service measurements, not a matched performance comparison. The former
capture-off table is intentionally removed: it was a historical diagnostic
without the repeated raw bundle required for a main claim.

---

## Capability Status

The Sage Mate dense `Qwen3.8-27B` target contains no routed experts, so
LatchMoE is **Not Applicable** for that model. This is not an incompatibility
workaround and no dense-model speedup or compatibility claim is made.

The separate adaptation target is unquantized `Qwen3-30B-A3B`, BF16, TP4,
PP1, Ascend PIECEWISE graph mode. It is functionally qualified against
vLLM-HUST `762f85b3`, vLLM-Ascend-HUST `4e57439e`, and MoE seam ABI 2 with
four concurrent sequences. All four ranks exercised routed expert mapping,
graph capture/replay, device/host movement, cancellation, malformed-request
recovery, and stable captured addresses. Its measured output throughput was
about 2.91 tok/s versus about 23.57 tok/s without LatchMoE, so this is a
**functional compatibility result with performance degradation**, not a
speedup or deployment recommendation. See the
[Sage Mate TP4 qualification record](docs/evidence/sage-mate-20260904-tp4-graph.md).
The TP1 Issue #7 bundle remains historical mechanism evidence only.

LatchMoE selects its graph seam from a serialized, model-name-independent
capability descriptor. The descriptor fixes the output ABI, shared-expert
representation, router ownership and selection semantics, weight lifecycle,
parallel mode, and overlap mode. Unknown combinations fail closed before a
native/eager fallback can be presented as an enabled LatchMoE run.

| Capability tuple | Status | Evidence boundary |
|---|---|---|
| Routed-only, BF16, builtin router, single NPU | NPU-qualified | Qwen3-30B-A3B Issue #7 graph bundle |
| External resident shared expert, fixed tuple ABI, builtin external/internal router, BF16, single NPU | Implemented | Host ABI/lifecycle/router tests; no NPU qualification claim yet |
| Gated external shared expert with grouped/sigmoid/correction/routed-scale semantics | Implemented | Host guard and router-parity artifact tests; no NPU qualification claim yet |
| Fused or mix-placement shared experts | Unsupported | Reserved for Issue #25 after a backend-qualified fixed shared lane |
| Shared compute/H2D/MLP overlap | Unsupported | Reserved for Issue #26; current shared path is correctness-first no-overlap |
| TP4 tensor-parallel, routed-only BF16 | NPU-qualified (functional; performance degraded) | Qwen3-30B-A3B, four Ascend ranks, PIECEWISE graph capture/replay, max_num_seqs=4 |
| Quantized weights, EP/DP/PCP multi-NPU | Unsupported | Rejected by the capability guard |

`implemented` means the model-generic seam and host tests exist. It does not
mean a checkpoint has completed native oracle, PIECEWISE capture/replay,
overflow/decode, or token-exactness gates. The checked-in
[`benchmark/registry/model_registry_v2.json`](benchmark/registry/model_registry_v2.json)
and
[`benchmark/registry/qualification_matrix_v2.json`](benchmark/registry/qualification_matrix_v2.json)
preserve this distinction. Each benchmark unit records the checkpoint-derived
descriptor digest and matching registry row; the materialized runner remains
the authoritative capability check because config files alone cannot always
identify router ownership.

Regenerate the Phase-A inventory with the locked environment:

```bash
/root/.cache/latchmoe-npu5-v021/venv/bin/python benchmark/scripts/model_registry_v2.py \
  --native-model-class-preflight \
  --model qwen3-30b-a3b=/root/data/shared_models/strict-models/Qwen3-30B-A3B \
  --model glm-4.7-flash=/root/data/shared_models/strict-models/GLM-4.7-Flash \
  --model qwen3-next-80b-a3b-instruct=/root/data/shared_models/strict-models/Qwen3-Next-80B-A3B-Instruct \
  --output benchmark/registry/model_registry_v2.json \
  --matrix-output benchmark/registry/qualification_matrix_v2.json
```

---

## Architecture

LatchMoE is organized into four layers:

```text
┌────────────────────────────────────────┐
│  vLLM / vllm-ascend Hook Seam          │  ← general plugin, explicit seam adapter
├────────────────────────────────────────┤
│  Compute-Protected Slot Lifecycle      │  ← transfer stream + event ordering
├────────────────────────────────────────┤
│  Capacity-Bounded Overflow Staging     │  ← multi-wave; full-layer fallback
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
- Qualified configuration: Qwen3-30B-A3B (unquantized MoE), BF16, TP4,
  PIECEWISE graph mode, `max_num_seqs=4`. Qualification proves functional
  compatibility; measured performance is substantially below the no-plugin
  baseline and must not be presented as a speedup.
- Do **not** combine with vLLM's native weight-offload flags
  (`--cpu-offload-gb`, `--offload-backend prefetch`, `--offload-group-size`);
  the plugin manages expert offload through its own dataplane

---

## 固定兼容版本

LatchMoE 只维护一组宿主源码 commit，并对两套成对的 Ascend 基础运行时做了
不同强度的验证。分支名不是版本锁，部署时必须 checkout 下表中的完整 commit：

| Component | Repository / branch | Locked commit | Version |
|---|---|---|---|
| vLLM | `vLLM-HUST/vllm-hust` | `762f85b311fbab0bcf8921dd216f5093cd58b9b8` | `0.28.1rc1.dev319` |
| Ascend hook seam | `vLLM-HUST/vllm-ascend-hust`, `feature/moe-offload-seam-v2-tp` | `2c8c722107a54127999a64c4eb0ec86139df8c26` | seam ABI 2, source-tested only |

`feature/moe-offload-seam-v2-tp` 是目标基线的 LatchMoE seam 分支。

完整机器可读锁位于
[`vllm_moe_offload_ascend/compatibility.lock`](vllm_moe_offload_ascend/compatibility.lock)。

## 安装

可行的部署方式是把两个宿主仓库固定到上述 commit，然后在同一个 Python
环境安装 LatchMoE。项目故意不把 vLLM 或 vLLM-Ascend 写入普通 PyPI
`dependencies`：在 Ascend 环境中让 pip 自动解析依赖，可能下载 CUDA/上游包并
覆盖已经匹配的 Torch-NPU 软件栈。

以下命令假设上述任一**完整 profile** 已由基础镜像或运维环境提供；不能跨行混搭
Torch-NPU 和 CANN 版本。推荐在镜像 Python 上创建 overlay venv，这不会替换镜像
持有的 Torch、Torch-NPU 或 CANN：

推荐先 clone 本仓库，再用机器可读 lock 驱动安装。自动安装与后面的手工安装是
两条**二选一**的路径，不能顺序执行。脚本会把两个宿主 checkout 到固定 commit、
补充 vLLM `v0.28.1rc0` tag、始终调用当前 `sys.executable -m pip`、跳过与 LatchMoE
seam 无关的自定义算子编译，并在最后执行环境检查：

```bash
git clone https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE.git
cd vllm-ascend-hust-LatchMoE

BASE_PYTHON=${BASE_PYTHON:-python3}
LATCHMOE_HOME=${LATCHMOE_HOME:-"$HOME/.local/share/latchmoe"}
"$BASE_PYTHON" -m venv --system-site-packages "$LATCHMOE_HOME/venv"
source "$LATCHMOE_HOME/venv/bin/activate"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 先检查将要执行的命令；不会 clone、checkout 或安装。
python tools/install_locked_stack.py \
  --workspace "$LATCHMOE_HOME/stack" --dry-run

# 安装到当前 python；目标目录必须不存在或是 clean Git checkout。
python tools/install_locked_stack.py \
  --workspace "$LATCHMOE_HOME/stack"
```

执行到这里即安装完成。需要自行控制宿主 checkout 或排查安装器时，应跳过上面的
两个 `install_locked_stack.py` 命令，复用当前 LatchMoE checkout，并执行下面的
等价步骤：

```bash
LATCHMOE_PLUGIN_ROOT=$PWD
mkdir -p "$LATCHMOE_HOME/stack"

# 1. 固定 vLLM-HUST
git clone https://github.com/vLLM-HUST/vllm-hust.git \
  "$LATCHMOE_HOME/stack/vllm-hust"
git -C "$LATCHMOE_HOME/stack/vllm-hust" fetch origin \
  762f85b311fbab0bcf8921dd216f5093cd58b9b8
git -C "$LATCHMOE_HOME/stack/vllm-hust" fetch \
  https://github.com/vllm-project/vllm.git \
  refs/tags/v0.28.1rc0:refs/tags/v0.28.1rc0
git -C "$LATCHMOE_HOME/stack/vllm-hust" checkout \
  762f85b311fbab0bcf8921dd216f5093cd58b9b8

# 2. 固定唯一的 vLLM-Ascend hook seam
git clone --branch feature/moe-offload-seam-v2-tp \
  https://github.com/vLLM-HUST/vllm-ascend-hust.git \
  "$LATCHMOE_HOME/stack/vllm-ascend-hust"
git -C "$LATCHMOE_HOME/stack/vllm-ascend-hust" checkout \
  2c8c722107a54127999a64c4eb0ec86139df8c26

# 3. 在同一解释器中安装两个宿主和 LatchMoE
VLLM_TARGET_DEVICE=empty python -m pip install \
  --no-deps --no-build-isolation -e "$LATCHMOE_HOME/stack/vllm-hust"
COMPILE_CUSTOM_KERNELS=0 python -m pip install \
  --no-deps --no-build-isolation -e "$LATCHMOE_HOME/stack/vllm-ascend-hust"

python -m pip install --no-deps --no-build-isolation \
  -e "$LATCHMOE_PLUGIN_ROOT"
```

额外 fetch `v0.28.1rc0` tag 用于让 setuptools-scm 从固定 commit 生成
`0.28.1rc1.dev319` 版本；不能用分支名或手写版本替代 commit 校验。

`VLLM_TARGET_DEVICE=empty` 只关闭 vLLM-HUST 自身的 CUDA 构建；运行时 NPU
platform 仍由 Ascend plugin 提供。`COMPILE_CUSTOM_KERNELS=0` 则避免安装 seam 时
编译整套与 Qwen3/LatchMoE 无关的 Ascend 自定义算子。需要这些额外算子的其他
模型应由基础 vLLM-Ascend 镜像单独提供，不能把它们的构建结果当作 LatchMoE
安装是否成功的门禁。

安装完成后无需复制源码、修改 `PYTHONPATH` 或手动调用 `register()`。目标 vLLM
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
commit、seam ABI、成对的基础运行时 profile 以及 `VLLM_PLUGINS` 白名单，并输出
当前 profile 的 qualification 范围。可编辑安装还会提供等价的 `latchmoe check`
命令。`check PASS` 只验证安装契约；真实 NPU 图验证见中文 quickstart。

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

# Serving-shape hint used by the qualified four-request profile
export VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT=4

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
| `VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT` | `0` | Serving-shape hint for the prefill-overflow handoff; the qualified TP4 profile uses `4` |
| `VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS` | auto | Expert override for the derived slot count |
| `VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS` | auto | Comma-separated layer IDs kept fully resident |
| `VLLM_ASCEND_MOE_OFFLOAD_POLICY` | `deadline` | Staging policy (`deadline` / `lru`) |
| `VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD` | `1` on SEW path | Load experts on a dedicated transfer stream |
| `VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE` | `multi_wave` | Prefer qualified native-recombine multi-wave execution when active experts exceed slot capacity; recoverable qualification failures automatically use `full_layer`. Set `full_layer` to force the blocking fallback; `experimental_wave` is a legacy alias for `multi_wave` |
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
The default `multi_wave` overflow mode uses a single native layer-level top-k
recombine so wave-local execution does not change the native combine order. It
has been qualified for the narrow single-card, TP1, `max_num_seqs=1`,
no-prefix-cache configuration documented in
`docs/evidence/issue-13-multi-wave-prefill.md`. Recoverable preflight or native
recombine qualification failures fall back to blocking `full_layer` execution;
NPU/ACL, OOM, and arbitrary runtime failures remain visible. Set the mode to
`full_layer` to force the fallback outside the validated boundary.

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
