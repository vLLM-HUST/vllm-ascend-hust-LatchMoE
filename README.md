# vllm-moe-offload-ascend

MoE（Mixture-of-Experts）Expert Offloading 插件，适用于带 MoE offload hook 接缝的 vllm-ascend-hust（Ascend NPU 后端）。

以独立插件形式提供，通过 vllm 的 `vllm.platform_plugins` 机制自动加载。当前 hook 尚未合入官方 vllm-ascend / vLLM-HUST 主分支，因此部署时必须使用 `vllm-ascend-hust` 的 `moe-offload-hooks` 分支。

---

## 功能

- **Expert Weight Offloading**：将不活跃的 MoE expert 权重卸载到 CPU，按需加载回 NPU，降低显存占用
- **Fixed-Slot Plan**：预分配 NPU slot，按需加载 expert 权重，减少常驻 NPU 显存
- **Phase Split（MVP-D.11）**：将 MLP 计算按 hit/miss 拆分为多阶段，提升 overlap 效率
- **Trace & Profiling**：记录 routing 分布和 pipeline 耗时，用于调优
- **AutoConfig**：通过环境变量 `VLLM_ASCEND_MOE_OFFLOAD_GB` 自动配置 offload 参数

---

## 前提条件

| 依赖 | 说明 |
|------|------|
| Python ≥ 3.10 | 当前验证环境使用 Python 3.11 |
| Ascend CANN | NPU 驱动环境，需在安装 vllm-ascend-hust 前配置好 |
| vllm-hust | vllm 主体（含 `vllm.platform_plugins` 支持） |
| vllm-ascend-hust | NPU 平台后端，必须切换到 `https://github.com/Li-changwu/vllm-ascend-hust/tree/moe-offload-hooks` |

> **重要**：当前插件依赖的 hook 尚未合入官方仓库。不能直接使用官方 vllm-ascend，也不能只使用 vllm-hust-dev-hub 默认拉下来的 vllm-ascend-hust 分支；必须先切换到 `Li-changwu/vllm-ascend-hust:moe-offload-hooks` 并重新安装 vllm-ascend-hust。

### 真实依赖路径

本插件不是单独的 vLLM fork。它必须安装到**运行 `vllm serve` 的同一个 Python/Conda 环境**，并通过 Python entry point 被 vLLM 自动发现：

```text
vllm-hust                         # vLLM 主体，提供 vllm 命令和 platform plugin loader
vllm-ascend-hust                  # Ascend platform backend，提供 MoE hook 接缝
vllm-moe-offload-ascend           # 本仓库，注册 moe_offload_ascend entry point
```

推荐的新服务器目录布局如下；如果路径不同，后续命令中的路径要同步替换：

```text
/root/vllm-hust
/root/vllm-ascend-hust
/root/vllm-moe-offload-ascend
/data/shared_models/modelscope_cache/Qwen/Qwen3-30B-A3B
/data/shared_datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```

论文 benchmark 默认读取上面的模型和数据集路径；新服务器如果使用其它路径，请修改 `benchmark/configs/sew_offload_v1.yaml`，或建立等价软链接。

---

## 安装

### 方式一：从源码可编辑安装（研究开发推荐）

**步骤 1：安装 vllm-hust 和 vllm-ascend-hust**

通过 [vllm-hust-dev-hub](https://github.com/vLLM-HUST/vllm-hust-dev-hub) 一键初始化整个 workspace：

```bash
git clone git@github.com:vLLM-HUST/vllm-hust-dev-hub.git
```
这个命令会下载这个代码仓。然后运行：
```bash
cd vllm-hust-dev-hub
bash scripts/quickstart.sh
```
首次使用请选择菜单项1。脚本会自动下载整个workspace的代码。

**步骤 2：切换 vllm-ascend-hust 到 hook 分支并刷新安装**

`quickstart.sh` 会先安装 vllm-hust 和 vllm-ascend-hust，但当前 MoE offload hook 还在个人分支中。安装本插件前，请先切换 `vllm-ascend-hust`：

```bash
# 进入 quickstart 创建的运行环境；如果你的环境名不同，请替换这里。
conda activate vllm-hust-dev
which python

cd /root/vllm-ascend-hust

# 如果 origin 已经是 Li-changwu/vllm-ascend-hust，可跳过 remote add/set-url。
git remote add moe-offload https://github.com/Li-changwu/vllm-ascend-hust.git 2>/dev/null || \
  git remote set-url moe-offload https://github.com/Li-changwu/vllm-ascend-hust.git
git fetch moe-offload moe-offload-hooks
git switch moe-offload-hooks || git switch -c moe-offload-hooks --track moe-offload/moe-offload-hooks
git pull --ff-only moe-offload moe-offload-hooks

# 将 hook 分支重新安装到当前 vllm-hust-dev 环境。
python -m pip install -e . --no-build-isolation
```

核查 hook 分支和关键文件：

```bash
git branch --show-current
test -f vllm_ascend/_moe_offload_null.py
python - <<'PY'
import vllm_ascend
print(vllm_ascend.__file__)
PY
```

**步骤 3：安装本插件**

```bash
git clone https://github.com/Li-changwu/vllm-moe-offload-ascend.git
python -m pip install -e vllm-moe-offload-ascend --no-deps
```

这里推荐 `--no-deps`：前提是 `vllm-hust` 和 hook 分支的 `vllm-ascend-hust` 已经在当前环境中安装好。这样可以避免 `pip` 在新服务器上额外拉取官方 `vllm-ascend` 包，意外覆盖或绕过本地 HUST hook 分支。

### 方式二：直接 pip 安装（vllm-hust 和 hook 分支 vllm-ascend-hust 已装）

```bash
python -m pip install --no-deps git+https://github.com/Li-changwu/vllm-moe-offload-ascend.git
```

安装后无需任何额外配置，vllm 启动时会自动发现并调用插件的 `register()` 函数。

### 安装后核查

先确认 `python`、`vllm`、`vllm_ascend` 和插件来自同一个环境：

```bash
python - <<'PY'
import sys
import vllm
import vllm_ascend
import vllm_moe_offload_ascend

print("python =", sys.executable)
print("vllm =", vllm.__file__)
print("vllm_ascend =", vllm_ascend.__file__)
print("plugin =", vllm_moe_offload_ascend.__file__)
PY
```

再确认 entry point 已注册：

```bash
python - <<'PY'
from importlib.metadata import entry_points

eps = [
    ep for ep in entry_points(group="vllm.platform_plugins")
    if ep.name == "moe_offload_ascend"
]
print(eps)
raise SystemExit(0 if eps else 1)
PY
```

如果设置了 `VLLM_PLUGINS`，必须保证它没有过滤掉本插件。最简单的做法是启动前不设置该变量；如果必须设置白名单，则要包含 `moe_offload_ascend`。

本仓库的轻量核查命令：

```bash
python -m pytest tests/test_autoconfig.py tests/test_patch_fused_moe.py -q
```

---

## 使用

### 通过环境变量启用（AutoConfig，推荐）

```bash
# 实验约定：不要使用 NPU 0-3；请选择 NPU 4-7 中空闲的一张
export ASCEND_RT_VISIBLE_DEVICES=4

# 设置目标 offload 显存大小（GiB），插件自动推导 resident layers 和 slot 容量
export VLLM_ASCEND_MOE_OFFLOAD_GB=14
export VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE=1
export VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT=1

vllm serve <model> --trust-remote-code ...
```

### 专家调试 override

```bash
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1
export VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=32
export VLLM_ASCEND_MOE_OFFLOAD_POLICY=deadline
export VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1

vllm serve <model> --trust-remote-code ...
```

通常不需要设置 `VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS`。当该变量未显式设置时，插件会根据 `VLLM_ASCEND_MOE_OFFLOAD_GB`、模型专家数/top-k、offloaded layer 数和服务配置在启动期自动推导 slot 容量；显式设置时会作为专家调试 override 保留。

以 Qwen3-30B-A3B 为例，`VLLM_ASCEND_MOE_OFFLOAD_GB=14` 通常会自动推导为 `32` slots；`28` 在 64 GiB 910B-class 单卡上通常也会被 KV-aware 物理 slot-bank 上限压到约 `32` slots，避免 slot bank 挤占 KV cache。`64` slots 只建议作为 slot sensitivity / KV failure 诊断点。最终 slot 数还会受真实 HBM 可用预算、`gpu_memory_utilization`、KV reserve、slot-bank fraction 和最小净显存收益约束。

不要和 vLLM 原生 weight offload 参数混用，例如 `--offload-backend prefetch`、`--offload-group-size`、`--cpu-offload-gb`。本插件通过 vllm-ascend-hust 的 MoE hooks 管理 expert offload，原生 offloader 是另一套路径。

如果设置 `VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE=1`，表示启用 graph-compatible 的 SEW fixed-slot 数据通路；这条路径会主动拒绝原生 prefetch offload 参数，并默认打开 B2 Prefill async load、pinned host store 和 transfer-aware wave schedule。未启用 SEW 数据通路时，AutoConfig 的普通分层路径仍可能通过 vLLM PrefetchOffloader 保留 high-fanout full-weight fallback，这是 legacy/layered 路径的一部分，不应和 SEW fixed-slot 实验混为同一组对比。

`VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT` 用于收窄 B2 overflow handoff 的边界：只有当 vLLM 没有暴露明确 prefill metadata，且当前 token 数大于该 serving-shape hint 时，插件才会把超过 slot 容量的 working set 视为 prompt-shaped overflow 交给 B2 wave prefill。年会低并发 demo 建议固定为 `1`，避免多请求 decode 被误交给 prefill wave 路径。

### 年会 Demo 主路径

8 月 demo 建议只展示 SEW fixed-slot 主路径和独立 baseline，不混用 legacy/layered 路径：

```bash
python tools/run_annual_demo_suite.py \
  --config demo/annual_demo_config.json \
  --case sew_14gb_slots32
```

该 runner 会为每个 case 生成 `case_manifest.json`、`server.log`、`benchmark.json`、`case_result.json` 和 `summary.md`，并在 case 结束时清理 vLLM server 进程组。支持边界见 [docs/compatibility.md](docs/compatibility.md)，论文 Method 骨架见 [docs/paper_method_skeleton.md](docs/paper_method_skeleton.md)。

### CPU-first expert loading（实验）

```bash
export VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD=1
```

该开关用于大模型启动期：offloaded MoE 层的 expert 参数在 `create_weights` 阶段直接分配到 CPU host 内存，后续只按层短暂搬到 NPU 做 Ascend 格式化，再回落 CPU host store，避免“所有 expert 先完整加载到 NPU，再整体拷回 CPU”的启动峰值。当前第一阶段只覆盖 unquantized fixed-slot offloaded 层；resident 层、非 MoE 权重和暂未适配的量化 MoE 权重仍走原始加载路径。

### 验证插件已加载

启动日志中应出现：

```
Enabled Ascend MoE offload autoconfig from VLLM_ASCEND_MOE_OFFLOAD_GB. ...
```

更完整的启动期信号包括：

```text
moe_offload_ascend -> vllm_moe_offload_ascend:register
Platform plugin ascend is activated
Enabled Ascend MoE offload autoconfig from VLLM_ASCEND_MOE_OFFLOAD_GB
```

如果启动日志里没有 `moe_offload_ascend`，优先检查：插件是否安装到了当前 `python` 所在环境、`VLLM_PLUGINS` 是否过滤了插件、`pip install -e` 是否漏掉了 `--no-deps` 导致 vLLM/Ascend 包被替换。

### Benchmark 路径核查

运行论文 benchmark 前，请先确认默认路径存在：

```bash
test -d /data/shared_models/modelscope_cache/Qwen/Qwen3-30B-A3B
test -f /data/shared_datasets/ShareGPT_V3_unfiltered_cleaned_split.json
python benchmark/scripts/sew_bench.py validate
```

准备 ShareGPT workload manifest：

```bash
python benchmark/scripts/sew_bench.py prepare-workloads \
  --bucket smoke \
  --requests-per-bucket 1
```

在 NPU 4-7 中选择一张空闲卡后再跑 smoke：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python benchmark/scripts/run_suite.py \
  --case sew_28gb_autoslots \
  --workload smoke
```

正式实验不要使用 NPU 0-3；请使用 `ASCEND_RT_VISIBLE_DEVICES=4/5/6/7` 中的空闲卡。

### 禁用插件

不设置 `VLLM_ASCEND_MOE_OFFLOAD_GB`，或：

```bash
pip uninstall vllm-moe-offload-ascend
```

卸载后 vllm-ascend-hust 自动回落到 null stubs，功能不受影响。

---

## 环境变量参考

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `VLLM_ASCEND_MOE_OFFLOAD_GB` | 未设置（禁用） | 目标 offload 大小（GiB），设置即启用 AutoConfig |
| `VLLM_ASCEND_MOE_OFFLOAD_ENABLED` | `1` | 是否启用 offload |
| `VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS` | 自动推导 | NPU 上预分配的 expert slot 数量；仅建议作为专家 override |
| `VLLM_ASCEND_MOE_OFFLOAD_POLICY` | `deadline` | 调度策略（`deadline` / `lru`） |
| `VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY` | `0` | 仅收集 routing trace，不做实际 offload |
| `VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME` | 普通路径 `1`，SEW 路径 `0` | 启用分层运行时（resident + offload 混合） |
| `VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD` | 普通路径 `0`，SEW 路径 `1` | 使用独立 transfer stream 异步加载 expert；Prefill B2 overlap 依赖该开关 |
| `VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD` | 跟随 `NUM_SLOTS` | 切换 slot cache / full weight 路径的 expert 数阈值 |
| `VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES` | `1` | Phase split 最大阶段数 |
| `VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS` | `0` | host store 注册后是否释放 offloaded layer 的原始 NPU expert 权重 |
| `VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS` | 自动推导 | 逗号分隔的常驻层 ID（不 offload） |
| `VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_BUDGET_GB` | 自动读取 | slot bank 可使用的 HBM 预算；通常由 `torch.npu.mem_get_info()` 和 `gpu_memory_utilization` 推导，设置该变量可显式覆盖 |
| `VLLM_ASCEND_MOE_OFFLOAD_MIN_NET_SAVING_RATIO` | `0.25` | 自动推导 slots 时至少保留的净显存收益比例 |
| `VLLM_ASCEND_MOE_OFFLOAD_MIN_NET_SAVING_GB` | 未设置 | 自动推导 slots 时至少保留的净显存收益 GiB 下限 |
| `VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_FRACTION` | `0.12` | 使用真实 `torch.npu.mem_get_info()` 时，slot bank 最多使用物理 NPU HBM 的比例；用于防止 28 GiB/64 slots 挤占 KV cache |
| `VLLM_ASCEND_MOE_OFFLOAD_KV_RESERVE_SEQS` | `4` | AutoConfig 预留 KV cache 的 sequence 数；不是默认 serving 并发，真实并发仍由 vLLM `--max-num-seqs` 控制 |
| `VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE` | `0` | 启用 SEW router-stage-MLP graph-compatible fixed-slot 数据通路 |
| `VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD` | `0` | 实验开关；offloaded unquantized MoE expert 在初始化时直接落到 CPU host store，降低启动期 NPU 峰值 |
| `VLLM_ASCEND_MOE_OFFLOAD_TRANSFER_AWARE_SCHEDULE` | SEW 路径 `1` | Prefill B2 根据每个 wave 的 H2D bytes 和 routed pair 数调整 stage/compute 顺序，尽量用当前 wave 计算覆盖下一波加载 |
| `VLLM_ASCEND_MOE_OFFLOAD_PREFILL_PREFETCH_DEPTH` | `1` | SEW B2 Prefill 软件流水预取深度 |
| `VLLM_ASCEND_MOE_OFFLOAD_PREFILL_BUFFER_COUNT` | `2` | SEW B2 Prefill stage buffer 数 |
| `VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT` | `0` | B2 overflow fallback 的 serving-shape hint；低并发 demo 建议设为 `1` |
| `VLLM_ASCEND_MOE_GMM_PROFILE_PATH` | 未设置 | MoE/GMM profile JSONL 输出路径 |
| `VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH` | 未设置 | offload/stage profile JSONL 输出路径 |

---

## 与 vllm-ascend-hust 的关系

```
vllm
 └── vllm-ascend-hust (NPU 平台后端)
      ├── vllm_ascend/_moe_offload_null.py  ← 无插件时的空实现
      └── ops/fused_moe/*.py                ← try/except 导入 hook 点

vllm-moe-offload-ascend (本插件，可选)
 └── 注册 vllm.platform_plugins
      └── register() → apply_patches()
           └── 将 null stubs 替换为真实实现
```

插件通过 Python 包的 `vllm.platform_plugins` entry point 注册，vllm 在平台初始化时自动调用 `register()`，将 null stubs monkey-patch 为本包提供的真实实现。

---

## 新服务器常见故障

| 现象 | 优先检查 |
|------|----------|
| 启动日志没有 `moe_offload_ascend` | 插件是否安装到当前 `python` 环境；entry point 是否存在；`VLLM_PLUGINS` 是否过滤插件 |
| `ModuleNotFoundError: vllm_ascend` | 当前环境没有安装 vllm-ascend-hust，或 `vllm` 命令来自另一个环境 |
| 插件导入成功但 hook 不生效 | `vllm-ascend-hust` 是否切到 `moe-offload-hooks` 分支并在同一环境里重新 `pip install -e .` |
| 安装插件后 vLLM 行为突然变化 | 是否未使用 `--no-deps`，导致 pip 安装/替换了官方 `vllm-ascend` |
| SEW case 仍然出现原生 prefetch 行为 | 是否混用了 `--offload-backend prefetch`、`--cpu-offload-gb`、`--offload-group-size` 等原生 offload 参数 |
| 28 GiB AutoConfig 仍选择 64 slots 并报 KV cache 不足 | 是否显式设置了 `VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=64`；插件是否为包含 KV-aware cap 的最新代码；临时可设 `VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=32` |
| benchmark validate 失败 | `benchmark/configs/sew_offload_v1.yaml` 中的模型、tokenizer、ShareGPT 数据集路径是否存在 |
| NPU 被其它任务占用 | 用 `npu-smi info` 选择 NPU 4-7 中空闲卡，并设置 `ASCEND_RT_VISIBLE_DEVICES` |

---

## 工具脚本

`tools/` 目录包含研究用工具：

| 脚本 | 用途 |
|------|------|
| `collect_moe_trace.py` | 收集 routing 分布 trace |
| `run_minimal_offload_benchmark.py` | 最小化 offload benchmark |
| `run_fixed_slot_smoke.py` | fixed-slot 冒烟测试 |
| `simulate_expert_slots.py` | 模拟 slot 命中率 |
| `estimate_fixed_slot_memory.py` | 估算 slot 显存占用 |
| `analyze_layered_strategy.py` | 分析分层 offload 策略 |
| `moe_offload_timeline.py` | 可视化 offload timeline |
| `run_annual_demo_suite.py` | 年会 demo runner：启动 server、跑 ShareGPT benchmark、记录 manifest/log/json、清理进程 |

---

## License

Apache 2.0
