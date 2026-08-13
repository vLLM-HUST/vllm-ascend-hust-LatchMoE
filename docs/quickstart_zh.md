# LatchMoE 安装与启动

本文档面向希望在已有 Ascend vLLM 软件栈上安装和启动 LatchMoE 的开发者。
核心原则只有一条：**始终用安装 LatchMoE 的同一个 Python 启动 vLLM**。

安装后推荐使用下面的统一入口：

```bash
python -m vllm_moe_offload_ascend check
python -m vllm_moe_offload_ascend serve <model> [vLLM 参数...]
```

可编辑安装还会生成等价的短命令：

```bash
latchmoe check
latchmoe serve <model> [vLLM 参数...]
```

统一入口会先检查当前 Python、vLLM、vLLM-Ascend、LatchMoE 插件入口和
`VLLM_PLUGINS`，检查通过后才在同一个 Python 进程中启动 vLLM。LatchMoE 只
注册为 `vllm.general_plugins`；`ascend` 仍是 `vllm.platform_plugins`。这样可以
避免激活虚拟环境后，裸 `vllm` 命令仍然由系统 Python 启动的问题。

## 环境要求

- Ascend NPU，以及可用的 CANN 和 torch-npu 环境；
- `vLLM-HUST/vllm-hust` commit
  `ad7125a431e176d4161099480a66f0169609a690`（vLLM 0.21.0）；
- hook-enabled `vllm-ascend-hust`：
  `feature/latchmoe-offload-seam-v1-v021`；
- seam commit：`4806367eeeb7d62b32078ae90cd929cc06d825fe`；
- Python 3.10 或更高版本。

## 方式一：源码可编辑安装（研究开发推荐）

### 自动安装固定宿主（推荐，可直接执行）

在已经提供 CANN、Torch 与 Torch-NPU 的 Ascend Python 环境中执行：

```bash
git clone https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE.git
cd vllm-ascend-hust-LatchMoE

BASE_PYTHON=${BASE_PYTHON:-python3}
LATCHMOE_HOME=${LATCHMOE_HOME:-"$HOME/.local/share/latchmoe"}
"$BASE_PYTHON" -m venv --system-site-packages "$LATCHMOE_HOME/venv"
source "$LATCHMOE_HOME/venv/bin/activate"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/install_locked_stack.py \
  --workspace "$LATCHMOE_HOME/stack" --dry-run
python tools/install_locked_stack.py \
  --workspace "$LATCHMOE_HOME/stack"
```

安装器直接读取 `compatibility.lock`，拒绝覆盖 dirty checkout，并完成以下容易
出错的步骤：

- 固定 vLLM-HUST 与唯一 seam 分支的完整 commit；
- 使用 filtered shallow fetch，只下载固定快照需要的 Git 对象；
- 从上游补充 vLLM `v0.21.0` tag，保证 setuptools-scm 版本正确；
- 全程使用当前 `sys.executable -m pip`，防止 venv 与系统 Python 混用；
- 以 `VLLM_TARGET_DEVICE=empty` 安装 vLLM-HUST；
- 以 `COMPILE_CUSTOM_KERNELS=0` 安装 hook seam，避免编译 37 个与本插件无关的
  自定义算子及其 CANN 子进程；
- 使用 `--no-deps --no-build-isolation` 保护镜像中的 Torch-NPU 软件栈；
- 最后自动运行 `python -m vllm_moe_offload_ascend check`。

### 手动安装步骤
#### 步骤 1：安装固定的 vLLM-HUST

```bash
git clone https://github.com/vLLM-HUST/vllm-hust.git
git -C vllm-hust fetch origin ad7125a431e176d4161099480a66f0169609a690
git -C vllm-hust fetch https://github.com/vllm-project/vllm.git \
  refs/tags/v0.21.0:refs/tags/v0.21.0
git -C vllm-hust checkout ad7125a431e176d4161099480a66f0169609a690
VLLM_TARGET_DEVICE=empty python -m pip install \
  --no-deps --no-build-isolation -e ./vllm-hust
```

第二次 fetch 用于补充 HUST 仓库当前没有发布的 `v0.21.0` tag；固定 commit 与
该上游 tag 完全相同，setuptools-scm 需要它生成正确的 `0.21.0` 包版本。

#### 步骤 2：安装固定的 vLLM-Ascend hook seam

```bash
git clone --branch feature/latchmoe-offload-seam-v1-v021 \
  https://github.com/vLLM-HUST/vllm-ascend-hust.git

git -C vllm-ascend-hust checkout \
  4806367eeeb7d62b32078ae90cd929cc06d825fe

COMPILE_CUSTOM_KERNELS=0 python -m pip install \
  --no-deps \
  --no-build-isolation \
  -e ./vllm-ascend-hust
```

核查实际导入路径和 commit：

```bash
git -C vllm-ascend-hust rev-parse HEAD
python - <<'PY'
import vllm_ascend
print("vllm_ascend =", vllm_ascend.__file__)
PY
```

#### 步骤 3：安装 LatchMoE

```bash
git clone https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE.git

python -m pip install \
  --no-deps \
  --no-build-isolation \
  -e ./vllm-ascend-hust-LatchMoE
```

这里必须使用 `python -m pip`，这样安装动作与之后的 Python 启动命令属于同一个
解释器。`--no-deps` 用来避免 pip 自动替换已经安装好的 vLLM、Torch 或
vLLM-Ascend。

这里的 hook seam 只提供 LatchMoE 所需的 Python hook ABI；不要把“成功编译
vLLM-Ascend 全量自定义算子”当作插件安装前置条件。全量构建会编译大量与
Qwen3/LatchMoE 无关的算子，还可能由 CANN 构建子进程选中 PATH 中另一个
Python。确有其他模型需要额外算子时，应在基础镜像层独立构建并验证。

安装后无需修改 vLLM/vLLM-Ascend 源码、设置 `PYTHONPATH` 或手动调用
`register()`。vLLM 会在 API、EngineCore 和 Worker 进程自动加载
`vllm.general_plugins`；LatchMoE 的 `register()` 会先把插件配置注册到
`vllm.envs` 与 `vllm_ascend.envs`，再安装 seam adapter。

#### 步骤 4：安装后核查

```bash
python -m vllm_moe_offload_ascend check
```

正常输出至少包含：

```text
LatchMoE environment check: PASS
python = <当前环境的 Python>
vllm = <vLLM 模块路径>
vllm_ascend = <hook seam 模块路径>
plugin = <LatchMoE 模块路径>
platform_plugins = ascend
general_plugins = moe_offload_ascend
runtime_profile = issue7 或 cann9_legacy
qualification = 对应 profile 的验证范围
```

机器部署脚本可以读取 JSON：

```bash
python -m vllm_moe_offload_ascend check --json
```

### 真实 NPU graph smoke

`check PASS` 只证明安装、版本和插件发现契约正确。确认目标卡空闲后，用已安装的
固定 seam 和本地 Qwen3-30B-A3B 跑真实 graph smoke：

```bash
npu-smi info
benchmark/scripts/run_issue4_graph_repro.sh \
  --seam-root "$LATCHMOE_HOME/stack/vllm-ascend-hust" \
  --model /path/to/Qwen3-30B-A3B \
  --device 0 \
  --output-root ./latchmoe-evidence
```

成功不能只看退出码；最新 run 的 `graph_verification.json` 必须包含
`"status": "passed"`、非零 `aclgraph_replay`、`h2d_events` 和
`graph_slot_probe`，且 `failures` 为空。该 smoke 固定关闭 prefix cache 并使用纯
`PIECEWISE` ACLGraph。

## 方式二：在已有 HUST 软件栈中直接安装

如果 vLLM 0.21.0 和固定的 hook seam 已经安装到当前 Python，可以直接安装
LatchMoE：

```bash
python -m pip install --no-deps \
  git+https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE.git

python -m vllm_moe_offload_ascend check
```

这条命令只安装 LatchMoE，不负责安装或切换 vLLM-Ascend hook seam。

## 启动服务

### 启动前资源检查

先确认目标 NPU 没有不属于本次验证的进程，并记录实际空闲 HBM：

```bash
npu-smi info
python - <<'PY'
import torch
import torch_npu  # noqa: F401

free, total = torch.npu.mem_get_info()
print(f"free={free / 2**30:.2f} GiB total={total / 2**30:.2f} GiB")
PY
```

如果 `npu-smi` 显示的是其他容器 PID，不要直接终止。等待资源释放或选择空闲卡。
`--gpu-memory-utilization` 只设置 vLLM 的预算/启动门禁，无法解决真实 HBM 已被
占用的问题。共享卡数据只能用于功能诊断，不能作为 TTFT/TPOT 性能基线。

### AutoConfig（推荐）

```bash
# 根据服务器实际空闲卡选择设备。
export ASCEND_RT_VISIBLE_DEVICES=4

# 目标 expert offload 容量，单位 GiB。
export VLLM_ASCEND_MOE_OFFLOAD_GB=14
export VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE=1
export VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -m vllm_moe_offload_ascend serve \
  /path/to/Qwen3-30B-A3B \
  --trust-remote-code \
  --max-num-seqs 1
```

`VLLM_ASCEND_MOE_OFFLOAD_GB=14` 表示生成约 14 GiB expert 的部分常驻规划，
不表示整个模型只占 14 GiB。若要在显存受限卡上验证“所有 MoE 层均卸载”，需显式
设置空的常驻层集合，并开启 CPU-first 与固定 slot override：

```bash
export VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS=
export VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD=1
export VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=16
```

该配置会以更多 H2D 传输换取更低 HBM 占用。常规服务仍建议使用 AutoConfig 的
部分常驻规划，除非已针对目标负载核验容量与性能取舍。

SEW AutoConfig 默认会在 host store 和 fixed-slot bank 就绪后释放 offloaded
expert 的原始 NPU 副本。只有在排查“保留完整权重”路径时才显式设置
`VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS=0`；该模式仍会执行
staging，但不会回收目标 expert 权重占用的 HBM。

### 专家调试 override

通常不需要显式指定 slot 数量。只有在复现实验或排查 AutoConfig 时才设置：

```bash
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1
export VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=32
export VLLM_ASCEND_MOE_OFFLOAD_POLICY=deadline
export VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1

python -m vllm_moe_offload_ascend serve \
  /path/to/Qwen3-30B-A3B --trust-remote-code
```

不要与 vLLM 原生 weight-offload 参数混用：

```text
--offload-backend prefetch
--offload-group-size
--cpu-offload-gb
```

LatchMoE 的 expert slot/Host Store 与 vLLM 原生 offloader 是两条不同的数据路径。

启用 SEW dataplane 后，Graph 服务会自动使用纯 `PIECEWISE` ACLGraph。不要手动改成
`FULL`、`FULL_DECODE_ONLY` 或 `FULL_AND_PIECEWISE`：这些模式会把 decode 外层整体
捕获，使依赖实时路由的 expert staging 在 replay 时不再执行，进而复用陈旧的 slot
映射。使用 `--enforce-eager` 时仍保持 `NONE`，不会被自动开启 Graph。

### CPU-first expert loading（实验）

```bash
export VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD=1
```

该开关降低模型启动时“所有 expert 先进入 NPU”的瞬时 HBM 峰值。目前主要覆盖
unquantized fixed-slot offloaded layers；resident layers 和未适配的量化权重仍走
原始加载路径。

## 验证插件已经加载

每个 vLLM 进程的启动日志都应包含：

```text
moe_offload_ascend -> vllm_moe_offload_ascend:register
ascend -> vllm_ascend:register
Platform plugin ascend is activated
Enabled Ascend MoE offload autoconfig from VLLM_ASCEND_MOE_OFFLOAD_GB
LATCHMOE_GRAPH_CONFIG cudagraph_mode=PIECEWISE splitting_op=vllm::moe_offload_stage status=enabled
```

`check` 通过只证明安装和自动发现契约正确，不证明模型加载、NPU Offload 或
ACLGraph 已经正确。发布或实验前仍需完成真实模型请求和 token 正确性验证。

## VLLM_PLUGINS 白名单

最简单的做法是不设置 `VLLM_PLUGINS`。如果部署必须使用白名单，应同时包含：

```bash
export VLLM_PLUGINS=ascend,moe_offload_ascend
```

统一启动入口会拒绝缺少其中任意一个插件的白名单。

## 常见故障

### 激活环境后仍然启动了系统 vLLM

```bash
command -v python
command -v vllm
head -1 "$(command -v vllm)"
```

如果 `vllm` shebang 不属于当前环境，不要使用它。改用：

```bash
python -m vllm_moe_offload_ascend serve <model>
```

### 缺少 moe_offload_ascend entry point

确认插件由当前 Python 安装：

```bash
python -m pip install --no-deps --no-build-isolation \
  -e /path/to/vllm-ascend-hust-LatchMoE
python -m vllm_moe_offload_ascend check
```

### 禁用或卸载插件

不设置 `VLLM_ASCEND_MOE_OFFLOAD_GB` 即不会通过 AutoConfig 启用 Offload。完全卸载：

```bash
python -m pip uninstall vllm-moe-offload-ascend
```

卸载后，hook-enabled vLLM-Ascend 应回落到 null stubs。
