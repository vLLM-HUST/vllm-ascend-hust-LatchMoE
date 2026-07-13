# Tier 1 基线对照 —— 失败根因核验与诚实判读 (2026-07-06)

本文件核验 `sharegpt_tier1_controls` run 的 4 个基线失败，逐个追到 server.log 的**真实根因**（不是外层 `Engine core initialization failed` 包装），并据此确定论文的基线对比故事。**整条链路遵守诚信红线：每个结论都标注原始日志出处。**

Run 目录:
`benchmark/artifacts/runs/sharegpt_tier1_controls/sew-offload-ascend-v1-20260706T092739Z`

补跑（NPU 4）: `legacy_layered_14gb_eager` smoke →
`benchmark/artifacts/runs/sew-offload-ascend-v1-20260706T144235Z/legacy_layered_14gb_eager/smoke`

## 逐个失败根因（已从 server.log 追证）

| Case | 模式 | 真实根因（server.log 原文） | 性质 | 能否用 |
|---|---|---|---|---|
| no_offload_capacity_probe | — | `ValueError: No available memory for the cache blocks`（`kv_cache_utils.py:696`，经插件 `patch_fused_moe.py:346` 转发） | 干净 KV OOM | ✅ Claim 1 可行性 |
| native_prefetch_14gb | capture | `RuntimeError: Tried to instantiate dummy base class _cuda_isCurrentStreamCapturing`（`prefetch.py:517 start_onload_to_static`） | **移植缺口**：vLLM 原生 prefetch offloader 是 CUDA-only，昇腾无此 API，`post_init` 加载期即崩 | ⚠️ 不能当 capture 冲突 |
| native_prefetch_14gb_eager | eager | **同上**，同一行 `is_current_stream_capturing()` | 同上，eager 也起不来 | ⚠️ 不能当 eager 基线 |
| legacy_layered_14gb | capture | `Not allow to synchronize captured-stream` + `rtMemcpy ... current capture mode does not support this operation`（`api_error.cc:1016`，`aclrtMemcpy` err 107030） | **真·图捕获冲突** | ✅ **Figure 1 / Claim 3a 金证据** |
| legacy_layered_14gb_eager | eager | `RuntimeError: Expected all tensors to be on the same device, but got weight is on cpu, different from other tensors on npu:0`（`moe_mlp.py:341 npu_grouped_matmul`） | 无固定槽 staging，专家权重留 CPU，算子设备不匹配 | ⚠️ legacy 无法在 eager 正确出 token |

### 关键结论

**紧 HBM 预算下，昇腾上没有任何现存卸载路径能真正服务 token**：native prefetch 未移植（eager/capture 都崩在同一 CUDA API）、legacy 在 eager 崩（权重没搬回 NPU）、legacy 在 capture 崩（捕获流不能同步搬运）。**只有 SEW 能跑。** 这比"别人开图会崩"更强——是"别人根本跑不起来"。

## 主结果 [Y]：Graph-Capture 消融（同一 SEW runtime，只切图开关）

同一峰值 HBM、同一 200 条 ShareGPT mixed_chat：

| Metric | capture-off (eager) | capture-on (SEW) | gain |
|---|---:|---:|---:|
| TTFT p50 (ms) | 2283.2 | 1373.6 | 1.66x ↓ |
| TPOT p50 (ms/tok) | 192.5 | 83.6 | **2.30x ↓** |
| Output throughput (tok/s) | 4.71 | 10.47 | **2.22x ↑** |
| Successful | 200/200 | 200/200 | both ok |

capture-off run: `sharegpt_tier1_controls/.../sew_14gb_capture_disabled/mixed_chat`
capture-on run: `sharegpt_mixed_main/sew-offload-ascend-v1-20260706T080714Z/sew_14gb_autoslots/mixed_chat`

**因为昇腾上没有其它能出 token 的 eager 卸载基线，SEW capture-off 本身就是"最佳可用 eager 卸载"，[Y] 同时充当了 [X] 的角色。**

## 论文基线故事（已定）

采用**能力对比 + [Y] 消融**框架：

1. **能力对比（Figure 1 / Claim 3a）**：现存卸载方案在昇腾紧预算下无一能出 token，各因不同原因失败（表见上）。诚实区分：native = 移植缺口，legacy-capture = 真图冲突，legacy-eager = 无固定槽的 device-mismatch。
2. **定量主结果（Claim 3b, [Y]）**：SEW capture-on vs capture-off = 2.30x TPOT / 2.22x 吞吐。同 runtime 同卸载只切图开关，最干净不可辩驳。
3. **可行性（Claim 1）**：no_offload 干净 KV OOM。

**禁止**把 native_prefetch 写成"capture 崩溃"——它 eager 也崩，是 CUDA-only 移植缺口。

## 已免费到手的机制图（来自 mixed_chat run 的 307K profile）

- 槽命中率预热曲线（smoke 观察：50%→75-88% 稳态），max active experts=126，max wave=4 —— 支撑 D2 B2 波 Claim 4。

## 仍缺（周三后补）

- 重复 3 次给误差棒（当前均为单次 run）。
- Claim 5 正确性：去掉算力保护生命周期复现具体错输出/崩溃。
- decode_heavy / prefill_heavy 两桶做 workload variation。
- （可选）若要 head-to-head 数值 [X]，需修 legacy eager 的 onload 逻辑让专家搬回 NPU——但有"自己给别人搭 baseline"之嫌，当前不做。

## PPT 已更新

`docs/PPT/2026-7-8-SEW-Offload论文汇报.pptx`（21 页，由 `build_sew_report_ppt.py` 生成）：
- Slide 13 兼容性表已改成诚实的四类失败判读。
- **新增 Slide 15「Evidence 3: Graph-Capture Ablation」= [Y] 主结果**（2.30x/2.22x 对比条形图 + 表）。
