# Phase 2A 立即执行计划 — ShareGPT 首版可展示结果

**结论先行**: 实验设置**已经在 `benchmark/configs/sew_offload_v1.yaml` 里定义好了**，且与文献规范（MoE offloading 主实验 batch=1、ShareGPT、同一峰值 HBM、eager/capture 配对）一致。本文档只做两件事：(1) 把已确定的设置写清楚供汇报引用；(2) 给出本周三前跑出 ShareGPT 结果的最小执行路径。

---

## 一、已确定的实验设置（来自 sew_offload_v1.yaml，无需再设计）

### 固定项

| 维度 | 值 | 出处 |
|---|---|---|
| 模型 | Qwen3-30B-A3B (bf16, TP=1) | `model:` |
| 硬件 | 单卡 Ascend 910B2 (64 GiB HBM) | `hardware:` |
| **并发** | **`max_num_seqs=1`, concurrency=1, request_rate=inf** | `serving_shape/concurrency:` |
| max_model_len | 4096 | `serving_shape:` |
| gpu_mem_util | 0.90 | `serving_shape:` |
| 数据集 | ShareGPT_V3（真实分布，禁止合成/随机） | `dataset:` |
| KV 预留 | AutoConfig 已扣 KV floor/激活/B2 buffer | roadmap Phase 0B |
| 设备选择 | **NPU 5**（当前唯一空闲的 4-7 卡；4=36G/6/7=55G 已占用） | 现场实测 2026-07-06 |

**并发说明（写进 PPT / 论文的口径）**: 主实验单请求 batch=1，跟随 HOBBIT（all batch=1）、MoE-Infinity（单卡低 RPS）。这是"内存受限单卡可行性"范式，**不是吞吐竞赛**。2/4 路并发只作"固定槽机制不崩、正确"的次要节。

### Workload buckets（输入/输出长度 —— 你问的核心）

| bucket | prompt tokens | output tokens | num_requests | 默认启用 | 用途 |
|---|---|---:|---:|:---:|---|
| `smoke` | 64–512 | 8 | 1 | ✗ | 链路自检 |
| **`mixed_chat`** | **mixed（真实 ShareGPT 分布）** | **128** | **200** | ✓ | **主实验（e1 端到端）** |
| `decode_heavy` | 64–256 | 256 | 128 | ✓ | decode 热路径 TPOT |
| `prefill_heavy` | 1024–2048 | 32 | 64 | ✓ | TTFT + B2 波 |
| `long_context_prefill` | 2048–4096 | 16 | 32 | ✗ | 极限 prefill/专家溢出 |

> `mixed` = 直接采样真实 ShareGPT 对话长度分布（非固定长度），最贴近文献主流做法。固定长度桶（decode/prefill_heavy）用来做机制的可控扫描。

### Cases（对比矩阵，已定义 17 个）

**核心 5 个（e1 端到端主线，同一 14 GiB 预算）**：

| case | 模式 | 预期 | 证明 |
|---|---|---|---|
| `no_offload_capacity_probe` | — | OOM/KV 失败 | Claim 1 可行性 |
| `native_prefetch_14gb` | capture 尝试 | **崩溃**（stream not joined） | Fig.1 / Claim 3a |
| `native_prefetch_14gb_eager` | eager | 出 token，性能基线 | Claim 3b 的 [X] 分母 |
| `sew_14gb_capture_disabled` | eager | 出 token | Claim 3b 的 [Y] 分母 |
| `sew_14gb_autoslots` | **capture-on** | **主结果** | Claim 3b 分子 |

其余：`legacy_layered_*`（另一基线）、`sew_28gb_*`（第二预算点）、`sew_*_no_b2/no_transfer_aware/no_cpu_first`（消融）、`sew_*_slots8/16/32/64`（槽敏感性）。全在 yaml 里，实验分组 e0–e5。

### 指标（已定义）

serving: TTFT/TPOT (mean/p50/p90/p99)、output/request throughput、成功/失败数。
offload: h2d_bytes、staging_time、slot 命中率、active_expert_count、b2_wave_count。
evidence: graph_capture_completed、moe_offload_stage_seen、failure_reason。

---

## 二、本周三前的最小执行路径

**约束**: 每个 case 要重启 server（加载 30B 权重 ~96s + 图捕获），batch=1 decode 约 1.5–3.9 tok/s。200 请求 × 128 token 的 mixed_chat 单 case 要跑 1–4 小时，**首版不能用满配**。

### 执行顺序（NPU 5）

```bash
cd /root/vllm-moe-offload-ascend
export ASCEND_RT_VISIBLE_DEVICES=5   # 唯一空闲的 4-7 卡

# 0) 已完成：建 manifest（425 请求）
#    python benchmark/scripts/sew_bench.py --config <cfg> prepare-workloads

# 1) 链路自检（1 请求，已在跑）——确认 SEW capture-on 通
python benchmark/scripts/run_suite.py --config benchmark/configs/sew_offload_v1.yaml \
  --case sew_14gb_autoslots --workload smoke

# 2) 首版可展示结果：decode_heavy 缩到小样本，跑核心 3 case
#    （见下方"缩样本"说明）
```

### 缩样本策略（关键）

首版目标是**拿到真实 TTFT/TPOT 数字 + 证明 capture-on 能跑/基线崩**，不追求统计功效。做法：临时把要跑的 bucket 的 `num_requests` 调小（8–16），或新增一个 `decode_quick`(prompt 64–256, output 64, n=8) 桶。

**首版跑 3 个 case × 1 个小 bucket**：
- `native_prefetch_14gb`（capture）→ 预期崩，留 server.log 报错 = Fig.1
- `sew_14gb_capture_disabled`（eager）→ TTFT/TPOT 基线
- `sew_14gb_autoslots`（capture-on）→ 主结果

时间估算：3 case × (~2min 加载 + ~2min 捕获 + 8 请求 × ~30s) ≈ 3 × 8min ≈ **25–35 分钟**。

### 收集结果

```bash
python benchmark/scripts/sew_bench.py summarize <run_dir> --output <run_dir>/summary.json
# 或直接看每个 unit 的 summary.md / benchmark.json
```

---

## 三、结果如何进 PPT

**Slide 13（证据状态）**: Claim 3a 从 "Smoke ✓" → "**ShareGPT 实测 ✓**"。

**Slide 15（关键结果 1 · 动机矩阵）**: 补真实数字列——
| 路径 | Eager | Capture |
|---|---|---|
| Native prefetch | TPOT=**XX** ms | ✗ stream not joined |
| SEW-Offload | TPOT=**YY** ms | ✓ TPOT=**ZZ** ms |

[X]=(ZZ−XX)/XX，[Y]=(ZZ−YY)/YY。

**诚实口径**: 若首版只跑 8 请求，PPT 明确标"n=8 预览，全负载 Phase 2 补齐"，绝不把小样本当最终结果（延续 roadmap 诚信红线）。

---

## 四、周三后的完整矩阵（Phase 2 全量）

直接用已定义的实验分组，无需再设计：

```bash
# e1 端到端（mixed_chat + decode_heavy + prefill_heavy × 9 cases）
python benchmark/scripts/run_suite.py --config <cfg> \
  --case no_offload_capacity_probe --case native_prefetch_14gb_eager \
  --case sew_14gb_capture_disabled --case sew_14gb_autoslots \
  --workload mixed_chat
# e5 槽敏感性、e3 B2、e4 消融同理，按 experiments: 分组选 case/workload
```

- Phase 2A 性能对齐 → e1
- Phase 2B 预算/槽扫描 → e5（14/28 GiB × slots 8/16/32/64）
- Phase 2C 正确性 → e4（no_cpu_first / 需另加 no-protect 变体复现错输出）
- B2 波 → e3（prefill_heavy + long_context_prefill）
- 真实 ShareGPT 分布 → mixed_chat 全 200 请求，报 P50/P90/P99

---

## 执行清单（本周）

- [x] 建 manifest（425 请求）
- [x] 确认 NPU 5 空闲、config validate OK、环境可跑
- [ ] smoke 自检通过（SEW capture-on 出 token）← 进行中
- [ ] 首版：3 core case × 小 bucket，拿 TTFT/TPOT
- [ ] native capture 崩溃留 log（Fig.1 证据）
- [ ] 数字填 PPT slide 13/15
