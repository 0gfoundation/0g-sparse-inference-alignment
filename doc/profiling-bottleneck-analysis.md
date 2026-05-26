# SIA 端到端 Profiling 与瓶颈分析

**日期**：2026-05-26
**目标**：基于代码里已有的 profiling 点位，把 `exp/` 下的真实日志聚合一遍，定位 SIA 当前的耗时瓶颈，并据此提出后续优化方向。
**核心发现**：单张共享 H200 上，每生成一个 token，33% 时间在 LLM forward、67% 时间在等 RM；**等 RM 的时间里 70% 是 RM 模型 forward（5 候选 × 短 extension）本身的算力**。这是后续所有优化的攻击目标。

**优化约束**：本报告所有方案都按"**不影响干预质量**"为前提筛选。任何牺牲 SIA 准确率或 reward 提升的招（缩 VM、降 topk、提 entropy_threshold）一律标记暂不实施。剩下的硬核路径汇集到 §6 路线图。

---

## 1. 代码层：profiling 点位

### 1.1 `src/sia_rm_server.py`（PyTorch RM）—— 9 个 phase 详细计时

环境变量 `RM_PROFILE=1` 启用（默认开），用 `_pf_record()` 累积每个 phase 的耗时，每 50 次 RM 调用打一次 summary。

| Phase | 含义 |
|---|---|
| `tokenize` | 服务端对 5 个 candidate 的 chat-formatted 字符串做 tokenize |
| `prefix_calc` | 找 5 个 candidate token 序列的公共 stable prefix |
| `state_lookup` | 查 per-request KV cache 状态字典 |
| `prep_tensor` | 准备 input tensors（pad、attention_mask 等） |
| `kv_expand` | 把单条 prefix 的 KV 扩展为 batch=5 |
| **`forward`** | **RM 模型 forward（HIT 路径只 forward extension 部分）** |
| `score_extract` | 从 pooler logits 抽 reward 分数 |
| `kv_save` | 把扩展后的 stable prefix KV 存回 cache |
| `miss_prefix_fwd` | 冷启动 MISS 路径全 prefix forward（少见） |

输出格式：
- `[RM-pf-hit] req=... ... | tokenize=X.X prefix_calc=X.X state_lookup=X.Xms` —— 每次 HIT 一行
- `[RM-pf-miss] req=... ... | tokenize=X.X prefix_calc=X.X miss_prefix_fwd=X.Xms` —— 每次 MISS 一行
- `[RM-pf-fwd]` / `[RM-pf-fwd-cg]` —— forward 完成后含 k/diff/kv 信息
- `[RM-pf-summary @N]` —— 每 50 次调用聚合 p50/p95/max
- `[RM] score: path=... fwd=Xms total=Yms` —— 每次 /score endpoint 退出前的最终一行

可选 `RM_PROFILE_LAYERS=1` 开 per-layer hook，打印每个 transformer decoder layer 的耗时分布（debug 用，~0.5ms/层 overhead）。

### 1.2 `src/sia_vllm_RM.py`（SIA processor）—— 只决策行

每个生成 token 打一行 `[SIA] step=N req=I {SKIP|INTERVENE} ...`，含 entropy / top-1 flip / pre/post token id。**没有 per-RM-call latency 计时**。

### 1.3 `vllm serve`（vLLM RM 服务端）—— 自带 throughput logger

每 10 秒打一行 `Engine 000: Avg prompt throughput: ... tokens/s, Avg generation throughput: ..., Prefix cache hit rate: XX%`。从这里可读出 prefix cache 命中率。

---

## 2. 实测瓶颈：5-20 RM_PROFILE 全 600 题，97,175 次 RM 调用

数据源：`exp/log_rm_server_SIA_202605201500.txt`（45 MB，196,293 行 profiling 输出）

### 2.1 最终 summary

末尾一条 `[RM-pf-summary @97150]`：

| Phase | p50 (ms) | p95 (ms) | max (ms) | 占 HIT 路径 |
|---|---:|---:|---:|---:|
| tokenize | 4.9 | 8.2 | 11.5 | 7.0% |
| prefix_calc | 0.2 | 0.3 | 0.5 | 0.3% |
| state_lookup | 0.0 | 0.1 | 0.1 | 0.0% |
| prep_tensor | 0.6 | 0.8 | 1.6 | 0.9% |
| kv_expand | 3.3 | 5.3 | 12.2 | 4.7% |
| **forward** | **57.0** | **73.4** | **81.3** | **🔴 81.9%** |
| score_extract | 0.1 | 0.1 | 0.1 | 0.1% |
| kv_save | 3.5 | 3.7 | 4.0 | 5.0% |
| **HIT path 合计** | **~69.6ms** | | | |
| miss_prefix_fwd | 80.1 | 95.5 | 121.4 | (MISS only) |

### 2.2 HIT vs MISS 分布

- **HIT: 96,335 次 (99.14%)** —— KV prefix cache 极有效
- **MISS: 840 次 (0.86%)** —— 仅冷启动 / 用户 turn 切换时触发

由于 MISS 比例极小，**HIT 路径的 70ms 就是 RM call 的典型成本**。

### 2.3 端到端反推 vs 内部 profile 的差额

从 `exp/log_SIA_202605201500.txt` 反推（348 题 × 平均 token 长度，约 345k tokens / 13606s 总 latency）：

```
total tokens          = 345,690
total wall latency    = 13,606.5s
推断 total steps      = 97,175 / 0.28 ≈ 347,053（按 28% 干预率）
LLM time (11.3ms/step)= 3,921.7s  (29%)
RM-related time       = 9,684.8s  (71%)
per-RM-call (端到端)  = 9,684.8 / 97,175 ≈ 99.7ms
per-RM-call (内部)    = 69.6ms (HIT path)
差额                  = ~30ms = HTTP + JSON 序列化 + FastAPI 排队
```

**协议/服务调度 overhead ~30ms / call，占 30% 的 per-RM-call 时间**。

---

## 3. 与 vLLM RM 时代（G3）对比

vLLM RM 没有 RM-pf-* 那么细的内部 profile，但能从端到端数据反推：

数据源：`exp/log_SIA_vllmrm_202605252120.txt` + `exp/log_llm_vllmrm_202605252120.txt`

```
total tokens          = 358,835
total wall latency    = 12,103.1s
total INTERVENE       = 102,476  ← grep "INTERVENE\b" 得到
total SKIP            = 255,746  ← grep "SKIP "      得到
干预率                = 102476 / 358222 = 28.6%
LLM time (11.3ms)     = 4,047.9s  (33%)
RM-related time       = 8,055.2s  (67%)
per-RM-call           = 8055.2 / 102476 ≈ 78.6ms
```

### 3.1 对照表

| 项 | PyTorch RM (5-20) | vLLM RM (G3) | 改善 |
|---|---:|---:|---:|
| 整体 tok/s | 25.41 | 29.65 | **+17%** |
| 干预率 | ~28% | 28.6% | 持平 |
| per-RM-call (端到端) | 99.7ms | 78.6ms | **-21ms** |
| 其中 forward 估算 | 57ms | ~40-50ms | -10ms |
| 其中协议+调度+tokenize | ~30ms | ~25ms | -5ms |
| 其中 kv_expand / kv_save | ~7ms | ~0（vLLM 内部）| -7ms |

**vLLM RM 主要节省的是"服务层"和"KV 维护"，forward 本身的算力时间几乎没动**。

### 3.2 vLLM RM 的 prefix cache 命中率

从 `log_vllm_rm_opt_202605252120.txt` 的 `Engine 000` 日志稳态读数：**Prefix cache hit rate 96-98%**。和 PyTorch RM 的 99.14% HIT 接近，结构上 prefix cache 都已经吃满。

---

## 4. 瓶颈定位与阿姆达尔分解

```
SIA per-token = LLM forward + 干预率 × RM call
              = 11.3ms       + 0.286 × 78.6ms
              = 11.3ms       + 22.5ms
              = 33.8ms       → 29.6 tok/s  ✓ (实测 G3 29.65 tok/s)

按"在哪干活"分:
  33% 在 LLM 推理
  67% 在等 RM
      └─ 67% × 70% ≈ 47% 是 RM forward 本身的算力（57ms × 0.286）
      └─ 67% × 30% ≈ 20% 是协议+调度+tokenize+KV 管理

按可压缩性看:
  LLM forward (33%)          — 跟 noSIA 共享，不动
  RM forward (47%)           — 🎯 最大可压目标
  RM 协议层 + KV (20%)        — 可压但收益不如 forward
```

> **核心结论：57ms 的 RM forward（5 候选 × 短 extension 通过 Qwen3-4B eager forward）就是当前架构的最大耗时块。任何不攻这部分的优化收益都有限。**

---

## 5. 优化方向：两条主路径

**约束**：以下所有列出的招式都按"**不影响干预质量**"为前提筛选。任何会降低 SIA 准确率 / reward 提升的招（缩 VM 规模、降 topk、提 entropy_threshold）都标记为"暂不实施"——理由是 SIA 的核心价值在干预效果，宁可慢也不能弱化。

### 5.1 方向 A：加速 Value Model 推理本身（零质量损失子集）

直接压 RM forward 的 57ms。

| 招 | 攻击点 | 复杂度 | 预期 RM 降幅 | 全局 tok/s | 质量影响 |
|---|---|---|---|---|---|
| **A2**. 校准的 FP8（`llm-compressor` 离线产出校准 q/k/v/prob scale 的 checkpoint，避免 G4 中 `--kv-cache-dtype fp8` 触发的退化路径） | weight + KV 量化 | 一次离线 quantize + A/B 验证 | 57 → ~35ms (-40%) | **+18%** | **零**（精度损失通常在 noise 内，需 A/B 确认） |
| **A5 / C2**. Pydantic monkey-patch + 客户端传 token_ids | 服务端 tokenize | ~10 行 hook + 客户端代码 | 协议层省 3-5ms | +5-8% | **零**（bit-exact 等价） |

**暂不实施（影响质量）**：
- ~~A1 换 1.7B VM~~：模型容量缩小，干预决策能力下降
- ~~A3 topk 5→3~~：top-4/5 候选不再被 RM 评分，干预面收窄
- ~~A4 entropy_threshold 1.0→1.3~~：干预次数减少，SIA 强度下降

### 5.2 方向 B：LLM 与 RM 并行 / overlap

| 招 | overlap 机制 | 复杂度 | 预期 | 质量影响 |
|---|---|---|---|---|
| **B1**. Tier 1：分 2 张 GPU | 消除 GPU 资源争抢 + CUDA context 切换 + L2 cache 互污 | 启动命令改 | +15-25% | **零**（纯硬件） |
| **B2**. 有状态 RM（fix_a_token + get_candidates_scores + KV 复用 2B+2C）| SKIP 步 `fix_a_token` 异步藏在 LLM 影子；INTERVENE 步只 forward 5 候选 × 1 token | ~1300 行 + 改 RM server | **+50-60%** | **零**（数学等价） |
| ~~B3. 推测式 SIA (2D rewind)~~ | 当前 flip 率 67% 下边际 < 5% | ~400 行 + vLLM hack | ❌ | (理论零，但收益太小) |

**B2 的核心机制**：当前 vLLM `/classify` 每次 INTERVENE 都要让 RM "重新拼出"完整 prefix（即便有 prefix cache，也得走一遍 vLLM 调度），forward 量 = 5 candidates × extension token 数。B2 用有状态 server 把 RM 的 KV 推进做成"每步异步推 1 token"（藏在 SKIP 步影子里），INTERVENE 时只需 forward 5 candidates × 1 token，**砍掉 4× 的 extension 长度**。详见 `doc/parallel-decoding-design.md`。

### 5.3 方向 C：协议层 / 客户端 / 进程结构（零硬件 + 零质量损失）

| 招 | 攻击点 | 复杂度 | 预期 | 质量影响 |
|---|---|---|---|---|
| **C1**. 客户端 chat template 缓存：把 user_content + chat_template 前缀**每个 query 只渲染一次**并缓存；INTERVENE 时只拼 `response_so_far + candidate` 后缀 | 客户端 5× `apply_chat_template` 重复劳动 | ~50 行 | -1-2ms/call → +2-3% | **零**（bit-exact 等价） |
| **C2**. Pydantic monkey-patch：启动时给 vLLM 的 `ClassificationRequest.input` 打补丁让它接受 `list[list[int]]`（vLLM 0.10.1.1 schema 限制），客户端直接发 token_ids | 服务端 re-tokenize 5 字符串 | ~10 行 hook + 客户端改 | -3-5ms/call → +5-8% | **零**（bit-exact 等价） |
| **C3**. vLLM Python AsyncLLMEngine 内嵌：把 RM 跑在 LLM 同一个 Python 进程里，直接调 `AsyncLLMEngine.encode()` 而非 HTTP | HTTP/JSON 全部协议层 | 中等（共享 engine handle，进程模型可能冲突） | -10-15ms/call → +15-20% | **零**（API 调用，无序列化） |
| C4. Unix socket 代 TCP HTTP | TCP 协议栈 | 启动参数改 | -1ms/call，噪声级 | **零** |

### 5.4 A / B / C 不冲突，可叠加

- A 攻击的是"单次 RM forward 的纯算力"
- B 攻击的是"何时调用 RM、调用时 forward 多少 token"
- C 攻击的是"RM 调用本身的协议层和进程结构开销"

理想上 A2 (校准 FP8) + B1 (分卡) + B2 (有状态 RM) + C1+C2+C3 全部上，per-token 可降到 ~13-14ms → **~65-75 tok/s**（达到 noSIA 88 的 75-85%）。

---

## 6. 实操优先级路线图（零质量损失版本）

按"性价比"和"硬件依赖度"排序：

| Step | 改动 | 预期 tok/s | 累计 vs G3 (29.65) | 准入条件 |
|---|---|---:|---:|---|
| Step 0 | G3 现状 | 29.65 | — | — |
| Step 1 | **C1**: 客户端 chat template 缓存 | ~31 | +5% | 无（纯客户端） |
| Step 2 | **C2**: token_ids monkey-patch | ~33 | +11% | 验证 vLLM `_preprocess_completion` 真接受 token_ids |
| Step 3 | **A2**: 校准 FP8 | ~40 | +35% | A/B 验证 reward 分数稳定 |
| Step 4 | **B1**: 分 2 张 GPU | ~46 | +55% | 拿到 2 张 ≥48GB 的卡 |
| Step 5 | **C3**: in-process RM (vLLM AsyncLLMEngine 内嵌) | ~52 | +75% | 解决进程模型冲突 |
| Step 6 | **B2**: 有状态 RM + KV 复用 | ~65-70 | +120%-135% | ~1300 行工程量 |

**最值得先做的 3 步**（无需 2 卡、无需大改架构）：

```
Step 1 (C1) + Step 2 (C2) + Step 3 (A2)  →  ~40 tok/s  (+35% over G3)
```

**全部零质量损失，全部低/中工程量**。其中 C1 几十行代码即可上，C2 需要先验证 vLLM 内部能吃 token_ids 但补丁极简，A2 需要写一次校准脚本但是一次性工作。

### 不推荐做的

- A1 / A3 / A4：动质量
- B3 推测式 SIA：当前 flip 率 67% 下收益太小
- C4 unix socket：收益噪声级

---

## 7. 引用

- 完整 profiling 数据：[`exp/log_rm_server_SIA_202605201500.txt`](../exp/log_rm_server_SIA_202605201500.txt) (5-20 RM_PROFILE 全 600 题，45 MB)
- 完整端到端对比数据：[`exp/log_SIA_vllmrm_202605252120.txt`](../exp/log_SIA_vllmrm_202605252120.txt) (G3)
- 实验全景与命令：[`exp/README.md`](../exp/README.md) 中「2026-05-20 15:00 — 开 RM_PROFILE」「2026-05-25 21:20 — G3」两节
- profiling 启动指南：[`doc/rm-profiling-guide.md`](rm-profiling-guide.md)
- 并行设计三个候选方案：[`doc/parallel-decoding-design.md`](parallel-decoding-design.md)（2B / 2C / 2D 详解）
- G2-G4 后续优化分析：[`doc/vllm-rm-followup-optimizations.md`](vllm-rm-followup-optimizations.md)
- 已训练好的 1.7B VM checkpoint：`Runyi-Hu/SIA/VM-Qwen3-1.7B-Base` (Hugging Face Hub)
