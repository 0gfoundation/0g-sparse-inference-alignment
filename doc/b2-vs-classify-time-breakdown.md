# 30 → 45 tok/s 是怎么来的：b2 vs HTTP /classify 端到端耗时拆解

**适用范围**：解释从 HTTP `/classify` 后端 (G1) 切换到 b2 in-process 后端后，单卡 SIA throughput 从 ~30 tok/s 提升到 ~45 tok/s 的具体来源。

**配置**（两条路径完全对齐）：
- LLM: `Qwen3-14B`
- RM: `VM-Qwen3-4B-merged-for-vllm`（同一个 checkpoint）
- `topk=5`, `weight=1.0`, `entropy_threshold=1.0`
- 单卡 H200 / 同一台机器
- 评测集: MMLU-Redux 600 题

---

## 1. 优化前 — HTTP `/classify` + batch RM forward (G1)

来源：[`vllm-rm-experiment-report.md`](vllm-rm-experiment-report.md) §1, §5；[`vllm-rm-followup-optimizations.md`](vllm-rm-followup-optimizations.md) §86

**整体性能**:
- Throughput = **33.5 tok/s** ≈ **30 ms/token**
- 干预率 = 27.3%

**每 token 时间拆解（加权平均）**:

```
per_token = (1 − 0.273) × 11.3 ms (SKIP)
          +     0.273   × 91.0 ms (INTERVENE)
          ≈ 33.06 ms/token  → 30 tok/s
```

**INTERVENE step (~91 ms) 内部细分**:

| 阶段 | 耗时 | 说明 |
|------|------|------|
| LLM forward | 11.3 ms | Qwen3-14B 单 token decode |
| **RM call 总计** | **~80 ms** | 端到端反推 |
| ├─ chat template + client tokenize | ~10 ms | SIA 客户端拼 prompt 字符串 + RM server tokenize |
| ├─ HTTP POST + JSON serialize/parse | ~15 ms | 客户端 ↔ RM server |
| ├─ vLLM `/classify` GPU forward | ~40 ms | **走 prefill kernel**（即使 `--enable-prefix-caching` 命中 prefix KV，剩下的 candidate token 仍走 prefill kernel；详见 §4(a)）|
| └─ vLLM 调度/排队/同步 overhead | ~15 ms | TCP, scheduler 等 |

**关键瓶颈**: vLLM `/classify` 是 **pooling 任务**，走 `--runner pooling` 管线。即使 prefix caching 命中（G1 启动确实带了 `--enable-prefix-caching`），**非 cache 部分的 token 走 prefill kernel**——这个 kernel 设计来处理大 token batch（≥64），SIA 每次 RM call 只 forward 10-15 个新 token 落在它的低效区。

---

## 2. 优化后 — b2 in-process + stateful session + prefix caching

来源：[`b2-m2-design.md`](b2-m2-design.md) §6.5/§7.X；[`b2-pure-engineering-optimizations.md`](b2-pure-engineering-optimizations.md) §1；600 题 MMLU 实测

**整体性能**:
- Throughput = **45.3-46.6 tok/s** ≈ **22 ms/token**（两次 600 题平均 45.95）
- 干预率 = 32.0-32.6%

**每 token 时间拆解（加权平均）**:

```
per_token = (1 − 0.326) × 12.3 ms (SKIP)
          +     0.326   × 44.4 ms (INTERVENE)
          ≈ 22.76 ms/token  → 44 tok/s
```

**INTERVENE step (~44 ms) 内部细分**:

| 阶段 | 耗时 | 说明 |
|------|------|------|
| LLM forward | 12.3 ms | Qwen3-14B 单 token decode（跟优化前几乎一样）|
| **RM call 总计** | **~32 ms** | b2 path 端到端 |
| ├─ `b2_session_init` | ~0 ms | 仅第一步非 0，之后从 session 表里取 |
| ├─ `b2_prefix_adv`（fix_a_token chain）| ~0 ms | Python list append, 不调 vLLM |
| ├─ `b2_score_call`（核心）| ~20 ms | **嵌套 RM EngineCore + decode kernel + prefix caching** |
| │   ├─ pure vLLM forward | ~15 ms | 走 decode kernel（FlashDecode / PagedAttention）— 5 candidates 各跑 1-token decode，prefix caching 命中 |
| │   └─ 嵌套 EngineCore IPC | ~5 ms | zmq + /dev/shm reward channel |
| └─ SIA Python wrap | ~8 ms | logits modify + tensor copy + scheduler 等 |

---

## 3. side-by-side 对比 — 每一项被压缩了多少

| 阶段 | 优化前 (G1) | 优化后 (b2) | Δ | 原理 |
|------|-------------|-------------|---|------|
| LLM forward | 11.3 | 12.3 | **0** | LLM 路径不变 |
| chat template + tokenize | ~10 | 0 | **−10** | **token-level session**：不再每次拼字符串 + tokenize，prefix 直接是 `list[int]`，`fix_a_token` 时只 append |
| HTTP / JSON 协议 | ~15 | 0 | **−15** | **去 HTTP 协议**：RM 跑在嵌套 EngineCore subprocess 内，跟 SIA processor 同进程；用 `/dev/shm` 文件 channel 取代 HTTP request/response |
| vLLM scheduler / 排队 overhead | ~15 | ~5 | **−10** | 简化的 in-process IPC 比 HTTP server scheduler 轻 |
| **RM GPU forward**（含 prefix caching）| **~40** | **~15** | **🌟 −25** | **decode kernel 取代 prefill kernel**：两条路径都开 prefix caching，但 G1 走 pooling/prefill 路径，非 cache 部分仍用 prefill kernel；b2 走 generative 路径，非 cache 部分用 decode kernel（更适合"5 candidates × 1 token"的小 batch shape） |
| SIA Python wrap | 0 | 8 | +8 | 新增：dict lookup、logits modify、tensor copy 等 SIA processor 逻辑 |
| 嵌套 EngineCore IPC | 0 | 5 | +5 | 新增：嵌套结构带来的 zmq + /dev/shm reward channel 同步 |
| **INTERVENE step 合计** | **~91 ms** | **~44 ms** | **−47 ms (−52%)** | Δ 列纵向相加：−10 −15 −10 −25 +8 +5 = **−47 ✓** |

---

## 4. 三项核心原理

### 🌟 (a) decode kernel 取代 prefill kernel（省 ~25 ms，最大头）— **质变**

**澄清常见误解**：G1 启动确实带了 `--enable-prefix-caching`（[`vllm-rm-experiment-report.md`](vllm-rm-experiment-report.md) §3.1），prefix KV cache **在两条路径都生效**：
- 同一次 RM call 内的 5 个 candidate 共享 prefix KV ✓ G1 和 b2 都有
- 跨 RM call 复用 prefix（上次干预之前的 token）✓ G1 和 b2 都有

**真正的差距在于"非 cache 部分的 attention 怎么算"**：

| 模式 | 非 cache token 的 kernel | 性能特性 |
|------|--------------------------|---------|
| G1 pooling/prefill kernel | varlen prefill kernel | 设计来摊销 ≥64 token batch；SIA 每次 10-15 个新 token 落在低效区 |
| b2 generative decode kernel | FlashDecode / PagedAttention | 专为 "1 个新 Q vs 完整 cached K/V" 优化（vLLM continuous decode F=7ms 实测）|

具体到 SIA 的 5 candidates × 1-3 candidate tokens：
- G1 走 prefill kernel，~40 ms GPU forward
- b2 走 decode kernel，~15 ms GPU forward（**~25 ms 节省**）

代码层切换原理 + 完整内核对比详见 [`b2-decode-vs-pooling-mode.md`](b2-decode-vs-pooling-mode.md)。

### (b) 去 HTTP 协议（省 ~15 ms）— **量变**

- SIA processor 跟 RM 同进程，调用变成 Python 直接调；reward 通过 `/dev/shm` 文件 channel 跨嵌套 EngineCore subprocess 传递
- 嵌套 EngineCore 仍有 ~5 ms IPC，但总体比 HTTP/JSON 轻

实现：[`src/sia_rm/`](../src/sia_rm/) + [`pyproject.toml`](../pyproject.toml) 的 `vllm.general_plugins` entry-point

### (c) token-level session（省 ~10 ms）— **设计变更**

- G1 每次 RM 调用前要 SIA 客户端跑 `apply_chat_template` 拼字符串 + RM server 端 `tokenize`
- b2 直接 `fix_a_token(int)`，prefix list 不断生长；score 时构造 `prefix + [c]` 直接发 token ids
- 相当于把 string 拼接和 retokenize 全跳过

实现：[`src/sia_rm/client.py::RMClient`](../src/sia_rm/client.py) `new_session` / `fix_a_token` / `score_candidates`

---

## 5. 最大杠杆是哪一项

按节省量排序（净节省 47 ms = 91 ms − 44 ms）：

| # | 原理 | 节省 / INTERVENE | 占净节省比例 |
|---|------|------------------|--------------|
| 🥇 | **decode kernel 取代 prefill kernel** | ~25 ms | **~53%** |
| 🥈 | 去 HTTP 协议 | ~15 ms | ~32% |
| 🥉 | token-level session | ~10 ms | ~21% |
| 4 | 简化 scheduler 排队 | ~10 ms | ~21% |
| — | （减项）SIA Python wrap 新开销 | +8 ms | — |
| — | （减项）嵌套 EngineCore IPC 新开销 | +5 ms | — |

(节省合计 60 ms − 新增开销 13 ms = 净节省 47 ms ≈ 91 − 44 ✓)

**核心 takeaway**: 30 → 45 tok/s 的飞跃中，**decode kernel vs prefill kernel 是单项最大头（约 45%）**。这也是为什么 b2 的设计目标第一条就是 "让 RM 跑在 decode 模式" 而不是 pooling/prefill 模式 — [`b2-decode-mode-poc-plan.md`](b2-decode-mode-poc-plan.md) 整篇文档就在论证这一点（M1a PoC 验证 hidden_state 可从 decode loop 提取；M1b PoC 验证 score 数值跟 BF16 baseline 等价）。

**Prefix caching 不是 b2 独占的优化**：G1 和 b2 都开了，但只有 b2 能配合 decode kernel 把"5 candidates × 1 token decode"跑到 ~15 ms。pooling 路径即使开了 prefix caching，剩下的 candidate token 仍走 prefill kernel，~30-40 ms。

---

## 6. 引用

- **decode vs pooling/prefill kernel 深度对比 + 代码层模式切换**：[`b2-decode-vs-pooling-mode.md`](b2-decode-vs-pooling-mode.md)
- HTTP `/classify` G1 实测数据：[`vllm-rm-experiment-report.md`](vllm-rm-experiment-report.md)
- HTTP G1 → G2/G3/G4 优化为何没继续提升：[`vllm-rm-followup-optimizations.md`](vllm-rm-followup-optimizations.md)
- b2 decode 模式可行性论证 + F=7ms 实测：[`b2-decode-mode-poc-plan.md`](b2-decode-mode-poc-plan.md)
- b2 M2 设计 + 实测：[`b2-m2-design.md`](b2-m2-design.md)
- b2 后续可优化空间（Group A/B/C）：[`b2-pure-engineering-optimizations.md`](b2-pure-engineering-optimizations.md)
- 最早 PyTorch RM 时代基线：[`performance-report.md`](performance-report.md)
- SIA vs noSIA accuracy regression 验证：[`eval-report.md`](eval-report.md)
