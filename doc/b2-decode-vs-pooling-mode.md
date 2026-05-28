# Decode 模式 vs Pooling 模式：80 ms → 24 ms 的真正分布

**适用范围**：精确解释从 G1 (HTTP `/classify`) 到 b2 (in-process) 这一跃中，**RM call 从 ~80 ms 压到 ~24 ms** 的每一项节省到底来自哪里。重点澄清一个容易误解的点：**两条路径都开了 vLLM prefix caching，节省的不是"prefix 重算"，而是"非 cache 部分的 attention kernel 选择"**。

参考：
- HTTP `/classify` 路径实测：[`vllm-rm-experiment-report.md`](vllm-rm-experiment-report.md)
- b2 decode 模式可行性论证：[`b2-decode-mode-poc-plan.md`](b2-decode-mode-poc-plan.md)
- 整体收益叙事：[`b2-vs-classify-time-breakdown.md`](b2-vs-classify-time-breakdown.md)

---

## 1. 先澄清一个常见误解

**误解**：G1 的 5 个 candidates 是不是每个都从 0 重算整个 prefix？

**事实**：**不是**。G1 启动命令明确带了 `--enable-prefix-caching` (`vllm-rm-experiment-report.md` §3.1)：

```bash
vllm serve VM-Qwen3-4B-merged-for-vllm \
  --runner pooling --convert classify \
  --enable-prefix-caching   # ← 是开的
  ...
```

所以在 G1 上：

| Prefix caching 复用层面 | G1 是否生效 |
|------------------------|------------|
| 同一次 `/classify` 调用内的 5 个 prompt 共享 prefix KV | ✅ 生效 |
| 跨 `/classify` 调用复用 prefix（上次干预之前的部分）| ✅ 生效（block 没被驱逐时）|

**所以 G1 和 b2 在"prefix KV 复用"这件事上是一样的。**

那 RM call 从 ~80 ms 压到 ~24 ms 的 56 ms 节省到底来自哪？这就是本文要拆解的。

---

## 2. 80 ms → 24 ms 的真正分布

| 组成 | G1 (HTTP `/classify`) | b2 (in-process) | 节省 | 节省来源 |
|------|----------------------|------------------|------|----------|
| HTTP POST + JSON serialize/parse | ~10-15 ms | 0 ms | **~12 ms** | 去 HTTP 协议 |
| 客户端 chat template 拼字符串 + 服务端 tokenize | ~5-10 ms | 0 ms | **~8 ms** | token-level session |
| vLLM 调度/排队/同步 | ~5-10 ms | ~5 ms | **~5 ms** | 简化的 in-process IPC |
| **vLLM RM GPU forward（即使 prefix caching 命中）** | **~30-40 ms** | **~15 ms** | **🌟 ~20-25 ms** | **decode 模式 vs pooling/prefill 模式** |
| score head + Python 处理 | ~3-5 ms | ~3 ms | ~2 ms | 类似 |
| **总计** | **~80 ms** | **~24 ms** | **~56 ms** | |

总节省 56 ms 大致分摊：

| 节省来源 | ms | 占比 |
|---------|-----|------|
| 去 HTTP 协议 | ~12 | ~21% |
| 去 chat template + tokenize | ~8 | ~14% |
| 简化 scheduler / IPC | ~5 | ~9% |
| **decode 模式 vs pooling/prefill 模式** | **~25** | **~45%** |
| 其他 | ~6 | ~11% |

**decode 模式 vs pooling 模式是最大头（约 45%）**。这正是本文要深挖的部分。

---

## 3. Decode 模式 vs Pooling/Prefill 模式 — 内核层差异

### 3.1 vLLM 的两条独立任务管线

vLLM 0.10.x 同一份模型权重支持**两种完全不同的执行路径**：

| 路径 | task type | 用途 | 内部对应的 attention kernel |
|------|-----------|------|---------------------------|
| **Generative path** | `generate` | LLM 生成式推理 | FlashAttention（prefill）+ **FlashDecode/PagedAttention（decode）** |
| **Pooling path** | `pooling` (含 `classify` / `embed` / `score`) | 分类、嵌入、reward 模型一次性打分 | **只有 prefill 一次性 forward**，最后取 hidden state 做 pooling |

两条路径在 vLLM v1 内部走完全不同的 scheduler、KV cache 管理、attention kernel dispatch。

### 3.2 关键差异：非 cache 部分的 attention 怎么算

即使 prefix caching 让大部分 prefix KV 命中，**"剩下没在 cache 里的几个 token"怎么算 attention** 才是性能差距的根本：

| 模式 | 怎么算非 cache 的 token | 性能特性 |
|------|------------------------|---------|
| **G1 pooling/prefill kernel** | 走 FlashAttention varlen prefill kernel：一次性 forward 一整段 sequence。即使 KV 在 cache，新加入的 token 仍以 batch-position 维度并行计算 attention | 设计目标是**大 batch token 数（≥64）摊销 kernel launch**；对小 batch token 数（10-15）效率不优 |
| **b2 generative decode kernel** | 走 FlashDecoding / PagedAttention decode kernel：专门为"1 个新 Q 对完整 K/V cache"优化 | 设计目标是**单 token decode 极快**（vLLM continuous decode F=7ms 实测，见 `b2-decode-mode-poc-plan.md` §2.8）|

具体到 SIA 工作负载（5 candidates × 1-3 new candidate tokens × ~700 token prefix）：

| 阶段 | G1 (prefill kernel) | b2 (decode kernel) |
|------|---------------------|---------------------|
| 共享 prefix KV cache 命中 | ✅（两边都有）| ✅（两边都有）|
| **新 token 数** | 5 × ~3 = ~15 tokens | 5 × 1 = 5 tokens（每 candidate 1 token decode）|
| **Kernel 选择** | varlen prefill kernel | FlashDecode kernel |
| Kernel launch overhead | 每次 ~1-2 ms（prefill setup）| 每次 ~0.3-0.5 ms（decode setup）|
| Per-token attention 计算 | 中等效率（batch 维度并行不饱和）| 高效率（专为 1 token 优化的 memory access pattern）|
| **GPU forward 总耗时** | **~30-40 ms** | **~15 ms** |

**核心原因**：vLLM 的 prefill kernel 是给"长序列 chunked prefill"设计的，**摊销点在 ≥64 token / chunk**。SIA 每次 RM call 只增加 10-15 token，落在 prefill kernel 的"低效区"。而 decode kernel 专为"每次 1 token"优化，正好匹配 SIA 单 candidate 单 token 的需求。

`doc/b2-decode-mode-poc-plan.md` 整篇文档就在论证"想要 RM 跑到 7-15 ms，必须切到 decode 路径"——pooling 路径**不管怎么调参数都到不了这个数字**。

### 3.3 实测对比 — F=7ms 是 decode 独占

`b2-decode-mode-poc-plan.md` §2.8 实测：
- vLLM continuous decode 单 token (Qwen3-4B) = **7.02 ms**
- 这是 decode kernel 在单 token + 长 prefix（已 cached）下的硬下限
- pooling/classify 路径无法触发这个 kernel

`b2-m2-design.md` §7.A 实测：
- batch=5 prompts × 1 token decode + prefix caching = **15.2 ms p50**（pure bench，无 SIA wrap）
- 这是 b2 的实际 GPU forward 时间

**G1 在同样硬件上 GPU forward 是 30-40 ms**（实验报告反推），跟 b2 的 15 ms 差距就是 prefill kernel vs decode kernel 的 kernel 效率差。

---

## 4. 代码层是怎么切换这两个模式的

两条路径**用同一份模型权重** (`VM-Qwen3-4B-merged-for-vllm`)，区别在 vLLM 启动时**架构 dispatch 路径**：

### 4.1 G1 模式 — 走 pooling 路径

启动方式（[`vllm-rm-experiment-report.md`](vllm-rm-experiment-report.md) §3.1）：

```bash
vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --runner pooling --convert classify \
  --enable-prefix-caching ...
```

关键开关：
- `--runner pooling` —— 强制走 pooling task 管线
- `--convert classify` —— 配合模型 config 里的 `architectures: ["Qwen3ForSequenceClassification"]` 让 vLLM dispatch 到分类头

VM 模型的 `config.json` 里架构是：
```json
"architectures": ["Qwen3ForSequenceClassification"]
```

vLLM 看到这个 architecture + `pooling` runner 标志 → 用 **`Qwen2Model` backbone + `Linear` 分类头** + **pooling task 调度器**，所有 forward 走 prefill kernel。

### 4.2 b2 模式 — 走 generative 路径

启动方式（[`src/sia_rm/client.py`](../src/sia_rm/client.py) 内部）：

```python
self.llm = LLM(
    model=model_path,                                       # VM checkpoint 路径
    hf_overrides={"architectures": ["Qwen3WithScoreForCausalLM"]},  # 🌟 强制覆盖
    enable_prefix_caching=True,
    ...
)
```

关键开关：
- `hf_overrides={"architectures": ["Qwen3WithScoreForCausalLM"]}` —— **运行时把 architecture 改成 generative 子类**
- 不传 `--runner`（vLLM 看到 `ForCausalLM` 架构自动走 generative path）

`Qwen3WithScoreForCausalLM` 定义（[`src/sia_rm/qwen3_with_score.py`](../src/sia_rm/qwen3_with_score.py)）：

```python
class Qwen3WithScoreForCausalLM(Qwen3ForCausalLM):   # 🌟 继承 ForCausalLM
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hidden = vllm_config.model_config.hf_config.hidden_size
        self.score = nn.Linear(hidden, 1, bias=False)    # 外接 score head

    def compute_logits(self, hidden_states, sampling_metadata):
        # hidden_states 已经是 sample positions (last token of each prompt)
        rewards = (hidden_states @ self.score.weight.T).squeeze(-1)
        _write_rewards(rewards.detach().to("cpu", torch.float32))  # 写 /dev/shm
        return super().compute_logits(hidden_states, sampling_metadata)


ModelRegistry.register_model(
    "Qwen3WithScoreForCausalLM",
    "sia_rm.qwen3_with_score:Qwen3WithScoreForCausalLM",
)
```

vLLM 看到 architecture = `Qwen3WithScoreForCausalLM`（继承自 `Qwen3ForCausalLM`）→ 用 **`Qwen3Model` backbone + `lm_head`** + **generative task 调度器** → forward 走 prefill kernel（首次 prefill）+ **decode kernel（后续 1-token decode，prefix cache 命中时）**。

### 4.3 关键设计技巧

这个切换的**精妙之处**：

1. **同一份权重，零拷贝**：`hf_overrides` 在运行时改 architecture string，但 vLLM 加载的还是 VM 那一份 safetensors 权重（chat_prefix tokens + transformer layers + 7.5 GB 模型参数全部复用）
2. **score head 复用**：VM 的 `score.weight`（1×2560 Linear）被 vLLM 的 `AutoWeightsLoader` 自动 routes 到 `self.score`（因为 module 名字相同）
3. **decode 路径 + 自定义 score**：`compute_logits` hook 在 decode 路径的末尾插入，拿到的 `hidden_states` 已经是 sample positions（gather 过的 last token），直接 @ score head 算 reward
4. **plugin 入口**：vLLM EngineCore subprocess 启动时通过 `vllm.general_plugins` entry-point 自动 import `sia_rm.qwen3_with_score`，触发 `ModelRegistry.register_model`，subprocess 里也能 dispatch 到这个自定义 architecture（详见 [`pyproject.toml`](../pyproject.toml) + [`src/sia_rm/plugin.py`](../src/sia_rm/plugin.py)）

---

## 5. 为什么 decode 模式在 SIA 项目里特别快

总结 SIA workload 跟 decode kernel 的天然匹配：

| SIA 特性 | 适合 decode kernel 的原因 |
|---------|--------------------------|
| Prefix 长（700+ token），candidate 短（每个 1 token）| decode kernel 专为"长 KV cache + 1 个新 Q"优化 |
| 同一个 session 内 prefix 增量式增长 | prefix caching 跨调用命中率高，每次 RM call 只算最末几个 token |
| 5 candidates 并行打分 | decode kernel 支持 batch-of-1-token-per-request 高效并发（vLLM continuous batching）|
| 频繁但短的 RM call | decode kernel launch overhead 比 prefill 低 |

反过来，pooling 模式天然适合：长文档分类、embedding 一次性提取、长序列 reward 评分。SIA 的 per-token 干预**根本用不上 pooling 的优势**，反而被它的 prefill-only 路径拖累。

---

## 6. 最终归因表

| 节省项 | ms | 优化原理 | 在哪个文件实现 |
|--------|-----|---------|----------------|
| 去 HTTP 协议 | ~12 | SIA processor 和 RM 同进程，调用变成 Python 直接调；reward 通过 `/dev/shm` 文件 channel | [`src/sia_rm/qwen3_with_score.py`](../src/sia_rm/qwen3_with_score.py) `_write_rewards` |
| 去 chat template + tokenize | ~8 | token-level session：`fix_a_token` 直接 append `int` 到 prefix list，免去字符串拼接和 retokenize | [`src/sia_rm/client.py`](../src/sia_rm/client.py) `fix_a_token` / `score_candidates` |
| 简化 scheduler / IPC | ~5 | 嵌套 EngineCore 但仍比 HTTP 简单 | 见 `pyproject.toml` plugin entry |
| **decode kernel 取代 prefill kernel** | **~25** | **`hf_overrides` 把 `Qwen3ForSequenceClassification` 切换到 `Qwen3WithScoreForCausalLM`（继承 ForCausalLM），让 vLLM dispatch 到 generative path，5 candidates 走 1-token decode** | [`src/sia_rm/qwen3_with_score.py`](../src/sia_rm/qwen3_with_score.py) class 定义 + `ModelRegistry.register_model` |
| 其他 | ~6 | 较少的 scheduler/queueing overhead | — |
| **合计** | **~56** | | |

---

## 7. 引用

- G1 (HTTP /classify) 实验报告：[`vllm-rm-experiment-report.md`](vllm-rm-experiment-report.md)
- b2 decode 模式可行性论证（包含 F=7ms 实测、为什么 /classify 不能跑 decode）：[`b2-decode-mode-poc-plan.md`](b2-decode-mode-poc-plan.md)
- b2 M2 整体设计 + §7.A 15.2ms p50 实测：[`b2-m2-design.md`](b2-m2-design.md)
- 整体收益叙事（含 75.7 tok/s MMLU 结果）：[`b2-vs-classify-time-breakdown.md`](b2-vs-classify-time-breakdown.md)
- 后续可优化方向（Group A/B/C）：[`b2-pure-engineering-optimizations.md`](b2-pure-engineering-optimizations.md)
