# 0GM-35B + Qwen3-4B-RM 跨进程方案的成本分析与重回 inproc 的可行路径

**日期**: 2026-06-02
**适用范围**: 主 LLM = 0GM-1.0-35B-A3B-0427 (Qwen3.5/3.6 MoE 架构), RM = VM-Qwen3-4B-merged-for-vllm, vllm 版本 0.19.0

---

## 1. 当前架构与速度损失

### 1.1 拓扑
```
Main LLM (port 8000)  ─ vllm 0.19, FULL_AND_PIECEWISE cudagraph
        │
        │ HTTP POST /classify (5 candidates per INTERVENE)
        ▼
RM      (port 8001)  ─ vllm 0.19, pooling/classify, PIECEWISE cudagraph
```

### 1.2 速度
- noSIA baseline (35B 单跑 vllm serve): **97–108 tok/s**
- SIA 当前 (跨进程 + 0GM-friendly sampling, top_k=20/top_p=0.95/rep_penalty=1.0): **~68 tok/s**
- **slowdown ≈ 30%**

### 1.3 成本分解 (基于 SIA-pf-summary 实测)

| phase | p50 | 备注 |
|-------|-----|------|
| `http_post` (RM /classify call) | **94 ms** | 单次 INTERVENE 同步阻塞主 LLM |
| `format_chat` (字符串拼接) | 0.04 ms | 缓存命中, 几乎零 |
| `tokenize_client` | 0.00 ms | 走字符串发送, RM 服务端 tokenize |
| `parse_response` | 0.19 ms | JSON 解析 |
| `apply_total` (含 http_post) | 94.6 ms | 单次 INTERVENE 总耗时 |
| `skip_step` | 3.85 ms | 非 INTERVENE step 的 SIA overhead |

实际干预率: **3–13% per request** (entropy_threshold=1.0, 稀疏 junction 干预, 跟 SIA 论文一致)。
平均每 token SIA overhead ≈ 0.10 × 94 + 0.90 × 3.85 ≈ **12.9 ms/token**。

---

## 2. 跟之前 Qwen14B b2 inproc 方案的对比

| 维度 | Qwen14B + Qwen3-4B-RM (b2 inproc, vllm 0.10.1.1) | 0GM-35B + Qwen3-4B-RM (vllm HTTP, vllm 0.19.0) |
|------|---------------------------------------------------|------------------------------------------------|
| RM 调用方式 | inproc, persistent session, 增量 fix_a_token + 5 candidate forward | 跨进程 HTTP, 每次发完整 prefix×5 |
| 单次 RM 调用 (b2_score_call / http_post) | **~5 ms** | **~94 ms** (≈ 19×) |
| 主 LLM cudagraph | PIECEWISE (vllm 0.10, AOT capture, 无 runtime flag) | FULL_AND_PIECEWISE (vllm 0.19, runtime capture, 有全局 flag) |
| 主 LLM forward 速度 | 全速 | 全速 |
| SIA 干预成功率 | 100% (无 cudagraph 冲突) | 100% (跨进程隔离) |
| 实测 SIA slowdown vs noSIA | ~12% | ~30% |

**结论**: 跨进程路径让 per-INTERVENE 慢了 **~19×**, 但因为干预率本身 sparse (~10%), 反映到每 token overhead 上是从 ~2.5 ms 涨到 ~13 ms, 让 SIA 慢了约 18 个百分点。

---

## 3. 为什么 0GM-35B 必须用跨进程?

### 3.1 vllm 0.19 的全局 cudagraph capturing flag

- vllm 0.10.1.1 时代只有 PIECEWISE cudagraph, 是 **inductor 编译时 AOT capture**, runtime 不设全局 flag → nested vllm 在同一进程畅通
- vllm 0.13+ 引入 FULL cudagraph, 它在 **runtime 调用 `torch.cuda.graph()` context manager**, 这会拉起 PyTorch 的全局 `is_currently_capturing` flag
- flag 拉起期间, **任何**其他 CUDA 操作都会报错 `CUDA graph capturing detected at an inappropriate time. This operation is currently disabled.`
- vllm 0.19 的 PIECEWISE 实现也变了 — 不再是纯 AOT, 而是 **per-piece runtime capture** (当遇到 attention 内部 dynamic seq_len 等没见过的 shape 时 trigger), 同样会拉 flag

### 3.2 我们已经实测验证过的失败配置

| 配置 (主 LLM cudagraph mode, RM cudagraph mode, 同进程?) | 结果 | 现象 |
|---|---|---|
| FULL_AND_PIECEWISE, PIECEWISE, same | ❌ | RM error 10%+ step, 数据无效 |
| FULL_AND_PIECEWISE, eager, same | ❌ | 仍 RM error (主 LLM 的 flag 阻断 RM) |
| PIECEWISE-only, PIECEWISE, same | ❌ | 100% INTERVENE 失败 (intervened=0/150) |
| PIECEWISE-only, eager, same | ❌ | 仍 RM error (主 LLM PIECEWISE 也会 runtime capture) |
| 跨进程 (vllm serve RM) | ✅ | **当前方案**, 0 RM error |

**核心事实**: **冲突来自主 LLM 这一侧的全局 flag**, 不是 RM 端。RM 怎么配置都没用, 只要跟主 LLM 同一进程就中招。跨进程拥有独立 CUDA context, 是 vllm 0.19 下的唯一干净路径。

### 3.3 为什么 0GM-35B 不能降回 vllm 0.10?

- 0GM-1.0-35B-A3B 是 **Qwen3.5/3.6 MoE 架构** (`Qwen3_5MoeForConditionalGeneration`), vllm 0.10.1.1 / 0.15.0 都不支持该 architecture
- vllm 0.19 是目前能加载 0GM 的最旧版本
- 所以必须接受 vllm 0.19 的全局 flag 设计

---

## 4. 单纯换 RM 模型能否消除冲突?

**不能**。换 RM 只改变 RM 这一侧的成本/质量, 不能让 RM 不做 CUDA 操作 — 而冲突来源是主 LLM 端的全局 flag, **任何**做 CUDA forward 的 inproc RM 都会撞墙, 跟 RM 是什么模型无关。

不可行的"换 RM"方向:
- **更小的 vllm RM** (1B / 0.5B): 仍是 vllm + cudagraph + CUDA forward, 仍中招
- **RM 用 enforce_eager** (无 RM 端 cudagraph): 实测仍 100% INTERVENE 失败 — flag 来自主 LLM, RM 自己 eager 无用
- **CPU 跑 RM**: 4B 模型 CPU forward >100ms, 比当前 HTTP 还慢, 失去意义
- **量化 RM (FP8/INT8)**: 不改变是否 CUDA 操作, 同样中招

---

## 5. 真正可行的 inproc 回归路径: 把 RM 换成 transformers + DynamicCache

### 5.1 核心思想

**RM 实现层面换掉 vllm**, 改用 raw transformers + 手动维护 KV cache。
- transformers 默认**不用 cudagraph**, 也不设 PyTorch 全局 capturing flag
- transformers 的 `DynamicCache` 可以让多个 forward 共享同一份 KV cache 状态, 实现跟 b2 InprocClient 等价的 "1+4 batch 优化" (prefix forward 一次, 5 candidates 各 append 1 token)

### 5.2 跟当前 b2 InprocClient 的对比

当前 `src/sia_rm/client.py` 的 RMClient 是基于 vllm 0.19 LLM SDK 的:
```python
self._llm = LLM(model=model_path, ...)  # 启动一个 nested vllm engine
session_id = self._llm.new_session(prefix_tokens)
self._llm.fix_a_token(session_id, tok)
rewards = self._llm.score_candidates(session_id, candidate_ids)
```

它跟主 LLM 共用 PyTorch CUDA context, 撞 vllm 0.19 全局 flag。

替换方案 — transformers-based RMClient:
```python
from transformers import AutoModelForCausalLM, DynamicCache
self._model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
self._score_head = ...  # 加载 token_reward_head

# new_session: 在 prefix 上 forward 一次, 存 KV cache
def new_session(self, prefix_tokens):
    cache = DynamicCache()
    with torch.inference_mode():
        self._model(input_ids=prefix_tokens, past_key_values=cache, use_cache=True)
    return cache  # 这就是 session_id (其实是 cache obj 本身)

# fix_a_token: 在已有 cache 上 forward 1 token
def fix_a_token(self, cache, tok):
    with torch.inference_mode():
        self._model(input_ids=tok, past_key_values=cache, use_cache=True)

# score_candidates: 复制 cache 5 份, 每份 forward 1 个 candidate, 取最后 hidden state -> score head
def score_candidates(self, cache, candidates):
    # 关键: 复用 cache, 只对每个 candidate forward 1 个 token, batch=5
    expanded_cache = self._expand_cache(cache, n=5)
    out = self._model(input_ids=candidates.unsqueeze(-1), past_key_values=expanded_cache, ...)
    scores = self._score_head(out.last_hidden_state[:, -1])
    return scores
```

### 5.3 预期收益 — 重新评估 (悲观 + 乐观区间)

| 场景 | per-INTERVENE | per-session-advance | per-token avg SIA overhead | SIA throughput |
|------|---------------|---------------------|---------------------------|----------------|
| **当前 vllm HTTP** | 94 ms | 0 (HTTP 内部 prefix cache) | 12.9 ms/tok | **68 tok/s** ✅ 实测 |
| transformers 乐观 (FlashAttn2 + 16ms/tok forward) | 25 ms (5-way) | 16 ms (10-tok 跳跃) | ≈ **7.5 ms/tok** | **~85 tok/s** |
| transformers 悲观 (无 FlashAttn2 + 28ms/tok forward) | 35 ms | 25 ms | ≈ **9.5 ms/tok** | **~78 tok/s** |
| transformers 灾难 (cudagraph 仍冲突) | N/A | N/A | N/A | 退回当前 HTTP |

**实际收益区间: +10 ~ +25 tok/s (+15% ~ +25%)**, 不是质变。

### 5.4 关键假设的可靠性 — 诚实审视

| 假设 | 实际可靠性 |
|------|------------|
| transformers 4B forward ~15-25 ms | ⚠️ **偏乐观**。无 cudagraph 单 batch decode 实测 15-25 ms, batch=5 估 20-30 ms。是否启用 FlashAttention 2 是关键变量 |
| session 推进可以 lazy 做, 累积一次性 forward | ✅ 合理 (attention 在 seq 维 sub-linear) |
| 共享 prefix + 5 路 branching 实现复杂度 "中等" | ⚠️ **偏乐观**。transformers `DynamicCache` 标准 API **不直接支持** shared-prefix batched forward。朴素做法是 cache 复制 5 份 (4B/600 tok ≈ 2.2 GB 显存); 优化做法需手动 hack attention 内部 |
| 不会撞 cudagraph 全局 flag | ⚠️ **未实测**。transformers 默认不调 `torch.cuda.graph()`, 但主 LLM 在 inference 期间 spontaneously 触发 cudagraph capture (e.g., 没见过的 batch size) 时, **同进程 transformers 的 CUDA op 仍会被全局 flag 阻断**。这是方案 A 唯一的"致命点", 必须先验证 |
| 实现 1-2 天可完成 | ⚠️ **偏乐观**。实际估 **3-5 天** (含 shared-prefix 实现 + cudagraph 兼容性验证 + 调试) |

### 5.5 推荐分两步走 — 先低成本验证关键假设

**Step 1 (1 小时, 接近零成本)**: 写 minimal 验证脚本
- 同进程加载主 LLM (0GM-35B, vllm 0.19, FULL_AND_PIECEWISE) + transformers Qwen3-4B
- 跑主 LLM forward 期间, 并发跑 transformers Qwen3-4B forward
- 观察是否报 `CUDA graph capturing detected at an inappropriate time`
- **这一步决定性回答方案 A 的 cudagraph 假设是否成立**, 不通过则方案 A 直接报废

**Step 2 (条件: Step 1 通过, 半天)**: benchmark 真实 forward 时延
- 单独跑 transformers Qwen3-4B 1-token decode + 5-batch + KV cache 命中
- 实测时延 ≤ 25 ms → 继续完整实现
- 实测 ≥ 35 ms → 收益太薄, 放弃

**Step 3 (条件: Step 2 通过, 2-3 天)**: 完整实现 + 600q eval 验证

### 5.6 替代方案的比较 (供选择)

| 方案 | 单 RM 调用 | 主 LLM 速度 | 实现成本 | 预期 SIA throughput |
|------|-----------|-----------|---------|---------------------|
| **当前** vllm 跨进程 HTTP | 94 ms | 100% | 0 (已工作) | **68 tok/s** ✅ 实测 |
| **方案 A** transformers + DynamicCache inproc | 25-35 ms (含 session advance) | 100% (理论上) | 中等 (实测后估 3-5 天) | ~78-85 tok/s, **未验证** |
| **方案 B** 主 LLM enforce_eager + b2 inproc | ~5 ms | ~60% (35B eager 严重慢) | 0 (已支持 `SIA_LLM_CUDAGRAPH=none`) | 估 ~40-50 tok/s, **更差** |
| **方案 C** 自训练超小 VM (e.g., 学习一个 linear head 接在主 LLM 隐状态上) | <1 ms | 100% | 大 (需要重新训练) | 估 ~95 tok/s, 接近 noSIA, **是 ML 工作** |

### 5.7 实事求是的判断

- 方案 A **理论可行**, 但 doc 早期版本写的 1-2 天 / 85-90 tok/s 都偏乐观, 无强证据
- 最大不确定性: **shared-prefix 5-way 实现** (transformers 标准 API 不直接支持) + **inproc transformers vs 主 LLM cudagraph 全局 flag 兼容性** (未实测)
- 收益是 +15-25% throughput, **不是质变**
- **建议先把当前 600q SIA eval 跑完**, 验证 SIA 在 0GM 上的 alignment 效果 (vs noSIA accuracy) — 如果效果不显著, 优化速度无意义

---

## 6. 当前的修复路径有哪些副作用?

在做"sampling param 修复"时, 我们已经:
1. `src/sia_vllm_server.py`: 在 `ChatCompletionRequest` 加 `top_k`/`repetition_penalty` 字段, `_build_sampling_params` 支持透传 (默认值保持 `top_k=-1`, `repetition_penalty=1.3` 不变)
2. `eval/mmlu_eval.py`: 加 `--top_k`/`--top_p`/`--repetition_penalty` CLI flag

**14B + Qwen3-4B-RM 之前的 baseline 不受影响** — 不传新参数时, server 走默认值 = 跟原来一致。

**0GM-35B 评测必须传**: `--top_k 20 --top_p 0.95 --repetition_penalty 1.0` (跟 0GM `generation_config.json` 一致), 否则 248K vocab + repetition_penalty=1.3 会让 sampler 漂到多语言 OOV rare token, 输出乱码。

---

## 7. 行动建议 (按优先级)

**立即可行 (零成本, 必须先做)**:
- 让当前 600q SIA eval 跑完, 拿到 0GM 上的 **SIA vs noSIA 准确率对比** — 如果 SIA 效果不显著, 后续所有速度优化都没意义

**短期低成本探针 (条件: SIA 效果验证 OK; 1 小时)**:
- 实现 Step 1 验证脚本: 同进程加载 vllm 0.19 主 LLM + transformers Qwen3-4B, 验证后者是否被前者的 cudagraph 全局 flag 阻断
- 这是方案 A 的"致命点"实验, 不通过则方案 A 直接报废

**中期 (条件: Step 1 通过; 3-5 天)**:
- benchmark transformers 真实 forward 时延 (Step 2)
- 实现 transformers + DynamicCache 版的 RMClient backend (Step 3)
- 跟 Qwen14B baseline (~12% slowdown) 对齐验证

**长期 (1+ 周)**:
- 方案 C: 探索学习一个超小 value head, 完全替代 RM forward — 但这是 ML 工作, 不是工程优化
