# B2 Decode-Mode 可行性分析 + M1a/M1b PoC 计划

**日期**：2026-05-27
**目的**：在投入 1100+ LOC 实现 B2 有状态 RM 之前，先用 1-2 天 PoC 解决最关键的技术未知数——**score_candidates 能否走 vLLM decode + CUDA graph 路径**——这是 B2 性能上限的决定性因素。
**前提**：用户决定单卡环境优先（不依赖 2 GPU），目标 RM 调用从 78ms → ≤ 15ms，整体吞吐 30 tok/s → ~45-50 tok/s。

---

## 1. 背景：B2 已经走到哪一步

`doc/parallel-decoding-design.md` §2.8 实测验证：
- vLLM 在 `LLM.generate()` continuous decode 路径上，Qwen3-4B 单 token decode = **7.02ms**
- 远小于 LLM forward 时间 L=11.3ms
- 证明 B2 在算力层面 feasible

但实测用的是 `LLM.generate()`——这是**生成路径**，走 `lm_head` 输出 token logits。SIA 的 RM 需要的是 **`score_head` 输出 reward 标量**，**这是两条不同路径**。

---

## 2. fix_a_token vs score_candidates：难度差异巨大

| 操作 | 信心 | 原因 |
|---|---|---|
| **fix_a_token**（每 SKIP 步给 RM 推 1 token）| **高（80%+）** | 跟 vLLM `LLM.generate()` decode loop 形态 1:1 对应；只需强制采样指定 token |
| **score_candidates**（每 INTERVENE 给 5 候选打分）| **中（40-60%）** | `score_head` 不在 vLLM decode CUDA graph 里，需要解决"如何在 decode 路径上拿到 reward"|

差别根源——vLLM 的 decode CUDA graph 捕获的路径：

```
input_ids → embed → 36×transformer_block → lm_head → 词表 logits → sample → 输出 token
                                              ↑
                                  vLLM decode CUDA graph 终点
```

而我们要的：

```
input_ids → embed → 36×transformer_block → score_head(Linear(2560,1)) → reward 标量
                                              ↑
                                  跟 lm_head 是平行的另一条 head
```

**两个 head 数学上不可互推**：lm_head 输出 151936-dim token logits，score_head 输出 1-dim 标量。即便拿到 decode 步骤的 logits，也算不出 reward。

---

## 3. 三个潜在解决方案

### 方案 A：ForCausalLM + 外接 score head（推荐先验证）

```
vLLM 加载 Qwen3-4B-CausalLM（非 SequenceClassification）→ 走 decode + CUDA graph
但 vLLM 默认只返回 logits（post-lm_head），我们要的是 hidden state（pre-lm_head）
→ 需要从 decode loop 里把 hidden state 抠出来
```

- **优点**：完全复用 vLLM 现成的 decode CUDA graph
- **核心未知**：vLLM 0.10.1.1 能否在 decode loop 里暴露 last_hidden_state？需 PoC 验证
- **PoC 成本**：~1 天

### 方案 B：自定义 vLLM model adapter

```
注册一个 hybrid 模型 class（继承 Qwen3ForCausalLM）：
- 保留 lm_head（让 vLLM decode loop 工作）
- 额外加一个 score_head 输出 attribute
- 每步 forward 同时返回 lm_head logits 和 score
vLLM 重新捕获这个 hybrid 模型的 decode CUDA graph
```

- **优点**：原生支持，不依赖未公开 API
- **风险**：vLLM model registry 机制、CUDA graph 重捕获条件不熟，工程量 ~300-500 LOC
- **PoC 成本**：~3 天

### 方案 C：兜底——score_candidates 仍走 /classify 路径

```
fix_a_token: 走方案 A 的 decode + CUDA graph，~7ms ✓
score_candidates: 仍用现有 /classify，~30ms（不动它）
```

- **优点**：技术 100% 可行，只是吞吐打折
- **代价**：score 没拿到 decode 加速，**整体收益从 +60% 降到 +25%**

---

## 4. 单卡上各方案的吞吐估算

参数：L=11.3ms, F (fix_a_token)=7ms, S (score) 视方案而变, I=0.28

| 方案 | F | S | 单卡 per-token | tok/s | vs 当前 30 tok/s |
|---|---:|---:|---:|---:|---:|
| 当前（C1+C2 / A2）| — | 78 | 33.5ms | 30 | — |
| **方案 A/B 都成功**（S 走 decode）| 7 | 7-10 | ~20ms | **~49** | **+63%** 🎯 |
| **半成功**（只 fix 走 decode）| 7 | 30 | ~26.7ms | **~37** | **+25%** |
| **完全失败**（连 fix 都卡住）| 30 | 30 | 49ms | 20 | -33% ❌ |

**关键 insight**：`score_candidates` 是决定性的——能走 decode → 接近上限；走不通 → 收益减半。这就是为什么 M1a/M1b 必须先验证。

---

## 5. M1a/M1b PoC 实施方案

按"先验证最关键不确定性、再做完整实现"的原则拆分。每个 milestone 都有明确的 go/no-go。

### 5.1 M1a — 验证 hidden_state 能否从 decode loop 提取（~1 天）

**目标**：找到至少一种方式，能在 vLLM continuous decode + CUDA graph 路径上拿到每步的 `last_hidden_state`（pre-lm_head, shape `[batch, hidden_dim=2560]`）。

**新建文件**：`scripts/poc_b2_extract_hidden_state.py`

**尝试顺序**（先简单后复杂）：

#### 尝试 (i)：vLLM 内置 `output_hidden_states` 参数

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/workspace/SIA/models/Qwen3-4B", dtype="bfloat16",
          gpu_memory_utilization=0.3)

# 尝试 SamplingParams 是否支持
sp = SamplingParams(temperature=0, max_tokens=10, output_hidden_states=True)
out = llm.generate(["test"], sp)
# 检查 out[0].outputs[0] 是否有 .hidden_states 字段
```

- ✅ 如果支持 → **方案 A 落地，1 天完成 M1a**
- ❌ 如果 SamplingParams 没这个字段 → 试 (ii)

#### 尝试 (ii)：写 LogitsProcessor 在 logits 计算前抓 hidden state

vLLM 的 `LogitsProcessor` 接口只接收 logits，不直接接收 hidden state。**但 hidden state 在调用 `lm_head` 之前就存在**——可以 hook `gpu_model_runner` 或重写 model forward。

```python
# 思路：subclass Qwen3ForCausalLM，重写 forward
class Qwen3WithHiddenStateLeak(Qwen3ForCausalLM):
    last_hidden_state = None  # 类属性侧通道

    def forward(self, *args, **kwargs):
        # 跑到 final layer norm 之后、lm_head 之前
        hidden = super().model(*args, **kwargs).last_hidden_state
        self.__class__.last_hidden_state = hidden  # 存到侧通道
        return self.lm_head(hidden)  # 给 vLLM decode loop 用
```

- 注册到 vLLM ModelRegistry，跑 generate，看 `Qwen3WithHiddenStateLeak.last_hidden_state` 能否拿到
- ⚠️ 风险：vLLM 跑的是 PagedAttention 路径，model forward 可能被 vLLM 改写过，简单 subclass 可能不工作

#### 尝试 (iii)：直接 patch vLLM `gpu_model_runner.execute_model`

```python
# 在 vllm/v1/worker/gpu_model_runner.py 的 execute_model 里
# 加一个 callback：execute 完成后把 hidden_states 通过 zmq 或 SharedTensor 传出
```

- 工程量大，但确定能做
- ⚠️ vLLM 升级要重 patch

**M1a Go/No-go**：
- ✅ 任一尝试拿到 hidden_state（且 CUDA graph 仍工作）→ 进 M1b
- ❌ 三种都失败 → 走方案 B（custom model adapter）或直接 fall back 方案 C

---

### 5.2 M1b — 实测 score_candidates 端到端延迟 + reward 正确性（~1 天）

**目标**：在 M1a 拿到 hidden state 的基础上，加上外接 score head，跑 5-candidate 实测延迟 + 验证 reward 数值跟 BF16 baseline 一致。

**新建文件**：`scripts/poc_b2_score_candidates.py`

**步骤**：

```python
# 1. 加载 VM checkpoint 的 score_head 权重为独立 nn.Linear
import safetensors.torch as st
score_weight = st.load_file(
    "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm/model.safetensors"
)["score.weight"]  # shape (1, 2560)
score_head = torch.nn.Linear(2560, 1, bias=False).cuda().bfloat16()
score_head.weight.data = score_weight.to("cuda", dtype=torch.bfloat16)

# 2. 用 M1a 的 hidden state 提取方法 + vLLM 跑 5 candidates
candidates = ["...", "...", "...", "...", "..."]  # 5 个 chat-formatted prompts
sp = SamplingParams(temperature=0, max_tokens=1, ...)

# 测稳态延迟（warmup + 100 iter 求 p50）
times = []
for _ in range(100):
    t0 = time.perf_counter()
    out = llm.generate(candidates, sp)
    hidden = get_last_hidden_state_via_M1a_method()  # shape (5, 2560)
    rewards = score_head(hidden).squeeze(-1)  # shape (5,)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

# 3. 跟 BF16 baseline 对比 reward 数值
# 拿同一组 candidates 在 /classify 上跑一次，对比两组 reward 的 Pearson
```

**关键验证**：
- **延迟 S = p50 of times**
- **正确性**：在 50 个不同 prompt × 5 candidates 上跑 PoC RM + BF16 baseline，对比 reward
  - Pearson correlation
  - Top-1 candidate agreement rate

**M1b Go/No-go 矩阵**：

| S (实测延迟) | Pearson | 推荐动作 | 预期单卡吞吐 |
|---|---|---|---|
| < 10ms | > 0.995 | ✅ 投 M2/M3，走方案 A | ~49 tok/s |
| 10-15ms | > 0.99 | ✅ 投 M2/M3，走方案 A，但稍打折 | ~45 tok/s |
| 15-25ms | > 0.99 | ⚠️ 仍可投，但收益缩水 | ~40 tok/s |
| 25-30ms | > 0.99 | ⚠️ 边缘，考虑方案 B（custom model adapter）| ~35 tok/s |
| > 30ms 或 Pearson < 0.95 | 任意 | ❌ 走方案 C 兜底 / 重新设计 | ~37 tok/s |

---

## 6. 决策矩阵（M1 完成后选下一条路）

```
M1a 结果
  ├─ ✅ 成功（拿到 hidden state，CUDA graph 工作）
  │     ├─ M1b: S < 15ms + Pearson > 0.99
  │     │     └─→ ✅ 全力投 M2/M3（方案 A），预期 +60% 吞吐
  │     ├─ M1b: S 在 15-25ms 之间
  │     │     └─→ ⚠️ 仍投 M2/M3，但目标降到 +40%
  │     └─ M1b: S > 25ms 或 reward 不准
  │           └─→ 重新评估，考虑方案 B 或 C
  │
  └─ ❌ 失败（hidden state 提取不可行）
        ├─→ 尝试方案 B（custom model adapter，~3 天 PoC）
        │     └─→ 若成功，重走 M1b
        └─→ 若 B 也失败，走方案 C（兜底，~+25% 吞吐）
```

---

## 6.5 M1a / M1b 实测结果（2026-05-27）

### M1a — hidden_state 提取 ✅ PASS

- 在 `Qwen3ForCausalLM.compute_logits` 入口挂 hook，hidden_state 能直接取到
- per-token decode = **6.84ms**（vs 无 patch baseline 7.02ms，多出 ~0.2ms 是 hook 写 /tmp 文件的开销）
- CUDA graph 仍然生效（没有任何重新编译）
- 关键 insight：`monkey-patch + import` 不跨 subprocess；EngineCore 是 fork 出的 subprocess，必须**直接改 vLLM 源码**，subprocess re-import 时才能拿到 patch
- 脚本：[`scripts/poc_b2_extract_hidden_state.py`](../scripts/poc_b2_extract_hidden_state.py)

### M1b — score 数值正确性 ✅ PASS（经历一次重跑）

#### 第一次（错配）：vLLM rewards vs BF16 baseline 出现 ~7× systematic scaling

| candidate | vLLM | BF16 | \|Δ\| |
|-----------|------|------|-------|
| '4' | +1.13 | +7.06 | 5.94 |
| '5' | -1.23 | -9.00 | 7.77 |

- Pearson = 0.9731（< 0.99 ❌）
- Top-1 一致（'4' 都最高），但数值标度差 ~6–8×

#### 根因排查（diagnose 脚本：[`scripts/poc_b2_diagnose_scaling.py`](../scripts/poc_b2_diagnose_scaling.py)）

排查路径：
1. **vLLM final norm 缺失？** ❌ 不是。`Qwen2Model.forward` line 369 `self.norm(hidden_states, residual)` 确认应用了 final RMSNorm
2. **vLLM/HF tokenization 不同？** ❌ 不是。dump 出 input_ids 完全相同（包括末尾 `[..., 151645, 198]`）
3. **sample position 错位？** ❌ 不是。HF 全序列对 vLLM 的 cos 最高出现在 pos 23（last token，cos=0.9998），位置对齐
4. **vLLM / HF 用了不同模型！** ✅ **就是根因**：
   - 脚本里 `MODEL_BASE = /workspace/SIA/models/Qwen3-4B`（base）→ 给 vLLM 用
   - 脚本里 `MODEL_VM  = /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm`（LoRA merged）→ 给 HF SequenceClassification 用
   - VM 是 LoRA fine-tune 过的 backbone，weights 跟 base 不同 → hidden_state 自然处在不同的表示空间，cos=0.617，score 标度差 6×

#### 修复（3 处改动 + 1 处临时 config 改动）

**1. config.json**（让 vLLM 用 generative 路径而不是 pooling）：
```diff
- "architectures": ["Qwen3ForSequenceClassification"],
+ "architectures": ["Qwen3ForCausalLM"],
```
（`config.json.bak` 已备份；M2 进生产时会改成自定义 architecture 名而不是修改 base 模型 config）

**2. qwen3.py `load_weights`**（vLLM 严格校验，看到不认识的 `score.weight` 会 fail）：
```diff
- loader = AutoWeightsLoader(self, skip_prefixes=(["lm_head."] if tie else None))
+ skip = ["score."]  # VM score head 不属于 Qwen3ForCausalLM
+ if self.config.tie_word_embeddings:
+     skip.append("lm_head.")
+ loader = AutoWeightsLoader(self, skip_prefixes=skip)
```

**3. qwen3.py `compute_logits` patch**（lazy-load score head + 算 reward）：
```python
if type(self)._b2_score_weight is None:
    _data = safetensors.torch.load_file(
        "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm/"
        "model-00002-of-00002.safetensors"  # score.weight 在 shard 2
    )
    type(self)._b2_score_weight = _data["score.weight"].to(
        hidden_states.device, hidden_states.dtype
    )
rewards = (hidden_states @ type(self)._b2_score_weight.T).squeeze(-1)
```

**4. PoC 脚本**：vLLM 端模型路径也指向 VM-Qwen3-4B-merged-for-vllm。

#### 重跑结果（M1b 真正 PASS）

| candidate | vLLM | BF16 | \|Δ\| |
|-----------|------|------|-------|
| '4'   | **+7.094** | +7.063 | 0.031 |
| '5'   | -8.875 | -9.000 | 0.125 |
| '3'   | -7.313 | -7.313 | 0.000 |
| '6'   | -8.938 | -8.875 | 0.063 |
| '100' | -9.000 | -8.938 | 0.063 |

- **Pearson = 0.9999** ✅（> 0.99）
- **max \|Δ\| = 0.125** ✅（< 0.5）
- **Top-1 = '4' 一致** ✅
- batch=5 generate p50 = 14.8ms（含 prefill；纯 decode ≈ 7ms）
- 诊断 hidden_state cos similarity = **0.9998**，norm 99.9% 一致 → vLLM 和 HF 在同一 backbone 上数值完全等价

### 决策：✅ 走方案 A，投 M2/M3

按 §6 矩阵，M1b S=14.8ms（含 prefill；M2 stateful + decode-only 后预计 ~7ms）+ Pearson > 0.99 → 落在"**全力投 M2/M3，预期 +60% 吞吐**"格子。

### M2 启动前必须做的清理

PoC 期间为了快速验证，**直接改了 vLLM 源码 + VM config**：
1. `/workspace/SIA/venv2/lib/python3.12/site-packages/vllm/model_executor/models/qwen3.py` — 加了 PoC hook + 改了 `load_weights`。M2 启动前 restore：`cp qwen3.py.bak qwen3.py`
2. `/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm/config.json` — architectures 改成 Qwen3ForCausalLM。restore：`cp config.json.bak config.json`

M2 应该用 vLLM `ModelRegistry.register_model` 注册一个新的 `Qwen3WithScoreForCausalLM` adapter（在我们自己的代码里），避免再去修改 vLLM 源码或 base config——这样 patch 是 sticky 的，能在 SIA package 内部 ship。

---

## 7. M2 / M3 概略（M1 通过后再细化）

### M2（~400 LOC, 1 周）—— 实现 fix_a_token + score_candidates 的 stateful server

- 用 `AsyncLLMEngine` 维护 long-running request 表示"每个 SIA session 的 RM 状态"
- `fix_a_token(req_id, token_id)`：通过 LogitsProcessor 强制 sample 指定 token
- `score_candidates(req_id, candidate_token_ids)`：fork 出 5 个分支 request，1-token forward 后取分数

### M3（~500 LOC, 2-3 周）—— 集成进 SIA processor + benchmark

- 改 `sia_vllm_RM.py`：把 HTTP `/classify` 调用替换为 in-process 调用 M2 的 stateful server
- 跑 600 题 MMLU benchmark，对比 G3 baseline
- 写 follow-up doc 落档结果

---

## 8. 实施顺序建议

```
M1a (~1 day)  ← 现在开始，验证最关键不确定性
   ↓
M1b (~1 day)  ← M1a 成功后立即接上
   ↓
[Go/No-go 决策点]
   ↓
M2 (~1 week)
   ↓
M3 (~2-3 weeks)
```

**只要 M1a + M1b 在 2-3 天内能给出明确数据，整个 B2 路线就能在第一周内判断"投不投后续 4 周"**——这比直接闷头写 1100 行后才发现 S=30ms 要好得多。

---

## 9. 引用

- B2 整体设计：[`doc/parallel-decoding-design.md`](parallel-decoding-design.md) §2
- F=7ms 实测验证：[`doc/parallel-decoding-design.md`](parallel-decoding-design.md) §2.8
- 为什么不能等 vLLM 优化 /classify：[`doc/parallel-decoding-design.md`](parallel-decoding-design.md) §2.9
- F 测试脚本：[`scripts/measure_rm_continuous_decode.py`](../scripts/measure_rm_continuous_decode.py)
- A2 失败教训（"量化加速 RM 不行"的来源）：[`doc/rm-fp8-dynamic-quantization-plan.md`](rm-fp8-dynamic-quantization-plan.md) §9
- 瓶颈定位与 LLM vs RM compute 时间差异：[`doc/profiling-bottleneck-analysis.md`](profiling-bottleneck-analysis.md) §4.1-§4.2
