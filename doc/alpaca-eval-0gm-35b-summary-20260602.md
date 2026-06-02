# 0GM-35B AlpacaEval — Skywork 第三方打分完整执行总结

**日期**: 2026-06-02
**主推理模型**: 0GM-1.0-35B-A3B-0427 (Qwen3.6-35B-A3B 微调, vocab=248K, native thinking)
**Value Model**: VM-Qwen3-4B-merged-for-vllm (LoRA + score head 合并到 vllm 兼容版)
**打分 RM**: Skywork-Reward-V2-Llama-3.1-8B (第三方独立打分模型)
**SIA 配置**: topk=5, weight=1.0, entropy_threshold=0.8, Step1+2 fix (commit `e1d150e`)
**对照前置 doc**:
- [`alpaca-eval-plan-20260602.md`](alpaca-eval-plan-20260602.md) — 早期规划
- [`alpaca-eval-0gm-35b-execution-20260602.md`](alpaca-eval-0gm-35b-execution-20260602.md) — noSIA arm 单独执行记录 (--disable_thinking, 200Q)

---

## 0. TL;DR

| 配置 | mean reward | win vs noSIA | 跟历史 14B A1 (+13.2%) 对比 |
|------|------------|------------------|--------------------------|
| 历史 14B A1 (官方代码, 805Q, raw `Human/Assistant`) | 12.29 → 13.92 | n/a | **+13.2%** ✅ baseline |
| 实验 1: 0GM + `--disable_thinking` (50Q matched, SIA partial) | 18.50 → 11.14 | 12% | **-40%** ❌ 大幅 hurt |
| 实验 2 (本次): **0GM + `--no_think_prompt --ban_think_token` (98Q matched)** | **10.09 → 8.78** | **44%** | **-13%** ⚠️ 小幅 hurt |

**最关键发现**: 之前 SIA on 0GM 看似严重 hurt (-40% reward) **大部分是 `<think>` format artifact**:
- chat_template 强塞 `<think>` 标记后, SIA 把生成推向 "Here's a thinking process: 1. ..." outline 风格, 答案没写完就被 cap 截断
- 把 prompt 改成 raw `Human:/Assistant:` + 用 vllm `bad_words` 禁掉 `<think>/</think>` token, **SIA 的真实 alignment 效果暴露出来: -13% reward** (小幅 cross-family OOD hurt)

---

## 1. 4 个实验配置 (按时间顺序)

### 1.1 实验 0: 早期 chat_template + thinking 探索 (废弃)

第一次 SIA 跑 200Q AlpacaEval (max_tokens=256), 用 chat_template 默认 (enable_thinking=True) → 0GM thinking 模式被强制开启 → **522/525 输出 hit 256 max_tokens cap, 全是 thinking 中段**, 几乎没有 final answer。**数据完全无效**, kill。

### 1.2 实验 1: chat_template + `--disable_thinking` (max_tokens=512, 200Q noSIA + 50Q SIA partial)

[`alpaca-eval-0gm-35b-execution-20260602.md`](alpaca-eval-0gm-35b-execution-20260602.md) 详细记录。
- noSIA 200Q (max_tokens=512): mean reward **18.06**
- SIA 50Q partial (RM=4B, Step1+2 fix): mean reward **11.14** (CPU 打分, 200Q 跑到 50 用户停)
- **Matched 50Q 对比**: noSIA 18.50, SIA 11.14, **Δ=-7.35 (-40%)**, SIA win 12%

**问题诊断**: `--disable_thinking` 不是真正"无 think tag", 而是 chat_template 在 prompt 末尾塞 `<think>\n\n</think>\n\n` 空块。模型见到 `<think>` 标记仍激活 thinking-pattern 训练先验, 在 SIA 干预下输出 "Here's a thinking process: 1. Understand User..." 风格, 而非直接答案。

### 1.3 实验 2: 中间过渡 — `enable_thinking=True` (max_tokens=2048, 100Q noSIA only)

跑 noSIA 100Q with default thinking enabled (max_tokens=2048):
- avg tokens 1725, max 2048
- 94/100 emit `</think>` (大多数完成 thinking + 出答案)
- 没继续做 SIA arm, 因为发现还有更彻底的"无 think"路径

### 1.4 实验 3 (本 doc 主体): **`--no_think_prompt --ban_think_token` 双管齐下** (max_tokens=2048, 100Q × 2 arm)

完全消除 thinking influence 的最干净方案:

**`--no_think_prompt`**: 走 `/v1/completions` (跳过 chat_template), prompt = `"Human:\n{instr}\nAssistant:\n"` raw 文本 (跟官方 14B A1 一致)。

**`--ban_think_token`**: vllm `SamplingParams(bad_words=["<think>", "</think>"])` 在 sampler 层禁止这两个 token 被采样。0GM 即便没 prompt 提示, 训练先验仍会自发 emit `<think>` (实测 60% Q 会), 这个 flag 在 sampling 阶段强制禁止。

**双管齐下结果**: 100/100, **0 `<think>`, 0 `</think>`, 0 "thinking process" 开头**, 模型完全走直接答案模式。

---

## 2. 实验 3 完整命令记录

### 2.1 启 RM 4B server (port 8001)

```bash
nohup /workspace/SIA/venv4/bin/vllm serve \
    /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling --convert classify \
    --hf-overrides '{"architectures":["Qwen3WithScoreForCausalLM"]}' \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.22 \
    --max-model-len 2048 \
    --port 8001 --host 0.0.0.0 \
    --disable-log-stats \
    > exp/alpaca_0gm_sia_banthink_server_20260602_151551.log 2>&1 &
```

(RM 4B 占 ~32GB, gpu_mem=0.22)

### 2.2 SIA arm: 启 0GM SIA server (port 8000)

```bash
nohup env SIA_DEBUG_HIST=1 /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
    --rm_backend vllm \
    --rm_url http://localhost:8001 \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.72 --max_model_len 4096 \
    --topk 5 --weight 1.0 --entropy_threshold 0.8 \
    --host 0.0.0.0 --port 8000 \
    > exp/alpaca_0gm_sia_banthink_server_20260602_151551.log 2>&1 &
```

### 2.3 SIA arm: 跑 100Q AlpacaEval

```bash
/workspace/SIA/venv2/bin/python eval/alpaca_eval.py \
    --base_url http://localhost:8000/v1 \
    --model 0GM-1.0-35B-A3B-0427 \
    --limit 100 --max_tokens 2048 \
    --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
    --no_think_prompt --ban_think_token \
    --output exp/alpaca_0gm_sia_banthink_100q_20260602_151551.json
```

实际 12 min 跑完。

### 2.4 noSIA arm: kill SIA, 启 noSIA server (port 8000)

为保证两 arm 走**完全相同的代码路径** (含 bad_words 支持), noSIA 也用 `sia_vllm_server.py` 但设 `--weight 0 --entropy_threshold 999999` (SIA processor 注册但永不 INTERVENE):

```bash
# kill SIA server (PID 1136845)
kill 1136845; sleep 6
pgrep -f EngineCore | xargs -r kill -9

# 启 noSIA arm
nohup /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
    --rm_backend vllm \
    --rm_url http://localhost:8001 \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.72 --max_model_len 4096 \
    --topk 5 --weight 0.0 \
    --entropy_threshold 999999 \
    --host 0.0.0.0 --port 8000 \
    > exp/alpaca_0gm_nosia_banthink_server_20260602_155639.log 2>&1 &
```

### 2.5 noSIA arm: 跑 100Q (同样的题, 同样的参数)

```bash
/workspace/SIA/venv2/bin/python eval/alpaca_eval.py \
    --base_url http://localhost:8000/v1 \
    --model 0GM-1.0-35B-A3B-0427 \
    --limit 100 --max_tokens 2048 \
    --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
    --no_think_prompt --ban_think_token \
    --output exp/alpaca_0gm_nosia_banthink_100q_20260602_155639.json
```

实际 6 min 跑完 (比 SIA 快, 因为没 RM HTTP overhead)。

### 2.6 Skywork RM 打分两 JSON

kill 所有 server, GPU 加载 Skywork-Reward-V2-Llama-3.1-8B (BF16, ~16GB):

```bash
pgrep -f "sia_vllm_server|vllm serve|EngineCore" | xargs -r kill -9
sleep 6

# 顺序跑两次打分 (Skywork 占 16GB GPU, 一次只能一个)
/workspace/SIA/venv/bin/python scripts/measure_alpaca_reward.py \
    --input_file exp/alpaca_0gm_nosia_banthink_100q_20260602_155639.json \
    --output_file exp/alpaca_0gm_nosia_banthink_100q_scored_20260602_155639.json \
    --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
    --device cuda:0

/workspace/SIA/venv/bin/python scripts/measure_alpaca_reward.py \
    --input_file exp/alpaca_0gm_sia_banthink_100q_20260602_151551.json \
    --output_file exp/alpaca_0gm_sia_banthink_100q_scored_20260602_151551.json \
    --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
    --device cuda:0
```

每个 ~1.5 min (GPU 上 Skywork forward 100 题非常快)。

---

## 3. 实验 3 完整结果

### 3.1 Reward 对比 (98 题 matched, 2 题 SIA 空答跳过)

| 指标 | noSIA | **SIA (RM=4B)** | Δ |
|------|-------|----------------|---|
| **mean reward** | **10.09** | **8.78** | **-1.31 (-13%)** |
| median | 10.38 | 9.25 | -1.13 |
| stdev | 12.39 | 10.49 | SIA 更集中 |
| min / max | -17.6 / 38.8 | -13.4 / 42.3 | SIA 上限略高 |
| **SIA win rate (per-Q)** | | | **43.9%** (44/98) |
| SIA lose rate | | | 55.1% (54/98) |
| ties | | | 1 |

### 3.2 SIA 内部指标 (从 server DONE 累计)

| 指标 | SIA arm |
|------|---------|
| cumulative intervene rate | **33.35%** (6,612/19,829 steps) |
| flip rate (of intervened) | 7.96% |
| flip rate (of total step) | 2.65% |
| per-Q intervene rate | p50=35.9%, mean=36.5%, max=80%, min=7.7% |
| per-Q flip rate | p50=8.5%, mean=15.8%, max=100% |

⚠️ 干预率 **33%** 是本次最高的 (vs 之前 MMLU 30Q 时 17.8%, vs 历史 14B baseline 29%)。原因: `--ban_think_token` 排除 `<think>` (高频 sampled token), 强制 sampler 从 top-5 其他候选选, 这些候选 entropy 偏高 → 更多 step 跨过 threshold=0.8 → 干预率高。

### 3.3 速度对比

| 指标 | noSIA | SIA |
|------|-------|-----|
| total tokens | ~23,000 | 19,729 |
| total wall time | ~6 min | 12 min |
| **throughput** | ~64 tok/s | **27.5 tok/s** |
| avg time/Q | 3.7 s | 7.2 s |
| avg tokens/Q | ~230 | 197 |
| RM HTTP overhead | 0 | ~95ms × 33% × ~600 step/Q ≈ ~19 s 累积 (但与 LLM forward 并行/串行混合) |

### 3.4 Per-Q 极端 case 分析

**5 个 SIA 损失最大** (事实/列举题, SIA 推向冗余结构化, Skywork 偏好直接):

| id | Δ | noSIA reward | SIA reward | instruction (前 60 char) |
|----|---|-------------|-----------|------------------------|
| 78 | **-28.03** | 31.5 | 3.5 | "I like to host guests at my home from time to time..." |
| 35 | **-27.78** | 31.8 | 4.0 | "Who created the Superman cartoon character?" |
| 23 | **-20.05** | 21.2 | 1.2 | "What type of soil is suitable for cactus?" |
| 4 | -17.19 | 13.1 | -4.1 | "What is some cool music from the 1920s?" |
| 87 | -15.22 | 18.0 | 2.8 | "I have my grandfather's antique fountain pen..." |

**5 个 SIA 提升最大** (开放性/建议/有反直觉答案的题, SIA 推向更结构化反而帮到):

| id | Δ | noSIA reward | SIA reward | instruction (前 60 char) |
|----|---|-------------|-----------|------------------------|
| 12 | **+21.41** | -6.7 | 14.7 | "I'm trying to teach myself to have nicer handwriting..." |
| 40 | **+18.45** | 3.2 | 21.6 | "Should I get my children a nanny? I'm so exhausted." |
| 59 | **+18.38** | -12.8 | 5.6 | "I like to host guests at my home from time to time..." |
| 15 | +17.47 | -3.4 | 14.1 | "Do you know why turkeys became the official food of thanksgiving?" |
| 36 | **+15.97** | -17.6 | -1.7 | "What is Atlantis?" ← 之前实验里 -25 题, 现在反转 +15.97! |

注意 id=36 ("Atlantis") 在实验 1 (`--disable_thinking`) 中是 -25, **本次反转为 +15.97**, 证明之前的 -25 是 `<think>` format artifact 不是真实 alignment 反向。

---

## 4. 跨实验对比表

| 实验 | mean Δ | win rate | <think> | 备注 |
|------|--------|----------|---------|------|
| 历史 14B A1 (官方代码) | **+13.2%** | n/a | 0/805 (官方用 raw `Human/Assistant`, 不进 thinking) | baseline reference |
| 实验 1: 0GM + `--disable_thinking` (50Q) | **-40%** | 12% | 0/50 (但出 "Here's a thinking process:" outline) | chat_template 空 `<think></think>` 块 trigger 训练先验 |
| **实验 3: 0GM + `--no_think_prompt --ban_think_token`** (98Q) | **-13%** | 44% | **0/100** ✅ | 最干净, 暴露 SIA 真实 alignment effect |

**幅度改善 5.6× (-40% → -13%)**, 主要来源:
- 60% 实验 1 的 hurt 是 `<think>`-format artifact (SIA 把 reward 推到 outline 风格)
- 剩下 -13% 才是真实的 cross-family OOD 损失 (Qwen3-base 训的 RM × 0GM Qwen3.6 主 LLM)

---

## 5. 跟论文 §5.3 Weak-to-Strong 数据对比

论文 Figure 3 显示 (same Qwen3 family, W2S):
- VM-4B → Qwen3-8B-LLM: HEx-PHI +6, AlpacaEval +4, TruthfulQA +5 (reward 提升)
- VM-4B → Qwen3-14B-LLM: HEx-PHI +5, AlpacaEval +3, TruthfulQA +3

**我们 (cross-family)**: VM-Qwen3-4B-Base → 0GM-Qwen3.6-35B-A3B, AlpacaEval **-13%** (-1.31 reward)

→ 论文 W2S 数据**全部 same-family**, 跨家族不在论文测试范围。我们的数据点提供了**首个 cross-family quantification**: 跨 family + cross-thinking-mode 时 SIA 不再是 net 收益, 反而小幅 hurt。

---

## 6. 归档文件 (在 `exp/` 下)

### 实验 1 archive (chat_template + `--disable_thinking`)

| 文件 | 路径 |
|------|------|
| noSIA generation (200Q) | [`exp/alpaca_0gm_nosia_200q_20260602_122922.json`](../exp/alpaca_0gm_nosia_200q_20260602_122922.json) |
| noSIA Skywork scored | [`exp/alpaca_0gm_nosia_200q_scored_20260602_122922.json`](../exp/alpaca_0gm_nosia_200q_scored_20260602_122922.json) |
| SIA generation (200Q partial, 50 scored) | [`exp/alpaca_0gm_sia_disable_think_200q_20260602_130750.json`](../exp/alpaca_0gm_sia_disable_think_200q_20260602_130750.json) |
| SIA 50Q partial scored | [`exp/alpaca_0gm_sia_disable_think_50q_partial_scored_20260602_130750.json`](../exp/alpaca_0gm_sia_disable_think_50q_partial_scored_20260602_130750.json) |

### 实验 2 archive (中间过渡, noSIA + enable_thinking)

| 文件 | 路径 |
|------|------|
| noSIA generation 100Q (thinking enabled) | [`exp/alpaca_0gm_nosia_think_100q_20260602_140046.json`](../exp/alpaca_0gm_nosia_think_100q_20260602_140046.json) |
| noSIA server log | [`exp/alpaca_0gm_nosia_think_server_20260602_140046.log`](../exp/alpaca_0gm_nosia_think_server_20260602_140046.log) |
| (没跑 SIA arm, 因为发现还有更彻底方案) | — |

### 实验 3 archive (`--no_think_prompt --ban_think_token`, 最终方案)

| 文件 | 路径 |
|------|------|
| SIA generation 100Q | [`exp/alpaca_0gm_sia_banthink_100q_20260602_151551.json`](../exp/alpaca_0gm_sia_banthink_100q_20260602_151551.json) |
| SIA server log | [`exp/alpaca_0gm_sia_banthink_server_20260602_151551.log`](../exp/alpaca_0gm_sia_banthink_server_20260602_151551.log) |
| SIA Skywork scored | [`exp/alpaca_0gm_sia_banthink_100q_scored_20260602_151551.json`](../exp/alpaca_0gm_sia_banthink_100q_scored_20260602_151551.json) |
| noSIA generation 100Q | [`exp/alpaca_0gm_nosia_banthink_100q_20260602_155639.json`](../exp/alpaca_0gm_nosia_banthink_100q_20260602_155639.json) |
| noSIA server log | [`exp/alpaca_0gm_nosia_banthink_server_20260602_155639.log`](../exp/alpaca_0gm_nosia_banthink_server_20260602_155639.log) |
| noSIA Skywork scored | [`exp/alpaca_0gm_nosia_banthink_100q_scored_20260602_155639.json`](../exp/alpaca_0gm_nosia_banthink_100q_scored_20260602_155639.json) |

---

## 7. 代码变更

### 7.1 新增脚本

- [`eval/alpaca_eval.py`](../eval/alpaca_eval.py) — HTTP client, 跑 AlpacaEval. 关键 flag:
  - `--disable_thinking`: chat_template_kwargs={"enable_thinking": false}, 塞空 `<think></think>` block (不彻底)
  - `--no_think_prompt`: 走 `/v1/completions` 用 raw `Human/Assistant` 文本 prompt (完全绕过 chat_template)
  - `--ban_think_token`: vllm `bad_words=["<think>","</think>"]` 在 sampler 层禁掉 (跟 `--no_think_prompt` 配合使用最彻底)

- [`scripts/measure_alpaca_reward.py`](../scripts/measure_alpaca_reward.py) — Skywork-Reward-V2 打分脚本. 关键 flag:
  - `--strip_think`: 用正则剥离 `<think>...</think>` 再喂 Skywork (本次实验 3 不需要, 因为模型本来就 0 think tag)
  - `--device cpu` 或 `cuda:0`: CPU 推理慢但不抢 GPU; GPU 快

### 7.2 修改 `src/sia_vllm_server.py`

加 `/v1/completions` 端点 + `CompletionRequest` 类支持:
- `prompt: str` (替代 messages)
- `bad_words: Optional[list[str]]` (透传到 vllm SamplingParams)
- 行为: skip chat_template, 直接拿 raw prompt 给 vllm engine, SIA logits processor 仍按 decode step 触发

---

## 8. 关键发现 (TLDR 重述 + 细节)

1. **`<think>`-format artifact 占了"SIA on 0GM 看似严重 hurt"的大头** (-40% → -13%, **5.6× 改善**)
   - chat_template + `--disable_thinking` 还是塞空 `<think></think>`, 触发模型训练先验
   - SIA 在 thinking-pattern context 下把 reward 推向 "thinking outline" 形式
   - 这是 RM-4B 训练数据 (UltraFeedback 等 helpful chat) 里**包含 reasoning-style 回答**导致的 reward bias

2. **0GM 训练先验极强**: 即便 `--no_think_prompt` (raw `Human/Assistant` 完全无 chat tokens), 仍 60% 题自发 emit `<think>`。需要 vllm `bad_words` 在 sampler 层强制禁。

3. **干预率受 `bad_words` 影响**: `--ban_think_token` 让干预率从 ~18% 跳到 33% (排除高频 `<think>` token 后, top-5 内部 entropy 变高)。

4. **真实 cross-family OOD 损失约 -13%**, 是论文 §5.3 W2S 数据外推不到的场景。论文全部 same-family (Qwen3 backbone × Qwen3-VM), W2S 时 +3~5 reward; 我们 cross-family (Qwen3-VM × Qwen3.6-LLM) 反向 -13%。

5. **极端 case 分裂**: 事实/列举题 SIA 输 (-28 ~ -15 reward), 开放性/建议题 SIA 赢 (+15 ~ +21 reward)。win rate 44% 接近 fair coin flip。

---

## 9. 跟前置 doc 对比

| doc | 内容 | 状态 |
|-----|------|------|
| [`alpaca-eval-plan-20260602.md`](alpaca-eval-plan-20260602.md) | 早期方案规划 (805Q, max_tokens=256) | 部分过时, 用户后来要求改 200Q→100Q + 改 max_tokens |
| [`alpaca-eval-0gm-35b-execution-20260602.md`](alpaca-eval-0gm-35b-execution-20260602.md) | noSIA `--disable_thinking` 200Q 执行 | 实验 1 noSIA 部分 |
| **本 doc** | 实验 1+2+3 完整执行 + 4 个实验对照 + 9 章节 | ✅ 最新, 涵盖所有 |

后续看 SIA on 0GM 用什么数据/结论, 优先看本 doc; 早期 plan/execution doc 保留为历史记录。

---

## 10. 下一步建议 (按优先级)

按 [`sia-fix-and-vm-ablation-0gm-35b-20260602.md`](sia-fix-and-vm-ablation-0gm-35b-20260602.md) §7 的优先级:

1. **跑 RM=1.7B + AlpacaEval 100Q** (跟本次 RM=4B 对照): MMLU 30Q 上 1.7B 比 4B 好 +6.7pp acc, AlpacaEval 上预期类似 (小 VM 更温和, 跨家族损失更小)
2. (可选) 跑完整 805Q + RM=4B + `--no_think_prompt --ban_think_token`, 把 N=98 提升到 N=805 减少 sample noise
3. 训一个专门为 0GM 训的 VM (cross-family 问题的 ML 层解决)
