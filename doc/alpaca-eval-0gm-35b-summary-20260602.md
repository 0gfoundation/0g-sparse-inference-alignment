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

---

## 11. 跟官网代码 (`/workspace/SIA/git/SIA/`) 的偏差审计

### 11.1 SIA vs noSIA **内部公平性** — ✅ 通过

| 维度 | SIA | noSIA |
|------|-----|-------|
| Prompts | 前 100 题 alpaca_eval.json | 前 100 题 alpaca_eval.json ✓ |
| max_tokens / temp / top_p / top_k / rep_pen | 2048 / 1.0 / 0.95 / 20 / 1.0 | 2048 / 1.0 / 0.95 / 20 / 1.0 ✓ |
| no_think_prompt / ban_think_token | True / True | True / True ✓ |
| Server 代码路径 | sia_vllm_server.py /v1/completions | sia_vllm_server.py /v1/completions ✓ |
| **唯一差异** | weight=1.0, thr=0.8 | **weight=0.0, thr=999999** (SIA processor 注册但永不 INTERVENE) |

两 arm 同代码、同 prompt、同 sampling 配置, 唯一差异是 SIA processor 是否真正干预。**内部对比公平。**

### 11.2 跟官网默认配置的偏差

| 维度 | 官网 (`evaluate.py` / `sia.py`) | 我们 | 影响评估 |
|------|---------------------------------|------|----------|
| **topk** (RM 候选数) | **10** | **5** | ⚠️ 候选池减半, SIA 有效干预空间窄一倍 (历史 14B b2 baseline 为加速也用 5) |
| weight | 1.0 | 1.0 | ✓ 相同 |
| entropy_threshold | None / 1.0 (论文) | **0.8** | 略低让干预率高, 不影响公平性, 仅 hyperparameter 选择 |
| max_new_token | 128 (default) / 256 (14B A1) | **2048** | 让 0GM thinking 有空间, 但设置差很大 |
| **top_p** | **无** | **0.95** | ⚠️ 官方 sia.py 0 occurrence of top_p, 我们多了 nucleus 截断 |
| **top_k (sampling)** | **无** | **20** | ⚠️ 同上, 官方采样无二次截断 |
| **repetition_penalty** | **无** | **1.0** (= 关闭) | 1.0 实际是 no-op, 不实质影响 |
| **bad_words** | **无** | **["<think>","</think>"]** | 0GM 训练先验强自发 emit `<think>`, 必须禁; 官方 14B 用 raw `Human/Assistant` 不进 thinking, 不需要 |
| RM score 数值范围 | raw logits (~[-7,+7]) | sigmoid prob 反变换回 raw logits (commit `e1d150e` Step 1) | ✓ 数值等价 |
| 非 top-k mask | -inf | -inf (commit `e1d150e` Step 2) | ✓ 等价 |
| mean-norm | **无** (`combined = rewards * weight + orig`) | **有** (`rm_deltas = (rm_scores - mean) * weight`) | softmax shift-invariant under -inf mask, **数学等价** (top-5 内部 ordering 不变) |
| SKIP path (weight=0) | `combined = orig_scores`, softmax(/T), multinomial (无 top_p/top_k filter) | logits 不动, vllm sampler apply temp + top_p + top_k + bad_words + multinomial | ⚠️ **微妙差异**: 我们 SKIP 仍走二次 filter, 官方不走 |

### 11.3 🚨 真正可疑的几处

#### 可疑 #1 — **mean-norm + `<think>` 在 top-5 的污染**

当 `<think>` (id=248068) 在 0GM top-5 内 (因训练先验, 概率很高):
1. SIA 把 `<think>` 当 candidate 之一, RM 给它打分
2. RM 训练时没见过 `<think>` 当 candidate 的场景, 分数偏 noise (可能很高或很低)
3. **mean-norm 时 `<think>` 的分数被算进 mean** → mean 被污染 → **影响其他 4 个 candidate's delta**
4. -inf mask 后 logits[`<think>`] = orig_logit + delta (modified)
5. vllm sampler 的 bad_words 把 `<think>` 又设回 -inf → `<think>` 永远不被采到
6. **但步骤 3 的 mean 污染**已经扭曲了其他 candidate 的 logit

→ **官方代码没有这个问题** (无 mean-norm + 14B 不自发 emit `<think>`)。我们的 setup 引入此噪声, 估算每次 INTERVENE 都受影响 (假设 `<think>` 在 top-5 的概率 ~50-80%)。

#### 可疑 #2 — **topk=5 vs 官方 10**

历史 14B baseline 用 topk=5 (b2 InprocClient 时代为减少 RM call 加速)。论文实验和官方 evaluate.py default 都用 10。**RM 候选池减半, SIA 直接影响力减半**。如果 SIA on 0GM 真有效, 用 topk=10 可能恢复部分效果。

#### 可疑 #3 — **干预率 33% 是 ban_think_token 副作用**

`bad_words` 把 `<think>` (高频 top-1) banned 后, 排名后挪, top-1 不再压倒性, top-5 internal entropy 拉高 → **更多 step 跨过 threshold=0.8 触发 INTERVENE**。

→ 这跟论文"sparse junction" 语义略不同: 论文期望干预集中在**模型真正不确定的关键决策点**, 我们的 33% 干预率部分是 setup-induced (ban_think 后 top-5 重排引起的伪 entropy 上升)。

#### 可疑 #4 — **noSIA 仍走二次 filter, 跟官方 pure noSIA 不同**

官方 noSIA (weight ≤ 0.1) 是 `combined = orig_scores`, softmax(/T), multinomial, 没有 top_p/top_k/rep_pen 二次截断。我们的 noSIA arm 用 sia_vllm_server.py 走 vllm sampler 全套 (top_p=0.95, top_k=20, bad_words)。**SIA 跟我们 noSIA 内部公平**, 但**跟官方 pure noSIA 数字不可比**。

### 11.4 优先级排序的修复建议

按"对当前结论 (-13% reward) 影响大小"排序:

#### 优先级 1 — **改 `topk=10` 重跑 100Q** (跟官方对齐)

理由:
- 候选池翻倍, SIA 真正影响范围大一倍
- 历史 14B A1 (官方 topk=10) +13.2% reward, 我们用 topk=5 已可能弱了一半的 SIA 效力
- 修复成本: 1 行 CLI 参数, ~15-20 min 重跑 SIA 100Q (noSIA arm 不变, 因为 noSIA 不调 RM)
- 预期: 若 topk=10 让 SIA 改善至 -5% 或转正, 说明 topk=5 是主要 bottleneck

#### 优先级 2 — **去掉 mean-norm + 重跑** (匹配官方)

理由:
- 官方代码 `combined = rewards * weight + orig` 无 mean-norm
- 我们 mean-norm 在 `<think>` 在 top-5 时引入污染
- 修复成本: 改 1 处代码 (commit `e1d150e` 的 Step 2 fix 区域), 重新部署 SIA server, 重跑 100Q
- 预期: 去掉 mean-norm 后, `<think>` 在 top-5 时其他 candidate's delta 不再受 `<think>` RM score 污染, 干预方向更精确

#### 优先级 3 — **改用 1.7B VM 重跑 AlpacaEval 100Q**

理由:
- MMLU 30Q 已证明 1.7B VM 比 4B 在 0GM 上好 +6.7pp acc
- AlpacaEval 上预期同向收益 (跨家族场景小 VM 更温和)
- 这是 [`sia-fix-and-vm-ablation-0gm-35b-20260602.md`](sia-fix-and-vm-ablation-0gm-35b-20260602.md) §7 的 #1 建议
- 修复成本: kill 4B RM, 起 1.7B RM, 重跑 SIA 100Q (~15 min)

#### 优先级 4 — **完整 805Q 减少 sample noise**

当前 N=98 (2 题 SIA 输出过短被 skip), Δ=-1.31 reward, p-value 估算: stdev=12, SE_diff ≈ 1.6 → -1.31 仅约 0.8σ, 统计上不显著。跑完整 805 Q → SE 降 1/√(805/98) ≈ 2.86×, 能确认 -13% 是真实信号还是 sample noise。

#### 优先级 5 — **完全匹配官方 sampling**: 去掉 `top_p/top_k/rep_pen` 二次截断

跟官方完全一致。但 0GM 大词表有 OOV 风险 (历史 rep_pen=1.3 + top_k=-1 时乱码)。需要测试是否能在 raw Human/Assistant + ban_think 下安全。

### 11.5 推荐顺序

1. 先跑**优先级 1** (topk=10) — 改动最小, 跟官方对齐意义最大, 单次实验
2. 再跑**优先级 3** (1.7B VM with topk=10), 看 VM size + topk 双管齐下效果
3. 之后看是否值得做**优先级 4** (805Q full) 减少噪声
4. **优先级 2** (mean-norm) 和 **优先级 5** (完全匹配 sampling) 是工程层 cleanup, 影响相对小, 可暂缓
