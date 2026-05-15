# SIA Regression 评测报告

## 评测目标

验证 SIA（Sparse Inference-time Alignment）在通用知识类评测集（MMLU）上的 regression 情况：

> **核心问题**：加了 SIA 干预之后，模型在与 Value Model 训练目标无关的领域（知识问答），是否会出现 accuracy 下降？

---

## 实验设置

### 模型配置

| 组件 | 模型 | 说明 |
|------|------|------|
| LLM | Qwen3-14B | 推理主模型 |
| Value Model（RM） | Qwen3-4B + VM-Qwen3-4B-Base LoRA | SIA 干预所用奖励模型 |

### SIA 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--topk` | 5 | 每步评分的候选 token 数 |
| `--weight` | 1.0 | RM score 乘以权重后加到 logits |
| `--entropy_threshold` | 1.0 | 只在 token entropy ≥ 1.0 时干预（稀疏策略） |

### 评测集

**MMLU-Redux**（edinburgh-dawg/mmlu-redux），覆盖 30 个子学科，包含解剖学、天文学、商业伦理、化学、物理、逻辑、经济等知识领域。

### 推理参数

| 参数 | 值 |
|------|----|
| temperature | 1.0 |
| max_tokens | 2048 |
| repetition_penalty | 1.3 |
| system prompt | "You are a helpful assistant. Answer the following multiple choice question. You may think before answering, but keep your reasoning concise and under 500 tokens. End your response with exactly: Answer: X (where X is A, B, C, or D)." |

### 超时处理

服务端设置 60s 超时：超时后 abort 生成，返回已有内容。评测时**排除** latency > 60s 的样本；同时在 noSIA 日志中同步排除对应顺序号的样本，保证两组比较的题目完全一致（side-by-side 公平对比）。

---

## 实验1：小规模验证（limit=3）

**运行命令（SIA）：**
```bash
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 --port 8001 > log_rm_server_202605141350.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 > log_llm_server_202605141350.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_SIA_202605141350.json \
    --limit 3 > log_SIA_202605141350.txt 2>&1 &
```

**运行命令（noSIA）：**
```bash
nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 0.0 --entropy_threshold 999999 \
    --host 0.0.0.0 --port 8000 > log_llm_server_noSIA_202605141540.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_noSIA_selfServer_202605141540.json \
    --limit 3 > log_noSIA_selfServer_202605141540.txt 2>&1 &
```

**结果：**

| 指标 | SIA | noSIA（同顺序号题目） |
|------|-----|---------------------|
| 原始总样本 | 90 | 90 |
| 超时排除（>60s） | 29 | — |
| **有效样本** | **61** | **61** |
| correct=True | 49 | 50 |
| correct=False | 12 | 11 |
| **Accuracy** | **80.33%** | **81.97%** |
| 95% 置信区间 | ±9.98% | ±9.65% |
| 平均 latency | 28.5s | 6.8s |

**小结：** 两组 accuracy 差值约 1.6 个百分点（仅 1 道题的差异），远小于 ±10% 的置信区间，统计上无显著差异。但 61 题的样本量过小，结论可靠性有限。

---

## 实验2：大规模验证（limit=20）

**运行命令（SIA）：**
```bash
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 --port 8001 > log_rm_server_SIA_202605142111.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 > log_llm_server_SIA_202605142111.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_SIA_202605142111.json \
    --limit 20 > log_SIA_202605142111.txt 2>&1 &
```

**运行命令（noSIA）：**
```bash
nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 0.0 --entropy_threshold 999999 \
    --host 0.0.0.0 --port 8000 > log_llm_server_noSIA_202605142111.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_noSIA_selfServer_202605142111.json \
    --limit 20 > log_noSIA_selfServer_202605142111.txt 2>&1 &
```

**结果：**

| 指标 | SIA | noSIA（同顺序号题目） |
|------|-----|---------------------|
| 原始总样本 | 600 | 600 |
| 超时排除（>60s） | 158 | — |
| **有效样本** | **442** | **442** |
| correct=True | 369 | 363 |
| correct=False | 73 | 79 |
| **Accuracy** | **83.48%** | **82.13%** |
| 95% 置信区间 | ±3.46% | ±3.57% |
| 平均 tokens | 325.5 | 507.8 |
| 平均 latency | 26.0s | 5.9s |

**小结：** 两组 accuracy 差值 1.35 个百分点，在 ±3.5% 的置信区间内，统计上无显著差异。实验2 有 442 个有效样本，置信区间显著收窄，结论可靠性更高。

---

## 分析

### 1. Accuracy：无显著差异

两次实验的结果高度一致：SIA 与 noSIA 的 accuracy 差值均在 1~2 个百分点以内，均在统计误差范围内。这说明 SIA 干预**既没有提升、也没有损害** Qwen3-14B 在 MMLU 上的表现。

这一结果是符合预期的。正如 Ming Wu 所指出的：

> "Value Model 是在 alignment 的场景里训练的，如果实际的场景跟 align 的目标毫无关系，就不会有所谓的提升。"

MMLU 是事实知识类评测，而 VM-Qwen3-4B-Base 是在 Helpfulness（UltraFeedback）和 Harmlessness（WildGuardMix）数据上训练的，训练目标与知识问答没有直接关联。Value Model 无法分辨"选 A 还是选 B 哪个更符合事实"，因此在此类任务上既不能提升，也不应造成干扰。

### 2. 为何做 MMLU Regression 测试

本次实验的目的**不是期望 SIA 提升 MMLU 准确率**，而是验证 SIA 在与 Value Model 训练目标无关的领域内是否引入负面效果（regression）。

此次测试对应 Jason 所说的：

> "本来也是看有没有 regression。"

结论是：无 regression。

### 3. noSIA 平均 tokens 更高的原因

观察到一个现象：noSIA 的平均 tokens（507.8）显著高于 SIA（325.5），但两者使用完全相同的 prompt。

原因在于 SIA 的 Value Model 改变了模型的**生成方式**：

- **noSIA**：无 RM 干预，模型自由选择是否 thinking。有时跳过 thinking 直接输出 `Answer: X`（仅 3 个 tokens），有时进行长篇推理（500~800+ tokens），导致分布呈两极，平均值被长回复拉高。
- **SIA**：RM 从第一个 token 开始干预。在 Helpfulness 数据上训练的 Value Model 倾向于给"开始推理"的 token 打高分，稳定地引导模型进入 thinking 模式，始终生成中等长度的推理过程（平均 325 tokens）。

这说明 SIA 不只改变"选哪个答案"，还改变了"用什么方式回答"。

### 4. 推理延迟开销

SIA 的平均 latency（26s）远高于 noSIA（5.9s），约为 4.4 倍。

根据 LLM server 日志的逐步分析：

| 类型 | 平均耗时/step |
|------|------------|
| SKIP（不调用 RM） | 17 ms |
| INTERVENE（调用 RM） | 279 ms |

INTERVENE 比 SKIP 慢约 **16 倍**，主要原因是当前实现对 topk=5 个候选 token 串行进行 5 次独立 RM forward pass。该性能问题有明确的优化路径（批量 forward、KV cache prefix 共享等），详见 `/tmp/sia_latency_optimization.md`。

---

## 结论

1. **无 regression（最核心结论）**：SIA 干预在 MMLU 知识类评测上没有造成 accuracy 下降，与预期一致。两次实验均显示 SIA 与 noSIA 的 accuracy 差值在统计误差范围内，可认为无显著影响。

2. **SIA 的适用边界**：Value Model 在哪个领域训练，就在哪个领域引导效果。在与训练目标无关的知识问答类任务上，SIA 既不提升也不损害，属于"透明"干预。

3. **延迟开销存在，有优化空间**：当前实现中，INTERVENE 步骤比无干预步骤慢约 16 倍，是主要的工程优化方向，但不影响 accuracy 的评测结论。

4. **后续实验建议**：在 SIA Value Model 对口的评测集（AlpacaEval、TruthfulQA、HEx-PHI）上验证 SIA 是否能带来 accuracy/reward 提升，以完整覆盖方案一的验收标准。
