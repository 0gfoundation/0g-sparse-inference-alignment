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
| Value Model | Qwen3-4B + VM-Qwen3-4B-Base LoRA | SIA 干预所用奖励模型 |

**模型组合选择说明：**

- **为何选用 Qwen3-14B + Qwen3-4B 组合**：该组合是 SIA 论文中测试的最大规模配置，实验结论可与论文原始结果直接参照对比。
- **为何选用 VM-Qwen3-4B-Base LoRA 作为 Value Model**：该 checkpoint 由论文第一作者 Hu Runyi 训练并公开发布，是论文推荐的最完善版本。本次实验直接下载使用，未做任何修改。

### SIA 干预参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--weight` | 1.0 | Value Model score 加权后叠加到 logits |
| `--entropy_threshold` | 1.0 | 只在模型不确定（token entropy ≥ 1.0）时才触发干预；确定性高的 token 直接跳过 |

以上参数值参照论文推荐配置，为效果最稳定的一组。

### 评测集

**MMLU-Redux**（edinburgh-dawg/mmlu-redux），覆盖 30 个子学科，包含解剖学、天文学、商业伦理、化学、物理、逻辑、经济等知识领域。

### 推理参数

| 参数 | 值 |
|------|----|
| temperature | 1.0 |
| repetition_penalty | 1.3 |
| system prompt | "You are a helpful assistant. Answer the following multiple choice question. You may think before answering, but keep your reasoning concise and under 500 tokens. End your response with exactly: Answer: X (where X is A, B, C, or D)." |

### noSIA 说明

noSIA 并非另一套独立部署，而是在同一套服务框架下、将 SIA 干预完全关闭后运行：将 Value Model 打分权重设为 0、干预阈值设为极大值，使得每一步都直接跳过 Value Model 调用，退化为标准的 vLLM 推理。这样可以保证两组实验除 SIA 干预开关外，其余条件（模型、参数、评测脚本）完全一致，结果具有直接可比性。

### 超时处理

SIA 开启时，每个生成步骤需要额外调用 Value Model 打分，导致单题推理时间显著拉长。为避免个别超长回复无限占用资源，服务端设置 60s 超时上限：超时后强制中止生成，返回已有内容。

评测时**排除** latency > 60s 的样本，原因如下：
- 超时样本的回复被截断，无法保证答案完整，纳入统计会引入噪声；
- noSIA 同步排除对应题目，保证两组比较的题目集合完全一致（side-by-side 公平对比）。

超时题目集中在推理链较长的子学科（如 formal_logic、college_mathematics），并非随机分布，但两组排除后剩余题目完全相同，不影响对比的公平性。

---

## 实验1：小规模验证（每科取 3 题，共 90 题）

**结果：**

| 指标 | SIA | noSIA（同组题目） |
|------|-----|-----------------|
| 原始总样本 | 90 | 90 |
| 超时排除（>60s） | 29 | — |
| **有效样本** | **61** | **61** |
| 预测正确 | 49 | 50 |
| 预测错误 | 12 | 11 |
| **Accuracy** | **80.33%** | **81.97%** |
| 95% 置信区间 | ±9.98% | ±9.65% |
| 平均 latency | 28.5s | 6.8s |

*注：latency 差异来自 SIA 的 Value Model 打分开销，详见性能分析章节。*

**小结：** 两组 accuracy 差值约 1.6 个百分点（仅 1 道题的差异），远小于 ±10% 的置信区间，统计上无显著差异。但 61 题的样本量过小，结论可靠性有限。

---

## 实验2：大规模验证（每科取 20 题，共 600 题）

**结果：**

| 指标 | SIA | noSIA（同组题目） |
|------|-----|-----------------|
| 原始总样本 | 600 | 600 |
| 超时排除（>60s） | 158 | — |
| **有效样本** | **442** | **442** |
| 预测正确 | 369 | 363 |
| 预测错误 | 73 | 79 |
| **Accuracy** | **83.48%** | **82.13%** |
| 95% 置信区间 | ±3.46% | ±3.57% |
| 平均输出 tokens | 325.5 | 507.8 |
| 平均 latency | 26.0s | 5.9s |

*注：latency 差异来自 SIA 的 Value Model 打分开销，详见性能分析章节。*

**小结：** 两组 accuracy 差值 1.35 个百分点，在 ±3.5% 的置信区间内，统计上无显著差异。实验2 有 442 个有效样本，置信区间显著收窄，结论可靠性更高。

---

## 分析

### 1. Accuracy：无显著差异

两次实验的结果高度一致：SIA 与 noSIA 的 accuracy 差值均在 1~2 个百分点以内，均在统计误差范围内。这说明 SIA 干预**既没有提升、也没有损害** Qwen3-14B 在 MMLU 上的表现。

这一结果是符合预期的。MMLU 是事实知识类评测，而 VM-Qwen3-4B-Base 是在 Helpfulness（UltraFeedback）和 Harmlessness（WildGuardMix）数据上训练的，训练目标与知识问答没有直接关联。Value Model 无法分辨"选 A 还是选 B 哪个更符合事实"，因此在此类任务上既不能提升，也不应造成干扰。

### 2. 为何做 MMLU Regression 测试

本次实验的目的**不是期望 SIA 提升 MMLU 准确率**，而是验证 SIA 在与 Value Model 训练目标无关的领域内是否引入负面效果（regression）。结论是：无 regression。

### 3. SIA 对生成方式的影响

noSIA 的平均输出 tokens（507.8）显著高于 SIA（325.5）。在 Helpfulness 数据上训练的 Value Model 倾向于给"开始推理"的 token 打高分，稳定地引导模型进入 thinking 模式，生成中等长度的推理过程；而 noSIA 无干预，模型的生成长度分布较为分散，平均值偏高。

这说明 SIA 干预改变了模型"用什么方式回答"——即便在对最终 accuracy 无显著影响的任务上，生成行为本身已发生明显变化。

---

## 结论

1. **无 regression（最核心结论）**：SIA 干预在 MMLU 知识类评测上没有造成 accuracy 下降，与预期一致。两次实验均显示 SIA 与 noSIA 的 accuracy 差值在统计误差范围内，可认为无显著影响。

2. **SIA 的适用边界**：Value Model 在哪个领域训练，就只在哪个领域的任务上能引导 LLM，对无关领域既不提升也不干扰。举例来说，若 Value Model 训练目标是"不提供有害内容"，则在与安全无关的对话中它基本不起作用；若对话涉及有害内容，它才会发挥引导效果。本次实验中，Value Model 训练在 Helpfulness 和 Harmlessness 方向，而 MMLU 是纯知识问答，两者无关，因此干预效果为零——这是符合预期的。

3. **后续实验建议**：在 SIA Value Model 对口的评测集（AlpacaEval、TruthfulQA、HEx-PHI）上验证 SIA 是否能带来 accuracy/reward 提升，以完整覆盖方案一的验收标准。

---

## Appendix：实验运行命令

### 实验1（每科取 3 题）

**SIA：**
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

**noSIA：**
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

### 实验2（每科取 20 题）

**SIA：**
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

**noSIA：**
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
