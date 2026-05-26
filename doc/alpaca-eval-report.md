# SIA AlpacaEval 评测报告

## 评测目标

验证 SIA（Sparse Inference-time Alignment）在 Helpfulness 类任务上的实际提升效果：

> **核心问题**：Value Model 在 Helpfulness 数据上训练，那么在 Helpfulness 导向的开放指令评测集（AlpacaEval）上，SIA 干预能否带来可测量的 reward 提升？

本次实验直接使用 SIA 论文作者提供的原始实验代码运行，结果可与论文数据直接参照对比。

---

## 实验设置

### 模型配置

| 组件 | 模型 | 说明 |
|------|------|------|
| LLM | Qwen3-14B | 推理主模型 |
| Value Model | Qwen3-4B + VM-Qwen3-4B-Base LoRA | SIA 干预所用模型，在 UltraFeedback（Helpfulness）和 WildGuardMix（Harmlessness）上训练 |
| 评分 RM | Skywork-Reward-V2-Llama-3.1-8B | **独立**评分模型，与引导用 Value Model 不同 |

使用独立 RM 打分的意义：评分模型与引导模型完全不同，避免"自评自"的偏差，结论更具客观性。

### SIA 干预参数

| 参数 | SIA | noSIA |
|------|-----|-------|
| `--rm_weight` | 1.0 | 0.0 |
| `--entropy_threshold` | 1.0 | 999999（不触发） |
| `--topk` | 10 | 10 |

其余参数（`--sample_temp 1.0`、`--max_new_token 256`）两组完全一致。

### 评测集

**AlpacaEval**，包含 805 条开放式指令，覆盖写作、问答、建议、推理等多类 Helpfulness 场景。这是与 Value Model 训练目标直接对口的评测集：Value Model 在 Helpfulness 数据上训练，AlpacaEval 测的正是 Helpfulness 质量。

---

## 实验结果

| 指标 | SIA | noSIA |
|------|-----|-------|
| 样本数 | 805 | 805 |
| 平均 Reward 分数 | **13.923** | 12.294 |
| 平均干预比例 | **16.14%** | 0% |

**Reward 提升：+1.629（绝对值），+13.2%（相对提升）**

---

## 分析

### 1. 在对口任务上 SIA 效果显著

MMLU 回归测试的结论是"Value Model 对无关领域没有干扰"；本次实验则给出了另一面：在 Value Model 训练目标对口的 Helpfulness 评测集上，SIA 带来了 **+13.2%** 的 reward 提升。两组实验共同验证了 SIA 的作用边界：有的放矢，不旁及无关领域。

### 2. 稀疏干预的效率

SIA 仅在 **16.14%** 的 token 位置触发了 Value Model 打分（即约 6 个 token 中只有 1 个触发干预），绝大多数低熵、确定性高的 token 直接跳过。这种稀疏策略在显著降低计算开销的同时，仍实现了可观的 reward 提升，与论文"20%–80% 干预率即可达到或超过全量干预效果"的结论一致。

### 3. 实验可信度

- **独立 RM 评分**：评分使用 Skywork-Reward-V2-Llama-3.1-8B，与引导用的 Qwen3-4B + VM LoRA 完全独立，排除了自评自偏差。
- **全量评测**：805 条样本全部有效，无超时排除，两组条件完全对称。
- **原始作者代码**：使用论文第一作者提供的 evaluate.py 和 measure_reward.py，结果可直接与论文对比。

---

## 结论

1. **SIA 在对口任务上带来显著提升**：AlpacaEval 平均 reward 从 12.294 提升至 13.923，相对提升 **+13.2%**，验证了 SIA 在 Helpfulness 方向的有效性。

2. **稀疏干预策略有效**：仅 16.14% 的 token 位置触发干预，即实现显著的 reward 提升，支持论文关于稀疏干预效率的核心主张。

3. **与 MMLU 回归测试形成完整闭环**：MMLU 测试证明 SIA 对无关领域无负面影响，AlpacaEval 测试证明 SIA 对对口领域有正向提升，两项实验共同完整验证了 SIA 的能力边界。

---

## Appendix

> 本报告的 4 个 artifact 文件（noSIA/SIA generation log + Skywork 打分后的 2 个 JSON）已归档至 `exp/`。完整命令与说明见 [`exp/README.md`](../exp/README.md) 中「2026-05-15 — A1: AlpacaEval」一节。

### A. 推理 Prompt 格式

本次实验使用论文原始 `evaluate.py`，未启用 `--add_sys_prompt`。每条指令的 prompt 格式为：

```
Human:
{instruction}
Assistant:
```

### B. AlpacaEval 数据集样本示例

AlpacaEval 共 805 条开放式指令，涵盖事实问答、实用建议、技能解释、创意写作等多类 Helpfulness 场景。以下为部分样本：

| # | 指令 |
|---|------|
| 1 | What are the names of some famous actors that started their careers on Broadway? |
| 11 | Do you think retinoid is effective on removing the acne? Because I have a lot of it. |
| 51 | What year was the Yamato Battleship built? |
| 101 | I am interested in trying some Indonesian dishes. Can you give me a recipe for Tahu Gejrot Cirebon? |
| 201 | What are five important topics for game design? |

这类指令没有固定标准答案，答案质量完全依赖模型的 Helpfulness——正是 Value Model 训练目标所对应的场景。

---

### C. 实验运行命令

#### 生成阶段（使用论文原始代码）

**noSIA（基线）：**
```bash
cd /workspace/SIA/git/SIA

nohup python3 evaluate.py \
      --llm    /workspace/SIA/models/Qwen3-14B \
      --rm      /workspace/SIA/models/Qwen3-4B \
      --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
      --dataset /workspace/SIA/data/alpaca_eval \
      --run_num -1 --run_percent 100 \
      --max_new_token 256 \
      --rm_weight 0.0 \
      --topk 10 \
      --sample_temp 1.0 \
      --entropy_threshold 999999 \
      --llm_gpu cuda:0 \
      --rm_gpu  cuda:0 2>&1 1>log_alpaca_eval_weight0_Qwen3_14B.txt &
```

**SIA：**
```bash
nohup python3 evaluate.py \
      --llm    /workspace/SIA/models/Qwen3-14B \
      --rm      /workspace/SIA/models/Qwen3-4B \
      --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
      --dataset /workspace/SIA/data/alpaca_eval \
      --run_num -1 --run_percent 100 \
      --max_new_token 256 \
      --rm_weight 1.0 \
      --topk 10 \
      --sample_temp 1.0 \
      --entropy_threshold 1.0 \
      --llm_gpu cuda:0 \
      --rm_gpu  cuda:0 2>&1 1>log_alpaca_eval_weight1_Qwen3_14B.txt &
```

### 打分阶段（Skywork RM 独立评分）

```bash
BASE_JSON="assets/generation_results/Qwen3-14B/alpaca_eval/Base/topk-10/20260515_112115/results.json"
SIA_JSON="assets/generation_results/Qwen3-14B/alpaca_eval/Ours/topk-10/rm-Qwen3-4B/weight-1.0/entropy-1.000/20260515_060002/results.json"

# noSIA 打分
python3 src/measure_reward.py \
      --input_file  $BASE_JSON \
      --output_file assets/generation_results/alpaca_base_scored_Qwen3-14B_202605152354.json \
      --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
      --device cuda:0

# SIA 打分
python3 src/measure_reward.py \
      --input_file  $SIA_JSON \
      --output_file assets/generation_results/alpaca_sia_scored_Qwen3-14B_202605152354.json \
      --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
      --device cuda:0
```
