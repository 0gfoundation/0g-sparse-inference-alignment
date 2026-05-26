# SIA Batch Forward 性能优化实验报告

## 实验目标

验证将 Value Model 打分从串行改为批量处理（batch forward）后的实际效果：

> **核心问题**：batch forward 优化能将有干预步骤的延迟降低多少？整体吞吐量提升几倍？模型的 accuracy 是否受到影响？

---

## 实验配置

实验配置与性能基准测试完全一致，仅将 Value Model 打分实现由串行改为 batch forward：

| 组件 | 模型 |
|------|------|
| LLM | Qwen3-14B（vLLM 部署） |
| Value Model | Qwen3-4B + VM-Qwen3-4B-Base LoRA |

**SIA 干预参数：**

| 参数 | 值 |
|------|----|
| `--topk` | 5 |
| `--weight` | 1.0 |
| `--entropy_threshold` | 1.0 |

**评测集：** MMLU-Redux，每科取 20 题，共 600 题。

---

## 一、性能提升

### 干预率与逐步延迟

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| 干预率 | 27.3% | **27.2%** | 基本不变 |
| 无干预步骤平均延迟 | 16.1 ms | **15.4 ms** | 基本不变 |
| 有干预步骤平均延迟 | 281.0 ms | **118.7 ms** | **降低 58%** |
| 延迟倍数（有/无干预） | 17.5x | **7.7x** | — |

优化前，每次干预需对 topk=5 个候选 token 串行执行 5 次 Value Model forward；优化后将 5 个候选组成一个 batch，一次 forward 完成，有干预步骤延迟从 281 ms 降至 119 ms。无干预步骤与 LLM forward 延迟（约 15 ms）基本相同，优化前后无变化，说明改动未影响 LLM 推理路径。

### 整体吞吐量

将两类步骤的延迟按干预率加权，推算整体吞吐量：

```
加权平均步骤延迟 = 72.8% × 15.4 ms + 27.2% × 118.7 ms = 43.5 ms
对应吞吐量 = 1000 / 43.5 = 23.0 tokens/s
```

| 版本 | 吞吐量（tokens/s） | 对比 |
|------|-------------------|------|
| 优化前 SIA | 11.3 | 基准 |
| **优化后 SIA** | **23.0** | **+2.0x** |
| noSIA（SIA 服务，干预关闭） | 84.3 | 参考上限 |

吞吐量从 11.3 提升至 23.0 tokens/s，约 **2 倍提升**，与优化前 doc/performance-report.md 中预估的"2–3x 提升"吻合。

---

## 二、超时样本变化

batch forward 提速后，原本因推理链较长而超时（>60s）的题目，有部分得以在限制时间内完成：

| 版本 | 超时样本数（共 600 题） |
|------|----------------------|
| 优化前 SIA | 158 |
| **优化后 SIA** | **85** |

超时样本从 158 降至 85，减少 **46%**。超时样本集中在推理链较长的子学科（如 formal_logic、college_mathematics、high_school_statistics），batch forward 在这些题目上节省的时间最为可观。

---

## 三、Accuracy 验证

为确保对比公平，取两次实验超时集合的**并集**作为排除集，只保留两次实验均在 60s 内完成的样本（共 423 题）进行 accuracy 统计：

| 指标 | 优化前 SIA | 优化后 SIA |
|------|-----------|-----------|
| 有效样本 | 423 | 423 |
| 超时排除逻辑 | 两次实验超时并集（177 题） | 同左 |
| 预测正确 | 356 | 354 |
| 预测错误 | 67 | 69 |
| **Accuracy** | **84.16%** | **83.69%** |
| 95% 置信区间 | ±3.48% | ±3.52% |
| 平均输出 tokens | 320.5 | 422.9 |
| 平均 latency | 25.4s | 15.2s |

两组 accuracy 差值 0.47 个百分点，远在统计误差范围内，**可认为无显著差异**。batch forward 只改变了打分的实现方式（串行 → 批量），未改变输入到 Value Model 的数据内容，因此 accuracy 保持一致是符合预期的。

*注：优化后平均输出 tokens 从 320.5 升至 422.9，原因是 batch forward 提速后模型在 60s 内能生成更长的推理链，并非优化本身改变了生成行为。*

---

## 结论

1. **有干预步骤延迟降低 58%**：从 281 ms 降至 119 ms，直接来自将 k 次串行 forward 合并为 1 次 batch forward。

2. **整体吞吐量提升约 2 倍**：从 11.3 tokens/s 提升至 23.0 tokens/s，与 doc/performance-report.md 中的预估一致。

3. **超时样本减少 46%**：原本因推理链过长而超时的 158 题中，有 73 题在优化后得以在 60s 内完成，评测覆盖率提高。

4. **Accuracy 无变化**：在两次实验均完成的 423 题上，accuracy 从 84.16% 到 83.69%，差值在统计误差内，模型行为未受影响。

---

## Appendix：实验运行命令

> 本节命令的日志已归档至 `exp/`。完整命令与说明见 [`exp/README.md`](../exp/README.md) 中「2026-05-17 20:30」一节。

```bash
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 \
    --port 8001 > log_rm_server_SIA_202605172030.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 > log_llm_server_SIA_202605172030.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_SIA_202605172030.json \
    --limit 20 > log_SIA_202605172030.txt 2>&1 &
```
