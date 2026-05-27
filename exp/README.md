# SIA 实验日志与命令归档

本目录组织 17 组实验：13 组 SIA MMLU 三元组（RM + LLM + eval）+ 3 组 noSIA MMLU 二元组（LLM + eval，无 RM）+ 1 组 AlpacaEval（generation 阶段 noSIA/SIA 双 log + scoring 阶段双 JSON）。本文件记录每组的启动命令与简要说明。

按时间排序，从早到晚。

---

## 总览：17 组实验一览表

> 所有实验：`Qwen3-14B`（LLM）+ MMLU 30 学科评测，**单张 H200 共享部署**。SIA 组用 `Qwen3-4B-VM`（Value Model）做 per-token 干预，noSIA 组只跑纯 LLM。

**SIA 组（13 组三元组：RM + LLM + eval）**

| # | 时间戳 | 模式 / 关键变化 | n | 整体 tok/s | 中位 tok/s | 准确率 |
|---|---|---|---:|---:|---:|---:|
| 1 | 2026-05-14 13:50 | SIA + PyTorch RM，**首次冒烟**（`--limit 3`，90 题）| 90 | n/a※※ | n/a※※ | 54.4% |
| 2 | 2026-05-14 17:40 | SIA + PyTorch RM，eager 路径首次 600 题 | 247※ | 9.94 | 9.70 | n/a |
| 3 | 2026-05-14 21:11 | SIA + PyTorch RM，**完整 600 题**，与 noSIA 对照 | 600 | 11.40 | 11.39 | 61.5% |
| 4 | 2026-05-17 20:30 | SIA + PyTorch RM + **batch forward 优化**（5 candidates 单次 batch）| 600 | 23.41 | 27.03 | 69.3% |
| 5 | 2026-05-18 23:50 | SIA + PyTorch RM，复测 eager | 600 | 18.47 | 19.43 | 67.5% |
| 6 | 2026-05-19 08:00 | SIA + PyTorch RM + **`torch.compile`** ❌ | 600 | **crash** | — | 0%（LLM server 挂掉）|
| 7 | 2026-05-19 23:18 | SIA + PyTorch RM，回退 eager | 395※ | 25.22 | 28.59 | 68.2% |
| 8 | 2026-05-20 15:00 | SIA + PyTorch RM + **`RM_PROFILE=True`**（定位瓶颈用）| 600 | 25.41 | 27.45 | 72.2% |
| 9 | 2026-05-21 00:50 ※※※ | SIA + PyTorch RM + **`--cuda_graph`**（静态 bucketing）| 381※ | 24.96 | 26.83 | 72.3% |
| **10** | **2026-05-22 10:45 (G1)** | **SIA + vLLM RM（架构换代）**：RM 从手写 PyTorch FastAPI 改为 `vllm serve --convert classify` | 600 | **30.23** | 34.12 | 71.5% |
| 11 | 2026-05-25 09:45 (G2) | G1 + RM 调度参数：关 chunked prefill、`--max-model-len 4096→2048`、`gpu_memory_utilization 0.3→0.4` | 600 | 30.30 (+0%) | 34.83 | 72.5% (+1.0%pt) |
| 12 | 2026-05-25 21:20 (G3) | G2 + **客户端代码 3 件套**：去掉 `entropy.item()` CUDA sync / response_so_far 增量解码缓存 / top-1 flip 统计日志 | 600 | 29.65 (-2%) | 33.84 | **73.7%** (+1.2%pt) |
| 13 | 2026-05-26 10:30 (G4) | G3 + **RM FP8 量化**：`--quantization fp8 --kv-cache-dtype fp8` | 600 | 28.08 (-5%) | 31.82 | 73.2% (-0.5%pt) |

**noSIA 对照组（3 组二元组：LLM + eval，无 RM）**

| # | 时间戳 | 模式 / 关键变化 | n | 整体 tok/s | 中位 tok/s | 准确率 |
|---|---|---|---:|---:|---:|---:|
| N1 | 2026-05-14 15:40 | noSIA via `sia_vllm_server.py`（`--weight 0` 关掉干预），`--limit 3` 冒烟 | 90 | n/a※※ | n/a※※ | 71.1% |
| N2 | 2026-05-14 21:11 | noSIA via `sia_vllm_server.py`，**完整 600 题**（对照 #3 SIA） | 600 | 85.86 | 85.80 | 73.8% |
| **N3** | **2026-05-14 21:11** | **noSIA via vLLM 原生 `api_server`（不走 SIA processor）**——**最权威的 noSIA 上限** | 600 | **88.43** | 88.51 | **74.8%** |

> ※ 跑到一半被截断 / 中止，未完成 600 题
> ※※ 旧版 mmlu_eval，日志无 token/latency 字段
> ※※※ 日志文件名为 `*_202505210050.txt`（年份 typo），实际 file birth time 是 2026-05-20 16:52 UTC（北京时间 2026-05-21 00:52），按真实时间排序

**对照标尺**：
- **速度天花板** = N3 (noSIA 原生 vLLM) ≈ **88.4 tok/s**；当前 SIA 最快版本（G2 = 30.3 tok/s）只到 **34%**
- **准确率参考**：N2 (noSIA via 同一 `sia_vllm_server.py`) = 73.8%、N3 (noSIA 原生 vLLM) = 74.8%、SIA G3 = 73.7%。**全 600 题原始数据下 SIA 与 noSIA 大体持平**（在 1%pt 内）
- 注：`doc/vllm-rm-experiment-report.md` 报告的 "+1.31%pt SIA > noSIA" 是**剔除 65 个 SIA 超时样本后**在 535 个共同样本上的对比，不是全 600 题的原始结果

---

**AlpacaEval 组（1 组双臂对比，805 条 Helpfulness 指令）**

不同于 MMLU 评测，AlpacaEval 用原论文 `evaluate.py` + 独立 Skywork RM 打分，关心的是 reward（不是 tok/s 或 accuracy）。

| # | 时间 | 模式 | 干预率 | mean reward | 备注 |
|---|---|---|---:|---:|---|
| A1 | 2026-05-15 | noSIA (rm_weight=0, entropy_threshold=999999) | 0% | 12.294 | 基线 |
| **A1** | 2026-05-15 | **SIA (rm_weight=1.0, entropy_threshold=1.0)** | **16.14%** | **13.923 (+13.2%)** | 与对口任务的对照 ✅ |

详见 [`doc/alpaca-eval-report.md`](../doc/alpaca-eval-report.md) 和下文「2026-05-15 — A1」一节。

### 一眼看懂的几个跨阶段

| 阶段跃迁 | tok/s 变化 | 准确率变化 | 说明 |
|---|---:|---:|---|
| 早期 PyTorch RM eager → batch forward (#3 → #4) | 11.4 → 23.4 (**+105%**) | 61.5% → 69.3% (+7.8%pt) | batch 5 candidates 是早期主要加速来源 |
| PyTorch RM 最佳 (#8 profile) → **vLLM RM** (#10 G1) | 25.4 → 30.2 (**+19%**) | 72.2% → 71.5% (-0.7%pt) | 架构换代；显著加速 + 准确率持平 |
| G1 → G4（vLLM RM 内继续调优）| 30.2 → 28.1 (**-7%**) | 71.5% → 73.2% (+1.7%pt) | **吞吐基本停滞**，但准确率持续涨 |
| SIA 最快 (G2) vs **noSIA 上限 (N3)** | 30.3 vs **88.4** (**SIA 仅 34%**) | 72.5% vs 74.8% (-2.3%pt) | 当前 SIA 速度只有 noSIA 的 1/3，准确率小幅偏低 |

### 结论指针

- 吞吐已逼近**单张 H200 共享部署**的上限（~30 tok/s，相对 noSIA 88 tok/s 只到 34%）
- 后续要破吞吐天花板，**必须分卡**（2 张独立 GPU），详见 `doc/parallel-decoding-design.md`
- FP8 在共享 GPU 上没带来收益（甚至轻微回退），需要分卡才能真正生效

---

## 2026-05-14 13:50 — 早期 SIA 冒烟（--limit 3）

**测试**：首次跑通 SIA 端到端，每科只 3 题快速验证。PyTorch RM + Qwen3-14B + Qwen3-4B VM LoRA。
**命令来源**：`doc/eval-report.md`。
**相关分析**：[`doc/eval-report.md`](../doc/eval-report.md)。

```bash
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 --port 8001 \
    > log_rm_server_202605141350.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_server_202605141350.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_SIA_202605141350.json --limit 3 \
    > log_SIA_202605141350.txt 2>&1 &
```

---

## 2026-05-14 15:40 — **N1**: noSIA 冒烟（--limit 3）

**测试**：与同日 13:50 SIA limit=3 配对的 noSIA 对照。通过 `sia_vllm_server.py` 走同一推理 harness，只是 `--weight 0 --entropy_threshold 999999` 关掉所有干预。
**命令来源**：`doc/eval-report.md`。
**相关分析**：[`doc/eval-report.md`](../doc/eval-report.md)。
**只有 2 个文件**（LLM + eval），noSIA 不需要 RM server。

```bash
nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 0.0 --entropy_threshold 999999 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_server_noSIA_202605141540.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_noSIA_selfServer_202605141540.json --limit 3 \
    > log_noSIA_selfServer_202605141540.txt 2>&1 &
```

---

## 2026-05-14 17:40 — SIA 第一次全量（--limit 20）

**测试**：放大到每科 20 题（600 题），eager 模式无加速 flag。
**命令来源**：从 log 头反查（日志头无 Compile / CUDA graph / Profile 字段）。

```bash
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 --port 8001 \
    > log_rm_server_202605141740.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_server_202605141740.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_SIA_202605141740.json --limit 20 \
    > log_SIA_202605141740.txt 2>&1 &
```

---

## 2026-05-14 21:11 — SIA vs noSIA 首次对比（SIA 部分）

**测试**：与同时间戳的 noSIA (N2、N3) 配对的 SIA 一组。PyTorch RM。
**命令来源**：`doc/eval-report.md`、`doc/performance-report.md`。
**相关分析**：[`doc/eval-report.md`](../doc/eval-report.md)、[`doc/performance-report.md`](../doc/performance-report.md)。
**对照组**：见下文 N2（noSIA via `sia_vllm_server.py`）、N3（noSIA via vLLM 原生 api_server）。

```bash
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 --port 8001 \
    > log_rm_server_SIA_202605142111.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_server_SIA_202605142111.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_SIA_202605142111.json --limit 20 \
    > log_SIA_202605142111.txt 2>&1 &
```

---

## 2026-05-14 21:11 — **N2**: noSIA via `sia_vllm_server.py`（--limit 20）

**测试**：与同时间戳 SIA 配对，走 SIA 同一推理 harness 但关掉干预（`--weight 0`），是评估"SIA processor 自身额外 overhead"的对照。
**命令来源**：`doc/eval-report.md`、`doc/performance-report.md`。
**相关分析**：[`doc/eval-report.md`](../doc/eval-report.md)、[`doc/performance-report.md`](../doc/performance-report.md)。
**结果**：85.86 tok/s，74% 准确率。即便干预关掉，走 SIA harness 仍比纯 vLLM (N3) 慢 ~3%。

```bash
nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 0.0 --entropy_threshold 999999 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_server_noSIA_202605142111.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_noSIA_selfServer_202605142111.json --limit 20 \
    > log_noSIA_selfServer_202605142111.txt 2>&1 &
```

---

## 2026-05-14 21:11 — **N3**: noSIA via vLLM 原生 api_server（--limit 20）

**测试**：完全绕开 SIA harness，直接用 `python -m vllm.entrypoints.openai.api_server`。这是**最纯粹的 noSIA 速度上限**，所有后续 SIA 优化都拿这个数字做对照。
**命令来源**：`doc/performance-report.md`、`doc/vllm-rm-experiment-report.md`。
**相关分析**：[`doc/performance-report.md`](../doc/performance-report.md)、[`doc/vllm-rm-experiment-report.md`](../doc/vllm-rm-experiment-report.md)。
**结果**：**88.43 tok/s**，74.8% 准确率 —— **整个项目的速度天花板**。

```bash
nohup python -m vllm.entrypoints.openai.api_server \
    --model /workspace/SIA/models/Qwen3-14B \
    --gpu-memory-utilization 0.6 \
    --max-model-len 4096 \
    --override-generation-config '{"repetition_penalty": 1.3}' \
    --host 0.0.0.0 --port 8000 \
    > log_llm_server_noSIA_vllm_202605142111.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_noSIA_selfServer_vllm_202605142111.json --limit 20 \
    > log_noSIA_selfServer_vllm_202605142111.txt 2>&1 &
```

---

## 2026-05-15 — **A1**: AlpacaEval（noSIA vs SIA 双臂对比）

**测试**：在 805 条 Helpfulness 指令上验证 SIA 干预对生成质量的影响。**不是 MMLU**，用原论文 `evaluate.py`（单进程 LLM + RM 同卡），用独立的 Skywork-Reward-V2-Llama-3.1-8B 模型打分。
**结果**：mean reward 从 12.294 → 13.923（**+13.2%**），干预率 16.14%。
**相关分析**：[`doc/alpaca-eval-report.md`](../doc/alpaca-eval-report.md)、[`doc/alpaca-eval-case-analysis.md`](../doc/alpaca-eval-case-analysis.md)。
**归档文件**（4 个）：
- `log_alpaca_eval_weight0_Qwen3_14B.txt`（noSIA 生成 log）
- `log_alpaca_eval_weight1_Qwen3_14B.txt`（SIA 生成 log）
- `alpaca_base_scored_Qwen3-14B_202605152354.json`（Skywork 打分后的 noSIA 结果）
- `alpaca_sia_scored_Qwen3-14B_202605152354.json`（Skywork 打分后的 SIA 结果）

### 生成阶段（使用论文原始代码 `evaluate.py`）

**noSIA 基线**（`--rm_weight 0.0 --entropy_threshold 999999`，等同于关掉所有干预）：

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

**SIA**（`--rm_weight 1.0 --entropy_threshold 1.0`）：

```bash
cd /workspace/SIA/git/SIA

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

### 打分阶段（独立 Skywork RM 评分）

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

> 注：原始命令在 `/workspace/SIA/git/SIA` 目录下跑，生成的 4 个文件已 copy 至本目录。

---

## 2026-05-17 20:30 — batch forward 优化

**测试**：把 5 个 candidate 改成单次 batch forward 而非循环 5 次，期望 RM 端 forward 减半。
**命令来源**：`doc/batch-forward-optimization-report.md` appendix。
**相关分析**：[`doc/batch-forward-optimization-report.md`](../doc/batch-forward-optimization-report.md)。

```bash
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 --port 8001 \
    > log_rm_server_SIA_202605172030.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_server_SIA_202605172030.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_SIA_202605172030.json --limit 20 \
    > log_SIA_202605172030.txt 2>&1 &
```

---

## 2026-05-18 23:50 — 复测 eager 路径

**测试**：与 17:40 几乎相同配置，复测稳定性。Eager 模式，无加速 flag。
**命令来源**：log 头反查。

```bash
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 --port 8001 \
    > log_rm_server_SIA_202605182350.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_server_SIA_202605182350.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_SIA_202605182350.json --limit 20 \
    > log_SIA_202605182350.txt 2>&1 &
```

---

## 2026-05-19 08:00 — torch.compile 加速尝试

**测试**：在 PyTorch RM 上启用 `torch.compile(dynamic=True, fullgraph=False)`。启动时跑 3 次 warmup（standard / KV cache HIT / MISS batch=1）。
**命令来源**：log 头反查（"Applying torch.compile..." + 3 次 warmup pass）。

```bash
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 --port 8001 \
    --compile \
    > log_rm_server_SIA_202605190800.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_server_SIA_202605190800.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_SIA_202605190800.json --limit 20 \
    > log_SIA_202605190800.txt 2>&1 &
```

---

## 2026-05-19 23:18 — 复测 eager 路径（torch.compile 回退）

**测试**：torch.compile 路线最终未稳定，回到 eager。同 18:50 配置。
**命令来源**：log 头反查（无 Compile / CUDA graph 字段）。

```bash
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 --port 8001 \
    > log_rm_server_SIA_202605192318.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_server_SIA_202605192318.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_SIA_202605192318.json --limit 20 \
    > log_SIA_202605192318.txt 2>&1 &
```

---

## 2026-05-20 15:00 — 开 RM_PROFILE 跑 profiling

**测试**：开 `RM_PROFILE=True` 在 RM 服务端打印每次 score 的耗时分解（`[RM-pf-miss]` / `[RM-pf-hit]` 等行），为后续优化定位瓶颈。
**命令来源**：log 头反查 + `doc/rm-profiling-guide.md` §1.1。
**相关分析**：[`doc/rm-profiling-guide.md`](../doc/rm-profiling-guide.md)（profiling 启动指南）、[`doc/profiling-bottleneck-analysis.md`](../doc/profiling-bottleneck-analysis.md)（**基于本组日志的瓶颈分析与下一步优化路径**）。

```bash
# 注：RM_PROFILE 默认就是开的；显式写法 RM_PROFILE=1 也可
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 --port 8001 \
    > log_rm_server_SIA_202605201500.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_server_SIA_202605201500.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_SIA_202605201500.json --limit 20 \
    > log_SIA_202605201500.txt 2>&1 &
```

---

## 2026-05-21 00:50 — CUDA graph 加速实验

**测试**：在 PyTorch RM (`sia_rm_server.py`) 上启用静态 bucketing + CUDA graph 捕获，把 ~360 个 kernel launch 压成 1 次，目标把 HIT path forward 从 56ms 压到 5-10ms。
**命令来源**：`doc/rm-profiling-guide.md` §1.4 + log 头确认。
**相关分析**：[`doc/cuda-graph-debugging-journal.md`](../doc/cuda-graph-debugging-journal.md)（CG 踩坑全过程），[`doc/rm-profiling-guide.md`](../doc/rm-profiling-guide.md)（CG 启动方式）。
**注**：实际日志文件名为 `*_202505210050.txt`（年份 typo，应为 `202605210050`）；按 file birth time（2026-05-20 16:52 UTC ≈ 北京时间 2026-05-21 00:52）排序至此。

```bash
nohup python src/sia_rm_server.py \
    --rm /workspace/SIA/models/Qwen3-4B \
    --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
    --rm_device cuda:0 \
    --port 8001 \
    --cuda_graph --cg_batch 5 --cg_diff_max 64 --cg_kv_max 2048 \
    > log_rm_cg_202505210050.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_cg_202505210050.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_cg_202505210050.json --limit 20 \
    > log_SIA_cg_202505210050.txt 2>&1 &
```

---

## 2026-05-22 10:45 — **G1 Baseline**：切换到 vLLM RM

**测试**：RM 从手写 PyTorch FastAPI 改为 `vllm serve` 原生部署（`/classify` endpoint），观测端到端延迟和准确率。**这是后续 G2-G4 的 baseline 起点**。
- RM gpu_memory_utilization 0.3
- RM max-model-len 4096
- chunked_prefill 默认开
- LLM gpu_memory_utilization 0.6

**相关分析**：[`doc/vllm-rm-experiment-report.md`](../doc/vllm-rm-experiment-report.md)。

```bash
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling --convert classify \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.3 \
    --max-model-len 4096 \
    --port 8001 \
    > log_vllm_rm_202605221045.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --rm_backend vllm \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_vllmrm_202605221045.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_vllmrm_202605221045.json --limit 20 \
    > log_SIA_vllmrm_202605221045.txt 2>&1 &
```

---

## 2026-05-25 09:45 — **G2**：RM 调度参数优化

**测试**：在 G1 基础上调三个 RM 端参数：
- 关 chunked prefill（pooling 任务用不上）
- gpu_memory_utilization 0.3 → 0.4（多 14 GB KV cache 池）
- max-model-len 4096 → 2048（调度元数据减半）

**命令来源**：从 log 反查（LLM mem 在日志里未打印，按惯例沿用 G1 的 0.6，未确证）。
**相关分析**：[`doc/vllm-rm-followup-optimizations.md`](../doc/vllm-rm-followup-optimizations.md) §4.2。

```bash
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling --convert classify \
    --enable-prefix-caching --no-enable-chunked-prefill \
    --gpu-memory-utilization 0.4 \
    --max-model-len 2048 \
    --port 8001 \
    > log_vllm_rm_opt_202605250945.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --rm_backend vllm \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.6 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_vllmrm_202605250945.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_vllmrm_202605250945.json --limit 20 \
    > log_SIA_vllmrm_202605250945.txt 2>&1 &
```

---

## 2026-05-25 21:20 — **G3**：客户端代码 3 件套 + LLM mem 调整

**测试**：在 G2 之上叠加客户端代码层 3 个优化（commits `a2357e9` / `368ef24` / `c802054`）：
1. 去掉每 token 的 `entropy.item()` CUDA 同步
2. response_so_far 增量解码缓存（O(n²) → O(1)）
3. top-1 flip 统计日志（per INTERVENE + per request）

LLM mem 同时 0.6 → 0.55 给 RM 让出 5%。
**相关分析**：[`doc/vllm-rm-followup-optimizations.md`](../doc/vllm-rm-followup-optimizations.md) §4.3。

```bash
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling --convert classify \
    --enable-prefix-caching --no-enable-chunked-prefill \
    --gpu-memory-utilization 0.4 \
    --max-model-len 2048 \
    --port 8001 \
    > log_vllm_rm_opt_202605252120.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --rm_backend vllm \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.55 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_vllmrm_202605252120.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_vllmrm_202605252120.json --limit 20 \
    > log_SIA_vllmrm_202605252120.txt 2>&1 &
```

---

## 2026-05-26 10:30 — **G4**：FP8 + FP8 KV cache

**测试**：在 G3 之上把 RM 权重和 KV cache 都量化到 FP8：
- `--quantization fp8`（权重 BF16 → FP8，GPU 占用 7.5 GB → 4.1 GB）
- `--kv-cache-dtype fp8`（KV 也 FP8）
- RM gpu_memory_utilization 0.4 → 0.3（FP8 后不需要那么多）

**注意**：log 里有 3 条 WARNING：`q_scale` / `k_scale` / `prob_scale` 均未校准，默认 1.0，可能触发 attention 退化路径。
**相关分析**：[`doc/vllm-rm-followup-optimizations.md`](../doc/vllm-rm-followup-optimizations.md) §4.4。

```bash
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling --convert classify \
    --enable-prefix-caching --no-enable-chunked-prefill \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 2048 \
    --port 8001 \
    > log_vllm_rm_fp8_kvfp8_202605261030.txt 2>&1 &

nohup python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-14B \
    --rm_url http://localhost:8001 \
    --rm_backend vllm \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.55 \
    --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > log_llm_vllmrm_202605261030.txt 2>&1 &

nohup python eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 --model /workspace/SIA/models/Qwen3-14B \
    --output results/test_vllmrm_202605261030.json --limit 20 \
    > log_SIA_vllmrm_202605261030.txt 2>&1 &
```

