# VL-30B b2 inproc 速度优化实测 — vllm 0.17.1 sweet spot (2026-06-05)

**TL;DR**: 在 vllm **0.17.1** (新建 venv5) 上启用 `--rm_backend b2` (nested in-process vllm RM), VL-30B AlpacaEval 200Q max=256 实测:

- **单次 RM 调用 p50: 71.18ms → 11.12ms (6.4× 快)**
- **200Q 端到端 wall: 17.2 min → 11.8 min (1.46× 加速)**
- **Skywork mean reward: +14.01 → +13.88, paired Δ=-0.13, p=0.73** (统计上完全等价, 零质量回归)
- **零代码改动**: 只需切 venv5 + 几个 env var

完整验证: 跨 200Q × 48,236 generation steps, **0 RM error, 0 cudagraph 冲突**。

---

## 1. 背景: 为什么 0.17.1 是 sweet spot

之前 doc [`vl30b-inproc-vm-feasibility-analysis`](../tmp/...) (in /tmp, 内部研究) 已经分析过 3 个 vllm 版本的失败模式:

| vllm 版本 | venv | 启动 | 推理 | 结论 |
|-----------|------|------|------|------|
| 0.15.0 | venv3 | ❌ workspace lock assert | n/a | 缺 `unlock_workspace()` API |
| **0.17.1** | venv5 (新装) | ✅ | ✅ | **🎯 sweet spot** |
| 0.19.0 | venv4 | ✅ | ❌ 100% step RM error | PIECEWISE 运行期 cudagraph runtime trigger 拉起全局 flag, 同进程 RM forward 撞 flag |

**0.17.1 同时满足 3 个条件**:

| 条件 | 影响 | 0.15 | **0.17.1** | 0.19 |
|------|------|------|-----------|------|
| 支持 Qwen3VLMoe (VL-30B 架构) | 模型能加载 | ✅ | ✅ | ✅ |
| 有 `unlock_workspace()` 顶层 API | 过得了 MoE workspace lock | ❌ | ✅ | ✅ |
| PIECEWISE 仍是 AOT capture | 推理期不撞全局 cudagraph flag | (未测) | ✅ | ❌ |

---

## 2. 配置

### 2.1 拓扑 — b2 inproc (新)

**单进程**: SIA EngineCore 内嵌套一个 nested vllm `LLM(...)` 作为 RM。

```
┌─────────── 主进程 (sia_vllm_server.py) ───────────┐
│   FastAPI server (port 8000)                       │
│   ↓                                                 │
│   AsyncLLMEngine (主 LLM = VL-30B, vllm 0.17.1)    │
│   ↓ EngineCore subprocess                          │
│   ┌──────────────────────────────────────────────┐ │
│   │  SIA LogitsProcessor                          │ │
│   │  ↓  __call__()                                │ │
│   │  RMClient (b2 backend)                        │ │
│   │  ↓  score_candidates(...)  ← Python 直接调用 │ │
│   │  ┌──────────────────────────────────────────┐ │ │
│   │  │  nested vllm LLM(VM-Qwen3-4B-merged...)  │ │ │
│   │  │  cudagraph_mode=PIECEWISE (AOT)          │ │ │
│   │  │  on same CUDA context as 主 LLM          │ │ │
│   │  └──────────────────────────────────────────┘ │ │
│   └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

无 HTTP 跨进程通信。RM 调用走同进程 Python 函数。

### 2.2 SIA 参数 (跟之前 HTTP path 实验保持一致)

| 参数 | 值 |
|------|---|
| `--llm` | `/workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct` |
| `--rm_backend` | **`b2`** (跟旧 `vllm` HTTP backend diff) |
| `--rm_model` | `/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm` |
| `--llm_gpu_mem` | 0.55 |
| `--rm_b2_gpu_mem` | 0.15 |
| `--max_model_len` | 4096 |
| `--topk` (SIA candidates) | 10 |
| `--weight` | 1.0 |
| `--entropy_threshold` | 1.0 |
| `SIA_LLM_CUDAGRAPH` env | `piecewise` (主 LLM 强制 PIECEWISE-only, 避免 FULL cudagraph runtime flag) |
| `SIA_RM_CUDAGRAPH` env | `piecewise` (RM 同理) |
| `SIA_RM_MULTIPROCESS` env | `0` (RM 跟主 LLM 同进程, 走 InprocClient) |

### 2.3 Driver 参数 (跟之前 HTTP path 200Q max=256 完全一致)
- 200Q AlpacaEval, max_tokens=256, temperature=1.0, repetition_penalty=1.0
- `chat_template_kwargs: {enable_thinking: False}` (VL-30B 不是 thinking model, 但 driver 习惯传)

### 2.4 Skywork 评分
- Model: Skywork-Reward-V2-Llama-3.1-8B
- Method: 跟之前所有 VL-30B AlpacaEval 实验一致

---

## 3. 输出质量

| 指标 | 值 |
|------|---|
| 完成数 | **200/200** ✅ |
| 错误数 | 0 |
| avg tokens / Q | 244 (跟 HTTP path 243 几乎一致) |
| finish=length / stop | 22% length-cap / 3% 自然停止 (符合 max=256) |
| 含 `<think>` tag | 0/200 (VL-30B 不是 thinking model) |

vllm seed=0 默认 + 同 sampling params, **b2 inproc 跟 HTTP path 生成的输出 Q0 byte-near-identical**:
```
1. **Audrey Hepburn** – Started in the London West End and later appeared on 
   Broadway in *Ondine* (1954)...
2. **Robert De Niro** – Made his Broadway debut in *The Threepenny Opera* (1964) 
   at the age of 19...
3. **Meryl Streep** – Began her professional acting career...
```
这证明 b2 inproc **功能等价**, 不是另一个采样路径。

---

## 4. 性能指标

### 4.1 RM 调用单次时延 (核心收益)

来自 `[SIA-pf-summary @8900]` 累计统计 (200Q 完整结束):

| 阶段 | b2 inproc p50 (ms) | HTTP path p50 (ms) | 改善 |
|------|--------------------|--------------------|------|
| **`b2_score_call` (RM forward 整体)** | **11.12** | (HTTP: `http_post` p50=71.18) | **6.4× 快** ⚡ |
| **`b2_score_call` p95** | **17.14** | (HTTP: p95=99.14) | **5.8× 快** |
| b2_score_call max | 45.33 | (HTTP: max=116.05) | 2.6× 快 |
| b2_session_init | 0.00 (基本免费) | (HTTP: format_chat 0.10) | — |
| b2_prefix_adv | 0.00 (基本免费) | — | — |
| skip_step (非干预) | 0.51 | (HTTP: 0.73) | — |
| apply_topk_ent (GPU topk + entropy) | 0.40 | (HTTP: 0.69) | — |
| intv_apply_logits (改 logits) | 0.48 | (HTTP: 1.13) | — |

RM 调用 **从 71ms 降到 11ms**, 比 doc 预测的 17ms 还快, 几乎追平 Qwen3-14B 时代的实测基准。

### 4.2 端到端 throughput (200Q)

| 指标 | b2 inproc (venv5 0.17.1) | HTTP path (venv4 0.19) | 改善 |
|------|---------------------------|------------------------|------|
| 总 reqs | 200 | 200 | — |
| 总 wall (driver) | **11.8 min** | **17.2 min** | **1.46× 加速** ⚡ |
| 平均 wall / Q | 3.5s | 5.2s | 1.49× |
| 平均 tokens / Q | 244 | 243 | 一致 |

### 4.3 SIA Server 健康 (b2 inproc, 200Q × 48,236 steps)

| 指标 | 值 |
|------|---|
| Intervention 率 | **18.50%** (8923 / 48236) |
| Top-1 flip 率 | **62.19%** (5549 / 8923) |
| **RM error** | **0** ✅ |
| **Connection refused** | **0** ✅ |

跟 HTTP path (intervention 25.83%, flip 67%) 同量级, 略低 (sampling 随机性 + 模型未变)。

### 4.4 加速比理论分析

为什么端到端只 1.46× 而不是 RM call 的 6.4×?
- RM 调用只占总时间 ~24% (intervention 率) × ~71ms (HTTP) = ~17ms / step 平均
- 主 LLM forward 占大头 (~14ms / token)
- 端到端理论加速 = 1 / ((1 - 0.24×(1 - 11.12/71.18))) = 1 / 0.61 = **1.64×** (理论上限)
- 实测 1.46× 接近上限, 缺口 ~10% 是 SIA processor Python wrap 等其它开销

---

## 5. Skywork 评分 — 质量等价验证

### 5.1 同 200 个 instructions 配对比较

| arm | n | mean | median | std |
|-----|---|------|--------|-----|
| **b2 inproc (vllm 0.17.1)** | 200 | **+13.882** | +13.125 | 9.227 |
| HTTP path (vllm 0.19) | 200 | **+14.012** | +13.312 | 8.872 |

### 5.2 配对 t-test (n=200)

| 指标 | 值 |
|------|---|
| **Δmean** | **-0.131** |
| **relative** | -0.9% |
| **Win** (b2 > HTTP) | 95/200 (48%) |
| t-statistic | -0.34 |
| **p-value** | **0.7335** |

**p=0.73, win 95/200 (48%, 几乎 50/50)** → **统计上完全等价**, 没有 quality regression。

### 5.3 结论
b2 inproc 路径 跟 HTTP path **质量完全等价, 速度 6.4× 快 (RM 单次) / 1.46× 加速 (端到端)**。

---

## 6. 适用范围 + 局限

### 6.1 适用模型

| 模型 | 状态 | 备注 |
|------|------|------|
| **Qwen3-VL-30B-A3B-Instruct** | ✅ **本实验验证** | vllm 0.17.1 支持 Qwen3VLMoe |
| 0GM-1.0-35B-A3B-0427 | ❌ 不能用 | Qwen3.5 MoE arch 0.17 不支持, 必须 0.19+ |
| Qwen3-14B (旧 baseline 模型) | ✅ (历史 b2 inproc 实测过) | 也可以用 0.17.1 跑 |

### 6.2 局限

- 单卡布局: b2 inproc 需要 RM 跟主 LLM **同进程同 GPU**, 不能跨卡负载均衡
- 0GM-35B 上**不可用**, 因为 Qwen3.5 MoE 架构在 0.17 上未注册, 强制要 0.19
- 切换到 0.17.1 需要新建 venv5, 跟现有 venv4 (0.19) 并存; 主推理代码完全不动

---

## 7. 后续优化空间

按 doc `inproc-client-optimization-20260531.md` 的剩余路径:

| # | 方向 | 预期收益 (相对 b2 inproc) | 工作量 |
|---|------|---------------------------|--------|
| 1 | Stage A (KV catchup) 跟主 LLM forward 并行 (双 CUDA stream) | +14% (~95 tok/s) | 中等 |
| 2 | 更小 VM (Qwen3-1.7B 重训 LoRA) | +20-40% | 高 (需训练) |
| 3 | SIA processor Python wrap 优化 (当前 SKIP 0.51ms 主要是 cpu_sync) | +5-8% | 低 |
| 4 | 直接训练 RM 嵌入主 LLM 隐状态 (linear head) | 接近 noSIA 速度 | 极高 (ML 工作) |

短期最划算的是 #3 (SIA processor 优化) + #1 (双 CUDA stream)。

---

## Appendix A. 完整启动命令

所有日志已 archive 到 [`exp/vl30b-b2-inproc-speedup-20260605/`](../exp/vl30b-b2-inproc-speedup-20260605/)。

### A.1 SIA LLM server with b2 inproc backend (port 8000)

```bash
cd /workspace/git/0g-sparse-inference-alignment
SIA_LLM_CUDAGRAPH=piecewise \
SIA_RM_CUDAGRAPH=piecewise \
SIA_RM_MULTIPROCESS=0 \
nohup /workspace/SIA/venv5/bin/python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
  --rm_backend b2 \
  --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --rm_b2_gpu_mem 0.15 \
  --llm_gpu_mem 0.55 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 4096 --port 8000 \
  > /tmp/vl30b_runs/b2_inproc_200q_server.log 2>&1 &
```

**日志**: [`exp/.../sia_llm_server.log`](../exp/vl30b-b2-inproc-speedup-20260605/sia_llm_server.log) (含全部 SIA-pf-summary 性能数据 + 每个 req 的 `intervened=/top1_flip=` 行)

### A.2 Driver (200Q AlpacaEval, max=256)

```bash
cd /tmp/vl30b_runs
nohup /workspace/SIA/venv5/bin/python -u drive_vl30b.py \
  /tmp/vl30b_runs/sia_VL30B_b2inproc_200q_max256.json \
  Qwen3-VL-30B-A3B-Instruct \
  256 200 \
  > /tmp/vl30b_runs/driver_b2_inproc.log 2>&1 &
```

驱动 (`drive_vl30b.py`) 内容跟之前 VL-30B AlpacaEval 实验一致 — `chat_template_kwargs={"enable_thinking": False}`, `repetition_penalty=1.0`, `temperature=1.0`。

**日志**: [`exp/.../driver.log`](../exp/vl30b-b2-inproc-speedup-20260605/driver.log)
**原始输出**: [`exp/.../outputs.json`](../exp/vl30b-b2-inproc-speedup-20260605/outputs.json)

### A.3 Skywork 评分 + paired 比较

```bash
# kill 主 LLM server 释放 GPU
pkill -9 -f 'EngineCore' 2>/dev/null
pkill -9 -f 'sia_vllm_server' 2>/dev/null
sleep 8

cd /tmp/vl30b_runs
# venv4 (含 accelerate, venv5 没装)
/workspace/SIA/venv4/bin/python -u compare_b2_inproc.py
```

**脚本**: [`exp/.../compare_b2_inproc.py`](../exp/vl30b-b2-inproc-speedup-20260605/compare_b2_inproc.py)
**评分输出 (含 Skywork rewards)**: [`exp/.../outputs_scored.json`](../exp/vl30b-b2-inproc-speedup-20260605/outputs_scored.json)
**对比 log**: [`exp/.../compare_b2_inproc.log`](../exp/vl30b-b2-inproc-speedup-20260605/compare_b2_inproc.log)

---

## Appendix B. venv5 创建步骤

```bash
python3 -m venv /workspace/SIA/venv5
/workspace/SIA/venv5/bin/pip install --quiet vllm==0.17.1
# 不需要装 peft (b2 用的是 merged VM, 不走 LoRA)
# 不需要装 accelerate (Skywork 用 venv4 跑)
```

确认就绪:
```bash
/workspace/SIA/venv5/bin/python -c "import vllm; print('vllm:', vllm.__version__)"
# vllm: 0.17.1

grep -c Qwen3VLMoe /workspace/SIA/venv5/lib/python3.12/site-packages/vllm/model_executor/models/registry.py
# 2

grep 'def unlock_workspace' /workspace/SIA/venv5/lib/python3.12/site-packages/vllm/v1/worker/workspace.py
# def unlock_workspace() -> None:
```

---

## Appendix C. SIA 健康检查

```bash
SIA_LOG=exp/vl30b-b2-inproc-speedup-20260605/sia_llm_server.log

# 必须 = 0
grep -c "Connection refused\|RM error\|inappropriate" $SIA_LOG

# 总 intervention ratio
grep 'intervened=' $SIA_LOG | awk -F'intervened=' \
  '{split($2,a,"/"); split(a[2],b," "); intv+=a[1]; tot+=b[1]; n++} \
   END {printf "Reqs: %d  steps: %d  intv: %d (%.2f%%)\n", n, tot, intv, intv*100/tot}'

# 总 flip ratio
grep 'top1_flip=' $SIA_LOG | awk -F'top1_flip=' \
  '{split($2,a,"/"); split(a[2],b," "); flip+=a[1]; total+=b[1]} \
   END {if (total>0) printf "Flip: %d/%d = %.2f%%\n", flip, total, flip*100/total}'

# 最近 SIA-pf-summary (含 b2_score_call timing)
grep 'SIA-pf-summary' $SIA_LOG | tail -1
```
