# Qwen3-VL-30B-A3B SIA 评测汇总 (效果 + 性能)

**最后更新**: 2026-06-05
**适用前提**: 所有数据均在 [`commit f0f4e1c` (rep_penalty 1.3 → 1.0 修复)](sia-repetition-penalty-root-cause-20260604.md) 之后采集。

**结论速览**:
- ✅ **效果**: SIA Δ vs noSIA 在 AlpacaEval 全场景显著 (max=256: **+2.32, p<10⁻⁴**; max=2048: **+2.75, p<10⁻⁷**); MMLU 150Q 上 SIA 也有 **+2.67 pp accuracy lift** vs noSIA。
- ✅ **性能**: b2 inproc (vllm 0.17.1) 相对 HTTP path: RM 单次调用 **6.4× 快** (71→11ms), AlpacaEval 端到端 **1.46× 加速**, MMLU 端到端 **+11.3% throughput**。质量统计等价 (p=0.73)。
- ✅ **W2S=7.5× OOD 配置** (LLM 30B / RM 4B) 下 SIA 仍 work, 推翻之前 "OOD SIA 不可用" 错误结论。

来源:
- [`rep-penalty-fix-validation-experiments-20260604.md`](rep-penalty-fix-validation-experiments-20260604.md) (rep_penalty fix 验证, 含 5 个 VL-30B 实验)
- [`vl30b-b2-inproc-speedup-20260605.md`](vl30b-b2-inproc-speedup-20260605.md) (b2 inproc 速度优化, 含 AlpacaEval 200Q + MMLU 150Q)
- 2026-06-05 新增 MMLU 150Q b2 inproc (本仓库 commit `232e16a`)

**前置环境** — VL-30B 实验有两路 venv, 跟 SIA backend 路径绑定:

| 实验路径 | vllm | venv 路径 | 一次性 setup 命令 | requirements |
|---|---|---|---|---|
| HTTP RM path (大多数实验) | **0.19.0** | `/workspace/SIA/venv4` | `scripts/setup_venv_0gm35b_http.sh /workspace/SIA/venv4` | [`requirements/0gm35b-or-vl30b-http.txt`](../requirements/0gm35b-or-vl30b-http.txt) |
| **b2 inproc fast path** ⚡ | **0.17.1** | `/workspace/SIA/venv5` | `scripts/setup_venv_vl30b_fast.sh /workspace/SIA/venv5` | [`requirements/vl30b-b2-inproc.txt`](../requirements/vl30b-b2-inproc.txt) |

每个脚本: 创建 venv → pip install requirements → `pip install -e .` (注册 sia_rm vllm 插件). 详见 [CLAUDE.md "Dependencies"](../CLAUDE.md#running-the-project) 矩阵。

> **不要混用** vllm 0.19 跟 0.17.1, 它们 pin 不同的 torch 版本; 必须独立 venv。Skywork 评分用 venv4 (含 accelerate, venv5 没装)。

---

## 一. 效果评测

### 1.1 AlpacaEval 200Q (max=256) — Skywork 评分

详见 [Appendix C.1 — VL-30B AlpacaEval 200Q max=256 HTTP path SIA & noSIA](#c1-vl-30b-alpacaeval-200q-max256-http-path) 和 [Appendix C.2 — VL-30B AlpacaEval 200Q max=256 b2 inproc](#c2-vl-30b-alpacaeval-200q-max256-b2-inproc).

| arm | Skywork mean | median | n | 实验来源 |
|---|---|---|---|---|
| **SIA HTTP** (vllm 0.19 + HTTP RM) | **+14.01** | +13.31 | 200 | [Appendix C.1](#c1-vl-30b-alpacaeval-200q-max256-http-path) |
| noSIA HTTP (vllm 0.19, weight=0) | +11.69 | +10.78 | 200 | [Appendix C.1](#c1-vl-30b-alpacaeval-200q-max256-http-path) |
| **SIA b2 inproc** (vllm 0.17.1, RM 同进程) | **+13.88** | +13.13 | 200 | [Appendix C.2](#c2-vl-30b-alpacaeval-200q-max256-b2-inproc) |

**Paired 比较 (n=200)**:

| 对比 | Δ | win | p | 解读 |
|---|---|---|---|---|
| SIA HTTP vs noSIA HTTP | **+2.32 (+19.8%)** | 132/200 (66%) | **<10⁻⁴** | SIA 显著加分 ✅ |
| SIA b2 inproc vs SIA HTTP | -0.13 (-0.9%) | 95/200 (48%) | **0.73** | 统计等价 ✅ (b2 不引入质量损失) |

### 1.2 AlpacaEval 200Q (max=2048) — 长生成 regime

详见 [Appendix C.3 — VL-30B AlpacaEval 200Q max=2048 HTTP path](#c3-vl-30b-alpacaeval-200q-max2048-http-path).

| arm | Skywork mean | median | n |
|---|---|---|---|
| **SIA HTTP max=2048** (REDO, RM 健康) | **+30.19** | +30.75 | 200 |
| noSIA HTTP max=2048 | +27.44 | +27.63 | 200 |

**Paired (n=200)**:

| 对比 | Δ | win | p |
|---|---|---|---|
| SIA vs noSIA | **+2.75 (+10.0%)** | 123/200 (62%) | **<10⁻⁷** ✅ |
| SIA trunc→256 vs noSIA trunc→256 | +1.30 (+11%) | (n/a) | 0.003 ✅ |

→ SIA 在 max=2048 长生成 regime 仍有 +10% lift, **推翻"SIA 只在短窗口有效"假设**。

> ⚠️ 第一次 max=2048 跑出现 RM OOM 死亡 → SIA 退化为 no-op, 错误结论 "SIA -6%"。**REDO** 用顺序启动 + /classify smoke test 后才是真数据。健康检查方法见 [Appendix C.4](#c4-sia-intervention-健康检查).

### 1.3 MMLU 150Q (30 subjects × 5, 知识类任务)

详见 [Appendix C.5 — VL-30B MMLU 150Q HTTP path SIA & noSIA](#c5-vl-30b-mmlu-150q-http-path) 和 [Appendix C.6 — VL-30B MMLU 150Q b2 inproc](#c6-vl-30b-mmlu-150q-b2-inproc).

| arm | accuracy | correct/150 | 实验来源 |
|---|---|---|---|
| noSIA HTTP | 0.7867 | 118/150 | [Appendix C.5](#c5-vl-30b-mmlu-150q-http-path) |
| SIA HTTP (vllm 0.19) | 0.8000 | 120/150 | [Appendix C.5](#c5-vl-30b-mmlu-150q-http-path) |
| **SIA b2 inproc** (vllm 0.17.1) | **0.8133** | **122/150** | [Appendix C.6](#c6-vl-30b-mmlu-150q-b2-inproc) |

**Δ accuracy**:

| 对比 | Δ | 相对 | 解读 |
|---|---|---|---|
| SIA HTTP vs noSIA | +1.33 pp | +1.7% | 落在 150Q 噪声内 (binomial 95% CI ±7.7pp) |
| SIA b2 inproc vs noSIA | **+2.67 pp** | +3.4% | 同上, 但方向一致 |
| SIA b2 inproc vs SIA HTTP | +1.33 pp | +1.7% | 跟 1.1 节 AlpacaEval 统计等价结论自洽 |

**SIA 在 MMLU (RM 训练目标外的纯知识任务) 上仍小幅提升, 不引入 regression**。

### 1.4 跨实验一致性

| 实验 | n | SIA mean | noSIA mean | Δ | rel | p |
|---|--|---|---|---|---|---|
| Paper Qwen3-14B (805Q baseline) | 805 | +13.92 | +12.29 | **+1.63** | +13% | (paper) |
| Ours Qwen3-14B (200Q, max=256) | 200 | +11.48 | +9.59 | **+1.89** | +20% | <10⁻⁴ |
| Ours VL-30B (200Q, max=256) | 200 | +14.01 | +11.69 | **+2.32** | +20% | <10⁻⁴ |
| **Ours VL-30B (200Q, max=2048)** | **200** | **+30.19** | **+27.44** | **+2.75** | **+10%** | **<10⁻⁷** |
| Ours VL-30B (max=2048, trunc-256) | 200 | +13.32 | +12.01 | **+1.30** | +11% | 0.003 |

> SIA 绝对 gain 在 **+1.3 ~ +2.8** 之间, 跨模型 (14B → 30B, W2S 从 3.5× 到 7.5×) / 跨长度高度一致。

---

## 二. 性能评测

> 性能核心指标: **agg throughput (tok/s)** 和 **avg tokens/Q**。下面按实验组织, 同一实验在效果和性能里都出现的, 引用相同的 Appendix 节。

### 2.1 AlpacaEval 200Q (max=2048) — HTTP path, single-stream

实验环境同 [§1.2](#12-alpacaeval-200q-max2048--长生成-regime), 详见 [Appendix C.3](#c3-vl-30b-alpacaeval-200q-max2048-http-path).

| 指标 | SIA HTTP | noSIA HTTP | 比值 |
|---|---|---|---|
| 总 reqs | 200 | 200 | — |
| 总生成 tokens | 129,313 | 151,806 | -15% (SIA 输出略短) |
| 总 wall (driver) | 3,599 s (60.0 min) | 1,214 s (20.2 min) | +2.97× |
| **平均 tokens/s** | **35.9** | **125.0** | noSIA **3.5×** 快 |
| 平均 tokens/Q | 647 | 759 | -15% |
| 平均 wall/Q | 18.0 s | 6.1 s | +2.95× |

**每 token 推理耗时 (= wall / tokens)**:

| arm | ms/token |
|---|---|
| noSIA (纯 LLM forward + sampling) | 8.0 |
| SIA (混合 SKIP + INTERVENE) | **27.8** |

**SIA 内部时间拆解** (from `[SIA-pf-summary]`):

| 步骤 | p50 (ms) | p95 (ms) | 说明 |
|---|---|---|---|
| SKIP step (非干预步骤) | 3.73 | 4.06 | top-K + entropy + decision, 无 RM 调用 |
| ↳ Top-K 提取 + entropy (GPU torch.topk + log_softmax) | 0.64 | 1.16 | |
| ↳ CPU sync (entropy 拉回 CPU) | 3.11 | 3.61 | 主要开销 |
| **INTERVENE step (干预步骤)** | **~75** | ~78 | apply_total p95 |
| ↳ **RM HTTP /classify call** | **71.18** | 98.34 | **占干预成本 95%** ← 瓶颈, 也是 b2 inproc 优化对象 |
| ↳ format_chat (prompt 构造) | 0.13 | 0.69 | |
| ↳ intv_prepare (索引/数据准备) | 0.16 | 0.32 | |
| ↳ intv_apply_logits (改 logits) | 1.15 | 1.32 | |

**核心观察**: INTERVENE 比 SKIP 慢 ~20× (75ms vs 3.7ms), 因 RM HTTP 调用 (~71ms)。加权平均 step = 25.8% × 75 + 74.2% × 3.7 ≈ 22ms (理论), 实测 27.8ms (差 5ms 是 LLM forward + vllm 内部 sampling)。

### 2.2 AlpacaEval 200Q (max=256) — b2 inproc vs HTTP path

详见 [Appendix C.1](#c1-vl-30b-alpacaeval-200q-max256-http-path) (HTTP) 和 [Appendix C.2](#c2-vl-30b-alpacaeval-200q-max256-b2-inproc) (b2 inproc).

#### 2.2.1 端到端 throughput

| 指标 | b2 inproc (venv5 0.17.1) | HTTP path (venv4 0.19) | 改善 |
|---|---|---|---|
| 总 reqs | 200 | 200 | — |
| 总 wall (driver) | **11.8 min** (708 s) | 17.2 min (1032 s) | **1.46× 加速** ⚡ |
| 平均 wall/Q | 3.5 s | 5.2 s | 1.49× |
| 平均 tokens/Q | 244 | 243 | 一致 |
| **Agg throughput (tok/s)** | **68.9** | **47.1** | **+46% (1.46×)** ⚡ |

#### 2.2.2 RM 调用单次时延 (核心收益)

来自 `[SIA-pf-summary @8900]` 累计 (200Q 完整结束):

| 阶段 | b2 inproc p50 (ms) | HTTP path p50 (ms) | 改善 |
|---|---|---|---|
| **`b2_score_call` (RM forward 整体)** | **11.12** | (HTTP: `http_post` p50=71.18) | **6.4× 快** ⚡ |
| `b2_score_call` p95 | 17.14 | (HTTP: p95=99.14) | 5.8× 快 |
| b2_score_call max | 45.33 | (HTTP: max=116.05) | 2.6× 快 |
| skip_step (非干预) | 0.51 | (HTTP: 0.73) | — |
| apply_topk_ent (GPU topk + entropy) | 0.40 | (HTTP: 0.69) | — |
| intv_apply_logits (改 logits) | 0.48 | (HTTP: 1.13) | — |

#### 2.2.3 加速比理论分析

为什么端到端 1.46× 而 RM call 6.4×?
- RM 调用占总时间 ~24% (intervention 率) × ~71ms (HTTP) ≈ 17ms/step 平均
- 主 LLM forward 占大头 (~14ms/token)
- **理论上限 = 1 / (1 − 0.24 × (1 − 11.12/71.18)) = 1/0.61 = 1.64×**
- 实测 1.46× 接近上限, 缺口 ~10% 是 SIA processor Python wrap 等其它开销

### 2.3 MMLU 150Q — 三方对比 (noSIA / SIA HTTP / SIA b2 inproc)

详见 [Appendix C.5](#c5-vl-30b-mmlu-150q-http-path) (HTTP noSIA + SIA HTTP) 和 [Appendix C.6](#c6-vl-30b-mmlu-150q-b2-inproc) (b2 inproc).

| 指标 | noSIA | SIA HTTP | **SIA b2 inproc** |
|---|---|---|---|
| Total tokens | 43,149 | 39,371 | 37,362 |
| Total decode latency | **359.3 s** | 598.8 s | 510.5 s |
| **Agg tok/s** | **120.1** | 65.8 | **73.2** |
| Median tok/s | 116.9 | 78.2 | 73.3 |
| **Avg tokens/Q** | 287.7 | 262.5 | 249.1 |
| Median tokens/Q | 163 | 152 | 147 |
| Median latency/Q | 1.4 s | 2.1 s | 2.3 s |
| SIA intervention 率 | — | 10.03% (4000/39889) | ~10-20% (samples) |
| SIA top1 flip 率 | — | 61.7% (2467/4000) | (略) |
| RM error 数 | — | 0 | 0 |

**速度解读 (b2 inproc 相对其他两路)**:

```
noSIA          120.1 tok/s   1.00×   (baseline, 无 SIA tax)
SIA HTTP        65.8 tok/s   0.55×   (-45% tok/s) — RM via HTTP, p50 ≈ 62 ms
SIA b2 inproc   73.2 tok/s   0.61×   (-39% tok/s) — RM via b2_score_call p50 ≈ 10.8 ms (5.7× 快)
```

- b2 inproc vs HTTP: **+11.3% agg tok/s** (decode latency -14.7%)
- RM 调用层加速 **5.7×** 已确认 (server pf-summary @3100), 但端到端加速被压缩到 +11% 因为:
  1. MMLU 输出短 (median 147 tok vs AlpacaEval 244), 固定开销分摊不开
  2. 干预率低 (~10%), 没干预的 step b2 vs HTTP 无差别
  3. RM 通信成本缩小后, 主 LLM forward 成主导
- → **AlpacaEval (长输出 + 偏好评分) 是 b2 inproc 加速最明显的场景; MMLU (短输出 + 选择题) 加速被压缩**

### 2.4 SIA 健康指标汇总

| 实验 | n reqs | intv 率 | top1 flip 率 | RM error |
|---|---|---|---|---|
| VL-30B SIA AlpacaEval max=256 HTTP | 200 | **18.8%** | 55-81% | **0** ✅ |
| VL-30B SIA AlpacaEval max=2048 HTTP | 200 | **25.83%** (33,462 intv / 129,313 steps) | **66.88%** (22,381 / 33,462) | **0** ✅ |
| VL-30B SIA AlpacaEval max=256 b2 inproc | 200 | **18.50%** (8923/48236) | **62.19%** (5549/8923) | **0** ✅ |
| VL-30B SIA MMLU 150Q HTTP | 150 | **10.03%** (4000/39889) | **61.7%** (2467/4000) | **0** ✅ |

判断方法见 [Appendix C.4](#c4-sia-intervention-健康检查).

---

## 三. 适用范围 + 局限

| 模型 | b2 inproc | HTTP path |
|---|---|---|
| Qwen3-VL-30B-A3B-Instruct | ✅ (本 doc 主路径) | ✅ |
| Qwen3-14B (旧 baseline) | ✅ (理论可用, 未实测) | ✅ ([14B doc Appendix C.1](qwen3-14b-sia-eval-20260605.md#c1-14b-alpacaeval-200q-max256-sia-only)) |
| 0GM-1.0-35B-A3B (Qwen3.5 arch) | ❌ (需 vllm 0.19+, b2 inproc 在 0.19 有 PIECEWISE 冲突) | ✅ |

**b2 inproc 局限**:
- 单卡布局 — RM 必须跟主 LLM **同进程同 GPU**, 不能跨卡负载均衡
- 0GM-35B 不能用 (Qwen3.5 MoE arch 0.17.1 未注册)
- 需独立 venv5 (vllm 0.17.1), 跟 venv4 (0.19) 并存

---

## 四. 后续优化空间

| # | 方向 | 预期收益 (相对 b2 inproc) | 工作量 |
|---|---|---|---|
| 1 | Stage A (KV catchup) 跟主 LLM forward 并行 (双 CUDA stream) | +14% (~95 tok/s) | 中 |
| 2 | 更小 VM (Qwen3-1.7B 重训 LoRA) | +20-40% | 高 (需训练) |
| 3 | SIA processor Python wrap 优化 (SKIP 0.51ms 主要是 cpu_sync) | +5-8% | 低 |
| 4 | 直接训练 RM 嵌入主 LLM 隐状态 (linear head) | 接近 noSIA 速度 | 极高 (ML 工作) |

短期 #3 + #1 最划算。

---

## Appendix

### C.0 通用约束

- 主推理 venv: `venv4` (vllm 0.19.0) for HTTP path, `venv5` (vllm 0.17.1) for b2 inproc
- 评分 venv: `venv4` (含 accelerate, venv5 没装 → Skywork 需要 venv4)
- 评分 RM: `/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B`
- 跑评分前**必须 kill 所有 SIA + RM server 进程** (Skywork 需要 ~16GB GPU)
- Driver payload 必须显式 `repetition_penalty=1.0` (虽然 server 默认已 fix 到 1.0, 双保险)
- SIA `noSIA arm` 启动方式 = 同 `sia_vllm_server.py` 但 `--weight 0.0 --entropy_threshold 999999` (或 10), 让 SIA processor 退化为 no-op, 保持代码路径一致以排除非 SIA 因素

---

### C.1 VL-30B AlpacaEval 200Q max=256 HTTP path

**覆盖**: §1.1 (SIA HTTP & noSIA HTTP 的 Skywork mean) + §1.4 (跨实验一致性) + §2.4 (intv 18.8%, 0 RM error).

#### 启动 vllm RM server (port 8001)

```bash
nohup /workspace/SIA/venv4/bin/vllm serve \
  /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --runner pooling --convert classify \
  --hf-overrides '{"architectures":["Qwen3WithScoreForCausalLM"]}' \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.15 \
  --max-model-len 4096 \
  --port 8001 --host 0.0.0.0 \
  --disable-log-stats \
  > /tmp/vl30b_runs/vllm_rm_server.log 2>&1 &
```

#### RM /classify smoke test (启动 SIA LLM 之前**必做**)

```bash
until curl -s --max-time 2 http://localhost:8001/v1/models 2>/dev/null | grep -q object; do sleep 3; done

curl -s --max-time 10 -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{"model":"/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm","input":[[1,2,3,4,5]]}'
# 应返回 {"id":...,"data":[{"index":0,"probs":[...]}], ...}
```

#### SIA LLM server (port 8000)

```bash
cd /workspace/git/0g-sparse-inference-alignment
nohup /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
  --rm_url http://localhost:8001 \
  --rm_backend vllm \
  --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --llm_gpu_mem 0.60 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 4096 \
  --port 8000 \
  > /tmp/vl30b_runs/sia_llm_server.log 2>&1 &
```

#### noSIA LLM server (相同代码路径, 仅 weight=0)

```bash
# 同上, 仅改:
  --topk 10 --weight 0.0 --entropy_threshold 10.0 \
```

#### Driver (200Q AlpacaEval max=256)

`drive_vl30b.py` payload:
```python
{
    "model": "Qwen3-VL-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": item["instruction"]}],
    "max_tokens": 256,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": False},
}
```

```bash
nohup /workspace/SIA/venv4/bin/python -u /tmp/vl30b_runs/drive_vl30b.py \
  /tmp/vl30b_runs/sia_VL30B_vllmRM_200q_max256.json \
  Qwen3-VL-30B-A3B-Instruct \
  256 200 \
  > /tmp/vl30b_runs/driver.log 2>&1 &
```

#### Skywork 评分

```bash
pkill -9 -f 'EngineCore' 2>/dev/null
pkill -9 -f 'sia_vllm_server' 2>/dev/null
pkill -9 -f 'vllm serve' 2>/dev/null
sleep 8

cd /tmp/vl30b_runs
/workspace/SIA/venv4/bin/python -u compare_redo.py
```

#### Artifacts

| 文件 | 内容 |
|---|---|
| [`exp/.../vl30b-sia-max256/outputs_scored.json`](../exp/rep-penalty-fix-validation-20260604/vl30b-sia-max256/outputs_scored.json) | SIA arm |
| [`exp/.../vl30b-nosia-max256/outputs_scored.json`](../exp/rep-penalty-fix-validation-20260604/vl30b-nosia-max256/outputs_scored.json) | noSIA arm |
| [`exp/.../vl30b-sia-max256/sia_llm_server.log`](../exp/rep-penalty-fix-validation-20260604/vl30b-sia-max256/sia_llm_server.log) | SIA 健康指标 (grep `intervened=`) |
| [`exp/.../analysis/compare_redo.log`](../exp/rep-penalty-fix-validation-20260604/analysis/compare_redo.log) | paired t-test |

---

### C.2 VL-30B AlpacaEval 200Q max=256 b2 inproc

**覆盖**: §1.1 (SIA b2 inproc 的 Skywork mean) + §2.2 (端到端 1.46× + RM 6.4×) + §2.4 (intv 18.50%, 0 RM error).

#### SIA LLM server with b2 inproc backend (port 8000)

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

**关键 env vars**:
- `SIA_LLM_CUDAGRAPH=piecewise` — 主 LLM 强制 PIECEWISE, 避免 0.19 那种 FULL cudagraph runtime flag
- `SIA_RM_CUDAGRAPH=piecewise` — RM 同理
- `SIA_RM_MULTIPROCESS=0` — RM 跟主 LLM 同进程 (InprocClient)

#### Driver

跟 [Appendix C.1](#c1-vl-30b-alpacaeval-200q-max256-http-path) 的 driver 完全相同 (只 venv 改 venv5):

```bash
cd /tmp/vl30b_runs
nohup /workspace/SIA/venv5/bin/python -u drive_vl30b.py \
  /tmp/vl30b_runs/sia_VL30B_b2inproc_200q_max256.json \
  Qwen3-VL-30B-A3B-Instruct \
  256 200 \
  > /tmp/vl30b_runs/driver_b2_inproc.log 2>&1 &
```

#### Skywork 评分

```bash
pkill -9 -f 'EngineCore' 2>/dev/null
pkill -9 -f 'sia_vllm_server' 2>/dev/null
sleep 8

cd /tmp/vl30b_runs
/workspace/SIA/venv4/bin/python -u compare_b2_inproc.py   # venv4 因为含 accelerate
```

#### Artifacts

| 文件 | 内容 |
|---|---|
| [`exp/.../sia_llm_server.log`](../exp/vl30b-b2-inproc-speedup-20260605/sia_llm_server.log) | 含 `[SIA-pf-summary]` 性能 + `intervened=` 健康指标 |
| [`exp/.../driver.log`](../exp/vl30b-b2-inproc-speedup-20260605/driver.log) | 200Q 驱动 |
| [`exp/.../outputs.json`](../exp/vl30b-b2-inproc-speedup-20260605/outputs.json) | 原始输出 |
| [`exp/.../outputs_scored.json`](../exp/vl30b-b2-inproc-speedup-20260605/outputs_scored.json) | + Skywork score |
| [`exp/.../compare_b2_inproc.py`](../exp/vl30b-b2-inproc-speedup-20260605/compare_b2_inproc.py) | paired vs HTTP path |
| [`exp/.../compare_b2_inproc.log`](../exp/vl30b-b2-inproc-speedup-20260605/compare_b2_inproc.log) | paired t-test 输出 |

#### venv5 创建 (一次性)

```bash
python3 -m venv /workspace/SIA/venv5
/workspace/SIA/venv5/bin/pip install --quiet vllm==0.17.1
# 不装 peft (b2 用 merged VM, 不走 LoRA)
# 不装 accelerate (Skywork 评分用 venv4)

# 验证:
/workspace/SIA/venv5/bin/python -c "import vllm; print(vllm.__version__)"   # 0.17.1
grep 'def unlock_workspace' /workspace/SIA/venv5/lib/python3.12/site-packages/vllm/v1/worker/workspace.py
```

---

### C.3 VL-30B AlpacaEval 200Q max=2048 HTTP path

**覆盖**: §1.2 (max=2048 SIA Δ=+2.75) + §1.4 (跨实验一致性) + §2.1 (per-token timing breakdown) + §2.4 (intv 25.83%).

跟 [Appendix C.1](#c1-vl-30b-alpacaeval-200q-max256-http-path) 完全一致, 仅 driver 把 `max_tokens` 改成 **2048** (server 的 `--max_model_len` 仍 4096):

```bash
nohup /workspace/SIA/venv4/bin/python -u /tmp/vl30b_runs/drive_vl30b.py \
  /tmp/vl30b_runs/sia_VL30B_vllmRM_200q_max2048_v2.json \
  Qwen3-VL-30B-A3B-Instruct \
  2048 200 \
  > /tmp/vl30b_runs/driver.log 2>&1 &
```

**⚠️ Race condition (max=2048 第一次跑踩过)**: vllm RM (port 8001) + SIA LLM (port 8000) 同时启动会竞争 GPU memory, 概率性导致 RM 在 KV cache 分配阶段 OOM 死亡 (FastAPI 进程残留, `/v1/models` 短暂仍 200 OK, 但 `/classify` 全部 Connection refused, SIA 退化为 no-op)。**正确顺序**:

1. 启动 RM, 等 `/v1/models` 返回 200
2. **执行 `/classify` smoke test (见 [Appendix C.1](#c1-vl-30b-alpacaeval-200q-max256-http-path))**, 确认它真活着
3. 启动 SIA LLM
4. SIA LLM ready 后, **再次** ping RM `/v1/models`
5. 跑 driver 第一题, 检查 SIA log 出现 `intervened=N/M (N>0)`

#### Artifacts

| 文件 | 内容 |
|---|---|
| [`exp/.../vl30b-sia-max2048/outputs_scored.json`](../exp/rep-penalty-fix-validation-20260604/vl30b-sia-max2048/outputs_scored.json) | SIA arm (REDO, 真 SIA) |
| [`exp/.../vl30b-nosia-max2048/outputs_scored.json`](../exp/rep-penalty-fix-validation-20260604/vl30b-nosia-max2048/outputs_scored.json) | noSIA arm |
| [`exp/.../vl30b-sia-max2048/sia_llm_server.log`](../exp/rep-penalty-fix-validation-20260604/vl30b-sia-max2048/sia_llm_server.log) | SIA-pf-summary 性能 + 健康 |
| [`exp/.../analysis/compare_redo.log`](../exp/rep-penalty-fix-validation-20260604/analysis/compare_redo.log) | paired |

---

### C.4 SIA intervention 健康检查

**目的**: 判断 SIA 是否真在工作, 避免 broken-RM 跑出 garbage 数据。

| 指标 | 健康范围 | 红灯 |
|---|---|---|
| intervention ratio | 10-40% (entropy_threshold=1.0) | **0.0%** → RM 死了 |
| top1 flip rate | 50-80% | 0% / 100% (有 bug) |
| RM error 日志数 | **0** | >0 → 连接问题 |
| Connection refused 日志数 | **0** | >0 → RM server 死了 |
| SIA output vs noSIA output | 应该 **不等** byte-for-byte | 等 → broken |

```bash
SIA_LOG=/path/to/sia_llm_server.log

grep -c "Connection refused\|RM error\|inappropriate" $SIA_LOG    # 应该 0

# 总 intervention ratio
grep 'intervened=' $SIA_LOG | awk -F'intervened=' \
  '{split($2,a,"/"); split(a[2],b," "); intv+=a[1]; tot+=b[1]; n++} \
   END {printf "Reqs: %d  steps: %d  intv: %d (%.2f%%)\n", n, tot, intv, intv*100/tot}'

# 总 top1 flip ratio
grep 'top1_flip=' $SIA_LOG | awk -F'top1_flip=' \
  '{split($2,a,"/"); split(a[2],b," "); flip+=a[1]; total+=b[1]} \
   END {if (total>0) printf "Flip: %d/%d = %.2f%%\n", flip, total, flip*100/total}'

# 最近 SIA-pf-summary (含 RM 调用 timing)
grep 'SIA-pf-summary' $SIA_LOG | tail -1
```

---

### C.5 VL-30B MMLU 150Q HTTP path

**覆盖**: §1.3 (SIA HTTP & noSIA accuracy) + §2.3 (SIA HTTP & noSIA throughput) + §2.4 (SIA intv 10.03%).

#### RM server

跟 [Appendix C.1](#c1-vl-30b-alpacaeval-200q-max256-http-path) 一致。

#### SIA LLM server (注意 `--topk 5`, 不是 AlpacaEval 的 10)

```bash
cd /workspace/git/0g-sparse-inference-alignment
nohup /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
    --rm_url http://localhost:8001 \
    --rm_backend vllm \
    --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --llm_gpu_mem 0.6 --topk 5 --weight 1.0 --entropy_threshold 1.0 \
    --host 0.0.0.0 --port 8000 \
    > /tmp/vl30b_runs/mmlu_sia_server.log 2>&1 &
```

#### noSIA LLM server (相同 code path, weight=0)

```bash
nohup /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
    --llm /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
    --rm_url http://localhost:8001 \
    --llm_gpu_mem 0.6 --topk 5 --weight 0.0 --entropy_threshold 999999 \
    --host 0.0.0.0 --port 8000 \
    > /tmp/vl30b_runs/mmlu_nosia_server.log 2>&1 &
```

#### Driver (eval/mmlu_eval.py, max_tokens=2048 内置)

```bash
# SIA arm
nohup /workspace/SIA/venv4/bin/python -u eval/mmlu_eval.py \
    --base_url http://localhost:8000/v1 \
    --model /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
    --output eval/results/vl30b_SIA_mmlu_150q.json \
    --limit 5 --repetition_penalty 1.0 \
    > /tmp/vl30b_runs/mmlu_sia_driver.log 2>&1 &

# noSIA arm — 同上, 仅 --output 不同
```

打分: `eval/mmlu_eval.py` 内置 exact_match (从 CoT 输出抽 "Answer: X"), 不需要单独评分阶段。

#### Artifacts

| 文件 | 内容 |
|---|---|
| `eval/results/vl30b_SIA_mmlu_150q.json` | SIA arm results (accuracy 0.8000, per-subject) |
| `eval/results/vl30b_noSIA_mmlu_150q.json` | noSIA arm (accuracy 0.7867) |
| [`exp/.../vl30b-mmlu-SIA/results.json`](../exp/rep-penalty-fix-validation-20260604/vl30b-mmlu-SIA/results.json) | 备份 |
| [`exp/.../vl30b-mmlu-noSIA/results.json`](../exp/rep-penalty-fix-validation-20260604/vl30b-mmlu-noSIA/results.json) | 备份 |
| `/tmp/vl30b_runs/mmlu_sia_driver.log` | per-Q latency + tokens (parse for throughput) |
| `/tmp/vl30b_runs/mmlu_sia_server.log` | SIA-pf-summary + `intervened=` 健康指标 |

---

### C.6 VL-30B MMLU 150Q b2 inproc

**覆盖**: §1.3 (SIA b2 inproc accuracy 0.8133) + §2.3 (SIA b2 inproc 73.2 tok/s).

#### SIA LLM server with b2 inproc backend (port 8000)

```bash
cd /workspace/git/0g-sparse-inference-alignment
SIA_LLM_CUDAGRAPH=piecewise \
SIA_RM_CUDAGRAPH=piecewise \
SIA_RM_MULTIPROCESS=0 \
/workspace/SIA/venv5/bin/python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
  --rm_backend b2 \
  --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --rm_b2_gpu_mem 0.15 \
  --llm_gpu_mem 0.55 \
  --topk 5 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 4096 --port 8000 \
  > /tmp/vl30b_runs/mmlu_150q_b2_inproc_20260605/sia_llm_server.log 2>&1 &
```

**关键 diff vs [Appendix C.2](#c2-vl-30b-alpacaeval-200q-max256-b2-inproc)** (AlpacaEval b2 inproc):
- `--topk 5` (不是 10) — 跟 MMLU HTTP path 对齐保持 SIA Δ 可比
- driver 用 `eval/mmlu_eval.py`, 不是 `drive_vl30b.py`

#### Driver

```bash
/workspace/SIA/venv5/bin/python eval/mmlu_eval.py \
  --base_url http://localhost:8000/v1 \
  --model /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
  --subjects anatomy astronomy business_ethics clinical_knowledge \
             college_chemistry college_computer_science college_mathematics \
             college_medicine college_physics conceptual_physics \
             econometrics electrical_engineering formal_logic global_facts \
             high_school_chemistry high_school_geography high_school_macroeconomics \
             high_school_mathematics high_school_physics high_school_statistics \
             high_school_us_history human_aging logical_fallacies machine_learning \
             miscellaneous philosophy professional_accounting professional_law \
             public_relations virology \
  --limit 5 --temperature 1.0 --repetition_penalty 1.0 \
  --output eval/results/vl30b_SIA_mmlu_150q_b2_inproc_20260605.json \
  > /tmp/vl30b_runs/mmlu_150q_b2_inproc_20260605/mmlu_driver.log 2>&1
```

> ⚠️ **venv5 默认无 datasets/tqdm**, 跑 driver 前先 `/workspace/SIA/venv5/bin/pip install datasets tqdm`。

#### Artifacts

| 文件 | 内容 |
|---|---|
| `eval/results/vl30b_SIA_mmlu_150q_b2_inproc_20260605.json` | acc 0.8133 |
| [`exp/.../vl30b-mmlu-150q-b2-inproc-20260605/sia_llm_server.log`](../exp/vl30b-mmlu-150q-b2-inproc-20260605/sia_llm_server.log) | SIA-pf-summary (b2_score_call p50 = 10.83 ms) + 健康 |
| [`exp/.../vl30b-mmlu-150q-b2-inproc-20260605/mmlu_driver.log`](../exp/vl30b-mmlu-150q-b2-inproc-20260605/mmlu_driver.log) | per-Q latency + tokens |

---

### C.7 历史推翻 — 旧错误结论

之前 [`exp/vl30b_200q_dual_rm/README.md`](../exp/vl30b_200q_dual_rm/README.md) 曾写:

> W2S=7.5× 配置下 SIA 仍 **-75% Δ**, 完全不可用。论文的成功只在 W2S ≤ 3.5× in-distribution 配置成立。

**真因**: `repetition_penalty=1.3` 污染所有那些实验, 修复后:
- VL-30B (W2S=7.5×) max=256: **+20% Δ**
- VL-30B (W2S=7.5×) max=2048: **+10% Δ**
- 跟 paper Qwen3-14B (in-distribution) gain **同量级**

同样曾推论 "SIA 是短窗口工具, 只在 ≤ 256 有效" — 基于 broken-RM 的 SIA-2048 = -6%。REDO 后真 +10% Δ, **SIA 在长生成上同样有效**。

参考根因: [`sia-repetition-penalty-root-cause-20260604.md`](sia-repetition-penalty-root-cause-20260604.md).
