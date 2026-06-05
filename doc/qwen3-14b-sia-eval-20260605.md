# Qwen3-14B SIA 评测汇总 (效果 + 性能)

**最后更新**: 2026-06-05
**适用前提**: 所有数据均在 [`commit f0f4e1c` (rep_penalty 1.3 → 1.0 修复)](sia-repetition-penalty-root-cause-20260604.md) 之后采集。

**结论速览**:
- ✅ **效果**: Qwen3-14B + VM-Qwen3-4B SIA 在 AlpacaEval 200Q 拿到 Skywork mean **+11.48**, 跟 paper SIA 805Q (+11.16) **统计上等价 (p=0.43)**, SIA Δ vs paper noSIA **+1.89, p<10⁻⁴**。
- ⚠️ **性能**: Qwen3-14B 没有本轮单独跑性能数据 (走 PyTorch RM HTTP path, 大致预期类似 [VL-30B max=256 HTTP path](qwen3-vl-30b-sia-eval-20260605.md#22-alpacaeval-200q-max256))。
- ⚠️ **范围**: 本轮 14B 实验只跑了 AlpacaEval 200Q max=256 一个配置, 没有 MMLU / max=2048 / b2 inproc 数据。

来源:
- [`rep-penalty-fix-validation-experiments-20260604.md`](rep-penalty-fix-validation-experiments-20260604.md)
- [`project-vs-official-inference-code-diff-20260604.md`](project-vs-official-inference-code-diff-20260604.md) (背景: 14B 复现 paper 的对照实验)

**前置环境** — 本 doc 所有实验在 vllm **0.19.0** 上跑 (venv4), 用 `--rm_backend pytorch` HTTP RM:

```bash
# 一次性创建 venv (~5 min)
scripts/setup_venv_0gm35b_http.sh /workspace/SIA/venv4
```

该脚本装 [`requirements/0gm35b-or-vl30b-http.txt`](../requirements/0gm35b-or-vl30b-http.txt) (vllm 0.19.0 + transformers 4.57.6 + peft) + `pip install -e .`。venv4 名字虽然来自 0gm35b, 但对 Qwen3-14B + HTTP RM 同样适用 (vllm 0.19 全模型兼容)。

> Qwen3-14B **也可以**用 vllm 0.10.1.1 (`requirements/qwen14b-b2-inproc.txt`) 跑 b2 inproc 加速, 但本 doc 实验没走那条路, 详见 [CLAUDE.md "Dependencies"](../CLAUDE.md#running-the-project) 矩阵。

---

## 一. 效果评测

### 1.1 AlpacaEval 200Q (max=256) — Skywork 评分

详见 [Appendix C.1 — 14B AlpacaEval 200Q max=256 实验](#c1-14b-alpacaeval-200q-max256-sia-only)。

| | Skywork mean | median | n |
|--|---|---|---|
| **Ours Qwen3-14B SIA (rep=1.0)** | **+11.48** | +11.59 | 200 |
| Paper Qwen3-14B SIA (805Q baseline) | +11.16 | — | 805 |
| Paper Qwen3-14B noSIA (805Q baseline) | +9.59 | — | 805 |

**Paired (n=200, 跟 paper 同 instructions 对比)**:

| 对比 | Δ | win | p |
|---|---|---|---|
| Ours SIA vs Paper SIA | +0.31 | (n.s.) | **0.43** ← 统计等价 ✅ |
| Ours SIA vs Paper noSIA | **+1.89** | (n/a) | **<10⁻⁴** ✅ 跟 paper +1.63 同量级 |

**单变量 rep_penalty 对照 (n=125, 同 instructions, 同代码, 只切 rep_penalty)**:

| 对比 | Δ | win | p |
|---|---|---|---|
| Ours rep=1.0 vs Ours rep=1.3 | **+7.71** | 109/125 (87%) | **<10⁻¹²** ← 锁定 rep_penalty bug |

**结论**: 项目代码 + rep_penalty=1.0 后, Qwen3-14B SIA 跟官方代码 **byte-equivalent**, 跟 paper 数据统计等价, paper SIA gain 完全复现。

### 1.2 MMLU

本轮没跑 Qwen3-14B MMLU。**历史数据**见 [`doc/eval-report.md`](eval-report.md), 但那是 rep_penalty=1.3 时期的, 结论 ("MMLU SIA 跟 RM 训练目标无关 → 无 regression 无 gain") **本质上仍成立** (rep_penalty bug 主要伤 alignment-class 任务, 对 exact_match accuracy 影响有限), 但绝对数值不再权威, 建议如需要请重跑。

---

## 二. 性能评测

### 2.1 本轮采集状态

⚠️ Qwen3-14B 本轮**没有独立性能采集**, 原因:

1. Qwen3-14B 跑的是 `--rm_backend pytorch` (官方 ValueModel HTTP), 跟 30B 的 `vllm` HTTP backend 不同, 时延特性不可直接套用
2. 30B 的 [b2 inproc 加速](qwen3-vl-30b-sia-eval-20260605.md#222-rm-调用单次时延-核心收益) **理论上对 14B 同样适用** (14B 完全支持 vllm 0.17.1, 见 [vl30b-b2-inproc-speedup §6.1](vl30b-b2-inproc-speedup-20260605.md#61-适用模型)), 但**未实测**

### 2.2 跟 30B 类似实验的对照参考

如需 14B 实际 throughput, 可参考 30B AlpacaEval max=256 HTTP path 数据 (作为粗略上界, 因为 30B 主 LLM 更大):

| 配置 | 30B HTTP path 200Q wall | 30B b2 inproc 200Q wall |
|---|---|---|
| AlpacaEval max=256 | 17.2 min (~5.2s/Q) | 11.8 min (~3.5s/Q) |

详见 [VL-30B doc § 2.2](qwen3-vl-30b-sia-eval-20260605.md#22-alpacaeval-200q-max256)。

---

## 三. 跟 paper 的对应关系

| 我们的实验 | Paper 对照 | 结果 |
|---|---|---|
| Qwen3-14B SIA AlpacaEval (max=256, topk=10, weight=1.0, entropy=1.0, rep=1.0) | Table 1: Qwen3-14B SIA on AlpacaEval | **统计等价** (+11.48 vs +11.16, p=0.43) |
| 同上, paired vs paper noSIA | Table 1: Qwen3-14B (SIA - noSIA) = +1.63 | 我们 **+1.89, p<10⁻⁴** (跟 paper 高度一致, 在 200Q 噪声内) |
| MMLU | (paper 未测) | — |

---

## Appendix

### C.0 通用约束

- Python venv: `venv4` (`/workspace/SIA/venv4/bin/python`, vllm 0.19.0)
- 评分 RM: `/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B`
- 评分要求: **跑评分前必须 kill 所有 SIA + RM server 进程, Skywork 需要 ~16GB GPU**
- Driver 必须在 payload 显式 `repetition_penalty=1.0` (即使 server 默认已修复 1.0, 双保险防止旧版 client 默认 1.3)

### C.1 14B AlpacaEval 200Q max=256 SIA-only

**rep_penalty bug 修复验证, 单 SIA arm, paired 比较走 paper 805Q baseline。**

#### 启动 RM (官方 PyTorch ValueModel, port 8002)

```bash
nohup /workspace/SIA/venv4/bin/python \
  /workspace/git/0g-sparse-inference-alignment/src/sia_rm_pytorch_official.py \
  --rm /workspace/SIA/models/Qwen3-4B \
  --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
  --device cuda:0 --port 8002 \
  > /tmp/qwen3_14b_runs/pytorch_rm_server.log 2>&1 &
```

#### 启动 SIA LLM server (Qwen3-14B + pytorch backend, port 8000)

```bash
cd /workspace/git/0g-sparse-inference-alignment
nohup /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/Qwen3-14B \
  --rm_url http://localhost:8002 \
  --rm_backend pytorch \
  --llm_gpu_mem 0.55 \
  --topk 10 --weight 1.0 --entropy_threshold 1.0 \
  --max_model_len 2048 \
  --port 8000 \
  > /tmp/qwen3_14b_runs/sia_server_rep10.log 2>&1 &
```

#### Driver (200Q, max=256, payload 显式 rep=1.0)

```bash
nohup /workspace/SIA/venv4/bin/python -u /tmp/qwen3_14b_runs/drive_805_rep10.py \
  /tmp/qwen3_14b_runs/sia_REP10_200q_max256.json \
  Qwen3-14B \
  256 200 \
  > /tmp/qwen3_14b_runs/driver_rep10_max256.log 2>&1 &
```

`drive_805_rep10.py` payload 关键字段:
```python
{
    "model": "Qwen3-14B",
    "messages": [{"role": "user", "content": item["instruction"]}],
    "max_tokens": 256,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
}
```

#### Skywork 评分 + 跟 paper paired

```bash
pkill -9 -f 'EngineCore' 2>/dev/null
pkill -9 -f 'sia_vllm_server' 2>/dev/null
pkill -9 -f 'sia_rm_pytorch_official' 2>/dev/null
sleep 8

cd /tmp/qwen3_14b_runs
/workspace/SIA/venv4/bin/python -u compare_qwen14b.py
```

#### Artifacts

| 文件 | 内容 |
|---|---|
| [`exp/.../qwen14b-sia-max256/outputs_scored.json`](../exp/rep-penalty-fix-validation-20260604/qwen14b-sia-max256/outputs_scored.json) | 200Q SIA 输出 + Skywork score |
| [`exp/.../analysis/compare_qwen14b.log`](../exp/rep-penalty-fix-validation-20260604/analysis/compare_qwen14b.log) | paired vs paper SIA + paper noSIA |
| [`exp/.../analysis/compare_qwen14b.py`](../exp/rep-penalty-fix-validation-20260604/analysis/compare_qwen14b.py) | 评分脚本 |

### C.2 SIA intervention 健康检查

跟 30B 一致, 见 [VL-30B doc Appendix C.4](qwen3-vl-30b-sia-eval-20260605.md#c4-sia-intervention-健康检查).
