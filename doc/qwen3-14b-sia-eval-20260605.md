# Qwen3-14B SIA Evaluation Summary (Quality + Performance)

**Last updated**: 2026-06-05
**Prerequisites**: All data collected after [`commit f0f4e1c` (rep_penalty 1.3 → 1.0 fix)](sia-repetition-penalty-root-cause-20260604.md).

**Summary**:
- **Quality**: Qwen3-14B + VM-Qwen3-4B SIA achieves Skywork mean **+11.48** on AlpacaEval 200Q, statistically equivalent to paper SIA 805Q (+11.16) **(p=0.43)**; SIA delta vs paper noSIA **+1.89, p<10⁻⁴**.
- **Performance**: No independent performance data collected for Qwen3-14B in this round (runs on PyTorch RM HTTP path; roughly expected to be similar to [VL-30B max=256 HTTP path](qwen3-vl-30b-sia-eval-20260605.md#22-alpacaeval-200q-max256)).
- **Scope**: This round of 14B experiments only ran the AlpacaEval 200Q max=256 configuration; no MMLU / max=2048 / b2 inproc data.

Sources:
- [`rep-penalty-fix-validation-experiments-20260604.md`](rep-penalty-fix-validation-experiments-20260604.md)
- [`project-vs-official-inference-code-diff-20260604.md`](project-vs-official-inference-code-diff-20260604.md) (context: 14B paper reproduction comparison experiment)

**Environment prerequisites** — all experiments in this doc run on vllm **0.19.0** (venv4) with `--rm_backend pytorch` HTTP RM:

```bash
# One-time venv creation (~5 min)
scripts/setup_venv_0gm35b_http.sh /workspace/SIA/venv4
```

This script installs [`requirements/0gm35b-or-vl30b-http.txt`](../requirements/0gm35b-or-vl30b-http.txt) (vllm 0.19.0 + transformers 4.57.6 + peft) + `pip install -e .`. The venv4 name comes from 0gm35b, but it also applies to Qwen3-14B + HTTP RM (vllm 0.19 is compatible with all models).

> Qwen3-14B **can also** use vllm 0.10.1.1 (`requirements/qwen14b-b2-inproc.txt`) for b2 inproc speedup, but the experiments in this doc did not take that path. See the [CLAUDE.md "Dependencies"](../CLAUDE.md#running-the-project) matrix.

---

## 1. Quality Evaluation

### 1.1 AlpacaEval 200Q (max=256) — Skywork Scoring

See [Appendix C.1 — 14B AlpacaEval 200Q max=256 experiment](#c1-14b-alpacaeval-200q-max256-sia-only).

| | Skywork mean | median | n |
|--|---|---|---|
| **Ours Qwen3-14B SIA (rep=1.0)** | **+11.48** | +11.59 | 200 |
| Paper Qwen3-14B SIA (805Q baseline) | +11.16 | — | 805 |
| Paper Qwen3-14B noSIA (805Q baseline) | +9.59 | — | 805 |

**Paired (n=200, compared with paper using same instructions)**:

| Comparison | Delta | win | p |
|---|---|---|---|
| Ours SIA vs Paper SIA | +0.31 | (n.s.) | **0.43** — statistically equivalent |
| Ours SIA vs Paper noSIA | **+1.89** | (n/a) | **<10⁻⁴** — same order of magnitude as paper's +1.63 |

**Single-variable rep_penalty comparison (n=125, same instructions, same code, only rep_penalty changed)**:

| Comparison | Delta | win | p |
|---|---|---|---|
| Ours rep=1.0 vs Ours rep=1.3 | **+7.71** | 109/125 (87%) | **<10⁻¹²** — confirms rep_penalty bug |

**Conclusion**: With project code + rep_penalty=1.0, Qwen3-14B SIA is **byte-equivalent** to the official code, statistically equivalent to paper data, and fully reproduces the paper's SIA gain.

### 1.2 MMLU

Qwen3-14B MMLU was not run in this round. **Historical data** is available in [`doc/eval-report.md`](eval-report.md), but that was from the rep_penalty=1.3 era. The conclusion ("MMLU SIA is unrelated to RM training objective → no regression, no gain") **still holds in essence** (the rep_penalty bug mainly affects alignment-class tasks, with limited impact on exact_match accuracy), but the absolute numbers are no longer authoritative — rerun if needed.

---

## 2. Performance Evaluation

### 2.1 Data Collection Status

Qwen3-14B has **no independent performance data** collected in this round, reasons:

1. Qwen3-14B runs on `--rm_backend pytorch` (official ValueModel HTTP), which differs from the 30B's `vllm` HTTP backend — latency characteristics cannot be directly applied
2. The 30B's [b2 inproc speedup](qwen3-vl-30b-sia-eval-20260605.md#222-rm-调用单次时延-核心收益) **theoretically applies to 14B as well** (14B fully supports vllm 0.17.1; see [vl30b-b2-inproc-speedup §6.1](vl30b-b2-inproc-speedup-20260605.md#61-适用模型)), but **has not been measured**

### 2.2 Reference Comparison with Similar 30B Experiments

For 14B actual throughput, refer to the 30B AlpacaEval max=256 HTTP path data (as a rough upper bound, since the 30B main LLM is larger):

| Configuration | 30B HTTP path 200Q wall time | 30B b2 inproc 200Q wall time |
|---|---|---|
| AlpacaEval max=256 | 17.2 min (~5.2s/Q) | 11.8 min (~3.5s/Q) |

See [VL-30B doc § 2.2](qwen3-vl-30b-sia-eval-20260605.md#22-alpacaeval-200q-max256).

---

## 3. Correspondence with the Paper

| Our experiment | Paper reference | Result |
|---|---|---|
| Qwen3-14B SIA AlpacaEval (max=256, topk=10, weight=1.0, entropy=1.0, rep=1.0) | Table 1: Qwen3-14B SIA on AlpacaEval | **Statistically equivalent** (+11.48 vs +11.16, p=0.43) |
| Same, paired vs paper noSIA | Table 1: Qwen3-14B (SIA - noSIA) = +1.63 | Ours **+1.89, p<10⁻⁴** (highly consistent with paper, within 200Q noise) |
| MMLU | (not measured in paper) | — |

---

## Appendix

### C.0 Common Constraints

- Python venv: `venv4` (`/workspace/SIA/venv4/bin/python`, vllm 0.19.0)
- Scoring RM: `/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B`
- Scoring requirement: **kill all SIA + RM server processes before scoring; Skywork requires ~16GB GPU**
- Driver must explicitly set `repetition_penalty=1.0` in the payload (even if the server default is already fixed to 1.0 — belt-and-suspenders to prevent old client defaulting to 1.3)

### C.1 14B AlpacaEval 200Q max=256 SIA-only

**Validates rep_penalty bug fix; single SIA arm; paired comparison against paper 805Q baseline.**

#### Start RM (official PyTorch ValueModel, port 8002)

```bash
nohup /workspace/SIA/venv4/bin/python \
  /workspace/git/0g-sparse-inference-alignment/src/sia_rm_pytorch_official.py \
  --rm /workspace/SIA/models/Qwen3-4B \
  --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
  --device cuda:0 --port 8002 \
  > /tmp/qwen3_14b_runs/pytorch_rm_server.log 2>&1 &
```

#### Start SIA LLM server (Qwen3-14B + pytorch backend, port 8000)

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

#### Driver (200Q, max=256, payload explicitly sets rep=1.0)

```bash
nohup /workspace/SIA/venv4/bin/python -u /tmp/qwen3_14b_runs/drive_805_rep10.py \
  /tmp/qwen3_14b_runs/sia_REP10_200q_max256.json \
  Qwen3-14B \
  256 200 \
  > /tmp/qwen3_14b_runs/driver_rep10_max256.log 2>&1 &
```

`drive_805_rep10.py` payload key fields:
```python
{
    "model": "Qwen3-14B",
    "messages": [{"role": "user", "content": item["instruction"]}],
    "max_tokens": 256,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
}
```

#### Skywork scoring + paired comparison with paper

```bash
pkill -9 -f 'EngineCore' 2>/dev/null
pkill -9 -f 'sia_vllm_server' 2>/dev/null
pkill -9 -f 'sia_rm_pytorch_official' 2>/dev/null
sleep 8

cd /tmp/qwen3_14b_runs
/workspace/SIA/venv4/bin/python -u compare_qwen14b.py
```

#### Artifacts

| File | Contents |
|---|---|
| [`exp/.../qwen14b-sia-max256/outputs_scored.json`](../exp/rep-penalty-fix-validation-20260604/qwen14b-sia-max256/outputs_scored.json) | 200Q SIA outputs + Skywork scores |
| [`exp/.../analysis/compare_qwen14b.log`](../exp/rep-penalty-fix-validation-20260604/analysis/compare_qwen14b.log) | Paired vs paper SIA + paper noSIA |
| [`exp/.../analysis/compare_qwen14b.py`](../exp/rep-penalty-fix-validation-20260604/analysis/compare_qwen14b.py) | Scoring script |

### C.2 SIA Intervention Health Check

Same as 30B; see [VL-30B doc Appendix C.4](qwen3-vl-30b-sia-eval-20260605.md#c4-sia-intervention-健康检查).
