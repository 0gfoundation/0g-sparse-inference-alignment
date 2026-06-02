# 在当前 dir 跑 AlpacaEval (Skywork RM 第三方打分) 的方案

**日期**: 2026-06-02
**目标主推理模型**: **0GM-1.0-35B-A3B-0427** (Qwen3.5/3.6 MoE thinking, vocab=248K)
**Value Model**: VM-Qwen3-4B-merged-for-vllm
**打分 RM**: Skywork-Reward-V2-Llama-3.1-8B (跟 helpfulness 训练对口的第三方 RM)
**对照**: 历史 A1 (2026-05-15, **14B + 同一个 VM**) 用官方 `evaluate.py`, mean reward 12.294 (noSIA) vs 13.923 (SIA) **+13.2%**。
**实施前提**: 在当前 dir 跑, 不依赖 `/workspace/SIA/git/SIA/` 的代码。

---

## 0. ⚠️ 0GM-35B 跟 14B 历史 A1 的关键差异 (必看)

**doc 第 4-5 节最初按 14B 写**, 修正后 0GM-35B 需要以下不同处理:

| 维度 | 14B (历史 A1) | **0GM-35B (本次目标)** |
|------|-------------|---------------------|
| vocab | 152K (跟 RM 同) | **248K** (比 RM 大 96K) |
| 跨 tokenizer | 不需要 | **需要** (vllm HTTP backend 走文本桥, 自动) |
| sampling rep_penalty | 1.3 (server hardcoded) → SIA 干预率 ~30% | **必须 1.0** (避免大词表 OOV 漂移到多语言乱码) → SIA 干预率 ~8% |
| sampling top_k/top_p | -1 / 1.0 (server hardcoded) | **20 / 0.95** (跟 0GM `generation_config.json` 一致) |
| 主 LLM 显存 | ~30GB | ~103GB (gpu_mem=0.72) |
| 主 LLM cudagraph | 默认即可 | FULL_AND_PIECEWISE — RM 必须**跨进程** (vllm HTTP) |
| RM-LLM 训练分布匹配 | ✅ 基本匹配 (同 Qwen3 家族) | ⚠️ **可能 OOD** (Qwen3 RM vs Qwen3.5/3.6 thinking 模型) |
| 已知 SIA 在该模型 MMLU 效果 | A1=+13.2% reward, MMLU +1.7% acc | **MMLU -1.2pp (near no-effect)** — alignment 信号在 0GM 上可能弱很多 |

**预期结果区间** (基于已观察的 MMLU 数据外推):
- 乐观: SIA 比 noSIA reward +5~+10% (VM 对 helpfulness 有训练, 即使 OOD 也可能有部分信号; AlpacaEval 比 MMLU 更对口)
- 中性: ±2% (SIA 基本无效, 跟 MMLU -1.2pp 一致)
- 悲观: SIA -5~-10% (SIA 的小幅扰动反而 hurt 自然生成质量)

实验本身仍**有信息量**: 即便结果是 near-zero, 也回答了"VM-Qwen3-4B-RM 跟 0GM-35B mismatch 到什么程度才让 helpfulness 信号失效"这个问题。

---

## 1. 官方 AlpacaEval 流程拆解 (`/workspace/SIA/git/SIA/`)

```
                          官方 pipeline (2 阶段)
       ┌─────────────────────────────────────────────────────────┐
       │ ① Generation 阶段 (evaluate.py)                          │
       │   - 加载 alpaca_eval.json (805 prompts)                  │
       │   - 每条 prompt 渲染成 "Human:\n{instruction}\nAssistant:\n" (raw text!) │
       │   - 调 src/sia.py 的 SIA.generate() — 用 raw transformers + 手写 logit 干预 │
       │   - 保存 JSON: [{id, prompt, result, intervene_ratio, ...}, ...]│
       └──────────────────────┬──────────────────────────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────────────────┐
       │ ② Scoring 阶段 (src/measure_reward.py)                   │
       │   - 加载 Skywork-Reward-V2-Llama-3.1-8B                  │
       │   - 把 (prompt, result) 解析成 [{user,...},{assistant,...}] │
       │   - 用 Skywork 自己的 chat template 渲染 + tokenize          │
       │   - model.forward() → logits[0][0] = reward score          │
       │   - 保存 JSON (原数据 + reward 字段) + 输出 mean reward       │
       └─────────────────────────────────────────────────────────┘
```

**两个关键模型**:
- **SIA Value Model** (引导): `Qwen3-4B + VM-Qwen3-4B-Base LoRA` — 在 Helpfulness 数据训过
- **评分 RM** (打分): `Skywork-Reward-V2-Llama-3.1-8B` — **独立**模型, 避免自评自

**关键路径**:
- 输入数据集: `/workspace/SIA/data/alpaca_eval/alpaca_eval.json` (805 条 `{dataset, instruction, output, generator}`)
- 评分模型: `/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B` (已下载)

---

## 2. 当前 dir 现状 vs 需要的能力

| 能力 | 官方 evaluate.py | 当前 dir | 需要 |
|------|-----------------|----------|------|
| 加载 alpaca_eval.json | ✅ | ❌ | **新增** |
| SIA-on-LLM 推理 | ✅ (raw transformers `SIA.generate`) | ✅ (vllm + `sia_vllm_server.py` HTTP API) | 复用现有 server |
| HTTP 客户端调 SIA server | n/a | ✅ (`eval/mmlu_eval.py`) | **写 alpaca 版** |
| Skywork RM 打分 | ✅ (`src/measure_reward.py`) | ❌ | **新增** |
| 结果对比/统计 | 手动 | 手动 | **新增 summary 脚本** |

**核心结论**: 当前 dir 的 SIA 推理基础设施 (`sia_vllm_server.py` + HTTP) **完全可以复用**, 只需补 2 个独立脚本: 一个 generation client, 一个 reward scoring script。

---

## 3. 方案: 最小代码改动

### 3.1 不改动的部分

- `src/sia_vllm_server.py` **零改动**: 已有的 OpenAI 兼容 `/v1/chat/completions` 端点直接能用, AlpacaEval 是单轮 user → assistant 的对话, 跟 MMLU 评测的 message 结构完全一致
- `src/sia_rm/`, `src/sia_vllm_RM.py` **零改动**

### 3.2 新增的 2 个文件

#### 文件 1: `eval/alpaca_eval.py` (Generation client, ~80 行)

```python
"""
AlpacaEval generation client. 跑 805 条 instruction → save JSON.
不引入 src/ 的代码 — 跟 mmlu_eval.py 一样是个 HTTP 客户端。
"""
import argparse, json, time
from pathlib import Path
import requests

def chat_completion(base_url, model, instruction, *,
                    max_tokens=256, temperature=1.0,
                    top_p=None, top_k=None, repetition_penalty=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": instruction}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for k, v in [("top_p", top_p), ("top_k", top_k),
                 ("repetition_penalty", repetition_penalty)]:
        if v is not None:
            payload[k] = v
    resp = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                         json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"], \
           data.get("usage", {}).get("completion_tokens", -1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="/workspace/SIA/data/alpaca_eval/alpaca_eval.json")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="只跑前 N 条 (default 全 805)")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--repetition_penalty", type=float, default=None)
    args = p.parse_args()

    data = json.load(open(args.dataset))
    if args.limit:
        data = data[:args.limit]
    print(f"loaded {len(data)} prompts from {args.dataset}")

    results = []
    for idx, row in enumerate(data):
        instr = row["instruction"]
        t0 = time.time()
        try:
            text, ntok = chat_completion(
                args.base_url, args.model, instr,
                max_tokens=args.max_tokens, temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
        except Exception as e:
            text, ntok = f"ERROR: {e}", -1
        elapsed = time.time() - t0
        results.append({
            "id": idx + 1,
            "instruction": instr,
            "output": text,
            "tokens": ntok,
            "elapsed": elapsed,
            # 沿用官方字段名 'result' 以便 measure_reward 直接复用
            "prompt": f"Human:\n{instr}\nAssistant:\n",
            "result": f"Human:\n{instr}\nAssistant:\n{text}",
        })
        if (idx + 1) % 25 == 0:
            print(f"[{idx+1}/{len(data)}] {elapsed:.1f}s")
            # 中途保存防止中断丢数据
            json.dump(results, open(args.output, "w"), ensure_ascii=False, indent=2)

    json.dump(results, open(args.output, "w"), ensure_ascii=False, indent=2)
    print(f"saved {len(results)} results to {args.output}")

if __name__ == "__main__":
    main()
```

#### 文件 2: `scripts/measure_alpaca_reward.py` (Reward scoring, ~70 行)

```python
"""
Reward scoring script for AlpacaEval results.
Port of /workspace/SIA/git/SIA/src/measure_reward.py, simplified.
"""
import argparse, json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--rm", default="/workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_length", type=int, default=2048)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.rm, use_fast=True)
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.rm, dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="sdpa", num_labels=1,
    )
    rm.eval()
    print(f"loaded Skywork RM: {args.rm}")

    data = json.load(open(args.input_file))
    print(f"scoring {len(data)} samples...")

    scored = []
    skipped = 0
    rewards = []
    for i, row in enumerate(data):
        instr = row["instruction"]
        output = row["output"]
        if output.startswith("ERROR:") or not output.strip():
            row["reward"] = None
            skipped += 1
            scored.append(row); continue
        # build conversation
        convs = [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": output},
        ]
        text = tok.apply_chat_template(convs, tokenize=False)
        if tok.bos_token and text.startswith(tok.bos_token):
            text = text[len(tok.bos_token):]
        enc = tok(text, return_tensors="pt").to(args.device)
        if enc.input_ids.shape[1] >= args.max_length:
            row["reward"] = None
            skipped += 1
            scored.append(row); continue
        with torch.no_grad():
            out = rm(**enc)
            r = out.logits[0][0].item()
        row["reward"] = r
        rewards.append(r)
        scored.append(row)
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(data)}] cur_mean={sum(rewards)/len(rewards):.4f}")

    json.dump(scored, open(args.output_file, "w"), ensure_ascii=False, indent=2)
    mean_reward = sum(rewards) / max(len(rewards), 1)
    print(f"\n=== summary ===")
    print(f"  total samples: {len(data)}")
    print(f"  scored: {len(rewards)}, skipped: {skipped}")
    print(f"  mean reward: {mean_reward:.4f}")
    print(f"  saved to: {args.output_file}")

if __name__ == "__main__":
    main()
```

---

## 4. 完整运行流程 (针对 **0GM-35B**)

### 4.1 阶段 1 — Generation (双臂, SIA vs noSIA)

**关键设计**: noSIA arm 用**原生 `vllm serve`** (不走 sia_vllm_server.py), 干净排除 SIA processor 开销; SIA arm 用 sia_vllm_server.py + RM 跨进程。两 arm 用**相同的 sampling params** 保证对比可靠。

```bash
# 假设 RM server (port 8001) 已在跑 (vllm serve VM-Qwen3-4B-merged-for-vllm)
# 当前会话中, RM server PID 918523, log /tmp/rm_vllm_serve_20260602_022935.log

# === Arm 1: noSIA — 原生 vllm serve, 不加 SIA processor ===
nohup /workspace/SIA/venv4/bin/vllm serve \
  /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
  --served-model-name 0GM-1.0-35B-A3B-0427 \
  --gpu-memory-utilization 0.72 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --host 0.0.0.0 --port 8000 \
  > /tmp/alpaca_0gm_nosia_server.log 2>&1 &

# Wait for "Application startup complete", then:
/workspace/SIA/venv2/bin/python eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model 0GM-1.0-35B-A3B-0427 \
  --max_tokens 512 \
  --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
  --output exp/alpaca_0gm_nosia_$(date +%Y%m%d_%H%M%S).json

# === Arm 2: SIA — kill noSIA, 启 sia_vllm_server.py ===
kill <pid_of_nosia_server>; sleep 10

nohup /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/0GM-1.0-35B-A3B-0427 \
  --rm_backend vllm --rm_url http://localhost:8001 \
  --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --llm_gpu_mem 0.72 --max_model_len 4096 \
  --topk 5 --weight 1.0 \
  --entropy_threshold 0.7 \
  --host 0.0.0.0 --port 8000 \
  > /tmp/alpaca_0gm_sia_server.log 2>&1 &
# 注: entropy_threshold=0.7 (而非历史 1.0), 因为 rep_penalty=1.0 下 0GM 用 1.0 干预率仅 ~8%,
#     用 0.7 干预率回到 ~25% (调整完整 600q eval 时同步调整后确定)。
#     具体阈值取决于 thr=0.7 60Q eval 结果, 可能改为 0.5。

# Wait for ready (~3-5 min for 35B load + cudagraph), then:
/workspace/SIA/venv2/bin/python eval/alpaca_eval.py \
  --base_url http://localhost:8000/v1 \
  --model 0GM-1.0-35B-A3B-0427 \
  --max_tokens 512 \
  --temperature 1.0 --top_p 0.95 --top_k 20 --repetition_penalty 1.0 \
  --output exp/alpaca_0gm_sia_$(date +%Y%m%d_%H%M%S).json
```

**注意**:
- 两 arm 用**完全相同的 sampling params** (`top_k=20 top_p=0.95 rep_penalty=1.0 max_tokens=512`), 唯一区别是 SIA processor 是否注册
- `max_tokens=512` (不是 256, 历史 A1 值): 0GM 的 thinking 比 14B 长 1.7×, 256 token 不够回答完整问题
- `rep_penalty=1.0` 是**硬性约束**, 不能用历史 1.3 — 否则 0GM 大词表会漂到多语言 OOV → 输出乱码 → reward 直接崩
- 因为 SIA processor 在 sia_vllm_server.py 里, 而 noSIA 用 raw vllm serve, 两个进程用**同一个**模型路径加载 (但启动开销分别 ~3-5 min 各算一次)

### 4.2 阶段 2 — Scoring (针对 0GM 的 `<think>` 内容做双版本)

```bash
# 等 LLM server 关闭, 释放 GPU 给 Skywork RM
kill <pid_of_sia_server>; sleep 5

# 也 kill RM server, 释放显存 (Skywork RM ~16GB BF16, 单卡足够装)
kill <pid_of_rm_server>; sleep 5
```

**0GM 输出包含 `<think>...thinking...</think>` block** (avg ~1000 token), Skywork RM 训练于 Llama-3.1 chat 数据, 没见过 `<think>` 标记。为公平对比, **打两个版本**:

```bash
# Version A: 不剥离 <think>, 整段输出喂 Skywork (跟历史 14B A1 处理一致)
/workspace/SIA/venv4/bin/python scripts/measure_alpaca_reward.py \
  --input_file exp/alpaca_0gm_sia_<TS>.json \
  --output_file exp/alpaca_0gm_sia_scored_with_think_<TS>.json \
  --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
  --device cuda:0

/workspace/SIA/venv4/bin/python scripts/measure_alpaca_reward.py \
  --input_file exp/alpaca_0gm_nosia_<TS>.json \
  --output_file exp/alpaca_0gm_nosia_scored_with_think_<TS>.json \
  --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
  --device cuda:0

# Version B: 剥离 <think>...</think> 后再打分 (更公平, 测纯 helpful 部分)
/workspace/SIA/venv4/bin/python scripts/measure_alpaca_reward.py \
  --input_file exp/alpaca_0gm_sia_<TS>.json \
  --output_file exp/alpaca_0gm_sia_scored_no_think_<TS>.json \
  --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
  --device cuda:0 \
  --strip_think                                           # ← 新增 flag

/workspace/SIA/venv4/bin/python scripts/measure_alpaca_reward.py \
  --input_file exp/alpaca_0gm_nosia_<TS>.json \
  --output_file exp/alpaca_0gm_nosia_scored_no_think_<TS>.json \
  --rm /workspace/SIA/models/Skywork-Reward-V2-Llama-3.1-8B \
  --device cuda:0 \
  --strip_think
```

**`--strip_think` 实现** (加到 `measure_alpaca_reward.py`):
```python
import re
if args.strip_think:
    # remove <think>...</think> (including the tags); robust to multi-line
    output = re.sub(r"<think>.*?</think>\s*", "", output, flags=re.DOTALL)
```

**预期**: noSIA 跟 SIA 的 reward 差应该在两种 scoring 方式下**一致** (因为 SIA 主要影响 thinking 内的 nudge, 最终 answer 文本风格差异小)。如果两种打法差异巨大, 说明 `<think>` 内容本身影响了 Skywork 评分, 此时 Version B 更可信。

### 4.3 阶段 3 — 对比

四个 scored JSON 之间的对比 (SIA vs noSIA × with/without `<think>`):

```bash
/workspace/SIA/venv4/bin/python - <<EOF
import json, statistics
def mean_r(p):
    d = json.load(open(p))
    rs = [r["reward"] for r in d if r.get("reward") is not None]
    return statistics.mean(rs), len(rs), len(d)

for tag in ["with_think", "no_think"]:
    nosia_mean, n1, t1 = mean_r(f"exp/alpaca_0gm_nosia_scored_{tag}_<TS>.json")
    sia_mean,   n2, t2 = mean_r(f"exp/alpaca_0gm_sia_scored_{tag}_<TS>.json")
    delta_abs = sia_mean - nosia_mean
    delta_rel = (delta_abs / abs(nosia_mean)) * 100 if nosia_mean else 0
    print(f"[{tag}]")
    print(f"  noSIA: {n1}/{t1} scored, mean reward = {nosia_mean:+.4f}")
    print(f"  SIA:   {n2}/{t2} scored, mean reward = {sia_mean:+.4f}")
    print(f"  Δ = {delta_abs:+.4f}  ({delta_rel:+.2f}%)")
    print()
EOF
```

**判读**:
- `with_think` 跟 `no_think` 的 Δ 方向**一致** → SIA 真有 alignment 影响 (不是 `<think>` 内容带偏)
- 两者方向**相反** → SIA 主要通过 thinking 内容影响 reward, 而非 final answer 质量
- 两者都接近 0 → SIA 在 0GM + VM-Qwen3-4B 配置下基本失效 (跟 MMLU -1.2pp 一致)

---

## 5. 跟历史 A1 (2026-05-15, **14B**) 对比的注意点

历史 A1 是 **14B**, 跟本次 **0GM-35B** 完全不同 model, **绝对数字不可比**, 只能定性参考。

| 维度 | 历史 A1 (14B, 官方) | **本次 (0GM-35B, vllm-based)** | 影响 |
|------|--------------------|--------------------------------|------|
| 主推理模型 | Qwen3-14B | **0GM-1.0-35B-A3B-0427** | 模型本身不同, 绝对 reward 不可比 |
| 推理引擎 | raw transformers `SIA.generate` | vllm + `sia_vllm_server.py` | 速度差异 (vllm 应快 3-5×) |
| prompt 格式 | `"Human:\n...\nAssistant:\n"` (raw text) | 0GM chat template `<\|im_start\|>user\n...\n<think>\n` | **输出风格会变** |
| max_tokens | 256 | **512** (0GM thinking 比 14B 长 1.7×, 256 不够) | 公平性: 都让模型充分回答 |
| temperature | 1.0 | 1.0 (一致) | — |
| topk (SIA 候选) | 10 | **5** | 候选少一半, 干预精度略差 |
| weight | 1.0 / 0.0 | 同 | — |
| entropy_threshold | 1.0 (SIA), 999999 (noSIA) | **0.7** (SIA), n/a (noSIA 用 raw vllm) | 0.7 是为补偿 rep_penalty=1.0 下干预率天然偏低 |
| sampling 参数 | rep=1.3, top_k=-1, top_p=1.0 (14B 安全) | **rep=1.0, top_k=20, top_p=0.95** (0GM 大词表强制要求, 见 §0) | **核心差异** |
| RM-LLM 训练分布匹配 | ✅ Qwen3 家族同源 | ⚠️ Qwen3 RM × Qwen3.5/3.6 0GM (可能 OOD) | A1 上 SIA +13.2%, 本次预期 ±10% 区间内 |

### 不可比的根本原因 #1: 模型不同

A1 是 14B, 本次是 35B MoE thinking 模型。base reward 量级不同 (35B 可能生成质量本来就更高), 绝对数字 (`13.923`) **不能** 直接拿来跟本次对比。**只能比 Δ 趋势** (SIA 相对 noSIA 的提升比例)。

### 不可比的根本原因 #2: prompt 格式不同

历史 A1 用 raw text `"Human:\n{instr}\nAssistant:\n"` (Qwen3-14B base 模型在 fine-tune 阶段见过的格式), 本次走 chat template `<|im_start|>user\n...\n<|im_start|>assistant\n<think>\n`。两者输出风格、token 数、reward 值都会有系统差异。

如果**严格要求**跟 A1 数字对齐, 需要:
- 改 `eval/alpaca_eval.py` 用 raw text prompt + OpenAI `/v1/completions` (而非 `/v1/chat/completions`)
- 或改 `sia_vllm_server.py` 加一个 raw-prompt 端点
- 但这跟 0GM thinking 模型的设计目标冲突 (0GM 期望 chat_template + `<think>` 引导)

### 不可比的根本原因 #3: VM 跟主 LLM 不同源

历史 A1: VM-Qwen3-4B-Base + Qwen3-14B → 同家族
本次:    VM-Qwen3-4B-Base + 0GM-1.0-35B-A3B → 跨家族 (0GM 后训练分布偏离 Qwen3 base 多远未知)

这是 SIA 在 0GM 上**可能效果显著弱于 14B** 的根本原因。MMLU 数据已经证实 (-1.2pp acc vs 14B 的 +1.7pp)。AlpacaEval 上预期类似 — 可能 SIA 也不会复现 +13.2% 的提升。

### 推荐做法

直接用 chat_template + 0GM 推荐 sampling 跑, **不强求**跟 A1 绝对数字对齐。**关心的是 Δ (本次 SIA - 本次 noSIA) 的方向和量级**:
- 即使 Δ 是 -2% (SIA hurt), 仍比"SIA 完全没在 0GM 上 work"信息量大: 提供了"VM 跨家族 mismatch 的代价 quantification"
- 即使 Δ 是 +3%, 也能说"SIA 在 0GM 上虽然 alignment 信号弱但 still 有方向性收益"

---

## 5b. 实施前要先回答的两个开放问题

1. **entropy_threshold 用 1.0 还是 0.7?**
   - 1.0: 跟历史 A1 一致, 但 0GM 上干预率仅 ~8%, SIA 影响可能太弱
   - 0.7: 干预率回到 ~25-30% (待当前正在跑的 thr=0.7 60Q eval 验证), 但跟 A1 不直接可比
   - **建议**: 等 thr=0.7 60Q eval 结果, 看 MMLU acc 是 (a) 改善 → 用 0.7 跑 AlpacaEval; (b) 持平/恶化 → 维持 1.0 跑

2. **是否用 b2 inproc backend 替代 vllm HTTP?**
   - vllm HTTP 跨进程: 每个 INTERVENE ~95ms HTTP overhead, SIA 慢 ~50%
   - b2 inproc: ~5ms, 但 vllm 0.19 + 0GM cudagraph 全局 flag 已知失败 (100% RM error)
   - **建议**: AlpacaEval 不重视速度, 维持 vllm HTTP (跟当前其他 0GM 实验一致), 拿到的 reward 数据更可靠

---

## 6. 资源 / 时间估算 (针对 0GM-35B, 单卡 H200 共享)

| 阶段 | 显存 | 时间 (805 题) | 备注 |
|------|------|--------------|------|
| RM server (8001, VM-Qwen3-4B-merged-for-vllm) | ~32 GB | (常驻) | 复用当前会话已加载的 |
| **Arm 1 (noSIA)**: 原生 `vllm serve 0GM-35B` | ~103 GB | 启动 ~5min + 跑 805 × avg 6s @ ~100 tok/s = **~80 min** | gpu_mem=0.72 |
| **Arm 2 (SIA)**: `sia_vllm_server.py 0GM-35B` | ~103 GB | 启动 ~5min + 跑 805 × avg 10s @ ~50 tok/s = **~135 min** | gpu_mem=0.72 |
| Server 切换 (kill noSIA + start SIA) | — | ~5-10 min | 主 LLM 重载 |
| Skywork-Reward-V2-Llama-3.1-8B 加载 | ~16 GB (BF16) | ~30s/次 | 单 GPU 装 (kill 主 LLM 后) |
| 打分 (2 input × 2 strip mode = 4 次) | ~16 GB | 4 × 805 × ~1s = **~55 min** | bf16 forward, no batching |
| **Total** | <140 GB peak (主 LLM + RM 同时) | **~5 小时** | 含 server 切换 + 全部打分 |

⚠️ **关键资源约束**:
- 主 LLM (103GB) + RM server (32GB) = 135GB 同时驻留, **H200 单卡 143GB 显存可以塞下, 留 8GB margin**
- 主 LLM 跑完后 kill (释放 103GB), **加载 Skywork RM 16GB** 不冲突
- 时间最大的不确定性是 **SIA arm 跑 805 题**: 已知当前 thr=1.0 时 ~50 tok/s, thr=0.7 时干预多, 速度可能掉到 ~35-40 tok/s, 总时间可能涨到 **3 小时**单 arm

---

## 7. 总结 — 需要写的代码 (针对 0GM-35B)

| 文件 | 角色 | 行数 | 复杂度 |
|------|------|------|--------|
| `eval/alpaca_eval.py` | HTTP client, 跑 805 prompts → JSON | ~80 | 低 (跟 `mmlu_eval.py` 高度相似, 只是去掉 MMLU 的 subjects/correctness 逻辑) |
| `scripts/measure_alpaca_reward.py` | 加载 Skywork RM, 打分 → JSON; **必须支持 `--strip_think`** flag | ~80 | 低-中 (单文件单 model 加载 + 正则剥离 `<think>...</think>`) |
| (可选) `scripts/compare_alpaca_reward.py` | 算 4 组 mean reward 差 (SIA/noSIA × with_think/no_think) | ~25 | 极低 |

**零改动现有代码**: `sia_vllm_server.py`, `sia_vllm_RM.py`, `eval/mmlu_eval.py`, `sia_rm/*` 全部保留。

## 8. 实施前置条件 (checklist)

- [ ] 当前 thr=0.7 60Q eval 结果出来 (用于决定 AlpacaEval 用 thr=1.0 还是 0.7)
- [ ] RM server (8001) 持续可用 (当前 PID 918523)
- [ ] 主 LLM server (8000) 当前是 sia_vllm_server.py thr=0.7 — 跑完 AlpacaEval 前先 kill, 重新启 raw vllm serve 作为 noSIA arm

## 9. 实施步骤 (顺序)

1. 等 thr=0.7 60Q eval 跑完, 看 MMLU acc 是否比 thr=1.0 改善, 决定 AlpacaEval 用哪个 threshold
2. 写 `eval/alpaca_eval.py` (~80 行)
3. 写 `scripts/measure_alpaca_reward.py` (~80 行, 含 --strip_think)
4. kill 当前 0GM SIA server, 启 noSIA arm (原生 `vllm serve 0GM`), 跑 805 题 → JSON-A
5. kill noSIA, 启 SIA arm (sia_vllm_server, thr=确定值), 跑 805 题 → JSON-B
6. kill SIA + RM server, 加载 Skywork RM, 打分 JSON-A + JSON-B (× with_think + no_think, 共 4 次)
7. 用 §4.3 的脚本输出 4 组 mean reward 对比

**总耗时**: ~5 小时 (含两次主 LLM 加载 + 805×2 generation + 805×4 scoring)
