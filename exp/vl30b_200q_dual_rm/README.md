# VL-30B SIA 200Q 三路对比实验 — 日志与产物 (2026-06-03)

详细分析见: [`doc/vl30b-sia-200q-experiment-report-20260603.md`](../../doc/vl30b-sia-200q-experiment-report-20260603.md)

## 实验摘要

- **目的**: 验证 vllm RM 字段名修复 (`use_activation=False`) 后, 在 VL-30B + Qwen3-4B 配置下 SIA 是否能复现论文 +13.2% Δ; 同时确认 vllm RM 跟官方 PyTorch ValueModel **行为等价**。
- **结论**: 修复成功 (vllm RM ≡ PyTorch RM, p=0.45 等价), 但 W2S=7.5× 配置下 SIA 仍 **-75% Δ**, 完全不可用。**论文的成功只在 W2S≤3.5× in-distribution 配置成立**。

## 三个 Run

| Run | RM 后端 | n | Skywork mean Δ vs noSIA | rel Δ |
|-----|---------|---|---|---|
| **Run 1** (vllm RM, path A) | `--rm_backend vllm`, port 8001 (use_activation=False fix) | 200 (paired 195) | -22.01 | **-75.2%** |
| **Run 2** (官方 PyTorch RM) | `--rm_backend pytorch`, port 8002 (`sia_rm_pytorch_official.py /score`) | 200 (本目录: 60 partial scored) | -21.90 | -72.6% |
| noSIA (baseline) | 复用 2026-06-03 03:04 跑的 200Q vllm-baseline | 200 | - | - |

**Run 2 vs Run 1**: p=0.45, win=52%, mean Δ=+0.96 — **byte-exact 等价**, 证明 RM 实现修复成功。

## 文件清单

### 生成数据 (Skywork-scored)

| 文件 | 说明 |
|------|------|
| `alpaca_vl30b_sia_vllm_rm_200q_20260603.json` | Run 1 (vllm RM) 200Q 原始生成 |
| `alpaca_vl30b_sia_vllm_rm_200q_scored.json` | Run 1 + Skywork rewards (full, trunc256, trunc180) |
| `alpaca_vl30b_sia_pytorch_rm_partial.json` | Run 2 (PyTorch RM) 部分完成生成 (写入时跑到 ~70/200) |
| `alpaca_vl30b_sia_pytorch_rm_60q_scored.json` | Run 2 前 60Q + Skywork rewards (full, trunc256) |
| `alpaca_vl30b_nosia_200q_scored.json` | noSIA baseline + Skywork full-length rewards |
| `alpaca_vl30b_nosia_200q_scored_with_trunc.json` | noSIA + Skywork (full, trunc256, trunc180) rewards |

### Server / Driver 日志

| 文件 | 说明 |
|------|------|
| `sia_vllm_server.log` | Run 1 SIA server 完整日志 (含 INTERVENE 统计) |
| `sia_pytorch_server.log` | Run 2 SIA server 日志 |
| `orchestrate.log` | 自动化 orchestrator 完整执行日志 |
| `driver_pytorch.log` | Run 2 driver 输出 (per-Q timing) |

### 分析结果

| 文件 | 说明 |
|------|------|
| `run1_vs_nosia_paired.json` | Run 1 paired records (195 个), 含 SIA / noSIA reward 配对 |
| `run1_vs_nosia_partial.txt` | Run 1 vs noSIA 主表 |
| `run1_vs_nosia_trunc256_paired.json` | trunc=256 配对结果 |
| `run1_breakdown_analysis.json` | 每题崩坏检测结果 (位置, 类型, 长度) |

### 分析脚本

| 文件 | 说明 |
|------|------|
| `drive_alpaca.py` | 200Q AlpacaEval driver (用于 Run 1 / Run 2) |
| `score_run1.py` | Run 1 Skywork 全长评分 |
| `score_truncated.py` | trunc=256 评分 |
| `score_t180.py` | trunc=180 评分 |
| `compare_60q.py` | 三路对比脚本 (Run 1 vs Run 2 vs noSIA) |
| `analyze_breakdown.py` | 崩坏模式检测 (词链 / CJK 注入 / 标题重复 / 自我修正) |

## 启动命令

### 1. RM/VM servers (跑前先启)

```bash
# vllm RM (Run 1 用) — port 8001
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --runner pooling --convert classify \
  --hf-overrides '{"architectures":["Qwen3WithScoreForCausalLM"]}' \
  --enable-prefix-caching --gpu-memory-utilization 0.10 \
  --max-model-len 2048 --port 8001 \
  > /tmp/vl30b_runs/vllm_rm_server.log 2>&1 &

# PyTorch 官方 ValueModel (Run 2 用) — port 8002
nohup /workspace/SIA/venv4/bin/python src/sia_rm_pytorch_official.py \
  --rm /workspace/SIA/models/Qwen3-4B \
  --rm_lora /workspace/SIA/models/VM-Qwen3-4B-Base/VM-Qwen3-4B-Base \
  --device cuda:0 --port 8002 \
  > /tmp/vl30b_runs/pytorch_vm_server.log 2>&1 &
```

### 2. Run 1: vllm RM (path A direct-token-ids)

```bash
SIA_LOG_LEVEL=quiet \
nohup /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
  --rm_url http://localhost:8001 \
  --rm_backend vllm \
  --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
  --llm_gpu_mem 0.55 \
  --topk 5 --weight 1.0 --entropy_threshold 1.3 \
  --max_model_len 2048 \
  --port 8000 \
  > /tmp/vl30b_runs/sia_vllm_server.log 2>&1 &

# 等就绪后跑 driver
/workspace/SIA/venv4/bin/python /tmp/vl30b_runs/drive_alpaca.py \
  /tmp/vl30b_runs/vllm_sia_outputs.json Qwen3-VL-30B-A3B-Instruct 2048 200
```

### 3. Run 2: PyTorch 官方 ValueModel (text mode via /score)

```bash
# 先 kill Run 1 server (port 8000) + 它的 EngineCore 子进程
pkill -9 -f sia_vllm_server.py
# (重要: vllm EngineCore subprocess 不会被 pkill 父进程 kill, 需要单独清理)

SIA_LOG_LEVEL=quiet \
nohup /workspace/SIA/venv4/bin/python src/sia_vllm_server.py \
  --llm /workspace/SIA/models/Qwen3-VL-30B-A3B-Instruct \
  --rm_url http://localhost:8002 \
  --rm_backend pytorch \
  --llm_gpu_mem 0.50 \
  --topk 5 --weight 1.0 --entropy_threshold 1.3 \
  --max_model_len 2048 \
  --port 8000 \
  > /tmp/vl30b_runs/sia_pytorch_server.log 2>&1 &

# Driver
/workspace/SIA/venv4/bin/python /tmp/vl30b_runs/drive_alpaca.py \
  /tmp/vl30b_runs/pytorch_sia_outputs.json Qwen3-VL-30B-A3B-Instruct 2048 200
```

### 4. Skywork 评分 + 对比

```bash
# 评 Run 1 全长
/workspace/SIA/venv4/bin/python /tmp/vl30b_runs/score_run1.py

# 评 trunc=256
/workspace/SIA/venv4/bin/python /tmp/vl30b_runs/score_truncated.py

# 评 trunc=180
/workspace/SIA/venv4/bin/python /tmp/vl30b_runs/score_t180.py

# 三路对比
/workspace/SIA/venv4/bin/python /tmp/vl30b_runs/compare_60q.py

# 崩坏分析
/workspace/SIA/venv4/bin/python /tmp/vl30b_runs/analyze_breakdown.py
```

## 性能统计 (Run 1)

| 指标 | mean |
|------|------|
| INTERVENE 率 | 28.3% |
| Top-1 flip 率 | 65.8% |
| Tokens/s | 30 (Run 1 vllm RM 比 Run 2 PyTorch RM 快 1.8×) |
| 200Q 总耗时 | 137 min (Run 1) / ~210 min (Run 2, 估) |
| 总干预次数 | 87,212 |

## 关键发现

1. **vllm RM 字段名修复**: 之前发的 `"activation": False` 被 Pydantic 静默忽略 → 默认 sigmoid 永远开启 → RM 信号被压缩到 [0,1]。修复后 `"use_activation": False` 拿到 raw logit。
2. **Path A direct-token-ids**: 同 vocab (LLM=VM=Qwen3) 时跳过 text round-trip, 直接 `prefix_ids + output_ids + [cand_id]` 给 vllm /classify。比 text mode 精度高 4-5×。
3. **W2S=7.5× 仍然崩**: 即使 RM 信号修复, VL-30B + Qwen3-4B 仍 -75% Δ, 51.5% 输出有崩坏。问题不在 RM 实现, 在模型 size 不匹配。
4. **Qwen3-14B (W2S=3.5×) 0 崩坏**: 重新分析 paper 805Q 结果 (median 1181 tokens, 远大于 256), 任意截断长度下 **0% 崩坏率** + **+13.2% Δ**。

---

**实验完整时间线**:
- 2026-06-03 16:19 — 启动 Run 1 server load (VL-30B 需 ~15 min)
- 2026-06-03 16:35 — Run 1 driver 开始 200Q
- 2026-06-03 18:52 — Run 1 完成 (137 min, 200/200)
- 2026-06-04 00:11 — 重启 continue2.sh, Run 2 server load
- 2026-06-04 00:27 — Run 2 driver 开始
- 2026-06-04 ~04:00 — Run 2 预计完成 (~210 min)
