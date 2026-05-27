# Value Model FP8_DYNAMIC 量化方案（A2 路线完整版）

**日期**：2026-05-26
**目标**：将 SIA 的 Value Model（`VM-Qwen3-4B-merged-for-vllm`）量化为 FP8，把 RM forward 算力时间从 ~50ms / call 压到 ~30-35ms，端到端吞吐预期从 G3 的 29.65 tok/s 提升至 ~33-36 tok/s（+10-22%）。
**约束**：**不影响干预质量**（reward ranking、SIA accuracy 与 BF16 等价）。
**与 G4 的关键区别**：G4 用 vLLM 的 runtime `--quantization fp8 --kv-cache-dtype fp8`，激活 scale 默认 1.0 触发退化路径；本方案用离线 `llm-compressor` 产出 **FP8_DYNAMIC** checkpoint，**动态激活量化（per-token 实时算 scale）**，无静态校准依赖。

---

## 0. 用户最在意的疑问的正面回答

> "用 MMLU 评测子集来校准，是不是假设了未来数据也要符合这个数据集的文本分布？"

**对 static 量化是 yes；对 FP8_DYNAMIC 是 no。**

FP8 量化里有两类 scale：

| Scale 类型 | 来源 | 数据分布依赖 |
|---|---|---|
| **Weight scale** | 直接从权重张量统计算出 | ❌ 无依赖 |
| **Activation scale (static)** | 校准数据跑一遍，离线固定一个 scale | ⚠️ **强依赖**：分布偏移导致质量下降 |
| **Activation scale (dynamic)** | 每个 forward step 用当前激活实时算 | ✅ **零依赖**：对任意输入都自适应 |

**FP8_DYNAMIC** = 权重静态 per-channel FP8 + 激活**动态 per-token** FP8。

- 权重 scale 只看权重张量本身，跟校准数据完全无关
- 激活 scale 每个 forward 现场算，**校准数据再有偏，也只影响"量化算子在哪些层上插入"，不影响 scale 数值**
- 所以 "calibration 集 = 部署集"这个假设**根本不存在**

`llm-compressor` 的 oneshot 流程在 FP8_DYNAMIC 下需要传几十条样本，但作用是**让 modifier 遍历模型图、正确插入量化算子**，不是拟合分布。传 MMLU、AlpacaEval、随机文本，**产出的量化模型是完全一样的**。

---

## 0.5 硬件前置要求

**FP8 E4M3 算子需要 NVIDIA GPU 计算能力 sm_89 及以上**：

| GPU | sm 版本 | 本方案能跑？ |
|---|---|---|
| H100, H200 | sm_90 | ✅ |
| RTX 4090, L40, L40S, RTX 6000 Ada | sm_89 | ✅ |
| A100 (40/80GB) | sm_80 | ❌（会报错或 emulation 退化） |
| V100, T4, RTX 3090 | sm_70 / sm_75 / sm_86 | ❌ |

本项目部署在 **H200 上没问题**。如果搬到 A100，本方案不能直接用，需要走 INT8 / AWQ 路线。

启动前自检：
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# 期望: NVIDIA H200, 9.0  （或 H100 9.0 / RTX 4090 8.9 / L40 8.9 等）
```

## 1. 背景：为什么是 A2，为什么不是别的

经过 17 组实验和 C1/C2 实测，瓶颈定位明确：

```
per-RM-call ≈ 78ms (G3) ≈ 60ms (C1+C2 profile total)

拆开看：
  RM 真实 forward 算力：~40-50ms  ← 这是大头，FP8 能动
  HTTP/scheduling/parse: ~10-15ms  ← C1/C2 已经摸过，余量小
  客户端 tokenize: ~5-7ms          ← C2 把这个工作从服务端搬到客户端，零净收益
```

C1+C2 收益接近零（实测 29.65 → 29.43 tok/s）的原因：动的是"协议层"，攻不到核心算力。**A2 直接攻 RM forward 算力**，是当前共享 GPU 部署下唯一还能动且零质量损失的牌。

详细分析见 `doc/profiling-bottleneck-analysis.md`。

---

## 2. 方案细节

### 2.1 量化目标层

`VM-Qwen3-4B-merged-for-vllm` 的层结构（已从 config.json 确认）：

```
36 个 transformer decoder layer，每层 7 个 Linear:
  self_attn: q_proj, k_proj, v_proj, o_proj
  mlp:       gate_proj, up_proj, down_proj
= 252 个 Linear → 全部量化为 FP8_DYNAMIC
+ embed_tokens (Linear-like, 不量化)
+ score head (Linear(2560, 1), 不量化 ← reward 精度关键)
```

### 2.2 ignore list（关键）

```python
ignore = [
    "score",         # classifier head，reward 精度
    "lm_head",       # 兜底（即便此模型实际无 lm_head）
    "embed_tokens",  # embedding lookup 不走 FP8 GEMM 路径
]
```

⚠️ **如果忘记 ignore "score"，reward 数值精度会大幅退化**，因为 score head 是 `Linear(2560, 1)` 的极小矩阵，FP8 量化误差会被放大。

### 2.3 KV cache 保持 BF16

明确**不动 KV cache**。理由：
- vLLM 0.10.1.1 的 `--kv-cache-dtype fp8` 在没有校准过的 q/k/v/prob scale 时触发退化路径（G4 实测 -7%）
- 校准 KV scale 需要额外流程，复杂度上一个台阶
- 单纯量化权重 + 动态激活已经能拿大部分 FP8 算力红利

### 2.4 已知待验证项

下面这条**未在本仓库实测过**，量化跑通后第一时间需验：

> **vLLM 0.10.1.1 的 `--runner pooling --convert classify` 是否能在 compressed-tensors 格式的 checkpoint 上正确挂 classifier head？**

之前 G4 用的是 vLLM 的 runtime FP8（`--quantization fp8`），并未走 compressed-tensors 路径。本方案换成离线 compressed-tensors checkpoint，`--convert classify` 的行为是新组合，存在两种可能：

1. ✅ 正常：vLLM 加载 compressed-tensors 权重后，仍按 `--convert classify` 套上 score head 的 pooler 行为
2. ❌ 失败：vLLM 加载逻辑分支让两个 flag 互斥（compressed-tensors 已认为模型自带分类头，再 `--convert classify` 报错）

→ Phase 2 启动 log 第一时间确认。如果失败转 §4 R3 应对。

### 2.5 量化方案配置

```python
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets=["Linear"],   # 用 list 形式，旧版接受 str、新版要求 list，list 双兼容
    scheme="FP8_DYNAMIC",
    ignore=["score", "lm_head", "embed_tokens"],
)
```

**`FP8_DYNAMIC` scheme 内部行为**：
- Weights: `static FP8 E4M3` per-output-channel，scale 从权重统计算出
- Activations: `dynamic FP8 E4M3` per-token，scale 在 forward 时按当前 batch 现算
- KV cache: 不动（不在 Linear 量化范围内）

---

## 3. 完整工作流

### Phase 1：离线量化（~20 分钟，一次性）

#### Step 1.0 — **验证 score head 真实模块名**（必做，否则 ignore list 漏写 = reward 精度灾难性下降）

`ignore=["score"]` 是按假设写的。`Qwen3ForSequenceClassification` 的 classifier head 实际可能叫 `score`、`model.score`、`classifier`、`score.dense` 等。**ignore 不匹配 = 整个 head 被量化 = reward 精度灾难性下降**，所以量化前必须先确认真实名字：

```python
from transformers import AutoModelForSequenceClassification
import torch.nn as nn

m = AutoModelForSequenceClassification.from_pretrained(
    "/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm",
    torch_dtype="auto", trust_remote_code=True,
)
print("=== 所有 nn.Linear 模块（含 classifier head 候选）===")
for name, mod in m.named_modules():
    if isinstance(mod, nn.Linear):
        out_features = mod.out_features
        # classifier head 特征：输出维度极小（通常 1 或 num_labels）
        if out_features <= 16:
            print(f"  *** classifier head 候选: {name} -> Linear({mod.in_features}, {out_features})")

print("\n=== embedding 模块的全名 ===")
for name, mod in m.named_modules():
    if isinstance(mod, nn.Embedding):
        print(f"  embedding: {name} -> Embedding({mod.num_embeddings}, {mod.embedding_dim})")
```

期望输出形如：
```
*** classifier head 候选: score -> Linear(2560, 1)
embedding: model.embed_tokens -> Embedding(151936, 2560)
```

**用这两个真实名字（不去前缀，完整模块路径）替换 Step 1.1 里 `ignore=[...]` 默认的 `"score"` 和 `"embed_tokens"`**。常见 Qwen3 真实形式：

| 默认占位 | 真实可能 |
|---|---|
| `"score"` | `"score"` 或 `"classifier"` |
| `"embed_tokens"` | `"model.embed_tokens"`（带 `model.` 前缀） |
| `"lm_head"` | `"lm_head"`（保留以防万一，即便此模型可能无 lm_head）|

#### Step 1.1 — 写量化脚本

新建 `scripts/quantize_rm_fp8_dynamic.py`：

```python
"""
将 VM-Qwen3-4B-merged-for-vllm 量化为 FP8_DYNAMIC，保留 score head 为 BF16。

用法:
    python scripts/quantize_rm_fp8_dynamic.py
        [--src /path/to/source] [--dst /path/to/output] [--n_samples 50]
"""
import argparse, random
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def build_calibration_dataset(tok, n_samples: int = 50, max_len: int = 1024):
    """构造 calibration 样本。FP8_DYNAMIC 不依赖分布，但 oneshot 需要一些样本走模型图。

    样本格式必须与部署时 RM 实际看到的文本一致：chat-formatted user+assistant turn。
    """
    # 给 30 条多样化样本：技术 / 科学 / 人文 / 数学 / 闲聊 / 多语言 / 长短文 各类
    # 注：FP8_DYNAMIC 不依赖数据分布，但为了让 oneshot 在多形态文本上跑过模型图，
    # 多样化的样本能更好覆盖所有 Linear 层的激活路径（即便 scale 仍是 per-token 动态算）
    sample_pairs = [
        # 数学 / 算术
        ("What is 2+2?", "The answer is 4."),
        ("Calculate 15% of 200.", "30."),
        ("Solve x^2 = 25.", "x = 5 or x = -5."),
        # 科学 / 自然
        ("Explain photosynthesis briefly.", "Plants convert sunlight to energy via chlorophyll."),
        ("What is the boiling point of water?", "100 degrees Celsius at sea level."),
        ("Why is the sky blue?", "Rayleigh scattering of sunlight by air molecules."),
        ("Name 3 noble gases.", "Helium, neon, argon."),
        # 人文 / 历史 / 文学
        ("Who wrote Hamlet?", "William Shakespeare."),
        ("When did World War II end?", "September 2, 1945."),
        ("Capital of France?", "Paris."),
        ("Translate 'hello' to Spanish.", "Hola."),
        # 编程 / 技术
        ("What is recursion in programming?", "A function calling itself with smaller inputs."),
        ("Difference between TCP and UDP?", "TCP is connection-oriented and reliable; UDP is connectionless."),
        ("What does HTTP stand for?", "HyperText Transfer Protocol."),
        # 日常 / 列表
        ("List 3 colors.", "Red, blue, green."),
        ("Suggest a quick breakfast.", "Toast with peanut butter and a banana."),
        ("Name 2 musical instruments.", "Piano and guitar."),
        # 长一些的指令
        ("Write a short poem about autumn.",
         "Leaves of amber dance in fading light, "
         "the crisp air whispers summer's end."),
        ("Explain the theory of relativity in one sentence.",
         "Space and time are interwoven, and both bend under gravity and motion."),
        ("Summarize the plot of Romeo and Juliet.",
         "Two young lovers from feuding families secretly marry, "
         "then die by misunderstanding."),
        # 推理 / 多步
        ("If a train travels 60 mph for 2 hours, how far did it go?", "120 miles."),
        ("Is 17 prime?", "Yes, 17 has only 1 and 17 as divisors."),
        # 否定 / 偏见检查
        ("What's the capital of the moon?", "The moon has no capital city; it's not a country."),
        # 中英混合 / 多语言
        ("北京的首都是哪？", "北京就是中国的首都。"),
        ("用英文说'谢谢'。", "Thank you."),
        # 代码
        ("Python print 'hello world'.", "print('hello world')"),
        ("What does this Python do: x = [1,2,3]; print(sum(x))?", "It prints 6, the sum of the list."),
        # 哲学 / 开放
        ("Is free will real?", "Philosophers debate this; both compatibilist and libertarian views exist."),
        ("Define consciousness.", "Subjective awareness of one's surroundings and inner experience."),
        # 计算密集 / 长输入
        ("List 10 fruits.",
         "Apple, banana, orange, grape, mango, strawberry, watermelon, pineapple, kiwi, peach."),
    ]
    random.seed(42)
    # 用尽 30 条多样样本，多出的部分按需轮转
    samples = (sample_pairs * (n_samples // len(sample_pairs) + 1))[:n_samples]

    texts = []
    for user, assistant in samples:
        convs = [
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ]
        text = tok.apply_chat_template(convs, tokenize=False)
        bos = tok.bos_token
        if bos and text.startswith(bos):
            text = text[len(bos):]
        texts.append(text)

    encodings = tok(
        texts, truncation=True, max_length=max_len, padding=False,
        return_tensors=None,
        return_attention_mask=True,   # 显式要 attention_mask，transformers 不同版本默认行为不一致
    )
    # 必须含 attention_mask，llm-compressor 的 data collator 默认需要两个字段
    return Dataset.from_dict({
        "input_ids":      encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm")
    p.add_argument("--dst", default="/workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic")
    p.add_argument("--n_samples", type=int, default=50)
    args = p.parse_args()

    print(f"[quantize] Loading model from {args.src}", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.src,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",   # 让模型加载到 GPU，oneshot forward 才能用 GPU 算激活
    )
    tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)

    print(f"[quantize] Building calibration dataset ({args.n_samples} samples)", flush=True)
    ds = build_calibration_dataset(tok, n_samples=args.n_samples)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["score", "lm_head", "embed_tokens"],
    )

    print(f"[quantize] Running oneshot → {args.dst}", flush=True)
    # 故意不传 num_calibration_samples / max_seq_length 这类 kwarg：
    #   - 不同 llm-compressor 版本对这两个名字接受程度不一致，传错会 TypeError
    #   - 校准样本数已由 dataset 长度决定
    #   - 序列长度已在 build_calibration_dataset 里截断到 max_len=1024
    # 只传必要的 4 个 kwarg，向前向后版本兼容最稳
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        output_dir=args.dst,
    )
    tok.save_pretrained(args.dst)
    print("[quantize] Done", flush=True)


if __name__ == "__main__":
    main()
```

#### Step 1.2 — 跑量化

```bash
# 量化端的库（产出 checkpoint）
pip install llmcompressor==0.5.1            # 或当前稳定版

# RM server 端运行时也需要的库（vLLM 加载 compressed-tensors checkpoint 必装）
# ⚠️ 版本必须与 llmcompressor 配对，不匹配可能加载失败：
#    llmcompressor 0.5.1 对应 compressed-tensors 0.7.x
#    llmcompressor 0.4.x 对应 compressed-tensors 0.6.x
# 实际版本以 llmcompressor 安装时拉下来的 compressed-tensors 为准，
# 跨进程时（量化端 vs vLLM 端）务必同版本
pip install compressed-tensors==0.7.1       # 与 llmcompressor 0.5.1 配对版本

python scripts/quantize_rm_fp8_dynamic.py \
    --src /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --dst /workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic
```

跑前用 `pip show llmcompressor compressed-tensors` 确认两边版本对齐。

#### Step 1.3 — 产物验证

```bash
# 1. 权重大小应从 ~7.5 GB 降到 ~4 GB
du -sh /workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic/

# 2. config.json 应含 quantization_config，且 ignore list 正确
python -c "
import json
c = json.load(open('/workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic/config.json'))
qc = c.get('quantization_config', {})
print('format:', qc.get('format'))
print('scheme:', qc.get('config_groups', {}))
print('ignore:', qc.get('ignore'))
"
# 期望:
#   format: 'float-quantized'
#   ignore: ['score', 'lm_head', 'embed_tokens']
#   config_groups 里 weights.num_bits=8 type='float' strategy='channel'
#                   input_activations.num_bits=8 type='float' strategy='token' dynamic=True

# 3. 检查 score head 真的没被量化（应为 BF16）
python -c "
import safetensors.torch as st
import glob
for f in sorted(glob.glob('/workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic/*.safetensors')):
    keys = st.load_file(f).keys()
    for k in keys:
        if 'score' in k:
            t = st.load_file(f)[k]
            print(f'{k}: dtype={t.dtype} shape={t.shape}')
"
# 期望: score.weight dtype=torch.bfloat16
```

### Phase 2：启动 RM + 烟测

**前置**：先确保端口 8001 / 8000 没被旧进程占用：
```bash
pkill -9 -f "vllm serve|sia_vllm_server|VLLM::EngineCore|vllm_serve_with_token_ids" 2>/dev/null
sleep 2
ss -tlnp 2>/dev/null | grep -E ":800[01]" && echo "端口仍有进程占用，手工处理" || echo "(端口已空闲)"
```

```bash
# 注意：不加 --quantization fp8 flag (那是 runtime quant，与 compressed-tensors 冲突)
# 首选: vLLM 0.10.1.1 自动从 config.json 的 quantization_config 识别
nohup python scripts/vllm_serve_with_token_ids.py serve \
      /workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic \
      --runner pooling \
      --convert classify \
      --enable-prefix-caching \
      --no-enable-chunked-prefill \
      --gpu-memory-utilization 0.3 \
      --max-model-len 2048 \
      --port 8001 \
      > log_vllm_rm_fp8d_$(date +%Y%m%d%H%M).txt 2>&1 &
```

**Fallback**：若启动 log 报 "unsupported quantization config" 或 "no quantization detected"，自动识别失败时显式加 flag：

```bash
# 在上面命令里加
--quantization compressed-tensors \
```

如果 `--quantization compressed-tensors` 仍报错，说明 vLLM 0.10.1.1 不支持当前 compressed-tensors 版本的格式，转 R3 风险分支。

**启动 log 必看几行**：

| Pattern | 期望 |
|---|---|
| `Model loading took X.X GiB` | ~**4.X GiB**（不是 7.5）|
| `quantization=...` 字段 | 应能识别 compressed-tensors / fp8 |
| `Using KV cache scaling factor 1.0` | **不应出现**（我们没动 KV）|
| `q_scale / k_scale / prob_scale` 警告 | **不应出现** |
| `[vllm-rm-patch]` | 应出现（Pydantic patch）|

**起来后立刻 sanity check**（先等 vLLM ready，再查 reward 数值）：

```bash
# 0) 先等 vLLM 完全 ready（启动 + CUDA graph capture 通常 30-60s）
echo "Waiting for vLLM ready..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
    echo "vLLM ready after ${i}s"
    break
  fi
  sleep 2
done
# 兜底：检查 log 末尾有 "Application startup complete"
grep "Application startup complete" log_vllm_rm_fp8d_*.txt | tail -1 || {
  echo "vLLM still not ready, abort sanity"; exit 1;
}

# 1) 基础 reward 数值 sanity（含 NaN/Inf 检测）
python - <<'EOF'
import math, requests
resp = requests.post("http://localhost:8001/classify", json={
    "model": "/workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic",
    "input": [
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n5<|im_end|>",
    ],
    "activation": False,
})
resp.raise_for_status()
data = sorted(resp.json()["data"], key=lambda x: x.get("index", 0))
scores = [d["probs"][0] for d in data]
print("scores:", scores)

# 必须断言（任意一条 fail 立即停手）
assert all(not math.isnan(s) and not math.isinf(s) for s in scores), \
    f"FP8 RM 返回 NaN/Inf: {scores}"
assert scores[0] > scores[1], \
    f"FP8 RM 给 '4' 的分数不应低于 '5'：scores={scores}"
print("Sanity pass.")
EOF
```

如果 sanity 不过（NaN/Inf、reward 数值倒置、HTTP 500 等），**立刻回退到 BF16，进 §4.1 回退树**，A2 路线放弃。

### Phase 3：质量 A/B 验证（不可跳过的 gate，**sequential 跑**）

⚠️ **不要同时起 BF16 + FP8 两个 vLLM**。同卡上跑两个 vLLM server：(a) 显存得各分 0.25，(b) 互相抢 GPU context 切换，**latency 数据会失真**，reward 对比也会引入额外噪声。**改成 sequential**：先跑 BF16 收集结果存盘，kill，再起 FP8 跑同样输入，最后对比两份存盘文件。

新建 2 个脚本：

**`scripts/collect_rm_scores.py`**（同一个脚本，跑两次，分别 query BF16/FP8 server，输出到不同 JSON）：

```python
"""
用法（sequential 跑两次）：
    # 先起 BF16 RM 在 8001
    python scripts/collect_rm_scores.py --label bf16 --port 8001 \
        --out results/ab_bf16.json
    # 杀掉 BF16 RM
    # 起 FP8 RM 在 8001
    python scripts/collect_rm_scores.py --label fp8 --port 8001 \
        --out results/ab_fp8.json

输入：固定 seed 抽取的 100 条 MMLU + 50 条 AlpacaEval prompts，
     每条 prompt 取 LLM 的 top-5 candidates（chat-formatted 文本，5 个一组）。
"""
import argparse, json, math, requests


def query_rm(port, model, formatted_texts):
    resp = requests.post(
        f"http://localhost:{port}/classify",
        json={"model": model, "input": formatted_texts, "activation": False},
        timeout=30,
    )
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda x: x.get("index", 0))
    return [d["probs"][0] for d in data]


def main():
    import os
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, choices=["bf16", "fp8"])
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--model", required=True, help="RM 模型路径，与启动 server 时一致")
    p.add_argument("--prompts_file", default="results/ab_prompts.json",
                   help="预生成的固定 prompts + 5-candidate 集合")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if not os.path.exists(args.prompts_file):
        raise FileNotFoundError(
            f"{args.prompts_file} 不存在。请先跑 scripts/gen_ab_prompts.py "
            f"生成 A/B 用的固定 prompt 集合。"
        )
    prompts = json.load(open(args.prompts_file))  # list[ {prompt_id, candidates: [5 chat-formatted texts]} ]
    out = []
    for item in prompts:
        scores = query_rm(args.port, args.model, item["candidates"])
        assert all(not math.isnan(s) and not math.isinf(s) for s in scores), \
            f"{args.label} RM returned NaN/Inf at prompt {item['prompt_id']}: {scores}"
        out.append({"prompt_id": item["prompt_id"], "scores": scores})

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Wrote {len(out)} entries to {args.out}")


if __name__ == "__main__":
    main()
```

**`scripts/compare_rm_ab.py`**（读两个 JSON，输出统计）：

```python
"""分析 sequential 收集的 BF16 vs FP8 reward 对比。"""
import argparse, json
import numpy as np
from scipy.stats import pearsonr, spearmanr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bf16", required=True)
    p.add_argument("--fp8",  required=True)
    args = p.parse_args()

    bf16 = json.load(open(args.bf16))
    fp8  = json.load(open(args.fp8))
    # 按 prompt_id 对齐
    bf16_map = {x["prompt_id"]: x["scores"] for x in bf16}
    fp8_map  = {x["prompt_id"]: x["scores"] for x in fp8}
    common = sorted(set(bf16_map) & set(fp8_map))
    assert common, "无对齐的 prompt_id"

    import math
    all_bf16, all_fp8 = [], []
    top1_agree = 0
    spearman_pass = 0
    for pid in common:
        b, f = bf16_map[pid], fp8_map[pid]
        all_bf16.extend(b)
        all_fp8.extend(f)
        if int(np.argmax(b)) == int(np.argmax(f)):
            top1_agree += 1
        sp = spearmanr(b, f).correlation
        # NaN 出现于：所有 5 个分数完全相同 (constant input)；视为完美一致计入 pass
        if math.isnan(sp) or sp > 0.8:
            spearman_pass += 1

    pearson = pearsonr(all_bf16, all_fp8).statistic
    n = len(common)
    print(f"=== A/B 报告 (n={n} prompts × 5 candidates = {len(all_bf16)} 个数据点) ===")
    print(f"Pearson (absolute reward):              {pearson:.4f}      期望 ≥ 0.99")
    print(f"Top-1 candidate agreement:              {top1_agree}/{n} ({top1_agree/n*100:.1f}%)   期望 ≥ 95%")
    print(f"Per-group Spearman > 0.8 比例:           {spearman_pass}/{n} ({spearman_pass/n*100:.1f}%)   期望 ≥ 90%")
    print(f"BF16 reward 中位/max: {np.median(all_bf16):.4f} / {np.max(all_bf16):.4f}")
    print(f"FP8  reward 中位/max: {np.median(all_fp8):.4f} / {np.max(all_fp8):.4f}")
    drift = abs(np.median(all_fp8) - np.median(all_bf16)) / (abs(np.median(all_bf16)) + 1e-6)
    print(f"中位绝对漂移: {drift*100:.1f}%   期望 < 10%")


if __name__ == "__main__":
    main()
```

**`scripts/gen_ab_prompts.py`**（一次性生成 A/B 用的固定 prompt 集合）：

```python
"""
从 MMLU + AlpacaEval 抽 150 个 prompt，每个用 LLM 跑出 top-5 candidates 的
chat-formatted 文本（5 个候选 = 取 LLM logits 的 top-5 个 token，分别 append 到
response_so_far 后 + chat template）。

输出 JSON 结构:
[
  {"prompt_id": "mmlu_anatomy_0", "candidates": [text1, text2, text3, text4, text5]},
  ...
]

注：candidates 应是真实部署时 SIA 第一步会送给 RM 的 5 个候选 (即 LLM step 0
的 top-5 token append 到一个空 response_so_far)，保证校准数据 = 部署分布。
最简化版（仅用于 A/B 验证 reward 数值是否一致，不要求是真实 top-5）：
固定 5 个 candidate token (如 ' A', ' B', ' C', ' D', ' E') 拼到 chat template 末尾。
"""
import argparse, json, random
from datasets import load_dataset
from transformers import AutoTokenizer

LLM = "/workspace/SIA/models/Qwen3-14B"
FIXED_CANDIDATES = [" A", " B", " C", " D", " E"]  # 简化版

def chat_format(tok, user, assistant_text):
    convs = [
        {"role": "user",      "content": user},
        {"role": "assistant", "content": assistant_text},
    ]
    text = tok.apply_chat_template(convs, tokenize=False)
    bos = tok.bos_token
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/ab_prompts.json")
    p.add_argument("--n_mmlu", type=int, default=100)
    p.add_argument("--n_alpaca", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    out = []

    # 100 条 MMLU，跨 5 个 subject 各取 20 题，更代表真实 30-subject 分布
    # ⚠️ load_dataset 需要联网（首次下载到 ~/.cache/huggingface）。
    # 离线环境请预先 `python -c "from datasets import load_dataset; load_dataset('cais/mmlu', 'all')"`
    # 或手动准备 JSON。
    MMLU_SUBJECTS = ["anatomy", "clinical_knowledge", "astronomy",
                     "high_school_mathematics", "philosophy"]
    per_subject = args.n_mmlu // len(MMLU_SUBJECTS)
    mmlu_samples = []
    try:
        for subj in MMLU_SUBJECTS:
            ds = load_dataset("cais/mmlu", subj, split="test")
            mmlu_samples += [(subj, ex) for ex in random.sample(list(ds), min(per_subject, len(ds)))]
    except Exception as e:
        print(f"⚠️ MMLU 加载失败（可能是网络/缓存问题）: {e}")
        print(f"   已收集 {len(mmlu_samples)} 条，继续后续步骤")
    for i, (subj, ex) in enumerate(mmlu_samples):
        q = ex["question"] + "\n" + "\n".join([f"{c}: {v}" for c, v in zip("ABCD", ex["choices"])])
        # 简化: 5 candidates = 同 prompt + 5 个固定后缀
        cands = [chat_format(tok, q, "Answer:" + s) for s in FIXED_CANDIDATES]
        out.append({"prompt_id": f"mmlu_{subj}_{i}", "candidates": cands})

    # 50 条 AlpacaEval
    try:
        alpaca = load_dataset("tatsu-lab/alpaca_eval", split="eval")
        alpaca_sub = random.sample(list(alpaca), min(args.n_alpaca, len(alpaca)))
        for i, ex in enumerate(alpaca_sub):
            q = ex["instruction"]
            cands = [chat_format(tok, q, "Sure" + s + ".") for s in FIXED_CANDIDATES]
            out.append({"prompt_id": f"alpaca_{i}", "candidates": cands})
    except Exception as e:
        print(f"AlpacaEval 加载失败，跳过: {e}")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} prompts → {args.out}")

if __name__ == "__main__":
    main()
```

**Sequential 跑的流程**：
```bash
# 0. 一次性生成固定 prompts 集合 (results/ab_prompts.json)
python scripts/gen_ab_prompts.py   # 见上方草稿，输出 ~150 prompt × 5 candidates

# 1. 起 BF16 RM
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --runner pooling --convert classify ... --port 8001 &
python scripts/collect_rm_scores.py --label bf16 --port 8001 \
    --model /workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm \
    --out results/ab_bf16.json

# 2. 杀 BF16 RM（包括 EngineCore 子进程；之前杀进程时发现 pkill 模式如果只匹配
#     "vllm serve" 会漏掉 VLLM::EngineCore 这个 spawn 出来的子进程）
pkill -9 -f "VM-Qwen3-4B-merged-for-vllm|VLLM::EngineCore" 2>/dev/null
sleep 5
# 验证显存释放（H200 idle 时单卡占用应回落到 ~几百 MB；如仍有 GB 级占用说明有子进程没死透）
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# 3. 起 FP8 RM（同样端口 8001）
nohup vllm serve /workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic ... --port 8001 &
python scripts/collect_rm_scores.py --label fp8 --port 8001 \
    --model /workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic \
    --out results/ab_fp8.json

# 4. 对比
python scripts/compare_rm_ab.py --bf16 results/ab_bf16.json --fp8 results/ab_fp8.json
```

**Pass criteria（必须全部满足才进 Phase 4）**：

| 指标 | 阈值 | 说明 |
|---|---|---|
| Pearson 相关系数 (absolute reward) | **≥ 0.99** | 数值上几乎等价 |
| Top-1 candidate agreement | **≥ 95%** | argmax 不变，SIA 干预决策一致 |
| Per-group Spearman > 0.8 比例 | **≥ 90%** ※※ | 5 候选内部排序一致。注：n=5 时 Spearman > 0.8 对应 p ≈ 0.1，统计显著性偏弱；但 5 candidates 本身样本少，没办法做更严，把它当弱信号用，主信号看 Pearson 和 Top-1 |
| Reward 中位绝对值漂移 | **|Δ| < 30%** ※ | scale 大致稳定。SIA 内部 `rm_scores - mean()` 会抵消绝对偏移，所以 30% 已较保守。**若此项 fail 但 Pearson + Top-1 + Spearman 三项全过，可放行**（注：Pearson 已包含 scale 一致性的更严格判定，median 漂移只是简易指标） |
| 端到端 MMLU 100 题 accuracy 差 | **|Δ| < 1%pt** | 整体质量不掉 |

任意一项 fail → 见 §6 回退方案。

### Phase 4：性能 benchmark

跑 600 题完整 MMLU 评测。**RM 沿用 Phase 2 起的实例**（已经在 8001 跑、sanity 过了，没必要重启）；**只需新起 LLM + eval**。两者都 nohup + log 重定向到 `/tmp/`——这样即便交互终端（如 Claude 会话）退出，后台进程仍然继续跑：

```bash
TS=$(date +%Y%m%d%H%M)

# === 前置：确认 Phase 2 的 RM 还活着 ===
# RM log 路径就是 Phase 2 启动时用的（本次实跑是 /tmp/vllm_rm_fp8d_phase2.txt）
ps aux | grep -E "vllm_serve_with_token_ids|VM-Qwen3-4B-merged-fp8-dynamic" | grep -v grep
curl -sf http://localhost:8001/health && echo "RM 8001 OK" || { echo "RM 没起，回 Phase 2 先把 RM 跑起来"; exit 1; }

# === 1) LLM server (C1 自动开 + C2 通过 flag 开) ===
nohup python src/sia_vllm_server.py \
      --llm /workspace/SIA/models/Qwen3-14B \
      --rm_url http://localhost:8001 \
      --rm_backend vllm \
      --rm_model /workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic \
      --llm_gpu_mem 0.55 \
      --topk 5 --weight 1.0 --entropy_threshold 1.0 \
      --use_token_ids \
      --host 0.0.0.0 --port 8000 \
      > /tmp/sia_llm_fp8d_${TS}.log 2>&1 &
echo "LLM PID: $!"

# 等 LLM ready（启动约 90-120s：weight load 60s + torch.compile 70s + CUDA graph 5s）
LLM_LOG=/tmp/sia_llm_fp8d_${TS}.log
until grep -q "Application startup complete\|Uvicorn running" $LLM_LOG 2>/dev/null \
   || grep -qE "Traceback|Error in|raise " $LLM_LOG 2>/dev/null
do sleep 5; done

# === 2) MMLU eval ===
mkdir -p results
nohup python eval/mmlu_eval.py \
      --base_url http://localhost:8000/v1 \
      --model /workspace/SIA/models/Qwen3-14B \
      --output results/test_vllmrm_fp8d_${TS}.json \
      --limit 20 \
      > /tmp/sia_eval_fp8d_${TS}.log 2>&1 &
echo "EVAL PID: $!"
echo "TS=$TS"
echo "LLM + eval 两个进程后台跑，可安全退出当前会话"
```

跑完后查结果（注意 RM log 路径与 LLM/eval 不同，因为 RM 是 Phase 2 起的）：
```bash
# 查最终吞吐 + accuracy
tail -20 /tmp/sia_eval_fp8d_${TS}.log
# 查 profiling aggregate（看 http_post 是否降下来了）
grep "SIA-pf-summary" /tmp/sia_llm_fp8d_${TS}.log | tail -1
# 查 RM 端 prefix cache hit rate（RM 的 log 是 Phase 2 那个）
grep "Prefix cache hit rate" /tmp/vllm_rm_fp8d_phase2.txt | tail -3
```

**完全冷启动场景**（Phase 2 RM 已经被 kill 或机器重启过）：先回 §Phase 2 起 RM，再回这里。或者一次性把 3 个服务一起起：

```bash
# === 完全冷启动：3 个进程一起起（备用） ===
TS=$(date +%Y%m%d%H%M)

# 1) RM
nohup python scripts/vllm_serve_with_token_ids.py serve \
      /workspace/SIA/models/VM-Qwen3-4B-merged-fp8-dynamic \
      --runner pooling --convert classify \
      --enable-prefix-caching --no-enable-chunked-prefill \
      --gpu-memory-utilization 0.3 \
      --max-model-len 2048 --port 8001 \
      > /tmp/sia_vllm_rm_fp8d_${TS}.log 2>&1 &
RM_LOG=/tmp/sia_vllm_rm_fp8d_${TS}.log
until grep -q "Application startup complete" $RM_LOG || \
      grep -qE "Traceback|Error in|raise " $RM_LOG; do sleep 5; done

# 2) LLM (同上)，3) eval (同上)
# ……
```

**期望读数**：

| 指标 | C1+C2 baseline | A2 (本方案) 预期 | 改善 |
|---|---:|---:|---:|
| `[SIA-pf-summary] http_post` p50 | 53.89ms | **~38-42ms** | -25% |
| `[SIA-pf-summary] total` p50 | 59.84ms | **~45-50ms** | -20% |
| 整体吞吐 | 29.43 tok/s | **~33-36 tok/s** | **+12-22%** |
| 准确率 | 72.0% | 71-73% | 噪声内 |

如果 http_post 没明显下降（< 5%），说明 FP8 GEMM 加速没有真正生效，需要诊断（见 §6.3）。

---

## 4. 风险矩阵 + 回退方案

| # | 风险 | 概率 | 检测点 | 应对 |
|---|---|---|---|---|
| R1 | `llm-compressor` 加载 `Qwen3ForSequenceClassification` 失败 | 中 | Step 1.2 oneshot 报错 | 改用 `AutoModelForCausalLM` 加载 + 手动加 score head；或 fork 兼容版本 |
| R1b | `QuantizationModifier(targets=...)` API 变更（str vs list 或要求正则） | 低-中 | oneshot 启动报 TypeError 或 ValueError | 我们已用 list 形式 `targets=["Linear"]`；如仍报错试 `targets="re:.*\\.proj$"` 这种正则形式，或按新版 API 文档调整 |
| R2 | `ignore=["score"]` 不生效，score head 被量化 | 低 | Step 1.3 验证脚本看到 `score.weight` 是 FP8 | 改 ignore 写法（如 `["model.score"]`、`["^score$"]`）；或量化后手动覆盖 score 权重为 BF16 |
| R3 | vLLM 加载 FP8_DYNAMIC checkpoint 报"unsupported quantization" | 中 | Phase 2 启动失败 | 先试 `--quantization compressed-tensors` 显式 flag；不行升 vLLM 0.10.x → 0.11+，**但升级 vLLM 是大改动**，本仓库 `src/sia_vllm_RM.py` 用了 `vllm.v1.sample.logits_processor.interface` 的 `BatchUpdate / MoveDirectionality` 内部 API，需同步验证兼容性；最后回退到 static FP8 scheme |
| R4 | Pearson < 0.95 | 中-低 | Phase 3 fail | 扩 ignore 列表（加最后 2-3 个 transformer block）；或试 `FP8` static scheme |
| R5 | CUDA graph 在 FP8 路径上 capture 失败 | 低 | 启动 log 看到 `capture_size` 相关 error | 加 `--enforce-eager` 关 graph（损失 5-10% 速度但能跑） |
| R6 | reward scale 漂移大（|Δ| > 20%）| 低 | Phase 3 显示 BF16 vs FP8 reward 中位差 > 20% | 调 `--weight` 适配新 scale；或扩 ignore |
| R7 | http_post p50 没降，FP8 加速失效 | 中 | Phase 4 实测 http_post ≈ 54ms | 检查 vLLM 是否真在 FP8 路径上 forward（用 nsys profile 或 `torch.cuda.is_current_stream_capturing`）；可能需要 vLLM 升级 |
| R8 | Phase 1 oneshot OOM（4B 模型 + activation 存储）| 低 | oneshot 时 OOM | 降 `n_samples`、降 `max_seq_length`；或在多 GPU 上跑 |
| R9 | 量化产物存了一半被中断 | 低 | dst 目录不完整 | 删 dst 重跑 |
| R10 | 跟 wrapper `vllm_serve_with_token_ids.py` 不兼容 | 极低 | Phase 2 启动报错 | wrapper 只 patch Pydantic schema，与 FP8 加载逻辑无关；理论不冲突 |

### 4.1 总体回退顺序

```
Phase 3 fail (Pearson < 0.95)
  ↓
  尝试: scheme="FP8" (static, 带校准)
  ↓ fail
  尝试: 扩 ignore list (加最后 4 layers)
  ↓ fail
  放弃 A2，回 BF16 G3 配置（29.65 tok/s 是当前可达最优）

Phase 4 fail (http_post 没降)
  ↓
  诊断: vLLM 是否真用 FP8 kernels?
  ↓ 不是
  升级 vLLM 0.10.1.1 → 0.11+
  ↓ 还不行
  放弃 A2
```

### 4.2 如何识别"GPU 上真的在跑 FP8"

vLLM 启动后用一个长 prompt 触发 forward，期间在另一个终端：

```bash
nvidia-smi dmon -s u  # 看 SM 利用率
# 或更准确：
nsys profile -t cuda --output=fp8_check.qdrep python -c "import requests; requests.post(...)"
nsys stats fp8_check.qdrep | grep -i "fp8\|e4m3"
```

应能看到 FP8 GEMM kernel（如 `cutlass_fp8_e4m3_gemm`）被调用。

---

## 5. 与 G4 失败的精确对照

| 维度 | G4（失败，-7%）| A2（本方案）|
|---|---|---|
| 量化路径 | vLLM runtime quant：`--quantization fp8` | 离线 `llm-compressor` 产出 compressed-tensors checkpoint |
| Activation scale | **未校准**，默认 1.0 → 触发退化路径 | **dynamic per-token**，每个 forward 现算，永远准 |
| Weight scale | 运行时计算 per-tensor | 离线计算 per-channel（更细粒度，更准）|
| KV cache | `--kv-cache-dtype fp8`，3 条 WARNING | **保持 BF16**，主动绕开 |
| score head | 一起量化（reward 精度受损）| **显式 ignore**（关键！）|
| 启动 WARNING | 3 条 "uncalibrated scale" | **0 条**（理想状态）|
| 实测 vs BF16 | -7%（更慢）| 预期 +12-22%（待实测）|

---

## 6. 时间线与里程碑

| Milestone | 工作 | 时长 | 通过标准 |
|---|---|---|---|
| M1 | 写 + 跑 `scripts/quantize_rm_fp8_dynamic.py` | 30min | dst 目录产出，配置正确 |
| M2 | Phase 2 启动 + sanity check | 10min | 服务起来，curl 测试 reward 数值合理 |
| M3 | Phase 3 质量 A/B（150 prompts）| 30min | Pearson ≥ 0.99，Top-1 ≥ 95% |
| M4 | Phase 4 完整 600 题 MMLU | 4-5h | 整体 tok/s ≥ 32（**+8% over G3**） |
| M5 | 落档：在 `doc/vllm-rm-followup-optimizations.md` **§5 全实验回顾**最后追加新行 "G5: + A2 FP8_DYNAMIC"，并在 §3 实验结果表也追加 G5 一行（含 tok/s / acc / 干预率），最后在 §6 结论部分把"FP8 在共享 GPU 上没收益"这条更新成"未校准 FP8 (G4) 没收益，校准过的 FP8_DYNAMIC (G5) 收益 +XX%"。同时给 `exp/README.md` 新增 "2026-XX-XX — G5: FP8_DYNAMIC" 一节，挂上 log 文件链接。 | 30min | 数据 + 结论清楚，PR-ready |

**Go/No-go 检查点**：M3 没过 → 启动回退方案，不跑 M4。

---

## 7. 引用

- 用户的核心疑问与 static vs dynamic 解释：本文 §0
- 瓶颈定位与 A2 路线提出：[`doc/profiling-bottleneck-analysis.md`](profiling-bottleneck-analysis.md) §5.1
- G4 失败的详细原因：[`doc/vllm-rm-followup-optimizations.md`](vllm-rm-followup-optimizations.md) §4.4
- C1+C2 实测净收益接近零的分析：[`doc/profiling-bottleneck-analysis.md`](profiling-bottleneck-analysis.md) §5.3
- `llm-compressor` 官方文档：https://github.com/vllm-project/llm-compressor
- vLLM compressed-tensors 集成：https://docs.vllm.ai/en/latest/quantization/auto_awq.html（FP8 在同样的 compressed-tensors 路径上）

---

## 8. 不在本方案范围内的事

明确**不做**的事，避免混淆：

- ❌ 不量化 LLM (Qwen3-14B)：本方案只动 RM
- ❌ 不动 KV cache 量化：避开 G4 退化路径
- ❌ 不做 static FP8 calibration：用 dynamic 绕开分布依赖
- ❌ 不换更小 RM (Qwen3-1.7B-VM)：动质量，违反约束
- ❌ 不动 entropy_threshold / topk：动质量
- ❌ 不实现 2B 有状态 RM：留给未来 2 卡环境
