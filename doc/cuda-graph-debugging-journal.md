# CUDA Graph 调试经验总结（RM Server）

本文档总结在为 `src/sia_rm_server.py` 引入 CUDA graph + 静态 bucketing 加速过程中遇到的所有坑、走过的弯路，以及最终的解决方案与未解之谜。所有判断和数据都来自实测，不是理论推断。

## 1. 背景与目标

### 1.1 出发点
SIA 框架在 LLM decode 每一步可能调用一次 Reward Model（Qwen3-4B + LoRA）打分 top-k 个 candidate token，**每次 RM 调用即使在 KV cache 完美复用前缀的情况下也要 60-65ms**，构成 LLM 总解码时延的主要部分。

### 1.2 Profiling 定位的瓶颈
详见 `doc/rm-profiling-guide.md`。用 `[RM-pf-fwd]` 单次 timing 加 sliding window 统计得出：

| fwd 阶段 | p50 |
|----------|-----|
| forward (model.forward, 含全部 36 层) | **56.7ms** |
| kv_expand | 3.2ms |
| kv_save | 3.5ms |
| tokenize | 3.9ms |
| 其他 | <1ms |

forward 占总时间 83%。进一步按 diff_len 分桶：

| diff_len | p50 forward |
|----------|------------|
| 8-13 | **56ms** |
| 14-15 | 62ms |
| 30-50 | 80ms |

**diff_len 在 8-13 范围内时 forward 时间几乎是常数 56ms**（KV cache 已经把绝大部分计算省掉了，理论值应该 < 5ms）。这个固定不变的 56ms 就是 **Qwen3-4B 36 层 transformer 在 PyTorch eager 下的 kernel launch overhead 物理下限**。

### 1.3 CUDA Graph 是消除 launch overhead 的标准方案
- CUDA graph 一次性捕获整张前向计算图（所有 GPU kernel 调用），replay 只触发 1 次启动，把 ~360 次 launch 折叠成 1 次
- 要求**所有输入张量形状固定**（静态 shape）
- 与 `torch.compile(mode="reduce-overhead")` 底层是同一套机制，区别在我们手动管理 capture/replay 而不依赖 dynamo 的 dispatch

## 2. 整体设计

### 2.1 静态 bucket
预分配一个 (`batch=5`, `diff_max=64`, `kv_max=2048`) 的 buffer 组合，捕获一次 CUDA graph。所有满足 `actual ≤ max` 的实际推理都 round up 到这个 shape，replay 同一张图。

### 2.2 稀疏 attention_mask 布局
真实的 `(kv_actual, diff_actual)` 远小于 `(kv_max, diff_max)`。布局方式：

```
KV buffer 总长 kv_max + diff_max = 2112：
  [0           .. kv_actual-1   ] ← 真实 prefix K/V，mask=1
  [kv_actual   .. kv_max-1      ] ← padding K/V（值 = 0），mask=0
  [kv_max      .. kv_max+diff_actual-1] ← 真实 diff K/V（本次 forward 算），mask=1
  [kv_max+diff_actual .. kv_max+diff_max-1] ← padding，mask=0
```

### 2.3 显式 position_ids（关键正确性保证）
默认 `cache_position = [kv_max..kv_max+diff_max-1]`，但 diff token 的真实位置应该是 `kv_actual..kv_actual+diff_actual-1`。
- 不传 position_ids → RoPE 用错位置 → 与 eager 输出不一致
- 必须显式传入 `position_ids = arange(kv_actual, kv_actual + diff_actual)`

### 2.4 新 prefix 提取
HIT 路径 forward 后，新 prefix 数据分散在两段：
- `full_K[0:1, :, 0:kv_actual, :]`：原始 prefix（未变）
- `full_K[0:1, :, kv_max:kv_max+extension_len, :]`：本次新算的 extension tokens

需要 stitch 到一个连续 `(1, h, kv_actual+extension_len, d)` 张量供下次 HIT 复用。

## 3. 调试过程中踩到的所有坑

按出现顺序排列。**每一项后面都有"教训"**。

### 3.1 `torch.arange()` 不指定 device → capture 失败

**症状**：
```
RuntimeError: CUDA error: operation not permitted when stream is capturing
File "src/sia_rm_server.py", line 91, in forward
    logits = token_rewards[torch.arange(token_rewards.size(0)), diff_lens].unsqueeze(-1)
```

**根因**：`torch.arange(5)` 默认在 CPU 上创建小张量，被用作 GPU tensor 的 index 时触发隐式 CPU→GPU 拷贝。CUDA graph capture 阶段禁止所有 CPU↔GPU 同步操作。

**修复**：
```python
_batch_idx = torch.arange(token_rewards.size(0), device=token_rewards.device)
logits = token_rewards[_batch_idx, diff_lens].unsqueeze(-1)
```

**教训**：CUDA graph capture 期间任何潜在的 CPU↔GPU 通讯都会报错。要做的：
- 所有 `torch.arange / torch.tensor / torch.zeros` 显式 `device=...`
- 避免任何 `.item()`、`.cpu()`、`.tolist()`、`.numpy()`
- 避免 Python 条件依赖张量值（如 `if x.sum() > 0`）

### 3.2 切换 `attn_implementation="eager"` → 另一种 capture 失败

**思路**：以为 SDPA 在 capture 时和 replay 时分发到不同 backend 导致 NaN，干脆全程用 eager attention 绕开 SDPA。

**结果**：
```
File "transformers/masking_utils.py", line 521, in eager_mask
    mask = torch.where(mask, torch.tensor(0.0, ...), min_dtype)
                             ^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: operation not permitted when stream is capturing
```

HF transformers 的 eager attention mask 构造代码用了 `torch.tensor(0.0, device=mask.device)` —— 创建小 scalar 张量，也是 CPU 端构造然后传 GPU，同样触发同步。

**教训**：换 attention 实现是绕远路。HF transformers 的代码并不一定都 CUDA-graph-safe，eager 路径甚至更不安全（用了更多 Python-level 张量构造）。

### 3.3 mark_dynamic 完全不解决 CUDA graph 的 NaN

试了两种用法：

**用法 A**：在 warmup tensor 上标记 dynamic 维度
- 没效果：`mark_dynamic` 标记的是特定 Python tensor 对象，不会传递给推理时新建的 tensor

**用法 B**：在每次推理 tensor 上标记
- 也没效果：CUDA graph 不像 `torch.compile`，它一旦 capture 就固化了所有形状决策。`mark_dynamic` 是给 dynamo 用的，CUDA graph 看不到这个标记。

**教训**：`mark_dynamic` 属于 `torch._dynamo` API，跟手动 CUDA graph 没关系。混用 `torch.compile` 和手动 CUDA graph 时要分清楚哪个是哪个。

### 3.4 SDPA Flash backend：捕获成功但输出错（NaN/wrong）

**症状**：capture 完美成功，bucket 跑了 ~270 次都得到 `cg_run=30-35ms`（理论速度），但 **88% 的输出含 NaN**。

**根因**：`F.scaled_dot_product_attention` 在 capture 时根据输入决定后端，可能选择 Flash Attention 2。Flash **不支持自定义 attention_mask**（只支持 `is_causal=True` 或全有效）。因此：
- Capture 时 SDPA 看到 attention_mask（即使全是 1）→ 实际可能仍选 Flash
- Flash kernel 内部完全忽略 mask 值
- Replay 时真实 mask 有 0 → Flash 还是忽略 → 结果错

**教训**：
- SDPA 的 dispatch 决策**是动态的**，且与 input data 形状/dtype/mask 是否提供有关
- Flash 与 mem_efficient 的能力差异：Flash 不支持任意 mask，mem_efficient 支持
- 把 attention_mask 当成参数传给 SDPA **不等于** SDPA 一定会处理它

### 3.5 SDPA Math backend：正确但慢

强制 `enable_flash_sdp(False)`、`enable_mem_efficient_sdp(False)`、只留 math：
- 输出正确（math 是最朴素的 `Q@K → mask → softmax → @V`）
- 但 `cg_run = 65-70ms`，比原 Flash eager（56ms）还慢
- 因为 math 比 Flash 慢约 2-3 倍 per op
- 而且**这个设置是全局的**，eager fallback 路径也被拖慢

**教训**：math backend 是"安全但慢"。不能为了 CG 正确性把 eager 路径也变慢。

### 3.6 SDPA Mem Efficient + Capture-then-Restore 妙招

**关键洞察**：CUDA graph capture 阶段的 backend dispatch 决策**会被固化进图里**，replay 时不再 dispatch。

利用这一点：
```python
# 保存原状
orig_flash = torch.backends.cuda.flash_sdp_enabled()
...

# Capture 期间强制 mem_efficient（关掉 Flash 避免被它捕获）
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
bucket.capture()

# Capture 完恢复原状（Flash 重新启用）→ eager 路径继续用 Flash 飞快
torch.backends.cuda.enable_flash_sdp(orig_flash)
...
```

效果：
- Bucket replay 路径（烧在图里）：**mem_efficient（正确处理 mask + 快）→ cg_run = 36ms**
- Eager 路径（动态 dispatch）：**Flash（极快）→ fwd = 56ms 维持 baseline**

**教训**：CUDA graph 的"捕获时机"很关键。**任何在 capture 时设置的 PyTorch 全局/上下文状态都会被固化**，可以利用这点做"双轨配置"。

### 3.7 capture 时用全 1 attention_mask → backend 选错

最初 capture 用 `attention_mask.fill_(1)`（全有效）。SDPA 看到 mask 是"trivially valid"，**可能内部把它当 None 处理 → 选 Flash**。

后来改成 capture 时也用稀疏布局（前 1024 个 1，后面 0，模拟真实推理）：
```python
self.attention_mask.zero_()
cap_kv_actual = self.kv_max // 2
cap_diff_actual = self.diff_max // 2
self.attention_mask[:, :cap_kv_actual] = 1
self.attention_mask[:, self.kv_max:self.kv_max + cap_diff_actual] = 1
```

这样 SDPA 看到的是"真实有 mask 的场景"，正确选 mem_efficient/math。

**教训**：**capture 输入要尽量贴近真实推理输入**。不只是形状要一致，**值的特征**（如 mask 是否有 0）也会影响 SDPA dispatch。

### 3.8 RM server 突然死亡（无 traceback）

**症状**：某次跑评测后期 RM server 进程 silently exit。LLM server 报 Connection Refused。日志文件没有 Python traceback。

**根因 1**：用 `nohup ... > log_xxx.txt 2>&1` 同名日志，重启时新进程覆盖了崩溃前的日志。

**根因 2**：CUDA 错误是异步的，可能在 PyTorch sync 点（如 `.item()`）才报出来，如果没有 catch 直接让 C++ abort 杀进程。

**修复**：
1. 文件名加时间戳（`log_rm_cg_$(date +%Y%m%d%H%M).txt`）
2. **bucket.run() 全包 try/except**：任何异常（含 CUDA 异步错误）都 catch 并写 traceback，返回 None 让上层 fallback
3. **/score endpoint top-level try/except**：万一异常逃出，至少返回降级 scores 而不是让进程死

```python
@app.post("/score")
def score(req: ScoreRequest):
    try:
        return _score_impl(req)
    except Exception as e:
        traceback.print_exc()
        return {"scores": [0.0] * len(req.candidate_texts), "error": str(e)}
```

**教训**：
- nohup 日志一定要带时间戳，别覆盖旧文件
- CUDA-related 代码必须用 try/except 全部包起来，防止异步错误杀进程
- 用 `print(..., flush=True)` 保证 traceback 写盘

### 3.9 NaN 只检查 logits 不够，要检查 K/V

**问题**：bucket forward 后只判 logits 是否 NaN，干净就保存 new_prefix_dc。但有时 K/V 部分位置含 NaN，被 attention_mask 屏蔽掉所以 logits 干净 → NaN K/V 被存进 `_req_kv` → 下一次 HIT 用了带毒的 prefix → bucket 必 NaN → 形成"链式污染"。

**修复**：保存前 stack 所有 36 层的 extension K/V，整体扫一遍 NaN：
```python
ext_K_stack = torch.stack([self.full_K[i][0:1, :, kv_max:kv_max+ext_len, :] for i in range(L)])
if not torch.isfinite(ext_K_stack).all().item():
    return None  # 不保存被污染的 prefix
```

**教训**：NaN 检测要覆盖**所有要保存或下游使用的张量**，不只是当下输出。

### 3.10 全局 disable 太激进 → per-request disable

**问题**：当某个 request 触发 bucket NaN 后，我设的"连续 20 次 NaN 关 bucket"会**全局**关掉，让所有 request 都失去加速。实测 1 个坏 request 5 次 NaN 后再叠加其他 request 几次就关了。

**修复**：分两层：
- **per-request**：同一 request_id 连续 5 次 NaN → 只关它，其他 request 仍享受 bucket
- **global**：跨 request 累计 50 次 NaN → safety net，全关

```python
_cg_request_nan: dict = {}      # request_id → 连续 NaN 次数
_cg_per_req_disabled: set = set()
_cg_consecutive_nan: int = 0    # 全局
```

**教训**：自适应降级粒度要够细。一个坏苹果不应该让一整箱苹果都不能吃。

### 3.11 padded K/V 残留 stale 数据（假设错误）

**症状**：bucket 在第 1 个 request 工作 270 次完美，之后所有 request 立刻 NaN。

**假设**：`self.prefix_K[i]` buffer 只覆盖 `[0:kv_actual]`，`[kv_actual:kv_max]` 保留上次调用的真实 K 值（不再是初始 zero）。虽然 `attention_mask=0` 应屏蔽这些位置，但 mem_efficient 内部对 stale 值可能有数值不稳定。

**修复尝试**：每次写入前 `self.prefix_K[i].zero_()`，整 buffer 清零再写真实 prefix。

**结果**：**完全没改善**。同样 NaN 模式。

**教训**：直觉是不可靠的，要靠数据驱动。这次修复花了时间但没用，但也排除了一个嫌疑根因。

### 3.12 ⚠️ 未解：首个 request 工作，后续 request 全失败

**事实**：
- 启动后第 1 个 request 用 bucket 跑 270 次，全部正确
- 该 request 结束，第 2 个 request 第 1 次 MISS 调用就 NaN
- 之后 10+ 个 request 全部立刻 NaN
- **输入 prefix_dc 经全 36 层检查都干净**（`INPUT prefix_dc has NaN` 计数 = 0）
- Eager fallback 对同样输入产生正确 scores

**这意味着**：bucket 的 GPU 内部状态在跑过几个 request 后**变得有问题**，相同形状、相同干净输入，bucket 输出 NaN，eager 不会。

**已尝试但失败的修复**：
- 清零 padded K/V buffer
- per-request disable（只能限制扩散，不能修根因）
- 验证 input 全 36 层 finite
- Capture-then-restore SDPA backend

**尚未尝试**：
- 缩小 kv_max（2048 → 1024）减少 padding 总量
- "warm-then-capture" 模式：先用真实请求跑一阵积累 GPU state，再 capture
- 用 `torch.cuda.graph_pool_handle()` 显式管理 memory pool
- 切换到 vLLM 部署 RM（vLLM 内部已经做了 CUDA graph + 调度，不用自己 reinvent）

## 4. 最终落地的设计要点

### 4.1 SDPA backend 双轨
- Capture 时：`flash=OFF, mem_efficient=ON, math=ON, cudnn=OFF`
- Capture 后立即恢复用户原配置（通常 `flash=ON`）
- Bucket replay 用 mem_efficient（图里烧死），eager fallback 用 Flash（动态 dispatch）

### 4.2 三层 NaN safety net
1. **Bucket 入口**：检查 input prefix_dc 全 36 层 K/V finite
2. **Bucket 出口**：检查 logits 和 extension K/V finite
3. **任何 NaN → return None → 上层 fallback 到 eager**

### 4.3 双层 disable
- **Per-request**：单个 request 连续 5 次 NaN → 只关它
- **Global**：跨 request 累计 50 次 NaN → 全关（safety net）

### 4.4 防御性 try/except
- `bucket.run()` 全包 → 任何 CUDA/Python 异常都不会让 server 死
- `/score` endpoint top-level wrap → 异常退化为 `scores=[0,0,...]` 但 server 继续

### 4.5 实测性能（n=20 MMLU 题）

| | fwd p50 | 说明 |
|--|---------|------|
| 原 baseline（Flash eager + KV cache）| 56-57ms | 我们要打的目标 |
| Bucket 工作时 | **36ms** | 38% 加速 |
| Per-req disabled 后该 request 走 eager | 56-58ms | 回到 baseline，无损 |
| Adaptive global disable 后全部走 eager | 56-58ms | 回到 baseline，无损 |

**净收益**：约 274 / 总 2237 = 12% 的调用享受 38% 加速 → 整体约 **4-5% 的 eval 时间节省**。

最坏情况：**回到 baseline**，绝不变慢。

## 5. 关键经验教训汇总

### 5.1 CUDA Graph 是脆弱的工程
- 任何 CPU↔GPU 同步、动态 shape、隐式分发都可能让 capture 失败或产生错误输出
- 而且失败模式可能是**捕获成功但运行时输出错**（如 SDPA 选错 backend）
- 一定要有 NaN 检测 + fallback 兜底

### 5.2 Backend 选择是"运行时"的
SDPA 的 backend 不是固定的，根据：
- 输入 dtype（fp32 vs bf16）
- 是否提供 attention_mask
- mask 的形状/类型
- 输入张量是否连续
- 全局 `torch.backends.cuda.*_sdp_enabled` 标志

任何一个变化都可能让 dispatch 选不同的 kernel。**CUDA graph 把这个决策固化下来时一定要在最具代表性的输入上 capture**。

### 5.3 防御编程比根因分析便宜
当遇到不明原因的 NaN 时，与其无止境调试根因，不如：
- 加 NaN 检测
- 加 fallback 路径
- 加 adaptive disable
- 加 try/except 全包

代价：每次调用多 1-2ms 检测开销
收益：系统永远不会比 baseline 慢，最坏情况退化到 baseline

### 5.4 nohup 日志名字带时间戳
CUDA-related crash 是不可避免的，至少要保住 traceback。

### 5.5 profiling 必须 fine-grained
单一 "forward = 56ms" 没有信息量。要分解到：
- 每个阶段（tokenize / prep / expand / forward / save）
- 按 input 形状分组（diff_len 段 / kv_len 段）
- 时间序列趋势

这样才能区分"是 forward 慢"还是"是 launch overhead 慢"还是"是 tokenize 慢"。

## 6. 未解之谜与后续探索方向

### 6.1 为什么第一个 request 工作而后续失败？
两种假设方向：

**A. CUDA 内存池假说**：bucket capture 时分配的 GPU 内存可能与后续 PyTorch 缓存分配器的内存有微妙互动。某些后续操作（eager fallback、tokenizer GPU 临时张量、vLLM KV cache 增长）可能踩到 graph memory pool 的边界。

**B. cuDNN 状态污染假说**：cuDNN（即使我们 disabled cuDNN_sdp）的某些内部 workspace 缓冲可能在多次推理后积累状态，导致 graph replay 读到非预期数据。

可探索：
- 用 `torch.cuda.memory.set_per_process_memory_fraction` 隔离 bucket 内存
- 显式 `torch.cuda.graph_pool_handle()` 给 bucket 私有 pool
- `CUDA_LAUNCH_BLOCKING=1` 强制同步，看是否更早暴露错误

### 6.2 vLLM-style 部署 RM
vLLM 内部已经实现了：
- KV cache 管理（含跨请求复用）
- CUDA graph 加速（per-shape bucketing）
- 连续批处理（不同请求合并 forward）

把 RM 当成"只 prefill 不 decode"的 vLLM 实例可能比手动 CUDA graph 收益大。但工程量也大，需要把 RM 的 reward head（`token_reward_head` linear layer）接到 vLLM 输出。

### 6.3 第二块 GPU
如果硬件允许，把 RM 放到 cuda:1：
- 完全消除与 LLM 的 GPU 资源竞争
- 单 RM kernel launch overhead 不变，但 LLM 不再等 RM 完成（异步流水）
- 收益可能比代码优化更大

### 6.4 减少 bucket padding
当前 `kv_max=2048` 是为了"覆盖最长情况"。但 90%+ 的调用 `kv_actual < 1000`。多个不同 size 的 bucket（如 512、1024、2048）配合分发，可以减少 padding 浪费。代价是 capture 时间和 GPU 内存 × bucket 数。

## 7. 参考代码

- `src/sia_rm_server.py`
  - `_CudaGraphHitBucket`: bucket 类
  - `_init_cuda_graph_buckets()`: 初始化 + 双轨 backend 切换
  - `_score_with_prefix_kv()`: 集成点，bucket 优先 + eager fallback
- `doc/rm-profiling-guide.md`：profiling 启动命令与日志格式
- 启动命令示例：
  ```bash
  python src/sia_rm_server.py --rm ... --rm_lora ... \
    --cuda_graph --cg_batch 5 --cg_diff_max 64 --cg_kv_max 2048
  ```
- 关闭方式：去掉 `--cuda_graph` 参数即可，完全回到原 KV cache 路径。

## 8. Commit 记录

- `ddba0b0` — add CUDA graph bucket for HIT path + detailed profiling
- `4021737` — ignore experiment log files and Python cache

之前的相关基础工作：
- `1c4cdaa` — fix DynamicCache API for transformers 4.57+
- `aabd5df` — optimize MISS path: batch=1 prefix + batch=k diff
- `16d4fa5` — add torch.compile support and fix output_hidden_states waste
- `3c8bc20` — implement single-pass KV prefix cache for Value Model scoring

---

## 9. 实验日志归档

CUDA graph 路线的代表性端到端实验（PyTorch RM + `--cuda_graph` flag）日志已归档至 `exp/`，完整命令与说明见 [`exp/README.md`](../exp/README.md) 中「2025-05-21 00:50 — CUDA graph 加速实验」一节。
