# SIA 高并发吞吐分析与 VM 批量打分优化方案

日期：2026-06-18  
模型：0GM-1.0-35B-A3B（35B）、Qwen3-VL-30B-A3B-Instruct（30B）  
数据来源：`exp/bench-0gm35b-bench-v2-20260617.md`、`exp/bench-vl30b-bench-v1-20260617.md`

---

## 一、吞吐量对比：SIA vs noSIA

### 35B（topk=10, max_model_len=32768）

Concurrency Sweep（input≈480 tokens, max_out=304）：

| Conc | SIA ITL | noSIA ITL | ITL 倍数 | SIA tok/s | noSIA tok/s | SIA/noSIA |
|------|---------|-----------|----------|-----------|-------------|-----------|
| 1    | 10.9ms  | 9.0ms     | 1.2×     | 88.5      | 105.4       | 84%       |
| 2    | 19.2ms  | 9.6ms     | 2.0×     | 100.7     | 191.5       | 53%       |
| 4    | 24.8ms  | 10.5ms    | 2.4×     | 156.6     | 352.7       | 44%       |
| 8    | 35.8ms  | 11.8ms    | 3.0×     | 217.3     | 617.4       | **35%**   |
| 16   | 65.4ms  | 13.7ms    | 4.8×     | 239.4     | 1040.3      | **23%**   |

**结论**：并发 ≥ 8 时 SIA tok/s 降至 noSIA 的 23–35%（下降 65–77%）；conc=16 时 ITL 是 noSIA 的 4.8×。

### 30B（topk=10, max_model_len=4096）

Concurrency Sweep（input≈512 tokens, max_out=128）：

| Conc | SIA ITL | noSIA ITL | ITL 倍数 | SIA tok/s | noSIA tok/s | SIA/noSIA |
|------|---------|-----------|----------|-----------|-------------|-----------|
| 1    | 8.0ms   | 6.6ms     | 1.2×     | 101.6     | 126.9       | 80%       |
| 4    | 8.2ms   | 8.1ms     | 1.0×     | 311.0     | 380.3       | 82%       |
| 8    | 11.5ms  | 8.4ms     | 1.4×     | 539.9     | 605.6       | 89%       |
| 16   | 17.0ms  | 9.3ms     | 1.8×     | 704.3     | 1067.5      | **66%**   |

30B 高并发开销明显低于 35B（conc=16 时 1.83× vs 4.8×），原因在后文分析。

---

## 二、SIA 高并发 ITL 急剧上升的原因

### 2.1 一个 decode step 的完整计算流程

设并发请求数 = N，VM topk = K（生产配置 N=16, K=10）：

```
═══════════════════════════════════════════════════════════════════
  每个 decode step（vLLM 触发一次 SIALogitsProcessor.apply()）
═══════════════════════════════════════════════════════════════════

【步骤 1：主 LLM forward — 一次，N 个请求并行】

  GPU: LLM_attention(batch=[req_0 ... req_{N-1}])
     → logits[N, vocab_size]           # N 个请求同时出结果，~14ms（N=16）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【步骤 2：SIALogitsProcessor.apply(logits)】

  # 2-A: 批量提取 top-K 候选 + 计算熵（全 N 个请求同时，1 次 GPU→CPU sync）
  topk_indices[N, K] = torch.topk(logits, K)          # GPU
  entropy[N] = -Σ softmax(topk) * log_softmax(topk)   # GPU
  entropy_cpu = entropy.cpu().tolist()                 # GPU→CPU sync，仅 1 次

  # 2-B: 顺序遍历每个请求，调用 VM 打分  ⚠️ 串行瓶颈所在
  for i in range(N):
      if entropy_cpu[i] < threshold:
          continue                 # SKIP：entropy 低，模型已确信，不干预

      candidates_i = topk_indices[i].cpu().tolist()   # [c_0 ... c_{K-1}]

      # ↓ 阻塞：必须等第 i 个 VM 打分完成才处理第 i+1 个 ↓
      rm_scores_i = VM.score_candidates(session_i, candidates_i)  # ~3ms

      logits[i, candidates_i] += rm_scores_i * weight

  return logits   # 交还 vLLM 采样

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【步骤 3：vLLM 采样：基于修改后的 logits 采样 N 个 token】
```

### 2.2 VM.score_candidates 内部流程

```
═══════════════════════════════════════════════════════════════
  VM.score_candidates(session_i, [c_0 ... c_{K-1}])
  每次 apply 串行调用 N 次，每次处理 1 个请求的 K 个候选
═══════════════════════════════════════════════════════════════

  # prefix_i = chat_template(user_msg) + 已生成的 output_ids
  # 每个请求独立维护，不同请求的 prefix 不能共享

  # fix_a_token：纯 Python append，不碰 GPU
  for tid in new_output_ids_since_last_call:
      self._sessions[sid].append(tid)       # O(1)，无 GPU

  # 构造 K 条完整序列，每条 = prefix_i + 一个候选 token
  prompts = [prefix_i + [c_0],
             prefix_i + [c_1],
             ...
             prefix_i + [c_{K-1}]]          # K 条序列

  # VM 内部 vLLM：一次 forward，K 条序列作为 mini-batch
  # APC 命中：prefix_i 已有 KV，只需 forward 最后 1 个 candidate token
  GPU: VM_attention(batch=K_prompts)        # 实际只计算 K 个末尾 token
     → hidden[K, d_model]

  scores[K] = linear(hidden[K])            # score head：K 个标量
  return scores[K]
```

### 2.3 为什么高并发时 VM 是瓶颈

```
noSIA（N=16）：
  ┌───────────── LLM forward 14ms ─────────────┐
  └─────────────────────────────────────────────┘
  → 1 步完成，出 16 个 token，ITL ≈ 14ms

SIA（N=16, K=10）：
  ┌── LLM forward 14ms ──┐
                          ├─ VM[req_0]  3ms ─┤
                                              ├─ VM[req_1]  3ms ─┤
                                                                  ├─ ...
                                                                      ├─ VM[req_15] 3ms ─┤
  总耗时 ≈ 14ms + 16×3ms = 62ms（实测 65.4ms），ITL 是 noSIA 的 4.8×
```

**本质原因**：
- LLM 的 batch decode 计算量随并发**线性增长但被 batching 均摊**，conc=1→16 时 ITL 仅从 9ms → 14ms（+56%）
- VM 的 N 次调用是**串行的**（`for i in range(N)` 顺序循环），overhead 随并发**线性累加**：conc=1→16 时 VM 部分从 3ms → 48ms（+1500%）

**30B 为何好于 35B**：30B 的 VM（同为 VM-Qwen3-4B）打分更快（APC block_size=16，命中率高），每次 VM 调用约 1.5ms，16 次串行仅 ~24ms vs 35B 的 48ms，因此 conc=16 时 ITL 倍数为 1.8× 而非 4.8×。

---

## 三、可行的并行优化方案

### 3.1 排除的方案

| 方案 | 结论 | 原因 |
|------|------|------|
| 多线程（N 线程各跑一次 `score_candidates`） | ❌ 不可行 | VM 使用 vLLM InprocClient（`multiprocessing=False`，`client.py:90`），EngineCore 在当前进程内无线程锁，多线程并发会打乱 reward 缓冲区顺序或 crash |
| asyncio 并发 | ❌ 不可行 | `llm.generate()` 是同步阻塞调用，vLLM 的异步接口（`AsyncLLMEngine`）与当前 `LLM` 实例不兼容 |

### 3.2 推荐方案：N×K prompt 合并为一次 `llm.generate()`

**核心思路**：把串行的 N 次 `score_candidates(K prompts)` 合并成一次 `score_candidates_batch(N×K prompts)`，只调用一次 `vm_llm.generate()`。

**可行性验证逐点**：

**① `fix_a_token` 不阻塞**  
`client.py:288`：`fix_a_token` 是纯 Python append，不调用 GPU。可以先把所有 N 个 session 的前缀全部推进，再统一做一次 GPU forward，无顺序约束。

**② inproc reward buffer 支持批量取回**  
现有模式：`clear → generate(K) → take(K)`。  
批量模式：`clear → generate(N×K) → take(N×K)`。  
`take_inproc_rewards` 返回 flat tensor，按提交顺序排列（`client.py:507` 已通过 split batch 验证顺序正确性）。取回后按 offset 切片还原每个请求的 K 个分数。

**③ APC 收益完全保留**  
vLLM APC 是基于内容 hash 的前缀缓存，与提交批次大小无关。N 个请求的 prefix 各不相同，互不干扰；每组 K 个候选仍共享各自的 prefix_i，APC 仍命中 K-1 次。

**④ cross-tokenizer bridge（35B）兼容**  
35B 的 LLM 和 VM tokenizer 不同（`client.py:327`）。每个 session 的前缀需单独 decode→encode。这是 CPU 操作，N 个 session 逐一处理后，仍只需一次 GPU forward，不影响批量化。

**⑤ 可变 topk 兼容**  
不同请求可能有不同的 K_i（`apply()` 第 956-961 行）。批量时记录每个请求的 offset，取回时用 `scores[offset_i : offset_i + K_i]` 切片，完全兼容。

### 3.3 改动范围

```
client.py：新增方法
    score_candidates_batch(
        requests: list[tuple[int, list[int]]]   # (sid, candidate_ids) × N
    ) → list[Tensor]
    # 一次 clear-generate-take 循环，返回 N 个 Tensor，每个 shape=(K_i,)

sia_vllm_RM.py：修改 apply() 中的循环，拆成三阶段

  # Phase 1：纯 Python，无 GPU（顺序处理所有请求，无延迟）
  intervene_items = []
  for i in range(N):
      if entropy[i] < threshold or weight_i == 0:
          continue
      advance session_i prefix（fix_a_token 链）
      intervene_items.append((sid_i, candidates_i))

  # Phase 2：1 次 GPU forward，替代原来的 N 次
  if intervene_items:
      all_scores = rm.score_candidates_batch(intervene_items)

  # Phase 3：纯 Python，应用分数
  for idx, (i, scores_i) in enumerate(zip(intervene_indices, all_scores)):
      logits[i, candidates_i] += scores_i * weight_i
```

### 3.4 预期性能收益

| 指标 | 当前（N=16 串行） | 改后（N=16 批量） | 提升 |
|------|-----------------|-----------------|------|
| `vm_llm.generate()` 调用次数/step | 16 次 | 1 次 | 16× 减少 |
| VM 耗时/step（35B 估算） | 16 × 3ms ≈ 48ms | ~8–15ms | ~3–6× 加速 |
| 总 ITL（35B, conc=16 估算） | ~65ms | ~22–30ms | ~2–3× 改善 |
| tok/s vs noSIA（35B, conc=16 估算） | 23% | 预期 50–70% | — |

改善的核心：vLLM 的调度开销（dispatch + EngineCore 通信 + GPU sync）从每步 16 次减少为 1 次；GPU 利用率因更大 batch size 提升。

### 3.5 注意事项

- VM 的 KV cache 需要容纳 N 个 session 的前缀同时活跃（N=16 时约多占 ~800 MB），需确认 `rm_b2_gpu_mem` 分配充足
- 若 N 个请求的 entropy 均低于 threshold（全部 SKIP），batch 为空，直接跳过，无 overhead
- 现有 `_score_candidates_b2()` 可保留作为 N=1 的兼容降级路径，或直接统一走 batch（N=1 的 batch 行为等价）
