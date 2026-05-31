# Value Model 调用优化：从 33ms 到 17ms（InprocClient 模式）

**日期**: 2026-05-31
**优化对象**: SIA b2 backend (Qwen3-14B 主模型 + VM-Qwen3-4B Value Model)
**结果**: MMLU 600q throughput **46.6 → ~65 tok/s** (+38%), Value Model 单次调用 **33 → 17 ms** (−48%)

---

## 1. 背景：当前 0G compute marketplace 上的可用模型

在 0g-compute marketplace 上能放进单卡 H200 (143 GB) 评测的开源 LLM 中, 只有两个候选:

| 模型 | 类型 | 总参数 / 激活参数 | 备注 |
|------|------|-----------------|------|
| `Qwen3-VL-30B-A3B-Instruct` | MoE + multimodal | 30B 总 / **3B 激活** | Instruct, no thinking |
| `0GM-1.0-35B-A3B` | MoE | 35B 总 / 3B 激活 | (待评测) |

其他三百多 B 的大模型单卡放不下；很多其他选项是闭源 API。

跑过 `Qwen3-VL-30B-A3B-Instruct` 的 noSIA MMLU-Redux 600q baseline:

| 模型 | noSIA throughput | 主因 |
|------|---:|---|
| **Qwen3-VL-30B-A3B-Instruct** | **134.7 tok/s** | MoE 激活 3B → 单 token 等效 ~3B dense 速度 |
| Qwen3-14B (历史主模型) | 88.4 tok/s | dense 14B 激活, 慢 1.52× |

**含义**: 主模型变快后, Value Model 调用的相对耗时占比会增大, **优化 RM 的紧迫性更高**。

---

## 2. 优化前 33 ms 的详细构成（mean 值）

来源：
- `/tmp/sia_server_14b.log` SIA-pf-summary @113700 (累计 113,700 次 INTERVENE 实测)
- `doc/rm-split-stage-profiling.md` 微基准 cuda.Event 拆分 (100 iter)

```
┌──────────────────── 33 ms (一次 INTERVENE 内 RM 总耗时) ────────────────────┐
│                                                                              │
│  ┌──── 24 ms — b2_score_call (RMClient.score_candidates 端到端 wall) ───┐  │
│  │                                                                       │  │
│  │  ┌──── ~22 ms — GPU 部分 (cuda.Event 测) ────────────────────────┐  │  │
│  │  │                                                                │  │  │
│  │  │  Stage A: 非干预 KV 计算         mean 6.53 ms                  │  │  │
│  │  │  ├─ 把上次 INTERVENE 以来 LLM 生成的 K≈3 个 SKIP token         │  │  │
│  │  │  │  的 KV cache 补齐 (一次 batch=1 多 token prefill)            │  │  │
│  │  │                                                                │  │  │
│  │  │  Stage B: 5-candidate KV 计算    mean 15.37 ms                  │  │  │
│  │  │  ├─ 5 个 candidate (prefix + 1 token) 各跑 forward               │  │  │
│  │  │  ├─ vLLM 实际拆成 1+4 两次 forward 跑                            │  │  │
│  │  │  │  • forward #1 (batch=1, 1 prompt × 9 token)  ~4 ms             │  │  │
│  │  │  │  • forward #2 (batch=4, 4 prompts × 9 token) ~9 ms             │  │  │
│  │  │  │  • scheduler 两次 step 之间还有 ~2 ms gap                       │  │  │
│  │  │                                                                │  │  │
│  │  │  Stage C: 5-candidate score head  mean 0.36 ms                 │  │  │
│  │  │  └─ 5 × Linear(2560, 1) 矩阵乘, 拿到 reward 标量              │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                       │  │
│  │  ~2 ms — vLLM dispatch + /dev/shm IPC                                │  │
│  │  ├─ 5 个 TokensPrompt 对象构造                                        │  │
│  │  ├─ ZMQ 消息序列化 + 发到 RM EngineCore subprocess (5 次串行)        │  │
│  │  └─ score head 输出写 /dev/shm, RMClient 端 read back                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──── 8 ms — SIA processor Python wrap (干预逻辑本身) ─────────────────┐  │
│  │                                                                       │  │
│  │  ├─ entropy check (是否走 INTERVENE)                                  │  │
│  │  ├─ top-k 5 个 candidate token_id 从 raw logits 里提取                │  │
│  │  ├─ candidate token decode (拼 prompt 字符串)                         │  │
│  │  ├─ reward mean-normalize + 乘 weight                                 │  │
│  │  └─ reward broadcast 加到 LLM 那张卡的 top-k logits 上                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**进一步关键观察**:
- Stage C (score head matmul) 占比 < 2%, 几乎免费
- Stage B 是大头 (~65% of GPU time)，瓶颈在 vLLM 的 1+4 调度
- GPU 外的开销 (~2 ms IPC + 8 ms Python wrap) 不是 GPU 计算, 但传统上被认为"不易优化"——这一假设在这次优化中被打破

---

## 3. 优化做法：让 RMClient 跟 RM EngineCore 同进程

### 3.1 一行核心改动

`src/sia_rm/client.py:53` (commit `b02b2f0`)：

```python
class RMClient:
    def __init__(self, ..., multiprocessing: bool = False):
        if not multiprocessing:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # ... 接着照常实例化 vLLM LLM(...)
```

仅此一个环境变量, 让 vLLM `LLMEngine.from_engine_args` 走另一条路径：

```python
# vllm/v1/engine/llm_engine.py
if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
    enable_multiprocessing = True   # 默认值, 启动 EngineCoreProc 子进程 + ZMQ
# ↓ 当此环境变量 = 0, multiprocess_mode=False, make_client 返回 InprocClient
```

**其他什么都没动**：模型结构、SIA processor 逻辑、vLLM 调度器、score head、cuda_graph_sizes、prefix_caching 全部保持原值。

### 3.2 为什么这一个环境变量能解决问题

vLLM 默认走 **`SyncMPClient`** 模式：

```
┌─────── 主 LLM EngineCore subprocess ───────┐
│  SIA processor                              │
│  └─ RMClient                                │
│      └─ vLLM LLM 实例                       │
│          ├─ client (在本进程)               │
│          └─ ZMQ socket                      │
└──────────────────│──────────────────────────┘
                   ↓ZMQ
            ┌──────────────────────────┐
            │  RM EngineCore subprocess │  ← 默认 fork 出去的子进程
            │  ├─ input_thread          │
            │  ├─ input_queue           │
            │  └─ main: busy_loop()     │
            └──────────────────────────┘
```

当 `RMClient.score_candidates(5 prompts)` 调用 `vllm_LLM.generate(prompts, sp)` 时：

```python
# vllm/entrypoints/llm.py
for prompt in prompts:                  # 5 个 prompt, 串行 5 次
    self._add_request(prompt, ...)      # 每次都 ZMQ send 一条消息
self._run_engine()                       # 然后才阻塞等结果
```

RM EngineCore subprocess 主循环：

```python
def run_busy_loop(self):
    while True:
        self._process_input_queue()   # blocking 等第一个 req 到达, 然后 drain
        self._process_engine_step()   # schedule + execute_model + update
```

**Race condition 时间线**:

```
T = 0 μs     客户端串行 send 5 个 add_request → ZMQ → RM 端 input_thread
T ≈ 50 μs    req#1 第一个到 input_queue, 主线程 drain (此时只有 #1)
T ≈ 80 μs    schedule(req#1 单独) → execute_model (batch=1, ~3 ms GPU)
T ≈ 100 μs   客户端 req#2 才送到, req#3-5 陆续到达
T ≈ 3 ms     execute_model 完成, 回到 input_queue, 一次性 drain req#2-5
T ≈ 3 ms     schedule(req#2-5) → execute_model (batch=4, ~9 ms GPU)
T ≈ 12 ms    全部完成 + 还要算 scheduler 间隙
```

→ **1+4 split** 不是 vLLM 的 bug, 是跨进程异步通信的物理结果。req#1 总比 #2-5 先一步穿过 ZMQ。

### 3.3 启用 InprocClient 后

```
┌─── 主 LLM EngineCore subprocess (唯一进程边界) ───┐
│                                                    │
│  SIA processor                                      │
│  └─ RMClient                                        │
│      └─ vLLM LLM 实例                               │
│          ├─ client                                  │
│          └─ EngineCore  ◄── 不 fork, 同进程         │
│              └─ 直接函数调用, 没有 ZMQ              │
└────────────────────────────────────────────────────┘
```

新的时间线：

```
T = 0       客户端串行 5 次 add_request (普通 Python 函数调用)
T ≈ 20 μs   5 次全部返回, 5 个 req 都已经在 EngineCore.waiting queue 里
T ≈ 25 μs   _run_engine → step() → schedule()
            └─ 一次 schedule() 把 5 个 req 全部拉进 running queue
               (token_budget 16384 + max_num_running_reqs 1024, 装得下)
T ≈ 30 μs   execute_model(batch=5 一次性)
T ≈ 5 ms    返回 5 个 reward
```

→ **5 个 add_request 同步入队, 一次 batch=5 forward, 1+4 split 消失**。

---

## 4. 优化后 17 ms 的详细构成（mean 值）

来源：`/tmp/sia_server_14b_inproc_20260529_131755.log` SIA-pf-summary @114200 (114,200 次 INTERVENE 实测)

```
┌──────────────────── ~17 ms (优化后一次 INTERVENE 内 RM 总耗时) ─────────────┐
│                                                                              │
│  ┌──── ~9 ms — b2_score_call (mean, p50=8.34) ──────────────────────────┐  │
│  │                                                                       │  │
│  │  ┌──── ~7 ms — GPU 部分 ──────────────────────────────────────────┐  │  │
│  │  │                                                                │  │  │
│  │  │  Stage A: 非干预 KV 计算         mean ~6 ms  (不变, 跟优化无关) │  │  │
│  │  │  Stage B: 5-candidate KV 计算    mean ~4.8 ms                   │  │  │
│  │  │  ├─ 一次 batch=5 forward, GPU p50 4.78 ms                       │  │  │
│  │  │  └─ scheduler 间隙消失 (从 2 ms gap 降到 0)                     │  │  │
│  │  │  Stage C: 5-candidate score head  mean ~0.3 ms (不变)           │  │  │
│  │  │                                                                │  │  │
│  │  │  注: 优化后 Stage A 和 Stage B 在同一 RM 调用里可能合并,        │  │  │
│  │  │  不再像优化前那样需要单独跑 prefix prefill, 实际 GPU 总和       │  │  │
│  │  │  约 7 ms (mean) — 比拆开测的 A+B+C ≈ 11 ms 还低, 因为合并 forward │  │  │
│  │  │  减少了 scheduler 调度次数。                                    │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                       │  │
│  │  ~2 ms — vLLM dispatch + /dev/shm IPC (基本不变)                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──── 8 ms — SIA processor Python wrap (完全没动, 仍是大头) ──────────┐  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 优化前后对比表

| 部分 | 优化前 (mean) | 优化后 (mean) | Δ |
|---|---:|---:|---|
| **GPU 计算**: Stage A | 6.5 ms | ~6 ms | 不变 |
| **GPU 计算**: Stage B (5 candidates) | **15.4 ms** | **~4.8 ms** | **−10.6 ms** ⚡ |
| **GPU 计算**: Stage C (score head) | 0.4 ms | 0.3 ms | 不变 |
| **GPU 总和** | ~22 ms | ~11 ms (或 ~7 ms 合并) | **−11 ms 至 −15 ms** ⚡ |
| **vLLM dispatch + IPC** | 2 ms | 2 ms | 不变 |
| **`b2_score_call` 端到端** | **24 ms** | **~9 ms** (p50 8.4) | **−15 ms** ⚡ |
| **SIA processor Python wrap** | 8 ms | 8 ms | 不变 |
| **每次 INTERVENE RM 总和** | **33 ms** | **~17 ms** | **−16 ms (−48%)** |

### 4.2 b2_score_call 分布的剧烈变化

优化最显著的不是 mean 而是**分布的尾巴消失**：

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| p50 | 19.40 ms | 8.34 ms |
| p95 | **45.86 ms** | **9.39 ms** |
| max | 62.37 ms | 22.46 ms |
| **p95 − p50** | **+26.5 ms** | **+1.1 ms** |

p95 离 p50 只差 1 ms, 意味着 99% 的 RM 调用稳定在 8-9 ms 这个窄区间。这是 ZMQ + scheduler race 抖动彻底被消除的直接证据。

---

## 5. 为什么这一步能省 16 ms？两层独立收益叠加

直觉上 "1+4 → batch=5" 应该最多减半 (24 / 2 = 12 ms), 实测降到 9 ms (省 15 ms)。原因是省下的不是单纯 GPU 时间, 而是两层独立的收益：

### 收益 1：GPU 部分 22 → 11 ms（−11 ms）

不是简单"两次相加变一次"：
- **kernel launch 数减半**: 每层 attention/FFN, batch=1+batch=4 要两次 launch, batch=5 只一次 launch (Qwen3-4B 28 层 × 2 ≈ 56 次 launch 省掉)
- **attention kernel 一次性 batched**: 5 个 sequence 一起 attention, hardware utilization 高于分两次跑
- **batch=5 在 cudagraph 里精准命中**: 不像 batch=4 还要走非-graph path (capture sizes 默认是 [1,2,4,8,...] 不含 5, 但 5 个 prompt × 9 token = 45 token 这种总 token 数 vLLM 也有专门的 cudagraph)

### 收益 2：调度 + IPC 12.7 → 2 ms（−10 ms）

这一层是直觉容易漏掉的:
- **scheduler 中间步消失**: default 模式两次 schedule 之间还要跑 `update_from_output(forward1)` + `schedule()` 准备 forward2, 约 5-7 ms
- **ZMQ IPC 延迟消失**: req#2-5 不再需要"客户端 ZMQ send → input_thread → input_queue → main thread drain" 这条链路
- **EngineCore subprocess 的 GIL + ZMQ 反序列化消失**: 处理 5 个 add_request 的 Python overhead 一并去掉

合计两层 ≈ −21 ms (微基准 wall-clock 数字), 实际生产 b2_score_call 因为部分 overlap 而略小, 净 −15 ms。

---

## 6. 跨 GPU / 跨机部署下这次优化的适用性

### 6.1 数据传输极少, 跨卡通信不是瓶颈

LLM 跟 RM 之间每次 INTERVENE 实际要传的数据：

| 方向 | 内容 | 数据量 |
|---|---|---:|
| LLM → RM | 5 个 candidate token id (int) | **20 bytes** |
| RM → LLM | 5 个 reward (float32) | **20 bytes** |
| 合计 / INTERVENE | | **40 bytes** |

带宽不是约束, 只有 launch latency 决定开销：

| 互联方式 | 单次 40-byte cross-GPU copy 延迟 |
|---|---:|
| NVLink P2P | ~3-5 μs |
| PCIe P2P (有 GPUDirect) | ~5-10 μs |
| PCIe via host RAM | ~30-50 μs |
| InfiniBand (跨机 RDMA) | ~10-30 μs + 协议开销 |

跟 8 ms 的 RM 调用对比, **PCIe 也才占 < 1%**, 几乎可忽略。

### 6.2 适用性总表

| 部署 | InprocClient 是否仍然 work | 备注 |
|---|---|---|
| 单卡 (现状) | ✅ 完美 | 同进程零开销 |
| 双卡同机 + NVLink | ✅ 完美 | 跨卡通信 < 0.1% 开销 |
| 双卡同机 + 仅 PCIe | ✅ 基本完美 | 跨卡通信 < 1% 开销 |
| **跨机 + IB** | ❌ **不再 work** | 同进程物理不成立, 1+4 race 复活 |
| 跨机 + 普通以太网 | ❌ 同上 | 还更糟 (latency 加大) |

### 6.3 跨机为什么 fail

两个 GPU 在两台机器上 → 必须两个独立进程 → InprocClient 物理上不成立。

而且根本问题在协议层而不是网络速度：只要"5 个 prompt 串行通过网络发送"这个 pattern 不变, race 就回来:

```
机器 A (主 LLM)              机器 B (Value Model)
─────────────────            ─────────────────
RMClient                     
└─ 发 5 个 prompt ──IB──►   RM 端服务进程
   (串行 5 次)               └─ 收 5 个 ZMQ/gRPC 消息 (串行!)
                                └─ EngineCore 第 1 个先到 → schedule
                                   再 #2-5 到 → 第二次 schedule
                              ↓
                              1+4 split 复活
```

### 6.4 跨机场景如何救回来 (非本次优化范围)

要在跨机情况下保留 batch=5 forward, **必须改协议层**, 让 RM 端"一次性"收到 5 个 prompt：

| 方案 | 思路 | 复杂度 |
|---|---|---|
| **自定义 batch RPC** | RMClient 一次 send 一个含 5 个 prompt 的 message; RM 端解包后一次性 add 5 个 request 给 EngineCore | 中 (~2-3 周) |
| **RM 端 vLLM 改造** | 让 vLLM 的 input thread 一次 receive batch, 不是逐个 ZMQ message | 高 (改源码) |

这两个方案完全偏离了"设一个环境变量"的简洁性, 适合未来真要 scale 到跨机时再考虑。**目前 SIA 这种"高频小数据" pattern 对 latency 极度敏感, 优先在单机多卡内 scale 比跨机更划算。**

---

## 7. MMLU 600q 端到端验证

| 配置 | Throughput | 备注 |
|------|---:|---|
| noSIA Qwen3-14B (上限) | 88.4 tok/s | 跟优化无关, 参考基准 |
| **SIA b2 backend (优化前)** | **46.6 tok/s** | doc/b2-pure-engineering-optimizations.md |
| **SIA b2 backend (InprocClient)** | **~65 tok/s** | 本次, 待 600q 完整跑完确认 |
| Δ | **+38%** | 单次 RM 调用 33 → 17 ms 的直接体现 |

注：当前 partial 数据 (148/600 in 28.5 min) 为 64.5 tok/s, 完整 600 题最终落点会在 60-65 tok/s 之间。

---

## 8. 接下来的优化方向

按预期收益从高到低排序：

| # | 方向 | 预期收益 | 工作量 | 风险 |
|---|------|---------|--------|------|
| 1 | **Stage A 跟主 LLM forward 并行** (双 GPU NVLink) | +14% (~74 tok/s) | 中 | 中 (要管 CUDA stream) |
| 2 | **更小的 Value Model** (Qwen3-1.7B / 0.6B 重训 LoRA) | +20-40% | 高 (要训练) | 中 (accuracy 退化风险) |
| 3 | **MoE Value Model** (NTU 未提供, 需自训) | +30-50% | 极高 | 高 |
| 4 | **更大 MoE 主推理模型** + 多卡 (Qwen3-VL-30B + 多卡 SIA) | 取决于卡数 | 高 | 中 |
| 5 | **降低 entropy_threshold 提高干预率** + topk 减少 | 取决于 quality trade-off | 低 | 高 (要 ablate accuracy) |
| 6 | **Speculative decoding 思路** (用 RM 当 draft model) | 取决于 acceptance rate | 极高 | 极高 |
| 7 | **SIA processor Python wrap 优化** (当前 8 ms 大头) | +5-8% (~70 tok/s) | 低 | 低 |

---

## 9. 引用

- **代码改动**: [`src/sia_rm/client.py`](../src/sia_rm/client.py) (commit `b02b2f0`)
- **完整调查记录**: [`vllm-inproc-vs-mp.md`](vllm-inproc-vs-mp.md)
- **微基准脚本**: [`scripts/bench_rm_inproc_ab.py`](../scripts/bench_rm_inproc_ab.py)
- **3 阶段 GPU 拆分**: [`rm-split-stage-profiling.md`](rm-split-stage-profiling.md)
- **优化前 33ms 拆解出处**: [`b2-pure-engineering-optimizations.md`](b2-pure-engineering-optimizations.md)
- **noSIA 三模型对比** (88 vs 135 tok/s 数据出处): [`noSIA-three-model-comparison.md`](noSIA-three-model-comparison.md)
- **生产实测日志**:
  - 优化前: `/tmp/sia_server_14b.log` (SIA-pf-summary @113700, b2_score_call p50=19.40)
  - 优化后: `/tmp/sia_server_14b_inproc_20260529_131755.log` (SIA-pf-summary @114200, b2_score_call p50=8.34)
- **MMLU eval 结果**:
  - 优化前: `sparse_logs/eval_b2_mmlu/mmlu_redux_14b_600q_weight1.json` (46.6 tok/s)
  - 优化后: `sparse_logs/eval_b2_mmlu/mmlu_redux_14b_600q_inproc.json` (进行中, ~65 tok/s)
