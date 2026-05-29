# 消除 vLLM 5-prompt 的 1+4 split — 根因 + 解决方案

> TL;DR — vLLM 默认的多进程 EngineCore 通过 ZMQ 串行接收 add_request, 导致
> SIA 的 5 个 shared-prefix prompt 在 EngineCore 子进程被拆成 1+4 两次 forward。
> 切换到 **InprocClient**（设 `VLLM_ENABLE_V1_MULTIPROCESSING=0`）就让 5 个
> request 同步入队再一次性 schedule 成一个 batch=5 forward。
>
> **实测**: Stage B GPU forward p50 **13.29 → 4.78 ms (−64%)**, wall-clock
> p50 **23.96 → 9.04 ms (−62%)**, 100% iteration 走单一 batch=5 forward。

## 背景：1+4 split 是怎么来的

[`doc/rm-batch5-ab-test.md`](rm-batch5-ab-test.md) 的 A/B 测试观察到一个奇怪现象：
SIA 一次 `score_candidates` 给 vLLM 提交 5 个 shared-prefix prompt（prefix + [c_i]），
本应一次 batch=5 forward 搞定, 但 **98% 的 iteration 被 vLLM 拆成两次 forward**
(batch=1 + batch=4)。即使在 `cuda_graph_sizes=[1,2,4,5,...]` 显式包含 batch=5,
batch=5 graph 也几乎从未被命中。

当时的 hypothesis 都是关于 **scheduler 内部 batching 限制**（max_num_partial_prefills,
chunked_prefill, max_num_seqs 等等）, 但全部排查后没有一个能解释这个现象。

## 调查过程

### 1. 排查 scheduler 内部限制

读 vLLM 0.10.1.1 的 v1 scheduler (`vllm/v1/core/sched/scheduler.py:schedule()`):

- WAITING queue 处理循环没有 "一步只能 schedule N 个 new request" 的限制
- 唯一的 break 条件: `len(self.running) == self.max_num_running_reqs` (默认 1024 in
  LLM_CLASS mode for H100/H200), 或 `token_budget == 0` (默认 16384)
- 5 个 prompt × 9 token/prompt = 45 token, 远低于 budget
- `max_num_partial_prefills=1` 这个 v0 配置 **不影响 v1** (v1 scheduler 不读它)
- comment 明确写: "There's no 'decoding phase' nor 'prefill phase' in the scheduler."

→ scheduler 层面**理论上应该一步 schedule 完全部 5 个**。

### 2. 排查 model runner 内部切分

读 `vllm/v1/worker/gpu_model_runner.py:execute_model()` — 它从 `SchedulerOutput`
直接取 `total_num_scheduled_tokens`, 调一次 `self.model(input_ids=..., positions=...)`,
**不会主动把 prefill 拆出来**。

→ model runner 层面也**不切分**。

### 3. 真正的根因：**EngineCoreProc 的 ZMQ race condition**

读 `vllm/v1/engine/core_client.py` 发现:

```python
class SyncMPClient(MPClient):  # vLLM 默认在 LLM() 里用这个
    """Synchronous client for multi-proc EngineCore."""
```

而 `EngineCoreProc` 是后台子进程, 通过 ZMQ socket 收 ADD requests:

```python
# vllm/v1/engine/core.py
def run_busy_loop(self):
    while True:
        self._process_input_queue()       # blocking get + drain (no wait)
        self._process_engine_step()       # schedule + execute + output

def _process_input_queue(self):
    waited = False
    while not self.engines_running and not self.scheduler.has_requests():
        req = self.input_queue.get()      # BLOCKS until first req arrives
        self._handle_client_request(*req)
    # drain remaining (non-blocking, takes whatever is already there)
    while not self.input_queue.empty():
        req = self.input_queue.get_nowait()
        self._handle_client_request(*req)
```

客户端 (LLM.generate) **串行** add 5 个 prompt:

```python
# vllm/entrypoints/llm.py
for prompt in prompts:                    # 5 次串行
    self._add_request(prompt, ...)        # → ZMQ socket send
self._run_engine()                        # 等输出
```

**Race 时间线** (实测):
```
T=0   us : 客户端 send req#1 → ZMQ → input_thread put → input_queue
T=10  us : EngineCore 主线程 _process_input_queue 拿到 req#1, drain (空)
T=50  us : EngineCore 进入 _process_engine_step → schedule(req#1) → execute_model
T=100 us : 客户端送来 req#2-5 (各间隔几十 μs IPC)
T=12 ms  : execute_model 结束 (batch=1 forward)
T=12 ms+ : 回到 _process_input_queue → drain 拿到 req#2-5
T=12 ms+ : schedule(req#2-5) → execute_model (batch=4 forward)
```

**这就是 1+4 split 的真凶**——不是 scheduler bug, 不是 cudagraph 配置问题, 而是
跨进程异步消息传递的 race。

## 解决方案：InprocClient

`vllm/v1/engine/core_client.py` 还提供了一个被忽略的 client:

```python
class InprocClient(EngineCoreClient):
    """In-process EngineCore. V0-style add_request() and step(). No busy loop."""

    def add_request(self, request):
        # synchronous, in-process method call
        self.engine_core.add_request(...)
```

启用方式: **设环境变量 `VLLM_ENABLE_V1_MULTIPROCESSING=0`** (必须在 `import vllm`
之前设置, 因为 `LLMEngine.from_engine_args` 在构造时读它):

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, ...
llm = LLM(model=..., ...)
```

效果: 5 个 add_request 同步入队, 全部加完才进入 `_run_engine()` → 第一次
schedule() 就把 5 个 request 一起拉进 running queue → 一次 batch=5 forward。

## 实测结果

[`scripts/bench_rm_inproc_ab.py`](../scripts/bench_rm_inproc_ab.py) —
跑 10 warmup + 100 measurement, 同 seed, 同 prefix (600 token), 同 5 candidates。

### Stage B GPU forward (cuda.Event 测量, 排除 wall-clock 噪声)

| metric | default (MP+ZMQ) | inproc | Δ |
|--------|--------:|--------:|---|
| min    | 9.02 ms  | 4.59 ms  | −49% |
| **p50** | **13.29 ms** | **4.78 ms** | **−64%** |
| mean   | 14.92 ms | 4.82 ms  | −68% |
| **p95** | **29.19 ms** | **5.06 ms** | **−83%** |
| max    | 43.48 ms | 6.02 ms  | −86% |

### Stage B wall-clock (端到端)

| metric | default | inproc | Δ |
|--------|--------:|--------:|---|
| **p50** | **23.96 ms** | **9.04 ms** | **−62%** |
| mean   | 27.59 ms | 9.19 ms  | −67% |
| p95    | 49.07 ms | 10.02 ms | −80% |

### Per-iteration forward 分布

| # forwards / iter | default | inproc |
|---:|---:|---:|
| 1 (batch=5) | 5 (5%) | **100 (100%)** |
| 2 (1+4) | 93 (93%) | 0 |
| 3 | 2 (2%) | 0 |

**所有 100 个 iteration 在 inproc 模式下都是单一 batch=5 forward**。
batch=5 graph 100% 命中, 没有 padding 浪费。

### 为什么节省比 "1+4 → 5" 还多

理论上 1+4 = 2 个 forward (~12 ms 总) → 1 个 batch=5 forward (~5 ms), 节省 ~7 ms。
实测 GPU 时间节省 8.5 ms (13.29 → 4.78), wall-clock 节省 15 ms — 超出预期的部分是:

1. **scheduler 间隙消失**: 2 次 forward 之间, scheduler 还要再跑一次 schedule()
   + update_from_output + 准备下一次 forward, 这个 gap wall-clock 约 5-7 ms。
2. **prefill mode → decode mode**: 1+4 都被 vLLM 当 "新 request 的 prefill" 处理,
   走 prefill-attention kernel; 而 inproc 一次性 5 个一起 schedule 也是 prefill mode,
   但 batch=5 一次 attention 调用要比 batch=1 + batch=4 两次调用更高效。
3. **p95 抖动消失**: ZMQ 跨进程通信本身就有 jitter (上下文切换, GIL, queue 锁等),
   inproc 没这些。

## 集成

[`src/sia_rm/client.py`](../src/sia_rm/client.py) `RMClient.__init__` 现在接受
`multiprocessing: bool = False` 参数 (默认 False, 即 inproc 模式):

```python
rm = RMClient(model_path=..., gpu_mem=0.3)
# 等价于 multiprocessing=False, 自动启用 InprocClient
```

如果需要回退 (例如真要并发多 client / async usecase):

```python
rm = RMClient(model_path=..., multiprocessing=True)
```

## Trade-offs / 注意事项

| 项 | InprocClient | SyncMPClient (default) |
|---|---|---|
| 5-prompt batch | **batch=5 一次 forward** ✓ | 1+4 拆 2 次 forward |
| Stage B latency | **−60% to −80%** | baseline |
| 内存 footprint | 同进程, 不需要额外子进程 | 多一个 Python 子进程 (~200 MB) |
| 健壮性 | 子进程崩溃 = 主进程崩溃 | 子进程崩溃可被父进程检测 |
| async/await usage | ✗ (asyncio_mode 不兼容 inproc) | ✓ (AsyncMPClient) |
| 跟主 LLM 进程隔离 | 同进程; GPU memory 共享要小心 | 完全隔离 |
| /dev/shm 文件 IPC | 仍然 work (但本可优化掉) | 仍然 work |

对于 SIA 的典型 use case (单 client, 同步 score, 一个 LLM + 一个 RM 跑在一个机器上),
**InprocClient 是绝对的赢家**。

对于 future asyncio-based server 或 multi-tenant 场景, 才需要 multiprocessing=True。

## 副作用：reward file IPC 的简化机会

[`src/sia_rm/qwen3_with_score.py`](../src/sia_rm/qwen3_with_score.py) 当前把 5 个
candidate 的 reward 写到 `/dev/shm/sia_reward_<id>.bin`, 是因为 EngineCore 在子进程,
score head 输出无法通过 Python return 传回主进程。

InprocClient 下, `compute_logits` 跟客户端**同进程**, 完全可以直接 return tensor 或
存在 instance 属性。当前实现仍然走 /dev/shm 文件（无 perf cost, 因为 tmpfs in-memory）,
不强求改, 但未来若进一步优化可以移除这层 IPC。

## 引用

- 客户端改动: [`src/sia_rm/client.py`](../src/sia_rm/client.py) (加 `multiprocessing` 参数)
- A/B bench: [`scripts/bench_rm_inproc_ab.py`](../scripts/bench_rm_inproc_ab.py)
- Repro 脚本: [`scripts/repro_vllm_5_prompts_split.py`](../scripts/repro_vllm_5_prompts_split.py)
- 输出 JSON: `/tmp/bench_rm_inproc_ab_default.json`, `/tmp/bench_rm_inproc_ab_inproc.json`
- vLLM 源码:
  - `vllm/v1/engine/core_client.py` (InprocClient vs SyncMPClient)
  - `vllm/v1/engine/core.py` (run_busy_loop, _process_input_queue)
  - `vllm/v1/engine/llm_engine.py:from_engine_args` (读 VLLM_ENABLE_V1_MULTIPROCESSING)
- 上一份相关 doc: [`rm-batch5-ab-test.md`](rm-batch5-ab-test.md) (cuda_graph batch=5
  ablation, 结论"1-2 ms 收益在测量噪声内, 而且 batch=5 graph 从未被命中" —
  现在终于找到为什么 batch=5 graph 从未被命中: 不是 cudagraph 问题, 是 ZMQ race)
