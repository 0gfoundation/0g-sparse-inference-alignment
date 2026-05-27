# B2 M2 设计：stateful RM in-process server

**前置**：M1a/M1b 已通过，见 [`b2-decode-mode-poc-plan.md`](b2-decode-mode-poc-plan.md) §6.5。本文落档 M2 的实现方案，等待 §7 micro-bench 三个疑点确认后正式开撸。

---

## 0. M2 一句话目标

把 PoC 期间散落在 vLLM 源码 + VM base config 上的 hack，全部收纳进 SIA package 内的一个 `sia_rm/` 模块；给 SIA processor 提供 `fix_a_token` / `score_candidates` 同步 in-process API，跑出 §6.5 里实测过的 ~7ms/call。

---

## 1. 范围与非目标

**做**：
- 实现自定义 vLLM model class `Qwen3WithScoreForCausalLM`，通过 `ModelRegistry.register_model` 注册
- 实现 `RMClient` 封装 vLLM `LLM` 实例 + stateful session 管理
- 把 SIA processor 改成 in-process 调用 `RMClient`（替代当前 HTTP `/classify`）

**不做**：
- 不再修改 vLLM 源代码或 VM 模型 config
- 不引入 HTTP/RPC 协议层（in-process 即可）
- 不做多 GPU 调度（单卡先跑通）
- 不做 quantization（quality-neutral 优化原则）

---

## 2. 整体架构

```
┌───────────────────────────────────────────────────────────┐
│ SIA process (单进程)                                       │
│                                                            │
│  ┌──────────────────┐       ┌──────────────────────────┐  │
│  │ vLLM (LLM 主模型) │       │ RMClient                  │  │
│  │  Qwen3-4B-base   │       │  - sessions: dict[id]    │  │
│  │  generate loop   │──┬───▶│  - llm: vllm.LLM (VM)    │  │
│  └──────────────────┘  │    │                          │  │
│                        │    │  API:                    │  │
│  ┌──────────────────┐  │    │   new_session(prompt_ids)│  │
│  │ SIA              │  │    │   fix_a_token(sid, tid)  │  │
│  │ LogitsProcessor  │──┘    │   score_candidates(sid,  │  │
│  │  on each token: │        │     [tid1..tidN]) →floats│  │
│  │   call RMClient  │       └──────────┬───────────────┘  │
│  └──────────────────┘                  │                  │
│                                        ▼                  │
│                          ┌──────────────────────────────┐ │
│                          │ vLLM RM 实例 (sync LLM)      │ │
│                          │  Qwen3WithScoreForCausalLM   │ │
│                          │  prefix caching=ON           │ │
│                          │                              │ │
│                          │  compute_logits hook:        │ │
│                          │   reward = score(hidden)     │ │
│                          │   threading.local.last_reward│ │
│                          └──────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

两个 vLLM 实例（LLM 主模型 + RM）共享同一张 GPU；通过 `gpu_memory_utilization` 拆分显存。

---

## 3. Custom model class（替代源码 patch）

文件位置：`src/sia_rm/qwen3_with_score.py`

```python
import torch
from torch import nn
from vllm import ModelRegistry
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.layers.linear import RowParallelLinear  # 或 plain Linear
from vllm.model_executor.sampling_metadata import SamplingMetadata


_TLS = __import__('threading').local()


class Qwen3WithScoreForCausalLM(Qwen3ForCausalLM):
    """
    Qwen3ForCausalLM + 外接 score head (Linear(hidden, 1)).

    score head 权重直接从 VM checkpoint 的 score.weight 加载,
    不需要 patch base model.
    """

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hidden = vllm_config.model_config.hf_config.hidden_size
        # score 是 row-parallel-friendly Linear; 单卡用 plain nn.Linear 即可
        self.score = nn.Linear(hidden, 1, bias=False)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ):
        # hook: 写 thread-local channel
        # hidden_states shape (n_samples, hidden) — 已经是 sample positions
        try:
            rewards = (hidden_states @ self.score.weight.T).squeeze(-1)
            _TLS.last_rewards = rewards.detach().to('cpu', torch.float32)
        except Exception:
            _TLS.last_rewards = None

        # 仍走原 lm_head logits 计算 — sample 一个 dummy token 走 vLLM scheduler
        return super().compute_logits(hidden_states, sampling_metadata)


# 模块 import 时副作用：注册到 vLLM
# 注意 vLLM 期望 lazy "<module>:<class>" 字符串
ModelRegistry.register_model(
    "Qwen3WithScoreForCausalLM",
    "sia_rm.qwen3_with_score:Qwen3WithScoreForCausalLM",
)


def get_last_rewards():
    return getattr(_TLS, 'last_rewards', None)
```

**vLLM 启动方式**：

```python
llm = vllm.LLM(
    model="/workspace/SIA/models/VM-Qwen3-4B-merged-for-vllm",
    hf_overrides={"architectures": ["Qwen3WithScoreForCausalLM"]},
    dtype="bfloat16",
    enable_prefix_caching=True,
    gpu_memory_utilization=0.3,
    ...
)
```

> **疑点 B (§7)**：vLLM v1 是否真的支持 `hf_overrides` 在 runtime 切换 architectures，需 micro-bench 确认。

---

## 4. RMClient API

文件位置：`src/sia_rm/client.py`

```python
class RMClient:
    """
    In-process stateful RM client.

    用法（SIA processor 内部）：
        rm = RMClient(model_path="...")
        sid = rm.new_session(prompt_token_ids)
        # 每次 SIA 生成一个 token 后:
        rm.fix_a_token(sid, generated_token_id)
        # 当 SIA 要选 candidate 时:
        rewards = rm.score_candidates(sid, [t1, t2, t3, t4, t5])
    """

    def __init__(self, model_path: str, gpu_mem: float = 0.3,
                 max_model_len: int = 4096):
        import sia_rm.qwen3_with_score  # noqa — 触发 ModelRegistry 注册
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=model_path,
            hf_overrides={"architectures": ["Qwen3WithScoreForCausalLM"]},
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_mem,
            max_model_len=max_model_len,
        )
        self.sp = SamplingParams(temperature=0.0, max_tokens=1,
                                 min_tokens=1, ignore_eos=True)
        self._sessions: dict[int, list[int]] = {}
        self._next_id = 0

    def new_session(self, prompt_token_ids: list[int]) -> int:
        sid = self._next_id
        self._next_id += 1
        self._sessions[sid] = list(prompt_token_ids)
        return sid

    def fix_a_token(self, sid: int, token_id: int) -> None:
        """Commit a token to the session prefix. No vLLM call."""
        self._sessions[sid].append(token_id)

    def score_candidates(self, sid: int,
                         candidate_token_ids: list[int]) -> list[float]:
        prefix = self._sessions[sid]
        # N 个 prompts = [prefix + [c] for c in candidates]
        prompts = [prefix + [c] for c in candidate_token_ids]
        # batch generate 1 token each
        from vllm import TokensPrompt
        outs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts],
            self.sp, use_tqdm=False,
        )
        # 从 thread-local channel 读 rewards
        from sia_rm.qwen3_with_score import get_last_rewards
        r = get_last_rewards()
        assert r is not None and r.numel() == len(candidate_token_ids), \
            f"reward channel returned {r}, expected {len(candidate_token_ids)}"
        return r.tolist()

    def end_session(self, sid: int) -> None:
        self._sessions.pop(sid, None)
```

**核心简化**：
- `fix_a_token` **不调 vLLM**，仅 append 到 `prefix_ids`。下次 `score_candidates` 时让 prefix caching 自动复用 KV
- `score_candidates` 用 batch generate（5 个 prompt 同时跑），vLLM v1 continuous batching 自动合并
- 因为所有 5 个 prompt 共享 N-1 token 前缀，vLLM 的 prefix cache 应该让每次 forward 只跑 1 个新 token（5 行 × 1 token）

> **疑点 A (§7)**：prefix caching 是否真的让 score_candidates 跑成 ~7ms（block-aligned 问题），需 micro-bench 确认。

---

## 5. SIA processor 接入

`src/sia_vllm_RM.py` 改动（删 HTTP 客户端，换 in-process）：

```diff
- import requests
- ...
- def _score_via_classify(self, prompts: list[str]) -> list[float]:
-     resp = requests.post(self.rm_url + "/classify", json={...})
-     return resp.json()["scores"]
+ from sia_rm.client import RMClient
+ ...
+ def __init__(self, ..., rm_model_path: str):
+     self.rm = RMClient(model_path=rm_model_path, gpu_mem=0.3)
+     # SIA processor 在 update_state 里给每个 request 分配 RM session
+     self._sid_by_req: dict[str, int] = {}
+
+ def update_state(self, batch_delta):
+     # 处理 added: new_session;
+     # 处理 removed: end_session
+     ...
+
+ def apply(self, ...):
+     # 在每个 token 上:
+     for req_id, ... in batch:
+         sid = self._sid_by_req[req_id]
+         self.rm.fix_a_token(sid, last_generated_token_id)
+         if should_intervene(entropy):
+             rewards = self.rm.score_candidates(sid, topk_token_ids)
+             logits[topk_indices] += weight * (rewards - rewards.mean())
```

---

## 6. 测试 / benchmark 计划

- **单元测试**：`tests/test_rm_client.py` —— 用 §6.5 的 5-candidate 验证 RMClient.score_candidates 结果跟 BF16 baseline `max|Δ| < 0.5`
- **micro-bench**：`scripts/bench_rm_client.py` —— 100 次 score_candidates，p50/p95 延迟，对比 §6.5 实测的 7ms 目标
- **端到端**：把 SIA processor 接好后跑 MMLU 600 题，跟 G3 baseline 比 tok/s（目标：单卡 ~49 tok/s，从 baseline 30 提升）

---

## 7. 三个疑点 micro-bench

进入正式实现前必须先跑三个 ~30 行的验证脚本，避免设计依赖于未验证的假设。

### 7.A prefix caching 真的能省到 ~7ms？

**问题**：vLLM v1 的 prefix caching 是 block-aligned（block_size=16），如果 prefix 长度 N 不是 16 倍数，最后那个 partial-block 是否仍 prefill？

**bench 脚本**：`scripts/bench_b2_prefix_cache.py`
- 启动 vLLM (启 prefix caching) 加载 Qwen3-4B (随便，不需要 score head)
- 维护一个 prefix list，从长度 50 开始
- 循环 100 次：每次给 prefix append 1 个 token，调 5 prompt batch generate（max_tokens=1），测 latency
- 期望：稳定在 ~7-15ms（含 prefill of 1 token + decode），不是 70ms（cold prefill 整 prefix）

**判定**：
- 若 p50 < 20ms：✅ 设计可行
- 若 p50 > 30ms：⚠️ 需手动 prefix-warm 或换 stateful API

### 7.B `hf_overrides` 在 vLLM v1 是否支持 architectures override？

**问题**：M2 设计依赖运行时把 VM 的 `Qwen3ForSequenceClassification` 重 dispatch 到 `Qwen3WithScoreForCausalLM`，没有这个开关就得改 base config 或起 fork。

**bench 脚本**：`scripts/bench_b2_hf_overrides.py`（10 行）
- 写一个 dummy `Qwen3WithScoreForCausalLM(Qwen3ForCausalLM)` 注册
- vllm.LLM(model=VM, hf_overrides={"architectures": ["Qwen3WithScoreForCausalLM"]})
- 看 supported_tasks 是不是 ['generate']，能不能 generate 出 token

**判定**：
- 启动成功且 generate 出 token：✅
- 启动失败：⚠️ 退而求其次，要么写一个 wrapper config，要么用 `--model-impl` 强制

### 7.C ModelRegistry register 跨 subprocess 是否生效？

**问题**：M1a 已踩坑——monkey-patch 不跨 EngineCore subprocess。`ModelRegistry.register_model` 是注册到全局变量，主进程改了，subprocess 用 fork (linux) 才能继承；如果 vLLM v1 用 spawn 启 subprocess 就完蛋。

**bench 脚本**：`scripts/bench_b2_registry_subprocess.py`
- 在主进程 register 一个 dummy model
- 启动 vLLM (会 fork/spawn EngineCore subprocess)
- 看 subprocess 是否能找到该 model

**判定**：
- 能找到：✅ 设计成立
- 找不到：⚠️ 需要把 register 调用放到 EngineCore subprocess 启动后能 import 到的地方（一般通过 `--model-impl` 配 environment variable + lazy module import）

---

## 7.X 实测结果（2026-05-27）

### §7.B + §7.C ✅ PASS

脚本：[`scripts/bench_b2_registry_and_override.py`](../scripts/bench_b2_registry_and_override.py) + [`scripts/b2_test_model.py`](../scripts/b2_test_model.py)

- vLLM 接受 `hf_overrides={"architectures": ["Qwen3WithScoreTest"]}`，把 VM 模型（架构=`Qwen3ForSequenceClassification`）成功 dispatch 到我们注册的 `Qwen3ForCausalLM` 子类
- `score.weight` 自动加载到 `self.score`（不需要 `skip_prefixes`），因为 `Qwen3WithScoreTest.__init__` 加了对应 module
- generate 成功输出文本，supported_tasks = `['generate']`

**踩坑**：vLLM 的 `_run_in_subprocess` 是用 `python -m vllm.model_executor.models.registry` 起一个**全新独立的 Python subprocess**（不是 fork），`sys.path` 修改不继承。必须显式设环境变量 `PYTHONPATH` 让 subprocess 找到自定义 model module：

```python
os.environ["PYTHONPATH"] = "/path/to/sia_rm:" + os.environ.get("PYTHONPATH", "")
```

或者把 `sia_rm` 安装为正式 site-packages package。M2 实现时应该走第二条路（pip install -e .）让 SIA 成为正常 importable package。

### §7.A ✅ PASS（带 spike 隐患）

脚本：[`scripts/bench_b2_prefix_cache.py`](../scripts/bench_b2_prefix_cache.py)

setup: 起始 prefix 50 token，每 iter append 1 token，跑 5 prompt batch generate (max_tokens=1)，100 iter 测稳态延迟。

| metric | 值 | 注释 |
|--------|------|------|
| min    | **7.88 ms** | ≈ F=7ms 实测，prefix cache 命中时只跑 5×1 token decode |
| **p50** | **15.22 ms** | ✅ < 20ms 阈值 |
| avg    | 16.9 ms | |
| p95    | 35.0 ms | ⚠️ 偶发 spike |
| max    | 51.4 ms | ⚠️ 估计是 block-boundary partial prefill |

**trajectory（每 10 个 iter 的 p50）**：
- iter 0-19 (prefix 55-75): p50 ~20ms（warm up 期）
- iter 20-59 (prefix 75-115): p50 9-14ms（最稳）
- iter 60-89 (prefix 115-145): 偶有 spike，p50 ~14-24ms（block boundary）
- iter 90-99: 稳回 14ms

**分析**：
- p50=15ms 达成（< 20ms），M2 设计可行
- 偶发 max=51ms 估计是 block_size=16 的 prefix cache 在每 16 个 token 处理一个新 block 的 partial prefill；可以通过更小 block_size 或 manual prewarm 缓解，但 SIA 看 throughput 而非 tail latency，p50 已够
- 实际 SIA 收益估算：current baseline 30 tok/s（RM HTTP ~30ms/call）→ B2 后约 **45-57 tok/s**（RM 15ms/call + 7ms LLM 主步），跟 §4 doc 预测一致

### 综合判定：✅ 全部 PASS，进 step 1 实现

---

## 8. 实施步骤

按以下顺序，每个 step 跑完即 commit，失败可以一键回退：

| step | 内容 | 估时 | 阻塞条件 |
|------|------|------|--------|
| 0 | 跑 §7 三个 micro-bench，确认设计前提 | 0.5 天 | 任何 ❌ 都得回头调设计 |
| 1 | 写 `sia_rm/qwen3_with_score.py` + 单测（5-candidate 数值对得上 BF16） | 1 天 | step 0 全 ✅ |
| 2 | 写 `sia_rm/client.py` + score_candidates micro-bench | 1 天 | step 1 OK |
| 3 | 写 fix_a_token 集成（session prefix 管理） + bench fix_a_token 链式调用延迟 | 1 天 | step 2 OK |
| 4 | 改 `sia_vllm_RM.py` 把 HTTP 客户端换成 RMClient | 1 天 | step 3 OK |
| 5 | 端到端：600 题 MMLU + tok/s 测量 | 1 天 | step 4 OK |
| 6 | 清理 PoC 期残留（restore vLLM 源码 + VM config.json） | 0.5 天 | step 5 PASS |

总 ~6 天，落 PoC 期估计的 ~1 周。

---

## 9. 已知风险点

| 风险 | 缓解 |
|------|------|
| prefix caching 命中率不够导致 score_candidates > 15ms | §7.A bench 验，若 fail 走方案 B（custom adapter） |
| hf_overrides 不支持 architectures | §7.B bench 验，若 fail 写一个 sym-link config 目录 |
| ModelRegistry 不跨 subprocess | §7.C bench 验，若 fail 改用环境变量 + lazy import |
| 两个 vLLM 实例同 GPU 显存竞争（LLM 0.3 + RM 0.3 = 0.6，还有 0.4 给 CUDA 临时空间） | 实际测，OOM 则降 RM 到 0.25 |
| `threading.local()` 在 vLLM subprocess 内是否生效 | vLLM 同步 generate 时调用线程和 hook 是同一线程，理论上 OK；step 1 单测会暴露 |
| SIA processor 现有 batch update 逻辑跟 RM session 生命周期对不上 | step 3 单独测；不通就把 session 表用 weak-ref 兜底 |

---

## 10. 引用

- M1 PoC 验收（hidden_state 提取 + score 正确性）: [`b2-decode-mode-poc-plan.md`](b2-decode-mode-poc-plan.md) §6.5
- vLLM continuous decode 7ms 实测: [`parallel-decoding-design.md`](parallel-decoding-design.md) §2.8
- 当前 SIA processor 实现: [`../src/sia_vllm_RM.py`](../src/sia_vllm_RM.py)
