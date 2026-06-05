# 0GM-35B SIA 无损加速 — 可行路径分析 (2026-06-05)

**目标**: 在不牺牲 SIA 评测打分效果的前提下 (即不动 SIA params / RM / 主 LLM / sampling), 给 0GM-35B SIA 找纯工程加速方案。

**现状基准** (来自 [`doc/0gm-1.0-35b-sia-eval-20260605.md`](0gm-1.0-35b-sia-eval-20260605.md)):
- AlpacaEval 805Q: SIA **31.9 tok/s** vs noSIA 98 tok/s, **slowdown 3.07×**
- MMLU 150Q thinking: SIA **54.5 tok/s** vs noSIA 108.5 tok/s, **slowdown 2.0×**
- 瓶颈: 9-24% step 触发 RM HTTP /classify 调用, 单次 **p50 ≈ 70 ms**

**先决问题**: 单纯换 vllm 版本能不能解决? **基本不能** — 详见 §1。

---

## 1. 已实测验证过的失败配置 (不要再试)

### 1.1 同进程 b2 inproc 全军覆没

来自 [`0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md) §3.2:

| 配置 (vllm 0.19, 都同进程) | 结果 |
|---|---|
| 主 LLM `FULL_AND_PIECEWISE` + RM `PIECEWISE` | ❌ RM error 10%+ step |
| 主 LLM `FULL_AND_PIECEWISE` + RM **`eager`** | ❌ 仍 RM error |
| 主 LLM `PIECEWISE-only` + RM `PIECEWISE` | ❌ 100% INTERVENE 失败 |
| 主 LLM `PIECEWISE-only` + RM **`eager`** | ❌ 仍 RM error |

**关键事实**: 冲突来自**主 LLM 这一侧的全局 cudagraph flag** (vllm 0.13+ 引入的 `torch.cuda.graph()` context manager 设全局 `is_currently_capturing`), 跟 RM 怎么配置无关。只要 RM 跟主 LLM 在同一进程, 任何做 CUDA forward 的 inproc RM 都中招。

> ⚠️ **我 (Claude) 在 2026-06-05 对话中曾建议 "vllm 0.19 + nested RM `enforce_eager=True`"**, 错了。这个配置上表第 2 行已实测失败。flag 不在 RM 端, eager 改 RM 端没用。

### 1.2 旧 vllm 版本都缺 Qwen3.5MoE arch

| vllm 版本 | venv | 加载 0GM-35B? | 来源 |
|---|---|---|---|
| 0.10.1.1 | venv2 | ❌ arch 未注册 | `0gm-35b-sia-rm-inproc-path-20260602.md` §3.3 |
| 0.15.0 | venv3 | ❌ arch 未注册 | 同上 |
| **0.17.1** | **venv5** | **❌ arch 未注册** | `vl30b-b2-inproc-speedup-20260605.md` §6.1 |
| 0.19.0 | venv4 | ✅ 唯一能加载的版本 | — |

0GM-35B 是 Qwen3.5/3.6 MoE 架构 (`Qwen3_5MoeForConditionalGeneration`), vllm 0.19+ 才注册该 arch。这就是为什么 VL-30B 能享受 0.17.1 sweet spot 加速 (1.46×), 35B 不能。

---

## 2. 仍未实测的剩余路径

下面这些**都没在 35B 上跑过**, 排序按工程成本递增。

### 2.1 vllm 0.18.x 探针 (30 分钟, 高 ROI 如果存在)

- **未验证**: 0.18.x 是否存在, 以及是否同时满足:
  1. 注册了 Qwen3.5MoE arch
  2. PIECEWISE 仍是 AOT capture (跟 0.17.1 一样, 不撞 b2 inproc 冲突)
- 这是**唯一靠纯换 vllm 版本"白捡 b2 inproc"的可能性**
- 工作量: `python3 -m venv venv-0.18 && pip install vllm==0.18.0` + 跑现成 b2 inproc smoke test
- 结果二选一:
  - ✅ work → 给 35B 找到 sweet spot, 跟 VL-30B 一样可享受 ~1.5× 加速
  - ❌ arch 未注册或仍撞冲突 → 写入失败表, 转其它路径

### 2.2 方案 A — transformers + DynamicCache inproc RM (3-5 天)

来自 [`0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md) §5, 状态: **未实施**。

**核心思想**: RM 不走 vllm, 改用 raw transformers (默认不调 `torch.cuda.graph()` 也不设全局 flag), 配 `DynamicCache` 手动维护 KV cache, 实现 1+4 batch 优化 (prefix forward 一次, 5 candidates 各 append 1 token)。

**预期 (悲观-乐观区间)**:
| 场景 | per-INTERVENE | per-token avg SIA overhead | SIA throughput |
|---|---|---|---|
| 现在 vllm HTTP | 94 ms | 12.9 ms/tok | 68 tok/s |
| transformers 乐观 (FlashAttn2 + 16ms/tok) | 25 ms | 7.5 ms/tok | **~85 tok/s (+25%)** |
| transformers 悲观 (无 FlashAttn2 + 28ms/tok) | 35 ms | 9.5 ms/tok | ~78 tok/s (+15%) |
| transformers 灾难 (cudagraph 仍冲突) | n/a | n/a | 退回 HTTP |

**关键风险点 — 必须先做 1 小时致命验证** ([`0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md) §5.5 Step 1):

主 LLM 在 inference 期间 **spontaneously 触发 cudagraph capture** (例如遇到没见过的 batch size, vllm 0.19 PIECEWISE runtime trigger), **同进程的 transformers CUDA op 仍会被全局 flag 阻断**。这一点 transformers 自己默认不调 cudagraph 不解决问题 — 主 LLM 的 capture 让所有同进程 CUDA op 撞墙。

→ **Step 1 验证脚本**: 同进程加载 vllm 0.19 主 LLM (0GM-35B, FULL_AND_PIECEWISE) + transformers Qwen3-4B; 主 LLM forward 期间并发 transformers forward, 看是否 `CUDA graph capturing detected at an inappropriate time`。**不通过则方案 A 直接报废**。

### 2.3 HTTP path 内部优化 (中等收益, 跟方案 A 正交)

这些跟 b2 inproc / 方案 A 独立, **任何时候可做**, 收益可以叠加:

| # | 改动 | 单 RM 调用预期 | SIA 端到端预期 | 工作量 | 风险 |
|---|---|---|---|---|---|
| 1 | gRPC / msgpack 替 HTTP/JSON | 70ms → 50ms (-30%) | 68 → ~80 tok/s (+18%) | 1-2 天 (协议重写) | 中 |
| 2 | RM 跟主 LLM 走 Unix socket (替 TCP) | 70ms → 55ms (-20%) | 68 → ~75 tok/s (+10%) | 半小时 (改 listen) | 低 |
| 3 | 主 LLM // RM 双 CUDA stream pipelining | overlap | **+20-40%** | 2-3 天 (改 logits processor 同步语义) | 中 |
| 4 | RM `--enable-prefix-caching` 实际命中率验证 | 如果命中率<50% → 加 -10% | 看实际 | 半天 (加 telemetry) | 低 |

#3 是单项最大潜在收益 (overlap RM call 跟主 LLM forward), 但需要修改 SIA processor 的同步语义 (现在 `apply(logits)` 是阻塞同步)。

### 2.4 自训练 linear-head VM (1+ 周 ML 工作)

方案 C in [`0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md) §5.6:
- 训一个 small linear head, 接在主 LLM 隐状态上
- 单 RM 调用 <1 ms, 接近 noSIA 速度
- **不是纯工程, 是 ML 工作**, 涉及训练数据、loss 设计、验证 RM 信号质量

如果用户严格要求"无损纯工程加速", **这条路出 scope**。但作为长期方案, 它是最终答案。

---

## 3. 主 LLM 端 enforce_eager (列为已知不可行)

方案 B in [`0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md) §5.6:
- 主 LLM `enforce_eager=True` → 不设 cudagraph 全局 flag → 同进程 b2 inproc 不撞冲突
- 但: **35B eager 主 LLM 慢 ~40%** → 净 throughput **40-50 tok/s, 比当前 HTTP 还差**

→ 不可行。列在这里只是提醒未来不要走这条路。

---

## 4. 推荐实施顺序

| 优先级 | 步骤 | 工作量 | 决策点 |
|---|---|---|---|
| **P0** | §2.1 vllm 0.18.x 探针 | 30 min | 成 / 败 二选一 |
| **P1** | §2.2 方案 A Step 1 (致命点验证脚本) | 1 hr | 决定方案 A 是否报废 |
| **P2** | §2.3 #2 Unix socket | 半天 | 跟 P0/P1 正交, 任何时候都可做 |
| **P3** | §2.3 #3 双 CUDA stream pipelining | 2-3 天 | 单项最大工程收益 |
| **P4** | (条件 P1 通过) §2.2 方案 A 完整实现 | 3-5 天 | benchmark 实测时延 ≤ 25ms 才继续 |
| **P5** | (长期) §2.4 linear-head VM | 1+ 周 ML | 出"纯工程"scope, 但是最终方向 |

**最小动作**: P0 (30 min vllm 0.18 探针) + P1 (1 hr 致命验证) — 2 小时总共, 决定 SIA-35B 加速的整体路径。

---

## 5. 引用 + 来源

- [`0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md): 35B inproc 路径成本分析 (基础 doc, 含失败配置表 §3.2 / 方案 A B C 比较 §5.6 / 致命点验证 §5.5)
- [`vl30b-b2-inproc-speedup-20260605.md`](vl30b-b2-inproc-speedup-20260605.md): VL-30B 上 b2 inproc 加速 1.46× 实测, 含 vllm 0.15/0.17.1/0.19 版本对比 (§1 / §6.1) — 35B 不可用此路径的 reference
- [`0gm-1.0-35b-sia-eval-20260605.md`](0gm-1.0-35b-sia-eval-20260605.md): 35B 当前 SIA throughput baseline (AlpacaEval / MMLU 数据, §二 性能评测)
- [`qwen3-vl-30b-sia-eval-20260605.md`](qwen3-vl-30b-sia-eval-20260605.md) §四 "后续优化空间": 类似的 inproc + 双 stream 优化方向, 已在 VL-30B 上提出

---

## 6. 笔记: 我之前的错误建议

2026-06-05 对话中我曾建议 "vllm 0.19 + nested RM `enforce_eager=True`" 是最有希望的路径。**这是错误建议** — `0gm-35b-sia-rm-inproc-path-20260602.md` §3.2 表第 2 行已实测过这个配置, 失败。冲突源是主 LLM 的全局 flag, 不是 RM 端的 cudagraph mode。**任何只改 RM 端的方案都不解决问题**, 必须主 LLM 端也 eager (方案 B, 但主 LLM eager 严重慢, 不可行)。记录在此, 避免之后再绕回去。
