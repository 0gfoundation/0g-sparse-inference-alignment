# SIA 干预率跟哪些 sampling 参数最相关 — 参数 isolation 实证

**日期**: 2026-06-02
**主推理模型**: Qwen3-14B (同步在 0GM-35B 上交叉验证)
**Value Model (RM)**: VM-Qwen3-4B-merged-for-vllm
**SIA 配置**: `topk=5, weight=1.0, entropy_threshold=1.0` (不变量)
**vllm 版本**: 0.19.0 (venv4)
**Instrumentation**: `SIA_DEBUG_HIST=1` (commit `b72764e`) — per-request entropy / gap / inthink 直方图

---

## 1. 背景: 为什么要做这个实证

历史上 (2026-05 G3/G4 等 14B SIA 实验, 详见 [`exp/README.md`](../exp/README.md)) 14B 模型在 `entropy_threshold=1.0` 下 SIA 干预率稳定在 **29-30%**, top-1 flip 率高达 67%。

但 2026-06-02 在 0GM-35B 跑同样配置时干预率掉到 **8.34%**, 14B 在同配置下也只剩 **10.65%** (见 [`doc/0gm-35b-sia-vs-nosia-eval-20260602.md`](0gm-35b-sia-vs-nosia-eval-20260602.md))。

初步怀疑过几个假设:
- ~~35B 比 14B 更自信 → top-5 entropy 更尖锐~~ **已排除** (14B/0GM entropy 分布几乎一致)
- ~~vllm 0.10 → 0.19 的 sampler kernel 差异~~ **已排除** (本实证证明)
- ~~thinking 模式让 entropy 偏低~~ **部分相关但不是主因**

真正可疑的是: 6 月修 0GM 输出乱码 bug 时, 我引入了 client 显式传 `--top_p 0.95 --top_k 20 --repetition_penalty 1.0`, 跟历史 (server 硬编码 `top_p=1.0, top_k=-1, rep_penalty=1.3`) **不同**。

这次实证就是要定量回答: **这三个 sampling 参数, 哪个对 SIA 干预率影响最大?**

---

## 2. 实验方法

- 同一个 14B server (SIA_DEBUG_HIST=1) 持续运行
- 用同一道高熵 prompt (anatomy Q1) 重复 curl, 每条件 3 次请求
- 每个请求结束 server 打 `[SIA] DONE intervened=X/Y ratio=Z% top1_flip=...`
- 取每条件 3 次的平均干预率作为该条件的指标

**6 个 condition**:

| Cond | top_k | top_p | rep_penalty | 说明 |
|------|-------|-------|-------------|------|
| A | -1 | 1.0 | **1.3** | 历史 server 硬编码默认 (baseline) |
| B | -1 | 1.0 | **1.0** | 只改 rep_penalty |
| C | **20** | **0.95** | 1.3 | top_k + top_p 一起改 |
| D | **20** | **0.95** | **1.0** | 全套修复 (现状) |
| E | **20** | 1.0 | 1.3 | 只改 top_k |
| F | -1 | **0.95** | 1.3 | 只改 top_p |

---

## 3. 实测结果

| Cond | top_k | top_p | rep_penalty | 3 req intervene rate | **avg** |
|------|-------|-------|-------------|---------------------|---------|
| **A** baseline (历史) | -1 | 1.0 | **1.3** | 39.0%, 58.3%, 59.2% | **52.2%** |
| **B** | -1 | 1.0 | **1.0** | 17.3%, 17.3%, 16.7% | **17.1%** |
| **C** | **20** | **0.95** | 1.3 | 33.5%, 34.5%, 34.3% | **34.1%** |
| **D** 全套修复 | **20** | **0.95** | **1.0** | 14.7%, 18.0%, 15.5% | **16.1%** |
| **E** | **20** | 1.0 | 1.3 | 43.0%, 38.3%, 39.8% | **40.4%** |
| **F** | -1 | **0.95** | 1.3 | 48.7%, 47.5%, 39.3% | **45.2%** |

> 注: 单 prompt × 3 req 样本量很小, 单条件方差 ~±5pp。Baseline A 三个 req 值 39-59% 跨度大是这个原因。完整 600Q eval 时 A 类条件累积平均稳在 ~29%, B/D 类条件累积平均稳在 ~10%。这里看的是**相对顺序**, 不是精确数字。

---

## 4. 三个参数的独立贡献

跟 baseline A (52.2%) 比较, 每个参数**单独**改动的边际效应:

| 参数变化 | 条件 | 干预率 | Δ vs A | 占总下降 36pp 的比例 |
|---------|------|--------|--------|---------------------|
| **`rep_penalty: 1.3 → 1.0`** | A → B | 17.1% | **-35.1pp** | **97%** ← 几乎独占 |
| `top_k: -1 → 20` | A → E | 40.4% | **-11.8pp** | 33% |
| `top_p: 1.0 → 0.95` | A → F | 45.2% | **-7.0pp** | 19% |

`top_k` 和 `top_p` 合并 (C 条件) 的效应 **-18pp** ≈ E + F 单独效应之和 (-12 + -7 = -19), **二者近似可加**。

---

## 5. 参数 saturation 互动效应

**rep_penalty 一旦改成 1.0, top_k 和 top_p 的影响被吸收**:

| 比较 | 干预率变化 |
|------|-----------|
| B (rep=1.0, top_k=-1) → D (rep=1.0, top_k=20+top_p=0.95) | 17.1% → 16.1% (**仅 -1pp**) |
| A (rep=1.3, top_k=-1) → C (rep=1.3, top_k=20+top_p=0.95) | 52.2% → 34.1% (-18pp) |

→ **top_k/top_p 的边际效应高度依赖于 rep_penalty 是否生效**。在 rep_penalty=1.3 下 top_k/top_p 改变能压低干预率 -18pp; 在 rep_penalty=1.0 下相同改变只压 -1pp。

这强烈暗示三个参数都**通过"修改 trajectory"间接影响后续 step 的 entropy 分布**, 而 rep_penalty 的 trajectory effect 远大于 top_k/top_p, 一旦它生效几乎把所有压低 entropy 的空间吃完。

---

## 6. 机制: 为什么这些 sampling 参数会改变 entropy?

**关键事实**: SIA processor 在 vllm sampler 的 **`apply_logits_processors`** 阶段调用, 看到的是 **pre-temperature / pre-penalty / pre-topk 的 raw logits** (vllm 0.10 / 0.19 都一样)。所以 sampling 参数**不直接修改 SIA 看到的当前 step logits**。

但 sampling 参数通过**间接 trajectory 路径**影响 SIA:

```
                     step N → SIA 看到 raw logits, 算 entropy
                          ↓                ↓
                       (干预 / SKIP)    (无关 sampling 参数)
                          ↓
                       sampling pipeline 应用 rep_penalty + temperature + top_k + top_p
                          ↓
                       sampler 选 token (受 sampling 参数影响)
                          ↓
                       chosen token 加入 context
                          ↓
                     step N+1 → 主 LLM forward 在新 context 上算 logits
                          ↑
                     **这里的 logit 分布形状取决于上一步的 token 选择**
```

具体机制:

| 参数 | trajectory 效应 | 对后续 entropy 的影响 |
|------|----------------|---------------------|
| **rep_penalty=1.3** | 每步把已出现 token 的 logit /1.3, 强制 sampler 偏冷门 token | 模型一直走"不自然"路径, 进 "less confident" context → 下一步 logit 更平 → entropy 升高 |
| **top_k=-1** (无截断) | 允许 sampler 从全 vocab 采, tail token 偶尔被选 | 类似上面但更弱 (大多数 sample 仍在 top-20 内, 长尾偶尔触发) |
| **top_p=1.0** (无 nucleus) | 跟 top_k=-1 类似, 但 5% 长尾概率被允许 | 最弱效应 |

`rep_penalty` 是**每步全 vocab 系统化扰动**, top_k/top_p 只在**偶发长尾采样**时影响。这就是为什么 rep_penalty 的效应远大。

---

## 7. 跟 0GM 乱码 bug 的关系: 两难局面

之前修 0GM 输出乱码时引入这套参数变化, 因为:
- 0GM vocab 248K (远大于 14B 的 152K), 多 ~96K multilingual rare token
- 历史 `rep_penalty=1.3` 在 0GM 上把英文常用 token 压低, sampler 转向 OOV 多语言 token → 乱码
- 改成 `rep_penalty=1.0` 后 0GM 输出干净

但本实证显示: `rep_penalty=1.3` **同时是历史 SIA 30% 干预率的主要驱动**。所以:

| 选择 | 0GM 输出质量 | SIA 干预率 |
|------|--------------|-----------|
| `rep_penalty=1.3` (历史) | ❌ 乱码 (大词表 trap) | ✅ ~30% |
| `rep_penalty=1.0` (现状) | ✅ 干净 | ❌ ~10% |

**两个都重要, 不能同时满足**, 需要折中。

---

## 8. 修复路径

### 方案 A — 接受 0GM 干预率天然 ~10%, 改 threshold

- 0GM 用 `rep_penalty=1.0`, 保证输出不乱码
- 把 `entropy_threshold` 从 1.0 降到 0.5-0.7 让干预率回到 ~30%
- 代价: threshold 是 per-setting calibrate 的 magic number, 跟 SIA 论文里 1.0 默认对不上 (但本来论文 threshold 也是 model-specific)

**估计成本**: 30 分钟跑 60Q 验证。

### 方案 B — 找一组对两个模型都 OK 的中间值

- 试 `rep_penalty=1.15 + top_k=40`, 既避免 0GM OOV 漂移又保留部分 trajectory diversity
- 14B 干预率预计 ~20-25% (B 跟 A 之间), 0GM 输出可能仍干净 (rep_penalty=1.15 < 1.3, OOV 推力减半)
- 需要小范围实验确定 sweet spot

**估计成本**: 1-2 小时, 3-5 个参数组合 × 各 30Q。

### 方案 C — 模型分流

- 14B 路径完全不动 (`rep_penalty=1.3`), 保 14B baseline 不受影响
- 0GM 单独传 `rep_penalty=1.0` + `entropy_threshold=0.6` (A 方案的 per-model 版本)
- 工程改动: 让 eval client 按 model 名自动选择参数, 或者文档化两套不同的启动命令

**估计成本**: 1 小时编码 + 30 分钟验证。

---

## 9. 结论 — 一句话

> **历史 SIA 30% 干预率 ≈ `repetition_penalty=1.3` 通过 trajectory effect 把模型推向 less-confident context, 让后续 step 的 top-5 entropy 升高造成的;** 6 月修 0GM 乱码 bug 时把 `rep_penalty` 改成 1.0, **副作用是 SIA 干预率掉到 ~10% (97% 的下降归因于这一个参数变化)**。0GM-35B 跟 14B 在同样 rep_penalty 下干预率几乎一致 (12.97% vs 10.65%), 跟模型大小/MoE/thinking 模式无关。

---

## 10. 相关文件

- 本次 14B debug eval: `/tmp/14b_server_sia_debug_20260602_083338.log`
- 14B with-params eval (15Q): `/tmp/14b_sia_debug_eval_20260602_083610.log`
- 14B no-params eval (15Q): `/tmp/14b_sia_debug_eval_noparams_20260602_085409.log`
- 0GM debug eval (15Q): `/tmp/sia_debug_eval_20260602_075302.log`
- 0GM 完整 600Q eval: [`exp/mmlu_0gm_600q_sia_20260602_032711.log`](../exp/mmlu_0gm_600q_sia_20260602_032711.log)
- Debug instrumentation commit: `b72764e` (`debug: per-request entropy/gap/think histograms`)
- 相关分析:
  - [`doc/0gm-35b-sia-vs-nosia-eval-20260602.md`](0gm-35b-sia-vs-nosia-eval-20260602.md) — 0GM 全 eval 数据 + 早期 entropy 分析 (本 doc 修正了那里的"35B 更自信"假设)
  - [`doc/0gm-35b-sia-rm-inproc-path-20260602.md`](0gm-35b-sia-rm-inproc-path-20260602.md) — 0GM 跨进程 RM 的成本分析
