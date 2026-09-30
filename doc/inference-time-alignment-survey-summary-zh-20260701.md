# 推理时对齐文献调研摘要（2026-07-01）

共发现182篇论文，精读60篇。完整调研详见 [inference-time-alignment-survey-20260701.md](inference-time-alignment-survey-20260701.md)。

---

## 核心结论

推理时对齐领域的所有系统都面临和SIA一样的VM调用瓶颈。学术界已收敛到三种架构层面的解法：SIA已实现其中之一，FaRMA覆盖第二种，第三种需要数月研究。

---

## 可行动优先级排名

| 优先级 | 动作 | 吞吐量提升 | 难度 |
|--------|------|-----------|------|
| **1** | **FaRMA vocab-wide head**（已规划）| **2.36× 已验证** | 需要重新训练VM |
| **2** | **调高 `--entropy_threshold` 1.0 → 1.5–2.0** | 1.1–1.3×，零成本 | 几天 |
| **3** | **审查b2 inproc是否对K=10候选共享prefix KV cache** | 1.1–1.5×（若未启用）| 几天profiling |
| **4** | **换用更小的VM（1.7B或0.6B，HuggingFace上有现成checkpoint）** | VM调用成本降低1.5–2× | 几天，零代码改动 |
| **5** | CriticControl风格：把VM信号蒸馏到LLM隐层一个小critic头 | 理论上可彻底消除VM调用 | 数月 |

**叠加效果估算：** FaRMA vocab-head + 收紧entropy_threshold + 换小VM，三项可叠加，预计达到 **700–1100 tok/s**，填补当前322→1340 tok/s差距的60–80%，无需改动serving架构。

---

## 论文分类总览（7大类）

### A类：稀疏/选择性干预（SIA所在的类）

- **SIA**（arXiv:2602.21215）— 本项目的基础论文，entropy gate，~25%干预率
- **Nudging**（ACL 2025, 2410.09300）— 用top-1概率而非entropy做gate，~10%干预率。**关键启示：当前SIA的25%干预率可能过于保守，调高entropy_threshold到1.5–2.0是最快的免费优化**
- **CARDS**（COLM 2025, 2406.16306）— 基于entropy的segment级别rejection sampling，每20个token调用1次RM。架构上与vLLM LogitsProcessor不兼容，无法直接采用
- **STARS**（arXiv:2511.03827）— 固定间隔同步batching，强制所有并发请求在同一步骤调用RM。与vLLM continuous batching不兼容
- **GGRO**（UAI 2026）— 需要RM backward pass，与b2 inproc不兼容，跳过
- **TARo**（arXiv:2603.18411）— 每个token都调用RM（100%），与SIA稀疏策略相反，跳过

### B类：单次前向传播覆盖全词表（最直接相关）

- **FaRMA**（ICML 2025, 2502.04517）— RM输出|vocab|维向量，一次前向传播读取全部K个候选分数，**已验证2.36×提速，最高优先级**
- **GenARM**（ICLR 2025, 2410.08193）— 自回归RM，next-token log-probs即为reward信号，一次前向传播覆盖全词表。优点：标准causal LM结构，b2 inproc无需修改即可serve。缺点：比FaRMA的单线性层更重。**可在FaRMA训练基础设施搭建时同步评估**
- **RAD-Q/ARM**（TMLR 2025, 2407.04615）— 低秩Q-head，通过prefix隐向量与LLM冻结output embedding的点积，O(1)覆盖全词表。关键限制：要求VM与LLM共享embedding空间，Qwen3-4B和Qwen3-VL-30B是否满足条件需要验证
- **GeDi**（EMNLP 2021）、**DExperts**（ACL 2021）、**Proxy-Tuning**（COLM 2024）— 均通过两个LM前向传播覆盖全词表，密集（非稀疏），不适合直接用于SIA

### C类：RM蒸馏进LLM（长期方向）

- **CriticControl**（ACL 2023, 2212.10938）— 用terminal-only VM reward训练小型value critic；推理时critic（微秒级）替代VM（10–22ms）。对简单任务（情感/毒性）已验证，对复杂对齐任务的保真度未知。**这是填补760→1340 tok/s剩余差距的唯一路径，需要2–3个月研究**
- **Otter**（arXiv:2408.10764）— 向冻结LLM层插入参数，reward从LLM自身前向传播中读取，无需外部VM。实现难度高

### D类：密集per-token logit bias（无稀疏，验证设计但不提速）

- **ARGS**（ICLR 2024）— SIA的直接前身，每个token都调用RM K次，无entropy gate
- **RAD**（EMNLP 2023）— 因果RM + KV-cache prefix复用。**关键问题：当前b2 inproc在对K=10候选打分时，是否对共享的prefix（system prompt + user + 已生成response）做了KV cache复用？若没有，启用prefix caching可带来1.1–1.5×免费提速**
- **PARGS**（COLM 2025）— 理论上证明必须在partial sequence级别训练RM，验证了SIA VM的训练方式正确
- **PPLM**、**FUDGE**、**IVG**、**PAD** 等 — 密集干预，无稀疏，不适用

### E类：序列级别Best-of-N（与SIA token级别无关）

- **BOND**（NeurIPS 2024）、**BoNBoN**（NeurIPS 2024）、**TreeBoN**（EMNLP 2025）、**Speculative Rejection**（NeurIPS 2024）等 — 均在完整序列级别做RM评分，与SIA的per-token架构无关，跳过

### F类：Speculative Decoding + Alignment

- **RSD**（arXiv:2501.19324）、**GSI**（ICLR 2026）、**Judge Decoding**（ICLR 2025）等 — 用draft model + PRM做accept/reject，与SIA的logit bias架构不同，不兼容

### G类：训练方法/RM建模（无推理时机制）

- **Q-RM**（ICML 2025）、**TinyRM**（ICML 2025 Workshop）、**CPMI**（ACL 2026）等 — 纯训练方法，无推理优化路径

---

## 下一步行动计划

```
近期（几天内）：
  1. entropy_threshold 从1.0调高到1.5或2.0，跑AlpacaEval对比质量
  2. 审查b2 inproc：K=10候选打分时是否复用prefix KV cache？
  3. 用VM-Qwen3-1.7B-Base做A/B测试：吞吐量 vs 质量

中期（几周）：
  4. FaRMA vocab-wide head训练（已有doc/farma-vm-training-plan-20260701.md）
  5. 训练时同步评估GenARM作为备选架构

长期（数月）：
  6. CriticControl蒸馏：在FaRMA验证质量后，若760 tok/s仍不够，再投入
```
