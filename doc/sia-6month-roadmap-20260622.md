# SIA 六个月 Roadmap（2026-07 ~ 2026-12）

**撰写日期**：2026-06-22  
**执笔**：工程侧  
**受众**：管理层  

---

## 当前基线（2026-06 末）

| 指标 | SIA（当前）| noSIA | 差距 |
|------|-----------|-------|------|
| conc=16 吞吐（tok/s）| 369 | 1040 | **−64%** |
| conc=16 ITL（ms）| 41.3 | 13.6 | **3.0×** |
| VM per-batch latency | ~30ms | — | 瓶颈所在 |
| 干预效果（AlpacaEval）| 难以量化 | — | 无评估基准 |
| 多模态支持 | ❌ 文字盲打 | — | VLM 场景缺失 |

**三个核心痛点**：P1 效果不显著 / P2 VM 延迟高 / P3 高并发吞吐下降严重

---

## 执行顺序总览

```
Month 1  ██ block-wise scoring ██ 双熵门控 ██ 评估基准
Month 2  ████████ 同词表 VM 训练（数据收集 + 训练 + 初步验证）████████
Month 3  ████████ 同词表 VM 上线（dense 4B，含 MoE 选型）████████
Month 4  ████ 更强 VM（8B）████  ████ accept/reject 干预模式 ████
Month 5  ████ PRM 训练 ████              ▶ 多模态VM训练启动
Month 6  ████████ 多模态 VM 上线 ████████  ██ DPO 蒸馏实验 ██
```

每个阶段严格依赖前一阶段：前期降低 VM 调用频率 → 中期换更强模型 → 后期做根本性改造。

---

## Month 1（2026-07）：工程快赢——降低 VM 调用频率

**主线任务**：不改变模型，纯代码修改，把 VM 的有效调用频率从 ~20% 降到 ~5%。

### 任务 1.1：block-wise scoring（B=4）

修改 `SIALogitsProcessor`：每生成 4 个 token 才触发一次 VM 打分，而非每个 token 都触发。

- **理论依据**：arxiv 2503.02368 实测 per-token RM 比 block-wise(4) 慢 2.7-4.4×
- **工作量**：约 1 周（代码改动集中在 `apply()` 方法，需调整 token buffer 逻辑）
- **风险**：效果可能略降，需 A/B 对比，找最优 B 值

### 任务 1.2：双熵门控替换单熵门控

现有门控：主模型熵 > θ 才干预（单熵）。  
改为：主模型熵 > θ₁ **且** VM 上一步评分也显示高不确定性时才干预（双熵）。  
减少低质量的无效干预，同时减少 VM 调用次数约 20-30%。

- **工作量**：约 1 周（`apply()` 中增加一个状态缓存）

### 任务 1.3：建立评估基准

选定 200 题 AlpacaEval 作为固定测试集，建立可复现的评估 pipeline。这是后续所有优化的衡量标准，不建基准则无法证明改进。

- **工作量**：约 1 周

### Month 1 交付标准

| 指标 | Month 1 目标 | 当前基线 |
|------|-------------|---------|
| conc=16 SIA tok/s | **≥ 550** | 369 |
| conc=16 ITL（ms）| **≤ 28** | 41.3 |
| AlpacaEval 评估 | **可运行、有结果** | 无 |

---

## Month 2（2026-08）：同词表 VM 训练

**主线任务**：全力投入同词表 VM 的数据收集与训练，为 Month 3 部署做好准备。本月无新的吞吐量提升，Month 1 的工程收益（block-wise + 双熵门控）持续生效。

**为什么同词表是最高优先级**：当前 VM（Qwen3-4B）和主 LLM（0GM-35B）使用不同 tokenizer（Qwen3 32K 词表 vs Qwen3.5 248K 词表）。这意味着 VM 看到的 prefix token ID 与主 LLM 生成的 token 不是一一对应的，评分信号存在系统性噪声。同词表训练一次性解决：

- 跨分词器噪声消除 → 评分精度提升（P1）
- CUDA graph 可能恢复 → per-call latency 30ms → ~17ms（P2）
- 每次调用读取的 HBM 数据量不变，但质量更好（P3 间接改善）

### 任务 2.1：偏好数据收集

从现有生产流量中收集 SIA 干预前后的 token 偏好对，或使用 synthetic 方法（让主 LLM 对同一 prompt 生成 N 个候选，用现有 VM 打标签）。

- **工作量**：约 1-2 周
- **风险**：数据量不足时用 synthetic 补充

### 任务 2.2：VM-Qwen3.5-4B 训练——使用 ARM 训练目标

以 Qwen3.5-4B（248K 词表，与 0GM-35B 相同）为 base 训练 reward head。

**关键：训练目标选 ARM（Autoregressive Reward Model），而非传统 ORM（Outcome Reward Model）。**  
理由来自 GenARM（ICLR 2025，arxiv 2410.08193）：ARM 参数化**在理论上可将 frozen LLM 引导至 KL 正则 RL 框架内任意传统 RM 可实现的分布**，而 ORM 只对最终输出打分，逐 token 干预时信号粗糙、位置错位。同等 4B 规模下，ARM 目标比 ORM 目标理论信号质量更高，先验证这一点，再决定是否需要扩大模型规模。

同步评估是否有合适的小 MoE base 可选（VRAM 允许时 MoE per-call 读带宽更低）。

- **工作量**：约 2-3 周（训练 + 初步 offline 验证）

### 任务 2.3：极小 judge PoC（与 2.2 并行，1-2 周）

**来源**：Judge Decoding（ICLR 2025，arxiv 2501.19309）用 16.4k 参数线性层替换 speculative decoding 的接受准则，500 条偏好对、1.5 小时训练，实现 3.9–9.7× 加速。

**SIA 类比**：在主 LLM（0GM-35B）的 LogitsProcessor 内部，基于 top-K logit 分布训练一个**极小线性评分头**（<10M 参数），替代外部 4B VM。如果可行，per-call latency 从 ~30ms 降至 <0.1ms，P2/P3 根本解决。

- **工作量**：约 1 周代码 + 1 周效果对比
- **风险**：精度可能不足以指导 token 选择（VM 信号比 speculative decoding 的接受/拒绝更细）
- **成功标准**：极小 judge vs 4B VM 的 AlpacaEval win-rate 差距 ≤ 3%
- **失败处理**：如精度不足，实验结论也有价值（证明该方向的边界）

### Month 2 交付标准

| 指标 | Month 2 目标 | Month 1 基线 |
|------|-------------|------------|
| 同词表 ARM VM 训练 | **训练完成，offline 验证通过** | — |
| MoE 选型结论 | **dense 4B 或 MoE 选型确定** | — |
| 极小 judge PoC | **完成实验，有效果对比数据** | — |
| conc=16 SIA tok/s | **维持 ≥ 550**（本月无新工程优化）| ≥ 550 |

---

## Month 3（2026-09）：同词表 VM 上线——质效双提

**主线任务**：完成同词表 VM 训练，完成选型（dense 4B vs MoE），部署上线，替换现有 VM。

### 任务 3.1：同词表 VM 训练与选型

**Dense 4B**（默认选项）：与当前 VM 规模相当，同词表后消除噪声，可恢复 CUDA graph。  
**MoE 选型评估**（如有合适的 pretrained MoE base）：MoE 在 memory-bandwidth-bound 场景下，每次 forward 只读激活参数（约 3B），理论上比同质量 dense 更快。以 30B-A3B 为例：

| 对比 | 当前 VM（Qwen3-4B dense）| MoE 30B-A3B |
|------|--------------------------|-------------|
| 每 GPU forward 读带宽 | ~0.5 GB | ~0.375 GB（激活参数 ÷TP=4）|
| 模型能力 | 4B | 接近 30B 水平 |
| VRAM 占用（TP=4）| ~2 GB/GPU | ~7.5 GB/GPU（需确认可放下）|

如果 VRAM 允许，MoE base 是更优选择：更快、更强，一举两得。否则使用 dense 4B。

### 任务 3.2：同词表 VM 部署与验证

替换生产 VM，重新跑 AlpacaEval，验证效果提升。

### Month 3 交付标准

| 指标 | Month 3 目标 | Month 2 基线 |
|------|-------------|------------|
| conc=16 SIA tok/s | **≥ 750** | ≥ 550 |
| b2_score_call p50 | **≤ 20ms**（同词表 + 可能恢复 CUDA graph）| ~30ms |
| AlpacaEval win-rate vs noSIA | **可量化，有正提升** | 难以量化 |

---

## Month 4（2026-10）：VM 能力升级 + 干预策略实验

**主线任务**：基于 Month 3 的 ARM 4B 验证结论，决定是否扩大模型规模；同步实验 accept/reject 干预模式（作为探索性实验，不作强承诺）。

### 任务 4.1：规模决策——ARM 4B 效果若不足，训练 ARM 8B

**决策逻辑**（来自文献调研）：文献（GenARM，ICLR 2025）表明 VM 的关键在于**训练目标**而非模型规模——ARM（Autoregressive RM）4B 理论上优于 ORM 8B。Month 3 完成 ARM 4B 上线后，先用 AlpacaEval 验证效果。

- 若 ARM 4B 的 win-rate 已达 ≥ +5%：**跳过 8B 训练**，把资源投入 Month 5 的 PRM 或多模态
- 若 ARM 4B 效果仍不足：在 ARM 训练目标下扩大到 8B（此时 block-wise 已降低调用频率，8B 的更高 per-call latency 可承受）

- **工作量**：2-3 周（视 Month 3 结果决定是否执行）

### 任务 4.2：accept/reject 干预模式实验（探索性）

当前模式：VM 分数加到 logit 上（logit biasing）。  
新模式：VM 分高 → 接受 top-1；VM 分低 → 强制从 top-K 重采样。

**注意**：RSD（ICML 2025，arxiv 2501.19324）的 accept/reject 工作在**推理步骤/序列级别**，而非 token 级别，且使用的是两个独立大小模型的架构，与 SIA 的单模型 + 外挂 VM 不同。**token 级 accept/reject 在顶会文献中缺乏直接验证**，本任务作为探索性实验，结果不确定，需 A/B 实测。

- **工作量**：1-2 周实验
- **成功标准**：accept/reject 模式的 AlpacaEval win-rate 高于 logit biasing 模式

### Month 4 交付标准

| 指标 | Month 4 目标 | Month 3 基线 |
|------|-------------|------------|
| AlpacaEval win-rate vs noSIA | **≥ +5%（ARM 4B 或 ARM 8B）** | 有正提升 |
| 干预模式对比 | **logit bias vs accept/reject 有实测数据** | — |
| conc=16 SIA tok/s | **维持 ≥ 750** | ≥ 750 |

---

## Month 5（2026-11）：PRM + 多模态 VM 启动

**两条并行主线**，面向不同场景。

### 任务 5.1：Process Reward Model（PRM）训练——聚焦推理/数学/代码任务

当前 VM 是 token 级 outcome reward model，信号粒度极细，噪声大。PRM 改为**推理步骤级**打分（在自然推理分隔点评估），调用频率从 O(token) 降至 O(step)，信号质量更高。

**适用范围说明**（基于文献）：ICLR 2024 对 ORM vs PRM 的系统对比（Lightman et al.，arxiv 2305.20050）实验全部在 MATH 数学数据集上进行。PRM 优于 ORM 的结论在**推理/数学/代码**类任务上有顶会支撑，在通用问答/对话类任务上缺乏直接验证。**本任务优先针对推理类任务，不宜对通用指令跟随场景过度承诺。**

- **数据需求**：构建 step-level preference 数据集（数学/推理类，每步打好/坏标签）
- **预期效果**：推理类任务 VM 调用次数降低 3-5×，推理类 benchmark 效果进一步提升
- **工作量**：4 周（数据准备 2 周 + 训练 + 验证 2 周）

### 任务 5.2：多模态 VM 训练启动（针对 Qwen3-VL-30B 场景）

**为什么需要多模态 VM**：当主 LLM 是 Qwen3-VL-30B-A3B-Instruct 时，用户输入包含图片，当前 text-only VM 完全看不到图片内容——VM 评分等于盲猜，干预可能适得其反。

**解法**：以多模态 VLM（如 Qwen3-VL-4B）为 base，训练 reward head，使 VM 能够同时理解图文 context。

- **适用范围**：仅对 VLM 主推理 LLM 有意义（纯文字 0GM-35B 无需此功能）
- **工作量**：4 周（多模态偏好数据构建 + 训练），Month 5 启动，Month 6 交付

### Month 5 交付标准

| 指标 | Month 5 目标 |
|------|-------------|
| PRM 上线（推理类任务）| 推理任务 VM 调用次数 ≤ 20% of token 数 |
| 多模态 VM 训练 | **已启动，训练中** |

---

## Month 6（2026-12）：多模态上线 + DPO 蒸馏实验

### 任务 6.1：多模态 VM 上线（Qwen3-VL 场景完整支持）

部署多模态 VM，Qwen3-VL-30B 场景下 SIA 对图文输入的干预质量从"盲猜"升级为"真正理解图片"。

### 任务 6.2：DPO 蒸馏实验（纯探索性，低优先级）

方向：收集 SIA 系统中 VM 偏好的生成轨迹，用 DPO 蒸馏进 0GM-35B 主模型，让主模型内化对齐信号。

**注意（来自文献调研）**：TITA（2025）展示了推理时 log-ratio 方法（DPO 等价形式）有效，但那是推理时校正，不是训练时蒸馏。训练时 DPO 蒸馏在本次调研中**没有直接顶会证据支撑**，效果不确定。

- **这是纯探索性方向**，不作为 Month 6 的主要交付
- 如果 Month 2 的**极小 judge PoC** 结果积极，Month 6 的精力优先转向把极小 judge 打磨到可生产的质量
- 如果极小 judge 结果为负，再转向 DPO 蒸馏方向

**Month 6 实际优先级**：多模态 VM 上线（主线）> 极小 judge 生产化（视 Month 2 结果）> DPO 蒸馏探路

### Month 6 交付标准

| 指标 | Month 6 目标 |
|------|-------------|
| Qwen3-VL + SIA 多模态干预 | **正式可用** |
| DPO 蒸馏实验 | **完成可行性验证（有无效果）** |

---

## 终态汇总（2026-12 末预期）

| 指标 | 当前（2026-06）| 6 个月后目标 |
|------|----------------|------------|
| conc=16 SIA tok/s | 369 | **≥ 750** |
| SIA/noSIA 吞吐比 | 35% | **≥ 72%** |
| SIA/noSIA ITL 倍数 | 3.0× | **≤ 1.5×** |
| AlpacaEval win-rate vs noSIA | 难以量化 | **≥ +5%（可量化证明）** |
| 多模态 VLM 场景支持 | ❌ | **✅** |
| VM 是否需要每 token 调用 | 是（~20% token）| **否（block-wise + PRM，~5% 以下）**|

---

## 里程碑时间线

```
2026-07 末  conc=16 tok/s ≥ 550，评估基准建立
2026-08 末  同词表 VM 训练完成，offline 验证通过
2026-09 末  conc=16 tok/s ≥ 750，同词表 VM 上线，效果首次可量化
2026-10 末  AlpacaEval win-rate ≥ +5%，最优干预模式确定
2026-11 末  PRM 上线，推理任务 VM 调用进一步降低
2026-12 末  多模态 VM 上线，DPO 蒸馏探路完成
```

---

## 学术调研关键发现（2026-06）

本 roadmap 在制定前进行了系统性文献调研（覆盖 NeurIPS、ICML、ICLR、ACL 等顶会，108 个并行验证智能体，25 篇论文，121 条论断经三票对抗验证）。以下三项发现值得管理层重点关注：

### 发现一：高并发吞吐是学术空白，SIA 有机会贡献原创研究

当前所有 token 级对齐论文（包括 ICLR 2025 的 GenARM、ICML 2025 的 RSD 等顶会工作）**均只在单请求或小并发（≤4）场景下评估效果**，没有任何顶会论文系统研究"高并发连续批处理下 token 级干预的吞吐-效果权衡"。

这意味着我们在生产环境中观察到的 P3 问题（conc=16 吞吐下降 64%）是一个**尚未有学术解答的真实工程难题**。如果 SIA 项目系统性地：

- 建立高并发场景下的评估基准（吞吐 vs 对齐效果 Pareto）
- 验证 block-wise scoring / PRM 等方案在高并发下的实际效果
- 发表相关结果

**有可能成为该方向的第一批顶会投稿**，填补当前学术空白。

---

### 发现二：Judge Decoding 的"极小 judge"思路对 SIA 有重大潜力，尚无人探索

ICLR 2025 发表的 Judge Decoding（arxiv 2501.19309）提出了一个反直觉的结论：用**仅 16,384 个参数（16.4k）的线性投影层**替换大模型的 token 接受准则，以 500 条标注数据、不到 1.5 小时训练，实现 Llama-405B 推理 **3.9–9.7× 加速**。

当前 SIA 使用 4B 参数的 VM 打分，每次调用延迟 ~30ms。

**尚未有任何工作**探索：能否将 Judge Decoding 的极小 judge 思路移植到 SIA——用一个参数量极小的打分头（如 10M 以下）替代 4B VM，训练数据同样来自偏好对，推理时几乎零延迟（参数量降低 400×，延迟可能降低 50–100×）？

这一方向的代价是打分精度可能下降，但对于 SIA 已有熵门控（只有 ~20% 的 token 触发打分）的场景，极小 judge 只需在关键位置做粗粒度好/坏判断，精度要求可能并不高。**这是 6 个月 roadmap 之外、值得单独立项探索的研究方向。**

---

### 发现三：该领域论文的性能数字普遍存在夸大，需谨慎参考

调研过程中对 25 条具体性能论断进行了三票对抗验证，**15 条（60%）被否定**，典型案例：

| 论文声称 | 验证结果 |
|---------|---------|
| TARo：比 baseline 提升 +22.4% | ❌ 0-3 否定（数字无法在原文中核实）|
| TARo：比现有 token 级方法提升 +8.4% | ❌ 0-3 否定 |
| TITA：在 LLaVA-1.5 上 MMVet +8.6%、POPE +6.7% | ❌ 1-2 否定 |
| RSD：比并行解码方法平均提升 +3.5 准确率 | ❌ 0-3 否定 |
| GSI：端到端延迟降低 28%、吞吐提升 51% | ❌ 1-2 否定 |

**实践建议**：在评估外部论文声称的效果改善时，不应直接引用其论文数字；在对 SIA 自身效果做对外宣传时，应确保数字来自可复现的独立评估（AlpacaEval、MMLU 等标准 benchmark），而非内部测试集。

---

## SIA 技术局限性

在推进 roadmap 的同时，需要明确 SIA 当前及可预见未来的技术边界，以便合理设定预期。

| # | 局限性 | 说明 |
|---|--------|------|
| L1 | **仅适用于自部署模型** | SIA 通过在 vLLM 内部注入 `LogitsProcessor` 修改每步的 logit 分布，需要白盒访问模型的内部推理过程。无法用于调用 OpenAI、Anthropic 等外部 API 的场景，也无法作用于只暴露生成接口的封闭推理服务。 |
| L2 | **效果高度依赖 VM 训练数据覆盖范围** | Value Model 只能在训练数据所覆盖的领域和任务类型上提供可靠的引导。对于训练数据未涉及的垂直领域（医疗、法律、特定代码库等），VM 的评分可能是噪声甚至负向引导，不能期望 SIA 在这些场景带来正向效果。 |
| L3 | **多模态输入需单独训练 VM** | 原始 SIA 论文及当前实现仅支持纯文本输入。当主 LLM 为视觉语言模型（VLM，如 Qwen3-VL-30B）且用户输入包含图片时，text-only VM 完全看不到图片内容，评分退化为盲猜。多模态场景需单独训练支持图文输入的 VM 并重新评测效果（roadmap Month 6 目标）。 |
| L4 | **效果是统计平均，单条请求不保证** | SIA 的对齐改善是在大量请求上的统计提升，对具体某条请求无法保证方向。VM 在 VM 训练数据未覆盖的领域或罕见 prompt 类型上，单次干预可能使输出变差。效果评估必须依赖批量统计指标（如 AlpacaEval win-rate），不能用单条结果下判断。 |
| L5 | **VM 与主 LLM 共享显存，互相制约** | b2 inproc 方案要求 VM 和主 LLM 运行在同一组 GPU 上并共享显存。VM 越大，主 LLM 可用的显存越少，反之亦然。这限制了：（a）主 LLM 可支持的最大 context length；（b）主 LLM 可承载的最大并发数；（c）VM 的最大模型规模。 |
| L6 | **与推理框架版本强绑定，升级有风险** | SIA 的 b2 inproc 实现深度依赖 vLLM 内部 API，已验证 vLLM 0.19+ 上对 0GM-35B 的 b2 inproc 全面失效。每次升级主 LLM 推理框架（vLLM 版本），SIA 层都需要重新适配和回归验证，升级成本不可忽视。 |

---

## 依赖关系与风险

| 风险 | 影响阶段 | 缓解措施 |
|------|---------|---------|
| block-wise B=4 效果损失超预期 | Month 1 | 备选 B=2（牺牲一半收益，保证效果）|
| 同词表 VM 训练数据不足 | Month 2 | 用 synthetic preference 数据补充 |
| MoE base VRAM 放不下（>10GB/GPU）| Month 2-3 | fallback dense 4B |
| 同词表 VM 训练效果不如预期 | Month 3 | A/B 对比，保留旧 VM 作兜底 |
| 多模态偏好数据难以获取 | Month 5-6 | 降级为探路实验，不作强承诺 |

---

## 参考文献

| 论文 | 对应任务 | 会议/状态 |
|------|---------|---------|
| SIA 原论文 (arxiv 2602.21215) | 整体框架基础 | 2026 preprint |
| GenARM: Autoregressive Reward Model (arxiv 2410.08193) | 任务 2.2 ARM 训练目标 | **ICLR 2025** ✅ |
| Judge Decoding (arxiv 2501.19309) | 任务 2.3 极小 judge PoC | **ICLR 2025** ✅ |
| Let's Verify Step by Step / ORM vs PRM (arxiv 2305.20050) | 任务 5.1 PRM 适用范围 | **ICLR 2024** ✅ |
| Scaling LLM Test-Time Compute (arxiv 2408.03314) | 整体方向验证 | NeurIPS 2024 Workshop ✅ |
| RSD: Reward-guided Speculative Decoding (arxiv 2501.19324) | 任务 4.2 accept/reject 参考 | **ICML 2025** ✅ |
| TITA: Token-level Inference-Time Alignment (arxiv 2510.21794) | 任务 6.2 DPO 蒸馏参考 | 2025 preprint |
| Iterative Value Function Optimization (arxiv 2503.02368) | 任务 1.1 block-wise scoring | 2025 preprint |
| EASD: Entropy-Aware Speculative Decoding (arxiv 2512.23765) | 任务 1.2 双熵门控 | 2025 preprint |
| TARo: Token-level Adaptive Routing (arxiv 2603.18411) | 任务 1.2 自适应路由参考 | 2026 preprint |
